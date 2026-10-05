# benchmark/tests/test_new_behaviour.py
"""Tests for the fixes made during the metric / pipeline review."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest
import soundfile as sf

from benchmark.metrics.audio_info import audio_duration_s
from benchmark.metrics.system_metrics import SystemSampler
from benchmark.runners.accuracy_runner import AccuracyRunner
from benchmark.runners.concurrency_runner import ConcurrencyRunner
from benchmark.runners.latency_runner import LatencyRunner
from benchmark.runners.load_timer import measure_load_with_engine
from benchmark.engine_loader import SUPPORTED_BACKENDS, load_engine


def _wav(path, seconds, sr=16000):
    sf.write(str(path), np.zeros(int(seconds * sr), dtype="float32"), sr)
    return str(path)


# ── audio duration ───────────────────────────────────────────────────────────
def test_audio_duration_from_file(tmp_path):
    assert abs(audio_duration_s(_wav(tmp_path / "a.wav", 2.5)) - 2.5) < 1e-3


def test_audio_duration_unreadable_is_none(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"\x00" * 10)
    assert audio_duration_s(str(p)) is None


# ── RTF uses the real file duration, not the engine's claim ──────────────────
def test_latency_rtf_uses_file_duration(tmp_path):
    audio = _wav(tmp_path / "a.wav", 4.0)

    class Liar:
        def transcribe(self, path, **kw):
            time.sleep(0.05)
            return {"text": "x", "timing": {"audio_duration_s": 999.0}}

    s = LatencyRunner("c", Liar(), [audio], n_runs=2, warmup_runs=0).run()[0]
    assert abs(s.audio_duration_s - 4.0) < 1e-3
    assert 0.05 / 4.0 <= s.rtf_stats["mean"] < 0.5      # ~0.0125, not 0.00005


# ── accuracy: counts, detected language, verified flag ───────────────────────
def test_accuracy_records_counts_and_flags(tmp_path):
    audio = _wav(tmp_path / "a.wav", 1.0)

    class E:
        def transcribe(self, path, **kw):
            return {"text": "the cat sat", "language": "en"}

    refs = [{"audio": audio, "lang": "en", "metric": "wer",
             "text": "the dog sat down", "verified": False}]
    r = AccuracyRunner("c", E(), refs).run()[0]
    assert (r.errors, r.ref_len) == (2, 4)
    assert r.score == 0.5
    assert r.detected_language == "en"
    assert r.verified is False


def test_accuracy_none_text_does_not_crash(tmp_path):
    audio = _wav(tmp_path / "a.wav", 1.0)

    class E:
        def transcribe(self, path, **kw):
            return {"text": None}

    refs = [{"audio": audio, "lang": "en", "metric": "wer", "text": "hello"}]
    assert AccuracyRunner("c", E(), refs).run()[0].score == 1.0


# ── concurrency: identical workload, warm-up, throughput maths ───────────────
def test_concurrency_same_workload_and_warmup(tmp_path):
    a = _wav(tmp_path / "a.wav", 2.0)
    calls = []

    class E:
        def transcribe(self, path, **kw):
            calls.append(path)
            time.sleep(0.02)
            return {"text": "x"}

    res = ConcurrencyRunner("c", lambda: E(), [a], legs_list=[1, 2], n_rounds=2).run()
    assert len(calls) == 1 + 2 + 4                     # warm-up + 1*2 + 2*2
    assert [r.n_calls for r in res] == [2, 4]
    for r in res:
        assert abs(r.total_audio_s - 2.0 * r.n_calls) < 1e-6
        assert abs(r.throughput_audio_hours_per_wall_hour - r.total_audio_s / r.wall_elapsed_s) < 1e-6


def test_concurrency_parallelism_is_real(tmp_path):
    a = _wav(tmp_path / "a.wav", 1.0)

    class E:
        def transcribe(self, path, **kw):
            time.sleep(0.2)
            return {"text": "x"}

    r = ConcurrencyRunner("c", lambda: E(), [a], legs_list=[4], n_rounds=1, warmup=False).run()[0]
    assert r.wall_elapsed_s < 0.6        # 4 x 0.2s in parallel, not 0.8s serial


# ── load timer returns the engine ────────────────────────────────────────────
def test_measure_load_with_engine_returns_engine():
    res, eng = measure_load_with_engine("c", lambda: {"engine": True})
    assert eng == {"engine": True}
    assert res.load_time_s >= 0


def test_measure_load_rss_delta_sees_big_allocation():
    res, eng = measure_load_with_engine("c", lambda: np.ones(40_000_000, dtype="float64"))  # 320 MB
    assert res.rss_delta_mb > 200
    del eng


# ── sampler: short runs still get a sample; no phantom zero first sample ─────
def test_sampler_short_run_has_sample():
    with SystemSampler(interval_s=5.0) as s:
        time.sleep(0.05)
    assert s.result()["rss_mb"]["count"] >= 1
    assert s.result()["peak_rss_mb"] > 0


def test_sampler_sees_cpu_busy_process():
    end = time.time() + 0.6
    with SystemSampler(interval_s=0.1) as s:
        while time.time() < end:
            pass
    assert s.result()["process_cpu_pct"]["max"] > 50


# ── engine loader: whisper is a supported backend ────────────────────────────
def test_whisper_backend_listed():
    assert "whisper" in SUPPORTED_BACKENDS


def test_unknown_backend_error_lists_whisper(tmp_path):
    cfg = tmp_path / "m.yaml"
    cfg.write_text("name: x\n")
    with pytest.raises(ValueError, match="whisper"):
        load_engine({"id": "x", "backend": "nope", "model_config": str(cfg)})


@pytest.mark.skipif(
    not os.path.exists("models/whisper_int8/tiny_encoder_11_int8.onnx"),
    reason="whisper tiny int8 model files not present (run from project root)",
)
def test_load_whisper_tiny_via_loader():
    eng = load_engine({"id": "w", "backend": "whisper",
                       "model_config": "config/models/whisper_int8_tiny.yaml"})
    r = eng.transcribe("test_audio/en/librispeech_1_1089_1.wav")
    assert "belly" in r["text"].lower()
    assert r["timing"]["audio_duration_s"] > 3
