# benchmark/tests/test_latency_runner.py
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dataclasses
from benchmark.runners.latency_runner import LatencyResult, LatencyRunSummary, LatencyRunner


class _FakeEngine:
    """Fake engine that returns deterministic timing without loading a real model."""
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        time.sleep(0.02)  # simulate 20ms inference
        return {
            "text": "hello world",
            "language": "en",
            "timing": {
                "total_s": 0.02,
                "audio_duration_s": 2.0,
                "rtf": 0.01,
                "tokens_generated": 2,
                "mel_s": 0.005,
                "encoder_s": 0.010,
                "prefill_s": 0.002,
                "decode_s": 0.003,
            }
        }


def test_latency_result_is_dataclass():
    assert dataclasses.is_dataclass(LatencyResult)


def test_latency_run_summary_is_dataclass():
    assert dataclasses.is_dataclass(LatencyRunSummary)


def test_latency_runner_returns_summaries(tmp_path):
    # Create a fake audio file (content doesn't matter for FakeEngine)
    audio_file = str(tmp_path / "test.wav")
    with open(audio_file, "wb") as f:
        f.write(b"\x00" * 100)

    engine = _FakeEngine()
    runner = LatencyRunner(
        config_id="test_config",
        engine=engine,
        audio_files=[audio_file],
        n_runs=2,
        warmup_runs=1,
    )
    summaries = runner.run()

    assert len(summaries) == 1
    s = summaries[0]
    assert s.config_id == "test_config"
    assert s.audio_file == audio_file
    assert s.n_runs == 2
    assert "mean" in s.latency_stats
    assert "p50" in s.latency_stats
    assert "p95" in s.latency_stats
    assert "mean" in s.rtf_stats
    assert "peak_rss_mb" in s.system_metrics


def test_latency_runner_warmup_discarded(tmp_path):
    """Warmup runs should not appear in the n_runs count."""
    audio_file = str(tmp_path / "test.wav")
    with open(audio_file, "wb") as f:
        f.write(b"\x00" * 100)

    call_count = {"n": 0}
    class _CountingEngine:
        def transcribe(self, audio_path, **kwargs):
            call_count["n"] += 1
            return {
                "text": "hi",
                "language": "en",
                "timing": {"total_s": 0.01, "audio_duration_s": 1.0,
                           "rtf": 0.01, "tokens_generated": 1,
                           "mel_s": 0, "encoder_s": 0, "prefill_s": 0, "decode_s": 0.01}
            }

    runner = LatencyRunner(
        config_id="count_test",
        engine=_CountingEngine(),
        audio_files=[audio_file],
        n_runs=3,
        warmup_runs=1,
    )
    summaries = runner.run()
    # warmup_runs=1 + n_runs=3 = 4 total calls
    assert call_count["n"] == 4
    assert summaries[0].n_runs == 3
