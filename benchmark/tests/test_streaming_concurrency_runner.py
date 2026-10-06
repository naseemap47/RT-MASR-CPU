# benchmark/tests/test_streaming_concurrency_runner.py
import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.reporters.json_reporter import to_serialisable
from benchmark.reporters.summary_reporter import SummaryReporter
from benchmark.runners.concurrency_runner import ConcurrencyRunner
from benchmark.runners.streaming_concurrency_runner import (
    StreamingConcurrencyResult,
    StreamingConcurrencyRunner,
)

SR = 16000
PACE = 25.0     # play audio 25x faster than real time so tests stay quick


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)   # RMS ~0.21 > speech gate


def _loader(seconds: dict[str, float] | None = None):
    seconds = seconds or {}
    return lambda path: _tone(seconds.get(path, 6.0))


class _FakeQwen:
    """transcribe_stream-only engine (VAD / utterance path)."""
    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay, self.fail = delay, fail

    def transcribe_stream(self, audio, language=None, max_new_tokens=None):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        yield "hello ", None
        yield "world", None
        yield "", {"total_s": self.delay}


class _FakeWhisper:
    """transcribe(window) -> segments; text grows with the window (sliding-window path)."""
    def __init__(self, delay: float = 0.0, fail: bool = False):
        self.delay, self.fail = delay, fail

    def transcribe(self, audio, language=None, beam_size=1, fallback=False):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("boom")
        secs = len(audio) / SR
        words = " ".join(f"w{i}" for i in range(int(secs)))
        return {"text": words, "language": "en",
                "segments": [{"text": words, "start": 0.0, "end": secs}], "timing": {}}


def _runner(engine, mode, legs=(1, 2), files=("a.wav",), **kw):
    kw.setdefault("pace", PACE)
    kw.setdefault("audio_loader", _loader())
    kw.setdefault("warmup", False)
    return StreamingConcurrencyRunner(
        config_id="t", engine_factory=lambda: engine, audio_files=list(files),
        legs_list=list(legs), stream_mode=mode, **kw,
    )


@pytest.mark.parametrize("mode,engine", [("vad_utterance", _FakeQwen()), ("sliding_window", _FakeWhisper())])
def test_one_result_per_level_with_every_leg_streaming(mode, engine):
    results = _runner(engine, mode, legs=(1, 3)).run()
    assert [r.n_legs for r in results] == [1, 3]
    for r in results:
        assert isinstance(r, StreamingConcurrencyResult)
        assert r.stream_mode == mode
        assert len(r.legs) == r.n_legs
        assert r.error_count == 0
        assert r.total_passes > 0 and r.pass_latency_stats["count"] == r.total_passes
        assert r.legs_with_text == r.n_legs
        assert r.legs_kept_up == r.n_legs
        assert r.first_text_stats["count"] == r.n_legs
        assert all(leg.final_text.strip() for leg in r.legs)
        assert r.system_metrics["peak_rss_mb"] > 0


def test_audio_is_paced_not_drained_as_fast_as_possible():
    r = _runner(_FakeQwen(), "vad_utterance", legs=(2,), audio_loader=_loader({"a.wav": 8.0})).run()[0]
    # 8 s of audio at 25x -> at least 0.32 s of wall time even though the engine is instant.
    assert r.wall_elapsed_s >= 8.0 / PACE * 0.95
    assert r.wall_elapsed_s < 4.0


def test_each_leg_is_an_independent_source_cycling_over_files():
    r = _runner(_FakeQwen(), "vad_utterance", legs=(3,), files=("a.wav", "b.wav"),
                audio_loader=_loader({"a.wav": 4.0, "b.wav": 6.0})).run()[0]
    assert [leg.audio_file for leg in r.legs] == ["a.wav", "b.wav", "a.wav"]
    assert [round(leg.audio_s) for leg in r.legs] == [4, 6, 4]
    assert r.total_audio_s == pytest.approx(14.0)


@pytest.mark.parametrize("mode,slow,fast", [
    ("vad_utterance", _FakeQwen(delay=0.3), _FakeQwen()),
    ("sliding_window", _FakeWhisper(delay=0.3), _FakeWhisper()),
])
def test_slow_engine_falls_behind_and_fast_engine_keeps_up(mode, slow, fast):
    # Chunks arrive every 0.5/25 = 20 ms; a 300 ms pass cannot keep up with that.
    behind = _runner(slow, mode, legs=(1,), lag_threshold_s=0.1).run()[0]
    assert behind.legs_kept_up == 0
    assert behind.staleness_stats["p95"] > 0.1

    ok = _runner(fast, mode, legs=(1,), lag_threshold_s=1.0).run()[0]
    assert ok.legs_kept_up == 1


@pytest.mark.parametrize("mode,engine", [
    ("vad_utterance", _FakeQwen(fail=True)),
    ("sliding_window", _FakeWhisper(fail=True)),
])
def test_engine_errors_are_counted_and_leg_does_not_keep_up(mode, engine):
    r = _runner(engine, mode, legs=(2,)).run()[0]
    assert r.error_count > 0
    assert r.legs_kept_up == 0
    assert r.legs_with_text == 0


def test_audio_load_failure_surfaces_before_any_leg_starts():
    def bad(path):
        raise OSError("nope")
    with pytest.raises(OSError):        # decoding happens up front, before any leg starts
        _runner(_FakeQwen(), "vad_utterance", audio_loader=bad).run()


def test_stagger_offsets_leg_start_times():
    r = _runner(_FakeQwen(), "vad_utterance", legs=(2,), stagger_s=0.2).run()[0]
    # leg 1 starts 0.2 s after leg 0, so the level takes at least that much longer than one stream.
    assert r.wall_elapsed_s >= 6.0 / PACE + 0.2 - 0.05


def test_warmup_runs_one_untimed_pass():
    calls = {"n": 0}

    class _Counting(_FakeQwen):
        def transcribe_stream(self, audio, language=None, max_new_tokens=None):
            calls["n"] += 1
            yield from super().transcribe_stream(audio, language)

    _runner(_Counting(), "vad_utterance", legs=(1,), warmup=True).run()
    assert calls["n"] >= 2      # warm-up pass + at least one measured pass


def test_invalid_arguments_rejected():
    with pytest.raises(ValueError):
        _runner(_FakeQwen(), "bogus")
    with pytest.raises(ValueError):
        _runner(_FakeQwen(), "vad_utterance", chunk_s=0)


def test_result_is_json_serialisable():
    r = _runner(_FakeWhisper(), "sliding_window", legs=(2,)).run()[0]
    blob = json.dumps(to_serialisable(r))
    assert '"n_legs": 2' in blob and '"stream_mode": "sliding_window"' in blob


def test_summary_reporter_renders_streaming_and_batch_sections(tmp_path):
    stream = _runner(_FakeQwen(), "vad_utterance", legs=(1, 2)).run()

    audio = str(tmp_path / "a.wav")
    open(audio, "wb").write(b"\x00" * 10)

    class _E:
        def transcribe(self, p, **kw):
            return {"timing": {"audio_duration_s": 1.0}}

    batch = ConcurrencyRunner("b", lambda: _E(), [audio], legs_list=[1], n_rounds=2, warmup=False).run()

    path = SummaryReporter(str(tmp_path)).render({"concurrency": stream + batch})
    text = open(path).read()
    assert "One leg = one independently streamed audio source" in text
    assert "Kept up" in text and "Max Legs Kept Up" in text
    assert "Batch (offline) mode" in text and "Requests" in text
    assert "1/1" in text and "2/2" in text


@pytest.mark.parametrize("mode,engine", [
    ("vad_utterance", _FakeQwen(delay=0.3)),
    ("sliding_window", _FakeWhisper(delay=0.3)),
])
def test_overload_guard_aborts_a_leg_that_falls_far_behind(mode, engine):
    t = time.perf_counter()
    r = _runner(engine, mode, legs=(1,), abort_lag_s=0.2).run()[0]
    assert r.legs_aborted == 1 and r.legs_kept_up == 0 and r.legs[0].aborted
    assert time.perf_counter() - t < 3.0     # stopped early instead of finishing the 6 s stream


def test_run_level_supports_global_leg_offsets_for_multi_process_slices():
    runner = _runner(_FakeQwen(), "vad_utterance", files=("a.wav", "b.wav"),
                     audio_loader=_loader({"a.wav": 4.0, "b.wav": 6.0}))
    eng = _FakeQwen()
    r = runner.run_level(eng, 2, stagger_s=0.0, leg_offset=1, start_at=time.time() + 0.1)
    assert [leg.leg for leg in r.legs] == [1, 2]
    assert [leg.audio_file for leg in r.legs] == ["b.wav", "a.wav"]
