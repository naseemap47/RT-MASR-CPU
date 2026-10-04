# benchmark/tests/test_concurrency_runner.py
import sys, os, time, dataclasses
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.runners.concurrency_runner import ConcurrencyResult, ConcurrencyRunner


class _FakeEngine:
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        time.sleep(0.02)
        return {
            "text": "hello",
            "language": "en",
            "timing": {"total_s": 0.02, "audio_duration_s": 2.0, "rtf": 0.01,
                       "tokens_generated": 1, "mel_s": 0, "encoder_s": 0,
                       "prefill_s": 0, "decode_s": 0.02}
        }


def test_concurrency_result_is_dataclass():
    assert dataclasses.is_dataclass(ConcurrencyResult)


def test_concurrency_runner_returns_one_result_per_legs(tmp_path):
    audio = str(tmp_path / "a.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)

    engine = _FakeEngine()
    runner = ConcurrencyRunner(
        config_id="test",
        engine_factory=lambda: engine,
        audio_files=[audio],
        legs_list=[1, 2],
        n_rounds=1,
    )
    results = runner.run()
    assert len(results) == 2  # one per legs value
    assert {r.n_legs for r in results} == {1, 2}


def test_concurrency_result_has_required_fields(tmp_path):
    audio = str(tmp_path / "b.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)

    engine = _FakeEngine()
    runner = ConcurrencyRunner(
        config_id="fields_test",
        engine_factory=lambda: engine,
        audio_files=[audio],
        legs_list=[1],
        n_rounds=1,
    )
    results = runner.run()
    r = results[0]
    assert "mean" in r.latency_stats
    assert "p95" in r.latency_stats
    assert "mean" in r.rtf_stats
    assert r.throughput_audio_hours_per_wall_hour > 0
    assert r.error_count == 0
    assert "peak_rss_mb" in r.system_metrics


def test_concurrency_runner_errors_counted(tmp_path):
    audio = str(tmp_path / "c.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)

    call_count = {"n": 0}
    class _BrokenEngine:
        def transcribe(self, audio_path, **kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("Simulated error")
            return {"text": "ok", "language": "en",
                    "timing": {"total_s": 0.01, "audio_duration_s": 1.0, "rtf": 0.01,
                               "tokens_generated": 1, "mel_s": 0, "encoder_s": 0,
                               "prefill_s": 0, "decode_s": 0.01}}

    runner = ConcurrencyRunner(
        config_id="error_test",
        engine_factory=lambda: _BrokenEngine(),
        audio_files=[audio],
        legs_list=[2],
        n_rounds=1,
    )
    results = runner.run()
    assert results[0].error_count > 0
