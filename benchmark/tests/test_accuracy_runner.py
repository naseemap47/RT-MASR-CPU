# benchmark/tests/test_accuracy_runner.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import dataclasses
from benchmark.runners.accuracy_runner import AccuracyResult, AccuracyRunner


class _PerfectEngine:
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        # Returns text matching the reference exactly
        return {
            "text": "hello world",
            "language": "en",
            "timing": {"total_s": 0.01, "audio_duration_s": 1.0, "rtf": 0.01,
                       "tokens_generated": 2, "mel_s": 0, "encoder_s": 0,
                       "prefill_s": 0, "decode_s": 0.01}
        }


class _EmptyEngine:
    def transcribe(self, audio_path: str, **kwargs) -> dict:
        return {
            "text": "",
            "language": "en",
            "timing": {"total_s": 0.01, "audio_duration_s": 1.0, "rtf": 0.01,
                       "tokens_generated": 0, "mel_s": 0, "encoder_s": 0,
                       "prefill_s": 0, "decode_s": 0.01}
        }


REFS = [
    {
        "audio": "fake_en.wav",
        "lang": "en",
        "metric": "wer",
        "text": "hello world",
    }
]


def test_accuracy_result_is_dataclass():
    assert dataclasses.is_dataclass(AccuracyResult)


def test_accuracy_runner_perfect(tmp_path):
    audio = str(tmp_path / "fake_en.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)

    refs = [{"audio": audio, "lang": "en", "metric": "wer", "text": "hello world"}]
    runner = AccuracyRunner("test", _PerfectEngine(), refs)
    results = runner.run()
    assert len(results) == 1
    assert results[0].score == 0.0


def test_accuracy_runner_empty_hypothesis(tmp_path):
    audio = str(tmp_path / "fake_en.wav")
    with open(audio, "wb") as f:
        f.write(b"\x00" * 10)

    refs = [{"audio": audio, "lang": "en", "metric": "wer", "text": "hello world"}]
    runner = AccuracyRunner("test", _EmptyEngine(), refs)
    results = runner.run()
    assert results[0].score == 1.0  # 2 deletions / 2 words = 1.0


def test_accuracy_runner_missing_audio_skips():
    refs = [{"audio": "/nonexistent/path.wav", "lang": "en", "metric": "wer",
             "text": "hello"}]
    runner = AccuracyRunner("test", _PerfectEngine(), refs)
    results = runner.run()
    # Missing audio file should be skipped, not crash
    assert len(results) == 0
