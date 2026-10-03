import numpy as np
import pytest
from src.engines.live_call_session import LiveCallSession

def test_live_call_session_pcm_conversion():
    session = LiveCallSession(sample_rate=16000)
    # 0.5s of 16kHz 16-bit int16 mono audio (8000 samples = 16000 bytes)
    raw_int16 = (np.sin(np.linspace(0, 100, 8000)) * 16384).astype(np.int16)
    raw_bytes = raw_int16.tobytes()

    result = session.process_pcm_bytes(raw_bytes)
    assert result["buffered_seconds"] == pytest.approx(0.5, abs=0.05)
    assert len(session.audio_buffer) == 8000
    assert session.audio_buffer.dtype == np.float32


# ── has_speech() ───────────────────────────────────────────────────────────

def test_has_speech_returns_false_on_silence():
    sess = LiveCallSession()
    sess.audio_buffer = np.zeros(8000, dtype=np.float32)
    assert sess.has_speech() is False


def test_has_speech_returns_true_on_tone():
    sess = LiveCallSession()
    t = np.linspace(0, 0.5, 8000)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sess.audio_buffer = tone
    assert sess.has_speech() is True


def test_has_speech_checks_only_last_window():
    sess = LiveCallSession()
    # 16000 samples: first half silence, second half speech
    silence = np.zeros(8000, dtype=np.float32)
    t = np.linspace(0, 0.5, 8000)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sess.audio_buffer = np.concatenate([silence, tone])
    # window_samples=8000 → checks only the last 8000 samples = tone half → True
    assert sess.has_speech(window_samples=8000) is True
