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


# ── VAD sentence chunking ──────────────────────────────────────────────────

def test_find_vad_boundary_returns_none_on_short_buffer():
    sess = LiveCallSession()
    sess.audio_buffer = np.zeros(1000, dtype=np.float32)
    assert sess.find_vad_boundary() is None


def test_find_vad_boundary_detects_silence_after_speech():
    sess = LiveCallSession()
    t = np.linspace(0, 0.5, 8000)
    speech = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(8000, dtype=np.float32)
    # speech then silence — boundary should land in the silence region
    sess.audio_buffer = np.concatenate([speech, silence])
    boundary = sess.find_vad_boundary(min_silence_samples=3200)
    assert boundary is not None
    assert boundary > 3200  # must be past the min guard


def test_find_vad_boundary_force_commits_on_overflow():
    sess = LiveCallSession()
    # All-speech buffer larger than max_utterance_samples
    t = np.linspace(0, 15, 240000)
    sess.audio_buffer = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    boundary = sess.find_vad_boundary(max_utterance_samples=240000)
    assert boundary == 240000


def test_pop_utterance_removes_samples_correctly():
    sess = LiveCallSession()
    sess.audio_buffer = np.arange(16000, dtype=np.float32)
    utterance = sess.pop_utterance(8000)
    assert len(utterance) == 8000
    assert len(sess.audio_buffer) == 8000
    assert utterance[0] == pytest.approx(0.0)
    assert sess.audio_buffer[0] == pytest.approx(8000.0)


def test_append_committed_joins_with_space():
    sess = LiveCallSession()
    sess.append_committed("Hello")
    sess.append_committed("world")
    assert sess.committed_text == "Hello world"


def test_append_committed_ignores_whitespace_only():
    sess = LiveCallSession()
    sess.append_committed("   ")
    assert sess.committed_text == ""


def test_mark_call_start_resets_committed_text():
    sess = LiveCallSession()
    sess.committed_text = "previous transcript"
    sess.mark_call_start()
    assert sess.committed_text == ""
