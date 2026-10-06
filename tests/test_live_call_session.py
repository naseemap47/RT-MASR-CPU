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


# ── Input format normalisation ─────────────────────────────────────────────

def test_declared_baseline_format_is_passed_through():
    sess = LiveCallSession()
    fmt = sess.set_input_format(sample_rate=16000, channels=1, encoding="pcm_s16le")
    assert fmt["resample"] is False and fmt["downmix"] is False

    raw = np.zeros(8000, dtype=np.int16).tobytes()
    assert sess.process_pcm_bytes(raw)["new_samples_count"] == 8000


def test_8khz_input_is_resampled_to_16khz():
    sess = LiveCallSession()
    assert sess.set_input_format(sample_rate=8000)["resample"] is True

    # 1 s of 8 kHz audio must become ~1 s of 16 kHz audio, i.e. ~16000 samples
    t = np.linspace(0, 1, 8000, endpoint=False)
    raw = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16).tobytes()
    sess.process_pcm_bytes(raw)

    assert len(sess.audio_buffer) == pytest.approx(16000, abs=4)
    assert sess.audio_buffer.dtype == np.float32
    # the resampled tone must survive as speech-level energy, not be rescaled
    assert sess.has_speech() is True


def test_resampling_is_continuous_across_chunks():
    sess = LiveCallSession()
    sess.set_input_format(sample_rate=8000)

    t = np.linspace(0, 1, 8000, endpoint=False)
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    for start in range(0, 8000, 2000):           # four 0.25 s packets
        sess.process_pcm_bytes(pcm[start:start + 2000].tobytes())

    one_shot = LiveCallSession()
    one_shot.set_input_format(sample_rate=8000)
    one_shot.process_pcm_bytes(pcm.tobytes())

    # splitting the same audio across packets must not change the result
    assert np.allclose(sess.audio_buffer, one_shot.audio_buffer, atol=1e-6)


def test_stereo_input_is_averaged_to_mono():
    sess = LiveCallSession()
    assert sess.set_input_format(channels=2)["downmix"] is True

    left = np.full(4000, 10000, dtype=np.int16)
    right = np.full(4000, -10000, dtype=np.int16)      # opposite phase → cancels
    interleaved = np.empty(8000, dtype=np.int16)
    interleaved[0::2] = left
    interleaved[1::2] = right

    result = sess.process_pcm_bytes(interleaved.tobytes())
    assert result["new_samples_count"] == 4000          # one mono frame per pair
    assert np.allclose(sess.audio_buffer, 0.0, atol=1e-4)


def test_partial_frame_is_carried_to_the_next_packet():
    sess = LiveCallSession()
    raw = np.arange(100, dtype=np.int16).tobytes()

    first = sess.process_pcm_bytes(raw + b"\x01")       # one byte past a frame
    assert first["new_samples_count"] == 100            # odd byte held back

    second = sess.process_pcm_bytes(b"\x00" + raw)      # completes it
    assert second["new_samples_count"] == 101
    assert len(sess.audio_buffer) == 201


def test_unsupported_encoding_is_rejected():
    sess = LiveCallSession()
    with pytest.raises(ValueError, match="unsupported encoding"):
        sess.set_input_format(encoding="mulaw")


def test_unsupported_sample_rate_is_rejected():
    sess = LiveCallSession()
    with pytest.raises(ValueError, match="unsupported sample_rate"):
        sess.set_input_format(sample_rate=100)


def test_mark_call_start_restores_baseline_format():
    sess = LiveCallSession()
    sess.set_input_format(sample_rate=8000, channels=2)
    sess.mark_call_start()
    fmt = sess.input_format()
    assert (fmt["sample_rate"], fmt["channels"]) == (16000, 1)


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
