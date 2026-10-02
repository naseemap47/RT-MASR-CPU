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
