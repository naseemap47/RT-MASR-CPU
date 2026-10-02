import time
from pathlib import Path
from fastapi.testclient import TestClient
from utils.audio_utils import load_audio
import numpy as np
from main import app

client = TestClient(app)

def test_full_stream_simulation():
    sample_path = Path("test_audio/librispeech_0_1089_0.wav")
    assert sample_path.exists()

    wav = load_audio(str(sample_path))
    int16_audio = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)
    raw_bytes = int16_audio.tobytes()

    chunk_size = 8000 * 2  # 0.5s chunks (16000 bytes)
    chunks = [raw_bytes[i:i + chunk_size] for i in range(0, len(raw_bytes), chunk_size)]

    with client.websocket_connect("/ws/call-stream") as websocket:
        init_data = websocket.receive_json()
        assert init_data["type"] == "connected"

        received_messages = []
        for chunk in chunks[:4]:
            websocket.send_bytes(chunk)
            msg = websocket.receive_json()
            received_messages.append(msg["type"])

        websocket.send_json({"type": "end_call"})
        end_msg = websocket.receive_json()
        while end_msg["type"] != "call_ended":
            received_messages.append(end_msg["type"])
            end_msg = websocket.receive_json()

        assert end_msg["type"] == "call_ended"
        assert "metrics" in end_msg
        assert end_msg["metrics"]["audio_duration_s"] > 0
