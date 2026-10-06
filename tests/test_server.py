from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_samples():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) > 0

def test_websocket_connect():
    with client.websocket_connect("/ws/call-stream") as websocket:
        data = websocket.receive_json()
        assert data["type"] == "connected"
        websocket.send_json({"type": "end_call"})
        end_data = websocket.receive_json()
        assert end_data["type"] == "call_ended"



def test_start_call_echoes_accepted_audio_format():
    """A non-baseline PCM format is accepted and the normalisation it triggers is reported."""
    with client.websocket_connect("/ws/call-stream") as websocket:
        assert websocket.receive_json()["type"] == "connected"
        websocket.send_json({
            "type": "start_call",
            "language": "en",
            "audio": {"sample_rate": 8000, "channels": 2, "encoding": "pcm_s16le"},
        })
        ready = websocket.receive_json()
        assert ready["type"] == "call_ready"
        fmt = ready["audio_format"]
        assert fmt["sample_rate"] == 8000 and fmt["channels"] == 2
        assert fmt["resample"] is True and fmt["downmix"] is True
        assert fmt["normalized_to"]["sample_rate"] == 16000
        websocket.send_json({"type": "end_call"})
        assert websocket.receive_json()["type"] == "call_ended"


def test_start_call_rejects_non_linear_pcm():
    with client.websocket_connect("/ws/call-stream") as websocket:
        assert websocket.receive_json()["type"] == "connected"
        websocket.send_json({
            "type": "start_call",
            "audio": {"sample_rate": 8000, "encoding": "mulaw"},
        })
        err = websocket.receive_json()
        assert err["type"] == "error"
        assert err["error"] == "unsupported_audio_format"


def test_health_reports_cpu_percent_and_core_count():
    """cpu_percent keeps psutil's per-core scale (100 = one busy core), so it can exceed 100;
    cpu_cores lets the UI explain that."""
    import main
    import psutil
    from fastapi.testclient import TestClient

    client = TestClient(main.app)
    main._cpu_latest = 400.0
    try:
        h = client.get("/api/health").json()
    finally:
        main._cpu_latest = 0.0
    assert h["cpu_percent"] == 400.0
    assert h["cpu_cores"] == psutil.cpu_count(logical=True)
    assert "cpu_percent_raw" not in h
