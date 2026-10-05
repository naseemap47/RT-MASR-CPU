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
