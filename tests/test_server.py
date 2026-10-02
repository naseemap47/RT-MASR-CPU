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
