import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.engines.qwen3_engine import ONNXQwen3ASR
from src.engines.live_call_session import LiveCallSession

_engine: Optional[ONNXQwen3ASR] = None
_model_ready: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_ready
    print("Preloading ONNX Qwen3 ASR Model Pipeline...")
    _engine = ONNXQwen3ASR()
    _model_ready = True
    print("ONNX Qwen3 ASR Model Pipeline Preloaded and Ready.")
    yield

app = FastAPI(title="RT-MASR Live Voice-Call Simulation", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

def get_engine() -> ONNXQwen3ASR:
    global _engine
    if _engine is None:
        _engine = ONNXQwen3ASR()
    return _engine

@app.get("/")
def get_ui():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "RT-MASR Web UI Server Ready"})

@app.get("/api/health")
def check_health():
    return JSONResponse({
        "status": "ok",
        "model_ready": _model_ready
    })

@app.get("/api/samples")
def list_samples():
    test_audio_dir = Path("test_audio")
    samples = []
    if test_audio_dir.exists():
        for wav_file in test_audio_dir.glob("*.wav"):
            samples.append({
                "name": wav_file.name,
                "size_bytes": wav_file.stat().st_size
            })
    return {"samples": samples}

@app.get("/api/samples/{filename}")
def get_sample_file(filename: str):
    file_path = Path("test_audio") / filename
    if file_path.exists() and file_path.suffix == ".wav":
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)

@app.websocket("/ws/call-stream")
async def websocket_call_stream(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_json({
        "type": "connected",
        "model_ready": _model_ready,
        "message": "Connected to RT-MASR Live Call Stream"
    })
    
    session = LiveCallSession()
    cumulative_text = ""
    start_call_time = time.time()
    
    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                break
            
            if message.get("type") == "websocket.disconnect":
                break
                
            if "bytes" in message and message["bytes"]:
                pcm_data = message["bytes"]
                stats = session.process_pcm_bytes(pcm_data)
                
                await websocket.send_json({
                    "type": "chunk_ack",
                    "buffered_seconds": stats["buffered_seconds"],
                    "chunks_received": stats["chunks_received"]
                })
                
                if len(session.audio_buffer) >= 8000 and stats["chunks_received"] % 2 == 0:
                    engine = get_engine()
                    t0 = time.time()
                    deltas = list(engine.transcribe_stream(session.audio_buffer))
                    proc_time = time.time() - t0
                    
                    new_text = "".join(deltas)
                    if new_text != cumulative_text:
                        cumulative_text = new_text
                        metrics = session.get_metrics(proc_time)
                        await websocket.send_json({
                            "type": "transcript_delta",
                            "full_text": cumulative_text,
                            "metrics": metrics
                        })
            
            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")
                
                if msg_type == "start_call":
                    start_call_time = time.time()
                    await websocket.send_json({
                        "type": "call_ready",
                        "message": "Model ready. Call leg starting."
                    })
                elif msg_type == "end_call":
                    total_call_time = time.time() - start_call_time
                    if len(session.audio_buffer) > 0:
                        engine = get_engine()
                        t0 = time.time()
                        deltas = list(engine.transcribe_stream(session.audio_buffer))
                        proc_time = time.time() - t0
                        cumulative_text = "".join(deltas)
                    metrics = session.get_metrics(total_call_time)
                    
                    await websocket.send_json({
                        "type": "call_ended",
                        "final_text": cumulative_text,
                        "metrics": metrics
                    })
                    break
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)