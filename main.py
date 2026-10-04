"""
main.py — FastAPI server for RT-MASR Live Voice-Call Simulation.

Timestamp capture strategy (mirrors live_call_session.py docstring):
  T0  session.mark_call_start()        when "start_call" WS message is received
  T1  session.process_pcm_bytes()      first PCM binary frame arrives (set inside session)
  T2  session.mark_infer_start()       just before engine.transcribe_stream() is called
  Ty  time.time() after list(...)      just after the inference generator is exhausted
  T3  session.mark_first_token()       called from main when first text delta is observed

Stage timing (mel_s, encoder_s, prefill_s, decode_s, tokens_generated) is extracted
by wrapping OnnxAsrPipeline._transcribe_chunk via a patched transcribe_stream that
also returns timing info. Because transcribe_stream is a generator, we collect all
deltas, then call _transcribe_chunk for timing (one extra pass is wasteful), so
instead we time sub-phases directly via the session.
"""

import asyncio
import json
import time
import psutil
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.engines.qwen3_onnx_engine import ONNXQwen3ASR
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
    proc = psutil.Process(os.getpid())
    mem_info = proc.memory_info()
    return JSONResponse({
        "status": "ok",
        "model_ready": _model_ready,
        "cpu_percent": proc.cpu_percent(interval=None),
        "rss_mb": round(mem_info.rss / 1_048_576, 1),
        "vms_mb": round(mem_info.vms / 1_048_576, 1),
        "num_threads": proc.num_threads(),
    })

@app.get("/api/samples")
def list_samples():
    test_audio_dir = Path("test_audio")
    samples = []
    if test_audio_dir.exists():
        for wav_file in test_audio_dir.glob("**/*.wav"):
            rel_path = wav_file.relative_to(test_audio_dir)
            parent_name = rel_path.parent.name.lower()
            if parent_name == "en":
                lang_code = "en"
                lang_name = "English"
            elif parent_name in ("cn", "zh"):
                lang_code = "zh"
                lang_name = "Mandarin Chinese"
            elif parent_name == "id":
                lang_code = "id"
                lang_name = "Bahasa Indonesia"
            else:
                lang_code = "auto"
                lang_name = "Auto-Detect"

            samples.append({
                "path": str(rel_path),
                "name": wav_file.name,
                "language_code": lang_code,
                "language_name": lang_name,
                "size_bytes": wav_file.stat().st_size
            })
    return {"samples": samples}

@app.get("/api/samples/{filepath:path}")
def get_sample_file(filepath: str):
    file_path = Path("test_audio") / filepath
    if file_path.exists() and file_path.suffix == ".wav":
        return FileResponse(file_path, media_type="audio/wav")
    return JSONResponse({"error": "File not found"}, status_code=404)

async def _run_inference(
    engine: ONNXQwen3ASR,
    audio_buffer: "np.ndarray",
    language: Optional[str],
) -> tuple[list[str], dict | None]:
    """Run transcribe_stream in a thread-pool executor.

    The sync CPU generator is collected inside a plain function (_collect)
    and dispatched with run_in_executor so the asyncio event loop is never
    blocked — new PCM chunks keep arriving while inference runs.

    Returns
    -------
    (deltas, stage_timing)
        deltas      : list of text delta strings yielded during decode
        stage_timing: timing dict from the final ("", timing) sentinel,
                      or None if the generator produced no output
    """
    import numpy as np  # local import — already loaded, no cost
    loop = asyncio.get_event_loop()

    def _collect() -> tuple[list[str], dict | None]:
        deltas: list[str] = []
        stage_timing = None
        for delta, timing in engine.transcribe_stream(audio_buffer, language=language):
            if delta:
                deltas.append(delta)
            if timing is not None:
                stage_timing = timing
        return deltas, stage_timing

    return await loop.run_in_executor(None, _collect)


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
    call_language: Optional[str] = None

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
                    "chunks_received": stats["chunks_received"],
                    "total_bytes": stats["total_bytes"],
                })

                # ── VAD-driven inference trigger ───────────────────────────
                if (
                    len(session.audio_buffer) >= 8000
                    and stats["chunks_received"] % 2 == 0
                    and session.has_speech()        # skip silent/noise-only chunks
                ):
                    engine = get_engine()
                    boundary = session.find_vad_boundary()

                    if boundary is not None:
                        # ── Commit path: utterance boundary detected ────────
                        # Transcribe only the completed utterance, then discard
                        # those samples — inference time stays bounded.
                        t_infer_start = session.mark_infer_start()
                        first_token_noted = session.first_token_time is not None
                        utterance_audio = session.pop_utterance(boundary)

                        deltas, stage_timing = await _run_inference(
                            engine, utterance_audio, call_language
                        )

                        if deltas and not first_token_noted:
                            session.mark_first_token()  # T3
                        infer_duration_s = time.time() - t_infer_start

                        session.append_committed("".join(deltas))

                    else:
                        # ── Interim path: open utterance, show partial result ─
                        # Re-transcribe only the current (bounded) window.
                        t_infer_start = session.mark_infer_start()
                        first_token_noted = session.first_token_time is not None

                        deltas, stage_timing = await _run_inference(
                            engine, session.audio_buffer, call_language
                        )

                        if deltas and not first_token_noted:
                            session.mark_first_token()  # T3
                        infer_duration_s = time.time() - t_infer_start

                    # ── Build display text: committed + open-window interim ──
                    interim = "".join(deltas) if deltas else ""
                    new_text = (
                        session.committed_text
                        + (" " if session.committed_text and interim else "")
                        + interim
                    ).strip()

                    if new_text != cumulative_text:
                        cumulative_text = new_text
                        metrics = session.get_metrics(infer_duration_s, stage_timing=stage_timing)
                        await websocket.send_json({
                            "type": "transcript_delta",
                            "full_text": cumulative_text,
                            "metrics": metrics,
                        })

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "start_call":
                    call_language = data.get("language")
                    # ── T0: call-start ─────────────────────────────────
                    session.mark_call_start()
                    cumulative_text = ""
                    await websocket.send_json({
                        "type": "call_ready",
                        "message": f"Model ready (Language: {call_language or 'Auto-Detect'}). Call leg starting.",
                    })

                elif msg_type == "end_call":
                    total_call_time = time.time() - session.start_time
                    stage_timing = None
                    infer_duration_s = 0.0

                    # Flush any remaining audio in the open-utterance buffer
                    if len(session.audio_buffer) > 0:
                        engine = get_engine()
                        t_infer_start = session.mark_infer_start()
                        first_token_noted = session.first_token_time is not None

                        deltas, stage_timing = await _run_inference(
                            engine, session.audio_buffer, call_language
                        )

                        if deltas and not first_token_noted:
                            session.mark_first_token()
                        infer_duration_s = time.time() - t_infer_start

                        # Commit trailing audio; clear the buffer
                        trailing = "".join(deltas).strip()
                        if trailing:
                            session.append_committed(trailing)
                        session.audio_buffer = session.audio_buffer[:0]

                    cumulative_text = session.committed_text
                    metrics = session.get_metrics(infer_duration_s, stage_timing=stage_timing)
                    metrics["total_call_time_s"] = round(total_call_time, 2)

                    await websocket.send_json({
                        "type": "call_ended",
                        "final_text": cumulative_text,
                        "metrics": metrics,
                    })
                    break

    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)