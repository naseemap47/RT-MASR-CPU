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

Backend selection
-----------------
The active inference backend is chosen by ``default_model`` in config/config.yaml:
  "qwen3_onnx_0.6b_int8" → ONNXQwen3ASR      (src/engines/qwen3_onnx_engine.py)
  "qwen3_0.6b"           → Qwen3ASR 0.6B     (src/engines/qwen3_engine.py)
  "qwen3_1.7b"           → Qwen3ASR 1.7B     (src/engines/qwen3_engine.py)
  "whisper_*"            → WhisperOnnxEngine (src/engines/whisper_engine.py)
Change the YAML key (or set the RT_MASR_MODEL environment variable) to switch
backends without touching this file.

Streaming strategy per backend
------------------------------
Qwen3 (onnx / transformers)  VAD-utterance mode: the open utterance is re-transcribed
                             and committed at a silence boundary (find_vad_boundary).
Whisper                      Sliding-window mode (src/engines/whisper_streaming.py):
                             the window is re-transcribed every hop, text confirmed by
                             LocalAgreement-2 is committed, the rest is tentative, and
                             the window slides forward so cost stays bounded.
"""

import asyncio
import json
import logging
import time
import psutil
import threading
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from src.core.config import load_server_config
from src.core.model_check import ModelSetupError, check_registry_model, report
from src.core.runlog import begin_run, in_pytest
from src.engines.live_call_session import LiveCallSession
from src.engines.whisper_streaming import StreamingConfig, WhisperSlidingWindowStreamer

logger = logging.getLogger("rtmasr.server")
_run_session = None

# ── Config paths ──────────────────────────────────────────────────────────────
CONFIG_PATH = "config/config.yaml"

# ── Type alias for either engine ──────────────────────────────────────────────
ASREngine = Union["ONNXQwen3ASR", "Qwen3ASR", "WhisperOnnxEngine"]  # noqa: F821  (resolved at runtime)

_engine: Optional[ASREngine] = None
_model_ready: bool = False
_active_model_name: str = "(not loaded)"
_active_backend: str = "(unknown)"
_streaming_cfg: StreamingConfig = StreamingConfig()


def _model_name_override() -> Optional[str]:
    """Optional model override: RT_MASR_MODEL=whisper_int8_tiny uvicorn main:app"""
    return os.environ.get("RT_MASR_MODEL") or None


def _stream_mode() -> str:
    return "sliding_window" if _active_backend == "whisper" else "vad_utterance"


def _build_engine(model_cfg: dict) -> ASREngine:
    """
    Instantiate the correct engine class based on the ``backend`` field
    in the resolved per-model config dict.

    Args:
        model_cfg: Per-model config dict (from check_registry_model).

    Returns:
        A fully constructed engine instance (ONNXQwen3ASR, Qwen3ASR or WhisperOnnxEngine).

    Raises:
        ValueError: If the backend discriminator is unknown.
    """
    backend = model_cfg.get("backend", "onnx")

    if backend == "onnx":
        from src.engines.qwen3_onnx_engine import ONNXQwen3ASR
        return ONNXQwen3ASR.from_config(model_cfg)

    elif backend == "transformers":
        from src.engines.qwen3_engine import Qwen3ASR
        return Qwen3ASR.from_config(model_cfg)

    elif backend == "whisper":
        from src.engines.whisper_engine import WhisperOnnxEngine
        return WhisperOnnxEngine.from_config(model_cfg)

    else:
        raise ValueError(
            f"Unknown backend '{backend}' in model config. "
            "Expected 'onnx', 'transformers' or 'whisper'."
        )


def _load_active_model() -> ASREngine:
    """Resolve the active model config, record its metadata and build the engine."""
    global _engine, _active_model_name, _active_backend, _streaming_cfg

    override = _model_name_override()
    name = override or load_server_config(CONFIG_PATH).get("default_model")
    where = "RT_MASR_MODEL" if override else f"default_model in {CONFIG_PATH}"
    model_cfg = check_registry_model(name, config_path=CONFIG_PATH, where=where)
    _active_model_name = model_cfg.get("display_name", model_cfg.get("name", "?"))
    _active_backend    = model_cfg.get("backend", "?")
    _streaming_cfg     = StreamingConfig.from_dict(model_cfg.get("streaming"))

    logger.info("Loading ASR engine: %s (backend=%s) ...", _active_model_name, _active_backend)
    _engine = _build_engine(model_cfg)
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_ready, _run_session

    server_cfg = load_server_config(CONFIG_PATH)
    level = os.environ.get("RT_MASR_LOG_LEVEL") or server_cfg.get("server", {}).get("log_level", "info")
    _run_session = begin_run("server", level=level, skip_if_pytest=True)

    try:
        _load_active_model()
    except ModelSetupError as exc:
        report(logger, exc.panel)
        logger.error("Server startup aborted: fix the model setup above and restart.")
        if _run_session is not None:
            _run_session.close(exit_code=1)
            _run_session = None
        if in_pytest():
            raise RuntimeError("ASR model setup failed (see the message above)") from None
        # Uvicorn would follow the panel with a lifespan traceback; nothing is serving yet.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    _model_ready = True
    logger.info("Engine ready: %s (stream mode: %s)", _active_model_name, _stream_mode())
    try:
        yield
    finally:
        if _run_session is not None:
            _run_session.close()
            _run_session = None


def get_engine() -> ASREngine:
    if _engine is None:
        _load_active_model()
    return _engine

app = FastAPI(title="RT-MASR Live Voice-Call Simulation", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    class _NoCacheStatic(StaticFiles):
        """Always revalidate UI assets so edits to app.js / style.css show up on a normal reload."""
        async def get_response(self, path, scope):
            resp = await super().get_response(path, scope)
            resp.headers["Cache-Control"] = "no-cache"
            return resp

    app.mount("/static", _NoCacheStatic(directory=str(static_dir)), name="static")


@app.get("/")
def get_ui():
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"message": "RT-MASR Web UI Server Ready"})


# psutil's cpu_percent(interval=None) reports usage *since the previous call on the same
# Process object*. A fresh Process per request therefore always returned 0.0, and a
# poll-driven reading would average over whatever gap separated two requests (e.g. since
# server start for the first one). So one long-lived Process is sampled by a small
# background thread once per second and /api/health just returns the latest value.
_PROC = psutil.Process(os.getpid())
_cpu_latest = 0.0
_cpu_sampler_started = False
_cpu_sampler_lock = threading.Lock()


def _cpu_sampler_loop() -> None:
    global _cpu_latest
    _PROC.cpu_percent(interval=None)            # prime: the first call is always 0.0
    while True:
        time.sleep(1.0)
        try:
            _cpu_latest = _PROC.cpu_percent(interval=None)
        except psutil.Error:
            pass


def _ensure_cpu_sampler() -> None:
    global _cpu_sampler_started
    with _cpu_sampler_lock:
        if not _cpu_sampler_started:
            threading.Thread(target=_cpu_sampler_loop, daemon=True, name="cpu-sampler").start()
            _cpu_sampler_started = True


_ensure_cpu_sampler()      # start now (server start / reload), so values are ready before the first call


@app.get("/api/health")
def check_health():
    proc = _PROC
    _ensure_cpu_sampler()      # no-op normally; covers the case of import-time start being skipped
    cpu = round(_cpu_latest, 1)     # psutil scale: 100 % = one fully used core, so it can exceed 100 %
    mem_info = proc.memory_info()
    return JSONResponse({
        "status": "ok",
        "model_ready": _model_ready,
        "active_model": _active_model_name,
        "backend": _active_backend,
        "stream_mode": _stream_mode(),
        "cpu_percent": cpu,
        "cpu_cores": psutil.cpu_count(logical=True),   # the UI uses it to explain values above 100 %
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
                lang_name = "Chinese"
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
    engine: ASREngine,
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
    import contextvars
    import numpy as np  # local import — already loaded, no cost
    loop = asyncio.get_event_loop()
    ctx = contextvars.copy_context()

    def _collect() -> tuple[list[str], dict | None]:
        deltas: list[str] = []
        stage_timing = None
        for delta, timing in engine.transcribe_stream(audio_buffer, language=language):
            if delta:
                deltas.append(delta)
            if timing is not None:
                stage_timing = timing
        return deltas, stage_timing

    return await loop.run_in_executor(None, lambda: ctx.run(_collect))


class _CallState:
    """Mutable per-connection state shared by the handler helpers."""
    def __init__(self) -> None:
        self.cumulative_text: str = ""
        self.language: Optional[str] = None     # as sent by the UI ("" = auto-detect)


async def _ws_reader(websocket: WebSocket, incoming: "asyncio.Queue[dict]") -> None:
    """Pump raw WebSocket messages into a queue so the handler can look ahead."""
    try:
        while True:
            message = await websocket.receive()
            await incoming.put(message)
            if message.get("type") == "websocket.disconnect":
                return
    except (WebSocketDisconnect, RuntimeError):
        await incoming.put({"type": "websocket.disconnect"})


def _drain_audio(session: LiveCallSession, incoming: "asyncio.Queue[dict]") -> Optional[dict]:
    """
    Move every already-queued audio frame into the session without inferring.

    Returns the first non-audio message (control / disconnect), which the caller
    must handle next, or None if the queue ran dry. Skipping ahead like this
    keeps the stream live when a Whisper pass takes longer than the audio it
    covers, instead of building an ever-growing backlog.
    """
    while True:
        try:
            message = incoming.get_nowait()
        except asyncio.QueueEmpty:
            return None
        if message.get("bytes"):
            session.process_pcm_bytes(message["bytes"])
        else:
            return message


async def _handle_audio_vad(
    websocket: WebSocket, session: LiveCallSession, state: _CallState, pcm_data: bytes,
) -> None:
    """Qwen path: energy-gated, VAD-committed utterances (unchanged behaviour)."""
    stats = session.process_pcm_bytes(pcm_data)

    await websocket.send_json({
        "type": "chunk_ack",
        "buffered_seconds": stats["buffered_seconds"],
        "chunks_received": stats["chunks_received"],
        "total_bytes": stats["total_bytes"],
    })

    # ── VAD-driven inference trigger ───────────────────────────────────────
    if not (
        len(session.audio_buffer) >= 8000
        and stats["chunks_received"] % 2 == 0
        and session.has_speech()        # skip silent/noise-only chunks
    ):
        return

    engine = get_engine()
    boundary = session.find_vad_boundary()
    try:
        from src.core.observe import bound as _obs_bound
    except ImportError:
        from contextlib import contextmanager as _cm

        @_cm
        def _obs_bound(**_k):
            yield

    if boundary is not None:
        # ── Commit path: utterance boundary detected ───────────────────────
        # Transcribe only the completed utterance, then discard those samples
        # — inference time stays bounded.
        t_infer_start = session.mark_infer_start()
        first_token_noted = session.first_token_time is not None
        utterance_audio = session.pop_utterance(boundary)

        with _obs_bound(pass_kind="commit"):
            deltas, stage_timing = await _run_inference(engine, utterance_audio, state.language)

        if deltas and not first_token_noted:
            session.mark_first_token()  # T3
        infer_duration_s = time.time() - t_infer_start

        session.append_committed("".join(deltas))
        deltas = []     # now part of committed_text; it must not be shown again as interim

    else:
        # ── Interim path: open utterance, show partial result ──────────────
        # Re-transcribe only the current (bounded) window.
        t_infer_start = session.mark_infer_start()
        first_token_noted = session.first_token_time is not None

        with _obs_bound(pass_kind="interim"):
            deltas, stage_timing = await _run_inference(engine, session.audio_buffer, state.language)

        if deltas and not first_token_noted:
            session.mark_first_token()  # T3
        infer_duration_s = time.time() - t_infer_start

    # ── Build display text: committed + open-window interim ────────────────
    interim = "".join(deltas) if deltas else ""
    new_text = (
        session.committed_text
        + (" " if session.committed_text and interim else "")
        + interim
    ).strip()

    if new_text != state.cumulative_text:
        state.cumulative_text = new_text
        metrics = session.get_metrics(infer_duration_s, stage_timing=stage_timing)
        metrics["stream_mode"] = "vad_utterance"
        await websocket.send_json({
            "type": "transcript_delta",
            "full_text": state.cumulative_text,
            "committed_text": session.committed_text,
            "tentative_text": interim,
            "metrics": metrics,
        })


def _sliding_metrics(session: LiveCallSession, update, infer_s: float) -> dict:
    """Telemetry for a sliding-window pass (RTF is measured against the window it covered)."""
    metrics = session.get_metrics(infer_s, stage_timing=update.timing or None)
    metrics["audio_duration_s"] = round(update.window_s, 3)
    metrics["rtf"] = round(infer_s / update.window_s, 4) if update.window_s > 0 else 0.0
    metrics["stream_mode"] = "sliding_window"
    metrics["window_s"] = round(update.window_s, 2)
    metrics["window_start_s"] = round(update.window_start_s, 2)
    metrics["language"] = update.language
    return metrics


async def _handle_audio_sliding(
    websocket: WebSocket,
    session: LiveCallSession,
    streamer: WhisperSlidingWindowStreamer,
    state: _CallState,
    pcm_data: bytes,
    incoming: "asyncio.Queue[dict]",
) -> Optional[dict]:
    """
    Whisper path: sliding window + LocalAgreement (see whisper_streaming.py).

    Returns a look-ahead control message that must be processed next, if any.
    """
    stats = session.process_pcm_bytes(pcm_data)
    await websocket.send_json({
        "type": "chunk_ack",
        "buffered_seconds": stats["buffered_seconds"],
        "chunks_received": stats["chunks_received"],
        "total_bytes": stats["total_bytes"],
    })

    if not streamer.ready():
        return None

    carry = _drain_audio(session, incoming)         # skip any backlog

    t_infer_start = session.mark_infer_start()
    try:
        import contextvars
        ctx = contextvars.copy_context()
        update = await asyncio.get_running_loop().run_in_executor(
            None, lambda: ctx.run(streamer.step),
        )
    except Exception as exc:                        # keep the call alive on a bad pass
        logger.exception("sliding-window pass failed: %s", exc)
        return carry
    infer_duration_s = time.time() - t_infer_start

    if update is None:
        return carry                                # silence, nothing to do
    if update.full_text and session.first_token_time is None:
        session.mark_first_token()                  # T3

    if update.changed:
        state.cumulative_text = update.full_text
        await websocket.send_json({
            "type": "transcript_delta",
            "full_text": update.full_text,
            "committed_text": update.committed_text,
            "tentative_text": update.tentative_text,
            "metrics": _sliding_metrics(session, update, infer_duration_s),
        })
    return carry


@app.websocket("/ws/call-stream")
async def websocket_call_stream(websocket: WebSocket):
    await websocket.accept()
    peer = getattr(websocket.client, "host", None)
    logger.info("websocket connected  peer=%s  model=%s  mode=%s",
                peer, _active_model_name, _stream_mode())
    await websocket.send_json({
        "type": "connected",
        "model_ready": _model_ready,
        "stream_mode": _stream_mode(),
        "active_model": _active_model_name,
        "message": "Connected to RT-MASR Live Call Stream"
    })

    session = LiveCallSession()
    state = _CallState()
    sliding = _active_backend == "whisper" and _engine is not None
    streamer = (
        WhisperSlidingWindowStreamer(_engine, session, _streaming_cfg) if sliding else None
    )

    incoming: asyncio.Queue = asyncio.Queue()
    reader = asyncio.create_task(_ws_reader(websocket, incoming))
    carry: Optional[dict] = None

    try:
        while True:
            message = carry if carry is not None else await incoming.get()
            carry = None

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                if streamer is not None:
                    carry = await _handle_audio_sliding(
                        websocket, session, streamer, state, message["bytes"], incoming
                    )
                else:
                    await _handle_audio_vad(websocket, session, state, message["bytes"])

            elif "text" in message and message["text"]:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "start_call":
                    state.language = data.get("language")
                    logger.info("call start  language=%s", state.language or "auto")
                    # ── T0: call-start ─────────────────────────────────
                    session.mark_call_start()

                    # Optional {"audio": {"sample_rate", "channels", "encoding"}}.
                    # Absent → the 16 kHz mono Int16 baseline is assumed.
                    audio_fmt = data.get("audio") or {}
                    try:
                        accepted = session.set_input_format(
                            sample_rate=audio_fmt.get("sample_rate"),
                            channels=audio_fmt.get("channels"),
                            encoding=audio_fmt.get("encoding"),
                        )
                    except (ValueError, TypeError) as exc:
                        logger.warning("unsupported audio format: %s", exc)
                        await websocket.send_json({
                            "type": "error",
                            "error": "unsupported_audio_format",
                            "message": str(exc),
                        })
                        break

                    state.cumulative_text = ""
                    if streamer is not None:
                        streamer.reset(language=state.language or None)
                    await websocket.send_json({
                        "type": "call_ready",
                        "stream_mode": _stream_mode(),
                        "audio_format": accepted,
                        "message": f"Model ready (Language: {state.language or 'Auto-Detect'}). Call leg starting.",
                    })

                elif msg_type == "end_call":
                    total_call_time = time.time() - session.start_time
                    stage_timing = None
                    infer_duration_s = 0.0

                    if streamer is not None:
                        # Finalise the window: transcribe what is left, commit everything.
                        t_infer_start = session.mark_infer_start()
                        import contextvars
                        ctx = contextvars.copy_context()
                        update = await asyncio.get_running_loop().run_in_executor(
                            None, lambda: ctx.run(streamer.finish),
                        )
                        infer_duration_s = time.time() - t_infer_start
                        if update is not None and update.full_text and session.first_token_time is None:
                            session.mark_first_token()
                        metrics = (
                            _sliding_metrics(session, update, infer_duration_s)
                            if update is not None and update.window_s > 0
                            else session.get_metrics(0.0)
                        )
                        metrics["stream_mode"] = "sliding_window"
                        state.cumulative_text = streamer.full_text
                        final_text = streamer.committed_text
                    else:
                        # Flush any remaining audio in the open-utterance buffer
                        if len(session.audio_buffer) > 0:
                            engine = get_engine()
                            t_infer_start = session.mark_infer_start()
                            first_token_noted = session.first_token_time is not None

                            deltas, stage_timing = await _run_inference(
                                engine, session.audio_buffer, state.language
                            )

                            if deltas and not first_token_noted:
                                session.mark_first_token()
                            infer_duration_s = time.time() - t_infer_start

                            # Commit trailing audio; clear the buffer
                            trailing = "".join(deltas).strip()
                            if trailing:
                                session.append_committed(trailing)
                            session.audio_buffer = session.audio_buffer[:0]

                        state.cumulative_text = session.committed_text
                        final_text = state.cumulative_text
                        metrics = session.get_metrics(infer_duration_s, stage_timing=stage_timing)
                        metrics["stream_mode"] = "vad_utterance"

                    metrics["total_call_time_s"] = round(total_call_time, 2)
                    logger.info("call ended  duration_s=%.2f  chars=%d",
                                total_call_time, len(final_text or ""))
                    session.mark_call_end(outputs={
                        "text": final_text,
                        "stream_mode": metrics.get("stream_mode"),
                        "total_call_time_s": metrics.get("total_call_time_s"),
                    })

                    await websocket.send_json({
                        "type": "call_ended",
                        "final_text": final_text,
                        "metrics": metrics,
                    })
                    break

    except WebSocketDisconnect:
        logger.info("websocket disconnected")
        session.mark_call_end()
    finally:
        session.mark_call_end()
        reader.cancel()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)