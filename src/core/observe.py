"""
AI observability traces (LangSmith-style, log-only — no UI).

Each model inference is a *run* (a span). Nested runs share a *trace_id*
(a live call, a load-test leg). Records are JSONL, one object per line::

    logs/<pipeline>/<UTC>/traces.jsonl

A run looks like::

    {
      "event": "run",
      "id": "...", "trace_id": "...", "parent_id": "...",
      "name": "transcribe_stream", "run_type": "llm",
      "status": "ok",
      "start_time": "2026-10-07T18:34:02.123Z",
      "end_time":   "...",
      "latency_ms": 412.3,
      "inputs":  {"audio_s": 2.0, "samples": 32000, "language": "en"},
      "outputs": {"text": "...", "language": "en", "tokens": 12},
      "error": null,
      "model": {"name": "qwen3_onnx_0.6b_int8", "backend": "onnx"},
      "metrics": {"rtf": 0.21, "encoder_s": 0.05, "prefill_s": 0.08, "decode_s": 0.27},
      "tags": ["call", "live"]
    }

Wired in automatically when a :class:`~src.core.runlog.RunSession` is active.
Disable with ``RT_MASR_OBSERVE=0``. No-op in pytest (no session).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

_TEXT_CAP = 4000
_METRIC_KEYS = (
    "rtf", "total_s", "audio_duration_s", "tokens_generated",
    "mel_s", "encoder_s", "prepare_s", "prefill_s", "decode_s",
    "load_s", "other_s", "segments", "sub_chunks", "window_s",
)

_trace_id: ContextVar[Optional[str]] = ContextVar("obs_trace_id", default=None)
_parent_id: ContextVar[Optional[str]] = ContextVar("obs_parent_id", default=None)
_model: ContextVar[Optional[dict]] = ContextVar("obs_model", default=None)
_extra: ContextVar[dict] = ContextVar("obs_extra", default={})
_tags: ContextVar[tuple] = ContextVar("obs_tags", default=())

_writer_lock = threading.Lock()
_writer: Optional["_FileWriter"] = None


def _observe_disabled() -> bool:
    return os.environ.get("RT_MASR_OBSERVE", "1").strip().lower() in ("0", "false", "no")


def enabled() -> bool:
    return _writer is not None and not _observe_disabled()


def utc_ms() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _new_id() -> str:
    return uuid.uuid4().hex


def _clip_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= _TEXT_CAP:
        return value
    return value[:_TEXT_CAP] + f"...[+{len(value) - _TEXT_CAP} chars]"


def _clean(obj: Any, *, drop_none: bool = True) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj in (float("inf"), float("-inf")):  # noqa: PLR0124
            return None
        return round(obj, 6)
    if isinstance(obj, dict):
        # Keep top-level schema keys (error, parent_id, …) even when null.
        return {
            str(k): _clean(v)
            for k, v in obj.items()
            if (not drop_none) or v is not None
        }
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return _clean(obj.item())
        except Exception:
            return str(obj)
    return str(obj)


class _FileWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self, record: dict) -> None:
        line = json.dumps(_clean(record, drop_none=False), ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass


def attach_file(path: str | Path) -> Path:
    """Send subsequent runs to ``path`` (JSONL). Replaces any previous writer."""
    global _writer
    path = Path(path)
    with _writer_lock:
        if _observe_disabled():
            return path
        if _writer is not None:
            _writer.close()
        _writer = _FileWriter(path)
        return path


def detach() -> None:
    global _writer
    with _writer_lock:
        if _writer is not None:
            _writer.close()
            _writer = None


def _emit(record: dict) -> None:
    w = _writer
    if w is None or _observe_disabled():
        return
    try:
        w.write(record)
    except Exception:
        pass


def annotate_engine(engine: Any, *, name: str | None, backend: str | None = None,
                    display_name: str | None = None, **extra: Any) -> None:
    """Stamp model identity on an engine (and its ``.pipeline``) for later runs."""
    meta = {k: v for k, v in {
        "name": name, "backend": backend, "display_name": display_name, **extra,
    }.items() if v is not None}
    if not meta:
        return
    engine._observe_model = {**getattr(engine, "_observe_model", {}), **meta}
    pipe = getattr(engine, "pipeline", None)
    if pipe is not None:
        pipe._observe_model = {**getattr(pipe, "_observe_model", {}), **meta}


def annotate_from_config(engine: Any, cfg: dict, **extra: Any) -> None:
    eng = cfg.get("engine") or {}
    annotate_engine(
        engine,
        name=cfg.get("name"),
        backend=cfg.get("backend"),
        display_name=cfg.get("display_name"),
        quantize=eng.get("quantize") or eng.get("precision"),
        **extra,
    )


def use_engine(engine: Any) -> None:
    meta = getattr(engine, "_observe_model", None)
    if meta:
        _model.set(dict(meta))


def audio_inputs(audio: Any, *, language: Any = None, **extra: Any) -> dict:
    """Describe audio without storing the waveform."""
    out: dict[str, Any] = dict(extra)
    if language not in (None, ""):
        out["language"] = language
    if isinstance(audio, (str, Path)):
        out["audio_path"] = str(audio)
        return out
    try:
        import numpy as np
        if isinstance(audio, np.ndarray) and audio.size:
            n = int(audio.shape[-1])
            out["samples"] = n
            out["audio_s"] = round(n / 16000.0, 3)
    except Exception:
        pass
    return out


@contextmanager
def bound(**extra: Any) -> Generator[None, None, None]:
    """Merge extra fields into every run started inside this block."""
    cur = dict(_extra.get() or {})
    cur.update({k: v for k, v in extra.items() if v is not None})
    token = _extra.set(cur)
    try:
        yield
    finally:
        _extra.reset(token)


class Span:
    """One observability run. Written to JSONL on :meth:`finish`."""

    def __init__(self, name: str, run_type: str = "llm") -> None:
        self.id = _new_id()
        self.name = name
        self.run_type = run_type
        self.trace_id = _trace_id.get() or self.id
        self.parent_id = _parent_id.get()
        self.start_time = utc_ms()
        self._t0 = time.perf_counter()
        self.inputs: dict[str, Any] = {}
        self.outputs: dict[str, Any] = {}
        self.metrics: dict[str, Any] = {}
        self.extra: dict[str, Any] = dict(_extra.get() or {})
        self.tags: list[str] = list(_tags.get() or ())
        self.model: dict[str, Any] = dict(_model.get() or {})
        self.error: str | None = None
        self.status = "ok"
        self.finished = False
        self._parent_token: Token | None = None
        self._trace_token: Token | None = None

    def _push(self, *, as_trace: bool) -> None:
        self._parent_token = _parent_id.set(self.id)
        if as_trace or not _trace_id.get():
            self.trace_id = self.id
            self._trace_token = _trace_id.set(self.id)

    def _pop(self) -> None:
        if self._parent_token is not None:
            _parent_id.reset(self._parent_token)
            self._parent_token = None
        if self._trace_token is not None:
            _trace_id.reset(self._trace_token)
            self._trace_token = None

    def set_inputs(self, **fields: Any) -> None:
        self.inputs.update({k: v for k, v in fields.items() if v is not None})

    def set_outputs(self, **fields: Any) -> None:
        for k, v in fields.items():
            if v is None:
                continue
            self.outputs[k] = _clip_text(v) if k in ("text", "committed", "tentative") else v

    def set_metrics(self, timing: dict | None) -> None:
        if not timing:
            return
        for k in _METRIC_KEYS:
            if k in timing and timing[k] is not None:
                self.metrics[k] = timing[k]

    def finish(self, error: BaseException | str | None = None,
               outputs: dict | None = None) -> None:
        if self.finished:
            return
        self.finished = True
        if outputs:
            self.set_outputs(**outputs)
        if error is not None:
            self.status = "error"
            self.error = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        latency_ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
        _emit({
            "event": "run",
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "run_type": self.run_type,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": utc_ms(),
            "latency_ms": latency_ms,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "model": self.model or None,
            "metrics": self.metrics,
            "tags": self.tags,
            "extra": self.extra or None,
        })
        self._pop()

    def __enter__(self) -> "Span":
        if enabled():
            self._push(as_trace=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finish(error=exc if exc_type else None)
        return None


def span(name: str, run_type: str = "llm", *, inputs: dict | None = None,
         tags: Iterable[str] | None = None, extra: dict | None = None) -> Span:
    """Open a nested run (context manager). Parent is the currently open span."""
    s = Span(name, run_type)
    if inputs:
        s.set_inputs(**inputs)
    if extra:
        s.extra.update(extra)
    if tags:
        s.tags.extend(str(t) for t in tags if t)
    return s


def start_trace(name: str = "call", *, run_type: str = "chain",
                inputs: dict | None = None, tags: Iterable[str] | None = None,
                extra: dict | None = None) -> Span:
    """Open a root chain (live call / load-test leg). Nested inferences share its id."""
    s = Span(name, run_type)
    if inputs:
        s.set_inputs(**inputs)
    if extra:
        s.extra.update(extra)
    if tags:
        s.tags.extend(str(t) for t in tags if t)
    if enabled():
        s._push(as_trace=True)
    return s


def record_asr(sp: Span, result: dict | None) -> None:
    """Copy a standard engine result dict onto a span."""
    if not result:
        return
    timing = result.get("timing") or {}
    sp.set_outputs(
        text=result.get("text", ""),
        language=result.get("language") or None,
        tokens=timing.get("tokens_generated"),
    )
    sp.set_metrics(timing)


def trace_stream(name: str, inner: Iterable, *, inputs: dict | None = None,
                 tags: Iterable[str] | None = None, extra: dict | None = None
                 ) -> Generator[tuple, None, None]:
    """Wrap a ``(delta, timing)`` generator as one llm run."""
    sp = span(name, "llm", inputs=inputs, tags=tags, extra=extra)
    sp.__enter__()
    chunks: list[str] = []
    timing: dict | None = None
    try:
        for delta, t in inner:
            if delta:
                chunks.append(delta)
            if t is not None:
                timing = t
            yield delta, t
        text = "".join(chunks)
        sp.set_outputs(text=text, tokens=(timing or {}).get("tokens_generated"))
        if timing:
            sp.set_metrics(timing)
            if timing.get("audio_duration_s") and "audio_s" not in sp.inputs:
                sp.set_inputs(audio_s=timing["audio_duration_s"])
        sp.finish()
    except BaseException as exc:
        sp.set_outputs(text="".join(chunks))
        sp.finish(error=exc)
        raise
