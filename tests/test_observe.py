"""LangSmith-style inference traces (JSONL, no UI)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core import observe, runlog


@pytest.fixture
def traces(tmp_path, monkeypatch):
    monkeypatch.delenv("RT_MASR_OBSERVE", raising=False)
    observe.detach()
    path = tmp_path / "traces.jsonl"
    observe.attach_file(path)
    yield path
    observe.detach()


def _read(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


def test_span_writes_llm_run(traces):
    with observe.span("transcribe", "llm", inputs={"audio_s": 1.5, "language": "en"}) as sp:
        sp.set_outputs(text="hello world", language="en", tokens=2)
        sp.set_metrics({"rtf": 0.2, "encoder_s": 0.05, "decode_s": 0.1, "tokens_generated": 2})

    rows = _read(traces)
    assert len(rows) == 1
    r = rows[0]
    assert r["event"] == "run"
    assert r["name"] == "transcribe" and r["run_type"] == "llm"
    assert r["status"] == "ok" and r["error"] is None
    assert r["inputs"]["audio_s"] == 1.5
    assert r["outputs"]["text"] == "hello world"
    assert r["outputs"]["tokens"] == 2
    assert r["metrics"]["rtf"] == 0.2
    assert r["latency_ms"] >= 0
    assert r["id"] and r["trace_id"]
    assert r["parent_id"] is None


def test_call_trace_nests_inference(traces):
    call = observe.start_trace("call", tags=["call"])
    with observe.span("transcribe_stream", inputs={"samples": 8000}) as sp:
        sp.set_outputs(text="hi")
    call.finish(outputs={"text": "hi there"})

    rows = _read(traces)
    assert [r["name"] for r in rows] == ["transcribe_stream", "call"]
    child, parent = rows
    assert child["parent_id"] == parent["id"]
    assert child["trace_id"] == parent["id"] == parent["trace_id"]
    assert parent["run_type"] == "chain"
    assert "call" in parent["tags"]


def test_error_is_recorded(traces):
    with pytest.raises(RuntimeError):
        with observe.span("transcribe"):
            raise RuntimeError("boom")
    rows = _read(traces)
    assert rows[0]["status"] == "error"
    assert "boom" in rows[0]["error"]


def test_trace_stream(traces):
    def gen():
        yield "hel", None
        yield "lo", None
        yield "", {"tokens_generated": 2, "rtf": 0.3, "audio_duration_s": 1.0}

    out = list(observe.trace_stream("transcribe_stream", gen(), inputs={"audio_s": 1.0}))
    assert out[0][0] == "hel" and out[-1][1]["rtf"] == 0.3
    r = _read(traces)[0]
    assert r["outputs"]["text"] == "hello"
    assert r["outputs"]["tokens"] == 2
    assert r["metrics"]["rtf"] == 0.3


def test_audio_inputs_does_not_store_waveform():
    wav = np.zeros(16000, dtype=np.float32)
    d = observe.audio_inputs(wav, language="en")
    assert d["samples"] == 16000 and d["audio_s"] == 1.0
    assert d["language"] == "en"
    assert "audio" not in d


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_MASR_OBSERVE", "0")
    observe.detach()
    observe.attach_file(tmp_path / "traces.jsonl")
    with observe.span("transcribe") as sp:
        sp.set_outputs(text="nope")
    observe.detach()
    assert not (tmp_path / "traces.jsonl").exists() or (tmp_path / "traces.jsonl").stat().st_size == 0


def test_start_run_creates_traces_file(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_MASR_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("RT_MASR_NO_LOG", raising=False)
    monkeypatch.delenv("RT_MASR_OBSERVE", raising=False)
    runlog._active = None
    observe.detach()
    with runlog.start_run("benchmark") as session:
        assert session is not None
        with observe.span("transcribe") as sp:
            sp.set_outputs(text="ok")
    rows = _read(session.traces_path)
    assert len(rows) == 1
    meta = json.loads(session.meta_path.read_text(encoding="utf-8"))
    assert "traces" in meta["artifacts"]
    if runlog._active is not None:
        runlog._active.close()
        runlog._active = None
    observe.detach()
