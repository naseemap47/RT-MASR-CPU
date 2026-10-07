"""Run-session logging: files, stdio capture, metadata, pytest skip."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

from src.core import runlog


@pytest.fixture
def log_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RT_MASR_LOG_DIR", str(tmp_path))
    monkeypatch.delenv("RT_MASR_NO_LOG", raising=False)
    # Isolate from a session this test process might have started.
    runlog._active = None
    yield tmp_path
    if runlog._active is not None:
        runlog._active.close()
        runlog._active = None


def test_start_run_writes_log_and_meta(log_root):
    with runlog.start_run("benchmark") as session:
        assert session is not None
        print("hello-stdout")
        logging.getLogger("rtmasr.benchmark").info("hello-logger")
        session.note_artifact("raw_json", log_root / "fake.json")

    run_dir = log_root / "benchmark" / session.stamp
    log_text = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "hello-stdout" in log_text
    assert "hello-logger" in log_text
    assert "[rtmasr.benchmark]" in log_text

    meta = json.loads((run_dir / "run.meta.json").read_text(encoding="utf-8"))
    assert meta["pipeline"] == "benchmark"
    assert meta["stamp"] == session.stamp
    assert meta["exit_code"] == 0
    assert meta["duration_s"] is not None
    assert "raw_json" in meta["artifacts"]
    latest = log_root / "benchmark" / "latest"
    assert latest.is_symlink()
    assert os.readlink(latest) == session.stamp


def test_start_run_records_systemexit_code(log_root):
    session = None
    with pytest.raises(SystemExit) as ei:
        with runlog.start_run("sizing") as session:
            raise SystemExit(2)
    assert ei.value.code == 2
    meta = json.loads(session.meta_path.read_text(encoding="utf-8"))
    assert meta["exit_code"] == 2


def test_stdio_restored_after_close(log_root):
    orig = sys.stdout
    with runlog.start_run("download"):
        assert sys.stdout is not orig
    assert sys.stdout is orig


def test_begin_run_skips_inside_pytest_when_asked(log_root, monkeypatch):
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_runlog.py::test")
    session = runlog.begin_run("server", skip_if_pytest=True)
    assert session is None
    assert not list(Path(log_root).rglob("run.log"))


def test_no_log_env_disables_files(log_root, monkeypatch):
    monkeypatch.setenv("RT_MASR_NO_LOG", "1")
    with runlog.start_run("loadtest") as session:
        assert session is None
    assert not list(Path(log_root).rglob("run.log"))


def test_utc_stamp_format():
    s = runlog.utc_stamp()
    assert len(s) == 16 and s.endswith("Z") and s[8] == "T"
    assert s[:8].isdigit() and s[9:15].isdigit()
