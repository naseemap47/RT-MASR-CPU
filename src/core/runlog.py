"""
Per-run logging for RT-MASR-CPU.

Every CLI / pipeline invocation creates a directory::

    logs/<pipeline>/<UTC stamp>/
        run.log          # stdout, stderr, and formatted log records
        run.meta.json    # argv, pid, timing, exit code, artifact paths
        workers/         # child-process captures (load-test workers, model checks)

The UTC stamp matches the one used for benchmark / load-test result files, so a
run and its reports can be joined by name.

Call :func:`start_run` from a CLI entry point (context manager) or
:func:`begin_run` from a long-lived process such as the FastAPI lifespan.
Library code just uses ``logging.getLogger("rtmasr....")``; it does not create
files. Pytest does not auto-start a session.

Environment:

    RT_MASR_LOG_DIR     override the logs root (default: <project>/logs)
    RT_MASR_LOG_LEVEL   DEBUG / INFO / WARNING / ERROR (default: INFO)
    RT_MASR_NO_LOG=1    disable file capture (console logging still works if
                        the caller configures it)
"""
from __future__ import annotations

import json
import logging
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

_LOG_FORMAT = "%(asctime)s %(levelname)-5s [%(name)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"

_lock = threading.RLock()
_active: Optional["RunSession"] = None


def utc_stamp(when: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ`` — same clock the reporters use for result filenames."""
    dt = when or datetime.now(tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logs_root() -> Path:
    env = os.environ.get("RT_MASR_LOG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "logs"


def in_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ


def logging_disabled() -> bool:
    return os.environ.get("RT_MASR_NO_LOG", "").strip() in ("1", "true", "yes")


def parse_level(value: str | None, default: int = logging.INFO) -> int:
    if not value:
        return default
    if isinstance(value, int):
        return value
    name = str(value).strip().upper()
    return getattr(logging, name, default)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("rtmasr"):
        name = f"rtmasr.{name}"
    return logging.getLogger(name)


def current_run() -> Optional["RunSession"]:
    return _active


class _Tee:
    """Write to the original stream and to the run log file."""

    def __init__(self, stream, log_file) -> None:
        self._stream = stream
        self._log = log_file
        self._lock = threading.Lock()

    def write(self, data) -> int:
        if not data:
            return 0
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = data
        with self._lock:
            self._stream.write(text)
            try:
                self._log.write(text)
                self._log.flush()
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        with self._lock:
            try:
                self._stream.flush()
            except Exception:
                pass
            try:
                self._log.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except Exception:
            return False

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self):
        return getattr(self._stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._stream, "errors", "replace")

    def __getattr__(self, name: str):
        return getattr(self._stream, name)


def _install_root_logger(level: int, stream) -> logging.StreamHandler:
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)
    handler.set_name("rtmasr-run")
    # Drop a previous run's handler so repeat begin_run() in one process is clean.
    root.handlers = [h for h in root.handlers if getattr(h, "name", None) != "rtmasr-run"]
    root.addHandler(handler)
    logging.captureWarnings(True)
    return handler


class RunSession:
    """One captured invocation: directory, log file, metadata."""

    def __init__(
        self,
        pipeline: str,
        *,
        stamp: str | None = None,
        level: int | str | None = None,
        capture_stdio: bool = True,
    ) -> None:
        self.pipeline = pipeline
        self.stamp = stamp or utc_stamp()
        self.level = parse_level(
            str(level) if not isinstance(level, int) and level is not None else None,
            default=parse_level(os.environ.get("RT_MASR_LOG_LEVEL")),
        )
        if isinstance(level, int):
            self.level = level
        self.capture_stdio = capture_stdio
        self.dir = logs_root() / pipeline / self.stamp
        self.log_path = self.dir / "run.log"
        self.meta_path = self.dir / "run.meta.json"
        self.started_at = ""
        self.finished_at = ""
        self.exit_code: int | None = None
        self.artifacts: dict[str, str] = {}
        self._t0 = 0.0
        self._log_fh: Any = None
        self._stdout_orig = None
        self._stderr_orig = None
        self._handler: logging.StreamHandler | None = None
        self._started = False
        self._closed = False
        self._old_excepthook = None
        self._root_level_before: int | None = None

    @property
    def workers_dir(self) -> Path:
        return self.dir / "workers"

    @property
    def closed(self) -> bool:
        return self._closed

    def start(self) -> "RunSession":
        global _active
        with _lock:
            if self._started:
                return self
            self.dir.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8", buffering=1)
            self.started_at = utc_now_iso()
            self._t0 = time.perf_counter()

            if self.capture_stdio:
                self._stdout_orig = sys.stdout
                self._stderr_orig = sys.stderr
                sys.stdout = _Tee(sys.stdout, self._log_fh)
                sys.stderr = _Tee(sys.stderr, self._log_fh)

            root = logging.getLogger()
            self._root_level_before = root.level
            self._handler = _install_root_logger(self.level, sys.stdout)
            self._old_excepthook = sys.excepthook
            sys.excepthook = self._excepthook
            self._write_meta(partial=True)
            self._point_latest()
            self._started = True
            _active = self

        logging.getLogger("rtmasr").info(
            "run started  pipeline=%s  stamp=%s  log=%s",
            self.pipeline, self.stamp, self.log_path,
        )
        return self

    def note_artifact(self, kind: str, path: str | Path) -> None:
        self.artifacts[kind] = str(Path(path).resolve())

    def close(self, exit_code: int = 0) -> None:
        global _active
        with _lock:
            if self._closed or not self._started:
                self._closed = True
                if _active is self:
                    _active = None
                return
            self.exit_code = exit_code
            self.finished_at = utc_now_iso()
            duration = time.perf_counter() - self._t0
            logging.getLogger("rtmasr").info(
                "run finished  pipeline=%s  stamp=%s  duration_s=%.2f  exit=%s",
                self.pipeline, self.stamp, duration, exit_code,
            )
            if self._handler is not None:
                root = logging.getLogger()
                root.removeHandler(self._handler)
                if self._root_level_before is not None:
                    root.setLevel(self._root_level_before)
                self._handler.close()
                self._handler = None
            if self._old_excepthook is not None:
                sys.excepthook = self._old_excepthook
                self._old_excepthook = None
            if self._stdout_orig is not None:
                sys.stdout = self._stdout_orig
                self._stdout_orig = None
            if self._stderr_orig is not None:
                sys.stderr = self._stderr_orig
                self._stderr_orig = None
            self._write_meta(partial=False, duration_s=duration)
            if self._log_fh is not None:
                try:
                    self._log_fh.flush()
                    self._log_fh.close()
                except Exception:
                    pass
                self._log_fh = None
            self._closed = True
            if _active is self:
                _active = None

    def _excepthook(self, exc_type, exc, tb) -> None:
        logging.getLogger("rtmasr").error(
            "unhandled exception", exc_info=(exc_type, exc, tb),
        )
        if self._old_excepthook is not None:
            self._old_excepthook(exc_type, exc, tb)

    def _point_latest(self) -> None:
        latest = self.dir.parent / "latest"
        try:
            if latest.is_symlink() or latest.is_file():
                latest.unlink()
            elif latest.exists():
                return
            latest.symlink_to(self.stamp)
        except OSError:
            pass

    def _write_meta(self, *, partial: bool, duration_s: float | None = None) -> None:
        meta = {
            "pipeline": self.pipeline,
            "stamp": self.stamp,
            "started_at": self.started_at,
            "finished_at": self.finished_at or None,
            "duration_s": None if duration_s is None else round(duration_s, 3),
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "python": sys.version.split()[0],
            "hostname": platform.node(),
            "log_file": str(self.log_path),
            "exit_code": self.exit_code,
            "artifacts": dict(self.artifacts),
        }
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.meta_path)

    def __enter__(self) -> "RunSession":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        code = 0
        if exc_type is SystemExit:
            raw = getattr(exc, "code", 0)
            if raw is None:
                code = 0
            elif isinstance(raw, int):
                code = raw
            else:
                code = 1
        elif exc_type is not None:
            code = 1
        self.close(exit_code=code)
        return None


def begin_run(
    pipeline: str,
    *,
    stamp: str | None = None,
    level: int | str | None = None,
    capture_stdio: bool = True,
    skip_if_pytest: bool = False,
) -> Optional[RunSession]:
    """
    Start a run session. The caller must :meth:`RunSession.close` it.

    Returns ``None`` when logging is disabled, or when ``skip_if_pytest`` is
    set and the process is inside pytest (so TestClient does not write logs).
    """
    if logging_disabled():
        return None
    if skip_if_pytest and in_pytest():
        return None
    with _lock:
        if _active is not None:
            return _active
    return RunSession(
        pipeline, stamp=stamp, level=level, capture_stdio=capture_stdio,
    ).start()


@contextmanager
def start_run(
    pipeline: str,
    *,
    stamp: str | None = None,
    level: int | str | None = None,
    capture_stdio: bool = True,
) -> Iterator[Optional[RunSession]]:
    """Context manager for CLI entry points. Always closes, including on SystemExit."""
    session = begin_run(
        pipeline, stamp=stamp, level=level, capture_stdio=capture_stdio,
    )
    code = 0
    try:
        yield session
    except SystemExit as exc:
        raw = exc.code
        if raw is None:
            code = 0
        elif isinstance(raw, int):
            code = raw
        else:
            code = 1
        raise
    except BaseException:
        code = 1
        raise
    finally:
        if session is not None:
            session.close(exit_code=code)


def configure_child_logging(run_dir: str | Path, name: str) -> Path:
    """
    Capture a spawned worker's stdout/stderr into ``<run_dir>/workers/<name>.log``.

    Used by load-test worker processes (fresh interpreter, inherited fds only
    hit the terminal). Parent run.log does not see these writes.
    """
    workers = Path(run_dir) / "workers"
    workers.mkdir(parents=True, exist_ok=True)
    path = workers / f"{name}.log"
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.stdout, fh)
    sys.stderr = _Tee(sys.stderr, fh)
    level = parse_level(os.environ.get("RT_MASR_LOG_LEVEL"))
    _install_root_logger(level, sys.stdout)
    logging.getLogger("rtmasr").info("worker log attached  name=%s  file=%s", name, path)
    return path
