# loadtest/runners/worker_pool.py
"""
Process layout for a load-test scenario: P worker processes, each

  * pinned to its own disjoint set of logical CPUs,
  * holding its own copy of the ASR engine (ORT thread pool sized to its CPU set),
  * running a slice of the simulated call legs in threads (the legs *within* a process share
    that process's engine and weights, exactly like the live server).

P = 1 reproduces the live server (one process, all legs share one engine). P > 1 is the
"one process per core group" strategy: more model memory (weights are duplicated per process)
but no cross-group thread contention and no shared-interpreter (GIL) bottleneck.

The parent only orchestrates and samples; it guards the machine against running out of RAM.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
import traceback
from typing import Any, Optional

import psutil

from src.core.runlog import configure_child_logging


class InsufficientMemory(RuntimeError):
    """The requested process layout does not fit in the RAM that is free right now."""
    def __init__(self, message: str, info: dict) -> None:
        super().__init__(message)
        self.info = info


class MemoryAbort(RuntimeError):
    """Free RAM fell below the hard floor during a level; the workers were killed."""


class WorkerCrashed(RuntimeError):
    """A worker process died (e.g. OOM-killed) or reported an error."""


# ── worker process ────────────────────────────────────────────────────────────

def _worker_main(conn: Any, cfg: dict) -> None:
    """Entry point of one worker process."""
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))
    try:
        if cfg.get("log_dir"):
            configure_child_logging(cfg["log_dir"], f"worker-{cfg.get('index', os.getpid())}")
        cpus = cfg["cpus"]
        if cpus:
            os.sched_setaffinity(0, cpus)

        from benchmark.engine_loader import load_engine
        from benchmark.runners.streaming_concurrency_runner import (
            StreamingConcurrencyRunner, _default_audio_loader,
        )
        from loadtest.audio import make_call_loader

        r = cfg["runner"]
        t0 = time.perf_counter()
        engine = load_engine(cfg["config_entry"])
        load_s = time.perf_counter() - t0

        runner = StreamingConcurrencyRunner(
            config_id=r["config_id"],
            engine_factory=lambda: engine,
            audio_files=r["audio_files"],
            legs_list=[],
            stream_mode=r["stream_mode"],
            streaming_cfg=r["streaming_cfg"],
            chunk_s=r["chunk_s"],
            lag_threshold_s=r["lag_threshold_s"],
            abort_lag_s=r["abort_lag_s"],
            audio_loader=make_call_loader(_default_audio_loader, r["call_duration_s"], r["gap_s"]),
        )
        runner.prepare(engine, len(r["audio_files"]))        # decode audio + un-timed warm-up pass

        conn.send(("ready", {
            "pid": os.getpid(),
            "load_s": load_s,
            "rss_mb": psutil.Process().memory_info().rss / (1024 * 1024),
            "cpus": cpus,
        }))

        while True:
            msg = conn.recv()
            if msg[0] == "stop":
                break
            if msg[0] == "level":
                _, n_legs, leg_offset, stagger_s, start_at = msg
                res = runner.run_level(engine, n_legs, stagger_s=stagger_s,
                                       leg_offset=leg_offset, start_at=start_at)
                conn.send(("result", res))
    except BaseException:                                   # report, then exit
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ── parent-side pool ──────────────────────────────────────────────────────────

class WorkerPool:
    """
    Args:
        config_entry:   bench-config entry (id, backend, model_config, optional overrides).
        cpu_groups:     one list of logical CPUs per worker process.
        runner_cfg:     dict with config_id, audio_files, stream_mode, streaming_cfg, chunk_s,
                        lag_threshold_s, abort_lag_s, call_duration_s, gap_s.
        reserve_mb:     RAM that must stay free for the rest of the system.
        hard_floor_mb:  if free RAM drops below this the workers are killed.
        startup_timeout_s: max time for one worker to load + warm up.
        log_dir:        parent run directory; each worker writes workers/worker-N.log.
    """

    def __init__(
        self,
        config_entry: dict,
        cpu_groups: list[list[int]],
        runner_cfg: dict,
        reserve_mb: float = 1500.0,
        hard_floor_mb: float = 600.0,
        startup_timeout_s: float = 600.0,
        log_dir: Optional[str] = None,
    ) -> None:
        self.config_entry = config_entry
        self.cpu_groups = cpu_groups
        self.runner_cfg = runner_cfg
        self.reserve_mb = reserve_mb
        self.hard_floor_mb = hard_floor_mb
        self.startup_timeout_s = startup_timeout_s
        self.log_dir = log_dir
        self._ctx = mp.get_context("spawn")
        self._procs: list[Any] = []
        self._conns: list[Any] = []
        self.ready_info: list[dict] = []

    # ── lifecycle ─────────────────────────────────────────────────────────

    @property
    def n_procs(self) -> int:
        return len(self.cpu_groups)

    @property
    def pids(self) -> list[int]:
        return [p.pid for p in self._procs]

    @property
    def cpus(self) -> list[int]:
        return sorted({c for g in self.cpu_groups for c in g})

    @staticmethod
    def _available_mb() -> float:
        return psutil.virtual_memory().available / (1024 * 1024)

    def _spawn(self, index: int) -> None:
        parent, child = self._ctx.Pipe()
        cfg = {"index": index, "cpus": self.cpu_groups[index],
               "config_entry": self.config_entry, "runner": self.runner_cfg,
               "log_dir": self.log_dir}
        proc = self._ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
        proc.start()
        child.close()
        self._procs.append(proc)
        self._conns.append(parent)

    def _recv(self, i: int, deadline: float) -> tuple:
        """Wait for worker ``i``'s next message, watching for crashes, timeouts and low RAM."""
        conn, proc = self._conns[i], self._procs[i]
        while True:
            if conn.poll(0.5):
                try:
                    return conn.recv()
                except EOFError:
                    raise WorkerCrashed(f"worker {i} closed its pipe (exit code {proc.exitcode})")
            if not proc.is_alive() and not conn.poll(0):
                raise WorkerCrashed(f"worker {i} died (exit code {proc.exitcode}); likely out of memory")
            if self._available_mb() < self.hard_floor_mb:
                self.close(kill=True)
                raise MemoryAbort(f"free RAM fell below {self.hard_floor_mb:.0f} MB; workers killed")
            if time.time() > deadline:
                self.close(kill=True)
                raise WorkerCrashed(f"worker {i} timed out")

    def _await_ready(self, i: int) -> dict:
        kind, payload = self._recv(i, time.time() + self.startup_timeout_s)
        if kind != "ready":
            self.close(kill=True)
            raise WorkerCrashed(f"worker {i} failed to start:\n{payload}")
        return payload

    def start(self) -> list[dict]:
        """
        Start every worker. The first one is started alone so its real RAM footprint
        (after load + warm-up) can be checked against free RAM before the rest are launched.
        """
        self._spawn(0)
        info0 = self._await_ready(0)
        self.ready_info = [info0]
        extra = (self.n_procs - 1) * info0["rss_mb"]
        avail = self._available_mb()
        if extra > avail - self.reserve_mb:
            self.close(kill=True)
            raise InsufficientMemory(
                f"{self.n_procs} processes need ~{info0['rss_mb'] * self.n_procs:.0f} MB "
                f"({info0['rss_mb']:.0f} MB each) but only {avail:.0f} MB is free "
                f"(reserve {self.reserve_mb:.0f} MB)",
                {"per_process_rss_mb": info0["rss_mb"], "available_mb": avail,
                 "load_s": info0["load_s"], "processes": self.n_procs},
            )
        for i in range(1, self.n_procs):
            self._spawn(i)
        for i in range(1, self.n_procs):
            self.ready_info.append(self._await_ready(i))
        return self.ready_info

    def close(self, kill: bool = False) -> None:
        for conn in self._conns:
            try:
                if not kill:
                    conn.send(("stop",))
            except Exception:
                pass
        for proc in self._procs:
            try:
                if kill:
                    proc.kill()
                else:
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()
                proc.join(timeout=5)
            except Exception:
                pass
        for conn in self._conns:
            try:
                conn.close()
            except Exception:
                pass
        self._procs, self._conns = [], []

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ── running a level ───────────────────────────────────────────────────

    def split_legs(self, n_legs: int) -> list[int]:
        """Legs per process (as even as possible; earlier processes take the remainder)."""
        base, rem = divmod(n_legs, self.n_procs)
        return [base + (1 if i < rem else 0) for i in range(self.n_procs)]

    def run_level(self, n_legs: int, stagger_s: float, timeout_s: float, lead_s: float = 1.0) -> list:
        """
        Run ``n_legs`` simultaneous legs split across the workers; returns one
        ``StreamingConcurrencyResult`` per worker that got at least one leg.
        All workers start together (``lead_s`` from now) and leg start times are staggered
        across the *whole* level, not per process.
        """
        counts = self.split_legs(n_legs)
        start_at = time.time() + lead_s
        offset = 0
        active: list[int] = []
        for i, c in enumerate(counts):
            if c > 0:
                self._conns[i].send(("level", c, offset, stagger_s, start_at))
                active.append(i)
            offset += c
        deadline = time.time() + lead_s + timeout_s
        results = []
        for i in active:
            kind, payload = self._recv(i, deadline)
            if kind != "result":
                self.close(kill=True)
                raise WorkerCrashed(f"worker {i} failed during a level:\n{payload}")
            results.append(payload)
        return results
