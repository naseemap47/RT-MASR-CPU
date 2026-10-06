# loadtest/metrics/tree_sampler.py
"""
Resource sampler for a set of worker processes (the "tree" under test).

Per load level it reports:
  * CPU utilisation of the CPUs the workers may use (the pinned set, not the whole box)
  * CPU time actually consumed by the workers (user + system) -> core-seconds per audio-second
  * summed RSS of the workers (what the deployment would need in RAM)
  * lowest system-wide available memory seen (OOM early-warning)
"""
from __future__ import annotations

import threading
from typing import Optional

import psutil

from benchmark.metrics.statistics import summarise


def _stats(values: list[float]) -> dict:
    return summarise(values) if values else {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                                             "min": 0.0, "max": 0.0, "count": 0}


class TreeSampler:
    """
    Context manager sampling ``pids`` every ``interval_s`` seconds.

    Args:
        pids:       worker process ids.
        cpus:       logical CPUs the workers are pinned to (None = all CPUs).
        interval_s: sampling period.
    """

    def __init__(self, pids: list[int], cpus: Optional[list[int]] = None, interval_s: float = 0.25) -> None:
        self.pids = list(pids)
        self.cpus = list(cpus) if cpus else None
        self.interval_s = interval_s
        self._procs = [p for p in (self._proc(pid) for pid in self.pids) if p is not None]
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cpu: list[float] = []
        self._rss: list[float] = []
        self._avail: list[float] = []
        self._threads: list[int] = []
        self._cpu_t0 = 0.0
        self._cpu_t1 = 0.0
        self._wall = 0.0
        self._result: Optional[dict] = None

    @staticmethod
    def _proc(pid: int) -> Optional[psutil.Process]:
        try:
            return psutil.Process(pid)
        except psutil.NoSuchProcess:
            return None

    def _cpu_time(self) -> float:
        total = 0.0
        for p in self._procs:
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total

    def _sample(self) -> None:
        per_cpu = psutil.cpu_percent(interval=None, percpu=True)
        sel = [per_cpu[c] for c in self.cpus if c < len(per_cpu)] if self.cpus else per_cpu
        if sel:
            self._cpu.append(sum(sel) / len(sel))
        rss, thr = 0.0, 0
        for p in self._procs:
            try:
                rss += p.memory_info().rss / (1024 * 1024)
                thr += p.num_threads()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._rss.append(rss)
        self._threads.append(thr)
        self._avail.append(psutil.virtual_memory().available / (1024 * 1024))

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> "TreeSampler":
        import time
        self._stop.clear()
        psutil.cpu_percent(interval=None, percpu=True)      # prime the counters
        self._cpu_t0 = self._cpu_time()
        self._wall = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        import time
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._wall = time.perf_counter() - self._wall
        self._cpu_t1 = self._cpu_time()
        if not self._rss:
            self._sample()
        rss = self._rss or [0.0]
        cpu_s = max(0.0, self._cpu_t1 - self._cpu_t0)
        self._result = {
            "duration_s": self._wall,
            "cpu_pct": _stats(self._cpu),                 # mean over the pinned CPUs
            "tree_cpu_s": cpu_s,                          # user+system seconds used by the workers
            "cores_used": cpu_s / self._wall if self._wall > 0 else 0.0,
            "tree_rss_mb_mean": sum(rss) / len(rss),
            "tree_rss_mb_peak": max(rss),
            "mem_available_min_mb": min(self._avail) if self._avail else 0.0,
            "peak_threads": max(self._threads) if self._threads else 0,
        }

    def result(self) -> dict:
        return self._result if self._result is not None else {}
