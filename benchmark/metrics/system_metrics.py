# benchmark/metrics/system_metrics.py
"""
Background CPU/memory sampler using psutil.

Usage::

    with SystemSampler(interval_s=0.1) as sampler:
        # ... run inference ...
    report = sampler.result()
    print(report["peak_rss_mb"])
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import psutil

from benchmark.metrics.statistics import summarise


class SystemSampler:
    """
    Daemon-thread sampler that polls psutil every `interval_s` seconds.

    Context manager usage::

        with SystemSampler() as s:
            do_work()
        result = s.result()
    """

    def __init__(self, interval_s: float = 0.1):
        self._interval_s = interval_s
        self._proc = psutil.Process()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Raw sample lists
        self._overall_cpu: list[float] = []
        self._proc_cpu: list[float] = []
        self._rss_mb: list[float] = []
        self._threads: list[int] = []
        self._result: Optional[dict] = None

    def _sample_once(self) -> None:
        try:
            self._overall_cpu.append(psutil.cpu_percent(interval=None))
            self._proc_cpu.append(self._proc.cpu_percent(interval=None))
            mem = self._proc.memory_info()
            self._rss_mb.append(mem.rss / (1024 * 1024))
            self._threads.append(self._proc.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _sample_loop(self) -> None:
        # Event.wait() returns True as soon as stop is requested, so the thread
        # exits promptly, and each sample covers a full `interval_s` window
        # (psutil CPU% is measured since the previous call).
        while not self._stop_event.wait(self._interval_s):
            self._sample_once()

    def __enter__(self) -> "SystemSampler":
        self._stop_event.clear()
        self._overall_cpu, self._proc_cpu = [], []
        self._rss_mb, self._threads = [], []
        self._result = None
        # Prime the CPU counters *before* the work starts: the first
        # cpu_percent() call always returns a meaningless 0.0.
        psutil.cpu_percent(interval=None)
        self._proc.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        # Runs shorter than one interval would otherwise have no samples at all;
        # this sample then covers the whole run.
        if not self._rss_mb:
            self._sample_once()
        self._result = self._build_result()

    def _build_result(self) -> dict:
        def _safe_summarise(data: list) -> dict:
            if not data:
                return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0,
                        "min": 0.0, "max": 0.0, "count": 0}
            return summarise([float(x) for x in data])

        rss_floats = [float(x) for x in self._rss_mb] if self._rss_mb else [0.0]
        threads_ints = self._threads if self._threads else [0]

        return {
            "overall_cpu_pct":  _safe_summarise(self._overall_cpu),
            "process_cpu_pct":  _safe_summarise(self._proc_cpu),
            "rss_mb":           _safe_summarise(rss_floats),
            "thread_count":     _safe_summarise([float(t) for t in threads_ints]),
            "peak_rss_mb":      max(rss_floats),
            "peak_threads":     max(threads_ints),
        }

    def result(self) -> dict:
        """
        Return the collected metrics.
        If called before __exit__, builds from samples collected so far.
        """
        if self._result is not None:
            return self._result
        return self._build_result()
