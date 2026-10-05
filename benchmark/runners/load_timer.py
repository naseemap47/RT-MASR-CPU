# benchmark/runners/load_timer.py
"""
Cold-start / model-load timing measurement.

Measures wall-clock load time and RSS memory delta for an engine constructor,
keeping load measurement completely separate from steady-state inference.

Note: "cold" means a cold *process-level* start (fresh Python objects, fresh ORT
sessions). Model files are usually already in the OS page cache after the first
benchmark run, so disk-read time is not included on repeat runs.
"""
from __future__ import annotations

import ctypes
import gc
import time
from dataclasses import dataclass
from typing import Any, Callable

import psutil


@dataclass
class LoadResult:
    """Result of a single model-load timing measurement."""
    config_id: str
    load_time_s: float
    rss_before_mb: float
    rss_after_mb: float
    rss_delta_mb: float


def release_memory() -> None:
    """
    Run the GC and ask glibc to return freed heap pages to the OS.

    Without this, RSS stays high after a previous engine is freed and the next
    engine's RSS delta is under-reported (it re-uses already-mapped memory).
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # not glibc (e.g. macOS / musl): best effort only


def measure_load_with_engine(
    config_id: str,
    engine_factory: Callable[[], Any],
) -> tuple[LoadResult, Any]:
    """
    Measure cold-start load time and RSS delta, and return the loaded engine so
    the caller can reuse it (avoids loading the model a second time).

    Steps:
      1. GC + malloc_trim + 500 ms settle to get a stable baseline RSS.
      2. Record RSS_before.
      3. Call engine_factory() and time it.
      4. Record RSS_after.

    The caller must drop all references to previously loaded engines *before*
    calling this, otherwise RSS_before includes them.
    """
    proc = psutil.Process()

    release_memory()
    time.sleep(0.5)

    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    engine = engine_factory()
    load_time_s = time.perf_counter() - t0

    rss_after_mb = proc.memory_info().rss / (1024 * 1024)

    result = LoadResult(
        config_id=config_id,
        load_time_s=load_time_s,
        rss_before_mb=rss_before_mb,
        rss_after_mb=rss_after_mb,
        rss_delta_mb=rss_after_mb - rss_before_mb,
    )
    return result, engine


def measure_load(
    config_id: str,
    engine_factory: Callable[[], Any],
) -> LoadResult:
    """Like :func:`measure_load_with_engine` but discards the engine."""
    result, engine = measure_load_with_engine(config_id, engine_factory)
    del engine
    return result
