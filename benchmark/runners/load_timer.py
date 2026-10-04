# benchmark/runners/load_timer.py
"""
Cold-start / model-load timing measurement.

Measures wall-clock load time and RSS memory delta for an engine constructor,
keeping load measurement completely separate from steady-state inference.
"""
from __future__ import annotations

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


def measure_load(
    config_id: str,
    engine_factory: Callable[[], Any],
) -> LoadResult:
    """
    Measure cold-start load time and RSS memory delta for an engine constructor.

    Steps:
      1. Force GC + 500 ms settle to get a stable baseline RSS.
      2. Record RSS_before.
      3. Call engine_factory() and time it.
      4. Record RSS_after.

    Args:
        config_id:      Identifier string for this configuration.
        engine_factory: Zero-argument callable that instantiates and returns
                        the ASR engine. Called exactly once.

    Returns:
        LoadResult with load_time_s, rss_before_mb, rss_after_mb, rss_delta_mb.
    """
    proc = psutil.Process()

    # Settle: force GC and wait for allocator to stabilise
    gc.collect()
    time.sleep(0.5)

    rss_before_mb = proc.memory_info().rss / (1024 * 1024)

    t0 = time.perf_counter()
    engine_factory()  # engine returned but not stored; load time is what matters
    load_time_s = time.perf_counter() - t0

    rss_after_mb = proc.memory_info().rss / (1024 * 1024)
    rss_delta_mb = rss_after_mb - rss_before_mb

    return LoadResult(
        config_id=config_id,
        load_time_s=load_time_s,
        rss_before_mb=rss_before_mb,
        rss_after_mb=rss_after_mb,
        rss_delta_mb=rss_delta_mb,
    )
