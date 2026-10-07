# loadtest/runners/ramp.py
"""
Ramp + saturation search.

Walk a list of leg counts upwards until a level is *unhealthy*, then bisect between the last healthy
and the first unhealthy level, then re-run the answer to confirm it:

    l_sat   highest leg count at which every leg kept up with live speech (the saturation point)
    l_fail  lowest leg count that was seen to fail (None if the ramp stopped for another reason,
            e.g. memory or the top of the ladder)

"Healthy" = every leg kept up (p95 staleness and end-of-call lag under the lag threshold), no errors, no
overload aborts, and every leg produced a transcript.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import psutil

from benchmark.runners.streaming_concurrency_runner import StreamingConcurrencyResult, StreamLegResult, _stats
from loadtest.metrics.tree_sampler import TreeSampler
from loadtest.runners.worker_pool import MemoryAbort, WorkerCrashed, WorkerPool


@dataclass
class LevelOutcome:
    n_legs: int
    healthy: bool
    reasons: list[str]
    result: StreamingConcurrencyResult
    resources: dict                 # TreeSampler result
    legs_per_process: list[int]
    cpu_s_per_audio_s: float        # worker CPU-seconds consumed per audio-second streamed
    passes_per_s: float             # inference passes completed per wall second (all legs)


@dataclass
class RampResult:
    levels: list[LevelOutcome] = field(default_factory=list)
    l_sat: int = 0
    l_fail: Optional[int] = None
    confirmed: bool = False
    stop_reason: str = ""           # "saturated" | "memory" | "top_of_ladder" | "memory_abort" | "worker_crash"
    note: str = ""


def merge_level_results(results: list[StreamingConcurrencyResult]) -> StreamingConcurrencyResult:
    """Combine the per-process slices of one level into a single level result."""
    if not results:
        raise ValueError("no results to merge")
    r0 = results[0]
    legs: list[StreamLegResult] = sorted((leg for r in results for leg in r.legs), key=lambda g: g.leg)
    lat = [x for g in legs for x in g.raw_pass_latencies]
    rtf = [x for g in legs for x in g.raw_pass_rtfs]
    stale = [x for g in legs for x in g.raw_staleness]
    first = [g.first_text_s for g in legs if g.first_text_s is not None]
    return StreamingConcurrencyResult(
        config_id=r0.config_id,
        n_legs=len(legs),
        stream_mode=r0.stream_mode,
        chunk_s=r0.chunk_s,
        pace=r0.pace,
        stagger_s=r0.stagger_s,
        lag_threshold_s=r0.lag_threshold_s,
        pass_latency_stats=_stats(lat),
        pass_rtf_stats=_stats(rtf),
        staleness_stats=_stats(stale),
        first_text_stats=_stats(first),
        end_lag_stats=_stats([g.end_lag_s for g in legs]),
        legs_kept_up=sum(1 for g in legs if g.kept_up),
        legs_with_text=sum(1 for g in legs if g.final_text.strip()),
        error_count=sum(g.errors for g in legs),
        system_metrics={},
        wall_elapsed_s=max(r.wall_elapsed_s for r in results),
        total_audio_s=sum(g.audio_s for g in legs),
        total_passes=sum(g.passes for g in legs),
        legs_aborted=sum(1 for g in legs if g.aborted),
        legs=legs,
    )


def evaluate_level(result: StreamingConcurrencyResult) -> tuple[bool, list[str]]:
    """Healthy iff every leg kept up, nothing errored or aborted, and every leg produced text."""
    n = result.n_legs
    reasons: list[str] = []
    if result.error_count:
        reasons.append(f"errors={result.error_count}")
    if result.legs_aborted:
        reasons.append(f"overload_abort={result.legs_aborted}/{n}")
    if result.legs_kept_up < n:
        reasons.append(f"kept_up={result.legs_kept_up}/{n}")
    if result.legs_with_text < n:
        reasons.append(f"empty_transcript={n - result.legs_with_text}/{n}")
    return (not reasons), reasons


class _MemoryModel:
    """Predicts the worker RSS of a bigger level from the levels already run."""

    def __init__(self, assumed_mb_per_leg: float) -> None:
        self.assumed = assumed_mb_per_leg
        self.points: list[tuple[int, float]] = []       # (n_legs, tree peak RSS)

    def add(self, n_legs: int, rss_peak_mb: float) -> None:
        self.points.append((n_legs, rss_peak_mb))

    def predict(self, n_legs: int) -> float:
        if not self.points:
            return 0.0
        base_n, base_rss = max(self.points, key=lambda p: p[0])
        slope = self.assumed
        distinct = sorted({n: r for n, r in self.points}.items())
        if len(distinct) >= 2:
            (n1, r1), (n2, r2) = distinct[-2], distinct[-1]
            slope = max(0.0, (r2 - r1) / (n2 - n1))
            slope = max(slope, 0.5 * self.assumed)      # never trust a flat early reading completely
        return base_rss + slope * max(0, n_legs - base_n)


def run_ramp(
    pool: WorkerPool,
    levels: list[int],
    *,
    spread_s: float,
    call_duration_s: float,
    abort_lag_s: float,
    reserve_mb: float = 1500.0,
    assumed_mb_per_leg: float = 250.0,
    refine: bool = True,
    refine_steps: int = 4,
    confirm: bool = True,
    max_confirm_steps: int = 2,
    base_rss_mb: float = 0.0,
    log: Callable[[str], None] | None = None,
) -> RampResult:
    """Run the ramp on an already-started pool. Never raises for expected stop conditions."""
    if log is None:
        _lg = logging.getLogger("rtmasr.loadtest")
        log = _lg.info
    ramp = RampResult()
    mem = _MemoryModel(assumed_mb_per_leg)
    if base_rss_mb:
        mem.add(0, base_rss_mb)

    def execute(n: int) -> LevelOutcome:
        stagger = spread_s / max(1, n)
        timeout = call_duration_s + spread_s + abort_lag_s + 90.0
        log(f"\n  [loadtest] level: {n} legs  ({'+'.join(str(c) for c in pool.split_legs(n))} per process)")
        with TreeSampler(pool.pids, pool.cpus) as sampler:
            results = pool.run_level(n, stagger, timeout)
        merged = merge_level_results(results)
        res = sampler.result()
        merged.system_metrics = res
        healthy, reasons = evaluate_level(merged)
        wall = max(res.get("duration_s", 0.0), 1e-9)
        out = LevelOutcome(
            n_legs=n, healthy=healthy, reasons=reasons, result=merged, resources=res,
            legs_per_process=pool.split_legs(n),
            cpu_s_per_audio_s=res.get("tree_cpu_s", 0.0) / merged.total_audio_s if merged.total_audio_s else 0.0,
            passes_per_s=merged.total_passes / wall,
        )
        mem.add(n, res.get("tree_rss_mb_peak", 0.0))
        ramp.levels.append(out)
        st = merged.staleness_stats
        log(f"  [loadtest] -> {'HEALTHY' if healthy else 'UNHEALTHY ' + ','.join(reasons)}: "
            f"kept up {merged.legs_kept_up}/{n}, stale p95 {st['p95']:.2f}s, "
            f"CPU {res.get('cpu_pct', {}).get('mean', 0):.0f}%, RSS peak {res.get('tree_rss_mb_peak', 0):.0f} MB")
        return out

    def memory_blocks(n: int) -> bool:
        if not mem.points:
            return False
        predicted = mem.predict(n)
        now = max(r for _, r in mem.points)
        budget = now + max(0.0, psutil.virtual_memory().available / (1024 * 1024) - reserve_mb)
        if predicted > budget:
            log(f"  [loadtest] stopping before {n} legs: predicted worker RSS {predicted:.0f} MB "
                f"exceeds the {budget:.0f} MB the machine can spare")
            ramp.note = f"predicted RSS {predicted:.0f} MB > budget {budget:.0f} MB at {n} legs"
            return True
        return False

    lo, hi = 0, None
    try:
        for n in sorted(set(levels)):
            if memory_blocks(n):
                ramp.stop_reason = "memory"
                break
            out = execute(n)
            if out.healthy:
                lo = n
            else:
                hi = n
                ramp.stop_reason = "saturated"
                break
        else:
            ramp.stop_reason = "top_of_ladder"

        if hi is not None and refine:
            steps = 0
            while hi - lo > 1 and steps < refine_steps:
                mid = (lo + hi) // 2
                if memory_blocks(mid):
                    break
                steps += 1
                if execute(mid).healthy:
                    lo = mid
                else:
                    hi = mid

        cand = lo
        if confirm and cand > 0:
            for _ in range(max_confirm_steps + 1):
                if cand <= 0:
                    break
                if execute(cand).healthy:
                    ramp.confirmed = True
                    break
                hi = cand if hi is None else min(hi, cand)
                cand -= 1
        lo = cand
    except MemoryAbort as exc:
        ramp.stop_reason, ramp.note = "memory_abort", str(exc)
        log(f"  [loadtest] ABORTED: {exc}")
    except WorkerCrashed as exc:
        ramp.stop_reason, ramp.note = "worker_crash", str(exc)[:300]
        log(f"  [loadtest] ABORTED: {exc}")

    ramp.l_sat = lo
    ramp.l_fail = hi
    return ramp
