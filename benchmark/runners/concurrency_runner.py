# benchmark/runners/concurrency_runner.py
"""
Concurrency benchmark runner.

Fires N simultaneous transcription calls using ThreadPoolExecutor,
then measures per-call latency, throughput, errors, and system metrics.

A single engine instance is shared by all threads (created once per run, warmed
up once). This reflects real-world server usage where one loaded model serves
multiple concurrent callers.

Every concurrency level processes the *same* audio workload (calls cycle through
`audio_files` in order), so latency / throughput are comparable across levels.
"""
from __future__ import annotations

import gc
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmark.metrics.audio_info import audio_duration_s
from benchmark.metrics.statistics import summarise
from benchmark.metrics.system_metrics import SystemSampler

_EMPTY_STATS = {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}


@dataclass
class ConcurrencyResult:
    """Results for one concurrency level (n_legs simultaneous calls)."""
    config_id: str
    n_legs: int
    n_rounds: int
    latency_stats: dict    # per-call wall latency: mean/p50/p95/p99/min/max/count
    rtf_stats: dict        # per-call latency / that call's audio duration
    # Audio seconds transcribed per wall-clock second (== audio-hours per wall-hour).
    # >1 means the box transcribes faster than real time in aggregate.
    throughput_audio_hours_per_wall_hour: float
    error_count: int
    system_metrics: dict   # from SystemSampler.result()
    raw_latencies: list[float] = field(default_factory=list)
    raw_rtfs: list[float] = field(default_factory=list)
    wall_elapsed_s: float = 0.0     # wall time to finish all calls at this level
    n_calls: int = 0                # calls submitted (n_legs * n_rounds)
    total_audio_s: float = 0.0      # audio seconds of the successful calls


class ConcurrencyRunner:
    """
    Benchmarks the engine under N simultaneous concurrent transcription legs.

    For each n_legs value in legs_list:
      - Fires n_legs × n_rounds transcription calls through a pool of n_legs workers.
      - Records per-call latency and RTF, error count, throughput, system metrics.

    Args:
        config_id:      Configuration identifier.
        engine_factory: Zero-argument callable returning an engine instance
                        (called once per run(); may return an already-loaded engine).
        audio_files:    Audio files for the workload; call i uses audio_files[i % len].
        legs_list:      List of concurrency levels to test (e.g. [1, 2, 4]).
        n_rounds:       Calls per worker at each level (total calls = n_legs * n_rounds).
        warmup:         Run one un-timed call before measuring.
    """

    def __init__(
        self,
        config_id: str,
        engine_factory: Callable[[], Any],
        audio_files: list[str],
        legs_list: list[int],
        n_rounds: int = 2,
        warmup: bool = True,
    ) -> None:
        self.config_id = config_id
        self.engine_factory = engine_factory
        self.audio_files = audio_files
        self.legs_list = legs_list
        self.n_rounds = n_rounds
        self.warmup = warmup

    def _run_one_legs(self, engine: Any, n_legs: int) -> ConcurrencyResult:
        """Benchmark one concurrency level."""
        n_calls = n_legs * self.n_rounds
        print(f"  [concurrency] n_legs={n_legs}, n_rounds={self.n_rounds} ({n_calls} calls)")

        # Identical workload prefix at every level -> comparable results.
        call_args = [self.audio_files[i % len(self.audio_files)] for i in range(n_calls)]

        latencies: list[float] = []
        rtfs: list[float] = []
        errors = 0
        total_audio_s = 0.0

        def _call(audio_path: str) -> dict:
            t0 = time.perf_counter()
            result = engine.transcribe(audio_path)
            elapsed = time.perf_counter() - t0
            duration = audio_duration_s(audio_path)
            if duration is None:
                duration = result.get("timing", {}).get("audio_duration_s", 0.0)
            return {"latency_s": elapsed, "audio_duration_s": duration}

        wall_start = time.perf_counter()
        with SystemSampler(interval_s=0.1) as sampler:
            with ThreadPoolExecutor(max_workers=n_legs) as executor:
                futures = [executor.submit(_call, a) for a in call_args]
                for future in as_completed(futures):
                    try:
                        res = future.result()
                    except Exception as exc:
                        print(f"    [concurrency] call error: {exc}")
                        errors += 1
                        continue
                    latencies.append(res["latency_s"])
                    dur = res["audio_duration_s"]
                    total_audio_s += dur
                    if dur > 0:
                        rtfs.append(res["latency_s"] / dur)
        wall_elapsed = time.perf_counter() - wall_start

        throughput = total_audio_s / wall_elapsed if wall_elapsed > 0 else 0.0

        return ConcurrencyResult(
            config_id=self.config_id,
            n_legs=n_legs,
            n_rounds=self.n_rounds,
            latency_stats=summarise(latencies) if latencies else dict(_EMPTY_STATS),
            rtf_stats=summarise(rtfs) if rtfs else dict(_EMPTY_STATS),
            throughput_audio_hours_per_wall_hour=throughput,
            error_count=errors,
            system_metrics=sampler.result(),
            raw_latencies=latencies,
            raw_rtfs=rtfs,
            wall_elapsed_s=wall_elapsed,
            n_calls=n_calls,
            total_audio_s=total_audio_s,
        )

    def run(self) -> list[ConcurrencyResult]:
        """
        Run all concurrency levels in legs_list.

        Returns:
            List of ConcurrencyResult — one per legs value.
        """
        engine = self.engine_factory()
        try:
            if self.warmup and self.audio_files:
                print("  [concurrency] warm-up call (not timed)")
                try:
                    engine.transcribe(self.audio_files[0])
                except Exception as exc:
                    print(f"    [concurrency] warm-up error: {exc}")

            return [self._run_one_legs(engine, n_legs) for n_legs in self.legs_list]
        finally:
            del engine
            gc.collect()
