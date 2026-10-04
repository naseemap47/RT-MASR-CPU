# benchmark/runners/concurrency_runner.py
"""
Concurrency benchmark runner.

Fires N simultaneous transcription calls using ThreadPoolExecutor,
then measures per-call latency, throughput, errors, and system metrics.

One engine instance is created per ConcurrencyRunner run (shared across
all threads). This reflects real-world server usage where one model
serves multiple concurrent callers.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmark.metrics.statistics import summarise
from benchmark.metrics.system_metrics import SystemSampler


@dataclass
class ConcurrencyResult:
    """Results for one concurrency level (n_legs simultaneous calls)."""
    config_id: str
    n_legs: int
    n_rounds: int
    latency_stats: dict    # mean/p50/p95/p99/min/max/count
    rtf_stats: dict        # mean/p50/p95/p99/min/max/count
    throughput_audio_hours_per_wall_hour: float
    error_count: int
    system_metrics: dict   # from SystemSampler.result()
    raw_latencies: list[float] = field(default_factory=list)
    raw_rtfs: list[float] = field(default_factory=list)


class ConcurrencyRunner:
    """
    Benchmarks the engine under N simultaneous concurrent transcription legs.

    For each n_legs value in legs_list:
      - Instantiates a fresh engine via engine_factory (one per legs config).
      - Fires n_legs × n_rounds transcription calls in parallel.
      - Records per-call latency and RTF, error count, throughput, system metrics.

    Args:
        config_id:      Configuration identifier.
        engine_factory: Zero-argument callable returning a fresh engine instance.
        audio_files:    List of audio files to use (cycled across legs if needed).
        legs_list:      List of concurrency levels to test (e.g. [1, 2, 4]).
        n_rounds:       Number of rounds per legs level (each round = n_legs calls).
    """

    def __init__(
        self,
        config_id: str,
        engine_factory: Callable[[], Any],
        audio_files: list[str],
        legs_list: list[int],
        n_rounds: int = 2,
    ) -> None:
        self.config_id = config_id
        self.engine_factory = engine_factory
        self.audio_files = audio_files
        self.legs_list = legs_list
        self.n_rounds = n_rounds

    def _run_one_legs(self, n_legs: int) -> ConcurrencyResult:
        """Benchmark one concurrency level."""
        print(f"  [concurrency] n_legs={n_legs}, n_rounds={self.n_rounds}")

        # Fresh engine for each legs configuration
        engine = self.engine_factory()

        # Build the list of (audio_file, call_index) pairs for all rounds
        call_args: list[str] = []
        for r in range(self.n_rounds):
            for leg in range(n_legs):
                call_args.append(self.audio_files[leg % len(self.audio_files)])

        latencies: list[float] = []
        rtfs: list[float] = []
        errors: int = 0
        total_audio_s: float = 0.0

        def _call(audio_path: str) -> dict:
            t0 = time.perf_counter()
            result = engine.transcribe(audio_path)
            elapsed = time.perf_counter() - t0
            timing = result.get("timing", {})
            audio_dur = timing.get("audio_duration_s", 0.0)
            return {"latency_s": elapsed, "audio_duration_s": audio_dur}

        wall_start = time.perf_counter()
        with SystemSampler(interval_s=0.1) as sampler:
            with ThreadPoolExecutor(max_workers=n_legs) as executor:
                futures = {executor.submit(_call, a): a for a in call_args}
                for future in as_completed(futures):
                    try:
                        res = future.result()
                        latencies.append(res["latency_s"])
                        audio_dur = res["audio_duration_s"]
                        total_audio_s += audio_dur
                        if audio_dur > 0:
                            rtfs.append(res["latency_s"] / audio_dur)
                    except Exception as exc:
                        print(f"    [concurrency] call error: {exc}")
                        errors += 1
        wall_elapsed = time.perf_counter() - wall_start

        # Throughput: audio-hours processed per wall-clock hour
        throughput = (total_audio_s / 3600.0) / (wall_elapsed / 3600.0) if wall_elapsed > 0 else 0.0

        return ConcurrencyResult(
            config_id=self.config_id,
            n_legs=n_legs,
            n_rounds=self.n_rounds,
            latency_stats=summarise(latencies) if latencies else
                          {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0},
            rtf_stats=summarise(rtfs) if rtfs else
                      {"mean": 0, "p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0},
            throughput_audio_hours_per_wall_hour=throughput,
            error_count=errors,
            system_metrics=sampler.result(),
            raw_latencies=latencies,
            raw_rtfs=rtfs,
        )

    def run(self) -> list[ConcurrencyResult]:
        """
        Run all concurrency levels in legs_list.

        Returns:
            List of ConcurrencyResult — one per legs value.
        """
        results: list[ConcurrencyResult] = []
        for n_legs in self.legs_list:
            results.append(self._run_one_legs(n_legs))
        return results
