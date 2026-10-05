# benchmark/runners/latency_runner.py
"""
Single-leg latency and RTF benchmark runner.

For each audio file: runs `warmup_runs` discarded passes then `n_runs`
measured passes, recording wall-clock latency and RTF each time.
Reports P50/P95/P99 across the measured runs. Captures system metrics
(CPU/memory) during the full measured block.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from benchmark.metrics.audio_info import audio_duration_s
from benchmark.metrics.statistics import summarise
from benchmark.metrics.system_metrics import SystemSampler


@dataclass
class LatencyResult:
    """Raw result for a single transcription call."""
    config_id: str
    audio_file: str
    audio_duration_s: float
    latency_s: float
    rtf: float
    timing_detail: dict = field(default_factory=dict)


@dataclass
class LatencyRunSummary:
    """Aggregated summary for one audio file across n_runs measured passes."""
    config_id: str
    audio_file: str
    audio_duration_s: float
    latency_stats: dict       # mean/p50/p95/p99/min/max/count
    rtf_stats: dict           # mean/p50/p95/p99/min/max/count
    system_metrics: dict      # from SystemSampler.result()
    n_runs: int
    raw_results: list[LatencyResult] = field(default_factory=list)


class LatencyRunner:
    """
    Measures single-leg transcription latency and RTF.

    Args:
        config_id:    Identifier for this configuration.
        engine:       ASR engine with a `.transcribe(audio_path, **kwargs) -> dict` method.
                      RTF = wall-clock latency of the call / audio duration read from
                      the file (falls back to timing["audio_duration_s"] if unreadable).
        audio_files:  List of audio file paths to benchmark.
        n_runs:       Number of measured runs per audio file (warmup excluded).
        warmup_runs:  Number of discarded warm-up runs before measurement starts.
    """

    def __init__(
        self,
        config_id: str,
        engine: Any,
        audio_files: list[str],
        n_runs: int = 3,
        warmup_runs: int = 1,
    ) -> None:
        self.config_id = config_id
        self.engine = engine
        self.audio_files = audio_files
        self.n_runs = n_runs
        self.warmup_runs = warmup_runs

    def _single_transcribe(self, audio_file: str) -> LatencyResult:
        """Run one transcription call and return a LatencyResult."""
        t0 = time.perf_counter()
        result = self.engine.transcribe(audio_file)
        wall_latency = time.perf_counter() - t0

        timing = result.get("timing", {})
        # Prefer the duration read from the file itself so RTF does not depend on
        # how each engine reports (or mis-reports) its own timing.
        duration = audio_duration_s(audio_file)
        if duration is None:
            duration = timing.get("audio_duration_s", 0.0)
        rtf = wall_latency / duration if duration > 0 else 0.0

        return LatencyResult(
            config_id=self.config_id,
            audio_file=audio_file,
            audio_duration_s=duration,
            latency_s=wall_latency,
            rtf=rtf,
            timing_detail=timing,
        )

    def run(self) -> list[LatencyRunSummary]:
        """
        Benchmark all audio files.

        For each audio file:
          1. Run `warmup_runs` calls (results discarded).
          2. Run `n_runs` calls under SystemSampler (results kept).
          3. Aggregate into LatencyRunSummary.

        Returns:
            List of LatencyRunSummary — one per audio file.
        """
        summaries: list[LatencyRunSummary] = []

        for audio_file in self.audio_files:
            print(f"  [latency] {audio_file}")

            # Warmup
            for _ in range(self.warmup_runs):
                self._single_transcribe(audio_file)

            # Measured runs
            raw_results: list[LatencyResult] = []
            with SystemSampler(interval_s=0.1) as sampler:
                for _ in range(self.n_runs):
                    r = self._single_transcribe(audio_file)
                    raw_results.append(r)

            latencies = [r.latency_s for r in raw_results]
            rtfs = [r.rtf for r in raw_results]
            audio_dur = raw_results[0].audio_duration_s if raw_results else 0.0

            summaries.append(LatencyRunSummary(
                config_id=self.config_id,
                audio_file=audio_file,
                audio_duration_s=audio_dur,
                latency_stats=summarise(latencies),
                rtf_stats=summarise(rtfs),
                system_metrics=sampler.result(),
                n_runs=self.n_runs,
                raw_results=raw_results,
            ))

        return summaries
