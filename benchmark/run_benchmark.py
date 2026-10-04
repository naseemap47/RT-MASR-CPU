#!/usr/bin/env python3
# benchmark/run_benchmark.py
"""
Qwen3-ASR CPU Benchmark Pipeline — CLI Entry Point.

Usage:
    python3 benchmark/run_benchmark.py
    python3 benchmark/run_benchmark.py --models onnx_int8,transformers_bf16_0.6b
    python3 benchmark/run_benchmark.py --legs 1,2,4 --runs 5
    python3 benchmark/run_benchmark.py --skip-accuracy
    python3 benchmark/run_benchmark.py --output-dir /tmp/bench_results

Must be run from the project root (RT-MASR-CPU/).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── Ensure project root + src/ are importable ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml

from benchmark.engine_loader import load_engine, engine_factory
from benchmark.reporters.hardware_info import collect_hardware_info
from benchmark.reporters.json_reporter import JsonReporter
from benchmark.reporters.summary_reporter import SummaryReporter
from benchmark.runners.accuracy_runner import AccuracyRunner
from benchmark.runners.concurrency_runner import ConcurrencyRunner
from benchmark.runners.latency_runner import LatencyRunner
from benchmark.runners.load_timer import measure_load


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR CPU Benchmarking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="benchmark/configs/bench_config.yaml",
        help="Path to bench_config.yaml (default: benchmark/configs/bench_config.yaml)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated config IDs to run (default: all configs in bench_config.yaml)",
    )
    parser.add_argument(
        "--legs",
        default=None,
        help="Comma-separated concurrency leg counts (default: from bench_config.yaml)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Number of measured inference runs per audio (default: from bench_config.yaml)",
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip the accuracy runner (faster, perf-only run)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: from bench_config.yaml)",
    )
    return parser.parse_args()


# ── Config helpers ────────────────────────────────────────────────────────────

def load_bench_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_audio_files(audio_cfg: dict) -> list[str]:
    """Flatten the lang→[files] audio dict into a single sorted list."""
    files: list[str] = []
    for lang_files in audio_cfg.values():
        files.extend(lang_files)
    return files


# ── Main orchestration ────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_bench_config(args.config)

    # Apply CLI overrides
    n_runs      = args.runs     if args.runs     is not None else cfg.get("runs", 3)
    warmup_runs = cfg.get("warmup_runs", 1)
    output_dir  = args.output_dir if args.output_dir else cfg.get("output_dir", "benchmark/results")
    legs_list   = (
        [int(x) for x in args.legs.split(",")]
        if args.legs else
        cfg.get("concurrency_legs", [1, 2, 4])
    )
    n_rounds    = cfg.get("concurrency_rounds", 2)

    # Filter configs
    all_configs = cfg.get("configs", [])
    if args.models:
        wanted = set(args.models.split(","))
        all_configs = [c for c in all_configs if c["id"] in wanted]
    if not all_configs:
        print("No configs selected. Check --models or bench_config.yaml.")
        sys.exit(1)

    # Audio files
    audio_cfg     = cfg.get("audio", {})
    all_audio     = collect_audio_files(audio_cfg)
    refs_file     = cfg.get("references_file", "benchmark/data/references.yaml")

    with open(refs_file, "r") as f:
        refs_data = yaml.safe_load(f)
    references: list[dict] = refs_data.get("references", [])

    # Result containers
    all_load_results:       list = []
    all_latency_results:    list = []
    all_accuracy_results:   list = []
    all_concurrency_results: list = []

    # ── Per-config loop ───────────────────────────────────────────────────
    for config_entry in all_configs:
        config_id   = config_entry["id"]
        display_name = config_entry.get("display_name", config_id)
        print(f"\n{'='*70}")
        print(f"  Config: {display_name}")
        print(f"{'='*70}")

        # ── 1. Load timing (cold start) ───────────────────────────────────
        print("\n[1/4] Measuring cold-start load time...")
        factory = engine_factory(config_entry)
        load_result = measure_load(config_id, factory)
        all_load_results.append(load_result)
        print(f"  Load time : {load_result.load_time_s:.2f}s")
        print(f"  RSS delta : {load_result.rss_delta_mb:.0f} MB")

        # ── 2. Load engine for steady-state runs ──────────────────────────
        print("\n[2/4] Loading engine for latency / accuracy / concurrency runs...")
        engine = load_engine(config_entry)

        # ── 3. Latency & RTF ─────────────────────────────────────────────
        print("\n[3/4] Latency & RTF benchmark...")
        latency_runner = LatencyRunner(
            config_id=config_id,
            engine=engine,
            audio_files=all_audio,
            n_runs=n_runs,
            warmup_runs=warmup_runs,
        )
        latency_summaries = latency_runner.run()
        all_latency_results.extend(latency_summaries)

        # ── 4. Accuracy ───────────────────────────────────────────────────
        if not args.skip_accuracy:
            print("\n[4/4a] Accuracy benchmark (WER/CER)...")
            acc_runner = AccuracyRunner(
                config_id=config_id,
                engine=engine,
                references=references,
            )
            accuracy_results = acc_runner.run()
            all_accuracy_results.extend(accuracy_results)
        else:
            print("\n[4/4a] Accuracy benchmark SKIPPED (--skip-accuracy).")

        # ── 5. Concurrency ────────────────────────────────────────────────
        print("\n[4/4b] Concurrency benchmark...")
        conc_runner = ConcurrencyRunner(
            config_id=config_id,
            engine_factory=engine_factory(config_entry),
            audio_files=all_audio,
            legs_list=legs_list,
            n_rounds=n_rounds,
        )
        conc_results = conc_runner.run()
        all_concurrency_results.extend(conc_results)

    # ── Reporting ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Generating Reports")
    print(f"{'='*70}")

    hardware_info = collect_hardware_info()
    all_results = {
        "hardware":    hardware_info,
        "load":        all_load_results,
        "latency":     all_latency_results,
        "accuracy":    all_accuracy_results,
        "concurrency": all_concurrency_results,
    }

    json_reporter    = JsonReporter(output_dir=output_dir)
    summary_reporter = SummaryReporter(output_dir=output_dir)

    json_path    = json_reporter.save(all_results)
    summary_path = summary_reporter.render(all_results)

    print(f"\n✓ Raw JSON   : {json_path}")
    print(f"✓ Summary MD : {summary_path}")
    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
