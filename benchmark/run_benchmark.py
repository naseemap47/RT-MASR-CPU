#!/usr/bin/env python3
# benchmark/run_benchmark.py
"""
ASR CPU Benchmark Pipeline (Qwen3-ASR + Whisper) — CLI Entry Point.

Usage:
    python3 benchmark/run_benchmark.py
    python3 benchmark/run_benchmark.py --models qwen3_onnx_int8_0.6b,whisper_int8_tiny
    python3 benchmark/run_benchmark.py --legs 1,2,4 --runs 5
    python3 benchmark/run_benchmark.py --concurrency-mode batch      # offline request-queue test
    python3 benchmark/run_benchmark.py --skip-accuracy --skip-concurrency
    python3 benchmark/run_benchmark.py --output-dir /tmp/bench_results

Relative paths in bench_config.yaml and in the model YAMLs are resolved from the
project root (the script changes into it), so it can be launched from anywhere.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Ensure project root + src/ are importable ─────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml

from benchmark.engine_loader import engine_factory, stream_mode_for, streaming_settings
from benchmark.reporters.hardware_info import collect_hardware_info
from benchmark.reporters.json_reporter import JsonReporter
from benchmark.reporters.summary_reporter import SummaryReporter
from benchmark.runners.accuracy_runner import AccuracyRunner
from benchmark.runners.concurrency_runner import ConcurrencyRunner
from benchmark.runners.latency_runner import LatencyRunner
from benchmark.runners.load_timer import measure_load_with_engine, release_memory
from benchmark.runners.streaming_concurrency_runner import StreamingConcurrencyRunner
from src.core.runlog import start_run

logger = logging.getLogger("rtmasr.benchmark")


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ASR CPU Benchmarking Pipeline",
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
        "--concurrency-mode",
        choices=["stream", "batch"],
        default=None,
        help="stream = each leg is a live audio stream at real-time pace (default); "
             "batch = offline queue of transcribe() calls (default: from bench_config.yaml)",
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
        "--skip-concurrency",
        action="store_true",
        help="Skip the concurrency runner",
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


class _StageError(Exception):
    """Carries the failing stage name; the real error is in __cause__."""
    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


def _benchmark_one_config(
    config_entry, args, references, all_audio, conc_audio,
    n_runs, warmup_runs, legs_list, n_rounds, conc_opts,
    load_out, latency_out, accuracy_out, concurrency_out,
) -> None:
    """
    Run every stage for one config.

    Kept in its own function so the engine and runners go out of scope when it
    returns -- the next config's cold-load RSS baseline is then not polluted by
    this config's memory.
    """
    config_id    = config_entry["id"]
    display_name = config_entry.get("display_name", config_id)
    logger.info("\n%s\n  Config: %s\n%s", "=" * 70, display_name, "=" * 70)

    stage = "load"
    try:
        # ── 1. Load (cold start) — the loaded engine is reused below ──────
        logger.info("[1/4] Loading engine (cold-start timing)...")
        load_result, engine = measure_load_with_engine(
            config_id, engine_factory(config_entry)
        )
        load_out.append(load_result)
        logger.info("  Load time : %.2fs", load_result.load_time_s)
        logger.info("  RSS delta : %.0f MB", load_result.rss_delta_mb)

        # ── 2. Latency & RTF ──────────────────────────────────────────────
        stage = "latency"
        logger.info("[2/4] Latency & RTF benchmark...")
        latency_out.extend(LatencyRunner(
            config_id=config_id, engine=engine, audio_files=all_audio,
            n_runs=n_runs, warmup_runs=warmup_runs,
        ).run())

        # ── 3. Accuracy ───────────────────────────────────────────────────
        if not args.skip_accuracy:
            stage = "accuracy"
            logger.info("[3/4] Accuracy benchmark (WER/CER)...")
            accuracy_out.extend(AccuracyRunner(
                config_id=config_id, engine=engine, references=references,
            ).run())
        else:
            logger.info("[3/4] Accuracy benchmark SKIPPED (--skip-accuracy).")

        # ── 4. Concurrency ────────────────────────────────────────────────
        if not args.skip_concurrency:
            stage = "concurrency"
            logger.info("[4/4] Concurrency benchmark...")
            if conc_opts["mode"] == "stream":
                concurrency_out.extend(StreamingConcurrencyRunner(
                    config_id=config_id,
                    engine_factory=lambda: engine,   # reuse the already-loaded engine
                    audio_files=conc_audio,
                    legs_list=legs_list,
                    stream_mode=stream_mode_for(config_entry),
                    streaming_cfg=streaming_settings(config_entry),
                    chunk_s=conc_opts["chunk_s"],
                    stagger_s=conc_opts["stagger_s"],
                    lag_threshold_s=conc_opts["lag_threshold_s"],
                ).run())
            else:
                concurrency_out.extend(ConcurrencyRunner(
                    config_id=config_id,
                    engine_factory=lambda: engine,   # reuse the already-loaded engine
                    audio_files=conc_audio,
                    legs_list=legs_list,
                    n_rounds=n_rounds,
                ).run())
        else:
            logger.info("[4/4] Concurrency benchmark SKIPPED (--skip-concurrency).")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise _StageError(stage) from exc


# ── Main orchestration ────────────────────────────────────────────────────────

def _run(args: argparse.Namespace, run) -> None:
    # Resolve user-supplied paths against the launch directory, *then* switch to
    # the project root so every relative path inside the configs works.
    config_path = os.path.abspath(args.config)
    cli_output_dir = os.path.abspath(args.output_dir) if args.output_dir else None
    os.chdir(_PROJECT_ROOT)

    cfg = load_bench_config(config_path)

    # Apply CLI overrides
    n_runs      = args.runs     if args.runs     is not None else cfg.get("runs", 3)
    warmup_runs = cfg.get("warmup_runs", 1)
    output_dir  = cli_output_dir or cfg.get("output_dir", "benchmark/results")
    legs_list   = (
        [int(x) for x in args.legs.split(",")]
        if args.legs else
        cfg.get("concurrency_legs", [1, 2, 4])
    )
    n_rounds    = cfg.get("concurrency_rounds", 2)
    conc_opts   = {
        "mode":            args.concurrency_mode or cfg.get("concurrency_mode", "stream"),
        "chunk_s":         float(cfg.get("concurrency_chunk_s", 0.5)),
        "stagger_s":       float(cfg.get("concurrency_stagger_s", 0.0)),
        "lag_threshold_s": float(cfg.get("concurrency_lag_threshold_s", 2.0)),
    }

    # Filter configs
    all_configs = cfg.get("configs", [])
    if args.models:
        # Explicitly named ids always run, even if `enabled: false` in the YAML.
        wanted = [m.strip() for m in args.models.split(",") if m.strip()]
        known = {c["id"] for c in all_configs}
        unknown = [m for m in wanted if m not in known]
        if unknown:
            logger.error("Unknown config id(s): %s. Available: %s", unknown, sorted(known))
            sys.exit(1)
        all_configs = [c for c in all_configs if c["id"] in set(wanted)]
    else:
        # Default run: skip configs marked `enabled: false` (default is enabled).
        disabled = [c["id"] for c in all_configs if not c.get("enabled", True)]
        if disabled:
            logger.info("Skipping disabled config(s): %s "
                        "(run explicitly with --models <id> or set enabled: true)",
                        ", ".join(disabled))
        all_configs = [c for c in all_configs if c.get("enabled", True)]
    if not all_configs:
        logger.error("No configs selected. Check --models or bench_config.yaml.")
        sys.exit(1)

    # Audio files (latency) — warn about, and drop, missing files up front
    all_audio = collect_audio_files(cfg.get("audio", {}))
    missing = [a for a in all_audio if not os.path.exists(a)]
    for a in missing:
        logger.warning("audio file missing, skipped: %s", a)
    all_audio = [a for a in all_audio if os.path.exists(a)]
    if not all_audio:
        logger.error("No audio files found. Check 'audio' in bench_config.yaml.")
        sys.exit(1)

    # Concurrency workload: identical for every config and every legs level
    # (stream mode: leg i streams file i mod len)
    conc_audio = [a for a in cfg.get("concurrency_audio", all_audio[:1]) if os.path.exists(a)]
    if not conc_audio:
        conc_audio = all_audio[:1]

    refs_file = cfg.get("references_file", "benchmark/data/references.yaml")
    with open(refs_file, "r") as f:
        references: list[dict] = yaml.safe_load(f).get("references", [])

    # Result containers
    all_load_results:        list = []
    all_latency_results:     list = []
    all_accuracy_results:    list = []
    all_concurrency_results: list = []
    failures:                list = []

    interrupted = False

    # ── Per-config loop ───────────────────────────────────────────────────
    for config_entry in all_configs:
        config_id = config_entry["id"]
        try:
            _benchmark_one_config(
                config_entry, args, references, all_audio, conc_audio,
                n_runs, warmup_runs, legs_list, n_rounds, conc_opts,
                all_load_results, all_latency_results,
                all_accuracy_results, all_concurrency_results,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted — writing partial results.")
            failures.append({"config_id": config_id, "stage": "interrupted", "error": "KeyboardInterrupt"})
            interrupted = True
        except _StageError as exc:
            # One broken config must not throw away the results of the others.
            logger.exception("Config %s failed at stage %s", config_id, exc.stage)
            failures.append({"config_id": config_id, "stage": exc.stage,
                             "error": f"{type(exc.__cause__).__name__}: {exc.__cause__}"})
        finally:
            # _benchmark_one_config's locals (engine, runners) are already gone;
            # hand the freed heap back to the OS before the next cold-load measurement.
            release_memory()

        if interrupted:
            break

    # ── Reporting ─────────────────────────────────────────────────────────
    logger.info("\n%s\n  Generating Reports\n%s", "=" * 70, "=" * 70)
    stamp = run.stamp if run is not None else None

    all_results = {
        "hardware":    collect_hardware_info(),
        "run_params": {
            "configs":            ", ".join(c["id"] for c in all_configs),
            "latency_runs":       n_runs,
            "latency_warmup":     warmup_runs,
            "latency_audio":      len(all_audio),
            "concurrency_legs":   legs_list if not args.skip_concurrency else "skipped",
            "concurrency_mode":   conc_opts["mode"],
            **({"concurrency_chunk_s":         conc_opts["chunk_s"],
                "concurrency_stagger_s":       conc_opts["stagger_s"],
                "concurrency_lag_threshold_s": conc_opts["lag_threshold_s"]}
               if conc_opts["mode"] == "stream" else {"concurrency_rounds": n_rounds}),
            "concurrency_audio":  ", ".join(os.path.basename(a) for a in conc_audio),
        },
        "load":        all_load_results,
        "latency":     all_latency_results,
        "accuracy":    all_accuracy_results,
        "concurrency": all_concurrency_results,
        "failures":    failures,
    }

    json_path    = JsonReporter(output_dir=output_dir, stamp=stamp).save(all_results)
    summary_path = SummaryReporter(output_dir=output_dir, stamp=stamp).render(all_results)
    if run is not None:
        run.note_artifact("raw_json", json_path)
        run.note_artifact("summary_md", summary_path)

    logger.info("Raw JSON   : %s", json_path)
    logger.info("Summary MD : %s", summary_path)
    if failures:
        logger.warning("%d config(s) failed: %s",
                       len(failures),
                       ", ".join(f"{f['config_id']} ({f['stage']})" for f in failures))
        sys.exit(2)
    logger.info("Benchmark complete.")


def main() -> None:
    args = parse_args()
    with start_run("benchmark") as run:
        _run(args, run)


if __name__ == "__main__":
    main()
