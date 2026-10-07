#!/usr/bin/env python3
# loadtest/run_loadtest.py
"""
ASR CPU Load Test: simulate many independent real-time call legs, find the saturation point.

Usage:
    python3 loadtest/run_loadtest.py                                   # everything in loadtest_config.yaml
    python3 loadtest/run_loadtest.py --models whisper_int8_tiny
    python3 loadtest/run_loadtest.py --cpus 8                          # pin 8 threads instead of the whole machine
    python3 loadtest/run_loadtest.py --levels 1,2,4 --duration 15      # quick smoke run
    python3 loadtest/run_loadtest.py --list                            # show the runs and exit

Then turn the measured data into a deployment sizing guide:
    python3 loadtest/run_sizing.py

Relative paths are resolved from the project root.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

import yaml

from benchmark.engine_loader import _deep_update, stream_mode_for, streaming_settings
from benchmark.reporters.hardware_info import collect_hardware_info
from loadtest.reporters.loadtest_reporter import LoadtestReporter
from loadtest.runners.ramp import RampResult, run_ramp
from loadtest.runners.worker_pool import InsufficientMemory, WorkerPool
from loadtest.topology import available_cpus, cpu_set, split_cpus
from src.core.model_check import (
    bench_entries_not_downloaded, bench_id_hints, bench_table, report, unknown_name_panel,
)
from src.core.runlog import current_run, start_run

logger = logging.getLogger("rtmasr.loadtest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR CPU load test", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="loadtest/configs/loadtest_config.yaml")
    p.add_argument("--models", default=None, help="Comma-separated config ids (default: all in the load-test config)")
    p.add_argument("--cpus", "--vcpus", dest="cpus", type=int, default=None,
                   help="Logical CPUs to pin (default: all on this machine)")
    p.add_argument("--processes", type=int, default=1,
                   help="Worker processes sharing those CPUs (default: 1)")
    p.add_argument("--levels", default=None, help="Override ramp.levels, e.g. 1,2,4,8")
    p.add_argument("--profiles", default=None, help="Comma-separated load profiles from the config (default: all), e.g. conversational")
    p.add_argument("--duration", type=float, default=None, help="Override call.duration_s")
    p.add_argument("--no-refine", action="store_true", help="Skip bisecting between last healthy / first failing level")
    p.add_argument("--no-confirm", action="store_true", help="Skip re-running the saturation point to confirm it")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--list", action="store_true", help="Print the runs and exit")
    return p.parse_args()


def _ints(text: str | None) -> list[int] | None:
    return [int(x) for x in text.split(",") if x.strip()] if text else None


def expand_scenarios(cfg: dict, n_cpus: int, models: list[str] | None,
                     processes: int = 1, profiles: list[str] | None = None) -> list[dict]:
    """One run per (profile, model) on this machine: ``n_cpus`` threads, ``processes`` workers."""
    if n_cpus < 1 or processes < 1:
        return []
    out: list[dict] = []
    for prof in (profiles or list(cfg["profiles"])):
        for m in cfg.get("models", []):
            mid = m["id"] if isinstance(m, dict) else m
            if models and mid not in models:
                continue
            out.append({"model_id": mid, "profile": prof, "vcpus": n_cpus, "processes": processes})
    return out


def speech_fraction(audio: list[str], gap_s: float) -> float | None:
    """Share of a call that is speech: mean over the clips of d / (d + gap)."""
    from benchmark.metrics.audio_info import audio_duration_s
    fr = []
    for a in audio:
        d = audio_duration_s(a)
        if d:
            fr.append(d / (d + gap_s))
    return round(sum(fr) / len(fr), 3) if fr else None


def run_scenario(entry: dict, sc: dict, cfg: dict, call_audio: list[str], ramp_levels: list[int],
                 ramp_opts: dict) -> dict:
    cpus = cpu_set(sc["vcpus"])
    groups = split_cpus(cpus, sc["processes"])
    threads = min(len(g) for g in groups)
    entry = dict(entry)
    entry["overrides"] = _deep_update(dict(entry.get("overrides") or {}), {"engine": {"num_threads": threads}})

    call, slo, safety = cfg["call"], cfg["slo"], cfg["safety"]
    prof_name = sc["profile"]
    prof = cfg["profiles"][prof_name]
    gap_s = float(prof["gap_s"])
    runner_cfg = {
        "config_id": entry["id"],
        "audio_files": call_audio,
        "stream_mode": stream_mode_for(entry),
        "streaming_cfg": streaming_settings(entry),
        "chunk_s": float(call.get("chunk_s", 0.5)),
        "lag_threshold_s": float(slo["lag_threshold_s"]),
        "abort_lag_s": float(slo["abort_lag_s"]),
        "call_duration_s": float(call["duration_s"]),
        "gap_s": gap_s,
    }
    result = {
        "model_id": entry["id"], "profile": prof_name, "gap_s": gap_s,
        "speech_fraction": speech_fraction(call_audio, gap_s),
        "display_name": entry.get("display_name", entry["id"]),
        "backend": entry.get("backend"), "stream_mode": runner_cfg["stream_mode"],
        "vcpus": sc["vcpus"], "processes": sc["processes"], "threads_per_process": threads,
        "cpus": cpus, "status": "ok", "error": "", "ready": [], "base_rss_mb": 0.0, "load_s": 0.0, "ramp": None,
    }
    logger.info("\n%s\n  %s [%s, gap %gs]: %s CPU threads, %s process(es) x %s threads "
                "(CPUs %s-%s)\n%s",
                "=" * 78, entry["id"], prof_name, gap_s, sc["vcpus"], sc["processes"],
                threads, cpus[0], cpus[-1], "=" * 78)

    session = current_run()
    pool = WorkerPool(entry, groups, runner_cfg,
                      reserve_mb=float(safety["reserve_mb"]), hard_floor_mb=float(safety["hard_floor_mb"]),
                      log_dir=str(session.dir) if session is not None else None)
    try:
        ready = pool.start()
        result["ready"] = ready
        result["base_rss_mb"] = sum(r["rss_mb"] for r in ready)
        result["load_s"] = max(r["load_s"] for r in ready)
        logger.info("  [loadtest] %s worker(s) ready: load %.1fs, RSS %.0f MB (after warm-up)",
                    len(ready), result["load_s"], result["base_rss_mb"])
        result["ramp"] = run_ramp(
            pool, ramp_levels,
            spread_s=float(call["start_spread_s"]), call_duration_s=float(call["duration_s"]),
            abort_lag_s=float(slo["abort_lag_s"]), reserve_mb=float(safety["reserve_mb"]),
            assumed_mb_per_leg=float(safety["assumed_mb_per_leg"]),
            base_rss_mb=result["base_rss_mb"], **ramp_opts,
        )
        r: RampResult = result["ramp"]
        logger.info("  [loadtest] RESULT: max legs kept up = %s, first failing = %s, "
                    "stopped: %s%s",
                    r.l_sat, r.l_fail, r.stop_reason, " (confirmed)" if r.confirmed else "")
    except InsufficientMemory as exc:
        result["status"], result["error"] = "infeasible_memory", str(exc)
        result["base_rss_mb"] = exc.info["per_process_rss_mb"] * sc["processes"]
        result["load_s"] = exc.info["load_s"]
        logger.warning("  [loadtest] SKIPPED (memory): %s", exc)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.exception("scenario failed")
        result["status"], result["error"] = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        pool.close()
    return result


def _run(args: argparse.Namespace, run) -> None:
    config_path = os.path.abspath(args.config)
    cli_out = os.path.abspath(args.output_dir) if args.output_dir else None
    os.chdir(_PROJECT_ROOT)

    cfg = yaml.safe_load(open(config_path))
    if args.duration is not None:
        cfg["call"]["duration_s"] = args.duration
    bench = yaml.safe_load(open(cfg["bench_config"]))
    entries = {c["id"]: c for c in bench["configs"]}

    roster = [m["id"] if isinstance(m, dict) else m for m in cfg.get("models", [])]
    models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
    unknown = [m for m in (models or []) if m not in roster]
    if unknown:
        hints = bench_id_hints(unknown, entries.values())
        not_in_roster = [m for m in unknown if m in entries]
        if not_in_roster:
            hints.append(f"{', '.join(not_in_roster)}: defined in bench_config.yaml but not in the load-test "
                         f"roster. Add it under `models:` in {os.path.relpath(config_path)}.")
        hints += ["", "Load-test ids come from `models:` in the load-test config:",
                  "$ uv run python loadtest/run_loadtest.py --models <id>[,<id>...]"]
        report(logger, unknown_name_panel(
            "load-test model id", unknown, roster,
            bench_table([entries[m] for m in roster if m in entries]),
            where="--models", hints=hints))
        sys.exit(1)
    missing_def = [m for m in roster if m not in entries]
    if missing_def:
        sys.exit(f"Load-test model id(s) {missing_def} are not defined in {cfg['bench_config']}.")

    n_cpus = len(available_cpus())
    if args.cpus is not None:
        if args.cpus < 1 or args.cpus > n_cpus:
            sys.exit(f"--cpus {args.cpus} is outside 1..{n_cpus} on this machine")
        n_cpus = args.cpus
    if args.processes < 1 or args.processes > n_cpus:
        sys.exit(f"--processes {args.processes} must be between 1 and {n_cpus}")
    profiles = [x.strip() for x in args.profiles.split(",")] if args.profiles else None
    bad_prof = [x for x in (profiles or []) if x not in cfg["profiles"]]
    if bad_prof:
        sys.exit(f"Unknown profile(s): {bad_prof}. Available: {sorted(cfg['profiles'])}")
    scenarios = expand_scenarios(cfg, n_cpus, models, args.processes, profiles)
    if not scenarios:
        sys.exit("No runs selected. Check --models and the load-test config.")

    selected = list(dict.fromkeys(sc["model_id"] for sc in scenarios))
    not_downloaded = bench_entries_not_downloaded([entries[m] for m in selected])
    missing_ids = {e["id"] for e, _ in not_downloaded}

    if args.list:
        for sc in scenarios:
            logger.info("%-15s %-28s %3s CPU threads  %s process(es)%s",
                        sc["profile"], sc["model_id"], sc["vcpus"], sc["processes"],
                        "  [NOT DOWNLOADED]" if sc["model_id"] in missing_ids else "")
        for _, panel in not_downloaded:
            panel.level = "warning"
            report(logger, panel)
        return

    for _, panel in not_downloaded:
        if not models:
            panel.level = "warning"
            panel.blank().note("Skipped in this run; the other models continue.")
        report(logger, panel)
    if not_downloaded:
        if models:
            sys.exit(1)
        scenarios = [sc for sc in scenarios if sc["model_id"] not in missing_ids]
        if not scenarios:
            sys.exit("No downloaded models left to load-test.")

    call_audio = [a for a in cfg["call"]["audio"] if os.path.exists(a)]
    if not call_audio:
        sys.exit("None of call.audio exists.")
    ramp_levels = _ints(args.levels) or cfg["ramp"]["levels"]
    ramp_opts = {
        "refine": bool(cfg["ramp"].get("refine", True)) and not args.no_refine,
        "refine_steps": int(cfg["ramp"].get("refine_steps", 4)),
        "confirm": bool(cfg["ramp"].get("confirm", True)) and not args.no_confirm,
    }

    stamp = run.stamp if run is not None else None
    reporter = LoadtestReporter(cli_out or cfg.get("output_dir", "loadtest/results"), stamp=stamp)
    results: dict = {
        "hardware": collect_hardware_info(),
        "params": {
            "call_duration_s": cfg["call"]["duration_s"],
            "profiles": {k: v for k, v in cfg["profiles"].items() if not profiles or k in profiles},
            "start_spread_s": cfg["call"]["start_spread_s"], "chunk_s": cfg["call"].get("chunk_s", 0.5),
            "audio": ", ".join(os.path.basename(a) for a in call_audio),
            "lag_threshold_s": cfg["slo"]["lag_threshold_s"], "abort_lag_s": cfg["slo"]["abort_lag_s"],
            "ramp_levels": ramp_levels, **{f"ramp_{k}": v for k, v in ramp_opts.items()},
            "reserve_mb": cfg["safety"]["reserve_mb"],
        },
        "scenarios": [],
    }

    interrupted = False
    for sc in scenarios:
        try:
            results["scenarios"].append(
                run_scenario(entries[sc["model_id"]], sc, cfg, call_audio, ramp_levels, ramp_opts))
        except KeyboardInterrupt:
            logger.warning("Interrupted: writing partial results.")
            interrupted = True
        reporter.write_json(results)               # after every scenario, so a crash loses nothing
        reporter.render_summary(results)
        if interrupted:
            break

    if run is not None:
        run.note_artifact("raw_json", reporter.json_path)
        run.note_artifact("summary_md", reporter.summary_path)
    logger.info("Raw JSON   : %s", reporter.json_path)
    logger.info("Summary MD : %s", reporter.summary_path)
    bad = [s for s in results["scenarios"] if s["status"] == "error"]
    if bad:
        logger.warning("%d scenario(s) errored", len(bad))
        sys.exit(2)


def main() -> None:
    args = parse_args()
    with start_run("loadtest") as run:
        _run(args, run)


if __name__ == "__main__":
    main()
