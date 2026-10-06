#!/usr/bin/env python3
# loadtest/run_loadtest.py
"""
ASR CPU Load Test: simulate many independent real-time call legs, find the saturation point.

Usage:
    python3 loadtest/run_loadtest.py                                   # everything in loadtest_config.yaml
    python3 loadtest/run_loadtest.py --models whisper_int8_tiny
    python3 loadtest/run_loadtest.py --vcpus 8,16 --processes 1,2      # override the scenario sweeps
    python3 loadtest/run_loadtest.py --levels 1,2,4 --duration 15      # quick smoke run
    python3 loadtest/run_loadtest.py --list                            # show the scenarios and exit

Then turn the measured data into a deployment sizing guide:
    python3 loadtest/run_sizing.py

Relative paths are resolved from the project root.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ASR CPU load test", epilog=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="loadtest/configs/loadtest_config.yaml")
    p.add_argument("--models", default=None, help="Comma-separated config ids (default: all in the load-test config)")
    p.add_argument("--vcpus", default=None, help="Override core_sweep, e.g. 8,16")
    p.add_argument("--processes", default=None, help="Override process_sweep (at the largest vCPU count), e.g. 1,2")
    p.add_argument("--levels", default=None, help="Override ramp.levels, e.g. 1,2,4,8")
    p.add_argument("--profiles", default=None, help="Comma-separated load profiles from the config (default: all), e.g. conversational")
    p.add_argument("--duration", type=float, default=None, help="Override call.duration_s")
    p.add_argument("--no-refine", action="store_true", help="Skip bisecting between last healthy / first failing level")
    p.add_argument("--no-confirm", action="store_true", help="Skip re-running the saturation point to confirm it")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--list", action="store_true", help="Print the scenario matrix and exit")
    return p.parse_args()


def _ints(text: str | None) -> list[int] | None:
    return [int(x) for x in text.split(",") if x.strip()] if text else None


def expand_scenarios(cfg: dict, max_vcpus: int, models: list[str] | None,
                     vcpus_override: list[int] | None, procs_override: list[int] | None,
                     profiles: list[str] | None = None) -> list[dict]:
    """Scenario matrix: per load profile, per model: the core sweep (1 process) then the process sweep at the top vCPU count."""
    out: list[dict] = []
    for prof in (profiles or list(cfg["profiles"])):
        out += _expand_one_profile(cfg, max_vcpus, models, vcpus_override, procs_override, prof)
    return out


def _expand_one_profile(cfg: dict, max_vcpus: int, models: list[str] | None,
                        vcpus_override: list[int] | None, procs_override: list[int] | None, profile: str) -> list[dict]:
    out: list[dict] = []
    for m in cfg.get("models", []):
        if models and m["id"] not in models:
            continue
        sweep = sorted({v for v in (vcpus_override or m.get("core_sweep", [max_vcpus])) if 0 < v <= max_vcpus})
        if not sweep:
            continue
        top = sweep[-1]
        for v in sweep:
            out.append({"model_id": m["id"], "profile": profile, "vcpus": v, "processes": 1})
        for p in (procs_override if procs_override is not None else m.get("process_sweep", [])):
            if p > 1 and top // p >= 1:
                out.append({"model_id": m["id"], "profile": profile, "vcpus": top, "processes": p})
    seen, uniq = set(), []
    for sc in out:
        key = (sc["profile"], sc["model_id"], sc["vcpus"], sc["processes"])
        if key not in seen:
            seen.add(key)
            uniq.append(sc)
    return uniq


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
    print(f"\n{'=' * 78}\n  {entry['id']} [{prof_name}, gap {gap_s:g}s]: {sc['vcpus']} vCPU, {sc['processes']} process(es) x {threads} threads "
          f"(CPUs {cpus[0]}-{cpus[-1]})\n{'=' * 78}")

    pool = WorkerPool(entry, groups, runner_cfg,
                      reserve_mb=float(safety["reserve_mb"]), hard_floor_mb=float(safety["hard_floor_mb"]))
    try:
        ready = pool.start()
        result["ready"] = ready
        result["base_rss_mb"] = sum(r["rss_mb"] for r in ready)
        result["load_s"] = max(r["load_s"] for r in ready)
        print(f"  [loadtest] {len(ready)} worker(s) ready: load {result['load_s']:.1f}s, "
              f"RSS {result['base_rss_mb']:.0f} MB (after warm-up)")
        result["ramp"] = run_ramp(
            pool, ramp_levels,
            spread_s=float(call["start_spread_s"]), call_duration_s=float(call["duration_s"]),
            abort_lag_s=float(slo["abort_lag_s"]), reserve_mb=float(safety["reserve_mb"]),
            assumed_mb_per_leg=float(safety["assumed_mb_per_leg"]),
            base_rss_mb=result["base_rss_mb"], **ramp_opts,
        )
        r: RampResult = result["ramp"]
        print(f"\n  [loadtest] RESULT: max legs kept up = {r.l_sat}, first failing = {r.l_fail}, "
              f"stopped: {r.stop_reason}{' (confirmed)' if r.confirmed else ''}")
    except InsufficientMemory as exc:
        result["status"], result["error"] = "infeasible_memory", str(exc)
        result["base_rss_mb"] = exc.info["per_process_rss_mb"] * sc["processes"]
        result["load_s"] = exc.info["load_s"]
        print(f"  [loadtest] SKIPPED (memory): {exc}")
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        traceback.print_exc()
        result["status"], result["error"] = "error", f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        pool.close()
    return result


def main() -> None:
    args = parse_args()
    config_path = os.path.abspath(args.config)
    cli_out = os.path.abspath(args.output_dir) if args.output_dir else None
    os.chdir(_PROJECT_ROOT)

    cfg = yaml.safe_load(open(config_path))
    if args.duration is not None:
        cfg["call"]["duration_s"] = args.duration
    bench = yaml.safe_load(open(cfg["bench_config"]))
    entries = {c["id"]: c for c in bench["configs"]}

    models = [m.strip() for m in args.models.split(",")] if args.models else None
    unknown = [m for m in (models or []) if m not in entries]
    if unknown:
        sys.exit(f"Unknown config id(s): {unknown}. Available: {sorted(entries)}")

    max_vcpus = len(available_cpus())
    profiles = [x.strip() for x in args.profiles.split(",")] if args.profiles else None
    bad_prof = [x for x in (profiles or []) if x not in cfg["profiles"]]
    if bad_prof:
        sys.exit(f"Unknown profile(s): {bad_prof}. Available: {sorted(cfg['profiles'])}")
    scenarios = expand_scenarios(cfg, max_vcpus, models, _ints(args.vcpus), _ints(args.processes), profiles)
    if not scenarios:
        sys.exit("No scenarios selected. Check --models / --vcpus and the load-test config.")

    if args.list:
        for sc in scenarios:
            print(f"{sc['profile']:<15} {sc['model_id']:<28} {sc['vcpus']:>3} vCPU  {sc['processes']} process(es)")
        return

    call_audio = [a for a in cfg["call"]["audio"] if os.path.exists(a)]
    if not call_audio:
        sys.exit("None of call.audio exists.")
    ramp_levels = _ints(args.levels) or cfg["ramp"]["levels"]
    ramp_opts = {
        "refine": bool(cfg["ramp"].get("refine", True)) and not args.no_refine,
        "refine_steps": int(cfg["ramp"].get("refine_steps", 4)),
        "confirm": bool(cfg["ramp"].get("confirm", True)) and not args.no_confirm,
    }

    reporter = LoadtestReporter(cli_out or cfg.get("output_dir", "loadtest/results"))
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
            print("\nInterrupted: writing partial results.")
            interrupted = True
        reporter.write_json(results)               # after every scenario, so a crash loses nothing
        reporter.render_summary(results)
        if interrupted:
            break

    print(f"\n✓ Raw JSON   : {reporter.json_path}\n✓ Summary MD : {reporter.summary_path}")
    bad = [s for s in results["scenarios"] if s["status"] == "error"]
    if bad:
        print(f"⚠ {len(bad)} scenario(s) errored")
        sys.exit(2)


if __name__ == "__main__":
    main()
