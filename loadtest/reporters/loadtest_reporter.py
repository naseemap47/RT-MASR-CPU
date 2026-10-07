# loadtest/reporters/loadtest_reporter.py
"""Raw JSON + Markdown summary for a load-test run (the *measured* data; sizing is separate)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.reporters.json_reporter import _json_default, to_serialisable


def _f(v: Any, d: int = 2) -> str:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = float(v)
        return f"{0.0 if round(v, d) == 0 else v:.{d}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


class LoadtestReporter:
    """Writes ``<UTC>_loadtest_raw.json`` and ``<UTC>_loadtest_summary.md`` (re-writable as a run progresses)."""

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @property
    def json_path(self) -> Path:
        return self.output_dir / f"{self.stamp}_loadtest_raw.json"

    @property
    def summary_path(self) -> Path:
        return self.output_dir / f"{self.stamp}_loadtest_summary.md"

    def write_json(self, results: dict) -> str:
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(to_serialisable(results), f, indent=2, ensure_ascii=False, default=_json_default)
        return str(self.json_path)

    def render_summary(self, results: dict) -> str:
        L: list[str] = ["# ASR CPU Load Test Report\n",
                        f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
                        "> Everything in this file is **measured** on the hardware below. Sizing for larger "
                        "deployments is derived from it in the separate sizing guide.\n"]

        hw = results.get("hardware", {})
        if hw:
            L.append("## Test machine\n")
            L.append(_table(["Item", "Value"], [
                ["CPU", str(hw.get("cpu_model", ""))],
                ["Physical / logical cores", f"{hw.get('physical_cores', '')} / {hw.get('logical_cores', '')}"],
                ["RAM", f"{hw.get('ram_gb', 0):.1f} GB"],
                ["OS", str(hw.get("os", ""))],
                ["onnxruntime", str(hw.get("lib_versions", {}).get("onnxruntime", ""))],
            ]) + "\n")

        params = results.get("params", {})
        if params:
            L.append("## Load profile\n")
            L.append(_table(["Parameter", "Value"], [[str(k), str(v)] for k, v in params.items()]) + "\n")

        L.append("## Saturation summary\n")
        L.append("> **One leg = one independently streamed audio source** (a simulated call at real-time pace, "
                 "same stream logic as the live server). **Max legs kept up** is the saturation point: the "
                 "highest tested leg count at which every leg stayed within the lag threshold. "
                 "**First failing** is the lowest leg count seen to fail.\n")
        scenarios = results.get("scenarios", [])
        rows = []
        for sc in scenarios:
            ramp = sc.get("ramp")
            rows.append([
                sc["model_id"], sc.get("profile", "dense"), str(sc["vcpus"]), f"{sc['processes']}x{sc['threads_per_process']}",
                _f(sc.get("base_rss_mb", 0), 0) + " MB", _f(sc.get("load_s", 0), 1) + "s",
                str(ramp.l_sat) if ramp else "-",
                str(ramp.l_fail) if ramp and ramp.l_fail is not None else "-",
                ("yes" if ramp.confirmed else "no") if ramp else "-",
                (ramp.stop_reason if ramp else sc.get("status", "")) or sc.get("status", ""),
                (sc.get("error") or (ramp.note if ramp else "") or "").replace("|", "/")[:140],
            ])
        L.append(_table(["Model", "Profile", "CPU threads", "Procs x threads", "Base RSS (all procs)", "Load",
                         "Max legs kept up", "First failing", "Confirmed", "Stopped because", "Note"], rows) + "\n")

        for sc in scenarios:
            ramp = sc.get("ramp")
            if not ramp or not ramp.levels:
                continue
            L.append(f"### {sc['model_id']} [{sc.get('profile', 'dense')}] - {sc['vcpus']} CPU threads, {sc['processes']} process(es) x "
                     f"{sc['threads_per_process']} threads ({sc['stream_mode']})\n")
            rows = []
            for lv in ramp.levels:
                r, res = lv.result, lv.resources
                rows.append([
                    str(lv.n_legs), "yes" if lv.healthy else "NO: " + ", ".join(lv.reasons),
                    f"{r.legs_kept_up}/{lv.n_legs}",
                    _f(r.pass_latency_stats["p50"], 3) + "s", _f(r.pass_latency_stats["p95"], 3) + "s",
                    _f(r.pass_rtf_stats["p95"], 3),
                    _f(r.staleness_stats["p50"], 2) + "s", _f(r.staleness_stats["p95"], 2) + "s",
                    _f(r.staleness_stats["max"], 2) + "s", _f(r.end_lag_stats["max"], 2) + "s",
                    _f(res.get("cpu_pct", {}).get("mean", 0), 0) + "%",
                    _f(res.get("cores_used", 0), 1),
                    _f(lv.cpu_s_per_audio_s, 2),
                    _f(lv.passes_per_s, 1),
                    _f(res.get("tree_rss_mb_peak", 0), 0) + " MB",
                ])
            L.append(_table(["Legs", "Healthy", "Kept up", "Pass P50", "Pass P95", "Pass RTF P95",
                             "Stale P50", "Stale P95", "Stale Max", "End Lag Max", "CPU (pinned set)",
                             "Cores used", "CPU-s per audio-s", "Passes/s", "Peak RSS"], rows) + "\n")
        L.append("> **Cores used** = CPU time consumed by the workers / wall time (includes ORT thread "
                 "spin-wait, so it overstates useful work). **CPU-s per audio-s** = worker CPU seconds per "
                 "second of audio streamed: the per-leg compute cost at that load.\n")

        text = "\n".join(L)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write(text)
        return str(self.summary_path)
