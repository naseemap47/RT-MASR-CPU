# benchmark/reporters/summary_reporter.py
"""
Human-readable Markdown + stdout benchmark summary reporter.

Renders tables for: hardware fingerprint, model load times,
latency/RTF per config and audio file, CPU/memory utilisation,
accuracy (WER/CER), and concurrency scaling behaviour.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _fmt(v: Any, decimals: int = 3) -> str:
    """Format a numeric value for table display."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = float(v)
        if round(v, decimals) == 0:   # avoid "-0" / "-0.000"
            v = 0.0
        return f"{v:.{decimals}f}"
    return str(v)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple Markdown table."""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return f"{header_row}\n{sep}\n{data_rows}"


class SummaryReporter:
    """
    Renders a benchmark summary as a Markdown file and prints to stdout.

    Args:
        output_dir: Directory where the Markdown file is written.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(self, all_results: dict) -> str:
        """
        Render all benchmark results to a Markdown summary.

        Args:
            all_results: Dict with optional keys: hardware, load, latency,
                         accuracy, concurrency.

        Returns:
            Path to the written Markdown file.
        """
        lines: list[str] = []
        ts = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        lines.append("# ASR CPU Benchmark Report\n")
        lines.append(f"**Generated:** {ts}\n")

        # ── Hardware Fingerprint ──────────────────────────────────────────
        hw = all_results.get("hardware", {})
        lines.append("## Hardware & Software Environment\n")
        if hw:
            rows = [
                ["CPU Model",       str(hw.get("cpu_model", ""))],
                ["Physical Cores",  str(hw.get("physical_cores", ""))],
                ["Logical Cores",   str(hw.get("logical_cores", ""))],
                ["RAM",             f"{hw.get('ram_gb', 0):.1f} GB"],
                ["OS",              str(hw.get("os", ""))],
                ["Python",          str(hw.get("python_version", "")).split()[0]],
            ]
            libs = hw.get("lib_versions", {})
            for lib, ver in libs.items():
                rows.append([lib, ver])
            lines.append(_table(["Item", "Value"], rows) + "\n")
        else:
            lines.append("_(no hardware info)_\n")

        # ── Run parameters ────────────────────────────────────────────────
        params = all_results.get("run_params", {})
        if params:
            lines.append("## Run Parameters\n")
            lines.append(_table(["Parameter", "Value"],
                                [[str(k), str(v)] for k, v in params.items()]) + "\n")

        # ── Model Load Times ──────────────────────────────────────────────
        load_results = all_results.get("load", [])
        lines.append("## Cold-Start Model Load Times\n")
        if load_results:
            rows = []
            for lr in load_results:
                cfg = getattr(lr, "config_id", "?")
                lt  = getattr(lr, "load_time_s", 0.0)
                rb  = getattr(lr, "rss_before_mb", 0.0)
                ra  = getattr(lr, "rss_after_mb", 0.0)
                rd  = getattr(lr, "rss_delta_mb", 0.0)
                rows.append([cfg, _fmt(lt, 2)+"s", _fmt(rb, 0)+" MB",
                             _fmt(ra, 0)+" MB", _fmt(rd, 0)+" MB"])
            lines.append(_table(
                ["Config", "Load Time", "RSS Before", "RSS After", "RSS Delta"],
                rows
            ) + "\n")
        else:
            lines.append("_(no load results)_\n")

        # ── Latency & RTF ─────────────────────────────────────────────────
        latency_results = all_results.get("latency", [])
        lines.append("## Latency & Real-Time Factor (RTF)\n")
        lines.append("> RTF = processing time / audio duration. RTF < 1.0 = faster than real-time.\n")
        if latency_results:
            rows = []
            for lr in latency_results:
                cfg  = getattr(lr, "config_id", "?")
                af   = os.path.basename(getattr(lr, "audio_file", "?"))
                dur  = _fmt(getattr(lr, "audio_duration_s", 0.0), 1) + "s"
                ls   = getattr(lr, "latency_stats", {})
                rs   = getattr(lr, "rtf_stats", {})
                rows.append([
                    cfg, af, dur,
                    _fmt(ls.get("mean", 0), 3)+"s",
                    _fmt(ls.get("p50",  0), 3)+"s",
                    _fmt(ls.get("p95",  0), 3)+"s",
                    _fmt(ls.get("p99",  0), 3)+"s",
                    _fmt(rs.get("mean", 0), 3),
                    _fmt(rs.get("p95",  0), 3),
                ])
            lines.append(_table(
                ["Config", "Audio", "Duration",
                 "Lat Mean", "Lat P50", "Lat P95", "Lat P99",
                 "RTF Mean", "RTF P95"],
                rows
            ) + "\n")
        else:
            lines.append("_(no latency results)_\n")

        # ── Overall latency per config ────────────────────────────────────
        lines.append("### Overall per config (all audio files, measured runs)\n")
        lines.append("> Overall RTF = total processing time / total audio time "
                     "(duration-weighted, so long files count more than short ones).\n")
        if latency_results:
            per_cfg: dict[str, dict] = {}
            for lr in latency_results:
                d = per_cfg.setdefault(getattr(lr, "config_id", "?"),
                                       {"lat": 0.0, "aud": 0.0, "n": 0})
                for r in getattr(lr, "raw_results", []):
                    d["lat"] += r.latency_s
                    d["aud"] += r.audio_duration_s
                    d["n"] += 1
            rows = []
            for cfg, d in per_cfg.items():
                rtf = d["lat"] / d["aud"] if d["aud"] > 0 else 0.0
                rows.append([cfg, str(d["n"]), _fmt(d["aud"], 1) + "s",
                             _fmt(d["lat"], 1) + "s", _fmt(rtf, 3)])
            lines.append(_table(
                ["Config", "Calls", "Total Audio", "Total Processing", "Overall RTF"], rows
            ) + "\n")
        else:
            lines.append("_(no latency results)_\n")

        # ── CPU / Memory Under Load ───────────────────────────────────────
        lines.append("## CPU & Memory Utilisation (Latency Runs)\n")
        if latency_results:
            rows = []
            for lr in latency_results:
                cfg = getattr(lr, "config_id", "?")
                af  = os.path.basename(getattr(lr, "audio_file", "?"))
                sm  = getattr(lr, "system_metrics", {})
                cpu = sm.get("overall_cpu_pct", {})
                rss = sm.get("rss_mb", {})
                rows.append([
                    cfg, af,
                    _fmt(cpu.get("mean", 0), 1)+"%",
                    _fmt(cpu.get("p95",  0), 1)+"%",
                    _fmt(rss.get("mean", 0), 0)+" MB",
                    _fmt(sm.get("peak_rss_mb", 0), 0)+" MB",
                    str(sm.get("peak_threads", 0)),
                ])
            lines.append(_table(
                ["Config", "Audio", "CPU Mean", "CPU P95",
                 "RSS Mean", "Peak RSS", "Peak Threads"],
                rows
            ) + "\n")
        else:
            lines.append("_(no CPU/memory data)_\n")

        # ── Accuracy ──────────────────────────────────────────────────────
        accuracy_results = all_results.get("accuracy", [])
        lines.append("## Accuracy (WER / CER)\n")
        lines.append("> WER = word error rate (English, Indonesian). CER = character error rate (Mandarin).\n")
        lines.append("> Normalisation: NFKC + lowercase + strip punctuation (apostrophes deleted) for EN/ID; "
                     "NFKC + strip all punctuation/symbols for ZH.\n")
        lines.append("> Corpus rate = total edits / total reference length (long files weigh more); "
                     "Mean = unweighted average of per-file scores.\n")
        if accuracy_results:
            unverified = any(not getattr(ar, "verified", True) for ar in accuracy_results)
            if unverified:
                lines.append("> **\\* = draft reference transcript (not human-verified). "
                             "Scores against it measure agreement with the draft, not true accuracy.**\n")

            groups: dict[tuple, dict] = {}
            for ar in accuracy_results:
                key = (getattr(ar, "config_id", "?"), getattr(ar, "lang", "?"),
                       str(getattr(ar, "metric", "?")).upper(),
                       bool(getattr(ar, "verified", True)))
                g = groups.setdefault(key, {"err": 0, "ref": 0, "scores": []})
                g["err"] += getattr(ar, "errors", 0)
                g["ref"] += getattr(ar, "ref_len", 0)
                g["scores"].append(getattr(ar, "score", 0.0))
            rows = []
            for (cfg, lang, metric, ok), g in groups.items():
                corpus = g["err"] / g["ref"] if g["ref"] > 0 else 0.0
                mean = sum(g["scores"]) / len(g["scores"])
                rows.append([cfg, lang + ("" if ok else "\\*"), metric, str(len(g["scores"])),
                             _fmt(corpus, 4), _fmt(mean, 4)])
            lines.append("### Aggregate\n")
            lines.append(_table(
                ["Config", "Lang", "Metric", "Files", "Corpus Rate", "Mean Rate"], rows
            ) + "\n")

            rows = []
            for ar in accuracy_results:
                cfg    = getattr(ar, "config_id", "?")
                af     = os.path.basename(getattr(ar, "audio_file", "?"))
                lang   = getattr(ar, "lang", "?") + ("" if getattr(ar, "verified", True) else "\\*")
                det    = getattr(ar, "detected_language", "") or "-"
                metric = str(getattr(ar, "metric", "?")).upper()
                score  = _fmt(getattr(ar, "score", 0.0), 4)
                hyp    = getattr(ar, "normalised_hyp", "")[:60]
                ref    = getattr(ar, "normalised_ref", "")[:60]
                rows.append([cfg, af, lang, det, metric, score, hyp, ref])
            lines.append("### Per file\n")
            lines.append(_table(
                ["Config", "Audio", "Lang", "Detected", "Metric", "Score",
                 "Hypothesis (normalised)", "Reference (normalised)"],
                rows
            ) + "\n")
        else:
            lines.append("_(no accuracy results)_\n")

        # ── Concurrency ───────────────────────────────────────────────────
        concurrency_results = all_results.get("concurrency", [])
        lines.append("## Concurrency Scaling\n")
        lines.append("> Every level runs the same audio workload on one shared engine. "
                     "Throughput = audio seconds transcribed per wall-clock second "
                     "(x real-time, aggregate across all legs). RTF is per call.\n")
        if concurrency_results:
            rows = []
            for cr in concurrency_results:
                cfg   = getattr(cr, "config_id", "?")
                legs  = str(getattr(cr, "n_legs", 0))
                ls    = getattr(cr, "latency_stats", {})
                rs    = getattr(cr, "rtf_stats", {})
                thru  = _fmt(getattr(cr, "throughput_audio_hours_per_wall_hour", 0), 2)
                wall  = _fmt(getattr(cr, "wall_elapsed_s", 0.0), 1) + "s"
                ncall = str(getattr(cr, "n_calls", 0))
                errs  = str(getattr(cr, "error_count", 0))
                sm    = getattr(cr, "system_metrics", {})
                prss  = _fmt(sm.get("peak_rss_mb", 0), 0)
                rows.append([
                    cfg, legs, ncall, wall,
                    _fmt(ls.get("p50",  0), 3)+"s",
                    _fmt(ls.get("p95",  0), 3)+"s",
                    _fmt(ls.get("p99",  0), 3)+"s",
                    _fmt(rs.get("mean", 0), 3),
                    thru+"x",
                    errs,
                    prss+" MB",
                ])
            lines.append(_table(
                ["Config", "Legs", "Calls", "Wall", "Lat P50", "Lat P95", "Lat P99",
                 "RTF Mean", "Throughput", "Errors", "Peak RSS"],
                rows
            ) + "\n")
        else:
            lines.append("_(no concurrency results)_\n")

        # ── Failures ──────────────────────────────────────────────────────
        failures = all_results.get("failures", [])
        if failures:
            lines.append("## Failed Configs\n")
            lines.append(_table(["Config", "Stage", "Error"],
                                [[str(f.get("config_id", "?")), str(f.get("stage", "?")),
                                  str(f.get("error", "")).replace("|", "/")[:200]] for f in failures]) + "\n")

        # ── Write file ────────────────────────────────────────────────────
        ts_file = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{ts_file}_summary.md"
        path = self.output_dir / filename
        content = "\n".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        print(content)
        print(f"\n[reporter] Summary saved: {path}")
        return str(path)
