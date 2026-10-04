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
    if isinstance(v, float):
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
        lines.append(f"# Qwen3-ASR CPU Benchmark Report\n")
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
        lines.append("> Normalisation: NFKC + lowercase + strip punctuation for EN/ID; NFKC + strip CJK punct for ZH.\n")
        if accuracy_results:
            rows = []
            for ar in accuracy_results:
                cfg    = getattr(ar, "config_id", "?")
                af     = os.path.basename(getattr(ar, "audio_file", "?"))
                lang   = getattr(ar, "lang", "?")
                metric = getattr(ar, "metric", "?").upper()
                score  = _fmt(getattr(ar, "score", 0.0), 4)
                hyp    = getattr(ar, "normalised_hyp", "")[:60]
                ref    = getattr(ar, "normalised_ref", "")[:60]
                rows.append([cfg, af, lang, metric, score, hyp, ref])
            lines.append(_table(
                ["Config", "Audio", "Lang", "Metric", "Score",
                 "Hypothesis (normalised)", "Reference (normalised)"],
                rows
            ) + "\n")
        else:
            lines.append("_(no accuracy results)_\n")

        # ── Concurrency ───────────────────────────────────────────────────
        concurrency_results = all_results.get("concurrency", [])
        lines.append("## Concurrency Scaling\n")
        if concurrency_results:
            rows = []
            for cr in concurrency_results:
                cfg   = getattr(cr, "config_id", "?")
                legs  = str(getattr(cr, "n_legs", 0))
                ls    = getattr(cr, "latency_stats", {})
                rs    = getattr(cr, "rtf_stats", {})
                thru  = _fmt(getattr(cr, "throughput_audio_hours_per_wall_hour", 0), 3)
                errs  = str(getattr(cr, "error_count", 0))
                sm    = getattr(cr, "system_metrics", {})
                prss  = _fmt(sm.get("peak_rss_mb", 0), 0)
                rows.append([
                    cfg, legs,
                    _fmt(ls.get("p50",  0), 3)+"s",
                    _fmt(ls.get("p95",  0), 3)+"s",
                    _fmt(ls.get("p99",  0), 3)+"s",
                    _fmt(rs.get("mean", 0), 3),
                    thru+" audio-hr/hr",
                    errs,
                    prss+" MB",
                ])
            lines.append(_table(
                ["Config", "Legs", "Lat P50", "Lat P95", "Lat P99",
                 "RTF Mean", "Throughput", "Errors", "Peak RSS"],
                rows
            ) + "\n")
        else:
            lines.append("_(no concurrency results)_\n")

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
