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

from src.core.runlog import get_logger

logger = get_logger("benchmark.reporter")


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
        stamp: UTC stamp shared with the raw JSON (and the run log). Default: now.
    """

    def __init__(self, output_dir: str, stamp: str | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stamp = stamp or datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ── Concurrency sections ─────────────────────────────────────────────

    @staticmethod
    def _streaming_concurrency_section(results: list) -> list[str]:
        out: list[str] = []
        r0 = results[0]
        out.append(
            "> **One leg = one independently streamed audio source.** Each leg plays its audio "
            f"into one shared engine in {_fmt(getattr(r0, 'chunk_s', 0.5), 1)} s chunks at real-time pace, "
            "using the live server's stream logic (Whisper: sliding window; Qwen3: VAD-cut utterances). "
            "N legs = N calls streaming at once.\n"
        )
        out.append(
            "> **Kept up** = legs whose p95 staleness and end-of-call lag both stayed within "
            f"{_fmt(getattr(r0, 'lag_threshold_s', 2.0), 1)} s, with no errors. "
            "**Staleness** = pass finish time minus arrival time of the newest audio it covered "
            "(how far the transcript trails the speaker, queueing included). "
            "**End lag** = end of audio until the final transcript is committed. "
            "**Pass RTF** = pass latency / audio seconds that pass covered. "
            "**Text** = legs that produced a non-empty transcript (fewer than Legs means the numbers "
            "are suspect).\n"
        )
        if any(abs(getattr(c, "pace", 1.0) - 1.0) > 1e-9 for c in results):
            out.append("> **Warning: not played at real-time pace; latency and lag are not meaningful.**\n")

        rows = []
        for cr in results:
            ps = getattr(cr, "pass_latency_stats", {})
            rf = getattr(cr, "pass_rtf_stats", {})
            st = getattr(cr, "staleness_stats", {})
            ft = getattr(cr, "first_text_stats", {})
            el = getattr(cr, "end_lag_stats", {})
            sm = getattr(cr, "system_metrics", {})
            cpu = sm.get("overall_cpu_pct", {})
            n = getattr(cr, "n_legs", 0)
            rows.append([
                getattr(cr, "config_id", "?"),
                getattr(cr, "stream_mode", "?"),
                str(n),
                f"{getattr(cr, 'legs_kept_up', 0)}/{n}",
                _fmt(ps.get("p50", 0), 3) + "s",
                _fmt(ps.get("p95", 0), 3) + "s",
                _fmt(rf.get("p95", 0), 3),
                _fmt(st.get("p50", 0), 2) + "s",
                _fmt(st.get("p95", 0), 2) + "s",
                _fmt(st.get("max", 0), 2) + "s",
                _fmt(ft.get("mean", 0), 2) + "s",
                _fmt(el.get("max", 0), 2) + "s",
                f"{getattr(cr, 'legs_with_text', 0)}/{n}",
                str(getattr(cr, "error_count", 0)),
                _fmt(cpu.get("mean", 0), 0) + "%",
                _fmt(sm.get("peak_rss_mb", 0), 0) + " MB",
            ])
        out.append(_table(
            ["Config", "Mode", "Legs", "Kept up", "Pass P50", "Pass P95", "Pass RTF P95",
             "Stale P50", "Stale P95", "Stale Max", "First Text (mean)", "End Lag (max)",
             "Text", "Errors", "CPU Mean", "Peak RSS"],
            rows,
        ) + "\n")

        # Capacity per config: highest tested level at which *every* leg kept up
        # (and the levels below it also did).
        per_cfg: dict[str, list] = {}
        for cr in results:
            per_cfg.setdefault(getattr(cr, "config_id", "?"), []).append(cr)
        rows = []
        for cfg_id, crs in per_cfg.items():
            crs = sorted(crs, key=lambda c: getattr(c, "n_legs", 0))
            best = 0
            for cr in crs:
                if getattr(cr, "legs_kept_up", 0) == getattr(cr, "n_legs", 0) and cr.n_legs > 0:
                    best = cr.n_legs
                else:
                    break
            tested = ", ".join(str(getattr(c, "n_legs", 0)) for c in crs)
            rows.append([cfg_id, str(best) if best else "none", tested])
        out.append("### Concurrent live legs supported (all legs kept up)\n")
        out.append(_table(["Config", "Max Legs Kept Up", "Levels Tested"], rows) + "\n")
        return out

    @staticmethod
    def _batch_concurrency_section(results: list) -> list[str]:
        out: list[str] = []
        out.append(
            "> **Batch (offline) mode:** N worker threads call `transcribe()` back to back on one shared "
            "engine (legs x rounds requests). Each leg is a *worker*, not a live stream, and nothing is "
            "paced at real time, so this measures raw capacity, not whether live calls keep up. "
            "Throughput = audio seconds transcribed per wall-clock second (x real-time, aggregate "
            "across all legs). RTF is per request.\n"
        )
        rows = []
        for cr in results:
            cfg   = getattr(cr, "config_id", "?")
            legs  = str(getattr(cr, "n_legs", 0))
            ls    = getattr(cr, "latency_stats", {})
            rs    = getattr(cr, "rtf_stats", {})
            thru  = _fmt(getattr(cr, "throughput_audio_hours_per_wall_hour", 0), 2)
            wall  = _fmt(getattr(cr, "wall_elapsed_s", 0.0), 1) + "s"
            nreq  = str(getattr(cr, "n_calls", 0))
            errs  = str(getattr(cr, "error_count", 0))
            sm    = getattr(cr, "system_metrics", {})
            prss  = _fmt(sm.get("peak_rss_mb", 0), 0)
            rows.append([
                cfg, legs, nreq, wall,
                _fmt(ls.get("p50",  0), 3)+"s",
                _fmt(ls.get("p95",  0), 3)+"s",
                _fmt(ls.get("p99",  0), 3)+"s",
                _fmt(rs.get("mean", 0), 3),
                thru+"x",
                errs,
                prss+" MB",
            ])
        out.append(_table(
            ["Config", "Legs (workers)", "Requests", "Wall", "Lat P50", "Lat P95", "Lat P99",
             "RTF Mean", "Throughput", "Errors", "Peak RSS"],
            rows
        ) + "\n")
        return out

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
        stream_results = [c for c in concurrency_results if hasattr(c, "stream_mode")]
        batch_results = [c for c in concurrency_results if not hasattr(c, "stream_mode")]
        lines.append("## Concurrency Scaling\n")
        if not concurrency_results:
            lines.append("_(no concurrency results)_\n")
        if stream_results:
            lines.extend(self._streaming_concurrency_section(stream_results))
        if batch_results:
            lines.extend(self._batch_concurrency_section(batch_results))

        # ── Failures ──────────────────────────────────────────────────────
        failures = all_results.get("failures", [])
        if failures:
            lines.append("## Failed Configs\n")
            lines.append(_table(["Config", "Stage", "Error"],
                                [[str(f.get("config_id", "?")), str(f.get("stage", "?")),
                                  str(f.get("error", "")).replace("|", "/")[:200]] for f in failures]) + "\n")

        # ── Write file ────────────────────────────────────────────────────
        filename = f"{self.stamp}_summary.md"
        path = self.output_dir / filename
        content = "\n".join(lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info("\n%s", content)
        logger.info("Summary saved: %s", path)
        return str(path)
