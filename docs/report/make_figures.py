# docs/report/make_figures.py
"""
Regenerate the figures of docs/report/technical_report.md from the final result files.

    python3 docs/report/make_figures.py

Needs only matplotlib + the JSON files below (no model, no project imports).
"""
from __future__ import annotations

import collections
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(ROOT, "benchmark/results/20261005T173127Z_raw.json")
LOAD = os.path.join(ROOT, "loadtest/results/20261007T202154Z_loadtest_raw.json")
SIZING = os.path.join(ROOT, "loadtest/results/20261007T203650Z_sizing.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# INT4 0.6B returned 0 tokens on this clip in the benchmark (auto language); excluded where marked.
BROKEN = ("qwen3_onnx_int4_0.6b", "librispeech_0_1089_0.wav")

SHORT = {
    "qwen3_onnx_int4_0.6b": "ONNX INT4 0.6B",
    "qwen3_onnx_int8_0.6b": "ONNX INT8 0.6B",
    "qwen3_onnx_int4_1.7b": "ONNX INT4 1.7B",
    "qwen3_onnx_fp32_0.6b": "ONNX FP32 0.6B",
    "qwen3_transformers_bf16_0.6b": "HF BF16 0.6B",
    "qwen3_transformers_bf16_1.7b": "HF BF16 1.7B",
    "whisper_int8_tiny": "Whisper tiny",
    "whisper_int8_base": "Whisper base",
    "whisper_int8_small": "Whisper small",
    "whisper_int8_medium": "Whisper medium",
}
ORDER = list(SHORT)


def colour(cfg: str) -> str:
    if cfg.startswith("qwen3_onnx"):
        return "#2b6cb0"
    if cfg.startswith("qwen3_transformers"):
        return "#90cdf4"
    return "#dd6b20"


def is_broken(cfg: str, audio_file: str) -> bool:
    return cfg == BROKEN[0] and audio_file.endswith(BROKEN[1])


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)


def bench_figures(b: dict) -> None:
    # Overall RTF (duration-weighted), INT4 0.6B without the broken clip.
    proc, aud = collections.Counter(), collections.Counter()
    for r in b["latency"]:
        if is_broken(r["config_id"], r["audio_file"]):
            continue
        for x in r["raw_results"]:
            proc[r["config_id"]] += x["latency_s"]
            aud[r["config_id"]] += x["audio_duration_s"]
    rtf = {c: proc[c] / aud[c] for c in ORDER}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar([SHORT[c] for c in ORDER], [rtf[c] for c in ORDER], color=[colour(c) for c in ORDER])
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.text(len(ORDER) - 0.5, 1.08, "real time (RTF 1.0)", color="red", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("Overall RTF (log scale, lower is faster)")
    ax.set_title("Single-request speed: duration-weighted RTF over 7 files (EN/ZH/ID)")
    for i, c in enumerate(ORDER):
        ax.text(i, rtf[c] * 1.08, f"{rtf[c]:.2f}", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "fig1_rtf_overall.png")

    # Accuracy vs speed (EN WER corpus, INT4 0.6B corrected; ZH CER).
    edits, refs = collections.Counter(), collections.Counter()
    for a in b["accuracy"]:
        if is_broken(a["config_id"], a["audio_file"]):
            continue
        k = (a["config_id"], a["lang"])
        edits[k] += a["errors"]
        refs[k] += a["ref_len"]
    markers = dict(zip(ORDER, "oDs^vPXo^s"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, lang, label in ((axes[0], "en", "EN WER (verified references)"),
                            (axes[1], "zh", "ZH CER (draft references)")):
        for c in ORDER:
            y = edits[(c, lang)] / refs[(c, lang)]
            ax.scatter(rtf[c], y, color=colour(c), marker=markers[c], s=60, zorder=3,
                       edgecolor="black", linewidth=0.5, label=SHORT[c])
        ax.axvline(1.0, color="red", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("Overall RTF (log)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Accuracy vs speed (lower-left is better; INT4 0.6B without the empty-output clip)")
    save(fig, "fig2_accuracy_vs_rtf.png")

    # Peak RSS per config.
    peak = collections.defaultdict(float)
    cpu = collections.defaultdict(list)
    for r in b["latency"]:
        peak[r["config_id"]] = max(peak[r["config_id"]], r["system_metrics"]["peak_rss_mb"])
        cpu[r["config_id"]].append(r["system_metrics"]["overall_cpu_pct"]["mean"])
    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    x = range(len(ORDER))
    ax1.bar(x, [peak[c] / 1024 for c in ORDER], color=[colour(c) for c in ORDER])
    ax1.set_ylabel("Peak RSS during latency runs (GB)")
    ax1.set_xticks(list(x), [SHORT[c] for c in ORDER], rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(list(x), [sum(cpu[c]) / len(cpu[c]) for c in ORDER], "ko-", ms=4)
    ax2.set_ylabel("Mean machine CPU % (black line)")
    ax2.set_ylim(0, 100)
    ax1.set_title("Memory and CPU utilisation (single request)")
    save(fig, "fig3_memory_cpu.png")

    # Stage breakdown.
    stages = ("mel_s", "encoder_s", "prefill_s", "decode_s")
    tot = collections.defaultdict(collections.Counter)
    for r in b["latency"]:
        if is_broken(r["config_id"], r["audio_file"]):
            continue
        for x_ in r["raw_results"]:
            td = x_.get("timing_detail") or {}
            for s in stages:
                tot[r["config_id"]][s] += td.get(s) or 0.0
            tot[r["config_id"]]["lat"] += x_["latency_s"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = [0.0] * len(ORDER)
    colours = {"mel_s": "#a0aec0", "encoder_s": "#38a169", "prefill_s": "#d69e2e", "decode_s": "#c53030"}
    for s in stages:
        vals = [100 * tot[c][s] / tot[c]["lat"] for c in ORDER]
        ax.bar([SHORT[c] for c in ORDER], vals, bottom=bottom, color=colours[s], label=s[:-2])
        bottom = [a + v for a, v in zip(bottom, vals)]
    ax.bar([SHORT[c] for c in ORDER], [100 - v for v in bottom], bottom=bottom, color="#edf2f7", label="other")
    ax.set_ylabel("% of request latency")
    ax.set_title("Where the time goes (single request)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "fig4_stage_breakdown.png")

    # Batch-mode concurrency throughput.
    fig, ax = plt.subplots(figsize=(9, 4.2))
    series = collections.defaultdict(list)
    for r in b["concurrency"]:
        series[r["config_id"]].append((r["n_legs"], r["throughput_audio_hours_per_wall_hour"]))
    palette = plt.get_cmap("tab10")
    for i, c in enumerate(ORDER):
        if c == BROKEN[0]:
            continue
        pts = sorted(series[c])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=palette(i), label=SHORT[c])
        ax.text(4.08, pts[-1][1], f"{pts[-1][1]:.2f}x", fontsize=7, va="center")
    ax.axhline(1.0, color="red", ls="--", lw=1)
    ax.set_xticks([1, 2, 4])
    ax.set_yscale("log")
    ax.set_xlabel("Workers calling transcribe() back to back")
    ax.set_ylabel("Throughput (x real time, log)")
    ax.set_title("Batch-mode concurrency (offline throughput, not live capacity)")
    ax.set_xlim(0.8, 4.6)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(alpha=0.3)
    save(fig, "fig5_batch_throughput.png")


def load_figures(lt: dict, b: dict, sizing: dict) -> None:
    style = {"qwen3_onnx_int8_0.6b": "#2b6cb0", "qwen3_onnx_int4_0.6b": "#63b3ed", "whisper_int8_tiny": "#dd6b20"}

    # P95 staleness vs legs (first run of each level).
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, profile in zip(axes, ("conversational", "dense")):
        for s in lt["scenarios"]:
            if s["profile"] != profile:
                continue
            seen = {}
            for lv in s["ramp"]["levels"]:
                seen.setdefault(lv["n_legs"], lv["result"]["staleness_stats"]["p95"])
            pts = sorted(seen.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=style[s["model_id"]],
                    label=f"{SHORT[s['model_id']]} (max {s['ramp']['l_sat']})")
        ax.axhline(2.0, color="red", ls="--", lw=1)
        ax.text(1, 2.1, "2 s lag threshold", color="red", fontsize=8)
        ax.set_title(f"{profile} profile")
        ax.set_xlabel("Simultaneous live legs on one 16-thread box")
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pooled P95 staleness (s)")
    fig.suptitle("Live streaming: how far the transcript trails the speaker as legs are added")
    save(fig, "fig6_staleness_vs_legs.png")

    # Streaming cost per audio second vs batch RTF of the same EN clips.
    proc, aud = collections.Counter(), collections.Counter()
    for r in b["latency"]:
        if "librispeech" not in r["audio_file"] or is_broken(r["config_id"], r["audio_file"]):
            continue
        for x in r["raw_results"]:
            proc[r["config_id"]] += x["latency_s"]
            aud[r["config_id"]] += x["audio_duration_s"]
    cost = {}
    for s in lt["scenarios"]:
        lv = next(lv for lv in s["ramp"]["levels"] if lv["n_legs"] == 1)
        r = lv["result"]
        busy = sum(sum(leg["raw_pass_latencies"]) for leg in r["legs"])
        cost[(s["model_id"], s["profile"])] = busy / r["total_audio_s"]
    models = list(style)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.27
    for i, m in enumerate(models):
        ax.bar(i - w, proc[m] / aud[m], w, color="#a0aec0", label="batch RTF (whole clip once)" if i == 0 else None)
        ax.bar(i, cost[(m, "conversational")], w, color="#63b3ed", label="streaming, conversational" if i == 0 else None)
        ax.bar(i + w, cost[(m, "dense")], w, color="#2b6cb0", label="streaming, dense" if i == 0 else None)
    ax.set_xticks(range(len(models)), [SHORT[m] for m in models])
    ax.set_ylabel("Inference seconds per audio second")
    ax.set_title("Cost of one live leg vs one batch transcription (same English clips)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig7_stream_vs_batch_cost.png")

    # Sizing: boxes vs legs, conversational.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    labels = {"qwen3_onnx_int8_0.6b": "Qwen3 0.6B INT8 = INT4 (1 leg/box)",
              "whisper_int8_tiny": "Whisper tiny (1.91 legs/box)"}
    for m in labels:
        rows = sizing["models"][f"{m}|conversational"]["rows"]
        xs = [r["legs"] for r in rows]
        ax.plot(xs, [r["nodes"] for r in rows], "o-", color=style[m], label=labels[m])
        ax.fill_between(xs, [r["nodes_low"] for r in rows], [r["nodes_high"] for r in rows],
                        color=style[m], alpha=0.12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Concurrent live legs (log)")
    ax.set_ylabel("Edge boxes incl. spares (log)")
    ax.set_title("EXTRAPOLATED sizing, conversational profile (band = saturation +/- 1 leg)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    save(fig, "fig8_sizing.png")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(BENCH) as f:
        b = json.load(f)
    with open(LOAD) as f:
        lt = json.load(f)
    with open(SIZING) as f:
        sizing = json.load(f)
    bench_figures(b)
    load_figures(lt, b, sizing)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
