# loadtest/sizing/report.py
"""Render the capacity sizing guide (Markdown) from measured scenarios and a ``SizingAssumptions``."""
from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timezone
from typing import Optional

from loadtest.sizing.model import (
    ModelSizing, Scenario, SizingAssumptions, memory_fit, size_all, size_model,
)

PROFILE_ORDER = ("conversational", "dense")
PROFILE_TEXT = {
    "conversational": "conversational (about half of each call is speech): the planning case",
    "dense": "dense speech (almost continuous talking): the worst-case stress profile",
}


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _f(v: Optional[float], d: int = 2, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "n/a"
    return f"{v:.{d}f}{unit}"


def _gb(v: float) -> str:
    return f"{v:,.0f} GB"


def _profile_rank(p: str) -> int:
    return PROFILE_ORDER.index(p) if p in PROFILE_ORDER else len(PROFILE_ORDER)


def _speech(scs: list[Scenario]) -> str:
    vals = [s.speech_fraction for s in scs if s.speech_fraction]
    return f"{vals[0]:.0%}" if vals else "n/a"


def sizing_table(ms: ModelSizing, a: SizingAssumptions) -> str:
    rows = []
    for r in ms.rows:
        if r.basis == "MEASURED":
            basis = f"MEASURED: fits on one measured edge box ({r.node_vcpus} CPU threads, max {ms.reference.l_sat} legs)"
        else:
            basis = (f"EXTRAPOLATED: {r.legs_per_node:.2f} legs/box = {a.headroom:.0%} of measured max "
                     f"{ms.reference.l_sat} / {a.serving_overhead:.2f} overhead; {r.nodes_base}+{r.spare_nodes} spare "
                     f"boxes; confidence {r.confidence}")
        rows.append([
            f"{r.legs:,}",
            f"{r.nodes:,} ({r.nodes_low:,}-{r.nodes_high:,})",
            str(r.node_vcpus),
            _gb(r.node_ram_gb),
            f"<= {_f(r.target_rtf, 2)}",
            f"<= {_f(r.target_p95_s, 1)} s",
            basis,
        ])
    return _table(
        ["Concurrent legs", "Edge boxes", "CPU threads / box", "RAM / box",
         "Target RTF", "Target P95 latency", "Basis"],
        rows,
    )


def _scenario_rows(scs: list[Scenario]) -> list[list[str]]:
    rows = []
    for s in sorted(scs, key=lambda s: (_profile_rank(s.profile), s.model_id, s.processes, s.vcpus)):
        mf = memory_fit(s)
        slope = _f(mf["slope_mb_per_leg"], 0, " MB") if mf["slope_mb_per_leg"] is not None else "n/a"
        r2 = f" (R2 {mf['r2']:.2f})" if mf["r2"] is not None else ""
        rows.append([
            s.model_id, s.profile, str(s.vcpus), f"{s.processes}x{s.threads}",
            _f(s.base_rss_mb, 0, " MB"), _f(s.load_s, 1, "s"),
            str(s.l_sat) if s.status == "ok" else "-",
            str(s.l_fail) if s.l_fail is not None else "-",
            ("yes" if s.confirmed else "no") if s.status == "ok" else "-",
            slope + r2,
            (s.stop_reason if s.status == "ok" else s.status) + (f": {s.error[:90]}" if s.error else ""),
        ])
    return rows


def _ordered(sized: dict[tuple, ModelSizing]):
    return sorted(sized.items(), key=lambda kv: (_profile_rank(kv[0][1]), kv[0][0]))


def render_sizing_guide(scenarios: list[Scenario], hardware: dict, a: Optional[SizingAssumptions] = None,
                        sources: Optional[list[str]] = None) -> tuple[str, dict]:
    a = a or SizingAssumptions()
    sized = size_all(scenarios, a)
    L: list[str] = []
    L.append("# ASR CPU Capacity Sizing Guide\n")
    L.append(f"**Generated:** {datetime.now(tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  ")
    if sources:
        L.append("**Measured data:** " + ", ".join(f"`{s}`" for s in sources) + "\n")

    L.append("""## 1. How to read this guide

**One concurrent call leg = one independently streamed audio source** (one call direction), transcribed live.
Every number is tagged by where it comes from:

| Tag | Meaning |
|---|---|
| **MEASURED** | read directly from the load test on the test machine (section 3) |
| **DERIVED** | arithmetic or regression on measured numbers (memory per leg, scaling-law fit) |
| **ASSUMED** | an input the load test cannot measure; listed in section 4 and editable in `SizingAssumptions` |
| **EXTRAPOLATED** | a prediction beyond the measured range. **Every row for 50+ legs below is extrapolated**: the test machine saturates at a handful of legs |

Nothing here is `legs x per-leg cost`. Sizes are built from the measured *saturation point* of this edge CPU, then adjusted
for headroom, serving overhead, spare boxes, and memory (section 4). Unmeasured larger chips are not predicted.

Two **load profiles** were measured because speech density changes the cost of a leg a lot: *conversational* (about half the
call is silence, which skips inference) and *dense* (almost continuous speech, a stress case). Read the conversational
table as the planning case and the dense table as the upper bound.
""")

    # ── tables ────────────────────────────────────────────────────────────
    L.append("## 2. Sizing tables (edge CPU)\n")
    L.append("> **Edge boxes** is how many boxes identical to the measured one are needed (central estimate, with "
             "the range if saturation is one leg higher or lower in parentheses), including spare boxes. "
             "**CPU threads / box** is how many logical CPUs the measured run pinned. **RAM / box** is the GB this box needs "
             "(measured weights + per-leg RSS, with RAM headroom and OS reserve, rounded up to a whole GB). "
             "**Target RTF** = P95 real-time factor of one inference pass at the operating point; **Target P95 latency** = "
             "P95 *staleness* (how far the live transcript trails the speaker, queueing included) at that point. Both "
             "targets are the values *measured* at the operating load on the reference layout; the capacity figures only "
             "hold if each box is operated at or below that load.\n")
    for (model, profile), ms in _ordered(sized):
        ref = ms.reference
        scs = [s for s in scenarios if s.key == (model, profile)]
        L.append(f"### {ms.display_name} (`{model}`), {PROFILE_TEXT.get(profile, profile)}\n")
        if not ms.rows:
            L.append("\n".join(f"- {n}" for n in ms.notes) + "\n")
            continue
        L.append(f"**Load profile:** speech about {_speech(scs)} of each call. "
                 f"**Box layout:** {ref.vcpus} CPU threads, {ref.processes} process(es) x {ref.threads} threads "
                 f"({ms.reference_reason}). **Measured:** saturation at **{ref.l_sat} legs/box** "
                 f"(first failing {ref.l_fail if ref.l_fail is not None else 'n/a'}, "
                 f"{'confirmed' if ref.confirmed else 'not re-confirmed'}). "
                 f"**Planning capacity:** {ms.legs_per_node:.2f} legs/box (ASSUMED headroom {a.headroom:.0%}, serving "
                 f"overhead x{a.serving_overhead:.2f}). **RAM / box:** {ms.node_ram_gb:.0f} GB "
                 f"({ms.ram_detail['legs_on_node']} legs x {_f(ms.ram_detail['per_leg_mb'], 0)} MB/leg + "
                 f"{_f(ms.ram_detail['weights_and_buffers_mb'], 0)} MB model/buffers, x{a.ram_headroom:.2f} headroom "
                 f"+ {a.os_reserve_gb:.0f} GB OS = {ms.ram_detail['need_gb']:.1f} GB, rounded up).\n")
        L.append(sizing_table(ms, a) + "\n")
        for n in ms.notes:
            L.append(f"> {n}\n")

    # ── measured ──────────────────────────────────────────────────────────
    L.append("## 3. Measured results (test machine)\n")
    if hardware:
        L.append(f"**Machine:** {hardware.get('cpu_model', '?')}, {hardware.get('physical_cores', '?')} cores / "
                 f"{hardware.get('logical_cores', '?')} threads, {hardware.get('ram_gb', 0):.1f} GB RAM, one socket / one NUMA node, "
                 f"onnxruntime {hardware.get('lib_versions', {}).get('onnxruntime', '?')}.\n")
    L.append("### 3.1 Saturation point per scenario (MEASURED)\n")
    L.append("> Each run pins the listed logical CPUs of this edge machine (hardware-thread sibling pairs, so whole physical "
             "cores; by default all of them, in one process). **Max legs kept up** is the highest simultaneous "
             "leg count at which every leg stayed within the lag threshold, found by ramping then bisecting, then "
             "re-run to confirm. A result of 0 means even one leg could not keep up on that layout. Memory slope is a "
             "DERIVED regression over the healthy levels.\n")
    L.append(_table(["Model", "Profile", "CPU threads", "Procs x threads", "Base RSS (all procs)", "Load", "Max legs kept up",
                     "First failing", "Confirmed", "Memory per leg (DERIVED)", "Stopped because"],
                    _scenario_rows(scenarios)) + "\n")

    for k, ((model, profile), ms) in enumerate(_ordered(sized), start=2):
        scs = sorted([s for s in scenarios if s.key == (model, profile)], key=lambda s: (s.processes, s.vcpus))
        tag = f"`{model}`, {profile}"
        ref = ms.reference
        L.append(f"### 3.{k} {tag} ({ref.label})\n")
        if ref.levels:
            L.append("**Latency vs load.** This is why a box is not run at its saturation point: staleness is flat, "
                     "then climbs steeply.\n")
            rows = []
            for lv in sorted(ref.levels, key=lambda x: x["n_legs"]):
                rows.append([str(lv["n_legs"]), _f(lv["n_legs"] / ref.l_sat, 2) if ref.l_sat else "n/a",
                             "yes" if lv["healthy"] else "no", _f(lv["stale_p95"], 2, " s"), _f(lv["pass_rtf_p95"], 2),
                             _f(lv["end_lag_max"], 2, " s"), _f(lv["cpu_pct"], 0, "%"), _f(lv["cores_used"], 1),
                             _f(lv["rss_peak_mb"], 0, " MB")])
            L.append(_table(["Legs", "Load / saturation", "Healthy", "Stale P95", "Pass RTF P95", "End lag max",
                             "CPU (pinned set)", "Cores used", "Peak RSS"], rows) + "\n")
            if ms.op_level:
                sat_lv = max((lv for lv in ref.levels if lv["healthy"]), key=lambda x: x["n_legs"], default=None)
                L.append(f"Operating point used for the targets: **{ms.op_level['n_legs']} legs** (stale P95 "
                         f"{_f(ms.op_level['stale_p95'], 2, ' s')}, pass RTF P95 {_f(ms.op_level['pass_rtf_p95'], 2)}); "
                         + (f"at the saturation point ({sat_lv['n_legs']} legs) stale P95 was "
                            f"{_f(sat_lv['stale_p95'], 2, ' s')}.\n" if sat_lv else "\n"))

        # Only older result files contain several layouts per model; a default run has one.
        sweep = [s for s in scs if s.processes == 1 and s.status == "ok"]
        if len(sweep) > 1:
            L.append("**More pinned threads on this CPU (MEASURED).** Single process, using that many logical CPUs.\n")
            rows = [[str(s.vcpus), str(s.l_sat)] for s in sweep]
            L.append(_table(["CPU threads pinned", "Max legs kept up (MEASURED)"], rows) + "\n")
            if ms.usl:
                u = ms.usl
                L.append(f"**Universal Scalability Law fit on these measured points only (DERIVED, {u['n_points']} points, "
                         f"RMSE {u['rmse_legs']:.2f} legs):** lambda = {u['lambda']:.3f} legs/thread, contention sigma = "
                         f"{u['sigma']:.3f}, coherency kappa = {u['kappa']:.4f}. This describes diminishing returns on *this* "
                         "chip; it is not used to predict other CPUs.\n")

        procs = [s for s in scs if s.vcpus == max((x.vcpus for x in scs), default=0)]
        if len(procs) > 1:
            L.append(f"**Process strategy at {procs[0].vcpus} CPU threads (MEASURED).** One process shares the weights "
                     "between all legs (least memory, but all legs contend inside one interpreter and one ORT thread pool). "
                     "Several pinned processes duplicate the weights but isolate the contention.\n")
            rows = []
            for s in sorted(procs, key=lambda s: s.processes):
                legs_per_gb = s.l_sat / (s.base_rss_mb / 1024) if s.base_rss_mb and s.status == "ok" else None
                rows.append([f"{s.processes} x {s.threads} threads", _f(s.base_rss_mb, 0, " MB"),
                             str(s.l_sat) if s.status == "ok" else s.status, _f(legs_per_gb, 2),
                             s.error[:100] if s.error else ""])
            L.append(_table(["Layout", "Base RSS (MEASURED)", "Max legs kept up", "Legs per GB of weights", "Note"], rows) + "\n")

    # ── model and assumptions ─────────────────────────────────────────────
    L.append("## 4. Model and assumptions\n")
    L.append("For each model and profile: `legs/box = max(1, measured_saturation x headroom / serving_overhead)`; "
             "`boxes = ceil(legs / legs_per_box) + max(min_spare, ceil(spare_fraction x boxes))`; "
             "`RAM/box = ceil(((processes x weights) + legs_on_box x MB_per_leg) x ram_headroom + OS)` in GB.\n")
    L.append(_table(["Factor", "Treatment", "Value", "Tag"], [
        ["Saturation point", "Highest simultaneous legs that all keep up (p95 staleness and end lag <= threshold), bisected to 1 leg, re-confirmed", "section 3.1", "MEASURED"],
        ["Speech density (profile)", "Silence skips inference, so a conversational leg costs less than a dense one. Both measured; real traffic must be checked against the profile used", "section 2 / 3.1", "MEASURED (profile) + ASSUMED (traffic mix)"],
        ["Headroom / queueing", "Waiting time grows without bound as utilisation approaches 1, so each box is run below the measured knee", f"{a.headroom:.0%} of saturation", "ASSUMED"],
        ["Serving-layer overhead", "WebSocket framing, JSON acks, resampling and process supervision are not in the in-process test", f"x{a.serving_overhead:.2f} CPU", "ASSUMED"],
        ["Spare capacity (N+k)", "A failed or draining box must not push the rest over the knee", f"max({a.min_spare_nodes}, {a.spare_fraction:.0%} of boxes)", "ASSUMED"],
        ["Shared model memory", "Weights are one copy per process and shared by all legs in it; a pinned-process layout multiplies them", "Base RSS, section 3.1", "MEASURED"],
        ["Per-leg memory", "Linear regression of peak RSS vs legs over healthy levels", "section 3.1", "DERIVED"],
        ["RAM headroom / OS", "Allocator fragmentation, page cache and bursts; OS and agents. Rounded up to a whole GB for this box", f"x{a.ram_headroom:.2f}, {a.os_reserve_gb:.0f} GB/box", "ASSUMED"],
        ["Thread contention / diminishing throughput", "Measured as the saturation of this machine (one process, all threads). Extra legs need extra boxes; a bigger chip is not assumed to add legs in proportion", "section 3.1", "MEASURED"],
        ["Process strategy", "Default is one process using the whole machine. Extra processes would duplicate weights; that trade-off is not swept", "1 process", "ASSUMED"],
        ["NUMA", "Test machine is one socket / one NUMA node: cross-socket effects were not measured", "n/a", "ASSUMED"],
        ["Batching across legs", "The engines decode one stream per pass; no cross-leg batching exists, so none is credited. Batching would raise capacity but is unmeasured", "none credited", "NOT MODELLED"],
        ["Scale-out", "Calls are sticky to a box, boxes share nothing, so capacity is additive across identical edge boxes; confidence falls with the extrapolation ratio", "ceil(N / legs per box)", "EXTRAPOLATED"],
        ["Target RTF / P95 latency", "Values measured at the operating load on the measured box", "section 3 (latency vs load)", "MEASURED"],
    ]) + "\n")

    # ── sensitivity ───────────────────────────────────────────────────────
    L.append("## 5. Sensitivity (edge boxes needed, including spares)\n")
    L.append("How much the answer moves when an ASSUMED input or the 1-leg resolution of the measurement changes.\n")
    for (model, profile), ms in _ordered(sized):
        if not ms.rows:
            continue
        probe = [n for n in (100, 1000) if n in a.targets] or [a.targets[-1]]
        variants = [
            ("central", a),
            ("headroom 60%", dataclasses.replace(a, headroom=0.60)),
            ("headroom 80%", dataclasses.replace(a, headroom=0.80)),
            ("serving overhead x1.00", dataclasses.replace(a, serving_overhead=1.00)),
            ("serving overhead x1.25", dataclasses.replace(a, serving_overhead=1.25)),
            ("no spare boxes", dataclasses.replace(a, spare_fraction=0.0, min_spare_nodes=0)),
        ]
        rows = []
        mine = [s for s in scenarios if s.key == (model, profile)]
        for name, av in variants:
            m2 = size_model(mine, av)
            rows.append([name] + [f"{next(r for r in m2.rows if r.legs == n).nodes:,}" for n in probe])
        rows.append(["measured saturation 1 leg higher"] + [f"{next(r for r in ms.rows if r.legs == n).nodes_low:,}" for n in probe])
        rows.append(["measured saturation 1 leg lower"] + [f"{next(r for r in ms.rows if r.legs == n).nodes_high:,}" for n in probe])
        L.append(f"**`{model}`, {profile}**\n")
        L.append(_table(["Variant"] + [f"Boxes @ {n:,} legs" for n in probe], rows) + "\n")

    # ── limitations ───────────────────────────────────────────────────────
    L.append("## 6. Confidence and limitations\n")
    L.append("""**Confidence rubric (judgement, stated so it can be challenged).** Measured = within the legs one test box sustained.
Beyond that confidence falls with the extrapolation ratio `N / measured saturation legs`: <= 10x Medium, <= 50x Low-Medium,
<= 200x Low, above Very low; one level lower when the saturation point is only 1-2 legs, because its 1-leg resolution is then
a +/-33-100% uncertainty on per-box capacity.

- **One machine, one CPU.** A laptop-class 8-core / 16-thread edge CPU with boost and thermal behaviour. Another edge box
  (different frequency, SIMD, cooling) will shift the saturation point. Re-run the load test on the target hardware; the pipeline is the same.
- **Shared test machine.** Anything else running on the test machine (IDE, browser, tight RAM) can only make results worse, not better,
  but it adds noise: a repeat run can land one leg higher or lower.
- **Load profile.** Three clean English read-speech clips looped into 30 s calls. Real calls also have noise, accents, other languages (more decoded
  tokens), overlap and talk ratios that vary by use case; the two profiles bracket speech density but not acoustic difficulty.
- **Scaling beyond one box is assumed, not measured.** Only one machine was available, so the guide treats extra edge boxes as independent
  (calls sticky to a box). Load-balancing skew, correlated peaks and failure domains are covered only by headroom and spares.
- **No network path.** WebSocket, TLS and JSON cost are not in the test; ASSUMED overhead stands in for them.
- **The lag threshold is a design choice.** Saturation is defined as every leg's p95 staleness and end lag <= the threshold in section 3 (2 s by default). Streaming ASR
  has a staleness floor of about 1.0-1.3 s even for one leg (hop + pass time), so the threshold sits close to it; at a 3 s threshold some models
  would sustain one more leg (see the latency-vs-load tables in section 3 for how near the knee the failing level was). Re-run with `slo.lag_threshold_s` set to your product SLO.
- **Saturation resolution is whole legs and was re-confirmed once**, not statistically repeated. The +/-1 leg sensitivity in section 5 is the honest error bar;
  at 1-2 legs per box it is large.
- **Short calls.** 30 s calls; memory growth over hour-long calls was not measured (the streaming code bounds buffers: Qwen force-commits at 15 s, Whisper windows cap at 20 s).
- **No batching credit.** Cross-leg batching of the encoder/decoder would reduce cost per leg but is not implemented here, so it is neither measured nor assumed.
""")
    text = "\n".join(L)
    return text, {f"{m}|{p}": dataclasses.asdict(ms) for (m, p), ms in sized.items()}
