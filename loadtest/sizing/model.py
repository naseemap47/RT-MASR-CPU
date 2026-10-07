# loadtest/sizing/model.py
"""
Capacity sizing model: measured load-test data in, deployment sizes out.

Every number the model produces is one of
  MEASURED     read straight from a load-test run on the test machine
  DERIVED      arithmetic on measured numbers (regression, ratios, interpolation)
  ASSUMED      an input the load test cannot measure (stated in ``SizingAssumptions``)
  EXTRAPOLATED a prediction beyond what was measured (anything for hundreds of legs)

It deliberately does NOT compute ``legs x per-leg cost``. Cost per leg is not constant: inference
threads spin-wait, so CPU time per leg *falls* as load rises, while latency collapses suddenly at
saturation. The model therefore sizes from the measured *saturation point* (the highest number of
simultaneous legs that all keep up) on this edge CPU and then applies, explicitly and separately:

  headroom        run each box at a fraction of its measured saturation point (queueing: waiting
                  time diverges as utilisation -> 1, so p95 latency is protected by staying off
                  the knee)
  serving layer   WebSocket / JSON / resampling overhead that the in-process test does not include
  spare boxes     N+k so a box failure or restart does not push the rest past the knee
  memory          weights are shared by all legs in one process (fixed cost per process, measured)
                  plus a per-leg increment (regression on measured levels); RAM is the GB this box
                  actually needs
  scale-up        diminishing returns with more pinned threads on this chip, from the measured core
                  sweep (used only to *describe* the curve; larger unmeasured CPUs are not predicted)
  scale-out       independent identical edge boxes (calls are sticky, nothing is shared): the one
                  place where capacity is taken as additive; its confidence falls with the
                  extrapolation ratio
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ── inputs ────────────────────────────────────────────────────────────────────

@dataclass
class SizingAssumptions:
    """Inputs that are NOT measured. Change them here; every table is recomputed."""
    headroom: float = 0.70               # fraction of the measured saturation point each edge box is run at
    spare_fraction: float = 0.10         # N+k spare boxes as a fraction of the boxes needed ...
    min_spare_nodes: int = 1             # ... but at least this many
    serving_overhead: float = 1.10       # extra CPU for WebSocket/JSON/ack handling (UNMEASURED)
    os_reserve_gb: float = 2.0           # OS + monitoring + serving process, per box
    ram_headroom: float = 1.20           # allocator fragmentation, page cache, bursts
    targets: tuple = (50, 60, 100, 200, 500, 1000)
    min_scale_out_legs: int = 1          # a box always hosts at least one leg


# ── reading measurements ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    """One measured (model, pinned CPU threads, processes) saturation search."""
    model_id: str
    display_name: str
    stream_mode: str
    vcpus: int
    processes: int
    threads: int
    status: str
    l_sat: int
    l_fail: Optional[int]
    confirmed: bool
    stop_reason: str
    base_rss_mb: float                  # all worker processes, after load + warm-up
    load_s: float
    levels: list[dict]                  # flattened healthy/unhealthy level records
    error: str = ""
    note: str = ""
    profile: str = "dense"              # load profile the scenario was measured under
    speech_fraction: Optional[float] = None

    @property
    def key(self) -> tuple:
        return (self.model_id, self.profile)

    @property
    def label(self) -> str:
        return f"{self.vcpus} CPU threads, {self.processes}x{self.threads}"


def _level_record(lv: dict) -> dict:
    r, res = lv["result"], lv.get("resources", {})
    return {
        "n_legs": lv["n_legs"], "healthy": lv["healthy"], "reasons": lv.get("reasons", []),
        "stale_p50": r["staleness_stats"]["p50"], "stale_p95": r["staleness_stats"]["p95"],
        "stale_max": r["staleness_stats"]["max"],
        "pass_p95": r["pass_latency_stats"]["p95"], "pass_rtf_p95": r["pass_rtf_stats"]["p95"],
        "end_lag_max": r["end_lag_stats"]["max"],
        "cpu_pct": res.get("cpu_pct", {}).get("mean", 0.0),
        "cores_used": res.get("cores_used", 0.0),
        "cpu_s_per_audio_s": lv.get("cpu_s_per_audio_s", 0.0),
        "rss_peak_mb": res.get("tree_rss_mb_peak", 0.0),
        "legs_kept_up": r["legs_kept_up"],
    }


def load_scenarios(raw: dict) -> list[Scenario]:
    out = []
    for sc in raw.get("scenarios", []):
        ramp = sc.get("ramp") or {}
        out.append(Scenario(
            model_id=sc["model_id"], display_name=sc.get("display_name", sc["model_id"]),
            stream_mode=sc.get("stream_mode", ""), vcpus=sc["vcpus"], processes=sc["processes"],
            threads=sc["threads_per_process"], status=sc.get("status", "ok"),
            l_sat=int(ramp.get("l_sat", 0)), l_fail=ramp.get("l_fail"),
            confirmed=bool(ramp.get("confirmed", False)), stop_reason=ramp.get("stop_reason", ""),
            base_rss_mb=float(sc.get("base_rss_mb", 0.0)), load_s=float(sc.get("load_s", 0.0)),
            levels=[_level_record(lv) for lv in ramp.get("levels", [])],
            error=sc.get("error", ""), note=ramp.get("note", ""),
            profile=sc.get("profile") or "dense", speech_fraction=sc.get("speech_fraction"),
        ))
    return out


def load_raw_files(paths: list[str | Path]) -> tuple[list[Scenario], dict]:
    """Merge several raw load-test JSON files (later files win for the same model/CPU-threads/process key)."""
    merged: dict[tuple, Scenario] = {}
    hardware: dict = {}
    for p in paths:
        raw = json.loads(Path(p).read_text())
        hardware = raw.get("hardware", hardware)
        for sc in load_scenarios(raw):
            merged[(sc.model_id, sc.profile, sc.vcpus, sc.processes)] = sc
    return list(merged.values()), hardware


# ── derived measurements ──────────────────────────────────────────────────────

def memory_fit(sc: Scenario) -> dict:
    """
    Per-leg RSS growth: least squares of measured peak RSS against leg count over the *healthy* levels
    (plus the idle point after warm-up). ``slope`` is MB per additional concurrent leg inside a process.
    """
    pts = [(0.0, sc.base_rss_mb)] if sc.base_rss_mb > 0 else []
    pts += [(float(lv["n_legs"]), lv["rss_peak_mb"]) for lv in sc.levels if lv["healthy"] and lv["rss_peak_mb"] > 0]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    if len(set(xs.tolist())) < 2:
        return {"slope_mb_per_leg": None, "intercept_mb": sc.base_rss_mb or None, "r2": None, "n_points": len(pts)}
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2)); ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope_mb_per_leg": max(0.0, float(slope)), "intercept_mb": float(intercept), "r2": r2, "n_points": len(pts)}


def fit_usl(points: list[tuple[float, float]]) -> Optional[dict]:
    """
    Universal Scalability Law  C(v) = lam * v / (1 + sigma*(v-1) + kappa*v*(v-1))
    fitted to (pinned CPU threads, saturation legs) points by linear least squares on v/C:

        v/C = a + b*(v-1) + c*v*(v-1),   a = 1/lam, b = sigma/lam, c = kappa/lam   (b, c >= 0)

    sigma = contention (serialisation), kappa = coherency (cross-talk) cost. Needs >= 2 points with C > 0;
    with 3 or fewer points the fit is only a description of the curve, not a validated model.
    """
    pts = [(float(v), float(c)) for v, c in points if c > 0]
    if len(pts) < 2:
        return None
    v = np.array([p[0] for p in pts]); c = np.array([p[1] for p in pts])
    y = v / c

    def solve(cols: list[np.ndarray]) -> np.ndarray:
        A = np.column_stack(cols)
        return np.linalg.lstsq(A, y, rcond=None)[0]

    one = np.ones_like(v)
    terms = [one, v - 1.0, v * (v - 1.0)]
    use = [0, 1, 2] if len(pts) >= 3 else [0, 1]
    coef = np.zeros(3)
    while True:
        sol = solve([terms[i] for i in use])
        coef = np.zeros(3)
        for i, s in zip(use, sol):
            coef[i] = s
        neg = [i for i in use if i > 0 and coef[i] < 0]
        if not neg:
            break
        use = [i for i in use if i not in neg]
    a, b, k = coef
    if a <= 0:
        return None
    lam, sigma, kappa = 1.0 / a, b / a, k / a
    pred = lam * v / (1 + sigma * (v - 1) + kappa * v * (v - 1))
    rmse = float(np.sqrt(np.mean((pred - c) ** 2)))
    return {"lambda": float(lam), "sigma": float(sigma), "kappa": float(kappa), "n_points": len(pts),
            "rmse_legs": rmse, "points": [(float(a_), float(b_)) for a_, b_ in pts]}


def usl_capacity(fit: dict, vcpus: float) -> float:
    v = float(vcpus)
    return fit["lambda"] * v / (1 + fit["sigma"] * (v - 1) + fit["kappa"] * v * (v - 1))


def _ceil_to(value: float, step: float) -> float:
    return math.ceil(value / step - 1e-9) * step


# ── sizing ────────────────────────────────────────────────────────────────────

@dataclass
class SizingRow:
    legs: int
    nodes_base: int
    spare_nodes: int
    nodes: int
    nodes_low: int                       # optimistic: saturation point 1 leg higher
    nodes_high: int                      # pessimistic: saturation point 1 leg lower
    node_vcpus: int                      # logical CPUs pinned on one edge box
    total_vcpus: int                     # boxes x pinned threads (kept for JSON; not shown as a fleet metric)
    node_ram_gb: float                   # GB this box needs (ceil of measured need)
    total_ram_gb: float
    legs_per_node: float
    target_rtf: float
    target_p95_s: float
    basis: str                           # "MEASURED" | "EXTRAPOLATED"
    confidence: str
    extrapolation_ratio: float


@dataclass
class ModelSizing:
    model_id: str
    display_name: str
    profile: str
    reference: Scenario                  # the node shape the sizing is built on
    reference_reason: str
    legs_per_node: float                 # planning capacity per node (after headroom + overhead)
    op_level: Optional[dict]             # measured level closest to the operating point
    node_ram_gb: float
    ram_detail: dict
    usl: Optional[dict]                  # fit on measured core-sweep points only; not used to predict other CPUs
    rows: list[SizingRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def choose_reference(scenarios: list[Scenario]) -> tuple[Optional[Scenario], str]:
    """
    Pick the layout to size from. A default load-test run has one layout (this whole machine).
    If older result files still contain several pin-widths, the best legs-per-thread among shapes
    that sustain >= 2 legs is used (a 1-leg result is too coarse to rank).
    """
    ok = [s for s in scenarios if s.status == "ok" and s.l_sat >= 1 and s.confirmed]
    if not ok:
        ok = [s for s in scenarios if s.status == "ok" and s.l_sat >= 1]
    if not ok:
        return None, "no scenario sustained even one leg"
    if len(ok) == 1:
        return ok[0], "the measured layout on this machine"
    pool = [s for s in ok if s.l_sat >= 2] or ok
    best = max(pool, key=lambda s: (round(s.l_sat / s.vcpus, 6), -s.base_rss_mb, -s.processes, s.vcpus))
    reason = (f"highest measured saturation legs per CPU thread ({best.l_sat}/{best.vcpus} = "
              f"{best.l_sat / best.vcpus:.3f}) among shapes sustaining >=2 legs"
              if best.l_sat >= 2 else "only shape that sustained a leg")
    return best, reason


def _op_level(ref: Scenario, op_legs: int) -> Optional[dict]:
    healthy = [lv for lv in ref.levels if lv["healthy"] and lv["n_legs"] <= max(1, op_legs)]
    return max(healthy, key=lambda lv: lv["n_legs"]) if healthy else None


def _confidence(ratio: float, ref: Scenario) -> str:
    if ratio <= 1:
        return "High (measured)"
    base = "Medium" if ratio <= 10 else "Low-Medium" if ratio <= 50 else "Low" if ratio <= 200 else "Very low"
    if ref.l_sat <= 2 and base in ("Medium", "Low-Medium"):
        return {"Medium": "Low-Medium", "Low-Medium": "Low"}[base]      # 1-leg resolution dominates
    return base


def _nodes(legs: int, cap: float, a: SizingAssumptions) -> tuple[int, int, int]:
    base = math.ceil(legs / max(cap, 1e-9))
    spare = max(a.min_spare_nodes, math.ceil(a.spare_fraction * base))
    return base, spare, base + spare


def size_model(scenarios: list[Scenario], a: SizingAssumptions) -> Optional[ModelSizing]:
    if not scenarios:
        return None
    ref, reason = choose_reference(scenarios)
    first = scenarios[0]
    if ref is None:
        return ModelSizing(first.model_id, first.display_name, first.profile, first, reason, 0.0, None, 0.0, {}, None,
                           notes=["No measured configuration sustained a single leg within the SLO: no sizing."])

    def cap_for(l_sat: float) -> float:
        # headroom + serving overhead applied to the saturation point; never below 1 leg per box
        return max(float(a.min_scale_out_legs), l_sat * a.headroom / a.serving_overhead)

    cap = cap_for(ref.l_sat)
    cap_hi = cap_for(ref.l_sat + 1)              # saturation really lies in [l_sat, l_fail): optimistic end
    cap_lo = cap_for(max(1, ref.l_sat - 1))      # run-to-run variance: pessimistic end
    op_legs = max(1, math.ceil(cap))
    op = _op_level(ref, op_legs)

    mem = memory_fit(ref)
    slope = mem["slope_mb_per_leg"] if mem["slope_mb_per_leg"] is not None else 0.0
    per_node_mb = ref.base_rss_mb + slope * math.ceil(cap)
    need_gb = (per_node_mb / 1024.0) * a.ram_headroom + a.os_reserve_gb
    node_ram = float(math.ceil(need_gb - 1e-9))                  # whole GB this edge box needs
    ram_detail = {"weights_and_buffers_mb": ref.base_rss_mb, "per_leg_mb": slope,
                  "legs_on_node": math.ceil(cap), "node_rss_mb": per_node_mb, "need_gb": need_gb,
                  "fit": mem}

    sweep = [(s.vcpus, s.l_sat) for s in scenarios if s.processes == 1 and s.status == "ok"]
    usl = fit_usl(sweep)

    target_rtf = _ceil_to(op["pass_rtf_p95"], 0.05) if op else float("nan")
    target_p95 = _ceil_to(op["stale_p95"], 0.1) if op else float("nan")

    ms = ModelSizing(first.model_id, first.display_name, first.profile, ref, reason, cap, op, float(node_ram),
                     ram_detail, usl)
    for n in a.targets:
        base, spare, total = _nodes(n, cap, a)
        _, _, total_hi = _nodes(n, cap_hi, a)
        _, _, total_lo = _nodes(n, cap_lo, a)
        measured = n <= ref.l_sat
        ratio = n / max(1, ref.l_sat)
        if measured:
            base, spare, total, total_hi, total_lo = 1, 0, 1, 1, 1
        ms.rows.append(SizingRow(
            legs=n, nodes_base=base, spare_nodes=spare, nodes=total, nodes_low=total_hi, nodes_high=total_lo,
            node_vcpus=ref.vcpus, total_vcpus=total * ref.vcpus, node_ram_gb=float(node_ram),
            total_ram_gb=float(node_ram) * total, legs_per_node=min(cap, n) if measured else cap,
            target_rtf=target_rtf, target_p95_s=target_p95,
            basis="MEASURED" if measured else "EXTRAPOLATED",
            confidence=_confidence(ratio, ref), extrapolation_ratio=ratio,
        ))
    if cap <= a.min_scale_out_legs + 1e-9:
        ms.notes.append("Headroom could not be applied: a box barely sustains one leg, so each box hosts a single leg "
                        "and capacity is not additive in a meaningful way below that.")
    return ms


def size_all(scenarios: list[Scenario], a: Optional[SizingAssumptions] = None) -> dict[tuple, ModelSizing]:
    """Size every (model, load profile) pair; keys are ``(model_id, profile)``."""
    a = a or SizingAssumptions()
    groups: dict[tuple, list[Scenario]] = {}
    for s in scenarios:
        groups.setdefault(s.key, []).append(s)
    out = {}
    for key, scs in groups.items():
        ms = size_model(sorted(scs, key=lambda s: (s.vcpus, s.processes)), a)
        if ms is not None:
            out[key] = ms
    return out


def to_jsonable(obj: Any) -> Any:
    from dataclasses import is_dataclass
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj
