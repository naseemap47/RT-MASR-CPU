# loadtest/tests/test_sizing_model.py
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from loadtest.sizing.model import (
    Scenario, SizingAssumptions, choose_reference, fit_usl, load_raw_files, load_scenarios, memory_fit,
    size_all, size_model, to_jsonable, usl_capacity,
)
from loadtest.sizing.report import render_sizing_guide, sizing_table


def _lv(n, healthy=True, stale=1.0, rtf=0.4, rss=4000.0):
    return {"n_legs": n, "healthy": healthy, "reasons": [], "stale_p50": stale / 2, "stale_p95": stale,
            "stale_max": stale * 1.2, "pass_p95": 0.8, "pass_rtf_p95": rtf, "end_lag_max": 0.5, "cpu_pct": 60.0,
            "cores_used": 4.0, "cpu_s_per_audio_s": 3.0, "rss_peak_mb": rss, "legs_kept_up": n if healthy else 0}


def _sc(vcpus=16, procs=1, l_sat=4, base=3000.0, status="ok", model="m", confirmed=True, levels=None):
    levels = levels if levels is not None else [
        _lv(1, stale=0.9, rtf=0.30, rss=3100), _lv(2, stale=1.1, rtf=0.40, rss=3200),
        _lv(3, stale=1.5, rtf=0.55, rss=3300), _lv(l_sat, stale=1.9, rtf=0.70, rss=3400),
        _lv(l_sat + 1, healthy=False, stale=6.0, rtf=1.4, rss=3600)]
    return Scenario(model_id=model, display_name=model.upper(), stream_mode="vad_utterance", vcpus=vcpus,
                    processes=procs, threads=vcpus // procs, status=status, l_sat=l_sat, l_fail=l_sat + 1,
                    confirmed=confirmed, stop_reason="saturated", base_rss_mb=base, load_s=4.0, levels=levels)


# ── derived measurements ──────────────────────────────────────────────────────

def test_memory_fit_recovers_slope_and_ignores_unhealthy_levels():
    sc = _sc(base=3000.0, levels=[_lv(1, rss=3100), _lv(2, rss=3200), _lv(4, rss=3400),
                                  _lv(9, healthy=False, rss=99999)])
    mf = memory_fit(sc)
    assert mf["slope_mb_per_leg"] == pytest.approx(100.0, rel=1e-6)
    assert mf["intercept_mb"] == pytest.approx(3000.0, rel=1e-6) and mf["r2"] == pytest.approx(1.0)


def test_memory_fit_with_one_point_has_no_slope():
    assert memory_fit(_sc(base=0.0, levels=[_lv(1, rss=3100)]))["slope_mb_per_leg"] is None


def test_usl_fit_recovers_known_parameters_and_never_goes_negative():
    lam, sigma, kappa = 0.5, 0.05, 0.002
    pts = [(v, lam * v / (1 + sigma * (v - 1) + kappa * v * (v - 1))) for v in (1, 2, 4, 8, 16)]
    fit = fit_usl(pts)
    assert fit["lambda"] == pytest.approx(lam, rel=1e-4)
    assert fit["sigma"] == pytest.approx(sigma, rel=1e-3) and fit["kappa"] == pytest.approx(kappa, rel=1e-2)
    assert usl_capacity(fit, 32) < 32 * lam                      # diminishing returns
    linear = fit_usl([(2, 1.0), (4, 2.0), (8, 4.0)])
    assert linear["sigma"] >= 0 and linear["kappa"] >= 0
    assert fit_usl([(4, 1.0)]) is None and fit_usl([(4, 0.0), (8, 0.0)]) is None


def test_choose_reference_prefers_legs_per_thread_and_ignores_coarse_one_leg_shapes():
    a = _sc(vcpus=16, l_sat=4)                                    # 0.25 / thread
    b = _sc(vcpus=8, l_sat=3, base=2500.0)                        # 0.375 / thread  <- best
    c = _sc(vcpus=4, l_sat=1)                                     # 0.25 but only 1 leg: ranked last
    ref, why = choose_reference([a, b, c])
    assert ref is b and "per CPU thread" in why
    assert choose_reference([_sc(l_sat=0), _sc(status="infeasible_memory", l_sat=0)])[0] is None
    only_one, why_one = choose_reference([_sc(vcpus=8, l_sat=1)])
    assert only_one.l_sat == 1 and "this machine" in why_one


# ── sizing arithmetic ─────────────────────────────────────────────────────────

def test_size_model_arithmetic_is_explicit_and_not_a_straight_multiple():
    a = SizingAssumptions(headroom=0.70, serving_overhead=1.0, spare_fraction=0.10, min_spare_nodes=1,
                          targets=(100, 1000))
    ms = size_model([_sc(l_sat=4)], a)
    assert ms.legs_per_node == pytest.approx(2.8)
    r100, r1000 = ms.rows
    assert (r100.nodes_base, r100.spare_nodes, r100.nodes) == (36, 4, 40)
    assert r100.total_vcpus == 40 * 16
    assert (r1000.nodes_base, r1000.spare_nodes, r1000.nodes) == (358, 36, 394)
    assert r100.basis == "EXTRAPOLATED"
    # capacity is NOT legs x (threads / saturation legs): headroom + spares always cost more than that
    naive_nodes = 1000 / 4
    assert r1000.nodes > naive_nodes
    assert r100.nodes_low < r100.nodes < r100.nodes_high            # +-1 leg range brackets the central estimate


def test_serving_overhead_and_headroom_move_the_answer_the_right_way():
    base = size_model([_sc(l_sat=4)], SizingAssumptions(targets=(500,)))
    heavier = size_model([_sc(l_sat=4)], SizingAssumptions(targets=(500,), serving_overhead=1.4))
    tighter = size_model([_sc(l_sat=4)], SizingAssumptions(targets=(500,), headroom=0.5))
    assert heavier.rows[0].nodes > base.rows[0].nodes and tighter.rows[0].nodes > base.rows[0].nodes


def test_one_leg_nodes_cannot_go_below_one_leg_per_node_and_say_so():
    ms = size_model([_sc(l_sat=1, vcpus=8)], SizingAssumptions(targets=(10,)))
    assert ms.legs_per_node == 1.0 and ms.rows[0].nodes_base == 10
    assert any("Headroom could not be applied" in n for n in ms.notes)


def test_measured_basis_when_the_load_fits_on_a_measured_node():
    ms = size_model([_sc(l_sat=4)], SizingAssumptions(targets=(3, 50)))
    assert ms.rows[0].basis == "MEASURED" and ms.rows[0].nodes == 1 and ms.rows[0].confidence.startswith("High")
    assert ms.rows[1].basis == "EXTRAPOLATED"


def test_confidence_falls_with_extrapolation_ratio():
    ms = size_model([_sc(l_sat=8)], SizingAssumptions(targets=(16, 100, 1000, 5000)))
    conf = [r.confidence for r in ms.rows]
    assert conf == ["Medium", "Low-Medium", "Low", "Very low"]
    low = size_model([_sc(l_sat=2)], SizingAssumptions(targets=(10,)))
    assert low.rows[0].confidence == "Low-Medium"                   # 1-2 leg resolution lowers it a notch


def test_ram_is_weights_per_process_plus_per_leg_increment_ceiled_to_whole_gb():
    one = size_model([_sc(l_sat=4, procs=1, base=3000.0)], SizingAssumptions(targets=(50,)))
    two = size_model([_sc(l_sat=4, procs=2, base=6000.0)], SizingAssumptions(targets=(50,)))
    assert two.ram_detail["weights_and_buffers_mb"] == 2 * one.ram_detail["weights_and_buffers_mb"]
    assert two.node_ram_gb >= one.node_ram_gb
    assert one.node_ram_gb == math.ceil(one.ram_detail["need_gb"] - 1e-9)
    assert one.ram_detail["need_gb"] > one.ram_detail["node_rss_mb"] / 1024            # headroom + OS added


def test_targets_come_from_the_measured_operating_point():
    ms = size_model([_sc(l_sat=4)], SizingAssumptions(targets=(100,), serving_overhead=1.0))
    # operating point = ceil(2.8) = 3 legs -> that measured level (stale 1.5, rtf 0.55)
    assert ms.op_level["n_legs"] == 3
    assert ms.rows[0].target_p95_s == pytest.approx(1.5) and ms.rows[0].target_rtf == pytest.approx(0.55)


def test_scenarios_without_a_working_shape_produce_no_rows():
    out = size_all([_sc(l_sat=0, status="ok"), _sc(vcpus=8, l_sat=0)])
    ms = out[("m", "dense")]
    assert ms.rows == [] and ms.notes


# ── reading raw files and rendering ───────────────────────────────────────────

def _raw(l_sat=3):
    def level(n, healthy):
        return {"n_legs": n, "healthy": healthy, "reasons": [] if healthy else ["kept_up=0/%d" % n],
                "legs_per_process": [n], "cpu_s_per_audio_s": 3.0, "passes_per_s": 1.0,
                "resources": {"cpu_pct": {"mean": 60.0}, "cores_used": 4.0, "tree_rss_mb_peak": 3000.0 + 100 * n},
                "result": {"staleness_stats": {"p50": 0.5, "p95": 1.0 + n, "max": 2.0},
                           "pass_latency_stats": {"p95": 0.8}, "pass_rtf_stats": {"p95": 0.3 * n},
                           "end_lag_stats": {"max": 0.5}, "legs_kept_up": n if healthy else 0}}
    return {"hardware": {"cpu_model": "Test CPU", "physical_cores": 8, "logical_cores": 16, "ram_gb": 15.0},
            "scenarios": [
                {"model_id": "m", "display_name": "M", "stream_mode": "vad_utterance", "vcpus": 16, "processes": 1,
                 "threads_per_process": 16, "status": "ok", "base_rss_mb": 3000.0, "load_s": 4.0,
                 "ramp": {"l_sat": l_sat, "l_fail": l_sat + 1, "confirmed": True, "stop_reason": "saturated",
                          "levels": [level(1, True), level(l_sat, True), level(l_sat + 1, False)]}},
                {"model_id": "m", "display_name": "M", "stream_mode": "vad_utterance", "vcpus": 16, "processes": 2,
                 "threads_per_process": 8, "status": "infeasible_memory", "base_rss_mb": 7600.0, "load_s": 4.0,
                 "error": "needs 7600 MB but only 5000 MB free", "ramp": None}]}


def test_load_raw_files_merges_and_reads_infeasible_scenarios(tmp_path):
    p = tmp_path / "x_loadtest_raw.json"
    p.write_text(json.dumps(_raw()))
    scs, hw = load_raw_files([p])
    assert hw["cpu_model"] == "Test CPU" and len(scs) == 2
    assert {s.status for s in scs} == {"ok", "infeasible_memory"}
    assert [s for s in scs if s.status == "ok"][0].l_sat == 3
    assert len(load_scenarios(_raw())[0].levels) == 3


def test_rendered_guide_uses_the_requested_table_and_separates_measured_from_extrapolated():
    scs = load_scenarios(_raw(l_sat=3))
    text, data = render_sizing_guide(scs, _raw()["hardware"], SizingAssumptions(targets=(2, 50, 60, 100, 500, 1000)))
    assert "| Concurrent legs | Edge boxes | CPU threads / box | RAM / box | Target RTF | Target P95 latency | Basis |" in text
    for n in ("50", "60", "100", "500", "1,000"):
        assert f"| {n} |" in text
    assert "MEASURED: fits on one measured edge box" in text and "EXTRAPOLATED:" in text
    assert "Estimated vCPU" not in text and "instance type" not in text and "2 GB per" not in text
    for heading in ("Measured results", "Model and assumptions", "Sensitivity", "Confidence and limitations"):
        assert heading in text
    assert "infeasible_memory" in text                                  # skipped layout is still reported
    assert "m|dense" in data and json.dumps(to_jsonable(data))                 # NaN-safe, serialisable


def test_sizing_table_rows_follow_the_model():
    ms = size_model(load_scenarios(_raw()), SizingAssumptions(targets=(100,)))
    table = sizing_table(ms, SizingAssumptions(targets=(100,)))
    row = table.splitlines()[-1]
    assert row.startswith("| 100 |") and f"{ms.rows[0].nodes:,}" in row and str(ms.rows[0].node_vcpus) in row


def test_guide_handles_a_model_that_never_sustains_a_leg():
    raw = _raw(l_sat=0)
    for sc in raw["scenarios"]:
        if sc["ramp"]:
            sc["ramp"]["levels"] = []
    text, _ = render_sizing_guide(load_scenarios(raw), raw["hardware"], SizingAssumptions())
    assert "no sizing" in text.lower()
