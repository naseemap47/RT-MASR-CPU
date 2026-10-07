# loadtest/tests/test_loadtest_core.py
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.runners.streaming_concurrency_runner import StreamingConcurrencyResult, StreamLegResult, _stats
from loadtest.audio import tile_call_audio
from loadtest.runners.ramp import evaluate_level, merge_level_results, run_ramp
from loadtest.runners.worker_pool import MemoryAbort, WorkerPool
from loadtest.run_loadtest import expand_scenarios
from loadtest.topology import cpu_set, ordered_cpus, split_cpus


# ── topology ──────────────────────────────────────────────────────────────────

def test_cpu_set_and_split_are_disjoint_and_sized():
    cpus = ordered_cpus()
    n = len(cpus)
    assert sorted(cpus) == sorted(set(cpus))
    sel = cpu_set(min(4, n))
    assert len(sel) == min(4, n)
    if n >= 4:
        groups = split_cpus(cpu_set(4), 2)
        assert [len(g) for g in groups] == [2, 2]
        assert not set(groups[0]) & set(groups[1])
        assert sorted(groups[0] + groups[1]) == sel


def test_expand_scenarios_is_one_run_per_model_and_profile():
    cfg = {"profiles": {"dense": {}, "conversational": {}},
           "models": [{"id": "a"}, {"id": "b"}]}
    runs = expand_scenarios(cfg, n_cpus=16, models=None, processes=1)
    assert [(r["profile"], r["model_id"], r["vcpus"], r["processes"]) for r in runs] == [
        ("dense", "a", 16, 1), ("dense", "b", 16, 1),
        ("conversational", "a", 16, 1), ("conversational", "b", 16, 1),
    ]
    only = expand_scenarios(cfg, n_cpus=8, models=["b"], processes=1, profiles=["conversational"])
    assert only == [{"model_id": "b", "profile": "conversational", "vcpus": 8, "processes": 1}]


def test_cpu_set_rejects_impossible_requests():
    with pytest.raises(ValueError):
        cpu_set(10_000)
    with pytest.raises(ValueError):
        split_cpus([0, 1], 3)


# ── call audio ────────────────────────────────────────────────────────────────

def test_tile_call_audio_length_and_pauses():
    sr = 16000
    clip = np.ones(sr * 2, dtype=np.float32)               # 2 s of "speech"
    out = tile_call_audio(clip, duration_s=10.0, gap_s=1.0)
    assert len(out) == 10 * sr
    assert out[: 2 * sr].min() == 1.0 and out[2 * sr: 3 * sr].max() == 0.0     # clip, then 1 s pause
    assert out[3 * sr: 5 * sr].min() == 1.0                                    # repeated
    with pytest.raises(ValueError):
        tile_call_audio(clip, 0.0, 1.0)


# ── level merge / evaluation / ramp ──────────────────────────────────────────

def _leg(i, ok=True, text="hi", aborted=False, errors=0):
    return StreamLegResult(leg=i, audio_file="a.wav", audio_s=30.0, passes=5, errors=errors,
                           first_text_s=1.0, end_lag_s=0.5, p95_staleness_s=1.0, max_staleness_s=1.5,
                           kept_up=ok, aborted=aborted, final_text=text,
                           raw_pass_latencies=[0.5] * 5, raw_pass_rtfs=[0.2] * 5, raw_staleness=[1.0] * 5)


def _result(legs):
    return StreamingConcurrencyResult(
        config_id="m", n_legs=len(legs), stream_mode="vad_utterance", chunk_s=0.5, pace=1.0, stagger_s=0.0,
        lag_threshold_s=2.0, pass_latency_stats=_stats([0.5]), pass_rtf_stats=_stats([0.2]),
        staleness_stats=_stats([1.0]), first_text_stats=_stats([1.0]), end_lag_stats=_stats([0.5]),
        legs_kept_up=sum(g.kept_up for g in legs), legs_with_text=sum(bool(g.final_text) for g in legs),
        error_count=sum(g.errors for g in legs), system_metrics={}, wall_elapsed_s=30.0,
        legs_aborted=sum(g.aborted for g in legs),
        total_audio_s=30.0 * len(legs), total_passes=5 * len(legs), legs=legs)


def test_merge_level_results_pools_slices():
    a, b = _result([_leg(0), _leg(1)]), _result([_leg(2)])
    m = merge_level_results([b, a])
    assert m.n_legs == 3 and [g.leg for g in m.legs] == [0, 1, 2]
    assert m.legs_kept_up == 3 and m.pass_latency_stats["count"] == 15 and m.total_passes == 15


def test_evaluate_level_reasons():
    assert evaluate_level(_result([_leg(0), _leg(1)])) == (True, [])
    ok, why = evaluate_level(_result([_leg(0), _leg(1, ok=False), _leg(2, text="")]))
    assert not ok and any(r.startswith("kept_up=") for r in why) and any(r.startswith("empty_transcript") for r in why)
    ok, why = evaluate_level(_result([_leg(0, ok=False, aborted=True)]))
    assert not ok and any(r.startswith("overload_abort") for r in why)


class _FakePool:
    """Stands in for WorkerPool: every leg keeps up iff n_legs <= capacity."""
    def __init__(self, capacity, n_procs=2, rss_per_leg=100.0, abort_at=None):
        self.capacity, self.n_procs, self.rss_per_leg, self.abort_at = capacity, n_procs, rss_per_leg, abort_at
        self.pids, self.cpus, self.calls = [os.getpid()], [0], []

    def split_legs(self, n):
        base, rem = divmod(n, self.n_procs)
        return [base + (1 if i < rem else 0) for i in range(self.n_procs)]

    def run_level(self, n, stagger_s, timeout_s, lead_s=1.0):
        self.calls.append(n)
        if self.abort_at is not None and len(self.calls) > self.abort_at:
            raise MemoryAbort("boom")
        ok = n <= self.capacity
        out, off = [], 0
        for c in self.split_legs(n):
            if c:
                out.append(_result([_leg(off + i, ok=ok) for i in range(c)]))
            off += c
        return out


def _ramp(pool, levels, **kw):
    kw.setdefault("spread_s", 0.0)
    kw.setdefault("call_duration_s", 1.0)
    kw.setdefault("abort_lag_s", 1.0)
    return run_ramp(pool, levels, log=lambda *_: None, **kw)


def test_ramp_finds_exact_saturation_point_by_bisection():
    pool = _FakePool(capacity=5)
    r = _ramp(pool, [1, 2, 4, 8, 16])
    assert (r.l_sat, r.l_fail, r.stop_reason, r.confirmed) == (5, 6, "saturated", True)
    assert pool.calls[:4] == [1, 2, 4, 8]            # ladder stops at the first failure
    assert 6 in pool.calls and 5 in pool.calls        # bisected, then confirmed
    assert 16 not in pool.calls
    assert all(lv.resources["tree_rss_mb_peak"] > 0 for lv in r.levels)


def test_ramp_without_refine_or_confirm():
    r = _ramp(_FakePool(capacity=5), [1, 2, 4, 8, 16], refine=False, confirm=False)
    assert (r.l_sat, r.l_fail, r.confirmed) == (4, 8, False)


def test_ramp_top_of_ladder_when_never_saturated():
    r = _ramp(_FakePool(capacity=100), [1, 2, 4])
    assert r.stop_reason == "top_of_ladder" and r.l_sat == 4 and r.l_fail is None


def test_ramp_fails_at_one_leg():
    r = _ramp(_FakePool(capacity=0), [1, 2, 4])
    assert r.l_sat == 0 and r.l_fail == 1


def test_ramp_memory_abort_is_reported_not_raised():
    r = _ramp(_FakePool(capacity=100, abort_at=2), [1, 2, 4, 8])
    assert r.stop_reason == "memory_abort" and r.l_sat == 2 and "boom" in r.note


def test_ramp_stops_before_a_level_that_would_not_fit_in_ram():
    # base 1 GB, then +10 TB per leg predicted: the very first level must be refused, and nothing run.
    pool = _FakePool(capacity=100)
    r = _ramp(pool, [1, 2, 4], assumed_mb_per_leg=10_000_000.0, base_rss_mb=1000.0)
    assert r.stop_reason == "memory" and r.l_sat == 0 and "budget" in r.note and pool.calls == []


def test_worker_pool_splits_legs_evenly():
    pool = WorkerPool({}, [[0], [1], [2]], {})
    assert pool.split_legs(7) == [3, 2, 2] and sum(pool.split_legs(1)) == 1 and pool.split_legs(2) == [1, 1, 0]
