# benchmark/tests/test_system_metrics.py
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.metrics.system_metrics import SystemSampler


def test_sampler_context_manager():
    with SystemSampler(interval_s=0.05) as s:
        time.sleep(0.3)  # let it collect at least 3 samples
    result = s.result()
    assert "overall_cpu_pct" in result
    assert "process_cpu_pct" in result
    assert "rss_mb" in result
    assert "thread_count" in result
    assert "peak_rss_mb" in result
    assert "peak_threads" in result


def test_sampler_rss_positive():
    with SystemSampler(interval_s=0.05) as s:
        time.sleep(0.2)
    result = s.result()
    assert result["peak_rss_mb"] > 0


def test_sampler_result_has_summary_keys():
    with SystemSampler(interval_s=0.05) as s:
        time.sleep(0.2)
    result = s.result()
    for field in ["overall_cpu_pct", "rss_mb"]:
        assert "mean" in result[field], f"Missing 'mean' in {field}"
        assert "p95" in result[field], f"Missing 'p95' in {field}"


def test_sampler_stops_after_exit():
    with SystemSampler(interval_s=0.05) as s:
        time.sleep(0.1)
    # After exit, thread should be stopped; calling result() again should not raise.
    r1 = s.result()
    r2 = s.result()
    assert r1["peak_rss_mb"] == r2["peak_rss_mb"]
