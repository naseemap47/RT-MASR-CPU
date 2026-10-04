# benchmark/tests/test_load_timer.py
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.runners.load_timer import LoadResult, measure_load


def test_load_result_is_dataclass():
    import dataclasses
    assert dataclasses.is_dataclass(LoadResult)


def test_measure_load_fields():
    # Use a trivial factory that allocates a small object
    def factory():
        time.sleep(0.05)
        return [0] * 10000  # allocate ~80KB

    result = measure_load("test_config", factory)
    assert result.config_id == "test_config"
    assert result.load_time_s >= 0.04   # at least 50ms sleep
    assert result.rss_before_mb > 0
    assert result.rss_after_mb > 0
    assert isinstance(result.rss_delta_mb, float)


def test_measure_load_timing_accuracy():
    def factory():
        time.sleep(0.1)
        return object()

    result = measure_load("timing_test", factory)
    assert result.load_time_s >= 0.09
    assert result.load_time_s <= 2.0   # sanity upper bound
