# benchmark/tests/test_statistics.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.metrics.statistics import percentile, summarise


def test_percentile_median():
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0


def test_percentile_p95():
    data = list(range(1, 101))  # 1..100
    result = percentile([float(x) for x in data], 95)
    assert 94.0 <= result <= 96.0


def test_summarise_keys():
    result = summarise([1.0, 2.0, 3.0, 4.0, 5.0])
    assert set(result.keys()) == {"mean", "p50", "p95", "p99", "min", "max", "count"}


def test_summarise_values():
    result = summarise([10.0, 20.0, 30.0])
    assert result["min"] == 10.0
    assert result["max"] == 30.0
    assert result["count"] == 3
    assert abs(result["mean"] - 20.0) < 0.001


def test_summarise_single_element():
    result = summarise([5.0])
    assert result["mean"] == 5.0
    assert result["p50"] == 5.0
    assert result["p95"] == 5.0
    assert result["p99"] == 5.0


def test_percentile_empty_raises():
    import pytest
    with pytest.raises(ValueError):
        percentile([], 50)


def test_percentile_known_values():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 50) == 3.0
    assert percentile(data, 100) == 5.0
    assert abs(percentile(data, 25) - 2.0) < 1e-9
    assert abs(percentile(data, 95) - 4.8) < 1e-9      # numpy.percentile default


def test_percentile_matches_numpy():
    import numpy as np
    data = [0.3, 9.1, 2.2, 7.7, 4.4, 1.0, 8.8]
    for p in (5, 50, 90, 95, 99):
        assert abs(percentile(data, p) - float(np.percentile(data, p))) < 1e-9


def test_percentile_rejects_out_of_range():
    import pytest
    with pytest.raises(ValueError):
        percentile([1.0], 101)
