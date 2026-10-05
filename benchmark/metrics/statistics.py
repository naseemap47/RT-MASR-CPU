# benchmark/metrics/statistics.py
"""
Pure-function statistics helpers for benchmark metric summarisation.

No external dependencies — stdlib only.
"""
from __future__ import annotations


def percentile(data: list[float], p: float) -> float:
    """
    Compute the p-th percentile of data using linear interpolation.

    Args:
        data: Non-empty list of floats.
        p:    Percentile in [0, 100].

    Returns:
        Interpolated p-th percentile value.

    Raises:
        ValueError: If data is empty.
    """
    if not data:
        raise ValueError("Cannot compute percentile of empty list.")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"Percentile must be in [0, 100], got {p}.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    # Linear interpolation (same as numpy percentile default)
    rank = (p / 100.0) * (n - 1)
    lower = int(rank)
    upper = lower + 1
    if upper >= n:
        return sorted_data[-1]
    frac = rank - lower
    return sorted_data[lower] + frac * (sorted_data[upper] - sorted_data[lower])


def summarise(data: list[float]) -> dict:
    """
    Compute summary statistics for a list of floats.

    Args:
        data: Non-empty list of floats.

    Returns:
        Dict with keys: mean, p50, p95, p99, min, max, count.
    """
    if not data:
        raise ValueError("Cannot summarise empty list.")
    return {
        "mean":  sum(data) / len(data),
        "p50":   percentile(data, 50),
        "p95":   percentile(data, 95),
        "p99":   percentile(data, 99),
        "min":   min(data),
        "max":   max(data),
        "count": len(data),
    }
