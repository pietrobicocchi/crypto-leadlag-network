"""Hayashi-Yoshida against an oracle, and against the baseline it replaces.

The estimator in `leadlag.estimators` is vectorised: for each of A's intervals
it locates the contiguous range of B's overlapping intervals with two binary
searches and sums them as a difference of cumulative sums. That is fast and
easy to get subtly wrong, and an off-by-one in an overlap condition does not
crash - it returns a slightly wrong covariance.

So this file carries a second implementation: the double sum written out
literally, O(n*m), far too slow for real data, and correct by inspection. Its
only job is to disagree with the fast one. It must never be edited to make a
test pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.estimators import hayashi_yoshida_correlation
from leadlag.synthetic import delayed_pair
from leadlag.types import TradeSeries

MS = 1_000_000


def reference_hy_covariance(a: TradeSeries, b: TradeSeries, lag_ns: int = 0) -> float:
    """The definition, transcribed. Every pair of intervals, tested for overlap.

    A's return over (t[i], t[i+1]] multiplies B's return over (s[j], s[j+1]]
    exactly when those half-open intervals intersect, which is
    `t[i] < s[j+1] and s[j] < t[i+1]`.
    """
    ts_a, ts_b = a.ts_ns, b.ts_ns - lag_ns
    log_a, log_b = np.log(a.price), np.log(b.price)
    total = 0.0
    for i in range(len(ts_a) - 1):
        for j in range(len(ts_b) - 1):
            if ts_a[i] < ts_b[j + 1] and ts_b[j] < ts_a[i + 1]:
                total += (log_a[i + 1] - log_a[i]) * (log_b[j + 1] - log_b[j])
    return total


def reference_hy_correlation(a: TradeSeries, b: TradeSeries, lag_ns: int = 0) -> float:
    covariance = reference_hy_covariance(a, b, lag_ns)
    var_a = float(np.sum(np.diff(np.log(a.price)) ** 2))
    var_b = float(np.sum(np.diff(np.log(b.price)) ** 2))
    return covariance / np.sqrt(var_a * var_b)


def tiny_pair(seed, n_a=9, n_b=7):
    """Few enough trades that the O(n*m) oracle is instant, and edge cases are dense."""
    rng = np.random.default_rng(seed)
    ts_a = np.unique(rng.integers(0, 200, size=n_a).astype(np.int64))
    ts_b = np.unique(rng.integers(0, 200, size=n_b).astype(np.int64))
    make = lambda sym, ts: TradeSeries(  # noqa: E731
        symbol=sym,
        ts_ns=ts,
        price=100.0 * np.exp(rng.normal(0, 0.01, size=ts.size)),
        qty=np.ones(ts.size),
    )
    return make("A", ts_a), make("B", ts_b)


def market_pair(lag_ns, *, rate_a=200.0, rate_b=20.0, duration_s=300.0, seed=0):
    return delayed_pair(
        duration_s=duration_s,
        rate_a_hz=rate_a,
        rate_b_hz=rate_b,
        lag_ns=lag_ns,
        volatility=1e-3,
        noise=0.0,
        rng=np.random.default_rng(seed),
    )


@pytest.mark.parametrize("seed", range(8))
def test_matches_the_reference_implementation(seed):
    """The oracle test. Different trade counts, ties in timing, uneven gaps."""
    a, b = tiny_pair(seed)
    lags = np.array([-30, -7, 0, 5, 40], dtype=np.int64)
    fast = hayashi_yoshida_correlation(a, b, lag_grid_ns=lags)
    slow = np.array([reference_hy_correlation(a, b, int(lag)) for lag in lags])
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("n_a,n_b", [(2, 2), (2, 30), (30, 2), (3, 4)])
def test_matches_the_reference_at_degenerate_sizes(n_a, n_b):
    """Where off-by-one errors hide: the shortest series that still has a return."""
    a, b = tiny_pair(3, n_a=n_a, n_b=n_b)
    lags = np.array([-11, 0, 11], dtype=np.int64)
    fast = hayashi_yoshida_correlation(a, b, lag_grid_ns=lags)
    slow = np.array([reference_hy_correlation(a, b, int(lag)) for lag in lags])
    np.testing.assert_allclose(fast, slow, rtol=1e-12, atol=1e-14)


def test_correlation_with_itself_is_one():
    """Every interval overlaps only itself, so the estimator must reduce to
    realised variance over realised variance."""
    a, _ = market_pair(0, duration_s=30.0)
    corr = hayashi_yoshida_correlation(a, a, lag_grid_ns=np.array([0], dtype=np.int64))
    assert corr[0] == pytest.approx(1.0, abs=1e-12)


def test_correlations_are_bounded():
    a, b = market_pair(0)
    corr = hayashi_yoshida_correlation(
        a, b, lag_grid_ns=np.arange(-6, 7, dtype=np.int64) * (50 * MS)
    )
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0)


def test_float_lag_grid_is_rejected():
    a, b = market_pair(0, duration_s=30.0)
    with pytest.raises(TypeError, match="lag_grid_ns"):
        hayashi_yoshida_correlation(a, b, lag_grid_ns=np.array([0.0, 1.0]))


def test_a_series_too_thin_to_correlate_is_rejected():
    a, _ = market_pair(0, duration_s=30.0)
    thin = TradeSeries(
        symbol="THIN",
        ts_ns=np.array([0], dtype=np.int64),
        price=np.array([100.0]),
        qty=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="at least two trades"):
        hayashi_yoshida_correlation(a, thin, lag_grid_ns=np.array([0], dtype=np.int64))


def test_arguments_are_keyword_only():
    a, b = market_pair(0, duration_s=30.0)
    with pytest.raises(TypeError):
        hayashi_yoshida_correlation(a, b, np.array([0], dtype=np.int64))
