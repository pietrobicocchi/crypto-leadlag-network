"""The gridded baseline: the standard wrong answer.

This estimator exists to fail. Chopping time into fixed buckets and carrying
the last price forward fabricates zero returns wherever an asset did not trade,
which both dilutes correlation as buckets shrink (Epps) and smears a thin
asset's returns backwards in time. Reproducing that failure on data with a
known answer is the argument for Hayashi-Yoshida.
"""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.estimators import gridded_correlation, peak_lag
from leadlag.synthetic import delayed_pair

MS = 1_000_000


def dense_pair(lag_ns, *, seed=0, rate_a=200.0, rate_b=200.0, duration_s=120.0):
    """Both assets trading densely: the case the baseline should handle."""
    return delayed_pair(
        duration_s=duration_s,
        rate_a_hz=rate_a,
        rate_b_hz=rate_b,
        lag_ns=lag_ns,
        volatility=1e-3,
        noise=0.0,
        rng=np.random.default_rng(seed),
    )


def grid(bucket_ns, half_width=8):
    return np.arange(-half_width, half_width + 1, dtype=np.int64) * bucket_ns


def test_returns_one_correlation_per_candidate_lag():
    a, b = dense_pair(0)
    lags = grid(50 * MS)
    corr = gridded_correlation(a, b, bucket_ns=50 * MS, lag_grid_ns=lags)
    assert corr.shape == lags.shape
    assert corr.dtype == np.float64
    assert np.all(np.isfinite(corr))


def test_correlations_are_bounded():
    a, b = dense_pair(0)
    corr = gridded_correlation(a, b, bucket_ns=50 * MS, lag_grid_ns=grid(50 * MS))
    assert np.all(corr >= -1.0) and np.all(corr <= 1.0)


def test_recovers_the_lag_when_both_assets_trade_densely():
    """With ~10 trades per bucket for both assets, forward-fill barely bites.

    If the baseline could not get this right, a later failure would prove
    nothing about asynchrony - it would just mean the implementation is broken.
    """
    bucket = 50 * MS
    a, b = dense_pair(4 * bucket)
    lags = grid(bucket)
    corr = gridded_correlation(a, b, bucket_ns=bucket, lag_grid_ns=lags)
    assert peak_lag(lags, corr) == 4 * bucket


def test_correlation_collapses_as_buckets_shrink():
    """The Epps effect, as an executable claim.

    Same data, finer clock: measured correlation falls, because more buckets
    contain no trade and are recorded as a zero return.
    """
    a, b = dense_pair(0, rate_a=50.0, rate_b=5.0)
    coarse = gridded_correlation(a, b, bucket_ns=200 * MS, lag_grid_ns=grid(200 * MS))
    fine = gridded_correlation(a, b, bucket_ns=2 * MS, lag_grid_ns=grid(2 * MS))
    assert fine.max() < 0.5 * coarse.max()


def test_peak_lag_returns_the_grid_value_at_the_maximum():
    lags = np.array([-100, 0, 100], dtype=np.int64)
    assert peak_lag(lags, np.array([0.1, 0.9, 0.3])) == 0
    assert peak_lag(lags, np.array([0.9, 0.1, 0.3])) == -100


@pytest.mark.parametrize("bucket_ns", [0, -1])
def test_non_positive_bucket_is_rejected(bucket_ns):
    a, b = dense_pair(0)
    with pytest.raises(ValueError, match="bucket_ns"):
        gridded_correlation(a, b, bucket_ns=bucket_ns, lag_grid_ns=grid(50 * MS))


def test_float_lag_grid_is_rejected():
    """Lags are integer nanoseconds everywhere else; no silent coercion here."""
    a, b = dense_pair(0)
    with pytest.raises(TypeError, match="lag_grid_ns"):
        gridded_correlation(a, b, bucket_ns=50 * MS, lag_grid_ns=np.array([0.0, 1.0]))


def test_two_dimensional_lag_grid_is_rejected():
    a, b = dense_pair(0)
    with pytest.raises(ValueError, match="1-D"):
        gridded_correlation(
            a, b, bucket_ns=50 * MS, lag_grid_ns=np.array([[0], [1]], dtype=np.int64)
        )


def test_a_series_too_thin_to_correlate_is_rejected():
    a, b = dense_pair(0)
    thin = type(a)(
        symbol="THIN",
        ts_ns=np.array([0], dtype=np.int64),
        price=np.array([100.0]),
        qty=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="at least two trades"):
        gridded_correlation(a, thin, bucket_ns=50 * MS, lag_grid_ns=grid(50 * MS))


def test_arguments_are_keyword_only():
    a, b = dense_pair(0)
    with pytest.raises(TypeError):
        gridded_correlation(a, b, 50 * MS, grid(50 * MS))
