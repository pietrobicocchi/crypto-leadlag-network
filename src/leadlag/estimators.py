"""Lead-lag estimators. Pure maths: arrays in, numbers out.

Nothing here reads a file or touches the network, so every claim these
functions make can be checked against data whose answer is already known.

Sign convention, used identically by every estimator in this module: a
**positive** lag means B trails A, so B's price at time `s` reflects
information A had at `s - lag`. A candidate lag is tested by moving B's clock
back by that amount and asking how well the two then line up.
"""

from __future__ import annotations

import numpy as np

from leadlag.types import TradeSeries


def gridded_correlation(
    a: TradeSeries,
    b: TradeSeries,
    *,
    bucket_ns: int,
    lag_grid_ns: np.ndarray,
) -> np.ndarray:
    """Correlation at each candidate lag, using fixed buckets and forward fill.

    This is the standard wrong answer, implemented faithfully so that its
    failure is the method's and not a straw man's. Time is chopped into buckets
    of `bucket_ns`; each bucket takes the last price at or before its end,
    carried forward from whenever that trade happened; returns are the log
    differences between consecutive buckets.

    The flaw is the carry-forward. A bucket in which an asset did not trade
    repeats the previous price and so records a return of exactly zero - a
    fabricated observation, indistinguishable downstream from a real one. As
    buckets shrink, more of them are empty, and measured correlation falls
    toward zero however strongly the assets actually move together. That is the
    Epps effect.

    Returns one correlation per entry of `lag_grid_ns`, in the same order. The
    whole curve rather than its peak, because a sharp peak and a flat smear
    have the same argmax and mean entirely different things.
    """
    _check_series(a, "a")
    _check_series(b, "b")
    if bucket_ns <= 0:
        raise ValueError(f"bucket_ns must be positive, got {bucket_ns}")
    if not isinstance(lag_grid_ns, np.ndarray) or lag_grid_ns.dtype != np.int64:
        dtype = getattr(lag_grid_ns, "dtype", type(lag_grid_ns).__name__)
        raise TypeError(f"lag_grid_ns must be an int64 numpy array, got {dtype}")
    if lag_grid_ns.ndim != 1:
        raise ValueError(f"lag_grid_ns must be 1-D, got {lag_grid_ns.ndim}-D")

    # One bucket grid, fixed across every candidate lag. If the window moved
    # with the lag, each correlation would be computed over a different number
    # of buckets and the curve would not be comparable with itself.
    start = min(a.ts_ns[0], b.ts_ns[0])
    end = max(a.ts_ns[-1], b.ts_ns[-1])
    n_buckets = int((end - start) // bucket_ns)
    if n_buckets < 3:
        raise ValueError(
            f"bucket_ns={bucket_ns} gives {n_buckets} buckets over a window of "
            f"{end - start} ns; too few to form returns"
        )
    edges = start + bucket_ns * np.arange(1, n_buckets + 1, dtype=np.int64)

    returns_a = _bucketed_log_returns(a.ts_ns, a.price, edges)

    correlation = np.empty(lag_grid_ns.size, dtype=np.float64)
    for i, lag in enumerate(lag_grid_ns):
        # Move B's clock back by the candidate lag: if B really does trail A by
        # this much, the two now describe the same moments.
        returns_b = _bucketed_log_returns(b.ts_ns - lag, b.price, edges)
        correlation[i] = _pearson(returns_a, returns_b)
    return correlation


def peak_lag(lag_grid_ns: np.ndarray, correlation: np.ndarray) -> int:
    """The candidate lag at which correlation is highest.

    Assumes the two assets move together, so the maximum is taken rather than
    the largest absolute value: a strongly negative correlation is a different
    finding, not a lag.
    """
    if lag_grid_ns.shape != correlation.shape:
        raise ValueError(
            f"lag grid and correlation must have the same shape, "
            f"got {lag_grid_ns.shape} and {correlation.shape}"
        )
    return int(lag_grid_ns[np.argmax(correlation)])


def lead_lag_ratio(lag_grid_ns: np.ndarray, correlation: np.ndarray) -> float:
    """How lopsided the cross-correlation curve is, after Huth-Abergel (2014).

    The sum of squared correlation at positive lags over the same at negative
    lags; the lag-zero entry belongs to neither side. Above 1 means B trails A.

    This, not the peak, is what exposes the gridded baseline. Under a 10:1
    trade-rate imbalance with a true lag of exactly zero, the peak stays
    innocently at zero while this ratio reaches the high tens or hundreds - a
    confident, entirely fabricated finding that A leads B.
    """
    if lag_grid_ns.shape != correlation.shape:
        raise ValueError(
            f"lag grid and correlation must have the same shape, "
            f"got {lag_grid_ns.shape} and {correlation.shape}"
        )
    trailing = np.sum(correlation[lag_grid_ns > 0] ** 2)
    leading = np.sum(correlation[lag_grid_ns < 0] ** 2)
    if leading == 0:
        raise ValueError("no negative lags in the grid, so the ratio is undefined")
    return float(trailing / leading)


def _check_series(series: TradeSeries, name: str) -> None:
    if len(series) < 2:
        raise ValueError(f"series {name!r} ({series.symbol}) needs at least two trades")


def _bucketed_log_returns(ts_ns: np.ndarray, price: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Log returns between consecutive bucket closes, with prices carried forward.

    `searchsorted(..., "right") - 1` is the index of the last trade at or before
    each bucket close, which is exactly the carry-forward rule. Buckets that
    close before the first trade have no price at all and become NaN rather
    than a guess; `_pearson` drops them.
    """
    index = np.searchsorted(ts_ns, edges, side="right") - 1
    closes = np.full(edges.size, np.nan, dtype=np.float64)
    known = index >= 0
    closes[known] = price[index[known]]
    return np.diff(np.log(closes))


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Correlation over the buckets where both series have a price."""
    both = np.isfinite(x) & np.isfinite(y)
    if both.sum() < 3:
        raise ValueError(f"only {both.sum()} buckets have a price in both series")
    x, y = x[both], y[both]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        raise ValueError("a series has zero variance over the bucket grid")
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (sx * sy))
