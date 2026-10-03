"""Lead-lag estimators. Pure maths: arrays in, numbers out.

Nothing here reads a file or touches the network, so every claim these
functions make can be checked against data whose answer is already known.

Sign convention, used identically by every estimator in this module: a
**positive** lag means B trails A, so B's price at time `s` reflects
information A had at `s - lag`. A candidate lag is tested by moving B's clock
back by that amount and asking how well the two then line up.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    _check_lag_grid(lag_grid_ns)

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


def hayashi_yoshida_correlation(
    a: TradeSeries,
    b: TradeSeries,
    *,
    lag_grid_ns: np.ndarray,
) -> np.ndarray:
    """Correlation at each candidate lag, imposing no clock at all.

    Hayashi-Yoshida (2005) uses each asset's own trade times. A's return over
    `(t[i], t[i+1]]` multiplies B's return over `(s[j], s[j+1]]` exactly when
    those half-open intervals intersect - `t[i] < s[j+1]` and `s[j] < t[i+1]` -
    and the estimator is the sum of those products, normalised by each asset's
    realised variance.

    There is no bucket size, and that absence is the method. Nothing is
    interpolated and no return is invented for an asset that did not trade, so
    there is no mechanism for the Epps effect to act through and no way for a
    trade-rate imbalance alone to manufacture a lead-lag finding.

    Realised variances do not depend on the lag: shifting all of B's timestamps
    moves which intervals overlap, never B's own returns. They are computed
    once.

    **The returned ratio is not bounded by 1.** A's return over one interval is
    counted against every B interval overlapping it, so the numerator carries a
    multiplicity the denominator's realised variances do not, and under a
    strong trade-rate imbalance the ratio passes 1. That is the estimator
    behaving as defined, not an error - but anything downstream that assumes a
    correlation coefficient, a Fisher z-transform above all, has to account for
    it.
    """
    _check_series(a, "a")
    _check_series(b, "b")
    _check_lag_grid(lag_grid_ns)

    log_returns_a = np.diff(np.log(a.price))
    log_returns_b = np.diff(np.log(b.price))
    normaliser = np.sqrt(np.sum(log_returns_a**2) * np.sum(log_returns_b**2))
    if normaliser == 0:
        raise ValueError("a series never moves, so correlation is undefined")

    # Prefix sums let the inner sum over B become one subtraction.
    cumulative_b = np.concatenate([[0.0], np.cumsum(log_returns_b)])

    correlation = np.empty(lag_grid_ns.size, dtype=np.float64)
    for k, lag in enumerate(lag_grid_ns):
        overlapping_b = _overlapping_b_returns(a.ts_ns, b.ts_ns - lag, cumulative_b)
        correlation[k] = float(np.sum(log_returns_a * overlapping_b)) / normaliser
    return correlation


@dataclass(frozen=True)
class HYBlocks:
    """Hayashi-Yoshida's three sums, split by the time block they started in.

    `covariance` has shape (n_blocks, n_lags); `variance_a` and `variance_b`
    have shape (n_blocks,). Summing every row recovers the full-sample
    estimator exactly:

        covariance.sum(axis=0) / sqrt(variance_a.sum() * variance_b.sum())
            == hayashi_yoshida_correlation(a, b, lag_grid_ns=...)

    That identity is what makes a block bootstrap honest: resampling rows and
    re-summing is resampling this estimator, the one Gate 1 validated - not an
    approximation of it.
    """

    covariance: np.ndarray
    variance_a: np.ndarray
    variance_b: np.ndarray


def hayashi_yoshida_blocks(
    a: TradeSeries,
    b: TradeSeries,
    *,
    lag_grid_ns: np.ndarray,
    block_edges_ns: np.ndarray,
) -> HYBlocks:
    """Hayashi-Yoshida's sums, unsummed over time, for a block bootstrap.

    `block_edges_ns` is a strictly increasing int64 array of n_blocks + 1
    edges; block k is the half-open span `[edges[k], edges[k+1])`.

    Each term is filed under the block where its interval *starts*:
    A's interval `(t[i], t[i+1]]` - and with it every B product it collects at
    every lag - goes to the block containing `t[i]`; B's squared return over
    `(s[j], s[j+1]]` goes to the block containing `s[j]`, on B's own clock,
    since realised variance does not depend on the lag.

    Why not cut each series into blocks and run the estimator on each piece:
    the interval that straddles a cut would vanish, and a bootstrap that
    stitched pieces back together would invent intervals at the seams. Filing
    terms by start time changes nothing about which products exist, only which
    row they are added to.

    Every interval must start inside `[edges[0], edges[-1])`. One that does not
    is a ValueError, not silently dropped: a dropped term breaks the identity
    with `hayashi_yoshida_correlation` without any other sign.
    """
    _check_series(a, "a")
    _check_series(b, "b")
    _check_lag_grid(lag_grid_ns)
    _check_block_edges(block_edges_ns)

    n_blocks = block_edges_ns.size - 1
    block_a = _block_of(a.ts_ns[:-1], block_edges_ns)
    block_b = _block_of(b.ts_ns[:-1], block_edges_ns)

    log_returns_a = np.diff(np.log(a.price))
    log_returns_b = np.diff(np.log(b.price))
    cumulative_b = np.concatenate([[0.0], np.cumsum(log_returns_b)])

    covariance = np.empty((n_blocks, lag_grid_ns.size), dtype=np.float64)
    for k, lag in enumerate(lag_grid_ns):
        overlapping_b = _overlapping_b_returns(a.ts_ns, b.ts_ns - lag, cumulative_b)
        # bincount with weights adds each A interval's product into its block's row.
        covariance[:, k] = np.bincount(
            block_a, weights=log_returns_a * overlapping_b, minlength=n_blocks
        )
    return HYBlocks(
        covariance=covariance,
        variance_a=np.bincount(block_a, weights=log_returns_a**2, minlength=n_blocks),
        variance_b=np.bincount(block_b, weights=log_returns_b**2, minlength=n_blocks),
    )


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


def _check_lag_grid(lag_grid_ns: np.ndarray) -> None:
    if not isinstance(lag_grid_ns, np.ndarray) or lag_grid_ns.dtype != np.int64:
        dtype = getattr(lag_grid_ns, "dtype", type(lag_grid_ns).__name__)
        raise TypeError(f"lag_grid_ns must be an int64 numpy array, got {dtype}")
    if lag_grid_ns.ndim != 1:
        raise ValueError(f"lag_grid_ns must be 1-D, got {lag_grid_ns.ndim}-D")


def _overlapping_b_returns(
    ts_a: np.ndarray, shifted_ts_b: np.ndarray, cumulative_b: np.ndarray
) -> np.ndarray:
    """For each of A's intervals, the summed B returns over intervals overlapping it.

    For A's interval i, the overlapping B intervals are contiguous because both
    clocks are sorted. Its bounds are two binary searches:
      lo: the first j with s[j+1] > t[i]          (B's interval ends after A's starts)
      hi: one past the last j with s[j] < t[i+1]  (B's interval starts before A's ends)
    and the sum over that range is one subtraction of prefix sums.
    """
    lo = np.searchsorted(shifted_ts_b, ts_a[:-1], side="right") - 1
    hi = np.searchsorted(shifted_ts_b, ts_a[1:], side="left")
    n_b = cumulative_b.size - 1
    np.clip(lo, 0, n_b, out=lo)
    np.clip(hi, 0, n_b, out=hi)
    np.maximum(hi, lo, out=hi)  # an empty range must stay empty, not go negative
    return cumulative_b[hi] - cumulative_b[lo]


def _check_block_edges(block_edges_ns: np.ndarray) -> None:
    if not isinstance(block_edges_ns, np.ndarray) or block_edges_ns.dtype != np.int64:
        dtype = getattr(block_edges_ns, "dtype", type(block_edges_ns).__name__)
        raise TypeError(f"block_edges_ns must be an int64 numpy array, got {dtype}")
    if block_edges_ns.ndim != 1 or block_edges_ns.size < 2:
        raise ValueError("block_edges_ns must be 1-D with at least two edges")
    if np.any(np.diff(block_edges_ns) <= 0):
        raise ValueError("block_edges_ns must be strictly increasing")


def _block_of(starts_ns: np.ndarray, block_edges_ns: np.ndarray) -> np.ndarray:
    """Index of the block `[edges[k], edges[k+1])` holding each interval start."""
    if starts_ns[0] < block_edges_ns[0] or starts_ns[-1] >= block_edges_ns[-1]:
        raise ValueError(
            "an interval starts outside the blocks "
            f"[{block_edges_ns[0]}, {block_edges_ns[-1]}); widen block_edges_ns"
        )
    return np.searchsorted(block_edges_ns, starts_ns, side="right") - 1


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
