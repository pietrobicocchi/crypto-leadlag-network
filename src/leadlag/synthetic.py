"""Synthetic trade data with a lag you already know.

An estimator cannot be validated on real data: there is no ground truth to
check it against. It can be validated here, because the lag is chosen rather
than discovered.

The model is the one the project's premise describes. A single latent log-price
path carries the information. Asset A observes that path at its own trade
times; asset B observes it as it was `lag_ns` earlier, because its market makers
have not yet repriced. Each observation carries independent noise, and — the
part that matters — the two assets' trade times are drawn independently, so
they are asynchronous in exactly the way real trade data is.
"""

from __future__ import annotations

import numpy as np

from leadlag.types import TradeSeries

NS_PER_S = 1_000_000_000


def delayed_pair(
    *,
    duration_s: float,
    rate_a_hz: float,
    rate_b_hz: float,
    lag_ns: int,
    volatility: float,
    noise: float,
    rng: np.random.Generator,
    initial_price: float = 100.0,
    symbols: tuple[str, str] = ("A", "B"),
) -> tuple[TradeSeries, TradeSeries]:
    """Two asynchronously traded assets, where B lags A by `lag_ns`.

    Arguments are keyword-only: a positional swap of `rate_a_hz` and
    `rate_b_hz` would silently invert the trade-rate asymmetry that the whole
    comparison against the gridded baseline depends on.

    - `duration_s` — length of the observation window, in seconds.
    - `rate_a_hz`, `rate_b_hz` — mean trades per second. Their ratio is the
      asymmetry under which a gridded estimator fabricates a lag.
    - `lag_ns` — how far B trails A, in nanoseconds. Non-negative: B's price at
      time `s` reflects the latent path at `s - lag_ns`. For the opposite
      direction, swap the returned pair.
    - `volatility` — standard deviation of the latent log-price per √second.
    - `noise` — standard deviation of independent observation noise, in
      log-price. Independent across assets, so it adds no spurious lead-lag.
    - `rng` — passed in rather than seeded here, so determinism is the caller's
      explicit choice and no global state is involved.

    Quantities are all 1.0; nothing models trade size yet.

    The latent path is exact Brownian motion, not a fine grid sampled by
    lookup: increments are drawn over the actual gaps between the times the
    path is needed at. So `lag_ns` can be any integer, rather than a multiple
    of some simulation step the estimator would then have an easier time
    landing on.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    if rate_a_hz <= 0:
        raise ValueError(f"rate_a_hz must be positive, got {rate_a_hz}")
    if rate_b_hz <= 0:
        raise ValueError(f"rate_b_hz must be positive, got {rate_b_hz}")
    if lag_ns < 0:
        raise ValueError(f"lag_ns must be non-negative, got {lag_ns}; swap the pair instead")
    if volatility < 0:
        raise ValueError(f"volatility must be non-negative, got {volatility}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if initial_price <= 0:
        raise ValueError(f"initial_price must be positive, got {initial_price}")

    ts_a = _poisson_times_ns(duration_s, rate_a_hz, rng)
    ts_b = _poisson_times_ns(duration_s, rate_b_hz, rng)

    # B reads the path as it was lag_ns ago, so those are the times the path is
    # needed at. The union is what the Brownian motion must be sampled on.
    read_times_b = ts_b - lag_ns
    latent_times = np.unique(np.concatenate([ts_a, read_times_b]))
    log_path = _brownian_log_path(latent_times, volatility, initial_price, rng)

    return (
        _observe(symbols[0], ts_a, latent_times, log_path, ts_a, noise, rng),
        _observe(symbols[1], ts_b, latent_times, log_path, read_times_b, noise, rng),
    )


def _poisson_times_ns(duration_s: float, rate_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Arrival times of a Poisson process over `[0, duration_s)`, in integer ns.

    Drawn as `N ~ Poisson(rate * duration)` uniform points rather than by
    accumulating exponential gaps: same process, one vectorised step.

    Rounding to integer nanoseconds can collide — at 36k trades in an hour,
    roughly once every few thousand runs. `np.unique` collapses those, because
    a repeated timestamp is a zero-length interval that TradeSeries rejects,
    and a test that fails one run in a few thousand is worse than one that
    never does.
    """
    count = rng.poisson(rate_hz * duration_s)
    offsets = rng.uniform(0.0, duration_s, size=count)
    return np.unique((offsets * NS_PER_S).astype(np.int64))


def _brownian_log_path(
    times_ns: np.ndarray,
    volatility: float,
    initial_price: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Brownian motion in log-price, sampled exactly at `times_ns`.

    A Brownian increment over a gap of `dt` seconds is Gaussian with standard
    deviation `volatility * sqrt(dt)`, so drawing one increment per actual gap
    samples the continuous-time process exactly — no simulation grid, and no
    resolution limit on the lag.

    Log-price rather than price: exponentiating guarantees the positivity
    TradeSeries requires, and makes the observation noise multiplicative, which
    is the realistic form.
    """
    if times_ns.size == 0:
        return np.empty(0, dtype=np.float64)
    gaps_s = np.diff(times_ns) / NS_PER_S
    increments = rng.normal(0.0, volatility * np.sqrt(gaps_s))
    path = np.empty(times_ns.size, dtype=np.float64)
    path[0] = np.log(initial_price)
    path[1:] = path[0] + np.cumsum(increments)
    return path


def _observe(
    symbol: str,
    ts_ns: np.ndarray,
    latent_times: np.ndarray,
    log_path: np.ndarray,
    read_times: np.ndarray,
    noise: float,
    rng: np.random.Generator,
) -> TradeSeries:
    """One asset's view: the path at `read_times`, stamped at `ts_ns`.

    `read_times` is where in the path this asset is looking; `ts_ns` is when it
    prints the trade. They differ by the lag, and that difference is the whole
    phenomenon under study.
    """
    log_price = log_path[np.searchsorted(latent_times, read_times)]
    observed = log_price + rng.normal(0.0, noise, size=ts_ns.size)
    return TradeSeries(
        symbol=symbol,
        ts_ns=ts_ns,
        price=np.exp(observed),
        qty=np.ones(ts_ns.size, dtype=np.float64),
    )
