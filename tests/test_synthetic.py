"""The synthetic generator's contract.

This is the measuring stick: everything the estimators claim is judged against
a lag chosen here. If the generator is wrong, every downstream result is wrong
in a way no amount of testing further down would reveal.
"""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.synthetic import delayed_pair
from leadlag.types import TradeSeries

MS = 1_000_000  # nanoseconds in a millisecond


def pair(**overrides):
    """A valid pair. Pass a parameter to replace it."""
    params = {
        "duration_s": 60.0,
        "rate_a_hz": 50.0,
        "rate_b_hz": 5.0,
        "lag_ns": 20 * MS,
        "volatility": 1e-3,
        "noise": 1e-5,
        "rng": np.random.default_rng(0),
    }
    return delayed_pair(**(params | overrides))


def test_returns_two_valid_trade_series():
    a, b = pair()
    assert isinstance(a, TradeSeries) and isinstance(b, TradeSeries)
    assert (a.symbol, b.symbol) == ("A", "B")
    assert len(a) > 0 and len(b) > 0


def test_symbols_can_be_named():
    a, b = pair(symbols=("BTCUSDT", "ADAUSDT"))
    assert (a.symbol, b.symbol) == ("BTCUSDT", "ADAUSDT")


def test_is_deterministic_given_the_same_seed():
    a1, b1 = pair(rng=np.random.default_rng(42))
    a2, b2 = pair(rng=np.random.default_rng(42))
    assert np.array_equal(a1.ts_ns, a2.ts_ns)
    assert np.array_equal(a1.price, a2.price)
    assert np.array_equal(b1.ts_ns, b2.ts_ns)
    assert np.array_equal(b1.price, b2.price)


def test_different_seeds_give_different_draws():
    a1, _ = pair(rng=np.random.default_rng(1))
    a2, _ = pair(rng=np.random.default_rng(2))
    assert not np.array_equal(a1.ts_ns, a2.ts_ns)


def test_arrival_rates_are_respected():
    """Poisson(lambda*T) has mean lambda*T; 6 sigma is a wide, non-flaky bound."""
    a, b = pair(duration_s=100.0, rate_a_hz=50.0, rate_b_hz=5.0)
    for series, expected in ((a, 5000), (b, 500)):
        assert abs(len(series) - expected) < 6 * np.sqrt(expected)


def test_asymmetric_rates_produce_asymmetric_counts():
    """The 10:1 imbalance is the condition under which the gridded baseline fails."""
    a, b = pair(rate_a_hz=100.0, rate_b_hz=10.0)
    assert len(a) > 5 * len(b)


def test_zero_volatility_and_noise_gives_a_flat_price():
    a, b = pair(volatility=0.0, noise=0.0, initial_price=123.5)
    assert np.allclose(a.price, 123.5)
    assert np.allclose(b.price, 123.5)


def test_timestamps_are_deduplicated_at_nanosecond_resolution():
    """At extreme rates, rounding to integer ns forces collisions.

    TradeSeries rejects repeated timestamps, so the generator must remove them
    by construction rather than raise once in several thousand runs.
    """
    a, _ = pair(duration_s=1e-6, rate_a_hz=1e9)
    assert np.all(np.diff(a.ts_ns) > 0)
    assert len(a) < 1000  # collisions happened and were collapsed


def test_an_empty_series_is_produced_rather_than_an_error():
    a, b = pair(duration_s=1e-9, rate_a_hz=1e-6, rate_b_hz=1e-6)
    assert len(a) == 0 and len(b) == 0


def test_b_lags_a_by_the_requested_amount_with_the_requested_sign():
    """The sign test, without an estimator.

    With no observation noise, A and B are the same Brownian path read at
    different times. Shifting B's clock back by the lag should align it with A;
    shifting it forward by the same amount moves it twice as far away. A sign
    error here is exactly the bug that would conclude altcoins lead Bitcoin.
    """
    lag = 200 * MS
    a, b = pair(
        duration_s=60.0,
        rate_a_hz=200.0,
        rate_b_hz=200.0,
        lag_ns=lag,
        volatility=1e-3,
        noise=0.0,
        rng=np.random.default_rng(7),
    )
    log_a, log_b = np.log(a.price), np.log(b.price)

    # Only compare where both shifts stay inside A's observed interval, so the
    # result is a statement about alignment and not about interpolation clamping.
    inside = (b.ts_ns - lag >= a.ts_ns[0]) & (b.ts_ns + lag <= a.ts_ns[-1])
    assert inside.sum() > 1000

    def misalignment(shift):
        target = (b.ts_ns[inside] + shift).astype(np.float64)
        return float(np.mean(np.abs(np.interp(target, a.ts_ns, log_a) - log_b[inside])))

    assert misalignment(-lag) * 3 < misalignment(+lag)


@pytest.mark.parametrize(
    ("param", "value"),
    [
        ("duration_s", 0.0),
        ("duration_s", -1.0),
        ("rate_a_hz", 0.0),
        ("rate_b_hz", -1.0),
        ("lag_ns", -1),
        ("volatility", -1e-3),
        ("noise", -1e-3),
        ("initial_price", 0.0),
    ],
)
def test_unusable_parameters_are_rejected(param, value):
    with pytest.raises(ValueError, match=param):
        pair(**{param: value})


def test_arguments_are_keyword_only():
    """A positional swap of rate_a_hz and rate_b_hz would silently invert the
    asymmetry the whole experiment depends on."""
    with pytest.raises(TypeError):
        delayed_pair(60.0, 50.0, 5.0, 20 * MS, 1e-3, 1e-5, np.random.default_rng(0))
