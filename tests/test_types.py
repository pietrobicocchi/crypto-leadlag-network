"""The TradeSeries contract.

TradeSeries is the vocabulary every estimator speaks. A silent violation of its
invariants — arrays misaligned by a bad slice, timestamps truncated to a coarser
unit, an unnoticed NaN — would not raise anywhere downstream. It would produce a
plausible number. This file is the only place that contract is enforced.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from leadlag.types import TradeSeries


def make_series(**overrides) -> TradeSeries:
    """A valid three-trade series. Pass a field to replace it."""
    fields = {
        "symbol": "BTCUSDT",
        "ts_ns": np.array([1_000, 2_000, 3_000], dtype=np.int64),
        "price": np.array([100.0, 101.0, 100.5], dtype=np.float64),
        "qty": np.array([0.5, 1.5, 0.25], dtype=np.float64),
    }
    return TradeSeries(**(fields | overrides))


def test_valid_series_constructs_and_reports_its_length():
    series = make_series()
    assert series.symbol == "BTCUSDT"
    assert len(series) == 3


def test_empty_series_is_valid():
    """A symbol with no trades in a window is normal, not an error."""
    series = make_series(
        ts_ns=np.array([], dtype=np.int64),
        price=np.array([], dtype=np.float64),
        qty=np.array([], dtype=np.float64),
    )
    assert len(series) == 0


def test_symbol_must_be_non_empty():
    with pytest.raises(ValueError, match="symbol"):
        make_series(symbol="")


def test_mismatched_array_lengths_are_rejected():
    """The most dangerous bug in the codebase: prices misaligned from their times."""
    with pytest.raises(ValueError, match="same length"):
        make_series(price=np.array([100.0, 101.0], dtype=np.float64))


def test_two_dimensional_arrays_are_rejected():
    with pytest.raises(ValueError, match="1-D"):
        make_series(price=np.array([[100.0], [101.0], [100.5]], dtype=np.float64))


def test_a_list_is_not_an_array():
    with pytest.raises(TypeError, match="numpy array"):
        make_series(qty=[0.5, 1.5, 0.25])


def test_float_timestamps_are_rejected():
    """float64 cannot represent nanoseconds; this project measures milliseconds."""
    with pytest.raises(TypeError, match="ts_ns"):
        make_series(ts_ns=np.array([1e3, 2e3, 3e3], dtype=np.float64))


def test_float32_prices_are_rejected():
    """No silent widening: precision is a decision the caller must make explicitly."""
    with pytest.raises(TypeError, match="price"):
        make_series(price=np.array([100.0, 101.0, 100.5], dtype=np.float32))


def test_simultaneous_trades_are_rejected():
    """Two trades at one instant form a zero-length interval.

    Hayashi-Yoshida pairs returns whose intervals overlap; a zero-length
    interval overlaps nothing, so the price change across it would vanish from
    the covariance silently. Ties are collapsed in `normalize`, never here.
    """
    with pytest.raises(ValueError, match="strictly increasing"):
        make_series(ts_ns=np.array([1_000, 2_000, 2_000], dtype=np.int64))


def test_out_of_order_timestamps_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        make_series(ts_ns=np.array([1_000, 3_000, 2_000], dtype=np.int64))


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("price", [100.0, np.nan, 100.5]),
        ("price", [100.0, np.inf, 100.5]),
        ("price", [100.0, 0.0, 100.5]),
        ("price", [100.0, -101.0, 100.5]),
        ("qty", [0.5, np.nan, 0.25]),
        ("qty", [0.5, -1.5, 0.25]),
    ],
)
def test_unusable_values_are_rejected(field, values):
    """A NaN here propagates into every covariance it touches, without raising."""
    with pytest.raises(ValueError, match=field):
        make_series(**{field: np.array(values, dtype=np.float64)})


def test_zero_quantity_is_allowed():
    """Unlike price, a zero quantity is merely odd, not unusable."""
    series = make_series(qty=np.array([0.0, 1.5, 0.25], dtype=np.float64))
    assert len(series) == 3


def test_fields_cannot_be_rebound():
    series = make_series()
    with pytest.raises(dataclasses.FrozenInstanceError):
        series.symbol = "ETHUSDT"


def test_arrays_cannot_be_mutated_in_place():
    """frozen=True alone would not catch this: it blocks rebinding, not writes."""
    series = make_series()
    with pytest.raises(ValueError, match="read-only"):
        series.price[0] = 999.0
