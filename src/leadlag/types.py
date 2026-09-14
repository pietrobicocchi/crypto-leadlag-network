"""The canonical trade series.

Every stage downstream of `normalize` speaks `TradeSeries` and nothing else.
Its invariants are checked once, at construction, because the bugs they catch
do not raise later — they produce a plausible number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Exact dtypes, never coerced. int64 nanoseconds because float64 cannot
# represent a nanosecond, and this project measures milliseconds; float64
# prices because ten million products accumulate error in float32.
_EXPECTED_DTYPES = {
    "ts_ns": np.int64,
    "price": np.float64,
    "qty": np.float64,
}


@dataclass(frozen=True, eq=False)
class TradeSeries:
    """Trades for one instrument, as parallel arrays in time order.

    Parallel arrays rather than a list of objects because ten million Python
    objects will not fit in memory. `ts_ns[i]`, `price[i]` and `qty[i]` describe
    one trade.

    Invariants, enforced at construction:

    - `symbol` is a non-empty string.
    - all three arrays are 1-D, the same length, and of exactly the dtype above.
    - `ts_ns` is **strictly** increasing. Simultaneous trades are collapsed in
      `normalize`, never here: two trades at one instant form a zero-length
      interval, which overlaps nothing, so Hayashi-Yoshida would silently drop
      the price change across it.
    - `price` is finite and positive; `qty` is finite and non-negative.

    The arrays are made read-only, because `frozen=True` only blocks rebinding
    a field — it does not stop `series.price[0] = 999`. Note this freezes the
    array the caller passed in. If that array is a *view* of a larger one, the
    parent stays writable, so mutation through the parent is still possible;
    this closes the common accident, not every conceivable one.

    Equality is identity (`eq=False`). The dataclass-generated `__eq__` compares
    the fields as a tuple, which calls `bool()` on an array comparison: that
    raises "truth value of an array is ambiguous" for a series of more than one
    trade, and — worse — silently returns a numpy array for a series of exactly
    one. Neither is usable, and the inconsistency is the dangerous part. A
    meaningful element-wise equality can be added when something needs it.
    """

    symbol: str
    ts_ns: np.ndarray
    price: np.ndarray
    qty: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError(f"symbol must be a non-empty string, got {self.symbol!r}")

        for name, expected in _EXPECTED_DTYPES.items():
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(array).__name__}")
            if array.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got {array.ndim}-D")
            if array.dtype != expected:
                raise TypeError(
                    f"{name} must have dtype {np.dtype(expected)}, got {array.dtype}. "
                    f"Convert deliberately; it is never coerced here."
                )

        lengths = {name: len(getattr(self, name)) for name in _EXPECTED_DTYPES}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"ts_ns, price and qty must have the same length, got {lengths}")

        if not np.all(np.diff(self.ts_ns) > 0):
            raise ValueError(
                "ts_ns must be strictly increasing; collapse simultaneous trades first"
            )

        if not np.all(np.isfinite(self.price)) or not np.all(self.price > 0):
            raise ValueError("price must be finite and positive")

        if not np.all(np.isfinite(self.qty)) or not np.all(self.qty >= 0):
            raise ValueError("qty must be finite and non-negative")

        for name in _EXPECTED_DTYPES:
            getattr(self, name).flags.writeable = False

    def __len__(self) -> int:
        """The number of trades."""
        return int(self.ts_ns.size)
