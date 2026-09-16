"""Binance aggregated trades, turned into the vocabulary the rest speaks.

This is the only module that knows what is inside an archive: which column
holds what, whether a header row is present, what unit the timestamps are in.
`ingest` knows where the bytes came from; nothing downstream of here knows
either. Adding a second venue means writing a sibling of this file.

Three quirks of the real archive are handled, and they were measured rather
than assumed:

- **Header rows.** USD-M `aggTrades` files from 2020 and 2021 carry none;
  2023 onward do. Detected per file.
- **Timestamp units.** Every file sampled from 2020-01 to 2026-09 was in
  milliseconds, but the unit is read from the magnitude regardless, because a
  wrong guess is a silent factor-of-a-thousand error in the one quantity this
  project measures.
- **Simultaneous trades.** On BTCUSDT, 67.7% of rows share a millisecond with
  another row, in clusters of up to 263. Collapsing them is the single largest
  transformation applied to the data, and it sets a floor on resolvable lag of
  roughly one millisecond.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from leadlag.types import TradeSeries

HEADER = (
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
)
PRICE, QUANTITY, TRANSACT_TIME = 1, 2, 5

# Plausible bounds on a real timestamp, used to identify its unit: 2010-01-01
# through 2040-01-01, expressed in seconds.
_EARLIEST_S = 1_262_304_000
_LATEST_S = 2_208_988_800
_UNITS_NS = {
    1_000_000_000: 1,  # seconds
    1_000_000: 1_000,  # milliseconds
    1_000: 1_000_000,  # microseconds
    1: 1_000_000_000,  # nanoseconds
}


def read_archive(path: Path, *, symbol: str) -> TradeSeries:
    """Parse one daily `aggTrades` archive into a `TradeSeries`.

    Each archive holds a single CSV named after itself. Only three of its seven
    columns survive: the rest identify trades within Binance's own bookkeeping
    and carry no information this project uses.
    """
    with zipfile.ZipFile(path) as archive:
        (name,) = archive.namelist()
        text = archive.read(name).decode()

    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{path} contains no rows")
    if len(rows[0]) != len(HEADER):
        raise ValueError(
            f"{path} has {len(rows[0])} columns, expected {len(HEADER)} "
            f"{HEADER}; the columns are read positionally, so a changed layout "
            f"must stop the run rather than be parsed as if nothing happened"
        )
    if _is_header(rows[0]):
        rows = rows[1:]
    if not rows:
        raise ValueError(f"{path} contains a header and no trades")

    count = len(rows)
    raw_ts = np.fromiter((int(r[TRANSACT_TIME]) for r in rows), dtype=np.int64, count=count)
    price = np.fromiter((float(r[PRICE]) for r in rows), dtype=np.float64, count=count)
    qty = np.fromiter((float(r[QUANTITY]) for r in rows), dtype=np.float64, count=count)

    ts_ns = raw_ts * detect_time_unit_ns(int(raw_ts[0]))
    ts_ns, price, qty = collapse_simultaneous(ts_ns, price, qty)
    return TradeSeries(symbol=symbol, ts_ns=ts_ns, price=price, qty=qty)


def detect_time_unit_ns(stamp: int) -> int:
    """How many nanoseconds one unit of `stamp` represents.

    Identified by magnitude: a timestamp for any date this project could care
    about falls in a narrow band, and the four plausible units are a thousand
    apart, so there is no ambiguity. Anything outside every band raises rather
    than being coerced into whichever is nearest.
    """
    for divisor, nanoseconds in _UNITS_NS.items():
        if _EARLIEST_S <= stamp // divisor <= _LATEST_S:
            return nanoseconds
    raise ValueError(
        f"timestamp {stamp} is not in seconds, milliseconds, microseconds or "
        f"nanoseconds for any date between 2010 and 2040"
    )


def collapse_simultaneous(
    ts_ns: np.ndarray, price: np.ndarray, qty: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One row per distinct timestamp: last price wins, quantities are summed.

    Two trades at one instant would form a zero-length interval, which overlaps
    nothing, so Hayashi-Yoshida would silently discard the price change across
    it. Collapsing solves that once, here, instead of obliging every future
    estimator to handle degenerate intervals.

    The last price is the previous-tick convention: it is what a participant
    sampling at the end of that millisecond would have seen, and it matches how
    the gridded baseline takes its bucket closes, so the two methods differ in
    their clock and in nothing else. Volume is preserved exactly.

    This is not a rare tidy-up. On BTCUSDT it merges roughly two thirds of all
    rows.
    """
    if ts_ns.size == 0:
        return ts_ns, price, qty
    if not np.all(np.diff(ts_ns) >= 0):
        raise ValueError("ts_ns must be sorted before simultaneous trades can be collapsed")

    distinct = np.unique(ts_ns)
    if distinct.size == ts_ns.size:
        return ts_ns, price, qty

    last = np.searchsorted(ts_ns, distinct, side="right") - 1
    first = np.searchsorted(ts_ns, distinct, side="left")
    return distinct, price[last], np.add.reduceat(qty, first)


def save_canonical(series: TradeSeries, path: Path) -> Path:
    """Write the canonical form: three arrays, compressed.

    Measured at 0.48x the source archive it came from, with no dependency
    beyond numpy. The source is kept: re-deriving after a bug in this file
    should cost a rerun, not a re-download.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ts_ns=series.ts_ns, price=series.price, qty=series.qty)
    return path


def load_canonical(path: Path, *, symbol: str) -> TradeSeries:
    """Read a canonical file back. Validation runs again, as for any series."""
    with np.load(path) as stored:
        return TradeSeries(
            symbol=symbol,
            ts_ns=stored["ts_ns"],
            price=stored["price"],
            qty=stored["qty"],
        )


def _is_header(row: list[str]) -> bool:
    """A header if the first field is not a trade id.

    Checking for the literal column names as well, so a file whose layout
    changed but still parses is caught rather than silently mis-read.
    """
    try:
        int(row[0])
    except ValueError:
        if tuple(field.strip() for field in row) != HEADER:
            raise ValueError(f"unexpected header columns {row}, expected {list(HEADER)}") from None
        return True
    return False
