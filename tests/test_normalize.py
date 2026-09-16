"""Binance CSV to TradeSeries: the only file that knows the archive's format.

Every quirk handled here is one the real archive produces. Files before 2022
carry no header row and files after it do; the timestamp unit is read from the
data rather than assumed; and two thirds of BTCUSDT's rows share a millisecond
with another row, so collapsing is not a tidy-up but the largest single
transformation applied to the data.

Test archives are built in-process. Nothing here touches the network.
"""

from __future__ import annotations

import datetime as dt
import zipfile

import numpy as np
import pytest

from leadlag.normalize import (
    HEADER,
    collapse_simultaneous,
    detect_time_unit_ns,
    load_canonical,
    read_archive,
    save_canonical,
)

DAY = dt.date(2026, 3, 2)
BASE_MS = 1_772_409_600_000  # 2026-03-02T00:00:00Z


def csv_rows(start_ms=BASE_MS, unit_scale=1):
    """Three trades, two of them sharing a millisecond."""
    return [
        # id, price, qty, first, last, transact_time, is_buyer_maker
        f"1,100.5,2.0,10,10,{(start_ms + 0) * unit_scale},true",
        f"2,100.7,3.0,11,11,{(start_ms + 5) * unit_scale},false",
        f"3,100.9,1.0,12,12,{(start_ms + 5) * unit_scale},true",
    ]


def write_zip(tmp_path, rows, *, header=True, name="ADAUSDT-aggTrades-2026-03-02"):
    body = ("\n".join([",".join(HEADER)] + rows) if header else "\n".join(rows)) + "\n"
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{name}.csv", body)
    return path


@pytest.mark.parametrize("header", [True, False])
def test_reads_files_with_and_without_a_header(tmp_path, header):
    """2020 and 2021 archives have no header row; 2023 onward do."""
    series = read_archive(write_zip(tmp_path, csv_rows(), header=header), symbol="ADAUSDT")
    assert series.symbol == "ADAUSDT"
    assert len(series) == 2  # three trades, two sharing a millisecond


def test_rejects_an_unexpected_schema(tmp_path):
    """A changed column layout must stop the run, not be parsed as if nothing
    happened - the columns are positional."""
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("bad.csv", "a,b,c\n1,2,3\n")
    with pytest.raises(ValueError, match="columns"):
        read_archive(path, symbol="ADAUSDT")


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        (1_772_409_600, 1_000_000_000),  # seconds
        (1_772_409_600_000, 1_000_000),  # milliseconds
        (1_772_409_600_000_000, 1_000),  # microseconds
        (1_772_409_600_000_000_000, 1),  # nanoseconds
    ],
)
def test_detects_the_timestamp_unit(stamp, expected):
    """Read from the magnitude, never assumed. The archive is documented to
    have changed unit, and a wrong guess is a silent factor-of-1000 error."""
    assert detect_time_unit_ns(stamp) == expected


@pytest.mark.parametrize("stamp", [0, 12345, 10**22])
def test_an_unrecognisable_timestamp_is_rejected(stamp):
    with pytest.raises(ValueError, match="timestamp"):
        detect_time_unit_ns(stamp)


def test_microsecond_stamps_produce_the_same_series(tmp_path):
    """Same instants, different unit in the file: the result must be identical."""
    ms = read_archive(write_zip(tmp_path, csv_rows()), symbol="X")
    us = read_archive(write_zip(tmp_path, csv_rows(unit_scale=1000), name="us"), symbol="X")
    assert np.array_equal(ms.ts_ns, us.ts_ns)


def test_collapsing_keeps_the_last_price_and_sums_quantity():
    ts = np.array([10, 10, 20], dtype=np.int64)
    price = np.array([1.0, 2.0, 3.0])
    qty = np.array([5.0, 7.0, 11.0])
    ts_c, price_c, qty_c = collapse_simultaneous(ts, price, qty)
    assert np.array_equal(ts_c, [10, 20])
    assert np.array_equal(price_c, [2.0, 3.0])  # last price in the millisecond
    assert np.array_equal(qty_c, [12.0, 11.0])  # volume preserved exactly
    assert qty_c.sum() == qty.sum()


def test_collapsing_leaves_untied_data_alone():
    ts = np.array([10, 20, 30], dtype=np.int64)
    price, qty = np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])
    ts_c, price_c, qty_c = collapse_simultaneous(ts, price, qty)
    assert np.array_equal(ts_c, ts) and np.array_equal(price_c, price)


def test_collapsing_rejects_unsorted_input():
    with pytest.raises(ValueError, match="sorted"):
        collapse_simultaneous(
            np.array([20, 10], dtype=np.int64), np.array([1.0, 2.0]), np.array([1.0, 1.0])
        )


def test_the_canonical_cache_round_trips_exactly(tmp_path):
    """A cache that changes the data is worse than no cache."""
    series = read_archive(write_zip(tmp_path, csv_rows()), symbol="ADAUSDT")
    path = tmp_path / "cached.npz"
    save_canonical(series, path)
    restored = load_canonical(path, symbol="ADAUSDT")
    assert np.array_equal(series.ts_ns, restored.ts_ns)
    assert np.array_equal(series.price, restored.price)
    assert np.array_equal(series.qty, restored.qty)
    assert restored.symbol == "ADAUSDT"


def test_the_result_obeys_the_trade_series_contract(tmp_path):
    """Strictly increasing timestamps, int64 ns, positive prices - enforced by
    TradeSeries itself, which is the point of collapsing before constructing."""
    series = read_archive(write_zip(tmp_path, csv_rows()), symbol="ADAUSDT")
    assert series.ts_ns.dtype == np.int64
    assert np.all(np.diff(series.ts_ns) > 0)
    assert not series.price.flags.writeable
