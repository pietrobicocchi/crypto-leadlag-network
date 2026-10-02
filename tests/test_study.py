"""Running Hayashi-Yoshida across the universe, cached per pair and day.

`study` ties `normalize` and `estimators` together. It is the one place a
gap - a date never ingested, a pair whose estimator cannot be computed - is
reported rather than raised, so one bad pair must not stop the other 27.

Canonical caching is built in-process with synthetic `TradeSeries`; the
on-demand-normalize path is exercised with a real zip, built the same way
`test_normalize.py` builds one. Nothing here touches the network or a real
archive.
"""

from __future__ import annotations

import datetime as dt
import zipfile

import numpy as np
import pytest

from leadlag.normalize import HEADER, canonical_name, save_canonical
from leadlag.study import PairResult, Status, run_study, save_results
from leadlag.synthetic import delayed_pair
from leadlag.types import TradeSeries

DAY = dt.date(2026, 3, 2)
LAG_GRID_NS = np.arange(-10, 11, dtype=np.int64) * 50_000_000  # +/-500ms, 50ms steps


def _pair(rng, *, lag_ns=0, rate_ratio=1.0):
    return delayed_pair(
        duration_s=60.0,
        rate_a_hz=20.0 * rate_ratio,
        rate_b_hz=20.0,
        lag_ns=lag_ns,
        volatility=1e-3,
        noise=1e-6,
        rng=rng,
    )


def _cache(tmp_path, symbol, date, series):
    path = tmp_path / "canonical" / symbol / canonical_name(symbol, date)
    save_canonical(series, path)
    return path


def test_computes_a_result_for_a_pair_with_cached_data(tmp_path):
    rng = np.random.default_rng(0)
    a, b = _pair(rng)
    _cache(tmp_path, "AAA", DAY, a)
    _cache(tmp_path, "BBB", DAY, b)

    (result,) = run_study(
        ["AAA", "BBB"],
        [DAY],
        canonical_dir=tmp_path / "canonical",
        raw_dir=tmp_path / "raw",
        lag_grid_ns=LAG_GRID_NS,
    )
    assert result.status is Status.OK
    assert result.symbol_a == "AAA" and result.symbol_b == "BBB"
    assert result.date == DAY
    assert not np.isnan(result.lead_lag_ratio)


def test_a_pair_with_no_data_anywhere_is_reported_missing(tmp_path):
    (result,) = run_study(
        ["AAA", "BBB"],
        [DAY],
        canonical_dir=tmp_path / "canonical",
        raw_dir=tmp_path / "raw",
        lag_grid_ns=LAG_GRID_NS,
    )
    assert result.status is Status.MISSING_DATA
    assert np.isnan(result.lead_lag_ratio)


def test_one_missing_pair_does_not_stop_the_others(tmp_path):
    rng = np.random.default_rng(1)
    a, b = _pair(rng)
    _cache(tmp_path, "AAA", DAY, a)
    _cache(tmp_path, "BBB", DAY, b)
    # CCC has no data anywhere.

    results = run_study(
        ["AAA", "BBB", "CCC"],
        [DAY],
        canonical_dir=tmp_path / "canonical",
        raw_dir=tmp_path / "raw",
        lag_grid_ns=LAG_GRID_NS,
    )
    statuses = {(r.symbol_a, r.symbol_b): r.status for r in results}
    assert statuses[("AAA", "BBB")] is Status.OK
    assert statuses[("AAA", "CCC")] is Status.MISSING_DATA
    assert statuses[("BBB", "CCC")] is Status.MISSING_DATA


def test_a_raw_archive_without_a_canonical_cache_is_normalized_on_demand(tmp_path):
    """study resumes the way ingest does: a raw archive already on disk must
    not require a separate manual normalize step before study can use it."""
    raw_dir = tmp_path / "raw"
    canonical_dir = tmp_path / "canonical"
    rows = [
        "1,100.5,2.0,10,10,1772409600000,true",
        "2,100.7,3.0,11,11,1772409600500,false",
    ]
    for symbol in ("AAA", "BBB"):
        path = raw_dir / symbol / f"{symbol}-aggTrades-{DAY}.zip"
        path.parent.mkdir(parents=True)
        body = "\n".join([",".join(HEADER)] + rows) + "\n"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"{symbol}-aggTrades-{DAY}.csv", body)

    results = run_study(
        ["AAA", "BBB"],
        [DAY],
        canonical_dir=canonical_dir,
        raw_dir=raw_dir,
        lag_grid_ns=LAG_GRID_NS,
    )
    assert (canonical_dir / "AAA" / canonical_name("AAA", DAY)).exists()
    assert (canonical_dir / "BBB" / canonical_name("BBB", DAY)).exists()
    assert results[0].status is not Status.MISSING_DATA


def test_an_unmovable_series_is_reported_as_an_estimator_error_not_raised(tmp_path):
    """Hayashi-Yoshida raises on a series with zero variance. A real pair
    cannot be this degenerate, but a crashed run over the other 27 pairs for
    one dead symbol would be a worse outcome than reporting it and moving on."""
    ts = np.array([0, 1_000_000, 2_000_000], dtype=np.int64)
    flat = TradeSeries(
        symbol="AAA", ts_ns=ts, price=np.full(3, 100.0), qty=np.ones(3)
    )
    moving = TradeSeries(
        symbol="BBB", ts_ns=ts, price=np.array([100.0, 101.0, 99.0]), qty=np.ones(3)
    )
    _cache(tmp_path, "AAA", DAY, flat)
    _cache(tmp_path, "BBB", DAY, moving)

    (result,) = run_study(
        ["AAA", "BBB"],
        [DAY],
        canonical_dir=tmp_path / "canonical",
        raw_dir=tmp_path / "raw",
        lag_grid_ns=LAG_GRID_NS,
    )
    assert result.status is Status.ESTIMATOR_ERROR
    assert np.isnan(result.lead_lag_ratio)


def test_one_result_per_pair_per_date(tmp_path):
    rng = np.random.default_rng(2)
    for date in (DAY, dt.date(2026, 3, 8)):
        a, b = _pair(rng)
        _cache(tmp_path, "AAA", date, a)
        _cache(tmp_path, "BBB", date, b)

    results = run_study(
        ["AAA", "BBB"],
        [DAY, dt.date(2026, 3, 8)],
        canonical_dir=tmp_path / "canonical",
        raw_dir=tmp_path / "raw",
        lag_grid_ns=LAG_GRID_NS,
    )
    assert len(results) == 2
    assert {r.date for r in results} == {DAY, dt.date(2026, 3, 8)}


def test_save_results_writes_one_row_per_result(tmp_path):
    results = [
        PairResult("AAA", "BBB", DAY, Status.OK, 50_000_000, 1.23),
        PairResult("AAA", "CCC", DAY, Status.MISSING_DATA),
    ]
    path = save_results(results, tmp_path / "results.csv")
    lines = path.read_text().splitlines()
    assert lines[0].startswith("symbol_a,symbol_b,date,status")
    assert len(lines) == 3
    assert "missing_data" in lines[2]


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        run_study(["AAA"], [DAY], "canon", "raw", LAG_GRID_NS)
