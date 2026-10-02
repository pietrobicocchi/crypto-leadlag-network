"""Running Hayashi-Yoshida across the universe, cached per pair and day.

This is the orchestration layer between `normalize` and `estimators`: it
never parses a Binance file itself, it only calls into the two modules that
do. For every (symbol, day) it needs a `TradeSeries`, taken from a canonical
cache that it fills in on demand - the same resumable shape as `ingest`,
one layer up the pipeline. A date never ingested, or a pair the estimator
cannot handle, is reported on that pair's row rather than raised, so one bad
pair does not stop the other 27.
"""

from __future__ import annotations

import csv
import datetime as dt
import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np

from leadlag.estimators import hayashi_yoshida_correlation, lead_lag_ratio, peak_lag
from leadlag.ingest import archive_name
from leadlag.normalize import canonical_name, load_canonical, read_archive, save_canonical
from leadlag.types import TradeSeries


class Status(StrEnum):
    """What happened for one (pair, day). A StrEnum, so it writes itself to CSV."""

    OK = "ok"
    MISSING_DATA = "missing_data"
    ESTIMATOR_ERROR = "estimator_error"


@dataclass(frozen=True)
class PairResult:
    symbol_a: str
    symbol_b: str
    date: dt.date
    status: Status
    peak_lag_ns: int = 0
    lead_lag_ratio: float = float("nan")


def run_study(
    symbols: Iterable[str],
    dates: Iterable[dt.date],
    *,
    canonical_dir: Path,
    raw_dir: Path,
    lag_grid_ns: np.ndarray,
) -> list[PairResult]:
    """Hayashi-Yoshida over every symbol pair, for every day, cached as it goes.

    Arguments are keyword-only past the two iterables, matching `ingest`: a
    positional swap of `canonical_dir` and `raw_dir` would silently point the
    cache at the wrong directory rather than raise.

    Each `TradeSeries` is loaded once per day even though it is needed by
    several pairs, and each symbol's canonical file is written at most once:
    `_series_cache` remembers both a hit and a miss within one call.
    """
    canonical_dir = Path(canonical_dir)
    raw_dir = Path(raw_dir)
    pairs = list(itertools.combinations(sorted(symbols), 2))

    results: list[PairResult] = []
    for date in dates:
        series_cache: dict[str, TradeSeries | None] = {}
        for symbol_a, symbol_b in pairs:
            series_a = _get_series(symbol_a, date, canonical_dir, raw_dir, series_cache)
            series_b = _get_series(symbol_b, date, canonical_dir, raw_dir, series_cache)
            if series_a is None or series_b is None:
                results.append(PairResult(symbol_a, symbol_b, date, Status.MISSING_DATA))
                continue
            try:
                correlation = hayashi_yoshida_correlation(
                    series_a, series_b, lag_grid_ns=lag_grid_ns
                )
            except ValueError:
                results.append(PairResult(symbol_a, symbol_b, date, Status.ESTIMATOR_ERROR))
                continue
            results.append(
                PairResult(
                    symbol_a,
                    symbol_b,
                    date,
                    Status.OK,
                    peak_lag(lag_grid_ns, correlation),
                    lead_lag_ratio(lag_grid_ns, correlation),
                )
            )
    return results


def _get_series(
    symbol: str,
    date: dt.date,
    canonical_dir: Path,
    raw_dir: Path,
    cache: dict[str, TradeSeries | None],
) -> TradeSeries | None:
    """A symbol's `TradeSeries` for one day, or None if nothing is on disk.

    Resumes the way `ingest` resumes a download: a canonical cache already
    there is trusted and loaded; a raw archive with no canonical cache yet is
    normalized once and the result is written for next time; neither existing
    is a gap, not an error.
    """
    if symbol in cache:
        return cache[symbol]

    canonical_path = canonical_dir / symbol / canonical_name(symbol, date)
    if canonical_path.exists():
        cache[symbol] = load_canonical(canonical_path, symbol=symbol)
        return cache[symbol]

    raw_path = raw_dir / symbol / archive_name(symbol, date)
    if raw_path.exists():
        series = read_archive(raw_path, symbol=symbol)
        save_canonical(series, canonical_path)
        cache[symbol] = series
        return series

    cache[symbol] = None
    return None


def save_results(results: list[PairResult], path: Path) -> Path:
    """Write one row per result, failures included.

    Mirrors `ingest._write_manifest`: a cache that records only successes
    cannot tell exp002 which pairs it still needs to explain.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol_a", "symbol_b", "date", "status", "peak_lag_ns", "lead_lag_ratio"])
        for r in results:
            writer.writerow(
                [
                    r.symbol_a,
                    r.symbol_b,
                    r.date.isoformat(),
                    r.status.value,
                    r.peak_lag_ns,
                    r.lead_lag_ratio,
                ]
            )
    return path


def main(config_name: str = "study") -> None:
    """Run the configured universe over the configured days. `make study`."""
    from leadlag.run import ROOT, load_config

    config = load_config(config_name)
    dates_cfg = config["dates"]
    start = dates_cfg["start"]
    dates = [
        start + dt.timedelta(days=i * int(dates_cfg["stride_days"]))
        for i in range(int(dates_cfg["count"]))
    ]

    grid = config["lag_grid"]
    half_width = int(grid["half_width"])
    lag_grid_ns = np.arange(-half_width, half_width + 1, dtype=np.int64) * int(grid["step_ns"])

    results = run_study(
        config["symbols"],
        dates,
        canonical_dir=ROOT / config["canonical_dir"],
        raw_dir=ROOT / config["raw_dir"],
        lag_grid_ns=lag_grid_ns,
    )

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status:>18}: {count}")

    out_path = ROOT / config["results_path"]
    save_results(results, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "study")
