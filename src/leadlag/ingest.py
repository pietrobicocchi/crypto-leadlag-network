"""Fetching Binance archives, verified.

This is the one module that talks to the network, and it is deliberately the
dumbest one. It knows *where* bytes come from - the URL layout of
data.binance.vision and its `.CHECKSUM` sidecars - and nothing whatsoever about
what is inside them. Columns, headers, timestamp units and every other
venue-specific quirk belong to `normalize`.

Archives are content-addressed by their published SHA-256, so nothing is
trusted until it has been verified, and nothing verified is fetched twice.

Every outcome is recorded, including the failures. A period that was never
published is normal in this archive and is reported rather than raised: one
gap must not end a run over three hundred other files.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades"

# A fetcher returns the body, or None if the resource was never published.
# Anything else - a timeout, a 500 - is an error and propagates.
Fetcher = Callable[[str], bytes | None]


class Status(StrEnum):
    """What happened to one (symbol, day). A StrEnum, so it writes itself to CSV."""

    DOWNLOADED = "downloaded"
    CACHED = "cached"
    MISSING = "missing"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    NO_CHECKSUM = "no_checksum"


@dataclass(frozen=True)
class IngestResult:
    symbol: str
    date: dt.date
    status: Status
    path: Path
    n_bytes: int = 0
    sha256: str = ""


def archive_url(symbol: str, date: dt.date) -> str:
    """URL of one day of aggregated trades for one USD-M perpetual."""
    return f"{BASE_URL}/{symbol}/{archive_name(symbol, date)}"


def checksum_url(symbol: str, date: dt.date) -> str:
    """URL of the SHA-256 sidecar published beside every archive."""
    return archive_url(symbol, date) + ".CHECKSUM"


def archive_name(symbol: str, date: dt.date) -> str:
    return f"{symbol}-aggTrades-{date:%Y-%m-%d}.zip"


def parse_checksum(text: str) -> str:
    """Pull the digest out of a sidecar.

    The format is the one `sha256sum` writes: the hex digest, two spaces, the
    filename. Only the digest is used - the filename is already known.
    """
    digest = text.split()[0] if text.split() else ""
    if len(digest) != 64:
        raise ValueError(f"expected a 64-character sha256 digest, got {digest!r}")
    return digest


def ingest(
    symbols: Iterable[str],
    dates: Iterable[dt.date],
    *,
    cache_dir: Path,
    fetch: Fetcher | None = None,
    budget_bytes: int | None = None,
) -> list[IngestResult]:
    """Fetch every (symbol, day), verify it, and write a manifest of all of it.

    Arguments are keyword-only past the two iterables: a positional swap of
    `cache_dir` and `fetch` would be caught, but one between symbols and dates
    would not.

    `budget_bytes` caps what may be written to `cache_dir` in this run. Real
    archives run to tens of gigabytes across a full universe, and a job that
    silently fills a disk overnight is a worse failure than one that stops.
    """
    fetch = download_bytes if fetch is None else fetch
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    known = _read_manifest(cache_dir)
    results: list[IngestResult] = []
    total = _cache_size(cache_dir)
    for symbol in symbols:
        for date in dates:
            result = _one(symbol, date, cache_dir, fetch, known)
            if result.status is Status.DOWNLOADED:
                total += result.n_bytes
                if budget_bytes is not None and total > budget_bytes:
                    result.path.unlink(missing_ok=True)
                    _write_manifest(cache_dir, results)
                    raise RuntimeError(
                        f"disk budget of {budget_bytes} bytes exceeded: the cache "
                        f"would reach {total} bytes after {len(results)} archives. "
                        f"Manifest written to {cache_dir / 'manifest.csv'}"
                    )
            results.append(result)
    _write_manifest(cache_dir, results)
    return results


def _one(
    symbol: str,
    date: dt.date,
    cache_dir: Path,
    fetch: Fetcher,
    known: dict[tuple[str, str], str],
) -> IngestResult:
    path = cache_dir / symbol / archive_name(symbol, date)

    # A previously verified archive is re-checked against the digest the
    # manifest recorded, on disk. Re-running ingestion therefore needs no
    # network at all, and still catches a file that rotted since.
    recorded = known.get((symbol, date.isoformat()))
    if recorded and path.exists() and _digest(path) == recorded:
        return IngestResult(symbol, date, Status.CACHED, path, path.stat().st_size, recorded)

    # The archive before the sidecar: a day that was never published is the
    # common case, and this way it costs one 404 rather than two.
    body = fetch(archive_url(symbol, date))
    if body is None:
        return IngestResult(symbol, date, Status.MISSING, path)

    sidecar = fetch(checksum_url(symbol, date))
    if sidecar is None:
        # The archive exists but nothing can verify it. An unverifiable file is
        # not worth the storage or the doubt.
        return IngestResult(symbol, date, Status.NO_CHECKSUM, path)
    expected = parse_checksum(sidecar.decode())

    actual = hashlib.sha256(body).hexdigest()
    if actual != expected:
        # Never keep it. A corrupted archive left on disk looks cached on the
        # next run and would never be fetched again.
        return IngestResult(symbol, date, Status.CHECKSUM_MISMATCH, path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return IngestResult(symbol, date, Status.DOWNLOADED, path, len(body), actual)


def _read_manifest(cache_dir: Path) -> dict[tuple[str, str], str]:
    """Digests recorded by a previous run, so caching needs no network."""
    manifest = cache_dir / "manifest.csv"
    if not manifest.exists():
        return {}
    with manifest.open(newline="") as handle:
        return {
            (row["symbol"], row["date"]): row["sha256"]
            for row in csv.DictReader(handle)
            if row["sha256"]
        }


def _cache_size(cache_dir: Path) -> int:
    """Bytes already stored, so the budget caps total disk and not just this run."""
    return sum(p.stat().st_size for p in cache_dir.rglob("*.zip"))


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(cache_dir: Path, results: list[IngestResult]) -> None:
    """One row per attempt, failures included.

    A manifest listing only what succeeded is a lie about what the dataset
    contains, and the gaps are exactly what a later analysis has to know.
    """
    with (cache_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["symbol", "date", "status", "n_bytes", "sha256", "path"])
        for r in results:
            writer.writerow(
                [r.symbol, r.date.isoformat(), r.status.value, r.n_bytes, r.sha256, r.path]
            )


def download_bytes(url: str, *, attempts: int = 3, timeout_s: float = 60.0) -> bytes | None:
    """Fetch a URL, or None if it was never published.

    A 404 in this archive means the period does not exist, which is ordinary.
    Everything else is retried with exponential backoff and then raised: a
    transient network fault must not be recorded as missing data.
    """
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            if attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2.0**attempt)
    return None


def main(config_name: str = "ingest") -> None:
    """Fetch everything `configs/<config_name>.yaml` asks for, and report.

    Run with `make ingest`. Safe to re-run: verified archives are not fetched
    again, so an interrupted download resumes where it stopped.
    """
    from leadlag.run import ROOT, load_config

    config = load_config(config_name)
    start = config["dates"]["start"]
    dates = [
        start + dt.timedelta(days=i * int(config["dates"]["stride_days"]))
        for i in range(int(config["dates"]["count"]))
    ]
    cache_dir = ROOT / config["cache_dir"]

    results = ingest(
        config["symbols"],
        dates,
        cache_dir=cache_dir,
        budget_bytes=config.get("budget_bytes"),
    )

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    for status, count in sorted(counts.items()):
        print(f"  {status:>18}: {count}")
    print(f"  {'cache size':>18}: {_cache_size(cache_dir) / 1e9:.2f} GB")
    print(f"wrote {cache_dir / 'manifest.csv'}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "ingest")
