"""Ingestion: the messy edge of the pipeline.

Every failure here is one the archive really produces - a period that was never
published, a truncated download, a disk filling up. None of them may be
silent, and none of them may abort a run over the other 300 files.

The network is injected, not mocked at the socket: `fetch` is an ordinary
callable, so a test can serve a corrupted payload or a 404 deterministically
and in microseconds. Nothing in this file touches the internet.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import pytest

from leadlag.ingest import (
    Status,
    archive_url,
    checksum_url,
    ingest,
    parse_checksum,
)

SYMBOL = "ADAUSDT"
DAY = dt.date(2025, 9, 5)
PAYLOAD = b"not really a zip, but bytes are bytes"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


def fake_archive(payload=PAYLOAD, digest=None):
    """A fetcher serving one archive and its sidecar. None means a 404."""
    digest = hashlib.sha256(payload).hexdigest() if digest is None else digest
    contents = {
        archive_url(SYMBOL, DAY): payload,
        checksum_url(SYMBOL, DAY): f"{digest}  {SYMBOL}-aggTrades-{DAY}.zip\n".encode(),
    }
    return lambda url: contents.get(url)


def test_urls_match_the_published_layout():
    """Checked against the live archive when this was written."""
    assert archive_url(SYMBOL, DAY) == (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        "ADAUSDT/ADAUSDT-aggTrades-2025-09-05.zip"
    )
    assert checksum_url(SYMBOL, DAY) == archive_url(SYMBOL, DAY) + ".CHECKSUM"


def test_parses_the_sidecar_format():
    """The sidecar is '<sha256>  <filename>', two spaces, as sha256sum writes it."""
    assert parse_checksum(f"{DIGEST}  ADAUSDT-aggTrades-2025-09-05.zip\n") == DIGEST


def test_downloads_verifies_and_caches(tmp_path):
    results = ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=fake_archive())
    (result,) = results
    assert result.status is Status.DOWNLOADED
    assert result.sha256 == DIGEST
    assert result.n_bytes == len(PAYLOAD)
    assert result.path.read_bytes() == PAYLOAD


def test_a_cached_archive_is_not_downloaded_again(tmp_path):
    ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=fake_archive())

    def refuse(url):
        raise AssertionError(f"should not have fetched {url}")

    (result,) = ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=refuse)
    assert result.status is Status.CACHED
    assert result.sha256 == DIGEST


def test_a_period_that_was_never_published_is_reported_not_raised(tmp_path):
    """Gaps in the archive are normal. One missing day must not end the run."""
    (result,) = ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=lambda url: None)
    assert result.status is Status.MISSING
    assert not result.path.exists()


def test_a_corrupted_download_is_rejected_and_not_cached(tmp_path):
    """The checksum says one thing and the bytes say another.

    The archive must not be kept: a bad file left on disk would be treated as
    cached on the next run and never re-fetched.
    """
    fetch = fake_archive(payload=b"truncated", digest=DIGEST)
    (result,) = ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=fetch)
    assert result.status is Status.CHECKSUM_MISMATCH
    assert not result.path.exists()


def test_a_missing_sidecar_is_reported(tmp_path):
    """An archive with no checksum cannot be verified, so it is not trusted."""
    fetch = lambda url: PAYLOAD if url.endswith(".zip") else None  # noqa: E731
    (result,) = ingest([SYMBOL], [DAY], cache_dir=tmp_path, fetch=fetch)
    assert result.status is Status.NO_CHECKSUM
    assert not result.path.exists()


def test_one_failure_does_not_stop_the_others(tmp_path):
    days = [dt.date(2025, 9, 5), dt.date(2025, 9, 6), dt.date(2025, 9, 7)]
    contents = {
        archive_url(SYMBOL, days[0]): PAYLOAD,
        checksum_url(SYMBOL, days[0]): f"{DIGEST}  x.zip".encode(),
        archive_url(SYMBOL, days[2]): PAYLOAD,
        checksum_url(SYMBOL, days[2]): f"{DIGEST}  x.zip".encode(),
    }
    results = ingest([SYMBOL], days, cache_dir=tmp_path, fetch=contents.get)
    assert [r.status for r in results] == [
        Status.DOWNLOADED,
        Status.MISSING,
        Status.DOWNLOADED,
    ]


def test_writes_a_manifest_of_every_attempt(tmp_path):
    """Including the failures. A manifest that lists only successes is a lie
    about what the dataset contains."""
    days = [dt.date(2025, 9, 5), dt.date(2025, 9, 6)]
    ingest([SYMBOL], days, cache_dir=tmp_path, fetch=fake_archive())
    lines = (tmp_path / "manifest.csv").read_text().splitlines()
    assert lines[0].startswith("symbol,date,status")
    assert len(lines) == 3
    assert "missing" in lines[2]


def test_the_disk_budget_is_enforced(tmp_path):
    days = [dt.date(2025, 9, d) for d in (5, 6, 7)]
    contents = {}
    for day in days:
        contents[archive_url(SYMBOL, day)] = PAYLOAD
        contents[checksum_url(SYMBOL, day)] = f"{DIGEST}  x.zip".encode()
    with pytest.raises(RuntimeError, match="budget"):
        ingest(
            [SYMBOL],
            days,
            cache_dir=tmp_path,
            fetch=contents.get,
            budget_bytes=len(PAYLOAD) * 2,
        )


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        ingest([SYMBOL], [DAY], "somewhere")
