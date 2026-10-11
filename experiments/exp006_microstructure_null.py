"""Does a null with correlation *and* microstructure reproduce exp002's per-symbol pattern?

exp005's two nulls each drop something the real data has: the shift null keeps
microstructure but destroys correlation, the latent null keeps correlation but
has no microstructure. exp002's odd pattern (ADA "leads" all seven others, AVAX
trails all seven) could come from the two interacting, which neither null can
show. This is the latent null of exp005 - real trade clocks, one shared random
walk, true lag exactly zero - plus a bid-ask bounce on every trade: +/- h in log
price, signed by the tick rule (the sign of the last non-zero price change in
that symbol's real series, carried forward through zeros). The canonical cache
has no buyer/seller flag, so the tick rule is the only direction available.

Decision rule, fixed in PROJECT.md before this was run: a statistic is excluded
from exp002's Benjamini-Hochberg step if, under this null, either
(1) the 95% Wilson interval of its false-positive rate at nominal 5% lies
entirely above 5%, or (2) in any replicate some symbol is on the same side of
leadership in at least `cluster_min_pairs` of its 7 pairs among significant
calls (p < 0.05).

Tests within one replicate share symbols, so they are not independent; the
Wilson intervals are somewhat narrower than they should be.

    make exp EXP=exp006
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict

import numpy as np

# Experiments run as scripts, so `experiments/` is on sys.path and exp005's
# helpers are importable. They are reused so both calibrations use literally
# the same test, grid and bootstrap.
from exp005_null_calibration import (
    NS_PER_DAY,
    NS_PER_MS,
    NS_PER_S,
    _dates,
    _lag_grid,
    _load,
    _pair_tests,
    _wilson,
)

from leadlag.estimators import hayashi_yoshida_blocks
from leadlag.plotting import STATISTIC_LABELS
from leadlag.run import load_config, start_run
from leadlag.types import TradeSeries


def main() -> None:
    config = load_config("exp006")
    out = start_run("exp006", config)

    lag_grid_ns = _lag_grid(config["lag_grid"])
    block_ns = int(config["bootstrap"]["block_s"]) * NS_PER_S
    n_draws = int(config["bootstrap"]["n_draws"])
    alpha = float(config["nominal_alpha"])
    cluster_min = int(config["cluster_min_pairs"])
    params = config["null"]
    dates = _dates(config["dates"])
    symbols = sorted(config["symbols"])
    pairs = list(itertools.combinations(symbols, 2))
    replicates = range(int(params["replicates"]))

    seeds = np.random.SeedSequence(config["seed"]).spawn(len(replicates))
    rngs = {r: np.random.default_rng(seed) for r, seed in zip(replicates, seeds, strict=True)}

    blocks: dict[tuple, list] = {(r, p): [] for r in replicates for p in pairs}
    for date in dates:
        series = {s: _load(config, s, date) for s in symbols}
        day_start = int(np.datetime64(date, "ns").astype(np.int64))
        edges = np.arange(day_start, day_start + NS_PER_DAY + 1, block_ns, dtype=np.int64)
        for rep in replicates:
            rng = rngs[rep]
            for a, b in pairs:
                if series[a] is None or series[b] is None:
                    continue
                sa, sb = _latent_bounce(series[a], series[b], params, rng)
                blocks[(rep, (a, b))].append(
                    hayashi_yoshida_blocks(sa, sb, lag_grid_ns=lag_grid_ns, block_edges_ns=edges)
                )
        print(f"  {date} done", flush=True)

    rows = []
    for (rep, (a, b)), day_blocks in blocks.items():
        if not day_blocks:
            continue
        p_asym, p_peak, asym, peak = _pair_tests(day_blocks, lag_grid_ns, n_draws, rngs[rep])
        rows.append(
            {
                "replicate": rep,
                "symbol_a": a,
                "symbol_b": b,
                "asymmetry": asym,
                "peak_lag_ms": peak / NS_PER_MS,
                "p_asymmetry": p_asym,
                "p_peak_lag": p_peak,
            }
        )
    _write(out / "null_pvalues.csv", rows)

    calibration, clusters = [], []
    for statistic in STATISTIC_LABELS:
        p = np.array([r[f"p_{statistic}"] for r in rows])
        hits = int(np.sum(p < alpha))
        lo, hi = _wilson(hits, p.size)
        calibration.append(
            {
                "statistic": statistic,
                "n_tests": p.size,
                "false_positives": hits,
                "rate": hits / p.size,
                "rate_lo": lo,
                "rate_hi": hi,
                "calibrated": lo <= alpha,
            }
        )
        clusters += _clusters(rows, statistic, alpha, cluster_min)
    _write(out / "calibration.csv", calibration)
    _write(out / "clusters.csv", clusters)

    for c in calibration:
        print(
            f"  {STATISTIC_LABELS[c['statistic']]:>16}: "
            f"{c['false_positives']:3d}/{c['n_tests']} = {c['rate']:.3f} "
            f"[{c['rate_lo']:.3f}, {c['rate_hi']:.3f}]  "
            f"{'calibrated' if c['calibrated'] else 'NOT calibrated'}"
        )
    for statistic in STATISTIC_LABELS:
        calibrated = next(c["calibrated"] for c in calibration if c["statistic"] == statistic)
        hit = [c for c in clusters if c["statistic"] == statistic and c["clustered"]]
        for c in hit:
            print(
                f"  {STATISTIC_LABELS[statistic]}: replicate {c['replicate']}, {c['symbol']} "
                f"leads {c['leads']} / trails {c['trails']} of {c['n_pairs']} pairs"
            )
        usable = calibrated and not hit
        print(f"{STATISTIC_LABELS[statistic]}: {'usable' if usable else 'EXCLUDED'} in BH")
    print(f"wrote {out}")


def _latent_bounce(
    a: TradeSeries, b: TradeSeries, params: dict, rng
) -> tuple[TradeSeries, TradeSeries]:
    """One random walk observed at A's and B's real timestamps, plus a tick-signed bounce."""
    union = np.union1d(a.ts_ns, b.ts_ns)
    dt_s = np.diff(union, prepend=union[0]) / NS_PER_S
    vol = float(params["volatility_per_sqrt_s"])
    path = np.cumsum(rng.normal(0.0, 1.0, union.size) * vol * np.sqrt(dt_s))
    noise = float(params["noise"])
    fraction = float(params["half_spread_fraction"])

    def observe(series: TradeSeries) -> TradeSeries:
        sign, half_spread = _tick_signs(series.price)
        log_price = path[np.searchsorted(union, series.ts_ns)]
        log_price = log_price + rng.normal(0.0, noise, series.ts_ns.size)
        log_price = log_price + fraction * half_spread * sign
        return TradeSeries(
            symbol=series.symbol,
            ts_ns=series.ts_ns,
            price=100.0 * np.exp(log_price),
            qty=series.qty,
        )

    return observe(a), observe(b)


def _tick_signs(price: np.ndarray) -> tuple[np.ndarray, float]:
    """Tick-rule trade signs (+1 up, -1 down, carried through zeros) and the typical move.

    The second value is the median absolute non-zero log price change - roughly
    one tick - which `_latent_bounce` scales into the bounce size.
    """
    change = np.diff(np.log(price), prepend=np.log(price[0]))
    nonzero = change != 0
    if not nonzero.any():
        return np.ones(price.size), 0.0
    # Index of the most recent non-zero change at or before each trade; trades
    # before the first one take that first change's sign.
    last = np.maximum.accumulate(np.where(nonzero, np.arange(price.size), 0))
    first = np.argmax(nonzero)
    sign = np.sign(change[np.where(nonzero[last], last, first)])
    return sign, float(np.median(np.abs(change[nonzero])))


def _clusters(rows: list[dict], statistic: str, alpha: float, minimum: int) -> list[dict]:
    """Per replicate and symbol: how many significant calls put it on the same side.

    A positive statistic means the first symbol of the pair leads, as in exp002.
    """
    leads: dict[tuple, int] = defaultdict(int)
    trails: dict[tuple, int] = defaultdict(int)
    npairs: dict[tuple, int] = defaultdict(int)
    value = "asymmetry" if statistic == "asymmetry" else "peak_lag_ms"
    for r in rows:
        for symbol in (r["symbol_a"], r["symbol_b"]):
            npairs[(r["replicate"], symbol)] += 1
        if r[f"p_{statistic}"] >= alpha or r[value] == 0:
            continue
        a_leads = r[value] > 0
        leads[(r["replicate"], r["symbol_a" if a_leads else "symbol_b"])] += 1
        trails[(r["replicate"], r["symbol_b" if a_leads else "symbol_a"])] += 1
    return [
        {
            "statistic": statistic,
            "replicate": rep,
            "symbol": symbol,
            "n_pairs": n,
            "leads": leads[(rep, symbol)],
            "trails": trails[(rep, symbol)],
            "clustered": max(leads[(rep, symbol)], trails[(rep, symbol)]) >= minimum,
        }
        for (rep, symbol), n in sorted(npairs.items())
    ]


def _write(path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
