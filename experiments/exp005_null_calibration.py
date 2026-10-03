"""Are exp002's tests calibrated: under a known zero lag, how often do they fire?

exp002 found 26 of 28 pairs "significant" on the curve-asymmetry statistic,
with effects so small and so per-symbol that the test itself is suspect. This
experiment measures each candidate statistic's false-positive rate, on data
built to have no lead-lag at all, using exactly exp002's grid, blocks and
bootstrap.

Two nulls, because each misses something the other keeps:

- shift: real prices and trades, with B's whole day rotated by a random
  offset of 1-60 minutes, wrapping at midnight. Every within-symbol quirk
  survives - bid-ask bounce, tick size, intraday seasonality - and all
  cross-asset timing is destroyed. The wrap creates one spurious return
  per day, at midnight.
- latent: real trade timestamps, prices replaced by one shared random walk
  observed at each symbol's own times plus independent noise. The true lag
  is exactly zero but the correlation is strong, so this asks whether the
  asynchronous sampling alone biases the direction.

Decision rule, fixed before this was run: a statistic is calibrated if, under
**both** nulls, the 95% Wilson interval of its false-positive rate at nominal
5% is not entirely above 5%. Only a calibrated statistic may go into exp002's
Benjamini-Hochberg step. If neither is calibrated, neither is used, and that
is reported.

Tests within one replicate share symbols, so they are not independent; the
Wilson intervals are somewhat narrower than they should be.

    make exp EXP=exp005
"""

from __future__ import annotations

import csv
import datetime as dt
import itertools

import numpy as np

from leadlag.estimators import hayashi_yoshida_blocks
from leadlag.normalize import canonical_name, load_canonical
from leadlag.plotting import (
    PAGE_WIDTH,
    STATISTIC_COLORS,
    STATISTIC_LABELS,
    label_panel,
    new_figure,
    save_figure,
)
from leadlag.run import ROOT, load_config, start_run
from leadlag.types import TradeSeries

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
NS_PER_DAY = 86_400 * NS_PER_S
NULLS = ("shift", "latent")


def main() -> None:
    config = load_config("exp005")
    out = start_run("exp005", config)

    lag_grid_ns = _lag_grid(config["lag_grid"])
    block_ns = int(config["bootstrap"]["block_s"]) * NS_PER_S
    n_draws = int(config["bootstrap"]["n_draws"])
    alpha = float(config["nominal_alpha"])
    dates = _dates(config["dates"])
    symbols = list(config["symbols"])
    pairs = list(itertools.combinations(sorted(symbols), 2))

    replicates = [(n, r) for n in NULLS for r in range(int(config["nulls"][n]["replicates"]))]
    seeds = dict(
        zip(replicates, np.random.SeedSequence(config["seed"]).spawn(len(replicates)), strict=True)
    )
    rngs = {key: np.random.default_rng(seed) for key, seed in seeds.items()}

    # (null, replicate, pair) -> list of per-day HYBlocks. Days are the outer
    # loop so each day's eight series are read once, not once per replicate.
    blocks: dict[tuple, list] = {(n, r, p): [] for n, r in replicates for p in pairs}
    for date in dates:
        series = {s: _load(config, s, date) for s in symbols}
        day_start = int(np.datetime64(date, "ns").astype(np.int64))
        edges = np.arange(day_start, day_start + NS_PER_DAY + 1, block_ns, dtype=np.int64)
        for null, rep in replicates:
            rng = rngs[(null, rep)]
            for a, b in pairs:
                if series[a] is None or series[b] is None:
                    continue
                if null == "shift":
                    sa, sb = series[a], _rotate(series[b], day_start, config["nulls"]["shift"], rng)
                else:
                    sa, sb = _latent(series[a], series[b], config["nulls"]["latent"], rng)
                blocks[(null, rep, (a, b))].append(
                    hayashi_yoshida_blocks(sa, sb, lag_grid_ns=lag_grid_ns, block_edges_ns=edges)
                )
        print(f"  {date} done", flush=True)

    rows = []
    for (null, rep, (a, b)), day_blocks in blocks.items():
        if not day_blocks:
            continue
        p_asym, p_peak, asym, peak = _pair_tests(
            day_blocks, lag_grid_ns, n_draws, rngs[(null, rep)]
        )
        rows.append(
            {
                "null": null,
                "replicate": rep,
                "symbol_a": a,
                "symbol_b": b,
                "asymmetry": asym,
                "peak_lag_ms": peak / NS_PER_MS,
                "p_asymmetry": p_asym,
                "p_peak_lag": p_peak,
            }
        )

    with (out / "null_pvalues.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for null in NULLS:
        for statistic in STATISTIC_LABELS:
            p = np.array([r[f"p_{statistic}"] for r in rows if r["null"] == null])
            hits = int(np.sum(p < alpha))
            lo, hi = _wilson(hits, p.size)
            summary.append(
                {
                    "null": null,
                    "statistic": statistic,
                    "n_tests": p.size,
                    "false_positives": hits,
                    "rate": hits / p.size,
                    "rate_lo": lo,
                    "rate_hi": hi,
                    "calibrated": lo <= alpha,
                }
            )
    with (out / "calibration.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    _figure(out, rows, alpha)

    for s in summary:
        print(
            f"  {s['null']:>6} {STATISTIC_LABELS[s['statistic']]:>16}: "
            f"{s['false_positives']:3d}/{s['n_tests']} = {s['rate']:.3f} "
            f"[{s['rate_lo']:.3f}, {s['rate_hi']:.3f}]  "
            f"{'calibrated' if s['calibrated'] else 'NOT calibrated'}"
        )
    for statistic in STATISTIC_LABELS:
        ok = all(s["calibrated"] for s in summary if s["statistic"] == statistic)
        print(f"{STATISTIC_LABELS[statistic]}: {'usable' if ok else 'not usable'} in BH")
    print(f"wrote {out}")


def _load(config: dict, symbol: str, date: dt.date) -> TradeSeries | None:
    path = ROOT / config["canonical_dir"] / symbol / canonical_name(symbol, date)
    return load_canonical(path, symbol=symbol) if path.exists() else None


def _rotate(series: TradeSeries, day_start: int, params: dict, rng) -> TradeSeries:
    """The whole day moved by a random whole-millisecond offset, wrapping at midnight."""
    seconds = rng.uniform(float(params["min_offset_s"]), float(params["max_offset_s"]))
    offset_ns = int(rng.choice([-1, 1]) * round(seconds * 1000)) * NS_PER_MS
    shifted = (series.ts_ns - day_start + offset_ns) % NS_PER_DAY
    order = np.argsort(shifted, kind="stable")
    return TradeSeries(
        symbol=series.symbol,
        ts_ns=day_start + shifted[order],
        price=series.price[order],
        qty=series.qty[order],
    )


def _latent(a: TradeSeries, b: TradeSeries, params: dict, rng) -> tuple[TradeSeries, TradeSeries]:
    """One random walk, observed at A's and B's real timestamps. True lag: zero."""
    union = np.union1d(a.ts_ns, b.ts_ns)
    dt_s = np.diff(union, prepend=union[0]) / NS_PER_S
    vol = float(params["volatility_per_sqrt_s"])
    path = np.cumsum(rng.normal(0.0, 1.0, union.size) * vol * np.sqrt(dt_s))
    noise = float(params["noise"])

    def observe(series: TradeSeries) -> TradeSeries:
        log_price = path[np.searchsorted(union, series.ts_ns)]
        log_price = log_price + rng.normal(0.0, noise, series.ts_ns.size)
        return TradeSeries(
            symbol=series.symbol,
            ts_ns=series.ts_ns,
            price=100.0 * np.exp(log_price),
            qty=series.qty,
        )

    return observe(a), observe(b)


def _pair_tests(day_blocks, lag_grid_ns, n_draws, rng) -> tuple[float, float, float, int]:
    """exp002's bootstrap, returning both statistics' two-sided p-values."""
    covariance = np.vstack([blk.covariance for blk in day_blocks])
    variance_a = np.concatenate([blk.variance_a for blk in day_blocks])
    variance_b = np.concatenate([blk.variance_b for blk in day_blocks])
    n = covariance.shape[0]
    weights = np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1.0 / n), size=n_draws)]).astype(
        np.float64
    )
    normaliser = np.sqrt((weights @ variance_a) * (weights @ variance_b))
    curves = (weights @ covariance) / normaliser[:, None]

    positive, negative = lag_grid_ns > 0, lag_grid_ns < 0
    asymmetry = curves[:, positive].mean(axis=1) - curves[:, negative].mean(axis=1)
    peak = lag_grid_ns[np.argmax(curves, axis=1)]
    return (
        _two_sided_bootstrap_p(asymmetry[1:]),
        _two_sided_bootstrap_p(peak[1:].astype(np.float64)),
        float(asymmetry[0]),
        int(peak[0]),
    )


def _two_sided_bootstrap_p(draws: np.ndarray) -> float:
    """As in exp002: twice the smaller share of draws on either side of zero."""
    n = draws.size
    below = (np.sum(draws <= 0) + 1) / (n + 1)
    above = (np.sum(draws >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def _wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    p = hits / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(centre - half), float(centre + half)


def _lag_grid(grid: dict) -> np.ndarray:
    positive = np.arange(int(grid["half_width"]) + 1, dtype=np.int64) * int(grid["step_ns"])
    positive += int(grid["tie_offset_ns"])
    return np.concatenate([-positive[::-1], positive])


def _dates(dates: dict) -> list[dt.date]:
    return [
        dates["start"] + dt.timedelta(days=i * int(dates["stride_days"]))
        for i in range(int(dates["count"]))
    ]


def _figure(out, rows, alpha) -> None:
    """Empirical CDF of null p-values against the uniform it should follow."""
    fig, axes = new_figure(width=PAGE_WIDTH, height=2.8, ncols=2, sharey=True)
    titles = {
        "shift": "Shift null: real prices, timing destroyed",
        "latent": "Latent null: real clocks, true lag zero",
    }
    for ax, null, letter in zip(axes, NULLS, "ab", strict=True):
        ax.plot([0, 1], [0, 1], color="0.75", linewidth=0.8, zorder=0)
        ax.axvline(alpha, color="0.75", linewidth=0.8, linestyle=":", zorder=0)
        for statistic, label in STATISTIC_LABELS.items():
            p = np.sort([r[f"p_{statistic}"] for r in rows if r["null"] == null])
            ax.step(
                p,
                np.arange(1, p.size + 1) / p.size,
                where="post",
                color=STATISTIC_COLORS[statistic],
                label=label,
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Nominal p-value")
        ax.set_title(titles[null])
        label_panel(ax, letter)
    axes[0].set_ylabel("Share of null tests at or below")
    axes[0].legend(loc="lower right")
    axes[0].annotate(
        "diagonal: a calibrated test\ndotted: 5%",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        color="0.45",
        fontsize=6,
    )
    save_figure(fig, out / "null_calibration.pdf")


if __name__ == "__main__":
    main()
