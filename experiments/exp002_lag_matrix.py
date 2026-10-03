"""Which pairs have a lead-lag structure that survives multiple-testing correction?

For every pair in the eight-symbol universe, Hayashi-Yoshida on the fine lag
grid (1 ms steps, +/-50.5 ms, half a millisecond off the clock), pooled over the
ten sampled days. Uncertainty comes from a block bootstrap: each day is split
into 5-minute blocks, the estimator's sums are filed by block
(`hayashi_yoshida_blocks`), and each draw re-weights the blocks and re-sums.
No series is cut and re-stitched, so no interval is invented at a seam.

The tested quantity is the curve's asymmetry - mean correlation at positive
lags minus mean at negative lags - not the peak lag: on curves this flat the
argmax is noisy, and a test on it would have little power. The peak lag is
reported with its own interval, as the human-readable size of the effect.
Benjamini-Hochberg then controls the false discovery rate over all 28 pairs.

Sign convention, as in `leadlag.estimators`: a positive lag or asymmetry for
pair (A, B) means B trails A, i.e. A leads.

    make exp EXP=exp002
"""

from __future__ import annotations

import csv
import datetime as dt
import itertools

import numpy as np

from leadlag.estimators import hayashi_yoshida_blocks
from leadlag.normalize import canonical_name, load_canonical
from leadlag.plotting import (
    LEAD_LAG_CMAP,
    METHOD_COLORS,
    NOT_SIGNIFICANT_COLOR,
    PAGE_WIDTH,
    label_panel,
    new_figure,
    save_figure,
)
from leadlag.run import ROOT, load_config, start_run

NS_PER_MS = 1_000_000
NS_PER_DAY = 86_400 * 1_000_000_000


def main() -> None:
    config = load_config("exp002")
    out = start_run("exp002", config)
    rng = np.random.default_rng(config["seed"])

    lag_grid_ns = _lag_grid(config["lag_grid"])
    positive, negative = lag_grid_ns > 0, lag_grid_ns < 0
    boot = config["bootstrap"]
    block_ns = int(boot["block_s"]) * 1_000_000_000
    dates = _dates(config["dates"])
    symbols = list(config["symbols"])
    pairs = list(itertools.combinations(sorted(symbols), 2))

    # Per pair, the per-block sums of every day, stacked into one table.
    blocks: dict[tuple[str, str], list] = {pair: [] for pair in pairs}
    days_used: dict[tuple[str, str], int] = {pair: 0 for pair in pairs}
    trade_counts = dict.fromkeys(symbols, 0)
    missing: list[str] = []
    for date in dates:
        series = {}
        for symbol in symbols:
            path = ROOT / config["canonical_dir"] / symbol / canonical_name(symbol, date)
            if path.exists():
                series[symbol] = load_canonical(path, symbol=symbol)
                trade_counts[symbol] += len(series[symbol])
            else:
                missing.append(f"{symbol} {date}")
        day_start = int(np.datetime64(date, "ns").astype(np.int64))
        edges = np.arange(day_start, day_start + NS_PER_DAY + 1, block_ns, dtype=np.int64)
        for a, b in pairs:
            if a in series and b in series:
                blocks[(a, b)].append(
                    hayashi_yoshida_blocks(
                        series[a], series[b], lag_grid_ns=lag_grid_ns, block_edges_ns=edges
                    )
                )
                days_used[(a, b)] += 1
        print(f"  {date}: {len(series)}/{len(symbols)} symbols", flush=True)

    rows = []
    alpha = 1.0 - float(boot["ci_level"])
    for a, b in pairs:
        covariance = np.vstack([blk.covariance for blk in blocks[(a, b)]])
        variance_a = np.concatenate([blk.variance_a for blk in blocks[(a, b)]])
        variance_b = np.concatenate([blk.variance_b for blk in blocks[(a, b)]])
        n_blocks = covariance.shape[0]

        # One bootstrap draw = how many times each block is picked: a multinomial
        # count vector. Re-summing is then a matrix product, (draws x blocks) @
        # (blocks x lags), instead of materialising every resampled table.
        weights = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks), size=boot["n_draws"])
        weights = weights.astype(np.float64)
        full = _correlation(np.ones(n_blocks), covariance, variance_a, variance_b)[0]
        draws = _correlation(weights, covariance, variance_a, variance_b)

        asymmetry = full[positive].mean() - full[negative].mean()
        asymmetry_draws = draws[:, positive].mean(axis=1) - draws[:, negative].mean(axis=1)
        peak_draws = lag_grid_ns[np.argmax(draws, axis=1)]

        rows.append(
            {
                "symbol_a": a,
                "symbol_b": b,
                "n_days": days_used[(a, b)],
                "n_blocks": n_blocks,
                "peak_lag_ms": lag_grid_ns[np.argmax(full)] / NS_PER_MS,
                "peak_lag_lo_ms": np.quantile(peak_draws, alpha / 2) / NS_PER_MS,
                "peak_lag_hi_ms": np.quantile(peak_draws, 1 - alpha / 2) / NS_PER_MS,
                "peak_correlation": float(full.max()),
                "asymmetry": float(asymmetry),
                "asymmetry_lo": float(np.quantile(asymmetry_draws, alpha / 2)),
                "asymmetry_hi": float(np.quantile(asymmetry_draws, 1 - alpha / 2)),
                "p_value": _two_sided_bootstrap_p(asymmetry_draws),
            }
        )

    q_values = _benjamini_hochberg([r["p_value"] for r in rows])
    for row, q in zip(rows, q_values, strict=True):
        row["q_value"] = q
        row["significant"] = bool(q <= float(config["fdr"]))

    with (out / "pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "missing.txt").open("w") as handle:
        handle.write("".join(f"{m}\n" for m in missing))

    # Rows and columns ordered by trading activity: distinct trade timestamps,
    # summed over the days. The hypothesis is that leadership tracks liquidity.
    order = sorted(symbols, key=lambda s: -trade_counts[s])
    _figure(out, rows, order)

    n_sig = sum(r["significant"] for r in rows)
    print(
        f"{n_sig}/{len(rows)} pairs significant at FDR {config['fdr']}; "
        f"missing files: {len(missing)}"
    )
    for r in sorted(rows, key=lambda r: r["q_value"]):
        # Named leader first, from the asymmetry's sign; lags oriented to match,
        # so a positive peak means the follower trails the named leader.
        sign = 1.0 if r["asymmetry"] >= 0 else -1.0
        leader, follower = _pair_label(r).split(" → ")
        lo, hi = sorted((sign * r["peak_lag_lo_ms"], sign * r["peak_lag_hi_ms"]))
        print(
            f"  {leader:>5} leads {follower:<5} peak {sign * r['peak_lag_ms']:+6.1f} ms "
            f"[{lo:+6.1f}, {hi:+6.1f}]  asymmetry {sign * r['asymmetry']:.4f}  "
            f"q={r['q_value']:.4f} {'*' if r['significant'] else ''}"
        )
    print(f"wrote {out}")


def _lag_grid(grid: dict) -> np.ndarray:
    """+/-(k*step + offset) for k = 0..half_width: symmetric, never on a whole tick."""
    positive = np.arange(int(grid["half_width"]) + 1, dtype=np.int64) * int(grid["step_ns"])
    positive += int(grid["tie_offset_ns"])
    return np.concatenate([-positive[::-1], positive])


def _dates(dates: dict) -> list[dt.date]:
    return [
        dates["start"] + dt.timedelta(days=i * int(dates["stride_days"]))
        for i in range(int(dates["count"]))
    ]


def _correlation(weights, covariance, variance_a, variance_b) -> np.ndarray:
    """HY correlation curve for each row of block weights: (draws, lags)."""
    weights = np.atleast_2d(weights)
    normaliser = np.sqrt((weights @ variance_a) * (weights @ variance_b))
    return (weights @ covariance) / normaliser[:, None]


def _two_sided_bootstrap_p(draws: np.ndarray) -> float:
    """Twice the smaller share of draws on either side of zero.

    The +1 in numerator and denominator counts the observed sample as one of
    the draws, so the p-value is never exactly 0 - 2000 draws cannot support a
    claim stronger than about 1 in 1000.
    """
    n = draws.size
    below = (np.sum(draws <= 0) + 1) / (n + 1)
    above = (np.sum(draws >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def _benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    """BH-adjusted q-values, in the input order.

    Sorted ascending, q_(i) = min over j >= i of p_(j) * m / j, capped at 1.
    The running minimum from the top keeps q monotone in p.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum(1.0, np.minimum.accumulate(scaled[::-1])[::-1])
    q = np.empty(m)
    q[order] = q_sorted
    return q


def _figure(out, rows, order) -> None:
    fig, (left, right) = new_figure(
        width=PAGE_WIDTH,
        height=3.6,
        ncols=2,
        gridspec_kw={"width_ratios": [1, 1.1], "wspace": 0.45},
    )

    # (a) Lead-lag matrix: cell (row, col) = how many ms the row symbol leads the
    # column symbol. Non-significant pairs are grey, not a pale colour.
    index = {s: i for i, s in enumerate(order)}
    n = len(order)
    lead_ms = np.full((n, n), np.nan)
    significant = np.zeros((n, n), dtype=bool)
    for r in rows:
        i, j = index[r["symbol_a"]], index[r["symbol_b"]]
        lead_ms[i, j], lead_ms[j, i] = r["peak_lag_ms"], -r["peak_lag_ms"]
        significant[i, j] = significant[j, i] = r["significant"]
    limit = np.nanmax(np.abs(lead_ms))
    shown = np.where(significant, lead_ms, np.nan)
    left.set_facecolor(NOT_SIGNIFICANT_COLOR)
    image = left.imshow(shown, cmap=LEAD_LAG_CMAP, vmin=-limit, vmax=limit)
    for i in range(n):
        left.add_patch(_diagonal_cell(i))
        for j in range(n):
            if i != j:
                left.text(j, i, f"{lead_ms[i, j]:+.1f}", ha="center", va="center", fontsize=5)
    labels = [s.removesuffix("USDT") for s in order]
    left.set_xticks(range(n), labels, rotation=90)
    left.set_yticks(range(n), labels)
    left.grid(False)
    left.set_xlabel("Follower (trailing symbol)")
    left.set_ylabel("Leader (row symbol)")
    colorbar = fig.colorbar(
        image, ax=left, orientation="horizontal", fraction=0.05, pad=0.22, aspect=30
    )
    colorbar.set_label("Peak lag, row leads column (ms)")
    left.annotate(
        "grey: not significant\n(BH, FDR 5%)",
        xy=(1.0, 1.02),
        xycoords="axes fraction",
        va="bottom",
        ha="right",
        color="0.45",
        fontsize=6,
    )
    label_panel(left, "a")

    # (b) Every pair's peak lag with its bootstrap interval, oriented so the
    # leader is named first and the lag is positive when the asymmetry says so.
    ranked = sorted(rows, key=lambda r: r["q_value"], reverse=True)
    for y, r in enumerate(ranked):
        sign = 1.0 if r["asymmetry"] >= 0 else -1.0
        lo, hi = sorted((sign * r["peak_lag_lo_ms"], sign * r["peak_lag_hi_ms"]))
        color = METHOD_COLORS["hayashi_yoshida"] if r["significant"] else METHOD_COLORS["gridded"]
        right.plot([lo, hi], [y, y], color=color, linewidth=1.0)
        right.plot(sign * r["peak_lag_ms"], y, "o", color=color, markersize=2.5)
    right.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    right.set_yticks(
        range(len(ranked)),
        [_pair_label(r) for r in ranked],
        fontsize=5,
    )
    right.set_xlabel("Peak lag of follower behind leader (ms)")
    right.annotate(
        "dot: pooled estimate\nbar: 95% block-bootstrap\n"
        "interval of the peak lag\ngreen: significant (BH)",
        xy=(0.0, -0.2),
        xycoords="axes fraction",
        va="top",
        color="0.45",
        fontsize=6,
    )
    label_panel(right, "b")

    save_figure(fig, out / "lag_matrix.pdf")


def _diagonal_cell(i: int):
    from matplotlib.patches import Rectangle

    return Rectangle((i - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor="none")


def _pair_label(row: dict) -> str:
    a, b = row["symbol_a"].removesuffix("USDT"), row["symbol_b"].removesuffix("USDT")
    return f"{a} → {b}" if row["asymmetry"] >= 0 else f"{b} → {a}"


if __name__ == "__main__":
    main()
