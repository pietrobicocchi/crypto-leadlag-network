"""Does Hayashi-Yoshida recover a known lag where the gridded baseline invents one?

Gate 1 of the project. Both estimators see identical data with a lag we chose
rather than discovered, so for once there is a right answer to compare against.

The discriminating case is a true lag of exactly zero under a trade-rate
imbalance. Asset A trades many times more often than B, and nothing else
differs. A method that reports lead-lag structure there is reporting an
artefact of its own clock.

    make exp EXP=exp001
"""

from __future__ import annotations

import csv

import numpy as np

from leadlag.estimators import (
    gridded_correlation,
    hayashi_yoshida_correlation,
    lead_lag_ratio,
    peak_lag,
)
from leadlag.plotting import (
    METHOD_COLORS,
    METHOD_LABELS,
    PAGE_WIDTH,
    label_panel,
    new_figure,
    save_figure,
)
from leadlag.run import load_config, start_run
from leadlag.synthetic import delayed_pair

NS_PER_MS = 1_000_000


def main() -> None:
    config = load_config("exp001")
    out = start_run("exp001", config)

    data, estimator = config["data"], config["estimator"]
    bucket_ns = int(estimator["bucket_ns"])
    half_width = int(estimator["lag_half_width"])
    lag_grid_ns = np.arange(-half_width, half_width + 1, dtype=np.int64) * bucket_ns

    # Independent streams from one recorded seed. SeedSequence.spawn guarantees
    # the children are statistically independent; seed+1, seed+2 does not.
    sweep = config["sweep"]
    seeds = np.random.SeedSequence(config["seed"]).spawn(int(sweep["n_seeds"]))

    illustration = config["illustration"]
    curves = _curves(
        data,
        bucket_ns,
        lag_grid_ns,
        true_lag_ns=int(illustration["true_lag_ns"]),
        rate_ratio=float(illustration["rate_ratio"]),
        seed=seeds[0],
    )

    rows = []
    for ratio in sweep["rate_ratios"]:
        for index, seed in enumerate(seeds):
            measured = _curves(data, bucket_ns, lag_grid_ns, 0, float(ratio), seed)
            for method, correlation in measured.items():
                rows.append(
                    {
                        "rate_ratio": ratio,
                        "seed_index": index,
                        "method": method,
                        "lead_lag_ratio": lead_lag_ratio(lag_grid_ns, correlation),
                        "peak_lag_ns": peak_lag(lag_grid_ns, correlation),
                        "peak_correlation": float(correlation.max()),
                    }
                )

    with (out / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    _figure(out, lag_grid_ns, curves, rows, sweep["rate_ratios"])

    for method in METHOD_LABELS:
        at_ten = [
            r["lead_lag_ratio"] for r in rows if r["method"] == method and r["rate_ratio"] == 10
        ]
        label = METHOD_LABELS[method]
        print(f"{label:>18}: lead-lag ratio at 10:1, true lag 0 = {np.median(at_ten):8.2f}")
    print(f"wrote {out}")


def _curves(
    data: dict,
    bucket_ns: int,
    lag_grid_ns: np.ndarray,
    true_lag_ns: int,
    rate_ratio: float,
    seed: np.random.SeedSequence,
) -> dict[str, np.ndarray]:
    """Both estimators' correlation curves over one synthetic pair."""
    a, b = delayed_pair(
        duration_s=float(data["duration_s"]),
        rate_a_hz=float(data["rate_a_hz"]),
        rate_b_hz=float(data["rate_a_hz"]) / rate_ratio,
        lag_ns=true_lag_ns,
        volatility=float(data["volatility"]),
        noise=float(data["noise"]),
        initial_price=float(data["initial_price"]),
        rng=np.random.default_rng(seed),
    )
    return {
        "gridded": gridded_correlation(a, b, bucket_ns=bucket_ns, lag_grid_ns=lag_grid_ns),
        "hayashi_yoshida": hayashi_yoshida_correlation(a, b, lag_grid_ns=lag_grid_ns),
    }


def _figure(out, lag_grid_ns, curves, rows, rate_ratios) -> None:
    fig, (left, right) = new_figure(width=PAGE_WIDTH, height=2.7, ncols=2)

    lags_ms = lag_grid_ns / NS_PER_MS
    left.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
    for method, correlation in curves.items():
        left.plot(lags_ms, correlation, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    left.set_xlabel("Candidate lag (ms)")
    left.set_ylabel("Correlation")
    left.legend(loc="upper left")
    left.annotate(
        "true lag = 0",
        xy=(0.0, left.get_ylim()[0]),
        xytext=(14, 6),
        textcoords="offset points",
        color="0.45",
        fontsize=6,
    )
    label_panel(left, "a")

    right.axhline(1.0, color="0.75", linewidth=0.8, zorder=0)
    for method in METHOD_LABELS:
        medians, lows, highs = [], [], []
        for ratio in rate_ratios:
            values = [
                r["lead_lag_ratio"]
                for r in rows
                if r["method"] == method and r["rate_ratio"] == ratio
            ]
            medians.append(np.median(values))
            lows.append(min(values))
            highs.append(max(values))
        right.fill_between(
            rate_ratios, lows, highs, color=METHOD_COLORS[method], alpha=0.18, linewidth=0
        )
        right.plot(rate_ratios, medians, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xticks(list(rate_ratios))
    right.set_xticklabels([f"{r:g}" for r in rate_ratios])
    right.minorticks_off()
    right.set_xlabel("Trade-rate ratio, A:B")
    right.set_ylabel("Lead–lag ratio")
    right.annotate(
        "line: median\nband: min-max\nover 10 seeds",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        color="0.45",
        fontsize=6,
    )
    label_panel(right, "b")

    save_figure(fig, out / "synthetic_recovery.pdf")


if __name__ == "__main__":
    main()
