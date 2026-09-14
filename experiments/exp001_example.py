"""exp001_example

Question:
    How does the test error of the ridge estimator vary with the penalty
    alpha, and does the best alpha beat a constant baseline?

Produces:
    outputs/exp001/metrics.csv
    outputs/exp001/alpha_sweep.pdf

This is the template's worked example. Delete it once exp002 exists.
"""

from __future__ import annotations

import csv

import numpy as np

from leadlag.data import make_dataset, train_test_split
from leadlag.methods import (
    fit_constant_baseline,
    fit_ridge,
    predict,
    predict_constant,
)
from leadlag.metrics import rmse
from leadlag.plotting import METHOD_COLORS, METHOD_LABELS, new_figure, save_figure
from leadlag.run import load_config, start_run

EXP_ID = "exp001"


def main() -> None:
    config = load_config(EXP_ID)
    out = start_run(EXP_ID, config)
    rng = np.random.default_rng(config["seed"])

    data = config["data"]
    X, y = make_dataset(data["n_samples"], data["n_features"], data["noise"], rng)
    X_train, y_train, X_test, y_test = train_test_split(X, y, data["train_frac"], rng)

    baseline = rmse(y_test, predict_constant(fit_constant_baseline(y_train), X_test))

    alphas = config["method"]["alphas"]
    errors = [rmse(y_test, predict(fit_ridge(X_train, y_train, a), X_test)) for a in alphas]

    with (out / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "alpha", "test_rmse"])
        for alpha, error in zip(alphas, errors, strict=True):
            writer.writerow(["ours", alpha, f"{error:.6f}"])
        writer.writerow(["baseline", "", f"{baseline:.6f}"])

    fig, ax = new_figure()
    ax.semilogx(
        alphas,
        errors,
        marker="o",
        color=METHOD_COLORS["ours"],
        label=METHOD_LABELS["ours"],
    )
    ax.axhline(
        baseline,
        linestyle="--",
        color=METHOD_COLORS["baseline"],
        label=METHOD_LABELS["baseline"],
    )
    ax.set_xlabel(r"Ridge penalty $\alpha$")
    ax.set_ylabel("Test RMSE")
    ax.legend()
    save_figure(fig, out / "alpha_sweep.pdf")

    best = int(np.argmin(errors))
    print(f"best alpha={alphas[best]:g}  rmse={errors[best]:.4f}  baseline={baseline:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
