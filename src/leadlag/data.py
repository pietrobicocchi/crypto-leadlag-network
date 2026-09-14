"""Data: generation, loading, preprocessing, splitting.

No model or evaluation logic belongs here. Replace `make_dataset` with real
data loading; keep `train_test_split` deterministic given its `rng`.
"""

from __future__ import annotations

import numpy as np


def make_dataset(
    n_samples: int,
    n_features: int,
    noise: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic regression data with a sparse true coefficient vector.

    Returns `X` of shape `(n_samples, n_features)` and `y` of shape `(n_samples,)`.
    """
    X = rng.standard_normal((n_samples, n_features))
    true_w = np.zeros(n_features)
    true_w[: max(1, n_features // 4)] = 1.0
    y = X @ true_w + noise * rng.standard_normal(n_samples)
    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split rows into train and test. Deterministic given `rng`.

    Returns `(X_train, y_train, X_test, y_test)`.
    """
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows, y has {len(y)}")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must lie in (0, 1), got {train_frac}")

    perm = rng.permutation(len(X))
    cut = int(round(train_frac * len(X)))
    if cut == 0 or cut == len(X):
        raise ValueError(f"train_frac={train_frac} leaves one side of the split empty")

    train, test = perm[:cut], perm[cut:]
    return X[train], y[train], X[test], y[test]
