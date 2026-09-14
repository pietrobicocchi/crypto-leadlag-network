"""The methods this project proposes, and the baselines they are compared to.

This is the canonical implementation. If a notebook or an experiment holds a
second copy of any of this, that copy is a bug.
"""

from __future__ import annotations

import numpy as np


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression by the normal equations.

    Minimises `mean((X w - y)**2) + alpha * ||w||**2`. Returns `w` of shape
    `(n_features,)`.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    n_samples, n_features = X.shape
    gram = X.T @ X + alpha * n_samples * np.eye(n_features)
    return np.linalg.solve(gram, X.T @ y)


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Linear prediction `X @ w`."""
    return X @ w


def fit_constant_baseline(y: np.ndarray) -> float:
    """Baseline: predict the training mean regardless of the input."""
    return float(np.mean(y))


def predict_constant(c: float, X: np.ndarray) -> np.ndarray:
    """Predict `c` for every row of `X`."""
    return np.full(X.shape[0], c)
