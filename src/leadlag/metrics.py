"""Evaluation metrics.

A silent error here invalidates every conclusion in the paper, so this module
is the first place to add a test.
"""

from __future__ import annotations

import numpy as np


def _as_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("cannot score an empty prediction")
    return y_true, y_pred


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true, y_pred = _as_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination, relative to the mean of `y_true`.

    Zero for a predictor that always returns the mean; one for a perfect fit.
    """
    y_true, y_pred = _as_pair(y_true, y_pred)
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    if total == 0.0:
        raise ValueError("y_true is constant; R^2 is undefined")
    return float(1.0 - residual / total)
