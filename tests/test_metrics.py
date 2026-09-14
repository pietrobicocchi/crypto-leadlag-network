"""Tests for scientifically dangerous code.

Risk-weighted, not coverage-weighted: these check the properties whose silent
failure would invalidate a conclusion.
"""

from __future__ import annotations

import numpy as np
import pytest

from leadlag.data import train_test_split
from leadlag.metrics import r2, rmse


def test_rmse_is_zero_for_exact_predictions():
    y = np.array([1.0, -2.0, 3.5])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_is_invariant_under_permutation():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.0, 3.2, 4.4])
    order = np.array([2, 0, 3, 1])
    assert rmse(y_true, y_pred) == pytest.approx(rmse(y_true[order], y_pred[order]))


def test_rmse_matches_closed_form():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))


def test_r2_of_the_mean_predictor_is_zero():
    y = np.array([1.0, 2.0, 6.0, 9.0])
    assert r2(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        rmse(np.zeros(3), np.zeros(4))


def test_split_is_a_partition_and_deterministic():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)

    a = train_test_split(X, y, 0.7, np.random.default_rng(0))
    b = train_test_split(X, y, 0.7, np.random.default_rng(0))
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)

    X_train, y_train, X_test, y_test = a
    assert len(X_train) == 7 and len(X_test) == 3
    assert sorted(np.concatenate([y_train, y_test])) == list(y)


def test_split_rejects_degenerate_fractions():
    X, y = np.zeros((4, 2)), np.zeros(4)
    with pytest.raises(ValueError, match="train_frac"):
        train_test_split(X, y, 1.0, np.random.default_rng(0))
