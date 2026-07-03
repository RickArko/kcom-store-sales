from __future__ import annotations

import numpy as np

from store_sales.metrics import rmsle


def test_rmsle_perfect():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmsle(y_true, y_pred) == 0.0


def test_rmsle_off_by_one():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    score = rmsle(y_true, y_pred)
    assert score > 0


def test_rmsle_negative_clamped():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([-1.0, 2.0, 3.0])
    score = rmsle(y_true, y_pred)
    assert score >= 0


def test_rmsle_zero_actual():
    y_true = np.array([0.0, 2.0, 3.0])
    y_pred = np.array([0.0, 2.0, 3.0])
    assert rmsle(y_true, y_pred) == 0.0
