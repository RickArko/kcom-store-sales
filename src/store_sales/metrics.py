"""Competition metrics — thin wrappers over ``kaggle_ml.evaluation``."""

from __future__ import annotations

import numpy as np
from kaggle_ml.evaluation import rmsle as _rmsle


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error.

    Formula: sqrt(mean((log(1+p) - log(1+y))^2))
    Both arrays must be non-negative. Predictions are clipped to >= 0.
    """
    return _rmsle(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64))
