from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone

from store_sales.metrics import rmsle

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, type] = {
    "lightgbm": LGBMRegressor,
}

SEED_PARAM_MAP: dict[str, str] = {
    "lightgbm": "random_state",
}


class TimeSeriesModel:
    """Time-series forecasting model with time-based cross-validation."""

    def __init__(
        self,
        model: object | None = None,
    ):
        self.model = model or LGBMRegressor(n_estimators=500, verbose=-1, n_jobs=-1)
        self.fold_models_: list[object] = []
        self.valid_scores_: list[float] = []
        self.overall_val_score_: float | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> TimeSeriesModel:
        self.feature_names_ = list(X_train.columns)
        m = clone(self.model)
        m.fit(X_train, y_train)
        self.fold_models_ = [m]

        if X_val is not None and y_val is not None:
            val_preds = m.predict(X_val)
            val_preds = np.maximum(val_preds, 0)
            score = rmsle(y_val.values, val_preds)
            self.valid_scores_ = [score]
            self.overall_val_score_ = score

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.fold_models_[0].predict(X[self.feature_names_])
        return np.maximum(preds, 0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TimeSeriesModel:
        return joblib.load(path)


def save_submission(
    test_ids: pd.Series,
    predictions: np.ndarray,
    output_path: str = "outputs/submissions/submission.csv",
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"id": test_ids, "sales": predictions})
    submission.to_csv(out, index=False)
    logger.info("  Submission saved → %s (%d rows)", out, len(submission))
