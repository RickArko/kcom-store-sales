from __future__ import annotations

import logging
import re
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


def _identify_lag_columns(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Find columns matching ``{col}_lag_{N}`` and return (full_name, col, N)."""
    matches: list[tuple[str, str, int]] = []
    for col in df.columns:
        m = re.match(r"(.+)_lag_(\d+)$", col)
        if m:
            matches.append((col, m.group(1), int(m.group(2))))
    return matches


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

    def recursive_validate(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        group_cols: list[str] | None = None,
        date_col: str = "date",
        store_col: str = "store_nbr",
        family_col: str = "family",
    ) -> tuple[float, np.ndarray]:
        """Walk-forward recursive validation.

        Predicts day by day, updating lag features with predicted values
        to mirror test-time conditions.  Only direct lag columns (e.g.
        ``sales_lag_1``) are updated — rolling statistics are left as-is.

        Returns (rmsle, predictions_array).
        """
        gc = group_cols or [store_col, family_col]

        X = X_val.copy()
        X[date_col] = pd.to_datetime(X[date_col])
        X = X.sort_values(gc + [date_col]).reset_index(drop=True)

        lag_cols = _identify_lag_columns(X)
        n_total = len(X)
        preds = np.full(n_total, np.nan)

        unique_dates = sorted(X[date_col].unique())

        for day_idx, current_date in enumerate(unique_dates):
            mask = X[date_col] == current_date
            idx = mask.values

            if idx.sum() == 0:
                continue

            X_day = X.loc[idx]

            day_preds = self.fold_models_[0].predict(X_day[self.feature_names_])
            day_preds = np.maximum(day_preds, 0)
            preds[idx] = day_preds

            # Update lag columns for future days in the same groups
            if day_idx < len(unique_dates) - 1 and lag_cols:
                X_day_aug = X_day.copy()
                X_day_aug["_predicted_sales_"] = day_preds
                X_day_aug.set_index(gc, inplace=False)

                X_future = X.loc[~idx].copy()
                X_future.set_index(gc, inplace=False)

                for (grp_val,), grp_future in X_future.groupby(gc):
                    grp_day = X_day_aug[X_day_aug[store_col] == grp_val[0]]
                    if family_col and family_col != store_col:
                        grp_day = grp_day[grp_day[family_col] == grp_val[1]]
                    if grp_day.empty:
                        continue
                    pred_val = grp_day["_predicted_sales_"].iloc[0]

                    for full_name, _, lag_n in lag_cols:
                        target_date = current_date + pd.Timedelta(days=lag_n)
                        match_idx = (X[store_col] == grp_val[0]) & (X[date_col] == target_date)
                        if family_col and family_col != store_col:
                            match_idx = match_idx & (X[family_col] == grp_val[1])
                        X.loc[match_idx, full_name] = pred_val

        preds = np.maximum(preds, 0)
        score = rmsle(y_val.values, preds)
        return score, preds

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
