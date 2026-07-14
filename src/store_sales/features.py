from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Engineer time-series features for store sales forecasting.

    Creates date-based features, lag features, rolling statistics,
    and encodes categorical variables.
    """

    def __init__(
        self,
        date_col: str = "date",
        store_col: str = "store_nbr",
        family_col: str = "family",
        onpromotion_col: str = "onpromotion",
        date_features: list[str] | None = None,
        drop_cols: list[str] | None = None,
        lag_config: list[list] | None = None,
        rolling_config: list[dict] | None = None,
        ref_date: str | None = None,
        fourier_config: dict[str, list[int]] | None = None,
        holiday_dates: list[str] | None = None,
    ):
        self.date_col = date_col
        self.store_col = store_col
        self.family_col = family_col
        self.onpromotion_col = onpromotion_col
        self.date_features = date_features or []
        self.drop_cols = drop_cols or []
        self.lag_config = lag_config or []
        self.rolling_config = rolling_config or []
        self.ref_date = pd.Timestamp(ref_date) if ref_date else None
        self.fourier_config = fourier_config or {}
        self.holiday_dates = sorted(pd.to_datetime(holiday_dates)) if holiday_dates else None

    def fit(self, X: pd.DataFrame, y=None) -> TimeSeriesFeatureEngineer:
        if self.date_col in X.columns and self.ref_date is None:
            self.ref_date = pd.to_datetime(X[self.date_col]).min()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # --- Date features ---
        if self.date_col in X.columns:
            dt = pd.to_datetime(X[self.date_col])
            ref = self.ref_date if self.ref_date is not None else dt.min()
            feat_map = {
                "year": dt.dt.year,
                "month": dt.dt.month,
                "dayofweek": dt.dt.dayofweek,
                "dayofmonth": dt.dt.day,
                "quarter": dt.dt.quarter,
                "weekofyear": dt.dt.isocalendar().week.astype(int),
                "dayofyear": dt.dt.dayofyear,
                "is_weekend": (dt.dt.dayofweek >= 5).astype(int),
                "time_step": (dt - ref).dt.days,
            }
            for feat in self.date_features:
                if feat in feat_map:
                    X[feat] = feat_map[feat]

        # --- Fourier features ---
        if self.date_col in X.columns and self.fourier_config:
            dt = pd.to_datetime(X[self.date_col])
            dt_vals = {
                "dayofyear": (365.25, dt.dt.dayofyear),
                "dayofweek": (7, dt.dt.dayofweek),
                "month": (12, dt.dt.month),
            }
            for col, harmonics in self.fourier_config.items():
                if col not in dt_vals:
                    continue
                period, vals = dt_vals[col]
                for h in harmonics:
                    angle = 2 * np.pi * h * vals / period
                    X[f"fourier_{col}_sin_{h}"] = np.sin(angle)
                    X[f"fourier_{col}_cos_{h}"] = np.cos(angle)

        # --- Holiday distance features ---
        if (
            self.date_col in X.columns
            and self.holiday_dates is not None
            and len(self.holiday_dates) > 0
        ):
            dt_ns = pd.to_datetime(X[self.date_col]).values.astype("datetime64[ns]")
            hd_ns = np.array(self.holiday_dates, dtype="datetime64[ns]")
            idx = np.searchsorted(hd_ns, dt_ns, side="right")
            # Days since the most recent past holiday
            prev_ix = np.clip(idx - 1, 0, len(hd_ns) - 1)
            prev_holiday = hd_ns[prev_ix]
            X["days_since_holiday"] = ((dt_ns - prev_holiday) / np.timedelta64(1, "D")).astype(int)
            # Days until the next holiday (0 if the date itself is a holiday)
            next_ix = np.clip(idx, 0, len(hd_ns) - 1)
            next_holiday = hd_ns[next_ix]
            X["days_until_holiday"] = ((next_holiday - dt_ns) / np.timedelta64(1, "D")).astype(int)

        # --- Categorical encoding (LightGBM rejects str/object dtype) ---
        for col in [self.store_col, self.family_col]:
            if col in X.columns:
                X[col] = X[col].astype("category")
        for col in X.columns:
            if str(X[col].dtype) in ("object", "str", "string"):
                X[col] = X[col].astype("category")

        # --- onpromotion ---
        if self.onpromotion_col in X.columns:
            X[self.onpromotion_col] = X[self.onpromotion_col].fillna(0).astype(int)

        # --- dcoilwtico (oil price) imputation ---
        if "dcoilwtico" in X.columns:
            X["dcoilwtico"] = X["dcoilwtico"].ffill().bfill()

        # --- Drop unwanted columns ---
        cols_to_drop = [c for c in self.drop_cols if c in X.columns]
        if self.date_col in X.columns:
            cols_to_drop.append(self.date_col)
        X = X.drop(columns=cols_to_drop, errors="ignore")

        return X

    def create_lag_features(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str = "sales",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create lag and rolling features using training data.

        Must be called on the full historical dataset *before* train/val split
        to avoid lookahead leakage.
        """
        n_train = len(train)
        # Preserve caller's train index so y.loc[X.index] stays aligned after
        # we drop first-row-per-group. Work on a positional RangeIndex internally.
        train_index = train.index.to_numpy()
        combined = pd.concat([train, test], axis=0, ignore_index=True)
        combined["_is_train"] = [True] * n_train + [False] * (len(combined) - n_train)
        combined = combined.sort_values([self.store_col, self.family_col, self.date_col])

        group_cols = [self.store_col, self.family_col]

        # --- Lag features ---
        for col, lags in self.lag_config:
            for lag in lags:
                combined[f"{col}_lag_{lag}"] = combined.groupby(group_cols)[col].shift(lag)

        # --- Rolling features ---
        for rc in self.rolling_config:
            col = rc["col"]
            windows = rc.get("windows", [])
            aggs = rc.get("aggs", ["mean"])
            for w in windows:
                for agg in aggs:
                    roll = combined.groupby(group_cols)[col].transform(
                        lambda s, w=w, agg=agg: s.shift(1).rolling(w, min_periods=1).agg(agg)
                    )
                    combined[f"{col}_roll_{w}_{agg}"] = roll

        # Drop train rows with no lag history (first-row-per-group). Test rows
        # are kept even if lags are NaN (target is unknown for the horizon); we
        # forward-fill from the last known value so the model still has features.
        lag_cols = [c for c in combined.columns if "lag_" in c or "roll_" in c]
        train_mask = combined["_is_train"]
        combined = combined[(~train_mask) | (combined[lag_cols].notna().any(axis=1))].copy()
        for col in lag_cols:
            combined[col] = combined.groupby(group_cols)[col].ffill()
            combined[col] = combined[col].fillna(0)

        # Split back — filter by _is_train flag (NOT positional, since dropna
        # removed first-row-per-group train rows, scrambling positional order).
        train_feat = combined[combined["_is_train"]].drop(columns=["_is_train"])
        test_feat = combined[~combined["_is_train"]].drop(columns=["_is_train"])
        # Map positional concat indices back to the caller's train index.
        train_feat.index = train_index[train_feat.index.to_numpy()]
        # Preserve a row-order proxy so callers can rejoin to the original test
        # frame (which is sorted by id). The combined frame was sorted by
        # (store, family, date); we expose that order via the reset RangeIndex.
        test_feat = test_feat.reset_index(drop=True)

        new_cols = [
            c
            for c in combined.columns
            if "lag_" in c
            or "roll_" in c
            or "transactions" in c
            or "is_holiday" in c
            or "dcoilwtico" in c
            or "onpromotion" in c
            or c in self.date_features
        ]
        logger.info("  Lag features added: %d new columns", len(new_cols))
        return train_feat, test_feat


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
    holiday_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: build feature engineer from config and apply."""
    from store_sales.data import apply_preprocessing

    train, _ = apply_preprocessing(train, cfg)
    feat_cfg = cfg["features"]
    engineer = TimeSeriesFeatureEngineer(
        date_col=feat_cfg.get("date_col", "date"),
        store_col=feat_cfg.get("store_col", "store_nbr"),
        family_col=feat_cfg.get("family_col", "family"),
        onpromotion_col=feat_cfg.get("onpromotion_col", "onpromotion"),
        date_features=feat_cfg.get("date_features", []),
        drop_cols=feat_cfg.get("drop_cols", []),
        lag_config=feat_cfg.get("lag_features", []),
        rolling_config=feat_cfg.get("rolling_features", []),
        fourier_config=feat_cfg.get("fourier_features", None),
        holiday_dates=holiday_dates,
    )

    train_feat, test_feat = engineer.create_lag_features(train, test, cfg["competition"]["target"])
    engineer.fit(train_feat)
    train_feat = engineer.transform(train_feat)
    test_feat = engineer.transform(test_feat)

    return train_feat, test_feat
