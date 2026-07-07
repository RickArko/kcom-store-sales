"""Linear regression baseline for Store Sales Time Series Forecasting.

Inspired by the Kaggle course "Linear Regression with Time Series"
(https://www.kaggle.com/code/ryanholbrook/exercise-linear-regression-with-time-series).

Uses time-step trend, seasonal dummies, lag features, and rolling
statistics in a single global Ridge regression across all store-family
series.

Usage:
    uv run python scripts/train_linear.py --config config/linear.yaml --run-name linear
    uv run python scripts/train_linear.py --config config/linear-smoke.yaml --run-name linear-smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, TweedieRegressor

from store_sales.data import load_config, load_data, merge_tables, timeseries_split
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.metrics import rmsle
from store_sales.models import TimeSeriesModel, save_submission
from store_sales.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

_CAT_COLS = ["store_nbr", "family"]


def _build_model(model_cfg: dict) -> Ridge | TweedieRegressor:
    linear_type = model_cfg.get("linear_type", "ridge")
    alpha = model_cfg.get("alpha", 1.0)
    fit_intercept = model_cfg.get("fit_intercept", True)
    if linear_type == "tweedie":
        power = model_cfg.get("power", 1.5)
        return TweedieRegressor(
            power=power, alpha=alpha, fit_intercept=fit_intercept, random_state=42
        )
    return Ridge(alpha=alpha, fit_intercept=fit_intercept, random_state=42)


def _onehot_encode(
    df: pd.DataFrame, cols: list[str], ref_cats: dict[str, list] | None = None
) -> pd.DataFrame:
    """One-hot encode categorical columns, optionally restricting to known categories."""
    for col in cols:
        if col not in df.columns:
            continue
        if ref_cats and col in ref_cats:
            df[col] = pd.Categorical(df[col], categories=ref_cats[col])
        else:
            df[col] = df[col].astype("category")
    return pd.get_dummies(
        df, columns=[c for c in cols if c in df.columns], drop_first=True, dtype=int
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear regression baseline.")
    parser.add_argument(
        "--config", type=str, default="config/linear.yaml", help="Path to config YAML"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Human-readable experiment name")
    return parser.parse_args()


def _apply_subsample(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = cfg.get("subsample", {})
    if not sub:
        return train, test

    max_stores = sub.get("max_stores")
    if max_stores is not None:
        keep_stores = sorted(train["store_nbr"].unique())[:max_stores]
        train = train[train["store_nbr"].isin(keep_stores)].copy()
        test = test[test["store_nbr"].isin(keep_stores)].copy()
        logger.info("  Subsample: %d stores -> %d rows", max_stores, len(train))

    start = sub.get("min_train_date")
    if start:
        train = train[train["date"] >= start].copy()
        logger.info("  Subsample: min_date %s -> %d rows", start, len(train))

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Store Sales — Linear Regression Baseline (Ridge)")
    logger.info("=" * 60)

    # -- 1. Load data --------------------------------------------------
    logger.info("[1/6] Loading data ...")
    t0 = time.time()
    tables = load_data(cfg["paths"]["data"])
    train, test = merge_tables(tables)
    train, test = _apply_subsample(train, test, cfg)
    logger.info("  Loaded in %.1fs", time.time() - t0)

    # -- 2. Feature engineering (lags need full history) ---------------
    logger.info("[2/6] Engineering features (lags + date features) ...")
    t0 = time.time()
    model_cfg = cfg["model"]
    target_col = cfg["competition"]["target"]
    target_transform = model_cfg.get("target_transform", "raw")
    y_train_full_raw = train[target_col].copy()
    if target_transform == "log1p":
        train["log_sales"] = np.log1p(train[target_col])
        test["log_sales"] = 0
        y_train_full = train["log_sales"].copy()
    else:
        y_train_full = y_train_full_raw.copy()
    ref_date = train["date"].min()
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
        ref_date=ref_date,
    )
    X_train_lag, X_test_feat = engineer.create_lag_features(train, test, target_col)
    test_ids = X_test_feat["id"].reset_index(drop=True) if "id" in X_test_feat.columns else None
    # Pre-fit to set ref_date for consistent time_step across splits
    engineer.fit(X_train_lag)
    logger.info(
        "  X_train: %s  X_test: %s  (%.1fs)",
        X_train_lag.shape,
        X_test_feat.shape,
        time.time() - t0,
    )

    # -- 3. Time-series split ------------------------------------------
    logger.info("[3/6] Time-series split ...")
    ts_cfg = cfg.get("timeseries", {})
    val_period = ts_cfg.get("test_period_days", 16)
    X_train_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
    y_train = y_train_full.loc[X_train_raw.index]
    y_val_raw = y_train_full_raw.loc[X_val_raw.index]
    X_train = engineer.transform(X_train_raw)
    X_val = engineer.transform(X_val_raw)
    X_test_feat = engineer.transform(X_test_feat)

    # -- 4. One-hot encode categoricals for linear regression ----------
    logger.info("[4/6] One-hot encoding categoricals ...")
    t0_oh = time.time()
    cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    known_cats: dict[str, list] = {}
    for col in cat_cols:
        known_cats[col] = sorted(X_train[col].cat.categories.tolist())
    X_train = _onehot_encode(X_train, cat_cols, known_cats)
    X_val = _onehot_encode(X_val, cat_cols, known_cats)
    X_test_feat = _onehot_encode(X_test_feat, cat_cols, known_cats)
    # Align columns across train/val/test (test may have different categories)
    train_cols = list(X_train.columns)
    train_cols_set = set(train_cols)
    for df_name in ("val", "test"):
        df = X_val if df_name == "val" else X_test_feat
        for c in train_cols_set - set(df.columns):
            df[c] = 0
        extra = set(df.columns) - train_cols_set
        if extra:
            logger.info("  Dropping %d cols from %s not in train", len(extra), df_name)
            df.drop(columns=list(extra), inplace=True)
        if df_name == "val":
            X_val = df[train_cols]
        else:
            X_test_feat = df[train_cols]
    # Final safety: fill any remaining NaN (future transactions, etc.)
    for df_name, df in [("train", X_train), ("val", X_val), ("test", X_test_feat)]:
        n_nan = df.isna().sum().sum()
        if n_nan:
            logger.info("  Filling %d NaN in %s", n_nan, df_name)
            df.fillna(0, inplace=True)
    logger.info("  Done (%.1fs). Train: %s", time.time() - t0_oh, X_train.shape)

    # -- 5. Build & train model ----------------------------------------
    linear_label = model_cfg.get("linear_type", "ridge")
    logger.info("[5/6] Training %s regression ...", linear_label.title())
    model = _build_model(model_cfg)

    with track_experiment(cfg, run_name=args.run_name) as run:
        ts_model = TimeSeriesModel(model)
        ts_model.fit(X_train, y_train)

        # Manual val evaluation (handles target_transform)
        val_preds_log = ts_model.fold_models_[0].predict(X_val[ts_model.feature_names_])
        if target_transform == "log1p":
            val_preds_raw = np.expm1(val_preds_log)
        else:
            val_preds_raw = val_preds_log
        val_preds_clamped = np.maximum(val_preds_raw, 0)
        val_score = rmsle(y_val_raw.values, val_preds_clamped)
        ts_model.overall_val_score_ = val_score
        ts_model.valid_scores_ = [val_score]

        run.log_metrics(
            {
                "val_rmsle": round(val_score, 6),
                "n_features": X_train.shape[1],
            }
        )
        run.log_params(
            {
                "model_type": cfg["model"]["type"],
                "linear_type": linear_label,
                "target_transform": target_transform,
                "alpha": model_cfg.get("alpha"),
                "fit_intercept": model_cfg.get("fit_intercept"),
                "val_period_days": val_period,
            }
        )

        model_path = run.models_dir / "model.joblib"
        ts_model.save(model_path)
        logger.info("  Model saved to %s", model_path)

        # -- 6. Generate submission ------------------------------------
        logger.info("[6/6] Generating predictions ...")
        test_preds_log = ts_model.predict(X_test_feat)
        if target_transform == "log1p":
            test_preds = np.expm1(test_preds_log)
        else:
            test_preds = test_preds_log
        test_preds = np.maximum(test_preds, 0)
        if test_ids is not None and len(test_ids) == len(test_preds):
            ids = test_ids
        else:
            ids = test.sort_values("id")["id"].reset_index(drop=True)
        save_submission(ids, test_preds, str(run.submission_path))
        save_submission(
            ids,
            test_preds,
            str(Path(cfg["paths"]["submissions"]) / "submission.csv"),
        )

        logger.info("  Val RMSLE: %.6f", val_score)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
