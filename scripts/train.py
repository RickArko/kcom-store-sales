"""End-to-end training pipeline for Store Sales Time Series Forecasting.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config config/config.yaml --run-name expr-001
    uv run python scripts/train.py --cv  # multi-window cross-validation
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from store_sales.data import (
    load_config,
    load_data,
    merge_tables,
    timeseries_split,
    walk_forward_split,
)
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.metrics import rmsle
from store_sales.models import TimeSeriesModel, save_submission
from store_sales.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def _build_model(model_cfg: dict) -> LGBMRegressor:
    model_cfg.pop("type", "lightgbm")
    model_cfg.setdefault("random_state", 42)
    model_cfg.setdefault("verbose", -1)
    model_cfg.setdefault("n_jobs", -1)
    return LGBMRegressor(**model_cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train store sales forecasting model.")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to config YAML"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Human-readable experiment name")
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run multi-window cross-validation instead of single split",
    )
    return parser.parse_args()


def _build_engineer(
    feat_cfg: dict, holiday_dates: list[str] | None = None
) -> TimeSeriesFeatureEngineer:
    return TimeSeriesFeatureEngineer(
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


def _run_cv(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    feat_cfg: dict,
    model_cfg: dict,
    ts_cfg: dict,
    holiday_dates: list[str] | None,
    run,
) -> list[dict]:
    """Run multi-window cross-validation with recursive evaluation.

    Returns list of result dicts, one per window.
    """
    engineer = _build_engineer(feat_cfg, holiday_dates=holiday_dates)
    X_train_lag, X_test_feat = engineer.create_lag_features(train, test, target_col)
    X_test_feat = engineer.transform(X_test_feat)

    cv_windows = ts_cfg.get("cv_windows", [30, 60, 90])
    splits = walk_forward_split(X_train_lag, windows=cv_windows)

    results = []
    all_recursive_scores = []

    for split_idx, (X_tr_raw, X_val_raw, label) in enumerate(splits):
        y_tr = train.loc[X_tr_raw.index, target_col]
        y_val = train.loc[X_val_raw.index, target_col]

        X_tr = engineer.transform(X_tr_raw)
        X_val = engineer.transform(X_val_raw)

        model = _build_model(model_cfg)
        ts_model = TimeSeriesModel(model)
        ts_model.fit(X_tr, y_tr)

        # One-shot (standard) evaluation
        val_preds = ts_model.predict(X_val)
        val_preds = np.maximum(val_preds, 0)
        one_shot_score = rmsle(y_val.values, val_preds)

        # Walk-forward (recursive) evaluation
        try:
            recursive_score, _ = ts_model.recursive_validate(X_val, y_val)
        except Exception:
            logger.warning("  Recursive validation failed for %s, skipping", label)
            recursive_score = None

        all_recursive_scores.append(recursive_score)

        logger.info(
            "  CV %s — one-shot: %.6f | recursive: %s",
            label,
            one_shot_score,
            f"{recursive_score:.6f}" if recursive_score is not None else "N/A",
        )

        results.append(
            {
                "window": label,
                "one_shot_rmsle": round(one_shot_score, 6),
                "recursive_rmsle": round(recursive_score, 6)
                if recursive_score is not None
                else None,
                "n_train": len(X_tr),
                "n_val": len(X_val),
            }
        )

    # Log summary
    one_shot_scores = [r["one_shot_rmsle"] for r in results]
    recursive_scores = [r["recursive_rmsle"] for r in results if r["recursive_rmsle"] is not None]

    logger.info("=" * 60)
    logger.info(
        "CV Summary (one-shot RMSLE): mean=%.6f  std=%.6f  min=%.6f  max=%.6f",
        np.mean(one_shot_scores),
        np.std(one_shot_scores),
        min(one_shot_scores),
        max(one_shot_scores),
    )
    if recursive_scores:
        logger.info(
            "CV Summary (recursive RMSLE): mean=%.6f  std=%.6f  min=%.6f  max=%.6f",
            np.mean(recursive_scores),
            np.std(recursive_scores),
            min(recursive_scores),
            max(recursive_scores),
        )
    logger.info("=" * 60)

    run.log_metrics(
        {
            "cv_one_shot_mean": round(float(np.mean(one_shot_scores)), 6),
            "cv_one_shot_std": round(float(np.std(one_shot_scores)), 6),
            "cv_recursive_mean": round(float(np.mean(recursive_scores)), 6)
            if recursive_scores
            else 0,
            "cv_recursive_std": round(float(np.std(recursive_scores)), 6)
            if recursive_scores
            else 0,
            "n_features": (
                X_train_lag.shape[1] - 1 if "date" in X_train_lag.columns else X_train_lag.shape[1]
            ),
        }
    )
    run.log_params(
        {
            "cv_windows": str(cv_windows),
            "model_type": model_cfg.get("type", "lightgbm"),
        }
    )

    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Store Sales — Time Series Forecasting")
    logger.info("=" * 60)

    # -- 1. Load data --------------------------------------------------
    logger.info("[1/5] Loading data ...")
    t0 = time.time()
    tables = load_data(cfg["paths"]["data"])
    train, test = merge_tables(tables)
    logger.info("  Loaded in %.1fs", time.time() - t0)

    # Extract holiday dates for feature engineering
    holiday_dates: list[str] | None = None
    holidays_df = tables.get("holidays")
    if holidays_df is not None and not holidays_df.empty:
        holiday_dates = holidays_df["date"].dropna().unique().tolist()
        logger.info("  Extracted %d holiday dates for features", len(holiday_dates))

    target_col = cfg["competition"]["target"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"].copy()
    ts_cfg = cfg.get("timeseries", {})

    # -- 2. Feature engineering ----------------------------------------
    logger.info("[2/5] Engineering features (lags + date features) ...")
    t0_feat = time.time()
    engineer = _build_engineer(feat_cfg, holiday_dates=holiday_dates)
    y_train_full = train[target_col].copy()
    X_train_lag, X_test_feat = engineer.create_lag_features(train, test, target_col)
    test_ids = X_test_feat["id"].reset_index(drop=True) if "id" in X_test_feat.columns else None
    logger.info(
        "  X_train: %s  X_test: %s  (%.1fs)",
        X_train_lag.shape,
        X_test_feat.shape,
        time.time() - t0_feat,
    )

    with track_experiment(cfg, run_name=args.run_name) as run:
        # -- 3. CV or single-split ------------------------------------
        if args.cv:
            logger.info("[3/5] Running multi-window CV ...")
            _run_cv(train, test, target_col, feat_cfg, model_cfg, ts_cfg, holiday_dates, run)

            # Train final model on full data for submission
            logger.info("  Training final model on full data ...")
            val_period = ts_cfg.get("test_period_days", 16)
            X_tr_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
            y_tr = y_train_full.loc[X_tr_raw.index]
            y_val = y_train_full.loc[X_val_raw.index]
            X_tr = engineer.transform(X_tr_raw)
            X_val = engineer.transform(X_val_raw)
            X_test = engineer.transform(X_test_feat)

            model = _build_model(model_cfg)
            ts_model = TimeSeriesModel(model)
            ts_model.fit(X_tr, y_tr, X_val, y_val)

            run.log_metrics(
                {
                    "val_rmsle": round(ts_model.overall_val_score_, 6),
                    "n_features": X_tr.shape[1],
                }
            )
            run.log_params(
                {
                    "model_type": model_cfg.get("type", "lightgbm"),
                    "val_period_days": val_period,
                    "run_scope": cfg.get("run_scope", "full"),
                }
            )

        else:
            # Standard single-split training (original behavior)
            logger.info("[3/5] Time-series split ...")
            val_period = ts_cfg.get("test_period_days", 16)
            X_tr_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
            y_tr = y_train_full.loc[X_tr_raw.index]
            y_val = y_train_full.loc[X_val_raw.index]
            X_tr = engineer.transform(X_tr_raw)
            X_val = engineer.transform(X_val_raw)
            X_test = engineer.transform(X_test_feat)
            logger.info("  Train: %s  Val: %s", X_tr.shape, X_val.shape)

            # -- 4. Build & train model --------------------------------
            logger.info("[4/5] Training model ...")
            model = _build_model(model_cfg)
            ts_model = TimeSeriesModel(model)
            ts_model.fit(X_tr, y_tr, X_val, y_val)

            run.log_metrics(
                {
                    "val_rmsle": round(ts_model.overall_val_score_, 6),
                    "n_features": X_tr.shape[1],
                }
            )
            run.log_params(
                {
                    "model_type": model_cfg.get("type", "lightgbm"),
                    "n_estimators": model_cfg.get("n_estimators", 500),
                    "val_period_days": val_period,
                    "run_scope": cfg.get("run_scope", "full"),
                }
            )

        # -- 5. Save model & submission --------------------------------
        model_path = run.models_dir / "model.joblib"
        ts_model.save(model_path)
        logger.info("  Model saved to %s", model_path)

        logger.info("[5/5] Generating predictions ...")
        test_preds = ts_model.predict(X_test)
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

        logger.info("  Val RMSLE: %.6f", ts_model.overall_val_score_)

    elapsed = time.time() - t0
    logger.info("Done in %.1fs", elapsed)


if __name__ == "__main__":
    main()
