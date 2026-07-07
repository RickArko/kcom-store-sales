"""End-to-end training pipeline for Store Sales Time Series Forecasting.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --config config/config.yaml --run-name expr-001
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from lightgbm import LGBMRegressor

from store_sales.data import load_config, load_data, merge_tables, timeseries_split
from store_sales.features import TimeSeriesFeatureEngineer
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
    return parser.parse_args()


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

    # -- 2. Feature engineering (lag features need full history) -------
    logger.info("[2/5] Engineering features (lags + date features) ...")
    t0 = time.time()
    target_col = cfg["competition"]["target"]
    y_train_full = train[target_col].copy()
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
    )
    # Lag features need full sorted history — build BEFORE split.
    # `date` is kept here so the split can use it; transform drops it after.
    X_train_lag, X_test_feat = engineer.create_lag_features(train, test, target_col)
    # Stash test ids (sorted by store/family/date) before transform drops them.
    test_ids = X_test_feat["id"].reset_index(drop=True) if "id" in X_test_feat.columns else None

    logger.info(
        "  X_train: %s  X_test: %s  (%.1fs)",
        X_train_lag.shape,
        X_test_feat.shape,
        time.time() - t0,
    )

    # -- 3. Time-series split (last N days for validation) -------------
    logger.info("[3/5] Time-series split ...")
    ts_cfg = cfg.get("timeseries", {})
    val_period = ts_cfg.get("test_period_days", 16)
    X_train_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
    y_train = y_train_full.loc[X_train_raw.index]
    y_val = y_train_full.loc[X_val_raw.index]
    # Now drop date / encode categoricals / add date features.
    X_train = engineer.transform(X_train_raw)
    X_val = engineer.transform(X_val_raw)
    X_test_feat = engineer.transform(X_test_feat)
    logger.info("  Train: %s  Val: %s", X_train.shape, X_val.shape)

    # -- 4. Build & train model ----------------------------------------
    logger.info("[4/5] Training model ...")
    model_cfg = cfg["model"].copy()
    model = _build_model(model_cfg)

    with track_experiment(cfg, run_name=args.run_name) as run:
        ts_model = TimeSeriesModel(model)
        ts_model.fit(X_train, y_train, X_val, y_val)

        run.log_metrics(
            {
                "val_rmsle": round(ts_model.overall_val_score_, 6),
                "n_features": X_train.shape[1],
            }
        )
        run.log_params(
            {
                "model_type": cfg["model"]["type"],
                "n_estimators": cfg["model"]["n_estimators"],
                "val_period_days": val_period,
                "run_scope": cfg.get("run_scope", "full"),
            }
        )

        model_path = run.models_dir / "model.joblib"
        ts_model.save(model_path)
        logger.info("  Model saved to %s", model_path)

        # -- 5. Generate submission ------------------------------------
        logger.info("[5/5] Generating predictions ...")
        test_preds = ts_model.predict(X_test_feat)
        if test_ids is not None and len(test_ids) == len(test_preds):
            ids = test_ids
        else:
            # Fallback: original test order (sorted by id)
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
