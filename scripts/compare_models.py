"""Compare all linear model variants + existing LightGBM/Nixtla runs.

Re-trains every linear model from scratch on the same data split
and computes: train_time, n_features, val_rmsle, val_mae, val_wmape,
val_bias_pct, val_rmse.

Usage:
    uv run python scripts/compare_models.py
    uv run python scripts/compare_models.py --output results.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, TweedieRegressor
from sklearn.preprocessing import StandardScaler

from store_sales.data import load_config, load_data, merge_tables, timeseries_split
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.metrics import rmsle
from store_sales.models import TimeSeriesModel

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


EXPERIMENTS: list[tuple[str, str]] = [
    ("Ridge (raw)", "config/linear.yaml"),
    ("Log-Ridge", "config/linear-log.yaml"),
    ("Log-Ridge+Fourier", "config/linear-log-fourier.yaml"),
]


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


def _align_ohe(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_cols = list(X_train.columns)
    train_cols_set = set(train_cols)
    for c in train_cols_set - set(X_val.columns):
        X_val[c] = 0
    extra = set(X_val.columns) - train_cols_set
    if extra:
        X_val = X_val.drop(columns=list(extra))
    X_val = X_val[train_cols]
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    return X_train, X_val


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    denom = max(np.sum(y_true), 1e-10)
    return {
        "val_rmsle": rmsle(y_true, y_pred),
        "val_mae": float(np.mean(np.abs(y_true - y_pred))),
        "val_wmape": float(np.sum(np.abs(y_true - y_pred)) / denom),
        "val_bias_pct": float(np.sum(y_pred - y_true) / denom * 100),
        "val_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def _scan_existing_runs(runs_dir: str = "outputs/runs") -> list[dict]:
    rows: list[dict] = []
    base = Path(runs_dir)
    if not base.exists():
        return rows
    for run_dir in sorted(base.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        params = data.get("params", {})
        model_type = params.get("model_type", "")
        if model_type not in ("lightgbm", "nixtla"):
            continue
        meta = data.get("metrics", {})
        label_stub = run_dir.name.split("_", 1)[-1]
        base_label = "LightGBM" if model_type == "lightgbm" else "Nixtla"
        rows.append(
            {
                "label": f"{base_label} ({label_stub})",
                "train_time_s": data.get("elapsed_seconds", 0),
                "n_features": meta.get("n_features"),
                "val_rmsle": meta.get("val_rmsle"),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare linear model variants.")
    parser.add_argument("--output", type=str, default=None, help="Save comparison CSV to path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading data ...", flush=True)
    tables = load_data()
    train, test = merge_tables(tables)
    y_train_full_raw = train["sales"].copy()

    results: list[dict] = []

    for label, config_path in EXPERIMENTS:
        print(f"\n── {label} ──", flush=True)
        cfg = load_config(config_path)
        feat_cfg = cfg["features"]
        model_cfg = cfg["model"]
        target_transform = model_cfg.get("target_transform", "raw")

        t_feat = time.time()

        # -- Prepare log_sales column if needed -------------------------
        if target_transform == "log1p":
            train_work = train.copy()
            train_work["log_sales"] = np.log1p(train_work["sales"])
            test_work = test.copy()
            test_work["log_sales"] = 0
            y_for_model = train_work["log_sales"].copy()
        else:
            train_work = train
            test_work = test
            y_for_model = y_train_full_raw.copy()

        # -- Feature engineering -----------------------------------------
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
            ref_date=train_work["date"].min(),
        )
        X_train_lag, X_test_feat = engineer.create_lag_features(
            train_work, test_work, cfg["competition"]["target"]
        )
        engineer.fit(X_train_lag)

        # -- Time-series split -------------------------------------------
        val_period = cfg.get("timeseries", {}).get("test_period_days", 16)
        X_train_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
        y_val_raw = y_train_full_raw.loc[X_val_raw.index]
        y_train = y_for_model.loc[X_train_raw.index]

        X_train = engineer.transform(X_train_raw)
        X_val = engineer.transform(X_val_raw)

        # -- One-hot encode ----------------------------------------------
        cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
        known_cats: dict[str, list] = {}
        for col in cat_cols:
            known_cats[col] = sorted(X_train[col].cat.categories.tolist())
        X_train = _onehot_encode(X_train, cat_cols, known_cats)
        X_val = _onehot_encode(X_val, cat_cols, known_cats)
        X_train, X_val = _align_ohe(X_train, X_val)

        # -- Standardize features for stability -------------------------
        scaler = StandardScaler()
        X_train_s = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_val_s = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index,
        )
        feat_time = time.time() - t_feat

        # -- Train ------------------------------------------------------
        t_train = time.time()
        model = _build_model(model_cfg)
        ts_model = TimeSeriesModel(model)
        ts_model.fit(X_train_s, y_train)
        train_time = time.time() - t_train

        # -- Evaluate on val --------------------------------------------
        val_preds_log = ts_model.fold_models_[0].predict(X_val_s[ts_model.feature_names_])
        if target_transform == "log1p":
            val_preds = np.expm1(val_preds_log)
        else:
            val_preds = val_preds_log
        val_preds = np.maximum(val_preds, 0)

        metrics = _compute_metrics(y_val_raw.values, val_preds)
        metrics["train_time_s"] = round(feat_time + train_time, 1)
        metrics["n_features"] = X_train.shape[1]
        metrics["label"] = label
        results.append(metrics)

        print(
            f"  {train_time:6.1f}s  |  n={X_train.shape[1]:3d}  "
            f"|  RMSLE={metrics['val_rmsle']:.4f}  "
            f"|  MAE={metrics['val_mae']:.1f}  "
            f"|  WMAPE={metrics['val_wmape']:.3f}  "
            f"|  Bias={metrics['val_bias_pct']:+.1f}%  "
            f"|  RMSE={metrics['val_rmse']:.1f}",
            flush=True,
        )

    # -- Scan for existing runs -----------------------------------------
    existing = _scan_existing_runs()
    for r in existing:
        r.update({k: None for k in ("val_mae", "val_wmape", "val_bias_pct", "val_rmse")})
        results.append(r)

    # -- Print comparison table -----------------------------------------
    df = pd.DataFrame(results)
    col_order = [
        "label",
        "train_time_s",
        "n_features",
        "val_rmsle",
        "val_mae",
        "val_wmape",
        "val_bias_pct",
        "val_rmse",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.colheader_justify", "right")

    print("\n" + "=" * 140)
    print("  Linear Model Comparison (re-trained from scratch)")
    print("=" * 140)
    print(df.to_string(index=False, na_rep="  —  "))
    print("=" * 140)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
