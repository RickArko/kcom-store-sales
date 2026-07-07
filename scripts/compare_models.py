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
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
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


def _label_to_run_name(label: str) -> str:
    return f"bench-{label.lower().replace(' ', '-').replace('(', '').replace(')', '')}"


def _find_existing_run(label: str) -> Path | None:
    """Return path to an existing run dir for this label, or None."""
    run_name = _label_to_run_name(label)
    runs_dir = Path("outputs/runs")
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.endswith(run_name):
            if (d / "submission.csv").exists() and (d / "metrics.json").exists():
                return d
    return None


def _load_run_metrics(run_dir: Path, label: str) -> dict:
    """Load metrics from an existing run directory."""
    with open(run_dir / "metrics.json") as f:
        data = json.load(f)
    meta = data.get("metrics", {})
    n_feat = meta.get("n_features")
    try:
        n_feat = int(n_feat) if n_feat is not None else None
    except (ValueError, TypeError):
        n_feat = None
    return {
        "label": label,
        "train_time_s": data.get("elapsed_seconds", 0),
        "n_features": n_feat,
        "val_rmsle": meta.get("val_rmsle"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model variants.")
    parser.add_argument("--output", type=str, default=None, help="Save comparison CSV to path")
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Path to benchmark YAML (overrides hardcoded experiment list)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip existing runs (default: on). Pass --no-skip-existing to re-train.",
    )
    return parser.parse_args()


def _run_experiment(label: str, config_path: str, script: str) -> dict | None:
    """Shell out to the training script and return metrics from the resulting run."""
    run_name = _label_to_run_name(label)
    cmd = [
        sys.executable,
        f"scripts/{script}.py",
        "--config",
        config_path,
        "--run-name",
        run_name,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {label} failed:\n{result.stderr[:500]}", flush=True)
        return None
    # Find the latest run with this run_name
    runs_dir = Path("outputs/runs")
    latest = None
    for d in sorted(runs_dir.iterdir(), reverse=True):
        if d.is_dir() and d.name.endswith(run_name):
            latest = d
            break
    if latest is None:
        return None
    metrics_path = latest / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    meta = data.get("metrics", {})
    n_feat = meta.get("n_features")
    try:
        n_feat = int(n_feat) if n_feat is not None else None
    except (ValueError, TypeError):
        n_feat = None
    return {
        "label": label,
        "train_time_s": data.get("elapsed_seconds", 0),
        "n_features": n_feat,
        "val_rmsle": meta.get("val_rmsle"),
    }


def _run_linear_experiment(
    label: str,
    config_path: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_train_full_raw: pd.Series,
) -> dict:
    """Run a single linear-model experiment in-process (fast)."""
    cfg = load_config(config_path)
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    target_transform = model_cfg.get("target_transform", "raw")
    t_feat = time.time()

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
    X_train_lag, _ = engineer.create_lag_features(
        train_work, test_work, cfg["competition"]["target"]
    )
    engineer.fit(X_train_lag)

    val_period = cfg.get("timeseries", {}).get("test_period_days", 16)
    X_train_raw, X_val_raw = timeseries_split(X_train_lag, val_period)
    y_val_raw = y_train_full_raw.loc[X_val_raw.index]
    y_train = y_for_model.loc[X_train_raw.index]

    X_train = engineer.transform(X_train_raw)
    X_val = engineer.transform(X_val_raw)

    cat_cols = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    known_cats: dict[str, list] = {}
    for col in cat_cols:
        known_cats[col] = sorted(X_train[col].cat.categories.tolist())
    X_train = _onehot_encode(X_train, cat_cols, known_cats)
    X_val = _onehot_encode(X_val, cat_cols, known_cats)
    X_train, X_val = _align_ohe(X_train, X_val)

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

    t_train = time.time()
    model = _build_model(model_cfg)
    ts_model = TimeSeriesModel(model)
    ts_model.fit(X_train_s, y_train)
    train_time = time.time() - t_train

    val_preds_log = ts_model.fold_models_[0].predict(X_val_s[ts_model.feature_names_])
    val_preds = np.expm1(val_preds_log) if target_transform == "log1p" else val_preds_log
    val_preds = np.maximum(val_preds, 0)

    metrics = _compute_metrics(y_val_raw.values, val_preds)
    metrics["train_time_s"] = round(feat_time + train_time, 1)
    metrics["n_features"] = X_train.shape[1]
    metrics["label"] = label
    return metrics


def main() -> None:
    args = parse_args()

    experiments: list[tuple[str, str, str | None]] = []
    if args.experiments:
        with open(args.experiments) as f:
            bench_cfg = yaml.safe_load(f)
        for exp in bench_cfg.get("experiments", []):
            experiments.append((exp["label"], exp["config"], exp.get("script")))
    else:
        experiments = [(label, cfg, None) for label, cfg in EXPERIMENTS]

    has_nonlinear = any(s not in (None, "train_linear") for _, _, s in experiments)

    results: list[dict] = []

    if has_nonlinear:
        # Dispatch each experiment to its training script
        for label, config_path, script in experiments:
            existing = _find_existing_run(label) if args.skip_existing else None
            if existing is not None:
                print(f"\n── {label} ── skipping (existing run)", flush=True)
                result = _load_run_metrics(existing, label)
                if result:
                    results.append(result)
                continue
            print(f"\n── {label} ──", flush=True)
            result = _run_experiment(label, config_path, script or "train_linear")
            if result:
                results.append(result)
                n_str = str(result["n_features"] or "—")
                r_str = str(result["val_rmsle"] or "—")
                print(
                    f"  {result['train_time_s']:6.1f}s  |  n={n_str:>4}  |  RMSLE={r_str:>8}",
                    flush=True,
                )
            else:
                # Try loading from a previous run as fallback
                fallback = _find_existing_run(label)
                if fallback:
                    result = _load_run_metrics(fallback, label)
                    if result:
                        results.append(result)
    else:
        # Fast in-process linear-only comparison
        print("Loading data ...", flush=True)
        tables = load_data()
        train, test = merge_tables(tables)
        y_train_full_raw = train["sales"].copy()

        for label, config_path, _ in experiments:
            print(f"\n── {label} ──", flush=True)
            m = _run_linear_experiment(label, config_path, train, test, y_train_full_raw)
            results.append(m)
            print(
                f"  {m['train_time_s']:6.1f}s  |  n={m['n_features']:3d}  "
                f"|  RMSLE={m['val_rmsle']:.4f}  "
                f"|  MAE={m['val_mae']:.1f}  "
                f"|  WMAPE={m['val_wmape']:.3f}  "
                f"|  Bias={m['val_bias_pct']:+.1f}%  "
                f"|  RMSE={m['val_rmse']:.1f}",
                flush=True,
            )

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

    subtitle = (
        "Full Benchmark" if has_nonlinear else "Linear Model Comparison (re-trained from scratch)"
    )
    print(f"\n{'=' * 160}")
    print(f"  {subtitle}")
    print(f"{'=' * 160}")
    print(df.to_string(index=False, na_rep="  —  "))
    print(f"{'=' * 160}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
