"""Re-run inference from a saved experiment run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from store_sales.data import apply_preprocessing, load_config
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.models import TimeSeriesModel


def build_feature_engineer(
    cfg: dict,
    *,
    ref_date,
    holiday_dates: list[str] | None = None,
) -> TimeSeriesFeatureEngineer:
    """Build a feature engineer matching the training pipeline."""
    feat_cfg = cfg["features"]
    return TimeSeriesFeatureEngineer(
        date_col=feat_cfg.get("date_col", "date"),
        store_col=feat_cfg.get("store_col", "store_nbr"),
        family_col=feat_cfg.get("family_col", "family"),
        onpromotion_col=feat_cfg.get("onpromotion_col", "onpromotion"),
        date_features=feat_cfg.get("date_features", []),
        drop_cols=feat_cfg.get("drop_cols", []),
        lag_config=feat_cfg.get("lag_features", []),
        rolling_config=feat_cfg.get("rolling_features", []),
        fourier_config=feat_cfg.get("fourier_features"),
        holiday_dates=holiday_dates,
        ref_date=ref_date,
    )


def _prepare_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    target_col = cfg["competition"]["target"]
    target_transform = cfg.get("model", {}).get("target_transform", "raw")

    run_train = train.copy()
    run_test = test.copy()
    if target_transform == "log1p":
        run_train["log_sales"] = np.log1p(run_train[target_col])
        run_test["log_sales"] = 0.0

    return run_train, run_test, target_col, target_transform


def _onehot_if_needed(
    X_all: pd.DataFrame,
    X_test: pd.DataFrame,
    ts_model: TimeSeriesModel,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_feats = set(ts_model.feature_names_)
    cat_cols = [c for c in X_all.columns if X_all[c].dtype.name == "category"]
    needs_ohe = bool(cat_cols) and not (cat_cols and cat_cols[0] in model_feats)
    if not needs_ohe:
        return X_all, X_test

    known_cats = {c: sorted(X_all[c].cat.categories.tolist()) for c in cat_cols}
    for df in (X_all, X_test):
        for c in cat_cols:
            df[c] = pd.Categorical(df[c], categories=known_cats[c])
    X_all = pd.get_dummies(X_all, columns=cat_cols, drop_first=True, dtype=int)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
    train_cols = list(X_all.columns)
    for c in set(train_cols) - set(X_test.columns):
        X_test[c] = 0
    X_test = X_test[train_cols]
    return X_all, X_test


def _decode_predictions(preds: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        preds = np.expm1(preds)
    return np.maximum(preds, 0)


def predict_from_run(
    run_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_full: pd.Series | None = None,
    *,
    holiday_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Load a run, re-run inference, return per-row actuals and predictions.

    Columns: date, store_nbr, family, actual, predicted, split (train|test).

    ``y_full`` is accepted for backward compatibility but ignored — actuals are
    taken from the post-preprocessing train frame so trim/index changes align.
    """
    _ = y_full
    cfg = load_config(str(run_dir / "config.yaml"))
    run_train_base, _ = apply_preprocessing(train.copy(), cfg)
    # Actuals must come from the post-preprocessing frame (trim may reset index).
    y_actual = run_train_base[cfg["competition"]["target"]].copy()
    run_train, run_test, target_col, target_transform = _prepare_frames(run_train_base, test, cfg)

    engineer = build_feature_engineer(
        cfg,
        ref_date=run_train["date"].min(),
        holiday_dates=holiday_dates,
    )
    X_lag, X_test_feat = engineer.create_lag_features(run_train, run_test, target_col)
    engineer.fit(X_lag)
    X_all = engineer.transform(X_lag)
    X_test = engineer.transform(X_test_feat)

    ts_model = TimeSeriesModel.load(run_dir / "models" / "model.joblib")
    X_all, X_test = _onehot_if_needed(X_all, X_test, ts_model)

    train_preds = _decode_predictions(ts_model.predict(X_all), target_transform)
    test_preds = _decode_predictions(ts_model.predict(X_test), target_transform)

    meta = X_lag[["date", "store_nbr", "family"]].copy()
    meta["actual"] = y_actual.loc[X_lag.index].values
    meta["predicted"] = train_preds
    meta["split"] = "train"

    meta_test = X_test_feat[["date", "store_nbr", "family"]].copy()
    meta_test["actual"] = np.nan
    meta_test["predicted"] = test_preds
    meta_test["split"] = "test"

    return pd.concat([meta, meta_test], ignore_index=True)


def predict_daily_from_run(
    run_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_full: pd.Series | None = None,
    *,
    holiday_dates: list[str] | None = None,
) -> pd.DataFrame:
    """Re-run inference and return daily aggregate (date, actual, predicted)."""
    _ = y_full
    cfg = load_config(str(run_dir / "config.yaml"))
    run_train_base, _ = apply_preprocessing(train.copy(), cfg)
    y_actual = run_train_base[cfg["competition"]["target"]].copy()
    run_train, run_test, target_col, target_transform = _prepare_frames(run_train_base, test, cfg)

    engineer = build_feature_engineer(
        cfg,
        ref_date=run_train["date"].min(),
        holiday_dates=holiday_dates,
    )
    X_lag, _ = engineer.create_lag_features(run_train, run_test, target_col)
    engineer.fit(X_lag)
    X_all = engineer.transform(X_lag)

    ts_model = TimeSeriesModel.load(run_dir / "models" / "model.joblib")
    X_all, _ = _onehot_if_needed(X_all, X_all, ts_model)
    preds = _decode_predictions(ts_model.predict(X_all), target_transform)

    result = X_lag[["date"]].copy()
    result["actual"] = y_actual.loc[X_lag.index].values
    result["predicted"] = preds
    return result


def _run_model_type(run_dir: Path) -> str:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return "?"
    with open(metrics_path) as f:
        return json.load(f).get("params", {}).get("model_type", "?")


def load_submission_predictions(run_dir: Path, test: pd.DataFrame) -> pd.DataFrame:
    """Load submission.csv and attach test metadata."""
    sub = pd.read_csv(run_dir / "submission.csv")
    meta = test[["id", "date", "store_nbr", "family"]].copy()
    out = meta.merge(sub.rename(columns={"sales": "predicted"}), on="id", how="left")
    out["actual"] = np.nan
    out["split"] = "test"
    out["run"] = run_dir.name
    out["model_type"] = _run_model_type(run_dir)
    return out


def summarize_submission(run_dir: Path) -> dict:
    """Summary stats for one run's test submission."""
    sub = pd.read_csv(run_dir / "submission.csv")
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    params = metrics.get("params", {})
    sales = sub["sales"]
    return {
        "run": run_dir.name,
        "model_type": params.get("model_type", "?"),
        "val_rmsle": metrics.get("metrics", {}).get("val_rmsle"),
        "test_median": round(float(sales.median()), 3),
        "test_mean": round(float(sales.mean()), 1),
        "pct_zero": round((sales == 0).mean() * 100, 1),
        "pct_lt1": round((sales < 1).mean() * 100, 1),
    }


def compare_submissions(run_dirs: list[Path]) -> pd.DataFrame:
    """Compare test-submission distributions across runs."""
    rows = [summarize_submission(d) for d in run_dirs if (d / "submission.csv").exists()]
    return pd.DataFrame(rows).sort_values("val_rmsle")


def load_all_submissions(run_dirs: list[Path], test: pd.DataFrame) -> pd.DataFrame:
    """Stack test predictions from multiple run submissions."""
    parts = [
        load_submission_predictions(d, test) for d in run_dirs if (d / "submission.csv").exists()
    ]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
