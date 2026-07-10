"""Plot daily aggregate sales: actual vs predicted for one or more models.

Shows the validation period only (where both actual and predicted exist).

Usage:
    # Single model
    uv run python scripts/plot_daily_aggregate.py --run outputs/runs/20260706_213830_linear-full

    # Multiple models (will run prediction for each)
    uv run python scripts/plot_daily_aggregate.py \\
        --run outputs/runs/20260706_213830_linear-full \\
        --run outputs/runs/20260703_071135_bench-lightgbm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from store_sales.data import load_config, load_data, merge_tables
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.metrics import rmsle
from store_sales.models import TimeSeriesModel

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")


def _predict_run(
    run_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_full: pd.Series,
) -> pd.DataFrame:
    """Load a run, re-run inference, return (date, actual, predicted) for all train rows."""
    cfg = load_config(str(run_dir / "config.yaml"))
    feat_cfg = cfg["features"]
    target_col = cfg["competition"]["target"]
    target_transform = cfg.get("model", {}).get("target_transform", "raw")

    run_train = train.copy()
    run_test = test.copy()
    if target_transform == "log1p":
        run_train["log_sales"] = np.log1p(run_train[target_col])
        run_test["log_sales"] = 0.0

    engineer = TimeSeriesFeatureEngineer(
        date_col=feat_cfg.get("date_col", "date"),
        store_col=feat_cfg.get("store_col", "store_nbr"),
        family_col=feat_cfg.get("family_col", "family"),
        onpromotion_col=feat_cfg.get("onpromotion_col", "onpromotion"),
        date_features=feat_cfg.get("date_features", []),
        drop_cols=feat_cfg.get("drop_cols", []),
        lag_config=feat_cfg.get("lag_features", []),
        rolling_config=feat_cfg.get("rolling_features", []),
        fourier_config=feat_cfg.get("fourier_features"),
        ref_date=run_train["date"].min(),
    )

    X_lag, X_test_feat = engineer.create_lag_features(run_train, run_test, target_col)
    engineer.fit(X_lag)
    X_all = engineer.transform(X_lag)

    # One-hot encode if needed
    ts_model = TimeSeriesModel.load(run_dir / "models" / "model.joblib")
    model_feats = set(ts_model.feature_names_)
    cat_cols = [c for c in X_all.columns if X_all[c].dtype.name == "category"]
    needs_ohe = bool(cat_cols) and not (cat_cols and cat_cols[0] in model_feats)

    if needs_ohe:
        known_cats = {c: sorted(X_all[c].cat.categories.tolist()) for c in cat_cols}
        for c in cat_cols:
            X_all[c] = pd.Categorical(X_all[c], categories=known_cats[c])
        X_all = pd.get_dummies(X_all, columns=cat_cols, drop_first=True, dtype=int)

    preds = ts_model.predict(X_all)
    if target_transform == "log1p":
        preds = np.expm1(preds)
    preds = np.maximum(preds, 0)

    result = X_lag[["date"]].copy()
    result["actual"] = y_full.loc[X_lag.index].values
    result["predicted"] = preds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot daily aggregate sales by model.")
    parser.add_argument(
        "--run",
        type=str,
        action="append",
        required=True,
        help="Run directory (repeatable for multiple models)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Save plot to this path (e.g. daily_aggregate.png)"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=90,
        help="Training days before val split to show (default: 90)",
    )
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.run]
    for d in run_dirs:
        if not (d / "models" / "model.joblib").exists():
            raise SystemExit(f"Model not found in {d}")

    print("Loading data ...", flush=True)
    tables = load_data()
    train, test = merge_tables(tables)
    y_full = train["sales"].copy()
    print(f"  Train: {train.shape}  Test: {test.shape}", flush=True)

    all_dates = sorted(train["date"].unique())
    val_split_date = str(all_dates[-16])
    val_start = pd.Timestamp(val_split_date)
    window_start = val_start - pd.Timedelta(days=args.days_back)

    fig, ax = plt.subplots(figsize=(16, 5))

    for run_dir in run_dirs:
        name = run_dir.name
        print(f"  Predicting {name} ...", flush=True)
        result = _predict_run(run_dir, train, test, y_full)
        result["date"] = pd.to_datetime(result["date"])

        # Filter to window
        result = result[result["date"] >= window_start].copy()

        # Daily aggregate
        daily = (
            result.groupby("date")
            .agg(
                actual=("actual", "sum"),
                predicted=("predicted", "sum"),
            )
            .reset_index()
            .sort_values("date")
        )

        # Validation period only (completed predictions)
        val_daily = daily[daily["date"] >= val_start].dropna()
        score = rmsle(val_daily["actual"].values, val_daily["predicted"].values)

        ax.plot(
            daily["date"],
            daily["predicted"],
            linewidth=0.8,
            alpha=0.7,
            label=f"{name}  (val RMSLE={score:.4f})",
        )

    # Plot actuals (once, from the last model's data)
    last_result = result.copy()
    daily_actual = last_result.groupby("date")["actual"].sum().reset_index().sort_values("date")
    ax.plot(
        daily_actual["date"],
        daily_actual["actual"],
        color="black",
        linewidth=1.0,
        alpha=0.8,
        label="Actual",
    )

    # Val split line
    vd = pd.Timestamp(val_split_date)
    ax.axvline(vd, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.text(vd, ax.get_ylim()[1] * 0.97, "val split", fontsize=8, rotation=90, va="top", alpha=0.6)

    ax.set_title("Daily aggregate sales — actual vs predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Total sales (sum across all stores)")
    ax.legend(fontsize=8, ncol=1, loc="upper left")
    fig.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"  Saved → {args.output}", flush=True)
    plt.show()


if __name__ == "__main__":
    main()
