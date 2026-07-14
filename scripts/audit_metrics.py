"""Offline Ilya-style metrics audit for a saved tabular run.

Usage:
    uv run python scripts/audit_metrics.py --run-dir outputs/runs/<run>
    uv run python scripts/audit_metrics.py --run-dir outputs/runs/<a> \\
        --compare-dir outputs/runs/<b>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from store_sales.data import (
    apply_preprocessing,
    extract_holiday_dates,
    load_config,
    load_data,
    merge_tables,
    timeseries_split,
)
from store_sales.inference import predict_from_run
from store_sales.metrics_audit import (
    assign_velocity_tiers,
    bootstrap_rmsle,
    compute_row_metrics,
    enrich_val_frame,
    error_tree_leaves,
    filter_validation,
    multiaccuracy_audit,
    paired_bootstrap_delta,
    render_report,
    residual_clusters,
    shift_summary,
    slice_scores,
    write_audit_outputs,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _require_tabular_model(run_dir: Path) -> None:
    model_path = run_dir / "models" / "model.joblib"
    if not model_path.exists():
        raise SystemExit(
            f"No models/model.joblib in {run_dir} — audit v1 supports tabular runs only"
            " (LightGBM / XGBoost / linear)."
        )


def _load_run_meta(run_dir: Path, cfg: dict) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "trim_pre_activation_zeros": bool(
            cfg.get("preprocessing", {}).get("trim_pre_activation_zeros", False)
        ),
        "test_period_days": int(cfg.get("timeseries", {}).get("test_period_days", 16)),
        "cv_one_shot_mean": None,
        "cv_one_shot_std": None,
        "cv_recursive_mean": None,
        "cv_recursive_std": None,
    }
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data = json.load(f)
        m = data.get("metrics", {})
        for key in (
            "cv_one_shot_mean",
            "cv_one_shot_std",
            "cv_recursive_mean",
            "cv_recursive_std",
        ):
            if key in m:
                meta[key] = m[key]
    return meta


def _build_val_frame(
    run_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_full: pd.Series,
    holiday_dates: list[str] | None,
    cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (val_enriched, train_fit, val_raw_merged, test)."""
    test_period_days = int(cfg.get("timeseries", {}).get("test_period_days", 16))
    run_train, _ = apply_preprocessing(train.copy(), cfg)
    train_fit, val_raw = timeseries_split(run_train, test_period_days=test_period_days)

    logger.info("Predicting from %s ...", run_dir.name)
    preds = predict_from_run(run_dir, train, test, y_full, holiday_dates=holiday_dates)
    val = filter_validation(preds, test_period_days=test_period_days)
    velocity = assign_velocity_tiers(train_fit)
    enriched = enrich_val_frame(val, run_train, velocity)
    return enriched, train_fit, val_raw, test


def audit_run(
    run_dir: Path,
    *,
    compare_dir: Path | None = None,
    n_boot: int = 500,
    seed: int = 42,
) -> Path:
    _require_tabular_model(run_dir)
    if compare_dir is not None:
        _require_tabular_model(compare_dir)

    cfg = load_config(str(run_dir / "config.yaml"))
    run_meta = _load_run_meta(run_dir, cfg)

    logger.info("Loading data ...")
    tables = load_data()
    train, test = merge_tables(tables)
    holiday_dates = extract_holiday_dates(tables)
    y_full = train["sales"].copy()

    val_a, train_fit, val_raw, test_merged = _build_val_frame(
        run_dir, train, test, y_full, holiday_dates, cfg
    )
    if val_a.empty:
        raise SystemExit(f"No validation rows for {run_dir}")

    global_metrics = compute_row_metrics(val_a["actual"].to_numpy(), val_a["predicted"].to_numpy())
    boot = bootstrap_rmsle(val_a, n_boot=n_boot, seed=seed)
    slices = slice_scores(val_a)
    shift = shift_summary(train_fit, val_raw, test_merged)
    leaves = error_tree_leaves(val_a)
    multiacc = multiaccuracy_audit(val_a, seed=seed)
    clusters = residual_clusters(val_a, seed=seed)

    paired = None
    compare_name = None
    if compare_dir is not None:
        cfg_b = load_config(str(compare_dir / "config.yaml"))
        val_b, _, _, _ = _build_val_frame(compare_dir, train, test, y_full, holiday_dates, cfg_b)
        paired = paired_bootstrap_delta(val_a, val_b, n_boot=n_boot, seed=seed)
        compare_name = compare_dir.name

    report = render_report(
        run_name=run_dir.name,
        global_metrics=global_metrics,
        bootstrap=boot,
        slices=slices,
        shift=shift,
        error_leaves=leaves,
        multiacc=multiacc,
        clusters=clusters,
        run_meta=run_meta,
        paired=paired,
        compare_run=compare_name,
    )

    summary: dict[str, Any] = {
        "run": run_dir.name,
        "global": global_metrics,
        "bootstrap": boot,
        "shift": shift,
        "run_meta": run_meta,
        "multiaccuracy": multiacc,
        "error_tree_leaves": [
            {k: v for k, v in leaf.items() if k != "tree_excerpt"} for leaf in leaves
        ],
        "residual_clusters": clusters,
    }
    if paired is not None:
        summary["paired_compare"] = {"run_b": compare_name, **paired}

    out_dir = run_dir / "metrics_audit"
    write_audit_outputs(out_dir, summary=summary, slices=slices, report_md=report)
    logger.info("Wrote %s", out_dir)
    print(report)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Ilya-style offline metrics audit.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Primary run directory under outputs/runs/",
    )
    parser.add_argument(
        "--compare-dir",
        type=str,
        default=None,
        help="Optional second run for paired bootstrap Δ RMSLE",
    )
    parser.add_argument("--n-boot", type=int, default=500, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    compare_dir = Path(args.compare_dir) if args.compare_dir else None
    audit_run(run_dir, compare_dir=compare_dir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
