"""Nixtla baseline training pipeline for Store Sales.

Usage:
    uv run python scripts/train_nixtla.py
    uv run python scripts/train_nixtla.py --config config/nixtla.yaml --run-name nixtla-stats

Requires the optional nixtla extra: `uv sync --extra nixtla`.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from pathlib import Path

from store_sales.data import load_config, load_data, merge_tables
from store_sales.nixtla_pipeline import (
    cross_validate,
    fit_predict,
    rmsle_for_models,
    to_long,
    to_submission,
)
from store_sales.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)
# AutoETS triggers harmless ACF divide-by-zero warnings on zero-variance series
warnings.filterwarnings("ignore", category=RuntimeWarning, module="statsmodels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nixtla baseline for store sales.")
    parser.add_argument("--config", type=str, default="config/nixtla.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("=" * 60)
    logger.info("Store Sales — Nixtla Baseline")
    logger.info("=" * 60)

    t0 = time.time()

    # -- 1. Load + merge ------------------------------------------------
    logger.info("[1/4] Loading data ...")
    tables = load_data(cfg["paths"]["data"])
    train, test = merge_tables(tables)

    nx_cfg = cfg["nixtla"]
    horizon = nx_cfg.get("horizon", 16)
    freq = nx_cfg.get("freq", "D")
    season_length = nx_cfg.get("season_length", 7)
    n_jobs = nx_cfg.get("n_jobs", -1)
    model_kind_cfg = {
        k: v
        for k, v in nx_cfg.items()
        if k not in ("freq", "horizon", "season_length", "n_jobs", "cv")
    }

    # -- 2. Reshape to Nixtla long format --------------------------------
    logger.info("[2/4] Reshaping to long format ...")
    history, future = to_long(train, test, target_col=cfg["competition"]["target"])
    logger.info("  History: %s  Future: %s", history.shape, future.shape)

    # -- 3. Cross-validate for RMSLE benchmark ---------------------------
    val_rmsle = None
    cv_scores = None
    if not args.no_cv:
        logger.info("[3/4] Cross-validating ...")
        cv_cfg = nx_cfg.get("cv", {})
        n_windows = cv_cfg.get("n_windows", 1)
        step_size = cv_cfg.get("step_size")
        cv_df = cross_validate(
            history,
            model_kind_cfg,
            horizon=horizon,
            n_windows=n_windows,
            step_size=step_size,
            freq=freq,
            season_length=season_length,
            n_jobs=n_jobs,
        )
        cv_scores = rmsle_for_models(cv_df)
        val_rmsle = float(cv_scores.iloc[0])
        logger.info("  CV RMSLE per model:")
        for name, score in cv_scores.items():
            logger.info("    %s: %.6f", name, score)
        logger.info("  Best: %s (%.6f)", cv_scores.index[0], val_rmsle)
    else:
        logger.info("[3/4] Skipping CV (--no-cv)")

    # -- 4. Fit on full history + generate submission --------------------
    logger.info("[4/4] Fitting + predicting ...")
    # If CV ran, only fit the best model (avoids refitting slow models like AutoETS)
    fit_cfg = model_kind_cfg
    if cv_scores is not None:
        best_model = cv_scores.index[0]
        best_alias = best_model
        fit_cfg = {
            **model_kind_cfg,
            "models": [
                m for m in nx_cfg.get("models", []) if m.get("alias", m["name"]) == best_alias
            ],
        }
        logger.info("  Fitting only best model: %s", best_model)
    else:
        best_model = None
    forecast = fit_predict(
        history,
        future,
        fit_cfg,
        horizon=horizon,
        freq=freq,
        season_length=season_length,
        n_jobs=n_jobs,
    )

    submission = to_submission(future, forecast, model_col=best_model)
    logger.info("  Submission: %d rows", len(submission))

    with track_experiment(cfg, run_name=args.run_name) as run:
        metrics = {"val_rmsle": round(val_rmsle, 6) if val_rmsle is not None else None}
        if cv_scores is not None:
            for name, score in cv_scores.items():
                metrics[f"cv_rmsle_{name}"] = round(float(score), 6)
        run.log_metrics(metrics)
        run.log_params(
            {
                "model_type": "nixtla_" + model_kind_cfg.get("kind", "stats"),
                "horizon": horizon,
                "n_models": len(nx_cfg.get("models", [])),
                "best_model": best_model,
                "run_scope": cfg.get("run_scope", "full"),
            }
        )

        sub_path = run.submission_path
        submission.to_csv(sub_path, index=False)
        logger.info("  Submission saved → %s", sub_path)
        submissions_dir = Path(cfg["paths"]["submissions"])
        submission.to_csv(submissions_dir / "submission_nixtla.csv", index=False)
        logger.info("  Submission saved → %s", submissions_dir / "submission_nixtla.csv")

    logger.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
