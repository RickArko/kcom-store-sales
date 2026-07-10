"""Zero-shot TOTO foundation-model forecasting for Store Sales.

Usage:
    uv run python scripts/train_toto.py --config config/toto.yaml --run-name toto-22m
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

from store_sales.data import load_config, load_data, merge_tables
from store_sales.models import save_submission
from store_sales.toto_pipeline import TotoPipeline
from store_sales.tracking import track_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TOTO zero-shot forecasting for Store Sales.")
    parser.add_argument(
        "--config", type=str, default="config/toto.yaml", help="Path to config YAML"
    )
    parser.add_argument("--run-name", type=str, default=None, help="Human-readable experiment name")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    cfg = load_config(args.config)
    toto_cfg = cfg["toto"]

    logger.info("=" * 60)
    logger.info("Store Sales — TOTO Zero-Shot Forecasting")
    logger.info("=" * 60)

    total_start = time.time()

    # -- 1. Load data --------------------------------------------------
    logger.info("[1/5] Loading data ...")
    t0 = time.time()
    tables = load_data(cfg["paths"]["data"])
    train, test = merge_tables(tables)
    logger.info("  Loaded in %.1fs", time.time() - t0)

    # -- 2. Build pipeline ---------------------------------------------
    logger.info("[2/5] Initializing TOTO pipeline (%s) ...", toto_cfg["model_name"])
    pipe = TotoPipeline(
        model_name=toto_cfg["model_name"],
        decode_block_size=toto_cfg.get("decode_block_size"),
        context_length=toto_cfg.get("context_length"),
        variate_batch_size=toto_cfg.get("variate_batch_size"),
        log_transform=toto_cfg.get("log_transform", False),
    )

    # -- 3. Validation -------------------------------------------------
    logger.info("[3/5] Running validation (last %d days) ...", toto_cfg.get("val_days", 16))
    t0 = time.time()
    val_score = pipe.validate(
        train,
        horizon=toto_cfg.get("horizon", 16),
        val_days=toto_cfg.get("val_days", 16),
    )
    logger.info("  Validation RMSLE: %.6f  (%.1fs)", val_score, time.time() - t0)

    # -- 4. Full forecast & submission ---------------------------------
    logger.info("[4/5] Generating full test forecasts ...")
    t0 = time.time()
    submission = pipe.predict(
        train,
        test,
        horizon=toto_cfg.get("horizon", 16),
    )
    logger.info("  Forecast complete (%.1fs)", time.time() - t0)

    # -- 5. Save results -----------------------------------------------
    logger.info("[5/5] Saving results ...")
    with track_experiment(cfg, run_name=args.run_name) as run:
        run.log_metrics(
            {
                "val_rmsle": round(val_score, 6),
            }
        )
        run.log_params(
            {
                "model_type": "toto",
                "model_name": toto_cfg["model_name"],
                "horizon": toto_cfg.get("horizon", 16),
                "val_days": toto_cfg.get("val_days", 16),
                "decode_block_size": toto_cfg.get("decode_block_size"),
                "context_length": toto_cfg.get("context_length"),
                "variate_batch_size": toto_cfg.get("variate_batch_size"),
                "log_transform": toto_cfg.get("log_transform", False),
                "run_scope": cfg.get("run_scope", "full"),
            }
        )

        submission.to_csv(run.submission_path, index=False)
        logger.info("  Submission saved → %s", run.submission_path)

        save_submission(
            submission["id"],
            submission["sales"].values,
            str(Path(cfg["paths"]["submissions"]) / "submission.csv"),
        )

    logger.info("Done in %.1fs", time.time() - total_start)


if __name__ == "__main__":
    main()
