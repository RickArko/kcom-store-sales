"""Inference from a saved model for Store Sales.

Usage:
    uv run python scripts/predict.py --run-dir outputs/runs/20260701_100000_expr-001
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from store_sales.data import load_config, load_data, merge_tables
from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.models import TimeSeriesModel, save_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submissions from a trained model.")
    parser.add_argument(
        "--run-dir", type=str, required=True, help="Path to experiment run directory"
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for submission CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = run_dir / "config.yaml"
    model_path = run_dir / "models" / "model.joblib"

    cfg = load_config(str(config_path))

    logger.info("[1/4] Loading data ...")
    tables = load_data(cfg["paths"]["data"])
    train, test = merge_tables(tables)

    logger.info("[2/4] Engineering features ...")
    target_col = cfg["competition"]["target"]
    X_train_feat, X_test_feat = TimeSeriesFeatureEngineer(
        date_col=cfg["features"].get("date_col", "date"),
        store_col=cfg["features"].get("store_col", "store_nbr"),
        family_col=cfg["features"].get("family_col", "family"),
        onpromotion_col=cfg["features"].get("onpromotion_col", "onpromotion"),
        date_features=cfg["features"].get("date_features", []),
        drop_cols=cfg["features"].get("drop_cols", []),
        lag_config=cfg["features"].get("lag_features", []),
        rolling_config=cfg["features"].get("rolling_features", []),
    ).create_lag_features(train, test, target_col)

    logger.info("[3/4] Loading model from %s ...", model_path)
    model = TimeSeriesModel.load(model_path)
    test_preds = model.predict(X_test_feat)

    logger.info("[4/4] Saving submission ...")
    output_path = args.output or str(run_dir / "submission.csv")
    save_submission(test["id"], test_preds, output_path)
    logger.info("Done!")


if __name__ == "__main__":
    main()
