"""Compare all experiment runs in a table.

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py --sort-by val_rmsle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare experiment runs.")
    parser.add_argument("--runs-dir", type=str, default="outputs/runs")
    parser.add_argument("--sort-by", type=str, default="val_rmsle", help="Column to sort by")
    parser.add_argument(
        "--ascending",
        action="store_true",
        default=True,
        help="Sort ascending (lower is better for error)",
    )
    return parser.parse_args()


def _load_runs(runs_dir: Path) -> list[dict]:
    rows = []
    if not runs_dir.exists():
        return rows

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)

        meta = data.get("metrics", {})
        params = data.get("params", {})
        submissions_exist = (run_dir / "submission.csv").exists()

        rows.append(
            {
                "run": run_dir.name,
                "val_rmsle": meta.get("val_rmsle"),
                "n_features": meta.get("n_features"),
                "model_type": params.get("model_type"),
                "n_estimators": params.get("n_estimators"),
                "elapsed_seconds": round(data.get("elapsed_seconds", 0), 1),
                "submission_exists": submissions_exist,
                "path": str(run_dir),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = _load_runs(runs_dir)

    if not rows:
        print("No experiment runs found.")
        return

    df = pd.DataFrame(rows)
    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.colheader_justify", "right")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
