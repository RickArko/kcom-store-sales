"""Pick the best submission from the latest benchmark runs.

Compares val_rmsle across the two most recent bench-* runs and copies the
lower-RMSLE submission to outputs/submissions/submission.csv.

Usage:
    uv run python scripts/pick_best.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick best submission from benchmark runs.")
    parser.add_argument("--runs-dir", type=str, default="outputs/runs")
    parser.add_argument("--prefix", type=str, default="bench-")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output = Path(args.output)

    candidates = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or not run_dir.name.startswith(args.prefix):
            continue
        metrics_path = run_dir / "metrics.json"
        sub_path = run_dir / "submission.csv"
        if not metrics_path.exists() or not sub_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        rmsle = data.get("metrics", {}).get("val_rmsle")
        if rmsle is None:
            continue
        candidates.append({"run": run_dir.name, "val_rmsle": rmsle, "path": str(sub_path)})

    if not candidates:
        print("No benchmark runs with val_rmsle found.")
        return

    df = pd.DataFrame(candidates).sort_values("val_rmsle")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest: {best['run']} (val_rmsle={best['val_rmsle']})")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["path"], output)
    print(f"Copied {best['path']} → {output}")


if __name__ == "__main__":
    main()
