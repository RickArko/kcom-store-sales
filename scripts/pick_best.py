"""Pick the best-submission run by val_rmsle.

Scans run directories, filters to full-dataset runs (run_scope=full),
and copies the lowest-RMSLE submission to outputs/submissions/submission.csv.

Usage:
    uv run python scripts/pick_best.py
    uv run python scripts/pick_best.py --runs-dir outputs/runs
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick best submission by val_rmsle.")
    parser.add_argument("--runs-dir", type=str, default="outputs/runs")
    parser.add_argument("--output", type=str, default="outputs/submissions/submission.csv")
    parser.add_argument(
        "--scope",
        type=str,
        default="full",
        help="Filter by run_scope (default: full). Set to 'all' to skip filtering.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Filter by model_type (e.g. 'toto', 'lightgbm'; default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    output = Path(args.output)

    candidates = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "metrics.json"
        sub_path = run_dir / "submission.csv"
        if not metrics_path.exists() or not sub_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        params = data.get("params", {})
        run_scope = params.get("run_scope", None)
        if args.scope != "all" and run_scope is not None and run_scope != args.scope:
            continue
        model_type = params.get("model_type", "?")
        if args.model_type is not None and model_type != args.model_type:
            continue
        rmsle = data.get("metrics", {}).get("val_rmsle")
        if rmsle is None:
            continue
        candidates.append(
            {
                "run": run_dir.name,
                "val_rmsle": rmsle,
                "run_scope": run_scope or "?",
                "model_type": params.get("model_type", "?"),
                "path": str(sub_path),
            }
        )

    if not candidates:
        print(f"No runs with scope='{args.scope}' and val_rmsle found.")
        return

    df = pd.DataFrame(candidates).sort_values("val_rmsle")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    print("All candidates (sorted by val_rmsle):")
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest: {best['run']}  (val_rmsle={best['val_rmsle']})")

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["path"], output)
    print(f"Copied → {output}")
    print(f"Submit via: make submit SUBMISSION_FILE={output}")


if __name__ == "__main__":
    main()
