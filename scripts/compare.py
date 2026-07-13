"""Compare all experiment runs in a table.

Reads metrics.json from each run directory under outputs/runs/.

Usage:
    uv run python scripts/compare.py
    uv run python scripts/compare.py --sort-by val_rmsle
    uv run python scripts/compare.py --scope full
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
    parser.add_argument(
        "--scope",
        type=str,
        default=None,
        help="Filter by run_scope (e.g. 'full', 'smoke'; default: show all)",
    )
    return parser.parse_args()


def _load_runs(runs_dir: Path, scope: str | None = None) -> list[dict]:
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
        run_scope = params.get("run_scope")
        if scope is not None and run_scope is not None and run_scope != scope:
            continue
        submissions_exist = (run_dir / "submission.csv").exists()

        rows.append(
            {
                "run": run_dir.name,
                "val_rmsle": meta.get("val_rmsle"),
                "n_features": meta.get("n_features"),
                "model_type": params.get("model_type"),
                "run_scope": run_scope or "?",
                "elapsed_seconds": round(data.get("elapsed_seconds", 0), 1),
                "submission": "✓" if submissions_exist else "—",
                "path": str(run_dir),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    rows = _load_runs(runs_dir, scope=args.scope)

    if not rows:
        filter_msg = f" (scope={args.scope})" if args.scope else ""
        print(f"No experiment runs found{filter_msg}.")
        return

    df = pd.DataFrame(rows)
    if args.sort_by and args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=args.ascending, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    pd.set_option("display.colheader_justify", "right")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
