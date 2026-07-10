"""Render pipeline visualisation GIF for README embedding.

Usage:
    uv run python scripts/viz_gif.py
    uv run python scripts/viz_gif.py --run outputs/runs/20260707_093203_bench-log-ridge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from store_sales.viz import DEFAULT_OUT_DIR, DEFAULT_RUNS_DIR, render_pipeline_viz

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render pipeline GIF visualisation.")
    parser.add_argument(
        "--run",
        type=str,
        action="append",
        default=None,
        help="Run directory (repeatable; default: auto-discover best per model type)",
    )
    parser.add_argument("--runs-dir", type=str, default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--scope", type=str, default="full", help="Filter runs by run_scope")
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-duration", type=float, default=10.0)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF rendering")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(r) for r in args.run] if args.run else None
    paths = render_pipeline_viz(
        run_dirs=run_dirs,
        runs_dir=Path(args.runs_dir),
        out_dir=Path(args.out_dir),
        scope=args.scope,
        gif=not args.no_gif,
        gif_fps=args.gif_fps,
        gif_duration=args.gif_duration,
    )
    for kind, path in paths.items():
        logger.info("viz: %s -> %s", kind, path)


if __name__ == "__main__":
    main()
