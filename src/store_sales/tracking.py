from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@contextmanager
def track_experiment(config: dict, run_name: str | None = None, base_dir: str = "outputs/runs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "run"
    run_dir = Path(base_dir) / f"{timestamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    submission_path = run_dir / "submission.csv"

    run = _RunTracker(run_dir, models_dir, submission_path)
    try:
        yield run
    finally:
        run.save()


class _RunTracker:
    def __init__(self, run_dir: Path, models_dir: Path, submission_path: Path):
        self.run_dir = run_dir
        self.models_dir = models_dir
        self.submission_path = submission_path
        self._metrics: dict[str, Any] = {}
        self._params: dict[str, Any] = {}
        self._start_time = time.time()
        self._saved = False

    def log_metric(self, key: str, value: float) -> None:
        self._metrics[key] = value

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._metrics.update(metrics)

    def log_params(self, params: dict[str, Any]) -> None:
        self._params.update(params)

    def save(self) -> None:
        if self._saved:
            return
        elapsed = time.time() - self._start_time
        data = {
            "elapsed_seconds": round(elapsed, 2),
            "params": self._params,
            "metrics": self._metrics,
        }
        with open(self.run_dir / "metrics.json", "w") as f:
            json.dump(data, f, indent=2, default=str)
        self._saved = True
