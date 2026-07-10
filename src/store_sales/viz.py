"""Animated pipeline visualisation for Store Sales forecasting.

Reads completed experiment runs and renders an animated GIF that walks through
feature engineering, validation predictions on a hero series, and a model
comparison bar chart (RMSLE).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from store_sales.data import extract_holiday_dates, load_data, merge_tables
from store_sales.inference import predict_from_run
from store_sales.metrics import rmsle

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "assets"
DEFAULT_RUNS_DIR = REPO_ROOT / "outputs" / "runs"

_COLORS = {
    "linear": "#58a6ff",
    "lightgbm": "#f778ba",
    "nixtla": "#d29922",
    "toto": "#a371f7",
    "other": "#8b949e",
}
_BG = "#0d1117"
_TEXT = "#f0f6fc"
_MUTED = "#8b949e"
_GRID = "#21262d"
_AXIS = "#30363d"
_TRAIN = "#c9d1d9"
_ACTUAL = "#58a6ff"
_PRED = "#f778ba"
_TEST = "#d62728"


@dataclass
class ModelRun:
    """One completed experiment run."""

    path: Path
    name: str
    model_type: str
    val_rmsle: float
    label: str


@dataclass
class HeroSeries:
    """Actual vs predicted for one store-family series."""

    store_nbr: int
    family: str
    dates: list[pd.Timestamp]
    actual: list[float]
    predicted: list[float]
    split: list[str]  # "train", "val", or "test"
    val_rmsle: float


@dataclass
class VizPayload:
    """Everything the GIF renderer needs."""

    hero: HeroSeries
    models: list[ModelRun] = field(default_factory=list)
    val_days: int = 16
    n_series: int = 0


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def discover_runs(
    runs_dir: Path = DEFAULT_RUNS_DIR,
    *,
    scope: str | None = "full",
    prefer: list[str] | None = None,
) -> list[Path]:
    """Return run directories, optionally filtered by scope and name preference."""
    if not runs_dir.exists():
        return []

    candidates: list[tuple[str, Path, float]] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "sandbox":
            continue
        metrics_path = run_dir / "metrics.json"
        model_path = run_dir / "models" / "model.joblib"
        if not metrics_path.exists() or not model_path.exists():
            continue
        with open(metrics_path) as f:
            data = json.load(f)
        run_scope = data.get("params", {}).get("run_scope")
        if scope is not None and run_scope is not None and run_scope != scope:
            continue
        val_rmsle = data.get("metrics", {}).get("val_rmsle")
        if val_rmsle is None:
            continue
        candidates.append((run_dir.name, run_dir, float(val_rmsle)))

    if prefer:
        picked: list[Path] = []
        used: set[str] = set()
        for pattern in prefer:
            matches = [
                (name, path, score)
                for name, path, score in candidates
                if pattern in name
                and name not in used
                and not (pattern == "log-ridge" and "+fourier" in name)
            ]
            if matches:
                best = min(matches, key=lambda x: x[2])
                picked.append(best[1])
                used.add(best[0])
        if picked:
            return picked

    # Default: best run per model_type
    by_type: dict[str, tuple[str, Path, float]] = {}
    for name, path, score in candidates:
        with open(path / "metrics.json") as f:
            model_type = json.load(f).get("params", {}).get("model_type", "other")
        prev = by_type.get(model_type)
        if prev is None or score < prev[2]:
            by_type[model_type] = (name, path, score)
    return [path for _, path, _ in sorted(by_type.values(), key=lambda x: x[2])]


def _model_label(name: str, model_type: str) -> str:
    short = name.split("_", 1)[-1] if "_" in name else name
    short = short.replace("bench-", "").replace("+", "+")
    return f"{model_type}\n({short})"


def _load_model_runs(run_dirs: list[Path]) -> list[ModelRun]:
    runs: list[ModelRun] = []
    for path in run_dirs:
        with open(path / "metrics.json") as f:
            data = json.load(f)
        model_type = data.get("params", {}).get("model_type", "other")
        val_rmsle = float(data["metrics"]["val_rmsle"])
        runs.append(
            ModelRun(
                path=path,
                name=path.name,
                model_type=model_type,
                val_rmsle=val_rmsle,
                label=_model_label(path.name, model_type),
            )
        )
    return sorted(runs, key=lambda r: r.val_rmsle)


def _predict_series(
    run_dir: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    y_full: pd.Series,
    *,
    store_nbr: int,
    family: str,
    val_days: int,
    days_back: int = 120,
    holiday_dates: list[str] | None = None,
) -> HeroSeries:
    """Re-run inference for one store-family series."""
    combined = predict_from_run(
        run_dir, train, test, y_full, holiday_dates=holiday_dates
    )
    combined = combined[
        (combined["store_nbr"] == store_nbr) & (combined["family"] == family)
    ].sort_values("date")

    all_dates = sorted(train["date"].unique())
    val_start = pd.Timestamp(all_dates[-val_days])
    cutoff = val_start - pd.Timedelta(days=days_back)
    combined = combined[combined["date"] >= cutoff].copy()
    combined.loc[combined["date"] >= val_start, "split"] = combined.loc[
        combined["date"] >= val_start, "split"
    ].replace("train", "val")

    val_mask = combined["split"] == "val"
    val_score = rmsle(
        combined.loc[val_mask, "actual"].values,
        combined.loc[val_mask, "predicted"].values,
    )

    return HeroSeries(
        store_nbr=store_nbr,
        family=family,
        dates=[pd.Timestamp(d) for d in combined["date"]],
        actual=[float(v) for v in combined["actual"].fillna(0)],
        predicted=[float(v) for v in combined["predicted"]],
        split=list(combined["split"]),
        val_rmsle=val_score,
    )


def _pick_hero(train: pd.DataFrame) -> tuple[int, str]:
    """Pick the highest-total-sales store-family for visual interest."""
    totals = train.groupby(["store_nbr", "family"])["sales"].sum().sort_values(ascending=False)
    store_nbr, family = totals.index[0]
    return int(store_nbr), str(family)


def build_payload(
    *,
    run_dirs: list[Path] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    scope: str | None = "full",
    val_days: int = 16,
    days_back: int = 120,
) -> VizPayload:
    """Load data and runs, build the visualisation payload."""
    if run_dirs is None:
        run_dirs = discover_runs(
            runs_dir,
            scope=scope,
            prefer=["log-ridge", "lightgbm", "nixtla"],
        )
    if not run_dirs:
        raise FileNotFoundError(
            f"No completed runs found in {runs_dir}. "
            "Run `make train-linear` or `make benchmark` first."
        )

    logger.info("viz: using runs %s", [p.name for p in run_dirs])
    tables = load_data()
    train, test = merge_tables(tables)
    holiday_dates = extract_holiday_dates(tables)
    y_full = train["sales"].copy()
    store_nbr, family = _pick_hero(train)
    logger.info("viz: hero series store=%d family=%s", store_nbr, family)

    hero = _predict_series(
        run_dirs[0],
        train,
        test,
        y_full,
        store_nbr=store_nbr,
        family=family,
        val_days=val_days,
        days_back=days_back,
        holiday_dates=holiday_dates,
    )
    models = _load_model_runs(run_dirs)
    n_series = train.groupby(["store_nbr", "family"]).ngroups

    return VizPayload(
        hero=hero,
        models=models,
        val_days=val_days,
        n_series=n_series,
    )


def render_gif(
    payload: VizPayload,
    out_path: Path,
    *,
    fps: int = 12,
    duration: float = 10.0,
    width: int = 900,
    height: int = 500,
) -> Path:
    """Render an animated GIF using matplotlib + Pillow."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    hero = payload.hero
    dates = hero.dates
    actual = np.asarray(hero.actual, dtype=float)
    predicted = np.asarray(hero.predicted, dtype=float)
    splits = hero.split
    n = len(dates)

    val_idx = [i for i, s in enumerate(splits) if s == "val"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_start_i = val_idx[0] if val_idx else n

    x = np.arange(n)
    y_max = float(max(actual.max(), predicted.max(), 1.0)) * 1.15

    models = payload.models
    model_colors = [_COLORS.get(m.model_type, _COLORS["other"]) for m in models]
    rmsle_max = max(m.val_rmsle for m in models) * 1.25

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor=_BG)

    fig.text(
        0.5,
        0.95,
        "Store Sales — Time Series Forecasting",
        color=_TEXT,
        fontsize=14,
        weight=700,
        ha="center",
    )
    fig.text(
        0.5,
        0.90,
        f"{payload.n_series:,} store-family series · metric: RMSLE",
        color=_MUTED,
        fontsize=9,
        ha="center",
    )

    ax_ts = fig.add_axes((0.07, 0.32, 0.55, 0.52))
    ax_ts.set_facecolor(_BG)
    ax_ts.set_xlim(-0.5, n - 0.5)
    ax_ts.set_ylim(0, y_max)
    ax_ts.tick_params(colors=_MUTED, labelsize=7)
    for spine in ax_ts.spines.values():
        spine.set_color(_AXIS)
    ax_ts.grid(True, color=_GRID, alpha=0.6, linestyle="--", linewidth=0.6)
    ax_ts.set_title(
        f"Store {hero.store_nbr} — {hero.family}",
        color=_MUTED,
        fontsize=10,
        loc="left",
        pad=6,
    )

    cutoff_line = ax_ts.axvline(
        val_start_i - 0.5, color=_PRED, linestyle="--", linewidth=1.2, visible=False
    )
    (line_train,) = ax_ts.plot([], [], color=_TRAIN, linewidth=1.0, alpha=0.7, label="train actual")
    (line_val_act,) = ax_ts.plot([], [], color=_ACTUAL, linewidth=1.4, label="val actual")
    (line_val_pred,) = ax_ts.plot(
        [], [], color=_PRED, linewidth=1.4, linestyle="--", label="val predicted"
    )
    (line_test_pred,) = ax_ts.plot(
        [], [], color=_TEST, linewidth=1.4, linestyle="--", label="test forecast"
    )
    ax_ts.legend(loc="upper left", frameon=False, fontsize=7, labelcolor=_MUTED, ncol=2)

    ax_bar = fig.add_axes((0.68, 0.32, 0.28, 0.52))
    ax_bar.set_facecolor(_BG)
    ax_bar.set_title("Val RMSLE by model", color=_MUTED, fontsize=10, loc="left", pad=6)
    ax_bar.tick_params(colors=_MUTED, labelsize=7)
    for spine in ax_bar.spines.values():
        spine.set_color(_AXIS)
    ax_bar.grid(True, axis="y", color=_GRID, alpha=0.6, linestyle="--", linewidth=0.6)
    ax_bar.set_xlim(-0.6, len(models) - 0.4)
    ax_bar.set_ylim(0, rmsle_max)
    ax_bar.set_xticks(range(len(models)))
    ax_bar.set_xticklabels([m.model_type for m in models], fontsize=7)

    bars = []
    for i, c in enumerate(model_colors):
        b = ax_bar.bar(i, 0, color=c, width=0.55, edgecolor=c, linewidth=0.5, alpha=0.9)
        bars.append(b)
    val_labels = []
    for i, (m, c) in enumerate(zip(models, model_colors, strict=True)):
        lbl = ax_bar.text(
            i,
            0,
            f"{m.val_rmsle:.3f}",
            color=c,
            fontsize=7,
            ha="center",
            va="bottom",
            weight=600,
            visible=False,
        )
        val_labels.append(lbl)

    caption = fig.text(0.5, 0.12, "", color=_TEXT, fontsize=10, ha="center", va="center")

    n_frames = int(fps * duration)

    def _phase(t: float) -> str:
        if t < 0.15:
            return "load"
        if t < 0.35:
            return "train"
        if t < 0.55:
            return "val"
        if t < 0.70:
            return "test"
        if t < 0.85:
            return "score"
        return "compare"

    captions = {
        "load": "Merge stores, oil, holidays → engineer lag & rolling features",
        "train": "Fit global Ridge on log1p(sales) across all store-family series",
        "val": f"Validation (last {payload.val_days}d): actual vs predicted",
        "test": "Forecast 16-day test horizon (no labels)",
        "score": f"Hero series val RMSLE = {hero.val_rmsle:.4f}",
        "compare": "Model comparison — lower RMSLE is better",
    }

    def update(frame: int):
        t = frame / max(n_frames - 1, 1)
        phase = _phase(t)
        caption.set_text(captions[phase])

        # Time series panel
        if phase == "load":
            line_train.set_data([], [])
            line_val_act.set_data([], [])
            line_val_pred.set_data([], [])
            line_test_pred.set_data([], [])
            cutoff_line.set_visible(False)
        elif phase == "train":
            prog = _ease(min(1.0, (t - 0.15) / 0.20))
            end = train_idx[0] + int(prog * len(train_idx)) if train_idx else val_start_i
            end = max(1, min(end, val_start_i))
            line_train.set_data(x[:end], actual[:end])
            line_val_act.set_data([], [])
            line_val_pred.set_data([], [])
            line_test_pred.set_data([], [])
            cutoff_line.set_visible(False)
        elif phase in ("val", "score"):
            line_train.set_data(x[:val_start_i], actual[:val_start_i])
            prog = 1.0 if phase == "score" else _ease(min(1.0, (t - 0.35) / 0.20))
            end = val_start_i + int(prog * len(val_idx)) if val_idx else val_start_i
            end = max(val_start_i, min(end, n))
            line_val_act.set_data(x[val_start_i:end], actual[val_start_i:end])
            line_val_pred.set_data(x[val_start_i:end], predicted[val_start_i:end])
            line_test_pred.set_data([], [])
            cutoff_line.set_visible(True)
        elif phase == "test":
            line_train.set_data(x[:val_start_i], actual[:val_start_i])
            line_val_act.set_data(
                x[val_start_i : val_idx[-1] + 1], actual[val_start_i : val_idx[-1] + 1]
            )
            line_val_pred.set_data(
                x[val_start_i : val_idx[-1] + 1], predicted[val_start_i : val_idx[-1] + 1]
            )
            prog = _ease(min(1.0, (t - 0.55) / 0.15))
            t_end = (val_idx[-1] + 1 if val_idx else val_start_i) + int(prog * len(test_idx))
            t_end = max(val_start_i, min(t_end, n))
            line_test_pred.set_data(
                x[val_idx[-1] + 1 if val_idx else val_start_i : t_end],
                predicted[val_idx[-1] + 1 if val_idx else val_start_i : t_end],
            )
            cutoff_line.set_visible(True)
        else:  # compare
            line_train.set_data(x[:val_start_i], actual[:val_start_i])
            if val_idx:
                line_val_act.set_data(
                    x[val_start_i : val_idx[-1] + 1], actual[val_start_i : val_idx[-1] + 1]
                )
                line_val_pred.set_data(
                    x[val_start_i : val_idx[-1] + 1], predicted[val_start_i : val_idx[-1] + 1]
                )
            if test_idx:
                line_test_pred.set_data(x[test_idx[0] :], predicted[test_idx[0] :])
            cutoff_line.set_visible(True)

        # Bar chart panel
        bar_prog = 0.0 if phase != "compare" else _ease(min(1.0, (t - 0.85) / 0.15))
        for i, (m, bar, lbl) in enumerate(zip(models, bars, val_labels, strict=True)):
            h = m.val_rmsle * bar_prog
            bar[0].set_height(h)
            lbl.set_position((i, h))
            lbl.set_visible(bar_prog > 0.5)

        return (
            line_train,
            line_val_act,
            line_val_pred,
            line_test_pred,
            cutoff_line,
            caption,
            *bars,
            *val_labels,
        )

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path


def render_pipeline_viz(
    *,
    run_dirs: list[Path] | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    scope: str | None = "full",
    gif: bool = True,
    gif_fps: int = 12,
    gif_duration: float = 10.0,
) -> dict[str, Path]:
    """Build payload and render pipeline visualisation assets."""
    payload = build_payload(run_dirs=run_dirs, runs_dir=runs_dir, scope=scope)
    paths: dict[str, Path] = {}
    if gif:
        gif_path = out_dir / "pipeline.gif"
        render_gif(payload, gif_path, fps=gif_fps, duration=gif_duration)
        logger.info("viz: wrote %s (%.1f KB)", gif_path, gif_path.stat().st_size / 1024)
        paths["gif"] = gif_path
    return paths
