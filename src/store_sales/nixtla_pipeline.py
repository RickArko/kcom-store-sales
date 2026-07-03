"""Nixtla-based forecasting pipeline for Store Sales.

Two model families, both config-driven:

* ``stats`` — :class:`statsforecast.StatsForecast` statistical baselines
  (SeasonalNaive, Theta, AutoETS, ...). One column per model in the output.
* ``lgbm`` — :class:`mlforecast.MLForecast` with a LightGBM regressor,
  lags, rolling transforms, date features, and static/exogenous features.

Data flow (mirrors the M5 nixtla pipeline):

    load_data() → merge_tables() → to_long()           # unique_id, ds, y
        → build_exogenous()                            # oil/holiday/transactions
        → build forecaster from config (stats | lgbm)
        → fit → predict(h=16)  OR  cross_validation(h, n_windows)
        → rmsle_for_models() over the wide CV frame
        → to_submission()  (long id, sales)

All Nixtla packages are an optional extra (``uv sync --extra nixtla``).
Importing this module without them raises a clear ``ImportError``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from store_sales.metrics import rmsle

logger = logging.getLogger(__name__)

try:
    from statsforecast import StatsForecast
    from statsforecast.models import (
        AutoETS,
        AutoTheta,
        Naive,
        SeasonalNaive,
        Theta,
    )

    _STATS_OK = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _STATS_OK = False

try:
    from mlforecast import MLForecast
    from mlforecast.lag_transforms import RollingMean
    from mlforecast.target_transforms import Differences

    _ML_OK = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _ML_OK = False


_ID_COL = "unique_id"
_TIME_COL = "ds"
_TARGET_COL = "y"
_DEFAULT_SEASON = 7
_DEFAULT_FREQ = "D"
_DEFAULT_HORIZON = 16

_STATS_REGISTRY: dict[str, Any] = {}


def _stats_models() -> dict[str, Any]:
    if not _STATS_OK:
        return {}
    if not _STATS_REGISTRY:
        _STATS_REGISTRY.update(
            {
                "Naive": Naive,
                "SeasonalNaive": SeasonalNaive,
                "Theta": Theta,
                "AutoTheta": AutoTheta,
                "AutoETS": AutoETS,
            }
        )
    return _STATS_REGISTRY


# ---------------------------------------------------------------------------
# Data shaping
# ---------------------------------------------------------------------------


def to_long(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    store_col: str = "store_nbr",
    family_col: str = "family",
    date_col: str = "date",
    target_col: str = "sales",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape merged train/test to Nixtla long format.

    Returns ``(history, future)`` where ``history`` has the target column ``y``
    and ``future`` carries the test rows (target NaN) with exogenous features.
    Both share ``unique_id = "{store}_{family}"`` and ``ds`` (datetime64[ns]).
    """
    history = train.copy()
    future = test.copy()

    for df in (history, future):
        df[date_col] = pd.to_datetime(df[date_col])
        df[_ID_COL] = df[store_col].astype(str) + "_" + df[family_col].astype(str)

    history = history.rename(columns={date_col: _TIME_COL, target_col: _TARGET_COL})
    future = future.rename(columns={date_col: _TIME_COL})

    keep_hist = [_ID_COL, _TIME_COL, _TARGET_COL, store_col, family_col, "onpromotion"]
    keep_hist += [c for c in ("dcoilwtico", "is_holiday", "transactions") if c in history.columns]
    keep_fut = [_ID_COL, _TIME_COL, store_col, family_col, "onpromotion"]
    keep_fut += [c for c in ("dcoilwtico", "is_holiday", "transactions") if c in future.columns]

    history = history[[c for c in keep_hist if c in history.columns]]
    future = future[[c for c in keep_fut if c in future.columns]]

    history = history.sort_values([_ID_COL, _TIME_COL]).reset_index(drop=True)
    future = future.sort_values([_ID_COL, _TIME_COL]).reset_index(drop=True)
    return history, future


def build_static_features(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    store_col: str = "store_nbr",
    family_col: str = "family",
    extra_static: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mark per-series static columns and coerce categoricals for LightGBM.

    Static features are constant within a ``unique_id`` (store/family metadata).
    LightGBM rejects ``object``/``str`` dtypes, so coerce to ``category``.
    """
    static = [store_col, family_col]
    if extra_static:
        static += [c for c in extra_static if c in history.columns]
    static = list(dict.fromkeys(static))

    for df in (history, future):
        for col in static:
            if col in df.columns and str(df[col].dtype) in ("object", "str", "string"):
                df[col] = df[col].astype("category")
    return history, future


def build_future_exogenous(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    """Build the ``X_df`` mlforecast needs for ``predict(h, X_df=...)``.

    Carries exogenous time-varying columns (oil, holidays, onpromotion) for the
    forecast horizon, indexed by ``unique_id`` × ``ds``. Static columns stay on
    the history frame (passed via ``static_features=``).
    """
    drop = [_TARGET_COL]
    exog_cols = [
        c
        for c in future.columns
        if c not in (_ID_COL, _TIME_COL, *drop)
        and c in ("onpromotion", "dcoilwtico", "is_holiday", "transactions")
    ]
    X_df = future[[_ID_COL, _TIME_COL, *exog_cols]].copy()
    for col in exog_cols:
        if str(X_df[col].dtype) in ("object", "str", "string"):
            X_df[col] = X_df[col].astype("category")
    return X_df


# ---------------------------------------------------------------------------
# Forecaster construction (config-driven)
# ---------------------------------------------------------------------------


def build_stats_forecaster(
    model_cfg: dict[str, Any],
    *,
    freq: str = _DEFAULT_FREQ,
    season_length: int = _DEFAULT_SEASON,
    n_jobs: int = 1,
) -> StatsForecast:
    """Build a :class:`StatsForecast` from a config dict.

    Expected ``model_cfg`` keys::

        models: [{name: SeasonalNaive, alias: SeasonalNaive}, ...]
        season_length: 7
        n_jobs: 1
    """
    if not _STATS_OK:
        raise ImportError("statsforecast is not installed. Run `uv sync --extra nixtla`.")
    registry = _stats_models()
    specs = model_cfg.get("models", [{"name": "SeasonalNaive"}])
    models = []
    for spec in specs:
        name = spec["name"]
        if name not in registry:
            raise ValueError(f"Unknown stats model: {name}. Available: {list(registry)}")
        kwargs = {k: v for k, v in spec.items() if k not in ("name", "alias")}
        kwargs.setdefault("season_length", season_length)
        alias = spec.get("alias", name)
        models.append(registry[name](**{**kwargs, "alias": alias}))
    return StatsForecast(models=models, freq=freq, n_jobs=n_jobs)


def build_lgbm_forecaster(
    model_cfg: dict[str, Any],
    *,
    freq: str = _DEFAULT_FREQ,
    n_jobs: int = -1,
    seed: int = 42,
) -> MLForecast:
    """Build an :class:`MLForecast` (LightGBM) from a config dict.

    Expected ``model_cfg`` keys::

        params: {n_estimators: 500, learning_rate: 0.05, ...}
        lags: [1, 7, 14, 28]
        rolling_means_lagged: {1: [7, 28]}   # RollingMean on lagged value
        differences: []                       # target transforms
        date_features: [dayofweek, month, ...]
    """
    if not _ML_OK:
        raise ImportError("mlforecast is not installed. Run `uv sync --extra nixtla`.")
    import lightgbm as lgb

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    params.update(model_cfg.get("params", {}))
    params["seed"] = seed

    lags = model_cfg.get("lags", [7, 14, 28])
    rolling = model_cfg.get("rolling_means_lagged", {})
    lag_transforms: dict[int, list[Any]] = {
        int(lag): [RollingMean(window_size=int(w)) for w in windows]
        for lag, windows in rolling.items()
    }
    target_transforms: list[Any] | None = None
    diffs = model_cfg.get("differences", [])
    if diffs:
        target_transforms = [Differences(diffs)]

    date_features = model_cfg.get("date_features", ["dayofweek", "month", "day", "year"])

    model = lgb.LGBMRegressor(**params, n_jobs=n_jobs)
    return MLForecast(
        models={"LGBM": model},
        freq=freq,
        lags=lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
        target_transforms=target_transforms,
        num_threads=n_jobs,
    )


# ---------------------------------------------------------------------------
# Fit / predict / CV
# ---------------------------------------------------------------------------


def fit_predict(
    history: pd.DataFrame,
    future: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    horizon: int = _DEFAULT_HORIZON,
    freq: str = _DEFAULT_FREQ,
    season_length: int = _DEFAULT_SEASON,
    static_cols: list[str] | None = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Fit a forecaster from ``cfg`` and emit future predictions.

    Returns a long frame ``[unique_id, ds, <model_cols>]`` with one row per
    series per future step.
    """
    kind = cfg.get("kind", "stats")
    base_cols = [_ID_COL, _TIME_COL, _TARGET_COL]

    if kind == "stats":
        sf = build_stats_forecaster(cfg, freq=freq, season_length=season_length, n_jobs=n_jobs)
        sf.fit(df=history[base_cols])
        return sf.predict(h=horizon)

    if kind == "lgbm":
        history, future = build_static_features(history, future, extra_static=static_cols)
        X_df = build_future_exogenous(history, future)
        keep = base_cols + [c for c in static_cols or [] if c in history.columns]
        keep += [
            c
            for c in ("onpromotion", "dcoilwtico", "is_holiday", "transactions")
            if c in history.columns
        ]
        fcst = build_lgbm_forecaster(cfg, freq=freq, n_jobs=n_jobs, seed=seed)
        present_static = [c for c in (static_cols or []) if c in history.columns]
        fcst.fit(history[keep], static_features=present_static)
        return fcst.predict(h=horizon, X_df=X_df)

    raise ValueError(f"Unknown model kind: {kind!r} (expected 'stats' or 'lgbm')")


def cross_validate(
    history: pd.DataFrame,
    cfg: dict[str, Any],
    *,
    horizon: int = _DEFAULT_HORIZON,
    n_windows: int = 1,
    step_size: int | None = None,
    freq: str = _DEFAULT_FREQ,
    season_length: int = _DEFAULT_SEASON,
    static_cols: list[str] | None = None,
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Run rolling-origin CV. Returns wide frame ``[unique_id, ds, cutoff, y, <model_cols>]``."""
    kind = cfg.get("kind", "stats")
    step = step_size or horizon
    base_cols = [_ID_COL, _TIME_COL, _TARGET_COL]

    if kind == "stats":
        sf = build_stats_forecaster(cfg, freq=freq, season_length=season_length, n_jobs=n_jobs)
        return sf.cross_validation(
            df=history[base_cols], h=horizon, n_windows=n_windows, step_size=step
        )

    if kind == "lgbm":
        # mlforecast cross_validation needs exogenous future features per window.
        # For pure-history CV the dynamic exogenous columns (oil/holiday) are in
        # the history frame; mlforecast re-uses them as future X_df automatically
        # when they are present in the input.
        history, _ = build_static_features(history, history, extra_static=static_cols)
        keep = base_cols + [c for c in static_cols or [] if c in history.columns]
        keep += [
            c
            for c in ("onpromotion", "dcoilwtico", "is_holiday", "transactions")
            if c in history.columns
        ]
        fcst = build_lgbm_forecaster(cfg, freq=freq, n_jobs=n_jobs, seed=seed)
        present_static = [c for c in (static_cols or []) if c in history.columns]
        return fcst.cross_validation(
            df=history[keep],
            h=horizon,
            n_windows=n_windows,
            step_size=step,
            static_features=present_static,
        )

    raise ValueError(f"Unknown model kind: {kind!r} (expected 'stats' or 'lgbm')")


# ---------------------------------------------------------------------------
# Scoring + submission
# ---------------------------------------------------------------------------


def rmsle_for_models(cv_df: pd.DataFrame) -> pd.Series:
    """Score every model column in a wide Nixtla CV frame with RMSLE.

    ``cv_df`` must have ``y`` and one+ model columns. Negatives clamped to 0.
    Returns a ``pd.Series`` sorted ascending (lower is better).
    """
    model_cols = [c for c in cv_df.columns if c not in (_ID_COL, _TIME_COL, "cutoff", "y")]
    scores = {}
    y_true = cv_df["y"].to_numpy()
    for col in model_cols:
        scores[col] = rmsle(y_true, cv_df[col].to_numpy())
    return pd.Series(scores, name="rmsle").sort_values()


def to_submission(
    future: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    id_col: str = "id",
    model_col: str | None = None,
) -> pd.DataFrame:
    """Pivot a Nixtla forecast back to the Kaggle submission layout.

    Result has columns ``[id, sales]`` sorted by ``id``. If the forecast has
    multiple model columns, ``model_col`` selects which to use (defaults to the
    last one, which for mlforecast is ``LGBM`` and for stats is the last model).
    """
    if model_col is None:
        model_cols = [c for c in forecast.columns if c not in (_ID_COL, _TIME_COL)]
        model_col = model_cols[-1]
    forecast = forecast.copy()
    forecast["sales"] = np.maximum(forecast[model_col], 0)
    future = future.copy()
    future[_TIME_COL] = pd.to_datetime(future[_TIME_COL])
    forecast[_TIME_COL] = pd.to_datetime(forecast[_TIME_COL])

    # future has unique_id + ds + id; join forecast's sales onto it.
    sub = future[[_ID_COL, _TIME_COL, id_col]].merge(
        forecast[[_ID_COL, _TIME_COL, "sales"]], on=[_ID_COL, _TIME_COL], how="left"
    )
    sub["sales"] = sub["sales"].fillna(0.0)
    return sub[[id_col, "sales"]].sort_values(id_col).reset_index(drop=True)
