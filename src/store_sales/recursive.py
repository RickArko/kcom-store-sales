"""Recursive multi-step forecasting for linear models.

A multi-step horizon (e.g. the competition's 16 test days) cannot be predicted
in one shot when the model uses lag features: the lag for test day *d* depends
on the (unknown) sales of day *d-1*.  Direct prediction feeds stale or zero
lags, so forecasts collapse toward zero.

``recursive_forecast`` predicts the horizon day-by-day instead, feeding each
day's prediction back as the lag/rolling source for subsequent days.  This
mirrors test-time conditions and is the standard way to roll autoregressive
features forward.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from store_sales.features import TimeSeriesFeatureEngineer

logger = logging.getLogger(__name__)

_AGG_FNS = {
    "mean": np.nanmean,
    "std": np.nanstd,
    "min": np.nanmin,
    "max": np.nanmax,
    "median": np.nanmedian,
}


def onehot_fit(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list], list[str]]:
    """One-hot encode category columns.

    Returns ``(df_encoded, known_cats, feature_names)`` so callers can align
    subsequent frames (val/test) to the same column space.
    """
    cat_cols = [c for c in df.columns if df[c].dtype.name == "category"]
    known_cats: dict[str, list] = {}
    for col in cat_cols:
        cats = sorted(df[col].cat.categories.tolist())
        known_cats[col] = cats
        df[col] = pd.Categorical(df[col], categories=cats)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    df = df.fillna(0)
    return df, known_cats, list(df.columns)


def onehot_align(
    df: pd.DataFrame,
    known_cats: dict[str, list],
    feature_names: list[str],
) -> pd.DataFrame:
    """One-hot encode ``df`` and align columns to ``feature_names``."""
    cat_cols = [c for c in df.columns if df[c].dtype.name == "category"]
    for col in cat_cols:
        cats = known_cats.get(col)
        if cats is not None:
            df[col] = pd.Categorical(df[col], categories=cats)
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
    for c in set(feature_names) - set(df.columns):
        df[c] = 0
    extra = set(df.columns) - set(feature_names)
    if extra:
        df = df.drop(columns=list(extra))
    df = df[feature_names]
    return df.fillna(0)


def _max_lookback(engineer: TimeSeriesFeatureEngineer) -> int:
    """Largest lag/window needed to compute horizon features."""
    best = 1
    for _col, lags in engineer.lag_config:
        best = max(best, max(lags))
    for rc in engineer.rolling_config:
        for w in rc.get("windows", []):
            best = max(best, w)
    return best


def recursive_forecast(
    model,
    engineer: TimeSeriesFeatureEngineer,
    history: pd.DataFrame,
    horizon: pd.DataFrame,
    *,
    lag_target_col: str,
    target_transform: str,
    known_cats: dict[str, list],
    feature_names: list[str],
    scaler=None,
) -> np.ndarray:
    """Recursively forecast ``horizon`` rows, updating lag/rolling from predictions.

    Parameters
    ----------
    model : fitted estimator
        Predicts in target space (``log1p`` values when ``target_transform``
        is ``"log1p"``).
    engineer : TimeSeriesFeatureEngineer
        Supplies ``lag_config`` / ``rolling_config`` and the ``transform``
        step (date features, categoricals, column drops).
    history : pd.DataFrame
        Training rows with the actual target column (e.g. ``log_sales``).
        Only the last ``max(lags, windows)`` days per group are retained —
        older rows do not influence any horizon feature.
    horizon : pd.DataFrame
        Rows to forecast (validation or test).  Same columns as ``history``;
        the target column values are ignored (overwritten with predictions).
    lag_target_col : str
        Column that lags/rolling reference (``"log_sales"`` or ``"sales"``).
    target_transform : str
        ``"log1p"`` or ``"raw"``.  Predictions are decoded to raw sales space.
    known_cats, feature_names : one-hot alignment from ``onehot_fit``.
    scaler : optional
        A fitted ``StandardScaler`` (or similar).  When provided, each day's
        feature matrix is scaled before ``model.predict``.

    Returns
    -------
    np.ndarray
        Predictions in raw sales space (clamped >= 0), aligned to ``horizon``
        row order.
    """
    group_cols = [engineer.store_col, engineer.family_col]
    date_col = engineer.date_col
    lookback = _max_lookback(engineer)

    # --- Truncate history to the recent lookback window per group ----------
    hist = history.copy()
    hist["_is_horizon"] = False
    if lookback > 0 and date_col in hist.columns:
        max_date = hist[date_col].max()
        cutoff = max_date - pd.Timedelta(days=lookback)
        hist = hist[hist[date_col] >= cutoff]

    hor = horizon.copy()
    hor["_is_horizon"] = True
    hor["_orig_order"] = np.arange(len(hor))

    combined = pd.concat([hist, hor], axis=0, ignore_index=True)
    combined = combined.sort_values(group_cols + [date_col]).reset_index(drop=True)

    combined.loc[combined["_is_horizon"], lag_target_col] = np.nan

    # --- Precompute arrays for vectorized lag/rolling updates -------------
    group_id = combined.groupby(group_cols, sort=False).ngroup().to_numpy()
    target = combined[lag_target_col].to_numpy(dtype=float)
    horizon_mask = combined["_is_horizon"].to_numpy()

    lag_specs = [(col, lag) for col, lags in engineer.lag_config for lag in lags]
    roll_specs = [
        (rc["col"], w, agg)
        for rc in engineer.rolling_config
        for w in rc.get("windows", [])
        for agg in rc.get("aggs", ["mean"])
    ]

    horizon_dates = sorted(combined.loc[combined["_is_horizon"], date_col].unique())
    dates_arr = combined[date_col].to_numpy()

    for date in horizon_dates:
        day_pos = np.where((dates_arr == np.datetime64(date)) & horizon_mask)[0]
        if day_pos.size == 0:
            continue

        # --- Lags (vectorized with group-boundary guard) ------------------
        for col, lag in lag_specs:
            src = day_pos - lag
            valid = src >= 0
            src_clipped = np.maximum(src, 0)
            valid[valid] = group_id[src_clipped[valid]] == group_id[day_pos[valid]]
            vals = np.where(valid, target[src_clipped], np.nan)
            col_name = f"{col}_lag_{lag}"
            if col_name in combined.columns:
                combined.iloc[day_pos, combined.columns.get_loc(col_name)] = vals

        # --- Rolling (vectorized window extraction + group mask) ----------
        for col, w, agg in roll_specs:
            offsets = np.arange(-w, 0)
            win_idx = day_pos[:, None] + offsets[None, :]
            win_vals = target[win_idx]
            same_group = group_id[win_idx] == group_id[day_pos, None]
            win_vals = np.where(same_group, win_vals, np.nan)
            fn = _AGG_FNS.get(agg, np.nanmean)
            with np.errstate(all="ignore"):
                vals = fn(win_vals, axis=1)
            vals = np.where(np.isnan(vals), 0.0, vals)
            col_name = f"{col}_roll_{w}_{agg}"
            if col_name in combined.columns:
                combined.iloc[day_pos, combined.columns.get_loc(col_name)] = vals

        # --- Predict ------------------------------------------------------
        day_df = combined.iloc[day_pos].copy()
        X_day = engineer.transform(day_df)
        X_day = onehot_align(X_day, known_cats, feature_names)
        X_input = scaler.transform(X_day) if scaler is not None else X_day
        preds = np.asarray(model.predict(X_input), dtype=float)

        # Store predictions in target space (log1p stays in log) for future lags.
        target[day_pos] = preds
        combined.iloc[day_pos, combined.columns.get_loc(lag_target_col)] = preds

    # --- Reassemble predictions in the caller's horizon order -------------
    hor_result = combined[combined["_is_horizon"]].sort_values("_orig_order")
    preds_target = hor_result[lag_target_col].to_numpy(dtype=float)

    if target_transform == "log1p":
        preds = np.expm1(preds_target)
    else:
        preds = preds_target
    return np.maximum(preds, 0)
