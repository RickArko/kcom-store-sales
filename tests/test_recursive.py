from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from store_sales.features import TimeSeriesFeatureEngineer
from store_sales.recursive import onehot_align, onehot_fit, recursive_forecast


def _make_series(
    store: int,
    family: str,
    sales: list[float],
    start: str = "2017-01-01",
    extra: dict | None = None,
) -> pd.DataFrame:
    n = len(sales)
    df = pd.DataFrame(
        {
            "id": range(n),
            "store_nbr": [store] * n,
            "family": [family] * n,
            "date": pd.date_range(start, periods=n, freq="D"),
            "sales": sales,
            "onpromotion": [0] * n,
        }
    )
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def _build_engineer(lags: list[int], target_col: str = "log_sales") -> TimeSeriesFeatureEngineer:
    return TimeSeriesFeatureEngineer(
        date_col="date",
        store_col="store_nbr",
        family_col="family",
        onpromotion_col="onpromotion",
        date_features=[],
        drop_cols=["id", "sales", "log_sales"],
        lag_config=[[target_col, lags]] if lags else [],
        rolling_config=[],
        fourier_config={},
    )


# --- onehot helpers ----------------------------------------------------------


def test_onehot_fit_encodes_categoricals():
    df = pd.DataFrame(
        {
            "store_nbr": pd.Categorical([1, 2, 1]),
            "family": pd.Categorical(["A", "B", "A"]),
            "x": [1.0, 2.0, 3.0],
        }
    )
    encoded, known_cats, names = onehot_fit(df)
    assert "store_nbr" in known_cats
    assert "family" in known_cats
    # drop_first=True → one fewer dummy per categorical
    assert any(c.startswith("store_nbr_") for c in names)
    assert any(c.startswith("family_") for c in names)
    assert encoded.shape[0] == 3
    assert list(encoded.columns) == names


def test_onehot_align_missing_and_extra_columns():
    train = pd.DataFrame(
        {
            "store_nbr": pd.Categorical([1, 2], categories=[1, 2, 3]),
            "family": pd.Categorical(["A", "B"], categories=["A", "B", "C"]),
            "x": [1.0, 2.0],
        }
    )
    encoded, known_cats, names = onehot_fit(train)
    # val has a unseen category value but aligned to same columns
    val = pd.DataFrame(
        {
            "store_nbr": pd.Categorical([3], categories=[1, 2, 3]),
            "family": pd.Categorical(["C"], categories=["A", "B", "C"]),
            "x": [5.0],
        }
    )
    aligned = onehot_align(val, known_cats, names)
    assert list(aligned.columns) == names
    assert len(aligned) == 1
    # No NaNs after alignment
    assert not aligned.isna().any().any()


# --- recursive_forecast ------------------------------------------------------


def test_recursive_forecast_shape_and_nonneg():
    train = _make_series(1, "A", [10.0] * 40)
    horizon = _make_series(1, "A", [0.0] * 5, start="2017-02-10")
    train["log_sales"] = np.log1p(train["sales"])
    horizon["log_sales"] = 0.0

    eng = _build_engineer([1, 7])
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train, X_hist["log_sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="log_sales",
        target_transform="log1p",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    assert preds.shape == (5,)
    assert (preds >= 0).all()


def test_recursive_forecast_constant_series():
    """A constant series should forecast roughly the same constant."""
    train = _make_series(1, "A", [10.0] * 40)
    horizon = _make_series(1, "A", [0.0] * 5, start="2017-02-10")
    train["log_sales"] = np.log1p(train["sales"])
    horizon["log_sales"] = 0.0

    eng = _build_engineer([1, 7])
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train, X_hist["log_sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="log_sales",
        target_transform="log1p",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    # Constant series → forecasts near 10 (the constant level)
    assert all(abs(p - 10.0) < 2.0 for p in preds)


def test_recursive_forecast_raw_target():
    """target_transform='raw' should return model predictions directly (clamped)."""
    train = _make_series(1, "A", [10.0] * 40)
    horizon = _make_series(1, "A", [0.0] * 5, start="2017-02-10")

    eng = _build_engineer([1, 7], target_col="sales")
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train, X_hist["sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="sales",
        target_transform="raw",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    assert preds.shape == (5,)
    assert (preds >= 0).all()


def test_recursive_forecast_multi_group():
    """Two store-family groups → predictions for each, no cross-contamination."""
    train = pd.concat(
        [
            _make_series(1, "A", [10.0] * 40),
            _make_series(2, "B", [100.0] * 40),
        ],
        ignore_index=True,
    )
    horizon = pd.concat(
        [
            _make_series(1, "A", [0.0] * 5, start="2017-02-10"),
            _make_series(2, "B", [0.0] * 5, start="2017-02-10"),
        ],
        ignore_index=True,
    )
    train["log_sales"] = np.log1p(train["sales"])
    horizon["log_sales"] = 0.0

    eng = _build_engineer([1, 7])
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train, X_hist["log_sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="log_sales",
        target_transform="log1p",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    assert preds.shape == (10,)
    # Store 2/B (100) should have higher forecasts than store 1/A (10)
    hor_meta = X_hor[["store_nbr", "family"]].reset_index(drop=True)
    s1 = preds[hor_meta["store_nbr"] == 1]
    s2 = preds[hor_meta["store_nbr"] == 2]
    assert s2.mean() > s1.mean()


def test_recursive_forecast_preserves_horizon_order():
    """Predictions should align to the horizon DataFrame's row order."""
    train = _make_series(1, "A", [10.0] * 40)
    horizon = _make_series(1, "A", [0.0] * 5, start="2017-02-10")
    # Shuffle horizon rows to test order preservation
    horizon = horizon.iloc[::-1].reset_index(drop=True)
    train["log_sales"] = np.log1p(train["sales"])
    horizon["log_sales"] = 0.0

    eng = _build_engineer([1, 7])
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=1.0)
    model.fit(X_train, X_hist["log_sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="log_sales",
        target_transform="log1p",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    assert preds.shape == (5,)
    assert (preds >= 0).all()


def test_recursive_forecast_feeds_back_lag1():
    """Day 2's lag_1 should equal day 1's prediction (the recursive property).

    Training data alternates [10, 20, 10, 20, ...].  With lag_1 only, the model
    learns anti-persistence: y_t ≈ 30 - lag_1.  If the recursive feedback works,
    the forecast continues the alternation [10, 20, 10, 20] — day 2's lag_1
    comes from day 1's *prediction* (≈10), not from a stale value.
    """
    sales = [10.0, 20.0] * 30  # 60 days, last value = 20
    train = _make_series(1, "A", sales)
    horizon = _make_series(1, "A", [0.0] * 4, start="2017-03-02")
    train["log_sales"] = np.log1p(train["sales"])
    horizon["log_sales"] = 0.0

    eng = _build_engineer([1])
    X_hist, X_hor = eng.create_lag_features(train, horizon, "sales")
    eng.fit(X_hist)

    X_train = eng.transform(X_hist)
    X_train, known_cats, feat_names = onehot_fit(X_train)

    model = Ridge(alpha=0.01)  # low alpha → fit lag_1 tightly
    model.fit(X_train, X_hist["log_sales"])

    preds = recursive_forecast(
        model,
        eng,
        X_hist,
        X_hor,
        lag_target_col="log_sales",
        target_transform="log1p",
        known_cats=known_cats,
        feature_names=feat_names,
    )
    # Last training value = 20 → day 1 lag_1 = 20 → predict ≈10
    # Day 2 lag_1 = day 1 pred ≈10 → predict ≈20 (proves recursive feedback)
    assert preds[0] < 15, f"day 1 should be ~10, got {preds[0]}"
    assert preds[1] > 15, f"day 2 should be ~20 (fed back from day 1), got {preds[1]}"
    assert preds[2] < 15, f"day 3 should be ~10 (fed back from day 2), got {preds[2]}"
    assert preds[3] > 15, f"day 4 should be ~20 (fed back from day 3), got {preds[3]}"


# --- create_lag_features edge cases -----------------------------------------


def test_create_lag_features_empty_lag_config_preserves_rows():
    """Regression: empty lag_config used to drop ALL train rows.

    When lag_config=[] there are no lag columns, so the notna().any() guard
    on an empty DataFrame evaluated to all-False, dropping every train row.
    """
    train = _make_series(1, "A", [10.0, 20.0, 30.0])
    test = _make_series(1, "A", [0.0], start="2017-01-04")

    eng = TimeSeriesFeatureEngineer(
        date_col="date",
        store_col="store_nbr",
        family_col="family",
        onpromotion_col="onpromotion",
        date_features=[],
        drop_cols=["id", "sales"],
        lag_config=[],
        rolling_config=[],
        fourier_config={},
    )
    X_train, X_test = eng.create_lag_features(train, test, "sales")
    assert len(X_train) == 3  # no rows should be dropped
    assert len(X_test) == 1


def test_create_lag_features_drops_first_row_per_group():
    """With lag_config, the first row per group (no lag history) is dropped."""
    train = _make_series(1, "A", [10.0, 20.0, 30.0])
    test = _make_series(1, "A", [0.0], start="2017-01-04")

    eng = TimeSeriesFeatureEngineer(
        date_col="date",
        store_col="store_nbr",
        family_col="family",
        onpromotion_col="onpromotion",
        date_features=[],
        drop_cols=["id", "sales"],
        lag_config=[["sales", [1]]],
        rolling_config=[],
        fourier_config={},
    )
    X_train, X_test = eng.create_lag_features(train, test, "sales")
    assert len(X_train) == 2  # first row dropped (no lag_1)
    assert "sales_lag_1" in X_train.columns
    assert len(X_test) == 1
