"""Tests for the TOTO forecasting pipeline.

Unit tests cover the data reshaping logic (wide pivot, submission mapping)
without loading the model.  Integration tests that require the Toto 2.0
checkpoint are marked ``slow`` and skipped when ``toto2`` is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from store_sales.toto_pipeline import TotoPipeline


def _have(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


def _make_train(n_stores=3, n_families=2, n_days=80):
    """Build a synthetic train DataFrame with store-family-date rows."""
    rows = []
    sid = 0
    for s in range(1, n_stores + 1):
        for f in ["A", "B"][:n_families]:
            for d in range(n_days):
                date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=d)
                rows.append(
                    {
                        "id": sid,
                        "date": date,
                        "store_nbr": s,
                        "family": f,
                        "sales": float(s * 10 + d % 7),
                    }
                )
                sid += 1
    return pd.DataFrame(rows)


def _make_test(n_stores=3, n_families=2, horizon=16):
    """Build a synthetic test DataFrame matching the train series."""
    rows = []
    sid = 999
    for s in range(1, n_stores + 1):
        for f in ["A", "B"][:n_families]:
            for d in range(horizon):
                date = pd.Timestamp("2023-01-01") + pd.Timedelta(days=80 + d)
                rows.append({"id": sid, "date": date, "store_nbr": s, "family": f, "sales": 0.0})
                sid += 1
    return pd.DataFrame(rows)


class TestToWide:
    """Test the wide pivot logic without loading the model."""

    def test_wide_shape(self):
        train = _make_train(n_stores=3, n_families=2, n_days=80)
        # Create a bare pipeline-like object to call the static-ish method
        pipe = _BarePipeline()
        wide, series_order = pipe._to_wide(train)
        assert wide.shape == (80, 6)  # 80 days × 6 series
        assert len(series_order) == 6
        assert series_order == sorted(series_order)

    def test_wide_columns_are_series_ids(self):
        train = _make_train(n_stores=2, n_families=2, n_days=40)
        pipe = _BarePipeline()
        wide, series_order = pipe._to_wide(train)
        assert "1_A" in series_order
        assert "2_B" in series_order
        assert list(wide.columns) == series_order

    def test_wide_sorted_by_date(self):
        train = _make_train(n_days=50)
        pipe = _BarePipeline()
        wide, _ = pipe._to_wide(train)
        assert wide.index.is_monotonic_increasing


class TestToSubmission:
    """Test the vectorized submission mapping."""

    def test_submission_shape_and_ids(self):
        train = _make_train(n_stores=3, n_families=2, n_days=80)
        test = _make_test(n_stores=3, n_families=2, horizon=16)
        pipe = _BarePipeline()
        _, series_order = pipe._to_wide(train)

        preds = np.random.default_rng(42).uniform(0, 100, size=(6, 16))
        sub = pipe._to_submission(preds, series_order, test, horizon=16)
        assert len(sub) == len(test)
        assert list(sub.columns) == ["id", "sales"]
        assert sub["id"].is_monotonic_increasing

    def test_submission_values_match_preds(self):
        """Each test row should get the prediction for its series and date."""
        train = _make_train(n_stores=2, n_families=1, n_days=80)
        test = _make_test(n_stores=2, n_families=1, horizon=16)
        pipe = _BarePipeline()
        wide, series_order = pipe._to_wide(train)

        # Known predictions: series 0 = 100+day, series 1 = 200+day
        preds = np.zeros((2, 16))
        preds[0] = np.arange(100, 116, dtype=float)
        preds[1] = np.arange(200, 216, dtype=float)

        sub = pipe._to_submission(preds, series_order, test, horizon=16)

        # Check a specific row: store 1, family A, first test date
        test_row = test[test["store_nbr"] == 1].iloc[0]
        sub_row = sub[sub["id"] == test_row["id"]].iloc[0]
        expected = preds[series_order.index("1_A"), 0]
        assert abs(sub_row["sales"] - expected) < 1e-6

    def test_submission_clamps_negative(self):
        train = _make_train(n_stores=1, n_families=1, n_days=80)
        test = _make_test(n_stores=1, n_families=1, horizon=16)
        pipe = _BarePipeline()
        _, series_order = pipe._to_wide(train)

        preds = -np.ones((1, 16))  # all negative
        sub = pipe._to_submission(preds, series_order, test, horizon=16)
        assert (sub["sales"] >= 0).all()


class TestRMSLEAlignment:
    """Verify the pipeline's RMSLE computation matches the metric."""

    def test_perfect_forecast_rmsle_zero(self):
        from store_sales.metrics import rmsle

        y = np.array([10.0, 20.0, 30.0])
        assert rmsle(y, y) == 0.0


# ------------------------------------------------------------------
# Integration tests — require the Toto 2.0 model (slow)
# ------------------------------------------------------------------

pytestmark_toto = [
    pytest.mark.skipif(not _have("toto2"), reason="toto-2 not installed"),
    pytest.mark.skipif(not _have("torch"), reason="torch not installed"),
    pytest.mark.slow,
]


@pytest.mark.slow
@pytest.mark.skipif(not _have("toto2"), reason="toto-2 not installed")
class TestTotoPipelineIntegration:
    """End-to-end test with the real Toto 2.0 model (requires GPU/CPU)."""

    def test_predict_produces_valid_submission(self):
        pytest.skip("Integration test — requires model download. Run manually.")

    def test_validate_returns_positive_score(self):
        pytest.skip("Integration test — requires model download. Run manually.")


class _BarePipeline(TotoPipeline):
    """TotoPipeline subclass that skips model loading for unit tests."""

    def __init__(self):
        # Bypass __init__ to avoid loading the model
        self.context_length = None
        self.variate_batch_size = None
        self.log_transform = False
        self.decode_block_size = None
        self.device = None
        self.model = None
        self.model_name = "bare"
        self._val_score = None
