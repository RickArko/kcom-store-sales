"""TOTO foundation-model forecasting pipeline for Store Sales.

Zero-shot forecasting using Datadog's Toto 2.0 time-series foundation model.
Reshapes store-family data into a wide (variate) format, feeds it through
the model, and converts quantile predictions back to submission layout.

Usage:
    from store_sales.toto_pipeline import TotoPipeline
    pipe = TotoPipeline(model_name="Datadog/Toto-2.0-22m")
    submission = pipe.predict(train_merged, test)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

from store_sales.metrics import rmsle

logger = logging.getLogger(__name__)

_QUANTILE_INDEX = 4  # median (0.5 quantile) for point forecasts


class TotoPipeline:
    """Zero-shot forecasting pipeline wrapping Toto 2.0.

    Parameters
    ----------
    model_name : str
        HuggingFace repo or local path for the Toto 2.0 checkpoint.
        Recommended: "Datadog/Toto-2.0-22m"
    device : str, optional
        "cuda" or "cpu". Auto-detected if not given.
    decode_block_size : int, optional
        Block decoding size for long horizons. ``None`` = single forward pass.
    """

    def __init__(
        self,
        model_name: str = "Datadog/Toto-2.0-22m",
        device: str | None = None,
        decode_block_size: int | None = 768,
    ):
        from toto2 import Toto2Model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.decode_block_size = decode_block_size

        logger.info("Loading Toto 2.0 from %s on %s ...", model_name, device)
        self.model = Toto2Model.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        self.model_name = model_name
        self._val_score: float | None = None

    @torch.inference_mode()
    def predict(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        horizon: int = 16,
        *,
        store_col: str = "store_nbr",
        family_col: str = "family",
        date_col: str = "date",
        target_col: str = "sales",
    ) -> pd.DataFrame:
        """Generate zero-shot forecasts for all store-family series.

        Returns a submission DataFrame with columns ``[id, sales]``.
        """
        # Pivot to wide: (date, series_id) matrix
        wide, series_order = self._to_wide(
            train,
            store_col=store_col,
            family_col=family_col,
            date_col=date_col,
            target_col=target_col,
        )
        n_series = wide.shape[1]
        n_timesteps = wide.shape[0]

        logger.info(
            "TOTO input: %d series × %d timesteps, horizon=%d",
            n_series,
            n_timesteps,
            horizon,
        )

        # TOTO requires context length divisible by patch_size (32).
        # Truncate from the start, keeping the most recent observations.
        patch_size = 32
        remainder = n_timesteps % patch_size
        if remainder:
            keep = patch_size * (n_timesteps // patch_size)
            wide = wide.iloc[-keep:]
            logger.info("  Trimmed %d→%d timesteps (patch_size=%d)", n_timesteps, keep, patch_size)

        # wide shape: (T, C). TOTO expects (C, T) → transpose.
        target = torch.from_numpy(wide.values.T.astype(np.float32)).unsqueeze(0)  # (1, C, T)
        target_mask = torch.isfinite(target)
        target[~target_mask] = 0.0

        series_ids = torch.arange(n_series, dtype=torch.long, device=self.device).unsqueeze(0)

        target = target.to(self.device)
        target_mask = target_mask.to(self.device)

        inputs = {
            "target": target,
            "target_mask": target_mask,
            "series_ids": series_ids,
        }

        quantiles = self.model.forecast(
            inputs,
            horizon=horizon,
            decode_block_size=self.decode_block_size,
            has_missing_values=False,
        )
        # quantiles shape: (9, batch=1, C, horizon)
        median_preds = quantiles[_QUANTILE_INDEX, 0].cpu().numpy()  # (C, horizon)
        median_preds = np.maximum(median_preds, 0)

        # Build submission: map back to original test rows sorted by id
        submission = self._to_submission(
            median_preds, series_order, test, horizon=horizon, date_col=date_col
        )
        return submission

    def _to_wide(
        self,
        train: pd.DataFrame,
        store_col: str = "store_nbr",
        family_col: str = "family",
        date_col: str = "date",
        target_col: str = "sales",
    ) -> tuple[pd.DataFrame, list[str]]:
        """Pivot training data to wide (date × series) matrix."""
        train = train.copy()
        train["_series"] = train[store_col].astype(str) + "_" + train[family_col].astype(str)
        series_order = sorted(train["_series"].unique())
        wide = train.pivot_table(
            index=date_col, columns="_series", values=target_col, aggfunc="sum"
        )
        # Ensure all series are present
        for s in series_order:
            if s not in wide.columns:
                wide[s] = np.nan
        wide = wide[series_order]
        wide = wide.sort_index()
        return wide, series_order

    def _to_submission(
        self,
        preds: np.ndarray,
        series_order: list[str],
        test: pd.DataFrame,
        horizon: int,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Map wide predictions back to Kaggle submission layout.

        ``preds`` has shape (n_series, horizon).
        """
        test = test.copy()
        test["_series"] = test["store_nbr"].astype(str) + "_" + test["family"].astype(str)
        test_dates = sorted(test[date_col].unique())
        assert len(test_dates) == horizon, f"Expected {horizon} test dates, got {len(test_dates)}"

        # Build mapping from series name -> predictions
        series_to_preds = {}
        for i, s in enumerate(series_order):
            series_to_preds[s] = preds[i]

        pred_list = []
        for _, row in test.iterrows():
            s = row["_series"]
            date_idx = test_dates.index(pd.Timestamp(row[date_col]))
            pred = series_to_preds.get(s, np.zeros(horizon))[date_idx]
            pred_list.append(max(pred, 0))

        result = test[["id"]].copy()
        result["sales"] = pred_list
        return result.sort_values("id").reset_index(drop=True)

    def validate(
        self,
        train: pd.DataFrame,
        horizon: int = 16,
        val_days: int = 16,
        *,
        store_col: str = "store_nbr",
        family_col: str = "family",
        date_col: str = "date",
        target_col: str = "sales",
    ) -> float:
        """Rolling validation: hold out last ``val_days`` and score with RMSLE."""
        dates = sorted(train[date_col].unique())
        # Train: everything before the last `val_days` days
        # Val: last `val_days` days (used as ground truth; we predict into it)
        train_part = train[train[date_col] < dates[-val_days]].copy()
        val_part = train[train[date_col] >= dates[-val_days]].copy()

        sub = self.predict(
            train_part,
            val_part,
            horizon=val_days,
            store_col=store_col,
            family_col=family_col,
            date_col=date_col,
            target_col=target_col,
        )

        # Merge predictions with actuals by id
        merged = (
            val_part[["id", target_col]]
            .rename(columns={target_col: "actual"})
            .merge(
                sub[["id", "sales"]].rename(columns={"sales": "predicted"}),
                on="id",
                how="inner",
            )
        )

        score = rmsle(
            merged["actual"].values,
            merged["predicted"].values,
        )
        self._val_score = score
        logger.info("TOTO validation RMSLE: %.6f", score)
        return score

    @property
    def val_score(self) -> float | None:
        return self._val_score
