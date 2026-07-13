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
_PATCH_SIZE = 32  # TOTO 2.0 patch size; context must be divisible by this


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
    context_length : int, optional
        Lookback window (days) fed to the model per series. ``None`` = use all
        available history. Shorter windows focus on recent patterns and use
        less memory; e.g. 512 or 768.
    variate_batch_size : int, optional
        Maximum number of variates (store-family series) per forward pass.
        ``None`` = feed all series as variates of a single multivariate tensor
        (captures cross-series correlations but uses more memory). Chunking
        splits variates into independent groups — memory-safe for small GPUs
        at the cost of losing cross-chunk attention.
    log_transform : bool, optional
        If True, apply ``log1p`` to the target before forecasting and ``expm1``
        to the predictions. Aligns the model with the RMSLE metric and tames
        right-skewed sales distributions.
    """

    def __init__(
        self,
        model_name: str = "Datadog/Toto-2.0-22m",
        device: str | None = None,
        decode_block_size: int | None = 768,
        context_length: int | None = None,
        variate_batch_size: int | None = None,
        log_transform: bool = False,
    ):
        from toto2 import Toto2Model

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.decode_block_size = decode_block_size
        self.context_length = context_length
        self.variate_batch_size = variate_batch_size
        self.log_transform = log_transform

        logger.info("Loading Toto 2.0 from %s on %s ...", model_name, device)
        self.model = Toto2Model.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()
        self.model_name = model_name
        self._val_score: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        wide, series_order = self._to_wide(
            train,
            store_col=store_col,
            family_col=family_col,
            date_col=date_col,
            target_col=target_col,
        )
        median_preds = self._forecast_wide(wide, horizon)
        submission = self._to_submission(
            median_preds, series_order, test, horizon=horizon, date_col=date_col
        )
        return submission

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

        merged = (
            val_part[["id", target_col]]
            .rename(columns={target_col: "actual"})
            .merge(
                sub[["id", "sales"]].rename(columns={"sales": "predicted"}),
                on="id",
                how="inner",
            )
        )

        score = rmsle(merged["actual"].values, merged["predicted"].values)
        self._val_score = score
        logger.info("TOTO validation RMSLE: %.6f", score)
        return score

    @property
    def val_score(self) -> float | None:
        return self._val_score

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        for s in series_order:
            if s not in wide.columns:
                wide[s] = np.nan
        wide = wide[series_order]
        wide = wide.sort_index()
        return wide, series_order

    def _forecast_wide(self, wide: pd.DataFrame, horizon: int) -> np.ndarray:
        """Run TOTO forecast on a wide (date × series) matrix.

        Returns a ``(n_series, horizon)`` array of median predictions.
        """
        # Optional log transform — aligns with RMSLE metric
        if self.log_transform:
            wide = wide.clip(lower=0)
            wide = np.log1p(wide)

        # Trim to context_length if configured
        n_timesteps = len(wide)
        if self.context_length is not None and n_timesteps > self.context_length:
            wide = wide.iloc[-self.context_length :]
            logger.info("  Trimmed to last %d timesteps (context_length)", self.context_length)

        # TOTO requires context length divisible by patch_size (32).
        # Truncate from the start, keeping the most recent observations.
        n_timesteps = len(wide)
        remainder = n_timesteps % _PATCH_SIZE
        if remainder:
            keep = n_timesteps - remainder
            wide = wide.iloc[-keep:]
            logger.info("  Trimmed %d→%d timesteps (patch_size=%d)", n_timesteps, keep, _PATCH_SIZE)

        n_series = wide.shape[1]
        logger.info(
            "TOTO input: %d series × %d timesteps, horizon=%d, log_transform=%s",
            n_series,
            len(wide),
            horizon,
            self.log_transform,
        )

        # Detect missing values for the model's mask handling
        has_missing = bool(wide.isna().any().any())

        # Forecast — optionally chunk variates for memory safety
        if self.variate_batch_size is not None and self.variate_batch_size < n_series:
            chunks = []
            for i in range(0, n_series, self.variate_batch_size):
                chunk = wide.iloc[:, i : i + self.variate_batch_size]
                logger.info("  Variate batch %d-%d / %d", i, i + len(chunk.columns), n_series)
                chunks.append(self._forecast_chunk(chunk, horizon, has_missing))
            median_preds = np.vstack(chunks)
        else:
            median_preds = self._forecast_chunk(wide, horizon, has_missing)

        # Inverse log transform
        if self.log_transform:
            median_preds = np.expm1(median_preds)

        median_preds = np.maximum(median_preds, 0)
        return median_preds

    def _forecast_chunk(self, wide: pd.DataFrame, horizon: int, has_missing: bool) -> np.ndarray:
        """Run a single TOTO forward pass on a subset of variates.

        Returns ``(n_variates_in_chunk, horizon)`` median predictions.
        """
        n_series = wide.shape[1]

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
            has_missing_values=has_missing,
        )
        # quantiles shape: (9, batch=1, C, horizon)
        median_preds = quantiles[_QUANTILE_INDEX, 0].cpu().numpy()  # (C, horizon)
        return median_preds

    def _to_submission(
        self,
        preds: np.ndarray,
        series_order: list[str],
        test: pd.DataFrame,
        horizon: int,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Map wide predictions back to Kaggle submission layout (vectorized).

        ``preds`` has shape ``(n_series, horizon)``.
        """
        test = test.copy()
        test["_series"] = test["store_nbr"].astype(str) + "_" + test["family"].astype(str)
        test_dates = sorted(test[date_col].unique())
        assert len(test_dates) == horizon, f"Expected {horizon} test dates, got {len(test_dates)}"

        # Build lookup arrays for vectorized indexing
        series_to_row = {s: i for i, s in enumerate(series_order)}
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(test_dates)}

        test_series = test["_series"].values
        test_rows = np.array([series_to_row.get(s, 0) for s in test_series])
        test_date_idx = np.array([date_to_idx[pd.Timestamp(d)] for d in test[date_col].values])

        pred_values = preds[test_rows, test_date_idx]
        pred_values = np.maximum(pred_values, 0)

        result = test[["id"]].copy()
        result["sales"] = pred_values
        return result.sort_values("id").reset_index(drop=True)
