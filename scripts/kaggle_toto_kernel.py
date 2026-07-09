"""TOTO 2.0 Zero-Shot Forecast — Store Sales Competition (Kaggle Notebook Kernel).

Copy-paste this entire script into a Kaggle notebook.  It is fully self-contained
and does not depend on the ``store_sales`` package.

Kaggle setup:
  1. Create a new notebook
  2. Add competition data:  Add Input → Competition → "Store Sales - Time Series Forecasting"
  3. Settings → Internet → ON   (needed to pip install toto-2 and download the HF model)
  4. Settings → Accelerator → GPU (T4 x2 or P100)
  5. Paste this script into a single cell and Run All

Expected output:
  - Validation RMSLE ~0.46 (zero-shot, no training)
  - submission.csv with 28 512 rows written to /kaggle/working/

Model: Datadog/Toto-2.0-22m  (https://github.com/datadog/toto)
Tuned: context_length=768, variate_batch_size=256, single-pass decode.
"""

# ============================================================
# Cell 0 — Install TOTO (run once; requires Internet ON)
# ============================================================
# !pip install -q toto-2

# ============================================================
# Cell 1 — Imports & configuration
# ============================================================
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

# --- Hyperparameters (tuned via local ablation) ---
MODEL_NAME = "Datadog/Toto-2.0-22m"
HORIZON = 16
VAL_DAYS = 16
CONTEXT_LENGTH = 768  # last 768 days as lookback (best val_rmsle)
VARIATE_BATCH_SIZE = 256  # chunk series for memory safety (no accuracy impact)
DECODE_BLOCK_SIZE = None  # single pass (horizon=16 ≤ patch_size=32)
LOG_TRANSFORM = False  # TOTO has internal normalization; log1p hurts
QUANTILE_INDEX = 4  # median (0.5 quantile)
PATCH_SIZE = 32

# --- Paths ---
import os

if os.path.exists("/kaggle/input/store-sales-time-series-forecasting"):
    DATA_DIR = "/kaggle/input/store-sales-time-series-forecasting/"
    OUTPUT_DIR = "/kaggle/working/"
else:
    DATA_DIR = "data/"
    OUTPUT_DIR = "outputs/submissions/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# Cell 2 — Load competition data
# ============================================================
def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs."""
    train = pd.read_csv(data_dir + "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir + "test.csv", parse_dates=["date"])
    print(
        f"Train: {len(train):,} rows | {train.store_nbr.nunique()} stores × "
        f"{train.family.nunique()} families = "
        f"{train.store_nbr.nunique() * train.family.nunique()} series"
    )
    print(f"  Date range: {train.date.min().date()} → {train.date.max().date()}")
    print(f"Test:  {len(test):,} rows | {test.date.min().date()} → {test.date.max().date()}")
    return train, test


train, test = load_data(DATA_DIR)


# ============================================================
# Cell 3 — Pivot to wide (date × series) format
# ============================================================
def to_wide(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pivot store-family data to a (date × series) matrix."""
    df = df.copy()
    df["_series"] = df["store_nbr"].astype(str) + "_" + df["family"].astype(str)
    series_order = sorted(df["_series"].unique())
    wide = df.pivot_table(index="date", columns="_series", values="sales", aggfunc="sum")
    wide = wide.sort_index()[series_order]
    return wide, series_order


wide, series_order = to_wide(train)
print(f"Wide matrix: {wide.shape} (dates × series)")
print(f"Missing values: {wide.isna().sum().sum()}")


# ============================================================
# Cell 4 — RMSLE metric
# ============================================================
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error (competition metric)."""
    y_true = np.maximum(np.asarray(y_true, dtype=np.float64), 0)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 0)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


# ============================================================
# Cell 5 — TOTO forecast function (zero-shot, batched variates)
# ============================================================
def toto_forecast(
    wide_df: pd.DataFrame,
    horizon: int,
    model,
    device: torch.device,
    *,
    context_length: int | None = CONTEXT_LENGTH,
    variate_batch_size: int | None = VARIATE_BATCH_SIZE,
    log_transform: bool = LOG_TRANSFORM,
    decode_block_size: int | None = DECODE_BLOCK_SIZE,
) -> np.ndarray:
    """Run TOTO 2.0 zero-shot forecast on a wide (date × series) matrix.

    Returns ``(n_series, horizon)`` array of median predictions.
    """
    # Optional log transform
    if log_transform:
        wide_df = wide_df.clip(lower=0)
        wide_df = np.log1p(wide_df)

    # Trim to context_length
    if context_length is not None and len(wide_df) > context_length:
        wide_df = wide_df.iloc[-context_length:]

    # Trim to be divisible by patch_size (32)
    remainder = len(wide_df) % PATCH_SIZE
    if remainder:
        wide_df = wide_df.iloc[:-(remainder)]
    n_timesteps = len(wide_df)
    n_series = wide_df.shape[1]

    print(f"  TOTO input: {n_series} series × {n_timesteps} timesteps, horizon={horizon}")

    has_missing = bool(wide_df.isna().any().any())

    def _forecast_chunk(chunk: pd.DataFrame) -> np.ndarray:
        """Single forward pass on a subset of variates."""
        nv = chunk.shape[1]
        # (T, C) → (1, C, T)
        target = torch.from_numpy(chunk.values.T.astype(np.float32)).unsqueeze(0).to(device)
        mask = torch.isfinite(target)
        target[~mask] = 0.0
        series_ids = torch.arange(nv, dtype=torch.long, device=device).unsqueeze(0)
        quantiles = model.forecast(
            {"target": target, "target_mask": mask, "series_ids": series_ids},
            horizon=horizon,
            decode_block_size=decode_block_size,
            has_missing_values=has_missing,
        )
        return quantiles[QUANTILE_INDEX, 0].cpu().numpy()  # (C, horizon)

    # Chunk variates for memory safety
    if variate_batch_size is not None and variate_batch_size < n_series:
        chunks = []
        for i in range(0, n_series, variate_batch_size):
            chunk = wide_df.iloc[:, i : i + variate_batch_size]
            chunks.append(_forecast_chunk(chunk))
        preds = np.vstack(chunks)
    else:
        preds = _forecast_chunk(wide_df)

    # Inverse log transform
    if log_transform:
        preds = np.expm1(preds)

    return np.maximum(preds, 0)


# ============================================================
# Cell 6 — Build submission from wide predictions (vectorized)
# ============================================================
def build_submission(
    preds: np.ndarray,
    series_order: list[str],
    test_df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Map (n_series, horizon) predictions → Kaggle (id, sales) layout."""
    test = test_df.copy()
    test["_series"] = test["store_nbr"].astype(str) + "_" + test["family"].astype(str)
    test_dates = sorted(test["date"].unique())
    assert len(test_dates) == horizon

    series_to_row = {s: i for i, s in enumerate(series_order)}
    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(test_dates)}

    test_rows = np.array([series_to_row.get(s, 0) for s in test["_series"].values])
    test_date_idx = np.array([date_to_idx[pd.Timestamp(d)] for d in test["date"].values])

    result = test[["id"]].copy()
    result["sales"] = np.maximum(preds[test_rows, test_date_idx], 0)
    return result.sort_values("id").reset_index(drop=True)


# ============================================================
# Cell 7 — Load TOTO 2.0 model
# ============================================================
from toto2 import Toto2Model

t0 = time.time()
model = Toto2Model.from_pretrained(MODEL_NAME).to(DEVICE).eval()
print(f"Model loaded on {DEVICE} in {time.time() - t0:.1f}s")


# ============================================================
# Cell 8 — Validation (hold out last 16 days, score RMSLE)
# ============================================================
print("\n=== Validation (last 16 days held out) ===")
val_wide = wide.iloc[:-VAL_DAYS]
val_actual = wide.iloc[-VAL_DAYS:]

t0 = time.time()
val_preds = toto_forecast(val_wide, HORIZON, model, DEVICE)  # (n_series, horizon)
val_score = rmsle(val_actual.values.T, val_preds)  # transpose actuals to match
print(f"Validation RMSLE: {val_score:.6f}  ({time.time() - t0:.1f}s)")

# Per-family breakdown (optional insight)
val_df = train[train["date"] >= train["date"].max() - pd.Timedelta(days=VAL_DAYS - 1)].copy()
val_df["_series"] = val_df["store_nbr"].astype(str) + "_" + val_df["family"].astype(str)
series_to_row = {s: i for i, s in enumerate(series_order)}
val_dates = sorted(val_df["date"].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(val_dates)}
val_rows = np.array([series_to_row.get(s, 0) for s in val_df["_series"].values])
val_didx = np.array([date_to_idx[pd.Timestamp(d)] for d in val_df["date"].values])
val_df["pred"] = val_preds[val_rows, val_didx]
family_scores = (
    val_df.groupby("family")
    .apply(lambda g: rmsle(g["sales"].values, g["pred"].values), include_groups=False)
    .sort_values()
)
print("\nPer-family RMSLE (best 5 / worst 5):")
print(family_scores.head().to_string())
print("  ...")
print(family_scores.tail().to_string())


# ============================================================
# Cell 9 — Full forecast & submission
# ============================================================
print("\n=== Full test forecast ===")
t0 = time.time()
full_preds = toto_forecast(wide, HORIZON, model, DEVICE)
submission = build_submission(full_preds, series_order, test, HORIZON)
print(f"Forecast complete ({time.time() - t0:.1f}s) — {len(submission):,} rows")

submission.to_csv(OUTPUT_DIR + "submission.csv", index=False)
print(f"Saved → {OUTPUT_DIR}submission.csv")
print(submission.head())
print(
    f"\nSales stats: min={submission.sales.min():.2f}  "
    f"mean={submission.sales.mean():.2f}  max={submission.sales.max():.2f}"
)
