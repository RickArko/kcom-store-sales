from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(
    data_dir: str = "data/",
    train_file: str = "train.csv",
    test_file: str = "test.csv",
    stores_file: str = "stores.csv",
    oil_file: str = "oil.csv",
    holidays_file: str = "holidays_events.csv",
    transactions_file: str = "transactions.csv",
    sample_submission: str = "sample_submission.csv",
) -> dict[str, pd.DataFrame]:
    """Load all competition tables into a dictionary of DataFrames."""
    data_path = Path(data_dir)
    tables = {}

    for name, fname in [
        ("train", train_file),
        ("test", test_file),
        ("stores", stores_file),
        ("oil", oil_file),
        ("holidays", holidays_file),
        ("transactions", transactions_file),
        ("sample_submission", sample_submission),
    ]:
        fpath = data_path / fname
        if fpath.exists():
            if name == "oil":
                tables[name] = pd.read_csv(fpath, parse_dates=["date"])
            elif name == "holidays":
                tables[name] = pd.read_csv(fpath, parse_dates=["date"])
            elif name == "test":
                tables[name] = pd.read_csv(fpath)
            else:
                tables[name] = pd.read_csv(fpath)
            logger.info("  Loaded %s: %s rows", name, len(tables[name]))
        else:
            logger.warning("  Missing file: %s", fpath)
            tables[name] = pd.DataFrame()

    return tables


def extract_holiday_dates(tables: dict[str, pd.DataFrame]) -> list[str] | None:
    """Return holiday dates for distance features (matches train.py)."""
    holidays_df = tables.get("holidays")
    if holidays_df is not None and not holidays_df.empty:
        return holidays_df["date"].dropna().unique().tolist()
    return None


def merge_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge store metadata and external signals into train/test.

    Returns (train_merged, test_merged), where train includes the target.
    """
    train = tables["train"].copy()
    test = tables["test"].copy()

    for df in [train, test]:
        df["date"] = pd.to_datetime(df["date"])

    # Merge stores
    stores = tables.get("stores")
    if stores is not None and not stores.empty:
        train = train.merge(stores, on="store_nbr", how="left")
        test = test.merge(stores, on="store_nbr", how="left")

    # Merge oil
    oil = tables.get("oil")
    if oil is not None and not oil.empty:
        oil = oil.set_index("date")
        for df in [train, test]:
            df["dcoilwtico"] = df["date"].map(oil["dcoilwtico"]).ffill()

    # Merge holidays
    holidays = tables.get("holidays")
    if holidays is not None and not holidays.empty:
        holidays["is_holiday"] = 1
        holiday_dates = holidays[["date", "is_holiday"]].drop_duplicates("date")
        train = train.merge(holiday_dates, on="date", how="left")
        test = test.merge(holiday_dates, on="date", how="left")
        train["is_holiday"] = train["is_holiday"].fillna(0).astype(int)
        test["is_holiday"] = test["is_holiday"].fillna(0).astype(int)

    # Merge transactions
    transactions = tables.get("transactions")
    if transactions is not None and not transactions.empty:
        transactions["date"] = pd.to_datetime(transactions["date"])
        train = train.merge(transactions, on=["date", "store_nbr"], how="left")
        test = test.merge(transactions, on=["date", "store_nbr"], how="left")
        train["transactions"] = train["transactions"].fillna(-1).astype(int)
        test["transactions"] = test["transactions"].fillna(-1).astype(int)

    logger.info("  Train merged: %s  Test merged: %s", train.shape, test.shape)
    return train, test


def trim_pre_activation_zeros(
    train: pd.DataFrame,
    *,
    target_col: str = "sales",
    group_cols: tuple[str, ...] = ("store_nbr", "family"),
    date_col: str = "date",
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Drop pre-activation zero rows per group; keep post-first-sale zeros.

    For each (store, family) series, removes rows before the first positive
    sale. Series that never record a positive sale are left unchanged.
    """
    n_before = len(train)
    if n_before == 0:
        return train.copy(), {
            "rows_dropped": 0,
            "series_trimmed": 0,
            "pct_dropped": 0.0,
        }

    sorted_df = train.sort_values([*group_cols, date_col]).copy()
    groups = list(group_cols)
    has_positive = sorted_df.groupby(groups, observed=True)[target_col].transform(
        lambda s: (s > 0).any()
    )
    had_sale = sorted_df.groupby(groups, observed=True)[target_col].transform(
        lambda s: (s > 0).cummax()
    )
    keep_mask = (~has_positive) | had_sale
    trimmed = sorted_df.loc[keep_mask].reset_index(drop=True)

    rows_dropped = n_before - len(trimmed)
    if rows_dropped:
        dropped_per_group = (
            sorted_df.assign(_keep=keep_mask)
            .groupby(groups, observed=True)["_keep"]
            .apply(lambda s: (~s).sum())
        )
        series_trimmed = int((dropped_per_group > 0).sum())
    else:
        series_trimmed = 0

    stats: dict[str, int | float] = {
        "rows_dropped": rows_dropped,
        "series_trimmed": series_trimmed,
        "pct_dropped": round(rows_dropped / n_before * 100, 2),
    }
    logger.info(
        "  Trim pre-activation zeros: dropped %d rows (%.2f%%), %d series trimmed",
        rows_dropped,
        stats["pct_dropped"],
        series_trimmed,
    )
    return trimmed, stats


def apply_preprocessing(
    train: pd.DataFrame,
    cfg: dict,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Apply optional preprocessing steps from config to train data."""
    prep = cfg.get("preprocessing", {})
    if not prep.get("trim_pre_activation_zeros", False):
        return train, {}

    feat = cfg.get("features", {})
    return trim_pre_activation_zeros(
        train,
        target_col=cfg["competition"]["target"],
        group_cols=(
            feat.get("store_col", "store_nbr"),
            feat.get("family_col", "family"),
        ),
        date_col=feat.get("date_col", "date"),
    )


def timeseries_split(
    train: pd.DataFrame,
    test_period_days: int = 16,
    date_col: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training data into train/validation by last N days."""
    dates = train[date_col].sort_values().unique()
    split_date = dates[-test_period_days]
    train_df = train[train[date_col] < split_date].copy()
    val_df = train[train[date_col] >= split_date].copy()
    logger.info(
        "  TS split: train %s rows (→%s), val %s rows (%s→)",
        len(train_df),
        split_date,
        len(val_df),
        split_date,
    )
    return train_df, val_df


def walk_forward_split(
    train: pd.DataFrame,
    windows: list[int] | None = None,
    date_col: str = "date",
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """Create multiple train/val splits at different cut points.

    Each split holds out the last N days for validation and uses everything
    before that for training.  Returns (train_df, val_df, label) tuples.

    Parameters
    ----------
    windows : list[int], optional
        Number of validation days for each split (default [16, 30, 60, 90]).
    """
    if windows is None:
        windows = [16, 30, 60, 90]
    dates = sorted(train[date_col].unique())
    splits: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    for n_days in sorted(set(windows)):
        if n_days >= len(dates):
            logger.warning("  Skipping window %d (only %d dates available)", n_days, len(dates))
            continue
        split_date = dates[-n_days]
        train_df = train[train[date_col] < split_date].copy()
        val_df = train[train[date_col] >= split_date].copy()
        label = f"val_{n_days}d"
        logger.info(
            "  WFV split %s: train %s rows (→%s), val %s rows (%s→)",
            label,
            len(train_df),
            split_date,
            len(val_df),
            split_date,
        )
        splits.append((train_df, val_df, label))
    if not splits:
        msg = f"No valid walk-forward windows for {windows} (only {len(dates)} dates)"
        raise ValueError(msg)
    return splits
