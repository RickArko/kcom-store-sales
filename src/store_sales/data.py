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
