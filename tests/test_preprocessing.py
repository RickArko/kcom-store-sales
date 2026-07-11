from __future__ import annotations

import pandas as pd

from store_sales.data import apply_preprocessing, trim_pre_activation_zeros


def _make_group(
    store: int,
    family: str,
    sales: list[float],
    dates: list[str] | None = None,
) -> pd.DataFrame:
    n = len(sales)
    if dates is None:
        dates = [f"2017-01-{i + 1:02d}" for i in range(n)]
    return pd.DataFrame(
        {
            "store_nbr": [store] * n,
            "family": [family] * n,
            "date": pd.to_datetime(dates),
            "sales": sales,
        }
    )


def test_trim_leading_zeros_before_first_sale():
    df = _make_group(1, "A", [0, 0, 5, 0, 3])
    trimmed, stats = trim_pre_activation_zeros(df)
    assert list(trimmed["sales"]) == [5, 0, 3]
    assert stats["rows_dropped"] == 2
    assert stats["series_trimmed"] == 1


def test_trim_never_activated_series_unchanged():
    df = _make_group(1, "A", [0, 0, 0])
    trimmed, stats = trim_pre_activation_zeros(df)
    assert len(trimmed) == 3
    assert stats["rows_dropped"] == 0
    assert stats["series_trimmed"] == 0


def test_trim_no_leading_zeros_unchanged():
    df = _make_group(1, "A", [1, 2, 3])
    trimmed, stats = trim_pre_activation_zeros(df)
    assert list(trimmed["sales"]) == [1, 2, 3]
    assert stats["rows_dropped"] == 0


def test_trim_multi_group():
    g1 = _make_group(1, "A", [0, 0, 2])
    g2 = _make_group(1, "B", [0, 0, 0])
    g3 = _make_group(2, "A", [1, 0, 4])
    df = pd.concat([g1, g2, g3], ignore_index=True)
    trimmed, stats = trim_pre_activation_zeros(df)
    assert len(trimmed) == 7  # g1: 1 row; g2: 3 rows; g3: 3 rows
    assert stats["rows_dropped"] == 2
    assert stats["series_trimmed"] == 1


def test_apply_preprocessing_disabled():
    df = _make_group(1, "A", [0, 0, 5])
    cfg = {"competition": {"target": "sales"}, "features": {}}
    out, stats = apply_preprocessing(df, cfg)
    assert len(out) == 3
    assert stats == {}


def test_apply_preprocessing_enabled():
    df = _make_group(1, "A", [0, 0, 5])
    cfg = {
        "competition": {"target": "sales"},
        "features": {"store_col": "store_nbr", "family_col": "family", "date_col": "date"},
        "preprocessing": {"trim_pre_activation_zeros": True},
    }
    out, stats = apply_preprocessing(df, cfg)
    assert list(out["sales"]) == [5]
    assert stats["rows_dropped"] == 2
