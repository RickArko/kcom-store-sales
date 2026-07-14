"""Offline Ilya-style metrics audit: slices, bootstrap UQ, shift, residual discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text

from store_sales.metrics import rmsle

SLICE_AXES = (
    "family",
    "store_nbr",
    "type",
    "cluster",
    "velocity_tier",
    "dayofweek",
    "onpromotion",
)

_DISCOVERY_FEATURES = (
    "store_nbr",
    "family",
    "type",
    "cluster",
    "velocity_mean",
    "dayofweek",
    "onpromotion",
    "is_holiday",
)


def compute_row_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Secondary metrics matching compare_models._compute_metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = max(float(np.sum(y_true)), 1e-10)
    return {
        "rmsle": rmsle(y_true, y_pred),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "wmape": float(np.sum(np.abs(y_true - y_pred)) / denom),
        "bias_pct": float(np.sum(y_pred - y_true) / denom * 100),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def squared_log_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row squared log error (RMSLE building block)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    return (np.log1p(y_true) - np.log1p(y_pred)) ** 2


def log_residual(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Signed log residual: log1p(y) - log1p(ŷ)."""
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    y_true = np.asarray(y_true, dtype=np.float64)
    return np.log1p(y_true) - np.log1p(y_pred)


def assign_velocity_tiers(
    history: pd.DataFrame,
    *,
    sales_col: str = "sales",
    store_col: str = "store_nbr",
    family_col: str = "family",
) -> pd.DataFrame:
    """Map (store, family) → mean sales and tertile velocity_tier (slow/mid/fast)."""
    means = (
        history.groupby([store_col, family_col], observed=True)[sales_col]
        .mean()
        .rename("velocity_mean")
        .reset_index()
    )
    # qcut can fail with too few unique values; fall back to equal-width labels
    try:
        means["velocity_tier"] = pd.qcut(
            means["velocity_mean"],
            q=3,
            labels=["slow", "mid", "fast"],
            duplicates="drop",
        )
    except ValueError:
        means["velocity_tier"] = "mid"
    means["velocity_tier"] = means["velocity_tier"].astype(str)
    return means


def filter_validation(
    preds: pd.DataFrame,
    *,
    test_period_days: int = 16,
    date_col: str = "date",
) -> pd.DataFrame:
    """Keep last ``test_period_days`` of rows marked split=='train' (holdout)."""
    train_rows = preds[preds["split"] == "train"].copy()
    train_rows[date_col] = pd.to_datetime(train_rows[date_col])
    dates = np.sort(train_rows[date_col].unique())
    if len(dates) == 0:
        return train_rows.iloc[0:0].copy()
    n = min(test_period_days, len(dates))
    split_date = dates[-n]
    return train_rows[train_rows[date_col] >= split_date].copy()


def enrich_val_frame(
    val: pd.DataFrame,
    train_merged: pd.DataFrame,
    velocity: pd.DataFrame,
) -> pd.DataFrame:
    """Attach store meta, promotions, holidays, oil, velocity tiers to val preds."""
    out = val.copy()
    out["date"] = pd.to_datetime(out["date"])

    meta_cols = [
        c
        for c in (
            "date",
            "store_nbr",
            "family",
            "type",
            "cluster",
            "city",
            "state",
            "onpromotion",
            "is_holiday",
            "dcoilwtico",
        )
        if c in train_merged.columns
    ]
    meta = train_merged[meta_cols].drop_duplicates(["date", "store_nbr", "family"])
    meta["date"] = pd.to_datetime(meta["date"])
    out = out.merge(meta, on=["date", "store_nbr", "family"], how="left")
    out = out.merge(velocity, on=["store_nbr", "family"], how="left")
    out["dayofweek"] = out["date"].dt.dayofweek
    if "onpromotion" in out.columns:
        out["onpromotion"] = out["onpromotion"].fillna(0).astype(int)
    if "is_holiday" in out.columns:
        out["is_holiday"] = out["is_holiday"].fillna(0).astype(int)
    if "velocity_tier" not in out.columns:
        out["velocity_tier"] = "mid"
    else:
        out["velocity_tier"] = out["velocity_tier"].fillna("mid").astype(str)
    if "velocity_mean" not in out.columns:
        out["velocity_mean"] = 0.0
    else:
        out["velocity_mean"] = out["velocity_mean"].fillna(0.0)
    out["sle"] = squared_log_error(out["actual"].to_numpy(), out["predicted"].to_numpy())
    out["log_resid"] = log_residual(out["actual"].to_numpy(), out["predicted"].to_numpy())
    out["abs_log_err"] = np.abs(out["log_resid"])
    return out


def slice_scores(
    df: pd.DataFrame,
    axes: tuple[str, ...] | list[str] = SLICE_AXES,
) -> pd.DataFrame:
    """Per-slice RMSLE / bias / error concentration for each available axis."""
    total_sle = float(df["sle"].sum()) if "sle" in df.columns else 0.0
    if total_sle <= 0:
        total_sle = 1e-10
    rows: list[dict[str, Any]] = []
    for axis in axes:
        if axis not in df.columns:
            continue
        for key, group in df.groupby(axis, observed=True):
            m = compute_row_metrics(group["actual"].to_numpy(), group["predicted"].to_numpy())
            sle_share = float(group["sle"].sum()) / total_sle if "sle" in group.columns else 0.0
            rows.append(
                {
                    "axis": axis,
                    "slice": str(key),
                    "n": int(len(group)),
                    "rmsle": m["rmsle"],
                    "mae": m["mae"],
                    "bias_pct": m["bias_pct"],
                    "wmape": m["wmape"],
                    "sle_share": sle_share,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["axis", "slice", "n", "rmsle", "mae", "bias_pct", "wmape", "sle_share"]
        )
    return pd.DataFrame(rows).sort_values(["axis", "rmsle"], ascending=[True, False])


def bootstrap_rmsle(
    df: pd.DataFrame,
    *,
    n_boot: int = 500,
    seed: int = 42,
    store_col: str = "store_nbr",
    family_col: str = "family",
) -> dict[str, float]:
    """Series-level bootstrap CI for global RMSLE."""
    y_true = df["actual"].to_numpy(dtype=np.float64)
    y_pred = df["predicted"].to_numpy(dtype=np.float64)
    point = rmsle(y_true, y_pred)

    series_ids = df[[store_col, family_col]].drop_duplicates().reset_index(drop=True)
    n_series = len(series_ids)
    if n_series == 0:
        return {
            "val_rmsle": point,
            "val_rmsle_boot_mean": point,
            "val_rmsle_ci_low": point,
            "val_rmsle_ci_high": point,
            "n_boot": 0,
            "n_series": 0,
        }

    # Pre-index rows by series for fast resampling
    keys = list(zip(df[store_col].to_numpy(), df[family_col].to_numpy(), strict=True))
    groups: dict[tuple[Any, Any], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    series_keys = list(groups.keys())

    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.choice(len(series_keys), size=n_series, replace=True)
        idx: list[int] = []
        for s in sampled:
            idx.extend(groups[series_keys[s]])
        scores[b] = rmsle(y_true[idx], y_pred[idx])

    return {
        "val_rmsle": float(point),
        "val_rmsle_boot_mean": float(scores.mean()),
        "val_rmsle_ci_low": float(np.quantile(scores, 0.025)),
        "val_rmsle_ci_high": float(np.quantile(scores, 0.975)),
        "n_boot": float(n_boot),
        "n_series": float(n_series),
    }


def paired_bootstrap_delta(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    n_boot: int = 500,
    seed: int = 42,
    store_col: str = "store_nbr",
    family_col: str = "family",
) -> dict[str, float]:
    """Paired series bootstrap of RMSLE_A - RMSLE_B (negative ⇒ A better)."""
    key_cols = [store_col, family_col, "date"]
    a = df_a[key_cols + ["actual", "predicted"]].copy()
    b = df_b[key_cols + ["predicted"]].copy().rename(columns={"predicted": "predicted_b"})
    merged = a.merge(b, on=key_cols, how="inner")
    if merged.empty:
        return {
            "delta_rmsle": float("nan"),
            "delta_ci_low": float("nan"),
            "delta_ci_high": float("nan"),
            "p_a_worse": float("nan"),
            "n_series": 0,
            "n_boot": 0,
        }

    y = merged["actual"].to_numpy(dtype=np.float64)
    pred_a = merged["predicted"].to_numpy(dtype=np.float64)
    pred_b = merged["predicted_b"].to_numpy(dtype=np.float64)
    point = rmsle(y, pred_a) - rmsle(y, pred_b)

    keys = list(zip(merged[store_col].to_numpy(), merged[family_col].to_numpy(), strict=True))
    groups: dict[tuple[Any, Any], list[int]] = {}
    for i, key in enumerate(keys):
        groups.setdefault(key, []).append(i)
    series_keys = list(groups.keys())
    n_series = len(series_keys)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b_i in range(n_boot):
        sampled = rng.choice(n_series, size=n_series, replace=True)
        idx: list[int] = []
        for s in sampled:
            idx.extend(groups[series_keys[s]])
        deltas[b_i] = rmsle(y[idx], pred_a[idx]) - rmsle(y[idx], pred_b[idx])

    return {
        "delta_rmsle": float(point),
        "delta_ci_low": float(np.quantile(deltas, 0.025)),
        "delta_ci_high": float(np.quantile(deltas, 0.975)),
        "p_a_worse": float(np.mean(deltas >= 0)),
        "n_series": float(n_series),
        "n_boot": float(n_boot),
    }


def _period_summaries(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {"n_rows": float(len(df))}
    if "onpromotion" in df.columns:
        out["promotion_rate"] = float(df["onpromotion"].fillna(0).mean())
    if "is_holiday" in df.columns:
        out["holiday_rate"] = float(df["is_holiday"].fillna(0).mean())
    if "dcoilwtico" in df.columns:
        oil = df["dcoilwtico"].dropna()
        out["oil_mean"] = float(oil.mean()) if len(oil) else float("nan")
        out["oil_std"] = float(oil.std()) if len(oil) else float("nan")
    if "sales" in df.columns and "family" in df.columns:
        mix = df.groupby("family", observed=True)["sales"].sum()
        total = float(mix.sum()) or 1.0
        # Shannon entropy of family mix as a compact diversity signal
        p = (mix / total).to_numpy(dtype=np.float64)
        p = p[p > 0]
        out["family_mix_entropy"] = float(-np.sum(p * np.log(p)))
    return out


def family_mix_js_divergence(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Jensen–Shannon divergence between family sales-mix distributions."""
    if "sales" not in a.columns or "family" not in a.columns:
        return float("nan")
    if "sales" not in b.columns or "family" not in b.columns:
        return float("nan")
    sa = a.groupby("family", observed=True)["sales"].sum()
    sb = b.groupby("family", observed=True)["sales"].sum()
    families = sorted(set(sa.index) | set(sb.index))
    pa = np.array([sa.get(f, 0.0) for f in families], dtype=np.float64)
    pb = np.array([sb.get(f, 0.0) for f in families], dtype=np.float64)
    pa = pa / (pa.sum() + 1e-12)
    pb = pb / (pb.sum() + 1e-12)
    m = 0.5 * (pa + pb)

    def _kl(p: np.ndarray, q: np.ndarray) -> float:
        mask = p > 0
        return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, None))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


def shift_summary(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, Any]:
    """Lightweight train/val/test distribution summaries + mix divergence."""
    return {
        "train": _period_summaries(train),
        "val": _period_summaries(val),
        "test": _period_summaries(test.assign(sales=test.get("sales", 0))),
        "family_mix_js_train_val": family_mix_js_divergence(train, val),
        "family_mix_js_train_test": family_mix_js_divergence(
            train, test.assign(sales=1.0) if "sales" not in test.columns else test
        ),
    }


def _encode_features(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, list[str], OrdinalEncoder | None]:
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return np.zeros((len(df), 0)), [], None
    X = df[cols].copy()
    cat_cols = [c for c in cols if X[c].dtype == object or str(X[c].dtype) == "category"]
    num_cols = [c for c in cols if c not in cat_cols]
    encoder = None
    parts: list[np.ndarray] = []
    names: list[str] = []
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = X[cat_cols].astype(str).fillna("missing")
        parts.append(encoder.fit_transform(X_cat))
        names.extend(cat_cols)
    if num_cols:
        parts.append(X[num_cols].fillna(0).to_numpy(dtype=np.float64))
        names.extend(num_cols)
    if not parts:
        return np.zeros((len(df), 0)), [], encoder
    return np.hstack(parts), names, encoder


def error_tree_leaves(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] | list[str] = _DISCOVERY_FEATURES,
    max_depth: int = 4,
    min_samples_leaf: int = 50,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Shallow tree on |log error|; return high-error leaves with rule text."""
    if len(df) < max(min_samples_leaf * 2, 20):
        return []
    X, names, _ = _encode_features(df, list(feature_cols))
    if X.shape[1] == 0:
        return []
    y = df["abs_log_err"].to_numpy(dtype=np.float64)
    tree = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(X, y)
    leaf_ids = tree.apply(X)
    rows: list[dict[str, Any]] = []
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        rows.append(
            {
                "leaf_id": int(leaf),
                "n": int(mask.sum()),
                "mean_abs_log_err": float(y[mask].mean()),
                "rmsle": rmsle(
                    df.loc[mask, "actual"].to_numpy(),
                    df.loc[mask, "predicted"].to_numpy(),
                ),
            }
        )
    rows.sort(key=lambda r: r["mean_abs_log_err"], reverse=True)
    tree_text = export_text(tree, feature_names=names, max_depth=max_depth)
    for r in rows[:top_k]:
        r["tree_excerpt"] = tree_text
    return rows[:top_k]


def multiaccuracy_audit(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] | list[str] = _DISCOVERY_FEATURES,
    alpha: float = 1.0,
    val_frac: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit Ridge on log residuals; nonzero R² ⇒ residual structure remains."""
    if len(df) < 40:
        return {"r2": float("nan"), "n_train": 0, "n_val": 0, "top_coefs": []}
    X, names, _ = _encode_features(df, list(feature_cols))
    if X.shape[1] == 0:
        return {"r2": float("nan"), "n_train": 0, "n_val": 0, "top_coefs": []}
    y = df["log_resid"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * val_frac))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[val_idx])
    r2 = float(r2_score(y[val_idx], pred))
    coefs = sorted(
        ({"feature": n, "coef": float(c)} for n, c in zip(names, model.coef_, strict=True)),
        key=lambda d: abs(d["coef"]),
        reverse=True,
    )
    return {
        "r2": r2,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "top_coefs": coefs[:10],
    }


def residual_clusters(
    df: pd.DataFrame,
    *,
    feature_cols: tuple[str, ...] | list[str] = _DISCOVERY_FEATURES,
    top_pct: float = 0.2,
    n_clusters: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """k-means on top-|error| rows; summarize cluster composition."""
    if len(df) < n_clusters * 5:
        return []
    cutoff = df["abs_log_err"].quantile(1.0 - top_pct)
    high = df[df["abs_log_err"] >= cutoff].copy()
    if len(high) < n_clusters:
        return []
    X, names, _ = _encode_features(high, list(feature_cols))
    if X.shape[1] == 0:
        return []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    k = min(n_clusters, len(high))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(Xs)
    high = high.assign(cluster=labels)
    summaries: list[dict[str, Any]] = []
    for c in range(k):
        g = high[high["cluster"] == c]
        top_family = g["family"].value_counts().head(3).to_dict() if "family" in g.columns else {}
        tier_mix = (
            g["velocity_tier"].value_counts(normalize=True).round(3).to_dict()
            if "velocity_tier" in g.columns
            else {}
        )
        type_mix = (
            g["type"].astype(str).value_counts(normalize=True).round(3).to_dict()
            if "type" in g.columns
            else {}
        )
        summaries.append(
            {
                "cluster": int(c),
                "n": int(len(g)),
                "mean_abs_log_err": float(g["abs_log_err"].mean()),
                "rmsle": rmsle(g["actual"].to_numpy(), g["predicted"].to_numpy()),
                "bias_pct": compute_row_metrics(g["actual"].to_numpy(), g["predicted"].to_numpy())[
                    "bias_pct"
                ],
                "top_families": {str(k_): int(v) for k_, v in top_family.items()},
                "velocity_tier_mix": {str(k_): float(v) for k_, v in tier_mix.items()},
                "type_mix": {str(k_): float(v) for k_, v in type_mix.items()},
                "feature_names": names,
            }
        )
    summaries.sort(key=lambda d: d["mean_abs_log_err"], reverse=True)
    return summaries


def render_report(
    *,
    run_name: str,
    global_metrics: dict[str, float],
    bootstrap: dict[str, float],
    slices: pd.DataFrame,
    shift: dict[str, Any],
    error_leaves: list[dict[str, Any]],
    multiacc: dict[str, Any],
    clusters: list[dict[str, Any]],
    run_meta: dict[str, Any],
    paired: dict[str, float] | None = None,
    compare_run: str | None = None,
) -> str:
    """Markdown audit report."""
    lines: list[str] = [
        f"# Metrics audit — `{run_name}`",
        "",
        "## Global",
        "",
        (
            f"- **val_rmsle:**"
            f" {bootstrap.get('val_rmsle', global_metrics.get('rmsle', float('nan'))):.6f}"
        ),
        f"- **95% CI:** [{bootstrap.get('val_rmsle_ci_low', float('nan')):.6f},"
        f" {bootstrap.get('val_rmsle_ci_high', float('nan')):.6f}]"
        f" (B={int(bootstrap.get('n_boot', 0))}, series={int(bootstrap.get('n_series', 0))})",
        f"- **bias_pct:** {global_metrics.get('bias_pct', float('nan')):.3f}%",
        f"- **mae / wmape:** {global_metrics.get('mae', float('nan')):.3f} /"
        f" {global_metrics.get('wmape', float('nan')):.4f}",
        "",
        "## Holdout honesty",
        "",
        f"- trim_pre_activation_zeros: `{run_meta.get('trim_pre_activation_zeros')}`",
        f"- test_period_days: `{run_meta.get('test_period_days')}`",
        f"- family_mix JS(train→val): `{shift.get('family_mix_js_train_val', float('nan')):.4f}`",
        f"- family_mix JS(train→test): `{shift.get('family_mix_js_train_test', float('nan')):.4f}`",
    ]
    if run_meta.get("cv_one_shot_mean") is not None:
        lines.append(
            f"- CV one-shot mean/std: `{run_meta.get('cv_one_shot_mean')}` /"
            f" `{run_meta.get('cv_one_shot_std')}`"
        )
    if run_meta.get("cv_recursive_mean") is not None:
        lines.append(
            f"- CV recursive mean/std: `{run_meta.get('cv_recursive_mean')}` /"
            f" `{run_meta.get('cv_recursive_std')}`"
        )
    lines.extend(
        [
            "",
            "> Large train→val mix shift ⇒ do not over-rank tiny `val_rmsle` deltas.",
            "",
        ]
    )

    if paired is not None and compare_run is not None:
        lines.extend(
            [
                f"## Paired compare vs `{compare_run}`",
                "",
                f"- Δ RMSLE (A−B): `{paired.get('delta_rmsle', float('nan')):.6f}`",
                f"- 95% CI: [{paired.get('delta_ci_low', float('nan')):.6f},"
                f" {paired.get('delta_ci_high', float('nan')):.6f}]",
                f"- P(A worse or equal): `{paired.get('p_a_worse', float('nan')):.3f}`",
                "",
                "> Overlapping CI / high P ⇒ difference may be noise.",
                "",
            ]
        )

    # Velocity bias (cancellation check)
    vel = slices[slices["axis"] == "velocity_tier"] if not slices.empty else pd.DataFrame()
    if not vel.empty:
        lines.extend(["## Velocity tiers (cancellation check)", ""])
        lines.append("| tier | n | rmsle | bias_pct | sle_share |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in vel.iterrows():
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['rmsle']:.4f} |"
                f" {r['bias_pct']:.2f} | {r['sle_share']:.3f} |"
            )
        signs = set(np.sign(vel["bias_pct"].to_numpy()))
        if len(signs - {0.0}) > 1:
            lines.append("")
            lines.append(
                "**Bias sign flip across velocity tiers** — global bias may cancel"
                " opposing segment errors."
            )
        lines.append("")

    # Worst families
    fam = slices[slices["axis"] == "family"] if not slices.empty else pd.DataFrame()
    if not fam.empty:
        worst = fam.nlargest(10, "rmsle")
        lines.extend(["## Worst families by RMSLE", ""])
        lines.append("| family | n | rmsle | bias_pct | sle_share |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in worst.iterrows():
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['rmsle']:.4f} |"
                f" {r['bias_pct']:.2f} | {r['sle_share']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Multiaccuracy audit",
            "",
            f"- residual Ridge R²: `{multiacc.get('r2', float('nan')):.4f}`"
            f" (train={multiacc.get('n_train')}, val={multiacc.get('n_val')})",
        ]
    )
    if multiacc.get("r2") is not None and not np.isnan(multiacc.get("r2", float("nan"))):
        if multiacc["r2"] > 0.05:
            lines.append("- **Nontrivial residual structure** — main model misses signal.")
        else:
            lines.append("- Low residual R² — little linear structure left in residuals.")
    if multiacc.get("top_coefs"):
        lines.append(
            "- Top residual coefs: "
            + ", ".join(f"`{c['feature']}={c['coef']:.3f}`" for c in multiacc["top_coefs"][:5])
        )
    lines.append("")

    if error_leaves:
        lines.extend(["## Error tree (high-error leaves)", ""])
        for leaf in error_leaves[:5]:
            lines.append(
                f"- leaf `{leaf['leaf_id']}`: n={leaf['n']},"
                f" mean|log-err|={leaf['mean_abs_log_err']:.4f},"
                f" rmsle={leaf['rmsle']:.4f}"
            )
        lines.extend(["", "```", error_leaves[0].get("tree_excerpt", ""), "```", ""])

    if clusters:
        lines.extend(["## Residual clusters (top 20% |error|)", ""])
        for c in clusters:
            lines.append(
                f"- cluster {c['cluster']}: n={c['n']}, rmsle={c['rmsle']:.4f},"
                f" bias_pct={c['bias_pct']:.2f},"
                f" families={c['top_families']}, tiers={c['velocity_tier_mix']}"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def write_audit_outputs(
    out_dir: Path,
    *,
    summary: dict[str, Any],
    slices: pd.DataFrame,
    report_md: str,
) -> None:
    """Write summary.json, slices.csv, report.md under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    slices.to_csv(out_dir / "slices.csv", index=False)
    (out_dir / "report.md").write_text(report_md)
