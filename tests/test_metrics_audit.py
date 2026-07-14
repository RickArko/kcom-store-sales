"""Tests for store_sales.metrics_audit (synthetic frames only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from store_sales.metrics_audit import (
    assign_velocity_tiers,
    bootstrap_rmsle,
    compute_row_metrics,
    enrich_val_frame,
    error_tree_leaves,
    filter_validation,
    multiaccuracy_audit,
    paired_bootstrap_delta,
    render_report,
    residual_clusters,
    shift_summary,
    slice_scores,
    write_audit_outputs,
)


def _synthetic_history(n_days: int = 30, n_stores: int = 3, n_families: int = 3) -> pd.DataFrame:
    rows = []
    families = [f"F{i}" for i in range(n_families)]
    for d in range(n_days):
        date = pd.Timestamp("2017-01-01") + pd.Timedelta(days=d)
        for s in range(1, n_stores + 1):
            for fam in families:
                # Fast movers: higher store × family index
                base = 10.0 * s * (families.index(fam) + 1)
                sales = base + (d % 7)
                rows.append(
                    {
                        "date": date,
                        "store_nbr": s,
                        "family": fam,
                        "sales": sales,
                        "type": "A" if s == 1 else "B",
                        "cluster": s,
                        "onpromotion": int(d % 5 == 0),
                        "is_holiday": int(d % 11 == 0),
                        "dcoilwtico": 40.0 + d * 0.1,
                        "city": "Quito",
                        "state": "Pichincha",
                    }
                )
    return pd.DataFrame(rows)


def _synthetic_val_preds(
    history: pd.DataFrame, val_days: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a preds frame with intentional velocity bias (over slow / under fast)."""
    dates = np.sort(history["date"].unique())
    val_start = dates[-val_days]
    hist = history[history["date"] < val_start]
    val = history[history["date"] >= val_start].copy()
    velocity = assign_velocity_tiers(hist)
    val = val.merge(velocity, on=["store_nbr", "family"], how="left")

    preds = val[["date", "store_nbr", "family", "sales"]].rename(columns={"sales": "actual"})
    # Bias by tier: overpredict slow, underpredict fast
    tier = val["velocity_tier"].astype(str)
    factor = np.where(tier == "slow", 1.3, np.where(tier == "fast", 0.7, 1.0))
    preds["predicted"] = preds["actual"].to_numpy() * factor
    preds["split"] = "train"
    return preds, hist, val, velocity


def test_compute_row_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = compute_row_metrics(y, y)
    assert m["rmsle"] == 0.0
    assert m["bias_pct"] == 0.0


def test_assign_velocity_tiers_three_labels():
    hist = _synthetic_history()
    tiers = assign_velocity_tiers(hist)
    assert set(tiers.columns) >= {"store_nbr", "family", "velocity_mean", "velocity_tier"}
    assert set(tiers["velocity_tier"].unique()) <= {"slow", "mid", "fast"}
    assert len(tiers) == 9  # 3 stores × 3 families


def test_filter_validation_last_n_days():
    hist = _synthetic_history(n_days=20)
    preds = hist[["date", "store_nbr", "family", "sales"]].rename(columns={"sales": "actual"})
    preds["predicted"] = preds["actual"]
    preds["split"] = "train"
    # Extra test rows must be ignored
    test_row = preds.iloc[:1].copy()
    test_row["split"] = "test"
    preds = pd.concat([preds, test_row], ignore_index=True)

    val = filter_validation(preds, test_period_days=5)
    assert val["split"].eq("train").all()
    assert val["date"].nunique() == 5


def test_slice_scores_and_velocity_bias_sign_flip():
    hist = _synthetic_history()
    preds, train_fit, val_raw, velocity = _synthetic_val_preds(hist)
    enriched = enrich_val_frame(preds, hist, velocity)
    slices = slice_scores(enriched)
    assert not slices.empty
    assert set(slices["axis"]) & {"family", "velocity_tier", "store_nbr"}

    vel = slices[slices["axis"] == "velocity_tier"].set_index("slice")
    if {"slow", "fast"}.issubset(vel.index):
        # Overpredict slow ⇒ positive bias; underpredict fast ⇒ negative bias
        assert vel.loc["slow", "bias_pct"] > 0
        assert vel.loc["fast", "bias_pct"] < 0


def test_bootstrap_rmsle_ci_contains_point():
    hist = _synthetic_history()
    preds, _, _, velocity = _synthetic_val_preds(hist)
    enriched = enrich_val_frame(preds, hist, velocity)
    boot = bootstrap_rmsle(enriched, n_boot=50, seed=0)
    assert boot["val_rmsle_ci_low"] <= boot["val_rmsle"] <= boot["val_rmsle_ci_high"]
    assert boot["n_series"] == 9


def test_paired_bootstrap_identical_runs_near_zero():
    hist = _synthetic_history()
    preds, _, _, velocity = _synthetic_val_preds(hist)
    enriched = enrich_val_frame(preds, hist, velocity)
    other = enriched.copy()
    other["predicted"] = enriched["actual"]  # perfect second model
    paired = paired_bootstrap_delta(enriched, other, n_boot=40, seed=1)
    assert paired["delta_rmsle"] > 0  # A worse than perfect B
    assert paired["n_series"] > 0


def test_shift_summary_keys():
    hist = _synthetic_history()
    dates = np.sort(hist["date"].unique())
    train = hist[hist["date"] < dates[-5]]
    val = hist[hist["date"] >= dates[-5]]
    test = val.drop(columns=["sales"]).copy()
    summary = shift_summary(train, val, test)
    assert "train" in summary and "val" in summary and "test" in summary
    assert "family_mix_js_train_val" in summary
    assert summary["train"]["n_rows"] > 0


def test_discovery_methods_run():
    hist = _synthetic_history(n_days=40, n_stores=4, n_families=4)
    preds, _, _, velocity = _synthetic_val_preds(hist, val_days=8)
    enriched = enrich_val_frame(preds, hist, velocity)

    leaves = error_tree_leaves(enriched, max_depth=3, min_samples_leaf=5, top_k=3)
    assert isinstance(leaves, list)

    multi = multiaccuracy_audit(enriched, seed=0)
    assert "r2" in multi
    assert "top_coefs" in multi

    clusters = residual_clusters(enriched, n_clusters=3, top_pct=0.3, seed=0)
    assert isinstance(clusters, list)


def test_write_audit_outputs(tmp_path: Path):
    hist = _synthetic_history()
    preds, _, _, velocity = _synthetic_val_preds(hist)
    enriched = enrich_val_frame(preds, hist, velocity)
    slices = slice_scores(enriched)
    boot = bootstrap_rmsle(enriched, n_boot=20, seed=0)
    global_metrics = compute_row_metrics(
        enriched["actual"].to_numpy(), enriched["predicted"].to_numpy()
    )
    report = render_report(
        run_name="synthetic",
        global_metrics=global_metrics,
        bootstrap=boot,
        slices=slices,
        shift={"family_mix_js_train_val": 0.0, "family_mix_js_train_test": 0.0},
        error_leaves=[],
        multiacc={"r2": 0.0, "n_train": 10, "n_val": 5, "top_coefs": []},
        clusters=[],
        run_meta={"trim_pre_activation_zeros": False, "test_period_days": 5},
    )
    out = tmp_path / "metrics_audit"
    write_audit_outputs(
        out,
        summary={"bootstrap": boot, "global": global_metrics},
        slices=slices,
        report_md=report,
    )
    assert (out / "summary.json").exists()
    assert (out / "slices.csv").exists()
    assert (out / "report.md").exists()
    assert "val_rmsle" in (out / "report.md").read_text()
