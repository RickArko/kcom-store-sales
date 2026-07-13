# Iteration Workflow — Store Sales

## Quick Start

```bash
make install          # dependencies + kaggle auth (one-time)
make download         # fetch competition data (one-time)

# Run baseline
make train CONFIG=config/baseline.yaml RUN_NAME=baseline

# Compare all experiments
uv run python scripts/compare.py
```

## How It Works

### 1. Config-driven experiments

Every experiment is a single YAML file controlling features, model hyperparams, and the time-series split.

```bash
make train CONFIG=config/experiments/my_idea.yaml RUN_NAME=v001
```

### 2. Automatic run tracking

Each run creates a timestamped directory under `outputs/runs/`:

```
outputs/runs/
  20260701_100000_baseline/
    config.yaml          # frozen config
    metrics.json         # val_rmsle, params, wall time
    models/model.joblib  # serialised model
    submission.csv       # competition submission
```

### 3. Compare experiments

```bash
uv run python scripts/compare.py
```

### 4. Re-predict from a saved model

```bash
uv run python scripts/predict.py --run-dir outputs/runs/20260701_100000_baseline
```

## Iteration Log

### baseline — LightGBM with lags and date features

**Config:** `config/baseline.yaml`

| Setting | Value |
|---|---|
| Model | LightGBM (500 estimators, 63 leaves) |
| Lags | sales_{1,7,14,28} |
| Rolling | mean/std 7/14/28d |
| Date features | year, month, dayofweek, dayofmonth, quarter, weekofyear |
| Validation | Last 16 days held out |

### toto-22m-tuned — TOTO 2.0 zero-shot foundation model

**Config:** `config/toto.yaml`

| Setting | Value |
|---|---|
| Model | Datadog/Toto-2.0-22m (zero-shot, no training) |
| context_length | 768 (last 768 days) |
| variate_batch_size | 256 (memory-safe chunking) |
| decode_block_size | null (single pass, horizon ≤ patch_size) |
| log_transform | false (TOTO has internal normalization) |
| Validation | Last 16 days held out, val RMSLE = 0.4597 |

**Ablation results** (see `config/experiments/toto-*.yaml`):

| Config | val_rmsle | Notes |
|---|---|---|
| context_length=768, no log | **0.4597** | Best — 2yr lookback |
| context_length=1280, no log | 0.4609 | Similar but 16× slower |
| default (full 1664d, no log) | 0.4674 | Old default |
| context_length=896, no log | 0.4645 | |
| context_length=1024, no log | 0.4686 | |
| log + context_length=768 | 0.4703 | log1p hurts |
| context_length=512, no log | 0.4766 | Too short for yearly seasonality |
| log_transform (full ctx) | 0.4733 | log1p hurts |
| log + context_length=512 | 0.4802 | Worst |

## Workflow Reference

```bash
# Run an experiment
make train CONFIG=config/experiments/my_config.yaml RUN_NAME=my_experiment

# Compare results
uv run python scripts/compare.py
uv run python scripts/compare.py --sort-by elapsed_seconds

# Re-predict from saved model
uv run python scripts/predict.py --run-dir outputs/runs/20260701_100000_baseline

# Submit
make submit SUBMISSION_FILE=outputs/runs/20260701_100000_baseline/submission.csv \
             SUBMISSION_MSG="baseline: LightGBM lags+rolling"

# Create a new experiment config
cp config/baseline.yaml config/experiments/my_idea.yaml
# edit, then run:
make train CONFIG=config/experiments/my_idea.yaml RUN_NAME=v001
```

## Next Directions

- **TOTO ensemble** — blend TOTO zero-shot predictions with LightGBM (e.g. 0.9×LGBM + 0.1×TOTO)
- **TOTO per-family context** — tune context_length per family (BOOKS needs 128, SCHOOL SUPPLIES needs 768+)
- **More lag windows** — try 56/84/365 day seasonal lags
- **Rolling features** — expanding window means, min/max over longer horizons
- **More models** — XGBoost, CatBoost, ensemble stacking
- **Hyperparameter tuning** — Optuna search over learning_rate, num_leaves, subsample
- **Feature engineering** — moving average cross-overs, price elasticity interactions (oil × family)
- **External data** — add weather, exchange rates
- **Time-series CV** — expanding window CV instead of fixed holdout
- **Post-processing** — clip outliers, cap at max historical sales per store-family


### Run CV

```bash
# Multi-window CV with recursive evaluation
make compare-cv

# Quick smoke test with no-lag1 config
make train CONFIG=config/experiments/no-lag1.yaml RUN_NAME=no-lag1-test

# Full CV on no-lag1 variant
uv run python scripts/train.py --config config/experiments/no-lag1.yaml --cv --run-name no-lag1-cv
```
