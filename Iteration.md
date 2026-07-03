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

- **More lag windows** — try 56/84/365 day seasonal lags
- **Rolling features** — expanding window means, min/max over longer horizons
- **More models** — XGBoost, CatBoost, ensemble stacking
- **Hyperparameter tuning** — Optuna search over learning_rate, num_leaves, subsample
- **Feature engineering** — moving average cross-overs, price elasticity interactions (oil × family)
- **External data** — add weather, exchange rates
- **Time-series CV** — expanding window CV instead of fixed holdout
- **Post-processing** — clip outliers, cap at max historical sales per store-family
