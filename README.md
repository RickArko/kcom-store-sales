# kcom-store-sales

Forecast store sales in Ecuador using time-series features, holiday effects, oil prices, and promotional activity.
[Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) · Metric: **RMSLE**

## Quick Start

```bash
make install          # uv sync (dev + nixtla extras) + kaggle auth (one-time)
make download         # fetch competition data (one-time)
make benchmark        # train LightGBM + nixtla stats baselines, compare
make submit-best      # submit the lower-RMSLE model to Kaggle
```

**Happy path:** `make all` (= install → download → benchmark → submit-best)

## Pipeline

```
Raw Data (train, test, stores, oil, holidays, transactions)
  → Merge tables (store metadata, oil prices, holiday flags, transactions)
  → Time-series feature engineering
      • Date features: year, month, dayofweek, quarter, is_weekend
      • Lag features: sales_{1,7,14,28}d ago
      • Rolling features: mean/std/min/max over 7/14/28/56d windows
      • External signals: dcoilwtico (oil), is_holiday, transactions, onpromotion
  → Time-based train/validation split (last 16 days held out)
  → LightGBM regressor
  → submission.csv
```

## Development

```bash
make lint           # ruff check
make format         # ruff format --check
make format-fix     # apply formatting
make test           # pytest
```

## Experiments

```bash
make train CONFIG=config/baseline.yaml RUN_NAME=baseline
make train-nixtla CONFIG=config/nixtla.yaml RUN_NAME=nixtla-stats
uv run python scripts/compare.py
```

Create your own config under `config/experiments/` (copy `baseline.yaml` and
edit features/model hyperparams), then `make train CONFIG=config/experiments/your_config.yaml`.

## Repository Structure

```
config/               # YAML configs (features, model, CV) — baseline.yaml, nixtla.yaml, experiments/
src/store_sales/       # data.py, features.py, models.py, metrics.py, tracking.py, nixtla_pipeline.py
scripts/               # train.py, train_nixtla.py, predict.py, compare.py, pick_best.py
tests/                 # unit tests
docs/                  # experiment reports
outputs/runs/          # timestamped run artifacts
```

## Kaggle API Setup

```bash
export KAGGLE_API_TOKEN=KGAT_<your-token>
echo -n "KGAT_<your-token>" > .kaggle/access_token
chmod 600 .kaggle/access_token
```

Get your token at https://www.kaggle.com/settings → API → Create New Token.
You must **join the competition** (Accept Rules) before `make download` works.
