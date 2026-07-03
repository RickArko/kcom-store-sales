# kcom-store-sales

Forecast store sales in Ecuador using time-series features, holiday effects, oil prices, and promotional activity.
[Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) · Metric: **RMSLE**

## Quick Start

```bash
make install          # uv sync + kaggle auth (one-time)
make download         # fetch competition data (one-time)
make train            # train LightGBM → submission.csv
make submit           # upload to Kaggle + show leaderboard
```

**Happy path:** `make all`

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
make train CONFIG=config/experiments/my_idea.yaml RUN_NAME=v001
uv run python scripts/compare.py
```

## Repository Structure

```
config/               # YAML configs (features, model, CV)
src/store_sales/       # data.py, features.py, models.py, metrics.py, tracking.py
scripts/               # train.py, predict.py, compare.py
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
