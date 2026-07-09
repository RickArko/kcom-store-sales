# kcom-store-sales

Forecast store sales in Ecuador using time-series features, holiday effects, oil prices, and promotional activity.
[Kaggle Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) · Metric: **RMSLE**

## Quick Start

```bash
make install          # uv sync (dev + nixtla extras) + kaggle auth (one-time)
make download         # fetch competition data (one-time)
make benchmark        # train LightGBM + nixtla stats baselines, compare
make submit-best      # pick the best run and submit to Kaggle
```

**Happy path:** `make all` (= install → download → benchmark → submit-best)

## Configuration

The Makefile loads variables from a `.env` file if present:

```bash
cp .env.example .env   # create from template (optional)
```

Edit `.env` to set your preferred defaults:

```bash
CONFIG=config/linear-log.yaml
RUN_NAME=
SUBMISSION_MSG="my baseline"
```

Command-line arguments override `.env`, which overrides the Makefile defaults.
Only `CONFIG` needs to be set — `RUN_NAME` is optional (timestamp is prepended).

## Training

```bash
make train CONFIG=config/baseline.yaml RUN_NAME=my-run          # LightGBM
make train-nixtla CONFIG=config/nixtla.yaml RUN_NAME=my-run     # Nixtla stats
make train-linear CONFIG=config/linear-log.yaml RUN_NAME=my-run # Ridge (log1p)
make train-toto CONFIG=config/toto.yaml RUN_NAME=my-run         # TOTO 2.0 zero-shot

# Quick smoke test (5 stores, fast)
make train CONFIG=config/experiments/smoke.yaml RUN_NAME=my-smoke
make train-linear CONFIG=config/linear-smoke.yaml RUN_NAME=my-smoke
```

All training scripts log `run_scope` to `metrics.json` — `"full"` for full-dataset
configs, `"smoke"` for smoke configs. New runs use whatever the config sets.

## Evaluation & Selection

```bash
# Compare all completed runs (sorted by val_rmsle)
uv run python scripts/compare.py

# Only show full-dataset runs
uv run python scripts/compare.py --scope full

# Auto-pick the lowest-RMSLE full run and copy its submission
uv run python scripts/pick_best.py --scope full

# Re-run all linear variants and print detailed metrics table
make benchmark-linear
make benchmark-linear ARGS="--output results.csv"
```

## Submission

```bash
# Submit a specific run's submission
make submit SUBMISSION_FILE=outputs/runs/<TIMESTAMP>_<NAME>/submission.csv \
            SUBMISSION_MSG="lightgbm n_est=500 lr=0.05"

# Auto-pick best full run and submit it
make submit-best

# Submit toto
make submit-toto SUBMISSION_MSG="toto-22m zero-shot"
```

`make submit-best` runs `pick_best.py --scope full` (lowest `val_rmsle` among
full-dataset runs with a submission file) then submits it to Kaggle.

### Kaggle Notebook Kernel

A self-contained, copy-pasteable TOTO 2.0 zero-shot script for Kaggle notebooks:

```bash
make kaggle-kernel    # prints the script; paste into a Kaggle notebook cell
```

Or copy `scripts/kaggle_toto_kernel.py` directly.  In Kaggle: enable **GPU** +
**Internet**, run `!pip install -q toto-2` in a cell, then paste and run the script.

## Pipeline

```
Raw Data (train, test, stores, oil, holidays, transactions)
  → Merge tables (store metadata, oil prices, holiday flags, transactions)
  → Time-series feature engineering
      • Date features: year, month, dayofweek, quarter, is_weekend, time_step
      • Lag features: sales_{1,7,14,28}d ago (or log_sales for log-Ridge)
      • Rolling features: mean/std/min/max over 7/14/28/56d windows
      • External signals: dcoilwtico (oil), is_holiday, transactions, onpromotion
  → Time-based train/validation split (last 16 days held out)
  → Model (LightGBM, Ridge, Tweedie, or Nixtla stats)
  → submission.csv
```

### Model Variants

| Model | Typical RMSLE | Train Time | Notes |
|---|---|---|---|
| LightGBM | 0.10 | ~25 min | Best accuracy |
| Log-Ridge | 0.42 | ~30 s | log1p target + log_sales lags |
| TOTO 2.0 (zero-shot) | 0.46 | ~13 s | Foundation model, no training |
| Nixtla (SeasonalNaive) | 0.51 | ~1 s | Stats-only baseline |
| Ridge (raw) | 1.38 | ~30 s | Original baseline |

## Development

```bash
make lint           # ruff check
make format         # ruff format --check
make format-fix     # apply formatting
make test           # pytest
```

## Experiments

Create your own config under `config/experiments/` (copy `baseline.yaml` and
edit features/model hyperparams), then:

```bash
make train CONFIG=config/experiments/your_config.yaml RUN_NAME=your-run
uv run python scripts/compare.py --scope full
uv run python scripts/pick_best.py --scope full && make submit
```

## Repository Structure

```
config/               # YAML configs (features, model, CV, run_scope)
src/store_sales/       # data.py, features.py, models.py, metrics.py, tracking.py, nixtla_pipeline.py, toto_pipeline.py
scripts/               # train.py, train_nixtla.py, train_linear.py, train_toto.py, predict.py
                       # compare.py, pick_best.py, plot_daily_aggregate.py, kaggle_toto_kernel.py
tests/                 # unit tests
outputs/runs/          # timestamped run artifacts (metrics.json, model.joblib, submission.csv)
outputs/runs/sandbox/  # smoke / test / debug runs (not included in --scope full)
```

## Kaggle API Setup

```bash
export KAGGLE_API_TOKEN=KGAT_<your-token>
echo -n "KGAT_<your-token>" > .kaggle/access_token
chmod 600 .kaggle/access_token
```

Get your token at https://www.kaggle.com/settings → API → Create New Token.
You must **join the competition** (Accept Rules) before `make download` works.
