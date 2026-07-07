# AGENTS.md — kcom-store-sales

Kaggle Store Sales — Time Series Forecasting. Metric: **RMSLE**.

## Commands

All Python invocations **must** be prefixed with `uv run`.

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev --extra nixtla` + editable install + kaggle auth |
| `make download` | Fetch competition CSVs into `data/` |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train LightGBM model |
| `make train-nixtla CONFIG=config/nixtla.yaml RUN_NAME=bar` | Train nixtla stats baseline |
| `make benchmark` | Train both LightGBM + nixtla, print comparison table |
| `make submit-best` | Pick lowest-RMSLE run, submit to Kaggle |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run pytest tests/ -v` |
| `make lint` | `ruff check src/ scripts/ tests/` |
| `make format` | `ruff format ... --check` |
| `make format-fix` | Apply ruff formatting |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle |

Verification: `make lint && make format && make test`.

## Architecture

- **Package** `src/store_sales/`: `data.py` (loaders + merges), `features.py` (`TimeSeriesFeatureEngineer` with lags/rolling/date features), `models.py` (`TimeSeriesModel`), `metrics.py` (`rmsle`), `tracking.py` (run logger), `nixtla_pipeline.py` (Nixtla long-format stats/ML baselines).
- **Data flow**: `load_data()` returns dict of tables → `merge_tables()` joins stores/oil/holidays/transactions → `create_lag_features()` builds lags/rolling on sorted store-family history → `timeseries_split()` holds last N days for validation → `TimeSeriesModel.fit()` trains on pre-split data.
- **Nixtla data flow**: `merge_tables()` → `to_long()` reshapes to `unique_id="{store}_{family}", ds, y` → `cross_validate()` scores stats models (SeasonalNaive/Theta/AutoETS) → `fit_predict()` generates forecast → `to_submission()` pivots back to Kaggle `id, sales` layout.
- **Config-driven**: YAML in `config/` controls features, model hyperparams, time-series split. `config/baseline.yaml` (LightGBM), `config/nixtla.yaml` (stats), `config/experiments/*` (variants).
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` with frozen `config.yaml`, `metrics.json`, `models/model.joblib`, and `submission.csv`.

## Conventions

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- `data/` is gitignored; data must be downloaded via `make download`.
- Lag features require the full sorted store-family history — must be created **before** train/val split.
- `rmsle()` clamps negative predictions to 0.
