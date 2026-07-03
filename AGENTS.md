# AGENTS.md — kcom-store-sales

Kaggle Store Sales — Time Series Forecasting. Metric: **RMSLE**.

## Commands

All Python invocations **must** be prefixed with `uv run`.

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev` + editable install + kaggle auth |
| `make download` | Fetch competition CSVs into `data/` |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train model |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run pytest tests/ -v` |
| `make lint` | `ruff check src/ scripts/ tests/` |
| `make format` | `ruff format ... --check` |
| `make format-fix` | Apply ruff formatting |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle |

Verification: `make lint && make format && make test`.

## Architecture

- **Package** `src/store_sales/`: `data.py` (loaders + merges), `features.py` (`TimeSeriesFeatureEngineer` with lags/rolling/date features), `models.py` (`TimeSeriesModel`), `metrics.py` (`rmsle`), `tracking.py` (run logger).
- **Data flow**: `load_data()` returns dict of tables → `merge_tables()` joins stores/oil/holidays/transactions → `create_lag_features()` builds lags/rolling on sorted store-family history → `timeseries_split()` holds last N days for validation → `TimeSeriesModel.fit()` trains on pre-split data.
- **Config-driven**: YAML in `config/` controls features, model hyperparams, time-series split.
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` with frozen `config.yaml`, `metrics.json`, `models/model.joblib`, and `submission.csv`.

## Conventions

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- `data/` is gitignored; data must be downloaded via `make download`.
- Lag features require the full sorted store-family history — must be created **before** train/val split.
- `rmsle()` clamps negative predictions to 0.
