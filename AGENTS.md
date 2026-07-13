# AGENTS.md — kcom-store-sales

Kaggle Store Sales — Time Series Forecasting. Metric: **RMSLE**.

## Commands

All Python invocations **must** be prefixed with `uv run`.

| Command | Purpose |
|---|---|
| `make install` | `uv sync --extra dev --extra nixtla --extra toto` + editable install + kaggle auth |
| `make download` | Fetch competition CSVs into `data/` |
| `make train CONFIG=config/foo.yaml RUN_NAME=bar` | Train LightGBM model |
| `make smoke-e2e` | Dogfood: smoke config train via shared `kaggle_ml.tracking` |
| `make train-nixtla CONFIG=config/nixtla.yaml RUN_NAME=bar` | Train nixtla stats baseline |
| `make train-toto CONFIG=config/toto.yaml RUN_NAME=bar` | Zero-shot TOTO foundation model forecast |
| `make benchmark` | Train LightGBM + nixtla + TOTO, print comparison table |
| `make benchmark-trim` | Log-Ridge + XGBoost with/without pre-activation zero trim |
| `make benchmark-toto` | Run TOTO zero-shot forecast only |
| `make kaggle-kernel` | Generate Kaggle notebook kernel script (self-contained) |
| `make compare` | Show existing run metrics table (no training/submission) |
| `make compare-cv` | Multi-window CV with recursive evaluation on baseline config |
| `make submit-best` | Pick lowest-RMSLE run, submit to Kaggle |
| `make predict` | `uv run python scripts/predict.py $(ARGS)` |
| `make test` | `uv run python -m pytest tests/ -v` |
| `make lint` | `uv run python -m ruff check src/ scripts/ tests/` |
| `make format` | `uv run python -m ruff format ... --check` |
| `make format-fix` | Apply ruff formatting |
| `make submit SUBMISSION_FILE=... SUBMISSION_MSG="..."` | Upload to Kaggle |
| `make viz-gif` | Render `assets/pipeline.gif` from latest experiment runs |

Verification: `make lint && make format && make test`.

## Architecture

- **Package** `src/store_sales/`: `data.py` (loaders + merges), `features.py` (`TimeSeriesFeatureEngineer` with lags/rolling/date features), `models.py` (`TimeSeriesModel`), `metrics.py` (wraps `kaggle_ml.evaluation.rmsle`), `tracking.py` (re-exports `kaggle_ml.tracking`), `nixtla_pipeline.py` (Nixtla long-format stats/ML baselines), `viz.py` (animated pipeline GIF).
- **Shared library**: editable path dep on `../../kaggle-ml` — run tracking and RMSLE come from `kaggle_ml`; domain merges stay local.
- **Data flow**: `load_data()` returns dict of tables → `merge_tables()` joins stores/oil/holidays/transactions → optional `apply_preprocessing()` (e.g. `trim_pre_activation_zeros`) → `create_lag_features()` builds lags/rolling on sorted store-family history → `timeseries_split()` holds last N days for validation → `TimeSeriesModel.fit()` trains on pre-split data.
- **Nixtla data flow**: `merge_tables()` → `to_long()` reshapes to `unique_id="{store}_{family}", ds, y` → `cross_validate()` scores stats models (SeasonalNaive/Theta/AutoETS) → `fit_predict()` generates forecast → `to_submission()` pivots back to Kaggle `id, sales` layout.
- **TOTO data flow**: `merge_tables()` → `_to_wide()` pivots to (date × series) matrix → `_forecast_wide()` optionally trims to `context_length` (tuned: 768) and chunks variates by `variate_batch_size` (256) for memory safety → `Toto2Model.forecast()` zero-shot median quantile → `_to_submission()` vectorized mapping to Kaggle `id, sales` layout. All exogenous variables are ignored (TOTO 2.0 zero-shot uses only target history). Tuned params: `context_length=768`, `variate_batch_size=256`, `log_transform=false`, `decode_block_size=null` (single pass, horizon ≤ patch_size). Val RMSLE: 0.4597.
- **Config-driven**: YAML in `config/` controls features, model hyperparams, time-series split, and `preprocessing.trim_pre_activation_zeros`. `config/baseline.yaml` (LightGBM), `config/xgboost.yaml` (XGBoost), `config/nixtla.yaml` (stats), `config/experiments/*` (variants), `config/benchmark-leading-zero-trim.yaml` (trim A/B).
- **Each `make train`** creates `outputs/runs/<timestamp>_<name>/` with frozen `config.yaml`, `metrics.json`, `models/model.joblib`, and `submission.csv`.

## Conventions

- `from __future__ import annotations` used in all source files.
- Ruff: line-length 100, `target-version = "py311"`, rules E/F/I.
- `data/` is gitignored; data must be downloaded via `make download`.
- Lag features require the full sorted store-family history — must be created **before** train/val split.
- `rmsle()` clamps negative predictions to 0.
