COMPETITION := store-sales-time-series-forecasting
DATA_DIR   := data
TOKEN_FILE := .kaggle/access_token

CONFIG          ?= config/config.yaml
RUN_NAME        ?=
SUBMISSION_FILE ?= outputs/submissions/submission.csv
SUBMISSION_MSG  ?= "baseline: LightGBM with lag/rolling features"

.PHONY: all install download train train-nixtla train-linear benchmark benchmark-linear submit-best predict submit test plot-daily lint format format-fix clean

all: install download benchmark submit-best
	@echo ""
	@echo "========================================================"
	@echo "  Happy path complete! Check leaderboard above."
	@echo "========================================================"

install: .uv_sync
	uv pip install -e .
	@$(MAKE) _ensure_kaggle_auth
	@echo ""
	@echo "All set. Run 'make download' to fetch the competition data."

download:
	@mkdir -p $(DATA_DIR); \
	$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	case "$$TOKEN" in KGAT_your-kaggle-api-token-here|"") TOKEN="";; esac; \
	[ -z "$$TOKEN" ] && TOKEN="$${KAGGLE_API_TOKEN:-}"; \
	if [ -n "$$TOKEN" ]; then export KAGGLE_API_TOKEN="$$TOKEN"; else unset KAGGLE_API_TOKEN; fi; \
	echo "Downloading $(COMPETITION) data..."; \
	uv run kaggle competitions download \
		-c $(COMPETITION) -p $(DATA_DIR) 2>&1 || { \
		exit_code=$$?; \
		echo ""; \
		echo "================================================================"; \
		echo " Download failed (403 Forbidden)."; \
		echo ""; \
		echo " Possible causes:"; \
		echo "   1. You haven't joined the competition yet."; \
		echo "      Go to the page and click 'Join' / 'Accept Rules':"; \
		echo "      https://www.kaggle.com/competitions/$(COMPETITION)"; \
		echo ""; \
		echo "   2. Your API token may be stale."; \
		echo "      Regenerate at https://www.kaggle.com/settings"; \
		echo "      then update $(TOKEN_FILE)"; \
		echo "================================================================"; \
		exit $$exit_code; \
	}; \
	echo "Extracting..."; \
	cd $(DATA_DIR) && unzip -o $(COMPETITION).zip && rm -f $(COMPETITION).zip; \
	echo "  Data ready in $(DATA_DIR)/"

train:
	@uv run python scripts/train.py --config $(CONFIG) $(if $(RUN_NAME),--run-name $(RUN_NAME),) $(ARGS)

train-nixtla:
	@uv run python scripts/train_nixtla.py --config $(CONFIG) $(if $(RUN_NAME),--run-name $(RUN_NAME),) $(ARGS)

train-linear:
	@uv run python scripts/train_linear.py --config $(CONFIG) $(if $(RUN_NAME),--run-name $(RUN_NAME),) $(ARGS)

benchmark:
	@echo "========================================================"
	@echo "  Benchmark: LightGBM baseline vs Nixtla stats baseline"
	@echo "========================================================"
	@uv run python scripts/train.py --config config/baseline.yaml --run-name bench-lightgbm
	@uv run python scripts/train_nixtla.py --config config/nixtla.yaml --run-name bench-nixtla
	@echo ""
	@echo "========================================================"
	@echo "  Benchmark results (sorted by val_rmsle, lower = better)"
	@echo "========================================================"
	@uv run python scripts/compare.py --sort-by val_rmsle

submit-best:
	@uv run python scripts/pick_best.py
	@$(MAKE) submit

predict:
	@uv run python scripts/predict.py $(ARGS)

submit:
	@$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	case "$$TOKEN" in KGAT_your-kaggle-api-token-here|"") TOKEN="";; esac; \
	[ -z "$$TOKEN" ] && TOKEN="$${KAGGLE_API_TOKEN:-}"; \
	if [ -n "$$TOKEN" ]; then export KAGGLE_API_TOKEN="$$TOKEN"; else unset KAGGLE_API_TOKEN; fi; \
	[ ! -f $(SUBMISSION_FILE) ] && { echo "ERROR: $(SUBMISSION_FILE) not found — run 'make train' first"; exit 1; }; \
	echo "Submitting $(SUBMISSION_FILE) to $(COMPETITION)..."; \
	uv run kaggle competitions submit \
		-c $(COMPETITION) \
		-f $(SUBMISSION_FILE) \
		-m "$(SUBMISSION_MSG)" && \
	echo "" && \
	echo "Submitted! Checking leaderboard..." && \
	uv run kaggle competitions leaderboard \
		-c $(COMPETITION) --show

test:
	@uv run pytest tests/ -v $(ARGS)

benchmark-linear:
	@uv run python scripts/compare_models.py $(ARGS)

plot-daily:
	@uv run python scripts/plot_daily_aggregate.py $(ARGS)

lint:
	@uv run ruff check src/ scripts/ tests/

format:
	@uv run ruff format src/ scripts/ tests/ --check

format-fix:
	@uv run ruff format src/ scripts/ tests/

.uv_sync: pyproject.toml uv.lock
	uv sync --extra dev --extra nixtla
	@touch .uv_sync

_ensure_kaggle_token:
	@mkdir -p .kaggle; \
	PLACEHOLDER="KGAT_your-kaggle-api-token-here"; \
	TOKEN=""; \
	\
	if [ -n "$$KAGGLE_API_TOKEN" ]; then \
		TOKEN="$$KAGGLE_API_TOKEN"; \
		if [ ! -f $(TOKEN_FILE) ] || [ "$$(cat $(TOKEN_FILE))" != "$$TOKEN" ]; then \
			printf '%s' "$$TOKEN" > $(TOKEN_FILE); \
			chmod 600 $(TOKEN_FILE); \
		fi; \
	elif [ -f $(TOKEN_FILE) ] && [ -s $(TOKEN_FILE) ] && ! grep -q "$$PLACEHOLDER" $(TOKEN_FILE) 2>/dev/null; then \
		TOKEN=$$(cat $(TOKEN_FILE)); \
	elif [ -f ~/.kaggle/access_token ] && [ -s ~/.kaggle/access_token ] && ! grep -q "$$PLACEHOLDER" ~/.kaggle/access_token 2>/dev/null; then \
		echo "Using credentials from ~/.kaggle/access_token."; \
		exit 0; \
	elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json"; \
		exit 0; \
	else \
		if [ -f .kaggle/access_token.example ] && [ -s .kaggle/access_token.example ]; then \
			cp .kaggle/access_token.example $(TOKEN_FILE); \
		else \
			printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
		fi; \
		chmod 600 $(TOKEN_FILE); \
		echo ""; \
		echo " Token template written to $(TOKEN_FILE)."; \
		echo ""; \
		echo " To configure:"; \
		echo "   1. Go to https://www.kaggle.com/settings"; \
		echo "   2. Under API, click 'Create New Token'"; \
		echo "   3. Copy the token (starts with KGAT_)"; \
		echo "   4. Paste it into $(TOKEN_FILE) (and nothing else)"; \
		echo "   5. Run 'make download' again to verify"; \
		exit 1; \
	fi

_ensure_kaggle_auth:
	@mkdir -p .kaggle; \
	PLACEHOLDER="KGAT_your-kaggle-api-token-here"; \
	TOKEN=""; \
	if [ -n "$$KAGGLE_API_TOKEN" ]; then \
		TOKEN="$$KAGGLE_API_TOKEN"; \
		if [ ! -f $(TOKEN_FILE) ] || [ "$$(cat $(TOKEN_FILE))" != "$$TOKEN" ]; then \
			printf '%s' "$$TOKEN" > $(TOKEN_FILE); \
			chmod 600 $(TOKEN_FILE); \
		fi; \
	elif [ -f $(TOKEN_FILE) ] && [ -s $(TOKEN_FILE) ] && ! grep -q "$$PLACEHOLDER" $(TOKEN_FILE) 2>/dev/null; then \
		TOKEN=$$(cat $(TOKEN_FILE)); \
	elif [ -f ~/.kaggle/access_token ] && [ -s ~/.kaggle/access_token ] && ! grep -q "$$PLACEHOLDER" ~/.kaggle/access_token 2>/dev/null; then \
		TOKEN=$$(cat ~/.kaggle/access_token); \
	elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json."; \
		exit 0; \
	else \
		if [ -f .kaggle/access_token.example ] && [ -s .kaggle/access_token.example ]; then \
			cp .kaggle/access_token.example $(TOKEN_FILE); \
		else \
			printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
		fi; \
		chmod 600 $(TOKEN_FILE); \
		echo ""; \
		echo " Kaggle token not configured yet (optional at install time)."; \
		echo ""; \
		echo " To configure:"; \
		echo "   1. Go to https://www.kaggle.com/settings"; \
		echo "   2. Under API, click 'Create New Token'"; \
		echo "   3. Copy the token (starts with KGAT_)"; \
		echo "   4. Paste it into $(TOKEN_FILE) (and nothing else)"; \
		echo "   5. Run 'make download' to verify"; \
		echo ""; \
		echo " Skipping auth check - 'make download' will fail until a token is set."; \
		exit 0; \
	fi; \
	echo "Verifying Kaggle auth..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions list >/dev/null 2>&1 && \
		echo "  Authenticated successfully." || \
		{ echo "  WARNING: Authentication check failed. Token may be stale."; \
		  echo "  'make download' may fail until you fix $(TOKEN_FILE)."; }

clean:
	rm -f .uv_sync
