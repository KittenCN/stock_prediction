PYTHON ?= python
PIP ?= pip

.PHONY: setup fmt lint test run build ci clean help

help:
	@echo "Available commands:"
	@echo "  setup          Install dependencies"
	@echo "  test           Run all tests"
	@echo "  test-quick     Run quick tests only"
	@echo "  run-train      Start model training"
	@echo "  run-getdata    Get stock data"
	@echo "  run-preprocess Preprocess data"
	@echo "  clean          Clean generated files"
	@echo "  ci             Run CI pipeline"

setup:
	$(PIP) install -r requirements.txt
	@echo "Setup completed. Recommend using conda environment: stock_prediction"

fmt:
	@echo "Code formatting (placeholder - add black/ruff if needed)"

lint:
	@echo "Code linting (placeholder - add ruff/mypy if needed)"

test:
	$(PYTHON) -m pytest -v

test-quick:
	$(PYTHON) -m pytest tests/test_import.py tests/test_config.py -v

run-train:
	$(PYTHON) scripts/predict.py --mode train --model transformer --epochs 5

run-getdata:
	$(PYTHON) scripts/getdata.py --api akshare --code ""

run-preprocess:
	$(PYTHON) scripts/data_preprocess.py --pklname train.pkl

build:
	@echo "No build step required for pure Python project"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

ci: fmt lint test build
	@echo "CI pipeline completed successfully"
