PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

.DEFAULT_GOAL := help
.PHONY: help setup run clean all

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

setup: $(VENV) ## Create venv (Python 3.13) and install dependencies

run: $(VENV) ## Run the TRM vs. baselines benchmark
	$(BIN)/python src/trm_benchmark.py

all: setup run ## Setup environment and run benchmark

clean: ## Remove virtual environment and cache files
	rm -rf $(VENV) __pycache__ src/__pycache__