# Every stage runs through `python -m src.cli <stage>` so the pipeline is testable and
# diffable. Stages are stubs until their feature lands (see PROJECT_LOG.md Feature Log).

UV := uv run
CLI := $(UV) python -m src.cli

.PHONY: setup lint format test data panel backtest benchmarks metrics robustness figures report \
        notebook all clean

setup:
	uv sync

lint:
	$(UV) ruff check .
	$(UV) ruff format --check .

format:
	$(UV) ruff format .
	$(UV) ruff check --fix .

test:
	$(UV) pytest

# Network: re-pulls from yfinance and rewrites data/ snapshot + manifest. Never run in CI.
data:
	$(CLI) data

panel:
	$(CLI) panel

backtest:
	$(CLI) backtest

benchmarks:
	$(CLI) benchmarks

metrics:
	$(CLI) metrics

robustness:
	$(CLI) robustness

figures:
	$(CLI) figures

report:
	$(CLI) report

notebook:
	$(UV) jupyter nbconvert --to notebook --execute --inplace notebooks/strategy.ipynb

all: panel backtest benchmarks metrics robustness figures report

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
