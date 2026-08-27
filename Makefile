# Convenience targets. Everything here also works as a plain command; this
# exists so the documented workflow is one word instead of three lines.
.PHONY: help install test test-fast lint fmt type security check run backtest clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	pip install -r requirements-dev.txt
	pip install -e .

test:  ## Run the full test suite (~3 minutes)
	pytest -q

test-fast:  ## Unit tests only — seconds, not minutes
	pytest tests/unit -q

lint:  ## Lint and format check
	ruff check src tests scripts
	ruff format --check src tests

fmt:  ## Apply formatting
	ruff format src tests scripts
	ruff check --fix src tests scripts

type:  ## Type check
	mypy src/tradebot

security:  ## Static analysis, dependency audit, secret scan
	bandit -r src -c pyproject.toml -q
	pip-audit -r requirements.txt
	python scripts/check_secrets.py

check: lint type test security  ## Everything CI runs

run:  ## Start the engine (PAPER unless the env says otherwise)
	python -m tradebot.app.cli run

validate:  ## Validate configuration without starting
	python -m tradebot.app.cli validate-config

doctor:  ## Report environment readiness
	python -m tradebot.app.cli doctor

clean:  ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
