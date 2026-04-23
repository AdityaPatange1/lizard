.PHONY: help install install-dev build clean lint format test coverage run interactive validate check typecheck

PY ?= python3
PIP ?= $(PY) -m pip

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "Lizard SIMULA — Make targets:\n\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install the package (editable) with dev extras
	$(PIP) install -e ".[dev]"

install-dev: install ## Alias for install

build: clean ## Build sdist + wheel into dist/
	$(PY) -m build

clean: ## Remove build artifacts and caches
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

lint: ## Lint with ruff and typecheck with mypy
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy src/lizard

format: ## Auto-format with ruff
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

typecheck: ## Run mypy only
	$(PY) -m mypy src/lizard

test: ## Run the test suite
	$(PY) -m pytest

coverage: ## Run tests with coverage report
	$(PY) -m pytest --cov=lizard --cov-report=term-missing

run: ## Run the interactive simulator (make run)
	$(PY) -m lizard --interactive

interactive: run ## Alias for `make run`

validate: lint test ## Full CI-style check: lint + tests
	@echo "\033[32mvalidate: OK\033[0m"

check: validate ## Alias for validate
