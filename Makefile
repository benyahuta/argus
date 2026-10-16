.PHONY: data features experiments figures all test lint

data:
	bash scripts/download_data.sh

features:
	uv run python scripts/build_dataset.py

experiments:
	uv run python scripts/run_experiments.py

figures:
	uv run python scripts/make_figures.py

all: data features experiments figures

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
