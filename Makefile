.PHONY: data features experiments rolling deep figures report all test lint

data:
	bash scripts/download_data.sh

features:
	uv run python scripts/build_dataset.py

experiments:
	uv run python scripts/run_experiments.py
	uv run python scripts/run_posthoc.py

rolling:
	uv run python scripts/run_rolling.py

deep:
	uv run python scripts/run_deep.py
	uv run python scripts/run_hybrid.py

figures:
	uv run python scripts/make_figures.py

report: figures
	uv run python scripts/render_tables.py
	uv run python scripts/export_site.py

all: data features experiments rolling deep report

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts tests
