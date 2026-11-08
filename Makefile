PORT ?= 8828

.PHONY: install test lint eval gate run data screenshot docker clean

install:
	uv sync --python 3.12

test:
	uv run pytest -q

lint:
	uvx ruff check .

eval:  ## baseline vs model-assisted (simulated client) on the 40 labeled contracts
	uv run contractreview eval --compare

gate:  ## the CI gate: high-severity deviation recall at least 80%
	uv run contractreview eval --gate --min-high-recall 0.8

run:  ## API and UI on http://127.0.0.1:$(PORT)
	uv run contractreview serve --port $(PORT)

data:  ## regenerate data/contracts and data/truth.jsonl
	uv run contractreview gen --out data

screenshot:  ## needs make run in another terminal, and Google Chrome
	uv run python scripts/screenshot.py --port $(PORT)

docker:
	docker build -t contract-review .
	docker run --rm -p $(PORT):8828 contract-review

clean:
	rm -rf .pytest_cache .ruff_cache reports
