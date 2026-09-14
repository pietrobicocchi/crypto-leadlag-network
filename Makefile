.PHONY: setup test lint fmt exp figures clean

setup:
	uv sync --extra dev

test:
	uv run pytest

lint:
	uv run ruff check src experiments tests
	uv run ruff format --check src experiments tests

fmt:
	uv run ruff format src experiments tests
	uv run ruff check --fix src experiments tests

# make exp EXP=exp001
exp:
	@test -n "$(EXP)" || { echo "usage: make exp EXP=exp001"; exit 1; }
	@f=$$(ls experiments/$(EXP)*.py 2>/dev/null | head -1); \
	 test -n "$$f" || { echo "no experiment matching '$(EXP)'"; exit 1; }; \
	 echo "running $$f"; uv run python "$$f"

figures:
	uv run python experiments/figures.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
