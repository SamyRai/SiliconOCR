.PHONY: install dev test lint format check

install:
	uv sync --all-extras

dev:
	uv run uvicorn src.api:app --reload

test:
	uv run python -m pytest

lint:
	uv run ruff check .
	uv run python -m mypy src

format:
	uv run ruff check --fix .
	uv run ruff format .

check: format lint test
