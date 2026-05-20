.PHONY: install dev test lint format check

install:
	uv sync

dev:
	uv run uvicorn src.api:app --reload

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run mypy src

format:
	uv run ruff check --fix .
	uv run ruff format .

check: format lint test
