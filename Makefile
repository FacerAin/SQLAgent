.PHONY: format lint typecheck all clean

format:
	isort .
	black .

lint:
	flake8 .
	mypy .

typecheck:
	mypy .

all: format lint typecheck

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

install-dev:
	pip install -e ".[dev]"
	pre-commit install
