# Trading Pipeline CLI Makefile

.PHONY: install install-dev test clean lint format help

# Default target
help:
	@echo "Trading Pipeline CLI - Available Commands:"
	@echo ""
	@echo "  install      Install the CLI application"
	@echo "  install-dev  Install in development mode"
	@echo "  test         Run tests"
	@echo "  clean        Clean build artifacts"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  example      Run example pipeline"
	@echo ""
	@echo "CLI Commands:"
	@echo "  make train       Quick training example"
	@echo "  make optimize    Hyperparameter optimization example"
	@echo "  make backtest    Backtesting example"
	@echo "  make pipeline    Full pipeline example"

# Installation
install:
	pip install .

install-dev:
	pip install -e .
	@echo "✅ CLI installed in development mode"
	@echo "You can now use: trading-cli --help"

# Testing
test:
	python -m pytest tests/ -v

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Code quality
lint:
	flake8 src/ --max-line-length=100
	pylint src/

format:
	black src/
	isort src/

# Examples
example:
	python examples/run_full_pipeline.py

# Quick CLI examples
train:
	@echo "🚀 Running quick training example..."
	trading-cli train --train-start 2023-06-01 --train-end 2023-08-31 --valid-start 2023-09-01 --valid-end 2023-10-31 --output-dir ./quick_models

optimize:
	@echo "🔧 Running optimization example..."
	trading-cli optimize --trials 20 --timeout 600

backtest:
	@echo "📊 Running backtest example..."
	@echo "Note: You need to provide --price-data path"
	trading-cli backtest --initial-balance 50000 --max-position 0.2 --sizing-method enhanced

pipeline:
	@echo "🔄 Running full pipeline example..."
	@echo "Note: You need to provide --price-data path"
	trading-cli pipeline --train-start 2023-06-01 --train-end 2023-08-31 --trials 10 --initial-balance 100000

# Configuration
config-show:
	trading-cli config --show

config-reset:
	trading-cli config --reset

# Development helpers
dev-setup: install-dev
	@echo "Setting up development environment..."
	pip install pytest flake8 pylint black isort
	@echo "✅ Development environment ready"

# Build distribution
build:
	python setup.py sdist bdist_wheel

# Check CLI is working
check:
	@echo "Checking CLI installation..."
	trading-cli --help
	@echo "✅ CLI is working correctly"