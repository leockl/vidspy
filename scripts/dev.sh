#!/bin/bash
# ViDSPy Development Script
# Usage: ./scripts/dev.sh [command]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

case "${1:-help}" in
    install)
        echo "Installing vidspy in development mode..."
        pip install -e ".[all]"
        ;;
    
    test)
        echo "Running tests..."
        pytest tests/ -v "${@:2}"
        ;;
    
    test-cov)
        echo "Running tests with coverage..."
        pytest tests/ -v --cov=vidspy --cov-report=html --cov-report=term-missing
        ;;
    
    lint)
        echo "Running linters..."
        ruff check vidspy/
        mypy vidspy/
        ;;
    
    format)
        echo "Formatting code..."
        black vidspy/ tests/ examples/
        isort vidspy/ tests/ examples/
        ;;
    
    build)
        echo "Building package..."
        rm -rf dist/ build/ *.egg-info
        python -m build
        ;;
    
    publish-test)
        echo "Publishing to TestPyPI..."
        python -m build
        twine upload --repository testpypi dist/*
        ;;
    
    publish)
        echo "Publishing to PyPI..."
        python -m build
        twine upload dist/*
        ;;
    
    clean)
        echo "Cleaning build artifacts..."
        rm -rf dist/ build/ *.egg-info .pytest_cache/ .mypy_cache/ htmlcov/
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
        find . -type f -name "*.pyc" -delete 2>/dev/null || true
        ;;
    
    setup-vbench)
        echo "Setting up VBench models..."
        python -c "from vidspy import setup_vbench_models; setup_vbench_models()"
        ;;
    
    help|*)
        echo "ViDSPy Development Script"
        echo ""
        echo "Usage: ./scripts/dev.sh [command]"
        echo ""
        echo "Commands:"
        echo "  install      Install vidspy in development mode"
        echo "  test         Run tests"
        echo "  test-cov     Run tests with coverage report"
        echo "  lint         Run linters (ruff, mypy)"
        echo "  format       Format code (black, isort)"
        echo "  build        Build package"
        echo "  publish-test Publish to TestPyPI"
        echo "  publish      Publish to PyPI"
        echo "  clean        Clean build artifacts"
        echo "  setup-vbench Setup VBench models"
        echo "  help         Show this help message"
        ;;
esac
