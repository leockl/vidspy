# Contributing to ViDSPy

Thank you for your interest in contributing to ViDSPy! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/vidspy.git
   cd vidspy
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode:**
   ```bash
   pip install -e ".[all]"
   ```

4. **Verify installation:**
   ```bash
   python -c "import vidspy; print(vidspy.__version__)"
   ```

## Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **Ruff** for linting
- **mypy** for type checking

Run formatting and linting:
```bash
./scripts/dev.sh format
./scripts/dev.sh lint
```

## Testing

Run the test suite:
```bash
./scripts/dev.sh test
```

Run with coverage:
```bash
./scripts/dev.sh test-cov
```

### Test Categories

- **Unit tests**: Fast tests that don't require external services
- **Integration tests**: Require VBench, OpenRouter, or other external services
  - Skip with: `pytest -m "not integration"`
- **GPU tests**: Require CUDA-capable GPU
  - Skip with: `pytest -m "not gpu"`

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for any new functionality
3. **Update documentation** as needed
4. **Run the test suite** and ensure all tests pass
5. **Run formatters and linters** to ensure code quality
6. **Submit a pull request** with a clear description

### PR Checklist

- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted with Black/isort
- [ ] Linting passes (Ruff, mypy)
- [ ] All tests pass

## Adding New Optimizers

To add a new optimizer:

1. Create a new class in `vidspy/optimizers.py` inheriting from `VidOptimizer`
2. Implement the `compile()` method
3. Add the optimizer to the `AVAILABLE_OPTIMIZERS` dict in `vidspy/core.py`
4. Add tests in `tests/test_optimizers.py`
5. Update documentation in README.md

Example:
```python
class VidNewOptimizer(VidOptimizer):
    """Your new optimizer description."""
    
    def __init__(self, metric: Callable, **kwargs):
        super().__init__(metric)
        # Initialize your optimizer
    
    def compile(self, module, trainset, valset=None, **kwargs):
        # Implement optimization logic
        return optimized_module
```

## Adding New Metrics

To add new VBench metrics:

1. Add the metric name to the appropriate list in `vidspy/metrics.py`:
   - `QUALITY_METRICS` for video-only metrics
   - `ALIGNMENT_METRICS` for prompt-conditioned metrics
2. Add any specific handling in `VBenchInterface`
3. Add tests in `tests/test_metrics.py`
4. Update the metrics table in README.md

## Documentation

- **README.md**: Main documentation, quick start, examples
- **Docstrings**: All public classes and functions should have docstrings
- **Type hints**: Use type hints for all function signatures

## Reporting Issues

When reporting issues, please include:

1. **ViDSPy version**: `python -c "import vidspy; print(vidspy.__version__)"`
2. **Python version**: `python --version`
3. **Operating system**: (e.g., Ubuntu 22.04, macOS 14, Windows 11)
4. **Steps to reproduce**
5. **Expected vs actual behavior**
6. **Full error traceback** (if applicable)

## Feature Requests

Feature requests are welcome! Please:

1. Check existing issues to avoid duplicates
2. Describe the use case clearly
3. Explain why existing features don't meet your needs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue for any questions about contributing!
