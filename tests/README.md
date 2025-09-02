# DIRE-JAX Test Suite

## Overview

The DIRE-JAX test suite provides comprehensive testing for the JAX-based dimensionality reduction implementation. Tests include unit tests, integration tests, and performance benchmarks to validate functionality and scalability.

## Test Structure

```
tests/
├── unit/                   # Core functionality tests
│   ├── test_dire.py       # JAX backend tests
│   └── test_utils.py      # Utility function tests
├── test_mpa_comparison.py # MPA algorithm validation
├── benchmark_memory.py    # Memory usage benchmarks
├── check_memory_methods.py # Memory method validation
└── simple_test.py         # Quick validation tests
```

## Running Tests

### Quick Start

```bash
# Run all unit tests (default)
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run all tests (unit + integration + benchmarks)
python run_tests.py --all

# Run specific test file
python run_tests.py --test tests/unit/test_dire.py
```

### Test Categories

#### Unit Tests
Core functionality testing for JAX implementation:

```bash
# Run unit tests with coverage
python run_tests.py --coverage -v

# Using pytest directly
pytest tests/unit/ --cov=dire_jax --cov-report=term-missing
```

#### Integration Tests
Test algorithm behavior and mathematical correctness:

```bash
# Run integration tests
python run_tests.py --integration

# Specific integration tests
pytest tests/test_mpa_comparison.py -v
```

### Command Line Options

```bash
python run_tests.py [OPTIONS]

Options:
  -h, --help            Show help message
  -c, --coverage        Enable coverage reporting
  -v, --verbose         Verbose output
  -i, --integration     Run integration tests
  -a, --all            Run all test suites
  -t, --test PATH      Run specific test file/directory
  -l, --list           List all available tests
```

## Test Descriptions

### Unit Tests

**test_dire.py**
- JAX backend core functionality
- Parameter validation
- Transform correctness
- Edge cases handling
- Force computation validation
- k-NN graph construction

**test_utils.py**
- Utility function correctness
- Data preprocessing
- Distance calculations
- Helper functions

### Integration Tests

**test_mpa_comparison.py**
- MPA (Mixture of Posteriors Approximation) validation
- Attraction/repulsion force calculations
- Numerical stability checks
- Algorithm convergence

**benchmark_memory.py**
- Memory usage patterns
- JAX memory allocation
- Performance benchmarks

**check_memory_methods.py**
- Memory management validation
- JAX device memory handling

## Testing Requirements

### Dependencies

```bash
# Core testing
pip install pytest pytest-cov

# Optional for enhanced testing
pip install pytest-benchmark  # Performance benchmarking
pip install pytest-timeout    # Timeout handling
pip install pytest-xdist      # Parallel test execution
```

### Hardware Requirements

**Minimum:**
- 8GB RAM
- Python 3.8+

**Recommended:**
- 16GB+ RAM
- Multi-core CPU for JAX optimization

## Writing New Tests

### Test Template

```python
import pytest
import numpy as np
from dire_jax import DiRe

class TestNewFeature:
    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        return np.random.randn(1000, 100).astype(np.float32)
    
    def test_feature_correctness(self, sample_data):
        """Test feature produces correct output."""
        reducer = DiRe(n_components=2)
        embedding = reducer.fit_transform(sample_data)
        
        assert embedding.shape == (1000, 2)
        assert np.isfinite(embedding).all()
```

### Best Practices

1. **Use Fixtures:** Share test data and setup across tests
2. **Parametrize:** Test multiple configurations efficiently
3. **Mark Tests:** Use pytest markers for categorization
4. **Check Edge Cases:** Empty data, single point, high dimensions
5. **Validate Numerics:** Check for NaN, Inf, numerical stability

### Test Markers

```python
# Mark slow tests
@pytest.mark.slow

# Mark tests requiring large memory
@pytest.mark.large_memory

# Run with: pytest -m "not slow"
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: python run_tests.py --coverage
```

## Performance Baselines

Expected performance for JAX backend:

### JAX Backend Performance
| Dataset Size | Dimensions | Time | Memory |
|-------------|------------|------|--------|
| 1K points | 100D | <1s | <1GB |
| 10K points | 100D | <5s | <2GB |
| 50K points | 100D | <30s | <8GB |
| 10K points | 1000D | <10s | <4GB |

## Troubleshooting

### Common Issues

**Memory Issues:**
- Reduce dataset size for testing
- Use smaller batch sizes
- Clear JAX cache between tests

**Test Timeouts:**
- Increase timeout limits
- Use smaller test datasets
- Skip slow tests: `pytest -m "not slow"`

**Import Errors:**
- Verify JAX installation
- Check Python path
- Install optional dependencies

**Numerical Instabilities:**
- Use FP32 for critical tests
- Add tolerance to assertions
- Check condition numbers

### Debug Mode

```bash
# Run with detailed output
pytest -vvs tests/unit/test_dire.py

# Run with pdb on failure
pytest --pdb tests/unit/test_dire.py

# Profile test execution
pytest --profile tests/test_mpa_comparison.py
```

## Contributing

When adding new features or fixing bugs:

1. Write tests first (TDD approach)
2. Ensure all existing tests pass
3. Add integration tests for new features
4. Update benchmarks if performance-related
5. Document test coverage in PR

Test coverage goals:
- Unit tests: >90% coverage
- Integration tests: All major workflows
- Performance tests: Regression detection