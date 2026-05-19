# MMA-AI-DB Tests

This directory contains tests for the MMA-AI-DB project.

## Test Structure

The tests are organized into layers:

- **Layer 1 Tests** (`tests_layer1/`): Tests for individual calculators with PostgreSQL-specific functions
- **Layer 2 Tests** (future): Tests for feature engineering pipelines
- **Layer 3 Tests** (future): Tests for model training and prediction

## Running Tests

### Prerequisites

1. PostgreSQL server running locally or accessible via network
2. Python dependencies installed (see `requirements-test.txt`)

### Configuration

Copy the `.env.example` file to `.env` and adjust the settings as needed:

```bash
cp tests/.env.example tests/.env
```

### Running Tests

#### On Windows:
```cmd
tests\run_tests.bat
```

#### On Linux/macOS:
```bash
./tests/run_tests.sh
```

#### Directly with pytest:
```bash
# Run all tests
pytest tests/

# Run specific test layer
pytest tests/tests_layer1/

# Run specific test file
pytest tests/tests_layer1/test_age_calc.py

# Run specific test
pytest tests/tests_layer1/test_age_calc.py::test_age_calculator_execution
```

## Test Layers

### Layer 1: Calculator Tests

Tests for individual calculators that use PostgreSQL-specific functions. These tests ensure that:

1. SQL generation works correctly
2. Calculation logic works correctly
3. Integration with PostgreSQL works correctly

See [tests_layer1/README.md](tests_layer1/README.md) for more details. 

# UFC Fight Prediction Tests

This directory contains the tests for the UFC fight prediction system.

## Running Tests

### Using pytest (recommended)

Tests in this project use pytest for simplified test writing and powerful assertions.

To run all tests:
```bash
pytest
```

To run tests in a specific directory:
```bash
pytest tests/test_inference
```

To run a specific test file:
```bash
pytest tests/test_inference/test_create_inference_data.py
```

### Using unittest (legacy)

Some older tests may still use the unittest framework. These can be run with:
```bash
python -m unittest discover
```

## Test Structure

The tests are organized into directories based on the component they are testing:

- `test_inference/`: Tests for inference data creation and prediction
- `tests_layer1/`: Tests for Layer 1 calculators 
- `tests_layer2/`: Tests for Layer 2 calculators
- `tests_layer3/`: Tests for Layer 3 calculators
- `tests_training_data/`: Tests for training data generation

## Writing New Tests

New tests should be written using pytest. Example:

```python
import pytest

def test_something():
    # Arrange
    expected = 42
    
    # Act
    result = calculate_answer()
    
    # Assert
    assert result == expected
```

Use fixtures for setup and teardown:

```python
@pytest.fixture
def test_data():
    # Setup
    data = {"value": 42}
    yield data
    # Teardown (optional)

def test_with_fixture(test_data):
    assert test_data["value"] == 42
``` 