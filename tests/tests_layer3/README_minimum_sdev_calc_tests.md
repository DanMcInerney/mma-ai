# MinimumSdevCalculator Tests

This directory contains tests for the `MinimumSdevCalculator` class, which is responsible for computing minimum standard deviation values for each stat by weightclass in the UFC fight prediction system.

## Test Files

- `test_minimum_sdev_calc.py`: Contains basic unit tests with mocked SQL execution
- `test_minimum_sdev_calc_real.py`: Contains tests that perform real calculations against dummy data

## Test Approach

The tests follow a two-pronged approach:

1. **Unit Tests**: Test the calculator's interface, SQL generation, and parameter handling with mocked dependencies
2. **Real Calculation Tests**: Test the actual calculation logic with dummy data to ensure mathematical correctness

## Real Calculation Tests

The `test_minimum_sdev_calc_real.py` file contains tests that:

1. Create a test subclass of `MinimumSdevCalculator` that performs real calculations on dummy data
2. Generate random test data for different stat tables (sig_str, td, sub)
3. Perform the actual 5th percentile calculation that would normally happen in the database
4. Verify the results match expected values

### Key Test Cases

1. **Basic Calculation**: Tests that the calculator correctly computes the 5th percentile of standard deviation values
2. **Decay Option**: Tests that the calculator works with both regular and time-decayed standard deviations
3. **Include/Exclude Patterns**: Tests that column filtering based on patterns works correctly
4. **Percentile Calculation**: Tests that the 5th percentile calculation is mathematically correct
5. **Run Method**: Tests that the main entry point correctly calls the precompute method
6. **Table Pattern**: Tests that table filtering based on patterns works correctly

## How to Run

Run the tests with pytest:

```bash
cd tests
python -m pytest tests_layer3/test_minimum_sdev_calc_real.py -v
```

## Implementation Details

The test implementation:

1. Creates a `RealCalculationMinimumSdevCalculator` class that extends `MinimumSdevCalculator`
2. Overrides key methods to avoid database dependencies
3. Implements the actual calculation logic in Python to match what would happen in SQL
4. Uses random test data with known characteristics
5. Verifies that the results match expected values

## Test Data Generation

The test data is generated with specific characteristics:

- Random standard deviation values within realistic ranges
- Multiple weightclasses to test grouping
- Date ranges matching the calculator's configuration
- Relationships between tables to test joining logic

## Validation Checks

The tests validate that:

1. The 5th percentile values are positive (as standard deviations should be)
2. The 5th percentile is less than the mean of the original values
3. Column filtering based on include/exclude patterns works correctly
4. The calculation is mathematically correct by comparing to a direct calculation 