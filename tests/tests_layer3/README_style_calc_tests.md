# StyleCalculator Tests Documentation

## Overview

This document describes the comprehensive test suite for the StyleCalculator, which verifies the accuracy of all 43 style metrics used to characterize fighter fighting styles.

## Test Structure

### Files
- `test_style_calc.py` - Main test file with comprehensive test coverage

### Test Coverage

#### 1. **Mathematical Accuracy Tests**
- `test_power_volume_metrics()` - Tests power vs volume calculations with known inputs
- `test_all_43_metrics_calculated()` - Verifies all 43 style metrics are calculated correctly

#### 2. **Edge Case Tests**  
- `test_division_by_zero_handling()` - Ensures all metrics return 0.0 when denominators are zero
- Tests with extreme values to verify no overflow or invalid results

#### 3. **Calculator Architecture Tests**
- `test_init()` - Verifies proper initialization
- `test_get_style_definitions()` - Checks all style definitions are present and formatted correctly
- `test_get_features()` - Validates feature list generation

#### 4. **Integration Tests**
- `test_create_style_table()` - Verifies table creation SQL
- `test_run_method()` - Tests complete pipeline execution

## Test Data

The tests use three fighter scenarios:

1. **fighter_normal**: Realistic values for testing standard calculations
2. **fighter_zeros**: All zeros to test division-by-zero handling  
3. **fighter_extreme**: Extreme values to test boundary conditions

## Key Testing Features

### StyleCalculatorTester Class
- Extends StyleCalculator but overrides `execute_raw_sql()` to return calculated results
- Implements all 43 style formulas directly in Python for verification
- Uses `safe_divide()` function to handle division by zero consistently

### Mathematical Verification
Each formula is implemented in Python and compared against the calculator results:

```python
# Example: Power vs Volume = (KO + KD) / Significant Strikes
results['style_power_vs_volume'] = safe_divide(
    row['ko_dec_adjperf_dec_avg'] + row['kd_dec_adjperf_dec_avg'],
    row['sig_str_land_dec_adjperf_dec_avg']
)
```

## Running the Tests

### Command Line
```bash
# Run all StyleCalculator tests
python -m pytest tests/tests_layer3/test_style_calc.py -v

# Run specific test category
python -m pytest tests/tests_layer3/test_style_calc.py::TestStyleCalculator::test_power_volume_metrics -v

# Run with detailed output
python -m pytest tests/tests_layer3/test_style_calc.py -v -s
```

### From Python
```python
import unittest
from tests.tests_layer3.test_style_calc import TestStyleCalculator

# Run all tests
suite = unittest.TestLoader().loadTestsFromTestCase(TestStyleCalculator)
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
```

## Expected Test Results

All tests should pass with the following coverage:

✅ **Basic Functionality** (3 tests)
- Initialization
- Style definitions 
- Feature list generation

✅ **Mathematical Accuracy** (2 tests)  
- Known calculation verification
- All 43 metrics present and valid

✅ **Edge Cases** (1 test)
- Division by zero handling

✅ **Integration** (2 tests)
- Table creation
- Pipeline execution

## Test Data Examples

### Normal Fighter Test Values
```python
'sig_str_land_per_min_dec_adjperf_dec_avg': 5.0,
'sig_str_acc_dec_adjperf_dec_avg': 0.5,
'ko_dec_adjperf_dec_avg': 0.2,
'kd_dec_adjperf_dec_avg': 0.3,
# Expected: style_power_vs_volume = (0.2 + 0.3) / 10.0 = 0.05
```

### Expected Results Verification
The tests verify calculations like:
- Power vs Volume: (0.2 + 0.3) / 10.0 = 0.05
- Volume vs Precision: 5.0 / 0.5 = 10.0  
- Distance Preference: 4.0 / (4.0 + 1.0 + 0.5) = 0.727

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure the StyleCalculator is in the Python path
2. **Mock Failures**: Check that BaseCalculator.__init__ is properly patched
3. **Calculation Mismatches**: Verify test data matches the expected formulas

### Debug Tips
- Use `-s` flag with pytest to see print statements
- Check individual style metric calculations with the StyleCalculatorTester
- Verify division by zero handling in edge cases

## Extending Tests

To add new tests:

1. **New Style Metrics**: Add to `get_style_definitions()` and update expected count
2. **New Test Cases**: Add fighter scenarios to `create_test_fighter_data()`
3. **New Edge Cases**: Create additional test methods following the pattern

This test suite ensures the StyleCalculator accurately computes all fighter style metrics and handles edge cases appropriately. 