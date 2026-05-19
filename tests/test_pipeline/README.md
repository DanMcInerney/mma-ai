# Full Pipeline Test

This directory contains a comprehensive test that validates the entire data processing pipeline from CSV input to database storage and feature calculation.

## Overview

The `test_full_pipeline.py` test creates dummy UFC fight data and runs it through the actual main.py logic to verify that:

1. **Data Loading Works Correctly**: CSV data is properly loaded into PostgreSQL tables
2. **Schema Creation**: All required database tables and indexes are created
3. **Data Integrity**: Loaded data matches expected values with proper transformations
4. **Calculator Functions**: Basic feature engineering calculators work with the test data
5. **Manual Verification**: Results are verified against manual calculations

## Test Data Structure

The test creates dummy data for:

- **2 Weight Classes**: Lightweight (155 lbs) and Welterweight (170 lbs)
- **6 Fighters per Weight Class**: 12 total fighters with realistic attributes
- **12 Fights**: 6 lightweight fights + 6 welterweight fights
- **Complete Round-by-Round Stats**: All required statistics for each round

### Dummy Fighters

**Lightweight (155 lbs):**
- john smith (the hammer) - Orthodox
- mike jones (lightning) - Southpaw  
- carlos rodriguez (el toro) - Orthodox
- alex petrov (the bear) - Orthodox
- tommy williams (wildcat) - Southpaw
- danny garcia (venom) - Switch

**Welterweight (170 lbs):**
- steve anderson (the crusher) - Orthodox
- frank miller (tank) - Orthodox
- ricardo silva (spider) - Southpaw
- ivan volkov (the wolf) - Orthodox
- bobby johnson (knockout) - Orthodox
- luis martinez (el fuego) - Switch

## What Gets Tested

### Database Operations
- Schema initialization (`features` schema)
- Table creation (fighter_mapping, event_mapping, fight_mapping, fight_stats_core, fight_stats_fe)
- Data loading through CoreFeatureStore
- Index creation and constraints

### Data Transformations
- Fighter name/nickname/stance conversion to lowercase
- Height conversion (e.g., "5' 10"" → 70 inches)
- Reach extraction (e.g., "72"" → 72)
- Date parsing and filtering
- Round-by-round stats parsing

### Feature Calculations
- TimeSecCalculator: Fight duration in seconds
- KOCalculator: Knockout indicators
- DecisionCalculator: Decision indicators  
- SubmissionslandCalculator: Submission success
- WinCalculator: Win/loss indicators

### Manual Verification
- Height/reach conversions are mathematically correct
- Fight duration calculations match expected values
- KO indicators match fight methods
- Win/loss assignments match fight results
- Database record counts match expected values

## Running the Test

### Option 1: Using the Test Runner
```bash
cd tests/test_pipeline
python run_pipeline_test.py
```

### Option 2: Using pytest directly
```bash
# From project root
pytest tests/test_pipeline/test_full_pipeline.py -v -s

# Run specific test method
pytest tests/test_pipeline/test_full_pipeline.py::TestFullPipeline::test_data_loading_pipeline -v -s
```

### Option 3: Using the main test runner
```bash
# From project root (Windows)
tests/run_tests.bat

# From project root (Linux/Mac)  
./tests/run_tests.sh
```

## Test Database

The test creates a temporary PostgreSQL database called `test_pipeline_db` that is:
- Created fresh for each test run
- Automatically cleaned up after tests complete
- Isolated from your main `mma-ai` database

**Database URL**: `postgresql://postgres@localhost:5432/test_pipeline_db`

## Test Structure

The test is organized into several phases:

1. **Setup Phase**: Create test database and dummy data
2. **Loading Phase**: Run actual main.py data loading logic
3. **Verification Phase**: Check database contents against expectations
4. **Calculator Phase**: Test basic feature engineering calculators
5. **Manual Verification Phase**: Verify calculations manually
6. **Cleanup Phase**: Drop test database

## Expected Output

When running successfully, you should see output like:

```
RUNNING FULL PIPELINE TEST
================================================================================
Test file: /path/to/tests/test_pipeline/test_full_pipeline.py

tests/test_pipeline/test_full_pipeline.py::TestFullPipeline::test_data_loading_pipeline 
Starting schema initialization...
Creating schema and tables...
Loading fighter features...
Loading fight features...
Loading fight stats...
Testing TimeSecCalculator...
Testing KOCalculator...
...
PASSED

tests/test_pipeline/test_full_pipeline.py::TestFullPipeline::test_basic_calculators
...
PASSED

✅ All pipeline tests passed!
```

## Troubleshooting

### Database Connection Issues
- Ensure PostgreSQL is running on localhost:5432
- Verify username/password: `your local Postgres credentials`
- Check that the postgres user has CREATE DATABASE privileges

### Import Issues  
- Run from project root directory
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that PYTHONPATH includes the project root

### Test Failures
- Check the detailed traceback for specific assertion failures
- Verify that your main.py logic hasn't changed significantly
- Ensure test database is properly cleaned up between runs

## Extending the Test

To add more test scenarios:

1. **More Fighters**: Add fighters to `dummy_individuals_df` fixture
2. **More Fights**: Add matchups to `dummy_competitions_df` fixture  
3. **More Calculators**: Add calculator tests to `test_basic_calculators`
4. **Edge Cases**: Add specific test methods for edge cases
5. **Performance**: Add timing assertions for performance testing

## Integration with CI/CD

This test can be integrated into continuous integration by:

1. Setting up a test PostgreSQL database in CI environment
2. Running the test as part of the test suite
3. Using the exit code to determine pass/fail status
4. Capturing test output for debugging failed builds
