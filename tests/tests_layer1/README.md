# PostgreSQL-Compatible Calculator Tests

This directory contains tests for the feature calculators that use PostgreSQL-specific functions and features.

## Testing Approach

The tests in this directory follow a comprehensive approach to ensure calculators work correctly with PostgreSQL:

1. **Unit Tests**: Test individual components of calculators without database interaction
   - SQL generation tests
   - Logic tests with mocked database connections

2. **Integration Tests**: Test calculators with a real PostgreSQL database
   - Uses isolated schemas for test isolation
   - Tests PostgreSQL-specific functions like `EXTRACT(EPOCH FROM AGE())`
   - Verifies correct data transformation

## Test Infrastructure

The test infrastructure is set up in `conftest.py` and provides:

1. **Dedicated Test Database**: Creates a unique PostgreSQL test database for the test session
   ```python
   @pytest.fixture(scope="session")
   def pg_engine():
       # Creates a dedicated test database
   ```

2. **Isolated Test Schemas**: Creates isolated schemas for each test
   ```python
   @pytest.fixture
   def pg_test_schema(pg_engine):
       # Creates an isolated schema with necessary tables
   ```

3. **Test Utilities**: Common test utilities in `test_utils.py`
   ```python
   class TestFeatureUtils(FeatureUtils):
       # A test version of FeatureUtils that doesn't query the database during initialization
   ```

## Running Tests

To run the tests, you need:

1. A PostgreSQL server running locally (or accessible via environment variables)
2. Python dependencies installed (`pytest`, `sqlalchemy`, `psycopg2`, etc.)

Set the following environment variables or use the defaults:
- `POSTGRES_USER`: PostgreSQL username (default: "postgres")
- `POSTGRES_PASSWORD`: PostgreSQL password, if your local server requires one
- `POSTGRES_HOST`: PostgreSQL host (default: "localhost")
- `POSTGRES_PORT`: PostgreSQL port (default: "5432")

### Running on Linux/macOS:
```bash
./tests/run_tests.sh
```

### Running on Windows:
```cmd
tests\run_tests.bat
```

Or directly with pytest:
```bash
pytest tests/tests_layer1/
```

## Test Examples

### AgeCalculator Tests

- `test_age_calculator_sql_generation`: Tests SQL generation without execution
- `test_age_calculator_execution`: Tests execution with a real PostgreSQL database
- `test_age_calculation_postgres_functions`: Tests PostgreSQL-specific age calculation functions

### TimeSecCalculator Tests

- `test_time_sec_calculator_get_features`: Tests data loading
- `test_time_sec_calculator_calculate`: Tests calculation logic
- `test_time_sec_calculator_save`: Tests saving results
- `test_time_sec_calculator_integration`: Tests end-to-end functionality with PostgreSQL

## Adding New Tests

When adding tests for new calculators:

1. Create a new test file named `test_<calculator_name>.py`
2. Use the `pg_test_schema` fixture for PostgreSQL integration tests
3. Use the `TestFeatureUtils` class to avoid database queries during initialization
4. Test both SQL generation and execution
5. Include assertions for expected results

## Troubleshooting

If you encounter errors related to the `features` schema:
- Make sure you're using the `TestFeatureUtils` class from `test_utils.py`
- Ensure you're patching the correct import path (`libs.feature_store.base.FeatureUtils` or `libs.feature_store.base_calculator.FeatureUtils`)
- Check that you're passing the test schema name to `TestFeatureUtils` 
