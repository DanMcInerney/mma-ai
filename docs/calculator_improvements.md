# Calculator Improvements

This document outlines the improvements made to the feature calculator system to enhance performance and testability.

## Overview

We've implemented two major improvements:

1. **SQL Template System**: Separates SQL logic from Python code
2. **Dependency Injection**: Makes calculators easier to test with mock data

## SQL Template System

### Implementation

- Created a `SQLTemplateManager` class that manages SQL templates using Jinja2
- Organized templates by calculator type in `libs/feature_store/sql_templates/`
- Templates use variables like `{{ schema }}` for dynamic content

### Benefits

- **Separation of Concerns**: SQL logic is separated from Python code
- **Maintainability**: SQL queries are easier to read and maintain
- **Performance**: SQL queries can be optimized independently
- **Testability**: SQL templates can be tested independently
- **Reusability**: Common SQL patterns can be reused across calculators

### Example

```python
# Before
def get_features(self):
    self.fight_mapping = pd.read_sql("""
        SELECT 
            fm.fight_id,
            fs.fighter_id,
            fm.end_round,
            fm.end_time,
            fm.time_format
        FROM features.fight_mapping fm
        JOIN features.fight_stats_core fs ON fm.fight_id = fs.fight_id
    """, self.conn)

# After
def get_features(self):
    query = self.context.sql_manager.render_template(
        'time_sec', 
        'get_features',
        {'schema': self.context.schema}
    )
    self.fight_mapping = pd.read_sql(query, self.conn)
```

## Dependency Injection

### Implementation

- Created a `CalculatorContext` class that encapsulates:
  - Database connection
  - Feature utilities
  - SQL template manager
  - Mock data for testing

### Benefits

- **Testability**: Calculators can be tested with mock data
- **Isolation**: Tests don't need a real database
- **Speed**: Tests run faster with in-memory data
- **Flexibility**: Different implementations can be injected for different environments

### Example

```python
# Before
calculator = TimeSecCalculator(conn)
calculator.run()

# After - Real execution
context = CalculatorContext(conn, schema='features')
calculator = TimeSecCalculator(context)
calculator.run()

# After - Testing with mock data
mock_data = {
    'fight_mapping': pd.DataFrame(...),
    'fight_stats_core': pd.DataFrame(...),
    'fight_stats_fe': pd.DataFrame(...)
}
context = CalculatorContext(mock_conn, mock_data=mock_data)
calculator = TimeSecCalculator(context)
calculator.run()
```

## Testing Improvements

### Before

- Tests required a real PostgreSQL database
- Tests were slow and could fail due to database issues
- Tests were difficult to set up and maintain

### After

- Tests can run with mock data
- Tests are faster and more reliable
- Tests are easier to set up and maintain
- Tests can cover more edge cases

### Example

```python
def test_time_sec_calculator_with_context():
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame(...),
        'fight_stats_core': pd.DataFrame(...),
        'fight_stats_fe': pd.DataFrame(...)
    }
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TimeSecCalculator(context)
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'time_sec_rd1' in mock_data['fight_stats_fe'].columns
```

## Performance Improvements

- **Reduced String Manipulation**: SQL templates reduce string manipulation overhead
- **Caching**: SQL templates are cached for better performance
- **Optimized SQL**: SQL can be optimized independently of Python code
- **Reduced Database Calls**: Mock data reduces database calls during testing

## Migration Guide

To migrate an existing calculator to use the new system:

1. Create SQL templates in `sql_templates/<calculator_type>/`
2. Update the calculator to accept a `CalculatorContext` in the constructor
3. Use the context to execute queries and update tables
4. Create tests using mock data

See `calculators/time_sec_calc.py` for an example of a migrated calculator. 