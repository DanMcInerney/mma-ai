# Feature Store Improvements

This directory contains the feature store implementation for the UFC fight prediction project. Recent improvements include:

## 1. SQL Template System

SQL templates are now separated from calculator code, making them easier to test, maintain, and optimize. See [sql_templates/README.md](sql_templates/README.md) for details.

## 2. Calculator Context

The `CalculatorContext` class provides dependency injection for calculators, making them easier to test with mock data.

### Usage

```python
# Real execution
context = CalculatorContext(conn, schema='features')
calculator = TimeSecCalculator(context)
calculator.run()

# Testing with mock data
mock_data = {
    'fight_mapping': pd.DataFrame(...),
    'fight_stats_core': pd.DataFrame(...),
    'fight_stats_fe': pd.DataFrame(...)
}
context = CalculatorContext(conn, mock_data=mock_data)
calculator = TimeSecCalculator(context)
calculator.run()
```

### Benefits

1. **Testability**: Calculators can be tested with mock data
2. **Isolation**: Tests don't need a real database
3. **Speed**: Tests run faster with in-memory data
4. **Flexibility**: Different implementations can be injected for different environments

## Implementation Details

### SQL Template Manager

The `SQLTemplateManager` class manages SQL templates using Jinja2:

```python
# Initialize
sql_manager = SQLTemplateManager(conn)

# Render template
sql = sql_manager.render_template(
    'calculator_type',
    'template_name',
    {'param1': value1, 'param2': value2}
)

# Validate SQL
is_valid = sql_manager.validate_sql(sql)

# Execute SQL
result = sql_manager.execute_sql(sql)
```

### Calculator Context

The `CalculatorContext` class provides a consistent interface for calculators:

```python
# Initialize
context = CalculatorContext(conn, mock_data=mock_data)

# Execute query
df = context.execute_query("SELECT * FROM table")

# Get columns
columns = context.get_columns('table_name', include_strs=['_land'], exclude_strs=['_id'])

# Update table
context.update_table('table_name', data_frame, new_columns=['col1', 'col2'])
```

## Migration Guide

To migrate an existing calculator to use the new system:

1. Create SQL templates in `sql_templates/<calculator_type>/`
2. Update the calculator to accept a `CalculatorContext` in the constructor
3. Use the context to execute queries and update tables
4. Create tests using mock data

See `calculators/time_sec_calc.py` for an example of a migrated calculator. 