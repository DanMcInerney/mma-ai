# SQL Template System for Feature Calculators

This directory contains SQL templates used by feature calculators. The templates are organized by calculator type and are rendered using Jinja2.

## Directory Structure

```
sql_templates/
├── time_sec/
│   └── get_features.sql
├── age/
│   └── get_features.sql
├── accuracy/
│   ├── snapshots.sql
│   └── calculations.sql
└── ...
```

## Template Variables

Templates can use the following variables:

- `schema`: Database schema (default: 'features')
- Calculator-specific variables (e.g., `table_name`, `columns`, etc.)

## Example Template

```sql
-- Example template for TimeSecCalculator.get_features
SELECT 
    fm.fight_id,
    fs.fighter_id,
    fm.end_round,
    fm.end_time,
    fm.time_format
FROM {{ schema }}.fight_mapping fm
JOIN {{ schema }}.fight_stats_core fs ON fm.fight_id = fs.fight_id
```

## Usage in Calculators

```python
def get_features(self):
    query = self.context.sql_manager.render_template(
        'time_sec',  # Calculator type
        'get_features',  # Template name
        {'schema': self.context.schema}  # Template variables
    )
    self.fight_mapping = pd.read_sql(query, self.conn)
```

## Benefits

1. **Separation of Concerns**: SQL logic is separated from Python code
2. **Testability**: SQL templates can be tested independently
3. **Maintainability**: SQL queries are easier to read and maintain
4. **Performance**: SQL queries can be optimized independently
5. **Reusability**: Common SQL patterns can be reused across calculators 