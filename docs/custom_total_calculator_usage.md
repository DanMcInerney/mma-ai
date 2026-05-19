# CustomTotalCalculator Usage Guide

The `CustomTotalCalculator` is a flexible calculator that follows the same structure as `OpponentCalculator` but calculates cumulative totals (`_total` suffix) for custom specified columns.

## Key Features

- **Custom Column Filtering**: Specify exactly which column patterns you want to calculate totals for
- **Multi-table Support**: Works across all feature-specific tables (e.g., `features.sig_str`, `features.td`, etc.)
- **Cumulative Totals**: Calculates running sums ordered by event date for each fighter
- **Safe Execution**: Includes validation and error handling

## Basic Usage

### Example 1: Calculate totals for all "_opp" columns

```python
from libs.feature_store.calculators.custom_total_calc import CustomTotalCalculator

# Calculate totals for all columns containing "_opp"
custom_columns = ['_opp']
CustomTotalCalculator(conn, custom_columns=custom_columns).run()
```

This will create columns like:
- `sig_str_land_opp` → `sig_str_land_opp_total`
- `td_land_opp` → `td_land_opp_total`
- `strikes_acc_opp` → `strikes_acc_opp_total`

### Example 2: Calculate totals for specific statistics

```python
# Calculate totals only for landing statistics
custom_columns = ['_land']
CustomTotalCalculator(conn, custom_columns=custom_columns).run()
```

This will create columns like:
- `sig_str_land` → `sig_str_land_total`
- `td_land` → `td_land_total`
- `head_land` → `head_land_total`

### Example 3: Multiple column patterns

```python
# Calculate totals for both opponent stats and accuracy stats
custom_columns = ['_opp', '_acc']
CustomTotalCalculator(conn, custom_columns=custom_columns).run()
```

### Example 4: Using include_patterns for additional filtering

```python
# Calculate totals for opponent stats, but only for sig_str table
custom_columns = ['_opp']
include_patterns = ['sig_str']
CustomTotalCalculator(conn, custom_columns=custom_columns, include_patterns=include_patterns).run()
```

## Integration with Main Pipeline

In your `main.py`, add the calculator right after the `OpponentCalculator`:

```python
# Make sure this step is AFTER ratio and opponent, BEFORE sdev and timedecay
print("Calculating opponent stats...")
OpponentCalculator(conn).run()

# Calculate totals for specific columns that contain "_opp" 
print("Calculating custom total stats for _opp columns...")
custom_columns = ['_opp']  # This will find all columns containing "_opp"
CustomTotalCalculator(conn, custom_columns=custom_columns).run()
```

## How It Works

1. **Feature Discovery**: The calculator scans all feature tables and finds columns matching your `custom_columns` patterns
2. **SQL Generation**: Creates window function SQL to calculate running sums partitioned by fighter and ordered by event date
3. **Table Updates**: Updates each feature table with new `_total` columns
4. **Validation**: Ensures all expected columns are created successfully

## Column Filtering Logic

The calculator automatically excludes certain column types from total calculations:
- `_total` (avoids totals of totals)
- `_id` (no totals for IDs)
- `_acc`, `_def`, `_ratio`, `_per_min` (no totals for rates/ratios)
- Static biographical info like `age`, `reach`, `height`

## Expected Output

For a column like `sig_str_land_opp`, the calculator will create `sig_str_land_opp_total` which contains:
- Fighter's cumulative sum of opponent significant strikes landed across all previous fights
- Ordered chronologically by event date
- Partitioned by fighter (each fighter has their own running total)

## Error Handling

The calculator includes comprehensive error handling:
- Validates table existence before processing
- Checks for column conflicts
- Provides detailed logging of success/failure
- Rollback on transaction errors

## Performance Notes

- Uses efficient window functions for cumulative calculations
- Processes multiple tables in parallel (if enabled)
- Minimal memory footprint by updating tables in place
