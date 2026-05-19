# AdjPerf Parameter Optimization

## Overview

This system automatically optimizes adjperf parameters (winsorization limits and MAD floor percentile) using the same infrastructure as smoothing parameter optimization.

## What Gets Optimized

### 1. Winsorization Limits
Controls how tightly we clip extreme adjperf values:
- **Healthy distributions**: ±7.0 (baseline) - Full variance, robust outlier handling
- **Sparse distributions**: ±3-5 (adaptive) - Scales with actual MAD
- **Degenerate distributions**: ±2.5 (baseline) - Tight clipping to reduce noise amplification

### 2. MAD Floor Percentile
Controls the minimum MAD value used in adjperf denominator:
- **Baseline**: 10th percentile
- **Optimized**: Analyzed empirically (10th-20th percentile range)
- **Purpose**: Prevent division by near-zero MAD in degenerate cases

## How It Works

### Automatic Integration (Same as Smoothing)

When you run `main.py`:

```python
# Lines 637-644: Check if optimization needed
should_optimize, reason = should_run_optimization(conn)
if should_optimize:
    run_parameter_optimization(conn)  # Runs BOTH smoothing AND adjperf optimization

# Lines 646-649: Load optimized parameters
param_loader = get_default_parameter_loader()

# Lines 740, 750, 756: Use optimized parameters
MinimumMadCalculator(conn, param_loader=param_loader).run()
AdjustedPerformanceCalculator(conn, param_loader=param_loader).run()
```

### What Happens During Optimization

**Step 1: Smoothing Optimization** (~30-60 min)
- `comprehensive_likelihood_tuner.py` optimizes beta-binomial and poisson-gamma tau values
- Saves to `config/optimized_parameters.json`

**Step 2: AdjPerf Optimization** (~5-10 min)
- `adjperf_optimizer.py` analyzes raw data from `fight_stats_fe` (2014-2023)
- Calculates MAD distribution percentiles
- Analyzes degenerate cases to validate winsorization limits
- Updates `config/optimized_parameters.json` with `adjperf` section

### File Structure

```
config/optimized_parameters.json
{
  "metadata": { ... },
  "beta_binomial": { ... },      // From smoothing optimization
  "poisson_gamma": { ... },      // From smoothing optimization
  "adjperf": {                   // From adjperf optimization
    "winsorization_limits": {
      "healthy": 7.0,
      "sparse": [3.0, 5.0],
      "degenerate": 2.5
    },
    "mad_floor_percentile": 0.10,
    "quality_thresholds": {
      "mad_degenerate": 0.010,
      "mad_sparse": 0.030,
      "mode_frequency_degenerate": 0.45
    },
    "optimization_metadata": {
      "optimized_at": "2024-01-01T12:00:00",
      "training_period": "2014-01-01 to 2023-01-01",
      ...
    }
  }
}
```

## Usage Examples

### Default (Automatic Optimization)

```bash
# First run: Optimizes both smoothing and adjperf parameters (~35-70 min)
python main.py

# Subsequent runs: Uses cached parameters (<5 sec)
python main.py
```

### Force Reoptimization

```bash
# Force both smoothing and adjperf reoptimization
FORCE_REOPTIMIZE=1 python main.py
```

### Baseline Mode (No Optimization)

```bash
# Use hardcoded baseline parameters
PARAM_MODE=baseline python main.py
```

### Manual Optimization

```bash
# Run just adjperf optimization
python tuning/adjperf_optimizer.py

# Run just smoothing optimization
python tuning/comprehensive_likelihood_tuner.py
```

## Optimization Methodology

### Data Source
- **Raw unsmoothed data**: `fight_stats_fe` table (2014-2023 training period)
- **Why raw?**: Winsorization should be based on actual data variance, not smoothed estimates

### Analysis Steps

**1. MAD Distribution Analysis**
```sql
-- Calculate MAD for all stats by weightclass
SELECT
    weightclass,
    PERCENTILE_CONT(0.5) WITHIN GROUP (...) as mad
FROM features.fight_stats_fe
WHERE event_date BETWEEN '2014-01-01' AND '2023-01-01'
GROUP BY weightclass
```

Calculate percentiles: 5th, 10th, 15th, 20th, 25th
Recommend percentile based on robustness vs. amplification trade-off

**2. Winsorization Limit Analysis**
- Sample degenerate cases (e.g., heavyweight ground stats)
- Simulate adjperf distributions with different winsorization limits
- Calculate 95th and 99th percentiles of |adjperf|
- Recommend limit that captures 95-98th percentile

### Baseline Parameters (Fallback)

From `config/parameters.py`:
```python
BASELINE_ADJPERF_PARAMS = {
    'winsorization_limits': {
        'healthy': 7.0,
        'sparse': [3.0, 5.0],
        'degenerate': 2.5
    },
    'mad_floor_percentile': 0.10,
    'quality_thresholds': {
        'mad_degenerate': 0.010,
        'mad_sparse': 0.030,
        'mode_frequency_degenerate': 0.45
    }
}
```

## Integration Points

### 1. MinimumMadCalculator
**Location**: `libs/feature_store/calculators/minimum_mad_calc.py`

**What changed**:
```python
# OLD (hardcoded)
PERCENTILE_CONT(0.10) WITHIN GROUP (...)

# NEW (optimized)
adjperf_params = self.param_loader.get_adjperf_params()
mad_percentile = adjperf_params['mad_floor_percentile']
PERCENTILE_CONT({mad_percentile}) WITHIN GROUP (...)
```

### 2. AdjustedPerformanceCalculator
**Location**: `libs/feature_store/calculators/adj_perf_calc.py`

**What changed**:
```python
# OLD (hardcoded ±7.0)
GREATEST(LEAST(adjperf, 7.0), -7.0)

# NEW (optimized, currently using 'healthy' tier)
adjperf_params = self.param_loader.get_adjperf_params()
winsor_limit = adjperf_params['winsorization_limits']['healthy']
GREATEST(LEAST(adjperf, {winsor_limit}), -{winsor_limit})
```

**Note**: Currently uses 'healthy' limit for all stats. Future enhancement could classify stats by quality tier and apply tier-specific limits dynamically.

### 3. ParameterLoader
**Location**: `libs/parameter_optimization/loaders/parameter_loader.py`

**New method**:
```python
def get_adjperf_params(self) -> dict:
    """
    Get adjperf parameters.

    Resolution order (mode='optimized'):
    1. optimized_parameters.json['adjperf']
    2. BASELINE_ADJPERF_PARAMS (fallback)

    For mode='baseline': always returns baseline
    """
```

## Cache Validation

Same cache validation as smoothing parameters:
- ✅ `config/optimized_parameters.json` exists
- ✅ Training period matches: `2014-01-01 to 2023-01-01`
- ✅ Fight count within ±100 of cached value
- ✅ `FORCE_REOPTIMIZE` not set

Cache is shared across smoothing and adjperf parameters for simplicity.

## Performance

- **Cache hit** (parameters valid): <5 seconds overhead
- **Cache miss** (needs optimization):
  - Smoothing optimization: ~30-60 minutes
  - AdjPerf optimization: ~5-10 minutes
  - **Total**: ~35-70 minutes

## Future Enhancements

### Per-Quality-Tier Winsorization

Currently uses 'healthy' winsorization limit for all stats. Could be enhanced to:
1. Classify each stat × weightclass by quality tier in SQL
2. Apply tier-specific winsorization limits dynamically:
   ```sql
   CASE
       WHEN quality_tier = 'degenerate' THEN LEAST(adjperf, 2.5)
       WHEN quality_tier = 'sparse' THEN LEAST(adjperf, interpolated_limit)
       ELSE LEAST(adjperf, 7.0)
   END
   ```

### Prediction-Based Optimization

Current approach uses empirical analysis. Could be enhanced to:
1. Grid search over winsorization limits and MAD percentiles
2. Run full pipeline for each combination
3. Evaluate prediction accuracy on test data (2023-2024)
4. Choose parameters that maximize accuracy

This would be more computationally expensive (~hours instead of minutes) but potentially more accurate.

## Troubleshooting

### "No optimized adjperf params found, using baseline"

**Cause**: `config/optimized_parameters.json` exists but has no 'adjperf' section

**Solution**: Run adjperf optimization manually:
```bash
python tuning/adjperf_optimizer.py
```

### "Parameter optimization failed"

**Cause**: Database connection error or optimization script error

**Solution**: Check database connection and run optimizers manually:
```bash
python tuning/comprehensive_likelihood_tuner.py  # Smoothing
python tuning/adjperf_optimizer.py              # AdjPerf
```

### Want to use only baseline winsorization

```bash
PARAM_MODE=baseline python main.py
```

This uses hardcoded ±7.0 and 10th percentile (ignores optimization).

## References

- **Original issue**: Degenerate distributions (heavyweight TD defense, etc.)
- **Statistical foundation**: `libs/feature_store/config/stat_quality_config.py`
- **Distribution analysis**: User's prior work in `analyze_distribution_quality.py`
- **Smoothing optimization**: `tuning/comprehensive_likelihood_tuner.py`
- **Integration pattern**: Same as smoothing parameters (lines 635-658 in main.py)
