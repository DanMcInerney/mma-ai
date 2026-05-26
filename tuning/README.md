# Comprehensive Smoothing Parameter Tuning

This folder contains the comprehensive likelihood-based tuning system for optimizing smoothing parameters across all three calculator types in the UFC prediction system.

## Quick Start

### Prerequisites

1. **Database Setup**: Ensure PostgreSQL is running with the `mma-ai` database
2. **Raw Data**: Must have `features.fight_stats_fe` table populated with raw fight data
3. **Date Range**: Default uses 2014-2023 for training (configurable)

### Running the Tuner

**Basic command (recommended):**
```bash
uv run python tuning/comprehensive_likelihood_tuner.py
```

**With verbose logging:**
```bash
uv run python tuning/comprehensive_likelihood_tuner.py --verbose
```

**Custom date range:**
```bash
uv run python tuning/comprehensive_likelihood_tuner.py \
  --start-date 2015-01-01 \
  --end-date 2024-01-01
```

**Custom output directory:**
```bash
uv run python tuning/comprehensive_likelihood_tuner.py \
  --output-dir data/tuning_results_2025
```

### Command Line Options

- `--db`: Database connection string (default: `DATABASE_URL` from `.env` or the repo default)
- `--output-dir`: Output directory for results (default: `data/comprehensive_tuning`)
- `--start-date`: Training start date in YYYY-MM-DD format (default: `2014-01-01`)
- `--end-date`: Training end date in YYYY-MM-DD format (default: `2023-01-01`)
- `--verbose`: Enable verbose logging for debugging

### Expected Runtime

- Full tuning: **10-30 minutes** depending on hardware
- Optimizes **49 stats × 8 weight classes = 392 optimizations**
- Uses time series cross-validation with 3 different configurations per stat

## Core Files

### `comprehensive_likelihood_tuner.py` ⭐ **Main Tuning Script**
The unified script for comprehensive likelihood-based parameter optimization across all smoothing types.

**What it does:**
- **Beta-Binomial tuning**: Binary outcomes (ko, win, decision, sub_land, ctrl)
- **Poisson-Gamma tuning**: Count data (striking, grappling counts)
- **Accuracy tuning**: Landed/attempted ratios (sig_str, head, body, leg, ground, clinch, distance, td, sub)
- Per-weight class optimization with proper statistical validation
- Time-series cross-validation to prevent data leakage
- Stability checks and improvement thresholds (≥0.5%)
- Handles both base and _rd1 variants separately

**Features:**
- Proper predictive likelihood functions (Beta-Binomial and Negative Binomial PMFs)
- Multiple CV configurations for stability validation
- Boundary detection and expanded tau ranges
- Comprehensive statistical reporting with per-weight class recommendations

## Output Files

The tuner generates three files in the output directory (default: `data/comprehensive_tuning/`):

### 1. `optimized_parameters.json` ⭐ **Main Output**

This is the critical file containing all optimized tau values ready for use. Structure:

```json
{
  "metadata": {
    "training_period": "2014-01-01 to 2023-01-01",
    "n_stats_tuned": 49,
    "n_weight_classes": 8,
    "total_optimizations": 392
  },
  "beta_binomial": {
    "per_weightclass": {
      "flyweight": {"ko": 22.44}
    },
    "global": {
      "ko": 7.29,
      "win": 43.11,
      "decision": 60.0,
      ...
    },
    "n_stats": 9
  },
  "poisson_gamma": {
    "per_weightclass": {...},
    "global": {
      "sig_str": 0.98,
      "head": 0.98,
      "kd": 20.0,
      ...
    },
    "n_stats": 22
  },
  "accuracy": {
    "per_weightclass": {...},
    "global": {
      "sig_str_acc": 18.67,
      "head_acc": 13.5,
      ...
    },
    "n_stats": 18
  }
}
```

**Understanding the structure:**
- **`per_weightclass`**: Parameters that showed ≥0.5% improvement over global for specific weight classes
- **`global`**: Default parameters used for all weight classes (or as fallback)
- Only weight classes with statistically significant improvements get per-class parameters

### 2. `detailed_results.json`

Contains full diagnostic information for each optimization:
- Log-likelihood scores
- Brier scores and calibration errors
- Effective sample sizes
- Shrinkage factors
- Tau stability metrics
- Search ranges and boundary detection

Useful for debugging and understanding why certain parameters were chosen.

### 3. `TUNING_SUMMARY.txt`

Human-readable summary report with:
- Overview of tuning process
- Recommendations for per-weight class vs global parameters
- Statistical validation results
- Implementation guidance

## Applying Results to Calculators

After tuning completes, you need to manually update the three calculator files with the new parameters.

### Step 1: Review the Output

```bash
cat data/comprehensive_tuning/optimized_parameters.json
```

### Step 2: Update Beta-Binomial Calculator

Edit `libs/feature_store/calculators/beta_binomial_calc.py`:

1. Find the `self.pseudo_counts` dictionary (around line 186)
2. Update the global parameters from `optimized_parameters.json` → `beta_binomial` → `global`
3. If there are per-weight class parameters, update those dictionaries too
4. Add a comment with the date: `# Updated YYYY-MM-DD with optimized values tuned from RAW fight data (fight_stats_fe)`

**Example:**
```python
# Global parameters (fallback for unknown weight classes)
# Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
self.pseudo_counts = {
    'ko': 7.29,       # From tuning
    'win': 43.11,     # From tuning
    'decision': 60.0, # From tuning
    'ctrl': 50.0,     # From tuning
    ...
}
```

### Step 3: Update Poisson-Gamma Calculator

Edit `libs/feature_store/calculators/poisson_gamma_smoothing_calc.py`:

1. Find the `self.pseudo_minutes` dictionary (around line 294)
2. Update from `optimized_parameters.json` → `poisson_gamma` → `global`
3. Add date comment

**Example:**
```python
# Global parameters (fallback for unknown weight classes)
# Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
self.pseudo_minutes = {
    'sig_str': 0.98,  # From tuning
    'head': 0.98,     # From tuning
    'kd': 20.0,       # From tuning
    ...
}
```

### Step 4: Update Accuracy Calculator

Edit `libs/feature_store/calculators/acc_calc.py`:

1. Find the `self.acc_tau` dictionary (around line 140)
2. Update from `optimized_parameters.json` → `accuracy` → `global`
3. Add date comment

**Example:**
```python
# Global parameters (fallback for unknown weight classes)
# Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
self.acc_tau = {
    'sig_str': 18.67,    # From tuning
    'head': 13.5,        # From tuning
    'td': 6.88,          # From tuning
    ...
}
```

### Step 5: Test the Changes

Run the feature pipeline to ensure everything works:

```bash
uv run python main.py
```

### Step 6: Commit Changes

```bash
git add libs/feature_store/calculators/
git commit -m "Update smoothing parameters from comprehensive tuning

- Tuned on raw data from fight_stats_fe (2014-2023)
- 49 stats optimized across 8 weight classes
- Used time-series CV with proper validation"
```

## Implementation Files

The optimized parameters are implemented in:
- `libs/feature_store/calculators/beta_binomial_calc.py` - Binary outcome smoothing (9 stats)
- `libs/feature_store/calculators/poisson_gamma_smoothing_calc.py` - Count data smoothing (22 stats)
- `libs/feature_store/calculators/acc_calc.py` - Accuracy ratio smoothing (18 stats)

All calculators use weight class aware tau values with proper statistical validation and fallback to global parameters where per-class benefits are minimal.

## Methodology

### Statistical Framework

- **Predictive Likelihood**: Uses proper Beta-Binomial and Negative Binomial PMFs (not ad-hoc loss functions)
- **Cross-Validation**: Time-series splits with gaps (50, 30, 70) to prevent data leakage
- **Statistical Rigor**: Multiple CV configurations, stability checks, improvement thresholds (≥0.5%)
- **Comprehensive Coverage**: Handles all three smoothing types in unified framework

### Data Requirements

- **Source**: `features.fight_stats_fe` table (raw fight data, NOT smoothed)
- **Joins**:
  - `features.fight_mapping` for weight class information
  - `features.event_mapping` for temporal ordering
- **Filters**:
  - Date range: 2014-2023 (configurable)
  - Weight classes: 8 main divisions only
  - Valid fights: `time_sec > 0`
- **Sample sizes**: Minimum 200 fights per weight class for per-class optimization

### Optimization Process

For each stat in each weight class:

1. **Load data**: Query raw counts/outcomes with exposure times
2. **Time-series CV**: Split data maintaining temporal order with anti-leakage gaps
3. **Grid search**: Test 25-30 tau values across adaptive range
4. **Boundary detection**: Expand search if optimum hits edge
5. **Stability check**: Optimize across 3 different CV configs, use median tau
6. **Statistical validation**: Compare per-class vs global improvement
7. **Decision**: Use per-class only if ≥0.5% improvement and <20% CV variation

### Why This Works

**No circular reasoning**: All parameters optimized from raw data in `fight_stats_fe`, never from smoothed values.

**Proper Bayesian inference**: Uses conjugate priors with exact predictive distributions, not approximate methods.

**Prevents overfitting**: Time-series CV with gaps, multiple validation metrics, stability checks, and conservative improvement thresholds.

**Handles sparse data**: Adaptive search ranges, minimum sample sizes, and graceful fallback to global parameters.

## Troubleshooting

### Issue: "Database not found"

```
psycopg2.OperationalError: database "mma-ai" does not exist
```

**Solution**: Check database name or create it:
```bash
# List databases
psql -U postgres -l

# Use correct database name in --db parameter
uv run python tuning/comprehensive_likelihood_tuner.py --db postgresql://your local Postgres credentials@localhost:5432/YOUR_DB_NAME
```

### Issue: "Loaded 0 fight records"

**Causes:**
1. Wrong database or schema
2. No data in date range
3. Weight class names don't match

**Solution**: Verify your data:
```sql
-- Check data exists
SELECT COUNT(*) FROM features.fight_stats_fe;

-- Check date range
SELECT MIN(em.event_date), MAX(em.event_date)
FROM features.fight_stats_fe fe
JOIN features.event_mapping em ON fe.event_id = em.event_id;

-- Check weight classes (must be lowercase)
SELECT DISTINCT fm.weightclass, COUNT(*)
FROM features.fight_stats_fe fe
JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
GROUP BY fm.weightclass;
```

### Issue: "TypeError: Object of type bool is not JSON serializable"

This was a bug in earlier versions. Update to the latest version of the script which converts numpy booleans to Python booleans.

### Issue: Tuning is very slow

**Expected**: 10-30 minutes for full tuning

**If slower**:
- Check if database queries are slow (add indexes on `fight_id`, `event_id`, `event_date`)
- Reduce date range with `--start-date` and `--end-date`
- Run on a machine with more CPU cores

### Issue: Results look unreasonable

Check the detailed diagnostics:
```bash
cat data/comprehensive_tuning/detailed_results.json | grep -A 5 "stat_name"
```

Look for:
- **boundary_hit: true** - Tau hit search limit, may need wider range
- **tau_stability > 50%** - Unstable across CV folds, use global instead
- **n_samples < 200** - Insufficient data for per-class optimization

## When to Re-tune

Re-run tuning when:

1. **New data added**: After scraping several months of new fights
2. **Rule changes**: UFC changes significantly affect fight dynamics
3. **Bug fixes**: If you discover issues with the raw data
4. **Different objectives**: Optimizing for different metrics or time periods

**Recommendation**: Re-tune every 6-12 months or after ~500 new fights added to database.

## References

- **Beta-Binomial**: Conjugate prior for binomial data (binary outcomes, attempts/successes)
- **Negative Binomial**: Predictive distribution for Poisson-Gamma (count data with exposure)
- **Time Series CV**: sklearn.model_selection.TimeSeriesSplit with gap parameter
- **Calibration**: Predictions should match observed frequencies (e.g., 70% predictions → 70% outcomes)
