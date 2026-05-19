# Configuration Guide

This directory consolidates ALL configuration for the MMA AI project into a single, well-organized location.

## Directory Structure

```
config/
├── __init__.py                  # Main config module (re-exports everything)
├── decay.py                     # Time decay configuration
├── parameters.py                # Parameter optimization configuration
├── optimized_parameters.json    # Generated optimized tau parameters (auto-created)
├── README.md                    # Overview of config directory
├── USAGE.md                     # Usage guide for parameter optimization
└── CONFIGURATION_GUIDE.md       # This file
```

## Configuration Modules

### 1. `config.decay` - Time Decay Configuration

Controls time-decay calculations for features like averages, standard deviations, and slopes.

**Key Settings:**
- `DECAY_HALF_LIFE_YEARS = 1.25` (default)
- `DECAY_RATE = log(2) / half_life`
- `DECAY_RATE_SQL` - Pre-formatted for SQL queries

**Usage:**
```python
from config.decay import get_decay_rate, DECAY_HALF_LIFE_YEARS

# In Python code
decay_rate = get_decay_rate()  # Returns 0.5545 for 1.25 year half-life

# In SQL generation
from config.decay import get_decay_rate_sql_constant
sql = f"SELECT EXP(-{get_decay_rate_sql_constant()} * years_ago) as weight"
```

**Environment Override:**
```bash
# Override default half-life
DECAY_HALF_LIFE_YEARS=2.0 python main.py
```

**Used By:**
- `AdjustedPerformanceCalculator` (adjperf decay)
- `TimedecAvgCalculator` (time-decayed averages)
- `TimedecStdDevCalculator` (time-decayed standard deviations)
- `TimedecSlopeCalculator` (time-decayed slopes)
- `StatUpdater` (inference updates)

---

### 2. `config.parameters` - Parameter Optimization Configuration

Controls parameter optimization system for smoothing tau values.

**Key Settings:**
- `TRAINING_START_DATE = '2014-01-01'`
- `TRAINING_END_DATE = '2023-01-01'`
- `TRAINING_PERIOD = "2014-01-01 to 2023-01-01"`
- `CACHE_PATH = PROJECT_ROOT / 'config' / 'optimized_parameters.json'`
- `FIGHT_COUNT_TOLERANCE = 100` (±100 fights before cache invalidation)

**Usage:**
```python
from config.parameters import CACHE_PATH, TRAINING_PERIOD

# Check if optimized parameters exist
if CACHE_PATH.exists():
    print(f"Using cached parameters from {CACHE_PATH}")
```

**Used By:**
- `JSONParameterStore` (loads/saves optimized parameters)
- `CacheManager` (validates cache)
- `comprehensive_likelihood_tuner.py` (generates optimized parameters)

---

### 3. `config.__init__` - Unified Imports

Re-exports everything for convenient access.

**Usage:**
```python
# Import everything from one place
from config import (
    DECAY_HALF_LIFE_YEARS,
    TRAINING_PERIOD,
    CACHE_PATH
)

# Or import from specific modules
from config.decay import get_decay_rate
from config.parameters import TRAINING_START_DATE
```

---

## Migration from Old Locations

### Deprecated Modules (Still Work, Show Warnings)

The following old import paths still work but show deprecation warnings:

```python
# DEPRECATED (but still works)
from libs.feature_store.config import get_decay_rate
# WARNING: libs.feature_store.config is deprecated.
# Please import from config.decay instead.

# DEPRECATED (but still works)
from libs.parameter_optimization.config import CACHE_PATH
# WARNING: libs.parameter_optimization.config is deprecated.
# Please import from config.parameters instead.
```

### Migration Guide

**Old Import → New Import:**

```python
# Time decay configuration
OLD: from libs.feature_store.config import get_decay_rate
NEW: from config.decay import get_decay_rate

# Parameter optimization configuration
OLD: from libs.parameter_optimization.config import CACHE_PATH
NEW: from config.parameters import CACHE_PATH

# Convenience (import both)
NEW: from config import DECAY_RATE, CACHE_PATH
```

### Updated Files

The following files have been updated to use the new config location:

- ✅ `libs/feature_store/calculators/adj_perf_calc.py`
- ✅ `libs/feature_store/create_inference_data.py`
- ✅ `libs/feature_store/inference/updaters/stat_updater.py`
- ✅ `libs/parameter_optimization/storage/cache_manager.py`
- ✅ `libs/parameter_optimization/storage/json_store.py`

---

## Configuration Data Files

### `optimized_parameters.json`

**Location:** `config/optimized_parameters.json`

**Auto-Generated:** Yes (by `main.py` if missing, or manually via `comprehensive_likelihood_tuner.py`)

**Structure:**
```json
{
  "metadata": {
    "training_period": "2014-01-01 to 2023-01-01",
    "n_fights": 5234,
    "optimized_at": "2024-01-01 12:00:00"
  },
  "beta_binomial": {
    "global": {
      "ko": 23.0,
      "win": 25.0,
      "sub_land": 9.0,
      "ctrl": 2.0
    },
    "per_weightclass": {
      "featherweight": {
        "sub_land": 3.0
      },
      "light heavyweight": {
        "ctrl": 1.5
      },
      "heavyweight": {
        "ctrl": 1.5
      }
    }
  },
  "poisson_gamma": {
    "global": { ... },
    "per_weightclass": { ... }
  }
}
```

**Lifecycle:**
1. **First Run:** `main.py` detects file is missing → Runs optimization (30-60 min) → Saves to `config/`
2. **Subsequent Runs:** Validates cache (training period, fight count) → Uses cached values (<5 sec)
3. **Cache Invalid:** Automatically reoptimizes
4. **Force Reoptimize:** `FORCE_REOPTIMIZE=1 python main.py`

---

## Complete Configuration Reference

### All Available Settings

| Setting | Module | Default | Override Method |
|---------|--------|---------|-----------------|
| `DECAY_HALF_LIFE_YEARS` | `config.decay` | `1.25` | `DECAY_HALF_LIFE_YEARS=2.0` env var |
| `DECAY_RATE` | `config.decay` | `0.5545` | Calculated from half-life |
| `DECAY_RATE_SQL` | `config.decay` | `"0.5545"` | Calculated from half-life |
| `TRAINING_START_DATE` | `config.parameters` | `'2014-01-01'` | Edit `config/parameters.py` |
| `TRAINING_END_DATE` | `config.parameters` | `'2023-01-01'` | Edit `config/parameters.py` |
| `TRAINING_PERIOD` | `config.parameters` | `"2014-01-01 to 2023-01-01"` | Calculated from dates |
| `CACHE_PATH` | `config.parameters` | `config/optimized_parameters.json` | Edit `config/parameters.py` |
| `FIGHT_COUNT_TOLERANCE` | `config.parameters` | `100` | Edit `config/parameters.py` |

### Environment Variables

| Variable | Effect | Example |
|----------|--------|---------|
| `DECAY_HALF_LIFE_YEARS` | Override time decay half-life | `DECAY_HALF_LIFE_YEARS=2.0` |
| `PARAM_MODE` | Set parameter mode | `PARAM_MODE=baseline` or `PARAM_MODE=optimized` |
| `FORCE_REOPTIMIZE` | Force parameter reoptimization | `FORCE_REOPTIMIZE=1` |

---

## Best Practices

### 1. Use Centralized Config

✅ **Good:**
```python
from config.decay import get_decay_rate
```

❌ **Bad:**
```python
# Hardcoding values
decay_rate = 0.5545  # Don't do this!
```

### 2. Override via Environment Variables

For temporary changes, use environment variables:

```bash
# Test different decay rates
DECAY_HALF_LIFE_YEARS=0.5 python main.py
DECAY_HALF_LIFE_YEARS=1.0 python main.py
DECAY_HALF_LIFE_YEARS=2.0 python main.py
```

### 3. Document Configuration Changes

If you modify defaults in `config/*.py`, document why:

```python
# Increased from 1.0 to 1.25 based on 2024-01-01 validation analysis
DECAY_HALF_LIFE_YEARS = 1.25
```

### 4. Version Control

- **DO commit:** `config/*.py` (code)
- **DO commit:** `config/*.md` (documentation)
- **CONSIDER:** `config/optimized_parameters.json` (generated data)
  - Commit if you want team to share same optimized parameters
  - Add to `.gitignore` if each environment should optimize independently

---

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'config'`

**Solution:** Ensure project root is in Python path. This should be automatic, but if running scripts directly:

```python
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
```

### Deprecation Warnings

**Warning:** `libs.feature_store.config is deprecated`

**Solution:** Update imports to use new location:
```python
# Change this:
from libs.feature_store.config import get_decay_rate

# To this:
from config.decay import get_decay_rate
```

### Missing optimized_parameters.json

**Error:** `Cache is invalid: File does not exist`

**Solution:** This is normal on first run! `main.py` will automatically generate the file. Just let it run (takes 30-60 minutes).

---

## Quick Reference

### Common Imports

```python
# Time decay
from config.decay import (
    DECAY_HALF_LIFE_YEARS,  # 1.25
    DECAY_RATE,              # 0.5545
    get_decay_rate,
    get_decay_rate_sql_constant
)

# Parameter optimization
from config.parameters import (
    TRAINING_START_DATE,     # '2014-01-01'
    TRAINING_END_DATE,       # '2023-01-01'
    TRAINING_PERIOD,         # "2014-01-01 to 2023-01-01"
    CACHE_PATH,              # Path to optimized_parameters.json
    FIGHT_COUNT_TOLERANCE    # 100
)

# Everything at once
from config import (
    DECAY_HALF_LIFE_YEARS,
    TRAINING_PERIOD,
    CACHE_PATH
)
```

### Common Commands

```bash
# Run with default config
python main.py

# Override decay rate
DECAY_HALF_LIFE_YEARS=2.0 python main.py

# Force parameter reoptimization
FORCE_REOPTIMIZE=1 python main.py

# Use baseline parameters (ignore optimized)
PARAM_MODE=baseline python main.py

# Manually regenerate optimized parameters
python tuning/comprehensive_likelihood_tuner.py
```
