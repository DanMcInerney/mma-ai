# Parameter Optimization - Usage Guide

## Quick Start

### First Time Setup

When you run `main.py` for the first time, it will automatically check for optimized parameters:

```bash
python main.py
```

**What happens:**

1. **File doesn't exist** → Runs optimization (30-60 min) → Saves to `config/optimized_parameters.json`
2. **File exists & valid** → Uses cached parameters (<5 sec)
3. **File exists but stale** → Reoptimizes automatically

## How It Works

### 1. Automatic Cache Management

`main.py` checks `config/optimized_parameters.json` before smoothing:

```python
# main.py lines 637-649
should_optimize, reason = should_run_optimization(conn)
if should_optimize:
    print(f"⚙️  Parameter optimization needed: {reason}")
    run_parameter_optimization(conn)  # Generates config/optimized_parameters.json
else:
    print(f"✓ Using cached parameters: {reason}")

param_loader = get_default_parameter_loader()  # Loads from config/
```

### 2. Cache Validation

The cache is considered **valid** when:
- ✅ `config/optimized_parameters.json` exists
- ✅ Training period matches: `2014-01-01 to 2023-01-01`
- ✅ Fight count within ±100 of database
- ✅ `FORCE_REOPTIMIZE` not set

The cache is **invalid** (triggers reoptimization) when:
- ❌ File doesn't exist
- ❌ Training period mismatch
- ❌ Fight count changed by >100
- ❌ `FORCE_REOPTIMIZE=1` environment variable set

### 3. Parameter Resolution (Per-Weightclass)

When smoothing a stat, the system follows this priority:

```
1. Per-weightclass parameter (if exists)
   └─> config/optimized_parameters.json → beta_binomial.per_weightclass.featherweight.sub_land

2. Global optimized parameter
   └─> config/optimized_parameters.json → beta_binomial.global.sub_land

3. Baseline hardcoded parameter
   └─> libs/parameter_optimization/loaders/parameter_loader.py → BASELINE_BETA_BINOMIAL
```

**Example:**
- Featherweight `sub_land`: Uses τ=3.0 (per-weightclass override)
- Lightweight `sub_land`: Uses τ=9.0 (global parameter)
- If file missing: Uses τ=13.98 (baseline fallback)

## Usage Modes

### Mode 1: Optimized (Default)

Uses parameters from `config/optimized_parameters.json`:

```bash
python main.py
# OR explicitly:
PARAM_MODE=optimized python main.py
```

### Mode 2: Baseline

Ignores `config/` and uses hardcoded parameters:

```bash
PARAM_MODE=baseline python main.py
```

**Use case**: A/B testing, debugging, or validating optimization improvements

### Mode 3: Force Reoptimization

Ignores cache and regenerates parameters (even if file is valid):

```bash
FORCE_REOPTIMIZE=1 python main.py
```

**Use case**:
- Database significantly changed
- Want to test new optimization algorithm
- Suspect cached parameters are incorrect

## Manual Parameter Generation

To regenerate parameters outside of `main.py`:

```bash
# Default: Saves to config/optimized_parameters.json
python tuning/comprehensive_likelihood_tuner.py

# Custom output location
python tuning/comprehensive_likelihood_tuner.py --output-dir custom_config/

# Custom training period
python tuning/comprehensive_likelihood_tuner.py --start-date 2015-01-01 --end-date 2024-01-01
```

## A/B Testing Example

Compare baseline vs optimized parameters:

```bash
# Run with baseline
PARAM_MODE=baseline python main.py
# Check model performance...

# Run with optimized
PARAM_MODE=optimized python main.py
# Check model performance...

# Compare results
```

## File Structure

```
config/
├── optimized_parameters.json    # Generated file (auto-created if missing)
├── README.md                     # What this directory contains
└── USAGE.md                      # This file
```

## Troubleshooting

### "Cache is invalid: Training period mismatch"

The cached file was generated with a different training period. Either:
- Let it reoptimize automatically (recommended)
- Delete `config/optimized_parameters.json` to force regeneration

### "Cache is invalid: Fight count changed significantly"

Database has grown/shrunk by >100 fights. Either:
- Let it reoptimize automatically (recommended)
- Increase `FIGHT_COUNT_TOLERANCE` in `libs/parameter_optimization/config.py`

### "No such file or directory: config/optimized_parameters.json"

This is **normal** on first run! The system will automatically generate the file.

### Want to inspect parameters

```bash
# Pretty-print the JSON
cat config/optimized_parameters.json | python -m json.tool | less

# Check metadata
cat config/optimized_parameters.json | python -m json.tool | grep -A 5 metadata
```

## Performance

- **Cache hit** (file valid): ~5 seconds overhead
- **Cache miss** (needs optimization): ~30-60 minutes
  - Beta-binomial optimization: ~15-30 min
  - Poisson-gamma optimization: ~15-30 min
  - Per-weightclass validation: Included in above

## Advanced: Custom Parameter Sets

You can maintain multiple parameter configurations:

```bash
# Save current optimized params
cp config/optimized_parameters.json config/optimized_2024_01_01.json

# Run with custom config by replacing the active file
cp config/optimized_2024_01_01.json config/optimized_parameters.json
python main.py
```
