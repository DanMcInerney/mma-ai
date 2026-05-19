# Engineered Features Smoothing Analysis

Analysis of which engineered features use raw vs smoothed data in the main.py pipeline.

## Pipeline Flow (main.py lines 627-686)

```
1. copy_to_derived()                    [Raw data: sig_str_land, sig_str_att, etc.]
2. BetaBinomialCalculator()             [Creates: ko_smooth, win_smooth, sub_land_smooth, ctrl_smooth, decision_smooth]
3. PoissonGammaCalculator()             [Creates: sig_str_land_smooth, td_land_smooth, etc.]
4. rename_smoothed_columns()            [Original → _raw, _smooth → original]
5. ─────────────── RAW DATA STILL AVAILABLE ────────────────
6. TotalCalculator()                    [Uses: smoothed counts] ✓ CORRECT
7. AccuracyCalculator()                 [Uses: _raw counts] ✓ CORRECT - Beta-Binomial smoothing
8. DefenseCalculator()                  [Uses: already-smoothed accuracy] ✓ CORRECT
9. ─────────────── delete_raw_columns() ────────────────
10. PerMinCalculator()                   [Uses: smoothed counts] ✓ CORRECT
11. RatioCalculator()                    [Uses: smoothed counts] ✓ CORRECT
12. PressureCalculator()                 [Uses: smoothed counts] ✓ CORRECT
```

## Feature-by-Feature Analysis

### ✓ AccuracyCalculator (line 644) - CORRECT
**What it does**: Beta-Binomial smoothing of accuracy (landed/attempted)

**Data source**: `_raw` columns (acc_calc.py:199-201)
```python
land_raw, att_raw → Beta-Binomial smoothing → sig_str_acc
```

**Why this is correct**:
- Accuracy is **binomial data** (each attempt is success/failure)
- Beta-Binomial is the **proper conjugate prior** for proportions
- Smooths towards a prior accuracy rate (e.g., 45% UFC average)
- Formula: `(prior_acc * tau + landed_raw) / (tau + attempted_raw)`

**Note**: This creates a **separate smoothed accuracy feature**, independent of the Poisson-smoothed counts.

---

### ✓ DefenseCalculator (line 646) - CORRECT
**What it does**: Calculates defense as `1 - opponent_accuracy`

**Data source**: Already-smoothed `_acc` columns (def_calc.py:70)

**Why this is correct**:
- Simple complement transformation of smoothed accuracy
- No additional smoothing needed
- Defense inherits the Beta-Binomial smoothing from accuracy

---

### ✓ TotalCalculator (line 642) - CORRECT
**What it does**: Cumulative career totals (running sum)

**Data source**: Smoothed counts (total_calc.py:48 excludes `'raw'`)

**Why this is correct**:
- Cumulative totals should be of **stable, smoothed values**
- Summing raw counts would propagate noise
- Comment in main.py says "ignore _raw" ✓

---

### ✓ PerMinCalculator (line 652) - CORRECT
**What it does**: Divides counts by time to get per-minute rates

**Data source**: Smoothed counts (per_min_calc.py:40 excludes `'_raw'`)

**Why this is correct**:
- Poisson-Gamma smoothing already models counts as **rates per minute**
- Mathematical model: `smoothed_count = time * smoothed_rate`
- Dividing by time recovers the smoothed rate: `smoothed_count / time = smoothed_rate`
- The smoothing is rate-aware, so division is just unit conversion

**From poisson_gamma_smoothing_calc.py:22-26**:
```
Weight class rate: μ_w = Σ counts / Σ exposure_time
Posterior rate: λ_post = (μ_w * τ + X) / (τ + t)
Smoothed count: X_smooth = t * λ_post
```

---

### ✓ RatioCalculator (line 654) - CORRECT
**What it does**: `fighter_stat / (fighter_stat + opponent_stat)`

**Data source**: Smoothed counts (ratio_calc.py:38 excludes `'_raw'`)

**Why this is correct**:
- Calculates ratios of already-smoothed values
- Both numerator and denominator are smoothed
- Produces stable, bounded ratios [0, 1]
- Alternative would be to smooth the ratio itself, but that's more complex

---

### ✓ PressureCalculator (line 657) - CORRECT
**What it does**: `sig_str_land_rd1 / sig_str_land`

**Data source**: Smoothed counts (pressure_calc.py:12)

**Why this is correct**:
- Ratio of two smoothed counts (round 1 / total)
- Both are Poisson-Gamma smoothed
- Measures proportion of output in round 1
- Similar logic to RatioCalculator

---

## Summary

**All engineered features are using the correct data source! ✓**

### Key Pattern Identified

1. **Accuracy is special** - Uses Beta-Binomial on raw data because it's modeling binomial trials
2. **All other derived features** - Use smoothed counts because:
   - They're transformations of already-smoothed data
   - The smoothing has already incorporated exposure/time properly
   - Using smoothed inputs creates stable derived features

### No Issues Found

The current implementation correctly:
- Applies Beta-Binomial smoothing to binary outcomes (ko, win, decision)
- Applies Poisson-Gamma smoothing to count data (strikes, takedowns)
- Applies Beta-Binomial smoothing to accuracy from raw counts
- Calculates all other derived features from smoothed values
- Preserves raw data during the critical window (lines 638-649)
- Cleans up raw data after accuracy calculation

### Tuning Validation

The comprehensive tuning script optimized:
- **49 stats** across 8 weight classes
- **Beta-Binomial parameters** for binary outcomes (9 stats)
- **Poisson-Gamma parameters** for count data (22 stats)
- **Accuracy parameters** for Beta-Binomial smoothing of ratios (18 stats)

All parameters were tuned on **raw fight data** from `fight_stats_fe` (2014-2023), ensuring no circular reasoning in the optimization.
