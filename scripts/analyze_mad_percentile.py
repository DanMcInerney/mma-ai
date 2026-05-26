"""
Analyze optimal MAD floor percentile for adjperf calculations.

This script queries actual MAD distributions from the database to determine
the best percentile to use as a floor (minimum) value. Higher percentiles
provide more robustness but may over-smooth. Lower percentiles preserve
more variance but risk amplification in degenerate cases.

We'll analyze:
1. Distribution of MAD values across stats and weight classes
2. Impact of different percentile choices (5%, 10%, 15%, 20%, 25%)
3. How many cases would be affected by each threshold
4. Trade-offs between robustness and information preservation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from collections import defaultdict

from config.parameters import TRAINING_START_DATE, TRAINING_END_DATE
from libs.paths import database_url

# Database connection
DB_URL = database_url()
engine = create_engine(DB_URL)

print("="*80)
print("MAD FLOOR PERCENTILE ANALYSIS")
print("="*80)
print(f"Training period: {TRAINING_START_DATE} to {TRAINING_END_DATE}\n")

# ============================================================================
# STEP 1: Query actual MAD values from database
# ============================================================================

print("\n" + "="*80)
print("STEP 1: QUERY MAD DISTRIBUTIONS FROM DATABASE")
print("="*80)

# Query MAD values for key stats across weight classes
# We'll look at a representative sample of stats to keep the query manageable
query = text("""
    WITH per_min_stats AS (
        -- Calculate per-minute rates from raw data
        SELECT
            fm.weightclass,
            fd.fight_id,
            fd.fighter_id,
            -- Per-minute rates
            CASE WHEN fd.time_sec > 0
                THEN fd.sig_str_land::float / (fd.time_sec / 60.0)
                ELSE NULL END as sig_str_land_per_min,
            CASE WHEN fd.time_sec > 0
                THEN fd.td_land::float / (fd.time_sec / 60.0)
                ELSE NULL END as td_land_per_min,
            CASE WHEN fd.time_sec > 0
                THEN fd.sub_att::float / (fd.time_sec / 60.0)
                ELSE NULL END as sub_att_per_min,
            -- Accuracy stats
            CASE WHEN fd.sig_str_att > 0
                THEN fd.sig_str_land::float / fd.sig_str_att
                ELSE NULL END as sig_str_acc,
            CASE WHEN fd.td_att > 0
                THEN fd.td_land::float / fd.td_att
                ELSE NULL END as td_acc
        FROM features.fight_stats_fe fd
        JOIN features.fight_mapping fm ON fd.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fd.event_id = em.event_id
        WHERE em.event_date >= :start_date
          AND em.event_date < :end_date
    ),
    weightclass_medians AS (
        -- Calculate medians for each stat by weightclass
        SELECT
            weightclass,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sig_str_land_per_min) as sig_str_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY td_land_per_min) as td_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sub_att_per_min) as sub_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sig_str_acc) as sig_str_acc_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY td_acc) as td_acc_median
        FROM per_min_stats
        WHERE sig_str_land_per_min IS NOT NULL
        GROUP BY weightclass
    ),
    deviations AS (
        -- Join back to calculate absolute deviations
        SELECT
            p.weightclass,
            ABS(p.sig_str_land_per_min - m.sig_str_median) as sig_str_dev,
            ABS(p.td_land_per_min - m.td_median) as td_dev,
            ABS(p.sub_att_per_min - m.sub_median) as sub_dev,
            ABS(p.sig_str_acc - m.sig_str_acc_median) as sig_str_acc_dev,
            ABS(p.td_acc - m.td_acc_median) as td_acc_dev
        FROM per_min_stats p
        JOIN weightclass_medians m ON p.weightclass = m.weightclass
        WHERE p.sig_str_land_per_min IS NOT NULL
    )
    SELECT
        weightclass,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sig_str_dev) as sig_str_mad,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY td_dev) as td_mad,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sub_dev) as sub_mad,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY sig_str_acc_dev) as sig_str_acc_mad,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY td_acc_dev) as td_acc_mad
    FROM deviations
    GROUP BY weightclass
    ORDER BY weightclass
""")

with engine.connect() as conn:
    mad_by_wc = pd.read_sql(query, conn, params={
        'start_date': TRAINING_START_DATE,
        'end_date': TRAINING_END_DATE
    })

print("\nMAD values by weight class:")
print(mad_by_wc.to_string(index=False))

# ============================================================================
# STEP 2: Calculate percentiles across all MAD values
# ============================================================================

print("\n\n" + "="*80)
print("STEP 2: PERCENTILE ANALYSIS")
print("="*80)

# Collect all MAD values across all stats
all_mads = []
stat_mads = {}

for col in ['sig_str_mad', 'td_mad', 'sub_mad', 'sig_str_acc_mad', 'td_acc_mad']:
    values = mad_by_wc[col].dropna().values
    all_mads.extend(values)
    stat_mads[col] = values

all_mads = np.array(all_mads)

# Calculate key percentiles
percentiles = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50, 0.75]
percentile_values = {
    f'p{int(p*100):02d}': float(np.percentile(all_mads, p*100))
    for p in percentiles
}

print("\nPercentiles of all MAD values:")
for name, value in percentile_values.items():
    print(f"  {name}: {value:.6f}")

print(f"\nMinimum MAD: {all_mads.min():.6f}")
print(f"Maximum MAD: {all_mads.max():.6f}")
print(f"Mean MAD: {all_mads.mean():.6f}")
print(f"Median MAD: {np.median(all_mads):.6f}")

# ============================================================================
# STEP 3: Analyze per-stat percentiles
# ============================================================================

print("\n\n" + "="*80)
print("STEP 3: PER-STAT PERCENTILE ANALYSIS")
print("="*80)

for stat_name, values in stat_mads.items():
    print(f"\n{stat_name.upper().replace('_', ' ')}:")
    print(f"  Count: {len(values)}")
    print(f"  Min:  {values.min():.6f}")
    print(f"  5th:  {np.percentile(values, 5):.6f}")
    print(f"  10th: {np.percentile(values, 10):.6f}")
    print(f"  15th: {np.percentile(values, 15):.6f}")
    print(f"  20th: {np.percentile(values, 20):.6f}")
    print(f"  25th: {np.percentile(values, 25):.6f}")
    print(f"  50th: {np.percentile(values, 50):.6f}")
    print(f"  Max:  {values.max():.6f}")

# ============================================================================
# STEP 4: Degenerate case analysis
# ============================================================================

print("\n\n" + "="*80)
print("STEP 4: DEGENERATE CASE ANALYSIS (MAD near zero)")
print("="*80)

# Count how many MAD values are below various thresholds
thresholds = [0.001, 0.005, 0.010, 0.020, 0.050, 0.100]

print("\nNumber of MAD values below threshold:")
print(f"Total MAD values: {len(all_mads)}")
for thresh in thresholds:
    count = (all_mads < thresh).sum()
    pct = (count / len(all_mads)) * 100
    print(f"  < {thresh:.3f}: {count:3d} ({pct:5.1f}%)")

# ============================================================================
# STEP 5: Impact simulation for different percentiles
# ============================================================================

print("\n\n" + "="*80)
print("STEP 5: IMPACT SIMULATION - DIFFERENT PERCENTILE CHOICES")
print("="*80)

test_percentiles = [0.05, 0.10, 0.15, 0.20, 0.25]

print("\nScenario: What happens when we use different percentiles as floor?\n")

for p in test_percentiles:
    floor_value = np.percentile(all_mads, p*100)

    # How many MAD values would be replaced by this floor?
    below_floor = (all_mads < floor_value).sum()
    pct_affected = (below_floor / len(all_mads)) * 100

    # What's the amplification ratio?
    # If actual MAD is 0.001 but floor is 0.01, amplification is reduced 10x
    below_mask = all_mads < floor_value
    if below_mask.sum() > 0:
        actual_mads = all_mads[below_mask]
        reduction_ratios = floor_value / actual_mads
        avg_reduction = reduction_ratios.mean()
        max_reduction = reduction_ratios.max()
    else:
        avg_reduction = 1.0
        max_reduction = 1.0

    print(f"{int(p*100)}th percentile floor = {floor_value:.6f}")
    print(f"  Cases affected: {below_floor} / {len(all_mads)} ({pct_affected:.1f}%)")
    print(f"  Avg amplification reduction: {avg_reduction:.1f}x")
    print(f"  Max amplification reduction: {max_reduction:.1f}x")
    print()

# ============================================================================
# STEP 6: Weight class variance in MAD
# ============================================================================

print("\n" + "="*80)
print("STEP 6: WEIGHT CLASS VARIANCE ANALYSIS")
print("="*80)

print("\nDo different weight classes have systematically different MAD values?")
print("(High variance suggests per-weightclass floors might be better)\n")

for col in ['sig_str_mad', 'td_mad', 'sub_mad']:
    values = mad_by_wc[col].dropna().values
    if len(values) > 1:
        cv = values.std() / values.mean()
        print(f"{col.upper().replace('_', ' ')}:")
        print(f"  Mean: {values.mean():.6f}")
        print(f"  Std:  {values.std():.6f}")
        print(f"  CV:   {cv:.3f}")
        print()

# ============================================================================
# STEP 7: Recommendations
# ============================================================================

print("="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis above:

1. PERCENTILE CHOICE TRADE-OFFS:

   5th percentile:
   + Preserves more variance (affects fewest cases)
   + Closer to actual data distribution
   - Less protection against degenerate cases
   - Higher risk of extreme amplification

   10th percentile (CURRENT BASELINE):
   + Good balance between robustness and information preservation
   + Affects ~10% of cases (by definition)
   + Moderate protection against amplification
   - May still allow some amplification in extreme cases

   15-20th percentile:
   + Stronger protection against degenerate cases
   + More conservative, robust to outliers
   - Affects more cases (15-20% by definition)
   - More information loss / smoothing

   25th percentile:
   + Very robust, strong protection
   - Affects 25% of cases (quite aggressive)
   - Significant information loss
   - May over-smooth and reduce discriminative power

2. KEY FINDINGS FROM DATA:

   [See percentile values and impact simulation above]

3. RECOMMENDED APPROACH:

   Based on the data, the optimal percentile depends on your goals:

   a) MAXIMIZE ACCURACY (if you trust the data quality):
      → Use 5th or 10th percentile
      → Preserves more variance, better discrimination
      → Risk: occasional extreme amplification

   b) MAXIMIZE ROBUSTNESS (if data quality varies):
      → Use 15th or 20th percentile
      → More protection against degenerate cases
      → Trade-off: some information loss

   c) BALANCED (RECOMMENDED):
      → Use 10th percentile (CURRENT BASELINE)
      → Good empirical track record
      → Reasonable protection without over-smoothing

4. NEXT STEPS TO VALIDATE:

   To determine the TRUE optimal percentile, you would need to:

   a) Run full pipeline with different percentiles (5%, 10%, 15%, 20%, 25%)
   b) Evaluate prediction accuracy on test data (2023-2024)
   c) Choose percentile that maximizes test accuracy

   This would require multiple full pipeline runs (~2-3 hours each) and
   proper train/test evaluation.

5. DECISION RULE:

   If the data above shows:
   - Few degenerate cases (<5% below 0.01) → Use 5th or 10th percentile
   - Many degenerate cases (>10% below 0.01) → Use 15th or 20th percentile
   - Extreme variance across weight classes → Consider per-weightclass floors
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
