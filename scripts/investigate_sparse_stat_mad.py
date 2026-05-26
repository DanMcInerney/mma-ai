"""
Investigate optimal MAD floor for sparse stats by querying actual database tables.

This script examines:
1. minimum_mad_<stat> tables - 10th percentile MAD floor per weightclass
2. <stat>_wc_mad tables - actual weightclass MAD values
3. Sparse stats like sub_att, td_att, ground_att where many fighters have 0 attempts

Goal: Determine the best floor when a stat has 0 MAD for a weightclass.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from libs.paths import database_url

DB_URL = database_url()
engine = create_engine(DB_URL)

print("="*80)
print("INVESTIGATING SPARSE STAT MAD FLOORS")
print("="*80)

# List of sparse stats to investigate
sparse_stats = ['sub_att', 'td_att', 'ground_att', 'ground_land', 'rev', 'kd']

print("\n" + "="*80)
print("STEP 1: CHECK MINIMUM_MAD TABLES (10th percentile per weightclass)")
print("="*80)

for stat in sparse_stats:
    table_name = f'minimum_mad_{stat}'

    try:
        query = text(f"""
            SELECT
                weightclass,
                {stat}_min_mad
            FROM features.{table_name}
            ORDER BY weightclass
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(f"\n{stat.upper()} - Minimum MAD (10th percentile):")
        print(df.to_string(index=False))

        # Statistics
        min_mad_col = f'{stat}_min_mad'
        if min_mad_col in df.columns:
            values = df[min_mad_col].dropna()
            if len(values) > 0:
                zeros = (values == 0).sum()
                nulls = df[min_mad_col].isna().sum()
                print(f"  Zeros: {zeros}/{len(df)} ({zeros/len(df)*100:.1f}%)")
                print(f"  NULLs: {nulls}/{len(df)} ({nulls/len(df)*100:.1f}%)")
                if values.max() > 0:
                    print(f"  Min (non-zero): {values[values > 0].min():.6f}")
                    print(f"  Max: {values.max():.6f}")
                    print(f"  Mean: {values.mean():.6f}")

    except Exception as e:
        print(f"\n{stat.upper()}: Table not found or error - {e}")

print("\n\n" + "="*80)
print("STEP 2: CHECK WC_MAD TABLES (actual weightclass MAD)")
print("="*80)

for stat in sparse_stats:
    table_name = f'{stat}_wc_mad'

    try:
        query = text(f"""
            SELECT
                weightclass,
                {stat}_wc_mad
            FROM features.{table_name}
            ORDER BY weightclass
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(f"\n{stat.upper()} - Weightclass MAD:")
        print(df.to_string(index=False))

        # Statistics
        wc_mad_col = f'{stat}_wc_mad'
        if wc_mad_col in df.columns:
            values = df[wc_mad_col].dropna()
            if len(values) > 0:
                zeros = (values == 0).sum()
                print(f"  Zeros: {zeros}/{len(df)} ({zeros/len(df)*100:.1f}%)")
                if values.max() > 0:
                    print(f"  Min (non-zero): {values[values > 0].min():.6f}")
                    print(f"  Max: {values.max():.6f}")
                    print(f"  Mean: {values.mean():.6f}")

    except Exception as e:
        print(f"\n{stat.upper()}: Table not found or error - {e}")

print("\n\n" + "="*80)
print("STEP 3: ANALYZE INDIVIDUAL FIGHTER MAD DISTRIBUTIONS")
print("="*80)

# For a few sparse stats, look at the actual distribution of MAD values
for stat in ['sub_att', 'td_att', 'ground_att']:
    try:
        # Query individual fighter MAD values from the base table
        query = text(f"""
            WITH fighter_stats AS (
                SELECT
                    fm.weightclass,
                    f.fighter_id,
                    f.{stat}_mad
                FROM features.{stat} f
                JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
                WHERE f.{stat}_mad IS NOT NULL
            )
            SELECT
                weightclass,
                COUNT(*) as n_fighters,
                COUNT(CASE WHEN {stat}_mad = 0 THEN 1 END) as n_zero_mad,
                COUNT(CASE WHEN {stat}_mad > 0 AND {stat}_mad < 0.01 THEN 1 END) as n_very_small,
                COUNT(CASE WHEN {stat}_mad >= 0.01 AND {stat}_mad < 0.1 THEN 1 END) as n_small,
                COUNT(CASE WHEN {stat}_mad >= 0.1 THEN 1 END) as n_healthy,
                MIN({stat}_mad) as min_mad,
                PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY CASE WHEN {stat}_mad > 0 THEN {stat}_mad END) as p05,
                PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY CASE WHEN {stat}_mad > 0 THEN {stat}_mad END) as p10,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY CASE WHEN {stat}_mad > 0 THEN {stat}_mad END) as p25,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY CASE WHEN {stat}_mad > 0 THEN {stat}_mad END) as p50,
                MAX({stat}_mad) as max_mad
            FROM fighter_stats
            GROUP BY weightclass
            ORDER BY weightclass
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(f"\n{stat.upper()} - Fighter MAD Distribution by Weightclass:")
        print(df.to_string(index=False))

        # Calculate what percentage have zero MAD
        if 'n_fighters' in df.columns and 'n_zero_mad' in df.columns:
            df['pct_zero'] = (df['n_zero_mad'] / df['n_fighters'] * 100).round(1)
            print(f"\nPercentage with zero MAD by weightclass:")
            print(df[['weightclass', 'pct_zero']].to_string(index=False))

    except Exception as e:
        print(f"\n{stat.upper()}: Error - {e}")

print("\n\n" + "="*80)
print("STEP 4: SIMULATE DIFFERENT FLOOR VALUES")
print("="*80)

print("""
When a weightclass has 0 MAD for a stat (meaning all fighters have identical values,
usually 0), we need a floor to prevent division by zero in adjperf calculations.

Current approach: COALESCE(mad_floor, 0.001)
- If 10th percentile exists: use it
- If NULL/0: fallback to 0.001

Let's analyze what happens with different fallback values:
""")

# Get an example sparse stat with many zeros
stat = 'sub_att'
try:
    query = text(f"""
        SELECT
            fm.weightclass,
            f.{stat}_mad
        FROM features.{stat} f
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        WHERE f.{stat}_mad IS NOT NULL
        LIMIT 10000
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"\nAnalyzing {stat.upper()} across {len(df)} fighter-fights:")

    # Group by weightclass
    for wc in df['weightclass'].unique()[:3]:  # Just show first 3 weightclasses
        wc_data = df[df['weightclass'] == wc][f'{stat}_mad']

        print(f"\n{wc}:")
        print(f"  Total samples: {len(wc_data)}")
        print(f"  Zero MAD: {(wc_data == 0).sum()} ({(wc_data == 0).sum()/len(wc_data)*100:.1f}%)")
        print(f"  Non-zero MAD: {(wc_data > 0).sum()} ({(wc_data > 0).sum()/len(wc_data)*100:.1f}%)")

        if (wc_data > 0).sum() > 0:
            nonzero = wc_data[wc_data > 0]
            print(f"  Non-zero stats:")
            print(f"    Min: {nonzero.min():.6f}")
            print(f"    5th: {np.percentile(nonzero, 5):.6f}")
            print(f"    10th: {np.percentile(nonzero, 10):.6f}")
            print(f"    25th: {np.percentile(nonzero, 25):.6f}")
            print(f"    Median: {nonzero.median():.6f}")

except Exception as e:
    print(f"Error: {e}")

print("\n\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
Based on the data above, here are the key considerations:

1. ZERO MAD PERCENTAGE:
   If a high percentage of fighters have 0 MAD (e.g., >50%), this indicates
   a degenerate/sparse stat where most fighters have the same value (usually 0).

2. CURRENT FALLBACK (0.001):
   - Conservative, prevents extreme amplification
   - If fighter has value=1 and median=0, adjperf = (1-0)/0.001 = 1000 (!)
   - Gets winsorized to ±7.0, so max impact is still bounded

3. ALTERNATIVE FALLBACKS:

   a) 0.005:
      - More conservative, reduces amplification 5x
      - adjperf = (1-0)/0.005 = 200, winsorized to 7.0
      - Less sensitive to single rare events

   b) 0.010:
      - Very conservative
      - adjperf = (1-0)/0.010 = 100, winsorized to 7.0
      - Heavily dampens sparse stat signals

   c) Use non-zero median:
      - Calculate median of non-zero MAD values
      - More data-driven but could be unstable with small samples

4. RECOMMENDED APPROACH:

   Based on the actual distributions, the optimal fallback depends on:
   - If >80% have zero MAD: Use 0.005 or 0.010 (very sparse, dampen signal)
   - If 50-80% have zero MAD: Use 0.003-0.005 (sparse, moderate dampening)
   - If 20-50% have zero MAD: Use 0.001-0.003 (somewhat sparse, light dampening)
   - If <20% have zero MAD: Use 0.001 (current approach is fine)

   The winsorization at ±7.0 provides a safety net regardless of floor choice.

5. NEXT STEP:
   Look at the actual percentages above to make a data-driven decision!
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
