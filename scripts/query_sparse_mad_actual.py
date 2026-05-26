"""
Query actual MAD data for sparse stats from existing database tables.
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
print("SPARSE STAT MAD INVESTIGATION - ACTUAL DATABASE VALUES")
print("="*80)

# Sparse stats to investigate
sparse_stats = ['sub', 'td', 'ground', 'rev', 'kd']

print("\n" + "="*80)
print("STEP 1: WEIGHTCLASS MAD VALUES (from _wc_mad tables)")
print("="*80)

for stat in sparse_stats:
    print(f"\n{stat.upper()} - Weightclass MAD:")
    try:
        query = text(f"SELECT * FROM features.{stat}_wc_mad ORDER BY weightclass")

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(df.to_string(index=False))

        # Get the MAD column
        mad_col = f'{stat}_wc_mad'
        if mad_col in df.columns:
            values = df[mad_col].dropna()
            if len(values) > 0:
                zeros = (values == 0).sum()
                very_small = ((values > 0) & (values < 0.01)).sum()
                small = ((values >= 0.01) & (values < 0.1)).sum()
                healthy = (values >= 0.1).sum()

                print(f"\n  Distribution:")
                print(f"    Zero MAD: {zeros}/{len(values)} ({zeros/len(values)*100:.1f}%)")
                print(f"    Very small (0-0.01): {very_small}/{len(values)} ({very_small/len(values)*100:.1f}%)")
                print(f"    Small (0.01-0.1): {small}/{len(values)} ({small/len(values)*100:.1f}%)")
                print(f"    Healthy (>0.1): {healthy}/{len(values)} ({healthy/len(values)*100:.1f}%)")

                if values.max() > 0:
                    nonzero = values[values > 0]
                    print(f"  Non-zero stats:")
                    print(f"    Min: {nonzero.min():.6f}")
                    print(f"    Mean: {nonzero.mean():.6f}")
                    print(f"    Max: {nonzero.max():.6f}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n\n" + "="*80)
print("STEP 2: MINIMUM MAD VALUES (10th percentile floors)")
print("="*80)

for stat in sparse_stats:
    print(f"\n{stat.upper()} - Minimum MAD (10th percentile per weightclass):")
    try:
        query = text(f"SELECT * FROM features.{stat}_minimum_mad ORDER BY weightclass")

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(df.to_string(index=False))

        # Get the min_mad column
        min_mad_col = f'{stat}_min_mad'
        if min_mad_col in df.columns:
            values = df[min_mad_col].dropna()
            if len(values) > 0:
                zeros = (values == 0).sum()
                nulls = df[min_mad_col].isna().sum()

                print(f"\n  Distribution:")
                print(f"    Zero: {zeros}/{len(df)} ({zeros/len(df)*100:.1f}%)")
                print(f"    NULL: {nulls}/{len(df)} ({nulls/len(df)*100:.1f}%)")

                if values.max() > 0:
                    nonzero = values[values > 0]
                    print(f"  Non-zero stats:")
                    print(f"    Min: {nonzero.min():.6f}")
                    print(f"    Mean: {nonzero.mean():.6f}")
                    print(f"    Max: {nonzero.max():.6f}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n\n" + "="*80)
print("STEP 3: COMPARISON - WC_MAD vs MIN_MAD")
print("="*80)

for stat in sparse_stats:
    print(f"\n{stat.upper()}:")
    try:
        query_wc = text(f"SELECT weightclass, {stat}_wc_mad FROM features.{stat}_wc_mad ORDER BY weightclass")
        query_min = text(f"SELECT weightclass, {stat}_min_mad FROM features.{stat}_minimum_mad ORDER BY weightclass")

        with engine.connect() as conn:
            wc_df = pd.read_sql(query_wc, conn)
            min_df = pd.read_sql(query_min, conn)

        # Merge
        merged = wc_df.merge(min_df, on='weightclass', how='outer')
        merged = merged.sort_values('weightclass')

        print(merged.to_string(index=False))

        # Calculate how many use fallback (min_mad is 0 or NULL)
        min_col = f'{stat}_min_mad'
        wc_col = f'{stat}_wc_mad'

        if min_col in merged.columns and wc_col in merged.columns:
            needs_fallback = (merged[min_col].isna() | (merged[min_col] == 0)).sum()
            print(f"\n  Weightclasses needing 0.001 fallback: {needs_fallback}/{len(merged)} ({needs_fallback/len(merged)*100:.1f}%)")

    except Exception as e:
        print(f"  Error: {e}")

print("\n\n" + "="*80)
print("STEP 4: ANALYZE IMPACT OF DIFFERENT FALLBACKS")
print("="*80)

print("""
Current implementation (from adj_perf_calc.py line 464):
    GREATEST(
        shrunk_mad,
        COALESCE(wp.{col}_mad_floor, 0.001)
    )

This means:
- If mad_floor (10th percentile) > 0: use it
- If mad_floor is NULL or 0: use 0.001

Let's analyze what this means for sparse stats...
""")

for stat in sparse_stats[:3]:  # Just show first 3 for clarity
    print(f"\n{stat.upper()} - Impact Analysis:")
    try:
        query = text(f"""
            SELECT
                wc.weightclass,
                wc.{stat}_wc_mad as weightclass_mad,
                mm.{stat}_min_mad as floor_10th_pct,
                CASE
                    WHEN mm.{stat}_min_mad IS NULL OR mm.{stat}_min_mad = 0
                    THEN 0.001
                    ELSE mm.{stat}_min_mad
                END as actual_floor
            FROM features.{stat}_wc_mad wc
            LEFT JOIN features.{stat}_minimum_mad mm ON wc.weightclass = mm.weightclass
            ORDER BY wc.weightclass
        """)

        with engine.connect() as conn:
            df = pd.read_sql(query, conn)

        print(df.to_string(index=False))

        # Simulate adjperf for a fighter with value=1, median=0
        df['adjperf_if_value_1'] = 1.0 / df['actual_floor']
        df['adjperf_winsorized'] = df['adjperf_if_value_1'].clip(-7.0, 7.0)

        print(f"\n  If fighter has {stat}=1, median=0:")
        print(df[['weightclass', 'actual_floor', 'adjperf_if_value_1', 'adjperf_winsorized']].to_string(index=False))

    except Exception as e:
        print(f"  Error: {e}")

print("\n\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
Based on the actual database values above, here are the key findings:

1. DEGENERATE STATS IDENTIFICATION:
   Look at the percentages of zero/NULL minimum MAD values.
   - High % (>50%) = very sparse, needs higher fallback
   - Low % (<20%) = mostly healthy, current 0.001 is fine

2. CURRENT FALLBACK IMPACT:
   For sparse stats with 0 MAD, using 0.001 fallback means:
   - Single rare event (value=1) creates adjperf = 1000 (!)
   - Winsorization clips to ±7.0, so max impact is bounded
   - But signal is still very strong for rare events

3. ALTERNATIVE FALLBACKS:

   OPTION A: Keep 0.001 (CURRENT)
   Pros:
   - Already implemented, working
   - Winsorization provides safety net
   - Preserves signal for rare but meaningful events
   Cons:
   - Very sensitive to single events in sparse stats
   - May overweight rare occurrences

   OPTION B: Raise to 0.005
   Pros:
   - Still preserves signal but less extreme
   - adjperf = 1/0.005 = 200, clips to 7.0
   - Better balance for sparse stats
   Cons:
   - Dampens rare event signals
   - Less discriminative power

   OPTION C: Raise to 0.010
   Pros:
   - Conservative, robust to noise
   - adjperf = 1/0.010 = 100, clips to 7.0
   - Heavily dampens sparse stat impact
   Cons:
   - May lose too much signal
   - Rare events become less important

4. DATA-DRIVEN DECISION:
   Based on the percentages shown above:
   - If SUB has >80% zero MAD: Consider 0.005-0.010
   - If TD has 50-80% zero MAD: Consider 0.003-0.005
   - If GROUND has <50% zero MAD: Keep 0.001

5. MY RECOMMENDATION:
   Look at the actual percentages above and choose:
   - If most sparse stats have >70% zero/NULL floors: Use 0.005
   - If mixed (some sparse, some not): Keep 0.001 (simpler)
   - If concerned about rare event overfitting: Use 0.003 (middle ground)

   The ±7.0 winsorization ensures no catastrophic outcomes regardless.
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
