"""
Analyze actual adjperf distributions to determine optimal winsorization limits.

This script examines:
1. Distribution of adjperf values across different stats
2. How many values hit the current ±7.0 limit
3. Whether different stats need different limits
4. Whether limits should vary by weightclass
5. Optimal winsorization strategy (global, per-stat, or per-stat-per-weightclass)
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from collections import defaultdict
from libs.paths import database_url

DB_URL = database_url()
engine = create_engine(DB_URL)

print("="*80)
print("ADJPERF WINSORIZATION ANALYSIS")
print("="*80)

# Get list of stat tables with adjperf columns
print("\nStep 1: Finding tables with adjperf columns...")

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'features'
          AND tablename NOT LIKE '%_wc_%'
          AND tablename NOT LIKE '%minimum%'
          AND tablename NOT LIKE '%_opp%'
        ORDER BY tablename
    """))
    all_tables = [row[0] for row in result]

# Sample some key stats for analysis
key_stats = ['sig_str', 'td', 'sub', 'ground', 'rev', 'kd', 'ko', 'head', 'body', 'clinch']
tables_to_analyze = [t for t in all_tables if any(t.startswith(s) for s in key_stats)]

print(f"Found {len(tables_to_analyze)} tables to analyze")

# ============================================================================
# STEP 1: Analyze adjperf distribution for each stat
# ============================================================================

print("\n" + "="*80)
print("STEP 1: ADJPERF DISTRIBUTION ANALYSIS")
print("="*80)

adjperf_stats = {}

for table in tables_to_analyze[:15]:  # Analyze first 15 tables
    print(f"\nAnalyzing: {table}")

    try:
        # Get column names
        with engine.connect() as conn:
            result = conn.execute(text(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'features'
                  AND table_name = '{table}'
                  AND column_name LIKE '%adjperf%'
                  AND column_name NOT LIKE '%opp%'
                ORDER BY column_name
                LIMIT 5
            """))
            adjperf_cols = [row[0] for row in result]

        if not adjperf_cols:
            print(f"  No adjperf columns found")
            continue

        # Analyze first adjperf column
        col = adjperf_cols[0]

        with engine.connect() as conn:
            query = text(f"""
                WITH adjperf_data AS (
                    SELECT
                        fm.weightclass,
                        t.{col}
                    FROM features.{table} t
                    JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
                    WHERE t.{col} IS NOT NULL
                      AND t.{col} != 'NaN'
                      AND t.{col} != 'Infinity'
                      AND t.{col} != '-Infinity'
                )
                SELECT
                    weightclass,
                    COUNT(*) as n_samples,
                    MIN({col}) as min_val,
                    PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY {col}) as p01,
                    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY {col}) as p05,
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {col}) as p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {col}) as p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {col}) as p75,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {col}) as p95,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {col}) as p99,
                    MAX({col}) as max_val,
                    -- Count at boundaries
                    COUNT(CASE WHEN {col} >= 6.9 THEN 1 END) as n_at_upper_7,
                    COUNT(CASE WHEN {col} <= -6.9 THEN 1 END) as n_at_lower_7,
                    COUNT(CASE WHEN ABS({col}) >= 6.9 THEN 1 END) as n_at_limit_7
                FROM adjperf_data
                GROUP BY weightclass
                ORDER BY weightclass
            """)

            df = pd.read_sql(query, conn)

        if len(df) > 0:
            # Store overall stats
            adjperf_stats[f"{table}.{col}"] = {
                'table': table,
                'column': col,
                'df': df
            }

            # Print summary
            total_samples = df['n_samples'].sum()
            total_at_limit = df['n_at_limit_7'].sum()
            pct_at_limit = (total_at_limit / total_samples * 100) if total_samples > 0 else 0

            global_p95 = df['p95'].mean()
            global_p99 = df['p99'].mean()
            global_max = df['max_val'].max()
            global_min = df['min_val'].min()

            print(f"  Column: {col}")
            print(f"  Total samples: {total_samples}")
            print(f"  At ±7.0 limit: {total_at_limit} ({pct_at_limit:.2f}%)")
            print(f"  Global range: [{global_min:.2f}, {global_max:.2f}]")
            print(f"  95th percentile: {global_p95:.2f}")
            print(f"  99th percentile: {global_p99:.2f}")

    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 2: Aggregate analysis across all stats
# ============================================================================

print("\n\n" + "="*80)
print("STEP 2: CROSS-STAT COMPARISON")
print("="*80)

if adjperf_stats:
    summary_data = []

    for stat_name, stat_info in adjperf_stats.items():
        df = stat_info['df']

        total_samples = df['n_samples'].sum()
        total_at_limit = df['n_at_limit_7'].sum()
        pct_at_limit = (total_at_limit / total_samples * 100) if total_samples > 0 else 0

        avg_p95 = df['p95'].mean()
        avg_p99 = df['p99'].mean()
        abs_max = max(abs(df['min_val'].min()), abs(df['max_val'].max()))

        summary_data.append({
            'stat': stat_name,
            'samples': total_samples,
            'pct_at_7.0': pct_at_limit,
            'avg_p95': avg_p95,
            'avg_p99': avg_p99,
            'abs_max': abs_max,
            'recommended_limit': None  # Will calculate
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('pct_at_7.0', ascending=False)

    print("\nStats ranked by % at winsorization limit:")
    print(summary_df[['stat', 'samples', 'pct_at_7.0', 'avg_p95', 'avg_p99', 'abs_max']].to_string(index=False))

    # Calculate recommended limits
    print("\n\nRecommended winsorization limits:")
    print("(Based on capturing 95-99th percentile)")

    for idx, row in summary_df.iterrows():
        # Recommend limit that captures 99th percentile + margin
        recommended = np.ceil(row['avg_p99'] * 1.1)  # 10% margin
        recommended = max(3.0, min(recommended, 10.0))  # Bound between 3 and 10

        summary_df.at[idx, 'recommended_limit'] = recommended

        print(f"\n{row['stat']}:")
        print(f"  Current: ±7.0")
        print(f"  99th percentile: {row['avg_p99']:.2f}")
        print(f"  Recommended: ±{recommended:.1f}")
        print(f"  Samples at limit: {row['pct_at_7.0']:.2f}%")

# ============================================================================
# STEP 3: Per-weightclass variation
# ============================================================================

print("\n\n" + "="*80)
print("STEP 3: PER-WEIGHTCLASS VARIATION ANALYSIS")
print("="*80)

print("\nDo different weightclasses need different limits?")

if adjperf_stats:
    # Pick a few representative stats
    for stat_name in list(adjperf_stats.keys())[:5]:
        stat_info = adjperf_stats[stat_name]
        df = stat_info['df']

        print(f"\n{stat_name}:")

        # Show p99 by weightclass
        wc_summary = df[['weightclass', 'p99', 'n_at_limit_7', 'n_samples']].copy()
        wc_summary['pct_at_limit'] = (wc_summary['n_at_limit_7'] / wc_summary['n_samples'] * 100).round(2)

        print(wc_summary[['weightclass', 'p99', 'pct_at_limit']].to_string(index=False))

        # Calculate coefficient of variation for p99 across weightclasses
        p99_values = df['p99'].values
        if len(p99_values) > 1 and p99_values.mean() > 0:
            cv = p99_values.std() / p99_values.mean()
            print(f"  CV of 99th percentile across WCs: {cv:.3f}")
            if cv > 0.2:
                print(f"  -> HIGH variation, consider per-weightclass limits")
            else:
                print(f"  -> LOW variation, global limit is fine")

# ============================================================================
# STEP 4: Optimal strategy recommendation
# ============================================================================

print("\n\n" + "="*80)
print("STEP 4: OPTIMAL WINSORIZATION STRATEGY")
print("="*80)

print("""
Based on the analysis above, here are the strategic options:

OPTION 1: GLOBAL LIMIT (CURRENT)
Pros:
- Simple, interpretable, easy to maintain
- Already implemented (±7.0)
- Consistent across all stats
Cons:
- May be too tight for some stats
- May be too loose for others
- Not optimized for specific distributions

OPTION 2: PER-STAT LIMITS
Pros:
- Tailored to each stat's distribution
- Captures appropriate percentile for each stat
- Relatively simple to implement
Cons:
- More complexity (need to store limits per stat)
- Less interpretable ("why does TD use ±5 but sub uses ±9?")
- Need to maintain/update limits

OPTION 3: PER-STAT-PER-WEIGHTCLASS LIMITS
Pros:
- Maximum optimization
- Accounts for weightclass-specific distributions
- Best theoretical performance
Cons:
- High complexity (need limits for each stat×weightclass)
- Risk of overfitting
- Hard to interpret and maintain
- Questionable benefit over per-stat

OPTION 4: TIERED LIMITS
Pros:
- Simple categorization (sparse/moderate/healthy)
- Easy to implement and understand
- Balances optimization and simplicity
Cons:
- Still requires stat categorization
- May not capture all nuances

MY RECOMMENDATION (as master ML engineer):

Look at the data above:
1. If most stats show <5% at limit: Keep ±7.0 (it's working)
2. If some stats show >10% at limit: Use per-stat limits
3. If CV across weightclasses >0.3: Consider per-stat-per-WC
4. If variation is minimal: Keep simple global limit

The key question: Does optimization justify complexity?
- For production models: probably yes (per-stat limits)
- For research/exploration: probably no (keep simple)
- For competition: definitely yes (maximize every advantage)
""")

# ============================================================================
# STEP 5: Implementation guidance
# ============================================================================

print("\n" + "="*80)
print("STEP 5: IMPLEMENTATION GUIDANCE")
print("="*80)

if 'summary_df' in locals():
    print("\nSuggested per-stat limits (based on 99th percentile + 10%):")
    print(summary_df[['stat', 'avg_p99', 'recommended_limit']].to_string(index=False))

    print("\n\nImplementation in adj_perf_calc.py:")
    print("""
# Define per-stat winsorization limits
WINSOR_LIMITS = {
    'sig_str': 7.0,
    'td': 5.0,
    'sub': 9.0,
    'ground': 6.0,
    'rev': 8.0,
    'kd': 10.0,
    'default': 7.0
}

# In _build_adjperf_expression():
stat_base = col.split('_')[0]  # Get base stat name
winsor_limit = WINSOR_LIMITS.get(stat_base, WINSOR_LIMITS['default'])
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
