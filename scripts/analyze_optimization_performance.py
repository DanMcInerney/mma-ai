"""
Analyze optimization performance to answer key questions:
1. Sample sizes per stat×weightclass
2. Data distribution patterns
3. Why improvements are so small
4. Optimal search ranges for different stat types
5. Statistical significance of improvements
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
import json
from collections import defaultdict

from config.parameters import TRAINING_START_DATE, TRAINING_END_DATE
from libs.paths import database_url

# Database connection
DB_URL = database_url()
engine = create_engine(DB_URL)

print("="*80)
print("OPTIMIZATION PERFORMANCE ANALYSIS")
print("="*80)
print(f"Training period: {TRAINING_START_DATE} to {TRAINING_END_DATE}\n")

# ============================================================================
# QUESTION 1: Sample sizes per stat×weightclass
# ============================================================================

print("\n" + "="*80)
print("Q1: SAMPLE SIZES PER STAT×WEIGHTCLASS")
print("="*80)

# Get fight counts and attempt counts per weightclass
query = text("""
    WITH fight_counts AS (
        SELECT
            fm.weightclass,
            COUNT(DISTINCT fd.fight_id) as n_fights,
            COUNT(DISTINCT fd.fighter_id) as n_fighters,

            -- Binary outcome sample sizes (per fight)
            SUM(CASE WHEN fd.ko = 1 OR fd.ko = 0 THEN 1 ELSE 0 END) as ko_attempts,
            SUM(CASE WHEN fd.win = 1 OR fd.win = 0 THEN 1 ELSE 0 END) as win_attempts,
            SUM(CASE WHEN fd.decision = 1 OR fd.decision = 0 THEN 1 ELSE 0 END) as decision_attempts,
            SUM(CASE WHEN fd.sub_land = 1 OR fd.sub_land = 0 THEN 1 ELSE 0 END) as sub_land_attempts,

            -- Count stat sample sizes (per fight with >0 attempts)
            SUM(CASE WHEN fd.sig_str_land > 0 OR fd.sig_str_att > 0 THEN 1 ELSE 0 END) as sig_str_samples,
            SUM(CASE WHEN fd.td_land > 0 OR fd.td_att > 0 THEN 1 ELSE 0 END) as td_samples,
            SUM(CASE WHEN fd.sub_att > 0 THEN 1 ELSE 0 END) as sub_samples,

            -- Accuracy stat sample sizes (per fight with >0 attempts)
            SUM(CASE WHEN fd.sig_str_att > 0 THEN 1 ELSE 0 END) as sig_str_acc_samples,
            SUM(CASE WHEN fd.td_att > 0 THEN 1 ELSE 0 END) as td_acc_samples,
            SUM(CASE WHEN fd.sub_att > 0 THEN 1 ELSE 0 END) as sub_acc_samples,

            -- Total attempts for accuracy
            SUM(fd.sig_str_att) as sig_str_total_att,
            SUM(fd.td_att) as td_total_att,
            SUM(fd.sub_att) as sub_total_att

        FROM features.fight_stats_fe fd
        JOIN features.fight_mapping fm ON fd.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fd.event_id = em.event_id
        WHERE em.event_date >= :start_date
          AND em.event_date < :end_date
        GROUP BY fm.weightclass
        ORDER BY n_fights DESC
    )
    SELECT * FROM fight_counts
""")

with engine.connect() as conn:
    sample_sizes = pd.read_sql(query, conn, params={
        'start_date': TRAINING_START_DATE,
        'end_date': TRAINING_END_DATE
    })

print("\nFight counts by weightclass:")
print(sample_sizes[['weightclass', 'n_fights', 'n_fighters']].to_string(index=False))

print("\n\nBinary outcome sample sizes:")
for stat in ['ko', 'win', 'decision', 'sub_land']:
    col = f'{stat}_attempts'
    print(f"\n{stat.upper()}:")
    print(sample_sizes[['weightclass', col]].to_string(index=False))

print("\n\nCount stat sample sizes (fights with >0 attempts):")
for stat in ['sig_str', 'td', 'sub']:
    col = f'{stat}_samples'
    print(f"\n{stat.upper()}:")
    print(sample_sizes[['weightclass', col]].to_string(index=False))

print("\n\nAccuracy stat sample sizes and total attempts:")
for stat in ['sig_str_acc', 'td_acc', 'sub_acc']:
    samples_col = f'{stat}_samples'
    att_col = stat.replace('_acc', '_total_att')
    print(f"\n{stat.upper()}:")
    df = sample_sizes[['weightclass', samples_col, att_col]].copy()
    df.columns = ['weightclass', 'n_fights_with_attempts', 'total_attempts']
    print(df.to_string(index=False))

# ============================================================================
# QUESTION 2: Data distribution patterns - variance across weightclasses
# ============================================================================

print("\n\n" + "="*80)
print("Q2: DATA DISTRIBUTION PATTERNS - VARIANCE ACROSS WEIGHTCLASSES")
print("="*80)

# Calculate variance in event rates and means across weightclasses
query = text("""
    WITH weightclass_stats AS (
        SELECT
            fm.weightclass,

            -- Binary outcome rates
            AVG(CASE WHEN fd.ko IS NOT NULL THEN fd.ko::float ELSE NULL END) as ko_rate,
            AVG(CASE WHEN fd.win IS NOT NULL THEN fd.win::float ELSE NULL END) as win_rate,
            AVG(CASE WHEN fd.sub_land IS NOT NULL THEN fd.sub_land::float ELSE NULL END) as sub_land_rate,

            -- Count stat per-minute rates (calculated from totals)
            AVG(CASE WHEN fd.time_sec > 0 THEN fd.sig_str_land::float / (fd.time_sec / 60.0) ELSE NULL END) as sig_str_mean,
            AVG(CASE WHEN fd.time_sec > 0 THEN fd.td_land::float / (fd.time_sec / 60.0) ELSE NULL END) as td_mean,
            AVG(CASE WHEN fd.time_sec > 0 THEN fd.sub_att::float / (fd.time_sec / 60.0) ELSE NULL END) as sub_mean,

            -- Accuracy rates (weighted by attempts)
            CASE WHEN SUM(fd.sig_str_att) > 0
                THEN SUM(fd.sig_str_land::float) / SUM(fd.sig_str_att)
                ELSE NULL END as sig_str_acc,
            CASE WHEN SUM(fd.td_att) > 0
                THEN SUM(fd.td_land::float) / SUM(fd.td_att)
                ELSE NULL END as td_acc,
            CASE WHEN SUM(fd.sub_att) > 0
                THEN SUM(fd.sub_land::float) / SUM(fd.sub_att)
                ELSE NULL END as sub_acc

        FROM features.fight_stats_fe fd
        JOIN features.fight_mapping fm ON fd.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fd.event_id = em.event_id
        WHERE em.event_date >= :start_date
          AND em.event_date < :end_date
        GROUP BY fm.weightclass
    )
    SELECT * FROM weightclass_stats
    ORDER BY weightclass
""")

with engine.connect() as conn:
    wc_stats = pd.read_sql(query, conn, params={
        'start_date': TRAINING_START_DATE,
        'end_date': TRAINING_END_DATE
    })

print("\nKO rates by weightclass (higher variance -> more benefit from per-class tau):")
print(wc_stats[['weightclass', 'ko_rate']].to_string(index=False))
print(f"  Variance: {wc_stats['ko_rate'].var():.6f}")
print(f"  Coefficient of variation: {wc_stats['ko_rate'].std() / wc_stats['ko_rate'].mean():.3f}")

print("\nSig strike per-minute means by weightclass:")
print(wc_stats[['weightclass', 'sig_str_mean']].to_string(index=False))
print(f"  Variance: {wc_stats['sig_str_mean'].var():.6f}")
print(f"  Coefficient of variation: {wc_stats['sig_str_mean'].std() / wc_stats['sig_str_mean'].mean():.3f}")

print("\nSub accuracy by weightclass:")
print(wc_stats[['weightclass', 'sub_acc']].to_string(index=False))
print(f"  Variance: {wc_stats['sub_acc'].var():.6f}")
print(f"  Coefficient of variation: {wc_stats['sub_acc'].std() / wc_stats['sub_acc'].mean():.3f}")

# ============================================================================
# QUESTION 3: Why are improvements so small?
# ============================================================================

print("\n\n" + "="*80)
print("Q3: WHY ARE IMPROVEMENTS SO SMALL?")
print("="*80)

# Check if global tau is already well-tuned by comparing to per-class optimal
results_file = project_root / 'data' / 'comprehensive_tuning' / 'detailed_results.json'
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)

    print("\nAnalyzing gap between global tau and per-class optimal tau:\n")

    # Group by stat type
    by_stat = defaultdict(list)
    for r in results:
        by_stat[r['stat_name']].append(r)

    for stat_name, stat_results in sorted(by_stat.items())[:5]:  # First 5 stats
        print(f"\n{stat_name.upper()}:")
        if stat_results:
            global_tau = stat_results[0]['global_tau']
            print(f"  Global tau: {global_tau:.3f}")

            # Calculate how far per-class optimal deviates from global
            deviations = []
            improvements = []
            for r in stat_results:
                if r.get('weightclass'):
                    optimal_tau = r['optimal_tau']
                    deviation = abs(optimal_tau - global_tau) / global_tau
                    improvement = r['improvement_pct']
                    deviations.append(deviation)
                    improvements.append(improvement)

                    if abs(improvement) > 0.1:  # Only show significant ones
                        print(f"    {r['weightclass']:20s} optimal={optimal_tau:6.2f}  "
                              f"deviation={deviation:5.1%}  improvement={improvement:+6.2%}")

            if deviations:
                print(f"  Mean deviation from global: {np.mean(deviations):.1%}")
                print(f"  Mean improvement: {np.mean(improvements):+.3%}")
else:
    print("detailed_results.json not found - run comprehensive_likelihood_tuner.py first")

# ============================================================================
# QUESTION 4: Optimal search ranges
# ============================================================================

print("\n\n" + "="*80)
print("Q4: OPTIMAL SEARCH RANGES FOR DIFFERENT STAT TYPES")
print("="*80)

if results_file.exists():
    print("\nCurrent optimal tau values by stat type:")

    stat_type_groups = defaultdict(list)
    for r in results:
        stat_type_groups[r['stat_type']].append(r['optimal_tau'])

    for stat_type, tau_values in sorted(stat_type_groups.items()):
        tau_arr = np.array(tau_values)
        print(f"\n{stat_type.upper()}:")
        print(f"  Min:  {tau_arr.min():.3f}")
        print(f"  25th: {np.percentile(tau_arr, 25):.3f}")
        print(f"  50th: {np.percentile(tau_arr, 50):.3f}")
        print(f"  75th: {np.percentile(tau_arr, 75):.3f}")
        print(f"  Max:  {tau_arr.max():.3f}")
        print(f"  Recommended range: ({tau_arr.min() * 0.5:.1f}, {tau_arr.max() * 1.5:.1f})")

        # Check for boundary hits
        boundary_hits = [r for r in results
                        if r['stat_type'] == stat_type and r.get('boundary_hit')]
        if boundary_hits:
            print(f"  [WARNING] BOUNDARY HITS: {len(boundary_hits)} cases")
            for r in boundary_hits[:3]:  # Show first 3
                print(f"     {r['stat_name']} @ {r.get('weightclass', 'global')}: "
                      f"tau={r['optimal_tau']:.1f}, range={r['search_range']}")

# ============================================================================
# QUESTION 5: Statistical significance of improvements
# ============================================================================

print("\n\n" + "="*80)
print("Q5: STATISTICAL SIGNIFICANCE OF IMPROVEMENTS")
print("="*80)

if results_file.exists():
    print("\nDistribution of improvement percentages:\n")

    improvements = [r['improvement_pct'] for r in results if 'weightclass' in r]
    imp_arr = np.array(improvements)

    print(f"Total per-class comparisons: {len(improvements)}")
    print(f"Mean improvement:   {imp_arr.mean():+.3%}")
    print(f"Median improvement: {np.median(imp_arr):+.3%}")
    print(f"Std deviation:      {imp_arr.std():.3%}")
    print()
    print(f"Positive improvements: {(imp_arr > 0).sum()} ({(imp_arr > 0).mean():.1%})")
    print(f"Negative improvements: {(imp_arr < 0).sum()} ({(imp_arr < 0).mean():.1%})")
    print(f"Near-zero (±0.1%):     {(np.abs(imp_arr) < 0.1).sum()} ({(np.abs(imp_arr) < 0.1).mean():.1%})")
    print()
    print(f"Significant (>0.5%):   {(imp_arr > 0.5).sum()} ({(imp_arr > 0.5).mean():.1%})")
    print(f"Very significant (>1%): {(imp_arr > 1.0).sum()} ({(imp_arr > 1.0).mean():.1%})")

    print("\n\nCorrelation between sample size and improvement magnitude:")

    # Get sample sizes and improvements
    sample_improvement_pairs = []
    for r in results:
        if 'weightclass' in r and 'n_samples' in r:
            sample_improvement_pairs.append((r['n_samples'], abs(r['improvement_pct'])))

    if sample_improvement_pairs:
        samples, abs_improvements = zip(*sample_improvement_pairs)
        correlation = np.corrcoef(samples, abs_improvements)[0, 1]
        print(f"  Correlation coefficient: {correlation:.3f}")
        if abs(correlation) > 0.3:
            print(f"  -> {'Positive' if correlation > 0 else 'Negative'} correlation detected")
            if correlation < 0:
                print("  -> Smaller sample sizes -> larger improvements (possible overfitting)")
        else:
            print("  -> Weak correlation (sample size not driving improvements)")

    print("\n\nTau stability vs improvement magnitude:")
    stability_improvement_pairs = []
    for r in results:
        if 'weightclass' in r and 'tau_stability' in r:
            stability_improvement_pairs.append((r['tau_stability'], abs(r['improvement_pct'])))

    if stability_improvement_pairs:
        stabilities, abs_improvements = zip(*stability_improvement_pairs)
        correlation = np.corrcoef(stabilities, abs_improvements)[0, 1]
        print(f"  Correlation coefficient: {correlation:.3f}")
        if abs(correlation) > 0.3:
            print(f"  -> {'Positive' if correlation > 0 else 'Negative'} correlation detected")
            if correlation > 0:
                print("  -> Higher instability -> larger improvements (overfitting signal)")
        else:
            print("  -> Weak correlation (stability not related to improvements)")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

print("""
Based on the analysis above, here are the key findings and recommended actions:

1. SAMPLE SIZE ISSUES:
   - Check if weightclasses with <500 fights should use global tau only
   - Consider minimum sample size threshold for per-class optimization

2. SMALL IMPROVEMENTS:
   - If mean improvement <0.5%, global tau may already be well-optimized
   - Consider: is the added complexity of per-class tau worth <1% gains?
   - Add statistical significance testing (bootstrap confidence intervals)

3. BOUNDARY HITS:
   - Expand search ranges for stat types hitting boundaries
   - Especially important for accuracy stats (sub_acc hitting both ends)

4. OVERFITTING SIGNALS:
   - High tau_stability (>20) suggests unstable optimization
   - Negative correlation between sample size and improvement is a red flag
   - Consider using median tau across CV folds for unstable cases

5. NEXT STEPS:
   - Add minimum improvement threshold (e.g., 0.5%) for per-class selection
   - Add maximum stability threshold (e.g., 15) for per-class selection
   - Implement bootstrap significance testing
   - Expand search ranges based on actual optimal values found
""")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
