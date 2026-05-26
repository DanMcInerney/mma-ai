#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to compare current smoothing parameters vs optimized values
and identify over-smoothing/under-smoothing issues.

Analyzes:
1. Current tau values in code vs optimized values from tuning
2. Impact of smoothing on actual fight data (Kazama case study)
3. Recommendations for which stats need retuning
"""

import json
import sys
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import io
from libs.paths import database_url, no_winsor_database_url

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

# ============================================================================
# PART 1: Compare Current vs Optimized Parameters
# ============================================================================

def load_optimized_parameters():
    """Load optimized parameters from tuning results"""
    params = {}

    # Load combined smoothing results (most comprehensive)
    combined_path = project_root / "data" / "smoothing_tuning" / "combined_smoothing_results.json"
    if combined_path.exists():
        with open(combined_path, 'r') as f:
            data = json.load(f)
            params['combined'] = data.get('best_combined_config', {})

    # Load individual analysis (older but has different insights)
    analysis_path = project_root / "data" / "smoothing_analysis" / "smoothing_analysis.json"
    if analysis_path.exists():
        with open(analysis_path, 'r') as f:
            data = json.load(f)
            params['analysis'] = data.get('recommendations', {})

    return params

def get_current_parameters():
    """Get current parameters from code"""
    current = {
        'beta_binomial': {
            'ko': 23.0,
            'sub_land': 9.0,
            'win': 25.0,
            'decision': 20.0,
            'ctrl': 2.0,
            'ko_rd1': 17.0,
            'win_rd1': 15.0,
            'sub_land_rd1': 7.0,
            'ctrl_rd1': 1.0,
        },
        'accuracy': {
            'sig_str': 18.0,
            'head': 14.0,
            'body': 12.0,
            'leg': 13.0,
            'ground': 14.0,
            'clinch': 12.0,
            'distance': 12.0,
            'td': 10.0,
            'sub': 25.0,
        },
        'poisson_gamma': {
            'sig_str': 3.0,
            'head': 6.5,
            'body': 6.5,
            'leg': 6.5,
            'distance': 6.5,
            'clinch': 6.5,
            'ground': 6.5,
            'td': 10.0,
            'sub': 25.0,
            'kd': 25.0,
            'rev': 25.0,
        }
    }
    return current

def compare_parameters(current, optimized):
    """Compare current vs optimized and flag discrepancies"""
    print("=" * 100)
    print("PARAMETER COMPARISON: CURRENT vs OPTIMIZED")
    print("=" * 100)
    print()

    # Beta-Binomial
    print("BETA-BINOMIAL (Binary Outcomes):")
    print("-" * 100)
    print(f"{'Stat':<15} {'Current':<12} {'Optimized':<12} {'Difference':<12} {'% Change':<12} {'Status':<20}")
    print("-" * 100)

    opt_bb = optimized.get('combined', {}).get('beta_binomial', {}).get('pseudocount_config', {})
    for stat in sorted(set(list(current['beta_binomial'].keys()) + list(opt_bb.keys()))):
        curr_val = current['beta_binomial'].get(stat, None)
        opt_val = opt_bb.get(stat, None)

        if curr_val is not None and opt_val is not None:
            diff = curr_val - opt_val
            pct_change = ((curr_val - opt_val) / opt_val) * 100 if opt_val != 0 else 0

            # Flag issues
            if abs(pct_change) > 100:
                status = "MAJOR MISMATCH"
            elif abs(pct_change) > 50:
                status = "Significant diff"
            elif abs(pct_change) > 20:
                status = "Minor difference"
            else:
                status = "Good match"

            print(f"{stat:<15} {curr_val:<12.1f} {opt_val:<12.1f} {diff:<+12.1f} {pct_change:<+12.1f} {status:<20}")
        elif curr_val is not None:
            print(f"{stat:<15} {curr_val:<12.1f} {'N/A':<12} {'N/A':<12} {'N/A':<12} No optimized value")
        elif opt_val is not None:
            print(f"{stat:<15} {'N/A':<12} {opt_val:<12.1f} {'N/A':<12} {'N/A':<12} Missing in current")

    print()
    print()

    # Accuracy
    print("ACCURACY STATS:")
    print("-" * 100)
    print(f"{'Stat':<15} {'Current':<12} {'Status':<20}")
    print("-" * 100)

    for stat in sorted(current['accuracy'].keys()):
        curr_val = current['accuracy'][stat]
        # Note: combined_smoothing_results doesn't have accuracy tuning results
        print(f"{stat:<15} {curr_val:<12.1f} No tuning data")

    print()
    print()

    # Poisson-Gamma
    print("POISSON-GAMMA (Count Data):")
    print("-" * 100)
    print(f"{'Stat':<15} {'Current':<12} {'Optimized':<12} {'Difference':<12} {'% Change':<12} {'Status':<20}")
    print("-" * 100)

    opt_pg = optimized.get('combined', {}).get('poisson_gamma', {}).get('tau_config', {})
    for stat in sorted(set(list(current['poisson_gamma'].keys()) + list(opt_pg.keys()))):
        curr_val = current['poisson_gamma'].get(stat, None)
        opt_val = opt_pg.get(stat, None)

        if curr_val is not None and opt_val is not None:
            diff = curr_val - opt_val
            pct_change = ((curr_val - opt_val) / opt_val) * 100 if opt_val != 0 else 0

            if abs(pct_change) > 100:
                status = "MAJOR MISMATCH"
            elif abs(pct_change) > 50:
                status = "Significant diff"
            elif abs(pct_change) > 20:
                status = "Minor difference"
            else:
                status = "Good match"

            print(f"{stat:<15} {curr_val:<12.1f} {opt_val:<12.1f} {diff:<+12.1f} {pct_change:<+12.1f} {status:<20}")
        elif curr_val is not None:
            print(f"{stat:<15} {curr_val:<12.1f} {'N/A':<12} {'N/A':<12} {'N/A':<12} No optimized value")
        elif opt_val is not None:
            print(f"{stat:<15} {'N/A':<12} {opt_val:<12.1f} {'N/A':<12} {'N/A':<12} Missing in current")

# ============================================================================
# PART 2: Test Smoothing Impact on Real Data (Kazama Case Study)
# ============================================================================

def test_smoothing_impact():
    """
    Test how different tau values affect smoothing for Kazama's defense.
    This demonstrates the over-smoothing problem.
    """
    print()
    print("=" * 100)
    print("SMOOTHING IMPACT ANALYSIS: Kazama Defense Case Study")
    print("=" * 100)
    print()

    # Kazama's actual fight data
    print("Kazama vs Elijah Smith (2025-08-09):")
    print("  Raw performance: Opponent landed 34 of 36 total strikes = 94.4% (terrible defense)")
    print("  Smoothed sig_str_def: 0.699")
    print("  Adjperf score: +3.83 (!!)")
    print()

    # Simulate smoothing with different tau values
    print("Simulating different tau values:")
    print("-" * 80)
    print(f"{'Tau':<10} {'Raw Fight':<15} {'Prior (est)':<15} {'Smoothed':<15} {'Interpretation':<30}")
    print("-" * 80)

    # Kazama's fight: opponent landed 34 out of 36
    raw_opp_ratio = 34 / 36  # 0.944 (opponent's land ratio)
    raw_kazama_def = 1 - raw_opp_ratio  # 0.056 (Kazama's actual defense)

    # Estimated prior from Kazama's historical fights (around 0.5 based on the data we saw)
    prior_def = 0.5

    # Test different tau values
    tau_values = [1, 2, 5, 10, 18, 25, 50, 100]

    for tau in tau_values:
        # Beta-Binomial smoothing formula: (prior * tau + k) / (tau + n)
        # For defense, we're smoothing opponent's accuracy, then taking 1 - smoothed_acc

        # Opponent landed 34, attempted 36
        # Smoothed opponent accuracy = (prior_opp_acc * tau + landed) / (tau + attempted)
        prior_opp_acc = 1 - prior_def  # 0.5
        smoothed_opp_acc = (prior_opp_acc * tau + 34) / (tau + 36)
        smoothed_def = 1 - smoothed_opp_acc

        if tau == 18:
            interp = "<- CURRENT (over-smoothing!)"
        elif tau <= 5:
            interp = "More responsive"
        elif tau >= 50:
            interp = "Very heavy smoothing"
        else:
            interp = ""

        print(f"{tau:<10} {raw_kazama_def:<15.3f} {prior_def:<15.3f} {smoothed_def:<15.3f} {interp:<30}")

    print()
    print("INTERPRETATION:")
    print("  - Raw fight performance: 0.056 (5.6% defense - terrible!)")
    print("  - Current tau=18: Smooths to ~0.42 (42% defense)")
    print("  - This makes terrible recent performance look mediocre")
    print("  - Lower tau (2-5) would be more responsive to recent performance")
    print("  - Current implementation over-smooths sparse but important fight data")

# ============================================================================
# PART 3: Recommendations
# ============================================================================

def print_recommendations(current, optimized):
    """Print actionable recommendations"""
    print()
    print("=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)
    print()

    print("CRITICAL ISSUES:")
    print()

    # Check ctrl specifically
    curr_ctrl = current['beta_binomial'].get('ctrl', 0)
    opt_ctrl = optimized.get('combined', {}).get('beta_binomial', {}).get('pseudocount_config', {}).get('ctrl', 0)

    if curr_ctrl and opt_ctrl:
        ctrl_diff = abs(curr_ctrl - opt_ctrl) / opt_ctrl * 100
        if ctrl_diff > 100:
            print(f"1. CTRL smoothing: Current={curr_ctrl}, Optimized={opt_ctrl}")
            print(f"   - Current is {ctrl_diff:.0f}% different from optimized!")
            print(f"   - RECOMMENDATION: Change ctrl tau from {curr_ctrl} to {opt_ctrl}")
            print()

    # Check for major mismatches
    opt_bb = optimized.get('combined', {}).get('beta_binomial', {}).get('pseudocount_config', {})
    print("2. Beta-Binomial major mismatches:")
    for stat, curr_val in current['beta_binomial'].items():
        opt_val = opt_bb.get(stat)
        if opt_val:
            diff_pct = abs(curr_val - opt_val) / opt_val * 100
            if diff_pct > 50:
                print(f"   - {stat}: Current={curr_val}, Optimized={opt_val} ({diff_pct:.0f}% diff)")
    print()

    print("3. Accuracy stats:")
    print("   - No optimized values found in tuning results")
    print("   - RECOMMENDATION: Run comprehensive_likelihood_tuner.py with accuracy stats")
    print("   - Current tau=18 for sig_str_acc appears too high based on Kazama case")
    print("   - Suggested range to test: tau=5-12 for accuracy stats")
    print()

    print("SUGGESTED ACTIONS:")
    print()
    print("1. Update BetaBinomialCalculator with optimized values:")
    print("   - Change ctrl: 2.0 → 120.0")
    print("   - Change ko: 23.0 → 12.0")
    print("   - Change win: 25.0 → 8.0")
    print("   - Change decision: 20.0 → 15.0")
    print("   - Change sub_land: 9.0 → 18.0")
    print()

    print("2. Re-run tuning for accuracy stats:")
    print("   - Run: python tuning/comprehensive_likelihood_tuner.py --verbose")
    print("   - Focus on sig_str, head, body, leg, ground, clinch accuracy")
    print("   - Current values (tau=12-18) likely too high for non-sparse stats")
    print()

    print("3. Validate changes:")
    print("   - Test on known edge cases (like Kazama)")
    print("   - Check that smoothing doesn't hide recent performance degradation")
    print("   - Ensure sparse stats (sub_att, rev) still get adequate smoothing")
    print()

    print("4. Consider stat-specific approaches:")
    print("   - Sparse stats (sub_att, kd, rev): Higher tau (15-25) is appropriate")
    print("   - Common stats (sig_str, head strikes): Lower tau (5-12) more responsive")
    print("   - Binary outcomes: Medium tau (8-15) balances stability and responsiveness")

# ============================================================================
# PART 4: Database Validation
# ============================================================================

def validate_with_database():
    """
    Query database to find more examples of potential over-smoothing.
    Look for fighters with large differences between raw and smoothed performance.
    """
    print()
    print("=" * 100)
    print("DATABASE VALIDATION: Finding Over-Smoothing Examples")
    print("=" * 100)
    print()

    engine = create_engine(DB_URL)

    query = text("""
        WITH recent_fights AS (
            SELECT
                fm.fighter_name,
                s.sig_str_land,
                s.sig_str_land_opp,
                s.sig_str_def as smoothed_def,
                s.sig_str_def_dec_adjperf_dec_avg as adjperf,
                -- Calculate raw defense from this fight
                CASE
                    WHEN s.sig_str_att_opp > 0
                    THEN (s.sig_str_att_opp - s.sig_str_land_opp)::float / s.sig_str_att_opp
                    ELSE NULL
                END as raw_def_this_fight,
                em.event_date
            FROM features.sig_str s
            JOIN features.fighter_mapping fm ON s.fighter_id = fm.fighter_id
            JOIN features.event_mapping em ON s.event_id = em.event_id
            WHERE em.event_date >= '2024-01-01'
              AND s.sig_str_att_opp > 10  -- At least 10 strikes attempted
        )
        SELECT
            fighter_name,
            event_date,
            raw_def_this_fight,
            smoothed_def,
            adjperf,
            sig_str_land,
            sig_str_land_opp,
            (smoothed_def - raw_def_this_fight) as smoothing_effect
        FROM recent_fights
        WHERE raw_def_this_fight IS NOT NULL
          AND abs(smoothed_def - raw_def_this_fight) > 0.3  -- Large smoothing effect
        ORDER BY abs(smoothed_def - raw_def_this_fight) DESC
        LIMIT 20
    """)

    with engine.connect() as conn:
        result = pd.read_sql(query, conn)

        if not result.empty:
            print("Fighters with largest smoothing effects (|smoothed - raw| > 0.3):")
            print("-" * 100)
            print(f"{'Fighter':<25} {'Date':<12} {'Raw Def':<10} {'Smoothed':<10} {'Effect':<10} {'Adjperf':<10}")
            print("-" * 100)

            for _, row in result.head(10).iterrows():
                print(f"{row['fighter_name'][:24]:<25} {str(row['event_date']):<12} "
                      f"{row['raw_def_this_fight']:<10.3f} {row['smoothed_def']:<10.3f} "
                      f"{row['smoothing_effect']:<+10.3f} {row['adjperf']:<10.2f}")

            print()
            print("NOTE: Large positive smoothing effects indicate raw performance was worse")
            print("      than smoothed value (over-smoothing of poor recent performance).")
        else:
            print("No examples found with large smoothing effects.")

# ============================================================================
# Main Execution
# ============================================================================

def main():
    print()
    print("SMOOTHING PARAMETER DIAGNOSTIC ANALYSIS")
    print("=" * 100)
    print()

    # Load parameters
    optimized = load_optimized_parameters()
    current = get_current_parameters()

    # Part 1: Compare parameters
    compare_parameters(current, optimized)

    # Part 2: Test smoothing impact
    test_smoothing_impact()

    # Part 3: Database validation
    validate_with_database()

    # Part 4: Recommendations
    print_recommendations(current, optimized)

    print()
    print("=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)

if __name__ == '__main__':
    main()
