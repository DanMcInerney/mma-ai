#!/usr/bin/env python3
"""
Validation script to verify smoothing is working correctly.

This script:
1. Selects sample fights from different weight classes (30 fights for comprehensive coverage)
2. Gets historical stats for BOTH fighters up to each fight
3. Calculates expected smoothed values using the smoothing formulas
4. Compares expected vs actual smoothed values
5. Reports any discrepancies with detailed diagnostics
6. Tests edge cases (0 attempts, extreme values, etc.)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Dict, List, Tuple, Optional
import json

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import calculators to dynamically load tau parameters
from libs.feature_store.calculators.beta_binomial_calc import BetaBinomialCalculator
from libs.feature_store.calculators.poisson_gamma_smoothing_calc import PoissonGammaCalculator
from libs.feature_store.calculators.acc_calc import AccuracyCalculator
from libs.paths import database_url

# Database connection
DB_URL = database_url()

EPS = 1e-6

# Load tau parameters dynamically from calculator implementations
def load_calculator_params(engine):
    """Load tau parameters directly from calculator implementations."""
    conn = engine.connect()
    try:
        bb_calc = BetaBinomialCalculator(conn)
        pg_calc = PoissonGammaCalculator(conn)
        acc_calc = AccuracyCalculator(conn)

        beta_binomial_params = {
            'global': bb_calc.pseudo_counts.copy(),
            'per_weightclass': bb_calc.per_weightclass_pseudo_counts.copy()
        }

        poisson_gamma_params = {
            'global': pg_calc.pseudo_minutes.copy(),
            'per_weightclass': pg_calc.per_weightclass_pseudo_minutes.copy()
        }

        accuracy_params = {
            'global': acc_calc.acc_tau.copy(),
            'per_weightclass': acc_calc.per_weightclass_acc_tau.copy()
        }

        return beta_binomial_params, poisson_gamma_params, accuracy_params
    finally:
        conn.close()


def get_tau_for_stat(stat_name: str, stat_type: str, weightclass: str, params: Dict) -> float:
    """Get tau parameter for a stat, checking per-weightclass first."""
    weightclass = weightclass.lower()

    # Check per-weightclass first
    if weightclass in params.get('per_weightclass', {}):
        wc_params = params['per_weightclass'][weightclass]
        if stat_name in wc_params:
            return wc_params[stat_name]
        # Try resolving stat key (e.g., 'ko' for 'ko_rd1')
        base_name = stat_name.replace('_rd1', '')
        if base_name in wc_params:
            return wc_params[base_name]

    # Fall back to global
    if stat_name in params.get('global', {}):
        return params['global'][stat_name]

    # Try base name
    base_name = stat_name.replace('_rd1', '')
    if base_name in params.get('global', {}):
        return params['global'][base_name]

    # Default
    return params.get('global', {}).get('default', 10.0)


def calculate_beta_binomial_smooth(
    successes: float,
    attempts: float,
    prior_rate: float,
    tau: float
) -> float:
    """Calculate Beta-Binomial smoothed probability."""
    if attempts == 0:
        return prior_rate

    smoothed_successes = successes + (tau * prior_rate)
    smoothed_attempts = attempts + tau
    return smoothed_successes / smoothed_attempts


def calculate_poisson_gamma_smooth(
    count: float,
    exposure_min: float,
    prior_rate: float,
    tau: float
) -> float:
    """Calculate Poisson-Gamma smoothed count."""
    if exposure_min <= 0:
        return count

    # Posterior rate
    posterior_rate = (prior_rate * tau + count) / (tau + exposure_min)
    # Smoothed count
    return exposure_min * posterior_rate


def get_historical_prior(
    conn,
    fighter_id: int,
    fight_date: str,
    stat_col: str,
    stat_type: str,
    weightclass: str,
    exposure_type: str = 'fight'
) -> Tuple[float, int]:
    """
    Get historical prior for a stat.

    CRITICAL: Uses fight_stats_fe (raw data) to calculate priors, NOT fight_stats_derived (smoothed data).
    Using smoothed data would create circular reasoning and invalid validation.

    Returns:
        (prior_rate, sample_size)
    """
    schema = 'features'
    table = 'fight_stats_fe'  # Use RAW data for priors, not smoothed!

    # Determine attempts/exposure expression
    if stat_type == 'beta_binomial':
        if stat_col.startswith('ctrl'):
            if exposure_type == 'rd1':
                attempts_expr = "LEAST(COALESCE(fd.time_sec_rd1, fd.time_sec), 300)"
            else:
                attempts_expr = "fd.time_sec"
        elif stat_col.startswith('sub_land'):
            attempts_col = stat_col.replace('sub_land', 'sub_att')
            attempts_expr = f"fd.{attempts_col}"
        else:
            attempts_expr = "1"  # Binary outcomes

        # Calculate WEIGHT CLASS prior (not per-fighter!)
        # The calculators use weight class averages, not individual fighter history
        query = text(f"""
            SELECT
                SUM(fd.{stat_col}::float) as total_successes,
                SUM({attempts_expr}) as total_attempts,
                COUNT(DISTINCT fd.fighter_id) as sample_size
            FROM {schema}.{table} fd
            JOIN {schema}.fight_mapping fm ON fd.fight_id = fm.fight_id
            JOIN {schema}.event_mapping em ON fd.event_id = em.event_id
            WHERE em.event_date < :fight_date
              AND em.event_date >= '2014-01-01'
              AND LOWER(fm.weightclass) = LOWER(:weightclass)
              AND {attempts_expr} > 0
        """)

        result = conn.execute(query, {
            'fight_date': fight_date,
            'weightclass': weightclass
        }).fetchone()

        if result and result[1] and float(result[1]) > 0:
            prior_rate = float(result[0]) / float(result[1])
            return prior_rate, int(result[2]) if result[2] else 0
        else:
            # Fall back to global prior
            query_global = text(f"""
                SELECT
                    SUM(fd.{stat_col}::float) as total_successes,
                    SUM({attempts_expr}) as total_attempts
                FROM {schema}.{table} fd
                JOIN {schema}.event_mapping em ON fd.event_id = em.event_id
                WHERE em.event_date < :fight_date
                  AND em.event_date >= '2014-01-01'
                  AND {attempts_expr} > 0
            """)
            result_global = conn.execute(query_global, {'fight_date': fight_date}).fetchone()
            if result_global and result_global[1] and float(result_global[1]) > 0:
                return float(result_global[0]) / float(result_global[1]), 0
            return 0.0, 0

    elif stat_type == 'poisson_gamma':
        if exposure_type == 'rd1':
            exposure_expr = "LEAST(COALESCE(fd.time_sec_rd1, fd.time_sec), 300) / 60.0"
        else:
            exposure_expr = "fd.time_sec / 60.0"

        # Calculate WEIGHT CLASS prior (not per-fighter!)
        query = text(f"""
            SELECT
                SUM(fd.{stat_col}::float) as total_count,
                SUM({exposure_expr}) as total_exposure,
                COUNT(DISTINCT fd.fighter_id) as sample_size
            FROM {schema}.{table} fd
            JOIN {schema}.fight_mapping fm ON fd.fight_id = fm.fight_id
            JOIN {schema}.event_mapping em ON fd.event_id = em.event_id
            WHERE em.event_date < :fight_date
              AND em.event_date >= '2014-01-01'
              AND LOWER(fm.weightclass) = LOWER(:weightclass)
              AND fd.time_sec > 0
        """)

        result = conn.execute(query, {
            'fight_date': fight_date,
            'weightclass': weightclass
        }).fetchone()

        if result and result[1] and float(result[1]) > 0:
            prior_rate = float(result[0]) / float(result[1])
            return prior_rate, int(result[2]) if result[2] else 0
        else:
            # Fall back to global prior
            query_global = text(f"""
                SELECT
                    SUM(fd.{stat_col}::float) as total_count,
                    SUM({exposure_expr}) as total_exposure
                FROM {schema}.{table} fd
                JOIN {schema}.event_mapping em ON fd.event_id = em.event_id
                WHERE em.event_date < :fight_date
                  AND em.event_date >= '2014-01-01'
                  AND fd.time_sec > 0
            """)
            result_global = conn.execute(query_global, {'fight_date': fight_date}).fetchone()
            if result_global and result_global[1] and float(result_global[1]) > 0:
                return float(result_global[0]) / float(result_global[1]), 0
            return 0.0, 0

    return 0.0, 0


def get_fight_stats(conn, fight_id: int, fighter_id: int) -> Optional[Dict]:
    """Get raw and smoothed stats for a fighter in a specific fight."""
    # Get smoothed stats from fight_stats_derived
    query_smooth = text("""
        SELECT
            fd.*,
            fm.weightclass,
            em.event_date
        FROM features.fight_stats_derived fd
        JOIN features.fight_mapping fm ON fd.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fd.event_id = em.event_id
        WHERE fd.fight_id = :fight_id
          AND fd.fighter_id = :fighter_id
    """)

    df_smooth = pd.read_sql(query_smooth, conn, params={'fight_id': fight_id, 'fighter_id': fighter_id})
    if df_smooth.empty:
        return None

    # Get raw stats from fight_stats_fe
    query_raw = text("""
        SELECT
            fe.*
        FROM features.fight_stats_fe fe
        WHERE fe.fight_id = :fight_id
          AND fe.fighter_id = :fighter_id
    """)

    df_raw = pd.read_sql(query_raw, conn, params={'fight_id': fight_id, 'fighter_id': fighter_id})

    # Combine: smoothed values from derived, raw values from fe
    result = df_smooth.iloc[0].to_dict()
    if not df_raw.empty:
        raw_dict = df_raw.iloc[0].to_dict()
        # Add _raw suffix to raw values for comparison
        for key, value in raw_dict.items():
            if key not in ['fight_id', 'fighter_id', 'event_id']:
                result[f"{key}_raw"] = value

    return result


def validate_fighter_stats(
    conn,
    fighter_id: int,
    fighter_name: str,
    fighter_stats: Dict,
    fight_date: str,
    weightclass: str,
    bb_params: Dict,
    pg_params: Dict,
    validations: List[Dict]
) -> None:
    """Validate smoothing for a single fighter and append results to validations list."""

    # Beta-Binomial stats to validate
    bb_stats = ['ko', 'win', 'decision', 'sub_land', 'ctrl', 'ko_rd1', 'win_rd1', 'sub_land_rd1', 'ctrl_rd1']

    for stat in bb_stats:
        if stat not in fighter_stats:
            continue

        # Get historical prior
        exposure_type = 'rd1' if stat.endswith('_rd1') else 'fight'
        prior_rate, sample_size = get_historical_prior(
            conn, fighter_id, fight_date, stat, 'beta_binomial', weightclass, exposure_type
        )

        # Get tau
        tau = get_tau_for_stat(stat, 'beta_binomial', weightclass, bb_params)

        # Get raw value (from _raw column if it exists, otherwise from original)
        raw_col = f"{stat}_raw"
        raw_value = fighter_stats.get(raw_col, fighter_stats.get(stat, 0))

        # Determine attempts
        if stat.startswith('ctrl'):
            if exposure_type == 'rd1':
                attempts = min(fighter_stats.get('time_sec_rd1', fighter_stats.get('time_sec', 0)), 300)
            else:
                attempts = fighter_stats.get('time_sec', 0)
        elif stat.startswith('sub_land'):
            attempts_col = stat.replace('sub_land', 'sub_att')
            attempts_raw_col = f"{attempts_col}_raw"
            attempts = fighter_stats.get(attempts_raw_col, fighter_stats.get(attempts_col, 0))
        else:
            attempts = 1

        # Calculate expected smoothed value
        if attempts > 0:
            expected_smooth = calculate_beta_binomial_smooth(raw_value, attempts, prior_rate, tau)
            if stat.startswith('ctrl'):
                expected_smooth = expected_smooth * attempts  # Control time returns seconds
        else:
            expected_smooth = prior_rate if not stat.startswith('ctrl') else 0

        # Get actual smoothed value (smoothed values are in original column, raw in _raw)
        actual_raw = fighter_stats.get(raw_col)
        actual_smooth = fighter_stats.get(stat)  # Smoothed value is in original column

        # Only validate if we have both raw and smoothed
        if actual_smooth is not None and actual_raw is not None:
            diff = abs(expected_smooth - actual_smooth)
            rel_diff = diff / (abs(actual_smooth) + EPS) if actual_smooth != 0 else diff

            validations.append({
                'stat': stat,
                'stat_type': 'beta_binomial',
                'fighter': fighter_name,
                'fighter_id': fighter_id,
                'raw_value': float(raw_value) if raw_value is not None else None,
                'attempts': float(attempts) if attempts is not None else None,
                'prior_rate': float(prior_rate),
                'tau': float(tau),
                'expected_smooth': float(expected_smooth),
                'actual_smooth': float(actual_smooth),
                'difference': float(diff),
                'relative_diff_pct': float(rel_diff * 100),
                'sample_size': int(sample_size),
                'edge_case': 'zero_attempts' if attempts == 0 else None,
                'status': 'PASS' if rel_diff < 0.01 else 'FAIL'  # 1% tolerance
            })

    # Poisson-Gamma stats to validate
    pg_stats = [
        'sig_str_land', 'head_land', 'body_land', 'leg_land',
        'distance_land', 'clinch_land', 'ground_land',
        'td_land', 'kd', 'rev', 'sub_att',
        'sig_str_land_rd1', 'head_land_rd1', 'body_land_rd1', 'leg_land_rd1',
        'distance_land_rd1', 'clinch_land_rd1', 'ground_land_rd1',
        'td_land_rd1', 'kd_rd1', 'rev_rd1', 'sub_att_rd1'
    ]

    for stat in pg_stats:
        if stat not in fighter_stats:
            continue

        # Get historical prior
        exposure_type = 'rd1' if stat.endswith('_rd1') else 'fight'
        prior_rate, sample_size = get_historical_prior(
            conn, fighter_id, fight_date, stat, 'poisson_gamma', weightclass, exposure_type
        )

        # Get tau - need to resolve stat key properly
        if stat.endswith('_land_rd1'):
            base_stat = stat.replace('_land_rd1', '_rd1')
        elif stat.endswith('_att_rd1'):
            base_stat = stat.replace('_att_rd1', '_rd1')
        elif stat.endswith('_land'):
            base_stat = stat.replace('_land', '')
        elif stat.endswith('_att'):
            base_stat = stat.replace('_att', '')
        else:
            base_stat = stat

        tau = get_tau_for_stat(base_stat, 'poisson_gamma', weightclass, pg_params)

        # Get raw value (from _raw column if it exists, otherwise from original)
        raw_col = f"{stat}_raw"
        raw_value = fighter_stats.get(raw_col, fighter_stats.get(stat, 0))

        # Get exposure
        if exposure_type == 'rd1':
            time_sec_rd1 = fighter_stats.get('time_sec_rd1_raw', fighter_stats.get('time_sec_rd1', fighter_stats.get('time_sec_raw', fighter_stats.get('time_sec', 0))))
            time_sec = fighter_stats.get('time_sec_raw', fighter_stats.get('time_sec', 0))
            exposure_min = min(time_sec_rd1, 300) / 60.0 if time_sec_rd1 else time_sec / 60.0
        else:
            time_sec = fighter_stats.get('time_sec_raw', fighter_stats.get('time_sec', 0))
            exposure_min = time_sec / 60.0

        # Calculate expected smoothed value
        if exposure_min > 0:
            expected_smooth = calculate_poisson_gamma_smooth(raw_value, exposure_min, prior_rate, tau)
        else:
            expected_smooth = raw_value

        # Get actual smoothed value (smoothed values are in original column, raw in _raw)
        actual_raw = fighter_stats.get(raw_col)
        actual_smooth = fighter_stats.get(stat)  # Smoothed value is in original column

        # Only validate if we have both raw and smoothed
        if actual_smooth is not None and actual_raw is not None:
            diff = abs(expected_smooth - actual_smooth)
            rel_diff = diff / (abs(actual_smooth) + EPS) if actual_smooth != 0 else diff

            validations.append({
                'stat': stat,
                'stat_type': 'poisson_gamma',
                'fighter': fighter_name,
                'fighter_id': fighter_id,
                'raw_value': float(raw_value) if raw_value is not None else None,
                'exposure_min': float(exposure_min),
                'prior_rate': float(prior_rate),
                'tau': float(tau),
                'expected_smooth': float(expected_smooth),
                'actual_smooth': float(actual_smooth),
                'difference': float(diff),
                'relative_diff_pct': float(rel_diff * 100),
                'sample_size': int(sample_size),
                'edge_case': 'zero_exposure' if exposure_min == 0 else None,
                'status': 'PASS' if rel_diff < 0.01 else 'FAIL'  # 1% tolerance
            })


def validate_smoothing_for_fight(
    conn,
    fight_id: int,
    fighter1_id: int,
    fighter2_id: int,
    fighter1_name: str,
    fighter2_name: str,
    fight_date: str,
    weightclass: str,
    bb_params: Dict,
    pg_params: Dict,
    acc_params: Dict
) -> Dict:
    """Validate smoothing for BOTH fighters in a fight."""
    results = {
        'fight_id': fight_id,
        'fight_date': fight_date,
        'weightclass': weightclass,
        'fighter1': fighter1_name,
        'fighter2': fighter2_name,
        'validations': []
    }

    # Get fight stats for both fighters
    fighter1_stats = get_fight_stats(conn, fight_id, fighter1_id)
    fighter2_stats = get_fight_stats(conn, fight_id, fighter2_id)

    if not fighter1_stats or not fighter2_stats:
        results['error'] = 'Could not fetch fight stats'
        return results

    # Validate Fighter 1
    validate_fighter_stats(
        conn, fighter1_id, fighter1_name, fighter1_stats,
        fight_date, weightclass, bb_params, pg_params, results['validations']
    )

    # Validate Fighter 2
    validate_fighter_stats(
        conn, fighter2_id, fighter2_name, fighter2_stats,
        fight_date, weightclass, bb_params, pg_params, results['validations']
    )

    return results


def get_sample_fights(conn, n_fights_per_wc: int = 4) -> List[Dict]:
    """
    Get sample fights ensuring coverage across all weight classes.

    Args:
        n_fights_per_wc: Number of fights to sample per weight class

    Returns:
        List of fight dictionaries
    """
    weight_classes = [
        'flyweight', 'bantamweight', 'featherweight', 'lightweight',
        'welterweight', 'middleweight', 'light heavyweight', 'heavyweight'
    ]

    all_fights = []

    for wc in weight_classes:
        query = text("""
            SELECT
                fm.fight_id,
                fm.fighter1_id,
                fm.fighter2_id,
                fm1.fighter_name as fighter1_name,
                fm2.fighter_name as fighter2_name,
                em.event_date,
                fm.weightclass
            FROM features.fight_mapping fm
            JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
            JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE em.event_date >= '2020-01-01'
              AND em.event_date < '2024-01-01'
              AND LOWER(fm.weightclass) = LOWER(:weightclass)
            ORDER BY RANDOM()
            LIMIT :n_fights
        """)

        results = conn.execute(query, {'weightclass': wc, 'n_fights': n_fights_per_wc}).fetchall()

        for r in results:
            all_fights.append({
                'fight_id': r[0],
                'fighter1_id': r[1],
                'fighter2_id': r[2],
                'fighter1_name': r[3],
                'fighter2_name': r[4],
                'fight_date': str(r[5]),
                'weightclass': r[6]
            })

    return all_fights


def main():
    """Main validation function."""
    print("=" * 80)
    print("SMOOTHING VALIDATION SCRIPT")
    print("=" * 80)
    print()

    engine = create_engine(DB_URL)

    # Load tau parameters from calculators
    print("Loading tau parameters from calculators...")
    bb_params, pg_params, acc_params = load_calculator_params(engine)
    print(f"  Beta-Binomial: {len(bb_params['global'])} global params, "
          f"{len(bb_params['per_weightclass'])} weight classes with overrides")
    print(f"  Poisson-Gamma: {len(pg_params['global'])} global params, "
          f"{len(pg_params['per_weightclass'])} weight classes with overrides")
    print(f"  Accuracy: {len(acc_params['global'])} global params, "
          f"{len(acc_params['per_weightclass'])} weight classes with overrides")
    print()

    conn = engine.connect()

    # Get sample fights with weight class coverage
    print("Selecting sample fights (4 per weight class = 32 total)...")
    sample_fights = get_sample_fights(conn, n_fights_per_wc=4)
    print(f"Selected {len(sample_fights)} fights")

    # Print weight class coverage
    wc_counts = {}
    for fight in sample_fights:
        wc = fight['weightclass']
        wc_counts[wc] = wc_counts.get(wc, 0) + 1
    print(f"Weight class coverage: {dict(sorted(wc_counts.items()))}")
    print()

    all_results = []

    for i, fight in enumerate(sample_fights, 1):
        print(f"[{i}/{len(sample_fights)}] Validating: {fight['fighter1_name']} vs {fight['fighter2_name']}")
        print(f"  Date: {fight['fight_date']}, Weightclass: {fight['weightclass']}")

        result = validate_smoothing_for_fight(
            conn,
            fight['fight_id'],
            fight['fighter1_id'],
            fight['fighter2_id'],
            fight['fighter1_name'],
            fight['fighter2_name'],
            fight['fight_date'],
            fight['weightclass'],
            bb_params,
            pg_params,
            acc_params
        )

        all_results.append(result)

        # Print summary
        validations = result.get('validations', [])
        if validations:
            passed = sum(1 for v in validations if v['status'] == 'PASS')
            failed = sum(1 for v in validations if v['status'] == 'FAIL')
            print(f"  Results: {passed} passed, {failed} failed (both fighters)")

            # Show failures
            failures = [v for v in validations if v['status'] == 'FAIL']
            if failures:
                for v in failures[:3]:  # Show first 3 failures
                    print(f"    FAIL: {v['fighter']} - {v['stat']} ({v['stat_type']})")
                    print(f"      Expected: {v['expected_smooth']:.4f}, Actual: {v['actual_smooth']:.4f}")
                    print(f"      Diff: {v['relative_diff_pct']:.2f}%")
                if len(failures) > 3:
                    print(f"    ... and {len(failures) - 3} more failures")
        print()

    # Save results
    output_path = project_root / "data" / "smoothing_validation_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"Results saved to: {output_path}")
    print()

    # Summary statistics
    all_validations = []
    for result in all_results:
        all_validations.extend(result.get('validations', []))

    if all_validations:
        passed = sum(1 for v in all_validations if v['status'] == 'PASS')
        failed = sum(1 for v in all_validations if v['status'] == 'FAIL')
        total = len(all_validations)

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total validations: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")

        # Breakdown by stat type
        bb_validations = [v for v in all_validations if v['stat_type'] == 'beta_binomial']
        pg_validations = [v for v in all_validations if v['stat_type'] == 'poisson_gamma']

        bb_passed = sum(1 for v in bb_validations if v['status'] == 'PASS')
        pg_passed = sum(1 for v in pg_validations if v['status'] == 'PASS')

        print(f"\nBeta-Binomial: {bb_passed}/{len(bb_validations)} passed ({bb_passed/len(bb_validations)*100:.1f}%)")
        print(f"Poisson-Gamma: {pg_passed}/{len(pg_validations)} passed ({pg_passed/len(pg_validations)*100:.1f}%)")

        # Edge cases
        edge_cases = [v for v in all_validations if v.get('edge_case')]
        if edge_cases:
            print(f"\nEdge cases found: {len(edge_cases)}")
            for ec_type in ['zero_attempts', 'zero_exposure']:
                ec_count = sum(1 for v in edge_cases if v.get('edge_case') == ec_type)
                if ec_count > 0:
                    ec_passed = sum(1 for v in edge_cases if v.get('edge_case') == ec_type and v['status'] == 'PASS')
                    print(f"  {ec_type}: {ec_passed}/{ec_count} passed")

        if failed > 0:
            print("\n" + "=" * 80)
            print("FAILED VALIDATIONS")
            print("=" * 80)

            # Group failures by stat
            failures_by_stat = {}
            for v in all_validations:
                if v['status'] == 'FAIL':
                    stat = v['stat']
                    if stat not in failures_by_stat:
                        failures_by_stat[stat] = []
                    failures_by_stat[stat].append(v)

            for stat, failures in sorted(failures_by_stat.items()):
                avg_diff = np.mean([v['relative_diff_pct'] for v in failures])
                max_diff = max([v['relative_diff_pct'] for v in failures])
                print(f"\n{stat} ({failures[0]['stat_type']}): {len(failures)} failures")
                print(f"  Avg diff: {avg_diff:.2f}%, Max diff: {max_diff:.2f}%")
                print(f"  Sample failures:")
                for v in failures[:2]:  # Show first 2
                    print(f"    {v['fighter']}: expected={v['expected_smooth']:.4f}, actual={v['actual_smooth']:.4f}, "
                          f"raw={v['raw_value']:.4f}, tau={v['tau']:.2f}")

    conn.close()
    print("\n" + "=" * 80)
    print("Validation complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
