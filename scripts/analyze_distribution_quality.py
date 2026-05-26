#!/usr/bin/env python3
"""
Distribution Quality Analysis for Adjperf Features

Analyzes all stats across all weight classes to identify degenerate distributions
and establish empirical quality thresholds for adaptive adjperf calculation.

This script:
1. Queries all stat tables and weight classes
2. Calculates distribution quality metrics
3. Identifies patterns of degeneracy
4. Generates recommendations for quality thresholds

Output: data/distribution_quality_analysis.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from libs.feature_store.feature_utils import FeatureUtils
from libs.paths import database_url, no_winsor_database_url

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URL = no_winsor_database_url()
ANALYSIS_DATE_RANGE = ('2014-01-01', '2023-01-01')  # Training period


def create_db_engine(db_url=DB_URL):
    """Create database engine with error handling."""
    if not database_exists(db_url):
        logger.error(f"Database does not exist at {db_url}")
        sys.exit(1)
    return create_engine(db_url, pool_size=10, max_overflow=20)


def get_stat_tables(engine) -> Dict[str, List[str]]:
    """Get all stat tables and their columns."""
    with engine.connect() as conn:
        feature_utils = FeatureUtils(conn)
        return feature_utils.get_stat_tables()


def analyze_stat_distribution(
    engine,
    table_name: str,
    stat_name: str,
    weightclass: str
) -> Dict[str, Any]:
    """
    Analyze distribution quality for a specific stat in a specific weight class.

    Args:
        engine: SQLAlchemy engine
        table_name: Feature table name (e.g., 'td')
        stat_name: Stat column name (e.g., 'td_def')
        weightclass: Weight class (e.g., 'heavyweight')

    Returns:
        Dictionary with quality metrics
    """
    query = text(f"""
    WITH stat_data AS (
        SELECT
            t.{stat_name} as stat_value,
            COUNT(*) OVER () as total_count
        FROM features.{table_name} t
        JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
        JOIN features.event_mapping em ON t.event_id = em.event_id
        WHERE fm.weightclass = :weightclass
          AND em.event_date BETWEEN :start_date AND :end_date
          AND t.{stat_name} IS NOT NULL
    ),
    percentiles AS (
        SELECT
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY stat_value) as median_val,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY stat_value) as p25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY stat_value) as p75,
            PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY stat_value) as p05,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY stat_value) as p95
        FROM stat_data
    ),
    mad_calc AS (
        SELECT
            PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY ABS(sd.stat_value - p.median_val)
            ) as mad_value
        FROM stat_data sd
        CROSS JOIN percentiles p
    ),
    mode_calc AS (
        SELECT
            MODE() WITHIN GROUP (ORDER BY stat_value) as mode_value
        FROM stat_data
    ),
    mode_count AS (
        SELECT
            COUNT(*)::float as mode_freq_count
        FROM stat_data sd
        CROSS JOIN mode_calc mc
        WHERE ABS(sd.stat_value - mc.mode_value) < 0.0001
    )
    SELECT
        COUNT(DISTINCT sd.stat_value) as unique_values,
        COUNT(*) as total_values,
        AVG(sd.stat_value) as mean_value,
        p.median_val,
        p.p25,
        p.p75,
        p.p05,
        p.p95,
        m.mad_value,
        STDDEV(sd.stat_value) as std_dev,
        -- Calculate % of values within 0.001 of median (effectively at prior)
        SUM(CASE WHEN ABS(sd.stat_value - p.median_val) < 0.001 THEN 1 ELSE 0 END)::float /
            COUNT(*) as pct_at_median,
        -- Most common value
        mc.mode_value,
        -- Mode frequency
        mf.mode_freq_count / COUNT(*) as mode_frequency
    FROM stat_data sd
    CROSS JOIN percentiles p
    CROSS JOIN mad_calc m
    CROSS JOIN mode_calc mc
    CROSS JOIN mode_count mf
    GROUP BY p.median_val, p.p25, p.p75, p.p05, p.p95, m.mad_value, mc.mode_value, mf.mode_freq_count
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {
                'weightclass': weightclass,
                'start_date': ANALYSIS_DATE_RANGE[0],
                'end_date': ANALYSIS_DATE_RANGE[1]
            }
        ).fetchone()

        if result is None:
            return None

        # Get wc_mad from existing table if it exists
        wc_mad_query = text(f"""
        SELECT {stat_name}_wc_mad
        FROM features.{table_name}_wc_mad
        WHERE weightclass = :weightclass
        """)

        try:
            wc_mad_result = conn.execute(
                wc_mad_query,
                {'weightclass': weightclass}
            ).fetchone()
            wc_mad = wc_mad_result[0] if wc_mad_result else None
        except Exception as e:
            logger.debug(f"Could not get wc_mad for {table_name}.{stat_name}: {e}")
            wc_mad = None

        return {
            'table': table_name,
            'stat': stat_name,
            'weightclass': weightclass,
            'unique_values': int(result[0]),
            'total_values': int(result[1]),
            'mean': float(result[2]) if result[2] is not None else None,
            'median': float(result[3]) if result[3] is not None else None,
            'p25': float(result[4]) if result[4] is not None else None,
            'p75': float(result[5]) if result[5] is not None else None,
            'p05': float(result[6]) if result[6] is not None else None,
            'p95': float(result[7]) if result[7] is not None else None,
            'mad': float(result[8]) if result[8] is not None else None,
            'std_dev': float(result[9]) if result[9] is not None else None,
            'pct_at_median': float(result[10]) if result[10] is not None else None,
            'mode_value': float(result[11]) if result[11] is not None else None,
            'mode_frequency': float(result[12]) if result[12] is not None else None,
            'wc_mad_from_table': float(wc_mad) if wc_mad is not None else None,
            # Classification helpers
            'is_degenerate': (
                result[8] is not None and result[8] < 0.010 or  # MAD < 0.01
                result[12] is not None and result[12] > 0.45     # >45% at mode
            ),
            'effective_sample_size': int(result[0]) if result[0] else 0  # unique values as proxy
        }


def get_all_weight_classes(engine) -> List[str]:
    """Get list of all weight classes in the database."""
    query = text("""
    SELECT DISTINCT weightclass
    FROM features.fight_mapping
    ORDER BY weightclass
    """)

    with engine.connect() as conn:
        result = conn.execute(query)
        return [row[0] for row in result]


def identify_adjperf_stats(columns: List[str]) -> List[str]:
    """
    Identify which columns should get adjperf treatment.

    Based on AdjPerfCalculator._is_adjperf_target logic.
    """
    adjperf_stats = []

    for col in columns:
        # Skip ID columns and total columns
        if col in ['fight_id', 'fighter_id', 'event_id']:
            continue
        if col.endswith('_total'):
            continue

        # Include stats that get adjperf
        if (col.endswith('_per_min') or
            col.endswith('_acc') or
            col.endswith('_def') or
            col.endswith('_ratio') or
            col.endswith('_pressure') or
            col in {
                'sub_att_per_ctrl', 'ground_land_per_ctrl', 'rev_per_ctrlopp',
                'sub_per_all_ctrl', 'ko_per_sig_str_land', 'sig_str_per_str_att',
                'distance_per_sig_str_land', 'clinch_per_sig_str_land',
                'ground_per_sig_str_land', 'head_per_sig_str_land',
                'body_leg_per_sig_str_land', 'td_per_sig_str_att',
                'ground_land_per_td_land', 'td_land_per_ctrl',
                'ko_sub_per_win', 'ko_sub_rd1_per_win',
                'win', 'decision', 'time_sec'
            }):
            adjperf_stats.append(col)

    return adjperf_stats


def analyze_all_distributions(engine) -> Dict[str, Any]:
    """
    Analyze all stat distributions across all weight classes.

    Returns comprehensive analysis results.
    """
    logger.info("Starting comprehensive distribution quality analysis")

    stat_tables = get_stat_tables(engine)
    weight_classes = get_all_weight_classes(engine)

    logger.info(f"Found {len(stat_tables)} stat tables")
    logger.info(f"Found {len(weight_classes)} weight classes: {weight_classes}")

    all_results = []
    total_combinations = 0
    processed = 0

    # Count total combinations
    for table_name, columns in stat_tables.items():
        adjperf_stats = identify_adjperf_stats(columns)
        total_combinations += len(adjperf_stats) * len(weight_classes)

    logger.info(f"Total stat × weightclass combinations to analyze: {total_combinations}")

    # Analyze each combination
    for table_name, columns in stat_tables.items():
        adjperf_stats = identify_adjperf_stats(columns)
        logger.info(f"Processing table {table_name}: {len(adjperf_stats)} adjperf stats")

        for stat_name in adjperf_stats:
            for weightclass in weight_classes:
                processed += 1
                if processed % 50 == 0:
                    logger.info(f"Progress: {processed}/{total_combinations} ({100*processed/total_combinations:.1f}%)")

                try:
                    result = analyze_stat_distribution(
                        engine, table_name, stat_name, weightclass
                    )
                    if result:
                        all_results.append(result)
                except Exception as e:
                    logger.warning(f"Error analyzing {table_name}.{stat_name} for {weightclass}: {e}")

    logger.info(f"Analysis complete: {len(all_results)} distributions analyzed")

    # Generate summary statistics
    df = pd.DataFrame(all_results)

    summary = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'database': DB_URL.split('@')[-1],  # Don't store credentials
            'date_range': ANALYSIS_DATE_RANGE,
            'total_distributions': len(all_results),
            'total_tables': len(stat_tables),
            'total_weight_classes': len(weight_classes)
        },
        'distributions': all_results,
        'summary_statistics': {
            'degenerate_count': int(df['is_degenerate'].sum()),
            'degenerate_pct': float(df['is_degenerate'].mean() * 100),
            'mad_distribution': {
                'min': float(df['mad'].min()) if not df['mad'].isna().all() else None,
                'p05': float(df['mad'].quantile(0.05)) if not df['mad'].isna().all() else None,
                'p10': float(df['mad'].quantile(0.10)) if not df['mad'].isna().all() else None,
                'p25': float(df['mad'].quantile(0.25)) if not df['mad'].isna().all() else None,
                'median': float(df['mad'].median()) if not df['mad'].isna().all() else None,
                'p75': float(df['mad'].quantile(0.75)) if not df['mad'].isna().all() else None,
                'p90': float(df['mad'].quantile(0.90)) if not df['mad'].isna().all() else None,
                'p95': float(df['mad'].quantile(0.95)) if not df['mad'].isna().all() else None,
                'max': float(df['mad'].max()) if not df['mad'].isna().all() else None,
            },
            'mode_frequency_distribution': {
                'min': float(df['mode_frequency'].min()) if not df['mode_frequency'].isna().all() else None,
                'p25': float(df['mode_frequency'].quantile(0.25)) if not df['mode_frequency'].isna().all() else None,
                'median': float(df['mode_frequency'].median()) if not df['mode_frequency'].isna().all() else None,
                'p75': float(df['mode_frequency'].quantile(0.75)) if not df['mode_frequency'].isna().all() else None,
                'max': float(df['mode_frequency'].max()) if not df['mode_frequency'].isna().all() else None,
            },
            'by_quality_tier': {
                'healthy': {
                    'count': int((df['mad'] >= 0.030).sum()),
                    'examples': df[df['mad'] >= 0.030].head(5)[['table', 'stat', 'weightclass', 'mad']].to_dict('records')
                },
                'sparse': {
                    'count': int(((df['mad'] >= 0.010) & (df['mad'] < 0.030)).sum()),
                    'examples': df[(df['mad'] >= 0.010) & (df['mad'] < 0.030)].head(5)[['table', 'stat', 'weightclass', 'mad']].to_dict('records')
                },
                'degenerate': {
                    'count': int((df['mad'] < 0.010).sum()),
                    'examples': df[df['mad'] < 0.010].head(10)[['table', 'stat', 'weightclass', 'mad', 'mode_frequency']].to_dict('records')
                }
            }
        }
    }

    return summary


def main():
    """Main analysis workflow."""
    logger.info("=== Distribution Quality Analysis ===")

    # Create output directory
    output_dir = project_root / 'data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'distribution_quality_analysis.json'

    # Create engine
    engine = create_db_engine()

    # Run analysis
    results = analyze_all_distributions(engine)

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total distributions analyzed: {results['metadata']['total_distributions']}")
    logger.info(f"Degenerate distributions: {results['summary_statistics']['degenerate_count']} "
                f"({results['summary_statistics']['degenerate_pct']:.1f}%)")

    logger.info("\nMAD Distribution:")
    for key, val in results['summary_statistics']['mad_distribution'].items():
        if val is not None:
            logger.info(f"  {key}: {val:.6f}")

    logger.info("\nQuality Tier Breakdown:")
    for tier, data in results['summary_statistics']['by_quality_tier'].items():
        logger.info(f"  {tier}: {data['count']} distributions")

    logger.info(f"\nFull results saved to: {output_file}")

    # Print some degenerate examples
    logger.info("\n=== Top 10 Degenerate Distributions ===")
    degenerate_examples = results['summary_statistics']['by_quality_tier']['degenerate']['examples']
    for example in degenerate_examples:
        logger.info(f"  {example['table']}.{example['stat']} ({example['weightclass']}): "
                   f"MAD={example.get('mad', 'N/A'):.6f}, "
                   f"Mode Freq={example.get('mode_frequency', 'N/A'):.1%}")


if __name__ == '__main__':
    main()
