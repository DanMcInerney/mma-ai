#!/usr/bin/env python3
"""
Raw Data Sparsity Analysis

Analyzes fight_stats_fe (raw unsmoothed data) to measure:
1. % of fights with zero attempts/opportunities for each stat
2. Sample sizes by weight class
3. Root causes of degenerate distributions

This complements analyze_distribution_quality.py which analyzes smoothed values.

Output: data/raw_sparsity_analysis.json
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
import json
from datetime import datetime
from typing import Dict, List, Any
import logging
from libs.paths import database_url, no_winsor_database_url

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DB_URL = no_winsor_database_url()
ANALYSIS_DATE_RANGE = ('2014-01-01', '2023-01-01')

# Map feature tables to fight_stats_fe columns and their attempt columns
STAT_MAPPINGS = {
    # Accuracy stats (land/att pairs)
    'td_acc': ('td_land', 'td_att'),
    'td_def': ('td_def_land', 'td_def_att'),  # Defensive perspective
    'sig_str_acc': ('sig_str_land', 'sig_str_att'),
    'sig_str_def': ('sig_str_def_land', 'sig_str_def_att'),
    'distance_acc': ('distance_land', 'distance_att'),
    'distance_def': ('distance_def_land', 'distance_def_att'),
    'clinch_acc': ('clinch_land', 'clinch_att'),
    'clinch_def': ('clinch_def_land', 'clinch_def_att'),
    'ground_acc': ('ground_land', 'ground_att'),
    'ground_def': ('ground_def_land', 'ground_def_att'),
    'head_acc': ('head_land', 'head_att'),
    'head_def': ('head_def_land', 'head_def_att'),
    'body_acc': ('body_land', 'body_att'),
    'body_def': ('body_def_land', 'body_def_att'),
    'leg_acc': ('leg_land', 'leg_att'),
    'leg_def': ('leg_def_land', 'leg_def_att'),

    # Per-minute rate stats (need fight duration)
    'sig_str_per_min': ('sig_str_land', 'time_sec'),
    'td_per_min': ('td_land', 'time_sec'),
    'sub_att_per_min': ('sub_att', 'time_sec'),

    # Submission stats
    'sub_att': ('sub_att', None),  # Count stat
    'sub_land': ('sub_land', None),  # Binary outcome

    # Control stats
    'ctrl': ('ctrl', None),  # Duration

    # Knockout stats
    'ko': ('ko', None),  # Binary outcome
    'kd': ('kd', None),  # Count stat
}


def create_db_engine(db_url=DB_URL):
    """Create database engine with error handling."""
    if not database_exists(db_url):
        logger.error(f"Database does not exist at {db_url}")
        sys.exit(1)
    return create_engine(db_url, pool_size=10, max_overflow=20)


def analyze_sparsity_for_stat(
    engine,
    stat_name: str,
    land_col: str,
    att_col: str,
    weightclass: str
) -> Dict[str, Any]:
    """
    Analyze raw data sparsity for a specific stat.

    Args:
        engine: SQLAlchemy engine
        stat_name: Name of the stat (e.g., 'td_acc')
        land_col: Column name for landed attempts
        att_col: Column name for total attempts (None for count stats)
        weightclass: Weight class

    Returns:
        Dictionary with sparsity metrics
    """
    # Different query for accuracy stats vs count/rate stats
    if att_col and att_col != 'time_sec':
        # Accuracy stat: measure % with zero attempts
        query = text(f"""
        SELECT
            COUNT(*) as total_fights,
            SUM(CASE WHEN fe.{att_col} = 0 THEN 1 ELSE 0 END) as zero_attempt_fights,
            SUM(CASE WHEN fe.{att_col} > 0 THEN 1 ELSE 0 END) as measurable_fights,
            AVG(CASE WHEN fe.{att_col} > 0 THEN fe.{land_col}::float / fe.{att_col} END) as avg_when_measurable,
            STDDEV(CASE WHEN fe.{att_col} > 0 THEN fe.{land_col}::float / fe.{att_col} END) as std_when_measurable,
            PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY CASE WHEN fe.{att_col} > 0 THEN fe.{land_col}::float / fe.{att_col} END
            ) as median_when_measurable,
            COUNT(DISTINCT CASE WHEN fe.{att_col} > 0 THEN fe.{land_col}::float / fe.{att_col} END) as unique_values_when_measurable
        FROM fight_stats_fe fe
        JOIN features.event_mapping em ON fe.event_id = em.event_id
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        WHERE fm.weightclass = :weightclass
          AND em.event_date BETWEEN :start_date AND :end_date
          AND fe.{att_col} IS NOT NULL
        """)
    elif att_col == 'time_sec':
        # Rate stat: always measurable if fight happened
        query = text(f"""
        SELECT
            COUNT(*) as total_fights,
            0 as zero_attempt_fights,
            COUNT(*) as measurable_fights,
            AVG(fe.{land_col}::float * 60 / NULLIF(fe.{att_col}, 0)) as avg_when_measurable,
            STDDEV(fe.{land_col}::float * 60 / NULLIF(fe.{att_col}, 0)) as std_when_measurable,
            PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY fe.{land_col}::float * 60 / NULLIF(fe.{att_col}, 0)
            ) as median_when_measurable,
            COUNT(DISTINCT fe.{land_col}::float * 60 / NULLIF(fe.{att_col}, 0)) as unique_values_when_measurable
        FROM fight_stats_fe fe
        JOIN features.event_mapping em ON fe.event_id = em.event_id
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        WHERE fm.weightclass = :weightclass
          AND em.event_date BETWEEN :start_date AND :end_date
          AND fe.{att_col} > 0
        """)
    else:
        # Count stat: measure % with zero occurrences
        query = text(f"""
        SELECT
            COUNT(*) as total_fights,
            SUM(CASE WHEN fe.{land_col} = 0 THEN 1 ELSE 0 END) as zero_attempt_fights,
            SUM(CASE WHEN fe.{land_col} > 0 THEN 1 ELSE 0 END) as measurable_fights,
            AVG(CASE WHEN fe.{land_col} > 0 THEN fe.{land_col}::float END) as avg_when_measurable,
            STDDEV(CASE WHEN fe.{land_col} > 0 THEN fe.{land_col}::float END) as std_when_measurable,
            PERCENTILE_CONT(0.50) WITHIN GROUP (
                ORDER BY CASE WHEN fe.{land_col} > 0 THEN fe.{land_col}::float END
            ) as median_when_measurable,
            COUNT(DISTINCT CASE WHEN fe.{land_col} > 0 THEN fe.{land_col} END) as unique_values_when_measurable
        FROM fight_stats_fe fe
        JOIN features.event_mapping em ON fe.event_id = em.event_id
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        WHERE fm.weightclass = :weightclass
          AND em.event_date BETWEEN :start_date AND :end_date
          AND fe.{land_col} IS NOT NULL
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

        if result is None or result[0] == 0:
            return None

        total_fights = int(result[0])
        zero_attempts = int(result[1])
        measurable = int(result[2])

        return {
            'stat': stat_name,
            'weightclass': weightclass,
            'total_fights': total_fights,
            'zero_attempt_fights': zero_attempts,
            'measurable_fights': measurable,
            'pct_zero_attempts': float(zero_attempts) / total_fights if total_fights > 0 else None,
            'pct_measurable': float(measurable) / total_fights if total_fights > 0 else None,
            'avg_when_measurable': float(result[3]) if result[3] is not None else None,
            'std_when_measurable': float(result[4]) if result[4] is not None else None,
            'median_when_measurable': float(result[5]) if result[5] is not None else None,
            'unique_values_when_measurable': int(result[6]) if result[6] is not None else 0,
            'is_sparse': (zero_attempts / total_fights) > 0.25 if total_fights > 0 else False,
            'is_very_sparse': (zero_attempts / total_fights) > 0.45 if total_fights > 0 else False,
        }


def analyze_all_raw_sparsity(engine) -> Dict[str, Any]:
    """Analyze raw data sparsity across all stats and weight classes."""
    logger.info("Starting raw data sparsity analysis")

    # Get weight classes
    query = text("""
    SELECT DISTINCT weightclass
    FROM features.fight_mapping
    ORDER BY weightclass
    """)

    with engine.connect() as conn:
        weight_classes = [row[0] for row in conn.execute(query)]

    logger.info(f"Found {len(weight_classes)} weight classes")
    logger.info(f"Analyzing {len(STAT_MAPPINGS)} stats")

    all_results = []
    total_combinations = len(STAT_MAPPINGS) * len(weight_classes)
    processed = 0

    for stat_name, (land_col, att_col) in STAT_MAPPINGS.items():
        for weightclass in weight_classes:
            processed += 1
            if processed % 50 == 0:
                logger.info(f"Progress: {processed}/{total_combinations} ({100*processed/total_combinations:.1f}%)")

            try:
                result = analyze_sparsity_for_stat(
                    engine, stat_name, land_col, att_col, weightclass
                )
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Error analyzing {stat_name} for {weightclass}: {e}")

    logger.info(f"Analysis complete: {len(all_results)} stat×weightclass combinations analyzed")

    # Generate summary
    df = pd.DataFrame(all_results)

    summary = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'database': DB_URL.split('@')[-1],
            'date_range': ANALYSIS_DATE_RANGE,
            'total_combinations': len(all_results),
            'total_stats': len(STAT_MAPPINGS),
            'total_weight_classes': len(weight_classes)
        },
        'raw_sparsity': all_results,
        'summary_statistics': {
            'sparse_count': int(df['is_sparse'].sum()),
            'sparse_pct': float(df['is_sparse'].mean() * 100),
            'very_sparse_count': int(df['is_very_sparse'].sum()),
            'very_sparse_pct': float(df['is_very_sparse'].mean() * 100),
            'sparsity_distribution': {
                'min': float(df['pct_zero_attempts'].min()) if not df['pct_zero_attempts'].isna().all() else None,
                'p05': float(df['pct_zero_attempts'].quantile(0.05)) if not df['pct_zero_attempts'].isna().all() else None,
                'p25': float(df['pct_zero_attempts'].quantile(0.25)) if not df['pct_zero_attempts'].isna().all() else None,
                'median': float(df['pct_zero_attempts'].median()) if not df['pct_zero_attempts'].isna().all() else None,
                'p75': float(df['pct_zero_attempts'].quantile(0.75)) if not df['pct_zero_attempts'].isna().all() else None,
                'p95': float(df['pct_zero_attempts'].quantile(0.95)) if not df['pct_zero_attempts'].isna().all() else None,
                'max': float(df['pct_zero_attempts'].max()) if not df['pct_zero_attempts'].isna().all() else None,
            },
            'top_sparse_stats': df.nlargest(20, 'pct_zero_attempts')[
                ['stat', 'weightclass', 'pct_zero_attempts', 'total_fights', 'measurable_fights']
            ].to_dict('records')
        }
    }

    return summary


def main():
    """Main analysis workflow."""
    logger.info("=== Raw Data Sparsity Analysis ===")

    # Create output directory
    output_dir = project_root / 'data'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'raw_sparsity_analysis.json'

    # Create engine
    engine = create_db_engine()

    # Run analysis
    results = analyze_all_raw_sparsity(engine)

    # Save results
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n=== Analysis Summary ===")
    logger.info(f"Total combinations analyzed: {results['metadata']['total_combinations']}")
    logger.info(f"Sparse stats (>25% zero attempts): {results['summary_statistics']['sparse_count']} "
                f"({results['summary_statistics']['sparse_pct']:.1f}%)")
    logger.info(f"Very sparse stats (>45% zero attempts): {results['summary_statistics']['very_sparse_count']} "
                f"({results['summary_statistics']['very_sparse_pct']:.1f}%)")

    logger.info("\nSparsity Distribution:")
    for key, val in results['summary_statistics']['sparsity_distribution'].items():
        if val is not None:
            logger.info(f"  {key}: {val:.1%}")

    logger.info(f"\n=== Top 20 Sparsest Stats ===")
    for item in results['summary_statistics']['top_sparse_stats']:
        logger.info(f"  {item['stat']} ({item['weightclass']}): "
                   f"{item['pct_zero_attempts']:.1%} zero attempts "
                   f"({item['measurable_fights']}/{item['total_fights']} measurable)")

    logger.info(f"\nFull results saved to: {output_file}")


if __name__ == '__main__':
    main()
