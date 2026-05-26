#!/usr/bin/env python3
"""
Find the top 10 most interesting statistical outliers for 2025 UFC fighters.
Focuses on dec_adjperf stats that tell compelling stories about fighter performance.
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from libs.paths import database_url, no_winsor_database_url

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

def create_db_engine(db_url=DB_URL):
    """Create and configure the database engine"""
    if not database_exists(db_url):
        print(f"ERROR: Database does not exist at {db_url}")
        sys.exit(1)

    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=14400
    )

def find_outlier_for_stat(conn, stat_info):
    """
    Find the biggest outlier for a given stat.

    Args:
        conn: Database connection
        stat_info: Dictionary with 'table', 'column', 'name', 'description'

    Returns:
        Dictionary with outlier information
    """
    table = stat_info['table']
    column = stat_info['column']

    # Query to find the fighter with the highest adjperf value for this stat in 2025
    # Requirements: minimum 3 UFC fights total, at least 1 fight in 2025
    query = text(f"""
        WITH fighters_2025 AS (
            -- Get fighters who fought in 2025
            SELECT DISTINCT fm.fighter1_id as fighter_id
            FROM features.fight_mapping fm
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE em.event_date >= '2025-01-01'
              AND em.event_date < '2026-01-01'
            UNION
            SELECT DISTINCT fm.fighter2_id as fighter_id
            FROM features.fight_mapping fm
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE em.event_date >= '2025-01-01'
              AND em.event_date < '2026-01-01'
        ),
        fighter_fight_counts AS (
            -- Count total UFC fights per fighter
            SELECT
                fighter_id,
                COUNT(*) as total_fights
            FROM (
                SELECT fighter1_id as fighter_id
                FROM features.fight_mapping
                UNION ALL
                SELECT fighter2_id as fighter_id
                FROM features.fight_mapping
            ) all_fights
            GROUP BY fighter_id
            HAVING COUNT(*) >= 3  -- At least 3 UFC fights
        ),
        eligible_fighters AS (
            SELECT
                f.fighter_id,
                LOWER(f.fighter_name) as fighter_name,
                ffc.total_fights
            FROM features.fighter_mapping f
            JOIN fighters_2025 f2025 ON f.fighter_id = f2025.fighter_id
            JOIN fighter_fight_counts ffc ON f.fighter_id = ffc.fighter_id
        ),
        stat_values AS (
            SELECT
                t.fighter_id,
                t."{column}" as stat_value,
                em.event_date,
                fm.fight_id,
                ROW_NUMBER() OVER (PARTITION BY t.fighter_id ORDER BY em.event_date DESC) as rn
            FROM features."{table}" t
            JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE em.event_date >= '2025-01-01'
              AND em.event_date < '2026-01-01'
              AND t."{column}" IS NOT NULL
        )
        SELECT
            ef.fighter_name,
            sv.stat_value,
            ef.total_fights,
            sv.event_date as most_recent_fight
        FROM eligible_fighters ef
        JOIN stat_values sv ON ef.fighter_id = sv.fighter_id
        WHERE sv.rn = 1  -- Most recent fight only
        ORDER BY sv.stat_value DESC
        LIMIT 1
    """)

    result = conn.execute(query).fetchone()

    if result:
        return {
            'stat_name': stat_info['name'],
            'description': stat_info['description'],
            'fighter_name': result[0],
            'stat_value': float(result[1]),
            'total_fights': result[2],
            'most_recent_fight': result[3],
            'table': table,
            'column': column
        }
    return None

def get_ufc_average_for_stat(conn, table, column):
    """Get the UFC average for a stat in 2025"""
    query = text(f"""
        SELECT
            AVG("{column}") as avg_value,
            STDDEV("{column}") as std_value
        FROM features."{table}" t
        JOIN features.event_mapping em ON t.event_id = em.event_id
        WHERE em.event_date >= '2025-01-01'
          AND em.event_date < '2026-01-01'
          AND "{column}" IS NOT NULL
    """)

    result = conn.execute(query).fetchone()
    return {
        'avg': float(result[0]) if result[0] else 0,
        'std': float(result[1]) if result[1] else 0
    }

def main():
    """Main execution function"""
    print("=" * 100)
    print("FINDING TOP 10 MOST INTERESTING STATISTICAL OUTLIERS FOR 2025 UFC FIGHTERS")
    print("=" * 100)
    print()

    # Define the 10 most interesting stats to check
    # These are chosen for their storytelling value and fan interest
    interesting_stats = [
        {
            'name': 'Striking Accuracy',
            'description': 'How accurately does this fighter land strikes? (Precision striker)',
            'table': 'sig_str',
            'column': 'sig_str_acc_dec_adjperf_dec_avg'
        },
        {
            'name': 'Takedown Accuracy',
            'description': 'How often does this fighter successfully complete takedowns? (Wrestling dominance)',
            'table': 'td',
            'column': 'td_acc_dec_adjperf_dec_avg'
        },
        {
            'name': 'Striking Defense',
            'description': 'How well does this fighter avoid getting hit? (Defensive mastery)',
            'table': 'sig_str',
            'column': 'sig_str_def_dec_adjperf_dec_avg'
        },
        {
            'name': 'Submission Attempt Rate',
            'description': 'How often does this fighter hunt for submissions? (Submission specialist)',
            'table': 'sub',
            'column': 'sub_att_per_min_dec_adjperf_dec_avg'
        },
        {
            'name': 'Control Time Per Minute',
            'description': 'How much control time does this fighter accumulate? (Grappling control)',
            'table': 'ctrl',
            'column': 'ctrl_per_min_dec_adjperf_dec_avg'
        },
        {
            'name': 'Striking Volume',
            'description': 'How many strikes does this fighter land? (High-volume striker)',
            'table': 'sig_str',
            'column': 'sig_str_land_per_min_dec_adjperf_dec_avg'
        },
        {
            'name': 'Knockout Power per Strike',
            'description': 'How efficiently does this fighter generate knockouts? (One-punch power)',
            'table': 'ko',
            'column': 'ko_per_sig_str_land_dec_adjperf_dec_avg'
        },
        {
            'name': 'Ground Strike Accuracy',
            'description': 'How accurately does this fighter land ground strikes? (Ground-and-pound specialist)',
            'table': 'ground',
            'column': 'ground_acc_dec_adjperf_dec_avg'
        },
        {
            'name': 'Clinch Strike Volume',
            'description': 'How many strikes does this fighter land in the clinch? (Clinch warfare)',
            'table': 'clinch',
            'column': 'clinch_land_per_min_dec_adjperf_dec_avg'
        },
        {
            'name': 'Head Strike Accuracy',
            'description': 'How accurately does this fighter land head strikes? (Headhunter)',
            'table': 'head',
            'column': 'head_acc_dec_adjperf_dec_avg'
        }
    ]

    # Create database engine
    engine = create_db_engine()

    results = []

    with engine.connect() as conn:
        for i, stat_info in enumerate(interesting_stats, 1):
            print(f"\n[{i}/10] Analyzing: {stat_info['name']}")
            print(f"  {stat_info['description']}")

            try:
                outlier = find_outlier_for_stat(conn, stat_info)

                if outlier:
                    # Get UFC average for context
                    ufc_avg = get_ufc_average_for_stat(conn, stat_info['table'], stat_info['column'])
                    outlier['ufc_avg'] = ufc_avg['avg']
                    outlier['ufc_std'] = ufc_avg['std']

                    results.append(outlier)

                    print(f"  OK Outlier: {outlier['fighter_name']}")
                    print(f"    Adjusted Performance: {outlier['stat_value']:.2f}")
                    print(f"    UFC Average: {outlier['ufc_avg']:.2f}")
                    print(f"    Total UFC Fights: {outlier['total_fights']}")
                else:
                    print(f"  WARNING: No data found")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Display final results
    print("\n" + "=" * 100)
    print("TOP 10 MOST INTERESTING STATISTICAL OUTLIERS")
    print("=" * 100)
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['stat_name']}: {result['fighter_name'].title()}")
        print(f"   Adjusted Performance: {result['stat_value']:.2f} (UFC Avg: {result['ufc_avg']:.2f})")
        print(f"   {result['description']}")
        print()

    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        output_path = project_root / "data" / "top_10_interesting_outliers_2025.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    main()
