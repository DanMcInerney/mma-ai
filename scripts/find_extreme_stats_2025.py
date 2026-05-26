#!/usr/bin/env python3
"""
Find the highest and lowest stat fighters who fought at least once in 2025
for each feature in vSeven_testing2.

This script queries the database to find fighters who fought in 2025 and
identifies the fighters with the highest and lowest values for each feature.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from contextlib import contextmanager
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from libs.feature_store.features import vSeven_testing2
from libs.paths import database_url, no_winsor_database_url

# Database connection URL
# Change this to 'mma-ai-no-winsor' if you want to query the non-winsorized database
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

@contextmanager
def get_db_connection(engine):
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def get_fighters_2025(conn):
    """Get all unique fighters who fought at least once in 2025 AND have at least 3 UFC fights total"""
    query = text("""
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
        )
        SELECT 
            f.fighter_id, 
            LOWER(f.fighter_name) as fighter_name,
            ffc.total_fights
        FROM features.fighter_mapping f
        JOIN fighters_2025 f2025 ON f.fighter_id = f2025.fighter_id
        JOIN fighter_fight_counts ffc ON f.fighter_id = ffc.fighter_id
        ORDER BY fighter_name
    """)
    
    df = pd.read_sql(query, conn)
    print(f"Found {len(df)} fighters who fought in 2025 with at least 3 UFC fights")
    if len(df) > 0:
        print(f"  Fight count range: {df['total_fights'].min()} - {df['total_fights'].max()} fights")
        print(f"  Average fights: {df['total_fights'].mean():.1f}")
    return df

def get_feature_table_and_column(feature_name):
    """
    Determine which table and column contains a given feature.
    Features ending in _diff are difference features, so we need to find
    the base feature name and its table.
    """
    # Remove _diff suffix if present
    base_feature = feature_name.replace('_diff', '')
    
    # Map feature prefixes to table names
    feature_to_table = {
        'age': 'age',
        'reach': 'reach',
        'days_since_last_fight': 'days_since_last_fight',
        'weightclass_encoded': 'fight_mapping',  # Special case
        'sig_str': 'sig_str',
        'head': 'head',
        'body': 'body',
        'leg': 'leg',
        'distance': 'distance',
        'clinch': 'clinch',
        'ground': 'ground',
        'ko': 'ko',
        'sub': 'sub',
        'decision': 'decision',
        'win': 'win',
        'rev': 'rev',
        'ctrl': 'ctrl',
        'td': 'td',
    }
    
    # Find the matching table prefix
    table_name = None
    for prefix, table in feature_to_table.items():
        if base_feature.startswith(prefix):
            table_name = table
            break
    
    if not table_name:
        # Try to infer from common patterns
        if 'weightclass' in base_feature:
            table_name = 'fight_mapping'
        else:
            # Default: try to find table by checking if column exists
            table_name = None
    
    return table_name, base_feature

def check_column_exists(conn, table_name, column_name):
    """Check if a column exists in a table"""
    query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = :table_name
        AND column_name = :column_name
    """)
    
    result = conn.execute(query, {"table_name": table_name, "column_name": column_name}).fetchone()
    return result is not None

def find_table_for_feature(conn, feature_name):
    """Find which table contains a given feature column"""
    # Use the feature name as-is (already converted by caller if needed)
    base_feature = feature_name
    
    # Map feature prefixes to likely table names
    prefix_to_table = {
        'age': 'age',
        'reach': 'reach',
        'days_since_last_fight': 'days_since_last_fight',
        'weightclass': 'fight_mapping',
        'sig_str': 'sig_str',
        'head': 'head',
        'body': 'body',
        'leg': 'leg',
        'distance': 'distance',
        'clinch': 'clinch',
        'ground': 'ground',
        'ko': 'ko',
        'sub': 'sub',
        'decision': 'decision',
        'win': 'win',
        'rev': 'rev',
        'ctrl': 'ctrl',
        'td': 'td',
    }
    
    # Try to find table based on prefix
    likely_table = None
    for prefix, table in prefix_to_table.items():
        if base_feature.startswith(prefix):
            likely_table = table
            break
    
    # Check the likely table first
    if likely_table and check_column_exists(conn, likely_table, base_feature):
        return likely_table
    
    # If not found, search all tables
    query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'features'
        AND table_type = 'BASE TABLE'
        AND table_name NOT IN ('fighter_mapping', 'event_mapping', 'fight_mapping', 
                               'fight_stats_core', 'fight_stats_fe', 'fight_stats_derived',
                               'first_time', 'minimum')
        ORDER BY table_name
    """)
    
    tables = [row[0] for row in conn.execute(query).fetchall()]
    
    # Check each table for the column
    for table in tables:
        if check_column_exists(conn, table, base_feature):
            return table
    
    return None

def get_fighter_stats_for_feature(conn, fighter_ids, feature_name, table_name, column_name):
    """
    Get the stat value for each fighter for a given feature.
    Uses the most recent fight value (or dec_avg if available).
    """
    # For weightclass_encoded, it's stored in fight_mapping
    if 'weightclass_encoded' in feature_name:
        query = text("""
            SELECT 
                fm.fighter1_id as fighter_id,
                fm.weightclass_encoded,
                em.event_date
            FROM features.fight_mapping fm
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE fm.fighter1_id = ANY(:fighter_ids)
              AND em.event_date >= '2025-01-01' 
              AND em.event_date < '2026-01-01'
            UNION ALL
            SELECT 
                fm.fighter2_id as fighter_id,
                fm.weightclass_encoded,
                em.event_date
            FROM features.fight_mapping fm
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            WHERE fm.fighter2_id = ANY(:fighter_ids)
              AND em.event_date >= '2025-01-01' 
              AND em.event_date < '2026-01-01'
            ORDER BY fighter_id, event_date DESC
        """)
    else:
        # For other features, get from the feature table
        # The column_name is already the base feature (without _diff)
        use_column = column_name
        
        # Use parameterized query to avoid SQL injection
        # Use quoted identifiers for table and column names
        query = text(f"""
            SELECT 
                f.fighter_id,
                f."{use_column}" as stat_value,
                em.event_date
            FROM features."{table_name}" f
            JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
            JOIN features.event_mapping em ON f.event_id = em.event_id
            WHERE f.fighter_id = ANY(:fighter_ids)
              AND em.event_date >= '2025-01-01' 
              AND em.event_date < '2026-01-01'
              AND f."{use_column}" IS NOT NULL
            ORDER BY f.fighter_id, em.event_date DESC
        """)
    
    try:
        df = pd.read_sql(query, conn, params={"fighter_ids": fighter_ids})
        
        # For each fighter, take the most recent value
        if len(df) > 0:
            df = df.groupby('fighter_id').first().reset_index()
        
        return df
    except Exception as e:
        print(f"  Error querying {table_name}.{column_name}: {e}")
        import traceback
        print(f"  Traceback: {traceback.format_exc()}")
        return pd.DataFrame()

def convert_feature_to_actual_column(feature_name):
    """
    Convert a feature name from vSeven_testing2 to the actual column name.
    For features with _dec_adjperf_dec_avg_diff, convert to _dec_adjperf_dec_avg.
    For other _diff features, convert to the base feature (remove _diff).
    """
    # Special case: weightclass_encoded stays as is
    if feature_name == 'weightclass_encoded':
        return feature_name
    
    # For features with _dec_adjperf_dec_avg_diff, convert to _dec_adjperf_dec_avg
    if '_dec_adjperf_dec_avg_diff' in feature_name:
        return feature_name.replace('_dec_adjperf_dec_avg_diff', '_dec_adjperf_dec_avg')
    
    # For other _diff features, remove _diff
    if feature_name.endswith('_diff'):
        return feature_name.replace('_diff', '')
    
    # For features that don't have _diff, return as is
    return feature_name

def analyze_feature(conn, fighter_ids_df, feature_name):
    """Analyze a single feature to find highest and lowest fighters"""
    print(f"\nAnalyzing feature: {feature_name}")
    
    # Convert feature name to actual column name
    actual_column = convert_feature_to_actual_column(feature_name)
    print(f"  Looking for column: {actual_column}")
    
    # Find the table and column for this feature
    table_name = find_table_for_feature(conn, actual_column)
    
    if not table_name:
        print(f"  WARNING: Could not find table for column: {actual_column}")
        return None
    
    # Check if column exists
    if not check_column_exists(conn, table_name, actual_column):
        print(f"  WARNING: Column {actual_column} not found in {table_name}")
        return None
    
    # Get stats for all fighters
    fighter_ids = fighter_ids_df['fighter_id'].tolist()
    stats_df = get_fighter_stats_for_feature(conn, fighter_ids, feature_name, table_name, actual_column)
    
    if stats_df.empty:
        print(f"  WARNING: No data found for {feature_name}")
        return None
    
    # Merge with fighter names
    stats_df = stats_df.merge(fighter_ids_df, on='fighter_id', how='left')
    
    # Determine stat column name
    if 'weightclass_encoded' in feature_name:
        stat_col = 'weightclass_encoded'
    else:
        stat_col = 'stat_value'
    
    # Filter out null values
    stats_df = stats_df[stats_df[stat_col].notna()]
    
    if stats_df.empty:
        print(f"  WARNING: No non-null values for {feature_name}")
        return None
    
    # Find highest and lowest
    highest_idx = stats_df[stat_col].idxmax()
    lowest_idx = stats_df[stat_col].idxmin()
    
    highest = stats_df.loc[highest_idx]
    lowest = stats_df.loc[lowest_idx]
    
    result = {
        'feature': feature_name,
        'actual_column': actual_column,
        'highest_fighter': highest['fighter_name'],
        'highest_value': highest[stat_col],
        'lowest_fighter': lowest['fighter_name'],
        'lowest_value': lowest[stat_col],
        'num_fighters': len(stats_df),
        'mean_value': stats_df[stat_col].mean(),
        'std_value': stats_df[stat_col].std()
    }
    
    print(f"  OK Highest: {highest['fighter_name']} ({highest[stat_col]:.4f})")
    print(f"  OK Lowest: {lowest['fighter_name']} ({lowest[stat_col]:.4f})")
    
    return result

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINDING EXTREME STATS FOR FIGHTERS WHO FOUGHT IN 2025")
    print("=" * 80)
    
    # Create database engine
    engine = create_db_engine()
    
    with get_db_connection(engine) as conn:
        # Get fighters who fought in 2025
        fighters_df = get_fighters_2025(conn)
        
        if fighters_df.empty:
            print("No fighters found who fought in 2025!")
            return
        
        # Analyze each feature
        results = []
        for feature in vSeven_testing2:
            result = analyze_feature(conn, fighters_df, feature)
            if result:
                results.append(result)
        
        # Create results dataframe
        if results:
            results_df = pd.DataFrame(results)
            
            # Display summary
            print("\n" + "=" * 80)
            print("SUMMARY RESULTS")
            print("=" * 80)
            print(f"\nAnalyzed {len(results)} features out of {len(vSeven_testing2)} total")
            
            # Save to CSV
            output_path = project_root / "data" / "extreme_stats_2025.csv"
            output_path.parent.mkdir(exist_ok=True)
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")
            
            # Display top 10 features by range (highest - lowest)
            results_df['range'] = results_df['highest_value'] - results_df['lowest_value']
            results_df['range_abs'] = results_df['range'].abs()
            
            print("\nTop 10 features with largest ranges:")
            print("-" * 80)
            top_ranges = results_df.nlargest(10, 'range_abs')[
                ['feature', 'highest_fighter', 'highest_value', 
                 'lowest_fighter', 'lowest_value', 'range']
            ]
            print(top_ranges.to_string(index=False))
        else:
            print("\nWARNING: No results found for any features!")

if __name__ == '__main__':
    main()

