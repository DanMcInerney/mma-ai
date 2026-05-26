#!/usr/bin/env python3
"""
Find the most interesting outlier stats for UFC fans by analyzing extreme values
across multiple categories.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from contextlib import contextmanager
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

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

@contextmanager
def get_db_connection(engine):
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def get_fighters_2025(conn):
    """Get all unique fighters who fought at least once in 2025 and have at least 3 UFC fights"""
    query = text("""
        WITH fighter_fight_counts AS (
            SELECT 
                fighter_id,
                COUNT(DISTINCT fight_id) as total_fights
            FROM (
                SELECT fighter1_id as fighter_id, fight_id FROM features.fight_mapping
                UNION ALL
                SELECT fighter2_id as fighter_id, fight_id FROM features.fight_mapping
            ) all_fighters
            GROUP BY fighter_id
            HAVING COUNT(DISTINCT fight_id) >= 3
        ),
        fighters_2025 AS (
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
        )
        SELECT DISTINCT f.fighter_id, LOWER(f.fighter_name) as fighter_name
        FROM features.fighter_mapping f
        WHERE f.fighter_id IN (SELECT fighter_id FROM fighters_2025)
        AND f.fighter_id IN (SELECT fighter_id FROM fighter_fight_counts)
        ORDER BY fighter_name
    """)
    df = pd.read_sql(query, conn)
    return df

def find_extreme_stats_by_category(conn, fighter_ids):
    """Find extreme stats across different interesting categories"""
    
    interesting_stats = [
        # Striking power and efficiency
        ('sig_str', 'sig_str_acc_dec_adjperf_dec_avg', 'Striking Accuracy'),
        ('sig_str', 'sig_str_land_per_min_dec_adjperf_dec_avg', 'Strikes Landed Per Minute'),
        ('head', 'head_land_ratio_dec_adjperf_dec_avg', 'Head Strike Dominance'),
        ('ko', 'ko_per_sig_str_land_dec_adjperf_dec_avg', 'Knockout Efficiency'),
        
        # Defense
        ('sig_str', 'sig_str_def_dec_adjperf_dec_avg', 'Striking Defense'),
        ('head', 'head_def_dec_adjperf_dec_avg', 'Head Defense'),
        
        # Grappling
        ('td', 'td_acc_dec_adjperf_dec_avg', 'Takedown Accuracy'),
        ('td', 'td_land_per_min_dec_adjperf_dec_avg', 'Takedown Rate'),
        ('sub', 'sub_acc_dec_adjperf_dec_avg', 'Submission Accuracy'),
        ('sub', 'sub_att_ratio_dec_adjperf_dec_avg', 'Submission Attempt Rate'),
        
        # Ground game
        ('ground', 'ground_land_per_ctrl_dec_avg', 'Ground Strikes Per Control'),
        ('ground', 'ground_land_ratio_dec_adjperf_dec_avg', 'Ground Strike Dominance'),
        
        # Clinch
        ('clinch', 'clinch_land_ratio_dec_adjperf_dec_avg', 'Clinch Dominance'),
        ('clinch', 'clinch_acc_dec_adjperf_dec_avg', 'Clinch Accuracy'),
        
        # Distance striking
        ('distance', 'distance_land_ratio_dec_adjperf_dec_avg', 'Distance Strike Dominance'),
        ('distance', 'distance_acc_dec_adjperf_dec_avg', 'Distance Accuracy'),
        
        # Body/leg strikes
        ('body', 'body_acc_dec_adjperf_dec_avg', 'Body Strike Accuracy'),
        ('leg', 'leg_land_per_min_dec_adjperf_dec_avg', 'Leg Kick Rate'),
        
        # Control and dominance
        ('ctrl', 'ctrl_ratio_dec_adjperf_dec_avg', 'Control Dominance'),
        ('ctrl', 'ctrl_per_min_dec_adjperf_dec_avg', 'Control Time Rate'),
        
        # Finishing ability
        ('ko', 'ko_ratio_dec_adjperf_dec_avg', 'Knockout Rate'),
        ('win', 'win_dec_adjperf_dec_avg', 'Win Rate'),
    ]
    
    results = []
    
    for table_name, column_name, description in interesting_stats:
        try:
            # Check if column exists
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = 'features' 
                AND table_name = :table_name
                AND column_name = :column_name
            """)
            exists = conn.execute(check_query, {"table_name": table_name, "column_name": column_name}).fetchone()
            
            if not exists:
                continue
            
            # Get extreme values
            query = text(f"""
                SELECT 
                    f.fighter_id,
                    LOWER(fm.fighter_name) as fighter_name,
                    f.{column_name} as stat_value
                FROM features.{table_name} f
                JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
                JOIN features.event_mapping em ON f.event_id = em.event_id
                WHERE f.fighter_id = ANY(:fighter_ids)
                  AND em.event_date >= '2025-01-01' 
                  AND em.event_date < '2026-01-01'
                  AND f.{column_name} IS NOT NULL
                ORDER BY f.{column_name} DESC
                LIMIT 5
            """)
            
            highest = pd.read_sql(query, conn, params={"fighter_ids": fighter_ids})
            
            query = text(f"""
                SELECT 
                    f.fighter_id,
                    LOWER(fm.fighter_name) as fighter_name,
                    f.{column_name} as stat_value
                FROM features.{table_name} f
                JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
                JOIN features.event_mapping em ON f.event_id = em.event_id
                WHERE f.fighter_id = ANY(:fighter_ids)
                  AND em.event_date >= '2025-01-01' 
                  AND em.event_date < '2026-01-01'
                  AND f.{column_name} IS NOT NULL
                ORDER BY f.{column_name} ASC
                LIMIT 5
            """)
            
            lowest = pd.read_sql(query, conn, params={"fighter_ids": fighter_ids})
            
            if not highest.empty and not lowest.empty:
                max_val = highest.iloc[0]['stat_value']
                min_val = lowest.iloc[0]['stat_value']
                max_fighter = highest.iloc[0]['fighter_name']
                min_fighter = lowest.iloc[0]['fighter_name']
                
                # Calculate spread
                spread = abs(max_val - min_val)
                
                results.append({
                    'table': table_name,
                    'column': column_name,
                    'description': description,
                    'max_fighter': max_fighter,
                    'max_value': max_val,
                    'min_fighter': min_fighter,
                    'min_value': min_val,
                    'spread': spread,
                    'max_abs': max(abs(max_val), abs(min_val))
                })
                
        except Exception as e:
            print(f"Error processing {table_name}.{column_name}: {e}")
            continue
    
    return pd.DataFrame(results)

def analyze_outlier_quality(df):
    """Analyze which outliers are most interesting"""
    
    # Sort by absolute value (most extreme)
    df_sorted = df.sort_values('max_abs', ascending=False)
    
    print("\n" + "=" * 80)
    print("TOP 20 MOST EXTREME STATS (by absolute value)")
    print("=" * 80)
    
    for idx, row in df_sorted.head(20).iterrows():
        print(f"\n{row['description']}")
        print(f"  Highest: {row['max_fighter']} ({row['max_value']:.2f})")
        print(f"  Lowest: {row['min_fighter']} ({row['min_value']:.2f})")
        print(f"  Spread: {row['spread']:.2f}")
    
    return df_sorted

def get_fight_details_for_stat(conn, fighter_id, table_name, column_name):
    """Get the fight that contributed most to an extreme stat"""
    query = text(f"""
        SELECT 
            em.event_date,
            CASE 
                WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
                ELSE LOWER(fm1.fighter_name)
            END as opponent_name,
            f.{column_name} as stat_value,
            fm.result,
            fm.method
        FROM features.{table_name} f
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
        JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
        WHERE f.fighter_id = :fighter_id
          AND em.event_date >= '2025-01-01' 
          AND em.event_date < '2026-01-01'
          AND f.{column_name} IS NOT NULL
        ORDER BY ABS(f.{column_name}) DESC
        LIMIT 1
    """)
    
    result = conn.execute(query, {"fighter_id": fighter_id}).fetchone()
    if result:
        return {
            'date': result[0],
            'opponent': result[1],
            'value': result[2],
            'result': result[3],
            'method': result[4]
        }
    return None

def main():
    engine = create_db_engine()
    
    with get_db_connection(engine) as conn:
        print("=" * 80)
        print("FINDING MOST INTERESTING OUTLIER STATS")
        print("=" * 80)
        
        # Get 2025 fighters
        fighters_df = get_fighters_2025(conn)
        fighter_ids = fighters_df['fighter_id'].tolist()
        print(f"\nAnalyzing {len(fighter_ids)} fighters with 3+ UFC fights active in 2025")
        
        # Find extreme stats
        extreme_stats_df = find_extreme_stats_by_category(conn, fighter_ids)
        
        print(f"\nFound {len(extreme_stats_df)} stat categories")
        
        # Analyze outliers
        sorted_df = analyze_outlier_quality(extreme_stats_df)
        
        # Get fight details for top outliers
        print("\n" + "=" * 80)
        print("TOP 10 MOST INTERESTING OUTLIERS WITH FIGHT DETAILS")
        print("=" * 80)
        
        top_outliers = []
        for idx, row in sorted_df.head(10).iterrows():
            fighter_id_val = int(fighters_df[fighters_df['fighter_name'] == row['max_fighter']]['fighter_id'].iloc[0])
            fight_details = get_fight_details_for_stat(
                conn, 
                fighter_id_val,
                row['table'],
                row['column']
            )
            
            outlier_info = {
                'description': row['description'],
                'fighter': row['max_fighter'],
                'value': row['max_value'],
                'spread': row['spread'],
                'fight': fight_details
            }
            top_outliers.append(outlier_info)
            
            print(f"\n{row['description']}")
            print(f"  Fighter: {row['max_fighter']}")
            print(f"  Value: {row['max_value']:.2f}")
            if fight_details:
                win_loss = "Win" if fight_details['result'] == 1 else "Loss" if fight_details['result'] == 0 else "Draw/NC"
                print(f"  Key Fight: vs {fight_details['opponent']} on {fight_details['date']}")
                print(f"  Fight Value: {fight_details['value']:.2f}")
                print(f"  Result: {win_loss} via {fight_details['method']}")
        
        # Save results
        output_df = pd.DataFrame(top_outliers)
        output_path = project_root / "data" / "most_interesting_outliers.csv"
        output_path.parent.mkdir(exist_ok=True)
        output_df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to: {output_path}")

if __name__ == '__main__':
    main()

