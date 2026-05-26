#!/usr/bin/env python3
"""
Get UFC averages for fighters with 3+ fights active in 2025,
and get opponent pre-fight stats for key fights.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

def create_db_engine(db_url=DB_URL):
    if not database_exists(db_url):
        print(f"ERROR: Database does not exist at {db_url}")
        sys.exit(1)
    return create_engine(db_url)

def get_qualified_fighter_ids(conn):
    """Get fighter IDs for fighters with 3+ UFC fights active in 2025"""
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
        SELECT fighter_id
        FROM fighters_2025
        WHERE fighter_id IN (SELECT fighter_id FROM fighter_fight_counts)
    """)
    df = pd.read_sql(query, conn)
    return df['fighter_id'].tolist()

def get_ufc_averages(conn, fighter_ids):
    """Get averages for qualified fighters"""
    stats_to_average = {
        'ctrl': 'ctrl_per_min_dec_adjperf_dec_avg',
        'td': 'td_land_per_min_dec_adjperf_dec_avg',
        'sig_str': 'sig_str_acc_dec_adjperf_dec_avg',
        'sig_str': 'sig_str_land_per_min_dec_adjperf_dec_avg',
        'head': 'head_land_ratio_dec_adjperf_dec_avg',
        'ko': 'ko_per_sig_str_land_dec_adjperf_dec_avg',
        'sub': 'sub_acc_dec_adjperf_dec_avg',
        'sub': 'sub_att_ratio_dec_adjperf_dec_avg',
        'ground': 'ground_land_per_ctrl_dec_avg',
        'ground': 'ground_land_ratio_dec_adjperf_dec_avg',
        'clinch': 'clinch_land_ratio_dec_adjperf_dec_avg',
        'clinch': 'clinch_acc_dec_adjperf_dec_avg',
        'distance': 'distance_land_ratio_dec_adjperf_dec_avg',
        'distance': 'distance_acc_dec_adjperf_dec_avg',
        'leg': 'leg_land_per_min_dec_adjperf_dec_avg',
        'ko': 'ko_ratio_dec_adjperf_dec_avg',
        'win': 'win_dec_adjperf_dec_avg',
    }
    
    averages = {}
    
    for table, column in [
        ('ctrl', 'ctrl_per_min_dec_adjperf_dec_avg'),
        ('td', 'td_land_per_min_dec_adjperf_dec_avg'),
        ('sig_str', 'sig_str_acc_dec_adjperf_dec_avg'),
        ('sig_str', 'sig_str_land_per_min_dec_adjperf_dec_avg'),
        ('head', 'head_land_ratio_dec_adjperf_dec_avg'),
        ('ko', 'ko_per_sig_str_land_dec_adjperf_dec_avg'),
        ('sub', 'sub_acc_dec_adjperf_dec_avg'),
        ('sub', 'sub_att_ratio_dec_adjperf_dec_avg'),
        ('ground', 'ground_land_per_ctrl_dec_avg'),
        ('ground', 'ground_land_ratio_dec_adjperf_dec_avg'),
        ('clinch', 'clinch_land_ratio_dec_adjperf_dec_avg'),
        ('clinch', 'clinch_acc_dec_adjperf_dec_avg'),
        ('distance', 'distance_land_ratio_dec_adjperf_dec_avg'),
        ('distance', 'distance_acc_dec_adjperf_dec_avg'),
        ('leg', 'leg_land_per_min_dec_adjperf_dec_avg'),
        ('ko', 'ko_ratio_dec_adjperf_dec_avg'),
        ('win', 'win_dec_adjperf_dec_avg'),
    ]:
        try:
            query = text(f"""
                SELECT AVG(f.{column}) as avg_value
                FROM features.{table} f
                JOIN features.event_mapping em ON f.event_id = em.event_id
                WHERE f.fighter_id = ANY(:fighter_ids)
                  AND em.event_date >= '2025-01-01' 
                  AND em.event_date < '2026-01-01'
                  AND f.{column} IS NOT NULL
            """)
            result = conn.execute(query, {"fighter_ids": fighter_ids}).fetchone()
            if result and result[0] is not None:
                key = f"{table}.{column}"
                averages[key] = result[0]
        except Exception as e:
            print(f"Error getting average for {table}.{column}: {e}")
    
    return averages

def get_opponent_prefight_stat(conn, opponent_name, fight_date, table_name, column_name):
    """Get opponent's stat value before the fight date"""
    query = text(f"""
        SELECT 
            f.{column_name} as stat_value,
            em.event_date,
            CASE 
                WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
                ELSE LOWER(fm1.fighter_name)
            END as opponent_name
        FROM features.{table_name} f
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
        JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
        WHERE LOWER(fm1.fighter_name) = LOWER(:opponent_name)
           OR LOWER(fm2.fighter_name) = LOWER(:opponent_name)
          AND em.event_date < :fight_date
          AND f.{column_name} IS NOT NULL
        ORDER BY em.event_date DESC
        LIMIT 1
    """)
    
    # Try to get the opponent's ID first
    opponent_id_query = text("""
        SELECT fighter_id FROM features.fighter_mapping 
        WHERE LOWER(fighter_name) = LOWER(:opponent_name)
    """)
    opponent_result = conn.execute(opponent_id_query, {"opponent_name": opponent_name}).fetchone()
    
    if not opponent_result:
        return None
    
    opponent_id = opponent_result[0]
    
    # Get opponent's most recent stat before the fight
    query = text(f"""
        SELECT 
            f.{column_name} as stat_value,
            em.event_date
        FROM features.{table_name} f
        JOIN features.event_mapping em ON f.event_id = em.event_id
        WHERE f.fighter_id = :opponent_id
          AND em.event_date < :fight_date
          AND f.{column_name} IS NOT NULL
        ORDER BY em.event_date DESC
        LIMIT 1
    """)
    
    result = conn.execute(query, {"opponent_id": opponent_id, "fight_date": fight_date}).fetchone()
    if result:
        return {
            'value': result[0],
            'date': result[1]
        }
    return None

def main():
    engine = create_db_engine()
    conn = engine.connect()
    
    print("Getting qualified fighter IDs...")
    fighter_ids = get_qualified_fighter_ids(conn)
    print(f"Found {len(fighter_ids)} qualified fighters")
    
    print("\nCalculating UFC averages...")
    averages = get_ufc_averages(conn, fighter_ids)
    
    print("\nUFC Averages (fighters with 3+ fights active in 2025):")
    for key, value in sorted(averages.items()):
        print(f"  {key}: {value:.4f}")
    
    # Get opponent stats for key fights
    key_fights = [
        ('jailton almeida', '2025-01-18', 'serghei spivac', 'ctrl', 'ctrl_per_min_dec_adjperf_dec_avg'),
        ('giga chikadze', '2025-04-26', 'david onama', 'td', 'td_land_per_ctrl_dec_adjperf_dec_avg'),
        ('kyoji horiguchi', '2025-11-22', 'tagir ulanbekov', 'ground', 'ground_land_ratio_dec_adjperf_dec_avg'),
        ('toshiomi kazama', '2025-08-09', 'elijah smith', 'sig_str', 'sig_str_def_dec_adjperf_dec_avg'),
        ('jamahal hill', '2025-06-21', 'khalil rountree jr.', 'ground', 'ground_land_per_ctrl_dec_avg'),
        ('daniel santos', '2025-10-04', 'joo sang yoo', 'win', 'win_dec_adjperf_dec_avg'),
        ('anshul jubli', '2025-02-08', 'quillan salkilld', 'ko', 'ko_per_sig_str_land_dec_adjperf_dec_avg'),
        ('valter walker', '2025-10-25', 'louie sutherland', 'sub', 'sub_att_ratio_dec_adjperf_dec_avg'),
        ('kyoji horiguchi', '2025-11-22', 'tagir ulanbekov', 'distance', 'distance_land_ratio_dec_adjperf_dec_avg'),
        ('merab dvalishvili', '2025-06-07', 'sean o\'malley', 'clinch', 'clinch_land_ratio_dec_adjperf_dec_avg'),
    ]
    
    print("\n\nOpponent Pre-Fight Stats:")
    print("=" * 80)
    
    opponent_stats = {}
    for fighter, fight_date, opponent, table, column in key_fights:
        opp_stat = get_opponent_prefight_stat(conn, opponent, fight_date, table, column)
        key = f"{fighter}_{opponent}_{column}"
        opponent_stats[key] = {
            'fighter': fighter,
            'opponent': opponent,
            'fight_date': fight_date,
            'stat': column,
            'opponent_value': opp_stat['value'] if opp_stat else None,
            'opponent_date': opp_stat['date'] if opp_stat else None
        }
        
        if opp_stat:
            print(f"\n{opponent} (before {fight_date}):")
            print(f"  {column}: {opp_stat['value']:.4f} (from fight on {opp_stat['date']})")
        else:
            print(f"\n{opponent}: No pre-fight data found")
    
    # Save results
    import json
    output = {
        'averages': averages,
        'opponent_stats': opponent_stats
    }
    
    output_path = project_root / "data" / "ufc_averages_and_opponent_stats.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n\nResults saved to: {output_path}")
    
    conn.close()

if __name__ == '__main__':
    main()

