#!/usr/bin/env python3
"""Review all stats in the blog for accuracy"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

engine = create_engine(DB_URL)
conn = engine.connect()

# Define all the stats to check
stats_to_check = [
    {
        'fighter': 'jailton almeida',
        'opponent': 'serghei spivac',
        'fight_date': '2025-01-18',
        'table': 'ctrl',
        'column': 'ctrl_per_min_dec_adjperf_dec_avg',
        'fight_column': 'ctrl_per_min_dec_adjperf',
        'opponent_column': 'ctrl_per_min_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'hyder amil',
        'opponent': 'william gomis',
        'fight_date': '2025-03-01',
        'table': 'sig_str',
        'column': 'sig_str_land_per_min_dec_adjperf_dec_avg',
        'fight_column': 'sig_str_land_per_min_dec_adjperf',
        'opponent_column': 'sig_str_land_per_min_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'kyoji horiguchi',
        'opponent': 'tagir ulanbekov',
        'fight_date': '2025-11-22',
        'table': 'ground',
        'column': 'ground_land_ratio_dec_adjperf_dec_avg',
        'fight_column': 'ground_land_ratio_dec_adjperf',
        'opponent_column': 'ground_land_ratio_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'toshiomi kazama',
        'opponent': 'elijah smith',
        'fight_date': '2025-08-09',
        'table': 'sig_str',
        'column': 'sig_str_def_dec_adjperf_dec_avg',
        'fight_column': 'sig_str_def_dec_adjperf',
        'opponent_column': 'sig_str_def_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'jamahal hill',
        'opponent': 'khalil rountree jr.',
        'fight_date': '2025-06-21',
        'table': 'ground',
        'column': 'ground_land_per_ctrl_dec_avg',
        'fight_column': 'ground_land_per_ctrl_dec_avg',  # This one doesn't have adjperf
        'opponent_column': 'ground_land_per_ctrl_dec_avg'
    },
    {
        'fighter': 'daniel santos',
        'opponent': 'joo sang yoo',
        'fight_date': '2025-10-04',
        'table': 'win',
        'column': 'win_dec_adjperf_dec_avg',
        'fight_column': 'win_dec_adjperf',
        'opponent_column': 'win_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'anshul jubli',
        'opponent': 'quillan salkilld',
        'fight_date': '2025-02-08',
        'table': 'ko',
        'column': 'ko_per_sig_str_land_dec_adjperf_dec_avg',
        'fight_column': 'ko_per_sig_str_land_dec_adjperf',
        'opponent_column': 'ko_per_sig_str_land_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'valter walker',
        'opponent': 'louie sutherland',
        'fight_date': '2025-10-25',
        'table': 'sub',
        'column': 'sub_att_ratio_dec_adjperf_dec_avg',
        'fight_column': 'sub_att_ratio_dec_adjperf',
        'opponent_column': 'sub_att_ratio_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'kyoji horiguchi',
        'opponent': 'tagir ulanbekov',
        'fight_date': '2025-11-22',
        'table': 'distance',
        'column': 'distance_land_ratio_dec_adjperf_dec_avg',
        'fight_column': 'distance_land_ratio_dec_adjperf',
        'opponent_column': 'distance_land_ratio_dec_adjperf_dec_avg'
    },
    {
        'fighter': 'merab dvalishvili',
        'opponent': 'sean o\'malley',
        'fight_date': '2025-06-07',
        'table': 'clinch',
        'column': 'clinch_land_ratio_dec_adjperf_dec_avg',
        'fight_column': 'clinch_land_ratio_dec_adjperf',
        'opponent_column': 'clinch_land_ratio_dec_adjperf_dec_avg'
    },
]

results = []

for stat_info in stats_to_check:
    fighter = stat_info['fighter']
    opponent = stat_info['opponent']
    fight_date = stat_info['fight_date']
    table = stat_info['table']
    column = stat_info['column']
    fight_column = stat_info['fight_column']
    opponent_column = stat_info['opponent_column']
    
    print(f"\n{'='*80}")
    print(f"Checking: {fighter} vs {opponent} ({fight_date})")
    print(f"{'='*80}")
    
    # Get fighter ID
    fighter_id_query = text("SELECT fighter_id FROM features.fighter_mapping WHERE LOWER(fighter_name) = LOWER(:name)")
    fighter_id = conn.execute(fighter_id_query, {"name": fighter}).fetchone()[0]
    opponent_id = conn.execute(fighter_id_query, {"name": opponent}).fetchone()[0]
    
    # Get fighter's career average at fight time and fight performance
    query = text(f"""
        SELECT 
            em.event_date,
            f.{column} as career_avg,
            f.{fight_column} as fight_value
        FROM features.{table} f
        JOIN features.event_mapping em ON f.event_id = em.event_id
        WHERE f.fighter_id = :fighter_id
          AND em.event_date = :fight_date
    """)
    
    fighter_data = conn.execute(query, {"fighter_id": fighter_id, "fight_date": fight_date}).fetchone()
    
    if fighter_data:
        career_avg = fighter_data[1]
        fight_value = fighter_data[2]
        
        print(f"Career avg ({column}): {career_avg}")
        print(f"Fight value ({fight_column}): {fight_value}")
        
        # Get opponent's pre-fight stat
        opp_query = text(f"""
            SELECT 
                em.event_date,
                f.{opponent_column} as stat_value
            FROM features.{table} f
            JOIN features.event_mapping em ON f.event_id = em.event_id
            WHERE f.fighter_id = :opponent_id
              AND em.event_date < :fight_date
            ORDER BY em.event_date DESC
            LIMIT 1
        """)
        
        opp_data = conn.execute(opp_query, {"opponent_id": opponent_id, "fight_date": fight_date}).fetchone()
        
        if opp_data:
            opp_stat = opp_data[1]
            opp_date = opp_data[0]
            print(f"Opponent pre-fight stat: {opp_stat} (from {opp_date})")
        else:
            opp_stat = None
            opp_date = None
            print(f"Opponent pre-fight stat: No data found")
        
        results.append({
            'fighter': fighter,
            'opponent': opponent,
            'fight_date': fight_date,
            'career_avg': career_avg,
            'fight_value': fight_value,
            'opponent_stat': opp_stat,
            'opponent_date': opp_date,
            'match': abs(career_avg - fight_value) < 0.01 if career_avg and fight_value else False
        })
    else:
        print(f"ERROR: Could not find fight data")
        results.append({
            'fighter': fighter,
            'opponent': opponent,
            'fight_date': fight_date,
            'career_avg': None,
            'fight_value': None,
            'opponent_stat': None,
            'opponent_date': None,
            'match': False
        })

print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

for r in results:
    match_str = "MATCH" if r['match'] else "DIFFERENT"
    print(f"\n{r['fighter']} vs {r['opponent']}:")
    print(f"  Career avg: {r['career_avg']}")
    print(f"  Fight value: {r['fight_value']}")
    print(f"  Status: {match_str}")
    if r['opponent_stat'] is not None:
        print(f"  Opponent pre-fight: {r['opponent_stat']} (from {r['opponent_date']})")
    else:
        print(f"  Opponent pre-fight: No data")

conn.close()

