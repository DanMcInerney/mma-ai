#!/usr/bin/env python3
"""Get raw unadjusted stats for all blog fights"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
from libs.paths import no_winsor_database_url

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

engine = create_engine(DB_URL)
conn = engine.connect()

# Define all stats with their raw equivalents
stats = [
    {
        'fighter': 'jailton almeida',
        'opponent': 'serghei spivac',
        'fight_date': '2025-01-18',
        'table': 'ctrl',
        'adjperf_col': 'ctrl_per_min_dec_adjperf_dec_avg',
        'raw_col': 'ctrl_per_min_dec_avg'
    },
    {
        'fighter': 'hyder amil',
        'opponent': 'william gomis',
        'fight_date': '2025-03-01',
        'table': 'sig_str',
        'adjperf_col': 'sig_str_land_per_min_dec_adjperf_dec_avg',
        'raw_col': 'sig_str_land_per_min_dec_avg'
    },
    {
        'fighter': 'kyoji horiguchi',
        'opponent': 'tagir ulanbekov',
        'fight_date': '2025-11-22',
        'table': 'ground',
        'adjperf_col': 'ground_land_ratio_dec_adjperf_dec_avg',
        'raw_col': 'ground_land_ratio_dec_avg'
    },
    {
        'fighter': 'toshiomi kazama',
        'opponent': 'elijah smith',
        'fight_date': '2025-08-09',
        'table': 'sig_str',
        'adjperf_col': 'sig_str_def_dec_adjperf_dec_avg',
        'raw_col': 'sig_str_def_dec_avg'
    },
    {
        'fighter': 'jamahal hill',
        'opponent': 'khalil rountree jr.',
        'fight_date': '2025-06-21',
        'table': 'ground',
        'adjperf_col': 'ground_land_per_ctrl_dec_avg',
        'raw_col': 'ground_land_per_ctrl_dec_avg'  # This one doesn't have adjperf
    },
    {
        'fighter': 'daniel santos',
        'opponent': 'joo sang yoo',
        'fight_date': '2025-10-04',
        'table': 'win',
        'adjperf_col': 'win_dec_adjperf_dec_avg',
        'raw_col': 'win_dec_avg'
    },
    {
        'fighter': 'anshul jubli',
        'opponent': 'quillan salkilld',
        'fight_date': '2025-02-08',
        'table': 'ko',
        'adjperf_col': 'ko_per_sig_str_land_dec_adjperf_dec_avg',
        'raw_col': 'ko_per_sig_str_land_dec_avg'
    },
    {
        'fighter': 'valter walker',
        'opponent': 'louie sutherland',
        'fight_date': '2025-10-25',
        'table': 'sub',
        'adjperf_col': 'sub_att_ratio_dec_adjperf_dec_avg',
        'raw_col': 'sub_att_ratio_dec_avg'
    },
    {
        'fighter': 'kyoji horiguchi',
        'opponent': 'tagir ulanbekov',
        'fight_date': '2025-11-22',
        'table': 'distance',
        'adjperf_col': 'distance_land_ratio_dec_adjperf_dec_avg',
        'raw_col': 'distance_land_ratio_dec_avg'
    },
    {
        'fighter': 'merab dvalishvili',
        'opponent': 'sean o\'malley',
        'fight_date': '2025-06-07',
        'table': 'clinch',
        'adjperf_col': 'clinch_land_ratio_dec_adjperf_dec_avg',
        'raw_col': 'clinch_land_ratio_dec_avg'
    },
]

results = []

for stat_info in stats:
    fighter = stat_info['fighter']
    fight_date = stat_info['fight_date']
    table = stat_info['table']
    adjperf_col = stat_info['adjperf_col']
    raw_col = stat_info['raw_col']
    
    # Get fighter ID
    fighter_id_query = text("SELECT fighter_id FROM features.fighter_mapping WHERE LOWER(fighter_name) = LOWER(:name)")
    fighter_id = conn.execute(fighter_id_query, {"name": fighter}).fetchone()[0]
    
    # Get both adjperf and raw stats
    query = text(f"""
        SELECT 
            f.{adjperf_col} as adjperf_value,
            f.{raw_col} as raw_value
        FROM features.{table} f
        JOIN features.event_mapping em ON f.event_id = em.event_id
        WHERE f.fighter_id = :fighter_id
          AND em.event_date = :fight_date
    """)
    
    result = conn.execute(query, {"fighter_id": fighter_id, "fight_date": fight_date}).fetchone()
    
    if result:
        results.append({
            'fighter': fighter,
            'opponent': stat_info['opponent'],
            'adjperf_col': adjperf_col,
            'raw_col': raw_col,
            'adjperf_value': result[0],
            'raw_value': result[1]
        })
        print(f"{fighter}: adjperf={result[0]}, raw={result[1]}")

# Save results
import json
output_path = project_root / "data" / "raw_stats_for_blog.json"
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")

conn.close()

