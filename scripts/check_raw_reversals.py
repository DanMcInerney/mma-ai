#!/usr/bin/env python3
"""Check raw reversal counts from fight_stats_core"""

import sys
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

engine = create_engine(DB_URL)
conn = engine.connect()

# Find Usman vs Buckley fight
query = text("""
    SELECT 
        fm.fight_id,
        fsc.fighter_id,
        LOWER(fm2.fighter_name) as fighter_name,
        fsc.rev_rd1,
        fsc.rev_rd2,
        fsc.rev_rd3,
        fsc.rev_rd4,
        fsc.rev_rd5,
        (COALESCE(fsc.rev_rd1, 0) + COALESCE(fsc.rev_rd2, 0) + 
         COALESCE(fsc.rev_rd3, 0) + COALESCE(fsc.rev_rd4, 0) + 
         COALESCE(fsc.rev_rd5, 0)) as total_rev
    FROM features.fight_mapping fm
    JOIN features.event_mapping em ON fm.event_id = em.event_id
    JOIN features.fight_stats_core fsc ON fm.fight_id = fsc.fight_id
    JOIN features.fighter_mapping fm2 ON fsc.fighter_id = fm2.fighter_id
    WHERE em.event_date = '2025-06-14'
      AND (fm.fighter1_id = 4038 OR fm.fighter2_id = 4038)
      AND (fm.fighter1_id = 1496 OR fm.fighter2_id = 1496)
    ORDER BY fsc.fighter_id
""")

results = conn.execute(query).fetchall()
print("Raw reversal counts from fight_stats_core:")
for row in results:
    print(f"\nFighter: {row[2]}")
    print(f"  Round 1: {row[3]}")
    print(f"  Round 2: {row[4]}")
    print(f"  Round 3: {row[5]}")
    print(f"  Round 4: {row[6]}")
    print(f"  Round 5: {row[7]}")
    print(f"  TOTAL: {row[8]}")

conn.close()

