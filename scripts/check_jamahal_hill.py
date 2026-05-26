#!/usr/bin/env python3
"""Check Jamahal Hill's ground strike stats"""

import sys
from sqlalchemy import create_engine, text
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

engine = create_engine(DB_URL)
conn = engine.connect()

# Get Jamahal Hill's fight vs Khalil Rountree
query = text("""
    SELECT 
        em.event_date,
        f.ground_land_per_ctrl_dec_avg,
        f.ground_land_per_ctrl,
        f.ground_land,
        f.ctrl
    FROM features.ground f
    JOIN features.event_mapping em ON f.event_id = em.event_id
    JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
    WHERE LOWER(fm.fighter_name) = 'jamahal hill'
      AND em.event_date = '2025-06-21'
""")

result = conn.execute(query).fetchone()
if result:
    print(f"Date: {result[0]}")
    print(f"ground_land_per_ctrl_dec_avg: {result[1]}")
    print(f"ground_land_per_ctrl: {result[2]}")
    print(f"ground_land: {result[3]}")
    print(f"ctrl: {result[4]}")

conn.close()

