#!/usr/bin/env python3
"""Investigate Hyder Amil vs William Gomis fight and stats"""

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

# Find the fight
query = text("""
    SELECT 
        fm.fight_id,
        em.event_date,
        fm.fighter1_id,
        fm.fighter2_id,
        fm.result,
        fm.method,
        LOWER(fm1.fighter_name) as fighter1_name,
        LOWER(fm2.fighter_name) as fighter2_name
    FROM features.fight_mapping fm
    JOIN features.event_mapping em ON fm.event_id = em.event_id
    JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
    JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
    WHERE em.event_date = '2025-03-01'
      AND (LOWER(fm1.fighter_name) = 'hyder amil' OR LOWER(fm2.fighter_name) = 'hyder amil')
      AND (LOWER(fm1.fighter_name) = 'william gomis' OR LOWER(fm2.fighter_name) = 'william gomis')
""")

fight = conn.execute(query).fetchone()

if not fight:
    print("Fight not found!")
    conn.close()
    exit(1)

fight_id = fight[0]
amil_id = fight[2] if 'hyder amil' in fight[6] else fight[3]
gomis_id = fight[2] if 'william gomis' in fight[6] else fight[3]

print("=" * 80)
print("FIGHT DETAILS")
print("=" * 80)
print(f"Fight ID: {fight_id}")
print(f"Date: {fight[1]}")
print(f"Fighter 1: {fight[6]} (ID: {fight[2]})")
print(f"Fighter 2: {fight[7]} (ID: {fight[3]})")
print(f"Result: {fight[4]}, Method: {fight[5]}")

# Get all of Amil's 2025 fights with this stat
print("\n" + "=" * 80)
print("HYDER AMIL'S 2025 FIGHTS - sig_str_land_per_min_dec_adjperf_dec_avg")
print("=" * 80)

query_amil = text("""
    SELECT 
        em.event_date,
        CASE 
            WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
            ELSE LOWER(fm1.fighter_name)
        END as opponent_name,
        f.sig_str_land_per_min_dec_adjperf_dec_avg as current_avg,
        f.sig_str_land_per_min_dec_adjperf as fight_value,
        f.sig_str_land_per_min_dec_avg as raw_rate,
        fm.result,
        fm.method
    FROM features.sig_str f
    JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    JOIN features.event_mapping em ON f.event_id = em.event_id
    JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
    JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
    WHERE f.fighter_id = :amil_id
      AND em.event_date >= '2025-01-01' 
      AND em.event_date < '2026-01-01'
    ORDER BY em.event_date
""")

df_amil = pd.read_sql(query_amil, conn, params={"amil_id": amil_id})
print("\nAmil's 2025 fights:")
print(df_amil.to_string())

# Get Gomis's pre-fight stats
print("\n" + "=" * 80)
print("WILLIAM GOMIS PRE-FIGHT STATS")
print("=" * 80)

query_gomis = text("""
    SELECT 
        em.event_date,
        CASE 
            WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
            ELSE LOWER(fm1.fighter_name)
        END as opponent_name,
        f.sig_str_land_per_min_dec_adjperf_dec_avg as stat_value,
        fm.result,
        fm.method
    FROM features.sig_str f
    JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    JOIN features.event_mapping em ON f.event_id = em.event_id
    JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
    JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
    WHERE f.fighter_id = :gomis_id
      AND em.event_date < '2025-03-01'
    ORDER BY em.event_date DESC
    LIMIT 5
""")

df_gomis = pd.read_sql(query_gomis, conn, params={"gomis_id": gomis_id})
print("\nGomis's fights before March 1, 2025:")
print(df_gomis.to_string())

# Get raw strike data for the fight
print("\n" + "=" * 80)
print("RAW STRIKE DATA FOR THE FIGHT")
print("=" * 80)

query_raw = text("""
    SELECT 
        fsc.fighter_id,
        LOWER(fm.fighter_name) as fighter_name,
        fsc.sig_str_land_rd1,
        fsc.sig_str_land_rd2,
        fsc.sig_str_land_rd3,
        fsc.sig_str_land_rd4,
        fsc.sig_str_land_rd5,
        (COALESCE(fsc.sig_str_land_rd1, 0) + COALESCE(fsc.sig_str_land_rd2, 0) + 
         COALESCE(fsc.sig_str_land_rd3, 0) + COALESCE(fsc.sig_str_land_rd4, 0) + 
         COALESCE(fsc.sig_str_land_rd5, 0)) as total_sig_str_landed,
        em.time_sec
    FROM features.fight_stats_core fsc
    JOIN features.fighter_mapping fm ON fsc.fighter_id = fm.fighter_id
    JOIN features.event_mapping em ON fsc.event_id = em.event_id
    WHERE fsc.fight_id = :fight_id
    ORDER BY fsc.fighter_id
""")

df_raw = pd.read_sql(query_raw, conn, params={"fight_id": fight_id})
print("\nRaw strike data:")
print(df_raw.to_string())

if not df_raw.empty:
    amil_raw = df_raw[df_raw['fighter_id'] == amil_id]
    if not amil_raw.empty:
        total_strikes = amil_raw.iloc[0]['total_sig_str_landed']
        fight_time = amil_raw.iloc[0]['time_sec']
        strikes_per_min = (total_strikes / (fight_time / 60)) if fight_time > 0 else 0
        print(f"\nAmil's raw strikes per minute: {strikes_per_min:.2f}")

conn.close()

