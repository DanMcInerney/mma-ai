#!/usr/bin/env python3
"""Investigate Giga Chikadze vs David Onama fight data"""

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
    WHERE em.event_date = '2025-04-26'
      AND (LOWER(fm1.fighter_name) = 'giga chikadze' OR LOWER(fm2.fighter_name) = 'giga chikadze')
      AND (LOWER(fm1.fighter_name) = 'david onama' OR LOWER(fm2.fighter_name) = 'david onama')
""")

fight = conn.execute(query).fetchone()

if not fight:
    print("Fight not found!")
    conn.close()
    exit(1)

fight_id = fight[0]
giga_id = fight[2] if 'giga chikadze' in fight[6] else fight[3]
onama_id = fight[2] if 'david onama' in fight[6] else fight[3]

print("=" * 80)
print("FIGHT DETAILS")
print("=" * 80)
print(f"Fight ID: {fight_id}")
print(f"Date: {fight[1]}")
print(f"Fighter 1: {fight[6]} (ID: {fight[2]})")
print(f"Fighter 2: {fight[7]} (ID: {fight[3]})")
print(f"Result: {fight[4]}, Method: {fight[5]}")

# Get raw stats from fight_stats_core
print("\n" + "=" * 80)
print("RAW STATS FROM fight_stats_core")
print("=" * 80)

query_raw = text("""
    SELECT 
        fsc.fighter_id,
        LOWER(fm.fighter_name) as fighter_name,
        fsc.td_land_rd1,
        fsc.td_land_rd2,
        fsc.td_land_rd3,
        fsc.td_land_rd4,
        fsc.td_land_rd5,
        (COALESCE(fsc.td_land_rd1, 0) + COALESCE(fsc.td_land_rd2, 0) + 
         COALESCE(fsc.td_land_rd3, 0) + COALESCE(fsc.td_land_rd4, 0) + 
         COALESCE(fsc.td_land_rd5, 0)) as total_td_land,
        fsc.ctrl_rd1,
        fsc.ctrl_rd2,
        fsc.ctrl_rd3,
        fsc.ctrl_rd4,
        fsc.ctrl_rd5
    FROM features.fight_stats_core fsc
    JOIN features.fighter_mapping fm ON fsc.fighter_id = fm.fighter_id
    WHERE fsc.fight_id = :fight_id
    ORDER BY fsc.fighter_id
""")

df_raw = pd.read_sql(query_raw, conn, params={"fight_id": fight_id})
print("\nRaw takedown and control data:")
print(df_raw.to_string())

# Get stats from td table
print("\n" + "=" * 80)
print("STATS FROM td TABLE")
print("=" * 80)

query_td = text("""
    SELECT 
        f.fighter_id,
        LOWER(fm.fighter_name) as fighter_name,
        f.td_land,
        f.td_land_per_ctrl,
        f.td_land_per_ctrl_dec_adjperf,
        f.td_land_per_ctrl_dec_adjperf_dec_avg,
        f.td_land_per_ctrl_dec_avg,
        f.td_land_per_ctrl_avg
    FROM features.td f
    JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
    WHERE f.fight_id = :fight_id
    ORDER BY f.fighter_id
""")

df_td = pd.read_sql(query_td, conn, params={"fight_id": fight_id})
print("\nTakedown stats:")
print(df_td.to_string())

# Get stats from ctrl table
print("\n" + "=" * 80)
print("STATS FROM ctrl TABLE")
print("=" * 80)

query_ctrl = text("""
    SELECT 
        f.fighter_id,
        LOWER(fm.fighter_name) as fighter_name,
        f.ctrl,
        f.ctrl_total,
        f.ctrl_per_min
    FROM features.ctrl f
    JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
    WHERE f.fight_id = :fight_id
    ORDER BY f.fighter_id
""")

df_ctrl = pd.read_sql(query_ctrl, conn, params={"fight_id": fight_id})
print("\nControl time stats:")
print(df_ctrl.to_string())

# Check if there's any control time at all
print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

giga_row_td = df_td[df_td['fighter_id'] == giga_id]
onama_row_td = df_td[df_td['fighter_id'] == onama_id]

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

if not giga_row_td.empty:
    giga_td_land_raw = df_raw[df_raw['fighter_id'] == giga_id].iloc[0]['total_td_land']
    giga_td_land_smoothed = giga_row_td.iloc[0]['td_land']
    giga_td_per_ctrl = giga_row_td.iloc[0]['td_land_per_ctrl']
    giga_ctrl_smoothed = df_ctrl[df_ctrl['fighter_id'] == giga_id].iloc[0]['ctrl']
    
    print(f"\nGiga Chikadze:")
    print(f"  RAW: {giga_td_land_raw} takedowns, 0 control time")
    print(f"  SMOOTHED: {giga_td_land_smoothed:.4f} takedowns, {giga_ctrl_smoothed:.4f} seconds control")
    print(f"  td_land_per_ctrl (smoothed): {giga_td_per_ctrl:.4f}")
    print(f"  td_land_per_ctrl_dec_adjperf_dec_avg: {giga_row_td.iloc[0]['td_land_per_ctrl_dec_adjperf_dec_avg']:.4f}")
    
    print(f"\n  [CRITICAL] Issue:")
    print(f"  Giga had 0 takedowns and 0 control time in this fight!")
    print(f"  The high adjperf score (115.17) is entirely from Bayesian smoothing")
    print(f"  creating non-zero values from zeros. This stat is misleading!")
    
    print(f"\n  Calculation:")
    print(f"  Smoothed td_land: {giga_td_land_smoothed:.4f} / Smoothed ctrl: {giga_ctrl_smoothed:.4f}")
    print(f"  = {giga_td_land_smoothed / giga_ctrl_smoothed:.4f} (when ctrl is tiny, ratio becomes huge)")
    print(f"  Then adjusted for opponent quality -> 115.17")

conn.close()

