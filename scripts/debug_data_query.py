#!/usr/bin/env python3
"""Debug why the tuning query returns 0 rows"""
from sqlalchemy import create_engine, text
import pandas as pd
from libs.paths import database_url, no_winsor_database_url

engine = create_engine(no_winsor_database_url())

with engine.connect() as conn:
    # Check total rows in fight_stats_fe
    result1 = pd.read_sql(text("SELECT COUNT(*) as cnt FROM features.fight_stats_fe"), conn)
    print(f"Total rows in fight_stats_fe: {result1['cnt'][0]}")

    # Check with JOIN to fight_mapping
    result2 = pd.read_sql(text("""
        SELECT COUNT(*) as cnt
        FROM features.fight_stats_fe fe
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
    """), conn)
    print(f"After JOIN with fight_mapping: {result2['cnt'][0]}")

    # Check with JOIN to event_mapping
    result3 = pd.read_sql(text("""
        SELECT COUNT(*) as cnt
        FROM features.fight_stats_fe fe
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fe.event_id = em.event_id
    """), conn)
    print(f"After JOIN with event_mapping: {result3['cnt'][0]}")

    # Check date filter
    result4 = pd.read_sql(text("""
        SELECT COUNT(*) as cnt, MIN(em.event_date) as min_date, MAX(em.event_date) as max_date
        FROM features.fight_stats_fe fe
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fe.event_id = em.event_id
        WHERE em.event_date >= '2014-01-01' AND em.event_date < '2023-01-01'
    """), conn)
    print(f"After date filter 2014-2023: {result4['cnt'][0]}, dates: {result4['min_date'][0]} to {result4['max_date'][0]}")

    # Check weight classes
    wc_sample = pd.read_sql(text("""
        SELECT DISTINCT fm.weightclass
        FROM features.fight_stats_fe fe
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        ORDER BY fm.weightclass
        LIMIT 20
    """), conn)
    print(f"\nWeight classes in data:")
    print(wc_sample['weightclass'].tolist())
