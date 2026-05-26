#!/usr/bin/env python3
"""Check fight_stats_fe schema"""
from sqlalchemy import create_engine, text
import pandas as pd
from libs.paths import database_url, no_winsor_database_url

engine = create_engine(no_winsor_database_url())

with engine.connect() as conn:
    # Get all columns
    cols = pd.read_sql(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'features'
          AND table_name = 'fight_stats_fe'
        ORDER BY ordinal_position
    """), conn)

    print("fight_stats_fe columns:")
    print("="*80)
    for _, row in cols.iterrows():
        print(f"{row['column_name']:<40} {row['data_type']}")

    print()
    print(f"Total columns: {len(cols)}")

    # Check sample row
    sample = pd.read_sql(text("""
        SELECT *
        FROM features.fight_stats_fe fe
        JOIN features.fight_mapping fm ON fe.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        WHERE em.event_date >= '2020-01-01'
        LIMIT 1
    """), conn)

    print()
    print("Sample row keys:")
    print(list(sample.columns))
