#!/usr/bin/env python3
"""Check sig_str table schema and sample data"""
from sqlalchemy import create_engine, text
import pandas as pd
from libs.paths import database_url, no_winsor_database_url

engine = create_engine(no_winsor_database_url())

with engine.connect() as conn:
    # Get column names
    cols = pd.read_sql(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'features'
          AND table_name = 'sig_str'
        ORDER BY ordinal_position
    """), conn)

    print("sig_str table columns:")
    print(cols.to_string(index=False))
    print()

    # Get a sample row with event_date
    sample = pd.read_sql(text("""
        SELECT s.*, em.event_date, fm.weight_class
        FROM features.sig_str s
        JOIN features.fight_mapping fm ON s.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        WHERE em.event_date >= '2020-01-01'
        LIMIT 1
    """), conn)

    print("Sample row:")
    for col in sample.columns:
        print(f"{col}: {sample[col].iloc[0]}")
