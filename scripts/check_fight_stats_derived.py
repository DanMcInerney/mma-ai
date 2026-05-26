#!/usr/bin/env python3
"""Quick check of fight_stats_derived table"""
from sqlalchemy import create_engine, text
import pandas as pd
from libs.paths import database_url, no_winsor_database_url

engine = create_engine(no_winsor_database_url())

with engine.connect() as conn:
    # Check basic stats
    result = pd.read_sql(text("""
        SELECT
            COUNT(*) as count,
            MIN(event_date) as min_date,
            MAX(event_date) as max_date
        FROM features.fight_stats_derived
    """), conn)

    print("fight_stats_derived table stats:")
    print(result)
    print()

    # Check a sample
    sample = pd.read_sql(text("""
        SELECT * FROM features.fight_stats_derived
        WHERE event_date >= '2014-01-01'
        LIMIT 5
    """), conn)

    print(f"Sample rows (filtered by event_date >= 2014-01-01): {len(sample)} rows")
    if len(sample) > 0:
        print("Columns:", list(sample.columns))
    else:
        print("NO ROWS FOUND with event_date >= 2014-01-01")

        # Check if there's any data at all
        any_data = pd.read_sql(text("SELECT * FROM features.fight_stats_derived LIMIT 5"), conn)
        print(f"\nTotal rows in table (no filter): {len(any_data)}")
        if len(any_data) > 0:
            print("Columns:", list(any_data.columns))
            print("\nFirst row:")
            print(any_data.iloc[0])
