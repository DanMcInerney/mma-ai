#!/usr/bin/env python3
"""
Investigate Kamaru Usman's reversal stats to understand the high score.
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists
from contextlib import contextmanager
from pathlib import Path
from libs.paths import database_url, no_winsor_database_url

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

DB_URL = no_winsor_database_url()

def create_db_engine(db_url=DB_URL):
    """Create and configure the database engine"""
    if not database_exists(db_url):
        print(f"ERROR: Database does not exist at {db_url}")
        sys.exit(1)
    
    return create_engine(
        db_url,
        pool_size=5,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=14400
    )

@contextmanager
def get_db_connection(engine):
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def get_fighter_id(conn, fighter_name):
    """Get fighter_id from fighter name"""
    query = text("""
        SELECT fighter_id 
        FROM features.fighter_mapping 
        WHERE LOWER(fighter_name) = LOWER(:fighter_name)
    """)
    result = conn.execute(query, {"fighter_name": fighter_name}).fetchone()
    return result[0] if result else None

def investigate_usman_buckley_fight(conn):
    """Investigate the Usman vs Buckley fight reversal data"""
    print("=" * 80)
    print("INVESTIGATING KAMARU USMAN vs JOAQUIN BUCKLEY")
    print("=" * 80)
    
    usman_id = get_fighter_id(conn, 'kamaru usman')
    buckley_id = get_fighter_id(conn, 'joaquin buckley')
    
    if not usman_id or not buckley_id:
        print("Could not find fighter IDs")
        return
    
    print(f"\nUsman ID: {usman_id}")
    print(f"Buckley ID: {buckley_id}")
    
    # Find the fight
    query = text("""
        SELECT 
            fm.fight_id,
            em.event_date,
            fm.fighter1_id,
            fm.fighter2_id,
            fm.result,
            fm.method
        FROM features.fight_mapping fm
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        WHERE em.event_date = '2025-06-14'
          AND ((fm.fighter1_id = :usman_id AND fm.fighter2_id = :buckley_id)
               OR (fm.fighter1_id = :buckley_id AND fm.fighter2_id = :usman_id))
    """)
    
    fight = conn.execute(query, {"usman_id": usman_id, "buckley_id": buckley_id}).fetchone()
    
    if not fight:
        print("Fight not found!")
        return
    
    fight_id = fight[0]
    print(f"\nFight ID: {fight_id}")
    print(f"Event Date: {fight[1]}")
    print(f"Result: {fight[4]}, Method: {fight[5]}")
    
    # First, check what columns exist in rev table
    print("\n" + "=" * 80)
    print("CHECKING REV TABLE COLUMNS")
    print("=" * 80)
    
    query_cols = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = 'rev'
        ORDER BY column_name
    """)
    
    cols = [row[0] for row in conn.execute(query_cols).fetchall()]
    print("\nAvailable columns in rev table:")
    for col in cols:
        print(f"  - {col}")
    
    # Get reversal stats for this fight
    print("\n" + "=" * 80)
    print("REVERSAL STATS FOR THIS FIGHT")
    print("=" * 80)
    
    # Get all rev columns for this fight
    query = text(f"""
        SELECT 
            f.fighter_id,
            LOWER(fm.fighter_name) as fighter_name,
            {', '.join([f'f.{col}' for col in cols if col not in ['fight_id', 'event_id']])}
        FROM features.rev f
        JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
        WHERE f.fight_id = :fight_id
        ORDER BY f.fighter_id
    """)
    
    df = pd.read_sql(query, conn, params={"fight_id": fight_id})
    print("\nRaw reversal data:")
    print(df.to_string())
    
    # Get control time stats
    print("\n" + "=" * 80)
    print("CONTROL TIME STATS FOR THIS FIGHT")
    print("=" * 80)
    
    query_ctrl_cols = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = 'ctrl'
        ORDER BY column_name
    """)
    
    ctrl_cols = [row[0] for row in conn.execute(query_ctrl_cols).fetchall()]
    
    query = text(f"""
        SELECT 
            f.fighter_id,
            LOWER(fm.fighter_name) as fighter_name,
            {', '.join([f'f.{col}' for col in ctrl_cols if col not in ['fight_id', 'event_id']])}
        FROM features.ctrl f
        JOIN features.fighter_mapping fm ON f.fighter_id = fm.fighter_id
        WHERE f.fight_id = :fight_id
        ORDER BY f.fighter_id
    """)
    
    df_ctrl = pd.read_sql(query, conn, params={"fight_id": fight_id})
    print("\nControl time data:")
    print(df_ctrl.to_string())
    
    # Calculate rev_per_ctrlopp manually from the data we already have
    print("\n" + "=" * 80)
    print("MANUAL CALCULATION: rev_per_ctrlopp")
    print("=" * 80)
    
    usman_row = df[df['fighter_id'] == usman_id].iloc[0]
    buckley_row = df[df['fighter_id'] == buckley_id].iloc[0]
    
    usman_rev = usman_row['rev']
    buckley_ctrl = buckley_row['ctrl'] if 'ctrl' in buckley_row else df_ctrl[df_ctrl['fighter_id'] == buckley_id]['ctrl'].iloc[0]
    
    print(f"\nUsman reversals (rev): {usman_rev}")
    print(f"Buckley control time: {buckley_ctrl} seconds")
    print(f"Usman rev_per_ctrlopp (raw): {usman_row['rev_per_ctrlopp']}")
    
    if buckley_ctrl > 0:
        calculated = usman_rev / buckley_ctrl
        print(f"Calculated: {usman_rev} / {buckley_ctrl} = {calculated}")
        print(f"\nNOTE: rev_per_ctrlopp = reversals / opponent_control_time")
        print(f"Even with 0 reversals, if opponent has very little control time,")
        print(f"the ratio can be high. But the ADJUSTED performance accounts for")
        print(f"opponent quality - Buckley's opponents normally allow very few reversals")
        print(f"per control time, so Usman's performance (even if 0 reversals) is")
        print(f"compared against that baseline.")
    else:
        print("Buckley had 0 control time!")
    
    # Get Usman's career reversal stats
    print("\n" + "=" * 80)
    print("USMAN'S CAREER REVERSAL STATS (all 2025 fights)")
    print("=" * 80)
    
    query = text("""
        SELECT 
            em.event_date,
            CASE 
                WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
                ELSE LOWER(fm1.fighter_name)
            END as opponent_name,
            f.rev,
            f.rev_per_ctrlopp,
            f.rev_per_ctrlopp_dec_adjperf,
            f.rev_per_ctrlopp_dec_adjperf_dec_avg,
            fm.result,
            fm.method
        FROM features.rev f
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
        JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
        WHERE f.fighter_id = :usman_id
          AND em.event_date >= '2025-01-01'
          AND em.event_date < '2026-01-01'
        ORDER BY em.event_date DESC
    """)
    
    df_career = pd.read_sql(query, conn, params={"usman_id": usman_id})
    print("\nUsman's 2025 fights:")
    print(df_career.to_string())
    
    # Check what rev_per_ctrlopp means - look at the calculation
    print("\n" + "=" * 80)
    print("UNDERSTANDING rev_per_ctrlopp")
    print("=" * 80)
    print("\nrev_per_ctrlopp = reversals / opponent's control time")
    print("If opponent has very little control time, even 0 reversals can result in a high ratio")
    print("(or if the calculation handles division by zero differently)")

def main():
    engine = create_db_engine()
    with get_db_connection(engine) as conn:
        investigate_usman_buckley_fight(conn)

if __name__ == '__main__':
    main()
