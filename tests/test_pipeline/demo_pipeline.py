#!/usr/bin/env python3
"""
Demo script that shows the pipeline test in action.
This creates dummy data and loads it through the actual pipeline logic.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import random

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database, drop_database
from contextlib import contextmanager

from libs.feature_store.core import CoreFeatureStore
from libs.feature_store.schema import initialize_schema


@contextmanager
def get_demo_db_connection(engine):
    """Context manager for demo database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()


def create_demo_data():
    """Create small demo dataset with predictable stats"""
    
    # Create 4 fighters (2 per weight class) - matching the main test
    individuals_data = [
        {
            'name': 'john smith',
            'nickname': 'the hammer',
            'url': 'http://ufcstats.com/fighter-details/john-smith-001',
            'dob': 'Jan 15, 1990',
            'weight': '155 lbs.',
            'reach': '70"',
            'height': '5\' 10"',
            'stance': 'Orthodox'
        },
        {
            'name': 'mike jones',
            'nickname': 'lightning',
            'url': 'http://ufcstats.com/fighter-details/mike-jones-002',
            'dob': 'Mar 22, 1988',
            'weight': '155 lbs.',
            'reach': '72"',
            'height': '6\' 0"',
            'stance': 'Southpaw'
        },
        {
            'name': 'steve anderson',
            'nickname': 'the crusher',
            'url': 'http://ufcstats.com/fighter-details/steve-anderson-005',
            'dob': 'Feb 28, 1985',
            'weight': '170 lbs.',
            'reach': '75"',
            'height': '6\' 3"',
            'stance': 'Orthodox'
        },
        {
            'name': 'frank miller',
            'nickname': 'tank',
            'url': 'http://ufcstats.com/fighter-details/frank-miller-006',
            'dob': 'Aug 14, 1990',
            'weight': '170 lbs.',
            'reach': '72"',
            'height': '6\' 0"',
            'stance': 'Orthodox'
        }
    ]
    
    # Create 2 fights
    competitions_data = []
    
    # Fight 1: Lightweight
    fight1 = {
        'result': 'W',
        'player1': 'john smith',
        'player2': 'mike jones',
        'player1_url': 'http://ufcstats.com/fighter-details/john-smith-001',
        'player2_url': 'http://ufcstats.com/fighter-details/mike-jones-002',
        'weightclass': 'Lightweight Bout',
        'method': 'Decision - Unanimous',
        'round': 3,
        'time': '5:00',
        'time_format': '3 Rnd (5-5-5)',
        'referee': 'John McCarthy',
        'details': 'Demo fight 1',
        'player1_nickname': 'the hammer',
        'player2_nickname': 'lightning',
        'event_date': 'January 15, 2020',
        'event_location': 'Las Vegas, Nevada, United States',
        'event_url': 'http://ufcstats.com/event-details/demo-event-001'
    }
    
    # Add predictable round stats for fight 1 (Decision fight - 3 rounds)
    round_stats = [
        # Round 1
        {'p1': {'Sig_str': '10 of 20', 'Total_str': '12 of 20', 'Head': '6 of 10', 'Body': '2 of 5', 'Leg': '2 of 5', 'Distance': '8 of 15', 'Clinch': '2 of 3', 'Ground': '0 of 2', 'Td': '1 of 3', 'KD': 0, 'Sub_att': 0, 'Rev': 1, 'Ctrl': '2:30'},
         'p2': {'Sig_str': '8 of 20', 'Total_str': '10 of 20', 'Head': '5 of 12', 'Body': '2 of 4', 'Leg': '1 of 4', 'Distance': '6 of 15', 'Clinch': '1 of 3', 'Ground': '1 of 2', 'Td': '0 of 2', 'KD': 0, 'Sub_att': 0, 'Rev': 0, 'Ctrl': '0:30'}},
        # Round 2  
        {'p1': {'Sig_str': '8 of 16', 'Total_str': '10 of 16', 'Head': '5 of 8', 'Body': '2 of 4', 'Leg': '1 of 4', 'Distance': '6 of 12', 'Clinch': '2 of 2', 'Ground': '0 of 2', 'Td': '1 of 2', 'KD': 0, 'Sub_att': 0, 'Rev': 1, 'Ctrl': '1:45'},
         'p2': {'Sig_str': '6 of 16', 'Total_str': '8 of 16', 'Head': '4 of 10', 'Body': '1 of 3', 'Leg': '1 of 3', 'Distance': '5 of 12', 'Clinch': '1 of 2', 'Ground': '1 of 2', 'Td': '0 of 2', 'KD': 0, 'Sub_att': 0, 'Rev': 0, 'Ctrl': '0:15'}},
        # Round 3
        {'p1': {'Sig_str': '5 of 10', 'Total_str': '6 of 10', 'Head': '3 of 5', 'Body': '1 of 3', 'Leg': '1 of 2', 'Distance': '4 of 8', 'Clinch': '1 of 2', 'Ground': '0 of 1', 'Td': '0 of 2', 'KD': 0, 'Sub_att': 0, 'Rev': 0, 'Ctrl': '1:00'},
         'p2': {'Sig_str': '4 of 10', 'Total_str': '5 of 10', 'Head': '2 of 6', 'Body': '1 of 2', 'Leg': '1 of 2', 'Distance': '3 of 8', 'Clinch': '1 of 2', 'Ground': '0 of 1', 'Td': '0 of 1', 'KD': 0, 'Sub_att': 0, 'Rev': 0, 'Ctrl': '0:00'}}
    ]
    
    for round_idx in range(1, 4):  # 3 rounds
        for stat in ['KD', 'Sig_str', 'Total_str', 'Td', 'Sub_att', 'Rev', 'Ctrl',
                    'Head', 'Body', 'Leg', 'Distance', 'Clinch', 'Ground']:
            fight1[f'p1_rd{round_idx}_{stat}'] = round_stats[round_idx-1]['p1'][stat]
            fight1[f'p2_rd{round_idx}_{stat}'] = round_stats[round_idx-1]['p2'][stat]
    
    # Add empty stats for rounds 4 and 5
    for round_idx in range(4, 6):
        for stat in ['KD', 'Sig_str', 'Total_str', 'Td', 'Sub_att', 'Rev', 'Ctrl',
                    'Head', 'Body', 'Leg', 'Distance', 'Clinch', 'Ground']:
            fight1[f'p1_rd{round_idx}_{stat}'] = None
            fight1[f'p2_rd{round_idx}_{stat}'] = None
    
    competitions_data.append(fight1)
    
    # Fight 2: Welterweight (similar structure)
    fight2 = fight1.copy()
    fight2.update({
        'player1': 'steve anderson',
        'player2': 'frank miller', 
        'player1_url': 'http://ufcstats.com/fighter-details/steve-anderson-007',
        'player2_url': 'http://ufcstats.com/fighter-details/frank-miller-008',
        'weightclass': 'Welterweight Bout',
        'method': 'TKO - Punches',
        'round': 2,
        'time': '3:45',
        'player1_nickname': 'the crusher',
        'player2_nickname': 'tank',
        'event_date': 'February 15, 2020',
        'event_url': 'http://ufcstats.com/event-details/demo-event-002'
    })
    
    competitions_data.append(fight2)
    
    return pd.DataFrame(individuals_data), pd.DataFrame(competitions_data)


def main():
    """Run the demo pipeline"""
    
    print("=" * 80)
    print("DEMO: UFC Fight Data Pipeline")
    print("=" * 80)
    
    # Create demo database
    demo_db_url = 'postgresql://your local Postgres credentials@localhost:5432/demo_pipeline_db'
    
    print(f"Setting up demo database: {demo_db_url}")
    
    # Drop existing demo database if it exists
    if database_exists(demo_db_url):
        print("Dropping existing demo database...")
        drop_database(demo_db_url)
    
    # Create fresh demo database
    print("Creating fresh demo database...")
    create_database(demo_db_url)
    
    # Create engine
    engine = create_engine(demo_db_url, echo=False)  # Set echo=True to see SQL queries
    
    try:
        with get_demo_db_connection(engine) as conn:
            print("\n1. Creating dummy data...")
            individuals_df, competitions_df = create_demo_data()
            
            print(f"   - Created {len(individuals_df)} fighters")
            print(f"   - Created {len(competitions_df)} fights")
            
            print("\n2. Initializing database schema...")
            initialize_schema(conn)
            print("   - Schema 'features' created")
            print("   - Core tables created (fighter_mapping, event_mapping, fight_mapping, etc.)")
            
            print("\n3. Loading data through CoreFeatureStore...")
            core_store = CoreFeatureStore(conn)
            
            # Filter data (following main.py logic)
            competitions_df = competitions_df[~competitions_df['weightclass'].str.lower().str.contains('women', na=False)]
            competitions_df['date'] = pd.to_datetime(competitions_df['event_date'])
            competitions_df = competitions_df[competitions_df['date'] > pd.to_datetime('1998-05-15')]
            competitions_df.drop(columns=['date'], inplace=True)
            competitions_df.reset_index(drop=True, inplace=True)
            
            # Load features
            print("   - Loading fighter features...")
            core_store.load_fighter_features(individuals_df)
            
            print("   - Loading fight features...")
            core_store.load_fight_features(competitions_df)
            
            print("   - Loading fight stats...")
            core_store.load_fight_stats(competitions_df)
            
            print("   - Creating fight_stats_fe...")
            conn.execute(text('''
                INSERT INTO features.fight_stats_fe 
                SELECT * FROM features.fight_stats_core;
            '''))
            conn.commit()
            
            print("\n4. Verifying loaded data...")
            
            # Check fighter_mapping
            fighters = pd.read_sql("SELECT COUNT(*) as count FROM features.fighter_mapping", conn)
            print(f"   - Fighters loaded: {fighters.iloc[0]['count']}")
            
            # Check fight_mapping
            fights = pd.read_sql("SELECT COUNT(*) as count FROM features.fight_mapping", conn)
            print(f"   - Fights loaded: {fights.iloc[0]['count']}")
            
            # Check fight_stats_core
            stats_core = pd.read_sql("SELECT COUNT(*) as count FROM features.fight_stats_core", conn)
            print(f"   - Fight stats (core): {stats_core.iloc[0]['count']} records")
            
            # Check fight_stats_fe
            stats_fe = pd.read_sql("SELECT COUNT(*) as count FROM features.fight_stats_fe", conn)
            print(f"   - Fight stats (fe): {stats_fe.iloc[0]['count']} records")
            
            print("\n5. Sample data verification...")
            
            # Show a sample fighter
            sample_fighter = pd.read_sql("""
                SELECT fighter_name, fighter_nickname, fighter_height, fighter_reach, fighter_weight
                FROM features.fighter_mapping 
                LIMIT 1
            """, conn)
            
            if len(sample_fighter) > 0:
                fighter = sample_fighter.iloc[0]
                print(f"   - Sample fighter: {fighter['fighter_name']} '{fighter['fighter_nickname']}'")
                print(f"     Height: {fighter['fighter_height']} inches, Reach: {fighter['fighter_reach']} inches")
                print(f"     Weight class: {fighter['fighter_weight']}")
            
            # Show a sample fight
            sample_fight = pd.read_sql("""
                SELECT fm.weightclass, fm.method, fm.end_round, fm.end_time,
                       f1.fighter_name as fighter1, f2.fighter_name as fighter2
                FROM features.fight_mapping fm
                JOIN features.fighter_mapping f1 ON fm.fighter1_id = f1.fighter_id
                JOIN features.fighter_mapping f2 ON fm.fighter2_id = f2.fighter_id
                LIMIT 1
            """, conn)
            
            if len(sample_fight) > 0:
                fight = sample_fight.iloc[0]
                print(f"   - Sample fight: {fight['fighter1']} vs {fight['fighter2']}")
                print(f"     {fight['weightclass']}, {fight['method']}")
                print(f"     End Round {fight['end_round']}, End Time {fight['end_time']} seconds")
            
            print("\n✅ Demo completed successfully!")
            print("\nThe pipeline successfully:")
            print("   1. Created a PostgreSQL database with proper schema")
            print("   2. Loaded fighter data with proper transformations")
            print("   3. Loaded fight data with all round-by-round statistics")
            print("   4. Created the foundation for feature engineering")
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        engine.dispose()
        if database_exists(demo_db_url):
            print(f"\nCleaning up demo database...")
            drop_database(demo_db_url)
            print("Demo database removed.")


if __name__ == "__main__":
    main()
