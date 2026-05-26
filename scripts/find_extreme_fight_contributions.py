#!/usr/bin/env python3
"""
Find the specific fight that most contributed to each fighter's extreme stat.
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

# Database connection URL
DB_URL = no_winsor_database_url()

# Fighters and their extreme stats from the blog
EXTREME_STATS = [
    {
        'fighter': 'kamaru usman',
        'stat': 'rev_per_ctrlopp_dec_adjperf_dec_avg',
        'value': 260.9973442594933,
        'description': 'Reversals per minute of opponent control time'
    },
    {
        'fighter': 'giga chikadze',
        'stat': 'td_land_per_ctrl_dec_adjperf_dec_avg',
        'value': 115.17167522864901,
        'description': 'Takedowns per minute of control time'
    },
    {
        'fighter': 'kyoji horiguchi',
        'stat': 'ground_land_ratio_dec_adjperf_dec_avg',
        'value': 5.921686024818788,
        'description': 'Ground strike ratio'
    },
    {
        'fighter': 'toshiomi kazama',
        'stat': 'sig_str_def_dec_adjperf_dec_avg',
        'value': 3.833285137401958,
        'description': 'Striking defense'
    },
    {
        'fighter': 'toshiomi kazama',
        'stat': 'sig_str_land_ratio_dec_adjperf_dec_avg',
        'value': -4.2214931274949805,
        'description': 'Striking offense (worst)'
    },
    {
        'fighter': 'jamahal hill',
        'stat': 'ground_land_per_ctrl_dec_avg',
        'value': 7.0911545339633975,
        'description': 'Ground strikes per minute of control time'
    },
    {
        'fighter': 'daniel santos',
        'stat': 'win_dec_adjperf_dec_avg',
        'value': 47.69157360933569,
        'description': 'Win rate'
    },
    {
        'fighter': 'kyle prepolec',
        'stat': 'win_dec_adjperf_dec_avg',
        'value': -46.135736208205316,
        'description': 'Win rate (worst)'
    },
    {
        'fighter': 'anshul jubli',
        'stat': 'ko_per_sig_str_land_dec_adjperf_dec_avg',
        'value': 35.13823589249298,
        'description': 'Knockouts per significant strike landed'
    },
    {
        'fighter': 'valter walker',
        'stat': 'sub_att_ratio_dec_adjperf_dec_avg',
        'value': 11.792833111717044,
        'description': 'Submission attempt ratio'
    },
    {
        'fighter': 'kyoji horiguchi',
        'stat': 'distance_land_ratio_dec_adjperf_dec_avg',
        'value': 4.809928915996663,
        'description': 'Distance strike ratio'
    },
    {
        'fighter': 'kyoji horiguchi',
        'stat': 'clinch_acc_dec_adjperf_dec_avg',
        'value': -5.413753328359169,
        'description': 'Clinch accuracy (worst)'
    },
    {
        'fighter': 'merab dvalishvili',
        'stat': 'clinch_land_ratio_dec_adjperf_dec_avg',
        'value': 4.310413728854347,
        'description': 'Clinch strike ratio'
    },
]

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

def find_table_for_stat(conn, stat_name):
    """Find which table contains a given stat"""
    # Map stat prefixes to likely table names
    prefix_to_table = {
        'age': 'age',
        'reach': 'reach',
        'days_since_last_fight': 'days_since_last_fight',
        'weightclass': 'fight_mapping',
        'sig_str': 'sig_str',
        'head': 'head',
        'body': 'body',
        'leg': 'leg',
        'distance': 'distance',
        'clinch': 'clinch',
        'ground': 'ground',
        'ko': 'ko',
        'sub': 'sub',
        'decision': 'decision',
        'win': 'win',
        'rev': 'rev',
        'ctrl': 'ctrl',
        'td': 'td',
    }
    
    # Find matching table
    for prefix, table in prefix_to_table.items():
        if stat_name.startswith(prefix):
            return table
    
    # Check all tables if not found
    query = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'features'
        AND table_type = 'BASE TABLE'
        AND table_name NOT IN ('fighter_mapping', 'event_mapping', 'fight_mapping', 
                               'fight_stats_core', 'fight_stats_fe', 'fight_stats_derived',
                               'first_time', 'minimum')
        ORDER BY table_name
    """)
    
    tables = [row[0] for row in conn.execute(query).fetchall()]
    
    # Check each table for the column
    for table in tables:
        check_query = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'features' 
            AND table_name = :table_name
            AND column_name = :column_name
        """)
        result = conn.execute(check_query, {"table_name": table, "column_name": stat_name}).fetchone()
        if result:
            return table
    
    return None

def get_fighter_id(conn, fighter_name):
    """Get fighter_id from fighter name"""
    query = text("""
        SELECT fighter_id 
        FROM features.fighter_mapping 
        WHERE LOWER(fighter_name) = LOWER(:fighter_name)
    """)
    result = conn.execute(query, {"fighter_name": fighter_name}).fetchone()
    return result[0] if result else None

def get_individual_stat_name(stat_name):
    """Convert dec_avg stat to individual fight stat"""
    # Remove _dec_avg or _dec_adjperf_dec_avg to get individual fight stat
    if '_dec_adjperf_dec_avg' in stat_name:
        return stat_name.replace('_dec_adjperf_dec_avg', '_dec_adjperf')
    elif '_dec_avg' in stat_name:
        return stat_name.replace('_dec_avg', '')
    return stat_name

def get_extreme_fight_for_stat(conn, fighter_id, fighter_name, stat_name, table_name, is_highest=True):
    """Get the fight that most contributed to the extreme stat"""
    
    # Try to get individual fight stat (without _dec_avg)
    individual_stat = get_individual_stat_name(stat_name)
    
    # Check if individual stat column exists
    check_query = text("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = :table_name
        AND column_name = :column_name
    """)
    
    use_stat = stat_name  # Default to dec_avg if individual doesn't exist
    result = conn.execute(check_query, {"table_name": table_name, "column_name": individual_stat}).fetchone()
    if result:
        use_stat = individual_stat
    
    # Get all fights for this fighter in 2025 with the stat value
    query = text(f"""
        SELECT 
            f.fight_id,
            f.{use_stat} as stat_value,
            em.event_date,
            fm.fighter1_id,
            fm.fighter2_id,
            fm.result,
            fm.method,
            CASE 
                WHEN f.fighter_id = fm.fighter1_id THEN LOWER(fm2.fighter_name)
                ELSE LOWER(fm1.fighter_name)
            END as opponent_name,
            CASE 
                WHEN f.fighter_id = fm.fighter1_id THEN fm.fighter2_id
                ELSE fm.fighter1_id
            END as opponent_id
        FROM features.{table_name} f
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fighter_mapping fm1 ON fm.fighter1_id = fm1.fighter_id
        JOIN features.fighter_mapping fm2 ON fm.fighter2_id = fm2.fighter_id
        WHERE f.fighter_id = :fighter_id
          AND em.event_date >= '2025-01-01' 
          AND em.event_date < '2026-01-01'
          AND f.{use_stat} IS NOT NULL
        ORDER BY f.{use_stat} {'DESC' if is_highest else 'ASC'}
        LIMIT 1
    """)
    
    try:
        result = conn.execute(query, {"fighter_id": fighter_id}).fetchone()
        if result:
            return {
                'fight_id': result[0],
                'stat_value': float(result[1]) if result[1] is not None else None,
                'event_date': result[2],
                'opponent_name': result[7],
                'result': result[5],
                'method': result[6]
            }
    except Exception as e:
        print(f"  Error querying {stat_name}: {e}")
    
    return None

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINDING EXTREME FIGHT CONTRIBUTIONS")
    print("=" * 80)
    
    engine = create_db_engine()
    results = []
    
    with get_db_connection(engine) as conn:
        for fighter_stat in EXTREME_STATS:
            fighter_name = fighter_stat['fighter']
            stat_name = fighter_stat['stat']
            
            print(f"\nAnalyzing: {fighter_name} - {stat_name}")
            
            # Get fighter ID
            fighter_id = get_fighter_id(conn, fighter_name)
            if not fighter_id:
                print(f"  WARNING: Fighter '{fighter_name}' not found")
                continue
            
            # Find table
            table_name = find_table_for_stat(conn, stat_name)
            if not table_name:
                print(f"  WARNING: Could not find table for stat '{stat_name}'")
                continue
            
            # Determine if we're looking for highest or lowest
            is_highest = fighter_stat['value'] > 0  # Positive values are typically "best"
            # Exception: for worst stats, we want the lowest value
            if 'worst' in fighter_stat['description'].lower():
                is_highest = False
            
            # Get extreme fight
            fight_data = get_extreme_fight_for_stat(
                conn, fighter_id, fighter_name, stat_name, table_name, is_highest
            )
            
            if fight_data:
                result = {
                    'fighter': fighter_name,
                    'stat': stat_name,
                    'description': fighter_stat['description'],
                    'career_value': fighter_stat['value'],
                    'fight_stat_value': fight_data['stat_value'],
                    'opponent': fight_data['opponent_name'],
                    'event_date': fight_data['event_date'],
                    'result': fight_data['result'],
                    'method': fight_data['method']
                }
                results.append(result)
                
                win_loss = "Win" if fight_data['result'] == 1 else "Loss" if fight_data['result'] == 0 else "Draw/NC"
                print(f"  Fight: vs {fight_data['opponent_name']} on {fight_data['event_date']}")
                print(f"  Stat value in fight: {fight_data['stat_value']:.4f}")
                print(f"  Result: {win_loss} via {fight_data['method']}")
            else:
                print(f"  WARNING: No fight data found")
    
    # Create results dataframe
    if results:
        df = pd.DataFrame(results)
        
        # Save to CSV
        output_path = project_root / "data" / "extreme_fight_contributions.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n\nResults saved to: {output_path}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for result in results:
            win_loss = "Win" if result['result'] == 1 else "Loss" if result['result'] == 0 else "Draw/NC"
            print(f"\n{result['fighter'].upper()}: {result['description']}")
            print(f"  Fight: vs {result['opponent']} on {result['event_date']}")
            print(f"  Fight stat value: {result['fight_stat_value']:.4f}")
            print(f"  Career average: {result['career_value']:.4f}")
            print(f"  Result: {win_loss} via {result['method']}")

if __name__ == '__main__':
    main()

