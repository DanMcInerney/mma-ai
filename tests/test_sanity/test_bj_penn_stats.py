"""
Test script to extract BJ Penn's career stats and opponent stats for sanity checking.

This script queries the database to extract the following stats for BJ Penn and his opponents:

KO Stats:
- ko, ko_dec_avg, ko_avg, ko_adjperf, ko_dec_adjperf, ko_dec_adjperf_dec_avg, ko_adjperf_dec_avg

KO Per Sig Str Land Stats:
- ko_per_sig_str_land, ko_per_sig_str_land_dec_avg, ko_per_sig_str_land_avg, 
- ko_per_sig_str_land_adjperf, ko_per_sig_str_land_dec_adjperf, 
- ko_per_sig_str_land_dec_adjperf_dec_avg, ko_per_sig_str_land_adjperf_dec_avg

Sig Str Land Stats:
- sig_str_land, sig_str_land_dec_avg, sig_str_land_avg

The output is formatted as JSON for easy analysis and sanity checking.

ADJPERF CALCULATION:
stat_adjperf = (fighter1_stat - fighter2_stat_opp_avg) / fighter2_stat_opp_mad

Where:
- fighter1_stat: The current fighter's stat value for this fight
- fighter2_stat_opp_avg: The opponent's historical average of this stat against their previous opponents
- fighter2_stat_opp_mad: The opponent's historical median absolute deviation of this stat against their previous opponents

For first-time fighters (no previous fights), fallback values are used from weightclass averages.
"""

import os
import sys
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
from contextlib import contextmanager
import pandas as pd
from tabulate import tabulate

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def create_db_engine(db_url='postgresql://postgres@localhost:5432/mma-ai'):
    """Create database engine with connection pooling"""
    if not database_exists(db_url):
        create_database(db_url)
    
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

def find_bj_penn_fighter_id(conn):
    """Find BJ Penn's fighter_id in the database"""
    query = text("""
        SELECT fighter_id, fighter_name, fighter_nickname
        FROM features.fighter_mapping 
        WHERE LOWER(fighter_name) LIKE '%bj penn%' 
           OR LOWER(fighter_name) LIKE '%b.j. penn%'
           OR LOWER(fighter_name) LIKE '%penn%bj%'
        ORDER BY fighter_name
    """)
    
    result = conn.execute(query)
    fighters = result.fetchall()
    
    print("Found potential BJ Penn matches:")
    for fighter in fighters:
        print(f"ID: {fighter.fighter_id}, Name: {fighter.fighter_name}, Nickname: {fighter.fighter_nickname}")
    
    # Return the first match (assuming it's BJ Penn)
    if fighters:
        return fighters[0].fighter_id
    else:
        raise ValueError("BJ Penn not found in database")

def get_bj_penn_fights(conn, fighter_id):
    """Get all of BJ Penn's fights with event information"""
    query = text("""
        SELECT DISTINCT
            fm.fight_id,
            em.event_date,
            em.event_location,
            CASE 
                WHEN fm.fighter1_id = :fighter_id THEN f2.fighter_name
                ELSE f1.fighter_name
            END as opponent_name,
            CASE 
                WHEN fm.fighter1_id = :fighter_id THEN fm.fighter2_id
                ELSE fm.fighter1_id
            END as opponent_id,
            fm.method,
            fm.result,
            fm.weightclass
        FROM features.fight_mapping fm
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        JOIN features.fighter_mapping f1 ON fm.fighter1_id = f1.fighter_id
        JOIN features.fighter_mapping f2 ON fm.fighter2_id = f2.fighter_id
        WHERE fm.fighter1_id = :fighter_id OR fm.fighter2_id = :fighter_id
        ORDER BY em.event_date ASC
    """)
    
    result = conn.execute(query, {"fighter_id": fighter_id})
    return result.fetchall()

def get_stats_for_fight(conn, fight_id, fighter_id, opponent_id, table_name, stat_columns):
    """Get stats for both BJ Penn and his opponent for a specific fight"""
    # Build column list for SELECT
    columns_str = ", ".join([f"t.{col}" for col in stat_columns])
    
    query = text(f"""
        SELECT 
            t.fighter_id,
            {columns_str}
        FROM features.{table_name} t
        WHERE t.fight_id = :fight_id 
          AND t.fighter_id IN (:fighter_id, :opponent_id)
    """)
    
    result = conn.execute(query, {
        "fight_id": fight_id, 
        "fighter_id": fighter_id, 
        "opponent_id": opponent_id
    })
    
    rows = result.fetchall()
    
    # Organize results by fighter
    stats = {}
    for row in rows:
        fighter_stats = {}
        for i, col in enumerate(stat_columns):
            fighter_stats[col] = getattr(row, col, None)
        stats[row.fighter_id] = fighter_stats
    
    return stats

def get_opponent_history_analysis(conn, bj_penn_id, current_fight_id, opponent_id, table_name, stat_columns):
    """
    Analyze what BJ Penn's historical opponents achieved against him.
    This helps validate the new reliability-weighted adjperf calculations.
    """
    # Get BJ Penn's historical fights before the current fight
    history_query = text(f"""
        WITH bj_historical_fights AS (
            SELECT DISTINCT
                fm.fight_id,
                em.event_date,
                CASE 
                    WHEN fm.fighter1_id = :bj_penn_id THEN fm.fighter2_id
                    ELSE fm.fighter1_id
                END as historical_opponent_id
            FROM features.fight_mapping fm
            JOIN features.event_mapping em ON fm.event_id = em.event_id
            JOIN features.event_mapping current_em ON current_em.fight_id = :current_fight_id
            WHERE (fm.fighter1_id = :bj_penn_id OR fm.fighter2_id = :bj_penn_id)
              AND fm.fight_id != :current_fight_id
              AND em.event_date < current_em.event_date
            ORDER BY em.event_date
        )
        SELECT 
            bhf.fight_id,
            bhf.event_date,
            bhf.historical_opponent_id,
            {', '.join([f't.{col}' for col in stat_columns])}
        FROM bj_historical_fights bhf
        JOIN features.{table_name} t ON t.fight_id = bhf.fight_id 
                                     AND t.fighter_id = bhf.historical_opponent_id
        ORDER BY bhf.event_date
    """)
    
    try:
        result = conn.execute(history_query, {
            "bj_penn_id": bj_penn_id,
            "current_fight_id": current_fight_id,
            "opponent_id": opponent_id
        })
        
        historical_data = []
        for row in result.fetchall():
            fight_data = {
                "fight_id": row.fight_id,
                "event_date": row.event_date,
                "opponent_id": row.historical_opponent_id
            }
            for col in stat_columns:
                fight_data[col] = getattr(row, col, None)
            historical_data.append(fight_data)
            
        return historical_data
    except Exception as e:
        print(f"Error getting opponent history for {table_name}: {e}")
        return []

def get_weightclass_priors(conn, table_name, weightclass, stat_columns):
    """Get weightclass prior values for fallback scenarios"""
    priors = {}
    
    # Get weightclass means
    try:
        mean_query = text(f"""
            SELECT {', '.join([f'{col}_wc_mean' for col in stat_columns if col.endswith(('_acc', '_per_min', '_ratio', '_def'))])}
            FROM features.{table_name}_wc_mean 
            WHERE weightclass = :weightclass
        """)
        result = conn.execute(mean_query, {"weightclass": weightclass})
        row = result.fetchone()
        if row:
            for col in stat_columns:
                if col.endswith(('_acc', '_per_min', '_ratio', '_def')):
                    priors[f"{col}_wc_mean"] = getattr(row, f"{col}_wc_mean", None)
    except Exception as e:
        print(f"Could not get weightclass means for {table_name}: {e}")
    
    # Get weightclass MADs
    try:
        mad_query = text(f"""
            SELECT {', '.join([f'{col}_wc_mad' for col in stat_columns if col.endswith(('_acc', '_per_min', '_ratio', '_def'))])}
            FROM features.{table_name}_wc_mad 
            WHERE weightclass = :weightclass
        """)
        result = conn.execute(mad_query, {"weightclass": weightclass})
        row = result.fetchone()
        if row:
            for col in stat_columns:
                if col.endswith(('_acc', '_per_min', '_ratio', '_def')):
                    priors[f"{col}_wc_mad"] = getattr(row, f"{col}_wc_mad", None)
    except Exception as e:
        print(f"Could not get weightclass MADs for {table_name}: {e}")
    
    # Get minimum MADs
    try:
        min_mad_query = text(f"""
            SELECT {', '.join([f'{col}_min_mad' for col in stat_columns if col.endswith(('_acc', '_per_min', '_ratio', '_def'))])}
            FROM features.{table_name}_minimum_mad 
            WHERE weightclass = :weightclass
        """)
        result = conn.execute(min_mad_query, {"weightclass": weightclass})
        row = result.fetchone()
        if row:
            for col in stat_columns:
                if col.endswith(('_acc', '_per_min', '_ratio', '_def')):
                    priors[f"{col}_min_mad"] = getattr(row, f"{col}_min_mad", None)
    except Exception as e:
        print(f"Could not get minimum MADs for {table_name}: {e}")
    
    return priors

def calculate_adjperf_breakdown(observed, opponent_history, weightclass_priors, stat_name, K_mean=4.0, K_mad=4.0):
    """
    Break down the new reliability-weighted adjperf calculation step by step.
    Formula: adjperf = clip((observed - mu_shrunk) / mad_shrunk, -7, +7)
    """
    breakdown = {
        "observed": observed,
        "n_fights": len(opponent_history) if opponent_history else 0,
        "K_mean": K_mean,
        "K_mad": K_mad
    }
    
    if not opponent_history:
        # No history case - pure weightclass priors
        breakdown.update({
            "w_mean": 0.0,
            "w_mad": 0.0,
            "opp_mean_pers": None,
            "opp_mad_pers": None,
            "wc_mean": weightclass_priors.get(f"{stat_name}_wc_mean", 0),
            "wc_mad": weightclass_priors.get(f"{stat_name}_wc_mad", 0.001),
            "mad_floor": weightclass_priors.get(f"{stat_name}_min_mad", 0.001),
            "mu_shrunk": weightclass_priors.get(f"{stat_name}_wc_mean", 0),
            "mad_shrunk": max(weightclass_priors.get(f"{stat_name}_wc_mad", 0.001), 
                             weightclass_priors.get(f"{stat_name}_min_mad", 0.001))
        })
    else:
        # Calculate opponent personal history stats
        values = [fight[stat_name] for fight in opponent_history if fight.get(stat_name) is not None]
        if values:
            import numpy as np
            opp_mean_pers = np.mean(values)
            opp_mad_pers = np.median(np.abs(np.array(values) - np.median(values)))
        else:
            opp_mean_pers = 0
            opp_mad_pers = 0
        
        n_fights = len(values)
        w_mean = n_fights / (n_fights + K_mean)
        w_mad = n_fights / (n_fights + K_mad)
        
        wc_mean = weightclass_priors.get(f"{stat_name}_wc_mean", 0)
        wc_mad = weightclass_priors.get(f"{stat_name}_wc_mad", 0.001)
        mad_floor = weightclass_priors.get(f"{stat_name}_min_mad", 0.001)
        
        # Shrinkage
        mu_shrunk = w_mean * opp_mean_pers + (1 - w_mean) * wc_mean
        mad_shrunk = max(w_mad * opp_mad_pers + (1 - w_mad) * wc_mad, mad_floor)
        
        breakdown.update({
            "w_mean": w_mean,
            "w_mad": w_mad,
            "opp_mean_pers": opp_mean_pers,
            "opp_mad_pers": opp_mad_pers,
            "wc_mean": wc_mean,
            "wc_mad": wc_mad,
            "mad_floor": mad_floor,
            "mu_shrunk": mu_shrunk,
            "mad_shrunk": mad_shrunk
        })
    
    # Final calculation
    if breakdown["mad_shrunk"] > 0:
        raw_adjperf = (observed - breakdown["mu_shrunk"]) / breakdown["mad_shrunk"]
        clipped_adjperf = max(min(raw_adjperf, 7.0), -7.0)
        breakdown.update({
            "raw_adjperf": raw_adjperf,
            "final_adjperf": clipped_adjperf,
            "was_clipped": raw_adjperf != clipped_adjperf
        })
    else:
        breakdown.update({
            "raw_adjperf": 0,
            "final_adjperf": 0,
            "was_clipped": False
        })
    
    return breakdown

def print_career_stats_table(career_data):
    """Print career stats in a readable table format"""
    print("\n" + "="*200)
    print("BJ PENN CAREER STATS SUMMARY")
    print("="*200)
    
    # Prepare data for table - BJ Penn's stats
    table_data = []
    headers = [
        "Fight", "Date", "Opponent", "Method", "Result", 
        "KO", "KO_DecAvg", "KO_AdjPerf", "KO_DecAdjPerf", 
        "KO_PerSigStr", "KO_PerSigStr_DecAvg", "KO_PerSigStr_AdjPerf",
        "SigStrLand", "SigStr_DecAvg", "SigStr_Avg"
    ]
    
    for fight in career_data["fights"]:
        bj_stats = fight["stats"]["bj_penn"]
        
        row = [
            f"#{len(table_data)+1}",
            fight["event_date"][:10] if fight["event_date"] else "N/A",
            fight["opponent"][:15],  # Truncate long names
            fight["method"][:12] if fight["method"] else "N/A",
            "W" if fight["result"] == 1 else "L" if fight["result"] == 0 else "N/A",
            f"{bj_stats.get('ko', 0):.3f}" if bj_stats.get('ko') is not None else "N/A",
            f"{bj_stats.get('ko_dec_avg', 0):.3f}" if bj_stats.get('ko_dec_avg') is not None else "N/A", 
            f"{bj_stats.get('ko_adjperf', 0):.2f}" if bj_stats.get('ko_adjperf') is not None else "N/A",
            f"{bj_stats.get('ko_dec_adjperf', 0):.2f}" if bj_stats.get('ko_dec_adjperf') is not None else "N/A",
            f"{bj_stats.get('ko_per_sig_str_land', 0):.4f}" if bj_stats.get('ko_per_sig_str_land') is not None else "N/A",
            f"{bj_stats.get('ko_per_sig_str_land_dec_avg', 0):.4f}" if bj_stats.get('ko_per_sig_str_land_dec_avg') is not None else "N/A",
            f"{bj_stats.get('ko_per_sig_str_land_adjperf', 0):.2f}" if bj_stats.get('ko_per_sig_str_land_adjperf') is not None else "N/A",
            f"{bj_stats.get('sig_str_land', 0):.1f}" if bj_stats.get('sig_str_land') is not None else "N/A",
            f"{bj_stats.get('sig_str_land_dec_avg', 0):.1f}" if bj_stats.get('sig_str_land_dec_avg') is not None else "N/A",
            f"{bj_stats.get('sig_str_land_avg', 0):.1f}" if bj_stats.get('sig_str_land_avg') is not None else "N/A"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Also print opponent stats table
    print("\n" + "="*200)
    print("OPPONENT STATS SUMMARY")
    print("="*200)
    
    opp_table_data = []
    opp_headers = [
        "Fight", "Opponent", "Opp_KO", "Opp_KO_DecAvg", "Opp_KO_AdjPerf", 
        "Opp_KO_PerSigStr", "Opp_KO_PerSigStr_DecAvg", 
        "Opp_SigStrLand", "Opp_SigStr_DecAvg", "Opp_SigStr_Avg"
    ]
    
    for fight in career_data["fights"]:
        opp_stats = fight["stats"]["opponent"]
        
        opp_row = [
            f"#{len(opp_table_data)+1}",
            fight["opponent"][:20],
            f"{opp_stats.get('ko', 0):.3f}" if opp_stats.get('ko') is not None else "N/A",
            f"{opp_stats.get('ko_dec_avg', 0):.3f}" if opp_stats.get('ko_dec_avg') is not None else "N/A",
            f"{opp_stats.get('ko_adjperf', 0):.2f}" if opp_stats.get('ko_adjperf') is not None else "N/A",
            f"{opp_stats.get('ko_per_sig_str_land', 0):.4f}" if opp_stats.get('ko_per_sig_str_land') is not None else "N/A",
            f"{opp_stats.get('ko_per_sig_str_land_dec_avg', 0):.4f}" if opp_stats.get('ko_per_sig_str_land_dec_avg') is not None else "N/A",
            f"{opp_stats.get('sig_str_land', 0):.1f}" if opp_stats.get('sig_str_land') is not None else "N/A",
            f"{opp_stats.get('sig_str_land_dec_avg', 0):.1f}" if opp_stats.get('sig_str_land_dec_avg') is not None else "N/A",
            f"{opp_stats.get('sig_str_land_avg', 0):.1f}" if opp_stats.get('sig_str_land_avg') is not None else "N/A"
        ]
        opp_table_data.append(opp_row)
    
    print(tabulate(opp_table_data, headers=opp_headers, tablefmt="grid"))
    
    # Print summary stats
    print(f"\nSUMMARY:")
    print(f"Total Fights: {len(career_data['fights'])}")
    print(f"Fighter: {career_data['fighter']}")
    print(f"Data extracted: {career_data['extraction_date'][:19]}")
    
    # Print a few key stats across career
    if career_data["fights"]:
        first_fight = career_data["fights"][0]
        last_fight = career_data["fights"][-1]
        print(f"Career span: {first_fight['event_date'][:10]} to {last_fight['event_date'][:10]}")
        
        # Show evolution of key stats
        print(f"\nKEY STAT EVOLUTION:")
        print(f"First Fight KO Rate: {first_fight['stats']['bj_penn'].get('ko', 0):.3f}")
        print(f"Last Fight KO Rate: {last_fight['stats']['bj_penn'].get('ko', 0):.3f}")
        print(f"First Fight Sig Strikes: {first_fight['stats']['bj_penn'].get('sig_str_land', 0):.1f}")
        print(f"Last Fight Sig Strikes: {last_fight['stats']['bj_penn'].get('sig_str_land', 0):.1f}")

def print_adjperf_analysis(career_data):
    """Print detailed adjusted performance analysis for key fights"""
    print("\n" + "="*120)
    print("ADJUSTED PERFORMANCE ANALYSIS (New Reliability-Weighted Formula)")
    print("="*120)
    print("Formula: adjperf = clip((observed - mu_shrunk) / mad_shrunk, -7, +7)")
    print("Where:")
    print("  mu_shrunk = w_mean * opp_mean_pers + (1 - w_mean) * wc_mean")
    print("  mad_shrunk = max(w_mad * opp_mad_pers + (1 - w_mad) * wc_mad, mad_floor)")
    print("  w_mean = n_fights / (n_fights + K_mean=4.0)")
    print("  w_mad = n_fights / (n_fights + K_mad=4.0)")
    print("-" * 120)
    
    # Analyze a few key fights with detailed breakdown
    key_fights_to_analyze = min(5, len(career_data["fights"]))  # First 5 fights for detailed analysis
    
    for i in range(key_fights_to_analyze):
        fight = career_data["fights"][i]
        print(f"\nFIGHT #{i+1}: BJ Penn vs {fight['opponent']} ({fight['event_date'][:10]})")
        print(f"Weightclass: {fight['weightclass']}, Result: {'W' if fight.get('result') == 1 else 'L' if fight.get('result') == 0 else 'N/A'}")
        
        # Analyze sig_str_acc if available
        if 'sig_str' in fight.get('adjperf_analysis', {}):
            sig_str_analysis = fight['adjperf_analysis']['sig_str']
            if 'sig_str_acc' in sig_str_analysis:
                breakdown = sig_str_analysis['sig_str_acc']
                print(f"\n  SIG_STR_ACC Analysis:")
                print(f"    Observed BJ Penn accuracy: {breakdown['observed']:.3f}")
                print(f"    Opponent history fights: {breakdown['n_fights']}")
                
                if breakdown['n_fights'] > 0:
                    print(f"    Historical opponents achieved vs BJ Penn:")
                    opponent_history = fight['opponent_history'].get('sig_str', [])
                    for j, hist_fight in enumerate(opponent_history):
                        if hist_fight.get('sig_str_acc') is not None:
                            print(f"      Fight {j+1}: {hist_fight['sig_str_acc']:.3f}")
                    
                    print(f"    Opponent personal mean: {breakdown['opp_mean_pers']:.3f}")
                    print(f"    Opponent personal MAD: {breakdown['opp_mad_pers']:.3f}")
                    print(f"    Shrinkage weights: w_mean={breakdown['w_mean']:.3f}, w_mad={breakdown['w_mad']:.3f}")
                else:
                    print(f"    No opponent history - using weightclass priors")
                
                print(f"    Weightclass mean: {breakdown['wc_mean']:.3f}")
                print(f"    Weightclass MAD: {breakdown['wc_mad']:.3f}")
                print(f"    MAD floor: {breakdown['mad_floor']:.3f}")
                print(f"    Shrunk mean (mu): {breakdown['mu_shrunk']:.3f}")
                print(f"    Shrunk MAD (denom): {breakdown['mad_shrunk']:.3f}")
                print(f"    Raw adjperf: {breakdown['raw_adjperf']:.3f}")
                print(f"    Final adjperf: {breakdown['final_adjperf']:.3f}")
                if breakdown['was_clipped']:
                    print(f"    ⚠️  Value was clipped to [-7, +7] range!")
                
                # Compare to database value
                db_adjperf = fight['stats']['bj_penn'].get('sig_str_acc_adjperf')
                if db_adjperf is not None:
                    print(f"    Database adjperf: {db_adjperf:.3f}")
                    diff = abs(breakdown['final_adjperf'] - db_adjperf)
                    if diff > 0.001:
                        print(f"    ⚠️  MISMATCH: Manual calc differs by {diff:.3f}")
                    else:
                        print(f"    ✅ Manual calculation matches database!")
        
        print("-" * 80)
    
    # Summary statistics
    print(f"\nADJPERF SUMMARY ACROSS CAREER:")
    
    # Collect all adjperf values for analysis
    adjperf_values = []
    shrinkage_weights = []
    
    for fight in career_data["fights"]:
        if 'sig_str' in fight.get('adjperf_analysis', {}):
            if 'sig_str_acc' in fight['adjperf_analysis']['sig_str']:
                breakdown = fight['adjperf_analysis']['sig_str']['sig_str_acc']
                adjperf_values.append(breakdown['final_adjperf'])
                shrinkage_weights.append(breakdown['w_mean'])
    
    if adjperf_values:
        import numpy as np
        print(f"  Sig Str Acc Adjperf: mean={np.mean(adjperf_values):.3f}, std={np.std(adjperf_values):.3f}")
        print(f"  Adjperf range: [{np.min(adjperf_values):.3f}, {np.max(adjperf_values):.3f}]")
        print(f"  Shrinkage weights: mean={np.mean(shrinkage_weights):.3f}, range=[{np.min(shrinkage_weights):.3f}, {np.max(shrinkage_weights):.3f}]")
        
        # Check for extreme values
        extreme_values = [v for v in adjperf_values if abs(v) > 3.0]
        if extreme_values:
            print(f"  ⚠️  {len(extreme_values)} extreme adjperf values (|adjperf| > 3.0): {extreme_values}")
    
    print("="*120)

def extract_bj_penn_career_stats():
    """Main function to extract BJ Penn's career stats"""
    
    # Database connection
    engine = create_db_engine()
    
    # Stats we want to extract (focused on adjperf target stats for analysis)
    requested_stats = [
        # KO stats (verified to exist from previous run)
        'ko', 'ko_dec_avg', 'ko_avg', 'ko_adjperf', 'ko_dec_adjperf', 
        'ko_dec_adjperf_dec_avg', 'ko_adjperf_dec_avg',
        
        # KO per sig str land stats (ADJPERF TARGET - ends with _per_min pattern)
        'ko_per_sig_str_land', 'ko_per_sig_str_land_dec_avg', 'ko_per_sig_str_land_avg',
        'ko_per_sig_str_land_adjperf', 'ko_per_sig_str_land_dec_adjperf',
        'ko_per_sig_str_land_dec_adjperf_dec_avg', 'ko_per_sig_str_land_adjperf_dec_avg',
        
        # Sig str stats - including accuracy (ADJPERF TARGET)
        'sig_str_land', 'sig_str_land_dec_avg', 'sig_str_land_avg',
        'sig_str_acc', 'sig_str_acc_adjperf', 'sig_str_acc_dec_adjperf',  # ADJPERF TARGET
        'sig_str_per_min', 'sig_str_per_min_adjperf', 'sig_str_per_min_dec_adjperf',  # ADJPERF TARGET
        
        # Takedown stats - including accuracy and defense (ADJPERF TARGETS)
        'td_acc', 'td_acc_adjperf', 'td_acc_dec_adjperf',  # ADJPERF TARGET
        'td_def', 'td_def_adjperf', 'td_def_dec_adjperf'   # ADJPERF TARGET
    ]
    
    # Map stats to their respective tables
    stat_table_mapping = {
        # KO stats -> ko table
        'ko': 'ko',
        'ko_dec_avg': 'ko',
        'ko_avg': 'ko', 
        'ko_adjperf': 'ko',
        'ko_dec_adjperf': 'ko',
        'ko_dec_adjperf_dec_avg': 'ko',
        'ko_adjperf_dec_avg': 'ko',
        
        # KO per sig str land stats -> ko table
        'ko_per_sig_str_land': 'ko',
        'ko_per_sig_str_land_dec_avg': 'ko',
        'ko_per_sig_str_land_avg': 'ko',
        'ko_per_sig_str_land_adjperf': 'ko',
        'ko_per_sig_str_land_dec_adjperf': 'ko',
        'ko_per_sig_str_land_dec_adjperf_dec_avg': 'ko',
        'ko_per_sig_str_land_adjperf_dec_avg': 'ko',
        
        # Sig str stats -> sig_str table  
        'sig_str_land': 'sig_str',
        'sig_str_land_dec_avg': 'sig_str',
        'sig_str_land_avg': 'sig_str',
        'sig_str_acc': 'sig_str',
        'sig_str_acc_adjperf': 'sig_str',
        'sig_str_acc_dec_adjperf': 'sig_str',
        'sig_str_per_min': 'sig_str',
        'sig_str_per_min_adjperf': 'sig_str',
        'sig_str_per_min_dec_adjperf': 'sig_str',
        
        # Takedown stats -> td table
        'td_acc': 'td',
        'td_acc_adjperf': 'td',
        'td_acc_dec_adjperf': 'td',
        'td_def': 'td',
        'td_def_adjperf': 'td',
        'td_def_dec_adjperf': 'td'
    }
    
    career_data = {
        "fighter": "BJ Penn",
        "extraction_date": datetime.now().isoformat(),
        "_metadata": {
            "description": "BJ Penn's career stats and opponent stats for sanity checking",
            "adjperf_calculation": {
                "formula": "stat_adjperf = (fighter1_stat - fighter2_stat_opp_avg) / fighter2_stat_opp_mad",
                "explanation": {
                    "fighter1_stat": "The current fighter's stat value for this fight",
                    "fighter2_stat_opp_avg": "The opponent's historical average of this stat against their previous opponents",
                    "fighter2_stat_opp_mad": "The opponent's historical median absolute deviation of this stat against their previous opponents"
                },
                "fallback": "For first-time fighters (no previous fights), fallback values are used from weightclass averages",
                "decay_variants": {
                    "_dec_avg": "Time-decayed average with 1.5-year half-life weighting recent fights more heavily",
                    "_adjperf": "Adjusted performance using opponent's historical stats",
                    "_dec_adjperf": "Adjusted performance using time-decayed opponent stats",
                    "_dec_adjperf_dec_avg": "Time-decayed average of adjusted performance values"
                }
            },
            "smoothing_methods": {
                "poisson_gamma": {
                    "description": "Poisson-Gamma smoothing applied to most count-based stats",
                    "applies_to": "All stats except ko, sub_land, decision, and win",
                    "purpose": "Smooths count data using Bayesian inference with Poisson likelihood and Gamma prior"
                },
                "beta_binomial": {
                    "description": "Beta-Binomial smoothing applied to binary outcome stats", 
                    "applies_to": "ko, sub_land, decision, and win stats only",
                    "purpose": "Smooths binary/proportion data using Bayesian inference with Beta prior"
                }
            }
        },
        "fights": []
    }
    
    with get_db_connection(engine) as conn:
        # Find BJ Penn's fighter ID
        bj_penn_id = find_bj_penn_fighter_id(conn)
        print(f"Found BJ Penn with fighter_id: {bj_penn_id}")
        
        # Get all his fights
        fights = get_bj_penn_fights(conn, bj_penn_id)
        print(f"Found {len(fights)} fights for BJ Penn")
        
        # Process each fight
        for fight_idx, fight in enumerate(fights):
            print(f"Processing fight {fight_idx + 1}/{len(fights)}: {fight.opponent_name} ({fight.event_date})")
            
            fight_data = {
                "fight_id": fight.fight_id,
                "event_date": fight.event_date.isoformat() if fight.event_date else None,
                "event_location": fight.event_location,
                "opponent": fight.opponent_name,
                "method": fight.method,
                "result": fight.result,
                "weightclass": fight.weightclass,
                "stats": {
                    "bj_penn": {},
                    "opponent": {}
                },
                "adjperf_analysis": {},
                "opponent_history": {},
                "weightclass_priors": {}
            }
            
            # Group stats by table
            tables_to_query = {}
            for stat in requested_stats:
                table = stat_table_mapping[stat]
                if table not in tables_to_query:
                    tables_to_query[table] = []
                tables_to_query[table].append(stat)
            
            # Query each table
            for table, stats_list in tables_to_query.items():
                try:
                    stats_data = get_stats_for_fight(
                        conn, fight.fight_id, bj_penn_id, fight.opponent_id, 
                        table, stats_list
                    )
                    
                    # Add BJ Penn's stats
                    if bj_penn_id in stats_data:
                        for stat, value in stats_data[bj_penn_id].items():
                            fight_data["stats"]["bj_penn"][stat] = value
                    
                    # Add opponent's stats
                    if fight.opponent_id in stats_data:
                        for stat, value in stats_data[fight.opponent_id].items():
                            fight_data["stats"]["opponent"][stat] = value
                    
                    # Get opponent history analysis for adjperf target stats
                    adjperf_target_stats = [s for s in stats_list if s.endswith(('_acc', '_per_min', '_ratio', '_def'))]
                    if adjperf_target_stats:
                        opponent_history = get_opponent_history_analysis(
                            conn, bj_penn_id, fight.fight_id, fight.opponent_id, table, adjperf_target_stats
                        )
                        fight_data["opponent_history"][table] = opponent_history
                        
                        # Get weightclass priors for this table
                        weightclass_priors = get_weightclass_priors(conn, table, fight.weightclass, adjperf_target_stats)
                        fight_data["weightclass_priors"][table] = weightclass_priors
                        
                        # Calculate adjperf breakdown for each target stat
                        fight_data["adjperf_analysis"][table] = {}
                        for stat in adjperf_target_stats:
                            if bj_penn_id in stats_data and stats_data[bj_penn_id].get(stat) is not None:
                                observed_value = stats_data[bj_penn_id][stat]
                                breakdown = calculate_adjperf_breakdown(
                                    observed_value, opponent_history, weightclass_priors, stat
                                )
                                fight_data["adjperf_analysis"][table][stat] = breakdown
                            
                except Exception as e:
                    print(f"Error querying {table} for fight {fight.fight_id}: {e}")
                    continue
            
            career_data["fights"].append(fight_data)
    
    return career_data

def main():
    """Main execution function"""
    try:
        print("Extracting BJ Penn's career stats...")
        career_stats = extract_bj_penn_career_stats()
        
        # Save to JSON file
        output_file = os.path.join(os.path.dirname(__file__), 'bj_penn_career_stats.json')
        with open(output_file, 'w') as f:
            json.dump(career_stats, f, indent=2, default=str)
        
        print(f"Career stats saved to: {output_file}")
        print(f"Total fights processed: {len(career_stats['fights'])}")
        
        # Print data in table format for human readability
        print_career_stats_table(career_stats)
        
        # Print detailed adjperf analysis
        print_adjperf_analysis(career_stats)
        
        return career_stats
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
