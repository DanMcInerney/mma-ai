import pandas as pd
import argparse
import os
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from libs.feature_store.core import CoreFeatureStore
# from libs.feature_store.stats import StatsFeatureStore
from sqlalchemy import text
from typing import Set
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from libs.feature_store.schema import initialize_schema
from libs.feature_store.schema import create_feature_specific_tables
from libs.feature_store.schema import initialize_new_schema
from libs.feature_store.schema import create_table
from libs.feature_store.calculators.time_sec_calc import TimeSecCalculator
from libs.feature_store.calculators.adj_perf_calc import AdjustedPerformanceCalculator
from libs.feature_store.calculators.ko_calc import KOCalculator
from libs.feature_store.calculators.sub_land_calc import SubmissionslandCalculator
from libs.feature_store.calculators.full_fight_stats import FullFightStatsCalculator
from libs.feature_store.calculators.win_calc import WinCalculator
from libs.feature_store.calculators.opp_calc import OpponentCalculator
from libs.feature_store.calculators.total_calc import TotalCalculator
from libs.feature_store.calculators.acc_calc import AccuracyCalculator
from libs.feature_store.calculators.def_calc import DefenseCalculator
from libs.feature_store.calculators.per_min_calc import PerMinCalculator
from libs.feature_store.calculators.pressure_calc import PressureCalculator
from libs.feature_store.calculators.per_calc import PerCalculator
from libs.feature_store.calculators.power_calc import KDPowerCalculator
from libs.feature_store.calculators.sdev_calc import StandardDeviationCalculator
from libs.feature_store.calculators.ratio_calc import RatioCalculator
from libs.feature_store.calculators.dslf_calc import DaysSinceLastFightCalculator
from libs.feature_store.calculators.decision_calc import DecisionCalculator
from libs.feature_store.calculators.ape_calc import ApeCalculator
from libs.feature_store.calculators.reach_calc import ReachCalculator
from libs.feature_store.calculators.height_calc import HeightCalculator
from libs.feature_store.calculators.age_calc import AgeCalculator
from libs.feature_store.calculators.avg_calc import AverageCalculator
from libs.feature_store.calculators.time_dec_avg_calc import TimedecAvgCalculator
from libs.feature_store.calculators.adj_perf_calc import AdjustedPerformanceCalculator
from libs.feature_store.calculators.odds_calc import OddsCalculator
from libs.feature_store.clean_training_data import CleanTrainingData
from libs.feature_store.calculators.time_dec_sdev_calc import TimedecStdDevCalculator
from libs.feature_store.create_training_data import CreateTrainingData
from libs.feature_store.calculators.time_dec_slope_calc import TimedecSlopeCalculator
from libs.feature_store.calculators.first_time_fighters_mad_calc import FirstTimeMadCalculator
from libs.feature_store.calculators.first_time_fighters_avg_calc import FirstTimeOpponentAverageCalculator
from libs.feature_store.calculators.minimum_mad_calc import MinimumMadCalculator
from libs.feature_store.calculators.ufc_age_calc import UfcAgeCalculator
from libs.feature_store.calculators.poisson_gamma_smoothing_calc import PoissonGammaCalculator
from libs.feature_store.calculators.beta_binomial_calc import BetaBinomialCalculator
from libs.feature_store.calculators.style_calc import StyleCalculator
from libs.parameter_optimization import get_default_parameter_loader
from libs.feature_store.calculators.custom_total_calc import CustomTotalCalculator
from libs.feature_store.fighter_balance import FighterBalancer
from libs.feature_store.calculators.weightclass_mean_calc import WeightclassMeanCalculator
from libs.feature_store.calculators.weightclass_mad_calc import WeightclassMadCalculator
from libs.feature_store.features import TEST_FEATS_NO_DIFF
#from libs.feature_store.feature_utils import FEATS, REQ_FEATS
from libs.odds import OddsAPI
from fuzzywuzzy import fuzz
from libs.feature_store.calculator_context import CalculatorContext
import time
from libs.feature_store.calculators.mad_calc import MedianAbsoluteDeviationCalculator
from libs.feature_store.calculators.time_dec_mad_calc import TimedecMadCalculator
from datetime import datetime
from pathlib import Path
# Import the BFO Scraper
from libs.bfo_scraper import BFOScraper
from libs.paths import data_dir, database_url, raw_ufcstats_dir

def copy_to_derived(conn):
    print("Copying data to fight_stats_derived...")
    conn.execute(text('''
        INSERT INTO features.fight_stats_derived (
            fight_id, fighter_id, event_id,
            -- Round 1 stats
            kd_rd1, sig_str_land_rd1, sig_str_att_rd1,
            strikes_land_rd1, strikes_att_rd1,
            td_land_rd1, td_att_rd1,
            sub_att_rd1, rev_rd1, ctrl_rd1,
            -- Strike locations round 1
            head_land_rd1, head_att_rd1,
            body_land_rd1, body_att_rd1,
            leg_land_rd1, leg_att_rd1,
            distance_land_rd1, distance_att_rd1,
            clinch_land_rd1, clinch_att_rd1,
            ground_land_rd1, ground_att_rd1,
            win_rd1,
            -- Derived features
            time_sec, time_sec_rd1,
            ko, ko_rd1,
            sub_land, sub_land_rd1,
            decision,
            win,
            -- Total stats
            kd,
            sig_str_land, sig_str_att,
            strikes_land, strikes_att,
            td_land, td_att,
            sub_att, rev, ctrl,
            head_land, head_att,
            body_land, body_att,
            leg_land, leg_att,
            distance_land, distance_att,
            clinch_land, clinch_att,
            ground_land, ground_att,
            -- Static derived features
            age, days_since_last_fight, reach, ape, ufcage
        )
        SELECT 
            fight_id, fighter_id, event_id,
            -- Round 1 stats
            kd_rd1, sig_str_land_rd1, sig_str_att_rd1,
            strikes_land_rd1, strikes_att_rd1,
            td_land_rd1, td_att_rd1,
            sub_att_rd1, rev_rd1, ctrl_rd1,
            -- Strike locations round 1
            head_land_rd1, head_att_rd1,
            body_land_rd1, body_att_rd1,
            leg_land_rd1, leg_att_rd1,
            distance_land_rd1, distance_att_rd1,
            clinch_land_rd1, clinch_att_rd1,
            ground_land_rd1, ground_att_rd1,
            win_rd1,
            -- Derived features
            time_sec, time_sec_rd1,
            ko, ko_rd1,
            sub_land, sub_land_rd1,
            decision,
            win,
            -- Total stats
            kd,
            sig_str_land, sig_str_att,
            strikes_land, strikes_att,
            td_land, td_att,
            sub_att, rev, ctrl,
            head_land, head_att,
            body_land, body_att,
            leg_land, leg_att,
            distance_land, distance_att,
            clinch_land, clinch_att,
            ground_land, ground_att,
            -- Static derived features
            age, days_since_last_fight, reach, ape, ufcage
        FROM features.fight_stats_fe;
    '''))
    conn.commit()
    print("Data copy complete!")

def create_db_engine(db_url=None):
    db_url = db_url or database_url()
    if not database_exists(db_url):
        create_database(db_url)
    
    # Configure the engine with a connection pool
    return create_engine(
        db_url,
        pool_size=5,  # Maximum number of permanent connections
        max_overflow=10,  # Maximum number of additional connections
        pool_timeout=30,  # Timeout waiting for a connection (seconds)
        pool_recycle=14400  # Recycle connections after 4h
    )


def reset_database(engine):
    """Drop generated schemas so the pipeline can rebuild from raw CSVs."""
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA IF EXISTS features CASCADE"))
        conn.execute(text("DROP SCHEMA IF EXISTS model_data CASCADE"))

@contextmanager
def get_db_connection(engine):
    """Context manager for database connections"""
    connection = engine.connect()
    try:
        yield connection
    finally:
        connection.close()

def populate_feature_tables(conn, feature_groups):
    print("Copying stats to feature-specific tables...")
    
    for feature, columns in feature_groups.items():
        # Skip feature groups with no columns (e.g., decision_rd1 which was removed)
        if not columns:
            print(f"Skipping {feature} - no columns to populate")
            continue
            
        # Create the column list for the INSERT statement
        column_list = ', '.join(col for col, _ in columns)
        
        # Insert data
        insert_sql = f"""
            INSERT INTO features.{feature} (
                fight_id, fighter_id, event_id,
                {column_list}
            )
            SELECT 
                fight_id, fighter_id, event_id,
                {column_list}
            FROM features.fight_stats_derived;
        """
        conn.execute(text(insert_sql))
        conn.commit()
        print(f"Copied {feature} stats...")

    print("Feature table population complete!")

# def update_combined_odds(conn):
#     """Add new historical odds to combined_odds.csvd"""
#     odds_api = OddsAPI(
#         conn=conn,
#         combined_odds_path="combined_odds.csv",
#         initial_date="2025-02-11T00:00:00Z" # last run: 2025-02-11
#     )
#     odds_api.update_historical_odds()

def fetch_fights_that_day(conn, event_date):
    """
    Return a list of dicts for fights matching the event_date +/- 1 day.
    [
      {
        'fight_id': 123,
        'fighter1_id': 11,
        'fighter2_id': 12,
        'fighter1_name': 'Nate Landwehr',
        'fighter2_name': 'Dooho Choi'
      },
      ...
    ]
    """
    query = text('''
        SELECT f.fight_id,
               f.fighter1_id,
               f.fighter2_id,
               fm1.fighter_name as fighter1_name,
               fm2.fighter_name as fighter2_name
        FROM features.fight_mapping f
        JOIN features.fighter_mapping fm1 ON f.fighter1_id = fm1.fighter_id
        JOIN features.fighter_mapping fm2 ON f.fighter2_id = fm2.fighter_id
        JOIN features.event_mapping e   ON f.event_id = e.event_id
        WHERE e.event_date BETWEEN :date - INTERVAL '1 day' AND :date + INTERVAL '1 day'
    ''')

    results = []
    rows = conn.execute(query, {"date": event_date}).fetchall()

    for r in rows:
        results.append({
            "fight_id":       r[0],
            "fighter1_id":    r[1],
            "fighter2_id":    r[2],
            "fighter1_name":  r[3],
            "fighter2_name":  r[4]
        })

    return results

def match_fights_by_fuzzy(fights_that_day, csv_f1, csv_f2, threshold=70):
    """
    Return a list of matches with scenario info:
    [
      {
        "fight_id": ...,
        "fighter1_id": ...,
        "fighter2_id": ...,
        "scenario": 1 or 2
      },
      ...
    ]
    """
    matched = []
    found_match = False
    
    for fight in fights_that_day:
        db_f1 = fight['fighter1_name']
        db_f2 = fight['fighter2_name']

        ratio_11 = fuzz.ratio(csv_f1.lower(), db_f1.lower())
        ratio_22 = fuzz.ratio(csv_f2.lower(), db_f2.lower())
        scenario1_score = ratio_11 + ratio_22

        ratio_12 = fuzz.ratio(csv_f1.lower(), db_f2.lower())
        ratio_21 = fuzz.ratio(csv_f2.lower(), db_f1.lower())
        scenario2_score = ratio_12 + ratio_21

        # If neither scenario is above (2 * threshold), skip
        best_scenario = None
        if scenario1_score >= 2 * threshold or scenario2_score >= 2 * threshold:
            found_match = True
            best_scenario = 1 if scenario1_score >= scenario2_score else 2
            matched.append({
                "fight_id": fight['fight_id'],
                "fighter1_id": fight['fighter1_id'],
                "fighter2_id": fight['fighter2_id'],
                'db_f1': fight['fighter1_name'],
                'db_f2': fight['fighter2_name'],
                "scenario": best_scenario
            })
    
    if not found_match:
        print(f"No match found for fighters: {csv_f1} vs {csv_f2}")
    
    return matched

def replace_with_smoothed_columns(conn):
    """
    Replace original columns with their smoothed versions (_smooth suffix)
    by renaming the smoothed columns and dropping the original ones.
    """
    print("Replacing original columns with smoothed versions...")
    
    # 1. First get a list of all columns with _smooth suffix
    query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'features' 
    AND table_name = 'fight_stats_derived' 
    AND column_name LIKE '%_smooth'
    """
    result = conn.execute(text(query)).fetchall()
    smooth_columns = [row[0] for row in result]
    
    if not smooth_columns:
        print("No smoothed columns found. Skipping replacement.")
        return
    
    print(f"Found {len(smooth_columns)} smoothed columns to replace originals.")
    
    # 2. For each smoothed column, drop the original and rename the smoothed
    for smooth_col in smooth_columns:
        original_col = smooth_col.replace('_smooth', '')
        
        # First rename smoothed column to a temporary name to avoid conflict
        temp_col = f"{original_col}_temp_{int(time.time())}"
        rename_query = f"""
        ALTER TABLE features.fight_stats_derived 
        RENAME COLUMN {smooth_col} TO {temp_col}
        """
        conn.execute(text(rename_query))
        
        # Drop the original column
        drop_query = f"""
        ALTER TABLE features.fight_stats_derived 
        DROP COLUMN {original_col}
        """
        conn.execute(text(drop_query))
        
        # Rename temp column to the original name
        final_rename_query = f"""
        ALTER TABLE features.fight_stats_derived 
        RENAME COLUMN {temp_col} TO {original_col}
        """
        conn.execute(text(final_rename_query))
        
        print(f"Replaced column: {original_col}")
    
    conn.commit()
    print("Column replacement complete.")

def rename_smoothed_columns(conn):
    """
    Rename original stat columns to <stat>_raw and _smooth columns to <stat>.
    This preserves the original values while making smoothed values the default.
    """
    print("Renaming original columns to _raw and _smooth columns to original names...")
    
    # 1. Get all columns with _smooth suffix
    query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'features' 
    AND table_name = 'fight_stats_derived' 
    AND column_name LIKE '%_smooth'
    """
    result = conn.execute(text(query)).fetchall()
    smooth_columns = [row[0] for row in result]
    
    if not smooth_columns:
        print("No smoothed columns found. Skipping renaming.")
        return
    
    print(f"Found {len(smooth_columns)} smoothed columns to rename.")
    
    # 2. For each smoothed column, rename original to _raw and smoothed to original
    for smooth_col in smooth_columns:
        original_col = smooth_col.replace('_smooth', '')
        raw_col = f"{original_col}_raw"
        
        # Check if original column exists
        check_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'features' 
        AND table_name = 'fight_stats_derived' 
        AND column_name = :column_name
        """
        original_exists = conn.execute(text(check_query), {"column_name": original_col}).fetchone()
        
        if original_exists:
            # Rename original column to _raw
            rename_original_query = f"""
            ALTER TABLE features.fight_stats_derived 
            RENAME COLUMN {original_col} TO {raw_col}
            """
            conn.execute(text(rename_original_query))
            print(f"Renamed {original_col} to {raw_col}")
        
        # Rename _smooth column to original name
        rename_smooth_query = f"""
        ALTER TABLE features.fight_stats_derived 
        RENAME COLUMN {smooth_col} TO {original_col}
        """
        conn.execute(text(rename_smooth_query))
        print(f"Renamed {smooth_col} to {original_col}")
    
    conn.commit()
    print("Column renaming complete.")

def delete_raw_columns(conn):
    """
    Delete all _raw columns from fight_stats_derived table to save space.
    These are the original unsmoothed values that are no longer needed.
    """
    print("Deleting _raw columns to save space...")
    
    # Get all columns with _raw suffix
    query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'features' 
    AND table_name = 'fight_stats_derived' 
    AND column_name LIKE '%_raw'
    """
    result = conn.execute(text(query)).fetchall()
    raw_columns = [row[0] for row in result]
    
    if not raw_columns:
        print("No _raw columns found. Skipping deletion.")
        return
    
    print(f"Found {len(raw_columns)} _raw columns to delete.")
    
    # Drop each _raw column
    for raw_col in raw_columns:
        drop_query = f"""
        ALTER TABLE features.fight_stats_derived 
        DROP COLUMN {raw_col}
        """
        conn.execute(text(drop_query))
        print(f"Deleted column: {raw_col}")
    
    conn.commit()
    print("_raw column deletion complete.")

def merge_raw_fighter_odds(df, conn):
    """Fetches raw odds from features.odds and merges them into the DataFrame for both fighters."""
    print("\nFetching raw opening/closing/sevenday odds (ip and vigless) for both fighters from features.odds...")
    odds_query = """
    SELECT
        fight_id,
        fighter_id,
        ip_opening_odds,         -- Keep this
        ip_closing_odds,         -- Keep this
        sevenday_ip_opening_odds, -- Keep this
        sevenday_vigless_ip_opening_odds -- Add this
    FROM features.odds;
    """
    try:
        odds_df = pd.read_sql(text(odds_query), conn)

        # Explicitly drop existing odds columns from df if they exist
        existing_odds_cols = [col for col in df.columns if 'odds' in col.lower() and ('f1_' in col or 'f2_' in col)]
        if existing_odds_cols:
            print(f"Dropping existing f1/f2 odds columns before merging raw odds: {existing_odds_cols}")
            df = df.drop(columns=existing_odds_cols)
            
        # Prepare odds for fighter 1 merge
        odds_f1 = odds_df.rename(columns={
            'fighter_id': 'fighter1_id',
            'ip_opening_odds': 'f1_ip_opening_odds',
            'ip_closing_odds': 'f1_ip_closing_odds',
            'sevenday_ip_opening_odds': 'f1_sevenday_ip_opening_odds',
            'sevenday_vigless_ip_opening_odds': 'f1_sevenday_vigless_ip_opening_odds' # Add rename
        })

        # Prepare odds for fighter 2 merge
        odds_f2 = odds_df.rename(columns={
            'fighter_id': 'fighter2_id',
            'ip_opening_odds': 'f2_ip_opening_odds',
            'ip_closing_odds': 'f2_ip_closing_odds',
            'sevenday_ip_opening_odds': 'f2_sevenday_ip_opening_odds',
            'sevenday_vigless_ip_opening_odds': 'f2_sevenday_vigless_ip_opening_odds' # Add rename
        })

        # Make a copy of the dataframe to avoid modifying the original
        df_merged = df.copy()

        # Specify columns for fighter 1 merge
        f1_merge_cols = ['fight_id', 'fighter1_id', 
                         'f1_ip_opening_odds', 'f1_ip_closing_odds', 
                         'f1_sevenday_ip_opening_odds', 
                         'f1_sevenday_vigless_ip_opening_odds'] # Add new col
        
        # Merge odds for fighter 1
        df_merged = pd.merge(
            df_merged,
            odds_f1[f1_merge_cols], # Use the specified columns
            on=['fight_id', 'fighter1_id'],
            how='left'
        )

        # Specify columns for fighter 2 merge
        f2_merge_cols = ['fight_id', 'fighter2_id', 
                         'f2_ip_opening_odds', 'f2_ip_closing_odds', 
                         'f2_sevenday_ip_opening_odds',
                         'f2_sevenday_vigless_ip_opening_odds'] # Add new col

        # Merge odds for fighter 2
        df_merged = pd.merge(
            df_merged,
            odds_f2[f2_merge_cols], # Use the specified columns
            on=['fight_id', 'fighter2_id'],
            how='left'
        )

        print(f"Merged raw odds (opening, closing, sevenday ip/vigless) for both fighters. New shape: {df_merged.shape}")

        # Check for NaNs introduced by the merges
        new_odds_cols = [
            'f1_ip_opening_odds', 'f1_ip_closing_odds', 'f1_sevenday_ip_opening_odds', 'f1_sevenday_vigless_ip_opening_odds', # Add new col
            'f2_ip_opening_odds', 'f2_ip_closing_odds', 'f2_sevenday_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds'  # Add new col
        ]
        nan_check = df_merged[new_odds_cols].isnull().sum()
        if nan_check.sum() > 0:
            print(f"Warning: NaNs found in new odds columns after merge: {nan_check[nan_check > 0]}")

        return df_merged

    except Exception as e:
        print(f"Error fetching or merging raw odds: {e}")
        print("Proceeding without merging raw odds.")
        return df # Return the original DataFrame if merge fails


def refresh_odds_features(conn, enabled=False, refresh_bfo=None):
    """Calculate odds features, optionally refreshing BestFightOdds first."""
    refresh_bfo = enabled if refresh_bfo is None else refresh_bfo
    calculate_features = enabled or refresh_bfo
    if not calculate_features:
        print("\nSkipping BFO odds refresh and odds feature calculation.")
        return {"enabled": False, "refresh_bfo": False, "records_scraped": None, "calculated": False}

    records_scraped = None
    if refresh_bfo:
        print("\nStarting BFO Odds Scraping...")
        bfo_scraper = BFOScraper(conn)
        records_scraped = bfo_scraper.scrape_all_fighters()
        print(f"Finished BFO Odds Scraping. Records saved: {records_scraped}\n")
    else:
        print("\nSkipping BFO odds web refresh; calculating features from the configured odds database.")

    print("Calculating odds features...")
    OddsCalculator(conn).run()
    print("Finished calculating odds features.")
    return {"enabled": True, "refresh_bfo": bool(refresh_bfo), "records_scraped": records_scraped, "calculated": True}


def main(odds=False, odds_features=False, db_url=None, raw_data_dir=None, output_data_dir=None, scrape=False, reset_db=False):
    from libs.feature_store.config import DECAY_HALF_LIFE_YEARS
    decay_rate_years = DECAY_HALF_LIFE_YEARS
    raw_data_dir = Path(raw_data_dir or raw_ufcstats_dir()).expanduser().resolve()
    output_data_dir = Path(output_data_dir or data_dir()).expanduser().resolve()
    output_data_dir.mkdir(parents=True, exist_ok=True)

    if scrape:
        from scripts.scrape_ufcstats import scrape_ufcstats

        counts = scrape_ufcstats(output_dir=raw_data_dir)
        print(f"UFCStats scrape complete: {counts}")

    competitions_path = raw_data_dir / "competitions.csv"
    individuals_path = raw_data_dir / "individuals.csv"
    missing_paths = [path for path in (competitions_path, individuals_path) if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Missing raw UFCStats CSVs: {missing_display}. "
            "Run `uv run python scripts/scrape_ufcstats.py` or pass --scrape."
        )

    # Initialize database engine once
    engine = create_db_engine(db_url)

    if reset_db:
        print("Resetting generated database schemas...")
        reset_database(engine)
    
    # Use a single connection for all operations
    with get_db_connection(engine) as conn:
        # Initialize schema
        initialize_schema(conn)

    with get_db_connection(engine) as conn:
        # Initialize core feature store with the connection
        core_store = CoreFeatureStore(conn)
        
        # Load data
        competitions_df = pd.read_csv(competitions_path)
        individuals_df = pd.read_csv(individuals_path)

        # Filter women's fights
        competitions_df = competitions_df[~competitions_df['weightclass'].str.lower().str.contains('women', na=False)]
        print(f"Fights after removing women's fights: {len(competitions_df)}")
        competitions_df['date'] = pd.to_datetime(competitions_df['event_date'])

        # Remove fights from before May 15, 1998 which was the end of the same day tournaments
        competitions_df = competitions_df[competitions_df['date'] > pd.to_datetime('1998-05-15')]
        # Also remove UFC 23 which was the last same day tournament
        competitions_df = competitions_df[competitions_df['date'] != pd.to_datetime('1999-11-19')]
        # Remove fights from before 2006-01-01
        #competitions_df = competitions_df[competitions_df['date'] > pd.to_datetime('2006-01-01')]

        competitions_df.drop(columns=['date'], inplace=True)
        competitions_df.reset_index(drop=True, inplace=True)

        # Check for new fights by comparing length of fight_mapping
        fight_mapping_count = pd.read_sql("""
            SELECT COUNT(*) as count 
            FROM features.fight_mapping
        """, conn).iloc[0]['count']

        new_fights = len(competitions_df) - (fight_mapping_count + 1)
        # if new_fights > 1000: # account for the duplicate in df not in db
        if fight_mapping_count == 0: # This fixes the issue where the first time the script is run, the fight_mapping table is empty
            print(f"New fights detected: {new_fights}")

            # Load features using the same connection
            print("Loading fighter features...")
            core_store.load_fighter_features(individuals_df)
            
            print("Loading fight features...")
            core_store.load_fight_features(competitions_df)

            print("Loading fight stats...")
            core_store.load_fight_stats(competitions_df)

            print("\nCreating fight_stats_fe...")
            conn.execute(text('''
                INSERT INTO features.fight_stats_fe 
                SELECT * FROM features.fight_stats_core;
            '''))
            conn.commit()
    
            print("Starting feature engineering...")
            # Base stats
            print("Calculating time sec...")
            # Create a CalculatorContext and pass it to TimeSecCalculator
            context = CalculatorContext(conn, schema='features')
            TimeSecCalculator(context).run()
            
            print("Calculating ko...")
            KOCalculator(context).run()
            print("Calculating decision...")
            DecisionCalculator(context).run()
            print("Calculating sub land...")
            SubmissionslandCalculator(context).run()
            print("Calculating win...")
            WinCalculator(context).run()
            print("Calculating full fight stats...")
            FullFightStatsCalculator(context).run()
            print("Calculating age...")
            AgeCalculator(context).run()
            print("Calculating days since last fight...")
            DaysSinceLastFightCalculator(context).run()
            print("Calculating reach...")
            ReachCalculator(context).run()
            print("Calculating height...")
            HeightCalculator(context).run()
            print("Calculating ape...")
            ApeCalculator(context).run()
            print("Calculating ufc age...")
            UfcAgeCalculator(context).run()

            # Copy to derived
            copy_to_derived(conn)

            # === PARAMETER OPTIMIZATION ===
            # Check if optimized parameters exist
            optimized_params_path = Path(__file__).parent / 'config' / 'optimized_parameters.json'

            if not optimized_params_path.exists():
                print("\n⚙️  No optimized parameters found - running tau optimization...")
                print("This may take 30-60 minutes...")
                from tuning.comprehensive_likelihood_tuner import main as run_tau_optimizer
                run_tau_optimizer()
                print("✓ Tau optimization complete!")
            else:
                print(f"\n✓ Using existing optimized parameters from {optimized_params_path}")

            # Initialize parameter loader
            param_loader = get_default_parameter_loader()

            # === SMOOTHING ===
            # Apply Beta-Binomial smoothing to binary outcomes like sub_land, win, decision, ko, must run before Poisson-Gamma smoothing so we use raw sub_att
            print("Calculating Beta-Binomial smoothing...")
            BetaBinomialCalculator(conn, param_loader=param_loader).run()

            # Apply Poisson-Gamma smoothing to count stats like sig_str_land, td_land, sub_att etc.
            print("Calculating Poisson-Gamma smoothing...")
            PoissonGammaCalculator(conn, param_loader=param_loader).run()

            # Rename original columns to _raw and smoothed columns to original names
            rename_smoothed_columns(conn)

            # Derived stats
            print("Calculating total...")
            TotalCalculator(conn).run() # ignore _raw
            print("Calculating accuracy...")
            AccuracyCalculator(conn).run()
            print("Calculating defense...")
            DefenseCalculator(conn).run()       

            # Delete _raw columns
            delete_raw_columns(conn)

            print("Calculating per min...")
            PerMinCalculator(conn).run() 
            print("Calculating ratio...")
            RatioCalculator(conn).run()

            print("Calculating pressure...") # only runs on: sig_str_land_pressure = sig_str_land_rd1 / sig_str_land
            PressureCalculator(conn).run() # ignore _raw

            # Populate feature tables
            feature_groups = create_feature_specific_tables(conn)
            populate_feature_tables(conn, feature_groups)

            # Calculate per features (after feature-specific tables are created)
            print("Calculating per features...")
            PerCalculator(conn).run()

            # Make sure this step is AFTER ratio, BEFORE sdev and timedecay. 
            print("Calculating opponent stats...")
            OpponentCalculator(conn).run()

            # Calculate totals for specific columns that contain "_opp" 
            # print("Calculating custom total stats for _opp columns...")
            # custom_columns = ['sig_str_land_opp', 'ko_opp', 'sub_land_opp', 'sub_att_opp', 'kd_opp', 'decision_opp']
            # CustomTotalCalculator(conn, custom_columns=custom_columns).run()

            # Calculate weightclass means
            print("Calculating weightclass means...")
            WeightclassMeanCalculator(conn).run()

            # Calculate weightclass MAD
            print("Calculating weightclass MAD...")
            WeightclassMadCalculator(conn).run()

            print("Calculating first time MAD...")
            exclude_patterns = set(['_mad']) 
            FirstTimeMadCalculator(conn, exclude_patterns=exclude_patterns).run()

            # Median absolute deviations
            print("Calculating MAD...")
            exclude_patterns = set(['_mad']) 
            MedianAbsoluteDeviationCalculator(conn, exclude_patterns=exclude_patterns).run()

            # Time-decayed MAD calculator
            # Uses centralized config (default: 1.0 year half-life)
            # print("Calculating dec_mad...")
            # exclude_patterns = set(['_mad'])
            # TimedecMadCalculator(conn, decay_rate_years=decay_rate_years, exclude_patterns=exclude_patterns).run()

            # Averages calculator
            print("Calculating avg...")
            exclude_patterns = set(['_avg', '_mad'])  # Updated to exclude _mad instead of _sdev
            AverageCalculator(conn, exclude_patterns=exclude_patterns).run()

            # Time-dec averages
            exclude_patterns = set(['_avg', '_mad'])  # Updated to exclude _mad instead of _sdev
            print("Calculating dec_avg...") 
            TimedecAvgCalculator(conn, decay_rate_years, exclude_patterns=exclude_patterns).run()

            # First time averages for adjperf in case fighter2 has no previous fight
            # print("Calculating first time avg...")
            # exclude_patterns = set() # we already exclude _opp_avg in the calc
            # FirstTimeOpponentAverageCalculator(conn, exclude_patterns=exclude_patterns).run()

            # Minimum mad for adjperf
            print("Calculating minimum mad...")
            exclude_patterns = set(['_avg']) # we only need mad for denominator of adjperf so no avg
            MinimumMadCalculator(conn, decay=False, exclude_patterns=exclude_patterns).run()

            # # Minimum mad for decayed adjperf
            # print("Calculating minimum dec_mad...")
            # exclude_patterns = set(['_avg']) # we only need mad for denominator of adjperf so no avg
            # MinimumMadCalculator(conn, decay=True, exclude_patterns=exclude_patterns).run()

            print("Calculating adj perf...")
            exclude_patterns = set(['_adjperf'])
            AdjustedPerformanceCalculator(conn, decay=False, exclude_patterns=exclude_patterns).run()

            print("Calculating dec adj perf...")
            exclude_patterns = set(['_adjperf'])
            # Columns are specified in calc()._is_adjperf_target()
            AdjustedPerformanceCalculator(conn, decay=True, exclude_patterns=exclude_patterns).run()

            # print("Calculating _adjperf_opp...")
            # include_patterns = set(['_adjperf'])
            # OpponentCalculator(conn, include_patterns=include_patterns).run()
            
            print("Calculating dec_avg...")
            include_patterns = set(['_adjperf'])                          
            exclude_patterns = set(['_avg']) # not necessary but whatevr
            TimedecAvgCalculator(conn, decay_rate_years, include_patterns=include_patterns, exclude_patterns=exclude_patterns).run()

            print("Calculating _avg...")
            include_patterns = {
                'land_adjperf', 'att_adjperf', 'ratio_adjperf', 'acc_adjperf', 'def_adjperf', 'per_min_adjperf', 'dec_adjperf', 'total_adjperf',
                # Round 1 patterns
                '_rd1_adjperf', 'rd1_ratio_adjperf', 'rd1_acc_adjperf', 'rd1_def_adjperf', 'rd1_per_min_adjperf',
                # Per-features from PerCalculator (ko_per_sig_str_land, td_per_sig_str_att, etc.)
                '_per_'
                }
            exclude_patterns = set(['_dec_avg'])
            AverageCalculator(conn, exclude_patterns=exclude_patterns, include_patterns=include_patterns).run()
######################
        # Style calculator
        #print("Calculating style...")
        #StyleCalculator(conn).run()

        refresh_odds_features(conn, enabled=odds_features or odds, refresh_bfo=odds)

        # print("Calculating slope...")
        # TimedecSlopeCalculator(conn, decay_rate_years).run()  # Uses centralized config

        # Update odds
        # if odds:
        #     update_combined_odds(conn)

        # Test features slim
        # include_patterns = set(['None'])
        # exclude_patterns = set()
        # feats = [f.replace('_diff', '') for f in FEATS]
        # required_features = set(feats)

        # Test features
        # include_patterns = set(['age', 'reach', 'days_since_last_fight', 'ufcage'])
        # exclude_patterns = set()
        # required_features = set(TEST_FEATS_NO_DIFF)

        # opp_dec_avg
        #include_patterns = set(['_opp_dec_avg', 'age', 'reach', 'days_since_last_fight'])
        # required_features = set()
        # exclude_patterns = set()

        # All test features including style
        include_patterns = set(['dec_avg', 'age', 'reach', 'ufcage', 'odds','days_since_last_fight', 'time_sec', 'weightclass_encoded', 'sig_str_land_total'])
        required_features = set()
        exclude_patterns = set()

        # dec_adjperf
        #include_patterns = set(['dec_adjperf_dec_avg', 'opp_dec_avg', 'age', 'reach', 'ufcage', 'days_since_last_fight'])
        #required_features = set()
        #exclude_patterns = set()
        
        # For final model
        #required_features = set(REQ_FEATS)
        #required_features = set()
        ctd = CreateTrainingData(conn, include_patterns=include_patterns, exclude_patterns=exclude_patterns, required_features=required_features)
        training_df = ctd.create_training_data()

        # Save training data to csv for use in prediction
        output_path = output_data_dir / 'prediction_data.csv'
        training_df.to_csv(output_path, index=False)
        print(f"\nSaved prediction DataFrame to {output_path}")
        training_df.drop(columns=['fighter_dob'], inplace=True) # Drop this because we only need it for prediction

        #include_patterns = set(REQ_FEATS)
        #include_patterns = set(['dec_adjperf_dec_avg'])
        #include_patterns = set(['_opp_dec_avg'])

        exclude_patterns = set()
        # Update to match the new CleanTrainingData constructor signature
        clean_td = CleanTrainingData(df=training_df, include_patterns=include_patterns, exclude_patterns=exclude_patterns)
        final_df = clean_td.clean_training_data()
        correlations = clean_td.correlations

        # Add raw odds from features.odds for fighter1
        final_df = merge_raw_fighter_odds(final_df, conn)

        # Save the final DataFrame to CSV
        output_path = output_data_dir / 'training_data.csv'
        final_df.to_csv(output_path, index=False)
        print(f"\nSaved training DataFrame to {output_path}")
        
        # Create decision-targeted training data (decision vs no decision, side-by-side columns)
        print("\n" + "="*80)
        print("Creating decision-targeted training data...")
        print("="*80)
        clean_td_dec = CleanTrainingData(
            df=training_df, 
            include_patterns=include_patterns, 
            exclude_patterns=exclude_patterns,
            target_type='decision'
        )
        final_df_dec = clean_td_dec.clean_training_data()
        correlations_dec = clean_td_dec.correlations
        
        # Save decision training data to CSV
        output_path_dec = output_data_dir / 'training_data_dec.csv'
        final_df_dec.to_csv(output_path_dec, index=False)
        print(f"\nSaved decision training DataFrame to {output_path_dec}")
        
        print("Finished")
        
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Build UFC feature tables and training CSVs from raw UFCStats data.")
    parser.add_argument("--db-url", default=os.getenv("DATABASE_URL"), help="PostgreSQL URL. Defaults to DATABASE_URL or local postgres.")
    parser.add_argument("--raw-data-dir", default=str(raw_ufcstats_dir()), help="Directory containing competitions.csv and individuals.csv.")
    parser.add_argument("--output-data-dir", default=str(data_dir()), help="Directory for prediction_data.csv and training_data*.csv.")
    parser.add_argument("--scrape", action="store_true", help="Scrape UFCStats before rebuilding.")
    parser.add_argument("--reset-db", action="store_true", help="Drop generated schemas before rebuilding.")
    parser.add_argument("--odds", action="store_true", help="Refresh BFO odds and calculate odds features.")
    parser.add_argument("--odds-features", action="store_true", help="Calculate odds features from the configured odds database without scraping BFO.")
    return parser.parse_args()


def cli():
    args = parse_args()
    main(
        odds=args.odds,
        db_url=args.db_url,
        raw_data_dir=args.raw_data_dir,
        output_data_dir=args.output_data_dir,
        scrape=args.scrape,
        reset_db=args.reset_db,
        odds_features=args.odds_features,
    )


if __name__ == "__main__":
    cli()
