import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.paths import odds_database_url

class OddsCalculator(BaseCalculator):
    """
    Calculator for processing odds data from the BFO database (bestfightodds.bfo table).
    Extracts opening, closing, and 7-day prior odds for each fight and adds them to the features.odds table.
    Uses exact name matching with lowercase comparisons for reliability.
    """
    
    def __init__(self, conn, bfo_db_url=None):
        """
        Initialize the OddsCalculator.
        
        Args:
            conn: Database connection for the mma-ai database
            bfo_db_url: Connection URL for the odds database containing the BFO table
        """
        super().__init__(conn, calculator_type='single_table')
        
        # Connect to the BFO database
        self.bfo_engine = create_engine(bfo_db_url or odds_database_url())
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Threshold for determining separate fights (in days)
        self.new_fight_threshold_days = 30
        
        # Batch size for updates
        self.batch_size = 1000
    
    def calculate(self):
        """
        Main calculation method that extracts BFO odds and adds them to the features.odds table.
        Optimized for speed with batch processing throughout the pipeline.
        """
        start_time = time.time()
        self.logger.info("Starting OddsCalculator calculation")
        
        # 1. Create the features.odds table if it doesn't exist
        self.create_odds_table()
        
        # 2. Extract and process BFO odds - optimized end-to-end pipeline
        self.process_odds_pipeline()
        
        end_time = time.time()
        self.logger.info(f"OddsCalculator calculation completed in {end_time - start_time:.2f} seconds")
    
    def create_odds_table(self):
        """
        Create the features.odds table if it doesn't exist.
        Ensures table exists before attempting modifications.
        """
        self.logger.info("Creating features.odds table if it doesn't exist")
        
        # Step 1: Create the table if it doesn't exist
        create_table_sql = text('''
            CREATE TABLE IF NOT EXISTS features.odds (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                fighter_name VARCHAR(255),
                opponent_name VARCHAR(255),
                opening_odds FLOAT,
                closing_odds FLOAT,
                ip_opening_odds FLOAT,
                ip_closing_odds FLOAT,
                vigless_ip_opening_odds FLOAT,
                vigless_ip_closing_odds FLOAT,
                -- Renamed 7-day columns
                sevenday_opening_odds FLOAT,
                sevenday_ip_opening_odds FLOAT,
                sevenday_vigless_ip_opening_odds FLOAT,
                -- Define PK here, it might be updated later if needed
                PRIMARY KEY (fight_id, fighter_id, event_id), 
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
        ''')
        self.conn.execute(create_table_sql)
        self.conn.commit() # Commit to make the table visible
        
        # Step 1.5: Add new columns if they don't exist (idempotent)
        add_columns_sql = text('''
            ALTER TABLE features.odds 
            ADD COLUMN IF NOT EXISTS sevenday_opening_odds FLOAT;
            
            ALTER TABLE features.odds 
            ADD COLUMN IF NOT EXISTS sevenday_ip_opening_odds FLOAT;
            
            ALTER TABLE features.odds 
            ADD COLUMN IF NOT EXISTS sevenday_vigless_ip_opening_odds FLOAT;
        ''')
        self.conn.execute(add_columns_sql)
        self.conn.commit() # Commit column additions
        
        self.logger.info("Table features.odds ensured to exist and columns added. Applying modifications...")

        # Step 2: Apply modifications (PK check/update, indexes, renames)
        modify_table_sql = text('''
            -- Check if the primary key needs updating (handles legacy PK)
            DO $$
            BEGIN
                -- Drop legacy primary key (fight_id, fighter_id) if it exists
                IF EXISTS (
                    SELECT 1 FROM pg_constraint 
                    WHERE conname = 'odds_pkey' AND conrelid = 'features.odds'::regclass
                      AND pg_get_constraintdef(oid) LIKE 'PRIMARY KEY (fight_id, fighter_id)' 
                      AND NOT pg_get_constraintdef(oid) LIKE 'PRIMARY KEY (fight_id, fighter_id, event_id)' 
                ) THEN
                    ALTER TABLE features.odds DROP CONSTRAINT odds_pkey;
                    ALTER TABLE features.odds ADD PRIMARY KEY (fight_id, fighter_id, event_id);
                -- Ensure the correct primary key exists if no PK is found (e.g., just created table)
                ELSIF NOT EXISTS (
                     SELECT 1 FROM pg_constraint 
                     WHERE conrelid = 'features.odds'::regclass AND contype = 'p'
                ) THEN
                     ALTER TABLE features.odds ADD PRIMARY KEY (fight_id, fighter_id, event_id);
                END IF;
            END
            $$;

            -- Add indexes
            CREATE INDEX IF NOT EXISTS idx_odds_fight ON features.odds(fight_id);
            CREATE INDEX IF NOT EXISTS idx_odds_fighter ON features.odds(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_odds_event ON features.odds(event_id);

            -- Rename columns if they exist with old names
            DO $$
            BEGIN
                IF EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_schema = 'features' AND table_name = 'odds' 
                          AND column_name = 'dec_opening_odds') THEN
                    ALTER TABLE features.odds RENAME COLUMN dec_opening_odds TO opening_odds;
                END IF;
                
                IF EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_schema = 'features' AND table_name = 'odds' 
                          AND column_name = 'dec_closing_odds') THEN
                    ALTER TABLE features.odds RENAME COLUMN dec_closing_odds TO closing_odds;
                END IF;
                
                IF EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_schema = 'features' AND table_name = 'odds' 
                          AND column_name = 'vigless_dec_opening_odds') THEN
                    ALTER TABLE features.odds RENAME COLUMN vigless_dec_opening_odds TO vigless_ip_opening_odds;
                END IF;
                
                IF EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_schema = 'features' AND table_name = 'odds' 
                          AND column_name = 'vigless_dec_closing_odds') THEN
                    ALTER TABLE features.odds RENAME COLUMN vigless_dec_closing_odds TO vigless_ip_closing_odds;
                END IF;
            END
            $$;
        ''')
        
        self.conn.execute(modify_table_sql)
        self.conn.commit() # Commit the modifications

        self.logger.info("features.odds table modifications complete.")
    
    def process_odds_pipeline(self):
        """
        Streamlined end-to-end pipeline to process all odds data efficiently.
        Uses batch processing and minimizes database round-trips.
        """
        start_time = time.time()
        
        # 1. Load all BFO odds data in a single query
        self.logger.info("Loading BFO odds data...")
        bfo_odds = self.load_bfo_odds()
        self.logger.info(f"Loaded {len(bfo_odds)} BFO odds entries in {time.time() - start_time:.2f} seconds")
        
        # 2. Load all fighter and event mappings
        stage_time = time.time()
        self.logger.info("Loading fighter and event mappings...")
        fight_mappings = self.load_fight_mappings()
        self.logger.info(f"Loaded {len(fight_mappings)} fight mappings in {time.time() - stage_time:.2f} seconds")
        
        # 3. Process and match fight odds
        stage_time = time.time()
        self.logger.info("Processing and matching odds...")
        processed_odds = self.match_fighter_odds(bfo_odds, fight_mappings)
        self.logger.info(f"Processed and matched {len(processed_odds)} odds records in {time.time() - stage_time:.2f} seconds")
        
        # 4. Batch insert all processed odds
        stage_time = time.time()
        self.logger.info("Inserting odds data in batches...")
        self.batch_insert_odds(processed_odds)
        self.logger.info(f"Completed batch inserts in {time.time() - stage_time:.2f} seconds")
        
        # 5. Normalize vigless odds
        stage_time = time.time()
        self.logger.info("Normalizing vigless odds...")
        self.bulk_normalize_vigless_odds()
        self.logger.info(f"Completed normalization in {time.time() - stage_time:.2f} seconds")
        
        self.logger.info(f"Total pipeline execution time: {time.time() - start_time:.2f} seconds")
    
    def load_bfo_odds(self) -> Dict:
        """
        Load all BFO odds data, grouped by fighter, opponent, and event month.
        Identifies opening, closing, and odds closest to 7 days before closing.
        
        Returns:
            Dictionary mapping (fighter, opponent, event_month_str) to a dict containing 
            lists of odds entries for 'opening', 'closing', and 'sevenday_opening'.
        """
        # Fetch all odds, ordered by timestamp. Grouping will happen in Python.
        # Use PostgreSQL's to_char for date formatting.
        query = """
            SELECT 
                fighter, 
                opponent, 
                timestamp, 
                odds,
                -- Extract event month for grouping using PostgreSQL's to_char
                to_char(timestamp, 'YYYY-MM') as event_month 
            FROM bestfightodds.bfo -- Use schema name here
            ORDER BY fighter, opponent, event_month, timestamp
        """
        
        # Temporary structure to hold all odds per group
        raw_odds_groups = defaultdict(list)
        
        with self.bfo_engine.connect() as conn:
            # Execute the query directly - no need for SQLite fallback now
            result = conn.execute(text(query))

            for row in result:
                fighter = row[0]
                opponent = row[1]
                event_month_str = row[4]
                timestamp = pd.to_datetime(row[2]) # Convert to pandas Timestamp for easier comparison
                
                raw_odds_groups[(fighter, opponent, event_month_str)].append({
                    'timestamp': timestamp,
                    'odds': row[3]
                })
        
        # Process groups to find opening, closing, and 7-day odds
        processed_odds_data = defaultdict(lambda: {'opening': None, 'closing': None, 'sevenday_opening': None})

        for key, entries in raw_odds_groups.items():
            if not entries:
                continue
                
            # Ensure entries are sorted by timestamp (should be from SQL, but double-check)
            entries.sort(key=lambda x: x['timestamp']) 
            
            opening_entry = entries[0]
            closing_entry = entries[-1]
            
            processed_odds_data[key]['opening'] = opening_entry
            processed_odds_data[key]['closing'] = closing_entry
            
            # Find 7-day prior odds
            closing_ts = closing_entry['timestamp']
            target_7day_ts = closing_ts - timedelta(days=7)
            
            best_7day_entry = None
            min_diff = timedelta.max
            
            # Iterate through entries to find the one closest to the 7-day target
            for entry in entries:
                diff = abs(entry['timestamp'] - target_7day_ts)
                if diff <= min_diff: # Prioritize the closest; if tied, takes the latest one due to loop order
                    # Ensure the timestamp is not *after* the closing timestamp (unlikely but possible with bad data)
                    if entry['timestamp'] <= closing_ts:
                         min_diff = diff
                         best_7day_entry = entry
            
            # If we found an entry near the 7-day mark, store it
            if best_7day_entry:
                 processed_odds_data[key]['sevenday_opening'] = best_7day_entry
                 # If opening/closing are the closest, they serve as 7day too
            else: # Handle case where no odds exist >= 7 days prior? 
                # If no odds found near 7 days prior, 'sevenday_opening' remains None
                processed_odds_data[key]['sevenday_opening'] = opening_entry

        return processed_odds_data
    
    def load_fight_mappings(self) -> Dict:
        """
        Load all fight mappings with fighter names in a single query.
        
        Returns:
            Dictionary mapping event dates to lists of fight details.
        """
        query = text('''
            SELECT 
                f.fight_id,
                f.fighter1_id,
                f.fighter2_id,
                f.event_id,
                LOWER(fm1.fighter_name) as fighter1_name,
                LOWER(fm2.fighter_name) as fighter2_name,
                e.event_date
            FROM features.fight_mapping f
            JOIN features.fighter_mapping fm1 ON f.fighter1_id = fm1.fighter_id
            JOIN features.fighter_mapping fm2 ON f.fighter2_id = fm2.fighter_id
            JOIN features.event_mapping e ON f.event_id = e.event_id
        ''')
        
        # Group by event date for efficient matching
        date_to_fights = defaultdict(list)
        
        rows = self.conn.execute(query).fetchall()
        for row in rows:
            event_date = row[6]
            
            # Build a +/- 3 day window to handle date mismatches
            for offset in range(-3, 4):
                date_key = event_date + timedelta(days=offset)
                
                # Add fight data to all date keys in window
                date_to_fights[date_key].append({
                    "fight_id": row[0],
                    "fighter1_id": row[1],
                    "fighter2_id": row[2],
                    "event_id": row[3],
                    "fighter1_name": row[4],
                    "fighter2_name": row[5],
                    "event_date": event_date
                })
        
        return date_to_fights
    
    def match_fighter_odds(self, odds_data, fight_mappings) -> List[Dict]:
        """
        Match BFO odds to fights and prepare data for batch insertion.
        
        Args:
            odds_data: Dictionary mapping (fighter, opponent, event_month) to odds data
            fight_mappings: Dictionary mapping dates to fights
            
        Returns:
            List of processed odds records ready for insertion
        """
        processed_records = []
        
        # Track processed fights to avoid duplicates in rematch scenarios
        processed_fights = set()
        
        # Process each fighter-opponent-eventmonth tuple with its opening and closing odds
        for (fighter, opponent, event_month), odds_group in odds_data.items():
            # Skip pairs with insufficient data
            if len(odds_group) < 1:
                continue
                
            # Extract opening and closing odds
            opening_entry = odds_group.get('opening')
            closing_entry = odds_group.get('closing')
            event_date_from_odds = None
            
            # Derive event_date_from_odds from the closing_entry if it exists
            if closing_entry:
                 event_date_from_odds = closing_entry['timestamp'].date() if hasattr(closing_entry['timestamp'], 'date') else pd.to_datetime(closing_entry['timestamp']).date()
                 
            # Skip if missing crucial data (opening or closing entry, or derived date)
            if not opening_entry or not closing_entry or not event_date_from_odds:
                continue
                
            # --- Match to Fights with +/- 1 day window ---
            matched_fights = []
            potential_matches = []
            
            # Check dates: day before, day of, day after the closing odds timestamp
            for day_offset in [-1, 0, 1]:
                lookup_date = event_date_from_odds + timedelta(days=day_offset)
                fights_on_lookup_day = fight_mappings.get(lookup_date, [])
                
                # Try exact name matching with lowercase comparison
                fighter_lower = fighter.lower()
                opponent_lower = opponent.lower()

                for fight in fights_on_lookup_day:
                    # Match by name
                    if ((fighter_lower == fight['fighter1_name'] and opponent_lower == fight['fighter2_name']) or
                        (fighter_lower == fight['fighter2_name'] and opponent_lower == fight['fighter1_name'])):
                        # Check if we already added this fight_id from a different date lookup
                        if not any(pm['fight_id'] == fight['fight_id'] for pm in potential_matches):
                             potential_matches.append(fight)
            
            # If we found potential matches across the date window, find the closest one by date
            if potential_matches:
                # Find the match with the minimum absolute difference in days 
                # between the fight's actual event_date and the closing odds date
                closest_match = min(potential_matches, 
                                  key=lambda f: abs((f['event_date'] - event_date_from_odds).days))
                
                fight = closest_match
                # Use a composite key including event_id to handle potential same-day events/rematches correctly
                fight_key = (fight['fight_id'], fight['event_id'])
                
                # Skip if this fight (fight_id, event_id combo) has already been processed
                if fight_key in processed_fights:
                    continue
                
                # Determine match scenario (fighter1 or fighter2 in BFO data?)
                fighter_lower = fighter.lower() # Recalculate just in case
                opponent_lower = opponent.lower()
                scenario = 0
                if fighter_lower == fight['fighter1_name'] and opponent_lower == fight['fighter2_name']:
                    scenario = 1
                elif fighter_lower == fight['fighter2_name'] and opponent_lower == fight['fighter1_name']:
                    scenario = 2
                    
                if scenario > 0:
                    matched_fights.append({
                        "fight_id": fight['fight_id'],
                        "event_id": fight['event_id'],
                        "fighter1_id": fight['fighter1_id'],
                        "fighter2_id": fight['fighter2_id'],
                        "scenario": scenario
                    })
                    # Mark as processed using the composite key
                    processed_fights.add(fight_key)
            
            # Process the matched fights (if any)
            for match in matched_fights:
                fight_id = match['fight_id']
                event_id = match['event_id']
                fighter1_id = match['fighter1_id']
                fighter2_id = match['fighter2_id']
                scenario = match['scenario']
                
                # Look for opponent's raw odds
                opponent_opening_odds = None
                opponent_closing_odds = None
                opponent_sevenday_odds = None # Initialize for opponent 7-day

                # Check if we have the opponent's odds data in the structure from load_bfo_odds
                reversed_key = (opponent, fighter, event_month)
                if reversed_key in odds_data:
                    opp_odds_group = odds_data[reversed_key]
                    
                    # Directly access opponent's entries (remove the old loop)
                    opp_opening_entry = opp_odds_group.get('opening')
                    opp_closing_entry = opp_odds_group.get('closing')
                    opp_sevenday_entry = opp_odds_group.get('sevenday_opening') # Get opponent 7day entry

                    if opp_opening_entry:
                        opponent_opening_odds = opp_opening_entry['odds']
                    if opp_closing_entry:
                        opponent_closing_odds = opp_closing_entry['odds']
                    if opp_sevenday_entry:
                        opponent_sevenday_odds = opp_sevenday_entry['odds']
                    # Note: Fallback for missing opponent 7day odds is handled below

                # Fallbacks if opponent odds (opening/closing) were not found directly
                if opponent_opening_odds is None and opening_entry:
                    opponent_opening_odds = self.convert_opponent_odds(opening_entry['odds'])
                if opponent_closing_odds is None and closing_entry:
                    opponent_closing_odds = self.convert_opponent_odds(closing_entry['odds'])

                # Get fighter's 7-day odds (using fallback to opening if necessary)
                sevenday_entry = odds_group.get('sevenday_opening')
                sevenday_odds = sevenday_entry['odds'] if sevenday_entry else opening_entry['odds']
                sevenday_ip = 1 / float(sevenday_odds) if float(sevenday_odds) > 0 else 0.5

                # Finalize opponent's 7-day odds (using fallbacks)
                if opponent_sevenday_odds is None:
                    # Fallback 1: Use opponent's standard opening odds if available
                    if opponent_opening_odds is not None:
                        opponent_sevenday_odds = opponent_opening_odds
                    # Fallback 2: Calculate from fighter's 7-day odds
                    else:
                        opponent_sevenday_odds = self.convert_opponent_odds(sevenday_odds)

                opponent_sevenday_ip = 1 / float(opponent_sevenday_odds) if float(opponent_sevenday_odds) > 0 else 0.5
                
                # Prepare records based on the scenario
                if scenario == 1:
                    # Fighter 1 record
                    processed_records.append({
                        "fight_id": fight_id,
                        "fighter_id": fighter1_id,
                        "event_id": event_id,
                        "fighter_name": fighter,
                        "opponent_name": opponent,
                        "opening_odds": float(opening_entry['odds']),
                        "closing_odds": float(closing_entry['odds']),
                        "sevenday_opening_odds": float(sevenday_odds),
                        "ip_opening_odds": 1 / float(opening_entry['odds']) if float(opening_entry['odds']) > 0 else 0.5,
                        "ip_closing_odds": 1 / float(closing_entry['odds']) if float(closing_entry['odds']) > 0 else 0.5,
                        "sevenday_ip_opening_odds": sevenday_ip
                    })
                    
                    # Fighter 2 record
                    processed_records.append({
                        "fight_id": fight_id,
                        "fighter_id": fighter2_id,
                        "event_id": event_id,
                        "fighter_name": opponent,
                        "opponent_name": fighter,
                        "opening_odds": float(opponent_opening_odds),
                        "closing_odds": float(opponent_closing_odds),
                        "sevenday_opening_odds": float(opponent_sevenday_odds),
                        "ip_opening_odds": 1 / float(opponent_opening_odds) if float(opponent_opening_odds) > 0 else 0.5,
                        "ip_closing_odds": 1 / float(opponent_closing_odds) if float(opponent_closing_odds) > 0 else 0.5,
                        "sevenday_ip_opening_odds": opponent_sevenday_ip
                    })
                else: # scenario == 2
                    # Fighter 2 record (BFO fighter is fighter2 in DB)
                    processed_records.append({
                        "fight_id": fight_id,
                        "fighter_id": fighter2_id,
                        "event_id": event_id,
                        "fighter_name": fighter,
                        "opponent_name": opponent,
                        "opening_odds": float(opening_entry['odds']),
                        "closing_odds": float(closing_entry['odds']),
                        "sevenday_opening_odds": float(sevenday_odds),
                        "ip_opening_odds": 1 / float(opening_entry['odds']) if float(opening_entry['odds']) > 0 else 0.5,
                        "ip_closing_odds": 1 / float(closing_entry['odds']) if float(closing_entry['odds']) > 0 else 0.5,
                        "sevenday_ip_opening_odds": sevenday_ip
                    })
                    
                    # Fighter 1 record (BFO opponent is fighter1 in DB)
                    processed_records.append({
                        "fight_id": fight_id,
                        "fighter_id": fighter1_id,
                        "event_id": event_id,
                        "fighter_name": opponent,
                        "opponent_name": fighter,
                        "opening_odds": float(opponent_opening_odds),
                        "closing_odds": float(opponent_closing_odds),
                        "sevenday_opening_odds": float(opponent_sevenday_odds),
                        "ip_opening_odds": 1 / float(opponent_opening_odds) if float(opponent_opening_odds) > 0 else 0.5,
                        "ip_closing_odds": 1 / float(opponent_closing_odds) if float(opponent_closing_odds) > 0 else 0.5,
                        "sevenday_ip_opening_odds": opponent_sevenday_ip
                    })
        
        # Add initial placeholders for vigless odds (will be normalized later)
        for record in processed_records:
            record["vigless_ip_opening_odds"] = record["ip_opening_odds"]
            record["vigless_ip_closing_odds"] = record["ip_closing_odds"]
            # Add placeholder for new vigless column
            record["sevenday_vigless_ip_opening_odds"] = record["sevenday_ip_opening_odds"]
            
        return processed_records
    
    def batch_insert_odds(self, records):
        """
        Insert records in efficient batches for better performance.
        Deletes existing records for a fight/event before inserting.

        Args:
            records: List of odds records to insert
        """
        if not records:
            self.logger.warning("No odds records to insert")
            return

        self.logger.info(f"Inserting {len(records)} odds records in batches of {self.batch_size}")

        # Delete query for removing existing odds records for a specific fight and event
        delete_query = text('''
            DELETE FROM features.odds
            WHERE fight_id = :fight_id AND event_id = :event_id
        ''')

        # Insert query for adding new odds records, including sevenday columns
        insert_query = text('''
            INSERT INTO features.odds
            (fight_id, fighter_id, event_id, fighter_name, opponent_name,
             opening_odds, closing_odds, sevenday_opening_odds,
             ip_opening_odds, ip_closing_odds, sevenday_ip_opening_odds,
             vigless_ip_opening_odds, vigless_ip_closing_odds, sevenday_vigless_ip_opening_odds)
            VALUES (:fight_id, :fighter_id, :event_id, :fighter_name, :opponent_name,
                    :opening_odds, :closing_odds, :sevenday_opening_odds,
                    :ip_opening_odds, :ip_closing_odds, :sevenday_ip_opening_odds,
                    :vigless_ip_opening_odds, :vigless_ip_closing_odds, :sevenday_vigless_ip_opening_odds)
        ''')

        # Track which fights we've deleted already in this function call
        deleted_fights = set()
            
        # Process in batches
        for i in range(0, len(records), self.batch_size):
            batch = records[i:i + self.batch_size]

            # Track which fights we've deleted already in this batch execution - MOVED OUTSIDE LOOP
            # deleted_fights = set()

            # Step 1: Delete existing records for fights in this batch
            for record in batch:
                # Only delete once per fight_id + event_id combination within this run
                fight_event_key = (record['fight_id'], record['event_id'])
                if fight_event_key not in deleted_fights:
                    # Execute the delete for this fight/event
                    self.conn.execute(delete_query, {
                        'fight_id': record['fight_id'],
                        'event_id': record['event_id']
                    })
                    deleted_fights.add(fight_event_key)

            # Step 2: Insert all records in the batch
            try:
                self.conn.execute(insert_query, batch)
            except Exception as e:
                self.logger.error(f"Error inserting batch {i//self.batch_size + 1}: {e}")
                # Optionally inspect the batch content here
                # self.logger.error(f"Batch data: {batch}")
                continue # Continue to next batch if possible

            # Commit each batch
            self.conn.commit()
            self.logger.info(f"Inserted batch {i//self.batch_size + 1}/{(len(records) - 1)//self.batch_size + 1}")
    
    def bulk_normalize_vigless_odds(self):
        """
        Normalize all vigless odds using a direct SQL update for maximum efficiency.
        Handles opening, closing, and 7-day opening odds.
        """
        # Direct SQL normalization for opening, closing, and 7-day odds
        normalize_query = text('''
            -- First, create a CTE with fight-level totals for all relevant implied probabilities
            WITH fight_totals AS (
                SELECT
                    fight_id,
                    event_id,
                    SUM(ip_opening_odds) as total_ip_opening,
                    SUM(ip_closing_odds) as total_ip_closing,
                    SUM(sevenday_ip_opening_odds) as total_sevenday_ip_opening -- Sum for the new 7-day IP
                FROM features.odds
                GROUP BY fight_id, event_id
            )

            -- Update all odds in a single pass
            UPDATE features.odds o
            SET
                vigless_ip_opening_odds = CASE
                    WHEN ft.total_ip_opening > 0 THEN o.ip_opening_odds / ft.total_ip_opening
                    ELSE 0.5
                END,
                vigless_ip_closing_odds = CASE
                    WHEN ft.total_ip_closing > 0 THEN o.ip_closing_odds / ft.total_ip_closing
                    ELSE 0.5
                END,
                sevenday_vigless_ip_opening_odds = CASE -- Normalize the 7-day vigless odds
                    WHEN ft.total_sevenday_ip_opening > 0 THEN o.sevenday_ip_opening_odds / ft.total_sevenday_ip_opening
                    ELSE 0.5
                END
            FROM fight_totals ft
            WHERE o.fight_id = ft.fight_id AND o.event_id = ft.event_id
        ''')

        # Execute the bulk update
        result = self.conn.execute(normalize_query)
        self.conn.commit()

        self.logger.info(f"Normalized {result.rowcount} odds records (including 7-day odds) in a single operation")
    
    def convert_opponent_odds(self, odds_value):
        """
        Convert the odds for the opponent based on the fighter's odds.
        Keep the odds in decimal format.
        
        Args:
            odds_value: Odds value for the fighter in decimal format
            
        Returns:
            Corresponding odds value for the opponent in decimal format
        """
        # Convert to float for safety
        odds_value = float(odds_value)
        
        # For decimal odds, the opponent's odds are calculated as:
        # opponent_odds = 1 + (1 / (odds_value - 1))
        # This ensures the implied probabilities sum to 1 (without vig)
        if odds_value > 1.0:
            return 1.0 + (1.0 / (odds_value - 1.0))
        else:
            # Handle edge case if odds are not in decimal format
            return 2.0  # Even odds as fallback
    
    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator sequentially.
        
        Returns:
            Empty dictionary as this calculator operates directly on tables
        """
        # Run the optimized end-to-end pipeline
        self.create_odds_table()
        self.process_odds_pipeline()
        
        # Return an empty dictionary since we're not returning any DataFrames
        return {} 
