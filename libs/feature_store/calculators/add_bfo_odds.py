import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import text, create_engine
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.paths import odds_database_url

class AddBFOodds(BaseCalculator):
    """
    Calculator for processing odds data from the BFO database (odds.bfo table).
    Extracts opening and closing odds for each fight and adds them to the features.odds table.
    """
    
    def __init__(self, conn, bfo_db_url=None):
        """
        Initialize the AddBFOodds calculator.
        
        Args:
            conn: Database connection for the mma-ai database
            bfo_db_url: Connection URL for the odds database containing the BFO table
        """
        super().__init__(conn, calculator_type='single_table')
        
        # Connect to the BFO database
        self.bfo_engine = create_engine(bfo_db_url or odds_database_url())
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Threshold for fuzzy matching (out of 100)
        self.fuzzy_match_threshold = 70
        
        # Threshold for determining separate fights (in days)
        self.new_fight_threshold_days = 30
    
    def calculate(self):
        """
        Main calculation method that extracts BFO odds and adds them to the features.odds table.
        """
        self.logger.info("Starting AddBFOodds calculation")
        
        # 1. Create the features.odds table if it doesn't exist
        self.create_odds_table()
        
        # 2. Extract and process BFO odds
        self.process_bfo_odds()
        
        self.logger.info("AddBFOodds calculation completed")
    
    def create_odds_table(self):
        """
        Create the features.odds table if it doesn't exist.
        """
        self.logger.info("Creating features.odds table if it doesn't exist")
        
        create_table_query = text('''
            CREATE TABLE IF NOT EXISTS features.odds (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                fighter_name VARCHAR(255),
                opening_odds FLOAT,
                closing_odds FLOAT,
                PRIMARY KEY (fight_id, fighter_id),
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_odds_fight ON features.odds(fight_id);
            CREATE INDEX IF NOT EXISTS idx_odds_fighter ON features.odds(fighter_id);
            CREATE INDEX IF NOT EXISTS idx_odds_event ON features.odds(event_id);
        ''')
        
        self.conn.execute(create_table_query)
        self.conn.commit()
    
    def process_bfo_odds(self):
        """
        Process odds from the BFO database and add them to the features.odds table.
        """
        self.logger.info("Processing BFO odds")
        
        try:
            # 1. Get all unique fighter-opponent combinations from BFO
            fighter_pairs = self.get_unique_fighter_pairs()
            
            # 2. Process each fighter pair to identify separate fights
            for fighter, opponent in fighter_pairs:
                self.logger.info(f"Processing fighter pair: {fighter} vs {opponent}")
                
                # Get all odds entries for this fighter pair
                odds_entries = self.get_odds_entries(fighter, opponent)
                
                # Group the entries by fight (based on timestamp gaps)
                fight_groups = self.group_by_fight(odds_entries)
                
                # Process each fight group
                for fight_group in fight_groups:
                    # Get opening and closing odds
                    opening_odds = fight_group[0]['odds']  # First entry
                    closing_odds = fight_group[-1]['odds']  # Last entry
                    
                    # Find the event date (use the timestamp of the last entry)
                    event_date = datetime.strptime(
                        fight_group[-1]['timestamp'], 
                        "%Y-%m-%d %H:%M:%S"
                    ).date()
                    
                    # Find matching fights in the mma-ai database
                    fights_that_day = self.fetch_fights_that_day(event_date)
                    
                    if not fights_that_day:
                        self.logger.warning(f"No fights found in database for date: {event_date}")
                        continue
                    
                    # Match fighters using fuzzy matching
                    matched_fights = self.match_fights_by_fuzzy(
                        fights_that_day, 
                        fighter, 
                        opponent
                    )
                    
                    # Update the odds for each matched fight
                    for match in matched_fights:
                        fight_id = match['fight_id']
                        fighter1_id = match['fighter1_id']
                        fighter2_id = match['fighter2_id']
                        scenario = match['scenario']
                        
                        self.logger.info(f"Updating fight_id: {fight_id} with scenario {scenario}")
                        
                        if scenario == 1:
                            # Direct match (fighter1 = fighter, fighter2 = opponent)
                            self.upsert_fighter_odds(
                                fight_id, fighter1_id, fighter, opening_odds, closing_odds
                            )
                            # For the opponent, we need to convert the moneyline odds
                            self.upsert_fighter_odds(
                                fight_id, fighter2_id, opponent, 
                                self.convert_opponent_odds(opening_odds),
                                self.convert_opponent_odds(closing_odds)
                            )
                        else:
                            # Swapped match (fighter1 = opponent, fighter2 = fighter)
                            self.upsert_fighter_odds(
                                fight_id, fighter2_id, fighter, opening_odds, closing_odds
                            )
                            # For the opponent, we need to convert the moneyline odds
                            self.upsert_fighter_odds(
                                fight_id, fighter1_id, opponent, 
                                self.convert_opponent_odds(opening_odds),
                                self.convert_opponent_odds(closing_odds)
                            )
        
        except Exception as e:
            self.logger.error(f"Error processing BFO odds: {str(e)}")
            raise
    
    def get_unique_fighter_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all unique fighter-opponent pairs from the BFO database.
        
        Returns:
            List of (fighter, opponent) tuples
        """
        query = """
            SELECT DISTINCT fighter, opponent
            FROM bfo
            ORDER BY fighter, opponent
        """
        
        with self.bfo_engine.connect() as conn:
            result = conn.execute(text(query))
            pairs = [(row[0], row[1]) for row in result]
            
        self.logger.info(f"Found {len(pairs)} unique fighter-opponent pairs")
        return pairs
    
    def get_odds_entries(self, fighter: str, opponent: str) -> List[Dict[str, Any]]:
        """
        Get all odds entries for a specific fighter-opponent pair, ordered by timestamp.
        
        Args:
            fighter: Fighter name
            opponent: Opponent name
            
        Returns:
            List of dictionaries with odds data
        """
        query = """
            SELECT id, fighter, opponent, timestamp, odds, fighter_url
            FROM bfo
            WHERE fighter = :fighter AND opponent = :opponent
            ORDER BY timestamp
        """
        
        with self.bfo_engine.connect() as conn:
            result = conn.execute(text(query), {"fighter": fighter, "opponent": opponent})
            entries = [
                {
                    'id': row[0],
                    'fighter': row[1],
                    'opponent': row[2],
                    'timestamp': row[3].strftime("%Y-%m-%d %H:%M:%S") if hasattr(row[3], 'strftime') else row[3],
                    'odds': row[4],
                    'fighter_url': row[5]
                }
                for row in result
            ]
            
        self.logger.info(f"Found {len(entries)} odds entries for {fighter} vs {opponent}")
        return entries
    
    def group_by_fight(self, entries: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group odds entries by fight based on timestamp gaps.
        If there's a gap of more than 30 days between entries, it's considered a different fight.
        
        Args:
            entries: List of odds entries sorted by timestamp
            
        Returns:
            List of entry groups, where each group represents a single fight
        """
        if not entries:
            return []
            
        # Initialize the result with the first group containing the first entry
        result = [[entries[0]]]
        
        # Iterate through the remaining entries
        for i in range(1, len(entries)):
            # Get timestamps for comparison
            prev_timestamp = datetime.strptime(entries[i-1]['timestamp'], "%Y-%m-%d %H:%M:%S")
            curr_timestamp = datetime.strptime(entries[i]['timestamp'], "%Y-%m-%d %H:%M:%S")
            
            # Calculate the time difference in days
            time_diff = (curr_timestamp - prev_timestamp).days
            
            # If the gap is more than the threshold, start a new group
            if time_diff > self.new_fight_threshold_days:
                result.append([entries[i]])
            else:
                # Add to the current group
                result[-1].append(entries[i])
        
        self.logger.info(f"Grouped {len(entries)} entries into {len(result)} fights")
        return result
    
    def fetch_fights_that_day(self, event_date) -> List[Dict[str, Any]]:
        """
        Return a list of dicts for fights matching the event_date +/- 1 day.
        
        Args:
            event_date: Date of the event
            
        Returns:
            List of dictionaries with fight data
        """
        query = text('''
            SELECT f.fight_id,
                   f.fighter1_id,
                   f.fighter2_id,
                   fm1.fighter_name as fighter1_name,
                   fm2.fighter_name as fighter2_name,
                   fm1.fighter_url as fighter1_url,
                   fm2.fighter_url as fighter2_url
            FROM features.fight_mapping f
            JOIN features.fighter_mapping fm1 ON f.fighter1_id = fm1.fighter_id
            JOIN features.fighter_mapping fm2 ON f.fighter2_id = fm2.fighter_id
            JOIN features.event_mapping e ON f.event_id = e.event_id
            WHERE e.event_date BETWEEN :date - INTERVAL '1 day' AND :date + INTERVAL '1 day'
        ''')

        results = []
        rows = self.conn.execute(query, {"date": event_date}).fetchall()

        for r in rows:
            results.append({
                "fight_id": r[0],
                "fighter1_id": r[1],
                "fighter2_id": r[2],
                "fighter1_name": r[3],
                "fighter2_name": r[4],
                "fighter1_url": r[5],
                "fighter2_url": r[6]
            })

        return results
    
    def match_fights_by_fuzzy(self, fights_that_day, csv_f1, csv_f2, threshold=70):
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
        
        Args:
            fights_that_day: List of fights on the event date
            csv_f1: First fighter name from BFO
            csv_f2: Second fighter name from BFO
            threshold: Fuzzy matching threshold
            
        Returns:
            List of matched fights with scenario info
        """
        from fuzzywuzzy import fuzz
        
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
            self.logger.warning(f"No match found for fighters: {csv_f1} vs {csv_f2}")
        
        return matched
    
    def convert_opponent_odds(self, odds_value):
        """
        Convert the odds for the opponent based on the fighter's odds.
        
        Args:
            odds_value: Odds value for the fighter
            
        Returns:
            Corresponding odds value for the opponent
        """
        # For moneyline odds, we need to calculate the corresponding odds for the opponent
        if odds_value > 0:
            # Fighter is underdog, opponent is favorite
            return 100 / (odds_value + 100) * 100 * -1
        else:
            # Fighter is favorite, opponent is underdog
            return 100 / (abs(odds_value) / 100) * 100
    
    def upsert_fighter_odds(self, fight_id, fighter_id, fighter_name, opening_odds, closing_odds):
        """
        Upsert opening and closing odds values for a given fight_id and fighter_id.
        
        Args:
            fight_id: The fight ID
            fighter_id: The fighter ID
            fighter_name: The fighter name
            opening_odds: The opening odds value
            closing_odds: The closing odds value
        """
        # First get the event_id for this fight
        get_event = text("""
            SELECT event_id 
            FROM features.fight_mapping 
            WHERE fight_id = :fight_id
        """)
        
        event_id = self.conn.execute(get_event, {"fight_id": fight_id}).scalar()
        
        if not event_id:
            self.logger.warning(f"Could not find event_id for fight_id {fight_id}")
            return

        query = text('''
            INSERT INTO features.odds 
            (fight_id, fighter_id, event_id, fighter_name, opening_odds, closing_odds)
            VALUES (:fight_id, :fighter_id, :event_id, :fighter_name, :opening_odds, :closing_odds)
            ON CONFLICT (fight_id, fighter_id)
            DO UPDATE SET 
                fighter_name = EXCLUDED.fighter_name,
                opening_odds = EXCLUDED.opening_odds,
                closing_odds = EXCLUDED.closing_odds
        ''')
        
        self.conn.execute(query, {
            "fight_id": fight_id,
            "fighter_id": fighter_id,
            "event_id": event_id,
            "fighter_name": fighter_name,
            "opening_odds": float(opening_odds),
            "closing_odds": float(closing_odds)
        })
        self.conn.commit()
    
    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator sequentially.
        
        Returns:
            Empty dictionary as this calculator operates directly on tables
        """
        # Create the features.odds table if it doesn't exist
        self.create_odds_table()
        
        # Process BFO odds and add them to the features.odds table
        self.process_bfo_odds()
        
        # Return an empty dictionary since we're not returning any DataFrames
        return {} 
