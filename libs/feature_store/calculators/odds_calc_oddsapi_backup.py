import pandas as pd
import logging
from datetime import datetime, timedelta
import requests
from sqlalchemy import text
from fuzzywuzzy import fuzz
from typing import Dict, List, Optional, Any, Tuple
import os

from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext

class OddsCalculator(BaseCalculator):
    """
    Calculator for processing odds data from combined_odds.csv and TheOddsAPI.
    Creates and populates the features.odds table with fighter odds and probabilities.
    """
    
    def __init__(self, conn, scrape_date=None, combined_odds_path="combined_odds.csv"):
        """
        Initialize the OddsCalculator.
        
        Args:
            conn: Database connection
            scrape_date: Date to scrape odds up to (default: current date)
            combined_odds_path: Path to the combined odds CSV file
        """
        super().__init__(conn, calculator_type='single_table')
        
        self.api_key = os.getenv("THE_ODDS_API_KEY")
        self.combined_odds_path = combined_odds_path
        
        # Set scrape date
        if scrape_date:
            if isinstance(scrape_date, str):
                self.scrape_date = datetime.strptime(scrape_date, "%Y-%m-%dT%H:%M:%SZ")
            else:
                self.scrape_date = scrape_date
        else:
            self.scrape_date = datetime.utcnow()
            
        # Set up API URLs
        self.historical_odds_base_url = (
            "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/events"
        )
        self.historical_events_base_url = "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/events"
        
        # Set up fuzzy matching threshold
        self.fuzzy_match_threshold = 70
        
        # Set up logger
        self.logger = logging.getLogger(__name__)

    def _require_api_key(self):
        if not self.api_key:
            raise RuntimeError("Set THE_ODDS_API_KEY before calling The Odds API.")
        return self.api_key
    
    def calculate(self):
        """
        Main calculation method that creates and populates the features.odds table.
        """
        self.logger.info("Starting OddsCalculator calculation")
        
        # 1. Create the features.odds table if it doesn't exist
        self.create_odds_table()
        
        # 2. Process the combined_odds.csv file
        self.process_combined_odds_csv()
        
        # 3. Scrape additional odds from TheOddsAPI
        self.scrape_odds_api()
        
        self.logger.info("OddsCalculator calculation completed")
        
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
                odds FLOAT,
                prob FLOAT,
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
        
    def process_combined_odds_csv(self):
        """
        Process the combined_odds.csv file and add the odds to the features.odds table.
        """
        self.logger.info(f"Processing combined odds from {self.combined_odds_path}")
        
        try:
            # Read the CSV file
            odds_df = pd.read_csv(self.combined_odds_path)
            
            # Normalize the implied probabilities
            odds_df = self.normalize_implied_probabilities(odds_df)
            
            # Process each row in the CSV
            for idx, row in odds_df.iterrows():
                # Skip if either probability is NaN
                if pd.isna(row['p1_prob']) or pd.isna(row['p2_prob']):
                    self.logger.warning(f"Skipping row {idx}: NaN probabilities found")
                    continue
                    
                event_date_str = row['event_date']
                fighter1_name = row['player1']
                fighter2_name = row['player2']
                fighter1_prob = row['p1_prob']
                fighter2_prob = row['p2_prob']
                fighter1_odds = row['p1_odds']
                fighter2_odds = row['p2_odds']
                
                # Convert the event date string to a datetime object
                dt = pd.to_datetime(event_date_str)
                db_date = dt.date()
                
                # Skip if the event date is after the scrape date
                if dt.date() > self.scrape_date.date():
                    self.logger.info(f"Skipping event on {db_date} as it's after the scrape date {self.scrape_date.date()}")
                    continue
                
                # Find matching fights for this date
                fights_that_day = self.fetch_fights_that_day(db_date)
                if not fights_that_day:
                    self.logger.warning(f"No fights found in database for date: {db_date}")
                    continue
                
                self.logger.info(f"Found {len(fights_that_day)} fights on {db_date}")
                
                # Match fighters using fuzzy matching
                matched_fights = self.match_fights_by_fuzzy(
                    fights_that_day, 
                    fighter1_name, 
                    fighter2_name
                )
                
                self.logger.info(f"Found {len(matched_fights)} matching fights")
                
                # Update the odds for each matched fight
                for match in matched_fights:
                    fight_id = match['fight_id']
                    fighter1_id = match['fighter1_id']
                    fighter2_id = match['fighter2_id']
                    scenario = match['scenario']
                    
                    self.logger.info(f"Updating fight_id: {fight_id} with scenario {scenario}")
                    
                    if scenario == 1:
                        # Direct match
                        self.upsert_fighter_odds(fight_id, fighter1_id, fighter1_odds, fighter1_prob)
                        self.upsert_fighter_odds(fight_id, fighter2_id, fighter2_odds, fighter2_prob)
                    else:
                        # Swapped match
                        self.upsert_fighter_odds(fight_id, fighter1_id, fighter2_odds, fighter2_prob)
                        self.upsert_fighter_odds(fight_id, fighter2_id, fighter1_odds, fighter1_prob)
                        
        except Exception as e:
            self.logger.error(f"Error processing combined odds CSV: {str(e)}")
            raise
    
    def scrape_odds_api(self):
        """
        Scrape additional odds from TheOddsAPI for events after the last event in the CSV.
        """
        self.logger.info("Scraping additional odds from TheOddsAPI")
        
        try:
            # Get the latest event date from the features.odds table
            latest_event_query = text('''
                SELECT MAX(e.event_date)
                FROM features.odds o
                JOIN features.event_mapping e ON o.event_id = e.event_id
            ''')
            
            latest_event_date = self.conn.execute(latest_event_query).scalar()
            
            if not latest_event_date:
                self.logger.warning("No events found in features.odds table, using default start date")
                latest_event_date = datetime(2020, 6, 7).date()
            else:
                # Add one day to avoid duplicates
                latest_event_date = latest_event_date + timedelta(days=1)
            
            # Fetch events from the API
            events = self.fetch_historical_events(latest_event_date)
            
            for event in events:
                event_id = event.get('id')
                event_date = datetime.strptime(event.get('commence_time'), "%Y-%m-%dT%H:%M:%SZ")
                
                # Skip if the event date is after the scrape date
                if event_date.date() > self.scrape_date.date():
                    self.logger.info(f"Skipping event on {event_date.date()} as it's after the scrape date {self.scrape_date.date()}")
                    continue
                
                # Fetch odds for this event
                odds_data = self.fetch_historical_odds_for_event(event_id, event_date)
                
                if not odds_data:
                    self.logger.warning(f"No odds data found for event {event_id} on {event_date}")
                    continue
                
                # Process the odds data
                for match in odds_data:
                    home_team = match.get('home_team', '').lower()
                    away_team = match.get('away_team', '').lower()
                    
                    # Find the best bookmaker with the most recent timestamp
                    best_bookmaker = None
                    best_timestamp = None
                    
                    for bookmaker in match.get('bookmakers', []):
                        timestamp = datetime.strptime(bookmaker.get('last_update'), "%Y-%m-%dT%H:%M:%SZ")
                        
                        if not best_timestamp or timestamp > best_timestamp:
                            best_timestamp = timestamp
                            best_bookmaker = bookmaker
                    
                    if not best_bookmaker:
                        self.logger.warning(f"No bookmakers found for match {home_team} vs {away_team}")
                        continue
                    
                    # Get the odds from the best bookmaker
                    markets = best_bookmaker.get('markets', [])
                    h2h_market = next((m for m in markets if m.get('key') == 'h2h'), None)
                    
                    if not h2h_market:
                        self.logger.warning(f"No h2h market found for match {home_team} vs {away_team}")
                        continue
                    
                    outcomes = h2h_market.get('outcomes', [])
                    
                    if len(outcomes) != 2:
                        self.logger.warning(f"Expected 2 outcomes, got {len(outcomes)} for match {home_team} vs {away_team}")
                        continue
                    
                    # Convert decimal odds to implied probabilities
                    home_odds = next((o.get('price') for o in outcomes if o.get('name').lower() == home_team), None)
                    away_odds = next((o.get('price') for o in outcomes if o.get('name').lower() == away_team), None)
                    
                    if not home_odds or not away_odds:
                        self.logger.warning(f"Missing odds for match {home_team} vs {away_team}")
                        continue
                    
                    # Convert decimal odds to implied probabilities
                    home_prob = 1 / home_odds
                    away_prob = 1 / away_odds
                    
                    # Normalize the probabilities
                    total_prob = home_prob + away_prob
                    home_prob_normalized = home_prob / total_prob
                    away_prob_normalized = away_prob / total_prob
                    
                    # Find matching fights in the database
                    db_date = event_date.date()
                    fights_that_day = self.fetch_fights_that_day(db_date)
                    
                    if not fights_that_day:
                        self.logger.warning(f"No fights found in database for date: {db_date}")
                        continue
                    
                    # Match fighters using fuzzy matching
                    matched_fights = self.match_fights_by_fuzzy(
                        fights_that_day, 
                        home_team, 
                        away_team
                    )
                    
                    # Update the odds for each matched fight
                    for match in matched_fights:
                        fight_id = match['fight_id']
                        fighter1_id = match['fighter1_id']
                        fighter2_id = match['fighter2_id']
                        scenario = match['scenario']
                        
                        self.logger.info(f"Updating fight_id: {fight_id} with scenario {scenario}")
                        
                        if scenario == 1:
                            # Direct match
                            self.upsert_fighter_odds(fight_id, fighter1_id, home_prob, home_prob_normalized)
                            self.upsert_fighter_odds(fight_id, fighter2_id, away_prob, away_prob_normalized)
                        else:
                            # Swapped match
                            self.upsert_fighter_odds(fight_id, fighter1_id, away_prob, away_prob_normalized)
                            self.upsert_fighter_odds(fight_id, fighter2_id, home_prob, home_prob_normalized)
                
        except Exception as e:
            self.logger.error(f"Error scraping odds from API: {str(e)}")
            raise
    
    def fetch_fights_that_day(self, event_date):
        """
        Return a list of dicts for fights matching the event_date +/- 1 day.
        
        Args:
            event_date: The event date to search for
            
        Returns:
            List of dicts with fight information
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
        rows = self.conn.execute(query, {"date": event_date}).fetchall()

        for r in rows:
            results.append({
                "fight_id":       r[0],
                "fighter1_id":    r[1],
                "fighter2_id":    r[2],
                "fighter1_name":  r[3],
                "fighter2_name":  r[4]
            })

        return results
    
    def match_fights_by_fuzzy(self, fights_that_day, csv_f1, csv_f2, threshold=None):
        """
        Match fights using fuzzy string matching.
        
        Args:
            fights_that_day: List of fights on the given day
            csv_f1: First fighter name from CSV
            csv_f2: Second fighter name from CSV
            threshold: Fuzzy matching threshold (default: self.fuzzy_match_threshold)
            
        Returns:
            List of matched fights with scenario information
        """
        if threshold is None:
            threshold = self.fuzzy_match_threshold
            
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
    
    def normalize_implied_probabilities(self, df):
        """
        Given a DataFrame with columns p1_odds, p2_odds that contain probabilities,
        normalize them to ensure they sum to 1.0
        
        Args:
            df: DataFrame with p1_odds and p2_odds columns
            
        Returns:
            DataFrame with normalized probabilities
        """
        # Since we already have probabilities, we just need to normalize them
        df['sum_p'] = df['p1_odds'] + df['p2_odds']
        
        # normalized
        df['p1_prob'] = df['p1_odds'] / df['sum_p']
        df['p2_prob'] = df['p2_odds'] / df['sum_p']
        
        # drop helper column
        df.drop(['sum_p'], axis=1, inplace=True)
        return df
    
    def upsert_fighter_odds(self, fight_id, fighter_id, odds_value, prob_value):
        """
        Upsert odds and probability values for a given fight_id and fighter_id.
        
        Args:
            fight_id: The fight ID
            fighter_id: The fighter ID
            odds_value: The odds value
            prob_value: The probability value
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
            INSERT INTO features.odds (fight_id, fighter_id, event_id, odds, prob)
            VALUES (:fight_id, :fighter_id, :event_id, :odds, :prob)
            ON CONFLICT (fight_id, fighter_id)
            DO UPDATE SET 
                odds = EXCLUDED.odds,
                prob = EXCLUDED.prob
        ''')

        self.conn.execute(query, {
            "fight_id": fight_id,
            "fighter_id": fighter_id,
            "event_id": event_id,
            "odds": odds_value,
            "prob": prob_value
        })

        # Commit the transaction
        self.conn.commit()
    
    def fetch_historical_events(self, date_timestamp):
        """
        Fetch historical events from TheOddsAPI.
        
        Args:
            date_timestamp: The date to fetch events from
            
        Returns:
            List of events
        """
        # Format the date as ISO string
        if isinstance(date_timestamp, datetime):
            date_str = date_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            date_str = datetime.combine(date_timestamp, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        params = {
            'apiKey': self._require_api_key(),
            'date': date_str
        }
        
        response = requests.get(self.historical_events_base_url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to fetch historical events: {response.status_code} - {response.text}")
            return []
    
    def fetch_historical_odds_for_event(self, theoddsapi_event_id, date_timestamp):
        """
        Fetch historical odds for a specific event from TheOddsAPI.
        
        Args:
            theoddsapi_event_id: The event ID from TheOddsAPI
            date_timestamp: The date of the event
            
        Returns:
            List of odds data
        """
        # Format the date as ISO string
        if isinstance(date_timestamp, datetime):
            date_str = date_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            date_str = datetime.combine(date_timestamp, datetime.min.time()).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        url = f"{self.historical_odds_base_url}/{theoddsapi_event_id}/odds"
        params = {
            'apiKey': self._require_api_key(),
            'date': date_str,
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'decimal'
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            self.logger.error(f"Failed to fetch historical odds: {response.status_code} - {response.text}")
            return []

    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator sequentially.
        
        Returns:
            Dictionary with results
        """
        # Create the features.odds table if it doesn't exist
        self.create_odds_table()
        
        # Process the combined_odds.csv file
        self.process_combined_odds_csv()
        
        # Scrape additional odds from TheOddsAPI
        self.scrape_odds_api()
        
        # Return an empty dictionary since we're not returning any DataFrames
        return {}
