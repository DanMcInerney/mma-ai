import pandas as pd
from collections import Counter
from fuzzywuzzy import fuzz
from datetime import datetime, timedelta
import requests
from sqlalchemy import text
import csv
import os


class OddsAPI:
    def __init__(self, initial_date='2020-06-07T00:00:00Z', end_date=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), combined_odds_path="combined_odds.csv", conn=None, api_key=None):
        self.api_key = api_key or os.getenv("THE_ODDS_API_KEY")
        self.initial_date = initial_date
        self.next_scrape_date = datetime.strptime(initial_date, "%Y-%m-%dT%H:%M:%SZ")
        self.previous_scrape_date = self.next_scrape_date
        self.end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
        self.max_positive_odds_threshold = 4000
        self.min_negative_odds_threshold = -4000
        self.conn = conn
                # The base URL for historical odds
        # e.g. GET /v4/historical/sports/mma_mixed_martial_arts/events/{eventId}/odds
        self.historical_odds_base_url = (
            "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/events"
        )
        self.historical_events_base_url = "https://api.the-odds-api.com/v4/historical/sports/mma_mixed_martial_arts/events"
        self.combined_odds_path = combined_odds_path

    def _require_api_key(self):
        if not self.api_key:
            raise RuntimeError("Set THE_ODDS_API_KEY before calling The Odds API.")
        return self.api_key

    def get_odds(self):
        base_url = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds/"
        params = {
            'apiKey': self._require_api_key(),
            'regions': 'us',
            'markets': 'h2h',
            'oddsFormat': 'decimal',
            'date': self.end_date
        }
        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": "Failed to fetch data from the API.",
                "status_code": response.status_code,
                "response_text": response.text
            }
        
    def upsert_fighter_prob(self, fight_id, fighter_id, value, col):
        """
        Upsert pattern for fighter probabilities.
        Updates or inserts probability values for a given fight_id and fighter_id.
        First gets the event_id from fight_mapping since it's required.
        """
        # First get the event_id for this fight
        get_event = text("""
            SELECT event_id 
            FROM features.fight_mapping 
            WHERE fight_id = :fight_id
        """)
        
        event_id = self.conn.execute(get_event, {"fight_id": fight_id}).scalar()
        
        if not event_id:
            print(f"Warning: Could not find event_id for fight_id {fight_id}")
            return

        query = text(f'''
            INSERT INTO model_data.static_stats_raw (fight_id, fighter_id, event_id, {col})
            VALUES (:fight_id, :fighter_id, :event_id, :{col})
            ON CONFLICT (fight_id, fighter_id)
            DO UPDATE SET {col} = EXCLUDED.{col}
        ''')

        self.conn.execute(query, {
            "fight_id": fight_id,
            "fighter_id": fighter_id,
            "event_id": event_id,
            col: value
        })

        # Commit the transaction
        self.conn.commit()

    def get_fighter_avg_odds(self, fighter_name):
        data = self.get_odds()
        all_odds = []

        # Dictionary with name replacements, key = masterml name, val = oddsapi name
        names = {'abus magomedov': 'Abusupyian Magomedov',
                 "khalil rountree jr.": "Khalil Rountree",
                 "jose aldo": "José Aldo",
                 "alexander hernandez": "Alex Hernandez",
                 "ovince saint preux": "Ovince St. Preux",
                 "geoff neal": "Geoffrey Neal",
                 "victor hugo": "Victor Hugo Silva",
                 "song kenan": "Kenan Song",
                 "ian machado garry": "Ian Garry",
                 }
        orig_name = fighter_name
        if fighter_name in names:
            fighter_name = names[fighter_name]

        # Prevent bug where sometimes the fighter_name and date might not be the same fight
        # like an event on 12-10, John Smith vs Jane Doe, but one market has odds for John Smith vs Jill Hanson same date
        # Most results will be for the correct fight, but some might not be so eliminate those
        home = []
        away = []
        remove = []
        for event in data:
            if any(fuzz.ratio(fighter_name.lower(), event_side.lower()) >= 90 for event_side in
                   [event['home_team'], event['away_team']]):
                home.append(event['home_team'])
                away.append(event['away_team'])

        # Count the most common names in home and away
        most_common_home = Counter(home).most_common(1)[0][0] if home else None
        most_common_away = Counter(away).most_common(1)[0][0] if away else None

        for event in data:
            # Check if the fighter is in this event using fuzzy matching
            # We check against both 'home_team' and 'away_team'
            if any(fuzz.ratio(fighter_name.lower(), event_side.lower()) >= 90 for event_side in
                   [event['home_team'], event['away_team']]):
                for bookmaker in event['bookmakers']:
                    for market in bookmaker['markets']:
                        #Make sure we're not getting odds for a different fight
                        out_names = [outcome['name'] for outcome in market['outcomes']]
                        for name in out_names:
                            if not fuzz.ratio(name.lower(), most_common_home.lower()) >= 90 and not fuzz.ratio(name.lower(), most_common_away.lower()) >= 90:
                            #if name != most_common_home and name != most_common_away:
                                continue
                        for outcome in market['outcomes']:
                            # Use fuzzy matching again to match the fighter name with the outcome name
                            if fuzz.ratio(fighter_name.lower(), outcome['name'].lower()) >= 90:
                                all_odds.append(outcome['price'])

        if len(all_odds) > 0:
            avg_odds = sum(all_odds) / len(all_odds)
            name_odds = {orig_name: avg_odds}
            return name_odds
        else:
            print(f'No odds found for {orig_name}')
            return None  # or an appropriate response indicating no matches found
    
    # -------------------------------------------------------------------------
    # 1) Fetch Event Dates
    # -------------------------------------------------------------------------
    def fetch_event_dates_from_db(self):
        """
        Fetch all event dates (and event IDs) from the database >= initial_date.
        """
        # Convert string date to datetime object first
        initial_date = datetime.strptime(self.initial_date, "%Y-%m-%dT%H:%M:%SZ").date()
        
        query = text('''
            SELECT event_id, event_date
            FROM features.event_mapping
            WHERE event_date >= :initial_date
            ORDER BY event_date ASC
        ''')

        event_list = []
        rows = self.conn.execute(query, {"initial_date": initial_date}).fetchall()
        for row in rows:
            event_list.append({
                "event_id": row[0],  # First element of tuple
                "event_date": row[1]  # Second element of tuple
            })
        return event_list

    # -------------------------------------------------------------------------
    # 2) Fetch Fight Data for Each Event
    # -------------------------------------------------------------------------
    def fetch_fights_for_event(self, event_id):
        """
        For a given event_id, returns a list of dicts:
        [
           {
              "fight_id": ...,
              "fighter1_name": ...,
              "fighter2_name": ...
              ...
           },
           ...
        ]
        """
        query = text('''
            SELECT
                f.fight_id,
                f.event_id,
                fm1.fighter_name AS fighter1_name,
                fm2.fighter_name AS fighter2_name
            FROM features.fight_mapping f
            INNER JOIN features.fighter_mapping fm1 ON f.fighter1_id = fm1.fighter_id
            INNER JOIN features.fighter_mapping fm2 ON f.fighter2_id = fm2.fighter_id
            WHERE f.event_id = :event_id
        ''')

        fights = []
        rows = self.conn.execute(query, {"event_id": event_id}).fetchall()
        for row in rows:
            fights.append({
                "fight_id": row[0],      # fight_id
                "event_id": row[1],      # event_id
                "fighter1_name": row[2],  # fighter1_name
                "fighter2_name": row[3]   # fighter2_name
            })
        return fights
    
    def fetch_historical_events(self, date_timestamp):
        """
        Calls The Odds API's 'GET /v4/historical/sports/mma_mixed_martial_arts/events'
        with the given date_timestamp, returning the JSON.
        
        date_timestamp: Python datetime
        """
        endpoint = self.historical_events_base_url
        params = {
            "apiKey": self._require_api_key(),
            "date": date_timestamp  # "YYYY-MM-DDTHH:MM:SSZ"
        }
        resp = requests.get(endpoint, params=params)
        if resp.status_code == 200:
            json = resp.json()
            if len(json['data']) == 0:
                print(f"No historical events found for date {date_timestamp}")
                return None
            return resp.json()  # Will contain { "timestamp":..., "data": [...] }
        else:
            print(f"Error fetching historical events: {resp.status_code} - {resp.text}")
            return None
        
    # -------------------------------------------------------------------------
    # 3) Fetch Historical Odds for an Event at a Specific Timestamp
    # -------------------------------------------------------------------------

    def fetch_historical_odds_for_event(self, theoddsapi_event_id, date_timestamp):
        """
        For the real TheOddsAPI event ID, calls:
          GET /v4/historical/sports/mma_mixed_martial_arts/events/{eventId}/odds
          ?apiKey=...&date=...
        Returns the JSON or None if error.
        """
        endpoint = f"{self.historical_odds_base_url}/{theoddsapi_event_id}/odds"
        params = {
            "apiKey": self._require_api_key(),
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "date": date_timestamp
        }
        resp = requests.get(endpoint, params=params)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Failed to fetch historical odds for TheOddsAPI event={theoddsapi_event_id}, date={date_timestamp} "
                  f"Code: {resp.status_code}, Text: {resp.text}")
            return None

    def append_odds_to_csv(self, snapshot_timestamp, fighter1_name, fighter2_name, fighter1_odds, fighter2_odds):
        """Write a row to the CSV if it doesn't already exist."""
        import csv
        
        # Format the timestamp consistently
        formatted_ts = snapshot_timestamp.strftime("%Y-%m-%d %H:%M:%S+00:00")
        new_row = [formatted_ts, fighter1_name, fighter2_name, fighter1_odds, fighter2_odds]
        print(new_row)
        
        # Check if file exists and read existing rows
        try:
            with open(self.combined_odds_path, mode="r", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file)
                existing_rows = list(reader)
        except FileNotFoundError:
            existing_rows = []
        
        # Check if this exact row already exists
        if new_row not in existing_rows:
            with open(self.combined_odds_path, mode="a", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(new_row)
                print(f"Added new odds entry for {fighter1_name} vs {fighter2_name}")
        else:
            print(f"Skipping duplicate odds entry for {fighter1_name} vs {fighter2_name}")

    # -------------------------------------------------------------------------
    # 5) Main Orchestrator: Update Historical Odds from 2020-01-01 to present
    # -------------------------------------------------------------------------
    def update_historical_odds(self):
        """
        Main orchestrator: 
         1. Get DB events >= initial_date
         2. For each DB event_date, fetch TheOddsAPI historical events
         3. Match TheOddsAPI events to local DB fights (by fighter name, date/time)
         4. For each match, get TheOddsAPI event's odds and append to CSV.
        """
        db_events = self.fetch_event_dates_from_db()

        for db_evt in db_events:
            db_event_id  = db_evt["event_id"]    # DB's event_id (no correlation to TheOddsAPI)
            db_event_date= db_evt["event_date"]  # type is date
            date_timestamp = datetime.combine(db_event_date, datetime.min.time())
            formatted_date = date_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')


            # 1) Fetch the local fights for this DB event
            fights = self.fetch_fights_for_event(db_event_id)
            if not fights:
                continue

            # 2) Call The Odds API to get all historical events for this date
            #    (We use the same date as the DB event. You could offset by some hours if needed.)
            historical_events_response = self.fetch_historical_events(formatted_date)
            if not historical_events_response or "data" not in historical_events_response:
                continue

            theoddsapi_events = historical_events_response["data"]
            snapshot_ts_str   = historical_events_response.get("timestamp")

            if snapshot_ts_str:
                snapshot_ts = datetime.fromisoformat(snapshot_ts_str.replace("Z", "+00:00"))
            else:
                snapshot_ts = formatted_date

            # 3) For each fight in the DB event, try to match to an event in TheOddsAPI data
            for fight in fights:
                f1 = fight["fighter1_name"]
                f2 = fight["fighter2_name"]
                
                # We'll look for a TheOddsAPI event whose home_team/away_team
                # matches one of our fighters (by fuzzy logic).
                matched_event_id = None  # TheOddsAPI event "id"
                
                for odds_evt in theoddsapi_events:
                    home_team = odds_evt["home_team"]
                    away_team = odds_evt["away_team"]

                    # Simple fuzzy approach: each fighter name must match either the home or away
                    # in some order. We also need to ensure both fighters match in the same event.
                    # For example:
                    ratio_f1_home = fuzz.ratio(f1.lower(), home_team.lower())
                    ratio_f1_away = fuzz.ratio(f1.lower(), away_team.lower())
                    ratio_f2_home = fuzz.ratio(f2.lower(), home_team.lower())
                    ratio_f2_away = fuzz.ratio(f2.lower(), away_team.lower())

                    # We consider a match if the best assignment of (f1->home/away, f2->home/away) is above some threshold
                    # This is just one possible logic. Tweak thresholds as needed:
                    if ((ratio_f1_home > 75 and ratio_f2_away > 75) or 
                        (ratio_f1_away > 75 and ratio_f2_home > 75)):
                        matched_event_id = odds_evt["id"]
                        break

                # If found a matched event, now fetch the actual odds for that event
                if matched_event_id:
                    odds_data = self.fetch_historical_odds_for_event(matched_event_id, formatted_date)
                    if not odds_data or "data" not in odds_data:
                        continue

                    # Parse the odds structure
                    # odds_data looks like:
                    # {
                    #   "timestamp": ...
                    #   "data": {
                    #       "id": "...",
                    #       "bookmakers": [
                    #           {
                    #               "key": "...",
                    #               "markets": [
                    #                   {
                    #                      "key": "h2h",
                    #                      "outcomes": [{"name": "...", "price": 1.9}, ...]
                    #                   }
                    #               ]
                    #           }
                    #       ]
                    #   }
                    # }
                    snapshot_ts_str2 = odds_data.get("timestamp")
                    if snapshot_ts_str2:
                        snapshot_ts2 = datetime.fromisoformat(snapshot_ts_str2.replace("Z", "+00:00"))
                    else:
                        snapshot_ts2 = snapshot_ts

                    f1_odds_list = []
                    f2_odds_list = []

                    bookmakers = odds_data["data"].get("bookmakers", [])
                    for bookmaker in bookmakers:
                        for market in bookmaker.get("markets", []):
                            if market["key"] == "h2h":
                                for outcome in market.get("outcomes", []):
                                    fighter_name = outcome["name"]
                                    price = outcome["price"]

                                    # Fuzzy match again for fighter name
                                    if fuzz.ratio(fighter_name.lower(), f1.lower()) > 75:
                                        f1_odds_list.append(price)
                                    elif fuzz.ratio(fighter_name.lower(), f2.lower()) > 75:
                                        f2_odds_list.append(price)

                    # Calculate average odds if we have odds from any bookmakers
                    f1_avg_odds = sum(f1_odds_list) / len(f1_odds_list) if f1_odds_list else None
                    f2_avg_odds = sum(f2_odds_list) / len(f2_odds_list) if f2_odds_list else None

                    # If we have odds for both fighters, write to CSV
                    if f1_avg_odds is not None and f2_avg_odds is not None:
                        f1_avg_odds = self.convert_odds(f1_avg_odds, 'probability')
                        f2_avg_odds = self.convert_odds(f2_avg_odds, 'probability')
                        print(f"Writing odds for {f1} vs {f2} in event {matched_event_id}, {f1_avg_odds} vs {f2_avg_odds}")
                        self.append_odds_to_csv(
                            snapshot_timestamp=snapshot_ts2,
                            fighter1_name=f1,
                            fighter2_name=f2,
                            fighter1_odds=f1_avg_odds,
                            fighter2_odds=f2_avg_odds
                        )
                    else:
                        print(f"No odds found for {f1} vs {f2} in event {matched_event_id}")

        print("Historical odds update complete!")

    def normalize_implied_probabilities(self, df):
        """
        Given a DataFrame with columns p1_odds, p2_odds that contain probabilities,
        normalize them to ensure they sum to 1.0
        """
        # Since we already have probabilities, we just need to normalize them
        df['sum_p'] = df['p1_odds'] + df['p2_odds']
        
        # normalized
        df['p1_prob'] = df['p1_odds'] / df['sum_p']
        df['p2_prob'] = df['p2_odds'] / df['sum_p']
        
        # drop helper column
        df.drop(['sum_p'], axis=1, inplace=True)
        return df

    def convert_odds(self, odds, target_format):
        """
        Convert odds between different formats.
        
        Args:
            odds (float/int): Odds in decimal, american, or implied probability format
            target_format (str): 'decimal', 'american', or 'probability'
            
        Returns:
            float: Converted odds in requested format
        """
        # First convert input to decimal odds
        if isinstance(odds, str):
            odds = float(odds)
            
        # Detect input format
        if odds > 0 and odds <= 1:  # Input is probability
            decimal = 1 / odds
        elif odds > 1 and odds < 99:  # Input is decimal
            decimal = odds
        else:  # Input is american
            if odds > 0:
                decimal = (odds / 100) + 1
            else:
                decimal = 1 - (100 / odds)

        # Convert decimal to target format
        if target_format.lower() == 'decimal':
            return round(decimal, 4)
        elif target_format.lower() == 'probability':
            return round(1 / decimal, 4)
        elif target_format.lower() == 'american':
            if decimal >= 2:
                return round((decimal - 1) * 100, 4)
            else:
                return round(-100 / (decimal - 1), 4)
        else:
            raise ValueError("target_format must be 'decimal', 'american', or 'probability'")
    
    def convert_odds_df(self, df: pd.DataFrame, target_format: str) -> pd.DataFrame:
        df['odds'] = df['odds'].apply(lambda x: self.convert_odds(x, target_format))
        df['prob'] = df['prob'].apply(lambda x: self.convert_odds(x, target_format))
        return df
        

    def ensure_odds_column_exists(self):
        """
        Ensures the odds and prob columns exist in model_data.static_stats_raw.
        Creates them if they don't exist.
        """
        # Check for odds column
        check_odds = text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_schema = 'model_data'
                AND table_name = 'static_stats_raw'
                AND column_name = 'odds'
            );
        """)
        
        # Check for prob column
        check_prob = text("""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.columns 
                WHERE table_schema = 'model_data'
                AND table_name = 'static_stats_raw'
                AND column_name = 'prob'
            );
        """)
        
        odds_exists = self.conn.execute(check_odds).scalar()
        prob_exists = self.conn.execute(check_prob).scalar()
        
        if not odds_exists:
            print("Adding odds column to model_data.static_stats_raw...")
            create_odds = text("""
                ALTER TABLE model_data.static_stats_raw
                ADD COLUMN odds FLOAT;
            """)
            self.conn.execute(create_odds)
            self.conn.commit()
            print("odds column added successfully!")
            
        if not prob_exists:
            print("Adding prob column to model_data.static_stats_raw...")
            create_prob = text("""
                ALTER TABLE model_data.static_stats_raw
                ADD COLUMN prob FLOAT;
            """)
            self.conn.execute(create_prob)
            self.conn.commit()
            print("prob column added successfully!")

# if __name__ == "__main__":
#     historical_csv_path = "my_historical_odds.csv"
#     api_key = "YOUR_ODDS_API_KEY"

#     # Initialize your class
#     odds_api = OddsAPI(
#         conn=conn,
#         historical_csv_path=historical_csv_path,
#         initial_date="2020-01-01T00:00:00Z"
#     )
    
#     # Run the main update
#     odds_api.update_historical_odds()
