import pandas as pd
import json
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy_utils import database_exists, create_database
from curl_cffi import requests
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from contextlib import contextmanager
import argparse
import sys
import os
from libs.paths import database_url, odds_database_url

# Path for the timestamp file
LAST_SCRAPE_TIMESTAMP_FILE = 'odds_last_scrape.txt'

class BFOScraper:
    """
    A scraper class for bestfightodds.com that retrieves betting odds for fighters
    and stores them in a dedicated database.
    
    Features:
    - Scrape historical odds data for all fighters
    - Scrape odds data for specific fighters
    - Get latest odds for specific fight matchups with vig removal
    - Handle name mappings between UFCStats and BestFightOdds formats
    - Manual input fallback for missing fighters
    """
    
    # Hardcoded name mappings from UFCStats to BestFightOdds
    NAME_MAPPINGS = {
        "yadier delvalle": "yadier del valle",
        "jose medina": "jose daniel medina",
        "aj matthews": "a.j. matthews",
        "aj mckee": "a.j. mckee",
        "alatengheili": "heili alateng",
        "alavutdin gadjiev": "alavutdin gadzhiyev",
        "andre amado": "andre amade",
        "antonio dos santos": "antonio dos santos jr.",
        "artenus young": "artenas young",
        "asu almabayev": "asu almabaev",
        "bazigit atajev": "bazigit ataev",
        "bj penn": "b.j. penn",
        "bret bergmark": "brett bergmark",
        "cj keith": "c.j. keith",
        "dan manasoiu": "daniel manasoiu",
        "dmitry sosnovskiy": "dmitriy sosnovskiy",
        "henrique shiguemoto": "henrique shigemoto",
        "jaime alvarez": "jamie alvarez",
        "jj ambrose": "j.j. ambrose",
        "jonathan piersma": "johnathan piersma",
        "joshua burkman": "josh burkman",
        "joshua wang-kim": "josh wang-kim",
        "kenyon jackson": "kenan jackson",
        "kyeungpyo kim": "kyung pyo kim",
        "luiz cane": "luis cane",
        "michael aswell jr.": "michael aswell",
        "ovince saint preux": "ovince st. preux",
        "rafael freitas": "rafael de freitas",
        "reyes cortez jr.": "reyes cortez",
        "rilley dutro": "riley dutro",
        "robert sanchez": "roberto sanchez",
        "rodrigo de lima": "rodrigo lima",
        "sangwon kim": "sang won kim",
        "steve kennedy": "steven kennedy",
        "suyoung you": "suyoung yu",
        "talisson teixeira": "tallison teixeira",
        "tj waldburger": "t.j. waldburger",
        "zachary micklewright": "zach micklewright",
        # Include the manual mappings added earlier
        "jacare souza": "ronaldo souza",
        "aj fonseca": "a.j. fonseca",
        "jc cottrell": "j.c. cottrell",
        "tj dillashaw": "t.j. dillashaw",
        'cb dollaway': 'c.b. dollaway',
        "alex torres": "alexander torres",
        "alvaro herrera mendoza": "alvaro herrera",
        "batgerel danaa": "danaa batgerel",
        "benoit saint denis": "benoit saint-denis",
        'da woon jung': 'da un jung',
        'junyong park': 'jun yong park',
        "changho lee": "chang ho lee",
        "jesus aguilar": "jesus santos aguilar",
        'donghun choi': "dong hoon choi",
        "jianping yang": "yang jianping",
        'jp buys': 'j.p. buys',
        'montserrat conejo ruiz': 'montserrat conejo',
        'mizuki': 'mizuki inoue',
        'tj grant': 't.j. grant',
        'raul rosas jr.': 'raul rosas jr',
    }
    
    # Build the reverse mapping dictionary (BFO → UFCStats)
    REVERSE_MAPPINGS = {v.lower(): k.lower() for k, v in NAME_MAPPINGS.items()}
    
    # BFO aliases that can appear as duplicate fighter pages or opponent rows.
    # Keys are UFCStats names; values are extra BestFightOdds names to try when
    # searching, reverse-mapping scraped opponents, and normalizing before save.
    DUPE_NAMES = {"timmy cuamba": ['timothy cuamba'],
                  'asu almabayev': ['asu almabaev'],
                  'song kenan': ['kenan song'],
                  'geoff neal': ['geoffrey neal'],
                  'nate maness': ['nathan maness'],
                  'daniel lacerda': ['daniel da silva lacerda'],
                  'alexandr romanov': ['alexander romanov'],
                  'jj aldrich': ['j.j. aldrich'],
                  'raul rosas jr': ['raul rosas jr.'],
                  'ricky glenn': ['rick glen'],
                  'diego ferreira': ['carlos diego ferreira'],
                  'maheshate': ['hayisaer maheshate', 'maheshate hayisaer'],
                  'elizeu zaleski': ['elizeu zaleski dos santos'],
                  'aoriqileng': ['aori qileng', 'qileng aori'],
                  'dan argueta': ['daniel argueta', 'daniel argueta.'],
                  'phil rowe': ['philip rowe', 'phillip rowe'],
                  'abus magomedov': ['abusupiyan magomedov'],
                  'elves brener': ['elves brenner', 'elves brenner.'],
                  'yanal ashmouz': ['yanal ashmoz'],
                  'aj dobson': ['a.j. dobson'],
                  'chepe mariscal': ['jose mariscal'],
                  'aj fletcher': ['a.j. fletcher'],
                  'muhammad naimov': ['muhammadjon naimov'],
                  'sergei pavlovich': ['sergey pavlovich'],
                  'sumudaerji': ['sumudaerji sumudaerji', 'su mudaerji'],
                  'hyunsung park': ['hyun sung park', 'park hyun-sung'],
                  'charles radtke': ['charlie radtke'],
                  'jeongyeong lee': ['lee jeong-yeong'],
                  'steve erceg': ['stephen erceg.', 'stephen erceg'],
                  "dooho choi": ['doo ho choi', 'choi doo ho'],
                  # Add apostrophe name mappings
                  "brendan o'reilly": ['brendan oreilly'],
                  "sean o'malley": ['sean omalley'],
                  "sean o'connell": ['sean oconnell'],
                  "chuck o'neil": ['chuck oneil'],
                  "anthony o'connor": ['anthony oconnor'],
                  "da'mon blackshear": ['damon blackshear'],
                  "don'tale mayes": ['dontale mayes'],
                  "ode osbourne": ["ode' osbourne"],
                  "jake o'brien": ['jake obrien'],
                  "kj noons": ['k.j. noons'],
    }
    
    def __init__(self, conn, odds_db_url=None):
        """
        Initialize the BFOScraper.
        
        Args:
            conn: Database connection object to the main MMA database
            odds_db_url: Connection string for the dedicated odds database
        """
        self.mma_conn = conn  # Connection to the main MMA database
        self.odds_db_url = odds_db_url or odds_database_url()
        self.odds_engine = self._create_odds_engine()
        self.base_url = "https://www.bestfightodds.com"
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Connection": "keep-alive",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
        }
        
        # Track not found fighters
        self.not_found_fighters = set()
        
        # Initialize database schema and tables
        self._initialize_database()
    
    def _create_odds_engine(self):
        """Create and configure the database engine for the odds database"""
        if not database_exists(self.odds_db_url):
            create_database(self.odds_db_url)
        
        return create_engine(
            self.odds_db_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=14400
        )
    
    @contextmanager
    def _get_odds_connection(self):
        """Context manager for connections to the odds database"""
        connection = self.odds_engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    def _initialize_database(self):
        """Create the odds database with bestfightodds schema and bfo table"""
        try:
            with self._get_odds_connection() as conn:
                # Create bestfightodds schema if it doesn't exist
                conn.execute(text('CREATE SCHEMA IF NOT EXISTS bestfightodds;'))
                
                # Create bfo table
                conn.execute(text('''
                    CREATE TABLE IF NOT EXISTS bestfightodds.bfo (
                        id SERIAL PRIMARY KEY,
                        fighter VARCHAR(255) NOT NULL,
                        fighter_url VARCHAR(255),
                        opponent VARCHAR(255) NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        odds FLOAT NOT NULL,
                        UNIQUE(fighter, opponent, timestamp)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_bfo_fighter ON bestfightodds.bfo(fighter);
                    CREATE INDEX IF NOT EXISTS idx_bfo_fighter_url ON bestfightodds.bfo(fighter_url);
                    CREATE INDEX IF NOT EXISTS idx_bfo_timestamp ON bestfightodds.bfo(timestamp);
                '''))
                
                conn.commit()
                print("Odds database, schema and BFO table initialized successfully.")
        
        except Exception as e:
            print(f"Error initializing odds database: {str(e)}")
            raise
    
    # --- Timestamp Handling ---
    
    def _read_last_scrape_timestamp(self):
        """Reads the last scrape timestamp from the file."""
        if not os.path.exists(LAST_SCRAPE_TIMESTAMP_FILE):
            print(f"Timestamp file '{LAST_SCRAPE_TIMESTAMP_FILE}' not found. Will scrape all fighters.")
            return None
        try:
            with open(LAST_SCRAPE_TIMESTAMP_FILE, 'r') as f:
                timestamp_str = f.read().strip()
                last_scrape_time = datetime.fromisoformat(timestamp_str)
                print(f"Found last scrape timestamp: {last_scrape_time}")
                return last_scrape_time
        except Exception as e:
            print(f"Error reading timestamp file '{LAST_SCRAPE_TIMESTAMP_FILE}': {str(e)}. Will scrape all fighters.")
            return None

    def _write_last_scrape_timestamp(self, timestamp):
        """Writes the given timestamp to the file."""
        try:
            with open(LAST_SCRAPE_TIMESTAMP_FILE, 'w') as f:
                f.write(timestamp.isoformat())
            print(f"Updated last scrape timestamp to: {timestamp.isoformat()}")
        except Exception as e:
            print(f"Error writing timestamp file '{LAST_SCRAPE_TIMESTAMP_FILE}': {str(e)}")
            
    # --- End Timestamp Handling ---
    
    def get_mapped_name(self, fighter_name):
        """
        Check if there's a mapping for this fighter name in the hardcoded mappings.
        
        Args:
            fighter_name: UFC Stats fighter name
            
        Returns:
            The mapped Best Fight Odds name or the original name if no mapping exists
        """
        if fighter_name is None:
            return None

        # Check the hardcoded mappings
        if fighter_name.lower() in self.NAME_MAPPINGS:
            mapped_name = self.NAME_MAPPINGS[fighter_name.lower()]
            print(f"Using hardcoded mapping: '{fighter_name}' → '{mapped_name}'")
            return mapped_name
        
        # No mapping found, return the original name
        return fighter_name
    
    def reverse_map_name(self, bfo_name):
        """
        Check if there's a reverse mapping for this BFO fighter name.
        
        Args:
            bfo_name: Best Fight Odds fighter name
            
        Returns:
            The mapped UFCStats name or the original name if no mapping exists
        """
        if bfo_name is None:
            return None

        # First check the regular reverse mappings
        if bfo_name.lower() in self.REVERSE_MAPPINGS:
            ufcstats_name = self.REVERSE_MAPPINGS[bfo_name.lower()]
            print(f"Using reverse mapping: '{bfo_name}' → '{ufcstats_name}'")
            return ufcstats_name
        
        # Then check if this name is in any of the DUPE_NAMES values
        for ufcstats_name, alt_names in self.DUPE_NAMES.items():
            if bfo_name.lower() in [alt.lower() for alt in alt_names]:
                print(f"Using DUPE_NAMES mapping: '{bfo_name}' → '{ufcstats_name}'")
                return ufcstats_name
        
        # No mapping found, return the original name
        return bfo_name
    
    def get_fighter_link(self, name):
        """
        Get the bestfightodds.com link for a fighter.
        
        Args:
            name: Fighter name to search for
            
        Returns:
            The fighter's URL path or None if not found
        """
        # Check if there's a mapping for this fighter name
        search_name = self.get_mapped_name(name)
        
        # If the fighter is in the skip list, search_name will be None
        if search_name is None:
            return None
        
        url = f"{self.base_url}/search"
        search_params = {'query': search_name}
        
        try:
            # First try with the standard/mapped name
            search_page = requests.get(
                url, 
                params=search_params, 
                impersonate="chrome", 
                headers=self.default_headers, 
                timeout=5
            )
            
            soup = BeautifulSoup(search_page.text, 'html.parser')
            text_to_find = re.compile(search_name, re.IGNORECASE)
            link = soup.find('a', string=text_to_find)
            
            # If found with the primary name, return it
            if link:
                return link['href']
            
            # If not found with the primary name but the fighter is in DUPE_NAMES,
            # try the alternate names
            if name.lower() in self.DUPE_NAMES:
                for alt_name in self.DUPE_NAMES[name.lower()]:
                    print(f"Trying alternate name: {alt_name} for {name}")
                    search_params = {'query': alt_name}
                    
                    alt_search_page = requests.get(
                        url, 
                        params=search_params, 
                        impersonate="chrome", 
                        headers=self.default_headers, 
                        timeout=5
                    )
                    
                    alt_soup = BeautifulSoup(alt_search_page.text, 'html.parser')
                    alt_text_to_find = re.compile(alt_name, re.IGNORECASE)
                    alt_link = alt_soup.find('a', string=alt_text_to_find)
                    
                    if alt_link:
                        print(f"Found using alternate name: {alt_name}")
                        return alt_link['href']
            
            # If still not found, add to not found fighters
            print(f"Fighter {search_name} not found")
            self.not_found_fighters.add(name)
            return None
                
        except Exception as e:
            print(f"Error getting fighter link for {search_name}: {str(e)}")
            # Add to not found fighters set in case of errors too
            self.not_found_fighters.add(name)
            return None
    
    def sanitize_name(self, name):
        """
        Clean a fighter name by removing @ symbols only.
        
        Args:
            name: Fighter name to sanitize
            
        Returns:
            Name with @ symbols removed
        """
        if not name:
            return name
            
        # Only remove @ symbols
        sanitized = name.replace('@', '')
        
        return sanitized
        
    def parse_fighter_page(self, fighterlink):
        """
        Parse a fighter's page to get opponent information and odds data.
        
        Args:
            fighterlink: Fighter's URL path
            
        Returns:
            List of dictionaries with opponent names and data IDs
        """
        if not fighterlink:
            return []
            
        try:
            url = f"{self.base_url}{fighterlink}"
            fighter_page = requests.get(
                url, 
                impersonate="chrome", 
                headers=self.default_headers, 
                timeout=5
            )
            
            soup = BeautifulSoup(fighter_page.text, 'html.parser')
            results = []
            rows = soup.find_all('tr')
            
            for i, row in enumerate(rows):
                if row.get('class') and row.get('class')[0] == "main-row":
                    opp_row = rows[i + 1]
                    opp_name = opp_row.find('a').text if opp_row.find('a') else None
                    
                    # Sanitize opponent name to remove any special characters
                    if opp_name:
                        opp_name = self.sanitize_name(opp_name)
                        
                    span = row.find('span', class_='teamPercChange')
                    data_li = span['data-li'] if span else None
                    
                    if opp_name and data_li:
                        data_arr = list(map(int, data_li.strip("[]").split(',')))
                        results.append({'opp_name': opp_name, 'data_li': data_arr})
            
            return results
            
        except Exception as e:
            print(f"Error parsing fighter page {fighterlink}: {str(e)}")
            return []
    
    def decode_base64(self, to_decode):
        """
        Decode the base64-encoded odds data from the API.
        
        Args:
            to_decode: Encoded data string
            
        Returns:
            Decoded JSON data
        """
        l = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
        decoded_text = ""
        base64_index = 0
        
        # Remove invalid characters
        to_decode = ''.join(c for c in to_decode if c in l)

        # Base64 decoding
        while base64_index < len(to_decode):
            binary_string = (l.index(to_decode[base64_index]) << 2) | (l.index(to_decode[base64_index + 1]) >> 4)
            next_index = (l.index(to_decode[base64_index + 1]) & 15) << 4 | (l.index(to_decode[base64_index + 2]) >> 2)
            emotional_resp = (l.index(to_decode[base64_index + 2]) & 3) << 6 | l.index(to_decode[base64_index + 3])

            decoded_text += chr(binary_string)

            if l.index(to_decode[base64_index + 2]) != 64:
                decoded_text += chr(next_index)

            if l.index(to_decode[base64_index + 3]) != 64:
                decoded_text += chr(emotional_resp)

            base64_index += 4

        # UTF-8 decoding
        base64_to_text = ""
        base64_decoder = 0

        while base64_decoder < len(decoded_text):
            char_code_at_64 = ord(decoded_text[base64_decoder])

            if char_code_at_64 < 128:
                base64_to_text += chr(char_code_at_64)
                base64_decoder += 1
            elif 191 < char_code_at_64 < 224:
                c2 = ord(decoded_text[base64_decoder + 1])
                base64_to_text += chr(((char_code_at_64 & 31) << 6) | (c2 & 63))
                base64_decoder += 2
            else:
                c2 = ord(decoded_text[base64_decoder + 1])
                c3 = ord(decoded_text[base64_decoder + 2])
                base64_to_text += chr(((char_code_at_64 & 15) << 12) | ((c2 & 63) << 6) | (c3 & 63))
                base64_decoder += 3

        # Final decoding step
        decode_base64_p = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        base64_decoded = ""
        decode_base64_h = len(decode_base64_p)
        
        for char in base64_to_text:
            dynamic_sort = decode_base64_p.find(char)
            if dynamic_sort >= 0:
                char = decode_base64_p[(dynamic_sort + decode_base64_h // 2) % decode_base64_h]
            base64_decoded += char
            
        return json.loads(base64_decoded)
    
    def get_perc_change(self, m, p):
        """
        Get the percentage change data for a specific match.
        
        Args:
            m: Match ID
            p: Position ID
            
        Returns:
            Odds data for the match
        """
        try:
            url = f"{self.base_url}/api/ggd"
            search_params = {'m': m, 'p': p}
            encrypted_resp = requests.get(
                url, 
                params=search_params, 
                impersonate="chrome", 
                headers=self.default_headers, 
                timeout=5
            )
            
            fight_data = self.decode_base64(encrypted_resp.text)
            return fight_data[0]["data"]
            
        except Exception as e:
            print(f"Error getting percentage change data: {str(e)}")
            return []
    
    def get_fighter_url(self, fighter_name):
        """
        Get the fighter_url from the main database for a given fighter name.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            The fighter_url or None if not found
        """
        try:
            result = self.mma_conn.execute(
                text("""
                    SELECT fighter_url 
                    FROM features.fighter_mapping 
                    WHERE LOWER(fighter_name) = LOWER(:fighter_name)
                    LIMIT 1
                """),
                {"fighter_name": fighter_name}
            )
            row = result.fetchone()
            return row[0] if row else None
        except Exception as e:
            print(f"Error getting fighter_url for {fighter_name}: {str(e)}")
            return None
    
    def get_odds_data(self, fighter_name):
        """
        Generate a DataFrame of odds data for a fighter.
        
        Args:
            fighter_name: Name of the fighter to get odds for
            
        Returns:
            DataFrame with odds data for all of the fighter's opponents
        """
        try:
            # Initialize a list to store all odds data
            all_odds_data = []
            names_to_search = []
            
            # Always add the primary fighter name
            names_to_search.append(fighter_name)
            
            # Add alternate names from DUPE_NAMES if available
            if fighter_name.lower() in self.DUPE_NAMES:
                names_to_search.extend(self.DUPE_NAMES[fighter_name.lower()])
                
            found_any = False
            
            # Search for each name variation
            for search_name in names_to_search:
                print(f"Searching for odds data for '{search_name}'")
                
                fighter_link = self.get_fighter_link(search_name)
                if not fighter_link:
                    print(f"Fighter '{search_name}' not found on bestfightodds.com")
                    continue
                    
                matches = self.parse_fighter_page(fighter_link)
                if not matches:
                    print(f"No matches found on fighter page for '{search_name}'")
                    continue
                
                print(f"Found {len(matches)} opponent matches for '{search_name}'")
                
                # Process match data for this name variant
                found_any = True
                for i, match_dict in enumerate(matches):
                    opp_name = match_dict['opp_name']
                    print(f"  Match {i+1}: vs {opp_name}")
                    
                    odds_data = self.get_perc_change(match_dict['data_li'][0], match_dict['data_li'][1])
                    print(f"    Found {len(odds_data)} odds data points")
                    
                    if odds_data:
                        # Get date range for this opponent
                        timestamps = [datetime.fromtimestamp(odds_dict['x'] / 1000) for odds_dict in odds_data]
                        if timestamps:
                            min_date = min(timestamps)
                            max_date = max(timestamps)
                            print(f"    Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                    
                    for odds_dict in odds_data:
                        ts = datetime.fromtimestamp(odds_dict['x'] / 1000)
                        all_odds_data.append({
                            'competitor': opp_name,
                            'timestamp': ts,
                            'odds': odds_dict['y']
                        })
            
            # If we didn't find odds for any name variation, return empty DataFrame
            if not found_any or not all_odds_data:
                print(f"No odds data found for {fighter_name} or any of its alternate names")
                return pd.DataFrame()

            # Create DataFrame from all odds data
            master_df = pd.DataFrame(all_odds_data)
            
            # Show overall date range
            if not master_df.empty:
                min_date = master_df['timestamp'].min()
                max_date = master_df['timestamp'].max()
                print(f"Overall odds data date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                print(f"Total odds records found: {len(master_df)}")
                
                # Show opponent summary
                opponent_counts = master_df['competitor'].value_counts()
                print(f"Opponents found: {', '.join(opponent_counts.head(5).index.tolist())}")
            
            # Add fighter name for database insertion - always use the primary UFCStats name
            master_df['fighter'] = fighter_name.lower()
            
            # Get fighter URL from the main database
            fighter_url = self.get_fighter_url(fighter_name)
            master_df['fighter_url'] = fighter_url
            
            # Process opponent names - convert BFO names to UFCStats names
            master_df['opponent'] = master_df['competitor'].apply(
                lambda x: self.reverse_map_name(x)
            ).str.lower()
            
            # Verify that all opponent and fighter names follow UFCStats format by checking
            # if any of them match BFO names in our mappings
            bfo_names = set(self.REVERSE_MAPPINGS.keys())
            for dupe_list in self.DUPE_NAMES.values():
                bfo_names.update([name.lower() for name in dupe_list])
            
            # Check opponent names
            for opp in master_df['opponent'].unique():
                if opp in bfo_names:
                    print(f"WARNING: Opponent name '{opp}' appears to be in BFO format, not UFCStats format!")
                    
            # Check fighter name too
            if fighter_name.lower() in bfo_names:
                print(f"WARNING: Fighter name '{fighter_name}' appears to be in BFO format, not UFCStats format!")
                # Since fighter name came from the caller, we should fix it now
                correct_name = self.reverse_map_name(fighter_name)
                print(f"Should be '{correct_name}' instead")
                # We don't update it here because we're just reporting the issue
                # The fix happens in save_to_database
            
            # Drop the temporary column
            master_df.drop('competitor', axis=1, inplace=True)
            
            return master_df
            
        except Exception as e:
            print(f"Error getting odds data for {fighter_name}: {str(e)}")
            return pd.DataFrame()
    
    def save_to_database(self, df):
        """
        Save odds data to the database.
        
        Args:
            df: DataFrame containing odds data
            
        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0
            
        try:
            print(f"\nAttempting to save {len(df)} records to database...")
            
            # Show date range of records being saved
            if not df.empty:
                min_date = df['timestamp'].min()
                max_date = df['timestamp'].max()
                print(f"Records date range: {min_date.strftime('%Y-%m-%d %H:%M:%S')} to {max_date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Final verification that all names are in UFCStats format
            # Get all BFO format names for checking
            bfo_names = set(self.REVERSE_MAPPINGS.keys())
            for dupe_list in self.DUPE_NAMES.values():
                bfo_names.update([name.lower() for name in dupe_list])
            
            # Check fighter names
            for fighter in df['fighter'].unique():
                # First sanitize the name
                sanitized_fighter = self.sanitize_name(fighter)
                if sanitized_fighter != fighter:
                    print(f"Sanitizing fighter name: '{fighter}' → '{sanitized_fighter}'")
                    df.loc[df['fighter'] == fighter, 'fighter'] = sanitized_fighter
                    fighter = sanitized_fighter
                
                # Then check if it's in BFO format    
                if fighter in bfo_names:
                    print(f"ERROR: Fighter name '{fighter}' is in BFO format! Fixing to UFCStats format.")
                    df.loc[df['fighter'] == fighter, 'fighter'] = self.reverse_map_name(fighter)
            
            # Check opponent names
            for opp in df['opponent'].unique():
                # First sanitize the name
                sanitized_opp = self.sanitize_name(opp)
                if sanitized_opp != opp:
                    print(f"Sanitizing opponent name: '{opp}' → '{sanitized_opp}'")
                    df.loc[df['opponent'] == opp, 'opponent'] = sanitized_opp
                    opp = sanitized_opp
                
                # Then check if it's in BFO format
                if opp in bfo_names:
                    print(f"ERROR: Opponent name '{opp}' is in BFO format! Fixing to UFCStats format.")
                    df.loc[df['opponent'] == opp, 'opponent'] = self.reverse_map_name(opp)
                    
            # Insert data into the odds database
            with self._get_odds_connection() as conn:
                # Insert each row individually to handle duplicates properly
                inserted = 0
                skipped = 0
                
                print(f"Checking each record against existing database entries...")
                
                # Group by date for better debugging output
                df_sorted = df.sort_values('timestamp')
                date_groups = df_sorted.groupby(df_sorted['timestamp'].dt.date)
                
                for date, day_df in date_groups:
                    print(f"\nProcessing {len(day_df)} records for date {date}:")
                    day_inserted = 0
                    day_skipped = 0
                    
                    for _, row in day_df.iterrows():
                        try:
                            # Check if this exact entry already exists
                            check_query = text("""
                                SELECT id FROM bestfightodds.bfo 
                                WHERE fighter = :fighter 
                                AND opponent = :opponent 
                                AND timestamp = :timestamp
                            """)
                            
                            existing = conn.execute(
                                check_query, 
                                {
                                    'fighter': row['fighter'],
                                    'opponent': row['opponent'],
                                    'timestamp': row['timestamp']
                                }
                            ).fetchone()
                            
                            if existing:
                                # Skip this record - it's a duplicate
                                day_skipped += 1
                                skipped += 1
                                continue
                                
                            # Insert the new record
                            conn.execute(
                                text("""
                                    INSERT INTO bestfightodds.bfo (fighter, fighter_url, opponent, timestamp, odds)
                                    VALUES (:fighter, :fighter_url, :opponent, :timestamp, :odds)
                                """),
                                {
                                    'fighter': row['fighter'],
                                    'fighter_url': row['fighter_url'],
                                    'opponent': row['opponent'],
                                    'timestamp': row['timestamp'],
                                    'odds': row['odds']
                                }
                            )
                            day_inserted += 1
                            inserted += 1
                        except Exception as e:
                            print(f"Error inserting row {row['fighter']} vs {row['opponent']} at {row['timestamp']}: {str(e)}")
                            day_skipped += 1
                            skipped += 1
                            continue
                    
                    print(f"  Date {date}: {day_inserted} inserted, {day_skipped} skipped as duplicates")
                        
                conn.commit()
                
                print(f"\nSummary:")
                print(f"  Total records processed: {len(df)}")
                print(f"  New records inserted: {inserted}")
                print(f"  Duplicate records skipped: {skipped}")
                
                if skipped > 0 and inserted == 0:
                    print(f"\nWARNING: All {skipped} records were duplicates! This suggests:")
                    print(f"  1. This fighter's odds data may already be up to date in the database")
                    print(f"  2. BestFightOdds may not have newer odds data")
                    print(f"  3. The fighter may not have recent fights with odds available")
                    
                    # Check what's the latest data we have for this fighter in the database
                    fighter_name = df['fighter'].iloc[0]
                    latest_query = text("""
                        SELECT MAX(timestamp) as latest_timestamp, 
                               COUNT(*) as total_records
                        FROM bestfightodds.bfo 
                        WHERE fighter = :fighter
                    """)
                    
                    latest_result = conn.execute(latest_query, {"fighter": fighter_name}).fetchone()
                    if latest_result and latest_result[0]:
                        print(f"  Latest odds data in database for {fighter_name}: {latest_result[0]}")
                        print(f"  Total records in database for {fighter_name}: {latest_result[1]}")
                    
                return inserted
                
        except Exception as e:
            print(f"Error saving odds data to database: {str(e)}")
            return 0
    
    # Latest Odds Functionality for Specific Fights
    
    def american_to_decimal(self, american_odds):
        """
        Convert American odds to decimal odds.
        
        Args:
            american_odds: American odds format (e.g. -150, +200)
            
        Returns:
            Decimal odds
        """
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def decimal_to_american(self, decimal_odds):
        """
        Convert decimal odds to American odds.
        
        Args:
            decimal_odds: Decimal odds format (e.g. 1.67, 3.0)
            
        Returns:
            American odds
        """
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    def remove_vig(self, odds1, odds2):
        """
        Remove the vig from two American odds to get fair odds.
        
        Args:
            odds1: American odds for fighter 1
            odds2: American odds for fighter 2
            
        Returns:
            Tuple of (fair_odds1, fair_odds2) in American format
        """
        # Convert to decimal odds
        decimal1 = self.american_to_decimal(odds1)
        decimal2 = self.american_to_decimal(odds2)
        
        # Calculate implied probabilities
        prob1 = 1 / decimal1
        prob2 = 1 / decimal2
        
        # Calculate overround (vig)
        total_prob = prob1 + prob2
        
        # Remove vig by normalizing probabilities
        fair_prob1 = prob1 / total_prob
        fair_prob2 = prob2 / total_prob
        
        # Convert back to decimal odds
        fair_decimal1 = 1 / fair_prob1
        fair_decimal2 = 1 / fair_prob2
        
        # Convert back to American odds
        fair_odds1 = self.decimal_to_american(fair_decimal1)
        fair_odds2 = self.decimal_to_american(fair_decimal2)
        
        return fair_odds1, fair_odds2
    
    def find_fight_odds(self, fighter1_name, fighter2_name):
        """
        Find the latest odds for a specific fight between two fighters.
        
        Args:
            fighter1_name: Name of the first fighter
            fighter2_name: Name of the second fighter
            
        Returns:
            Dictionary with latest odds data or None if not found
        """
        print(f"\nSearching for fight odds: {fighter1_name} vs {fighter2_name}")
        
        # Try to find fighter1's page and look for fighter2 as opponent
        fighter1_link = self.get_fighter_link(fighter1_name)
        if fighter1_link:
            matches = self.parse_fighter_page(fighter1_link)
            
            # Look for fighter2 in the opponents
            mapped_fighter2 = self.get_mapped_name(fighter2_name)
            
            for match in matches:
                opp_name = match['opp_name']
                
                # Check if this opponent matches fighter2 (try various name formats)
                if (opp_name.lower() == mapped_fighter2.lower() or 
                    opp_name.lower() == fighter2_name.lower() or
                    self.reverse_map_name(opp_name).lower() == fighter2_name.lower()):
                    
                    print(f"Found match: {fighter1_name} vs {opp_name}")
                    
                    # Get odds data for this match
                    odds_data = self.get_perc_change(match['data_li'][0], match['data_li'][1])
                    
                    if odds_data:
                        # Get the latest odds (most recent timestamp)
                        latest_odds = max(odds_data, key=lambda x: x['x'])
                        latest_timestamp = datetime.fromtimestamp(latest_odds['x'] / 1000)
                        
                        print(f"Latest odds from {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"{fighter1_name}: {latest_odds['y']}")
                        
                        # Now we need to find the odds for fighter2 (opponent)
                        # The odds we got are for fighter1, we need to calculate fighter2's odds
                        fighter1_odds = latest_odds['y']
                        
                        return {
                            'fighter1': fighter1_name,
                            'fighter2': fighter2_name,
                            'fighter1_odds': fighter1_odds,
                            'timestamp': latest_timestamp,
                            'found_match': True
                        }
        
        # If not found, try the reverse (fighter2's page looking for fighter1)
        fighter2_link = self.get_fighter_link(fighter2_name)
        if fighter2_link:
            matches = self.parse_fighter_page(fighter2_link)
            
            mapped_fighter1 = self.get_mapped_name(fighter1_name)
            
            for match in matches:
                opp_name = match['opp_name']
                
                if (opp_name.lower() == mapped_fighter1.lower() or 
                    opp_name.lower() == fighter1_name.lower() or
                    self.reverse_map_name(opp_name).lower() == fighter1_name.lower()):
                    
                    print(f"Found match: {fighter2_name} vs {opp_name}")
                    
                    odds_data = self.get_perc_change(match['data_li'][0], match['data_li'][1])
                    
                    if odds_data:
                        latest_odds = max(odds_data, key=lambda x: x['x'])
                        latest_timestamp = datetime.fromtimestamp(latest_odds['x'] / 1000)
                        
                        print(f"Latest odds from {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        print(f"{fighter2_name}: {latest_odds['y']}")
                        
                        fighter2_odds = latest_odds['y']
                        
                        return {
                            'fighter1': fighter1_name,
                            'fighter2': fighter2_name,
                            'fighter2_odds': fighter2_odds,
                            'timestamp': latest_timestamp,
                            'found_match': True
                        }
        
        print(f"No odds found for {fighter1_name} vs {fighter2_name}")
        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'found_match': False
        }
    
    def get_manual_odds_input(self, fighter1_name, fighter2_name):
        """
        Prompt user to manually input odds for a fight.
        
        Args:
            fighter1_name: Name of the first fighter
            fighter2_name: Name of the second fighter
            
        Returns:
            Dictionary with manually input odds
        """
        print(f"\nManual odds input required for: {fighter1_name} vs {fighter2_name}")
        print("Please enter the American odds (e.g., -150, +200):")
        
        while True:
            try:
                fighter1_odds_str = input(f"{fighter1_name} odds: ").strip()
                fighter1_odds = int(fighter1_odds_str)
                break
            except ValueError:
                print("Please enter a valid integer (e.g., -150, +200)")
        
        while True:
            try:
                fighter2_odds_str = input(f"{fighter2_name} odds: ").strip()
                fighter2_odds = int(fighter2_odds_str)
                break
            except ValueError:
                print("Please enter a valid integer (e.g., -150, +200)")
        
        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'fighter1_odds': fighter1_odds,
            'fighter2_odds': fighter2_odds,
            'manual_input': True,
            'timestamp': datetime.now()
        }
    
    def get_latest_fight_odds(self, fight_list):
        """
        Get the latest odds for a list of fights, remove vig, and return vigless odds.
        
        ⚠️  WARNING: This method requires database connections for initialization.
        For database-free latest odds retrieval, use BFOLatestOddsOnly class in predict.py
        
        Args:
            fight_list: List of tuples [(fighter1_name, fighter2_name), ...]
            
        Returns:
            Dictionary {fighter_name: vigless_odds, ...}
            
        Note: This method does NOT save odds to the database - it only retrieves latest odds.
        Database updates only occur in scrape_all_fighters() and scrape_fighter() methods.
        """
        result_odds = {}
        
        print(f"Processing {len(fight_list)} fights...")
        
        for fighter1, fighter2 in fight_list:
            print(f"\n{'='*60}")
            print(f"Processing fight: {fighter1} vs {fighter2}")
            print(f"{'='*60}")
            
            # Try to find odds automatically
            fight_odds = self.find_fight_odds(fighter1, fighter2)
            
            if not fight_odds['found_match']:
                # Ask for manual input
                fight_odds = self.get_manual_odds_input(fighter1, fighter2)
            
            # Extract odds (handle different response formats)
            if 'fighter1_odds' in fight_odds and 'fighter2_odds' in fight_odds:
                fighter1_odds = fight_odds['fighter1_odds']
                fighter2_odds = fight_odds['fighter2_odds']
            elif 'fighter1_odds' in fight_odds:
                # We only have fighter1's odds, need to derive fighter2's odds
                # This is a simplified approach - in reality, we'd need both odds from the same source
                print("Warning: Only found odds for one fighter. Please enter the other fighter's odds manually.")
                fighter1_odds = fight_odds['fighter1_odds']
                while True:
                    try:
                        fighter2_odds_str = input(f"{fighter2} odds: ").strip()
                        fighter2_odds = int(fighter2_odds_str)
                        break
                    except ValueError:
                        print("Please enter a valid integer (e.g., -150, +200)")
            elif 'fighter2_odds' in fight_odds:
                # We only have fighter2's odds
                print("Warning: Only found odds for one fighter. Please enter the other fighter's odds manually.")
                fighter2_odds = fight_odds['fighter2_odds']
                while True:
                    try:
                        fighter1_odds_str = input(f"{fighter1} odds: ").strip()
                        fighter1_odds = int(fighter1_odds_str)
                        break
                    except ValueError:
                        print("Please enter a valid integer (e.g., -150, +200)")
            else:
                # No odds found, already got manual input
                continue
            
            # Remove vig from the odds
            try:
                fair_odds1, fair_odds2 = self.remove_vig(fighter1_odds, fighter2_odds)
                
                print(f"\nOriginal odds:")
                print(f"  {fighter1}: {fighter1_odds}")
                print(f"  {fighter2}: {fighter2_odds}")
                print(f"\nVig-free odds:")
                print(f"  {fighter1}: {fair_odds1}")
                print(f"  {fighter2}: {fair_odds2}")
                
                # Add to results
                result_odds[fighter1] = fair_odds1
                result_odds[fighter2] = fair_odds2
                
            except Exception as e:
                print(f"Error calculating vig-free odds: {str(e)}")
                # Fallback to original odds if vig removal fails
                result_odds[fighter1] = fighter1_odds
                result_odds[fighter2] = fighter2_odds
        
        return result_odds

    def scrape_all_fighters(self):
        """
        Scrape odds data for all fighters in the database who have fought at least once,
        or only those who fought since the last recorded scrape time.
        Updates the last scrape timestamp only if a full scrape occurred or new fights were processed.
        
        🗃️  DATABASE OPERATION: This method SAVES historical odds data to the database.
        Use this method only when building/updating the historical odds database (main.py).
        
        Returns:
            Total number of odds records saved
        """
        # Record the start time of this scrape attempt
        scrape_start_time = datetime.now(timezone.utc)
        should_update_timestamp = False # Flag to control timestamp update
        processed_new_fights = False # Flag to track if we processed fighters since last scrape
        
        try:
            # Reset not found fighters
            self.not_found_fighters = set()
            
            # Determine the list of fighters to scrape
            last_scrape_timestamp = self._read_last_scrape_timestamp()
            
            if last_scrape_timestamp:
                print(f"Fetching fighters who fought after {last_scrape_timestamp.date()}...")
                # Get fighters who fought since the last scrape date
                query = text("""
                    SELECT DISTINCT fm.fighter_name
                    FROM features.fighter_mapping fm
                    JOIN features.fight_mapping fig ON fig.fighter1_id = fm.fighter_id OR fig.fighter2_id = fm.fighter_id
                    JOIN features.event_mapping ev ON ev.event_id = fig.event_id
                    WHERE ev.event_date > :last_scrape_date
                    ORDER BY fm.fighter_name;
                """)
                result = self.mma_conn.execute(query, {"last_scrape_date": last_scrape_timestamp.date()})
                fighters_to_scrape = [row[0] for row in result.fetchall()]
                print(f"Found {len(fighters_to_scrape)} fighters with fights since {last_scrape_timestamp.date()}")
                
                if not fighters_to_scrape:
                    print("No new fighters to scrape since the last run. Timestamp will not be updated.")
                    # DO NOT update timestamp here
                    return 0
                else:
                    # We found fighters who fought since the last scrape, so we might update the timestamp later
                    processed_new_fights = True 
                    
            else:
                print("No last scrape timestamp found. Fetching all fighters with at least one fight...")
                # Get all fighter names from the database who have at least one fight
                result = self.mma_conn.execute(text("""
                    SELECT DISTINCT fm.fighter_name 
                    FROM features.fighter_mapping fm
                    WHERE EXISTS (
                        SELECT 1 FROM features.fight_mapping f 
                        WHERE f.fighter1_id = fm.fighter_id OR f.fighter2_id = fm.fighter_id
                    )
                    ORDER BY fm.fighter_name
                """))
                fighters_to_scrape = [row[0] for row in result.fetchall()]
                print(f"Found {len(fighters_to_scrape)} fighters total")
                # Since this is a full scrape (no previous timestamp), we should update the timestamp
                should_update_timestamp = True 
                
            total_saved = 0
            # Iterate over the determined list of fighters
            for fighter in tqdm(fighters_to_scrape, desc="Scraping fighter odds"):
                    
                df = self.get_odds_data(fighter)
                if not df.empty:
                    saved = self.save_to_database(df)
                    total_saved += saved
                    # Optional: Reduced print frequency
                    # print(f"Saved {saved} odds records for {fighter}") 
            
            print(f"\nTotal odds records saved in this run: {total_saved}")
            
            # Print not found fighters at the end
            if self.not_found_fighters:
                print("\nFighters not found on BestFightOdds during this run:")
                for fighter in sorted(self.not_found_fighters):
                    print(f"- {fighter}")
                print(f"Total not found: {len(self.not_found_fighters)}")
            
            # Conditionally update the last scrape timestamp
            # Update if it was a full scrape OR if we processed new fights since the last timestamp
            if should_update_timestamp or processed_new_fights:
                self._write_last_scrape_timestamp(scrape_start_time)
                print("Last scrape timestamp updated.")
            else:
                print("Last scrape timestamp not updated as no new fights were processed.")
            
            return total_saved
            
        except Exception as e:
            print(f"\nError during scrape_all_fighters: {str(e)}")
            print("Last scrape timestamp will not be updated due to error.")
            import traceback
            traceback.print_exc()
            return 0
    
    def scrape_fighter(self, fighter_name):
        """
        Scrape odds data for a specific fighter.
        
        🗃️  DATABASE OPERATION: This method SAVES historical odds data to the database.
        Use this method only when building/updating the historical odds database (main.py).
        
        Args:
            fighter_name: Name of the fighter to scrape
            
        Returns:
            Number of odds records saved
        """
        try:
            # Reset not found fighters
            self.not_found_fighters = set()
            
            print(f"\n=== DEBUGGING INFO FOR {fighter_name.upper()} ===")
            
            # Show recent UFC fights for this fighter
            ufc_fights = self.get_fighter_ufc_fights(fighter_name)
            if ufc_fights:
                print(f"\nRecent UFC fights for {fighter_name}:")
                for fight in ufc_fights:
                    print(f"  {fight['event_date']} vs {fight['opponent']} ({fight['event_name']})")
            else:
                print(f"\nNo recent UFC fights found for {fighter_name}")
            
            print(f"\n=== SCRAPING BESTFIGHTODDS DATA ===")
            
            df = self.get_odds_data(fighter_name)
            if not df.empty:
                saved = self.save_to_database(df)
                print(f"\nSaved {saved} odds records for {fighter_name}")
                
                # Print not found fighters at the end
                if self.not_found_fighters:
                    print("\nFighters not found on BestFightOdds:")
                    for fighter in sorted(self.not_found_fighters):
                        print(f"- {fighter}")
                
                return saved
            
            # Print not found fighters at the end
            if self.not_found_fighters:
                print("\nFighters not found on BestFightOdds:")
                for fighter in sorted(self.not_found_fighters):
                    print(f"- {fighter}")
            
            return 0
            
        except Exception as e:
            print(f"Error scraping fighter {fighter_name}: {str(e)}")
            return 0

    def get_fighter_ufc_fights(self, fighter_name):
        """
        Get a fighter's recent fights from the UFC database for comparison.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            List of fight information from the UFC database
        """
        try:
            query = text("""
                SELECT 
                    e.event_date,
                    e.event_name,
                    CASE 
                        WHEN f.fighter1_id = fm.fighter_id THEN fm2.fighter_name 
                        ELSE fm1.fighter_name 
                    END as opponent
                FROM features.fighter_mapping fm
                JOIN features.fight_mapping f ON (f.fighter1_id = fm.fighter_id OR f.fighter2_id = fm.fighter_id)
                JOIN features.fighter_mapping fm1 ON fm1.fighter_id = f.fighter1_id
                JOIN features.fighter_mapping fm2 ON fm2.fighter_id = f.fighter2_id
                JOIN features.event_mapping e ON e.event_id = f.event_id
                WHERE LOWER(fm.fighter_name) = LOWER(:fighter_name)
                ORDER BY e.event_date DESC
                LIMIT 10
            """)
            
            result = self.mma_conn.execute(query, {"fighter_name": fighter_name})
            fights = []
            for row in result.fetchall():
                fights.append({
                    'event_date': row[0],
                    'event_name': row[1], 
                    'opponent': row[2]
                })
            
            return fights
            
        except Exception as e:
            print(f"Error getting UFC fights for {fighter_name}: {str(e)}")
            return []

if __name__ == "__main__":
    def create_db_engine(db_url, create_if_not_exists=True):
        """
        Create the SQLAlchemy database engine.
        
        Args:
            db_url: Database connection URL
            create_if_not_exists: Whether to create the database if it doesn't exist
            
        Returns:
            SQLAlchemy engine
        """
        if create_if_not_exists:
            from sqlalchemy_utils import database_exists, create_database
            
            # Create the database if it doesn't exist
            engine = create_engine(db_url)
            if not database_exists(engine.url):
                create_database(engine.url)
        else:
            engine = create_engine(db_url)
        
        return engine

    @contextmanager
    def get_db_connection(engine):
        """
        Context manager for database connections.
        
        Args:
            engine: SQLAlchemy engine
            
        Yields:
            Database connection
        """
        connection = engine.connect()
        try:
            yield connection
        finally:
            connection.close()
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Scrape fighter odds data from bestfightodds.com')
    parser.add_argument('--fighter', type=str, help='Specific fighter name to scrape (optional)')
    parser.add_argument('--latest-odds', action='store_true', 
                      help='Get latest odds for specific fights (example mode)')
    parser.add_argument('--mma-db-url', type=str, 
                      default=database_url(),
                      help='Main MMA database connection URL')
    parser.add_argument('--odds-db-url', type=str,
                      default=odds_database_url(),
                      help='Odds database connection URL')
    
    args = parser.parse_args()
    
    try:
        # Create database engines
        mma_engine = create_db_engine(args.mma_db_url, create_if_not_exists=False)  # Don't create the main DB
        
        # Scrape the odds data
        with get_db_connection(mma_engine) as mma_conn:
            # Initialize scraper with both connections
            scraper = BFOScraper(mma_conn, odds_db_url=args.odds_db_url)

            if args.latest_odds:
                # Example of using the latest odds functionality
                print("=== LATEST ODDS FUNCTIONALITY EXAMPLE ===")
                fight_list = [
                    ("jon jones", "stipe miocic"),
                    ("alex pereira", "khalil rountree jr."),
                ]
                
                print(f"Getting latest vig-free odds for {len(fight_list)} fights...")
                latest_odds = scraper.get_latest_fight_odds(fight_list)
                
                print(f"\n{'='*60}")
                print("FINAL RESULTS - VIG-FREE ODDS:")
                print(f"{'='*60}")
                for fighter, odds in latest_odds.items():
                    print(f"{fighter.title()}: {odds:+d}")
                    
            elif args.fighter:
                print(f"Scraping odds data for {args.fighter}...")
                records = scraper.scrape_fighter(args.fighter)
                print(f"Saved {records} odds records for {args.fighter}")
            else:
                print("Scraping odds data for all fighters in the database...")
                records = scraper.scrape_all_fighters()
                print(f"Total odds records saved: {records}")
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 
