from libs.upcoming_fights import UpcomingFights
import pandas as pd
from libs.feature_store.create_inference_data import CreateInferenceData  # Keep for backward compatibility
from libs.feature_store.inference import InferenceDataBuilder, filter_features_for_model
from libs.visualization import FightVisualizer
from libs.shap_visualization import ShapVisualizer
from itertools import combinations
from datetime import datetime
import os
import argparse
from pathlib import Path
from libs.screenshot import take_screenshots
import json
import re
from curl_cffi import requests
from bs4 import BeautifulSoup
import urllib3
import warnings
from libs.bfo_scraper import BFOScraper
from libs.modeling.discovery import is_loadable_prediction_model_dir
from libs.modeling.portable_artifacts import (
    install_pathlib_pickle_compatibility,
    load_joblib_artifact,
    load_tabular_predictor,
    pathlib_pickle_compatibility,
)
from libs.paths import data_file, models_dir, picks_dir

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

class BFOLatestOddsOnly:
    """
    Lightweight class for getting latest odds from BestFightOdds without database operations.
    Only fetches current odds for predictions - does not store or update any data.
    """
    
    # Share the historical scraper's BFO aliases so dashboard predictions and
    # odds-database refreshes resolve fighter names identically.
    NAME_MAPPINGS = BFOScraper.NAME_MAPPINGS
    REVERSE_MAPPINGS = BFOScraper.REVERSE_MAPPINGS
    DUPE_NAMES = BFOScraper.DUPE_NAMES
    
    def __init__(self, use_flaresolverr=False):
        self.base_url = "https://www.bestfightodds.com"
        self.use_flaresolverr = bool(use_flaresolverr)
        self.flaresolverr_url = os.getenv("FLARESOLVERR_URL", "http://localhost:8192/v1")
        self.default_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Connection": "keep-alive",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
        }
        self._flaresolverr_session = None
        self._flaresolverr_cookies = None
        self._flaresolverr_user_agent = None
    
    def get_mapped_name(self, fighter_name):
        """Get mapped BFO name if available, otherwise return original."""
        if fighter_name is None:
            return None
        if fighter_name.lower() in self.NAME_MAPPINGS:
            return self.NAME_MAPPINGS[fighter_name.lower()]
        return fighter_name
    
    def reverse_map_name(self, bfo_name):
        """Convert BFO name back to UFCStats format."""
        if bfo_name is None:
            return None
        if bfo_name.lower() in self.REVERSE_MAPPINGS:
            return self.REVERSE_MAPPINGS[bfo_name.lower()]
        
        for ufcstats_name, alt_names in self.DUPE_NAMES.items():
            if bfo_name.lower() in [alt.lower() for alt in alt_names]:
                return ufcstats_name
        
        return bfo_name
    
    def get_flaresolverr_session(self):
        """Get cookies and user-agent from FlareSolverr by solving Cloudflare challenge."""
        try:
            # Create session
            session_data = {
                "cmd": "sessions.create",
                "session": "bfo_session"
            }
            
            response = requests.post(self.flaresolverr_url, json=session_data, timeout=30, verify=False)
            if response.status_code != 200:
                print(f"Failed to create FlareSolverr session: {response.status_code}")
                return False
            
            # Solve challenge by visiting the main page
            solve_data = {
                "cmd": "request.get",
                "url": self.base_url,
                "session": "bfo_session"
            }
            
            response = requests.post(self.flaresolverr_url, json=solve_data, timeout=60, verify=False)
            if response.status_code != 200:
                print(f"FlareSolverr request failed: {response.status_code}")
                return False
            
            result = response.json()
            if result.get("status") != "ok":
                print(f"FlareSolverr solve failed: {result.get('message', 'Unknown error')}")
                return False
            
            # Extract cookies and user-agent
            solution = result.get("solution", {})
            self._flaresolverr_cookies = solution.get("cookies", [])
            self._flaresolverr_user_agent = solution.get("userAgent", "")
            self._flaresolverr_session = "bfo_session"
            
            print(f"FlareSolverr session created successfully. Got {len(self._flaresolverr_cookies)} cookies.")
            return True
            
        except Exception as e:
            print(f"Error getting FlareSolverr session: {str(e)}")
            return False
    
    def get_cookies_dict(self):
        """Convert FlareSolverr cookies list to dict format for requests."""
        if not self._flaresolverr_cookies:
            return {}
        
        cookies_dict = {}
        for cookie in self._flaresolverr_cookies:
            cookies_dict[cookie.get("name")] = cookie.get("value")
        
        return cookies_dict
    
    def make_request_with_flaresolverr(self, url, params=None):
        """Make a request using FlareSolverr cookies and user-agent."""
        if not self.use_flaresolverr:
            return requests.get(url, params=params, headers=self.default_headers, timeout=10, verify=False)

        # Get session if we don't have one
        if not self._flaresolverr_session:
            if not self.get_flaresolverr_session():
                # Fallback to regular request with SSL verification disabled
                return requests.get(url, params=params, headers=self.default_headers, timeout=5, verify=False)
        
        # Prepare headers with FlareSolverr user-agent
        headers = self.default_headers.copy()
        if self._flaresolverr_user_agent:
            headers["User-Agent"] = self._flaresolverr_user_agent
        
        # Get cookies
        cookies = self.get_cookies_dict()
        
        try:
            return requests.get(url, params=params, headers=headers, cookies=cookies, timeout=10, verify=False)
        except Exception as e:
            print(f"Request with FlareSolverr failed: {str(e)}")
            # Try to refresh session
            if self.get_flaresolverr_session():
                cookies = self.get_cookies_dict()
                return requests.get(url, params=params, headers=headers, cookies=cookies, timeout=10, verify=False)
            else:
                # Final fallback to regular request with SSL verification disabled
                return requests.get(url, params=params, headers=self.default_headers, timeout=5, verify=False)
    
    def sanitize_name(self, name):
        """Remove @ symbols from fighter names."""
        if not name:
            return name
        return name.replace('@', '')
    
    def decimal_to_american(self, decimal_odds):
        """Convert decimal odds to American odds."""
        try:
            decimal_odds = float(decimal_odds)
            if decimal_odds >= 2.0:
                return int(round((decimal_odds - 1) * 100))
            else:
                return int(round(-100 / (decimal_odds - 1)))
        except Exception as e:
            print(f"Error converting decimal odds {decimal_odds}: {e}")
            return 100  # Fallback
    
    def remove_vig(self, odds1, odds2):
        """Remove vig from two American odds to get fair odds."""
        try:
            # Convert to decimal odds
            if odds1 > 0:
                decimal1 = (odds1 / 100) + 1
            else:
                decimal1 = (100 / abs(odds1)) + 1
                
            if odds2 > 0:
                decimal2 = (odds2 / 100) + 1
            else:
                decimal2 = (100 / abs(odds2)) + 1
            
            # Calculate implied probabilities
            prob1 = 1 / decimal1
            prob2 = 1 / decimal2
            
            # Remove vig by normalizing probabilities
            total_prob = prob1 + prob2
            fair_prob1 = prob1 / total_prob
            fair_prob2 = prob2 / total_prob
            
            # Convert back to decimal then American odds
            fair_decimal1 = 1 / fair_prob1
            fair_decimal2 = 1 / fair_prob2
            
            # Convert to American odds
            if fair_decimal1 >= 2.0:
                fair_odds1 = int((fair_decimal1 - 1) * 100)
            else:
                fair_odds1 = int(-100 / (fair_decimal1 - 1))
                
            if fair_decimal2 >= 2.0:
                fair_odds2 = int((fair_decimal2 - 1) * 100)
            else:
                fair_odds2 = int(-100 / (fair_decimal2 - 1))
            
            return fair_odds1, fair_odds2
            
        except Exception as e:
            print(f"Error removing vig: {e}")
            return odds1, odds2
    
    def get_fighter_link(self, name):
        """Get the bestfightodds.com link for a fighter."""
        search_name = self.get_mapped_name(name)
        
        url = f"{self.base_url}/search"
        search_params = {'query': search_name}
        
        try:
            # Use FlareSolverr for the request
            search_page = self.make_request_with_flaresolverr(url, params=search_params)
            
            soup = BeautifulSoup(search_page.text, 'html.parser')
            text_to_find = re.compile(search_name, re.IGNORECASE)
            link = soup.find('a', string=text_to_find)
            
            if link:
                return link['href']
            
            # Try alternate names if available
            if name.lower() in self.DUPE_NAMES:
                for alt_name in self.DUPE_NAMES[name.lower()]:
                    search_params = {'query': alt_name}
                    
                    # Use FlareSolverr for alternate name search too
                    alt_search_page = self.make_request_with_flaresolverr(url, params=search_params)
                    
                    alt_soup = BeautifulSoup(alt_search_page.text, 'html.parser')
                    alt_text_to_find = re.compile(alt_name, re.IGNORECASE)
                    alt_link = alt_soup.find('a', string=alt_text_to_find)
                    
                    if alt_link:
                        return alt_link['href']
            
            return None
                
        except Exception as e:
            print(f"Error getting fighter link for {search_name}: {str(e)}")
            return None
    
    def parse_fighter_page(self, fighterlink):
        """Parse a fighter's page to get opponent information and odds data."""
        if not fighterlink:
            return []
            
        try:
            url = f"{self.base_url}{fighterlink}"
            # Use FlareSolverr for the request
            fighter_page = self.make_request_with_flaresolverr(url)
            
            soup = BeautifulSoup(fighter_page.text, 'html.parser')
            results = []
            rows = soup.find_all('tr')
            
            for i, row in enumerate(rows):
                if row.get('class') and row.get('class')[0] == "main-row":
                    opp_row = rows[i + 1]
                    opp_name = opp_row.find('a').text if opp_row.find('a') else None
                    
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
        """Decode the base64-encoded odds data from the API."""
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
        """Get the percentage change data for a specific match."""
        try:
            url = f"{self.base_url}/api/ggd"
            search_params = {'m': m, 'p': p}
            # Use FlareSolverr for the request
            encrypted_resp = self.make_request_with_flaresolverr(url, params=search_params)
            
            fight_data = self.decode_base64(encrypted_resp.text)
            return fight_data[0]["data"]
            
        except Exception as e:
            print(f"Error getting percentage change data: {str(e)}")
            return []
    
    def calculate_opponent_odds(self, main_fighter_odds):
        """
        Calculate the opponent's odds based on the main fighter's odds.
        This assumes a fair two-way market with standard vig.
        """
        try:
            # Ensure main_fighter_odds is a number and convert to float first
            main_odds = float(main_fighter_odds)
            
            # Convert to decimal odds
            if main_odds > 0:
                main_decimal = (main_odds / 100) + 1
            else:
                main_decimal = (100 / abs(main_odds)) + 1
            
            # Calculate implied probability
            main_prob = 1 / main_decimal
            
            # Opponent probability (assuming ~5% total vig)
            opp_prob = 1 - main_prob
            
            # Convert back to decimal then American odds
            if opp_prob > 0:
                opp_decimal = 1 / opp_prob
                
                if opp_decimal >= 2.0:
                    opp_odds = int(round((opp_decimal - 1) * 100))
                else:
                    opp_odds = int(round(-100 / (opp_decimal - 1)))
                    
                return opp_odds
            else:
                return None
                
        except Exception as e:
            print(f"Error calculating opponent odds: {e}")
            return None

    def find_single_fighter_odds(self, main_fighter, opponent_fighter):
        """Find odds for main_fighter against opponent_fighter."""
        fighter_link = self.get_fighter_link(main_fighter)
        if not fighter_link:
            return None
            
        matches = self.parse_fighter_page(fighter_link)
        mapped_opponent = self.get_mapped_name(opponent_fighter)
        
        for match in matches:
            opp_name = match['opp_name']
            
            # Check if this opponent matches the target opponent
            if (opp_name.lower() == mapped_opponent.lower() or 
                opp_name.lower() == opponent_fighter.lower() or
                self.reverse_map_name(opp_name).lower() == opponent_fighter.lower()):
                
                # Get odds data for this match
                odds_data = self.get_perc_change(match['data_li'][0], match['data_li'][1])
                
                if odds_data:
                    # Get the latest odds (most recent timestamp)
                    latest_odds = max(odds_data, key=lambda x: x['x'])
                    latest_timestamp = datetime.fromtimestamp(latest_odds['x'] / 1000)
                    
                    # BFO returns decimal odds, convert to American odds
                    decimal_odds = latest_odds['y']
                    american_odds = self.decimal_to_american(decimal_odds)
                    
                    return {
                        'fighter': main_fighter,
                        'opponent': opponent_fighter,
                        'decimal_odds': decimal_odds,
                        'american_odds': american_odds,
                        'timestamp': latest_timestamp
                    }
        
        return None

    def find_fight_odds(self, fighter1_name, fighter2_name):
        """Find the latest odds for both fighters in a specific fight."""
        print(f"  Searching: {fighter1_name} vs {fighter2_name}")
        
        # Try to get both fighters' odds
        fighter1_data = self.find_single_fighter_odds(fighter1_name, fighter2_name)
        fighter2_data = self.find_single_fighter_odds(fighter2_name, fighter1_name)
        
        # Check what we found
        if fighter1_data and fighter2_data:
            print("  [ok] Found both fighters' odds.")
            print(f"  [odds] {fighter1_name}: decimal {fighter1_data['decimal_odds']:.3f} -> American {fighter1_data['american_odds']:+d}")
            print(f"  [odds] {fighter2_name}: decimal {fighter2_data['decimal_odds']:.3f} -> American {fighter2_data['american_odds']:+d}")
            
            # Use the most recent timestamp
            latest_timestamp = max(fighter1_data['timestamp'], fighter2_data['timestamp'])
            
            return {
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': fighter1_data['american_odds'],
                'fighter2_odds': fighter2_data['american_odds'],
                'timestamp': latest_timestamp,
                'found_match': True,
                'both_odds_found': True
            }
            
        elif fighter1_data:
            print(f"  [warn] Found only {fighter1_name}'s odds; {fighter2_name} remains missing.")
            print(f"  [odds] {fighter1_name}: decimal {fighter1_data['decimal_odds']:.3f} -> American {fighter1_data['american_odds']:+d}")
            
            # Return partial result without calculating opponent odds
            return {
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': fighter1_data['american_odds'],
                'fighter2_odds': None,
                'timestamp': fighter1_data['timestamp'],
                'found_match': True,
                'both_odds_found': False
            }
            
        elif fighter2_data:
            print(f"  [warn] Found only {fighter2_name}'s odds; {fighter1_name} remains missing.")
            print(f"  [odds] {fighter2_name}: decimal {fighter2_data['decimal_odds']:.3f} -> American {fighter2_data['american_odds']:+d}")
            
            # Return partial result without calculating opponent odds
            return {
                'fighter1': fighter1_name,
                'fighter2': fighter2_name,
                'fighter1_odds': None,
                'fighter2_odds': fighter2_data['american_odds'],
                'timestamp': fighter2_data['timestamp'],
                'found_match': True,
                'both_odds_found': False
            }
        
        print("  [warn] No odds found for either fighter.")
        return {
            'fighter1': fighter1_name,
            'fighter2': fighter2_name,
            'found_match': False,
            'both_odds_found': False
        }
    
    def get_latest_fight_odds_no_db(self, fight_list):
        """
        Get latest odds for fights without any database operations.
        
        Args:
            fight_list: List of tuples [(fighter1_name, fighter2_name), ...]
            
        Returns:
            Dictionary {fighter_name: vigless_odds, ...}
        """
        print(f"\n[prediction] Getting latest BFO odds for {len(fight_list)} fights (no DB operations)...")
        result_odds = {}
        
        for fighter1, fighter2 in fight_list:
            print(f"\n[prediction] Processing odds: {fighter1} vs {fighter2}")
            
            # Try to find odds automatically
            fight_odds = self.find_fight_odds(fighter1, fighter2)
            
            if not fight_odds['found_match']:
                # No odds found
                print("  [warn] No odds found; setting to N/A.")
                result_odds[fighter1] = "N/A"
                result_odds[fighter2] = "N/A"
                continue
            
            # Extract odds - should have both now
            fighter1_odds = fight_odds.get('fighter1_odds')
            fighter2_odds = fight_odds.get('fighter2_odds')
            
            if fighter1_odds is None or fighter2_odds is None:
                # If one fighter's odds are missing, record the known one and mark the other as missing
                print("  [warn] Partial odds found. Marking missing fighter as N/A for later manual input.")
                if fighter1_odds is not None:
                    try:
                        result_odds[fighter1] = int(round(fighter1_odds))
                    except Exception:
                        result_odds[fighter1] = fighter1_odds
                    result_odds[fighter2] = "N/A"
                elif fighter2_odds is not None:
                    try:
                        result_odds[fighter2] = int(round(fighter2_odds))
                    except Exception:
                        result_odds[fighter2] = fighter2_odds
                    result_odds[fighter1] = "N/A"
                else:
                    # Shouldn't happen, but safeguard
                    result_odds[fighter1] = "N/A"
                    result_odds[fighter2] = "N/A"
                continue
            
            # Check if we found both real odds or had to calculate one
            both_real_odds = fight_odds.get('both_odds_found', False)
            
            # Remove vig from the odds
            try:
                fair_odds1, fair_odds2 = self.remove_vig(fighter1_odds, fighter2_odds)
                
                # Convert to integers for display
                orig1_int = int(round(fighter1_odds))
                orig2_int = int(round(fighter2_odds))
                fair1_int = int(round(fair_odds1))
                fair2_int = int(round(fair_odds2))
                
                # Show whether odds are real or calculated
                odds_source = "REAL BFO ODDS" if both_real_odds else "REAL + MANUAL (after input)"
                print(f"  {odds_source}")
                print(f"  [odds] Original odds: {fighter1}({orig1_int:+d}) vs {fighter2}({orig2_int:+d})")
                print(f"  [odds] Vig-free odds: {fighter1}({fair1_int:+d}) vs {fighter2}({fair2_int:+d})")
                
                # Add to results - store both original and vigless odds
                result_odds[fighter1] = {
                    'original': orig1_int,
                    'vigless': fair1_int
                }
                result_odds[fighter2] = {
                    'original': orig2_int,
                    'vigless': fair2_int
                }
                
            except Exception as e:
                print(f"  [error] Error calculating vig-free odds: {str(e)}")
                # Fallback to original odds if vig removal fails (convert to int)
                orig1_fallback = int(round(fighter1_odds)) if isinstance(fighter1_odds, (int, float)) else "N/A"
                orig2_fallback = int(round(fighter2_odds)) if isinstance(fighter2_odds, (int, float)) else "N/A"
                result_odds[fighter1] = {
                    'original': orig1_fallback,
                    'vigless': orig1_fallback  # Use same odds if vig removal fails
                }
                result_odds[fighter2] = {
                    'original': orig2_fallback,
                    'vigless': orig2_fallback  # Use same odds if vig removal fails
                }
        
        return result_odds

def get_manual_fighter_odds(fighter_name):
    """
    Prompt user to manually input American odds for a fighter.
    
    Args:
        fighter_name: Name of the fighter to input odds for
        
    Returns:
        American odds as integer or None if skipped
    """
    print(f"\n[input] Manual odds input required for: {fighter_name}")
    print("Please enter American odds (e.g., -150, +200) or hit enter to leave as N/A:")
    
    while True:
        try:
            user_input = input(f"{fighter_name} odds: ").strip()
            
            if user_input.lower() in ['skip', 's', 'n/a', '']:
                print(f"Skipping odds for {fighter_name}")
                return None
                
            # Try to parse as integer (handle + prefix)
            odds_str = user_input.replace('+', '')
            odds = int(odds_str)
            print(f"[ok] Set {fighter_name} odds to {odds:+d}")
            return odds
            
        except ValueError:
            print("[error] Please enter a valid integer (e.g., -150, +200) or 'skip'")

def _empty_odds_for_fights(fight_list):
    odds = {}
    for fight in fight_list:
        odds[fight[1]] = "N/A"
        odds[fight[2]] = "N/A"
    return odds


def get_bfo_odds(fight_list, allow_manual_input=True, use_flaresolverr=False):
    """
    Get the latest odds for fights using lightweight BFO scraper (no database operations).
    If odds are missing, optionally prompt for manual input.
    
    Args:
        fight_list: List of tuples [(fight_date, fighter1_name, fighter2_name), ...]
        
    Returns:
        Dictionary with fighter odds: {fighter_name: american_odds, ...}
    """
    try:
        # Convert fight_list format (only needs fighter names)
        bfo_fight_list = [(fight[1], fight[2]) for fight in fight_list]
        
        if use_flaresolverr:
            print("[prediction] Using FlareSolverr for BestFightOdds requests.")
        scraper = BFOLatestOddsOnly(use_flaresolverr=use_flaresolverr)
        latest_odds = scraper.get_latest_fight_odds_no_db(bfo_fight_list)
        
        print(f"[ok] Retrieved odds for {len(latest_odds)} fighters from BFO")
        
        # Check for missing odds and prompt for manual input
        all_fighters = set()
        for fight in fight_list:
            all_fighters.add(fight[1])  # fighter1
            all_fighters.add(fight[2])  # fighter2
        
        missing_fighters = [fighter for fighter in all_fighters if latest_odds.get(fighter, "N/A") == "N/A"]
        
        if missing_fighters and not allow_manual_input:
            print(f"\nMissing odds for {len(missing_fighters)} fighters; leaving them as N/A in non-interactive mode.")

        if missing_fighters and allow_manual_input:
            print(f"\n[warn] Missing odds for {len(missing_fighters)} fighters:")
            for fighter in missing_fighters:
                print(f"  - {fighter}")
            
            print("\n[input] You can manually input odds for missing fighters.")
            
            # Process manual input by fight pairs to enable devigging
            for fight in fight_list:
                fighter1, fighter2 = fight[1], fight[2]
                
                # Check if either fighter in this fight needs manual input
                needs_input = []
                if fighter1 in missing_fighters:
                    needs_input.append(fighter1)
                if fighter2 in missing_fighters:
                    needs_input.append(fighter2)
                
                if needs_input:
                    print(f"\n[prediction] Fight: {fighter1} vs {fighter2}")
                    
                    # Get manual input for missing fighters
                    for fighter in needs_input:
                        manual_odds = get_manual_fighter_odds(fighter)
                        if manual_odds is not None:
                            latest_odds[fighter] = manual_odds
                        else:
                            latest_odds[fighter] = "N/A"
                    
                    # If both fighters have valid odds now, devig them
                    f1_odds = latest_odds.get(fighter1)
                    f2_odds = latest_odds.get(fighter2)
                    
                    if (f1_odds != "N/A" and f2_odds != "N/A" and 
                        isinstance(f1_odds, (int, float)) and isinstance(f2_odds, (int, float))):
                        
                        print(f"  [odds] Devigging odds for {fighter1} vs {fighter2}...")
                        try:
                            fair_odds1, fair_odds2 = scraper.remove_vig(f1_odds, f2_odds)
                            print(f"  [odds] Original: {fighter1}({int(f1_odds):+d}) vs {fighter2}({int(f2_odds):+d})")
                            print(f"  [odds] Vig-free: {fighter1}({int(fair_odds1):+d}) vs {fighter2}({int(fair_odds2):+d})")
                            
                            # Store both original and vigless odds
                            latest_odds[fighter1] = {
                                'original': int(f1_odds),
                                'vigless': int(fair_odds1)
                            }
                            latest_odds[fighter2] = {
                                'original': int(f2_odds),
                                'vigless': int(fair_odds2)
                            }
                        except Exception as e:
                            print(f"  [error] Error devigging odds: {e}")
                            print("  [warn] Using original odds without devigging")
                            # Store original odds in both fields if devigging fails
                            latest_odds[fighter1] = {
                                'original': int(f1_odds),
                                'vigless': int(f1_odds)
                            }
                            latest_odds[fighter2] = {
                                'original': int(f2_odds),
                                'vigless': int(f2_odds)
                            }
        
        return latest_odds
        
    except Exception as e:
        print(f"[error] Error getting BFO odds: {str(e)}")
        print("[warn] Falling back to manual input for all fighters")
        
        if not allow_manual_input:
            print("Non-interactive mode enabled; leaving all odds as N/A")
            return _empty_odds_for_fights(fight_list)

        # If BFO scraping fails entirely, ask for manual input for all fighters
        manual_odds = {}
        scraper = BFOLatestOddsOnly(use_flaresolverr=use_flaresolverr)  # Need scraper instance for devigging
        
        # Process by fight pairs to enable devigging
        for fight in fight_list:
            fighter1, fighter2 = fight[1], fight[2]
            print(f"\n[prediction] Fight: {fighter1} vs {fighter2}")
            
            # Get manual input for both fighters
            f1_odds = get_manual_fighter_odds(fighter1) or "N/A"
            f2_odds = get_manual_fighter_odds(fighter2) or "N/A"
            
            # If both fighters have valid odds, devig them
            if (f1_odds != "N/A" and f2_odds != "N/A" and 
                isinstance(f1_odds, (int, float)) and isinstance(f2_odds, (int, float))):
                
                print(f"  [odds] Devigging odds for {fighter1} vs {fighter2}...")
                try:
                    fair_odds1, fair_odds2 = scraper.remove_vig(f1_odds, f2_odds)
                    print(f"  [odds] Original: {fighter1}({int(f1_odds):+d}) vs {fighter2}({int(f2_odds):+d})")
                    print(f"  [odds] Vig-free: {fighter1}({int(fair_odds1):+d}) vs {fighter2}({int(fair_odds2):+d})")
                    
                    manual_odds[fighter1] = fair_odds1
                    manual_odds[fighter2] = fair_odds2
                except Exception as e:
                    print(f"  [error] Error devigging odds: {e}")
                    print("  [warn] Using original odds without devigging")
                    manual_odds[fighter1] = f1_odds
                    manual_odds[fighter2] = f2_odds
            else:
                manual_odds[fighter1] = f1_odds
                manual_odds[fighter2] = f2_odds
        
        return manual_odds


def build_manual_odds(fight_list, fighter1_odds=None, fighter2_odds=None):
    """Build prediction odds from explicit CLI/UI inputs without prompting."""
    odds = _empty_odds_for_fights(fight_list)
    if len(fight_list) != 1:
        return odds

    _fight_date, fighter1, fighter2 = fight_list[0]
    if fighter1_odds is None and fighter2_odds is None:
        return odds

    odds[fighter1] = int(fighter1_odds) if fighter1_odds is not None else "N/A"
    odds[fighter2] = int(fighter2_odds) if fighter2_odds is not None else "N/A"

    if fighter1_odds is None or fighter2_odds is None:
        return odds

    scraper = BFOLatestOddsOnly()
    try:
        fair1, fair2 = scraper.remove_vig(int(fighter1_odds), int(fighter2_odds))
    except Exception:
        fair1, fair2 = int(fighter1_odds), int(fighter2_odds)

    odds[fighter1] = {"original": int(fighter1_odds), "vigless": int(fair1)}
    odds[fighter2] = {"original": int(fighter2_odds), "vigless": int(fair2)}
    return odds


def manual_odds_cover_fight_list(fight_list, manual_odds):
    """Return True when supplied fighter odds cover every fighter in the fights."""
    if not manual_odds:
        return False
    supplied = {
        str(name).strip().lower()
        for name, value in manual_odds.items()
        if str(name).strip() and value not in (None, "")
    }
    required = set()
    for fight in fight_list:
        required.add(str(fight[1]).strip().lower())
        required.add(str(fight[2]).strip().lower())
    return bool(required) and required.issubset(supplied)


def parse_manual_odds_json(raw_value):
    """Parse CLI/UI supplied fighter odds as a JSON object of fighter name to American odds."""
    if not raw_value:
        return {}
    parsed = json.loads(raw_value)
    if not isinstance(parsed, dict):
        raise ValueError("--manual-odds-json must be a JSON object like {\"fighter name\": -150}")

    manual_odds = {}
    for fighter_name, odds in parsed.items():
        name = str(fighter_name).strip()
        if not name or odds in (None, ""):
            continue
        if isinstance(odds, str):
            odds = odds.strip().replace("+", "")
        manual_odds[name] = int(odds)
    return manual_odds


def _extract_original_odds(odds_value):
    if isinstance(odds_value, dict):
        odds_value = odds_value.get("original", "N/A")
    if odds_value == "N/A" or odds_value is None:
        return None
    try:
        if isinstance(odds_value, str):
            odds_value = odds_value.strip().replace("+", "")
        return int(odds_value)
    except (TypeError, ValueError):
        return None


def apply_manual_odds(fight_list, odds, manual_odds):
    """Apply explicit fighter odds to an odds map and devig complete fight pairs."""
    if not manual_odds:
        return odds

    resolved_odds = dict(odds or _empty_odds_for_fights(fight_list))
    manual_by_name = {str(name).strip().lower(): int(value) for name, value in manual_odds.items()}
    scraper = BFOLatestOddsOnly()

    print(f"\nApplying manually supplied odds for {len(manual_by_name)} fighters...")
    for fight in fight_list:
        _fight_date, fighter1, fighter2 = fight
        fighter1_key = fighter1.lower()
        fighter2_key = fighter2.lower()
        if fighter1_key in manual_by_name:
            resolved_odds[fighter1] = manual_by_name[fighter1_key]
            print(f"  Manual odds: {fighter1}({manual_by_name[fighter1_key]:+d})")
        if fighter2_key in manual_by_name:
            resolved_odds[fighter2] = manual_by_name[fighter2_key]
            print(f"  Manual odds: {fighter2}({manual_by_name[fighter2_key]:+d})")

        fighter1_odds = _extract_original_odds(resolved_odds.get(fighter1))
        fighter2_odds = _extract_original_odds(resolved_odds.get(fighter2))
        if fighter1_odds is None or fighter2_odds is None:
            continue

        try:
            fair1, fair2 = scraper.remove_vig(fighter1_odds, fighter2_odds)
        except Exception as exc:
            print(f"  Warning: could not devig manual odds for {fighter1} vs {fighter2}: {exc}")
            fair1, fair2 = fighter1_odds, fighter2_odds

        resolved_odds[fighter1] = {"original": int(fighter1_odds), "vigless": int(fair1)}
        resolved_odds[fighter2] = {"original": int(fighter2_odds), "vigless": int(fair2)}
    return resolved_odds


def resolve_prediction_odds(
    fight_list,
    *,
    odds_enabled,
    manual_odds=None,
    fighter1_odds=None,
    fighter2_odds=None,
    allow_manual_input=True,
    use_flaresolverr=False,
):
    """Resolve event or matchup odds without doing avoidable interactive/network work."""
    manual_odds = manual_odds or {}

    if fighter1_odds is not None or fighter2_odds is not None:
        bfo_odds = build_manual_odds(fight_list, fighter1_odds, fighter2_odds)
    elif odds_enabled and manual_odds_cover_fight_list(fight_list, manual_odds):
        print("\nManual odds supplied for every fighter; skipping BFO odds lookup.")
        bfo_odds = _empty_odds_for_fights(fight_list)
    elif odds_enabled:
        bfo_odds = get_bfo_odds(
            fight_list,
            allow_manual_input=allow_manual_input,
            use_flaresolverr=use_flaresolverr,
        )
    else:
        bfo_odds = _empty_odds_for_fights(fight_list)

    if manual_odds:
        bfo_odds = apply_manual_odds(fight_list, bfo_odds, manual_odds)
    return bfo_odds


def convert_american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    try:
        if isinstance(american_odds, str):
            if american_odds == "N/A":
                return 0
            american_odds = int(american_odds.replace('+', ''))
        
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    except (ValueError, TypeError):
        return 0

def get_fights(df, upcoming_number):
    uf = UpcomingFights(df, upcoming_number)
    # Get list of upcoming fights
    events = uf.run()   
    fight_list = [fight for event in events.values() for fight in event]
    event_names = [x for x in events.keys()]
    return fight_list, event_names


def get_manual_fight(fighter1, fighter2, fight_date=None):
    """Return a one-fight list compatible with the event prediction pipeline."""
    if not fighter1 or not fighter2:
        raise ValueError("Both --fighter1 and --fighter2 are required for manual matchup prediction.")
    if fighter1.strip().lower() == fighter2.strip().lower():
        raise ValueError("Manual matchup prediction requires two different fighters.")

    parsed_date = datetime.strptime(fight_date, "%Y-%m-%d") if fight_date else datetime.now()
    clean_fighter1 = fighter1.strip()
    clean_fighter2 = fighter2.strip()
    return [(parsed_date, clean_fighter1, clean_fighter2)], [f"{clean_fighter1}_vs_{clean_fighter2}"]

def load_model_and_calibrator(model_path, use_calibrated=True):
    """
    Universal loader that works with:
    - Old single-model folders (predictor.pkl + scaler.pkl)
    - New walk-forward ensemble folders (window_0/, final_model/, ensemble_info.txt)
    
    :param model_path: Path to the model directory
    :param use_calibrated: Whether to load and use calibrator if available
    :return: Tuple of (predictor, calibrator or None)
    """
    from autogluon.tabular import TabularPredictor
    import os
    
    ensemble_info_path = os.path.join(model_path, 'ensemble_info.txt')
    
    # === DETECT WHICH FORMAT WE HAVE ===
    has_ensemble_dirs = os.path.isdir(os.path.join(model_path, 'final_model')) or any(
        os.path.isdir(os.path.join(model_path, item))
        for item in os.listdir(model_path)
        if item.startswith('window_')
    ) if os.path.isdir(model_path) else False

    if os.path.exists(ensemble_info_path) and has_ensemble_dirs:
        # Walk-forward ensemble format
        print(f"Detected WALK-FORWARD ENSEMBLE model at {model_path}")
        print("Loading EnsemblePredictor...")
        from libs.modeling.train import EnsemblePredictor
        predictor = EnsemblePredictor.load(model_path)
        print(f"Successfully loaded ensemble with {len(predictor.predictors)} models")
    else:
        # Old single-model format
        print(f"Detected SINGLE model at {model_path}")
        predictor = load_tabular_predictor(
            TabularPredictor,
            model_path,
            require_version_match=False,
            require_py_version_match=False,
        )
    
    # Try to load calibrator if requested
    calibrator = None
    if use_calibrated:
        calibrator_path = os.path.join(model_path, 'calibrator.pkl')
        if os.path.exists(calibrator_path):
            try:
                calibrator = load_joblib_artifact(calibrator_path)
                print(f"[ok] Loaded calibrator from {calibrator_path}")
            except Exception as e:
                print(f"Warning: Could not load calibrator from {calibrator_path}: {e}")
                calibrator = None
        else:
            print(f"No calibrator found at {calibrator_path}")
    else:
        print("Calibrated predictions disabled by user")
    
    return predictor, calibrator

def get_predictions(model, calibrator, scaled_X_df, use_calibrated=None):
    """
    Get model predictions, optionally using calibrator.
    
    :param model: The AutoGluon predictor
    :param calibrator: The calibrator (can be None)
    :param scaled_X_df: Scaled feature DataFrame
    :param use_calibrated: Whether to use calibrator (auto-detect if None)
    :return: Prediction probabilities
    """
    # Determine whether to use calibrated predictions
    if use_calibrated is None:
        use_calibrated = calibrator is not None
    elif use_calibrated and calibrator is None:
        print("Warning: Calibrated predictions requested but no calibrator available. Using original predictions.")
        use_calibrated = False
    
    # Get original predictions - handle models trained with sample_weight
    try:
        with pathlib_pickle_compatibility():
            y_pred_proba = model.predict_proba(scaled_X_df)
    except KeyError as e:
        if 'sample_weight' in str(e):
            scaled_X_df_with_weights = scaled_X_df.copy()
            scaled_X_df_with_weights['sample_weight'] = 1.0
            with pathlib_pickle_compatibility():
                y_pred_proba = model.predict_proba(scaled_X_df_with_weights)
            print(f"Added sample_weight column for prediction (model was trained with recency weights)")
        else:
            raise e

    # Ensure prediction output is a DataFrame with columns [0, 1]
    if isinstance(y_pred_proba, pd.DataFrame):
        y_pred_df = y_pred_proba.copy()
    else:
        try:
            y_pred_df = pd.DataFrame(y_pred_proba, columns=[0, 1])
        except Exception:
            # Fallback: best effort conversion
            y_pred_df = pd.DataFrame(y_pred_proba)
            if y_pred_df.shape[1] == 2:
                y_pred_df.columns = [0, 1]

    # Apply calibration if requested and available
    if use_calibrated and calibrator is not None:
        # Preferred path: probability-only calibrator that maps class-1 probabilities to calibrated probabilities.
        try:
            prob_class1 = y_pred_df[1].values
            cal_prob_class1 = calibrator.predict_proba(prob_class1)
            # Handle calibrators that return Nx2 vs 1D arrays
            if hasattr(cal_prob_class1, 'ndim') and getattr(cal_prob_class1, 'ndim', 1) == 2 and cal_prob_class1.shape[1] == 2:
                cal_prob_class1 = cal_prob_class1[:, 1]
            # Reconstruct calibrated DataFrame
            cal_series = pd.Series(cal_prob_class1, index=y_pred_df.index)
            y_pred_df[1] = cal_series
            y_pred_df[0] = 1 - cal_series
            return y_pred_df
        except Exception:
            # Fallback: feature-based calibrator (e.g., CalibratedClassifierCV with embedded estimator)
            try:
                scaled_X_clean = scaled_X_df.drop(columns=['sample_weight'], errors='ignore')
                cal_output = calibrator.predict_proba(scaled_X_clean)
                cal_df = pd.DataFrame(cal_output, index=y_pred_df.index, columns=y_pred_df.columns if y_pred_df.shape[1] == 2 else None)
                # Ensure [0,1] columns
                if list(cal_df.columns) != [0, 1] and cal_df.shape[1] == 2:
                    cal_df.columns = [0, 1]
                return cal_df
            except Exception as e2:
                print(f"Warning: Could not generate calibrated predictions (fallback failed): {e2}. Using original predictions.")

    return y_pred_df

def load_model(model_name):
    """Load an AutoGluon model from the specified path (deprecated - use load_model_and_calibrator)"""
    from autogluon.tabular import TabularPredictor
    predictor = load_tabular_predictor(TabularPredictor, model_name)
    return predictor

def create_conf_parlays(results):
    """
    Creates 3-leg parlays based on average AI confidence with the following rules:
      1) No fighter appears in more than 2 parlays.
      2) Parlays are selected starting with the highest combined AI confidence, then the next highest, and so on.
    
    For each parlay:
      - Combined AI confidence is computed as the average of the individual AI win probabilities.
      - Combined decimal odds is computed as the product of each fight's decimal odds (derived from Vegas odds).
      - Combined odds in American format is derived from the combined decimal odds.
    
    The function prints out a formatted table of the selected parlays and pretty prints the final dictionary.
    """
    from itertools import combinations
    from datetime import datetime
    from pprint import pprint

    # Count how many parlays each fighter is in
    fighter_count = {}
    
    # Get all 3-fight combinations, sorted by descending average AI confidence
    all_parlays = list(combinations(results, 3))
    parlay_data = []
    
    for i, parlay in enumerate(all_parlays):
        fighter_names = [fight['fighter1_name'] for fight in parlay] + [fight['fighter2_name'] for fight in parlay]
        # Skip parlays where any fighter appears twice
        if len(set(fighter_names)) < 6:
            continue
            
        avg_ai_confidence = sum(fight['proba'] for fight in parlay) / 3
        combined_decimal_odds = 1
        for fight in parlay:
            if fight['winner'] == 'fighter1':
                combined_decimal_odds *= fight['fighter1_decimal_odds']
            else:
                combined_decimal_odds *= fight['fighter2_decimal_odds']
                
        parlay_data.append({
            'parlay': parlay,
            'avg_ai_confidence': avg_ai_confidence,
            'combined_decimal_odds': combined_decimal_odds,
            'fighter_names': fighter_names
        })
    
    # Sort by average AI confidence (descending)
    parlay_data.sort(key=lambda x: x['avg_ai_confidence'], reverse=True)
    
    # Select parlays, no fighter appears in more than 2 parlays
    selected_parlays = []
    
    for parlay in parlay_data:
        # Check if any fighter is already in 2 parlays
        if any(fighter_count.get(fighter, 0) >= 2 for fighter in parlay['fighter_names']):
            continue
            
        # Add this parlay
        selected_parlays.append(parlay)
        
        # Update fighter counts
        for fighter in parlay['fighter_names']:
            fighter_count[fighter] = fighter_count.get(fighter, 0) + 1
            
        # Only select top 5 parlays
        if len(selected_parlays) >= 5:
            break
    
    # Format and display the selected parlays
    formatted_parlays = []
    
    print("\n=== TOP 3-FIGHT PARLAYS BY AI CONFIDENCE ===")
    print(f"Selected on {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'Avg Conf':^10} | {'Decimal':^10} | {'American':^10} | {'Parlay':^50}")
    print("-" * 85)
    
    for parlay in selected_parlays:
        fights_text = " + ".join([
            f"{f['fighter1_name'] if f['winner'] == 'fighter1' else f['fighter2_name']}"
            for f in parlay['parlay']
        ])
        
        # Convert decimal odds to American
        american_odds = 0
        if parlay['combined_decimal_odds'] >= 2:
            american_odds = int((parlay['combined_decimal_odds'] - 1) * 100)
        else:
            american_odds = int(-100 / (parlay['combined_decimal_odds'] - 1))
            
        american_str = f"+{american_odds}" if american_odds > 0 else f"{american_odds}"
        
        print(f"{parlay['avg_ai_confidence']*100:^10.1f}% | {parlay['combined_decimal_odds']:^10.2f} | {american_str:^10} | {fights_text}")
        
        formatted_parlays.append({
            'fighters': [f['fighter1_name'] if f['winner'] == 'fighter1' else f['fighter2_name'] for f in parlay['parlay']],
            'avg_confidence': f"{parlay['avg_ai_confidence']*100:.1f}%",
            'combined_decimal': f"{parlay['combined_decimal_odds']:.2f}",
            'american_odds': american_str
        })
    
    print("\nParlay Dictionary:")
    pprint(formatted_parlays)
    
    return formatted_parlays

def convert_prob_to_american_odds(prob):
    """Convert a probability to American odds format"""
    if prob > 0.5:
        odds = -100 * (prob / (1 - prob))
        return f"{int(odds)}"
    else:
        odds = 100 * ((1 - prob) / prob)
        return f"+{int(odds)}"

def american_odds_to_prob(odds):
    """Convert American odds to implied probability"""
    try:
        if isinstance(odds, str):
            odds = int(odds.replace('+', ''))
        else:
            odds = int(odds)
            
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        else:
            return 100 / (odds + 100)
    except (ValueError, TypeError):
        return None

def calculate_expected_value(ai_odds_str, bookie_odds_str):
    """Calculate expected value as percentage difference between AI and bookie implied probabilities"""
    try:
        # Handle N/A cases
        if isinstance(bookie_odds_str, str) and bookie_odds_str == "N/A":
            return 0.0
        if pd.isna(bookie_odds_str):
            return 0.0
            
        # Convert to probabilities
        ai_prob = american_odds_to_prob(ai_odds_str)
        bookie_prob = american_odds_to_prob(bookie_odds_str)
        
        if ai_prob is None or bookie_prob is None:
            return 0.0
            
        # Calculate EV as percentage point difference
        # If AI thinks 70% and Vegas thinks 60%, EV = 10 percentage points
        ev = (ai_prob - bookie_prob) * 100
        return ev
        
    except (ValueError, TypeError, AttributeError):
        return 0.0

def has_positive_ev(ai_odds_str, bookie_odds_str):
    """Compare AI odds with bookie odds to determine if there's positive EV"""
    ev = calculate_expected_value(ai_odds_str, bookie_odds_str)
    return ev > 0


def maybe_take_screenshots(output_dir, enabled=False):
    """Optionally create PNG screenshots without failing an otherwise successful prediction."""
    if not enabled:
        print("Skipping screenshots. Use --screenshots to create PNG captures from generated HTML visualizations.")
        return False

    print("\nTaking screenshots of visualizations...")
    try:
        take_screenshots(output_dir)
    except Exception as exc:
        print(f"Warning: Screenshot generation failed: {exc}. Prediction artifacts were still written.")
        return False

    print(f"Screenshots saved to: {os.path.join(output_dir, 'screenshots')}")
    return True


def latest_model_path(model_type):
    candidates = [
        path for path in models_dir().glob(f"ag-*-{model_type}-*")
        if is_loadable_prediction_model_dir(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No loadable {model_type} model found in {models_dir()}. "
            "Pass --model-path or run `uv run python -m libs.modeling.train` first."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions for an upcoming UFC event.")
    parser.add_argument("--model-type", choices=["win", "decision"], default="win")
    parser.add_argument("--model-path", default=None, help="Path to an AutoGluon model directory.")
    parser.add_argument("--prediction-data-csv", default=str(data_file("prediction_data.csv")))
    parser.add_argument("--training-data-csv", default=None, help="Training CSV for SHAP background data.")
    parser.add_argument("--upcoming-number", type=int, default=1, help="1 is the next event, 2 is the event after next.")
    parser.add_argument("--fighter1", default=None, help="Manual matchup fighter 1. Skips Wikipedia event lookup when paired with --fighter2.")
    parser.add_argument("--fighter2", default=None, help="Manual matchup fighter 2. Skips Wikipedia event lookup when paired with --fighter1.")
    parser.add_argument("--fight-date", default=None, help="Manual matchup date in YYYY-MM-DD format. Defaults to now.")
    parser.add_argument("--fighter1-odds", type=int, default=None, help="Manual American odds for fighter 1.")
    parser.add_argument("--fighter2-odds", type=int, default=None, help="Manual American odds for fighter 2.")
    parser.add_argument("--manual-odds-json", default=None, help="JSON object mapping fighter names to manual American odds.")
    parser.add_argument("--odds", action="store_true", help="Include latest BFO odds in the output.")
    parser.add_argument("--flaresolverr", action="store_true", help="Use FlareSolverr for BestFightOdds requests when BFO is blocking normal scraping.")
    parser.add_argument("--no-manual-odds", action="store_true", help="Do not prompt for missing odds; use N/A instead.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP visualizations.")
    parser.add_argument("--use-calibrated", action="store_true", help="Use calibrated predictions when calibrator.pkl exists.")
    parser.add_argument("--output-dir", default=None, help="Directory for prediction images and CSVs.")
    parser.add_argument("--screenshots", action="store_true", help="Create PNG screenshots from generated HTML visualizations when Chrome is available.")
    return parser.parse_args()


def cli():
    compat_class = install_pathlib_pickle_compatibility()
    if compat_class:
        print(f"[runtime] Enabled cross-OS pathlib pickle compatibility for {compat_class} artifacts")

    args = parse_args()
    # Which event to predict
    upcoming_number = args.upcoming_number
    model_type = args.model_type

    # === CONFIGURATION FLAGS ===
    use_calibrated = args.use_calibrated

    model_path = Path(args.model_path).expanduser().resolve() if args.model_path else latest_model_path(model_type)
    if model_type == 'win':
        training_data_csv = args.training_data_csv or str(data_file("training_data.csv"))
    elif model_type == 'decision':
        training_data_csv = args.training_data_csv or str(data_file("training_data_dec.csv"))
    else:
        print(f"Invalid model type: {model_type}")
        exit(1)
    
    odds = args.odds
    SHAP = not args.no_shap
    manual_odds = parse_manual_odds_json(args.manual_odds_json)
    
    prediction_data_csv = args.prediction_data_csv
    
    # Use the model's prediction_data.csv if it exists, otherwise use the default path
    # training_data_csv = os.path.join(model_path, "prediction_data.csv")
    # if not os.path.exists(training_data_csv):
    #     training_data_csv = prediction_data_csv
    
    # Load the training data for SHAP background (if needed)
    if SHAP:
        df_train = pd.read_csv(training_data_csv)
    df_pred = pd.read_csv(prediction_data_csv)

    # Read features from the model's feats.txt file
    feats_file_path = os.path.join(model_path, 'feats.txt')
    try:
        with open(feats_file_path, 'r') as f:
            feats = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: Features file not found at {feats_file_path}")
        # Handle the error appropriately, maybe exit or use default features
        # For now, let's exit if the file is crucial
        exit(1)
    except Exception as e:
        print(f"Error reading features file: {e}")
        exit(1)

    #feats = FEATS_MAD
    #feats = LAYERED_TEST_FEATS
    
    model, calibrator = load_model_and_calibrator(model_path, use_calibrated)
    
    # Check if this is an EnsemblePredictor (walkforward model)
    from libs.modeling.train import EnsemblePredictor
    is_ensemble = isinstance(model, EnsemblePredictor)
    
    # Load scaler only for single models (ensembles handle scaling internally)
    scaler = None
    if not is_ensemble:
        # Single model: load scaler from root directory
        scaler_path = os.path.join(model_path, 'scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = load_joblib_artifact(scaler_path)
            print(f"Loaded scaler from root directory (single model)")
        else:
            raise FileNotFoundError(f"scaler.pkl not found in {model_path}")
    else:
        print(f"Ensemble model detected - scaling will be handled internally by each window model")
    
    # Helper function to identify which columns should be scaled (matching training logic)
    def get_features_to_scale(df):
        """
        Get list of features that should be scaled, excluding date_cols and categorical static features.
        
        This matches the logic used during training: excludes date columns, categorical static features
        (like weightclass_encoded and odds), sample_weight, and y_true from scaling.
        """
        date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
        
        # Categorical/encoded static features that should NOT be normalized
        # (continuous static features like age, reach, days_since_last_fight, ufcage WILL be normalized)
        categorical_static_feats = ['weightclass_encoded', 'odds', 'sample_weight']
        
        def should_exclude_col(col_name):
            """Check if column should be excluded from scaling."""
            if col_name in date_cols:
                return True
            # Exclude if column name contains categorical static feature strings
            for cat_feat in categorical_static_feats:
                if cat_feat in col_name:
                    return True
            return False
        
        # Get features to scale (exclude categorical features, date columns, sample_weight, y_true)
        return [col for col in df.columns 
                if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
    
    # Print calibration status
    if use_calibrated:
        if calibrator is not None:
            print("[prediction] Using calibrated predictions")
        else:
            print("[warn] Calibrated predictions requested but not available - using original predictions")
    else:
        print("[prediction] Using original (uncalibrated) predictions")

    ## Single fight or event
    if args.fighter1 or args.fighter2:
        fight_list, event_names = get_manual_fight(args.fighter1, args.fighter2, args.fight_date)
    else:
        fight_list, event_names = get_fights(df_pred, upcoming_number)
    print(fight_list)

    # Single fight
    # date = datetime.strptime('2025-05-30', '%Y-%m-%d')
    # fight_list = [(date, 'bryce mitchell', 'jean silva')]
    # event_names = ['Custom']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = event_names[0].replace(':', ' ').replace('.', '').replace(' ', '_') + "_" + timestamp + "_" + model_type
    suffix = "odds" if odds else "no_odds"
    ss_output_dir = args.output_dir or str(picks_dir() / f"{folder_name}_{suffix}")
    
    os.makedirs(ss_output_dir, exist_ok=True)
    
    bfo_odds = resolve_prediction_odds(
        fight_list,
        odds_enabled=odds,
        manual_odds=manual_odds,
        fighter1_odds=args.fighter1_odds,
        fighter2_odds=args.fighter2_odds,
        allow_manual_input=not args.no_manual_odds,
        use_flaresolverr=args.flaresolverr,
    )
    
    # Get inference data using the prediction_data.csv file
    # New refactored system: generates all features (fighter1_*, fighter2_*, *_diff)
    builder = InferenceDataBuilder(prediction_data_csv, fight_list, bfo_odds)
    fighter_dfs = builder.build()  # Dictionary: fighter_name -> dataframe with ALL features
    
    # Build a set of fighter1 names from the fight list
    fighter1_names = {fight[1] for fight in fight_list}
    
    # Create a background dataset for SHAP explanations
    if SHAP:
        # Sample 200 rows randomly from the training data
        background_data = df_train[feats].sample(100, random_state=42)
        
        # Scale only if using single model (ensembles handle scaling internally)
        if scaler is not None:
            features_to_scale = get_features_to_scale(background_data)
            background_data_scaled = background_data.copy()
            if len(features_to_scale) > 0:
                background_data_scaled[features_to_scale] = scaler.transform(background_data[features_to_scale])
        else:
            # Ensemble model - don't scale here (each model will scale internally)
            background_data_scaled = background_data.copy()
    
    results = []
    raw_stats_dfs = []  # Store raw dataframes with _diffs columns
    visualizer = FightVisualizer(feats, output_dir=ss_output_dir)
    if SHAP:
        shap_viz = ShapVisualizer(model, feats, output_dir=ss_output_dir)
    
    # Process each fighter (fighter1 only) once
    for fighter_name, fighter_df in fighter_dfs.items():
        # Skip if this fighter is not fighter1
        if fighter_name not in fighter1_names:
            continue
            
        # Find the corresponding fight in the fight_list
        fight = next((f for f in fight_list if f[1] == fighter_name), None)
        if not fight:
            print(f"Warning: Could not find fight for {fighter_name} in fight_list")
            continue
            
        fight_date, fighter1_name, fighter2_name = fight
        
        # Store raw dataframe with metadata for prediction_stats.csv
        raw_df_copy = fighter_df.copy()
        raw_df_copy['fighter1_name'] = fighter1_name
        raw_df_copy['fighter2_name'] = fighter2_name
        raw_df_copy['fight_date'] = fight_date
        raw_stats_dfs.append(raw_df_copy)
        
        # Filter features for this specific model
        filtered_df = filter_features_for_model(fighter_df, feats)
        
        # Extract features for prediction (filtered_df already contains only the required features)
        X = filtered_df if len(filtered_df) > 0 else pd.DataFrame()
        
        if len(X.columns) == 0:
            print(f"Warning: No features available for {fighter_name}")
            continue
        
        # Rename features to match training data format (TEMPORARY - will be removed when inference data generation is fixed)
        rename_dict = {}
        if 'fighter1_weightclass_encoded' in X.columns:
            rename_dict['fighter1_weightclass_encoded'] = 'weightclass_encoded'
        if model_type == 'win' and 'fighter1_days_since_last_fight_dec_avg' in X.columns:
            rename_dict['fighter1_days_since_last_fight_dec_avg'] = 'days_since_last_fight_dec_avg'
        if rename_dict:
            X = X.rename(columns=rename_dict)
        
        # Scale data only for single models (ensembles handle scaling internally per window)
        if scaler is not None:
            # Single model: scale the data
            features_to_scale = get_features_to_scale(X)
            scaled_X_df = X.copy()
            if len(features_to_scale) > 0:
                scaled_X_df[features_to_scale] = scaler.transform(X[features_to_scale])
        else:
            # Ensemble model: don't scale here - EnsemblePredictor will scale using each window's scaler
            scaled_X_df = X.copy()
        
        # Make prediction using the new helper function
        # Note: For ensembles, scaled_X_df is actually unscaled - EnsemblePredictor handles scaling internally
        y_pred_proba = get_predictions(model, calibrator, scaled_X_df, use_calibrated)
        
        # Get prediction probability for fighter1 winning
        fighter1_win_prob = y_pred_proba.iloc[0][1]  # Assuming 1 corresponds to fighter1 win
        
        # Get BFO odds (already retrieved at the beginning)
        fighter1_odds_data = bfo_odds.get(fighter1_name, "N/A")
        fighter2_odds_data = bfo_odds.get(fighter2_name, "N/A")
        
        # Extract original and vigless odds
        if isinstance(fighter1_odds_data, dict):
            fighter1_original_odds = fighter1_odds_data['original']
            fighter1_vigless_odds = fighter1_odds_data['vigless']
        else:
            fighter1_original_odds = fighter1_odds_data
            fighter1_vigless_odds = fighter1_odds_data
            
        if isinstance(fighter2_odds_data, dict):
            fighter2_original_odds = fighter2_odds_data['original']
            fighter2_vigless_odds = fighter2_odds_data['vigless']
        else:
            fighter2_original_odds = fighter2_odds_data
            fighter2_vigless_odds = fighter2_odds_data
        
        # Convert vigless odds to decimal for market probability calculations
        fighter1_decimal_odds = convert_american_to_decimal(fighter1_vigless_odds)
        fighter2_decimal_odds = convert_american_to_decimal(fighter2_vigless_odds)
        
        # Calculate market implied probabilities from vigless odds
        if fighter1_vigless_odds != "N/A" and fighter2_vigless_odds != "N/A":
            fighter1_market_prob = 1 / fighter1_decimal_odds if fighter1_decimal_odds > 0 else 0
            fighter2_market_prob = 1 / fighter2_decimal_odds if fighter2_decimal_odds > 0 else 0
        else:
            fighter1_market_prob = 0
            fighter2_market_prob = 0
        
        if fighter1_original_odds == "N/A":
            print(f"Warning: No BFO odds found for {fighter1_name}")
        if fighter2_original_odds == "N/A":
            print(f"Warning: No BFO odds found for {fighter2_name}")
        
        # Determine AI's pick
        if fighter1_win_prob > 0.5:
            winner = "fighter1"
            proba = fighter1_win_prob
        else:
            winner = "fighter2"
            proba = 1 - fighter1_win_prob
            
        # Store the result
        result = {
            "fighter1_name": fighter1_name,
            "fighter2_name": fighter2_name,
            "fight_date": fight_date,
            "fighter1_win_prob": fighter1_win_prob,     # AI prediction
            "fighter2_win_prob": 1 - fighter1_win_prob, # AI prediction
            "fighter1_market_prob": fighter1_market_prob,  # Market implied prob from vigless odds
            "fighter2_market_prob": fighter2_market_prob,  # Market implied prob from vigless odds
            "fighter1_odds": fighter1_original_odds,      # Original Vegas odds
            "fighter2_odds": fighter2_original_odds,      # Original Vegas odds
            "fighter1_vigless_odds": fighter1_vigless_odds,  # Vigless odds for EV calculations
            "fighter2_vigless_odds": fighter2_vigless_odds,  # Vigless odds for EV calculations
            "fighter1_decimal_odds": fighter1_decimal_odds,  # Decimal odds from vigless
            "fighter2_decimal_odds": fighter2_decimal_odds,  # Decimal odds from vigless
            "winner": winner,  # AI's pick
            "proba": proba,    # AI's confidence
        }
        
        results.append(result)
        
        # Generate visualization
        # Convert the Series to a DataFrame (add single row to DataFrame)
        fighter_stats_df = pd.DataFrame([scaled_X_df.iloc[0]])
        
        # Create individual chart for this fight
        fig_individual = visualizer.create_individual_chart(fighter_stats_df, fighter1_name, fighter2_name, win_prob=fighter1_win_prob)
        
        # Create grouped chart for this fight
        fig_grouped = visualizer.create_grouped_chart(fighter_stats_df, fighter1_name, fighter2_name, win_prob=fighter1_win_prob)
        
        # Save both visualizations
        visualizer.save_visualization(fig_individual, fighter1_name, fighter2_name, grouped=False)
        visualizer.save_visualization(fig_grouped, fighter1_name, fighter2_name, grouped=True)

        # SHAP visualization with background data
        if SHAP:
            shap_paths = shap_viz.explain_prediction(
                prediction_data=scaled_X_df,  # The current fight data
                background_data=background_data_scaled,  # Sample of training data as background
                fighter1_name=fighter1_name, 
                fighter2_name=fighter2_name, 
                win_prob=fighter1_win_prob,
                nsamples=500  # Use 500 samples for more accurate SHAP values
            )
            print(f"SHAP visualization saved to: {shap_paths['force_plot']}")
    
    # Sort results by AI confidence
    results.sort(key=lambda x: x['proba'], reverse=True)
    
    # Print the location of saved visualizations
    if hasattr(visualizer, 'output_dir') and os.path.exists(visualizer.output_dir):
        print(f"\nFight visualizations saved to: {os.path.abspath(visualizer.output_dir)}")
    
    # Print fight predictions with improved formatting - more compact version
    header_width = 174  # Correct width to match table structure
    prediction_type = "CALIBRATED" if (use_calibrated and calibrator is not None) else "ORIGINAL"
    print(f"\n+{'-' * (header_width - 2)}+")
    print(f"|{f' FIGHT PREDICTIONS ({prediction_type}) ':^{header_width - 2}}|")
    print(f"+{'-' * (header_width - 2)}+")
    
    # Column headers with more compact spacing and increased width for names
    print(f"| {'Fighter 1':^26} | {'Fighter 2':^26} | {'F1 Vegas Odds':^15} | {'F2 Vegas Odds':^15} | {'F1 Prob':^7} | {'F2 Prob':^7} | {'AI Pick':^26} | {'AI Prob':^8} | {'AI Odds':^7} | {'EV':^6} |")
    print(f"+{'-' * 28}+{'-' * 28}+{'-' * 17}+{'-' * 17}+{'-' * 9}+{'-' * 9}+{'-' * 28}+{'-' * 10}+{'-' * 9}+{'-' * 8}+")
    
    # Format each row with improved spacing and alignment
    for r in results:
        # Format odds values to show cleaner numbers
        f1_odds = r['fighter1_odds']
        f2_odds = r['fighter2_odds']
        
        # If odds are numerical values, round them
        if isinstance(f1_odds, (int, float)):
            f1_odds = f"{int(f1_odds)}" if f1_odds % 1 == 0 else f"{f1_odds:.1f}"
        if isinstance(f2_odds, (int, float)):
            f2_odds = f"{int(f2_odds)}" if f2_odds % 1 == 0 else f"{f2_odds:.1f}"
            
        # Always show + for positive American odds
        if isinstance(f1_odds, str) and f1_odds.replace('.', '', 1).isdigit() and float(f1_odds) > 0:
            f1_odds = f"+{f1_odds}"
        if isinstance(f2_odds, str) and f2_odds.replace('.', '', 1).isdigit() and float(f2_odds) > 0:
            f2_odds = f"+{f2_odds}"
            
        # Determine winner for highlighting
        winner_name = r['fighter1_name'] if r['winner'] == 'fighter1' else r['fighter2_name']
        
        # Convert AI confidence to American odds
        ai_odds = convert_prob_to_american_odds(r['proba'])
        
        # Calculate true mathematical expected value
        ai_win_prob = r['proba']  # AI win probability (0-1)
        
        # Get Vegas odds for the fighter the AI is picking
        if r['winner'] == 'fighter1':
            vegas_odds_str = str(r['fighter1_odds'])
        else:
            vegas_odds_str = str(r['fighter2_odds'])
        
        # Convert Vegas odds to decimal payout multiplier
        try:
            if vegas_odds_str.startswith('+'):
                vegas_odds_num = int(vegas_odds_str[1:])
                payout_multiplier = vegas_odds_num / 100
            elif vegas_odds_str.startswith('-'):
                vegas_odds_num = int(vegas_odds_str[1:])
                payout_multiplier = 100 / vegas_odds_num
            else:
                vegas_odds_num = int(vegas_odds_str)
                if vegas_odds_num > 0:
                    payout_multiplier = vegas_odds_num / 100
                else:
                    payout_multiplier = 100 / abs(vegas_odds_num)
            
            # Calculate EV: (AI_win_prob * payout) - (AI_lose_prob * 1)
            # For a $1 bet: win = get back $1 + payout, lose = lose $1
            ev_value = (ai_win_prob * payout_multiplier) - ((1 - ai_win_prob) * 1)
            ev_percentage = ev_value * 100  # Convert to percentage
            ev_display = f"{ev_percentage:+.0f}%"
            
        except (ValueError, TypeError):
            ev_display = "N/A"
        
        # Truncate long names with ellipsis if necessary (increased character limit)
        f1_name = r['fighter1_name']
        f2_name = r['fighter2_name']
        winner_display = winner_name
        
        if len(f1_name) > 26:
            f1_name = f1_name[:23] + "..."
        if len(f2_name) > 26:
            f2_name = f2_name[:23] + "..."
        if len(winner_display) > 26:
            winner_display = winner_display[:23] + "..."
        
        print(f"| {f1_name:^26} | {f2_name:^26} | {f1_odds:^15} | {f2_odds:^15} | "
              f"{r['fighter1_market_prob']*100:^7.1f} | {r['fighter2_market_prob']*100:^7.1f} | "
              f"{winner_display:^26} | {r['proba']*100:^8.1f} | {ai_odds:^7} | {ev_display:^6} |")
    
    print(f"+{'-' * 28}+{'-' * 28}+{'-' * 17}+{'-' * 17}+{'-' * 9}+{'-' * 9}+{'-' * 28}+{'-' * 10}+{'-' * 9}+{'-' * 8}+")
    
    # Add CSV output
    prediction_type = "CALIBRATED" if (use_calibrated and calibrator is not None) else "ORIGINAL"
    print(f"\nCSV Format ({prediction_type} predictions):")
    print("Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV")
    
    # Create a CSV file in the output directory
    csv_file_path = os.path.join(ss_output_dir, "fight_predictions.csv")
    with open(csv_file_path, "w") as csv_file:
        # Write header with prediction type comment
        csv_file.write(f"# Fight Predictions using {prediction_type} model predictions\n")
        csv_file.write("Fighter1,Fighter2,Fighter1_Odds,Fighter2_Odds,Fighter1_AI_Prob,Fighter2_AI_Prob,Fighter1_Market_Prob,Fighter2_Market_Prob,AI_Pick,Confidence,AI_Odds,EV\n")
        
        # Write data rows
        for r in results:
            # Determine winner for the CSV row
            winner_name = r['fighter1_name'] if r['winner'] == 'fighter1' else r['fighter2_name']
            # Convert AI confidence to American odds
            ai_odds = convert_prob_to_american_odds(r['proba'])
            # Check for positive EV (use original odds - what you'd actually bet at)
            bookie_odds = r['fighter1_odds'] if r['winner'] == 'fighter1' else r['fighter2_odds']
            has_ev = has_positive_ev(ai_odds, bookie_odds)
            ev_value = "1" if has_ev else "0"
            
            csv_row = f"{r['fighter1_name']},{r['fighter2_name']},{r['fighter1_odds']},{r['fighter2_odds']}," \
                     f"{r['fighter1_win_prob']*100:.1f},{r['fighter2_win_prob']*100:.1f}," \
                     f"{r['fighter1_market_prob']*100:.1f},{r['fighter2_market_prob']*100:.1f}," \
                     f"{winner_name},{r['proba']*100:.1f},{ai_odds},{ev_value}\n"
            
            # Write to file and print to console
            csv_file.write(csv_row)
            print(csv_row.strip())
    
    print(f"CSV data saved to: {csv_file_path}")
    
    # Combine and save raw prediction stats with _diffs columns
    if raw_stats_dfs:
        # Combine all raw dataframes
        combined_stats_df = pd.concat(raw_stats_dfs, ignore_index=True)
        
        # Reorder columns to put metadata first
        metadata_cols = ['fighter1_name', 'fighter2_name', 'fight_date']
        other_cols = [col for col in combined_stats_df.columns if col not in metadata_cols]
        combined_stats_df = combined_stats_df[metadata_cols + other_cols]
        
        # Save to CSV
        prediction_stats_csv_path = os.path.join(ss_output_dir, "prediction_stats.csv")
        combined_stats_df.to_csv(prediction_stats_csv_path, index=False)
        print(f"Raw prediction stats (with _diffs) saved to: {prediction_stats_csv_path}")
        print(f"  Total columns: {len(combined_stats_df.columns)}")
        print(f"  Total rows: {len(combined_stats_df)}")

    # Create parlays
    #conf_parlays = create_conf_parlays(results)
    #kelly_parlays = create_parlays(results)
    
    maybe_take_screenshots(ss_output_dir, enabled=args.screenshots)


if __name__ == "__main__":
    cli()
