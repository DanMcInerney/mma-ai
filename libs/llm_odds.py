from sqlalchemy import create_engine, text
import google.generativeai as genai
import time
import csv
import os
from datetime import datetime
from dotenv import load_dotenv

from libs.paths import database_url

load_dotenv()

engine = create_engine(database_url())
_model = None


def get_gemini_model():
    """Create the Gemini client lazily so importing this module does not require a key."""
    global _model
    if _model is None:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running libs.llm_odds.")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
    return _model


def get_fight_odds(event_date, fights):
    """
    Fetches fight odds for all fights in a UFC event using the Gemini API.

    Args:
      event_date: The date of the UFC event (as a DATE object).
      fights: A list of tuples, where each tuple contains the names of 
             the two fighters in a fight (fighter1_name, fighter2_name).

    Returns:
      A string containing the fight odds for the entire event in <odds> tags, 
      or None if no odds are found for any of the fights.
    """
    print("Getting odds for", event_date)

    # Construct the prompt with all fighter pairings for the event
    prompt = f"""
    What were the closing odds on all the fights for the UFC event on {event_date.strftime('%Y-%m-%d')}.
    The fights on the card were:
    """
    for fighter1_name, fighter2_name in fights:
        prompt += f"{fighter1_name} vs {fighter2_name}\n"

    prompt += """
    If you find multiple sources for the odds, average the results together for each fighter's odds for each fight.
    If you can't find the odds, then add the fight to the output but put their Fighter1_odds and Fighter2_odds as "nan".
    Put odds in American Moneyline format.
    Output your response in the following format:

    <odds>
    Fighter1,Fighter2,Fighter1_odds,Fighter2_odds
    </odds>

    Do not include any other information or output.
    Use exactly the same fighter names as in the input in the output, case sensitive.
    Double check your output to make sure it's in the correct format and that you include
    all necessary information: both fighter names and their odds. Should be 4 columns, fighter1, fighter2, fighter1_odds, fighter2_odds.
    """

    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = get_gemini_model().generate_content(prompt)
            try:
                odds_data = response.text.split("<odds>")[1].split("</odds>")[0].strip()
                
                # Verify data format
                is_valid = True
                for line in odds_data.split('\n'):
                    if line.strip():  # Skip empty lines
                        parts = line.strip().split(',')
                        if len(parts) != 4 or not all(parts):  # Check if we have all 4 parts and none are empty
                            is_valid = False
                            break
 
                if is_valid:
                    print(odds_data)
                    return f"<odds>\n{odds_data}\n</odds>"
                else:
                    print("Invalid data format received, retrying...")
                    print(odds_data)
                    retry_count += 1
                    if retry_count < max_retries:
                        time.sleep(15)
                        continue
                    return None

            except IndexError:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"No odds found, attempt {retry_count} of {max_retries}...")
                    time.sleep(15)  # Add small delay between retries
                    continue
                return None
                
        except Exception as e:
            if "429" in str(e):  # Rate limit exceeded
                print(f"Rate limit hit, waiting 15 seconds before retry...")
                time.sleep(15)
                continue
            else:
                print(f"Unexpected error: {e}")
                return None


def get_events_and_fights(conn):
    """
    Fetches event dates and fight details from the database starting from 2014-01-01.

    Args:
      conn: The SQLAlchemy connection object.

    Returns:
      A list of tuples, where each tuple contains:
        - event_date (DATE)
        - list of tuples with fighter names (fighter1_name, fighter2_name)
    """
    try:
        sql = text("""
            SELECT e.event_date, f1.fighter_name, f2.fighter_name
            FROM features.event_mapping e
            JOIN features.fight_mapping fm ON e.event_id = fm.event_id
            JOIN features.fighter_mapping f1 ON fm.fighter1_id = f1.fighter_id
            JOIN features.fighter_mapping f2 ON fm.fighter2_id = f2.fighter_id
            WHERE e.event_date >= '2014-01-01'
            ORDER BY e.event_date;
        """)
        
        result = conn.execute(sql)
        
        events_and_fights = []
        current_date = None
        fights = []
        for row in result:
            event_date, fighter1_name, fighter2_name = row
            if event_date != current_date:
                if current_date:
                    events_and_fights.append((current_date, fights))
                current_date = event_date
                fights = []
            fights.append((fighter1_name, fighter2_name))
        if current_date:
            events_and_fights.append((current_date, fights))
        return events_and_fights
    except Exception as error:
        print("Error while fetching data from PostgreSQL:", error)
        return None


def get_latest_processed_date(csv_file):
    """Returns the latest event_date from the CSV file, or None if empty/non-existent"""
    if not os.path.exists(csv_file):
        return None
        
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        dates = [datetime.strptime(row[4], '%Y-%m-%d').date() for row in reader]
        return max(dates) if dates else None

def write_odds_to_csv(odds_output, event_date, csv_file):
    """Writes odds data to CSV file"""
    if not odds_output:
        return
        
    odds_lines = odds_output.split('\n')[1:-1]  # Remove <odds> tags
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        for line in odds_lines:
            if line.strip():  # Skip empty lines
                fighter1, fighter2, odds1, odds2 = line.strip().split(',')
                writer.writerow([fighter1, fighter2, odds1, odds2, event_date.strftime('%Y-%m-%d')])

def process_odds(conn):
    """Modified to handle CSV writing and incremental updates"""
    csv_file = 'llm_odds.csv'
    
    # Create file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['fighter1', 'fighter2', 'fighter1_odds', 'fighter2_odds', 'event_date'])
    
    # Get latest processed date
    latest_date = get_latest_processed_date(csv_file)
    
    events_and_fights = get_events_and_fights(conn)
    if events_and_fights:
        for event_date, fights in events_and_fights:
            # Skip events we've already processed
            if latest_date and event_date <= latest_date:
                continue
                
            odds_output = get_fight_odds(event_date, fights)
            if odds_output:
                write_odds_to_csv(odds_output, event_date, csv_file)
                print(f"Processed odds for {event_date}")
            else:
                print(f"No odds found for UFC event on {event_date.strftime('%Y-%m-%d')}")
    else:
        print("No events and fights found in the database.")

if __name__ == "__main__":
    with engine.connect() as conn:
        process_odds(conn)
