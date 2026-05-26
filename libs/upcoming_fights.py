from libs.wikipedia_scraper import WikiTableScraper
from unidecode import unidecode
from datetime import datetime
import pandas as pd
from fuzzywuzzy import fuzz

class UpcomingFights():
    def __init__(self, df, upcoming_number):
        self.df = df
        self.upcoming_number = upcoming_number # 1 is next event, 2 is event after next

    def run(self):
        try:
            scheduled_events = self.get_scheduled_events()
        except Exception:
            scheduled_events = []
        if scheduled_events:
            event_index = self.upcoming_number - 1
            if event_index < 0 or event_index >= len(scheduled_events):
                return {}
            event_links = [scheduled_events[event_index]['url']]
        else:
            event_links = self.get_upcoming_event_links()
            event_links = [event_links[-self.upcoming_number]]
        events = self.get_upcoming_cards(event_links)
        return events

    def get_upcoming_event_links(self):
        url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
        id = {'id': 'Scheduled_events'}
        ws = WikiTableScraper(url, id)
        event_links = ws.get_table_links(1)
        return event_links

    def get_scheduled_events(self):
        url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'
        id = {'id': 'Scheduled_events'}
        ws = WikiTableScraper(url, id)
        if ws.table is None:
            return []

        scheduled_events = self._scheduled_events_from_table(ws)
        if scheduled_events:
            return sorted(scheduled_events, key=self._scheduled_event_sort_key)

        event_links = ws.get_table_links(1)
        scheduled_table = ws.get_table_by_id()
        for index, row in scheduled_table.iterrows():
            if index >= len(event_links):
                continue
            event_date = pd.to_datetime(row.get('Date'), errors='coerce')
            scheduled_events.append({
                'name': str(row.get('Event', '')).strip(),
                'date': None if pd.isna(event_date) else event_date.to_pydatetime(),
                'url': event_links[index],
            })

        return sorted(scheduled_events, key=self._scheduled_event_sort_key)

    def _scheduled_events_from_table(self, ws):
        if not hasattr(ws.table, 'select'):
            return []

        scheduled_events = []
        for row in ws.table.select('tr'):
            cells = row.find_all('td', recursive=False)
            if len(cells) < 2:
                continue
            event_link = cells[0].find('a', href=True)
            if event_link is None:
                continue
            href = event_link.get('href', '')
            if not href:
                continue
            event_url = href if href.startswith('http') else ws.base_url + href
            event_date = pd.to_datetime(cells[1].get_text(' ', strip=True), errors='coerce')
            scheduled_events.append({
                'name': cells[0].get_text(' ', strip=True),
                'date': None if pd.isna(event_date) else event_date.to_pydatetime(),
                'url': event_url,
            })

        return scheduled_events

    @staticmethod
    def _scheduled_event_sort_key(event):
        event_date = event.get('date')
        if event_date is None:
            event_date = datetime.max
        return event_date, event.get('name') or event.get('url') or ''

    def name_matching(self, name, names_list):
        best_match = None
        for n in names_list:
            ratio = fuzz.ratio(name.lower(), n.lower())
            if ratio > 90:
                print(f'[Wikipedia Name Matching] Ratio: {ratio} for {name} and {n}')
                if ratio == 100:
                    return n

                if best_match:
                    if ratio > best_match[1]:
                        best_match = (n, ratio)
                else:
                    best_match = (n, ratio)

        if best_match:
            return best_match[0]

    def get_upcoming_cards(self, event_links):
        events = {}
        id = {'class': 'toccolours'}
        
        # Get all fighter names from the DataFrame
        # Check which columns exist and extract fighter names accordingly
        if 'fighter_name' in self.df.columns:
            # If we have the new format with fighter_name column
            df_names = pd.unique(self.df['fighter_name'].values)
        else:
            # No recognized name columns
            print("Warning: Could not find fighter name columns in DataFrame")
            return {}

        # BREAK HERE IF WIKIPEDIA HAS WRONG URL
        for url in event_links:
            ws = WikiTableScraper(url, id)

            # Check for valid event data
            if 'Wikipedia does not have an article with this exact name.' in ws.soup.text or ws.table is None:
                print("No table for URL: " + url)
                continue

            fighters = ws.get_table_column(2)
            fighters = [unidecode(fighter.replace(' (c)', '').replace(' (ic)', '').replace('.', '')) for fighter in fighters]
            fighters = ws.wikipedia_name_conversion(fighters)
            opponents = ws.get_table_column(4)
            opponents = [unidecode(opponent.replace(' (c)', '').replace(' (ic)', '')) for opponent in opponents]
            opponents = ws.wikipedia_name_conversion(opponents)

            if len(fighters) != len(opponents):
                print(f'[!] Length mismatch for {url}')
                continue

            # Find first time fighters
            all_names = self.df['fighter_name'] if 'fighter_name' in self.df.columns else pd.concat([self.df['fighter1_name'], self.df['fighter2_name']])
            name_counts = all_names.value_counts()

            # Match names with DataFrame and track valid fighter pairs
            valid_pairs = {}  # Track valid fighter pairs using index
            for i, name in enumerate(fighters + opponents):
                fight_index = i if i < len(fighters) else i - len(fighters)
                if name not in df_names:
                    n = self.name_matching(name, df_names)
                    if n:
                        print(f'[*] Matched: {name} to {n}')
                        if i < len(fighters):
                            fighters[i] = n
                        else:
                            opponents[fight_index] = n
                    else:
                        print(name + ' not in df, may need name conversion in wikipedia_name_conversion()')
                        valid_pairs[fight_index] = False
                        continue
                
                # Initialize pair as valid if not already marked invalid
                if fight_index not in valid_pairs:
                    valid_pairs[fight_index] = True

            # Get event date
            date_id = {'class': 'infobox'}
            wsdate = WikiTableScraper(url, date_id)
            rdate = wsdate.get_table_column(1)
            strdate = rdate[2 if 'poster' in rdate[0].lower() else 1].split('(', 1)[1][:-1]
            date = datetime.strptime(strdate, '%Y-%m-%d')

            # Event name
            event = url.rsplit('/', 1)[1].replace('_', ' ')

            # Construct the list of tuples for each fight, only including valid fights
            fight_list = [(date, fighters[i], opponents[i]) 
                         for i in range(len(fighters)) 
                         if valid_pairs.get(i, False)]

            # Add to events dictionary
            if fight_list:  # Only add event if it has valid fights
                events[event] = fight_list

        return events

# df = pd.read_csv('../master-odds.csv')
# uf = UpcomingFights(df, latest_event=True)
# future_fights = uf.run()
