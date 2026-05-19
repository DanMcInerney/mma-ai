from typing import Union, List, Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor
import psycopg2.extras
import io
from libs.feature_store.base import BaseFeatureStore
import re

class CoreFeatureStore(BaseFeatureStore):

    def __init__(self, connection):
        """Initialize with an open connection instead of engine"""
        super().__init__(connection)
        self.chunk_size = 1000

    def convert_height_to_inches(self, height_str):
        """Convert height string to total inches"""
        if pd.isna(height_str):
            return None
        try:
            feet, inches = height_str.split("'")
            return int(feet) * 12 + int(inches.strip('"'))
        except (ValueError, AttributeError):
            return None

    def convert_reach_to_inches(self, reach_str):
        """Convert reach string to inches"""
        if pd.isna(reach_str):
            return None
        try:
            return int(reach_str.strip('"'))
        except (ValueError, AttributeError):
            return None

    def load_fighter_features(self, df: pd.DataFrame) -> None:
        fighter_data = df[[
            'url', 'name', 'nickname', 'stance', 
            'weight', 'dob', 'height', 'reach'
        ]].copy()

        # Drop duplicates first
        fighter_data = fighter_data.drop_duplicates(subset=['url'], keep='last')

        # Replace '--' and empty strings with NaN for all columns
        fighter_data = fighter_data.replace(['--', ''], np.nan)
        
        # Rename columns to match schema
        fighter_data = fighter_data.rename(columns={
            'url': 'fighter_url',
            'name': 'fighter_name',
            'nickname': 'fighter_nickname',
            'stance': 'fighter_stance',
            'weight': 'fighter_weight',
            'height': 'fighter_height',
            'reach': 'fighter_reach',
            'dob': 'fighter_dob'
        })
        
        # Clean and transform data
        fighter_data['fighter_height'] = fighter_data['fighter_height'].apply(self.convert_height_to_inches).astype('Int64')
        fighter_data['fighter_reach'] = fighter_data['fighter_reach'].apply(self.convert_reach_to_inches).astype('Int64')
        fighter_data['fighter_weight'] = fighter_data['fighter_weight'].apply(
            lambda x: x.replace(' lbs.', '').lower() if pd.notna(x) else None
        )
        
        # Handle dates - convert to datetime and replace NaT with None
        fighter_data['fighter_dob'] = pd.to_datetime(fighter_data['fighter_dob'], errors='coerce')
        fighter_data['fighter_dob'] = fighter_data['fighter_dob'].where(fighter_data['fighter_dob'].notna(), None)
        
        # Convert string columns to lowercase, only if they contain strings
        string_columns = ['fighter_name', 'fighter_nickname', 'fighter_stance']
        for col in string_columns:
            mask = fighter_data[col].notna()
            if mask.any():
                fighter_data.loc[mask, col] = fighter_data.loc[mask, col].str.lower()
        
        # Drop duplicates to avoid unique constraint violations
        fighter_data = fighter_data.drop_duplicates(subset=['fighter_url'])
        
        # Use the most efficient insertion method
        self.bulk_insert_dataframe(
            fighter_data,
            'fighter_mapping',
            if_exists='append',
            schema='features'
        )

    def load_fight_features(self, df: pd.DataFrame) -> None:
        print(f"Loading {len(df)} new fights...")
        
        # Check if events exist first
        event_urls = df['event_url'].unique()
        existing_events = pd.read_sql(
            text('SELECT event_url FROM features.event_mapping WHERE event_url = ANY(:urls)'),
            self.conn,
            params={'urls': list(event_urls)}
        )
        
        # Only load missing events
        missing_events = set(event_urls) - set(existing_events['event_url'])
        if missing_events:
            print(f"Loading {len(missing_events)} missing events...")
            self.load_event_features(df[df['event_url'].isin(missing_events)])
        
        # Replace '--' with None
        df = df.replace('--', None)

        # Process in chunks for memory efficiency
        for chunk in np.array_split(df, max(1, len(df) // self.chunk_size)):
            chunk = chunk.drop_duplicates(subset=['event_url', 'player1_url', 'player2_url'], keep='last')
            fight_data = self._transform_fight_data(chunk)
            self.bulk_insert_dataframe(fight_data, 'fight_mapping', if_exists='append')

    def load_event_features(self, fight_df: pd.DataFrame) -> None:            
        
        print(f"Loading {len(fight_df)} new events...")
        """Extract and load event data from fight features"""
        # Extract unique event data
        event_data = fight_df[[
            'event_url', 'event_date', 'event_location'
        ]].drop_duplicates(subset=['event_url'], keep='last')

        # Replace '--' and empty strings with NaN
        event_data = event_data.replace(['--', ''], np.nan)
        
        # Convert date to datetime
        event_data['event_date'] = pd.to_datetime(event_data['event_date'], errors='coerce')
        event_data['event_date'] = event_data['event_date'].where(event_data['event_date'].notna(), None)
        
        # Clean location strings
        event_data['event_location'] = event_data['event_location'].str.strip()
        
        # Use the most efficient insertion method
        self.bulk_insert_dataframe(
            event_data,
            'event_mapping',
            if_exists='append',
            schema='features'
        )

    def _convert_time_to_seconds(self, time_str):
        """Convert MM:SS time string to total seconds"""
        if pd.isna(time_str):
            return None
        try:
            minutes, seconds = time_str.split(':')
            result = int(minutes) * 60 + int(seconds)
            return result  # No minimum constraint applied
        except (ValueError, AttributeError):
            return 0  # Return 0 for invalid values

    def _convert_time_format(self, format_str, end_time):
        """Convert time format string to list of seconds per round"""
        if pd.isna(format_str):
            return None
        try:
            if format_str == "No Time Limit":
                return end_time
            
            # Extract numbers between parentheses
            match = re.search(r'\(([\d-]+)\)', format_str)
            if match:
                # Convert each number to seconds (minutes * 60)
                round_times = [int(x) * 60 for x in match.group(1).split('-')]
                # Return comma-separated string of times
                return ','.join(str(x) for x in round_times)
            return None
        except (ValueError, AttributeError):
            return None

    def _transform_fight_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform the fight data for the fight_mapping table
        fight_data = df[[
            'event_url', 'player1_url', 'player2_url', 
            'weightclass', 'method', 'details', 'round', 'time', 
            'time_format', 'result'
        ]].copy()
        
        # Get the IDs from the mapping tables using existing connection
        # Get event_id
        event_ids = pd.read_sql(
            text('SELECT event_url, event_id FROM features.event_mapping'),
            self.conn
        ).set_index('event_url')['event_id']
        
        # Get fighter_ids
        fighter_ids = pd.read_sql(
            text('SELECT fighter_url, fighter_id FROM features.fighter_mapping'),
            self.conn
        ).set_index('fighter_url')['fighter_id']
        
        # Map the IDs
        fight_data['event_id'] = fight_data['event_url'].map(event_ids)
        fight_data['fighter1_id'] = fight_data['player1_url'].map(fighter_ids)
        fight_data['fighter2_id'] = fight_data['player2_url'].map(fighter_ids)
        # Convert result to integers, W = 1, L = 0, D = 2, NC = 3
        fight_data['result'] = fight_data['result'].str.lower().map({
            'w': 1,
            'l': 0,
            'd': 2,
            'nc': 3
        })
        
        # Convert end_time to seconds
        fight_data['end_time'] = fight_data['time'].apply(self._convert_time_to_seconds).astype('Int64')
        
        # Convert time_format
        fight_data['time_format'] = fight_data.apply(
            lambda x: self._convert_time_format(x['time_format'], x['end_time']), 
            axis=1
        )

        # Clean weightclass
        fight_data['weightclass'] = fight_data['weightclass'].apply(self._clean_weightclass)
        
        # Encode weightclass
        fight_data['weightclass_encoded'] = fight_data['weightclass'].apply(self._encode_weightclass)
        
        # Clean up and reorder columns
        fight_data = fight_data.rename(columns={
            'round': 'end_round'
        })
        
        return fight_data[[
            'event_id', 'fighter1_id', 'fighter2_id',
            'weightclass', 'weightclass_encoded', 'method', 'details', 'end_round', 'end_time',
            'time_format', 'result'
        ]]
    
    def _clean_weightclass(self, weightclass_str):
        """Clean weightclass string"""
        if pd.isna(weightclass_str):
            return None
            
        weightclasses = [
            "strawweight", "flyweight", "bantamweight",
            "featherweight", "lightweight", "welterweight",
            "middleweight", "light heavyweight", "heavyweight",
            "catchweight", "open weight"
        ]
        
        is_womens = "women's" in weightclass_str.lower()
        weightclass_str = weightclass_str.lower()
        
        if "catch weight" in weightclass_str or "catchweight" in weightclass_str:
            weightclass_str = weightclass_str.replace("catch weight", "catchweight")
        
        found_weightclass = None
        for wc in weightclasses:
            if wc in weightclass_str:
                found_weightclass = wc
                break
        
        if found_weightclass:
            if is_womens:
                return f"women's {found_weightclass}"
            return found_weightclass
        
        return "open weight"
    
    def _encode_weightclass(self, weightclass_str):
        """
        Encode weightclass as numeric values (lightest to heaviest).
        
        Encoding:
        0 = Catchweight
        1 = Strawweight (lightest)
        2 = Flyweight
        3 = Bantamweight
        4 = Featherweight
        5 = Lightweight
        6 = Welterweight
        7 = Middleweight
        8 = Light Heavyweight
        9 = Heavyweight / Open Weight (heaviest)
        
        Args:
            weightclass_str: Cleaned weightclass string
            
        Returns:
            Integer encoding (0-9) or None if weightclass is None
        """
        if pd.isna(weightclass_str):
            return None
        
        weightclass_str = str(weightclass_str).lower().strip()
        
        # Catchweight = 0
        if "catchweight" in weightclass_str:
            return 0
        
        # Open weight = same as heavyweight (9)
        if "open weight" in weightclass_str or "openweight" in weightclass_str:
            return 9
        
        # Remove "women's" prefix if present for encoding
        if "women's" in weightclass_str:
            weightclass_str = weightclass_str.replace("women's", "").strip()
        
        # Map weightclasses from lightest (1) to heaviest (9)
        weightclass_map = {
            'strawweight': 1,
            'flyweight': 2,
            'bantamweight': 3,
            'featherweight': 4,
            'lightweight': 5,
            'welterweight': 6,
            'middleweight': 7,
            'light heavyweight': 8,
            'heavyweight': 9
        }
        
        # Find matching weightclass
        for wc, encoding in weightclass_map.items():
            if wc in weightclass_str:
                return encoding
        
        # Default to heavyweight if not found
        return 9

    def _split_values(self, series):
        # Handle NaN and '--' values
        mask = series.notna() & (series != '--')
        result = pd.DataFrame(index=series.index, columns=['land', 'att'])
        result.loc[~mask] = 0
        
        # Split valid values
        valid = series[mask].str.split(' of ', expand=True).astype(float)
        if not valid.empty:
            result.loc[mask] = valid
        
        return result.astype(int)

    def _process_split_stats(self, df: pd.DataFrame, prefix: str, stat: str, new_stat: str) -> tuple:
        """Vectorized processing of X of Y stats"""
        cols = [f'{prefix}rd{i}_{stat}' for i in range(1, 6)]
        
        # Process all rounds at once
        results = {}
        for i in range(1, 6):
            results[f'{new_stat}_land_rd{i}'] = []
            results[f'{new_stat}_att_rd{i}'] = []
        
        for i, col in enumerate(cols, 1):
            if col in df.columns:
                split = self._split_values(df[col])
                results[f'{new_stat}_land_rd{i}'] = split['land']
                results[f'{new_stat}_att_rd{i}'] = split['att']
            else:
                results[f'{new_stat}_land_rd{i}'] = pd.Series(0, index=df.index)
                results[f'{new_stat}_att_rd{i}'] = pd.Series(0, index=df.index)
        
        return results

    def load_fight_stats(self, df: pd.DataFrame) -> None:
        print(f"Loading stats for {len(df)} new fights...")

        # 1) Pull the relevant mapping tables from the DB
        fight_mapping = pd.read_sql(text('''
            SELECT fight_id, event_id, fighter1_id, fighter2_id 
            FROM features.fight_mapping
        '''), self.conn)

        fighter_mapping = pd.read_sql(text('''
            SELECT fighter_id, fighter_url 
            FROM features.fighter_mapping
        '''), self.conn)

        event_mapping = pd.read_sql(text('''
            SELECT event_id, event_url, event_date
            FROM features.event_mapping
        '''), self.conn)

        # 2) Merge your new stats df with the fighter_mapping twice
        #    to get fighter_id_1 and fighter_id_2:
        df = (
            df.merge(
                fighter_mapping,
                how='left',
                left_on='player1_url',
                right_on='fighter_url'
            )
            .merge(
                fighter_mapping,
                how='left',
                left_on='player2_url',
                right_on='fighter_url',
                suffixes=('_1', '_2')
            )
        )

        # 3) Merge df with event_mapping to get event_id
        #    (assuming df has an 'event_url' or 'event_date'; pick the one you rely on).
        #    If your DataFrame has 'event_url', do this:
        df = df.merge(
            event_mapping[['event_id', 'event_url', 'event_date']],
            how='left',
            on='event_url'
        )

        # Now df has:
        #   fighter_id_1, fighter_id_2, event_id
        #   (plus possibly event_date too, if needed)

        # 4) Merge df with fight_mapping on (event_id, fighter1_id, fighter2_id)
        df = df.merge(
            fight_mapping[['fight_id', 'event_id', 'fighter1_id', 'fighter2_id']],
            how='left',
            left_on=['event_id', 'fighter_id_1', 'fighter_id_2'],
            right_on=['event_id', 'fighter1_id', 'fighter2_id']
        )

        # Now df finally has a unique fight_id for each row.

        # 5) Drop duplicates if necessary (some data sources can repeat rows)
        df = df.drop_duplicates(
            subset=['fight_id', 'fighter_id_1', 'fighter_id_2', 'event_id'],
            keep='last'
        )

        # 6) Validate that every fight_id is not null
        #    (Optional) If you want to skip stats for fights not in fight_mapping:
        df = df.dropna(subset=['fight_id'])
        # or convert to int:
        df['fight_id'] = df['fight_id'].astype(int)

        # 7) Process each fighter's stats
        all_stats = []
        for fighter_num in [1, 2]:
            prefix = f'p{fighter_num}_'
            base_data = {
                'fight_id': df['fight_id'],
                'fighter_id': df[f'fighter_id_{fighter_num}'],
                'event_id': df['event_id']
            }

            # Process all split stats at once
            split_stats = {
                'Sig_str': 'sig_str',
                'Total_str': 'strikes',
                'Td': 'td',
                'Head': 'head',
                'Body': 'body',
                'Leg': 'leg',
                'Distance': 'distance',
                'Clinch': 'clinch',
                'Ground': 'ground'
            }

            stats_data = {}
            for old_stat, new_stat in split_stats.items():
                results = self._process_split_stats(df, prefix, old_stat, new_stat)
                stats_data.update(results)

            # Process integer stats
            int_stats = {
                'KD': 'kd',
                'Sub_att': 'sub_att',
                'Rev': 'rev'
            }

            for old_stat, new_stat in int_stats.items():
                for round_num in range(1, 6):
                    col = f'{prefix}rd{round_num}_{old_stat}'
                    if col in df.columns:
                        stats_data[f'{new_stat}_rd{round_num}'] = df[col].fillna(0).astype(int)
                    else:
                        stats_data[f'{new_stat}_rd{round_num}'] = pd.Series(0, index=df.index)

            # Process control time separately
            for round_num in range(1, 6):
                col = f'{prefix}rd{round_num}_Ctrl'
                if col in df.columns:
                    stats_data[f'ctrl_rd{round_num}'] = df[col].apply(
                        lambda x: self._convert_time_to_seconds(x) if pd.notna(x) else 0
                    ).fillna(0).astype(int)
                else:
                    stats_data[f'ctrl_rd{round_num}'] = pd.Series(0, index=df.index)

            # Combine into a single DataFrame for these fighter stats
            fighter_stats = pd.DataFrame({**base_data, **stats_data})
            all_stats.append(fighter_stats)

        # 8) Concatenate fighter1 + fighter2 rows, insert
        all_stats_df = pd.concat(all_stats, ignore_index=True)

        # 9) Insert into fight_stats_core
        self.bulk_insert_dataframe(
            all_stats_df,
            'fight_stats_core',
            if_exists='append',
            schema='features'
        )