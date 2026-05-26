import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from typing import Set, List, Tuple, Dict
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.features import BASE_STATIC_FEATS
from libs.feature_store.inference.loaders.base import select_duplicate_fighter_id
from config.decay import get_decay_rate
import math

class CreateInferenceData:
    def __init__(self, csv_path: str, feats: List[str], fight_list: List[Tuple[str, str, str]], bfo_odds: Dict[str, int] = None):
        """
        filter down to only include the upcoming fighters
        recalc age
        recalc ufcage
        recalc days_since_last_fight
        recalc static diffs
        recalc dynamic diffs
        
        Args:
            csv_path: Path to the CSV file containing prediction data
            feats: List of features to use for prediction
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            bfo_odds: Dictionary mapping fighter names to their vigless American odds
        """
        self.csv_path = csv_path
        self.fight_list = fight_list
        self.feats = feats
        self.bfo_odds = bfo_odds or {}
        self.static_stats = [c.replace('_diff', '') for c in feats if any(x in c for x in BASE_STATIC_FEATS)]
        self.static_fighter_dfs = {}
        self.dynamic_fighter_dfs = {}
        self.fighter_dfs = {}
        self.all_data = None
        self.metadata_cols = ['fighter_name', 'opponent', 'event_date', 'fighter1_id', 'fighter2_id', 'fighter1']
        
        # Load the full dataset from CSV
        print(f"\n=== Loading data from {csv_path} ===")
        self.all_data = pd.read_csv(csv_path)
        
        # Convert date columns to datetime
        if 'event_date' in self.all_data.columns:
            self.all_data['event_date'] = pd.to_datetime(self.all_data['event_date'])
        if 'fighter_dob' in self.all_data.columns:
            self.all_data['fighter_dob'] = pd.to_datetime(self.all_data['fighter_dob'])
        
        print(f"Loaded {len(self.all_data)} rows with {len(self.all_data.columns)} columns")

    def load_static_data(self):
        """Load individual fighter stats into separate DataFrames."""
        print("\n=== Loading Fighter Data ===")
        
        # Define the static columns we're interested in
        base_static_columns = ['fighter_id', 'fighter_name', 'fighter_dob', 'event_id', 'event_date'] + BASE_STATIC_FEATS
        
        # Add ratio versions of the static stats (exclude categorical/encoded features)
        exclude_from_ratios = ['weightclass_encoded', 'odds']
        stats_for_ratios = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_ratios]
        ratio_columns = [f'{stat}_ratio' for stat in stats_for_ratios]
        
        # Combine them into a single list of columns to extract
        static_columns = base_static_columns + ratio_columns
        
        fights_to_remove = set()
        
        for fight_date, fighter_1, fighter_2 in self.fight_list:
            for fighter_name in [fighter_1, fighter_2]:
                # Filter data for this fighter
                fighter_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter_name.lower()]

                # Check for and handle multiple fighter_ids *before* creating static_data
                if 'fighter_id' in fighter_data.columns:
                    unique_ids = fighter_data['fighter_id'].dropna().unique()
                    if len(unique_ids) > 1:
                        print(f"Warning: Multiple fighter_ids found for {fighter_name}: {unique_ids}")
                        
                        print(f"--- Information to help select the correct ID for {fighter_name} ---")
                        for fid in unique_ids:
                            fighter_history_for_id = self.all_data[self.all_data['fighter_id'] == fid].copy()
                            if not fighter_history_for_id.empty:
                                # Ensure event_date is datetime, already done in __init__ if column exists
                                # but good to be sure if accessing directly before __init__ guarantees it for all_data
                                fighter_history_for_id['event_date'] = pd.to_datetime(fighter_history_for_id['event_date'])
                                fighter_history_for_id = fighter_history_for_id.sort_values(by='event_date', ascending=False)
                                if not fighter_history_for_id.empty:
                                    last_recorded_fight = fighter_history_for_id.iloc[0]
                                    date_of_last_fight = last_recorded_fight['event_date'].strftime('%Y-%m-%d')
                                    print(f"  ID {fid}: Last fought on {date_of_last_fight}.")
                                else:
                                    print(f"  ID {fid}: No fights found for this ID after sorting.") # Should not happen if first check passed
                            else:
                                print(f"  ID {fid}: No fight history found in the dataset for this specific ID.")
                        print("--- End of disambiguation information ---")
                        
                        correct_fighter_id = select_duplicate_fighter_id(fighter_data, fighter_name)
                        if correct_fighter_id is None:
                            print(f"Error: Could not select a fighter_id for {fighter_name}. Skipping fight involving {fighter_name}.")
                            fights_to_remove.add((fight_date, fighter_1, fighter_2))
                            continue # Skip processing this fighter for this fight
                        # Filter fighter_data to use only the selected ID
                        fighter_data = fighter_data[fighter_data['fighter_id'] == correct_fighter_id].copy()
                    elif len(unique_ids) == 0:
                         print(f"Warning: No fighter_id found for {fighter_name}. Skipping fight involving {fighter_name}.")
                         fights_to_remove.add((fight_date, fighter_1, fighter_2))
                         continue # Skip processing this fighter for this fight

                # Select only the columns we need
                static_data = fighter_data[
                    [col for col in static_columns if col in fighter_data.columns]
                ].copy()
                
                if len(static_data) > 0:
                    # Check if fighter has only 1 previous fight
                    if len(static_data) <= 1:
                        print(f"Warning: {fighter_name} has insufficient fight history (only {len(static_data)} fights)")
                        fights_to_remove.add((fight_date, fighter_1, fighter_2))
                        continue
                        
                    static_data = static_data.sort_values(by='event_date').reset_index(drop=True)
                    last_row = static_data.iloc[-1:].copy()
                    static_data = pd.concat([static_data, last_row], ignore_index=True)
                    static_data.loc[static_data.index[-1], 'event_date'] = pd.to_datetime(fight_date)
                    
                    # Add opponent and fighter1 columns
                    opponent = fighter_2 if fighter_name == fighter_1 else fighter_1
                    static_data.loc[static_data.index[-1], 'opponent'] = opponent
                    static_data.loc[static_data.index[-1], 'fighter1'] = fighter_name == fighter_1
                    
                    # Ensure proper datetime conversions
                    static_data['event_date'] = pd.to_datetime(static_data['event_date'])
                    static_data['fighter_dob'] = pd.to_datetime(static_data['fighter_dob'])
                    
                    # Store the DataFrame
                    self.static_fighter_dfs[fighter_name] = static_data
                    
                    print(f"Loaded {len(static_data) - 1} fights for {fighter_name}")
                else:
                    print(f"Warning: No data found for {fighter_name}")
                    fights_to_remove.add((fight_date, fighter_1, fighter_2))

        # Remove fights where either fighter has insufficient history
        self.fight_list = [fight for fight in self.fight_list if fight not in fights_to_remove]
        if fights_to_remove:
            print(f"\nRemoved {len(fights_to_remove)} fights due to insufficient fight history")

    def update_age(self):
        """Update fighter's age for their upcoming fight using their DOB."""
        for fighter_name, df in self.static_fighter_dfs.items():
            # Calculate age in years
            age = (
                pd.Timedelta(df.loc[df.index[-1], 'event_date'] - df.loc[df.index[-1], 'fighter_dob']).total_seconds() 
                / (365.25 * 24 * 60 * 60)
            )
            df.loc[df.index[-1], 'age'] = round(age, 3)
    
    def update_days_since_last_fight(self):
        """Update fighter's days_since_last_fight for their upcoming fight."""
        for fighter_name, df in self.static_fighter_dfs.items():
            df.loc[df.index[-1], 'days_since_last_fight'] = (
                pd.Timedelta(df.loc[df.index[-1], 'event_date'] - df.loc[df.index[-2], 'event_date']).total_seconds() 
                / (24 * 60 * 60)
            )
    
    def update_ufcage(self):
        """Update fighter's UFC age (time since debut) for their upcoming fight."""
        for fighter_name, df in self.static_fighter_dfs.items():
            # Get the fighter's first UFC fight date
            first_fight_date = df['event_date'].min()
            upcoming_fight_date = df.loc[df.index[-1], 'event_date']
            
            # Calculate UFC age in years
            ufcage = (
                pd.Timedelta(upcoming_fight_date - first_fight_date).total_seconds() 
                / (365.25 * 24 * 60 * 60)
            )
            df.loc[df.index[-1], 'ufcage'] = round(ufcage, 3)

    def update_ratios(self):
        """Update fighter's bounded ratios for their upcoming fight."""
        fighters_to_remove = set()
        
        # Exclude categorical/encoded features that shouldn't have ratios calculated
        exclude_from_ratios = ['weightclass_encoded', 'odds']
        stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_ratios]
        
        for fighter_name, df in self.static_fighter_dfs.items():
            opponent_name = df.loc[df.index[-1], 'opponent']
            
            # Check if opponent exists in static_fighter_dfs
            if opponent_name not in self.static_fighter_dfs:
                print(f"Warning: No data found for opponent {opponent_name} of {fighter_name}")
                fighters_to_remove.add(fighter_name)
                continue
            
            opp_df = self.static_fighter_dfs[opponent_name]
            for stat in stats_to_process:
                if stat in df.columns and stat in opp_df.columns:
                    df.loc[df.index[-1], f'{stat}_ratio'] = df.loc[df.index[-1], stat] / (df.loc[df.index[-1], stat] + opp_df.loc[opp_df.index[-1], stat])
        
        # Remove fighters with missing opponent data
        for fighter_name in fighters_to_remove:
            del self.static_fighter_dfs[fighter_name]
            # Also remove the fight from fight_list
            self.fight_list = [fight for fight in self.fight_list 
                              if fighter_name.lower() not in (fight[1].lower(), fight[2].lower())]

    def update_avgs(self):
        """Update fighter's avg stats for their upcoming fight."""
        # Exclude categorical/encoded features that shouldn't have averages calculated
        exclude_from_avgs = ['weightclass_encoded', 'odds']
        base_stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_avgs]
        
        for fighter_name, df in self.static_fighter_dfs.items():
            stats = [stat for stat in base_stats_to_process if stat in df.columns]
            stats += [f'{stat}_ratio' for stat in stats if f'{stat}_ratio' in df.columns]
            
            for stat in stats:
                df.loc[df.index[-1], f'{stat}_avg'] = df[stat].mean()
    
    def update_dec_avgs(self):
        """Update fighter's time-decay weighted average stats for their upcoming fight.
        Uses 1.0 year half-life, consistent with TimedecAvgCalculator.
        
        Includes the current fight for static features (age, reach, etc.) since they are known
        pre-fight. Current fight gets weight=1.0 (time_diff=0), matching training behavior.
        """
        decay_rate = get_decay_rate()  # Uses centralized config (default: 1.0 year half-life)
        
        for fighter_name, df in self.static_fighter_dfs.items():
            # Get upcoming fight date (last row)
            upcoming_date = df.loc[df.index[-1], 'event_date']
            
            # Include ALL fights for static features (current fight has time_diff=0, weight=1.0)
            all_fights = df.copy()
            
            if len(all_fights) == 0:
                print(f"Warning: No fights for {fighter_name}, skipping dec_avg calculation")
                continue
            
            # Calculate time differences in years (current fight will have 0 difference)
            all_fights['time_diff_years'] = (upcoming_date - all_fights['event_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
            
            # Calculate weights using exponential decay (current fight gets weight = exp(0) = 1.0)
            all_fights['weight'] = all_fights['time_diff_years'].apply(lambda x: math.exp(-1 * decay_rate * x))
            
            # Get stats to process (exclude categorical/encoded features)
            exclude_from_dec_avgs = ['weightclass_encoded', 'odds']
            base_stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_dec_avgs]
            stats = [stat for stat in base_stats_to_process if stat in all_fights.columns]
            stats += [f'{stat}_ratio' for stat in stats if f'{stat}_ratio' in all_fights.columns]
            
            # Calculate time-decay weighted averages
            for stat in stats:
                # Filter out null values
                valid_fights = all_fights[all_fights[stat].notnull()].copy()
                
                if len(valid_fights) == 0:
                    df.loc[df.index[-1], f'{stat}_dec_avg'] = None
                    continue
                    
                # Calculate weighted sum
                weighted_sum = (valid_fights[stat] * valid_fights['weight']).sum()
                # Calculate sum of weights
                sum_weights = valid_fights['weight'].sum()
                
                # Calculate time-decay weighted average
                if sum_weights > 0:
                    df.loc[df.index[-1], f'{stat}_dec_avg'] = weighted_sum / sum_weights
                else:
                    df.loc[df.index[-1], f'{stat}_dec_avg'] = None

    def load_dynamic_data(self):
        """Load dynamic stats for the upcoming fight."""
        # Define columns that would be considered dynamic (exclude static columns)
        dynamic_columns_to_exclude = ['fighter_id', 'fighter_name', 'fighter_dob', 'event_id'] + BASE_STATIC_FEATS
        
        for fight_date, fighter_1, fighter_2 in self.fight_list:
            for fighter_name in [fighter_1, fighter_2]:
                # Filter data for this fighter
                fighter_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter_name.lower()]

                # Check for and handle multiple fighter_ids *before* creating dynamic_data
                if 'fighter_id' in fighter_data.columns:
                    unique_ids = fighter_data['fighter_id'].dropna().unique()
                    if len(unique_ids) > 1:
                        print(f"Warning: Multiple fighter_ids found for {fighter_name} (dynamic data): {unique_ids}")
                        correct_fighter_id = select_duplicate_fighter_id(fighter_data, fighter_name)
                        if correct_fighter_id is None:
                            print(f"Error: Could not select a fighter_id for {fighter_name}. Skipping dynamic data for {fighter_name}.")
                            continue
                        # Filter fighter_data to use only the selected ID
                        fighter_data = fighter_data[fighter_data['fighter_id'] == correct_fighter_id].copy()
                    elif len(unique_ids) == 0:
                        print(f"Warning: No fighter_id found for {fighter_name} (dynamic data). Skipping dynamic data.")
                        continue

                # Select all columns except static ones
                dynamic_data = fighter_data.drop(
                    [col for col in dynamic_columns_to_exclude if col in fighter_data.columns],
                    axis=1, 
                    errors='ignore'
                ).copy()
                
                # Add back essential columns
                for col in ['fighter_name', 'event_date', 'fighter_id']:
                    if col in fighter_data.columns:
                        dynamic_data[col] = fighter_data[col]
            
                if len(dynamic_data) > 0:
                    dynamic_data = dynamic_data.sort_values(by='event_date').reset_index(drop=True)
                    last_row = dynamic_data.iloc[-1:].copy()
                    dynamic_data = pd.concat([dynamic_data, last_row], ignore_index=True)
                    dynamic_data.loc[dynamic_data.index[-1], 'event_date'] = pd.to_datetime(fight_date)
                    
                    # Add opponent and fighter1 columns
                    opponent = fighter_2 if fighter_name == fighter_1 else fighter_1
                    dynamic_data.loc[dynamic_data.index[-1], 'opponent'] = opponent
                    dynamic_data.loc[dynamic_data.index[-1], 'fighter1'] = fighter_name == fighter_1
                    
                    # Ensure proper datetime conversions
                    dynamic_data['event_date'] = pd.to_datetime(dynamic_data['event_date'])
                    
                    # Store the DataFrame
                    self.dynamic_fighter_dfs[fighter_name] = dynamic_data

                    print(f"Loaded {len(dynamic_data) - 1} dynamic fights for {fighter_name}") # Adjusted print statement index
                else:
                    print(f"Warning: No dynamic data found for {fighter_name}")

    def keep_final_row(self, fighter_dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Keep only the final row of each fighter's DataFrame."""
        result_dfs = {}
        for fighter_name, fighter_df in fighter_dfs.items():
            row = fighter_df.iloc[-1:].copy()
            result_dfs[fighter_name] = row
        return result_dfs

    def combine_static_and_dynamic(self):
        """Combine the static and dynamic stats."""
        fighter_dfs = {}
        for fighter_name in self.static_fighter_dfs:
            if fighter_name not in self.dynamic_fighter_dfs:
                print(f"Warning: No dynamic data for {fighter_name}, using static data only")
                fighter_dfs[fighter_name] = self.static_fighter_dfs[fighter_name].copy()
                continue
                
            static_df = self.static_fighter_dfs[fighter_name]
            dynamic_df = self.dynamic_fighter_dfs[fighter_name]
            
            # Identify common columns for merging
            common_cols = [col for col in ['fighter_name', 'event_date', 'fighter_id', 'opponent', 'fighter1'] 
                          if col in static_df.columns and col in dynamic_df.columns]
            
            # Merge on common columns
            fighter_dfs[fighter_name] = pd.merge(
                static_df, 
                dynamic_df,
                on=common_cols,
                how='outer',
                suffixes=('', '_dynamic')
            )
            # Just keep the non-diff columns that actually exist in the dataframe
            non_diff_cols = [col.replace('_diff', '') for col in self.feats]
            available_cols = self.metadata_cols + [col for col in non_diff_cols if col in fighter_dfs[fighter_name].columns]
            fighter_dfs[fighter_name] = fighter_dfs[fighter_name][available_cols]
            # Remove duplicate columns that may arise from static/dynamic overlap
            if fighter_dfs[fighter_name].columns.duplicated().any():
                fighter_dfs[fighter_name] = fighter_dfs[fighter_name].loc[:, ~fighter_dfs[fighter_name].columns.duplicated()]

        return fighter_dfs
    
    def subtract_fighter2_from_fighter1(self):
        """Subtract fighter2 from fighter1 and create diff columns."""
        fighter_dfs_diff = {}
        
        for fighter_name, df in self.fighter_dfs.items():
            # Ensure we don't have duplicate columns before calculations
            if df.columns.duplicated().any():
                df = df.loc[:, ~df.columns.duplicated()]
                self.fighter_dfs[fighter_name] = df

            if df.fighter1.iloc[-1]:
                fighter2_name = df.opponent.iloc[-1]
                if fighter2_name not in self.fighter_dfs:
                    print(f"Warning: No data for opponent {fighter2_name}, skipping diff calculation for {fighter_name}")
                    continue
                    
                fighter2_df = self.fighter_dfs[fighter2_name]

                if fighter2_df.columns.duplicated().any():
                    fighter2_df = fighter2_df.loc[:, ~fighter2_df.columns.duplicated()]
                    self.fighter_dfs[fighter2_name] = fighter2_df

                # Create dictionary with all columns at once
                data = {
                    'fighter_name': fighter_name,
                    'opponent': fighter2_name
                }
                
                # Get all columns that aren't metadata and should be diff'd
                # Exclude odds features and categorical/encoded features from being diff'd
                # - odds: standalone implied probabilities
                # - weightclass_encoded: categorical encoding (0-9), doesn't make sense to diff
                cols_to_diff = [col for col in df.columns 
                              if col not in self.metadata_cols 
                              and 'odds' not in col.lower()
                              and col != 'weightclass_encoded']
                
                # Add all diff calculations to the data dictionary
                for col in cols_to_diff:
                    if col in fighter2_df.columns:
                        value_f1 = df[col]
                        if isinstance(value_f1, pd.DataFrame):
                            value_f1 = value_f1.iloc[:, 0]
                        value_f2 = fighter2_df[col]
                        if isinstance(value_f2, pd.DataFrame):
                            value_f2 = value_f2.iloc[:, 0]

                        value_f1 = value_f1.iloc[-1] if isinstance(value_f1, pd.Series) else value_f1
                        value_f2 = value_f2.iloc[-1] if isinstance(value_f2, pd.Series) else value_f2

                        data[f"{col}_diff"] = value_f1 - value_f2
                    else:
                        print(f"Warning: Column {col} not found in opponent data, skipping diff calculation")
                
                # Add raw non-diff features that are needed (like odds features)
                raw_features_needed: List[str] = []
                # First, include any raw features explicitly requested in feats
                for feat in self.feats:
                    if not feat.endswith('_diff') and feat not in self.metadata_cols and feat not in raw_features_needed:
                        raw_features_needed.append(feat)
                # Ensure static stats (like days_since_last_fight_dec_avg) are included even if only _diff versions were requested
                for stat in self.static_stats:
                    if not stat.endswith('_diff') and stat not in self.metadata_cols and stat not in raw_features_needed:
                        raw_features_needed.append(stat)
                for feat in raw_features_needed:
                    if feat in df.columns:
                        value = df[feat]
                        if isinstance(value, pd.DataFrame):
                            value = value.iloc[:, 0]
                        value = value.iloc[-1] if isinstance(value, pd.Series) else value
                        data[feat] = value
                
                # Create DataFrame all at once
                fighter_dfs_diff[fighter_name] = pd.DataFrame([data])
                
        return fighter_dfs_diff

    def american_to_implied_probability(self, american_odds):
        """Convert American odds to implied probability (0.0 to 1.0)."""
        if american_odds == "N/A" or american_odds is None:
            return None
        
        try:
            odds = float(american_odds)
            if odds > 0:
                # Positive odds: +150 = 150/(150+100) = 0.4 = 40% implied probability
                return 100 / (odds + 100)
            else:
                # Negative odds: -150 = 150/(150+100) = 0.6 = 60% implied probability
                return abs(odds) / (abs(odds) + 100)
        except (ValueError, TypeError):
            return None

    def add_odds_features(self):
        """Add BFO odds features to fighter dataframes."""
        if not self.bfo_odds:
            print("Warning: No BFO odds provided, skipping odds features")
            return
            
        for fighter_name, df in self.fighter_dfs.items():
            # Only add odds features to Fighter1's dataframe since Fighter2's dataframe gets discarded anyway
            is_fighter1 = df['fighter1'].iloc[0] if 'fighter1' in df.columns else None
            
            if is_fighter1:
                # This is Fighter1's dataframe - add the odds feature
                opponent_name = df['opponent'].iloc[0] if 'opponent' in df.columns else None
                
                # Get Fighter1's odds for the f1_sevenday_vigless_ip_opening_odds feature
                fighter1_odds = self.bfo_odds.get(fighter_name, "N/A")
                fighter1_ip = self.american_to_implied_probability(fighter1_odds)
                
                # f1_sevenday_vigless_ip_opening_odds = Fighter1's implied probability of winning
                df['f1_sevenday_vigless_ip_opening_odds'] = fighter1_ip
                
                if fighter1_ip is not None:
                    print(f"Added odds feature for {fighter_name} (Fighter1): f1_ip={fighter1_ip:.3f}")
                    print(f"  → Fighter1 odds: {fighter1_odds} → IP: {fighter1_ip:.3f}")
                else:
                    print(f"Added odds feature for {fighter_name} (Fighter1): f1_ip=None (no odds)")
                    print(f"  → Fighter1 odds: {fighter1_odds} → IP: None")
            # Skip Fighter2 dataframes since they're only used for diff calculation then discarded

    def add_fight_experience(self):
        """Add total fights for each fighter and combined fights for the matchup."""
        fighter_dfs_copy = self.fighter_dfs.copy()
        
        for fighter_name, df in fighter_dfs_copy.items():
            # Get fighter1 and fighter2 names
            fighter1_name = df['fighter_name'].iloc[0]
            fighter2_name = df['opponent'].iloc[0]
            
            # Count total fights from the CSV data (using the fight_id column for counting)
            fighter1_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter1_name.lower()]
            fighter2_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter2_name.lower()]
            
            # Get total fights for both fighters
            fighter1_fights = fighter1_data['fight_id'].nunique() if len(fighter1_data) > 0 else 0
            fighter2_fights = fighter2_data['fight_id'].nunique() if len(fighter2_data) > 0 else 0
            
            # Add to DataFrame
            df['fighter1_total_fights'] = fighter1_fights
            df['fighter2_total_fights'] = fighter2_fights
            df['combined_fights'] = fighter1_fights + fighter2_fights
            self.fighter_dfs[fighter_name] = df

    def run(self):
        # Create the static stats
        self.load_static_data()
        self.update_age()
        self.update_days_since_last_fight()
        self.update_ufcage()
        self.update_ratios()
        self.update_avgs()
        self.update_dec_avgs()
        self.static_fighter_dfs = self.keep_final_row(self.static_fighter_dfs)

        # Create the dynamic stats
        self.load_dynamic_data()
        self.dynamic_fighter_dfs = self.keep_final_row(self.dynamic_fighter_dfs)

        # Combine the static and dynamic stats
        self.fighter_dfs = self.combine_static_and_dynamic()
        
        # Add BFO odds features to the combined dataframes
        self.add_odds_features()

        # Subtract fighter2 from fighter1
        self.fighter_dfs = self.subtract_fighter2_from_fighter1()
        
        # Add fight experience data
        self.add_fight_experience()

        return self.fighter_dfs
    
