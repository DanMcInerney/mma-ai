"""Static data loader for inference data creation."""

from typing import List, Tuple, Dict
import pandas as pd
from libs.feature_store.inference.loaders.base import DataLoader
from libs.feature_store.features import BASE_STATIC_FEATS


class StaticDataLoader(DataLoader):
    """Loads static features (age, reach, height, etc.) from CSV."""
    
    def __init__(self, all_data: pd.DataFrame, fight_list: List[Tuple[str, str, str]]):
        """
        Initialize static data loader.
        
        Args:
            all_data: Full dataset from CSV
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
        """
        self.all_data = all_data
        self.fight_list = fight_list
        self.fighter_dfs: Dict[str, pd.DataFrame] = {}
        
        # Define the static columns we're interested in
        base_static_columns = ['fighter_id', 'fighter_name', 'fighter_dob', 'event_id', 'event_date'] + BASE_STATIC_FEATS
        
        # Add ratio versions of the static stats (exclude categorical/encoded features)
        exclude_from_ratios = ['weightclass_encoded', 'odds']
        stats_for_ratios = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_ratios]
        ratio_columns = [f'{stat}_ratio' for stat in stats_for_ratios]
        
        # Combine them into a single list of columns to extract
        self.static_columns = base_static_columns + ratio_columns
    
    def load_fighter_data(self, fighter_name: str) -> pd.DataFrame:
        """
        Load static data for a specific fighter.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with fighter's static data, or empty DataFrame if not found
        """
        # Filter data for this fighter
        fighter_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter_name.lower()].copy()
        
        if len(fighter_data) == 0:
            return pd.DataFrame()
        
        # Handle duplicate IDs
        fighter_data = self.handle_duplicate_ids(fighter_data, fighter_name)
        if fighter_data.empty:
            return pd.DataFrame()
        
        # Select only the columns we need
        static_data = fighter_data[
            [col for col in self.static_columns if col in fighter_data.columns]
        ].copy()
        
        return static_data
    
    def load_all_fighters(self) -> Dict[str, pd.DataFrame]:
        """
        Load static data for all fighters in fight_list.
        
        Returns:
            Dictionary mapping fighter_name to DataFrame
        """
        fights_to_remove = set()
        
        for fight_date, fighter_1, fighter_2 in self.fight_list:
            for fighter_name in [fighter_1, fighter_2]:
                if fighter_name in self.fighter_dfs:
                    continue  # Already loaded
                
                static_data = self.load_fighter_data(fighter_name)
                
                if len(static_data) == 0:
                    print(f"Warning: No data found for {fighter_name}")
                    fights_to_remove.add((fight_date, fighter_1, fighter_2))
                    continue
                
                # Check if fighter has only 1 previous fight
                if len(static_data) <= 1:
                    print(f"Warning: {fighter_name} has insufficient fight history (only {len(static_data)} fights)")
                    fights_to_remove.add((fight_date, fighter_1, fighter_2))
                    continue
                
                static_data = static_data.sort_values(by='event_date').reset_index(drop=True)
                
                # Add placeholder row for upcoming fight
                last_row = static_data.iloc[-1:].copy()
                static_data = pd.concat([static_data, last_row], ignore_index=True)
                static_data.loc[static_data.index[-1], 'event_date'] = pd.to_datetime(fight_date)
                
                # Add opponent and fighter1 columns
                opponent = fighter_2 if fighter_name == fighter_1 else fighter_1
                static_data.loc[static_data.index[-1], 'opponent'] = opponent
                static_data.loc[static_data.index[-1], 'fighter1'] = fighter_name == fighter_1
                
                # Ensure proper datetime conversions
                static_data['event_date'] = pd.to_datetime(static_data['event_date'])
                if 'fighter_dob' in static_data.columns:
                    static_data['fighter_dob'] = pd.to_datetime(static_data['fighter_dob'])
                
                # Store the DataFrame
                self.fighter_dfs[fighter_name] = static_data
                
                print(f"Loaded {len(static_data) - 1} static fights for {fighter_name}")
        
        # Remove fights where either fighter has insufficient history
        self.fight_list = [fight for fight in self.fight_list if fight not in fights_to_remove]
        if fights_to_remove:
            print(f"\nRemoved {len(fights_to_remove)} fights due to insufficient fight history")
        
        return self.fighter_dfs
    
    def filter_fighters(self, fight_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Filter fight list to remove fights with insufficient data.
        
        This is handled during load_all_fighters, so we just return the filtered list.
        
        Args:
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            
        Returns:
            Filtered fight list
        """
        return self.fight_list

