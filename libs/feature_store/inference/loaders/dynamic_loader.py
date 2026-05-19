"""Dynamic data loader for inference data creation."""

from typing import List, Tuple, Dict
import pandas as pd
from libs.feature_store.inference.loaders.base import DataLoader
from libs.feature_store.features import BASE_STATIC_FEATS


class DynamicDataLoader(DataLoader):
    """Loads dynamic features (fight statistics) from CSV."""
    
    def __init__(self, all_data: pd.DataFrame, fight_list: List[Tuple[str, str, str]]):
        """
        Initialize dynamic data loader.
        
        Args:
            all_data: Full dataset from CSV
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
        """
        self.all_data = all_data
        self.fight_list = fight_list
        self.fighter_dfs: Dict[str, pd.DataFrame] = {}
        
        # Define columns that would be considered dynamic (exclude static columns)
        self.dynamic_columns_to_exclude = ['fighter_id', 'fighter_name', 'fighter_dob', 'event_id'] + BASE_STATIC_FEATS
    
    def load_fighter_data(self, fighter_name: str) -> pd.DataFrame:
        """
        Load dynamic data for a specific fighter.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with fighter's dynamic data, or empty DataFrame if not found
        """
        # Filter data for this fighter
        fighter_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter_name.lower()].copy()
        
        if len(fighter_data) == 0:
            return pd.DataFrame()
        
        # Handle duplicate IDs
        fighter_data = self.handle_duplicate_ids(fighter_data, fighter_name)
        if fighter_data.empty:
            return pd.DataFrame()
        
        # Select all columns except static ones
        dynamic_data = fighter_data.drop(
            [col for col in self.dynamic_columns_to_exclude if col in fighter_data.columns],
            axis=1,
            errors='ignore'
        ).copy()
        
        # Add back essential columns
        for col in ['fighter_name', 'event_date', 'fighter_id']:
            if col in fighter_data.columns:
                dynamic_data[col] = fighter_data[col]
        
        return dynamic_data
    
    def load_all_fighters(self) -> Dict[str, pd.DataFrame]:
        """
        Load dynamic data for all fighters in fight_list.
        
        Returns:
            Dictionary mapping fighter_name to DataFrame
        """
        for fight_date, fighter_1, fighter_2 in self.fight_list:
            for fighter_name in [fighter_1, fighter_2]:
                if fighter_name in self.fighter_dfs:
                    continue  # Already loaded
                
                dynamic_data = self.load_fighter_data(fighter_name)
                
                if len(dynamic_data) == 0:
                    print(f"Warning: No dynamic data found for {fighter_name}")
                    continue
                
                dynamic_data = dynamic_data.sort_values(by='event_date').reset_index(drop=True)
                
                # Add placeholder row for upcoming fight
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
                self.fighter_dfs[fighter_name] = dynamic_data
                
                print(f"Loaded {len(dynamic_data) - 1} dynamic fights for {fighter_name}")
        
        return self.fighter_dfs
    
    def filter_fighters(self, fight_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Filter fight list to remove fights with insufficient data.
        
        This is handled during load_all_fighters, so we just return the list.
        
        Args:
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            
        Returns:
            Filtered fight list
        """
        return self.fight_list

