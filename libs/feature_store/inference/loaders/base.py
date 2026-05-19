"""Base class for data loaders."""

from abc import ABC, abstractmethod
from typing import List, Tuple
import pandas as pd


class DataLoader(ABC):
    """Abstract base class for loading fighter data from CSV."""
    
    @abstractmethod
    def load_fighter_data(self, fighter_name: str) -> pd.DataFrame:
        """
        Load data for a specific fighter.
        
        Args:
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with fighter's historical data
        """
        pass
    
    @abstractmethod
    def filter_fighters(self, fight_list: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Filter fight list to remove fights with insufficient data.
        
        Args:
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            
        Returns:
            Filtered fight list
        """
        pass
    
    def handle_duplicate_ids(self, fighter_data: pd.DataFrame, fighter_name: str) -> pd.DataFrame:
        """
        Handle cases where a fighter has multiple fighter_ids.
        
        Args:
            fighter_data: DataFrame with potential duplicate IDs
            fighter_name: Name of the fighter
            
        Returns:
            DataFrame with single fighter_id
        """
        if 'fighter_id' not in fighter_data.columns:
            return fighter_data
            
        unique_ids = fighter_data['fighter_id'].unique()
        if len(unique_ids) > 1:
            print(f"Warning: Multiple fighter_ids found for {fighter_name}: {unique_ids}")
            
            print(f"--- Information to help select the correct ID for {fighter_name} ---")
            for fid in unique_ids:
                fighter_history_for_id = fighter_data[fighter_data['fighter_id'] == fid].copy()
                if not fighter_history_for_id.empty:
                    if 'event_date' in fighter_history_for_id.columns:
                        fighter_history_for_id['event_date'] = pd.to_datetime(fighter_history_for_id['event_date'])
                        fighter_history_for_id = fighter_history_for_id.sort_values(by='event_date', ascending=False)
                        if not fighter_history_for_id.empty:
                            last_recorded_fight = fighter_history_for_id.iloc[0]
                            date_of_last_fight = last_recorded_fight['event_date'].strftime('%Y-%m-%d')
                            print(f"  ID {fid}: Last fought on {date_of_last_fight}.")
                        else:
                            print(f"  ID {fid}: No fights found for this ID after sorting.")
                    else:
                        print(f"  ID {fid}: No event_date column found.")
                else:
                    print(f"  ID {fid}: No fight history found in the dataset for this specific ID.")
            print("--- End of disambiguation information ---")
            
            correct_fighter_id_str = input(f"Please enter the correct fighter_id for {fighter_name}: ")
            try:
                correct_fighter_id = pd.to_numeric(correct_fighter_id_str)
                if correct_fighter_id not in unique_ids:
                    print(f"Error: Entered ID {correct_fighter_id} is not one of the found IDs {unique_ids}.")
                    return pd.DataFrame()  # Return empty DataFrame to signal error
                fighter_data = fighter_data[fighter_data['fighter_id'] == correct_fighter_id].copy()
                print(f"Using fighter_id {correct_fighter_id} for {fighter_name}.")
            except ValueError:
                print(f"Error: Invalid input '{correct_fighter_id_str}'.")
                return pd.DataFrame()
        elif len(unique_ids) == 0:
            print(f"Warning: No fighter_id found for {fighter_name}.")
            return pd.DataFrame()
            
        return fighter_data

