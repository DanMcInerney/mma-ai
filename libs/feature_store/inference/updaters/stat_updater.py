"""Stat updater for updating time-dependent and derived statistics."""

from typing import Dict
import pandas as pd
import math
from libs.feature_store.features import BASE_STATIC_FEATS


class StatUpdater:
    """Updates time-dependent and derived statistics for fighters."""
    
    def __init__(self, fighter_dfs: Dict[str, pd.DataFrame]):
        """
        Initialize stat updater.
        
        Args:
            fighter_dfs: Dictionary mapping fighter_name to DataFrame
        """
        self.fighter_dfs = fighter_dfs.copy()  # Work with copy for immutability
    
    def update_age(self) -> Dict[str, pd.DataFrame]:
        """Update fighter's age for their upcoming fight using their DOB."""
        updated_dfs = {}
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            if 'fighter_dob' in df_copy.columns and 'event_date' in df_copy.columns:
                # Calculate age in years
                age = (
                    pd.Timedelta(df_copy.loc[df_copy.index[-1], 'event_date'] - 
                                df_copy.loc[df_copy.index[-1], 'fighter_dob']).total_seconds() 
                    / (365.25 * 24 * 60 * 60)
                )
                df_copy.loc[df_copy.index[-1], 'age'] = round(age, 3)
            updated_dfs[fighter_name] = df_copy
        return updated_dfs
    
    def update_days_since_last_fight(self) -> Dict[str, pd.DataFrame]:
        """Update fighter's days_since_last_fight for their upcoming fight."""
        updated_dfs = {}
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            if len(df_copy) > 1 and 'event_date' in df_copy.columns:
                df_copy.loc[df_copy.index[-1], 'days_since_last_fight'] = (
                    pd.Timedelta(df_copy.loc[df_copy.index[-1], 'event_date'] - 
                                df_copy.loc[df_copy.index[-2], 'event_date']).total_seconds() 
                    / (24 * 60 * 60)
                )
            updated_dfs[fighter_name] = df_copy
        return updated_dfs
    
    def update_ufcage(self) -> Dict[str, pd.DataFrame]:
        """Update fighter's UFC age (time since debut) for their upcoming fight."""
        updated_dfs = {}
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            if 'event_date' in df_copy.columns:
                # Get the fighter's first UFC fight date
                first_fight_date = df_copy['event_date'].min()
                upcoming_fight_date = df_copy.loc[df_copy.index[-1], 'event_date']
                
                # Calculate UFC age in years
                ufcage = (
                    pd.Timedelta(upcoming_fight_date - first_fight_date).total_seconds() 
                    / (365.25 * 24 * 60 * 60)
                )
                df_copy.loc[df_copy.index[-1], 'ufcage'] = round(ufcage, 3)
            updated_dfs[fighter_name] = df_copy
        return updated_dfs
    
    def update_ratios(self) -> Dict[str, pd.DataFrame]:
        """Update fighter's bounded ratios for their upcoming fight."""
        updated_dfs = {}
        fighters_to_remove = set()
        
        # Exclude categorical/encoded features that shouldn't have ratios calculated
        exclude_from_ratios = ['weightclass_encoded', 'odds']
        stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_ratios]
        
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            opponent_name = df_copy.loc[df_copy.index[-1], 'opponent']
            
            # Check if opponent exists
            if opponent_name not in self.fighter_dfs:
                print(f"Warning: No data found for opponent {opponent_name} of {fighter_name}")
                fighters_to_remove.add(fighter_name)
                continue
            
            opp_df = self.fighter_dfs[opponent_name]
            for stat in stats_to_process:
                if stat in df_copy.columns and stat in opp_df.columns:
                    df_copy.loc[df_copy.index[-1], f'{stat}_ratio'] = (
                        df_copy.loc[df_copy.index[-1], stat] / 
                        (df_copy.loc[df_copy.index[-1], stat] + opp_df.loc[opp_df.index[-1], stat])
                    )
            
            updated_dfs[fighter_name] = df_copy
        
        # Remove fighters with missing opponent data
        for fighter_name in fighters_to_remove:
            if fighter_name in updated_dfs:
                del updated_dfs[fighter_name]
        
        return updated_dfs
    
    def update_avgs(self) -> Dict[str, pd.DataFrame]:
        """Update fighter's avg stats for their upcoming fight."""
        updated_dfs = {}
        
        # Exclude categorical/encoded features that shouldn't have averages calculated
        exclude_from_avgs = ['weightclass_encoded', 'odds']
        base_stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_avgs]
        
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            stats = [stat for stat in base_stats_to_process if stat in df_copy.columns]
            stats += [f'{stat}_ratio' for stat in stats if f'{stat}_ratio' in df_copy.columns]
            
            for stat in stats:
                df_copy.loc[df_copy.index[-1], f'{stat}_avg'] = df_copy[stat].mean()
            
            updated_dfs[fighter_name] = df_copy
        
        return updated_dfs
    
    def update_dec_avgs(self) -> Dict[str, pd.DataFrame]:
        """
        Update fighter's time-decay weighted average stats for their upcoming fight.
        Uses 1.0 year half-life, consistent with TimedecAvgCalculator.
        """
        updated_dfs = {}
        from config.decay import get_decay_rate
        decay_rate = get_decay_rate()  # Uses centralized config (default: 1.25 year half-life)
        
        # Exclude categorical/encoded features that shouldn't have dec_avgs calculated
        exclude_from_dec_avgs = ['weightclass_encoded', 'odds']
        base_stats_to_process = [stat for stat in BASE_STATIC_FEATS if stat not in exclude_from_dec_avgs]
        
        for fighter_name, df in self.fighter_dfs.items():
            df_copy = df.copy()
            
            # Get upcoming fight date (last row)
            upcoming_date = df_copy.loc[df_copy.index[-1], 'event_date']
            
            # Get all previous fights (exclude the last row which is the upcoming fight)
            previous_fights = df_copy.iloc[:-1].copy()
            
            if len(previous_fights) == 0:
                print(f"Warning: No previous fights for {fighter_name}, skipping dec_avg calculation")
                updated_dfs[fighter_name] = df_copy
                continue
            
            # Calculate time differences in years
            previous_fights['time_diff_years'] = (
                (upcoming_date - previous_fights['event_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
            )
            
            # Calculate weights using exponential decay
            previous_fights['weight'] = previous_fights['time_diff_years'].apply(
                lambda x: math.exp(-1 * decay_rate * x)
            )
            
            stats = [stat for stat in base_stats_to_process if stat in previous_fights.columns]
            stats += [f'{stat}_ratio' for stat in stats if f'{stat}_ratio' in previous_fights.columns]
            
            # Calculate time-decay weighted averages
            for stat in stats:
                # Filter out null values
                valid_fights = previous_fights[previous_fights[stat].notnull()].copy()
                
                if len(valid_fights) == 0:
                    df_copy.loc[df_copy.index[-1], f'{stat}_dec_avg'] = None
                    continue
                
                # Calculate weighted sum
                weighted_sum = (valid_fights[stat] * valid_fights['weight']).sum()
                # Calculate sum of weights
                sum_weights = valid_fights['weight'].sum()
                
                # Calculate time-decay weighted average
                if sum_weights > 0:
                    df_copy.loc[df_copy.index[-1], f'{stat}_dec_avg'] = weighted_sum / sum_weights
                else:
                    df_copy.loc[df_copy.index[-1], f'{stat}_dec_avg'] = None
            
            updated_dfs[fighter_name] = df_copy
        
        return updated_dfs
    
    def update_all(self) -> Dict[str, pd.DataFrame]:
        """
        Update all statistics in sequence.
        
        Returns:
            Dictionary with updated DataFrames
        """
        updated = self.update_age()
        self.fighter_dfs = updated
        
        updated = self.update_days_since_last_fight()
        self.fighter_dfs = updated
        
        updated = self.update_ufcage()
        self.fighter_dfs = updated
        
        updated = self.update_ratios()
        self.fighter_dfs = updated
        
        updated = self.update_avgs()
        self.fighter_dfs = updated
        
        updated = self.update_dec_avgs()
        return updated

