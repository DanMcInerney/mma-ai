"""Feature transformer that generates all feature types (fighter1_*, fighter2_*, *_diff)."""

from typing import List, Set
import pandas as pd


class FeatureTransformer:
    """
    Transforms fighter stats into standardized feature format.
    
    Always generates three column types for each stat:
    - fighter1_<stat>: Fighter1's absolute stat value
    - fighter2_<stat>: Fighter2's absolute stat value
    - <stat>_diff: Difference (fighter1 - fighter2)
    """
    
    def __init__(self, metadata_cols: List[str] = None):
        """
        Initialize feature transformer.
        
        Args:
            metadata_cols: List of metadata column names to exclude from transformation
        """
        self.metadata_cols = metadata_cols or [
            # Core metadata
            'fighter_name', 'opponent', 'event_date', 'fighter1_id', 
            'fighter2_id', 'fighter1', 'fighter_id', 'event_id',
            # CSV metadata columns
            'fight_id', 'method', 'result', 'weightclass', 'fighter_dob',
            # Dynamic suffixes that might be added
            'fighter_name_dynamic', 'event_date_dynamic', 'fighter_id_dynamic'
        ]
    
    def transform(
        self,
        fighter1_df: pd.DataFrame,
        fighter2_df: pd.DataFrame,
        available_stats: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate all three column types for each available stat.
        
        Args:
            fighter1_df: DataFrame with fighter1's stats (should have final row for upcoming fight)
            fighter2_df: DataFrame with fighter2's stats (should have final row for upcoming fight)
            available_stats: Optional list of stat names to include. If None, uses all columns.
            
        Returns:
            DataFrame with columns: fighter1_<stat>, fighter2_<stat>, <stat>_diff
        """
        result = {}
        
        # Get the final row (upcoming fight) for each fighter
        fighter1_row = fighter1_df.iloc[-1] if len(fighter1_df) > 0 else pd.Series()
        fighter2_row = fighter2_df.iloc[-1] if len(fighter2_df) > 0 else pd.Series()
        
        # Determine which stats to process
        if available_stats is None:
            # Use all columns from both dataframes, excluding metadata
            all_cols = set(fighter1_df.columns) | set(fighter2_df.columns)
            stats_to_process = [col for col in all_cols if col not in self.metadata_cols]
        else:
            stats_to_process = available_stats
        
        # Generate features for each stat
        for stat in stats_to_process:
            # Skip metadata columns
            if stat in self.metadata_cols:
                continue
            
            # Skip odds features and categorical features from being diff'd
            if 'odds' in stat.lower() or stat == 'weightclass_encoded':
                # For these, only include fighter1 version (not diff)
                if stat in fighter1_row.index:
                    result[f'fighter1_{stat}'] = fighter1_row[stat]
                continue
            
            # Generate fighter1_<stat>
            if stat in fighter1_row.index:
                result[f'fighter1_{stat}'] = fighter1_row[stat]
            
            # Generate fighter2_<stat>
            if stat in fighter2_row.index:
                result[f'fighter2_{stat}'] = fighter2_row[stat]
            
            # Generate <stat>_diff (only if both fighters have the stat)
            if stat in fighter1_row.index and stat in fighter2_row.index:
                f1_val = fighter1_row[stat]
                f2_val = fighter2_row[stat]
                
                # Handle NaN values
                if pd.isna(f1_val) or pd.isna(f2_val):
                    result[f'{stat}_diff'] = None
                else:
                    result[f'{stat}_diff'] = f1_val - f2_val
        
        # Create DataFrame from result dictionary
        if result:
            return pd.DataFrame([result])
        else:
            return pd.DataFrame()
    
    def get_available_features(self, fighter1_df: pd.DataFrame, fighter2_df: pd.DataFrame) -> Set[str]:
        """
        Get set of all available feature names that would be generated.
        
        Args:
            fighter1_df: DataFrame with fighter1's stats
            fighter2_df: DataFrame with fighter2's stats
            
        Returns:
            Set of feature names (fighter1_*, fighter2_*, *_diff)
        """
        features = set()
        
        # Get all columns from both dataframes
        all_cols = set(fighter1_df.columns) | set(fighter2_df.columns)
        stats_to_process = [col for col in all_cols if col not in self.metadata_cols]
        
        for stat in stats_to_process:
            if stat in self.metadata_cols:
                continue
            
            if 'odds' in stat.lower() or stat == 'weightclass_encoded':
                if stat in fighter1_df.columns:
                    features.add(f'fighter1_{stat}')
                continue
            
            if stat in fighter1_df.columns:
                features.add(f'fighter1_{stat}')
            
            if stat in fighter2_df.columns:
                features.add(f'fighter2_{stat}')
            
            if stat in fighter1_df.columns and stat in fighter2_df.columns:
                features.add(f'{stat}_diff')
        
        return features

