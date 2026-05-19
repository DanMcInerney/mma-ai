"""Experience enricher for adding fight experience features."""

import pandas as pd
from libs.feature_store.inference.enrichers.base import Enricher


class ExperienceEnricher(Enricher):
    """Adds fight experience features to DataFrames."""
    
    def __init__(self, all_data: pd.DataFrame):
        """
        Initialize experience enricher.
        
        Args:
            all_data: Full dataset from CSV for counting fights
        """
        self.all_data = all_data
    
    def enrich(self, df: pd.DataFrame, fighter1_name: str = None, fighter2_name: str = None) -> pd.DataFrame:
        """
        Add fight experience features to DataFrame.
        
        Args:
            df: DataFrame to enrich
            fighter1_name: Name of fighter1
            fighter2_name: Name of fighter2 (opponent)
            
        Returns:
            Enriched DataFrame
        """
        df_copy = df.copy()
        
        if fighter1_name and fighter2_name:
            # Count total fights from the CSV data (using the fight_id column for counting)
            fighter1_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter1_name.lower()]
            fighter2_data = self.all_data[self.all_data['fighter_name'].str.lower() == fighter2_name.lower()]
            
            # Get total fights for both fighters
            fighter1_fights = fighter1_data['fight_id'].nunique() if len(fighter1_data) > 0 else 0
            fighter2_fights = fighter2_data['fight_id'].nunique() if len(fighter2_data) > 0 else 0
            
            # Add to DataFrame
            df_copy['fighter1_total_fights'] = fighter1_fights
            df_copy['fighter2_total_fights'] = fighter2_fights
            df_copy['combined_fights'] = fighter1_fights + fighter2_fights
        
        return df_copy

