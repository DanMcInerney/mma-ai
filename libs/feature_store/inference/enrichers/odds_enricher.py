"""Odds enricher for adding BFO odds features."""

from typing import Dict
import pandas as pd
from libs.feature_store.inference.enrichers.base import Enricher


class OddsEnricher(Enricher):
    """Adds BFO odds features to DataFrames."""
    
    def __init__(self, bfo_odds: Dict[str, int] = None):
        """
        Initialize odds enricher.
        
        Args:
            bfo_odds: Dictionary mapping fighter names to their vigless American odds
        """
        self.bfo_odds = bfo_odds or {}
    
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
    
    def enrich(self, df: pd.DataFrame, fighter_name: str = None) -> pd.DataFrame:
        """
        Add BFO odds features to DataFrame.
        
        Args:
            df: DataFrame to enrich
            fighter_name: Name of the fighter (for Fighter1 only)
            
        Returns:
            Enriched DataFrame
        """
        if not self.bfo_odds:
            return df
        
        df_copy = df.copy()
        
        # Only add odds features to Fighter1's dataframe
        is_fighter1 = df_copy['fighter1'].iloc[0] if 'fighter1' in df_copy.columns else False
        
        if is_fighter1 and fighter_name:
            # Get Fighter1's odds for the f1_sevenday_vigless_ip_opening_odds feature
            fighter1_odds = self.bfo_odds.get(fighter_name, "N/A")
            fighter1_ip = self.american_to_implied_probability(fighter1_odds)
            
            # f1_sevenday_vigless_ip_opening_odds = Fighter1's implied probability of winning
            df_copy['f1_sevenday_vigless_ip_opening_odds'] = fighter1_ip
            
            if fighter1_ip is not None:
                print(f"Added odds feature for {fighter_name} (Fighter1): f1_ip={fighter1_ip:.3f}")
                print(f"  → Fighter1 odds: {fighter1_odds} → IP: {fighter1_ip:.3f}")
            else:
                print(f"Added odds feature for {fighter_name} (Fighter1): f1_ip=None (no odds)")
                print(f"  → Fighter1 odds: {fighter1_odds} → IP: None")
        
        return df_copy

