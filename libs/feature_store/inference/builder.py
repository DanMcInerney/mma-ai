"""Inference data builder - orchestrates the entire inference data creation pipeline."""

from typing import List, Tuple, Dict
import pandas as pd
from libs.feature_store.inference.loaders.static_loader import StaticDataLoader
from libs.feature_store.inference.loaders.dynamic_loader import DynamicDataLoader
from libs.feature_store.inference.updaters.stat_updater import StatUpdater
from libs.feature_store.inference.transformers.feature_transformer import FeatureTransformer
from libs.feature_store.inference.enrichers.odds_enricher import OddsEnricher
from libs.feature_store.inference.enrichers.experience_enricher import ExperienceEnricher


class InferenceDataBuilder:
    """
    Orchestrates the creation of inference data for UFC fight prediction models.
    
    Always generates all feature types:
    - fighter1_<stat>: Fighter1's absolute stat value
    - fighter2_<stat>: Fighter2's absolute stat value
    - <stat>_diff: Difference (fighter1 - fighter2)
    
    Feature filtering happens later in predict.py based on model requirements.
    """
    
    def __init__(
        self,
        csv_path: str,
        fight_list: List[Tuple[str, str, str]],
        bfo_odds: Dict[str, int] = None
    ):
        """
        Initialize inference data builder.
        
        Args:
            csv_path: Path to the CSV file containing prediction data
            fight_list: List of tuples (fight_date, fighter1_name, fighter2_name)
            bfo_odds: Dictionary mapping fighter names to their vigless American odds
        """
        self.csv_path = csv_path
        self.fight_list = fight_list
        self.bfo_odds = bfo_odds or {}
        
        # Load the full dataset from CSV
        print(f"\n=== Loading data from {csv_path} ===")
        self.all_data = pd.read_csv(csv_path)
        
        # Convert date columns to datetime
        if 'event_date' in self.all_data.columns:
            self.all_data['event_date'] = pd.to_datetime(self.all_data['event_date'])
        if 'fighter_dob' in self.all_data.columns:
            self.all_data['fighter_dob'] = pd.to_datetime(self.all_data['fighter_dob'])
        
        print(f"Loaded {len(self.all_data)} rows with {len(self.all_data.columns)} columns")
        
        # Initialize components
        self.static_loader = StaticDataLoader(self.all_data, self.fight_list)
        self.dynamic_loader = DynamicDataLoader(self.all_data, self.fight_list)
        self.transformer = FeatureTransformer()
        self.odds_enricher = OddsEnricher(self.bfo_odds)
        self.experience_enricher = ExperienceEnricher(self.all_data)
    
    def build(self) -> Dict[str, pd.DataFrame]:
        """
        Build inference data for all fights.
        
        Returns:
            Dictionary mapping fighter1_name to DataFrame with all features
            (fighter1_<stat>, fighter2_<stat>, <stat>_diff)
        """
        # Step 1: Load static data
        print("\n=== Loading Static Data ===")
        static_fighter_dfs = self.static_loader.load_all_fighters()
        self.fight_list = self.static_loader.fight_list  # Update fight list after filtering
        
        if not static_fighter_dfs:
            print("Warning: No static data loaded. Returning empty dictionary.")
            return {}
        
        # Step 2: Update time-dependent stats
        print("\n=== Updating Time-Dependent Stats ===")
        stat_updater = StatUpdater(static_fighter_dfs)
        static_fighter_dfs = stat_updater.update_all()
        
        # Step 3: Keep only final row (upcoming fight) for static data
        static_fighter_dfs = {
            name: df.iloc[-1:].copy() 
            for name, df in static_fighter_dfs.items()
        }
        
        # Step 4: Load dynamic data
        print("\n=== Loading Dynamic Data ===")
        self.dynamic_loader.fight_list = self.fight_list  # Use filtered fight list
        dynamic_fighter_dfs = self.dynamic_loader.load_all_fighters()
        
        # Step 5: Keep only final row (upcoming fight) for dynamic data
        dynamic_fighter_dfs = {
            name: df.iloc[-1:].copy() 
            for name, df in dynamic_fighter_dfs.items()
        }
        
        # Step 6: Combine static and dynamic stats
        print("\n=== Combining Static and Dynamic Stats ===")
        combined_fighter_dfs = {}
        
        for fighter_name in static_fighter_dfs:
            static_df = static_fighter_dfs[fighter_name]
            
            if fighter_name in dynamic_fighter_dfs:
                dynamic_df = dynamic_fighter_dfs[fighter_name]
                
                # Identify common columns for merging
                common_cols = [
                    col for col in ['fighter_name', 'event_date', 'fighter_id', 'opponent', 'fighter1'] 
                    if col in static_df.columns and col in dynamic_df.columns
                ]
                
                # Merge on common columns
                combined_df = pd.merge(
                    static_df,
                    dynamic_df,
                    on=common_cols,
                    how='outer',
                    suffixes=('', '_dynamic')
                )
                
                # Remove duplicate columns
                if combined_df.columns.duplicated().any():
                    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                
                combined_fighter_dfs[fighter_name] = combined_df
            else:
                print(f"Warning: No dynamic data for {fighter_name}, using static data only")
                combined_fighter_dfs[fighter_name] = static_df
        
        # Step 7: Transform features (generate fighter1_*, fighter2_*, *_diff)
        print("\n=== Transforming Features ===")
        final_fighter_dfs = {}
        
        for fighter_name, df in combined_fighter_dfs.items():
            # Only process Fighter1 dataframes (Fighter2 will be included via transformer)
            is_fighter1 = df['fighter1'].iloc[0] if 'fighter1' in df.columns else False
            
            if is_fighter1:
                fighter2_name = df['opponent'].iloc[0] if 'opponent' in df.columns else None
                
                if fighter2_name and fighter2_name in combined_fighter_dfs:
                    fighter1_df = df
                    fighter2_df = combined_fighter_dfs[fighter2_name]
                    
                    # Transform features
                    transformed_df = self.transformer.transform(fighter1_df, fighter2_df)
                    
                    # Add metadata columns
                    metadata_cols = ['fighter_name', 'opponent', 'event_date', 'fighter1_id', 'fighter2_id', 'fighter1']
                    for col in metadata_cols:
                        if col in fighter1_df.columns:
                            transformed_df[col] = fighter1_df[col].iloc[0] if len(fighter1_df) > 0 else None
                    
                    # Add odds features
                    transformed_df = self.odds_enricher.enrich(transformed_df, fighter_name)
                    
                    # Add experience features
                    transformed_df = self.experience_enricher.enrich(
                        transformed_df, 
                        fighter1_name=fighter_name,
                        fighter2_name=fighter2_name
                    )
                    
                    final_fighter_dfs[fighter_name] = transformed_df
                else:
                    print(f"Warning: No opponent data found for {fighter_name}, skipping transformation")
            # Skip Fighter2 dataframes - they're included via transformer
        
        print(f"\n=== Inference Data Creation Complete ===")
        print(f"Created inference data for {len(final_fighter_dfs)} fighters")
        
        return final_fighter_dfs

