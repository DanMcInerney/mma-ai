"""
Features module for UFC fight prediction.
Contains baseline features and constants.
"""
from sqlalchemy import create_engine, text
from typing import List, Optional
from libs.paths import database_url

class FeatureSelector:
    """
    A class to help create feature lists based on patterns for different stat categories.
    
    Usage:
        selector = FeatureSelector(available_features)
        patterns = {
            'age_': ['age_dec_avg_diff', 'age_ratio_diff'],
            'ctrl_': ['ctrl_rd1_*', 'ctrl_*_per_min_*'],
            'head_': ['head_acc_*', 'head_def_*', 'head_land_dec_avg_diff']
        }
        features = selector.select_features(patterns)
    """
    
    def __init__(self, available_features=None, db_url=None):
        """
        Initialize the FeatureSelector.
        
        Args:
            available_features: List of available feature names. If None, queries database.
            db_url: Database URL for querying available features.
        """
        if available_features is None:
            self.available_features = self._query_database_features(db_url or database_url())
        else:
            self.available_features = available_features
    
    def _query_database_features(self, db_url: str) -> List[str]:
        """
        Query the database to get all available features from features.<stat> tables.
        
        Args:
            db_url: Database connection URL
            
        Returns:
            List of available feature names
        """
        try:
            engine = create_engine(db_url)
            all_features = []
            
            # Query features from BASE_STATIC_FEATS and BASE_DYNAMIC_FEATS tables
            all_stats = BASE_STATIC_FEATS + BASE_DYNAMIC_FEATS
            
            with engine.connect() as conn:
                for stat in all_stats:
                    try:
                        # Check if the table exists and get its columns
                        query = text(f"""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_schema = 'features' 
                            AND table_name = '{stat}'
                            AND column_name NOT IN ('fight_id', 'fighter_id', 'event_id')
                            ORDER BY column_name
                        """)
                        result = conn.execute(query)
                        columns = [row[0] for row in result.fetchall()]
                        all_features.extend(columns)
                    except Exception as e:
                        print(f"Warning: Could not query features.{stat} table: {e}")
                        continue
            
            # Remove duplicates and sort
            unique_features = sorted(list(set(all_features)))
            print(f"Found {len(unique_features)} features from database")
            return unique_features
            
        except Exception as e:
            print(f"Error querying database for features: {e}")
            print("Falling back to TEST_FEATS")
            return TEST_FEATS
    
    def _match_pattern(self, pattern, features):
        """
        Match a single pattern against a list of features.
        
        For patterns with multiple wildcards like 'ctrl_*per_min*', this applies
        sequential filtering: first finds features matching the first part,
        then filters those to match subsequent parts.
        
        Args:
            pattern: Pattern string with wildcards (*) 
            features: List of feature names to match against
            
        Returns:
            List of matching feature names
        """
        # Handle exact matches (no wildcards)
        if '*' not in pattern:
            return [f for f in features if f == pattern]
        
        # Split pattern by * to get individual filter parts
        parts = pattern.split('*')
        
        # Remove empty parts (happens with consecutive * or leading/trailing *)
        parts = [part for part in parts if part]
        
        if not parts:
            # Pattern was just '*' - return all features
            return list(features)
        
        # Apply sequential filtering
        current_features = list(features)
        
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                continue
                
            filtered_features = []
            for feature in current_features:
                if i == 0:
                    # First part: must start with this part
                    if feature.startswith(part):
                        filtered_features.append(feature)
                else:
                    # Subsequent parts: must contain this part
                    if part in feature:
                        filtered_features.append(feature)
            
            current_features = filtered_features
            
            # If no features left, break early
            if not current_features:
                break
        
        return current_features
    
    def _filter_by_category(self, category_prefix, features):
        """
        Filter features that start with the category prefix.
        
        Args:
            category_prefix: Prefix to filter by (e.g., 'ctrl_', 'ko_')
            features: List of features to filter
            
        Returns:
            List of features that start with the prefix
        """
        return [f for f in features if f.startswith(category_prefix)]
    
    def select_features(self, pattern_dict):
        """
        Select features based on category patterns.
        
        Args:
            pattern_dict: Dictionary where keys are category prefixes (e.g., 'ctrl_') 
                         and values are lists of patterns to match within that category
                         
        Returns:
            List of unique feature names that match the patterns
        """
        selected_features = []
        
        for category_prefix, patterns in pattern_dict.items():
            # First filter by category
            category_features = self._filter_by_category(category_prefix, self.available_features)
            
            # Then apply each pattern within the category
            for pattern in patterns:
                # Always apply patterns to the category-filtered features first
                # This ensures patterns like 'ctrl_*_per_min_*' only match ctrl features
                matches = self._match_pattern(pattern, category_features)
                selected_features.extend(matches)
        
        # Return unique features, preserving order
        seen = set()
        unique_features = []
        for feature in selected_features:
            if feature not in seen:
                seen.add(feature)
                unique_features.append(feature)
        
        return unique_features
    
    def get_pattern_dict_example(self):
        """
        Returns an example pattern dictionary based on the user's requirements.
        """
        return {
            'age_': [
                'age_dec_avg_diff',
                'age_ratio_diff'
            ],
            'reach_': [
                'reach_ratio_dec_avg_diff'
            ],
            'ufcage_': [
                'ufcage_dec_avg_diff'
            ],
            'days_since_last_fight_': [
                'days_since_last_fight_dec_avg_diff'
            ],
            'sig_str_': [
                'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
                'sig_str_land_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'head_': [
                'head_acc_*dec_avg',
                'head_def_*dec_avg',
                'head_land_dec_avg_diff'
            ],
            'body_': [
                'body_acc_*dec_avg',
                'body_def_*dec_avg'
            ],
            'leg_': [
                'leg_land_rd1_*dec_avg',
                'leg_*_per_min_*dec_avg'
            ],
            'distance_': [
                'distance_land*_dec_adjperf_*dec_avg',
                'distance_acc_dec_adjperf_dec_avg_diff',
                'distance_acc_dec_adjperf_opp_dec_avg_diff',
                'distance_def_dec_adjperf_dec_avg_diff',
                'distance_def_dec_adjperf_opp_dec_avg_diff',
                'distance_att_rd1_total_dec_adjperf_dec_avg_diff',
                'distance_att_rd1_total_dec_adjperf_opp_dec_avg_diff',
                'distance_att_rd1_ratio_dec_adjperf_dec_avg_diff',
                'distance_att_rd1_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'clinch_': [
                'clinch_att_rd1_*dec_avg',
                'clinch_*acc*dec_avg'
            ],
            'ground_': [
                'ground_att_*dec_adjperf*dec_avg',
                'ground_land_dec_adjperf_opp_dec_avg_diff',
                'ground_land_dec_adjperf_dec_avg_diff',
                'ground_rd1_*dec_avg'
            ],
            'ctrl_': [
                'ctrl_rd1_*dec_avg',
                'ctrl_*per_min*dec_avg'
            ],
            'td_': [
                'td_*dec_avg'
            ],
            'rev_': [
                'rev_dec_adjperf_opp_dec_avg_diff',
                'rev_dec_adjperf_dec_avg_diff',
                'rev_total_dec_adjperf_opp_dec_avg_diff',
                'rev_ratio_dec_adjperf_dec_avg_diff',
                'rev_total_dec_adjperf_dec_avg_diff',
                'rev_dec_avg_diff',
                'rev_total_dec_avg_diff',
                'rev_per_min_opp_dec_avg_diff',
                'rev_rd1_dec_adjperf_dec_avg_diff'
            ],
            'sub_': [
                'sub_att_*dec_avg',
                'sub_def_*dec_avg'
            ],
            'ko_': [
                'ko_rd1_dec_adjperf_opp_dec_avg_diff',
                'ko_total_dec_adjperf_opp_dec_avg_diff',
                'ko_rd1_total_ratio_dec_adjperf_opp_dec_avg_diff',
                'ko_rd1_total_dec_adjperf_dec_avg_diff',
                'ko_rd1_per_min_dec_adjperf_dec_avg_diff',
                'ko_rd1_dec_adjperf_dec_avg_diff',
                'ko_total_dec_adjperf_dec_avg_diff',
                'ko_per_min_dec_adjperf_dec_avg_diff',
                'ko_opp_dec_avg_diff',
                'ko_total_ratio_dec_avg_diff',
                'ko_rd1_per_min_opp_dec_avg_diff'
            ],
            'decision_': [
                'decision_total_dec_adjperf_opp_dec_avg_diff',
                'decision_total_dec_adjperf_dec_avg_diff',
                'decision_per_min_dec_adjperf_dec_avg_diff',
                'decision_opp_dec_avg_diff',
                'decision_total_ratio_dec_avg_diff'
            ],
            'kd_': [
                'kd_rd1_total_dec_avg_diff',
                'kd_opp_dec_avg_diff',
                'kd_rd1_total_dec_adjperf_dec_avg_diff',
                'kd_rd1_total_ratio_dec_adjperf_dec_avg_diff',
                'kd_rd1_ratio_dec_adjperf_dec_avg_diff',
                'kd_total_dec_adjperf_dec_avg_diff',
                'kd_dec_adjperf_opp_dec_avg_diff',
                'kd_per_min_dec_adjperf_opp_dec_avg_diff',
                'kd_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'time_sec_': [
                'time_sec_dec_avg_diff',
                'time_sec_opp_dec_avg_diff',
                'time_sec_rd1_dec_adjperf_dec_avg_diff',
                'time_sec_dec_adjperf_dec_avg_diff',
                'time_sec_total_dec_adjperf_dec_avg_diff',
                'time_sec_rd1_dec_adjperf_opp_dec_avg_diff',
                'time_sec_dec_adjperf_opp_dec_avg_diff'
            ],
            'style_': [
                'style_*'
            ]
        }

# Basic static features
BASE_STATIC_FEATS = ['age', 'days_since_last_fight', 'reach', 'height', 'ufcage', 'odds', 'weightclass_encoded']

BASE_DYNAMIC_FEATS = ['head', 'body', 'leg', 'sig_str', 'strikes','distance', 'clinch', 'ground', 'td', 'ctrl', 'sub', 'rev', 'ko', 'kd', 'decision', 'win', 'time_sec']
CUSTOM_FEATS = [
    'weightclass_encoded',
    # PerCalculator features
    'ko_per_sig_str_land',
    'sig_str_per_str_att',
    'distance_per_sig_str_land',
    'clinch_per_sig_str_land',
    'ground_per_sig_str_land',
    'head_per_sig_str_land',
    'body_leg_per_sig_str_land',
    'td_per_sig_str_att',
    'td_land_per_ctrl',
    'ground_land_per_ctrl',
    'ground_land_per_td_land',
    'sub_att_per_ctrl',
    'sub_per_all_ctrl',
    'rev_per_ctrlopp',
    'ko_sub_rd1_per_win',
    'ko_sub_per_win',
]

LAYERED_TEST_FEATS = [
    # Basic stats (already have _diff)
    "age_diff",
    "age_ratio_diff",
    "age_dec_avg_diff",
    "age_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "days_since_last_fight_ratio_dec_avg_diff",
    "reach_diff",
    "reach_ratio_diff",
    "reach_dec_avg_diff",
    "reach_ratio_dec_avg_diff",
    'ufcage_diff',
    'ufcage_ratio_diff',
    'ufcage_dec_avg_diff',
    'ufcage_ratio_dec_avg_diff',
    
    # Group 1: head_* (unchanged)
    "head_acc_opp_dec_avg_diff",
    "head_acc_dec_adjperf_dec_avg_diff",
    "head_def_opp_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    "head_land_opp_dec_avg_diff",
    "head_land_total_opp_dec_avg_diff",
    "head_land_dec_adjperf_dec_avg_diff",
    "head_land_total_dec_adjperf_dec_avg_diff",
    "head_land_rd1_opp_dec_avg_diff",
    "head_land_rd1_total_opp_dec_avg_diff",
    "head_land_rd1_dec_adjperf_dec_avg_diff",
    "head_land_rd1_total_dec_adjperf_dec_avg_diff",
    "head_land_ratio_opp_dec_avg_diff",
    "head_land_total_ratio_opp_dec_avg_diff",
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_land_total_ratio_dec_adjperf_dec_avg_diff",
    "head_land_rd1_ratio_opp_dec_avg_diff",
    "head_land_rd1_total_ratio_opp_dec_avg_diff",
    "head_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "head_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "head_land_per_min_opp_dec_avg_diff",
    "head_land_per_min_dec_adjperf_dec_avg_diff",
    "head_land_rd1_per_min_opp_dec_avg_diff",
    "head_land_rd1_per_min_dec_adjperf_dec_avg_diff",

    # Group 4: body_*
    "body_acc_opp_dec_avg_diff",
    "body_acc_dec_adjperf_dec_avg_diff",
    "body_def_opp_dec_avg_diff",
    "body_def_dec_adjperf_dec_avg_diff",
    "body_land_opp_dec_avg_diff",
    "body_land_total_opp_dec_avg_diff",
    "body_land_dec_adjperf_dec_avg_diff",
    "body_land_total_dec_adjperf_dec_avg_diff",
    "body_land_rd1_opp_dec_avg_diff",
    "body_land_rd1_total_opp_dec_avg_diff",
    "body_land_rd1_dec_adjperf_dec_avg_diff",
    "body_land_rd1_total_dec_adjperf_dec_avg_diff",
    "body_land_ratio_opp_dec_avg_diff",
    "body_land_total_ratio_opp_dec_avg_diff",
    "body_land_ratio_dec_adjperf_dec_avg_diff",
    "body_land_total_ratio_dec_adjperf_dec_avg_diff",
    "body_land_rd1_ratio_opp_dec_avg_diff",
    "body_land_rd1_total_ratio_opp_dec_avg_diff",
    "body_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "body_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "body_land_per_min_opp_dec_avg_diff",
    "body_land_per_min_dec_adjperf_dec_avg_diff",
    "body_land_rd1_per_min_opp_dec_avg_diff",
    "body_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 2: leg_*
    "leg_acc_opp_dec_avg_diff",
    "leg_acc_dec_adjperf_dec_avg_diff",
    "leg_def_opp_dec_avg_diff",
    "leg_def_dec_adjperf_dec_avg_diff",
    "leg_land_opp_dec_avg_diff",
    "leg_land_total_opp_dec_avg_diff",
    "leg_land_dec_adjperf_dec_avg_diff",
    "leg_land_total_dec_adjperf_dec_avg_diff",
    "leg_land_rd1_opp_dec_avg_diff",
    "leg_land_rd1_total_opp_dec_avg_diff",
    "leg_land_rd1_dec_adjperf_dec_avg_diff",
    "leg_land_rd1_total_dec_adjperf_dec_avg_diff",
    "leg_land_ratio_opp_dec_avg_diff",
    "leg_land_total_ratio_opp_dec_avg_diff",
    "leg_land_ratio_dec_adjperf_dec_avg_diff",
    "leg_land_total_ratio_dec_adjperf_dec_avg_diff",
    "leg_land_rd1_ratio_opp_dec_avg_diff",
    "leg_land_rd1_total_ratio_opp_dec_avg_diff",
    "leg_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "leg_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_opp_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_land_rd1_per_min_opp_dec_avg_diff",
    "leg_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 3: sig_str_*
    "sig_str_acc_opp_dec_avg_diff",
    "sig_str_acc_dec_adjperf_dec_avg_diff",
    "sig_str_def_opp_dec_avg_diff",
    "sig_str_def_dec_adjperf_dec_avg_diff",
    "sig_str_land_opp_dec_avg_diff",
    "sig_str_land_total_opp_dec_avg_diff",
    "sig_str_land_dec_adjperf_dec_avg_diff",
    "sig_str_land_total_dec_adjperf_dec_avg_diff",
    "sig_str_land_rd1_opp_dec_avg_diff",
    "sig_str_land_rd1_total_opp_dec_avg_diff",
    "sig_str_land_rd1_dec_adjperf_dec_avg_diff",
    "sig_str_land_rd1_total_dec_adjperf_dec_avg_diff",
    "sig_str_land_ratio_opp_dec_avg_diff",
    "sig_str_land_total_ratio_opp_dec_avg_diff",
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_total_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_rd1_ratio_opp_dec_avg_diff",
    "sig_str_land_rd1_total_ratio_opp_dec_avg_diff",
    "sig_str_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_land_per_min_opp_dec_avg_diff",
    "sig_str_land_per_min_dec_adjperf_dec_avg_diff",
    "sig_str_land_rd1_per_min_opp_dec_avg_diff",
    "sig_str_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 4: strikes
    "strikes_acc_opp_dec_avg_diff",
    "strikes_acc_dec_adjperf_dec_avg_diff",
    "strikes_def_opp_dec_avg_diff",
    "strikes_def_dec_adjperf_dec_avg_diff",
    "strikes_land_opp_dec_avg_diff",
    "strikes_land_total_opp_dec_avg_diff",
    "strikes_land_dec_adjperf_dec_avg_diff",
    "strikes_land_total_dec_adjperf_dec_avg_diff",
    "strikes_land_rd1_opp_dec_avg_diff",
    "strikes_land_rd1_total_opp_dec_avg_diff",
    "strikes_land_rd1_dec_adjperf_dec_avg_diff",
    "strikes_land_rd1_total_dec_adjperf_dec_avg_diff",
    "strikes_land_ratio_opp_dec_avg_diff",
    "strikes_land_total_ratio_opp_dec_avg_diff",
    "strikes_land_ratio_dec_adjperf_dec_avg_diff",
    "strikes_land_total_ratio_dec_adjperf_dec_avg_diff",
    "strikes_land_rd1_ratio_opp_dec_avg_diff",
    "strikes_land_rd1_total_ratio_opp_dec_avg_diff",
    "strikes_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "strikes_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "strikes_land_per_min_opp_dec_avg_diff",
    "strikes_land_per_min_dec_adjperf_dec_avg_diff",
    "strikes_land_rd1_per_min_opp_dec_avg_diff",
    "strikes_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 6: distance_*
    "distance_acc_opp_dec_avg_diff",
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_def_opp_dec_avg_diff",
    "distance_def_dec_adjperf_dec_avg_diff",
    "distance_land_opp_dec_avg_diff",
    "distance_land_total_opp_dec_avg_diff",
    "distance_land_dec_adjperf_dec_avg_diff",
    "distance_land_total_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_opp_dec_avg_diff",
    "distance_land_rd1_total_opp_dec_avg_diff",
    "distance_land_rd1_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_total_dec_adjperf_dec_avg_diff",
    "distance_land_ratio_opp_dec_avg_diff",
    "distance_land_total_ratio_opp_dec_avg_diff",
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "distance_land_total_ratio_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_ratio_opp_dec_avg_diff",
    "distance_land_rd1_total_ratio_opp_dec_avg_diff",
    "distance_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "distance_land_per_min_opp_dec_avg_diff",
    "distance_land_per_min_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_per_min_opp_dec_avg_diff",
    "distance_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 5: clinch_*
    "clinch_acc_opp_dec_avg_diff",
    "clinch_acc_dec_adjperf_dec_avg_diff",
    "clinch_def_opp_dec_avg_diff",
    "clinch_def_dec_adjperf_dec_avg_diff",
    "clinch_land_opp_dec_avg_diff",
    "clinch_land_total_opp_dec_avg_diff",
    "clinch_land_dec_adjperf_dec_avg_diff",
    "clinch_land_total_dec_adjperf_dec_avg_diff",
    "clinch_land_rd1_opp_dec_avg_diff",
    "clinch_land_rd1_total_opp_dec_avg_diff",
    "clinch_land_rd1_dec_adjperf_dec_avg_diff",
    "clinch_land_rd1_total_dec_adjperf_dec_avg_diff",
    "clinch_land_ratio_opp_dec_avg_diff",
    "clinch_land_total_ratio_opp_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_total_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_rd1_ratio_opp_dec_avg_diff",
    "clinch_land_rd1_total_ratio_opp_dec_avg_diff",
    "clinch_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "clinch_land_per_min_opp_dec_avg_diff",
    "clinch_land_per_min_dec_adjperf_dec_avg_diff",
    "clinch_land_rd1_per_min_opp_dec_avg_diff",
    "clinch_land_rd1_per_min_dec_adjperf_dec_avg_diff",

    # Group 13: ground_*
    "ground_acc_opp_dec_avg_diff",
    "ground_acc_dec_adjperf_dec_avg_diff",
    "ground_def_opp_dec_avg_diff",
    "ground_def_dec_adjperf_dec_avg_diff",
    "ground_land_opp_dec_avg_diff",
    "ground_land_total_opp_dec_avg_diff",
    "ground_land_dec_adjperf_dec_avg_diff",
    "ground_land_total_dec_adjperf_dec_avg_diff",
    "ground_land_rd1_opp_dec_avg_diff",
    "ground_land_rd1_total_opp_dec_avg_diff",
    "ground_land_rd1_dec_adjperf_dec_avg_diff",
    "ground_land_rd1_total_dec_adjperf_dec_avg_diff",
    "ground_land_ratio_opp_dec_avg_diff",
    "ground_land_total_ratio_opp_dec_avg_diff",
    "ground_land_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_total_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_rd1_ratio_opp_dec_avg_diff",
    "ground_land_rd1_total_ratio_opp_dec_avg_diff",
    "ground_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_per_min_opp_dec_avg_diff",
    "ground_land_per_min_dec_adjperf_dec_avg_diff",
    "ground_land_rd1_per_min_opp_dec_avg_diff",
    "ground_land_rd1_per_min_dec_adjperf_dec_avg_diff",
    
    # Group 12: td_*
    "td_acc_opp_dec_avg_diff",
    "td_acc_dec_adjperf_dec_avg_diff",
    "td_def_opp_dec_avg_diff",
    "td_def_dec_adjperf_dec_avg_diff",
    "td_land_opp_dec_avg_diff",
    "td_land_total_opp_dec_avg_diff",
    "td_land_dec_adjperf_dec_avg_diff",
    "td_land_total_dec_adjperf_dec_avg_diff",
    "td_land_rd1_opp_dec_avg_diff",
    "td_land_rd1_total_opp_dec_avg_diff",
    "td_land_rd1_dec_adjperf_dec_avg_diff",
    "td_land_rd1_total_dec_adjperf_dec_avg_diff",
    "td_land_ratio_opp_dec_avg_diff",
    "td_land_total_ratio_opp_dec_avg_diff",
    "td_land_ratio_dec_adjperf_dec_avg_diff",
    "td_land_total_ratio_dec_adjperf_dec_avg_diff",
    "td_land_rd1_ratio_opp_dec_avg_diff",
    "td_land_rd1_total_ratio_opp_dec_avg_diff",
    "td_land_rd1_ratio_dec_adjperf_dec_avg_diff",
    "td_land_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "td_att_rd1_opp_dec_avg_diff",
    "td_att_rd1_total_opp_dec_avg_diff",
    "td_att_rd1_dec_adjperf_dec_avg_diff",
    "td_att_rd1_total_dec_adjperf_dec_avg_diff",
    "td_att_ratio_opp_dec_avg_diff",
    "td_att_total_ratio_opp_dec_avg_diff",
    "td_att_ratio_dec_adjperf_dec_avg_diff",
    "td_att_total_ratio_dec_adjperf_dec_avg_diff",
    "td_att_rd1_ratio_opp_dec_avg_diff",
    "td_att_rd1_total_ratio_opp_dec_avg_diff",
    "td_att_rd1_ratio_dec_adjperf_dec_avg_diff",
    "td_att_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "td_att_per_min_opp_dec_avg_diff",
    "td_att_per_min_dec_adjperf_dec_avg_diff",
    "td_att_rd1_per_min_opp_dec_avg_diff",
    "td_att_rd1_per_min_dec_adjperf_dec_avg_diff",
    

    # Group 11: ctrl and related
    "ctrl_opp_dec_avg_diff",
    "ctrl_total_opp_dec_avg_diff",
    "ctrl_dec_adjperf_dec_avg_diff",
    "ctrl_total_dec_adjperf_dec_avg_diff",
    "ctrl_rd1_opp_dec_avg_diff",
    "ctrl_rd1_total_opp_dec_avg_diff",
    "ctrl_rd1_dec_adjperf_dec_avg_diff",
    "ctrl_rd1_total_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_opp_dec_avg_diff",
    "ctrl_total_ratio_opp_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_total_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_rd1_ratio_opp_dec_avg_diff",
    "ctrl_rd1_total_ratio_opp_dec_avg_diff",
    "ctrl_rd1_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "ctrl_per_min_dec_adjperf_dec_avg_diff",
    "ctrl_rd1_per_min_opp_dec_avg_diff",
    "ctrl_rd1_per_min_dec_adjperf_dec_avg_diff",

    # Group 12: rev
    "rev_opp_dec_avg_diff",
    "rev_total_opp_dec_avg_diff",
    "rev_dec_adjperf_dec_avg_diff",
    "rev_total_dec_adjperf_dec_avg_diff",
    "rev_rd1_opp_dec_avg_diff",
    "rev_rd1_total_opp_dec_avg_diff",
    "rev_rd1_dec_adjperf_dec_avg_diff",
    "rev_rd1_total_dec_adjperf_dec_avg_diff",
    "rev_ratio_opp_dec_avg_diff",
    "rev_total_ratio_opp_dec_avg_diff",
    "rev_ratio_dec_adjperf_dec_avg_diff",
    "rev_total_ratio_dec_adjperf_dec_avg_diff",
    "rev_rd1_ratio_opp_dec_avg_diff",
    "rev_rd1_total_ratio_opp_dec_avg_diff",
    "rev_rd1_ratio_dec_adjperf_dec_avg_diff",
    "rev_rd1_total_ratio_dec_adjperf_dec_avg_diff",
    "rev_per_min_opp_dec_avg_diff",
    "rev_per_min_dec_adjperf_dec_avg_diff",
    "rev_rd1_per_min_opp_dec_avg_diff",
    "rev_rd1_per_min_dec_adjperf_dec_avg_diff",

    # Group 7: win
    "win_opp_dec_avg_diff",
    "win_total_opp_dec_avg_diff",
    "win_dec_adjperf_dec_avg_diff",
    "win_total_dec_adjperf_dec_avg_diff",
    "win_ratio_opp_dec_avg_diff",
    "win_total_ratio_opp_dec_avg_diff",
    "win_ratio_dec_adjperf_dec_avg_diff",
    "win_total_ratio_dec_adjperf_dec_avg_diff",
    "win_per_min_opp_dec_avg_diff",
    
    # Group 8: sub_
    "sub_att_opp_dec_avg_diff",
    "sub_att_total_opp_dec_avg_diff",
    "sub_att_per_min_opp_dec_avg_diff",
    "sub_att_dec_adjperf_dec_avg_diff",
    "sub_att_total_dec_adjperf_dec_avg_diff",
    "sub_land_opp_dec_avg_diff",
    "sub_land_total_opp_dec_avg_diff",
    "sub_land_per_min_opp_dec_avg_diff",
    "sub_acc_opp_dec_avg_diff",
    "sub_acc_dec_adjperf_dec_avg_diff",
    "sub_def_opp_dec_avg_diff",
    "sub_def_dec_adjperf_dec_avg_diff",
    
    # Group 9: kd
    "kd_opp_dec_avg_diff",
    "kd_total_opp_dec_avg_diff",
    "kd_dec_adjperf_dec_avg_diff",
    "kd_total_dec_adjperf_dec_avg_diff",
    "kd_per_min_opp_dec_avg_diff",
    "kd_ratio_opp_dec_avg_diff",
    "kd_total_ratio_opp_dec_avg_diff",
    "kd_ratio_dec_adjperf_dec_avg_diff",
    "kd_total_ratio_dec_adjperf_dec_avg_diff",
    
    # Group 10: ko
    "ko_opp_dec_avg_diff",
    "ko_total_opp_dec_avg_diff",
    "ko_dec_adjperf_dec_avg_diff",
    "ko_total_dec_adjperf_dec_avg_diff",
    "ko_per_min_opp_dec_avg_diff",
    "ko_ratio_opp_dec_avg_diff",
    "ko_total_ratio_opp_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    "ko_total_ratio_dec_adjperf_dec_avg_diff",

    # Group 11: decision
    "decision_opp_dec_avg_diff",
    "decision_total_opp_dec_avg_diff",
    "decision_dec_adjperf_dec_avg_diff",
    "decision_total_dec_adjperf_dec_avg_diff",
    "decision_per_min_opp_dec_avg_diff",
    "decision_ratio_opp_dec_avg_diff",
    "decision_total_ratio_opp_dec_avg_diff",
    "decision_ratio_dec_adjperf_dec_avg_diff",
    "decision_total_ratio_dec_adjperf_dec_avg_diff",

    # Group 14: time_sec
    "time_sec_opp_dec_avg_diff",
    "time_sec_total_opp_dec_avg_diff",
    "time_sec_dec_adjperf_dec_avg_diff",
    "time_sec_total_dec_adjperf_dec_avg_diff",
    "time_sec_rd1_opp_dec_avg_diff",
    "time_sec_rd1_total_opp_dec_avg_diff",
    "time_sec_rd1_dec_adjperf_dec_avg_diff",
    "time_sec_rd1_total_dec_adjperf_dec_avg_diff",
]

BASIC_TEST_FEATS = [
    # Group 1: head_*
    "head_acc_dec_avg_diff",
    "head_def_dec_avg_diff",
    "head_land_dec_avg_diff",
    "head_land_total_dec_avg_diff",
    "head_land_rd1_dec_avg_diff",
    "head_land_rd1_total_dec_avg_diff",
    "head_land_ratio_dec_avg_diff",
    "head_land_total_ratio_dec_avg_diff",
    "head_land_rd1_ratio_dec_avg_diff",
    "head_land_rd1_total_ratio_dec_avg_diff",
    "head_land_per_min_dec_avg_diff",
    "head_land_rd1_per_min_dec_avg_diff",
    
    # Group 4: body_*
    "body_acc_dec_avg_diff",
    "body_def_dec_avg_diff",
    "body_land_dec_avg_diff",
    "body_land_total_dec_avg_diff",
    "body_land_rd1_dec_avg_diff",
    "body_land_rd1_total_dec_avg_diff",
    "body_land_ratio_dec_avg_diff",
    "body_land_total_ratio_dec_avg_diff",
    "body_land_rd1_ratio_dec_avg_diff",
    "body_land_rd1_total_ratio_dec_avg_diff",
    "body_land_per_min_dec_avg_diff",
    "body_land_rd1_per_min_dec_avg_diff",
    
    # Group 2: leg_*
    "leg_acc_dec_avg_diff",
    "leg_def_dec_avg_diff",
    "leg_land_dec_avg_diff",
    "leg_land_total_dec_avg_diff",
    "leg_land_rd1_dec_avg_diff",
    "leg_land_rd1_total_dec_avg_diff",
    "leg_land_ratio_dec_avg_diff",
    "leg_land_total_ratio_dec_avg_diff",
    "leg_land_rd1_ratio_dec_avg_diff",
    "leg_land_rd1_total_ratio_dec_avg_diff",
    "leg_land_per_min_dec_avg_diff",
    "leg_land_rd1_per_min_dec_avg_diff",
    
    # Group 3: sig_str_*
    "sig_str_acc_dec_avg_diff",
    "sig_str_def_dec_avg_diff",
    "sig_str_land_dec_avg_diff",
    "sig_str_land_total_dec_avg_diff",
    "sig_str_land_rd1_dec_avg_diff",
    "sig_str_land_rd1_total_dec_avg_diff",
    "sig_str_land_ratio_dec_avg_diff",
    "sig_str_land_total_ratio_dec_avg_diff",
    "sig_str_land_rd1_ratio_dec_avg_diff",
    "sig_str_land_rd1_total_ratio_dec_avg_diff",
    "sig_str_land_per_min_dec_avg_diff",
    "sig_str_land_rd1_per_min_dec_avg_diff",

    # Group 4: strikes
    "strikes_acc_dec_avg_diff",
    "strikes_def_dec_avg_diff",
    "strikes_land_dec_avg_diff",
    "strikes_land_total_dec_avg_diff",
    "strikes_land_rd1_dec_avg_diff",
    "strikes_land_rd1_total_dec_avg_diff",
    "strikes_land_ratio_dec_avg_diff",
    "strikes_land_total_ratio_dec_avg_diff",
    "strikes_land_rd1_ratio_dec_avg_diff",
    "strikes_land_rd1_total_ratio_dec_avg_diff",
    "strikes_land_per_min_dec_avg_diff",
    "strikes_land_rd1_per_min_dec_avg_diff",
    
    # Group 6: distance_*
    "distance_acc_dec_avg_diff",
    "distance_def_dec_avg_diff",
    "distance_land_dec_avg_diff",
    "distance_land_total_dec_avg_diff",
    "distance_land_rd1_dec_avg_diff",
    "distance_land_rd1_total_dec_avg_diff",
    "distance_land_ratio_dec_avg_diff",
    "distance_land_total_ratio_dec_avg_diff",
    "distance_land_rd1_ratio_dec_avg_diff",
    "distance_land_rd1_total_ratio_dec_avg_diff",
    "distance_land_per_min_dec_avg_diff",
    "distance_land_rd1_per_min_dec_avg_diff",
    
    # Group 5: clinch_*
    "clinch_acc_dec_avg_diff",
    "clinch_def_dec_avg_diff",
    "clinch_land_dec_avg_diff",
    "clinch_land_total_dec_avg_diff",
    "clinch_land_rd1_dec_avg_diff",
    "clinch_land_rd1_total_dec_avg_diff",
    "clinch_land_ratio_dec_avg_diff",
    "clinch_land_total_ratio_dec_avg_diff",
    "clinch_land_rd1_ratio_dec_avg_diff",
    "clinch_land_rd1_total_ratio_dec_avg_diff",
    "clinch_land_per_min_dec_avg_diff",
    "clinch_land_rd1_per_min_dec_avg_diff",

    # Group 13: ground_*
    "ground_acc_dec_avg_diff",
    "ground_def_dec_avg_diff",
    "ground_land_dec_avg_diff",
    "ground_land_total_dec_avg_diff",
    "ground_land_rd1_dec_avg_diff",
    "ground_land_rd1_total_dec_avg_diff",
    "ground_land_ratio_dec_avg_diff",
    "ground_land_total_ratio_dec_avg_diff",
    "ground_land_rd1_ratio_dec_avg_diff",
    "ground_land_rd1_total_ratio_dec_avg_diff",
    "ground_land_per_min_dec_avg_diff",
    "ground_land_rd1_per_min_dec_avg_diff",
    
    # Group 12: td_*
    "td_acc_dec_avg_diff",
    "td_def_dec_avg_diff",
    "td_land_dec_avg_diff",
    "td_land_total_dec_avg_diff",
    "td_land_rd1_dec_avg_diff",
    "td_land_rd1_total_dec_avg_diff",
    "td_land_ratio_dec_avg_diff",
    "td_land_total_ratio_dec_avg_diff",
    "td_land_rd1_ratio_dec_avg_diff",
    "td_land_rd1_total_ratio_dec_avg_diff",
    "td_land_per_min_dec_avg_diff",
    "td_land_rd1_per_min_dec_avg_diff",
    "td_att_rd1_dec_avg_diff",
    "td_att_rd1_total_dec_avg_diff",
    "td_att_ratio_dec_avg_diff",
    "td_att_total_ratio_dec_avg_diff",
    "td_att_rd1_ratio_dec_avg_diff",
    "td_att_rd1_total_ratio_dec_avg_diff",
    "td_att_per_min_dec_avg_diff",
    "td_att_rd1_per_min_dec_avg_diff",
    
    # Group 11: ctrl and related
    "ctrl_dec_avg_diff",
    "ctrl_total_dec_avg_diff",
    "ctrl_rd1_dec_avg_diff",
    "ctrl_rd1_total_dec_avg_diff",
    "ctrl_ratio_dec_avg_diff",
    "ctrl_total_ratio_dec_avg_diff",
    "ctrl_rd1_ratio_dec_avg_diff",
    "ctrl_rd1_total_ratio_dec_avg_diff",
    "ctrl_per_min_dec_avg_diff",
    "ctrl_rd1_per_min_dec_avg_diff",

    # Group 12: rev
    "rev_dec_avg_diff",
    "rev_total_dec_avg_diff",
    "rev_rd1_dec_avg_diff",
    "rev_rd1_total_dec_avg_diff",
    "rev_ratio_dec_avg_diff",
    "rev_total_ratio_dec_avg_diff",
    "rev_rd1_ratio_dec_avg_diff",
    "rev_rd1_total_ratio_dec_avg_diff",
    "rev_per_min_dec_avg_diff",
    "rev_rd1_per_min_dec_avg_diff",

    # Group 7: win
    "win_dec_avg_diff",
    "win_total_dec_avg_diff",
    "win_ratio_dec_avg_diff",
    "win_total_ratio_dec_avg_diff",
    
    # Group 8: sub_
    "sub_att_dec_avg_diff",
    "sub_att_total_dec_avg_diff",
    "sub_att_per_min_dec_avg_diff",
    "sub_land_dec_avg_diff",
    "sub_land_total_dec_avg_diff",
    "sub_land_per_min_dec_avg_diff",

    # Group 9: kd
    "kd_dec_avg_diff",
    "kd_total_dec_avg_diff",
    "kd_per_min_dec_avg_diff",
    "kd_ratio_dec_avg_diff",
    "kd_total_ratio_dec_avg_diff",

    # Group 10: ko
    "ko_dec_avg_diff",
    "ko_total_dec_avg_diff",
    "ko_per_min_dec_avg_diff",
    "ko_ratio_dec_avg_diff",
    "ko_total_ratio_dec_avg_diff",

    # Group 11: decision
    "decision_dec_avg_diff",
    "decision_total_dec_avg_diff",
    "decision_per_min_dec_avg_diff",
    "decision_ratio_dec_avg_diff",
    "decision_total_ratio_dec_avg_diff",

    # Group 14: time_sec, no ratio
    "time_sec_dec_avg_diff",
    "time_sec_total_dec_avg_diff",
    "time_sec_rd1_dec_avg_diff",
    "time_sec_rd1_total_dec_avg_diff",
]

TEST_FEATS = BASIC_TEST_FEATS + LAYERED_TEST_FEATS
TEST_FEATS_NO_DIFF = [c.replace('_diff', '') for c in TEST_FEATS]

FEAT_MAD2_AND_STYLES = [ # v61 819train 732test exp default hp feats_mad2
    # Basic stats
    'age_dec_avg_diff',
    'age_ratio_diff',
    'reach_ratio_dec_avg_diff',
    'ufcage_dec_avg_diff',
    'days_since_last_fight_dec_avg_diff',

    # Significant strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    #'sig_str_land_ratio_dec_avg_diff',

    # Strikes
    'strikes_land_rd1_dec_adjperf_dec_avg_diff',

    # Head strikes
	'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_acc_adjperf_dec_avg_diff',
    'head_def_adjperf_dec_avg_diff',
    'head_land_dec_avg_diff',

    # Body strikes
    #'body_acc_dec_adjperf_dec_avg_diff',
    'body_def_dec_adjperf_dec_avg_diff',

    # Distance strikes
    #'distance_land_ratio_dec_adjperf_dec_avg_diff',
	
    # Clinch strikes
    'clinch_att_rd1_per_min_adjperf_opp_dec_avg_diff',

    # Leg strikes
    'leg_att_rd1_per_min_opp_dec_avg_diff', # Probably remove

    # Ground strikes
    'ground_att_dec_adjperf_dec_avg_diff',

    # Takedowns
    'td_acc_dec_avg_diff',
    'td_def_dec_avg_diff',
    'td_att_opp_dec_avg_diff',

    # Control time
    'ctrl_rd1_per_min_opp_dec_avg_diff',
    'ctrl_rd1_dec_avg_diff',

    # Reversals
    'rev_dec_adjperf_opp_dec_avg_diff',
	#'rev_rd1_ratio_opp_dec_avg_diff', # Prob remove

    # Submissions
    'sub_att_dec_avg_diff',
    'sub_att_per_min_opp_dec_avg_diff',

    # Wins
    'win_opp_dec_avg_diff',
    'win_dec_adjperf_opp_dec_avg_diff',
    
    # Knockdowns
    #'kd_opp_dec_avg_diff',
    
    # Knockouts
	'ko_opp_dec_avg_diff',

    # Styles
    'style_power_vs_volume_diff',
    'style_damage_efficiency_diff',

    # Time_sec
    'time_sec_dec_adjperf_opp_dec_avg_diff',
]

FEAT_MAD2_AND_STYLES_TEST = [ 
    # Basic stats
    'age_dec_avg_diff',
    'age_diff',
    'age_ratio_diff',
    'reach_ratio_dec_avg_diff',
    'reach_diff',
    'ufcage_dec_avg_diff',
    'days_since_last_fight_dec_avg_diff',

    # Significant strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_land_ratio_dec_adjperf_opp_dec_avg_diff',
    'sig_str_land_ratio_dec_avg_diff',
    'sig_str_land_ratio_opp_dec_avg_diff',
    'sig_str_land_rd1_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_land_rd1_ratio_dec_adjperf_opp_dec_avg_diff',
    #'sig_str_land_ratio_dec_avg_diff',

    # Strikes
    'strikes_att_rd1_ratio_dec_adjperf_dec_avg_diff',
    'strikes_att_rd1_ratio_dec_adjperf_opp_dec_avg_diff',
    'strikes_att_rd1_per_min_dec_adjperf_dec_avg_diff',
    'strikes_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',

    # Head strikes
	'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_land_ratio_dec_adjperf_opp_dec_avg_diff',
    'head_land_ratio_dec_avg_diff',
    'head_land_ratio_opp_dec_avg_diff',
    'head_acc_adjperf_dec_avg_diff',
    'head_acc_adjperf_opp_dec_avg_diff',
    'head_def_adjperf_dec_avg_diff',
    'head_def_adjperf_opp_dec_avg_diff',
    'head_land_dec_avg_diff',
    'head_land_opp_dec_avg_diff',

    # Body strikes
    #'body_acc_dec_adjperf_dec_avg_diff',
    'body_def_dec_adjperf_dec_avg_diff',
    'body_def_dec_adjperf_opp_dec_avg_diff',
    'body_def_dec_avg_diff',
    'body_def_opp_dec_avg_diff',
    'body_acc_dec_adjperf_dec_avg_diff',
    'body_acc_dec_adjperf_opp_dec_avg_diff',
    'body_acc_dec_avg_diff',
    'body_acc_opp_dec_avg_diff',

    # Distance strikes
    'distance_land_ratio_dec_adjperf_dec_avg_diff',
    'distance_land_ratio_dec_adjperf_opp_dec_avg_diff',
    'distance_land_rd1_per_min_dec_avg_diff',
    'distance_land_rd1_per_min_opp_dec_avg_diff',
	
    # Clinch strikes
    'clinch_att_rd1_per_min_adjperf_opp_dec_avg_diff',
    'clinch_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',

    # Leg strikes   
    'leg_att_rd1_per_min_dec_adjperf_dec_avg_diff',
    'leg_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',
    'leg_acc_dec_avg_diff',
    'leg_acc_opp_dec_avg_diff',
    'leg_acc_dec_adjperf_dec_avg_diff',
    'leg_acc_dec_adjperf_opp_dec_avg_diff',

    # Ground strikes
    'ground_att_per_min_opp_dec_avg_diff',
    'ground_att_per_min_dec_avg_diff',
    'ground_att_per_min_dec_adjperf_opp_dec_avg_diff',
    'ground_att_per_min_dec_adjperf_dec_avg_diff',
    'ground_att_ratio_dec_adjperf_dec_avg_diff',
    'ground_att_ratio_dec_adjperf_opp_dec_avg_diff',

    # Takedowns
    'td_acc_dec_avg_diff',
    'td_def_dec_avg_diff',
    'td_att_opp_dec_avg_diff',
    'td_att_dec_adjperf_dec_avg_diff',
    'td_land_rd1_ratio_dec_adjperf_opp_dec_avg_diff',

    # Control time
    'ctrl_per_min_dec_adjperf_opp_dec_avg_diff',
    'ctrl_per_min_dec_adjperf_dec_avg_diff',
    'ctrl_per_min_opp_dec_avg_diff',
    'ctrl_per_min_dec_avg_diff',
    'ctrl_rd1_per_min_opp_dec_avg_diff',
    'ctrl_rd1_per_min_dec_avg_diff',
    'ctrl_rd1_per_min_dec_adjperf_opp_dec_avg_diff',
    'ctrl_rd1_per_min_dec_adjperf_dec_avg_diff',
    'ctrl_rd1_dec_adjperf_dec_avg_diff',

    # Reversals
    'rev_dec_adjperf_opp_dec_avg_diff',
	#'rev_rd1_ratio_opp_dec_avg_diff', # Prob remove

    # Submissions
    'sub_att_dec_avg_diff',
    'sub_att_per_min_opp_dec_avg_diff',

    # Wins
    'win_opp_dec_avg_diff',
    'win_dec_adjperf_opp_dec_avg_diff',
    'win_dec_avg_diff',
    'win_per_min_opp_dec_avg_diff',

    
    # Knockdowns
    #'kd_opp_dec_avg_diff',
    
    # Knockouts
	'ko_opp_dec_avg_diff',

    # Styles
    'style_power_vs_volume_diff',
    'style_damage_efficiency_diff',

    # Time_sec
    'time_sec_dec_adjperf_opp_dec_avg_diff',
    'time_sec_dec_adjperf_dec_avg_diff',
    'time_sec_rd1_dec_adjperf_dec_avg_diff',
    'time_sec_rd1_dec_adjperf_opp_dec_avg_diff',
]

FEAT_MAD2_AND_STYLES_TEST_FILTERED = [ 
    # Basic stats
    'age_dec_avg_diff',
    'age_diff',
    'reach_ratio_dec_avg_diff',
    'ufcage_dec_avg_diff',
    'days_since_last_fight_dec_avg_diff',

    # Significant strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_land_ratio_dec_avg_diff',

    # Strikes
    'strikes_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',

    # Head strikes
	'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_acc_adjperf_dec_avg_diff',
    'head_def_adjperf_dec_avg_diff',
    'head_land_dec_avg_diff',

    # Body strikes
    'body_def_dec_avg_diff',
    'body_acc_dec_adjperf_dec_avg_diff',

    # Distance strikes
    'distance_land_ratio_dec_adjperf_dec_avg_diff',
	
    # Clinch strikes
    'clinch_att_rd1_per_min_adjperf_opp_dec_avg_diff',

    # Leg strikes   
    #'leg_acc_dec_adjperf_dec_avg_diff',
    'leg_att_rd1_per_min_dec_adjperf_dec_avg_diff',

    # Ground strikes
    'ground_att_per_min_dec_adjperf_dec_avg_diff',

    # Takedowns
    'td_def_dec_avg_diff',
    'td_acc_dec_avg_diff',
    'td_att_dec_adjperf_dec_avg_diff',


    # Control time
    'ctrl_per_min_opp_dec_avg_diff',
    'ctrl_rd1_per_min_opp_dec_avg_diff',

    # Reversals
    'rev_dec_adjperf_opp_dec_avg_diff',

    # Submissions
    'sub_att_dec_avg_diff',
    'sub_att_per_min_opp_dec_avg_diff',

    # Wins
    'win_opp_dec_avg_diff',
    #'win_per_min_opp_dec_avg_diff',

    
    # Knockdowns
    #'kd_opp_dec_avg_diff',
    
    # Knockouts
	'ko_opp_dec_avg_diff',

    # Styles
    'style_power_vs_volume_diff',
    'style_damage_efficiency_diff',

    # Time_sec
    'time_sec_dec_adjperf_opp_dec_avg_diff',
]

NEW = [ 
    # Basic stats
    'age_dec_avg_diff',
    'age_diff',
    'reach_ratio_dec_avg_diff',
    'days_since_last_fight_dec_avg_diff',

    # Significant strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_land_ratio_dec_avg_diff',

    # Strikes
    'strikes_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',

    # Head strikes
	'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_acc_adjperf_dec_avg_diff',
    'head_def_adjperf_dec_avg_diff',
    'head_land_dec_avg_diff',

    # Body strikes
    'body_def_dec_avg_diff',
    'body_acc_dec_adjperf_dec_avg_diff',

    # Distance strikes
    'distance_land_ratio_dec_adjperf_dec_avg_diff',
	
    # Clinch strikes
    'clinch_att_rd1_per_min_adjperf_opp_dec_avg_diff',

    # Leg strikes   
    #'leg_acc_dec_adjperf_dec_avg_diff',
    'leg_att_rd1_per_min_dec_adjperf_dec_avg_diff',

    # Ground strikes
    'ground_att_per_min_dec_adjperf_dec_avg_diff',

    # Takedowns
    'td_def_dec_avg_diff',
    'td_acc_dec_avg_diff',
    'td_att_dec_adjperf_dec_avg_diff',


    # Control time
    'ctrl_per_min_opp_dec_avg_diff',
    'ctrl_rd1_per_min_opp_dec_avg_diff',

    # Reversals
    'rev_dec_adjperf_opp_dec_avg_diff',

    # Submissions
    'sub_att_dec_avg_diff',
    'sub_att_per_min_opp_dec_avg_diff',

    # Wins
    'win_opp_dec_avg_diff',
    #'win_per_min_opp_dec_avg_diff',

    
    # Knockdowns
    #'kd_opp_dec_avg_diff',
    
    # Knockouts
	'ko_opp_dec_avg_diff',

    # Styles
    'style_power_vs_volume_diff',
    'style_damage_efficiency_diff',

    # Time_sec
    'time_sec_dec_adjperf_opp_dec_avg_diff',
]

vSeven = [
    # Static
    'age_dec_avg_diff',
    'age_ratio_diff',
    'ufcage_dec_avg_diff',
    'reach_ratio_dec_avg_diff',
    'days_since_last_fight_dec_avg_diff',

    # Strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_def_dec_adjperf_dec_avg_diff',
    'sig_str_per_str_att_dec_adjperf_dec_avg_diff',

    # Target
    'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_def_dec_adjperf_dec_avg_diff',

    'body_def_dec_adjperf_dec_avg_diff',
    'body_acc_dec_adjperf_dec_avg_diff',

    'leg_att_per_min_dec_adjperf_dec_avg_diff',
    'leg_def_dec_adjperf_dec_avg_diff',

    # Range
    'distance_acc_dec_adjperf_dec_avg_diff',
    'distance_land_ratio_dec_adjperf_dec_avg_diff',
    'distance_land_per_min_dec_adjperf_dec_avg_diff',

    'clinch_per_sig_str_land_dec_adjperf_dec_avg_diff',
    'clinch_def_dec_adjperf_dec_avg_diff',
    'clinch_acc_dec_adjperf_dec_avg_diff',

    'ground_def_dec_adjperf_dec_avg_diff',
    'ground_acc_dec_adjperf_dec_avg_diff',
    'ground_land_per_td_land_dec_adjperf_dec_avg_diff',

    # Finish
    'ko_per_sig_str_land_dec_adjperf_dec_avg_diff',
    'ko_ratio_dec_adjperf_dec_avg_diff',
    'kd_ratio_dec_adjperf_dec_avg_diff',

    'sub_att_ratio_dec_adjperf_dec_avg_diff',
    'sub_att_per_ctrl_dec_adjperf_dec_avg_diff',

    'decision_per_min_dec_adjperf_dec_avg_diff',

    # Grappling
    'rev_per_ctrlopp_dec_adjperf_dec_avg_diff',

    'ctrl_ratio_dec_adjperf_dec_avg_diff',

    'td_acc_dec_adjperf_dec_avg_diff',
    'td_land_per_ctrl_dec_adjperf_dec_avg_diff',
    'td_land_ratio_dec_adjperf_dec_avg_diff'
]

top_22 = [
    'age_dec_avg_diff',  # 1. 0.0241
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',  # 2. 0.0148
    'head_land_ratio_dec_adjperf_dec_avg_diff',  # 3. 0.0135
    'age_ratio_diff',  # 4. 0.0102
    'reach_ratio_dec_avg_diff',  # 5. 0.0098
    'sub_att_ratio_dec_adjperf_dec_avg_diff',  # 6. 0.0091
    'distance_acc_dec_adjperf_dec_avg_diff',  # 7. 0.0077
    'ctrl_ratio_dec_adjperf_dec_avg_diff',  # 8. 0.0074
    'leg_att_per_min_dec_adjperf_dec_avg_diff',  # 9. 0.0074
    'body_acc_dec_adjperf_dec_avg_diff',  # 10. 0.0073
    'body_def_dec_adjperf_dec_avg_diff',  # 11. 0.0071
    'ko_per_sig_str_land_dec_adjperf_dec_avg_diff',  # 12. 0.0067
    'head_def_dec_adjperf_dec_avg_diff',  # 13. 0.0064
    'days_since_last_fight_dec_avg_diff',  # 14. 0.0063
    'distance_land_ratio_dec_adjperf_dec_avg_diff',  # 15. 0.0056
    'ko_ratio_dec_adjperf_dec_avg_diff',  # 16. 0.0056
    'sig_str_def_dec_adjperf_dec_avg_diff',  # 17. 0.0055
    'ufcage_dec_avg_diff',  # 18. 0.0053
    'td_land_ratio_dec_adjperf_dec_avg_diff',  # 19. 0.0052
    'ground_land_per_td_land_dec_adjperf_dec_avg_diff',  # 20. 0.0052
    'ground_def_dec_adjperf_dec_avg_diff',  # 21. 0.0052
    'decision_per_min_dec_adjperf_dec_avg_diff'  # 22. 0.0050
]

vSeven_testing = [
    # Static
    'age_dec_avg_diff',
    'age_ratio_diff',
    'age_diff',
    'time_sec_total_diff',
    'reach_ratio_dec_avg_diff',
    'reach_diff',
    'days_since_last_fight_dec_avg_diff',

    # Strikes
    'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
    'sig_str_def_dec_adjperf_dec_avg_diff',
    'sig_str_acc_dec_adjperf_dec_avg_diff',
    'sig_str_per_str_att_dec_avg_diff',
    'sig_str_land_per_min_dec_adjperf_dec_avg_diff',
    'sig_str_land_pressure_dec_adjperf_dec_avg_diff',

    # Target
    'head_land_ratio_dec_adjperf_dec_avg_diff',
    'head_def_dec_adjperf_dec_avg_diff',
    'head_acc_dec_adjperf_dec_avg_diff',
    'head_per_sig_str_land_dec_avg_diff',
    'head_land_per_min_dec_adjperf_dec_avg_diff',

    'body_land_ratio_dec_adjperf_dec_avg_diff',
    'body_def_dec_adjperf_dec_avg_diff',
    'body_acc_dec_adjperf_dec_avg_diff',
    'body_leg_per_sig_str_land_dec_avg_diff',  # Fixed: body_leg not body_land
    'body_land_per_min_dec_adjperf_dec_avg_diff',

    'leg_land_ratio_dec_adjperf_dec_avg_diff',
    'leg_def_dec_adjperf_dec_avg_diff',
    'leg_acc_dec_adjperf_dec_avg_diff',
    'leg_land_per_min_dec_adjperf_dec_avg_diff',

    # Range
    'distance_land_ratio_dec_adjperf_dec_avg_diff',
    'distance_def_dec_adjperf_dec_avg_diff',
    'distance_acc_dec_adjperf_dec_avg_diff',
    'distance_per_sig_str_land_dec_avg_diff',  # Fixed: from PerCalculator
    'distance_land_per_min_dec_adjperf_dec_avg_diff',

    'clinch_land_ratio_dec_adjperf_dec_avg_diff',
    'clinch_def_dec_adjperf_dec_avg_diff',
    'clinch_acc_dec_adjperf_dec_avg_diff',
    'clinch_per_sig_str_land_dec_avg_diff',  # Fixed: from PerCalculator
    'clinch_land_per_min_dec_adjperf_dec_avg_diff',

    'ground_land_ratio_dec_adjperf_dec_avg_diff',
    'ground_def_dec_adjperf_dec_avg_diff',
    'ground_acc_dec_adjperf_dec_avg_diff',
    'ground_per_sig_str_land_dec_avg_diff',  # Fixed: from PerCalculator
    'ground_land_per_min_dec_adjperf_dec_avg_diff',
    'ground_land_per_td_land_dec_adjperf_dec_avg_diff',
    'ground_land_per_ctrl_dec_avg_diff',  # Fixed: from PerCalculator

    # Finish
    'ko_per_sig_str_land_dec_adjperf_dec_avg_diff',
    'ko_ratio_dec_adjperf_dec_avg_diff',
    'ko_opp_dec_avg_diff',

    'sub_att_ratio_dec_adjperf_dec_avg_diff',
    'sub_att_per_ctrl_dec_adjperf_dec_avg_diff',
    'sub_att_opp_dec_avg_diff',
    'sub_acc_dec_adjperf_dec_avg_diff',
    'sub_def_dec_adjperf_dec_avg_diff',

    'decision_per_min_dec_adjperf_dec_avg_diff',
    'decision_ratio_dec_adjperf_dec_avg_diff',

    # Grappling
    'rev_per_ctrlopp_dec_adjperf_dec_avg_diff',  # Fixed: from PerCalculator
    'rev_ratio_dec_adjperf_dec_avg_diff',

    'ctrl_ratio_dec_adjperf_dec_avg_diff',
    'ctrl_per_min_dec_adjperf_dec_avg_diff',
    'ctrl_per_min_dec_avg_diff',  # Fixed: removed duplicate _diff
    'ctrl_per_min_opp_dec_avg_diff',

    'td_acc_dec_adjperf_dec_avg_diff',
    'td_def_dec_adjperf_dec_avg_diff',
    'td_land_per_ctrl_dec_adjperf_dec_avg_diff',  # Fixed: from PerCalculator
    'td_land_ratio_dec_adjperf_dec_avg_diff',
    'td_per_sig_str_att_dec_adjperf_dec_avg_diff',  # Fixed: from PerCalculator
    'td_att_per_min_dec_adjperf_dec_avg_diff',
    'td_att_opp_dec_avg_diff'
]

vSeven_testing2 = [ #v7
    # Static
    "age_dec_avg_diff",
    "age_ratio_diff",
    "reach_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "days_since_last_fight_dec_avg",
    "weightclass_encoded", # THIS HAS TO BE fighter1_weightclass_encoded for next training run

    # Strikes
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_def_dec_adjperf_dec_avg_diff",
    "sig_str_acc_dec_adjperf_dec_avg_diff",

    # Target
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    "head_per_sig_str_land_dec_avg_diff",
    "body_def_dec_adjperf_dec_avg_diff",
    "body_acc_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_def_dec_adjperf_dec_avg_diff",

    # Range
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "distance_land_rd1_ratio_dec_adjperf_dec_avg_diff", # NEW
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_land_per_min_dec_adjperf_dec_avg_diff",
    "distance_per_sig_str_land_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "clinch_acc_dec_adjperf_dec_avg_diff",
    "ground_land_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff",

    # Finish
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_dec_avg_diff",
    "sub_def_dec_adjperf_dec_avg_diff",
    #"decision_opp_dec_avg_diff", # NEW trying to remove
    "win_dec_adjperf_dec_avg_diff",

    # Grappling
    "rev_per_ctrlopp_dec_adjperf_dec_avg_diff",
    "rev_per_ctrlopp_dec_avg_diff",
    "rev_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "td_att_opp_dec_avg_diff",
    "td_att_rd1_opp_dec_avg_diff", # NEW
    "td_land_per_ctrl_dec_adjperf_dec_avg_diff",
    "td_per_sig_str_att_dec_adjperf_dec_avg_diff",
]
# This is for testing, vSeven_testing2 is the final features list
vSeven_testing2_with_f1 = [ #v7
    # Static
    "age_dec_avg_diff",
    "age_ratio_diff",
    "reach_ratio_dec_avg_diff",
    "days_since_last_fight_dec_avg_diff",
    "weightclass_encoded",

    # Strikes
    "sig_str_land_ratio_dec_adjperf_dec_avg_diff",
    "sig_str_def_dec_adjperf_dec_avg_diff",
    #"sig_str_acc_dec_adjperf_dec_avg_diff",

    # Target
    "head_land_ratio_dec_adjperf_dec_avg_diff",
    "head_def_dec_adjperf_dec_avg_diff",
    #"head_acc_dec_adjperf_dec_avg_diff",
    #"head_per_sig_str_land_dec_avg_diff",
    #"head_acc_dec_adjperf_dec_avg_diff",
    "leg_land_per_min_dec_adjperf_dec_avg_diff",
    "leg_land_ratio_dec_adjperf_dec_avg_diff",

    # Range
    "distance_land_ratio_dec_adjperf_dec_avg_diff",
    "distance_acc_dec_adjperf_dec_avg_diff",
    "distance_per_sig_str_land_dec_avg_diff",
    "clinch_land_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_ratio_dec_adjperf_dec_avg_diff",
    "ground_land_per_ctrl_dec_avg_diff",
    #"ground_acc_dec_adjperf_dec_avg_diff",

    # Finish
    "ko_per_sig_str_land_dec_adjperf_dec_avg_diff",
    "ko_ratio_dec_adjperf_dec_avg_diff",
    #"ko_opp_dec_avg_diff",
    "sub_att_ratio_dec_adjperf_dec_avg_diff",
    "sub_att_dec_avg_diff",
    #"decision_opp_dec_avg_diff",
    #"win_opp_dec_avg_diff",
    "win_dec_adjperf_dec_avg_diff",
    #"win_dec_avg_diff",

    # Totals
    #"time_sec_total_diff",
    #"sig_str_land_total_diff",

    # Grappling
    #"rev_per_ctrlopp_dec_adjperf_dec_avg_diff",
    "ctrl_ratio_dec_adjperf_dec_avg_diff",
    "ctrl_per_min_opp_dec_avg_diff",
    "td_att_opp_dec_avg_diff",
    "td_att_dec_avg_diff",
    #"td_land_ratio_dec_avg_diff",
    #"rev_per_ctrlopp_opp_dec_avg_diff", 
]

DECISION_TEST_FEATS = [
    "fighter1_reach",
    "fighter2_reach",
    "weightclass_encoded",

    # --- TIME / DURABILITY ---
    "fighter1_time_sec_avg",
    "fighter2_time_sec_avg",
    "fighter1_time_sec_mad",
    "fighter2_time_sec_mad",
    "fighter1_time_sec_dec_avg",
    "fighter2_time_sec_dec_avg",
    "fighter2_time_sec_opp_dec_avg",
    "fighter1_time_sec_opp_dec_avg",
    "fighter2_time_sec_rd1_avg",
    "fighter2_time_sec_rd1_total_opp_avg",
    "fighter2_time_sec_total_opp_avg",

    # --- DISTANCE STRIKING (pace/output & ratios) ---
    "fighter1_distance_att_dec_avg",
    "fighter2_distance_att_opp_dec_avg",
    "fighter1_distance_att_opp_dec_avg",
    "fighter1_distance_att_rd1_dec_avg",
    "fighter1_distance_att_total_dec_avg",
    "fighter1_distance_att_ratio_dec_avg",
    "fighter2_distance_att_rd1_opp_dec_avg",
    "fighter1_distance_att_rd1_ratio_dec_avg",
    "fighter1_distance_att_rd1_ratio_opp_dec_avg",
    "fighter1_distance_att_ratio_opp_dec_avg",
    "fighter1_distance_att_ratio_dec_adjperf_dec_avg",
    "fighter2_strikes_att_rd1_per_min_dec_adjperf_dec_avg",  # tempo proxy
    "fighter2_strikes_att_ratio_dec_adjperf_dec_avg",
    "fighter2_strikes_att_ratio_adjperf_dec_avg",
    "fighter1_strikes_att_dec_avg",
    "fighter1_strikes_att_rd1_dec_avg",
    "fighter1_strikes_att_rd1_opp_dec_avg",
    "fighter2_strikes_att_rd1_dec_avg",
    "fighter2_strikes_att_rd1_opp_dec_avg",
    "fighter1_strikes_att_opp_dec_avg",
    "fighter2_strikes_att_per_min_opp_dec_avg",

    # --- SUBMISSION THREAT / CONTROL EFFICIENCY ---
    "fighter1_sub_att_dec_avg",
    "fighter1_sub_att_per_min_dec_avg",
    "fighter1_sub_att_per_min_adjperf_dec_avg",
    "fighter1_sub_att_ratio_dec_adjperf_dec_avg",
    "fighter2_sub_att_per_min_dec_avg",
    "fighter2_sub_att_per_min_opp_dec_avg",
    "fighter2_sub_acc_opp_dec_avg",
    "fighter1_sub_acc_adjperf_dec_avg",
    "fighter1_sub_acc_opp_dec_avg",
    "fighter2_sub_acc_adjperf_dec_avg",
    "fighter1_sub_land_per_min_dec_avg",
    "fighter1_sub_land_rd1_ratio_adjperf_dec_avg",
    "fighter1_sub_land_ratio_dec_avg",
    "fighter1_sub_land_ratio_opp_dec_avg",
    "fighter2_sub_land_ratio_dec_avg",
    "fighter2_sub_land_opp_dec_avg",
    "fighter2_sub_land_rd1_per_min_adjperf_dec_avg",
    "fighter1_sub_per_all_ctrl_dec_avg",
    "fighter2_sub_per_all_ctrl_dec_avg",
    "fighter2_sub_per_all_ctrl_adjperf_dec_avg",
    "fighter2_sub_per_all_ctrl_opp_dec_avg",
    "fighter2_sub_per_all_ctrl_dec_adjperf_dec_avg",

    # --- KO / POWER (finish threat) ---
    "fighter1_ko_dec_avg",
    "fighter1_ko_per_min_dec_avg",
    "fighter2_ko_per_min_dec_avg",
    "fighter1_ko_per_min_opp_dec_avg",
    "fighter2_ko_per_min_opp_dec_avg",
    "fighter1_ko_per_min_dec_adjperf_dec_avg",
    "fighter2_ko_per_min_adjperf_dec_avg",
    "fighter2_ko_rd1_per_min_dec_adjperf_dec_avg",
    "fighter2_ko_rd1_per_min_adjperf_dec_avg",
    "fighter1_ko_rd1_per_min_opp_dec_avg",
    "fighter1_ko_rd1_dec_avg",
    "fighter1_ko_rd1_total_opp_dec_avg",
    "fighter2_ko_rd1_total_dec_avg",
    "fighter1_ko_rd1_ratio_adjperf_dec_avg",
    "fighter1_ko_rd1_ratio_dec_adjperf_dec_avg",
    "fighter1_ko_per_sig_str_land_opp_dec_avg",
    "fighter2_ko_per_sig_str_land_opp_dec_avg",
    "fighter2_ko_per_sig_str_land_dec_avg",

    # --- DECISION / DURABILITY METRICS ---
    "fighter2_decision_dec_avg",
    "fighter1_decision_per_min_adjperf_dec_avg",
    "fighter2_decision_per_min_dec_adjperf_dec_avg",
    "fighter1_decision_per_min_dec_adjperf_dec_avg",
    "fighter1_decision_opp_dec_avg",
    "fighter1_decision_ratio_adjperf_dec_avg",
    "fighter1_decision_ratio_dec_adjperf_dec_avg",

    # --- KNOCKDOWNS / DANGER ---
    "fighter2_kd_dec_avg",
    "fighter1_kd_dec_avg",
    "fighter2_kd_opp_dec_avg",
    "fighter2_kd_per_min_opp_dec_avg",
    "fighter2_kd_rd1_per_min_adjperf_dec_avg",
    "fighter2_kd_rd1_per_min_opp_dec_avg",
    "fighter1_kd_rd1_per_min_opp_dec_avg",

    # --- ROUND-1 & MOMENTUM (explosiveness / front-loaded risk) ---
    "fighter2_win_rd1_per_min_dec_adjperf_dec_avg",
    "fighter2_win_rd1_opp_dec_avg",
    "fighter2_win_per_min_dec_adjperf_dec_avg",
    "fighter2_win_per_min_adjperf_dec_avg",
    "fighter2_win_per_min_opp_dec_avg",
    "fighter1_win_per_min_dec_adjperf_dec_avg",
    "fighter1_win_per_min_dec_avg",
    "fighter2_win_opp_dec_avg",
    "fighter2_win_rd1_per_min_opp_dec_avg",

    # --- TD / CONTROL (positional dominance & stalling likelihood) ---
    "fighter1_td_rd1_acc_opp_dec_avg",
    "fighter2_td_acc_opp_dec_avg",
    "fighter2_td_rd1_acc_dec_avg",
    "fighter2_td_att_per_min_opp_dec_avg",
    "fighter2_td_att_rd1_per_min_opp_dec_avg",
    "fighter2_td_att_per_min_dec_avg",
    "fighter2_td_land_ratio_adjperf_dec_avg",
    "fighter2_td_land_ratio_opp_dec_avg",
    "fighter2_td_land_total_opp_dec_avg",
    "fighter2_td_land_rd1_opp_dec_avg",
    "fighter1_td_land_opp_dec_avg",
    "fighter1_td_land_total_ratio_adjperf_dec_avg",
    "fighter1_td_land_rd1_per_min_opp_dec_avg",
    "fighter1_td_att_per_min_opp_dec_avg",

    "fighter1_ctrl_dec_avg",
    "fighter1_ctrl_total_dec_avg",
    "fighter1_ctrl_rd1_total_opp_dec_avg",
    "fighter1_ctrl_per_min_opp_dec_avg",
    "fighter2_ctrl_rd1_per_min_dec_avg",
    "fighter2_ctrl_total_ratio_adjperf_dec_avg",
    "fighter2_ctrl_total_ratio_dec_adjperf_dec_avg",
    "fighter2_ctrl_rd1_ratio_adjperf_dec_avg",

    "fighter1_ground_land_per_td_land_opp_dec_avg",
    "fighter1_ground_land_per_td_land_dec_avg",
    "fighter2_ground_land_per_td_land_opp_dec_avg",
    "fighter2_ground_land_per_td_land_dec_adjperf_dec_avg",
]

# ============================================================================
# DECISION PREDICTION FEATURE LIST - EXPERIMENTATION NOTES
# ============================================================================
# 
# LATEST RESULTS (70 features after Round 1 removal):
# - Previous: 85 features
# - Removed: 15 lowest-importance features (#71-85, all < 0.0015)
# - Current: 70 features
# - Test Accuracy: 0.6030 (improved from 0.5966, +0.0064 improvement! Total +0.0236 from baseline)
# - Training Accuracy: 0.8471
# - Key Insight: Time features dominate (#1, #2, #5), reach_dec_avg jumped to #4 (0.0070)!
# 
# LATEST RESULTS (56 features after Round 2 removal):
# - Previous: 70 features
# - Removed: 15 redundant/low-value features (actually 14 removed, 1 feature count discrepancy)
# - Current: 56 features
# - Test Accuracy: 0.5987 (slight drop from 0.6030, but still +0.0193 from baseline)
# - Training Accuracy: 0.8004
# - Key Insight: Time features remain dominant (#1, #2, #5), reach_dec_avg is #4 (0.0074)
# 
# LATEST RESULTS (26 features after Round 4 removal):
# - Previous: 36 features
# - Removed: 10 lowest-importance/redundant features
# - Current: 26 features
# - Test Accuracy: 0.5987 (from previous 36-feature run)
# - Training Accuracy: 0.8869
# - Key Insight: Time features remain #1-2, finishing power #3, opponent time #4
# 
# FEATURE REMOVAL STRATEGY:
# Starting to pare down feature list - removing lowest importance features (10-15 at a time)
# Priority: Keep high-importance features, remove redundant/low-value ones
# 
# FEATURE IMPORTANCE INSIGHTS (Latest Run - 56 features):
# 1. Time features are CRITICAL and INCREASING:
#    - #1 time_sec_rd1_avg: 0.0156 (increased from 0.0136 - MOST IMPORTANT!)
#    - #2 time_sec_dec_avg: 0.0104 (increased from 0.0091)
#    - #5 time_sec_opp_dec_avg: 0.0065 (slight decrease, still critical)
# 2. Physical attributes matter:
#    - #4 reach_dec_avg: 0.0074 (increased from 0.0070 - very important!)
# 3. Finishing power remains strong:
#    - #3 ko_per_min_dec_adjperf_dec_avg: 0.0079 (increased from 0.0074)
#    - #9 ko_rd1_per_min_dec_adjperf_dec_avg_diff: 0.0057 (increased from 0.0049)
# 4. Decision-specific features are STRONG:
#    - #8 decision_opp_dec_avg_diff: 0.0060 (stable, critical!)
#    - #10 decision_total_opp_dec_avg_diff: 0.0057 (increased from 0.0035!)
#    - #18 decision_dec_avg_diff: 0.0043 (stable)
#    - #22 decision_per_min_dec_adjperf_dec_avg_diff: 0.0042 (stable)
# 5. Defense features remain valuable but some are lower importance:
#    - #15 sub_def_dec_adjperf_dec_avg_diff: 0.0045 (high importance)
#    - #20 td_def_dec_adjperf_dec_avg_diff: 0.0042 (high importance)
#    - #21 body_def_dec_adjperf_dec_avg_diff: 0.0042 (high importance)
#    - #40 clinch_def_dec_adjperf_dec_avg_diff: 0.0034 (medium)
#    - #47-49 sig_str_def, leg_def, head_def: 0.0030 (lower, candidates for removal)
#    - #50 distance_def_dec_adjperf_dec_avg_diff: 0.0029 (low, candidate)
#    - #55 ground_def_dec_adjperf_dec_avg_diff: 0.0027 (low, candidate)
# 6. Ground striking efficiency is excellent: #13 ground_land_per_ctrl_dec_adjperf_dec_avg_diff: 0.0050
# 7. Head strike ratio is important: #6 head_land_ratio_dec_adjperf_dec_avg: 0.0062 (non-diff!)
#    - #56 head_land_ratio_dec_adjperf_dec_avg_diff: 0.0025 (lowest importance - candidate)
# 8. Submission success: #7 sub_per_all_ctrl_dec_adjperf_dec_avg_diff: 0.0062 (increased!)
# 9. Knockdown rate: #11 kd_per_min_dec_adjperf_dec_avg_diff: 0.0050, #24 kd_per_min_dec_adjperf_dec_avg: 0.0042
# 10. Range profile features: Some are lower importance (#37, #45, #50) - candidates for removal
# 11. Control/wrestling: #42 ctrl_per_min_dec_adjperf_dec_avg: 0.0032, #43 rev_per_ctrlopp_dec_adjperf_dec_avg: 0.0031 (low, candidates)
# 12. Lowest importance features (candidates for removal):
#    - #56 head_land_ratio_dec_adjperf_dec_avg_diff: 0.0025
#    - #55 ground_def_dec_adjperf_dec_avg_diff: 0.0027
#    - #54 ko_per_sig_str_land_dec_adjperf_dec_avg_diff: 0.0028
#    - #53 td_per_sig_str_att_dec_adjperf_dec_avg_diff: 0.0028
#    - #52 time_sec_opp_dec_avg_diff: 0.0029
#    - #51 ground_land_per_td_land_dec_adjperf_dec_avg_diff: 0.0029
#    - #50 distance_def_dec_adjperf_dec_avg_diff: 0.0029
# 
# EXPERIMENTS CONDUCTED:
# Round 1:
# - Added decision_dec_avg_diff and decision_total_dec_avg_diff (working - #13, #79)
# - Added decision_per_min_dec_adjperf_dec_avg_diff (working - #11, importance DOUBLED!)
# - Added ufcage_dec_avg_diff (working - #44, importance DOUBLED!)
# - Added time_sec_rd1_dec_avg_diff (working - #66)
# - Removed style features (user removed - not helpful)
# Round 2:
# - Added td_per_sig_str_att_dec_adjperf_dec_avg_diff (wrestling pressure - #77)
# - Added ground_land_per_ctrl_dec_adjperf_dec_avg_diff (ground striking efficiency - #21, EXCELLENT!)
# - Added body_leg_per_sig_str_land_dec_adjperf_dec_avg_diff (body/leg distribution - #56)
# 
# ROUND 3 EXPERIMENTS:
# - Added win_opp_dec_avg_diff (opponent's win rate - available)
# - Attempted win_dec_adjperf_dec_avg_diff (NOT AVAILABLE in database - removed)
# - Attempted strikes_land_dec_adjperf_dec_avg_diff (NOT AVAILABLE - removed)
# - Attempted body_land_dec_adjperf_dec_avg_diff (NOT AVAILABLE - removed)
# - Attempted leg_land_dec_adjperf_dec_avg_diff (NOT AVAILABLE - removed)
# 
# POTENTIAL FUTURE EXPERIMENTS:
# - ko_sub_per_win features (combined finishing rate - strong inverse signal) - NEED TO VERIFY EXISTS
# - More non-diff opponent features (e.g., decision_opp_dec_avg without _diff)
# - Round 1 specific decision features
# - Interaction features between time and finishing power
# 
# LAYER EFFECTIVENESS:
# - dec_avg (time-decayed): Very important - most features use this
# - opp_dec_avg (opponent's decayed avg): Critical - top 4 includes this
# - dec_adjperf_dec_avg (opponent-adjusted): Important for finishing/defense - #3 feature uses this
# - Simple _avg (for time_sec_rd1): MOST IMPORTANT - #1 feature!
# 
# ACCURACY PROGRESSION:
# - Baseline: 0.5794
# - After Round 1: 0.5944 (+0.0150)
# - After Round 2: 0.5987 (+0.0043, total +0.0193 from baseline)
# - After Round 3: 0.5966 (-0.0021, still +0.0172 from baseline)
# - After Round 1 Removal (70 features): 0.6030 (+0.0064, total +0.0236 from baseline) - EXCELLENT!
# - After Round 2 Removal (56 features): 0.5987 (-0.0043, still +0.0193 from baseline) - slight drop but acceptable
# - After Round 3 Removal (40 features): 0.6030 (rebounded to previous high, total +0.0236 from baseline) - EXCELLENT!
# - After Round 4 Removal (26 features): TBD (will update after next run)
# 
# FEATURE REMOVAL ROUND 1:
# Removing 15 lowest-importance features (#71-85, all < 0.0015):
# - reach_dec_avg_diff (#85, 0.0011)
# - time_sec_rd1_dec_avg_diff (#84, 0.0011)
# - head_land_per_min_dec_adjperf_dec_avg_diff (#83, 0.0011)
# - td_land_per_ctrl_dec_adjperf_dec_avg (#82, 0.0012)
# - sub_att_per_ctrl_dec_adjperf_dec_avg (#81, 0.0012)
# - td_att_per_min_dec_adjperf_dec_avg (#80, 0.0013)
# - td_att_per_min_dec_adjperf_dec_avg_diff (#79, 0.0013)
# - decision_total_ratio_dec_adjperf_dec_avg_diff (#78, 0.0013)
# - td_land_per_ctrl_dec_adjperf_dec_avg_diff (#77, 0.0013)
# - time_sec_rd1_opp_dec_avg_diff (#76, 0.0013)
# - sig_str_land_per_min_dec_adjperf_dec_avg (#75, 0.0013)
# - rev_per_ctrlopp_dec_adjperf_dec_avg_diff (#74, 0.0014)
# - clinch_per_sig_str_land_dec_adjperf_dec_avg_diff (#73, 0.0015)
# - body_leg_per_sig_str_land_dec_adjperf_dec_avg_diff (#72, 0.0015)
# - time_sec_rd1_total_dec_avg_diff (#71, 0.0015)
# 
# FEATURE REMOVAL ROUND 2 (Thoughtful removal based on redundancy and low value):
# Strategy: Remove redundant features where we have better versions, and low-importance features
# Removed 15 features:
# TIME REDUNDANCIES (4 features):
# - time_sec_total_dec_avg_diff (#39, 0.0033) - redundant with time_sec_dec_avg_diff (#22, 0.0039)
# - time_sec_total_opp_avg (#41, 0.0033) - redundant with time_sec_opp_dec_avg (#5, 0.0070)
# - time_sec_total_opp_avg_diff (#35, 0.0034) - redundant with time_sec_opp_dec_avg_diff (#46, 0.0031)
# - time_sec_rd1_total_opp_dec_avg_diff (#57, 0.0028) - low importance, have better rd1 features
# KO REDUNDANCIES (2 features):
# - ko_per_sig_str_land_dec_adjperf_dec_avg (#54, 0.0029) - redundant with diff version (#67, 0.0024)
# - ko_rd1_per_min_dec_adjperf_dec_avg (#61, 0.0026) - redundant with diff version (#10, 0.0049)
# DECISION LOW VALUE (2 features):
# - decision_total_dec_avg_diff (#70, 0.0021) - lowest importance, have decision_dec_avg_diff (#16, 0.0044)
# - decision_ratio_dec_adjperf_dec_avg_diff (#33, 0.0034) - redundant with decision_opp_dec_avg_diff (#7, 0.0061)
# RANGE PROFILE REDUNDANCIES (3 features):
# - distance_per_sig_str_land_dec_adjperf_dec_avg_diff (#66, 0.0024) - redundant with non-diff (#50, 0.0030)
# - clinch_per_sig_str_land_dec_adjperf_dec_avg (#63, 0.0036) - low importance, have diff version (#45, 0.0032)
# - ground_per_sig_str_land_dec_adjperf_dec_avg_diff (#52, 0.0029) - redundant with non-diff (#27, 0.0037)
# CONTROL/WRESTLING LOW VALUE (2 features):
# - ctrl_per_min_dec_adjperf_dec_avg_diff (#65, 0.0024) - low importance, have non-diff (#30, 0.0036)
# - ground_land_per_td_land_dec_adjperf_dec_avg (#62, 0.0025) - redundant with diff version (#44, 0.0032)
# ACCURACY LOW VALUE (1 feature):
# - sig_str_per_str_att_dec_adjperf_dec_avg_diff (#58, 0.0027) - redundant with non-diff (#37, 0.0034)
# SUBMISSION LOW VALUE (1 feature):
# - sub_per_all_ctrl_dec_adjperf_dec_avg (#55, 0.0029) - redundant with diff version (#8, 0.0052)
# 
# FEATURE REMOVAL ROUND 3 (Target: ~40 features, focus on decision prediction):
# Strategy: Remove lowest-importance features and redundant ones that don't directly measure decision likelihood
# Removed 16 features:
# LOWEST IMPORTANCE (7 features - all < 0.0030):
# - head_land_ratio_dec_adjperf_dec_avg_diff (#56, 0.0025) - lowest importance, have non-diff version (#6, 0.0062)
# - ground_def_dec_adjperf_dec_avg_diff (#55, 0.0027) - low importance defense feature
# - ko_per_sig_str_land_dec_adjperf_dec_avg_diff (#54, 0.0028) - finishing efficiency diff, low importance
# - td_per_sig_str_att_dec_adjperf_dec_avg_diff (#53, 0.0028) - wrestling pressure, low importance
# - time_sec_opp_dec_avg_diff (#52, 0.0029) - redundant with time_sec_opp_dec_avg (#5, 0.0065)
# - ground_land_per_td_land_dec_adjperf_dec_avg_diff (#51, 0.0029) - ground control efficiency, low importance
# - distance_def_dec_adjperf_dec_avg_diff (#50, 0.0029) - low importance defense feature
# DEFENSE REDUNDANCIES (3 features - keep high-importance ones):
# - sig_str_def_dec_adjperf_dec_avg_diff (#47, 0.0030) - redundant with other defense features, keep sub/td/body_def
# - leg_def_dec_adjperf_dec_avg_diff (#48, 0.0030) - lower importance than body/head defense
# - head_def_dec_adjperf_dec_avg_diff (#49, 0.0030) - lower importance than body_def (#21, 0.0042)
# RANGE PROFILE LOW VALUE (2 features):
# - distance_per_sig_str_land_dec_adjperf_dec_avg (#37, 0.0036) - range distribution, keep clinch/ground
# - clinch_per_sig_str_land_dec_adjperf_dec_avg (#45, 0.0030) - range distribution, low importance
# CONTROL/WRESTLING LOW VALUE (2 features):
# - ctrl_per_min_dec_adjperf_dec_avg (#42, 0.0032) - control time, low importance for decision prediction
# - rev_per_ctrlopp_dec_adjperf_dec_avg (#43, 0.0031) - scrambling, low importance
# STATIC CONTEXT LOW VALUE (1 feature):
# - days_since_last_fight_dec_avg (#46, 0.0030) - redundant with diff version (#38, 0.0034)
# OUTPUT REDUNDANCY (1 feature):
# - head_land_per_min_dec_adjperf_dec_avg (#14, 0.0047) - have head_land_ratio which is more important (#6, 0.0062)
# 
# FEATURE REMOVAL ROUND 4 (Target: ~26 features, remove lowest-importance):
# Strategy: Remove 10 lowest-importance features from 36-feature list, focusing on redundant or low-value features
# Removed 10 features (from importance ranking #27-36):
# RANGE PROFILE LOW VALUE (2 features):
# - ground_per_sig_str_land_dec_adjperf_dec_avg (#36, 0.0085) - ground fighting style, lowest importance
# - distance_att_per_min_dec_adjperf_dec_avg (#34, 0.0087) - redundant with diff version (#14, 0.0110)
# STATIC CONTEXT LOW VALUE (2 features):
# - weightclass_encoded (#35, 0.0087) - static feature, low importance
# - age_dec_avg_diff (#33, 0.0088) - redundant with age_dec_avg (#29, 0.0093)
# RANGE ATTACK LOW VALUE (2 features):
# - clinch_att_per_min_dec_adjperf_dec_avg (#31, 0.0091) - redundant with diff version (#25, 0.0096)
# - ground_att_per_min_dec_adjperf_dec_avg (#24, 0.0096) - redundant with diff version (#27, 0.0095)
# - ground_att_per_min_dec_adjperf_dec_avg_diff (#27, 0.0095) - ground attack, low importance
# ACCURACY LOW VALUE (1 feature):
# - sig_str_per_str_att_dec_adjperf_dec_avg (#23, 0.0097) - accuracy feature, low importance
# OUTPUT LOW VALUE (1 feature):
# - sig_str_land_per_min_dec_adjperf_dec_avg_diff (#22, 0.0098) - output volume, low importance
# TIME VARIABILITY LOW VALUE (1 feature):
# - time_sec_mad (#28, 0.0095) - redundant with time_sec_mad_diff (#19, 0.0102)
# STATIC CONTEXT LOW VALUE (1 feature):
# - ufcage_dec_avg_diff (#26, 0.0096) - UFC experience, low importance
# 
# FEATURES TO MONITOR FOR POTENTIAL RE-ADDITION:
# - ground_per_sig_str_land_dec_adjperf_dec_avg: If ground fighting style becomes more important
# - weightclass_encoded: If weight class differences become more significant
# - distance_att_per_min_dec_adjperf_dec_avg: If non-diff range features become valuable
# - clinch_att_per_min_dec_adjperf_dec_avg: If non-diff clinch features become valuable
# - ground_att_per_min_dec_adjperf_dec_avg: If non-diff ground features become valuable
# - sig_str_per_str_att_dec_adjperf_dec_avg: If accuracy features become more important
# - sig_str_land_per_min_dec_adjperf_dec_avg_diff: If output volume differences become critical
# - time_sec_mad: If variability without diff becomes important
# - age_dec_avg_diff: If age differences become more significant
# - ufcage_dec_avg_diff: If UFC experience differences become critical
# 
# ============================================================================

DECISION_TEST_FEATS2 = [
    "fighter1_time_sec_avg",
    "fighter2_time_sec_avg",
    "fighter2_time_sec_mad",
    "fighter2_sub_per_all_ctrl_dec_avg",
    "fighter1_sub_att_dec_avg",
    "fighter2_ko_per_min_adjperf_dec_avg",
    "fighter2_ko_sub_per_win_dec_avg",
    "fighter1_ko_per_min_adjperf_dec_avg",
    "fighter1_distance_att_per_min_adjperf_dec_avg",
    "fighter2_time_sec_rd1_avg",
    "fighter2_distance_att_per_min_adjperf_dec_avg",
    "fighter1_time_sec_rd1_avg",
    "fighter1_time_sec_mad",
    "fighter2_ground_land_per_ctrl_dec_avg",
    "fighter1_sub_att_per_min_dec_avg",
    "fighter1_ground_land_per_ctrl_dec_avg",
    "fighter2_sig_str_land_per_min_dec_avg",
    "fighter2_sub_att_per_min_dec_avg",
    "fighter2_sub_att_dec_avg",
    "fighter1_ko_sub_per_win_dec_avg",
    "fighter2_ko_per_sig_str_land_adjperf_dec_avg",
    "fighter1_ko_per_sig_str_land_adjperf_dec_avg",
    "fighter1_sig_str_land_per_min_dec_avg",
    "fighter1_sub_per_all_ctrl_dec_avg",
    "weightclass_encoded",
]

DECISION_TEST_FEATS3 = [
    "fighter1_time_sec_rd1_avg",
    "fighter2_time_sec_rd1_avg",
    "fighter1_time_sec_avg",
    "fighter2_time_sec_avg",
    "fighter2_time_sec_mad",
    "fighter1_time_sec_mad",
    #"fighter1_time_sec_adjperf_dec_avg", doestn exist
    #"fighter2_time_sec_adjperf_dec_avg",
    
    "fighter2_ko_sub_per_win_dec_avg",
    "fighter1_ko_sub_per_win_dec_avg",
    "fighter1_ko_per_sig_str_land_dec_avg",
    "fighter2_ko_per_sig_str_land_dec_avg",
    
    "fighter1_ko_per_min_adjperf_dec_avg",
    "fighter2_ko_per_min_adjperf_dec_avg",
    "fighter1_ko_per_min_dec_avg",
    "fighter2_ko_per_min_dec_avg",
    
    "fighter1_distance_att_per_min_adjperf_dec_avg",
    "fighter2_distance_att_per_min_adjperf_dec_avg",

    "fighter1_sig_str_land_per_min_dec_avg",
    "fighter2_sig_str_land_per_min_dec_avg",
    
    "fighter1_ground_land_per_ctrl_dec_avg",
    "fighter2_ground_land_per_ctrl_dec_avg",
    
    "fighter1_sub_per_all_ctrl_dec_avg",
    "fighter2_sub_per_all_ctrl_dec_avg",
    "fighter1_sub_att_dec_avg",
    "fighter2_sub_att_dec_avg",
    "sub_att_dec_avg_diff",
    
    "weightclass_encoded",# THIS HAS TO BE fighter1_weightclass_encoded for next training run
]
# best, 61% acc
DECISION_TEST_FEATS4 = [
    "fighter1_time_sec_rd1_avg",
    "fighter2_time_sec_rd1_avg",
    "time_sec_rd1_avg_diff",
    "fighter1_time_sec_avg",
    "fighter2_time_sec_avg",
    "time_sec_avg_diff",
    "fighter1_time_sec_mad",
    "fighter2_time_sec_mad",
    "time_sec_mad_diff",
    
    "fighter1_ko_sub_per_win_dec_avg",
    "fighter2_ko_sub_per_win_dec_avg",
    "ko_sub_per_win_dec_avg_diff",
    "fighter1_ko_per_sig_str_land_dec_avg",
    "fighter2_ko_per_sig_str_land_dec_avg",
    "ko_per_sig_str_land_dec_avg_diff",
    
    "fighter1_ko_per_min_adjperf_dec_avg",
    "fighter2_ko_per_min_adjperf_dec_avg",
    "ko_per_min_adjperf_dec_avg_diff",
    "fighter1_ko_per_min_dec_avg",
    "fighter2_ko_per_min_dec_avg",
    "ko_per_min_dec_avg_diff",
    
    "fighter1_distance_att_per_min_dec_avg",
    "fighter2_distance_att_per_min_dec_avg",
    "distance_att_per_min_dec_avg_diff",
    
    "fighter1_sig_str_land_per_min_dec_avg",
    "fighter2_sig_str_land_per_min_dec_avg",
    "sig_str_land_per_min_dec_avg_diff",
    
    "fighter1_ground_land_per_ctrl_dec_avg",
    "fighter2_ground_land_per_ctrl_dec_avg",
    "ground_land_per_ctrl_dec_avg_diff",
    
    "fighter1_sub_per_all_ctrl_dec_avg",
    "fighter2_sub_per_all_ctrl_dec_avg",
    "sub_per_all_ctrl_dec_avg_diff",
    "fighter1_sub_att_dec_avg",
    "fighter2_sub_att_dec_avg",
    "sub_att_dec_avg_diff",
    
    "weightclass_encoded", # THIS HAS TO BE fighter1_weightclass_encoded for next training run
]

class FeatureSelector:
    """
    A class to help create feature lists based on patterns for different stat categories.
    
    Wildcards (*) work by sequential filtering:
    - 'ctrl_*_per_min_*' will find features containing 'ctrl_' AND '_per_min_'
    - 'head_acc_*' will find features containing 'head_acc_'
    
    Usage:
        selector = FeatureSelector(available_features)
        patterns = {
            'age_': ['age_dec_avg_diff', 'age_ratio_diff'],
            'ctrl_': ['ctrl_rd1_*', 'ctrl_*_per_min_*'],
            'head_': ['head_acc_*', 'head_def_*', 'head_land_dec_avg_diff']
        }
        features = selector.select_features(patterns)
    """
    
    def __init__(self, available_features=None, db_url=None):
        """
        Initialize the FeatureSelector.
        
        Args:
            available_features: List of available feature names. If None, queries database.
            db_url: Database URL for querying available features.
        """
        if available_features is None:
            self.available_features = self._query_database_features(db_url or database_url())
        else:
            self.available_features = available_features
    
    def _query_database_features(self, db_url: str) -> List[str]:
        """
        Query the database to get all available features from features.<stat> tables.
        
        Args:
            db_url: Database connection URL
            
        Returns:
            List of available feature names
        """
        try:
            engine = create_engine(db_url)
            all_features = []
            
            # Query features from BASE_STATIC_FEATS and BASE_DYNAMIC_FEATS tables
            all_stats = BASE_STATIC_FEATS + BASE_DYNAMIC_FEATS
            
            with engine.connect() as conn:
                for stat in all_stats:
                    try:
                        # Check if the table exists and get its columns
                        query = text(f"""
                            SELECT column_name 
                            FROM information_schema.columns 
                            WHERE table_schema = 'features' 
                            AND table_name = '{stat}'
                            AND column_name NOT IN ('fight_id', 'fighter_id', 'event_id')
                            ORDER BY column_name
                        """)
                        result = conn.execute(query)
                        columns = [row[0] for row in result.fetchall()]
                        all_features.extend(columns)
                    except Exception as e:
                        print(f"Warning: Could not query features.{stat} table: {e}")
                        continue
            
            # Remove duplicates and sort
            unique_features = sorted(list(set(all_features)))
            print(f"Found {len(unique_features)} features from database")
            return unique_features
            
        except Exception as e:
            print(f"Error querying database for features: {e}")
            print("Falling back to TEST_FEATS")
            return TEST_FEATS
    
    def _match_pattern(self, pattern, features):
        """
        Match a single pattern against a list of features.
        
        For patterns with wildcards like 'ctrl_*_per_min_*', this applies
        sequential filtering: first finds features containing the first part,
        then filters those to also contain subsequent parts.
        
        Args:
            pattern: Pattern string with wildcards (*) 
            features: List of feature names to match against
            
        Returns:
            List of matching feature names
        """
        # Handle exact matches (no wildcards)
        if '*' not in pattern:
            return [f for f in features if f == pattern]
        
        # Split pattern by * to get individual filter parts
        parts = pattern.split('*')
        
        # Remove empty parts (happens with consecutive * or leading/trailing *)
        parts = [part for part in parts if part]
        
        if not parts:
            # Pattern was just '*' - return all features
            return list(features)
        
        # Apply sequential filtering
        current_features = list(features)
        
        for part in parts:
            if not part:  # Skip empty parts
                continue
                
            # Filter current features to only those containing this part
            current_features = [f for f in current_features if part in f]
            
            # If no features left, break early
            if not current_features:
                break
        
        return current_features
    
    def _filter_by_category(self, category_prefix, features):
        """
        Filter features that start with the category prefix.
        
        Args:
            category_prefix: Prefix to filter by (e.g., 'ctrl_', 'ko_')
            features: List of features to filter
            
        Returns:
            List of features that start with the prefix
        """
        return [f for f in features if f.startswith(category_prefix)]
    
    def select_features(self, pattern_dict):
        """
        Select features based on category patterns.
        
        Args:
            pattern_dict: Dictionary where keys are category prefixes (e.g., 'ctrl_') 
                         and values are lists of patterns to match within that category
                         
        Returns:
            List of unique feature names that match the patterns
        """
        selected_features = []
        
        for category_prefix, patterns in pattern_dict.items():
            # First filter by category
            category_features = self._filter_by_category(category_prefix, self.available_features)
            
            # Then apply each pattern within the category
            for pattern in patterns:
                if pattern.startswith(category_prefix):
                    # Pattern includes the prefix, match directly
                    matches = self._match_pattern(pattern, self.available_features)
                else:
                    # Pattern is relative to category, apply to category features
                    matches = self._match_pattern(pattern, category_features)
                
                selected_features.extend(matches)
        
        # Return unique features, preserving order
        seen = set()
        unique_features = []
        for feature in selected_features:
            if feature not in seen:
                seen.add(feature)
                unique_features.append(feature)
        
        return unique_features
    
    def get_pattern_dict_example(self):
        """
        Returns an example pattern dictionary based on the user's requirements.
        """
        return {
            'age_': [
                'age_dec_avg_diff',
                'age_ratio_diff'
            ],
            'reach_': [
                'reach_ratio_dec_avg_diff'
            ],
            'ufcage_': [
                'ufcage_dec_avg_diff'
            ],
            'days_since_last_fight_': [
                'days_since_last_fight_dec_avg_diff'
            ],
            'sig_str_': [
                'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
                'sig_str_land_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'head_': [
                'head_acc_*',
                'head_def_*',
                'head_land_dec_avg_diff'
            ],
            'body_': [
                'body_acc_*',
                'body_def_*'
            ],
                        'leg_': [
                'leg_rd1_acc_dec_adjperf_opp_dec_avg_diff',
                'leg_land_rd1_ratio_dec_adjperf_opp_dec_avg_diff',
                'leg_acc_dec_adjperf_opp_dec_avg_diff',
                'leg_att_rd1_per_min_dec_adjperf_opp_dec_avg_diff',
                'leg_att_rd1_dec_adjperf_opp_dec_avg_diff',
                'leg_land_rd1_dec_adjperf_opp_dec_avg_diff',
                'leg_land_rd1_ratio_dec_adjperf_dec_avg_diff',
                'leg_att_rd1_per_min_dec_adjperf_dec_avg_diff',
                'leg_def_dec_adjperf_dec_avg_diff',
                'leg_att_per_min_dec_adjperf_dec_avg_diff',
                'leg_rd1_def_dec_adjperf_dec_avg_diff',
                'leg_att_total_ratio_dec_adjperf_dec_avg_diff',
                'leg_rd1_acc_opp_dec_avg_diff',
                'leg_att_per_min_opp_dec_avg_diff',
                'leg_land_per_min_opp_dec_avg_diff',
                'leg_att_rd1_per_min_opp_dec_avg_diff',
                'leg_land_rd1_opp_dec_avg_diff',
                'leg_acc_opp_dec_avg_diff'
            ],
            'distance_': [
                'distance_land_*_dec_adjperf_*',
                'distance_acc_dec_adjperf_dec_avg_diff',
                'distance_acc_dec_adjperf_opp_dec_avg_diff',
                'distance_def_dec_adjperf_dec_avg_diff',
                'distance_def_dec_adjperf_opp_dec_avg_diff',
                'distance_att_rd1_total_dec_adjperf_dec_avg_diff',
                'distance_att_rd1_total_dec_adjperf_opp_dec_avg_diff',
                'distance_att_rd1_ratio_dec_adjperf_dec_avg_diff',
                'distance_att_rd1_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'clinch_': [
                'clinch_att_rd1_*_dec_avg',
                'clinch_*_acc'
            ],
            'ground_': [
                'ground_att_*_dec_adjperf_*',
                'ground_land_dec_adjperf_opp_dec_avg_diff',
                'ground_land_dec_adjperf_dec_avg_diff',
                'ground_rd1_*'
            ],
            'ctrl_': [
                'ctrl_rd1_*',
                'ctrl_*_per_min_*'
            ],
                        'td_': [
                'td_rd1_def_dec_adjperf_dec_avg_diff',
                'td_att_dec_adjperf_dec_avg_diff',
                'td_land_dec_adjperf_dec_avg_diff',
                'td_land_per_min_dec_adjperf_dec_avg_diff',
                'td_def_dec_adjperf_dec_avg_diff',
                'td_acc_dec_adjperf_dec_avg_diff',
                'td_land_rd1_dec_adjperf_dec_avg_diff',
                'td_land_rd1_total_ratio_dec_adjperf_dec_avg_diff',
                'td_att_rd1_total_dec_adjperf_dec_avg_diff',
                'td_att_ratio_dec_adjperf_dec_avg_diff',
                'td_land_rd1_per_min_dec_adjperf_dec_avg_diff',
                'td_att_rd1_per_min_dec_adjperf_dec_avg_diff',
                'td_att_rd1_total_ratio_dec_adjperf_opp_dec_avg_diff',
                'td_land_rd1_ratio_dec_adjperf_opp_dec_avg_diff',
                'td_att_ratio_dec_adjperf_opp_dec_avg_diff',
                'td_land_rd1_total_dec_adjperf_opp_dec_avg_diff',
                'td_att_rd1_total_dec_adjperf_opp_dec_avg_diff',
                'td_rd1_acc_dec_avg_diff',
                'td_att_dec_avg_diff',
                'td_att_rd1_dec_avg_diff',
                'td_att_opp_dec_avg_diff',
                'td_acc_dec_avg_diff',
                'td_acc_opp_dec_avg_diff',
                'td_land_opp_dec_avg_diff',
                'td_land_rd1_per_min_dec_avg_diff'
            ],
            'rev_': [
                'rev_dec_adjperf_opp_dec_avg_diff',
                'rev_dec_adjperf_dec_avg_diff',
                'rev_total_dec_adjperf_opp_dec_avg_diff',
                'rev_ratio_dec_adjperf_dec_avg_diff',
                'rev_total_dec_adjperf_dec_avg_diff',
                'rev_dec_avg_diff',
                'rev_total_dec_avg_diff',
                'rev_per_min_opp_dec_avg_diff',
                'rev_rd1_dec_adjperf_dec_avg_diff'
            ],
                        'sub_': [
                'sub_att_dec_avg_diff',
                'sub_att_rd1_total_opp_dec_avg_diff',
                'sub_land_rd1_total_opp_dec_avg_diff',
                'sub_att_ratio_dec_avg_diff',
                'sub_att_rd1_per_min_opp_dec_avg_diff',
                'sub_land_rd1_opp_dec_avg_diff',
                'sub_land_rd1_per_min_opp_dec_avg_diff',
                'sub_att_opp_dec_avg_diff',
                'sub_def_dec_adjperf_dec_avg_diff',
                'sub_att_per_min_dec_adjperf_dec_avg_diff',
                'sub_att_rd1_dec_adjperf_dec_avg_diff',
                'sub_att_rd1_total_ratio_dec_adjperf_dec_avg_diff',
                'sub_rd1_def_dec_adjperf_dec_avg_diff',
                'sub_land_rd1_dec_adjperf_opp_dec_avg_diff',
                'sub_att_per_min_dec_adjperf_opp_dec_avg_diff',
                'sub_rd1_def_dec_adjperf_opp_dec_avg_diff',
                'sub_att_dec_adjperf_opp_dec_avg_diff',
                'sub_acc_opp_dec_avg_diff'
            ],
            'ko_': [
                'ko_rd1_dec_adjperf_opp_dec_avg_diff',
                'ko_total_dec_adjperf_opp_dec_avg_diff',
                'ko_rd1_total_ratio_dec_adjperf_opp_dec_avg_diff',
                'ko_rd1_total_dec_adjperf_dec_avg_diff',
                'ko_rd1_per_min_dec_adjperf_dec_avg_diff',
                'ko_rd1_dec_adjperf_dec_avg_diff',
                'ko_total_dec_adjperf_dec_avg_diff',
                'ko_per_min_dec_adjperf_dec_avg_diff',
                'ko_opp_dec_avg_diff',
                'ko_total_ratio_dec_avg_diff',
                'ko_rd1_per_min_opp_dec_avg_diff'
            ],
            'decision_': [
                'decision_total_dec_adjperf_opp_dec_avg_diff',
                'decision_total_dec_adjperf_dec_avg_diff',
                'decision_per_min_dec_adjperf_dec_avg_diff',
                'decision_opp_dec_avg_diff',
                'decision_total_ratio_dec_avg_diff'
            ],
            'kd_': [
                'kd_rd1_total_dec_avg_diff',
                'kd_opp_dec_avg_diff',
                'kd_rd1_total_dec_adjperf_dec_avg_diff',
                'kd_rd1_total_ratio_dec_adjperf_dec_avg_diff',
                'kd_rd1_ratio_dec_adjperf_dec_avg_diff',
                'kd_total_dec_adjperf_dec_avg_diff',
                'kd_dec_adjperf_opp_dec_avg_diff',
                'kd_per_min_dec_adjperf_opp_dec_avg_diff',
                'kd_ratio_dec_adjperf_opp_dec_avg_diff'
            ],
            'time_sec_': [
                'time_sec_dec_avg_diff',
                'time_sec_opp_dec_avg_diff',
                'time_sec_rd1_dec_adjperf_dec_avg_diff',
                'time_sec_dec_adjperf_dec_avg_diff',
                'time_sec_total_dec_adjperf_dec_avg_diff',
                'time_sec_rd1_dec_adjperf_opp_dec_avg_diff',
                'time_sec_dec_adjperf_opp_dec_avg_diff'
            ],
            'style_': [
                'style_*'
            ]
        }


# Example usage:
def create_custom_feature_set():
    """
    Example function showing how to use FeatureSelector to create a custom feature set.
    """
    selector = FeatureSelector()
    patterns = selector.get_pattern_dict_example()
    return selector.select_features(patterns)
