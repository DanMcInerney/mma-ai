"""Feature filtering utility for selecting model-specific features."""

from typing import List
import pandas as pd


def filter_features_for_model(df: pd.DataFrame, required_features: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame columns to match required_features list.
    
    Handles feature name patterns:
    - 'fighter1_<stat>' -> selects fighter1_<stat> column
    - 'fighter2_<stat>' -> selects fighter2_<stat> column
    - '<stat>_diff' -> selects <stat>_diff column
    - '<stat>' -> selects fighter1_<stat> column (for backward compatibility)
    
    Args:
        df: DataFrame with all features (fighter1_*, fighter2_*, *_diff)
        required_features: List of required feature names
        
    Returns:
        DataFrame with only the required features
        
    Example:
        >>> df = pd.DataFrame({
        ...     'fighter1_time_sec_rd1_avg': [100],
        ...     'fighter2_time_sec_rd1_avg': [90],
        ...     'time_sec_rd1_avg_diff': [10]
        ... })
        >>> features = ['fighter1_time_sec_rd1_avg', 'time_sec_rd1_avg_diff']
        >>> filtered = filter_features_for_model(df, features)
    """
    if df.empty:
        return df
    
    available_cols = set(df.columns)
    selected_cols = []
    missing_features = []
    
    for feat in required_features:
        # Exact match
        if feat in available_cols:
            selected_cols.append(feat)
        # Handle _diff features - check if they exist
        elif feat.endswith('_diff'):
            if feat in available_cols:
                selected_cols.append(feat)
            else:
                missing_features.append(feat)
        # Handle backward compatibility: 'stat' -> 'fighter1_stat'
        elif not feat.startswith('fighter1_') and not feat.startswith('fighter2_'):
            # Try fighter1_ prefix for backward compatibility
            fighter1_feat = f'fighter1_{feat}'
            if fighter1_feat in available_cols:
                selected_cols.append(fighter1_feat)
            else:
                missing_features.append(feat)
        else:
            missing_features.append(feat)
    
    if missing_features:
        print(f"Warning: {len(missing_features)} required features not found in DataFrame:")
        for feat in missing_features[:10]:  # Show first 10
            print(f"  - {feat}")
        if len(missing_features) > 10:
            print(f"  ... and {len(missing_features) - 10} more")
    
    if not selected_cols:
        print("Warning: No features selected. Returning empty DataFrame.")
        return pd.DataFrame()
    
    # Ensure we maintain the same index
    filtered_df = df[selected_cols].copy()
    
    print(f"Filtered to {len(selected_cols)} features from {len(required_features)} required")
    
    return filtered_df

