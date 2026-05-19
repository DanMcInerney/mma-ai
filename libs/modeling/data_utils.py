import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, FunctionTransformer
import joblib
from libs.feature_store.features import FeatureSelector


def split_data_three_way(X, y, train_size=0.7, val_size=0.15):
    """
    Performs chronological three-way split: train/validation/test.
    
    Args:
        X: Features DataFrame
        y: Target Series  
        train_size: Fraction for training (default 0.7)
        val_size: Fraction for validation (default 0.15)
        
    Returns:
        tuple: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    total_rows = len(X)
    test_size = 1 - train_size - val_size
    
    # Calculate split points
    train_end = int(total_rows * train_size)
    val_end = int(total_rows * (train_size + val_size))
    
    # Split X and y
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    print(f"Three-way split:")
    print(f"Training data:   {len(X_train)} fights ({train_size*100:.0f}%) - {X_train.index.min()} to {X_train.index.max()}")
    print(f"Validation data: {len(X_val)} fights ({val_size*100:.0f}%) - {X_val.index.min()} to {X_val.index.max()}")
    print(f"Test data:       {len(X_test)} fights ({test_size*100:.0f}%) - {X_test.index.min()} to {X_test.index.max()}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def split_data_simple(X, y, train_size=0.9):
    """
    Performs simple chronological split: train/test.
    
    Args:
        X: Features DataFrame
        y: Target Series  
        train_size: Fraction for training (default 0.9)
        
    Returns:
        tuple: ((X_train, y_train), (X_test, y_test))
    """
    total_rows = len(X)
    
    # Calculate split point
    train_end = int(total_rows * train_size)
    
    # Split X and y
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end:]
    y_test = y.iloc[train_end:]
    
    print(f"Simple chronological split:")
    print(f"Training data: {len(X_train)} fights ({train_size*100:.0f}%) - {X_train.index.min()} to {X_train.index.max()}")
    print(f"Test data:     {len(X_test)} fights ({(1-train_size)*100:.0f}%) - {X_test.index.min()} to {X_test.index.max()}")
    
    return (X_train, y_train), (X_test, y_test)


def load_training_data(df, feats: List[str]):
    """Load training data from dataframe."""
    # Select features
    if not feats:
        selector = FeatureSelector(df.columns.tolist())
        patterns = selector.get_pattern_dict_example()
        feature_cols_filtered = selector.select_features(patterns)
    else:
        feature_cols_filtered = [f for f in feats if f != 'y_true']  # Explicitly exclude y_true

    X = df[feature_cols_filtered]
    y = df['y_true']

    print(f"Loaded {len(X)} rows with {len(feature_cols_filtered)} features")

    return X, y


def filter_min_fights(df, min_fights=2):
    """
    Filter fights to include only those where both fighters have at least 2 previous fights.
    Assumes that the DataFrame contains columns 'fighter1_name' and 'fighter2_name',
    and that the fights are ordered chronologically by 'event_date' (and 'fight_id' for tie-breaking).
    """
    # Ensure the DataFrame is sorted chronologically
    df_sorted = df.sort_values(['event_date', 'fight_id']).copy()
    
    # Create a "melted" DataFrame with one row per fighter per fight.
    # Each row records the fight_id, event_date, the fighter's name, and which fighter they are.
    df_f1 = df_sorted[['fight_id', 'event_date', 'fighter1_name']].copy()
    df_f1['role'] = 'fighter1'
    df_f1.rename(columns={'fighter1_name': 'fighter_name'}, inplace=True)
    
    df_f2 = df_sorted[['fight_id', 'event_date', 'fighter2_name']].copy()
    df_f2['role'] = 'fighter2'
    df_f2.rename(columns={'fighter2_name': 'fighter_name'}, inplace=True)
    
    df_melt = pd.concat([df_f1, df_f2], ignore_index=True)
    
    # Sort by fighter name, then chronologically.
    df_melt = df_melt.sort_values(['fighter_name', 'event_date', 'fight_id'])
    
    # Compute the cumulative count for each fighter.
    # For example, if a fighter's first fight gets cumcount = 0, their third fight will get cumcount = 2.
    df_melt['cumcount'] = df_melt.groupby('fighter_name').cumcount()
    
    # Pivot the melted DataFrame back so that each fight_id has the cumulative count for fighter1 and fighter2.
    df_cum = df_melt.pivot(index='fight_id', columns='role', values='cumcount').reset_index()
    df_cum.rename(columns={'fighter1': 'fighter1_cum', 'fighter2': 'fighter2_cum'}, inplace=True)
    
    # Merge the cumulative counts back into the original DataFrame.
    df_merged = pd.merge(df_sorted, df_cum, on='fight_id', how='left')
    
    # Filter: Keep only fights where BOTH fighters have at least 2 previous fights
    filtered_df = df_merged[(df_merged['fighter1_cum'] >= min_fights) & (df_merged['fighter2_cum'] >= min_fights)]

    print(f"Filtered {len(df_merged)} fights to {len(filtered_df)} fights")
    
    return filtered_df


def apply_zscore_normalization(X_train, X_test, path):
    """Apply Z-score normalization to features only."""
    # Remove date columns and categorical static features before scaling
    date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
    
    # Categorical/encoded static features that should NOT be normalized
    # (continuous static features like age, reach, days_since_last_fight, ufcage WILL be normalized)
    categorical_static_feats = ['weightclass_encoded', 'odds']
    
    # Columns that should NEVER be normalized (metadata, labels, weights, encoded categories)
    never_normalize = ['sample_weight', 'y_true', 'weightclass_encoded']
    
    def should_exclude_col(col_name):
        """Check if column should be excluded from scaling."""
        if col_name in date_cols:
            return True
        if col_name in never_normalize:
            return True
        # Exclude if column name contains categorical static feature strings
        for cat_feat in categorical_static_feats:
            if cat_feat in col_name:
                return True
        return False
    
    features_to_scale = [col for col in X_train.columns if not should_exclude_col(col)]

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train[features_to_scale])
    joblib.dump(scaler, path)

    
    # Transform the data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[features_to_scale] = scaler.transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    return X_train_scaled, X_test_scaled


def apply_robust_normalization(X_train, X_test, path):
    """Apply Robust Scaling to features (i.e. use median and IQR)."""
    # Remove date columns and categorical static features before scaling
    date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
    
    # Categorical/encoded static features that should NOT be normalized
    # (continuous static features like age, reach, days_since_last_fight, ufcage WILL be normalized)
    categorical_static_feats = ['weightclass_encoded', 'odds']
    
    # Columns that should NEVER be normalized (metadata, labels, weights, encoded categories)
    never_normalize = ['sample_weight', 'y_true', 'weightclass_encoded']
    
    def should_exclude_col(col_name):
        """Check if column should be excluded from scaling."""
        if col_name in date_cols:
            return True
        if col_name in never_normalize:
            return True
        # Exclude if column name contains categorical static feature strings
        for cat_feat in categorical_static_feats:
            if cat_feat in col_name:
                return True
        return False
    
    features_to_scale = [col for col in X_train.columns if not should_exclude_col(col)]

    scaler = RobustScaler()
    scaler.fit(X_train[features_to_scale])
    # Save the scaler to disk if needed
    joblib.dump(scaler, path)

    # Transform both train and test sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[features_to_scale] = scaler.transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    return X_train_scaled, X_test_scaled


def apply_no_normalization(X_train, X_test, path):
    """Apply no scaling (identity transform) but still create and save a scaler object."""
    
    # Create an identity transformer that does nothing
    scaler = FunctionTransformer(func=None, validate=False)
    
    # Remove date columns and categorical static features for consistency with other scalers
    date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
    
    # Categorical/encoded static features that should NOT be normalized
    # (continuous static features like age, reach, days_since_last_fight, ufcage WILL be normalized)
    categorical_static_feats = ['weightclass_encoded', 'odds']
    
    # Columns that should NEVER be normalized (metadata, labels, weights, encoded categories)
    never_normalize = ['sample_weight', 'y_true', 'weightclass_encoded']
    
    def should_exclude_col(col_name):
        """Check if column should be excluded from scaling."""
        if col_name in date_cols:
            return True
        if col_name in never_normalize:
            return True
        # Exclude if column name contains categorical static feature strings
        for cat_feat in categorical_static_feats:
            if cat_feat in col_name:
                return True
        return False
    
    features_to_scale = [col for col in X_train.columns if not should_exclude_col(col)]
    
    # "Fit" the identity scaler (doesn't actually do anything)
    scaler.fit(X_train[features_to_scale])
    
    # Save the scaler to disk for consistency with other scaling methods
    joblib.dump(scaler, path)
    
    # Return the data unchanged
    return X_train.copy(), X_test.copy()


def balance_dataset(df, target_balance=0.5):
    """Balance the dataset by swapping fighter1/fighter2 roles for some fights.
    
    Args:
        df: DataFrame with fight data including y_true, fighter IDs/names, and _diff columns
        target_balance: Target proportion of fighter1 wins (default 0.5 for 50/50)
        
    Returns:
        DataFrame: Balanced dataset
    """
    print(f"\n=== Balancing Dataset (Target: {target_balance:.1%} Fighter1 wins) ===")
    
    if 'y_true' not in df.columns:
        print("Warning: y_true column not found. Skipping balancing.")
        return df
    
    # Work on a copy to avoid modifying the original
    balanced_df = df.copy()
    
    # Filter to only binary outcomes (0 or 1)
    binary_mask = balanced_df['y_true'].isin([0, 1])
    if not binary_mask.all():
        excluded_count = (~binary_mask).sum()
        print(f"Excluding {excluded_count} fights with non-binary outcomes")
        balanced_df = balanced_df[binary_mask].copy()
    
    # Show current distribution
    current_dist = balanced_df['y_true'].value_counts().sort_index()
    total_fights = len(balanced_df)
    f1_wins = current_dist.get(1, 0)
    f2_wins = current_dist.get(0, 0)
    current_f1_rate = f1_wins / total_fights if total_fights > 0 else 0
    
    print(f"Current distribution:")
    print(f"  Fighter1 wins: {f1_wins} ({current_f1_rate:.1%})")
    print(f"  Fighter2 wins: {f2_wins} ({1-current_f1_rate:.1%})")
    print(f"  Total fights: {total_fights}")
    
    # Calculate how many fights need to be swapped
    target_f1_wins = int(total_fights * target_balance)
    current_f1_wins = f1_wins
    
    if abs(current_f1_rate - target_balance) < 0.01:  # Within 1%
        print("Dataset is already balanced. No swapping needed.")
        return balanced_df
    
    if current_f1_wins > target_f1_wins:
        # Too many F1 wins - need to swap some F1 wins to F2 wins
        swaps_needed = current_f1_wins - target_f1_wins
        swap_from_class = 1  # Swap from fighter1 wins
        print(f"Need to swap {swaps_needed} fighter1 wins to fighter2 wins")
    else:
        # Too many F2 wins - need to swap some F2 wins to F1 wins  
        swaps_needed = target_f1_wins - current_f1_wins
        swap_from_class = 0  # Swap from fighter2 wins
        print(f"Need to swap {swaps_needed} fighter2 wins to fighter1 wins")
    
    # Randomly select fights to swap
    swap_candidates = balanced_df[balanced_df['y_true'] == swap_from_class].index
    if len(swap_candidates) < swaps_needed:
        print(f"Warning: Only {len(swap_candidates)} fights available to swap, but {swaps_needed} needed")
        swaps_needed = len(swap_candidates)
    
    np.random.seed(42)  # For reproducibility
    swap_indices = np.random.choice(swap_candidates, size=swaps_needed, replace=False)
    
    print(f"Randomly selected {len(swap_indices)} fights to swap roles")
    
    # Perform the swaps
    _swap_fighter_roles(balanced_df, swap_indices)
    
    # Show final distribution
    final_dist = balanced_df['y_true'].value_counts().sort_index()
    final_f1_wins = final_dist.get(1, 0)
    final_f2_wins = final_dist.get(0, 0)
    final_f1_rate = final_f1_wins / total_fights if total_fights > 0 else 0
    
    print(f"\nFinal distribution after balancing:")
    print(f"  Fighter1 wins: {final_f1_wins} ({final_f1_rate:.1%})")
    print(f"  Fighter2 wins: {final_f2_wins} ({1-final_f1_rate:.1%})")
    print(f"  Difference from target: {abs(final_f1_rate - target_balance):.1%}")
    
    return balanced_df


def _swap_fighter_roles(df, indices):
    """Swap fighter1 and fighter2 roles for the specified fight indices.
    
    Args:
        df: DataFrame to modify in place
        indices: Array-like of DataFrame indices where roles should be swapped
    """
    print("Swapping fighter roles...")
    
    # 1. Swap fighter IDs and names
    if 'fighter1_id' in df.columns and 'fighter2_id' in df.columns:
        df.loc[indices, ['fighter1_id', 'fighter2_id']] = \
            df.loc[indices, ['fighter2_id', 'fighter1_id']].values
    
    if 'fighter1_name' in df.columns and 'fighter2_name' in df.columns:
        df.loc[indices, ['fighter1_name', 'fighter2_name']] = \
            df.loc[indices, ['fighter2_name', 'fighter1_name']].values
    
    # 2. Swap odds columns to maintain proper alignment
    odds_column_pairs = [
        ('f1_ip_closing_odds', 'f2_ip_closing_odds'),
        ('f1_ip_opening_odds', 'f2_ip_opening_odds'),
        ('f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds'),
        ('f1_sevenday_vigless_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds'),
        # Add any other fighter1/fighter2 paired columns here
    ]
    
    odds_swapped = 0
    for f1_col, f2_col in odds_column_pairs:
        if f1_col in df.columns and f2_col in df.columns:
            df.loc[indices, [f1_col, f2_col]] = \
                df.loc[indices, [f2_col, f1_col]].values
            odds_swapped += 1
    
    # 3. Flip y_true (1->0, 0->1)
    df.loc[indices, 'y_true'] = 1 - df.loc[indices, 'y_true']
    
    # 4. Flip the sign of all _diff columns (since they represent fighter1 - fighter2)
    diff_columns = [col for col in df.columns if col.endswith('_diff')]
    for col in diff_columns:
        df.loc[indices, col] = -df.loc[indices, col]
    
    print(f"Swapped roles for {len(indices)} fights")
    print(f"Swapped {odds_swapped} pairs of odds columns")
    print(f"Flipped {len(diff_columns)} _diff columns")


def calculate_recency_weights(df, indices, decay_rate=0.1):
    """
    Calculate sample weights based on fight recency using exponential decay.
    
    Parameters:
    -----------
    df : DataFrame
        Original dataframe with event_date column
    indices : Index
        Indices of the samples to calculate weights for
    decay_rate : float
        Decay rate for exponential weighting. Higher values = more emphasis on recent fights
        0.1 = ~10% less weight per year
        
    Returns:
    --------
    numpy array of sample weights
    """
    # Get the dates for the specified indices
    fight_dates = pd.to_datetime(df.loc[indices, 'event_date'])
    
    # Calculate days from most recent fight
    max_date = fight_dates.max()
    days_ago = (max_date - fight_dates).dt.days
    
    # Convert to years and apply exponential decay
    years_ago = days_ago / 365.25
    weights = np.exp(-decay_rate * years_ago)
    
    # Normalize weights to sum to the number of rows (as recommended by AutoGluon)
    weights = weights * len(weights) / weights.sum()
    
    print(f"\nRecency weights applied:")
    print(f"Oldest fight weight: {weights.min():.3f}")
    print(f"Newest fight weight: {weights.max():.3f}")
    print(f"Weight ratio (newest/oldest): {weights.max()/weights.min():.2f}x")
    print(f"Sum of weights: {weights.sum():.1f} (should equal {len(weights)})")
    
    return weights.values


def filter_fights(df, threshold, date='2014-01-01', include_split_dec=False, data_cutoff=None):
    """
    Filter fights based on:
      - Binary results (y_true in [0, 1])
      - Both fighters must have had at least num_fights previous fights
          (i.e. their current fight number is at least num_fights + 1)
      - Removing unwanted fight methods
      - Fights from specified start date onward
      - Optionally, fights up to specified cutoff date
      
    Parameters:
      df : DataFrame
          A DataFrame where each row represents a fight. It must contain:
            'fight_id', 'event_date', 'fighter1_id', 'fighter2_id',
            'y_true', and 'method'
      threshold : int
          Minimum number of previous fights required for each fighter.
          (For 2 previous fights, the fighter's current fight number should be ≥ 3.)
      date : str
          Start date for filtering fights (YYYY-MM-DD format)
      include_split_dec : bool
          Whether to include split decisions
      data_cutoff : str, optional
          End date for filtering fights in YYYY-MM-DD format. If None, no cutoff is applied.
    
    Returns:
      Filtered DataFrame.
    """
    print("\n=== Filtering Fights ===")
    orig_df = df.copy()
    
    # --- Step 1. Compute overall fight counts for each fighter ---
    # Build a "long" DataFrame with one row per fighter per fight.
    df_f1 = df[['fight_id', 'event_date', 'fighter1_id']].rename(columns={'fighter1_id': 'fighter_id'})
    df_f2 = df[['fight_id', 'event_date', 'fighter2_id']].rename(columns={'fighter2_id': 'fighter_id'})
    df_long = pd.concat([df_f1, df_f2], ignore_index=True)
    
    # Ensure event_date is datetime and sort for proper ordering.
    df_long['event_date'] = pd.to_datetime(df_long['event_date'])
    df_long = df_long.sort_values(['fighter_id', 'event_date', 'fight_id'])
    
    # Compute each fighter's fight number (first fight will be 1, second 2, etc.)
    df_long['fight_num'] = df_long.groupby('fighter_id').cumcount() + 1
    
    # --- Step 2. Merge fight numbers back onto the original DataFrame ---
    # For fighter1:
    df_long_f1 = df_long.rename(columns={'fighter_id': 'fighter1_id', 'fight_num': 'fighter1_fight_num'})
    df = pd.merge(df, 
                  df_long_f1[['fight_id', 'fighter1_id', 'fighter1_fight_num']],
                  on=['fight_id', 'fighter1_id'],
                  how='left')
    
    # For fighter2:
    df_long_f2 = df_long.rename(columns={'fighter_id': 'fighter2_id', 'fight_num': 'fighter2_fight_num'})
    df = pd.merge(df, 
                  df_long_f2[['fight_id', 'fighter2_id', 'fighter2_fight_num']],
                  on=['fight_id', 'fighter2_id'],
                  how='left')
    
    # --- Step 3. Filter out fights where either fighter has insufficient experience ---
    # If a fighter must have had at least num_fights previous fights, then the current fight
    # number must be at least (num_fights + 1). For example, for 2 previous fights, current fight ≥ 3.
    before_exp = len(df)

    df = df[(df['fighter1_fight_num'] > threshold) & (df['fighter2_fight_num'] > threshold)]
    after_exp = len(df)
    print(f"Filtered out {before_exp - after_exp} rows due to insufficient previous fights (need current fight number > {threshold})")
    
    # --- Step 4. Remove fights with unwanted methods ---
    original_len = len(df)
    #unwanted_methods = ['dq', 'decision - split', 'decision - majority', 'other', 'overturned']
    if include_split_dec:
        unwanted_methods = ['dq', 'other', 'overturned']
    else:
        unwanted_methods = ['dq', 'other', 'decision - split', 'decision - majority', 'overturned']

    # Use the string accessor (.str.lower()) to lowercase each value
    df['method_lower'] = df['method'].str.lower()

    method_mask = ~df['method_lower'].str.contains('|'.join(unwanted_methods), na=False)
    df = df[method_mask].copy()
    print(f"Removed {original_len - len(df)} rows with unwanted methods")
    df.drop(columns=['method_lower'], inplace=True)

    # --- Step 5. Filter to only binary results ---
    original_len = len(df)
    df = df[df['y_true'].isin([0, 1])].copy()
    print(f"Filtered out {original_len - len(df)} rows with non-binary results")
    

    # --- Step 6. Filter by event date (start date and optional cutoff date) ---
    df['event_date'] = pd.to_datetime(df['event_date'])
    initial_date_count = len(df)
    
    # Filter by start date
    df = df[df['event_date'] >= pd.Timestamp(date)]
    after_start_filter = len(df)
    print(f"Filtered out {initial_date_count - after_start_filter} rows before start date {date}")
    
    # Filter by cutoff date if provided
    if data_cutoff is not None:
        cutoff_timestamp = pd.Timestamp(data_cutoff)
        before_cutoff_filter = len(df)
        df = df[df['event_date'] <= cutoff_timestamp]
        after_cutoff_filter = len(df)
        print(f"Filtered out {before_cutoff_filter - after_cutoff_filter} rows after cutoff date {data_cutoff}")
    
    print(f"Final number of rows after date filtering: {len(df)}")

    # --- Step 7. Filer NaN values ---
    original_len = len(df)

    df = df.dropna()
    print(f"Filtered out {original_len - len(df)} rows with NaN values")

    # Reset the index
    df.sort_values(by=['event_date', 'fight_id'], inplace=True)
    df = df.reset_index(drop=True)
    
    return df
