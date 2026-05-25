import pandas as pd
from datetime import datetime
import numpy as np
from typing import Set, List
import logging
from libs.feature_store.features import BASE_STATIC_FEATS

STATIC_STATS = BASE_STATIC_FEATS


class CleanTrainingData:
    def __init__(self, df: pd.DataFrame, include_patterns: Set[str], exclude_patterns: Set[str] = None,
                 target_type: str = 'win_loss'):
        """Initialize the CleanTrainingData class.
        
        Args:
            df: DataFrame containing training data with all stats
            include_patterns: Set of patterns to include in column selection
            exclude_patterns: Set of patterns to exclude from column selection (optional)
            target_type: Type of target variable - 'win_loss' (default) or 'decision'
        """
        self.training_df = df.copy() if df is not None else None
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns or set()
        self.target_type = target_type
        self.logger = logging.getLogger(__name__)
        
    def load_data(self):
        """Prepare data by separating static and non-static stats."""
        print("\n=== Loading Training Data ===")
        
        if self.training_df is None:
            raise ValueError("No training DataFrame provided")
            
        # Copy the input DataFrame to avoid modifying it
        combined_df = self.training_df.copy()
        
        # Ensure we have essential columns
        required_columns = ['fight_id', 'fighter_id', 'event_id']
        for col in required_columns:
            if col not in combined_df.columns:
                raise ValueError(f"Required column '{col}' not found in training DataFrame")
        
        # Identify static columns (age, reach, odds, etc.)
        static_columns = []
        for col in combined_df.columns:
            if col in required_columns:
                continue
            if any(static in col for static in BASE_STATIC_FEATS):
                static_columns.append(col)
        
        print(f"Identified {len(static_columns)} static columns:")
        print(static_columns[:5], '...' if len(static_columns) > 5 else '')
        
        # Create DataFrames from the combined one
        self.stats_df = combined_df.copy()
        self.static_df = combined_df[required_columns + static_columns].copy()
        
        # Remove static columns from stats_df to avoid duplication
        self.stats_df = self.stats_df.drop(columns=static_columns, errors='ignore')
        
        # Drop duplicates
        print(f"Dropping duplicates from stats_df")
        original_len = len(self.stats_df)
        self.stats_df = self.stats_df.drop_duplicates(subset=['fighter_id', 'event_id', 'fight_id'])
        print(f"Dropped {original_len - len(self.stats_df)} rows")

        print(f"Dropping duplicates from static_df")
        original_len = len(self.static_df)
        self.static_df = self.static_df.drop_duplicates(subset=['fighter_id', 'event_id', 'fight_id'])
        print(f"Dropped {original_len - len(self.static_df)} rows")

        print(f"Loaded {len(self.stats_df)} rows from stats data")
        print(f"Number of unique fighters: {self.stats_df['fighter_id'].nunique()}")
        
        print(f"Date range: {self.stats_df['event_date'].min()} to {self.stats_df['event_date'].max()}")
            
        print(f"Number of columns in stats_df: {len(self.stats_df.columns)}")
        print(f"Number of columns in static_df: {len(self.static_df.columns)}")
        
    def shift_fighter_stats(self):
        """Shift each fighter's non-static stats to use values from their previous fight."""
        print("\n=== Shifting Fighter Stats ===")
        
        print("Sorting by fighter_id and event_date for proper chronological ordering")
        self.stats_df = self.stats_df.sort_values(['fighter_id', 'event_date', 'fight_id'])
        
        # Get list of stat columns to shift (excluding IDs, event_date, and static stats)
        # CRITICAL: Only static features (age, reach, height, days_since_last_fight, ufcage, odds, weightclass_encoded)
        # are excluded from shifting. All dynamic features (decision, win, ko, kd, sub, time_sec, etc.) MUST be shifted.
        non_shift_columns = ['fight_id', 'fighter_id', 'fighter1_id', 'fighter2_id', 'event_id', 'event_date', 'method', 'result', 'fighter_name', 'fighter_dob', 'weightclass', 'weightclass_encoded']
        for static in BASE_STATIC_FEATS:
            non_shift_columns.extend([col for col in self.stats_df.columns if static in col])
        
        stat_columns = [col for col in self.stats_df.columns 
                       if col not in non_shift_columns]
        
        print(f"Number of columns to shift: {len(stat_columns)}")
        print("Sample columns being shifted:")
        print(stat_columns[:5])
        
        # Create a dictionary to store all shifted columns
        shifted_stats = {}
        for col in stat_columns:
            shifted_stats[f"{col}_prev"] = self.stats_df.groupby('fighter_id')[col].shift(1)
        
        # Get only required identifier columns for the base DataFrame
        id_columns = ['fight_id', 'fighter_id', 'fighter1_id', 'fighter2_id', 'event_id', 'event_date', 'method', 'result', 'fighter_name', 'fighter_dob']
        base_df = self.stats_df[[col for col in id_columns if col in self.stats_df.columns]]
        
        # Combine base DataFrame with all shifted stats at once
        self.shifted_df = pd.concat([
            base_df,
            pd.DataFrame(shifted_stats, index=self.stats_df.index)
        ], axis=1)
        
        print(f"\nShifted DataFrame shape: {self.shifted_df.shape}")
        print(f"NaN values (expected for first fights): {self.shifted_df.isna().sum().sum()}")

        
    def calculate_stat_differences(self):
        """Calculate differences between fighter1 and fighter2 stats."""
        print("\n=== Calculating Stat Differences ===")
        
        # Extract unique fights with fighter1_id and fighter2_id
        id_columns = ['fight_id', 'event_id', 'fighter1_id', 'fighter2_id', 'result', 'method']
        if 'event_date' in self.training_df.columns:
            id_columns.append('event_date')
        if 'weightclass' in self.training_df.columns:
            id_columns.append('weightclass')
        if 'weightclass_encoded' in self.training_df.columns:
            id_columns.append('weightclass_encoded')
        
        # Extract unique fight records - one row per fight
        fight_mapping = self.training_df[id_columns].drop_duplicates(subset=['fight_id'])
        print(f"Number of fights in mapping: {len(fight_mapping)}")
        
        # Create fighter name mapping from the training data if available
        if 'fighter_name' in self.training_df.columns and 'fighter_id' in self.training_df.columns:
            fighter_name_mapping = self.training_df[['fighter_id', 'fighter_name']].drop_duplicates()
            
            # Join fighter1_name
            fight_mapping = pd.merge(
                fight_mapping,
                fighter_name_mapping,
                left_on='fighter1_id',
                right_on='fighter_id',
                how='left'
            )
            
            # Rename fighter_name to fighter1_name and drop the redundant fighter_id
            fight_mapping = fight_mapping.rename(columns={'fighter_name': 'fighter1_name'})
            fight_mapping = fight_mapping.drop('fighter_id', axis=1)
            
            # Join fighter2_name
            fight_mapping = pd.merge(
                fight_mapping,
                fighter_name_mapping,
                left_on='fighter2_id',
                right_on='fighter_id',
                how='left'
            )
            
            # Rename fighter_name to fighter2_name and drop the redundant fighter_id
            fight_mapping = fight_mapping.rename(columns={'fighter_name': 'fighter2_name'})
            fight_mapping = fight_mapping.drop('fighter_id', axis=1)
            
            print("Fighter names added to fight mapping")
        
        print(f"Number of rows in shifted_df: {len(self.shifted_df)}")
        print(f"Number of rows in static_df: {len(self.static_df)}")
        
        # Combine static and performance stats for each fighter
        fighter_stats = pd.merge(
            self.shifted_df,
            self.static_df,
            on=['fighter_id', 'event_id', 'fight_id'],
            how='left'
        )
        
        # Join fighter1's stats - explicitly list columns that should not be suffixed
        join_keys_left = ['fighter1_id', 'fight_id']
        join_keys_right = ['fighter_id', 'fight_id']
        if 'event_id' in fight_mapping.columns:
            join_keys_left.append('event_id')
            join_keys_right.append('event_id')
        
        # Get metadata columns from fight_mapping to preserve
        fight_mapping_cols = list(fight_mapping.columns)
        
        # Create a list of columns that should be suffixed in fighter_stats
        stat_cols = [col for col in fighter_stats.columns 
                    if col not in join_keys_right and col != 'fighter_id']
        
        # Do the merge with controlled suffixes - only suffix stat columns, not metadata
        # First drop any conflicting columns from fighter_stats
        fighter_stats_f1 = fighter_stats.copy()
        for col in fight_mapping_cols:
            if col in fighter_stats_f1.columns and col not in join_keys_right:
                fighter_stats_f1 = fighter_stats_f1.drop(col, axis=1)
        
        fight_pairs = pd.merge(
            fight_mapping,
            fighter_stats_f1,
            left_on=join_keys_left,
            right_on=join_keys_right,
            how='inner'
        )
        
        # Rename all stat columns to have _1 suffix
        for col in stat_cols:
            if col in fight_pairs.columns and col not in fight_mapping_cols:
                fight_pairs = fight_pairs.rename(columns={col: f"{col}_1"})
        
        print(f"After merging fighter1 stats: {len(fight_pairs)}")
        
        # Remove redundant fighter_id column to avoid confusion
        if 'fighter_id' in fight_pairs.columns:
            fight_pairs = fight_pairs.drop('fighter_id', axis=1)
        
        # Join fighter2's stats with same approach
        join_keys_left = ['fighter2_id', 'fight_id']
        join_keys_right = ['fighter_id', 'fight_id']
        if 'event_id' in fight_mapping.columns:
            join_keys_left.append('event_id')
            join_keys_right.append('event_id')
        
        # Do the same for fighter2 merge - prepare a clean dataframe
        fighter_stats_f2 = fighter_stats.copy()
        
        # First drop any columns that might conflict with fight_pairs
        for col in fight_pairs.columns:
            if col in fighter_stats_f2.columns and col not in join_keys_right:
                fighter_stats_f2 = fighter_stats_f2.drop(col, axis=1)
        
        # Then do the merge
        fight_pairs = pd.merge(
            fight_pairs,
            fighter_stats_f2,
            left_on=join_keys_left,
            right_on=join_keys_right,
            suffixes=('', '_2'),  # Empty first suffix to avoid changing existing columns
            how='inner'
        )
        
        print(f"After merging fighter2 stats: {len(fight_pairs)}")
        
        # Now rename any remaining non-suffixed stat columns to have _2 suffix
        for col in stat_cols:
            if col in fight_pairs.columns and not col.endswith('_1') and not col.endswith('_2') and col not in fight_mapping_cols:
                fight_pairs = fight_pairs.rename(columns={col: f"{col}_2"})
        
        # Identify columns to diff, excluding metadata
        metadata_columns = [
            'fighter_id', 'fight_id', 'event_id', 'fighter_name',
            'fighter1_id', 'fighter2_id', 'fighter1_name', 'fighter2_name',
            'event_date', 'method', 'result', 'weightclass', 'weightclass_encoded'
        ]
        
        # Get clean lists of columns with _1 and _2 suffixes
        fighter1_cols = [col for col in fight_pairs.columns if col.endswith('_1')]
        fighter2_cols = [col for col in fight_pairs.columns if col.endswith('_2')]

        # Separate static and non-static columns
        static_fighter1_cols = [col for col in fighter1_cols if any(static in col for static in BASE_STATIC_FEATS)]
        static_fighter2_cols = [col for col in fighter2_cols if any(static in col for static in BASE_STATIC_FEATS)]

        # CRITICAL FILTER: Only use shifted (_prev) columns for non-static features
        # This ensures we're using historical data from previous fights, not current fight data
        # Static features (age, reach, etc.) don't need shifting as they're pre-fight attributes
        fighter1_cols = [col for col in fighter1_cols if '_prev' in col or col in static_fighter1_cols]
        fighter2_cols = [col for col in fighter2_cols if '_prev' in col or col in static_fighter2_cols]

        print(f"After filtering:")
        print(f"Number of fighter1 static columns: {len(static_fighter1_cols)}")
        print(f"Number of fighter1 stat columns: {len(fighter1_cols)}")
        print(f"Number of fighter2 stat columns: {len(fighter2_cols)}")
        
        
        # Create base DataFrame with only the required metadata columns
        base_columns = {}
        
        # Track which result column belongs to fighter1
        fighter1_result_col = None
        for col in fight_pairs.columns:
            if col == 'result' or col == 'result_x':
                fighter1_result_col = col
                break
        
        if fighter1_result_col is None:
            print("WARNING: Could not find fighter1's result column. Using first result column found.")
            for col in fight_pairs.columns:
                if 'result' in col and not col.endswith('_2'):
                    fighter1_result_col = col
                    break
        
        # Add required metadata columns, explicitly handling result
        for col in ['fight_id', 'event_id', 'fighter1_id', 'fighter2_id', 'method', 'event_date',
                   'fighter1_name', 'fighter2_name']:
            if col in fight_pairs.columns:
                base_columns[col] = fight_pairs[col]
        
        # Create target variable based on target_type
        if self.target_type == 'decision':
            # Create binary target: 1 if method contains "Decision", 0 otherwise
            if 'method' in fight_pairs.columns:
                base_columns['y_true'] = fight_pairs['method'].str.contains('Decision', case=False, na=False).astype(int)
                print("Using 'method' column to create decision/no decision target (y_true)")
            else:
                print("ERROR: 'method' column not found for decision target!")
                base_columns['y_true'] = None
            
        else:  # default to 'win_loss'
            # Use fighter1's result for win/loss target
            if fighter1_result_col is not None and fighter1_result_col in fight_pairs.columns:
                base_columns['y_true'] = fight_pairs[fighter1_result_col]
                print(f"Using {fighter1_result_col} as the source for win/loss target (y_true)")
            else:
                print("ERROR: Could not find fighter1's result column for y_true!")
                base_columns['y_true'] = None
        
        # Add weightclass_encoded for both target types (now created in feature pipeline)
        if 'weightclass_encoded' in fight_pairs.columns:
            base_columns['weightclass_encoded'] = fight_pairs['weightclass_encoded']
            print("Added weightclass_encoded to training data")
        else:
            print("WARNING: 'weightclass_encoded' column not found in fight_pairs")
        
        # Create the base DataFrame
        self.final_df = pd.DataFrame(base_columns)
        
        # Create both fighter1 absolute stats, fighter2 absolute stats, and diff stats
        fighter1_absolute_stats, fighter2_absolute_stats, diff_stats = self._create_feature_columns(
            fight_pairs, fighter1_cols, fighter2_cols, metadata_columns
        )
        
        # Combine all feature columns
        # For decision target type: include fighter1_<stat>, fighter2_<stat>, and <stat>_diff
        # For win_loss target type: include only <stat> and <stat>_diff (no fighter2 absolute stats)
        # Note: All stats are shifted (using _prev columns) except static stats (age, reach, etc.)
        all_feature_columns = {**fighter1_absolute_stats, **diff_stats}
        if self.target_type == 'decision':
            all_feature_columns.update(fighter2_absolute_stats)
            print(f"Added {len(fighter1_absolute_stats)} fighter1 features (fighter1_*), {len(fighter2_absolute_stats)} fighter2 features (fighter2_*), and {len(diff_stats)} diff features (*_diff)")
            print(f"Decision training data format: fighter1_<stat>, fighter2_<stat>, <stat>_diff")
        else:
            print(f"Added {len(fighter1_absolute_stats)} fighter1 absolute features and {len(diff_stats)} diff features")
        
        # Add feature columns to final DataFrame
        if all_feature_columns:
            self.final_df = pd.concat([
                self.final_df,
                pd.DataFrame(all_feature_columns)
            ], axis=1)

        print(f"\nFinal DataFrame shape: {self.final_df.shape}")
        metadata_cols_for_count = ['fight_id', 'event_id', 'fighter1_id', 'fighter2_id', 
                                   'method', 'event_date', 'fighter1_name', 'fighter2_name', 
                                   'y_true', 'weightclass', 'weightclass_encoded']
        
        if self.target_type == 'decision':
            # For decision: fighter1_<stat>, fighter2_<stat>, <stat>_diff
            fighter1_abs_cols = [c for c in self.final_df.columns 
                                if c.startswith('fighter1_') and c not in metadata_cols_for_count]
            fighter2_abs_cols = [c for c in self.final_df.columns 
                                if c.startswith('fighter2_') and c not in metadata_cols_for_count]
            diff_cols = [c for c in self.final_df.columns if c.endswith('_diff')]
            print(f"Number of fighter1 features (fighter1_*): {len(fighter1_abs_cols)}")
            print(f"Number of fighter2 features (fighter2_*): {len(fighter2_abs_cols)}")
            print(f"Number of difference features (*_diff): {len(diff_cols)}")
        else:
            # For win_loss: <stat>, <stat>_diff (no fighter2 absolute stats)
            fighter1_abs_cols = [c for c in self.final_df.columns 
                                if not c.endswith('_diff') and not c.endswith('_2') 
                                and c not in metadata_columns and c not in metadata_cols_for_count]
            diff_cols = [c for c in self.final_df.columns if c.endswith('_diff')]
            print(f"Number of fighter1 absolute features: {len(fighter1_abs_cols)}")
            print(f"Number of difference features: {len(diff_cols)}")
    
    def _create_feature_columns(self, fight_pairs: pd.DataFrame, fighter1_cols: List[str], 
                                fighter2_cols: List[str], metadata_columns: List[str]) -> tuple:
        """
        Create both fighter1 absolute stats, fighter2 absolute stats, and diff stats.
        
        Args:
            fight_pairs: DataFrame with fighter1 and fighter2 stats (with _1 and _2 suffixes)
            fighter1_cols: List of fighter1 column names (ending with _1)
            fighter2_cols: List of fighter2 column names (ending with _2)
            metadata_columns: List of metadata columns to exclude from feature creation
            
        Returns:
            Tuple of (fighter1_absolute_stats_dict, fighter2_absolute_stats_dict, diff_stats_dict)
        """
        fighter1_absolute_stats = {}
        fighter2_absolute_stats = {}
        diff_stats = {}
        excluded_count = 0
        
        for col1 in fighter1_cols:
            # Remove the _1 suffix to get the base column name
            # Example: "decision_dec_avg_prev_1" -> "decision_dec_avg_prev"
            base_col = col1[:-2]
            # Remove the _prev suffix to get the clean feature name
            # CRITICAL: The _prev suffix is removed from the NAME, but the VALUES are still from previous fights
            # because the column was shifted in shift_fighter_stats(). This ensures no data leakage.
            # Example: "decision_dec_avg_prev" -> "decision_dec_avg" (but values are from previous fight)
            clean_col = base_col.replace('_prev', '')
            
            # Skip metadata columns
            if clean_col in metadata_columns:
                excluded_count += 1
                continue
            
            # Find corresponding fighter2 column
            col2 = f"{base_col}_2"
            
            if col2 in fighter2_cols:
                try:
                    # For decision target type, use fighter1_ and fighter2_ prefixes
                    # For win_loss target type, use clean name for fighter1 and _2 suffix for fighter2
                    if self.target_type == 'decision':
                        fighter1_name = f"fighter1_{clean_col}"
                        fighter2_name = f"fighter2_{clean_col}"
                    else:
                        fighter1_name = clean_col
                        fighter2_name = f"{clean_col}_2"
                    
                    # Add fighter1 absolute stat
                    fighter1_absolute_stats[fighter1_name] = fight_pairs[col1]
                    
                    # Add fighter2 absolute stat
                    fighter2_absolute_stats[fighter2_name] = fight_pairs[col2]
                    
                    # Add diff stat (fighter1 - fighter2)
                    diff_stats[f"{clean_col}_diff"] = fight_pairs[col1] - fight_pairs[col2]
                    
                except TypeError as te:
                    # Log type errors specifically, often due to non-numeric data
                    self.logger.warning(
                        f"TypeError calculating features for {clean_col} "
                        f"(Columns: {col1}, {col2}): {te} - skipping"
                    )
                    excluded_count += 1
                except Exception as e:
                    self.logger.error(
                        f"Error calculating features for {clean_col} "
                        f"(Columns: {col1}, {col2}): {e}"
                    )
                    excluded_count += 1
            else:
                # Fighter1 column exists but no matching fighter2 column
                # Still add fighter1 absolute stat
                try:
                    if self.target_type == 'decision':
                        fighter1_name = f"fighter1_{clean_col}"
                    else:
                        fighter1_name = clean_col
                    fighter1_absolute_stats[fighter1_name] = fight_pairs[col1]
                except Exception as e:
                    self.logger.warning(f"Could not add fighter1 stat for {clean_col}: {e}")
        
        if excluded_count > 0:
            print(f"Excluded {excluded_count} columns from feature creation")
        
        return fighter1_absolute_stats, fighter2_absolute_stats, diff_stats
    
    def remove_correlated_features(self, df, print_corr=False):
        """Remove features that are highly correlated (>= 0.95) with other features."""
        print("\n=== Removing Highly Correlated Features ===")

        if any(col.endswith('_prev') for col in df.columns):
            static = False
        else:
            static = True
        
        # Get list of feature columns (those ending with _diff)
        remove = ['result', 'event_date', 'event_id', 'fighter_id', 'result_count_prev', 'result_count', 'method', 
                 'fight_id', 'fighter1_id', 'fighter2_id', 'y_true']

        feature_cols = [col for col in df.columns if col not in remove]
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Create upper triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find pairs of highly correlated features
        to_drop = set()
        for col in upper.columns:
            # Get features highly correlated with this column
            correlated = upper[col][upper[col] >= 0.95]
            
            for other_col, correlation in correlated.items():
                if print_corr:
                    print(f"\nFound correlation of {correlation:.3f} between:")
                    print(f"  1. {col}")
                    print(f"  2. {other_col}")
                    
                # Rule 1: Check for include_patterns preference
                if any(s in col for s in self.include_patterns) and not any(s in other_col for s in self.include_patterns):
                    to_drop.add(other_col)
                    if print_corr:
                        print(f"  → Keeping {col} (matches include_patterns)")
                    continue
                elif any(s in other_col for s in self.include_patterns) and not any(s in col for s in self.include_patterns):
                    to_drop.add(col)
                    if print_corr:
                        print(f"  → Keeping {other_col} (matches include_patterns)")
                    continue
                    
                # Rule 2: rd1 preference (keep non-rd1)
                if "rd1" in col and "rd1" not in other_col:
                    to_drop.add(col)
                    if print_corr:
                        print(f"  → Keeping {other_col} (no rd1)")
                    continue
                elif "rd1" in other_col and "rd1" not in col:
                    to_drop.add(other_col)
                    if print_corr:
                        print(f"  → Keeping {col} (no rd1)")
                    continue
                    
                # Rule 3: slope preference (keep non-slope)
                if "slope" in col and "slope" not in other_col:
                    to_drop.add(col)
                    if print_corr:
                        print(f"  → Keeping {other_col} (no slope)")
                    continue
                elif "slope" in other_col and "slope" not in col:
                    to_drop.add(other_col)
                    if print_corr:
                        print(f"  → Keeping {col} (no slope)")
                    continue
                    
                # If no rules apply, arbitrarily keep the first one
                to_drop.add(other_col)
                if print_corr:
                    print(f"  → Keeping {col} (default choice)")
        
        # Remove the correlated features
        if static:
            self.static_df = df.drop(columns=list(to_drop))
        else:
            self.shifted_df = df.drop(columns=list(to_drop))
    
        print(f"\nFeatures remaining after correlation removal: {len(feature_cols) - len(to_drop)}")
        return df.drop(columns=list(to_drop))

    def clean_training_data(self):
        """Execute the full data cleaning process."""
        print("\n=== Starting Training Data Cleaning Process ===")
        self.load_data()
        self.shift_fighter_stats()
        self.calculate_stat_differences()
        correlations = pd.Series(dtype=float)
        
        self.final_df.sort_values(by=['event_date', 'fight_id'], inplace=True)
        self.final_df.reset_index(drop=True, inplace=True)
        
        # Print summary statistics
        if 'y_true' in self.final_df.columns:
            print("Target distribution:")
            print(self.final_df['y_true'].value_counts(normalize=True).rename('proportion'))
        
        if 'event_date' in self.final_df.columns:
            print("\nDate range of final dataset:")
            print(f"First fight: {self.final_df['event_date'].min()}")
            print(f"Last fight: {self.final_df['event_date'].max()}")

        # Data validation
        print("\nData validation:")
        if 'event_id' in self.final_df.columns:
            print(f"Number of unique events: {self.final_df['event_id'].nunique()}")
        print(f"Number of unique fighter1s: {self.final_df['fighter1_id'].nunique()}")
        print(f"Number of unique fighter2s: {self.final_df['fighter2_id'].nunique()}")

        print("\nSample of last few fights:")
        print(self.final_df.tail())
        
        # Check correlations with target if y_true exists
        corr_check_df = self.final_df.copy()
        corr_check_df['event_date'] = pd.to_datetime(corr_check_df['event_date'])
        
        # Filter to fights from 2014-01-01 to present
        start_date = pd.to_datetime('2014-01-01')
        corr_check_df = corr_check_df[corr_check_df['event_date'] >= start_date]
        
        # Create list of columns to exclude from correlation calculation
        exclude_from_corr = ['fight_id', 'fighter1_id', 'fighter2_id', 'event_id', 
                            'event_date', 'method', 'fighter1_name', 'fighter2_name', 'fighter_dob']
        
        # Drop non-numeric columns to avoid correlation errors
        for col in exclude_from_corr:
            if col in corr_check_df.columns:
                corr_check_df.drop(columns=[col], inplace=True)
        
        # Now calculate correlations with numeric data only
        if 'y_true' in corr_check_df.columns:
            correlations = corr_check_df.corr()['y_true'].sort_values(ascending=False)

            # Print top positive and negative correlations
            print("\nTop 50 Positive Correlations with y_true:")
            print(correlations.head(50))
            print("\nTop 50 Negative Correlations with y_true:")
            print(correlations.tail(50))

        self.correlations = correlations
        return self.final_df
