import os
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from libs.feature_store.features import FEAT_MAD2_AND_STYLES_TEST_FILTERED
from libs.modeling.data_utils import (
    filter_fights, load_training_data, balance_dataset, 
    apply_robust_normalization, apply_zscore_normalization, apply_no_normalization,
    calculate_recency_weights, split_data_three_way, split_data_simple
)
from libs.modeling.split_date_utils import write_test_start_date
from libs.paths import data_file


class DataPreparation:
    """
    Handles all data preparation steps for model training, including:
    - Loading and filtering data
    - Feature selection
    - Data splitting (train/validation/test)
    - Normalization
    - Sample weighting
    - Dataset balancing
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 feats: Optional[List[str]] = None,
                 odds: bool = False,
                 train_size: float = 0.75,
                 val_size: Optional[float] = 0.15,
                 test_size: Optional[float] = 0.1,
                 normalize: str = 'robust',
                 use_recency_weights: bool = True,
                 decay_rate: float = 0.13,
                 balance_fighters: bool = False,
                 target_balance: float = 0.5,
                 start_date: str = '2014-01-01',
                 num_fights: int = 2,
                 include_split_dec: bool = False,
                 data_cutoff: Optional[str] = None,
                 preserve_fold_order: bool = False):
        """
        Initialize DataPreparation with configuration parameters.
        
        Args:
            data_path: Path to training data CSV
            feats: List of features to use (defaults to FEAT_MAD2_AND_STYLES_TEST_FILTERED)
            odds: Whether to include odds features
            train_size: Fraction for training data
            val_size: Fraction for validation data (None for 2-way split)
            test_size: Fraction for test data
            normalize: Normalization method ('robust', 'zscore', 'none')
            use_recency_weights: Whether to apply recency-based sample weights
            decay_rate: Decay rate for recency weights
            balance_fighters: Whether to balance fighter1/fighter2 distribution
            target_balance: Target proportion of fighter1 wins
            start_date: Start date for filtering fights
            num_fights: Minimum number of previous fights required
            include_split_dec: Whether to include split decisions
            data_cutoff: End date for filtering fights in YYYY-MM-DD format (None = no cutoff)
            preserve_fold_order: Whether to create groups column for preserving chronological fold order
        """
        self.data_path = data_path or str(data_file('training_data.csv'))
        self.feats = feats
        self.odds = odds
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.normalize = normalize
        self.use_recency_weights = use_recency_weights
        self.decay_rate = decay_rate
        self.balance_fighters = balance_fighters
        self.target_balance = target_balance
        self.start_date = start_date
        self.num_fights = num_fights
        self.include_split_dec = include_split_dec
        self.data_cutoff = data_cutoff
        self.preserve_fold_order = preserve_fold_order
        
        # Will be populated during preparation
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.test_start_date = None
        
    def load_and_clean_data(self, model_dir: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load, filter, balance, and select features for the dataset.
        Does NOT split or normalize.
        
        Args:
            model_dir: Directory to save artifacts (feats.txt)
            
        Returns:
            Tuple of (X, y) containing the full processed dataset
        """
        # Define paths for artifacts
        training_data_path = os.path.join(model_dir, 'training_data.csv')
        
        # Step 1: Load data
        print("Loading data...")
        orig_df = pd.read_csv(self.data_path)
        orig_df.sort_values(by=['event_date', 'fight_id'], inplace=True)
        orig_df.to_csv(training_data_path, index=False)
        
        # Step 2: Handle features and odds
        feats_for_model = self._prepare_features(model_dir)
        
        # Step 3: Remove odds columns if not using odds
        if not self.odds:
            odds_columns = [col for col in orig_df.columns if '_odds' in col]
            orig_df = orig_df.drop(columns=odds_columns)
        
        # Step 4: Filter fights
        cutoff_msg = f", cutoff_date={self.data_cutoff}" if self.data_cutoff else ""
        print(f"Filtering fights (min_fights={self.num_fights}, start_date={self.start_date}{cutoff_msg})...")
        self.df = filter_fights(orig_df, self.num_fights, date=self.start_date, 
                               include_split_dec=self.include_split_dec, data_cutoff=self.data_cutoff)
        
        # Step 5: Balance dataset if enabled (MUST be done before load_training_data)
        if self.balance_fighters:
            print(f"Balancing dataset (target={self.target_balance})...")
            self.df = balance_dataset(self.df, target_balance=self.target_balance)
        
        # Step 6: Load training data with selected features
        print(f"Loading training data with {len(feats_for_model)} features...")
        X, y = load_training_data(self.df, feats_for_model)
        
        return X, y

    def prepare_data(self, model_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series, pd.Series]:
        """
        Execute full data preparation pipeline.
        
        Args:
            model_dir: Directory to save artifacts (scaler, feats, training data, etc.)
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
            Note: X_val and y_val will be None if val_size is None
        """
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        # Steps 1-6: Load and clean data
        X, y = self.load_and_clean_data(model_dir)
        
        # Step 7: Split data
        self._split_data(X, y)
        
        # Step 8: Save test start date
        self._save_test_start_date(model_dir)
        
        # Step 9: Apply normalization
        self._apply_normalization(scaler_path)
        
        # Step 10: Add recency weights if enabled
        self._apply_recency_weights()
        
        # Step 11: Add groups column for fold ordering if enabled
        self._apply_fold_groups()
        
        print("Data preparation complete!")
        return self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test
    
    def _prepare_features(self, model_dir: str) -> List[str]:
        """Prepare feature list and save to feats.txt"""
        feats_for_model = self.feats.copy() if self.feats else []
        
        # Add odds feature if enabled
        if self.odds:
            odds_feature = 'f1_sevenday_vigless_ip_opening_odds'
            if odds_feature not in feats_for_model:
                feats_for_model.append(odds_feature)
        
        # Save feats as feats.txt
        if feats_for_model:
            feats_path = os.path.join(model_dir, 'feats.txt')
            with open(feats_path, 'w') as f:
                for feat in feats_for_model:
                    f.write(f"{feat}\n")
            print(f"Saved {len(feats_for_model)} features to {feats_path}")
        
        return feats_for_model
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series):
        """Split data into train/validation/test sets"""
        if self.val_size is not None:
            print(f"\n=== Three-way split for calibration: {self.train_size*100:.0f}% train, "
                  f"{self.val_size*100:.0f}% calibration, {self.test_size*100:.0f}% test ===")
            (self.X_train, self.y_train), (self.X_val, self.y_val), (self.X_test, self.y_test) = \
                split_data_three_way(X, y, train_size=self.train_size, val_size=self.val_size)
        else:
            print(f"\n=== Two-way split: {self.train_size*100:.0f}% train, "
                  f"{(1-self.train_size)*100:.0f}% test ===")
            # Use simple split if no validation split
            (self.X_train, self.y_train), (self.X_test, self.y_test) = \
                split_data_simple(X, y, train_size=self.train_size)
            self.X_val, self.y_val = None, None
    
    def _save_test_start_date(self, model_dir: str):
        """Get and save test data start date"""
        self.test_start_date = self.df.loc[self.X_test.index, 'event_date'].min()
        print(f"Test data start date: {self.test_start_date}")
        
        # Write test start date to shared file for other scripts to access
        test_start_path = os.path.join(model_dir, 'test_start_date.txt')
        write_test_start_date(self.test_start_date, test_start_path)
        
        # Also save data cutoff if one was used
        if self.data_cutoff is not None:
            cutoff_path = os.path.join(model_dir, 'data_cutoff.txt')
            with open(cutoff_path, 'w') as f:
                f.write(str(self.data_cutoff))
            print(f"Saved data cutoff date: {self.data_cutoff} to {cutoff_path}")
    
    def _apply_normalization(self, scaler_path: str):
        """Apply normalization to features"""
        print(f"Applying {self.normalize} normalization...")
        
        if self.val_size is not None:
            # Three-way split: fit scaler on training data, apply to train/val/test
            if self.normalize == 'robust':
                self.X_train, X_temp = apply_robust_normalization(
                    self.X_train, pd.concat([self.X_val, self.X_test]), scaler_path)
                self.X_val = X_temp.iloc[:len(self.X_val)]
                self.X_test = X_temp.iloc[len(self.X_val):]
            elif self.normalize == 'zscore':
                self.X_train, X_temp = apply_zscore_normalization(
                    self.X_train, pd.concat([self.X_val, self.X_test]), scaler_path)
                self.X_val = X_temp.iloc[:len(self.X_val)]
                self.X_test = X_temp.iloc[len(self.X_val):]
            elif self.normalize == 'none':
                self.X_train, X_temp = apply_no_normalization(
                    self.X_train, pd.concat([self.X_val, self.X_test]), scaler_path)
                self.X_val = X_temp.iloc[:len(self.X_val)]
                self.X_test = X_temp.iloc[len(self.X_val):]
            else:
                # Default to robust if normalize value is not recognized
                print(f"Warning: Unknown normalization '{self.normalize}', defaulting to 'robust'")
                self.X_train, X_temp = apply_robust_normalization(
                    self.X_train, pd.concat([self.X_val, self.X_test]), scaler_path)
                self.X_val = X_temp.iloc[:len(self.X_val)]
                self.X_test = X_temp.iloc[len(self.X_val):]
        else:
            # Two-way split: fit scaler on training data, apply to train/test
            if self.normalize == 'robust':
                self.X_train, self.X_test = apply_robust_normalization(
                    self.X_train, self.X_test, scaler_path)
            elif self.normalize == 'zscore':
                self.X_train, self.X_test = apply_zscore_normalization(
                    self.X_train, self.X_test, scaler_path)
            elif self.normalize == 'none':
                self.X_train, self.X_test = apply_no_normalization(
                    self.X_train, self.X_test, scaler_path)
            else:
                # Default to robust if normalize value is not recognized
                print(f"Warning: Unknown normalization '{self.normalize}', defaulting to 'robust'")
                self.X_train, self.X_test = apply_robust_normalization(
                    self.X_train, self.X_test, scaler_path)
    
    def _apply_recency_weights(self):
        """Add recency weights to training data if enabled"""
        if not self.use_recency_weights:
            return
            
        print(f"Applying recency weights (decay_rate={self.decay_rate})...")
        sample_weight_col = 'sample_weight'
        
        # Calculate and add sample weights to training data
        sample_weights = calculate_recency_weights(self.df, self.X_train.index, 
                                                  decay_rate=self.decay_rate)
        self.X_train[sample_weight_col] = sample_weights
        print(f"Added sample_weight column with recency-based weights")
        
        # For validation and test data, add uniform weights for consistency
        if self.val_size is not None and self.X_val is not None:
            self.X_val.loc[:, sample_weight_col] = 1.0
        self.X_test.loc[:, sample_weight_col] = 1.0
    
    def _apply_fold_groups(self):
        """Add groups column to preserve chronological order in folds if enabled"""
        if not self.preserve_fold_order:
            return
            
        print("Creating groups column for preserving chronological fold order...")
        groups_col = 'groups'
        
        # Get the training data indices sorted chronologically (they should already be sorted)
        train_indices = self.X_train.index.tolist()
        
        # Create groups for n_splits folds
        # For example, with 4 folds and 1000 training samples:
        # - Group 0: samples 0-249
        # - Group 1: samples 250-499  
        # - Group 2: samples 500-749
        # - Group 3: samples 750-999
        n_splits = 4  # Default value used in train.py
        n_samples = len(train_indices)
        samples_per_fold = n_samples // n_splits
        
        # Create group assignments
        group_assignments = []
        for i in range(n_samples):
            group_id = min(i // samples_per_fold, n_splits - 1)  # Ensure last group gets remaining samples
            group_assignments.append(group_id)
        
        # Add groups column to training data
        self.X_train[groups_col] = group_assignments
        
        # For validation and test data, add dummy groups for consistency (won't be used in training)
        if self.val_size is not None and self.X_val is not None:
            self.X_val.loc[:, groups_col] = 0  # All validation samples in group 0
        self.X_test.loc[:, groups_col] = 0  # All test samples in group 0
        
        print(f"Added groups column with {n_splits} groups ({samples_per_fold} samples per group)")
        print(f"Group distribution: {pd.Series(group_assignments).value_counts().sort_index().to_dict()}")

    def get_sample_weight_column(self) -> Optional[str]:
        """Get the name of the sample weight column if recency weights are used"""
        return 'sample_weight' if self.use_recency_weights else None
    
    def get_groups_column(self) -> Optional[str]:
        """Get the name of the groups column if preserve_fold_order is enabled"""
        return 'groups' if self.preserve_fold_order else None
