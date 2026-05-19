"""
Walk-forward validation implementation for UFC fight prediction.

Implements expanding window cross-validation to address temporal distribution shift.
Follows AutoGluon 1.4 best practices for small datasets (~3k rows):
- Uses 'extreme' preset (maps to best_quality with aggressive stacking)
- Leverages bagging and stacking to prevent overfitting
- Validates on future data only (no leakage)
"""

import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import shutil
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from autogluon.tabular import TabularPredictor
from libs.modeling.train import TrainingConfig, FeatureSelector
from libs.modeling.data_preparation import DataPreparation
from libs.modeling.data_utils import (
    apply_robust_normalization, 
    apply_zscore_normalization, 
    apply_no_normalization,
    calculate_recency_weights
)
from libs.paths import data_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    base_config: TrainingConfig
    n_folds: int = 5
    initial_train_years: int = 6  # Start with 6 years of history
    test_period_years: int = 1    # Predict 1 year at a time
    gap_days: int = 0             # Gap between train and test to prevent lookahead
    rolling_window_years: int = 0 # If > 0, use rolling window of this size instead of expanding
    save_fold_models: bool = False
    output_dir: str = 'walk_forward_results'
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

class WalkForwardValidator:
    """
    Executes walk-forward validation using expanding or rolling window strategy.
    """
    
    def __init__(self, config: WalkForwardConfig):
        self.config = config
        self.results: List[Dict] = []
        self.all_predictions: List[pd.DataFrame] = []
        self.data_prep = None # Will be initialized in _load_data
        
    def run(self) -> pd.DataFrame:
        """Execute the full walk-forward validation pipeline."""
        logger.info("Starting Walk-Forward Validation")
        
        # 1. Load and Preprocess Full Dataset
        full_X, full_y = self._load_data()
        
        # Combine for fold generation (need dates)
        full_df = self.data_prep.df.copy() # This has event_date and all cols
        
        # 2. Generate Folds
        folds = self._create_folds(full_df)
        logger.info(f"Generated {len(folds)} folds (Window Type: {'Rolling' if self.config.rolling_window_years > 0 else 'Expanding'})")
        
        # 3. Train and Validate Each Fold
        for i, fold in enumerate(folds):
            logger.info(f"\nProcessing Fold {i+1}/{len(folds)}")
            logger.info(f"Train: {fold['train_range']} (n={len(fold['train_df'])})")
            logger.info(f"Test:  {fold['test_range']} (n={len(fold['test_df'])})")
            
            fold_result, fold_preds = self._process_fold(fold, full_X, full_y, fold_idx=i)
            self.results.append(fold_result)
            self.all_predictions.append(fold_preds)
            
        # 4. Aggregate and Save
        summary_df = self._save_results()
        logger.info("\nWalk-Forward Validation Complete")
        logger.info(f"Mean Accuracy: {summary_df['accuracy'].mean():.4f}")
        logger.info(f"Mean Log Loss: {summary_df['log_loss'].mean():.4f}")
        
        return summary_df

    def _load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data using DataPreparation."""
        # Resolve features
        data_path = self._get_data_path()
        temp_df = pd.read_csv(data_path)
        
        if self.config.base_config.features:
            features = self.config.base_config.features
        else:
            available = FeatureSelector.get_features_from_dataframe(temp_df)
            features = FeatureSelector.select_features(
                available,
                self.config.base_config.included_strings,
                self.config.base_config.excluded_strings,
                self.config.base_config.required_strings
            )
            
        # Initialize DataPreparation
        self.data_prep = DataPreparation(
            data_path=data_path,
            feats=features,
            odds=False, # Odds handled via features list if included
            normalize=self.config.base_config.normalize,
            use_recency_weights=self.config.base_config.use_recency_weights,
            decay_rate=self.config.base_config.decay_rate,
            start_date=self.config.base_config.start_date,
            num_fights=self.config.base_config.num_fights,
            include_split_dec=self.config.base_config.include_split_dec
        )
        
        # Load and clean
        return self.data_prep.load_and_clean_data(self.config.output_dir)

    def _get_data_path(self) -> str:
        """Determine data path from model type."""
        return str(data_file('training_data_dec.csv' if self.config.base_config.model_type == 'decision' else 'training_data.csv'))

    def _create_folds(self, df: pd.DataFrame) -> List[Dict]:
        """Generate expanding or rolling window folds."""
        folds = []
        df['event_date'] = pd.to_datetime(df['event_date'])
        years = df['event_date'].dt.year.unique()
        min_year = int(min(years))
        max_year = int(max(years))
        
        current_train_end = min_year + self.config.initial_train_years - 1
        
        while True:
            test_start = current_train_end + 1
            test_end = test_start + self.config.test_period_years - 1
            
            if test_end > max_year:
                break
                
            # Determine start year for training
            if self.config.rolling_window_years > 0:
                train_start_year = current_train_end - self.config.rolling_window_years + 1
                # Ensure we don't go before min_year (though initial_train_years should prevent this)
                train_start_year = max(min_year, train_start_year)
            else:
                train_start_year = min_year
                
            # Create indices for split
            train_mask = (df['event_date'].dt.year >= train_start_year) & (df['event_date'].dt.year <= current_train_end)
            test_mask = (df['event_date'].dt.year >= test_start) & (df['event_date'].dt.year <= test_end)
            
            # Apply gap if needed (purge)
            if self.config.gap_days > 0:
                train_end_date = df[train_mask]['event_date'].max()
                test_start_date = df[test_mask]['event_date'].min()
                if (test_start_date - train_end_date).days < self.config.gap_days:
                    # Adjust train mask to enforce gap
                    cutoff = test_start_date - pd.Timedelta(days=self.config.gap_days)
                    train_mask = train_mask & (df['event_date'] <= cutoff)
            
            train_idx = df[train_mask].index
            test_idx = df[test_mask].index
            
            if len(test_idx) == 0:
                current_train_end += 1
                continue
                
            folds.append({
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_range': f"{train_start_year}-{current_train_end}",
                'test_range': f"{test_start}-{test_end}",
                'train_df': df.loc[train_idx], # For metadata
                'test_df': df.loc[test_idx]    # For metadata
            })
            
            current_train_end += 1
            
            if len(folds) >= self.config.n_folds:
                break
            
        return folds

    def _process_fold(self, fold: Dict, full_X: pd.DataFrame, full_y: pd.Series, fold_idx: int) -> Tuple[Dict, pd.DataFrame]:
        """Train and evaluate a single fold."""
        # Slice data
        X_train = full_X.loc[fold['train_idx']].copy()
        y_train = full_y.loc[fold['train_idx']].copy()
        X_test = full_X.loc[fold['test_idx']].copy()
        y_test = full_y.loc[fold['test_idx']].copy()
        
        # 1. Normalize (Per Fold)
        scaler_path = os.path.join(self.config.output_dir, f"scaler_fold_{fold_idx}.pkl")
        if self.config.base_config.normalize == 'robust':
            X_train, X_test = apply_robust_normalization(X_train, X_test, scaler_path)
        elif self.config.base_config.normalize == 'zscore':
            X_train, X_test = apply_zscore_normalization(X_train, X_test, scaler_path)
        elif self.config.base_config.normalize == 'none':
            X_train, X_test = apply_no_normalization(X_train, X_test, scaler_path)
            
        # 2. Apply Recency Weights (Per Fold)
        if self.config.base_config.use_recency_weights:
            # We need event dates for weights
            train_dates = self.data_prep.df.loc[fold['train_idx'], 'event_date']
            # Calculate weights relative to end of training set
            weights = calculate_recency_weights(
                self.data_prep.df.loc[fold['train_idx']], 
                X_train.index, 
                decay_rate=self.config.base_config.decay_rate
            )
            X_train['sample_weight'] = weights
            X_test['sample_weight'] = 1.0 # Uniform for test
            
        # 3. Setup AutoGluon
        model_path = os.path.join(self.config.output_dir, f"fold_{fold_idx}_model")
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        predictor = TabularPredictor(
            label='y_true',
            eval_metric='log_loss',
            problem_type='binary',
            path=model_path,
            sample_weight='sample_weight' if self.config.base_config.use_recency_weights else None,
            verbosity=2
        )
        
        # Prepare train data
        train_data = X_train.copy()
        train_data['y_true'] = y_train
        
        # 4. Train
        fit_kwargs = {
            'train_data': train_data,
            'presets': self.config.base_config.preset,
            'time_limit': self.config.base_config.time_limit,
            'ag_args_fit': {'shuffle': False} 
        }
        
        if self.config.base_config.included_model_types:
            fit_kwargs['included_model_types'] = self.config.base_config.included_model_types
            
        predictor.fit(**fit_kwargs)
        
        # 5. Evaluate
        test_data = X_test.copy()
        test_data['y_true'] = y_test
        
        scores = predictor.evaluate(test_data)
        y_pred_proba = predictor.predict_proba(test_data).iloc[:, 1]
        
        # Cleanup
        if not self.config.save_fold_models:
            shutil.rmtree(model_path)
            if os.path.exists(scaler_path):
                os.remove(scaler_path)
            
        # Result
        result = {
            'fold': fold_idx,
            'train_range': fold['train_range'],
            'test_range': fold['test_range'],
            'accuracy': scores['accuracy'],
            'log_loss': scores['log_loss'],
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        # Predictions with metadata
        preds_df = fold['test_df'][['fighter1_name', 'fighter2_name', 'event_date']].copy()
        preds_df['y_true'] = y_test
        preds_df['y_pred_proba'] = y_pred_proba
        preds_df['fold'] = fold_idx
        
        return result, preds_df

    def _save_results(self) -> pd.DataFrame:
        """Save all validation artifacts."""
        summary_df = pd.DataFrame(self.results)
        summary_df.to_csv(os.path.join(self.config.output_dir, 'fold_results.csv'), index=False)
        
        all_preds_df = pd.concat(self.all_predictions)
        all_preds_df.to_csv(os.path.join(self.config.output_dir, 'all_predictions.csv'), index=False)
        
        with open(os.path.join(self.config.output_dir, 'report.txt'), 'w') as f:
            f.write("Walk-Forward Validation Report\n")
            f.write("==============================\n\n")
            f.write(f"Model Type: {self.config.base_config.model_type}\n")
            f.write(f"Preset: {self.config.base_config.preset}\n")
            f.write(f"Folds: {len(self.results)}\n\n")
            f.write("Aggregate Metrics:\n")
            f.write(f"Mean Accuracy: {summary_df['accuracy'].mean():.4f} ± {summary_df['accuracy'].std():.4f}\n")
            f.write(f"Mean Log Loss: {summary_df['log_loss'].mean():.4f} ± {summary_df['log_loss'].std():.4f}\n\n")
            f.write("Per-Fold Results:\n")
            f.write(summary_df.to_string(index=False))
            
        return summary_df

if __name__ == "__main__":
    # Example usage
    from libs.feature_store.features import vSeven_testing2
    
    base_config = TrainingConfig(
        model_type='win',
        preset='extreme',
        time_limit=600,
        features=vSeven_testing2,
        use_recency_weights=False,
        decay_rate=0.125
    )
    
    wf_config = WalkForwardConfig(
        base_config=base_config,
        n_folds=5,
        initial_train_years=6,
        output_dir='walk_forward_test_run'
    )
    
    validator = WalkForwardValidator(wf_config)
    validator.run()
