"""Training script for UFC fight prediction models.

This module provides a clean, testable interface for training models to predict
either fight outcomes (win/loss) or fight methods (decision/no decision).
"""

import os
import sys
import argparse
import numpy as np
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
from autogluon.tabular import TabularPredictor

from libs.feature_store.features import vSeven_testing2, vSeven_testing2_with_f1, DECISION_TEST_FEATS, DECISION_TEST_FEATS2, DECISION_TEST_FEATS3, DECISION_TEST_FEATS4
from libs.modeling.data_preparation import DataPreparation
from libs.modeling.data_utils import balance_dataset, _swap_fighter_roles, filter_fights, load_training_data
from libs.modeling.model_utils import ModelUtils
from libs.modeling.portable_artifacts import load_joblib_artifact, load_tabular_predictor
from libs.paths import data_file


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    model_type: str  # 'win' or 'decision'
    preset: str  # 'extreme' or 'best'
    time_limit: int = 1500  # seconds
    
    # Data split configuration
    test_size: Optional[str] = None  # Date string (e.g., "2025-06-01") or None for no holdout
    val_date: Optional[str] = None  # Start date of validation set for timeseries_split (e.g., "2024-01-01"). If None, defaults to "2024-01-01" when test_size is set, else "2025-01-01"
    
    # Feature configuration
    features: Optional[List[str]] = None  # Explicit feature list
    included_strings: Optional[List[str]] = None  # Strings that must be in feature name
    excluded_strings: Optional[List[str]] = None  # Strings that must NOT be in feature name
    required_strings: Optional[List[str]] = None  # Exact features that override excluded_strings
    
    # Data filtering
    start_date: str = '2014-01-01'
    num_fights: int = 2
    include_split_dec: bool = False
    
    # Normalization
    normalize: str = 'robust'  # 'robust', 'zscore', or 'none'
    
    # Recency weighting
    use_recency_weights: bool = False
    decay_rate: float = 0.1
    
    # Feature importance
    calculate_importance: bool = False
    
    # Model type configuration
    included_model_types: Optional[List[str]] = None  # Model types to include (e.g., ['TABPFNV2', 'TABM', 'TABICL', 'GBM', 'XGB', 'CAT', 'MITRA'])
    
    # Split strategy: 'standard', 'timeseries_split', or 'walkforward'
    split_strategy: str = 'standard'  # Data splitting strategy
    
    # Walk-forward validation configuration
    walkforward_n_windows: int = 4  # Number of validation windows
    walkforward_initial_year: int = 2021  # First validation year (window 0 trains up to 2021, validates on 2022)
    
    # Backward compatibility: timeseries_split (deprecated)
    timeseries_split: Optional[bool] = None  # Deprecated: use split_strategy instead
    
    # Refit on full data
    refit_all: bool = False  # Retrain on all data (including test/holdout) after evaluations. Only applies to timeseries_split; walkforward always refits internally.
    refit_full: bool = False  # Call AutoGluon's refit_full() after validation evaluation in timeseries_split mode
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_type not in ['win', 'decision']:
            raise ValueError(f"model_type must be 'win' or 'decision', got '{self.model_type}'")
        
        if self.preset not in ['extreme', 'best']:
            raise ValueError(f"preset must be 'extreme' or 'best', got '{self.preset}'")
        
        if self.split_strategy not in ['standard', 'timeseries_split', 'walkforward']:
            raise ValueError(f"split_strategy must be 'standard', 'timeseries_split', or 'walkforward', got '{self.split_strategy}'")
        
        # Warn if refit_all is set but using walkforward (walkforward always refits internally)
        if self.refit_all and self.split_strategy == 'walkforward':
            warnings.warn(
                "refit_all=True is ignored for walkforward split strategy. Walkforward always refits models internally.",
                UserWarning,
                stacklevel=2
            )
        
        # Handle backward compatibility for timeseries_split
        if self.timeseries_split is not None:
            warnings.warn(
                "timeseries_split parameter is deprecated. Use split_strategy='timeseries_split' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            if self.timeseries_split:
                self.split_strategy = 'timeseries_split'
            else:
                self.split_strategy = 'standard'
        
        # Validate test_size if provided (must be valid date string)
        if self.test_size is not None:
            try:
                pd.Timestamp(self.test_size)
            except (ValueError, TypeError) as e:
                raise ValueError(f"test_size must be a valid date string (e.g., '2025-06-01') or None, got '{self.test_size}': {e}")


class FeatureSelector:
    """Handles feature selection with string-based filtering."""
    
    @staticmethod
    def select_features(
        available_features: List[str],
        included_strings: Optional[List[str]] = None,
        excluded_strings: Optional[List[str]] = None,
        required_strings: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select features based on string matching criteria.
        
        Args:
            available_features: List of all available feature names
            included_strings: Features must contain at least one of these strings
            excluded_strings: Features must NOT contain any of these strings
            required_strings: Exact feature names that override excluded_strings
            
        Returns:
            List of selected feature names
        """
        selected = set(available_features)
        
        # Apply included_strings filter
        if included_strings:
            included_set = set()
            for feature in selected:
                if any(included_str in feature for included_str in included_strings):
                    included_set.add(feature)
            selected = included_set
        
        # Apply excluded_strings filter
        if excluded_strings:
            excluded_set = set()
            for feature in selected:
                if any(excluded_str in feature for excluded_str in excluded_strings):
                    excluded_set.add(feature)
            selected -= excluded_set
        
        # Add required_strings (override excluded_strings)
        if required_strings:
            for required_feature in required_strings:
                if required_feature in available_features:
                    selected.add(required_feature)
        
        return sorted(list(selected))
    
    @staticmethod
    def get_features_from_dataframe(df: pd.DataFrame) -> List[str]:
        """
        Get all feature columns from a dataframe.
        
        Excludes metadata columns and target variable.
        
        Args:
            df: Input dataframe
            
        Returns:
            List of feature column names
        """
        metadata_columns = {
            'fight_id', 'event_id', 'fighter1_id', 'fighter2_id',
            'fighter1_name', 'fighter2_name', 'event_date', 'method',
            'y_true', 'sample_weight', 'groups', 'fighter_dob'
        }
        
        return [col for col in df.columns if col not in metadata_columns]


class EnsemblePredictor:
    """
    Ensemble predictor that averages predictions from multiple TabularPredictor models.
    
    This class wraps multiple AutoGluon predictors and provides the same interface
    as a single TabularPredictor by averaging predictions across all models.
    
    Each model uses its own scaler (fitted on its training data) to ensure
    predictions match the scaling used during training.
    """
    
    def __init__(self, predictors: List[TabularPredictor], path: str, scaler_paths: Optional[List[str]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            predictors: List of TabularPredictor instances to ensemble
            path: Path where the ensemble predictor should be saved
            scaler_paths: Optional list of scaler paths (one per predictor). 
                         If None, will try to infer from predictor paths.
        """
        self.predictors = predictors
        self.path = path
        self._model_best = f"Ensemble_{len(predictors)}_folds"
        
        # Store scaler paths for each predictor
        if scaler_paths is None:
            # Infer scaler paths from predictor paths
            self.scaler_paths = []
            for predictor in predictors:
                # Scaler is typically in the same directory as the predictor
                scaler_path = os.path.join(predictor.path, 'scaler.pkl')
                self.scaler_paths.append(scaler_path if os.path.exists(scaler_path) else None)
        else:
            self.scaler_paths = scaler_paths
        
        # Ensure we have the same number of scalers as predictors
        if len(self.scaler_paths) != len(self.predictors):
            raise ValueError(f"Mismatch: {len(self.predictors)} predictors but {len(self.scaler_paths)} scaler paths")
        
        # Copy model info from first predictor for compatibility
        if predictors:
            self.label = predictors[0].label
            self.eval_metric = predictors[0].eval_metric
            self.problem_type = predictors[0].problem_type
    
    def _scale_data_for_predictor(self, data: pd.DataFrame, scaler_path: Optional[str]) -> pd.DataFrame:
        """
        Scale data using the specified scaler.
        
        Args:
            data: Input data to scale
            scaler_path: Path to scaler pickle file (None if no scaling needed)
            
        Returns:
            Scaled data DataFrame
        """
        if scaler_path is None or not os.path.exists(scaler_path):
            return data.copy()
        
        try:
            scaler = load_joblib_artifact(scaler_path)
            
            # Identify columns to exclude from scaling
            date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
            categorical_static_feats = ['weightclass_encoded', 'odds']
            
            def should_exclude_col(col_name):
                if col_name in date_cols:
                    return True
                for cat_feat in categorical_static_feats:
                    if cat_feat in col_name:
                        return True
                return False
            
            # Get features to scale
            features_to_scale = [col for col in data.columns 
                                if not should_exclude_col(col) and col not in ['sample_weight', 'y_true', self.label]]
            
            # Scale only the features that should be scaled
            scaled_data = data.copy()
            if len(features_to_scale) > 0:
                scaled_data[features_to_scale] = scaler.transform(data[features_to_scale])
            
            return scaled_data
        except Exception as e:
            print(f"Warning: Could not load/apply scaler from {scaler_path}: {e}")
            return data.copy()
    
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        Predict class labels by averaging predictions from all models.
        
        Each model uses its own scaler (fitted on its training data) before prediction.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional arguments passed to each predictor
            
        Returns:
            Array of predicted class labels
        """
        # Get probability predictions from all models
        all_probs = []
        for predictor, scaler_path in zip(self.predictors, self.scaler_paths):
            # Scale data using this predictor's scaler
            scaled_data = self._scale_data_for_predictor(data, scaler_path)
            
            # Make prediction with scaled data
            probs = predictor.predict_proba(scaled_data, **kwargs)
            if isinstance(probs, pd.DataFrame):
                # Handle both integer (1) and string ('1') column names
                # AutoGluon typically uses integer columns, but check both for robustness
                if 1 in probs.columns:
                    probs = probs[1].values
                elif '1' in probs.columns:
                    probs = probs['1'].values
                else:
                    probs = probs.iloc[:, 1].values
            else:
                probs = probs[:, 1] if probs.ndim > 1 else probs
            all_probs.append(probs)
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        
        # Convert to binary predictions
        return (avg_probs > 0.5).astype(int)
    
    def predict_proba(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Predict class probabilities by averaging probabilities from all models.
        
        Each model uses its own scaler (fitted on its training data) before prediction.
        
        Args:
            data: Input data for prediction
            **kwargs: Additional arguments passed to each predictor
            
        Returns:
            DataFrame with probability predictions (columns: [0, 1])
        """
        # Get probability predictions from all models
        all_probs = []
        for predictor, scaler_path in zip(self.predictors, self.scaler_paths):
            # Scale data using this predictor's scaler
            scaled_data = self._scale_data_for_predictor(data, scaler_path)
            
            # Make prediction with scaled data
            probs = predictor.predict_proba(scaled_data, **kwargs)
            if isinstance(probs, pd.DataFrame):
                all_probs.append(probs.values)
            else:
                all_probs.append(probs)
        
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        
        # Return as DataFrame with same structure as TabularPredictor
        # Use integer column names to match AutoGluon's format (0, 1)
        return pd.DataFrame(avg_probs, columns=[0, 1], index=data.index)
    
    def evaluate(self, data: pd.DataFrame, **kwargs) -> dict:
        """
        Evaluate ensemble on data by averaging predictions.
        
        Args:
            data: Evaluation data (must include label column)
            **kwargs: Additional arguments passed to evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(data, **kwargs)
        y_pred_proba = self.predict_proba(data, **kwargs)
        
        # Get true labels
        if self.label in data.columns:
            y_true = data[self.label].values
        else:
            raise ValueError(f"Label column '{self.label}' not found in data")
        
        # Calculate metrics (matching AutoGluon's evaluation format)
        from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, brier_score_loss
        
        # Handle column access - AutoGluon uses integer column names (0, 1)
        # Our predict_proba returns columns [0, 1], so use 1 or fallback to index 1
        if 1 in y_pred_proba.columns:
            proba_values = y_pred_proba[1].values
        elif '1' in y_pred_proba.columns:
            proba_values = y_pred_proba['1'].values
        else:
            proba_values = y_pred_proba.iloc[:, 1].values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, proba_values),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'brier_score': brier_score_loss(y_true, proba_values)
        }
        
        return metrics
    
    def info(self) -> dict:
        """
        Get information about the ensemble predictor.
        
        Returns:
            Dictionary with ensemble information
        """
        return {
            'model_type': 'EnsemblePredictor',
            'num_models': len(self.predictors),
            'model_best': self.model_best,
            'label': self.label,
            'eval_metric': self.eval_metric,
            'problem_type': self.problem_type
        }
    
    def leaderboard(self, *args, **kwargs):
        """
        Compatibility method - leaderboard not available for ensemble predictor.
        
        Returns:
            None
        """
        print("Leaderboard not available in EnsemblePredictor (time-series split mode)")
        return None
    
    def feature_importance(self, *args, **kwargs):
        """
        Compatibility method - feature importance not yet supported for ensemble predictor.
        
        Returns:
            None
        """
        print("Feature importance not supported in EnsemblePredictor yet")
        return None
    
    @property
    def model_best(self):
        """Return the model_best identifier."""
        return self._model_best
    
    def save(self):
        """Save ensemble predictor metadata (models are already saved in their respective folders)."""
        # Save ensemble info
        ensemble_info_path = os.path.join(self.path, 'ensemble_info.txt')
        with open(ensemble_info_path, 'w') as f:
            f.write(f"Ensemble Predictor with {len(self.predictors)} folds\n")
            f.write(f"Model Type: {self.model_best}\n")
            f.write(f"Label: {self.label}\n")
            f.write(f"Eval Metric: {self.eval_metric}\n")
            f.write(f"Problem Type: {self.problem_type}\n")
            f.write(f"\nFold Models:\n")
            for i, predictor in enumerate(self.predictors):
                f.write(f"  Fold {i}: {predictor.path}\n")
    
    @classmethod
    def load(cls, path: str) -> 'EnsemblePredictor':
        """
        Load ensemble predictor from disk.
        
        Only supports walk-forward format:
        - window_0, window_1, ..., final_model (for walkforward split_strategy)
        
        Note: timeseries_split creates a single model, not an ensemble.
        
        Args:
            path: Path to ensemble predictor directory
            
        Returns:
            Loaded EnsemblePredictor instance
        """
        predictors = []
        scaler_paths = []
        
        # Walk-forward format: window_* + final_model
        window_dirs = []
        final_model_dir = None
        
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                if item.startswith('window_'):
                    window_num = int(item.split('_')[1])
                    window_dirs.append((window_num, item_path))
                elif item == 'final_model':
                    final_model_dir = item_path
        
        if not window_dirs:
            raise ValueError(f"No walk-forward models found in {path}. Expected 'window_*' directories. "
                           f"This loader is only for walk-forward ensemble models. "
                           f"For single models (standard/timeseries_split), use TabularPredictor.load() directly.")
        
        # Walk-forward format: load windows + final_model
        window_dirs.sort(key=lambda x: x[0])
        for window_num, window_path in window_dirs:
            predictor = load_tabular_predictor(
                TabularPredictor,
                window_path,
                require_version_match=False,
                require_py_version_match=False,
            )
            predictors.append(predictor)
            # Get scaler path for this window
            scaler_path = os.path.join(window_path, 'scaler.pkl')
            scaler_paths.append(scaler_path if os.path.exists(scaler_path) else None)
        
        if final_model_dir:
            predictor = load_tabular_predictor(
                TabularPredictor,
                final_model_dir,
                require_version_match=False,
                require_py_version_match=False,
            )
            predictors.append(predictor)
            # Get scaler path for final model
            scaler_path = os.path.join(final_model_dir, 'scaler.pkl')
            scaler_paths.append(scaler_path if os.path.exists(scaler_path) else None)
        
        if not predictors:
            raise ValueError(f"No valid models found in {path}")
        
        return cls(predictors, path, scaler_paths=scaler_paths)


class EnsemblePredictorWrapper:
    """
    Wrapper class that makes EnsemblePredictor compatible with ModelUtils.
    This wrapper mimics TabularPredictor interface for use with ModelUtils.
    """
    def __init__(self, ensemble_predictor: EnsemblePredictor):
        """Initialize wrapper with ensemble predictor."""
        self.ensemble_predictor = ensemble_predictor
        self.path = ensemble_predictor.path
        self.label = ensemble_predictor.label
        self.eval_metric = ensemble_predictor.eval_metric
        self.problem_type = ensemble_predictor.problem_type
        self.model_best = ensemble_predictor.model_best
    
    def predict(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """Predict using ensemble."""
        return self.ensemble_predictor.predict(data, **kwargs)
    
    def predict_proba(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Predict probabilities using ensemble."""
        return self.ensemble_predictor.predict_proba(data, **kwargs)
    
    @classmethod
    def load(cls, path: str) -> 'EnsemblePredictorWrapper':
        """Load ensemble predictor and wrap it."""
        ensemble = EnsemblePredictor.load(path)
        return cls(ensemble)


# ============================================================================
# Shared Utility Classes
# ============================================================================

class DataLoader:
    """Handles data loading, filtering, and feature preparation."""
    
    @staticmethod
    def get_data_path(model_type: str) -> str:
        """Get the appropriate data path based on model type."""
        if model_type == 'win':
            return str(data_file('training_data.csv'))
        elif model_type == 'decision':
            return str(data_file('training_data_dec.csv'))
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    @staticmethod
    def load_and_filter_data(
        data_path: str,
        num_fights: int,
        start_date: str,
        include_split_dec: bool
    ) -> pd.DataFrame:
        """Load CSV, filter fights, and return sorted DataFrame."""
        full_df = pd.read_csv(data_path)
        full_df['event_date'] = pd.to_datetime(full_df['event_date'])
        full_df = full_df.sort_values('event_date').reset_index(drop=True)
        
        from libs.modeling.data_utils import filter_fights
        full_df = filter_fights(
            full_df,
            threshold=num_fights,
            date=start_date,
            include_split_dec=include_split_dec,
            data_cutoff=None
        )
        
        return full_df
    
    @staticmethod
    def create_holdout_set(
        df: pd.DataFrame,
        test_size: Optional[str]
    ) -> pd.DataFrame:
        """
        Create holdout set based on date cutoff.
        
        If test_size is None, returns empty DataFrame (no holdout).
        If test_size is a date string (e.g., "2025-06-01"), returns all fights after that date.
        
        Args:
            df: Full dataframe with event_date column
            test_size: Date string cutoff or None
            
        Returns:
            DataFrame containing holdout fights (all fights after cutoff date)
        """
        if test_size is None:
            return pd.DataFrame()
        
        cutoff_date = pd.Timestamp(test_size)
        holdout_df = df[df['event_date'] > cutoff_date].copy()
        
        return holdout_df
    
    @staticmethod
    def prepare_features(
        df: pd.DataFrame,
        config: TrainingConfig
    ) -> List[str]:
        """
        Prepare feature list based on configuration.
        
        Args:
            df: Training dataframe
            config: TrainingConfig instance
            
        Returns:
            List of feature names to use
        """
        if config.features:
            return config.features
        
        available_features = FeatureSelector.get_features_from_dataframe(df)
        return FeatureSelector.select_features(
            available_features=available_features,
            included_strings=config.included_strings,
            excluded_strings=config.excluded_strings,
            required_strings=config.required_strings
        )
    
    @staticmethod
    def load_training_features(
        df: pd.DataFrame,
        features: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load X, y from DataFrame using specified features.
        
        Args:
            df: Input DataFrame
            features: List of feature column names
            
        Returns:
            Tuple of (X, y) where X is features DataFrame and y is labels Series
        """
        from libs.modeling.data_utils import load_training_data
        return load_training_data(df, features)


class NormalizationManager:
    """Handles data normalization with proper exclusion of special columns."""
    
    NEVER_NORMALIZE = ['sample_weight', 'y_true', 'weightclass_encoded']
    DATE_COLS = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
    CATEGORICAL_STATIC_FEATS = ['weightclass_encoded', 'odds']
    
    @classmethod
    def _should_exclude_col(cls, col_name: str) -> bool:
        """Check if column should be excluded from scaling."""
        if col_name in cls.DATE_COLS:
            return True
        if col_name in cls.NEVER_NORMALIZE:
            return True
        for cat_feat in cls.CATEGORICAL_STATIC_FEATS:
            if cat_feat in col_name:
                return True
        return False
    
    @classmethod
    def fit_scaler(
        cls,
        X_train: pd.DataFrame,
        normalize: str
    ) -> Tuple[object, List[str]]:
        """
        Fit scaler on training data only.
        
        Args:
            X_train: Training features DataFrame
            normalize: Normalization method ('robust', 'zscore', or 'none')
            
        Returns:
            Tuple of (fitted scaler, list of feature columns to scale)
        """
        features_to_scale = [
            col for col in X_train.columns 
            if not cls._should_exclude_col(col)
        ]
        
        if normalize == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        elif normalize == 'zscore':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        else:  # 'none'
            from sklearn.preprocessing import FunctionTransformer
            scaler = FunctionTransformer(func=None, validate=False)
        
        if len(features_to_scale) > 0:
            scaler.fit(X_train[features_to_scale])
        
        return scaler, features_to_scale
    
    @classmethod
    def transform(
        cls,
        scaler: object,
        X: pd.DataFrame,
        features_to_scale: List[str]
    ) -> pd.DataFrame:
        """
        Apply scaler transformation to data.
        
        Args:
            scaler: Fitted scaler object
            X: Features DataFrame to transform
            features_to_scale: List of feature columns to scale
            
        Returns:
            Transformed DataFrame
        """
        X_transformed = X.copy()
        if len(features_to_scale) > 0:
            X_transformed[features_to_scale] = scaler.transform(X[features_to_scale])
        return X_transformed
    
    @staticmethod
    def save_scaler(scaler: object, path: str):
        """Save scaler to disk."""
        import joblib
        joblib.dump(scaler, path)
    
    @staticmethod
    def load_scaler(path: str) -> object:
        """Load scaler from disk."""
        import joblib
        return joblib.load(path)


class EvaluationManager:
    """Handles model evaluation and result saving."""
    
    @staticmethod
    def evaluate_model(
        predictor: TabularPredictor,
        data: pd.DataFrame
    ) -> Dict:
        """
        Evaluate predictor on dataset.
        
        Args:
            predictor: TabularPredictor instance
            data: DataFrame with 'y_true' column
            
        Returns:
            Dictionary of evaluation metrics
        """
        return predictor.evaluate(data)
    
    @staticmethod
    def save_evaluations(
        results: Dict,
        config: TrainingConfig,
        model_dir: str,
        best_model: str,
        weights: Optional[Dict] = None,
        feature_importance: Optional[pd.DataFrame] = None
    ):
        """
        Save evaluation results to evals.txt.
        
        Args:
            results: Dictionary with 'train_scores', 'val_scores', 'test_scores'
            config: TrainingConfig instance
            model_dir: Model directory path
            best_model: Best model name
            weights: Optional model weights dictionary
            feature_importance: Optional feature importance DataFrame
        """
        evals_path = os.path.join(model_dir, 'evals.txt')
        with open(evals_path, 'w') as f:
            f.write("Model Performance:\n")
            
            if results.get('train_scores') is not None:
                f.write(f"Training accuracy: {results['train_scores']['accuracy']:.4f}\n")
                f.write(f"Training log loss: {results['train_scores']['log_loss']:.4f}\n")
            else:
                f.write("Training scores: N/A\n")
            
            if results.get('val_scores') is not None:
                f.write(f"Validation accuracy: {results['val_scores']['accuracy']:.4f}\n")
                f.write(f"Validation log loss: {results['val_scores']['log_loss']:.4f}\n")
            else:
                f.write("Validation scores: N/A\n")
            
            if results.get('test_scores') is not None:
                f.write(f"Holdout accuracy: {results['test_scores']['accuracy']:.4f}\n")
                f.write(f"Holdout log loss: {results['test_scores']['log_loss']:.4f}\n")
            else:
                f.write("Holdout scores: N/A (no holdout set)\n")
            
            f.write(f"\nBest Model: {best_model}\n")
            
            if weights:
                f.write(f"\nModel weights:\n")
                for model_name, weight in weights.items():
                    f.write(f"{model_name}: {weight:.3f}\n")
            else:
                f.write(f"\nModel weights: N/A (not a weighted ensemble or unavailable)\n")
            
            if feature_importance is not None:
                f.write(f"\nTop Most Important Features:\n")
                for i, (feature, importance_value) in enumerate(feature_importance.head(250).iterrows(), 1):
                    f.write(f"{i}. {feature}: {importance_value['importance']:.4f}\n")
            else:
                f.write(f"\nFeature importance was not calculated.\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"Model Type: {config.model_type}\n")
            f.write(f"Preset: {config.preset}\n")
            f.write(f"Time Limit: {config.time_limit}\n")
            f.write(f"Test Size (cutoff date): {config.test_size if config.test_size else 'None (no holdout)'}\n")
            f.write(f"Split Strategy: {config.split_strategy}\n")
            f.write(f"Refit All: {config.refit_all}\n")
            f.write(f"Start Date: {config.start_date}\n")
            f.write(f"Normalize: {config.normalize}\n")
            f.write(f"Use Recency Weights: {config.use_recency_weights}\n")
            f.write(f"Decay Rate: {config.decay_rate}\n")
            f.write(f"Calculate Importance: {config.calculate_importance}\n")
            f.write(f"Included Model Types: {config.included_model_types}\n")
            f.write(f"Include Split Dec: {config.include_split_dec}\n")
            f.write(f"dec_avg rate: {config.decay_rate}\n")
    
    @staticmethod
    def save_predictions(
        predictor: TabularPredictor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_dir: str,
        metadata_df: Optional[pd.DataFrame] = None
    ):
        """
        Save test predictions to CSV.
        
        Args:
            predictor: TabularPredictor instance
            X_test: Test features DataFrame
            y_test: Test labels Series
            model_dir: Model directory path
            metadata_df: Optional DataFrame with fighter names and event_date
        """
        try:
            X_test_clean = X_test.drop(columns=['sample_weight'], errors='ignore')
            probs = predictor.predict_proba(X_test_clean)
            if isinstance(probs, pd.DataFrame):
                if '1' in probs.columns:
                    test_probs = probs['1'].values
                else:
                    test_probs = probs.iloc[:, 1].values
            else:
                test_probs = probs[:, 1]
            
            test_preds = predictor.predict(X_test_clean)
            if isinstance(test_preds, pd.Series):
                test_preds = test_preds.values
            
            if len(X_test) > 0 and metadata_df is not None:
                test_indices = X_test.index
                required_cols = ['fighter1_name', 'fighter2_name']
                if 'event_date' in metadata_df.columns:
                    required_cols.append('event_date')
                
                metadata_info = metadata_df.reindex(test_indices)[required_cols].copy()
                
                if len(metadata_info) == len(test_indices) and not metadata_info.isna().all().all():
                    predictions_df = pd.DataFrame({
                        'fighter1_name': metadata_info['fighter1_name'].values,
                        'fighter2_name': metadata_info['fighter2_name'].values,
                        'y_pred_proba': test_probs,
                        'y_pred': test_preds,
                        'y_true': y_test.values
                    })
                    
                    if 'event_date' in metadata_info.columns:
                        predictions_df['event_date'] = metadata_info['event_date'].values
                    
                    column_order = ['fighter1_name', 'fighter2_name', 'y_pred_proba', 'y_pred', 'y_true', 'event_date']
                    column_order = [col for col in column_order if col in predictions_df.columns]
                    predictions_df = predictions_df[column_order]
                else:
                    raise ValueError(f"Metadata indices don't align: {len(metadata_info)} vs {len(test_indices)}")
            else:
                predictions_df = pd.DataFrame({
                    'y_pred_proba': test_probs,
                    'y_pred': test_preds,
                    'y_true': y_test.values
                })
            
            predictions_path = os.path.join(model_dir, 'test_predictions.csv')
            predictions_df.to_csv(predictions_path, index=False)
            print(f"\nTest predictions saved to: {predictions_path}")
        except Exception as e:
            print(f"Warning: Could not save test predictions: {e}")
            import traceback
            traceback.print_exc()
    
    @staticmethod
    def prepare_evaluation_data(
        holdout_df: pd.DataFrame,
        features: List[str],
        normalize: str,
        scaler_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare holdout data for evaluation with proper normalization.
        
        Args:
            holdout_df: Holdout DataFrame
            features: List of feature names
            normalize: Normalization method
            scaler_path: Path to saved scaler
            
        Returns:
            Tuple of (X_holdout, y_holdout)
        """
        from libs.modeling.data_utils import load_training_data
        X_holdout, y_holdout = load_training_data(holdout_df, features)
        
        if normalize != 'none' and scaler_path and os.path.exists(scaler_path):
            scaler = NormalizationManager.load_scaler(scaler_path)
            features_to_scale = [
                col for col in X_holdout.columns 
                if not NormalizationManager._should_exclude_col(col)
            ]
            X_holdout = NormalizationManager.transform(scaler, X_holdout, features_to_scale)
        
        return X_holdout, y_holdout


class FileManager:
    """Handles file I/O operations for model training."""
    
    @staticmethod
    def create_model_directory(
        model_type: str,
        preset: str,
        suffix: str = ""
    ) -> str:
        """
        Create timestamped model directory.
        
        Args:
            model_type: Model type ('win' or 'decision')
            preset: Preset name ('extreme' or 'best')
            suffix: Optional suffix for directory name
            
        Returns:
            Path to created model directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = os.path.join(
            'AutogluonModels',
            f'ag-{timestamp}-{model_type}-{preset}{suffix}'
        )
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    @staticmethod
    def save_features(features: List[str], model_dir: str):
        """Save feature list to feats.txt."""
        feats_path = os.path.join(model_dir, 'feats.txt')
        with open(feats_path, 'w') as f:
            for feat in features:
                f.write(f"{feat}\n")
        print(f"Features saved to: {feats_path}")
    
    @staticmethod
    def save_holdout_ids(holdout_df: pd.DataFrame, model_dir: str):
        """Save holdout fight IDs to holdout_fight_ids.txt."""
        if len(holdout_df) == 0:
            return
        
        holdout_fight_ids_path = os.path.join(model_dir, 'holdout_fight_ids.txt')
        with open(holdout_fight_ids_path, 'w') as f:
            for fight_id in holdout_df['fight_id'].values:
                f.write(f"{fight_id}\n")
        print(f"Saved {len(holdout_df)} holdout fight IDs to {holdout_fight_ids_path}")
    
    @staticmethod
    def save_training_data(df: pd.DataFrame, model_dir: str):
        """Save training data CSV."""
        training_data_path = os.path.join(model_dir, 'training_data.csv')
        df.to_csv(training_data_path, index=False)
        print(f"Training data saved to: {training_data_path}")
    
    @staticmethod
    def save_test_start_date(holdout_df: pd.DataFrame, model_dir: str):
        """Save test start date for backward compatibility."""
        if len(holdout_df) == 0:
            return
        
        from libs.modeling.split_date_utils import write_test_start_date
        holdout_start_date = holdout_df['event_date'].min()
        test_start_path = os.path.join(model_dir, 'test_start_date.txt')
        write_test_start_date(holdout_start_date, test_start_path)
        print(f"Saved test start date: {holdout_start_date}")


# ============================================================================
# Trainer Classes
# ============================================================================

class TimeseriesSplitTrainer:
    """Handles time-series split training with chronological train/val split."""
    
    def __init__(self, config: TrainingConfig, model_dir: str, original_df: pd.DataFrame):
        """
        Initialize timeseries split trainer.
        
        Args:
            config: TrainingConfig instance
            model_dir: Model directory path
            original_df: Original DataFrame with event_date for temporal splitting
        """
        self.config = config
        self.model_dir = model_dir
        self.original_df = original_df
        self.train_scores = None
        self.tune_scores = None
    
    def split_data(
        self,
        timeseries_data: pd.DataFrame,
        y_timeseries: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data chronologically into train and validation sets.
        
        Args:
            timeseries_data: Features DataFrame
            y_timeseries: Labels Series
            
        Returns:
            Tuple of (train_data, tune_data, y_train, y_tune)
        """
        # Combine features and labels
        timeseries_data = timeseries_data.copy()
        timeseries_data['y_true'] = y_timeseries
        
        # Get event dates for temporal splitting
        if 'event_date' not in self.original_df.columns:
            raise ValueError("event_date column required for time-series split training")
        
        event_dates = self.original_df.reindex(timeseries_data.index)['event_date'].values
        
        # Verify chronological order
        if not np.all(event_dates[:-1] <= event_dates[1:]):
            raise ValueError("Data is not in chronological order! This could cause data leakage.")
        
        # Determine validation start date
        if self.config.val_date:
            val_start_date = pd.Timestamp(self.config.val_date)
        else:
            val_start_date = pd.Timestamp('2024-01-01') if self.config.test_size else pd.Timestamp('2025-01-01')
        val_end_date = pd.Timestamp(self.config.test_size) if self.config.test_size else None
        
        # Create date masks
        train_mask = event_dates < val_start_date
        if val_end_date:
            val_mask = (event_dates >= val_start_date) & (event_dates < val_end_date)
        else:
            val_mask = event_dates >= val_start_date
        
        # Get indices
        train_indices = timeseries_data.index[train_mask]
        tune_indices = timeseries_data.index[val_mask]
        
        # Validate splits
        if len(train_indices) == 0:
            raise ValueError(f"No training data found before validation start date {val_start_date}")
        if len(tune_indices) == 0:
            if val_end_date:
                raise ValueError(f"No validation data found between {val_start_date} and {val_end_date}")
            else:
                raise ValueError(f"No validation data found on or after {val_start_date}")
        
        # Split data
        train_data_pre_norm = timeseries_data.loc[train_indices].copy()
        tune_data_pre_norm = timeseries_data.loc[tune_indices].copy()
        
        train_dates = event_dates[train_mask]
        tune_dates = event_dates[val_mask]
        
        # Verify no data leakage
        if len(train_dates) > 0 and len(tune_dates) > 0:
            if train_dates.max() >= tune_dates.min():
                raise ValueError(f"Data leakage detected! Train max date ({train_dates.max()}) >= Val min date ({tune_dates.min()})")
        
        return train_data_pre_norm, tune_data_pre_norm, train_dates, tune_dates, val_start_date, val_end_date
    
    def _prepare_training_data(
        self,
        train_data_pre_norm: pd.DataFrame,
        tune_data_pre_norm: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply normalization and prepare training data.
        
        Args:
            train_data_pre_norm: Training data before normalization
            tune_data_pre_norm: Validation data before normalization
            
        Returns:
            Tuple of (train_data, tune_data) after normalization
        """
        if self.config.normalize != 'none':
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            
            # Extract sample weights before normalization
            train_sample_weight = train_data_pre_norm.get('sample_weight', None)
            tune_sample_weight = tune_data_pre_norm.get('sample_weight', None)
            
            # Extract features for normalization
            X_train_features = train_data_pre_norm.drop(columns=['y_true', 'sample_weight'], errors='ignore')
            X_tune_features = tune_data_pre_norm.drop(columns=['y_true', 'sample_weight'], errors='ignore')
            
            # Fit scaler on training data only
            scaler, features_to_scale = NormalizationManager.fit_scaler(X_train_features, self.config.normalize)
            
            # Transform both datasets
            X_train_norm = NormalizationManager.transform(scaler, X_train_features, features_to_scale)
            X_tune_norm = NormalizationManager.transform(scaler, X_tune_features, features_to_scale)
            
            # Save scaler
            NormalizationManager.save_scaler(scaler, scaler_path)
            
            # Recombine with labels and sample weights
            train_data = X_train_norm.copy()
            train_data['y_true'] = train_data_pre_norm['y_true']
            if train_sample_weight is not None:
                train_data['sample_weight'] = train_sample_weight
            
            tune_data = X_tune_norm.copy()
            tune_data['y_true'] = tune_data_pre_norm['y_true']
            if self.config.use_recency_weights:
                tune_data['sample_weight'] = 1.0
            elif tune_sample_weight is not None:
                tune_data['sample_weight'] = tune_sample_weight
            
            print(f"Normalization applied (scaler fit on training data only to prevent data leakage)")
            print(f"Sample weights excluded from normalization and preserved separately")
        else:
            train_data = train_data_pre_norm.copy()
            tune_data = tune_data_pre_norm.copy()
            if self.config.use_recency_weights and 'sample_weight' not in tune_data.columns:
                tune_data['sample_weight'] = 1.0
        
        return train_data, tune_data
    
    def _create_fit_kwargs(self) -> Dict:
        """Create fit_kwargs for AutoGluon predictor."""
        # Map 'extreme' to 'extreme_quality' for AutoGluon
        preset = 'extreme_quality' if self.config.preset == 'extreme' else self.config.preset
        
        fit_kwargs = {
            'presets': preset,
            'time_limit': self.config.time_limit,
            'num_bag_folds': 0,  # Disables bagging/CV folds
            'num_stack_levels': 0,  # Disables stacking
            'auto_stack': False,  # Overrides preset's auto_stack=True
            'dynamic_stacking': False,  # Disables dynamic stacking validation splits
        }
        
        if self.config.included_model_types:
            fit_kwargs['included_model_types'] = self.config.included_model_types
        
        return fit_kwargs
    
    def _handle_refit_full(
        self,
        predictor: TabularPredictor,
        train_data: pd.DataFrame,
        tune_data: pd.DataFrame
    ) -> Tuple[TabularPredictor, Optional[dict]]:
        """
        Handle AutoGluon's refit_full() call if requested.
        
        Args:
            predictor: Trained TabularPredictor
            train_data: Training data
            tune_data: Validation data
            
        Returns:
            Tuple of (updated predictor, refit_train_scores dict or None if refit_full not enabled)
        """
        if not self.config.refit_full:
            return predictor, None
        
        print(f"\n--- Calling refit_full() to Retrain on All Data (Train + Val) ---")
        refit_result = predictor.refit_full()
        
        if isinstance(refit_result, dict):
            print(f"refit_full() returned model paths: {list(refit_result.keys())}")
            predictor = TabularPredictor.load(self.model_dir)
        elif refit_result is not None:
            predictor = refit_result
        
        # Evaluate refitted model on train+val combined
        full_train_data = pd.concat([train_data, tune_data], axis=0)
        refit_train_scores = predictor.evaluate(full_train_data)
        
        return predictor, refit_train_scores
    
    def train(
        self,
        X_timeseries: pd.DataFrame,
        y_timeseries: pd.Series
    ) -> TabularPredictor:
        """
        Train time-series split model.
        
        Args:
            X_timeseries: Features DataFrame
            y_timeseries: Labels Series
            
        Returns:
            Trained TabularPredictor instance
        """
        print(f"\n=== Time-Series Split Training ===")
        print("Time-series split mode enabled — using time-respecting train/tune split...")
        print(f"Preset: {self.config.preset}")
        print(f"Time limit: {self.config.time_limit}s")
        
        # Split data chronologically
        train_data_pre_norm, tune_data_pre_norm, train_dates, tune_dates, val_start_date, val_end_date = self.split_data(
            X_timeseries, y_timeseries
        )
        
        # Prepare training data (normalization)
        train_data, tune_data = self._prepare_training_data(train_data_pre_norm, tune_data_pre_norm)
        
        # Print split structure
        print(f"\nTime-series split structure:")
        if self.config.test_size:
            print(f"  Train: {len(train_data)} samples (before {val_start_date.strftime('%Y-%m-%d')})")
            print(f"  Val:   {len(tune_data)} samples ({val_start_date.strftime('%Y-%m-%d')} to {val_end_date.strftime('%Y-%m-%d')}, exclusive)")
            print(f"  Test:  Holdout set ({pd.Timestamp(self.config.test_size).strftime('%Y-%m-%d')} onward)")
        else:
            print(f"  Train: {len(train_data)} samples (before {val_start_date.strftime('%Y-%m-%d')})")
            print(f"  Val:   {len(tune_data)} samples ({val_start_date.strftime('%Y-%m-%d')} onward)")
        print(f"  Train date range: {train_dates.min()} to {train_dates.max()}")
        print(f"  Val date range:   {tune_dates.min()} to {tune_dates.max()}")
        
        # Create predictor
        sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
        predictor = TabularPredictor(
            label='y_true',
            eval_metric='log_loss',
            problem_type='binary',
            path=self.model_dir,
            verbosity=2,
            sample_weight=sample_weight_col,
            weight_evaluation=False
        )
        
        # Train model
        fit_kwargs = self._create_fit_kwargs()
        fit_kwargs['train_data'] = train_data
        fit_kwargs['tuning_data'] = tune_data
        
        print(f"\n--- Training Model ---")
        predictor.fit(**fit_kwargs)
        
        # Evaluate
        train_scores = EvaluationManager.evaluate_model(predictor, train_data)
        tune_scores = EvaluationManager.evaluate_model(predictor, tune_data)
        
        # Store scores (will be printed together after refit_full)
        self.train_scores = train_scores
        self.tune_scores = tune_scores
        
        # Handle refit_full if requested
        predictor, refit_train_scores = self._handle_refit_full(predictor, train_data, tune_data)
        
        # Store refit scores for later printing
        self.refit_train_scores = refit_train_scores
        
        return predictor


class RefitAllTrainer:
    """Handles refitting model on all available data (including holdout)."""
    
    def __init__(
        self,
        config: TrainingConfig,
        original_model_dir: str,
        original_df: pd.DataFrame,
        holdout_df: pd.DataFrame,
        selected_features: List[str],
        train_scores: Optional[Dict] = None,
        tune_scores: Optional[Dict] = None
    ):
        """
        Initialize refit all trainer.
        
        Args:
            config: TrainingConfig instance
            original_model_dir: Original model directory path
            original_df: Original training DataFrame
            holdout_df: Holdout DataFrame
            selected_features: List of selected feature names
            train_scores: Optional initial training scores for comparison
            tune_scores: Optional initial validation scores for comparison
        """
        self.config = config
        self.original_model_dir = original_model_dir
        self.original_df = original_df
        self.holdout_df = holdout_df
        self.selected_features = selected_features
        self.train_scores = train_scores
        self.tune_scores = tune_scores
    
    def prepare_refit_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Combine original and holdout data, prepare features.
        
        Returns:
            Tuple of (full_df, X_full, y_full)
        """
        # Combine original and holdout data
        full_df = pd.concat([self.original_df, self.holdout_df], ignore_index=True)
        full_df = full_df.sort_values('event_date').reset_index(drop=True)
        
        print(f"Combined data: {len(full_df)} samples")
        print(f"Date range: {full_df['event_date'].min()} to {full_df['event_date'].max()}")
        
        # Load features
        X_full, y_full = DataLoader.load_training_features(full_df, self.selected_features)
        
        return full_df, X_full, y_full
    
    def train(self) -> TabularPredictor:
        """
        Train refitted model on all data.
        
        Returns:
            Refitted TabularPredictor instance
        """
        if self.config.split_strategy != 'timeseries_split':
            raise ValueError(
                f"refit_all is only supported for timeseries_split strategy, "
                f"got {self.config.split_strategy}"
            )
        
        print(f"\n{'='*70}")
        print("=== REFIT FULL: Retraining on All Available Data ===")
        print(f"{'='*70}")
        
        # Create refit directory
        refit_dir = f"{self.original_model_dir}_refit_all"
        if os.path.exists(refit_dir):
            import shutil
            shutil.rmtree(refit_dir)
        os.makedirs(refit_dir, exist_ok=True)
        print(f"Refit model directory: {refit_dir}")
        
        # Prepare refit data
        full_df, X_full, y_full = self.prepare_refit_data()
        
        # Apply recency weights if enabled
        if self.config.use_recency_weights:
            from libs.modeling.data_utils import calculate_recency_weights
            weights = calculate_recency_weights(
                full_df.loc[X_full.index],
                X_full.index,
                decay_rate=self.config.decay_rate
            )
            X_full['sample_weight'] = weights
        
        # Combine features and labels
        full_data = X_full.copy()
        full_data['y_true'] = y_full
        
        # Use TimeseriesSplitTrainer for the actual training
        # Create a temporary DataFrame with event_date for splitting
        temp_df = full_df.copy()
        trainer = TimeseriesSplitTrainer(self.config, refit_dir, temp_df)
        
        # Split and train
        train_data_pre_norm, tune_data_pre_norm, train_dates, tune_dates, val_start_date, val_end_date = trainer.split_data(
            X_full, y_full
        )
        
        # Prepare training data (normalization)
        train_data, tune_data = trainer._prepare_training_data(train_data_pre_norm, tune_data_pre_norm)
        
        # Print split structure
        print(f"\nRefit split structure:")
        print(f"  Train: {len(train_data)} samples (before {val_start_date.strftime('%Y-%m-%d')})")
        print(f"  Val:   {len(tune_data)} samples ({val_start_date.strftime('%Y-%m-%d')} onward)")
        print(f"  Train date range: {train_dates.min()} to {train_dates.max()}")
        print(f"  Val date range:   {tune_dates.min()} to {tune_dates.max()}")
        
        # Create predictor
        sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
        refit_predictor = TabularPredictor(
            label='y_true',
            eval_metric='log_loss',
            problem_type='binary',
            path=refit_dir,
            verbosity=2,
            sample_weight=sample_weight_col,
            weight_evaluation=False
        )
        
        # Train model
        fit_kwargs = trainer._create_fit_kwargs()
        fit_kwargs['train_data'] = train_data
        fit_kwargs['tuning_data'] = tune_data
        
        print(f"\n--- Training Refit Model ---")
        refit_predictor.fit(**fit_kwargs)
        
        # Evaluate
        train_scores_refit = EvaluationManager.evaluate_model(refit_predictor, train_data)
        print(f"\nRefit Train set - Accuracy: {train_scores_refit['accuracy']:.4f}, "
              f"Log Loss: {train_scores_refit['log_loss']:.4f}")
        
        tune_scores_refit = EvaluationManager.evaluate_model(refit_predictor, tune_data)
        print(f"\nRefit Tune set (Validation) - Accuracy: {tune_scores_refit['accuracy']:.4f}, "
              f"Log Loss: {tune_scores_refit['log_loss']:.4f}")
        
        # Compare with initial scores
        if self.train_scores and self.tune_scores:
            print(f"\nComparison with initial model:")
            print(f"  Initial Train Accuracy: {self.train_scores['accuracy']:.4f}")
            print(f"  Initial Tune Accuracy: {self.tune_scores['accuracy']:.4f}")
            print(f"  Refit Train Accuracy: {train_scores_refit['accuracy']:.4f} (Δ{train_scores_refit['accuracy'] - self.train_scores['accuracy']:+.4f})")
            print(f"  Refit Tune Accuracy: {tune_scores_refit['accuracy']:.4f} (Δ{tune_scores_refit['accuracy'] - self.tune_scores['accuracy']:+.4f})")
        
        # Handle refit_full
        refit_predictor, _ = trainer._handle_refit_full(refit_predictor, train_data, tune_data)
        
        # Save evaluation results
        refit_results = {
            'train_scores': train_scores_refit,
            'val_scores': tune_scores_refit,
            'test_scores': None,
            'best_model': refit_predictor.model_best,
            'feature_importance': None
        }
        
        # Get model weights
        weights = None
        try:
            _, weights = self._get_model_weights(refit_predictor)
        except Exception:
            pass
        
        # Save evaluations
        EvaluationManager.save_evaluations(
            refit_results,
            self.config,
            refit_dir,
            refit_predictor.model_best,
            weights=weights
        )
        
        # Save features and training data
        FileManager.save_features(self.selected_features, refit_dir)
        FileManager.save_training_data(full_df, refit_dir)
        
        return refit_predictor
    
    @staticmethod
    def _get_model_weights(predictor: TabularPredictor) -> Tuple[str, Optional[dict]]:
        """Extract model weights from predictor."""
        best_model = predictor.model_best
        weights = None
        
        try:
            best_model_info = predictor.info()['model_info'][best_model]
            if 'children_info' in best_model_info:
                if 'S1F1' in best_model_info['children_info']:
                    if 'model_weights' in best_model_info['children_info']['S1F1']:
                        weights = best_model_info['children_info']['S1F1']['model_weights']
                else:
                    for child_key, child_info in best_model_info['children_info'].items():
                        if 'model_weights' in child_info:
                            weights = child_info['model_weights']
                            break
        except (KeyError, TypeError):
            pass
        
        return best_model, weights


class WalkForwardFoldGenerator:
    """Generates walk-forward validation folds based on date boundaries."""
    
    def __init__(self, config: TrainingConfig, full_df: pd.DataFrame):
        """
        Initialize fold generator.
        
        Args:
            config: TrainingConfig instance
            full_df: Full DataFrame with event_date column
        """
        self.config = config
        self.full_df = full_df
    
    def create_folds(self, X_full: pd.DataFrame) -> List[Dict]:
        """
        Create walk-forward validation folds based on date boundaries.
        
        Args:
            X_full: Features DataFrame (for index alignment)
            
        Returns:
            List of fold dictionaries with train/val indices and date ranges
        """
        folds = []
        
        # Create folds based on walkforward_initial_year and n_windows
        for window_idx in range(self.config.walkforward_n_windows):
            # Calculate train cutoff date (end of year)
            train_cutoff_year = self.config.walkforward_initial_year + window_idx
            train_cutoff_date = pd.Timestamp(f'{train_cutoff_year}-12-31')
            
            # Calculate validation period (next year)
            val_start_year = train_cutoff_year + 1
            val_end_year = val_start_year
            val_end_date = pd.Timestamp(f'{val_end_year}-12-31')
            
            # Create masks for train and validation
            train_mask = self.full_df['event_date'] <= train_cutoff_date
            val_mask = (self.full_df['event_date'] > train_cutoff_date) & (self.full_df['event_date'] <= val_end_date)
            
            train_indices = self.full_df[train_mask].index
            val_indices = self.full_df[val_mask].index
            
            # Skip if validation set is empty
            if len(val_indices) == 0:
                print(f"Warning: Window {window_idx + 1} has no validation data, skipping...")
                continue
            
            # Align with X_full indices
            train_indices = train_indices[train_indices.isin(X_full.index)]
            val_indices = val_indices[val_indices.isin(X_full.index)]
            
            if len(train_indices) == 0 or len(val_indices) == 0:
                print(f"Warning: Window {window_idx + 1} has empty train or val after alignment, skipping...")
                continue
            
            folds.append({
                'window_idx': window_idx,
                'train_indices': train_indices,
                'val_indices': val_indices,
                'train_cutoff_date': train_cutoff_date,
                'val_start_date': pd.Timestamp(f'{val_start_year}-01-01'),
                'val_end_date': val_end_date,
                'train_range': f"≤ {train_cutoff_date.strftime('%Y-%m-%d')}",
                'val_range': f"{val_start_year}"
            })
        
        return folds
    
    def validate_folds(self, folds: List[Dict]) -> bool:
        """
        Validate that folds have sufficient data.
        
        Args:
            folds: List of fold dictionaries
            
        Returns:
            True if all folds are valid
        """
        if len(folds) == 0:
            return False
        
        for fold in folds:
            if len(fold['train_indices']) == 0 or len(fold['val_indices']) == 0:
                return False
        
        return True


class WalkForwardTrainer:
    """
    Walk-forward validation trainer that trains multiple expanding window models
    and a final production model, then ensembles predictions via simple arithmetic mean.
    
    Implements strict temporal splits with 5 expanding windows + 1 final model:
    - Window 1: train ≤ 2020-12-31 → val 2021
    - Window 2: train ≤ 2021-12-31 → val 2022
    - Window 3: train ≤ 2022-12-31 → val 2023
    - Window 4: train ≤ 2023-12-31 → val 2024
    - Window 5: train ≤ 2024-12-31 → val latest (current date in 2025)
    - Final model: train on everything ≤ last completed fight (no validation)
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize walk-forward trainer with configuration."""
        self.config = config
        self.model_dir = None
        self.selected_features = None
        self.full_df = None
        self.window_models: List[TabularPredictor] = []
        self.window_results: List[Dict] = []
        self.ensemble_predictor: Optional[EnsemblePredictor] = None
    
    def _load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare full dataset for walk-forward validation, excluding holdout."""
        data_path = DataLoader.get_data_path(self.config.model_type)
        
        print(f"\n=== Walk-Forward Data Preparation ===")
        print(f"Model type: {self.config.model_type}")
        print(f"Data path: {data_path}")
        
        # Load and filter data
        full_df = DataLoader.load_and_filter_data(
            data_path,
            self.config.num_fights,
            self.config.start_date,
            self.config.include_split_dec
        )
        
        # Create holdout set
        holdout_df = DataLoader.create_holdout_set(full_df, self.config.test_size)
        
        # Exclude holdout from training data
        if len(holdout_df) > 0:
            holdout_indices = set(holdout_df.index)
            training_df = full_df[~full_df.index.isin(holdout_indices)].copy()
            print(f"Training data (excluding holdout): {len(training_df)} samples")
        else:
            training_df = full_df.copy()
            print(f"Warning: No holdout set created, using all data for training")
        
        # Store holdout for later evaluation
        self.holdout_data = holdout_df
        
        print(f"Filtered data: {len(training_df)} samples")
        print(f"Date range: {training_df['event_date'].min()} to {training_df['event_date'].max()}")
        
        # Prepare features
        features = DataLoader.prepare_features(training_df, self.config)
        print(f"Selected {len(features)} features")
        
        # Load training data
        X_full, y_full = DataLoader.load_training_features(training_df, features)
        
        # Store for later use
        self.selected_features = features
        self.full_df = training_df
        self.full_df_with_holdout = full_df
        
        return X_full, y_full
    
    def _create_walkforward_folds(self, X_full: pd.DataFrame, y_full: pd.Series) -> List[Dict]:
        """
        Create walk-forward validation folds based on date boundaries.
        
        Returns list of fold dictionaries with train/test indices and date ranges.
        """
        generator = WalkForwardFoldGenerator(self.config, self.full_df)
        folds = generator.create_folds(X_full)
        
        if not generator.validate_folds(folds):
            raise ValueError("Invalid folds generated - some folds have empty train or validation sets")
        
        return folds
    
    def _train_window_model(self, fold: Dict, X_full: pd.DataFrame, y_full: pd.Series, window_idx: int) -> Tuple[TabularPredictor, Dict]:
        """
        Train a single window model with refit_all.
        
        Returns:
            Tuple of (trained predictor, evaluation results)
        """
        print(f"\n{'='*70}")
        print(f"=== Training Window {window_idx + 1}/{self.config.walkforward_n_windows} ===")
        print(f"{'='*70}")
        print(f"Train: {fold['train_range']} (n={len(fold['train_indices'])})")
        print(f"Val:   {fold['val_range']} (n={len(fold['val_indices'])})")
        
        # Extract train and validation data
        X_train = X_full.loc[fold['train_indices']].copy()
        y_train = y_full.loc[fold['train_indices']].copy()
        X_val = X_full.loc[fold['val_indices']].copy()
        y_val = y_full.loc[fold['val_indices']].copy()
        
        # Apply normalization per-window (fit on train, transform val)
        window_dir = os.path.join(self.model_dir, f'window_{window_idx}')
        os.makedirs(window_dir, exist_ok=True)
        scaler_path = os.path.join(window_dir, 'scaler.pkl')
        
        if self.config.normalize != 'none':
            import joblib
            from libs.modeling.data_utils import (
                apply_robust_normalization,
                apply_zscore_normalization,
                apply_no_normalization
            )
            
            if self.config.normalize == 'robust':
                X_train, X_val = apply_robust_normalization(X_train, X_val, scaler_path)
            elif self.config.normalize == 'zscore':
                X_train, X_val = apply_zscore_normalization(X_train, X_val, scaler_path)
            else:
                X_train, X_val = apply_no_normalization(X_train, X_val, scaler_path)
        
        # Apply recency weights if enabled
        if self.config.use_recency_weights:
            from libs.modeling.data_utils import calculate_recency_weights
            train_df = self.full_df.loc[X_train.index]
            weights = calculate_recency_weights(
                train_df,
                X_train.index,
                decay_rate=self.config.decay_rate
            )
            X_train['sample_weight'] = weights
            X_val['sample_weight'] = 1.0  # Uniform for validation
        
        # Prepare training data
        train_data = X_train.copy()
        train_data['y_true'] = y_train
        
        val_data = X_val.copy()
        val_data['y_true'] = y_val
        
        # Create predictor
        sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
        predictor = TabularPredictor(
            label='y_true',
            eval_metric='log_loss',
            problem_type='binary',
            path=window_dir,
            verbosity=2,
            sample_weight=sample_weight_col,
            weight_evaluation=False
        )
        
        # Train with 'extreme' preset, no bagging, 2 stacking layers, no shuffling
        fit_kwargs = {
            'train_data': train_data,
            'tuning_data': val_data,
            'presets': 'extreme_quality',  # Corrected to full preset name; use 'extreme' if that's a custom alias
            'time_limit': self.config.time_limit,
            'num_bag_folds': 0,  # Disables bagging/CV folds
            'num_stack_levels': 0,  # Disables stacking
            'auto_stack': False,  # Overrides preset's auto_stack=True
            'dynamic_stacking': False,  # Disables dynamic stacking validation splits
        }
        
        if self.config.included_model_types:
            fit_kwargs['included_model_types'] = self.config.included_model_types
        
        print(f"\n--- Training Window {window_idx + 1} Model ---")
        predictor.fit(**fit_kwargs)
        
        # Evaluate on training set (BEFORE refit_all)
        train_scores = predictor.evaluate(train_data)
        print(f"Window {window_idx + 1} Training - Accuracy: {train_scores['accuracy']:.4f}, "
              f"Log Loss: {train_scores['log_loss']:.4f}")
        
        # Evaluate on validation set (BEFORE refit_all)
        val_scores = predictor.evaluate(val_data)
        print(f"Window {window_idx + 1} Validation - Accuracy: {val_scores['accuracy']:.4f}, "
              f"Log Loss: {val_scores['log_loss']:.4f}")
        
        # Refit on full training data (always refit_all for walk-forward, independent of config.refit_all setting)
        print(f"--- Refitting Window {window_idx + 1} Model on Full Training Data ---")
        refit_result = predictor.refit_full()
        
        # Handle refit_full return value - it may return a dict or a predictor
        if isinstance(refit_result, dict):
            # refit_full returned a dict of model paths, reload predictor to use refitted models
            print(f"Refit returned dict with {len(refit_result)} models, reloading predictor")
            predictor = TabularPredictor.load(window_dir)
        else:
            # refit_full returned a new predictor
            predictor = refit_result
        
        # Evaluate on test/holdout set if available
        test_scores = None
        X_test = None
        y_test = None
        if self.holdout_data is not None and len(self.holdout_data) > 0:
            # Prepare test data with same normalization/scaling as this window
            from libs.modeling.data_utils import load_training_data
            X_test_raw, y_test_raw = load_training_data(self.holdout_data, self.selected_features)
            
            # Apply normalization using this window's scaler
            if self.config.normalize != 'none' and os.path.exists(scaler_path):
                import joblib
                try:
                    window_scaler = joblib.load(scaler_path)
                    date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
                    categorical_static_feats = ['weightclass_encoded', 'odds']
                    
                    def should_exclude_col(col_name):
                        if col_name in date_cols:
                            return True
                        for cat_feat in categorical_static_feats:
                            if cat_feat in col_name:
                                return True
                        return False
                    
                    features_to_scale = [col for col in X_test_raw.columns 
                                        if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
                    
                    X_test = X_test_raw.copy()
                    if len(features_to_scale) > 0:
                        X_test[features_to_scale] = window_scaler.transform(X_test_raw[features_to_scale])
                except Exception as e:
                    print(f"Warning: Could not apply scaler to test data: {e}")
                    X_test = X_test_raw.copy()
            else:
                X_test = X_test_raw.copy()
            
            # Add sample_weight if needed
            if self.config.use_recency_weights:
                X_test['sample_weight'] = 1.0
            
            y_test = y_test_raw
            
            # Evaluate on test set
            test_data_eval = X_test.copy()
            test_data_eval['y_true'] = y_test
            test_scores = predictor.evaluate(test_data_eval)
            print(f"Window {window_idx + 1} Test - Accuracy: {test_scores['accuracy']:.4f}, "
                  f"Log Loss: {test_scores['log_loss']:.4f}")
        
        # Save window evaluations (train, val, test)
        self._save_window_evaluations(
            window_dir, window_idx, train_scores, val_scores, test_scores,
            X_train, y_train, X_val, y_val, X_test, y_test, predictor
        )
        
        # Save window results
        window_result = {
            'window_idx': window_idx,
            'train_range': fold['train_range'],
            'val_range': fold['val_range'],
            'train_size': len(X_train),
            'val_size': len(X_val),
            'train_cutoff_date': fold['train_cutoff_date'].strftime('%Y-%m-%d'),
            'val_start_date': fold['val_start_date'].strftime('%Y-%m-%d'),
            'val_end_date': fold['val_end_date'].strftime('%Y-%m-%d'),
            'train_accuracy': train_scores['accuracy'],
            'train_log_loss': train_scores['log_loss'],
            'accuracy': val_scores['accuracy'],
            'log_loss': val_scores['log_loss'],
            'precision': val_scores.get('precision', 0),
            'recall': val_scores.get('recall', 0),
            'f1': val_scores.get('f1', 0),
            'brier_score': val_scores.get('brier_score', 0)
        }
        if test_scores:
            window_result['test_accuracy'] = test_scores['accuracy']
            window_result['test_log_loss'] = test_scores['log_loss']
        
        return predictor, window_result
    
    def _get_model_weights(self, predictor: TabularPredictor) -> Tuple[str, Optional[dict]]:
        """
        Extract model weights from a TabularPredictor if it's a weighted ensemble.
        
        Returns:
            Tuple of (best_model_name, weights_dict) where weights_dict is None if not a weighted ensemble
        """
        best_model = predictor.model_best
        weights = None
        
        try:
            best_model_info = predictor.info()['model_info'][best_model]
            if 'children_info' in best_model_info:
                # Check for weighted ensemble at S1F1 (most common path)
                if 'S1F1' in best_model_info['children_info']:
                    if 'model_weights' in best_model_info['children_info']['S1F1']:
                        weights = best_model_info['children_info']['S1F1']['model_weights']
                else:
                    # Try to find any weighted ensemble in children_info
                    for child_key, child_info in best_model_info['children_info'].items():
                        if 'model_weights' in child_info:
                            weights = child_info['model_weights']
                            break
        except (KeyError, TypeError):
            # Not a weighted ensemble or weights not found
            pass
        
        return best_model, weights
    
    def _save_window_evaluations(self, window_dir: str, window_idx: int, 
                                 train_scores: dict, val_scores: dict, test_scores: Optional[dict],
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series,
                                 X_test: Optional[pd.DataFrame], y_test: Optional[pd.Series],
                                 predictor: TabularPredictor):
        """
        Save evaluations for a window model: train, val, and test (if exists).
        Also saves vegas comparison if test exists.
        
        Args:
            window_dir: Directory for this window model
            window_idx: Window index
            train_scores: Training evaluation scores
            val_scores: Validation evaluation scores
            test_scores: Test evaluation scores (None if no test)
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data (None if no test)
            predictor: Trained predictor for this window
        """
        # Save evals.txt
        evals_path = os.path.join(window_dir, 'evals.txt')
        with open(evals_path, 'w') as f:
            f.write("Model Performance:\n")
            f.write(f"Training accuracy: {train_scores['accuracy']:.4f}\n")
            f.write(f"Training log loss: {train_scores['log_loss']:.4f}\n")
            f.write(f"Validation accuracy: {val_scores['accuracy']:.4f}\n")
            f.write(f"Validation log loss: {val_scores['log_loss']:.4f}\n")
            if test_scores:
                f.write(f"Test accuracy: {test_scores['accuracy']:.4f}\n")
                f.write(f"Test log loss: {test_scores['log_loss']:.4f}\n")
            else:
                f.write("Test scores: N/A (no test set)\n")
            best_model, weights = self._get_model_weights(predictor)
            f.write(f"\nBest Model: {best_model}\n")
            
            # Write model weights if available
            if weights:
                f.write(f"\nModel weights:\n")
                for model_name, weight in weights.items():
                    f.write(f"{model_name}: {weight:.3f}\n")
            else:
                f.write(f"\nModel weights: N/A (not a weighted ensemble)\n")
            
            f.write(f"\nConfiguration:\n")
            f.write(f"Window: {window_idx + 1}\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Preset: {self.config.preset}\n")
            f.write(f"Time Limit: {self.config.time_limit}\n")
            f.write(f"Test Size (cutoff date): {self.config.test_size if self.config.test_size else 'None (no holdout)'}\n")
            f.write(f"Split Strategy: {self.config.split_strategy}\n")
            f.write(f"Normalize: {self.config.normalize}\n")
            f.write(f"Use Recency Weights: {self.config.use_recency_weights}\n")
            f.write(f"Decay Rate: {self.config.decay_rate}\n")
        
        print(f"\nWindow {window_idx + 1} evaluations saved to: {evals_path}")
        
        # Save test predictions if test exists
        if test_scores is not None and X_test is not None and y_test is not None:
            try:
                X_test_clean = X_test.drop(columns=['sample_weight'], errors='ignore')
                probs = predictor.predict_proba(X_test_clean)
                if isinstance(probs, pd.DataFrame):
                    if '1' in probs.columns:
                        test_probs = probs['1'].values
                    else:
                        test_probs = probs.iloc[:, 1].values
                else:
                    test_probs = probs[:, 1]
                
                test_preds = predictor.predict(X_test_clean)
                if isinstance(test_preds, pd.Series):
                    test_preds = test_preds.values
                
                # Get metadata from holdout_data
                if len(X_test) > 0:
                    test_indices = X_test.index
                    required_cols = ['fighter1_name', 'fighter2_name']
                    
                    if hasattr(self, 'holdout_data') and self.holdout_data is not None and len(self.holdout_data) > 0:
                        metadata_df = self.holdout_data
                    else:
                        metadata_df = None
                    
                    if metadata_df is not None:
                        if 'event_date' in metadata_df.columns:
                            required_cols.append('event_date')
                        
                        metadata_info = metadata_df.reindex(test_indices)[required_cols].copy()
                        
                        if len(metadata_info) == len(test_indices) and not metadata_info.isna().all().all():
                            predictions_df = pd.DataFrame({
                                'fighter1_name': metadata_info['fighter1_name'].values,
                                'fighter2_name': metadata_info['fighter2_name'].values,
                                'y_pred_proba': test_probs,
                                'y_pred': test_preds,
                                'y_true': y_test.values
                            })
                            
                            if 'event_date' in metadata_info.columns:
                                predictions_df['event_date'] = metadata_info['event_date'].values
                        else:
                            raise ValueError(f"Metadata indices don't align: {len(metadata_info)} vs {len(test_indices)}")
                    else:
                        raise ValueError("No metadata dataframe available")
                    
                    column_order = ['fighter1_name', 'fighter2_name', 'y_pred_proba', 'y_pred', 'y_true', 'event_date']
                    column_order = [col for col in column_order if col in predictions_df.columns]
                    predictions_df = predictions_df[column_order]
                else:
                    predictions_df = pd.DataFrame({
                        'y_pred_proba': test_probs,
                        'y_pred': test_preds,
                        'y_true': y_test.values
                    })
                
                predictions_path = os.path.join(window_dir, 'test_predictions.csv')
                predictions_df.to_csv(predictions_path, index=False)
                print(f"Window {window_idx + 1} test predictions saved to: {predictions_path}")
            except Exception as e:
                print(f"Warning: Could not save test predictions for window {window_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
            
            # Save vegas comparison using ModelUtils (if test exists)
            # Need to create a temporary training_data.csv and holdout_fight_ids.txt for this window
            try:
                # Save window-specific training data (for ModelUtils)
                window_training_data_path = os.path.join(window_dir, 'training_data.csv')
                # Use full_df_with_holdout to include all data
                df_to_save = self.full_df_with_holdout if hasattr(self, 'full_df_with_holdout') else self.full_df
                df_to_save.to_csv(window_training_data_path, index=False)
                
                # Save holdout fight IDs for this window
                holdout_fight_ids_path = os.path.join(window_dir, 'holdout_fight_ids.txt')
                with open(holdout_fight_ids_path, 'w') as f:
                    for fight_id in self.holdout_data['fight_id'].values:
                        f.write(f"{fight_id}\n")
                
                scaler_path = os.path.join(window_dir, 'scaler.pkl')
                
                if os.path.exists(window_training_data_path) and os.path.exists(scaler_path):
                    from libs.modeling.model_utils import ModelUtils
                    utils = ModelUtils(
                        model_path=window_dir,
                        training_data_path=window_training_data_path,
                        scaler_path=scaler_path,
                        feats=self.selected_features or [],
                        predictor=predictor
                    )
                    utils.save_model_stats()
                    utils.plot_calibration_curve()
                    print(f"Window {window_idx + 1} vegas comparison and calibration curve saved")
            except Exception as e:
                print(f"Warning: Could not save vegas comparison for window {window_idx + 1}: {e}")
                import traceback
                traceback.print_exc()
    
    def _save_results(self):
        """Save walk-forward validation results and summary."""
        # Save summary JSON
        summary = {
            'config': {
                'model_type': self.config.model_type,
                'preset': self.config.preset,
                'time_limit': self.config.time_limit,
                'walkforward_n_windows': self.config.walkforward_n_windows,
                'walkforward_initial_year': self.config.walkforward_initial_year,
                'normalize': self.config.normalize,
                'use_recency_weights': self.config.use_recency_weights,
                'decay_rate': self.config.decay_rate,
                'included_model_types': self.config.included_model_types
            },
            'window_results': self.window_results,
            'aggregate_metrics': {
                'mean_accuracy': np.mean([r['accuracy'] for r in self.window_results]),
                'std_accuracy': np.std([r['accuracy'] for r in self.window_results]),
                'mean_log_loss': np.mean([r['log_loss'] for r in self.window_results]),
                'std_log_loss': np.std([r['log_loss'] for r in self.window_results])
            }
        }
        
        summary_path = os.path.join(self.model_dir, 'walkforward_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nWalk-forward summary saved to: {summary_path}")
        
        # Save window results CSV
        if self.window_results:
            results_df = pd.DataFrame(self.window_results)
            results_path = os.path.join(self.model_dir, 'window_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Window results saved to: {results_path}")
        
        # Save features list
        if self.selected_features:
            feats_path = os.path.join(self.model_dir, 'feats.txt')
            with open(feats_path, 'w') as f:
                for feat in self.selected_features:
                    f.write(f"{feat}\n")
            print(f"Features saved to: {feats_path}")
        
        # Save training data CSV (includes all data, holdout will be filtered by ModelUtils)
        training_data_path = os.path.join(self.model_dir, 'training_data.csv')
        # Save full_df_with_holdout if available, otherwise full_df
        df_to_save = self.full_df_with_holdout if hasattr(self, 'full_df_with_holdout') else self.full_df
        df_to_save.to_csv(training_data_path, index=False)
        print(f"Training data saved to: {training_data_path}")
        
        # Save holdout fight IDs for ModelUtils
        if hasattr(self, 'holdout_data') and self.holdout_data is not None and len(self.holdout_data) > 0:
            holdout_fight_ids_path = os.path.join(self.model_dir, 'holdout_fight_ids.txt')
            with open(holdout_fight_ids_path, 'w') as f:
                for fight_id in self.holdout_data['fight_id'].values:
                    f.write(f"{fight_id}\n")
            print(f"Saved {len(self.holdout_data)} holdout fight IDs to {holdout_fight_ids_path}")
            
            # Also save test start date for backward compatibility
            from libs.modeling.split_date_utils import write_test_start_date
            holdout_start_date = self.holdout_data['event_date'].min()
            test_start_path = os.path.join(self.model_dir, 'test_start_date.txt')
            write_test_start_date(holdout_start_date, test_start_path)
            print(f"Saved test start date: {holdout_start_date}")
    
    def _save_ensemble_evaluations(self, train_scores: Optional[dict], val_scores: Optional[dict], 
                                   test_scores: Optional[dict], X_test: Optional[pd.DataFrame], 
                                   y_test: Optional[pd.Series]):
        """
        Save evaluation results and test predictions for ensemble (similar to ModelTrainer._save_results).
        
        Args:
            train_scores: Dictionary of training evaluation metrics (None if no train data)
            val_scores: Dictionary of validation evaluation metrics (None if no val data)
            test_scores: Dictionary of test evaluation metrics (None if no test)
            X_test: Test features DataFrame (None if no test)
            y_test: Test labels Series (None if no test)
        """
        # Save evaluation results (evals.txt)
        evals_path = os.path.join(self.model_dir, 'evals.txt')
        with open(evals_path, 'w') as f:
            f.write("Model Performance:\n")
            if train_scores is not None:
                f.write(f"Training accuracy: {train_scores['accuracy']:.4f}\n")
                f.write(f"Training log loss: {train_scores['log_loss']:.4f}\n")
            else:
                f.write("Training scores: N/A\n")
            if val_scores is not None:
                f.write(f"Validation accuracy: {val_scores['accuracy']:.4f}\n")
                f.write(f"Validation log loss: {val_scores['log_loss']:.4f}\n")
            else:
                f.write("Validation scores: N/A\n")
            if test_scores is not None:
                f.write(f"Test accuracy: {test_scores['accuracy']:.4f}\n")
                f.write(f"Test log loss: {test_scores['log_loss']:.4f}\n")
                if 'precision' in test_scores:
                    f.write(f"Test precision: {test_scores['precision']:.4f}\n")
                if 'recall' in test_scores:
                    f.write(f"Test recall: {test_scores['recall']:.4f}\n")
                if 'f1' in test_scores:
                    f.write(f"Test f1: {test_scores['f1']:.4f}\n")
                if 'brier_score' in test_scores:
                    f.write(f"Test brier score: {test_scores['brier_score']:.4f}\n")
            else:
                f.write("Test scores: N/A (no test set)\n")
            f.write(f"\nBest Model: Ensemble (averaged from {len(self.window_models)} windows)\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"Model Type: {self.config.model_type}\n")
            f.write(f"Preset: {self.config.preset}\n")
            f.write(f"Time Limit: {self.config.time_limit}\n")
            f.write(f"Test Size (cutoff date): {self.config.test_size if self.config.test_size else 'None (no holdout)'}\n")
            f.write(f"Split Strategy: {self.config.split_strategy}\n")
            f.write(f"Refit All: {self.config.refit_all}\n")
            f.write(f"Start Date: {self.config.start_date}\n")
            f.write(f"Normalize: {self.config.normalize}\n")
            f.write(f"Use Recency Weights: {self.config.use_recency_weights}\n")
            f.write(f"Decay Rate: {self.config.decay_rate}\n")
            f.write(f"Calculate Importance: {self.config.calculate_importance}\n")
            f.write(f"Included Model Types: {self.config.included_model_types}\n")
            f.write(f"Include Split Dec: {self.config.include_split_dec}\n")
            f.write(f"dec_avg rate: {self.config.decay_rate}\n")
            f.write(f"Walk-Forward Windows: {self.config.walkforward_n_windows}\n")
            f.write(f"Walk-Forward Initial Year: {self.config.walkforward_initial_year}\n")
        
        print(f"\nEnsemble evaluation results saved to: {evals_path}")
        
        # Save test predictions (test_predictions.csv) if test exists
        if test_scores is not None and X_test is not None and y_test is not None:
            try:
                X_test_clean = X_test.drop(columns=['sample_weight'], errors='ignore')
                
                # Get ensemble predictions (averaged from all folds + final model)
                probs = self.ensemble_predictor.predict_proba(X_test_clean)
                if isinstance(probs, pd.DataFrame):
                    if '1' in probs.columns:
                        test_probs = probs['1'].values
                    else:
                        test_probs = probs.iloc[:, 1].values
                else:
                    test_probs = probs[:, 1]
                
                # Get binary predictions
                test_preds = self.ensemble_predictor.predict(X_test_clean)
                if isinstance(test_preds, pd.Series):
                    test_preds = test_preds.values
                
                # Get fighter names and event_date from holdout_data
                if len(X_test) > 0:
                    test_indices = X_test.index
                    required_cols = ['fighter1_name', 'fighter2_name']
                    
                    # Use holdout_data for metadata
                    if hasattr(self, 'holdout_data') and self.holdout_data is not None and len(self.holdout_data) > 0:
                        metadata_df = self.holdout_data
                    else:
                        metadata_df = None
                    
                    if metadata_df is not None:
                        if 'event_date' in metadata_df.columns:
                            required_cols.append('event_date')
                        
                        # Use reindex to safely align indices
                        metadata_info = metadata_df.reindex(test_indices)[required_cols].copy()
                        
                        # Check if we got valid metadata
                        if len(metadata_info) == len(test_indices) and not metadata_info.isna().all().all():
                            # Build predictions dataframe
                            predictions_df = pd.DataFrame({
                                'fighter1_name': metadata_info['fighter1_name'].values,
                                'fighter2_name': metadata_info['fighter2_name'].values,
                                'y_pred_proba': test_probs,
                                'y_pred': test_preds,
                                'y_true': y_test.values
                            })
                            
                            # Add event_date if available
                            if 'event_date' in metadata_info.columns:
                                predictions_df['event_date'] = metadata_info['event_date'].values
                        else:
                            raise ValueError(f"Metadata indices don't align: {len(metadata_info)} vs {len(test_indices)}")
                    else:
                        raise ValueError("No metadata dataframe available")
                    
                    # Reorder columns to match requested order
                    column_order = ['fighter1_name', 'fighter2_name', 'y_pred_proba', 'y_pred', 'y_true', 'event_date']
                    # Only include columns that exist
                    column_order = [col for col in column_order if col in predictions_df.columns]
                    predictions_df = predictions_df[column_order]
                else:
                    # Fallback if no data
                    predictions_df = pd.DataFrame({
                        'y_pred_proba': test_probs,
                        'y_pred': test_preds,
                        'y_true': y_test.values
                    })
                    print("Warning: Original dataframe not available, fighter names and event_date not included")
                
                predictions_path = os.path.join(self.model_dir, 'test_predictions.csv')
                predictions_df.to_csv(predictions_path, index=False)
                print(f"\nEnsemble test predictions saved to: {predictions_path}")
                print(f"Columns: {list(predictions_df.columns)}")
            except Exception as e:
                print(f"Warning: Could not save test predictions: {e}")
                import traceback
                traceback.print_exc()
            
            # Save vegas comparison using ModelUtils (if test exists)
            try:
                # Save ensemble training data (for ModelUtils)
                ensemble_training_data_path = os.path.join(self.model_dir, 'training_data.csv')
                # Use full_df_with_holdout to include all data
                df_to_save = self.full_df_with_holdout if hasattr(self, 'full_df_with_holdout') else self.full_df
                df_to_save.to_csv(ensemble_training_data_path, index=False)
                
                # Save holdout fight IDs for ensemble
                holdout_fight_ids_path = os.path.join(self.model_dir, 'holdout_fight_ids.txt')
                with open(holdout_fight_ids_path, 'w') as f:
                    for fight_id in self.holdout_data['fight_id'].values:
                        f.write(f"{fight_id}\n")
                
                # For ensemble models, pass None as scaler_path since EnsemblePredictor handles scaling internally
                # ModelUtils will skip scaling if scaler_path is None and predictor is an EnsemblePredictor
                scaler_path = None  # Ensemble handles scaling internally
                
                if os.path.exists(ensemble_training_data_path):
                    from libs.modeling.model_utils import ModelUtils
                    from libs.modeling.train import EnsemblePredictorWrapper
                    # Wrap ensemble predictor for ModelUtils compatibility
                    wrapped_predictor = EnsemblePredictorWrapper(self.ensemble_predictor)
                    utils = ModelUtils(
                        model_path=self.model_dir,
                        training_data_path=ensemble_training_data_path,
                        scaler_path=scaler_path,  # None - ensemble handles scaling
                        feats=self.selected_features or [],
                        predictor=wrapped_predictor
                    )
                    utils.save_model_stats()
                    utils.plot_calibration_curve()
                    print(f"Ensemble vegas comparison and calibration curve saved")
            except Exception as e:
                print(f"Warning: Could not save vegas comparison for ensemble: {e}")
                import traceback
                traceback.print_exc()
    
    def train(self) -> EnsemblePredictor:
        """
        Execute full walk-forward validation pipeline.
        
        Returns:
            EnsemblePredictor that averages predictions from all window models + final model
        """
        # Create model directory
        self.model_dir = FileManager.create_model_directory(
            self.config.model_type,
            self.config.preset,
            suffix="-walkforward"
        )
        print(f"\n=== Walk-Forward Validation Pipeline ===")
        print(f"Model directory: {self.model_dir}")
        
        # Load and prepare data
        X_full, y_full = self._load_and_prepare_data()
        
        # Create folds
        folds = self._create_walkforward_folds(X_full, y_full)
        print(f"\nCreated {len(folds)} walk-forward validation folds")
        
        # Train each window model
        for fold in folds:
            predictor, window_result = self._train_window_model(
                fold, X_full, y_full, fold['window_idx']
            )
            self.window_models.append(predictor)
            self.window_results.append(window_result)
        
        # Create ensemble predictor with scaler paths
        all_models = self.window_models.copy()
        scaler_paths = []
        
        # Collect scaler paths for window models
        for i in range(len(self.window_models)):
            window_scaler_path = os.path.join(self.model_dir, f'window_{i}', 'scaler.pkl')
            scaler_paths.append(window_scaler_path if os.path.exists(window_scaler_path) else None)
        
        self.ensemble_predictor = EnsemblePredictor(all_models, self.model_dir, scaler_paths=scaler_paths)
        self.ensemble_predictor.save()  # Save ensemble metadata
        
        # Save results
        self._save_results()
        
        # Evaluate ensemble on train, val, and test separately
        print(f"\n{'='*70}")
        print("=== Evaluating Ensemble ===")
        print(f"{'='*70}")
        
        # Collect all train and val data from all windows
        # Note: Don't scale here - EnsemblePredictor will scale using each model's scaler
        all_train_data = []
        all_val_data = []
        
        for fold in folds:
            # Get train data for this fold (unscaled - EnsemblePredictor will handle scaling)
            X_train_fold = X_full.loc[fold['train_indices']].copy()
            y_train_fold = y_full.loc[fold['train_indices']].copy()
            
            if self.config.use_recency_weights:
                X_train_fold['sample_weight'] = 1.0
            
            train_data_fold = X_train_fold.copy()
            train_data_fold['y_true'] = y_train_fold
            all_train_data.append(train_data_fold)
            
            # Get val data for this fold (unscaled - EnsemblePredictor will handle scaling)
            X_val_fold = X_full.loc[fold['val_indices']].copy()
            y_val_fold = y_full.loc[fold['val_indices']].copy()
            
            if self.config.use_recency_weights:
                X_val_fold['sample_weight'] = 1.0
            
            val_data_fold = X_val_fold.copy()
            val_data_fold['y_true'] = y_val_fold
            all_val_data.append(val_data_fold)
        
        # Combine all train and val data
        if all_train_data:
            combined_train_data = pd.concat(all_train_data, ignore_index=False)
            # Remove duplicates (fights that appear in multiple train sets)
            combined_train_data = combined_train_data[~combined_train_data.index.duplicated(keep='first')]
        else:
            combined_train_data = pd.DataFrame()
        
        if all_val_data:
            combined_val_data = pd.concat(all_val_data, ignore_index=False)
        else:
            combined_val_data = pd.DataFrame()
        
        # Evaluate ensemble on train
        train_scores = None
        if len(combined_train_data) > 0:
            train_scores = self.ensemble_predictor.evaluate(combined_train_data)
            print(f"Ensemble Training - Accuracy: {train_scores['accuracy']:.4f}, "
                  f"Log Loss: {train_scores['log_loss']:.4f}")
        
        # Evaluate ensemble on val
        val_scores = None
        if len(combined_val_data) > 0:
            val_scores = self.ensemble_predictor.evaluate(combined_val_data)
            print(f"Ensemble Validation - Accuracy: {val_scores['accuracy']:.4f}, "
                  f"Log Loss: {val_scores['log_loss']:.4f}")
        
        # Evaluate ensemble on test/holdout if available
        test_scores = None
        X_test = None
        y_test = None
        if self.holdout_data is not None and len(self.holdout_data) > 0:
            print(f"\n=== Evaluating Ensemble on Test Set ===")
            
            from libs.modeling.data_utils import load_training_data
            X_test_raw, y_test_raw = load_training_data(self.holdout_data, self.selected_features)
            
            # Don't scale here - EnsemblePredictor will scale using each model's scaler
            X_test = X_test_raw.copy()
            
            # Add sample_weight if needed
            if self.config.use_recency_weights:
                X_test['sample_weight'] = 1.0
            
            y_test = y_test_raw
            
            # Evaluate ensemble on test
            test_data_eval = X_test.copy()
            test_data_eval['y_true'] = y_test
            test_scores = self.ensemble_predictor.evaluate(test_data_eval)
            
            print(f"Ensemble Test - Accuracy: {test_scores['accuracy']:.4f}, "
                  f"Log Loss: {test_scores['log_loss']:.4f}")
            
            # Save test results to summary
            test_result = {
                'test_accuracy': test_scores['accuracy'],
                'test_log_loss': test_scores['log_loss'],
                'test_size': len(X_test)
            }
            summary_path = os.path.join(self.model_dir, 'walkforward_summary.json')
            if os.path.exists(summary_path):
                import json
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                summary['test_results'] = test_result
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
        
        # Save evaluations and predictions (train, val, test)
        self._save_ensemble_evaluations(train_scores, val_scores, test_scores, X_test, y_test)
        
        # Print summary
        print(f"\n{'='*70}")
        print("=== Walk-Forward Validation Complete ===")
        print(f"{'='*70}")
        if self.window_results:
            print(f"Mean Accuracy: {np.mean([r['accuracy'] for r in self.window_results]):.4f} ± "
                  f"{np.std([r['accuracy'] for r in self.window_results]):.4f}")
            print(f"Mean Log Loss: {np.mean([r['log_loss'] for r in self.window_results]):.4f} ± "
                  f"{np.std([r['log_loss'] for r in self.window_results]):.4f}")
        all_models = self.window_models.copy()
        print(f"Total models: {len(all_models)} ({len(self.window_models)} windows)")
        print(f"{'='*70}\n")
        
        return self.ensemble_predictor


class ModelTrainer:
    """Handles model training with AutoGluon."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.predictor = None
        self.model_dir = None
        self.selected_features = None
        self.original_df = None  # Store original dataframe for metadata
        # Store data for refit_all
        self.X_train_refit = None
        self.X_test_refit = None
        self.y_train_refit = None
        self.y_test_refit = None
        self.holdout_data = None  # Store holdout data for final evaluation
    
    
    def _prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training and test data.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
            For time-series split mode: X_train/X_test are the first 95% (for train/tune split),
            and the last 5% is stored separately as holdout_data
        """
        data_path = DataLoader.get_data_path(self.config.model_type)
        
        print(f"\n=== Data Preparation ===")
        print(f"Model type: {self.config.model_type}")
        print(f"Data path: {data_path}")
        
        if self.config.split_strategy == 'timeseries_split':
            print("Time-series split mode enabled — preparing time-respecting data split...")
        
        # Load data to determine available features
        print("Loading data to determine available features...")
        temp_df = pd.read_csv(data_path)
        features = DataLoader.prepare_features(temp_df, self.config)
        
        print(f"Selected {len(features)} features")
        
        # Load full data and filter fights (common for all strategies)
        full_df = DataLoader.load_and_filter_data(
            data_path,
            self.config.num_fights,
            self.config.start_date,
            self.config.include_split_dec
        )
        
        # Create holdout set based on date cutoff
        holdout_df = DataLoader.create_holdout_set(full_df, self.config.test_size)
        
        # Exclude holdout from training data
        # Note: holdout_df.index refers to indices in full_df after filtering
        if len(holdout_df) > 0:
            # Use fight_id or create a set of identifiers to exclude
            # Since indices are reset after filter_fights, we can use index directly
            holdout_indices = set(holdout_df.index)
            training_df = full_df[~full_df.index.isin(holdout_indices)].copy()
            print(f"Training data (excluding holdout): {len(training_df)} samples")
            
            # Verify no data leakage: training dates must be <= holdout dates
            if len(training_df) > 0 and len(holdout_df) > 0:
                training_max_date = training_df['event_date'].max()
                holdout_min_date = holdout_df['event_date'].min()
                if training_max_date > holdout_min_date:
                    raise ValueError(
                        f"Data leakage detected! Training max date ({training_max_date}) > Holdout min date ({holdout_min_date}). "
                        f"This should not happen with date-based holdout."
                    )
                print(f"✓ Verified: Training data ends at {training_max_date}, holdout starts at {holdout_min_date}")
        else:
            training_df = full_df.copy()
            print(f"Warning: No holdout set created, using all data for training")
        
        # Store holdout data for later evaluation
        self.holdout_data = holdout_df
        
        # For time-series split mode, we need to handle data splitting differently
        if self.config.split_strategy == 'timeseries_split':
            # Use training_df (already excludes holdout)
            timeseries_df = training_df.copy()
            
            print(f"Time-series data: {len(timeseries_df)} samples")
            
            # Prepare features for time-series data
            X_timeseries, y_timeseries = DataLoader.load_training_features(timeseries_df, features)
            
            # Note: Normalization is now done AFTER train/val split in _train_timeseries
            # to ensure scaler is fit only on training data (prevents data leakage)
            
            # Apply recency weights if enabled
            if self.config.use_recency_weights:
                from libs.modeling.data_utils import calculate_recency_weights
                weights = calculate_recency_weights(
                    timeseries_df.loc[X_timeseries.index],
                    X_timeseries.index,
                    decay_rate=self.config.decay_rate
                )
                X_timeseries['sample_weight'] = weights
            
            # Store selected features and original dataframe
            self.selected_features = features
            self.original_df = timeseries_df  # Store for metadata
            
            # Save full filtered dataframe for ModelUtils (needed for calibration curve)
            training_data_path = os.path.join(self.model_dir, 'training_data.csv')
            full_df.to_csv(training_data_path, index=False)
            
            # Save holdout fight IDs and test start date for ModelUtils
            if len(holdout_df) > 0:
                # Save holdout fight IDs
                holdout_fight_ids_path = os.path.join(self.model_dir, 'holdout_fight_ids.txt')
                with open(holdout_fight_ids_path, 'w') as f:
                    for fight_id in holdout_df['fight_id'].values:
                        f.write(f"{fight_id}\n")
                print(f"Saved {len(holdout_df)} holdout fight IDs to {holdout_fight_ids_path}")
                
                # Also save test start date for backward compatibility
                from libs.modeling.split_date_utils import write_test_start_date
                holdout_start_date = holdout_df['event_date'].min()
                test_start_path = os.path.join(self.model_dir, 'test_start_date.txt')
                write_test_start_date(holdout_start_date, test_start_path)
                print(f"Saved test start date: {holdout_start_date}")
            
            # For time-series split, we'll use the full timeseries data for train/tune split
            # Return it as "train" data (will be split into train/tune later)
            return X_timeseries, pd.DataFrame(), y_timeseries, pd.Series(dtype=float)
        else:
            # Standard mode: use DataPreparation but exclude holdout
            # Save training_df (without holdout) to temp CSV for DataPreparation
            import tempfile
            temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            training_df.to_csv(temp_csv.name, index=False)
            temp_csv.close()
            
            data_prep = DataPreparation(
                data_path=temp_csv.name,
                feats=features,
                odds=False,
                train_size=1.0,  # Use all data (holdout already excluded)
                val_size=None,  # No validation set
                test_size=0.0,  # No test split (we already have holdout)
                normalize=self.config.normalize,
                use_recency_weights=self.config.use_recency_weights,
                decay_rate=self.config.decay_rate,
                balance_fighters=False,  # No fighter balancing
                target_balance=0.5,
                start_date=self.config.start_date,
                num_fights=self.config.num_fights,
                include_split_dec=self.config.include_split_dec,
                data_cutoff=None,  # No data cutoff
                preserve_fold_order=False
            )
            
            # Prepare data (will use all as train since train_size=1.0, test_size=0.0)
            X_train, _, X_test, y_train, _, y_test = data_prep.prepare_data(self.model_dir)
            
            # X_test should be empty since test_size=0, use all as train
            if len(X_test) == 0:
                X_train = pd.concat([X_train, X_test]) if len(X_test) > 0 else X_train
                y_train = pd.concat([y_train, y_test]) if len(y_test) > 0 else y_train
            
            # Store selected features and original dataframe for later use
            self.selected_features = features
            self.original_df = training_df  # Store training dataframe (without holdout)
            
            # Save full filtered dataframe for ModelUtils (needed for calibration curve)
            training_data_path = os.path.join(self.model_dir, 'training_data.csv')
            full_df.to_csv(training_data_path, index=False)
            
            # Save holdout fight IDs and test start date for ModelUtils
            if len(holdout_df) > 0:
                # Save holdout fight IDs
                holdout_fight_ids_path = os.path.join(self.model_dir, 'holdout_fight_ids.txt')
                with open(holdout_fight_ids_path, 'w') as f:
                    for fight_id in holdout_df['fight_id'].values:
                        f.write(f"{fight_id}\n")
                print(f"Saved {len(holdout_df)} holdout fight IDs to {holdout_fight_ids_path}")
                
                # Also save test start date for backward compatibility
                from libs.modeling.split_date_utils import write_test_start_date
                holdout_start_date = holdout_df['event_date'].min()
                test_start_path = os.path.join(self.model_dir, 'test_start_date.txt')
                write_test_start_date(holdout_start_date, test_start_path)
                print(f"Saved test start date: {holdout_start_date}")
            
            # Clean up temp file
            os.unlink(temp_csv.name)
            
            print(f"Training samples: {len(X_train)}")
            print(f"Holdout samples: {len(holdout_df)}")
            
            # Return empty test set (holdout is stored separately)
            return X_train, pd.DataFrame(), y_train, pd.Series(dtype=float)
    
    def _create_predictor(self) -> TabularPredictor:
        """Create and configure AutoGluon predictor."""
        sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
        
        predictor = TabularPredictor(
            label='y_true',
            eval_metric='log_loss',
            problem_type='binary',
            path=self.model_dir,
            verbosity=2,
            sample_weight=sample_weight_col,
            weight_evaluation=False
        )
        
        return predictor
    
    def _train_timeseries(self, X_timeseries: pd.DataFrame, y_timeseries: pd.Series) -> TabularPredictor:
        """
        Train time-series split model with time-respecting train/tune split.
        
        Args:
            X_timeseries: Features for time-series training (first 95% of data)
            y_timeseries: Labels for time-series training
            
        Returns:
            Trained TabularPredictor instance
        """
        trainer = TimeseriesSplitTrainer(self.config, self.model_dir, self.original_df)
        predictor = trainer.train(X_timeseries, y_timeseries)
        
        # Store scores for later use
        self.train_scores = trainer.train_scores
        self.tune_scores = trainer.tune_scores
        self.refit_train_scores = getattr(trainer, 'refit_train_scores', None)
        
        return predictor
    
    def _train_model(self, predictor: TabularPredictor, 
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> TabularPredictor:
        """Train the model with specified preset."""
        print(f"\n=== Training Model ===")
        print(f"Preset: {self.config.preset}")
        print(f"Time limit: {self.config.time_limit}s")
        
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        
        if self.config.preset == 'extreme':
            fit_kwargs = {
                'train_data': train_data,
                'presets': 'extreme',
                'time_limit': self.config.time_limit,
                'ag_args_fit': {'shuffle': False}
            }
            if self.config.included_model_types:
                fit_kwargs['included_model_types'] = self.config.included_model_types
            
            predictor.fit(**fit_kwargs)
        
        elif self.config.preset == 'best':
            excluded_model_types = ['KNN']
            predictor.fit(
                train_data=train_data,
                #tuning_data=test_data,  # Use test set for tuning
                presets='best',
                excluded_model_types=excluded_model_types,
                time_limit=self.config.time_limit,
                use_bag_holdout=True,
                ag_args_fit={
                    'stopping_metric': 'log_loss',
                    'num_gpus': 1,
                    'shuffle': False
                },
                ag_args_ensemble={
                    'use_orig_features': False,
                    'max_base_models': 15,
                    'max_base_models_per_type': 3,
                    'fold_fitting_strategy': 'sequential_local'
                }
            )
        
        return predictor
    
    def _refit_all(self) -> TabularPredictor:
        """
        Retrain model on all available data (including test/holdout sets).
        
        Only applies to timeseries_split strategy. Walkforward always refits internally.
        
        Returns:
            Refitted TabularPredictor instance
        """
        trainer = RefitAllTrainer(
            config=self.config,
            original_model_dir=self.model_dir,
            original_df=self.original_df,
            holdout_df=self.holdout_data,
            selected_features=self.selected_features,
            train_scores=self.train_scores,
            tune_scores=self.tune_scores
        )
        return trainer.train()
    
    def _refit_all_old(self) -> TabularPredictor:
        """
        OLD IMPLEMENTATION - Kept for reference.
        Retrain model on all available data (including test/holdout sets).
        
        Only applies to timeseries_split strategy. Walkforward always refits internally.
        
        For timeseries_split: Combines timeseries + holdout, then splits 85/15 chronologically
        
        Returns:
            Refitted TabularPredictor instance
        """
        # Ensure this is only called for timeseries_split
        if self.config.split_strategy != 'timeseries_split':
            raise ValueError(
                f"refit_all is only supported for timeseries_split strategy, "
                f"got {self.config.split_strategy}. Walkforward always refits internally."
            )
        
        print(f"\n{'='*70}")
        print("=== REFIT FULL: Retraining on All Available Data ===")
        print(f"{'='*70}")
        
        # Create refit directory
        refit_dir = f"{self.model_dir}_refit_all"
        if os.path.exists(refit_dir):
            import shutil
            shutil.rmtree(refit_dir)
        os.makedirs(refit_dir, exist_ok=True)
        print(f"Refit model directory: {refit_dir}")
        
        if self.config.split_strategy == 'timeseries_split':
            # Timeseries split mode: combine timeseries + holdout, then split 85/15
            print("\n--- Timeseries Split Mode: Combining All Data ---")
            
            # Combine original timeseries data and holdout data
            full_df = pd.concat([self.original_df, self.holdout_data], ignore_index=True)
            full_df = full_df.sort_values('event_date').reset_index(drop=True)
            
            print(f"Combined data: {len(full_df)} samples")
            print(f"Date range: {full_df['event_date'].min()} to {full_df['event_date'].max()}")
            
            # Prepare features for full dataset
            from libs.modeling.data_utils import load_training_data
            X_full, y_full = load_training_data(full_df, self.selected_features)
            
            # Apply normalization (fit scaler on training data only to avoid data leakage)
            # CRITICAL: Fit scaler on train_data only, then apply to train_data and tune_data separately
            # Fitting on full data would leak validation statistics into training normalization
            if self.config.normalize != 'none':
                scaler_path = os.path.join(refit_dir, 'scaler.pkl')
                import joblib
                from libs.modeling.data_utils import (
                    apply_robust_normalization, 
                    apply_zscore_normalization,
                    apply_no_normalization
                )
                
                # Get event dates for temporal splitting BEFORE normalization
                event_dates = full_df.reindex(X_full.index)['event_date'].values
                # Use val_date from config if provided, otherwise default to 2025-01-01
                if self.config.val_date:
                    val_start_date = pd.Timestamp(self.config.val_date)
                else:
                    val_start_date = pd.Timestamp('2025-01-01')
                train_mask = event_dates < val_start_date
                
                # Split into train and validation BEFORE normalization
                train_indices_pre_norm = X_full.index[train_mask]
                tune_indices_pre_norm = X_full.index[~train_mask]
                
                X_train_pre_norm = X_full.loc[train_indices_pre_norm]
                X_tune_pre_norm = X_full.loc[tune_indices_pre_norm]
                
                # Fit scaler on training data only, then transform both train and tune
                if self.config.normalize == 'robust':
                    X_train_norm, X_tune_norm = apply_robust_normalization(X_train_pre_norm, X_tune_pre_norm, scaler_path)
                elif self.config.normalize == 'zscore':
                    X_train_norm, X_tune_norm = apply_zscore_normalization(X_train_pre_norm, X_tune_pre_norm, scaler_path)
                else:
                    X_train_norm, X_tune_norm = apply_no_normalization(X_train_pre_norm, X_tune_pre_norm, scaler_path)
                
                # Recombine normalized data
                X_full = pd.concat([X_train_norm, X_tune_norm], axis=0)
                print(f"Normalization applied (scaler fit on training data only to prevent data leakage)")
            
            # Apply recency weights if enabled
            if self.config.use_recency_weights:
                from libs.modeling.data_utils import calculate_recency_weights
                weights = calculate_recency_weights(
                    full_df.loc[X_full.index],
                    X_full.index,
                    decay_rate=self.config.decay_rate
                )
                X_full['sample_weight'] = weights
            
            # Combine features and labels
            full_data = X_full.copy()
            full_data['y_true'] = y_full
            
            # Get event dates for temporal splitting
            event_dates = full_df.reindex(full_data.index)['event_date'].values
            
            # Create time-respecting train/tune split using date-based logic for refit_all
            # For refit_all: use val_date from config if provided, otherwise default to 2025-01-01
            # train before val_date, val from val_date to today
            if self.config.val_date:
                val_start_date = pd.Timestamp(self.config.val_date)
            else:
                val_start_date = pd.Timestamp('2025-01-01')
            
            # Create date masks
            train_mask = event_dates < val_start_date
            val_mask = event_dates >= val_start_date
            
            # Get indices for train and validation sets
            train_indices = full_data.index[train_mask]
            tune_indices = full_data.index[val_mask]
            
            # Validate that we have data in both sets
            if len(train_indices) == 0:
                raise ValueError(f"No training data found before validation start date {val_start_date}")
            if len(tune_indices) == 0:
                raise ValueError(f"No validation data found on or after {val_start_date}")
            
            # Create train and validation dataframes
            train_data = full_data.loc[train_indices].copy()
            tune_data = full_data.loc[tune_indices].copy()
            
            train_dates = event_dates[train_mask]
            tune_dates = event_dates[val_mask]
            
            # Verify no future leakage: train dates must be < val dates
            if len(train_dates) > 0 and len(tune_dates) > 0:
                if train_dates.max() >= tune_dates.min():
                    raise ValueError(f"Data leakage detected! Train max date ({train_dates.max()}) >= Val min date ({tune_dates.min()})")
            
            # Print split structure
            print(f"\nRefit split structure:")
            print(f"  Train: {len(train_data)} samples (before {val_start_date.strftime('%Y-%m-%d')})")
            print(f"  Val:   {len(tune_data)} samples ({val_start_date.strftime('%Y-%m-%d')} onward)")
            print(f"  Train date range: {train_dates.min()} to {train_dates.max()}")
            print(f"  Val date range:   {tune_dates.min()} to {tune_dates.max()}")
            
            # Create predictor for refit
            sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
            refit_predictor = TabularPredictor(
                label='y_true',
                eval_metric='log_loss',
                problem_type='binary',
                path=refit_dir,
                verbosity=2,
                sample_weight=sample_weight_col,
                weight_evaluation=False
            )
            
            # Train with same settings
            # CRITICAL: No shuffling, bagging, or stacking to preserve temporal order
            fit_kwargs = {
                'train_data': train_data,
                'tuning_data': tune_data,
                'presets': 'extreme_quality',  # Corrected to full preset name; use 'extreme' if that's a custom alias
                'time_limit': self.config.time_limit,
                'num_bag_folds': 0,  # Disables bagging/CV folds
                'num_stack_levels': 0,  # Disables stacking
                'auto_stack': False,  # Overrides preset's auto_stack=True
                'dynamic_stacking': False,  # Disables dynamic stacking validation splits
            }
            
            if self.config.included_model_types:
                fit_kwargs['included_model_types'] = self.config.included_model_types
            
            print(f"\n--- Training Refit Model ---")
            refit_predictor.fit(**fit_kwargs)
            
            # Evaluate on training set
            train_scores_refit = refit_predictor.evaluate(train_data)
            print(f"\nRefit Train set - Accuracy: {train_scores_refit['accuracy']:.4f}, "
                  f"Log Loss: {train_scores_refit['log_loss']:.4f}")
            print(f"  Train set size: {len(train_data)} samples")
            print(f"  Train date range: {train_dates.min()} to {train_dates.max()}")
            print(f"  Train label distribution: {train_data['y_true'].value_counts().to_dict()}")
            
            # Evaluate on tune set (validation)
            tune_scores_refit = refit_predictor.evaluate(tune_data)
            print(f"\nRefit Tune set (Validation) - Accuracy: {tune_scores_refit['accuracy']:.4f}, "
                  f"Log Loss: {tune_scores_refit['log_loss']:.4f}")
            print(f"  Tune set size: {len(tune_data)} samples")
            print(f"  Tune date range: {tune_dates.min()} to {tune_dates.max()}")
            print(f"  Tune label distribution: {tune_data['y_true'].value_counts().to_dict()}")
            
            # Compare with initial model scores for reference
            if hasattr(self, 'train_scores') and hasattr(self, 'tune_scores'):
                print(f"\nComparison with initial model:")
                print(f"  Initial Train Accuracy: {self.train_scores['accuracy']:.4f}")
                print(f"  Initial Tune Accuracy: {self.tune_scores['accuracy']:.4f}")
                print(f"  Refit Train Accuracy: {train_scores_refit['accuracy']:.4f} (Δ{train_scores_refit['accuracy'] - self.train_scores['accuracy']:+.4f})")
                print(f"  Refit Tune Accuracy: {tune_scores_refit['accuracy']:.4f} (Δ{tune_scores_refit['accuracy'] - self.tune_scores['accuracy']:+.4f})")
            
            # Call refit_full to retrain on all data (train + tune)
            print(f"\n--- Calling refit_full() to Retrain on All Data ---")
            refit_result = refit_predictor.refit_full()
            
            # Handle refit_full return value - it may return a dict or a predictor
            if isinstance(refit_result, dict):
                # refit_full returned a dict of model paths, reload predictor to use refitted models
                print(f"Refit returned dict with {len(refit_result)} models, reloading predictor")
                refit_predictor = TabularPredictor.load(refit_dir)
            else:
                # refit_full returned a new predictor
                refit_predictor = refit_result
            
            # Evaluate on training set after refit_all
            train_scores_after_refit = refit_predictor.evaluate(train_data)
            print(f"Refit Full Train set - Accuracy: {train_scores_after_refit['accuracy']:.4f}, "
                  f"Log Loss: {train_scores_after_refit['log_loss']:.4f}")
            
            # Save evaluation results for refit model
            refit_results = {
                'train_scores': train_scores_refit,
                'val_scores': tune_scores_refit,
                'test_scores': None,  # No holdout for refit (it's included in training)
                'best_model': refit_predictor.model_best,
                'feature_importance': None
            }
            
            # Create a temporary empty DataFrame for _save_results (it expects X_test, y_test)
            # We'll save evals.txt manually for refit
            evals_path_refit = os.path.join(refit_dir, 'evals.txt')
            with open(evals_path_refit, 'w') as f:
                f.write("Model Performance (Refit Full):\n")
                f.write(f"Training accuracy: {train_scores_refit['accuracy']:.4f}\n")
                f.write(f"Training log loss: {train_scores_refit['log_loss']:.4f}\n")
                f.write(f"Validation accuracy: {tune_scores_refit['accuracy']:.4f}\n")
                f.write(f"Validation log loss: {tune_scores_refit['log_loss']:.4f}\n")
                f.write("Holdout scores: N/A (refit includes all data)\n")
                
                # Get model weights
                best_model_refit, weights_refit = self._get_model_weights(refit_predictor)
                f.write(f"\nBest Model: {best_model_refit}\n")
                
                # Write model weights if available
                if weights_refit:
                    f.write(f"\nModel weights:\n")
                    for model_name, weight in weights_refit.items():
                        f.write(f"{model_name}: {weight:.3f}\n")
                else:
                    f.write(f"\nModel weights: N/A (not a weighted ensemble or unavailable)\n")
                
                f.write(f"\nConfiguration:\n")
                f.write(f"Model Type: {self.config.model_type}\n")
                f.write(f"Preset: {self.config.preset}\n")
                f.write(f"Time Limit: {self.config.time_limit}\n")
                f.write(f"Test Size (cutoff date): {self.config.test_size if self.config.test_size else 'None (no holdout)'}\n")
                f.write(f"Split Strategy: {self.config.split_strategy}\n")
                f.write(f"Refit All: {self.config.refit_all}\n")
                f.write(f"Start Date: {self.config.start_date}\n")
                f.write(f"Normalize: {self.config.normalize}\n")
                f.write(f"Use Recency Weights: {self.config.use_recency_weights}\n")
                f.write(f"Decay Rate: {self.config.decay_rate}\n")
                f.write(f"Calculate Importance: {self.config.calculate_importance}\n")
                f.write(f"Included Model Types: {self.config.included_model_types}\n")
                f.write(f"Include Split Dec: {self.config.include_split_dec}\n")
                f.write(f"dec_avg rate: {self.config.decay_rate}\n")
            
            print(f"\nRefit model evaluations saved to: {evals_path_refit}")
            
        else:
            # Normal mode: combine train + test, retrain on all
            print("\n--- Normal Mode: Combining Train + Test Data ---")
            
            # Reset indices before concatenating to avoid conflicts
            X_train_clean = self.X_train_refit.reset_index(drop=True)
            X_test_clean = self.X_test_refit.reset_index(drop=True)
            y_train_clean = self.y_train_refit.reset_index(drop=True)
            y_test_clean = self.y_test_refit.reset_index(drop=True)
            
            # Combine train and test data
            X_combined = pd.concat([X_train_clean, X_test_clean], ignore_index=True)
            y_combined = pd.concat([y_train_clean, y_test_clean], ignore_index=True)
            
            print(f"Combined data: {len(X_combined)} samples")
            
            # Combine features and labels
            combined_data = X_combined.copy()
            combined_data['y_true'] = y_combined
            
            # Create predictor for refit
            sample_weight_col = 'sample_weight' if self.config.use_recency_weights else None
            refit_predictor = TabularPredictor(
                label='y_true',
                eval_metric='log_loss',
                problem_type='binary',
                path=refit_dir,
                verbosity=2,
                sample_weight=sample_weight_col,
                weight_evaluation=False
            )
            
            # Train on all data (no tuning_data for normal mode refit)
            fit_kwargs = {
                'train_data': combined_data,
                'presets': self.config.preset,
                'time_limit': self.config.time_limit,
                'time_limit': self.config.time_limit,
                'num_bag_folds': 0,  # Disables bagging/CV folds
                'num_stack_levels': 0,  # Disables stacking
                'auto_stack': False,  # Overrides preset's auto_stack=True
                'dynamic_stacking': False,  # Disables dynamic stacking validation splits
            }
            
            if self.config.included_model_types:
                fit_kwargs['included_model_types'] = self.config.included_model_types
            
            print(f"\n--- Training Refit Model on All Data ---")
            refit_predictor.fit(**fit_kwargs)
            
            # Evaluate on combined data
            combined_scores = refit_predictor.evaluate(combined_data)
            print(f"Refit Combined data - Accuracy: {combined_scores['accuracy']:.4f}, "
                  f"Log Loss: {combined_scores['log_loss']:.4f}")
        
        # Copy necessary files to refit directory
        # Copy feats.txt
        if self.selected_features:
            feats_path = os.path.join(refit_dir, 'feats.txt')
            with open(feats_path, 'w') as f:
                for feat in self.selected_features:
                    f.write(f"{feat}\n")
            print(f"Features saved to: {feats_path}")
        
        # Copy training_data.csv and scaler
        if self.config.split_strategy == 'timeseries_split':
            training_data_path = os.path.join(refit_dir, 'training_data.csv')
            full_df.to_csv(training_data_path, index=False)
            # Scaler already saved during normalization above
        else:
            # For normal mode, copy from original model directory if it exists
            original_training_data = os.path.join(self.model_dir, 'training_data.csv')
            if os.path.exists(original_training_data):
                import shutil
                shutil.copy(original_training_data, os.path.join(refit_dir, 'training_data.csv'))
            
            # Copy scaler if it exists
            original_scaler = os.path.join(self.model_dir, 'scaler.pkl')
            if os.path.exists(original_scaler):
                import shutil
                shutil.copy(original_scaler, os.path.join(refit_dir, 'scaler.pkl'))
        
        print(f"\n{'='*70}")
        print("=== REFIT FULL Complete ===")
        print(f"Refit model saved to: {refit_dir}")
        print(f"{'='*70}\n")
        
        return refit_predictor
    
    def _get_model_weights(self, predictor: TabularPredictor) -> Tuple[str, Optional[dict]]:
        """
        Extract model weights from a TabularPredictor if it's a weighted ensemble.
        
        Returns:
            Tuple of (best_model_name, weights_dict) where weights_dict is None if not a weighted ensemble
        """
        best_model = predictor.model_best
        weights = None
        
        try:
            best_model_info = predictor.info()['model_info'][best_model]
            if 'children_info' in best_model_info:
                # Check for weighted ensemble at S1F1 (most common path)
                if 'S1F1' in best_model_info['children_info']:
                    if 'model_weights' in best_model_info['children_info']['S1F1']:
                        weights = best_model_info['children_info']['S1F1']['model_weights']
                else:
                    # Try to find any weighted ensemble in children_info
                    for child_key, child_info in best_model_info['children_info'].items():
                        if 'model_weights' in child_info:
                            weights = child_info['model_weights']
                            break
        except (KeyError, TypeError):
            # Not a weighted ensemble or weights not found
            pass
        
        return best_model, weights
    
    def _print_all_evaluations(self, predictor: TabularPredictor,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_holdout: Optional[pd.DataFrame], y_holdout: Optional[pd.Series]):
        """
        Collect and print all evaluations together in a clean format.
        
        Prints:
        - Train accuracy/logloss (original model)
        - Val accuracy/logloss (original model)
        - Test accuracy/logloss (original model on holdout, if available)
        - Refit Full Train+Val accuracy/logloss (refitted model, if refit_full enabled)
        - Refit Full Test accuracy/logloss (refitted model on holdout, if available)
        """
        print(f"\n{'='*70}")
        print("=== Model Evaluation Summary ===")
        print(f"{'='*70}")
        
        # Original model evaluations
        train_scores = getattr(self, 'train_scores', None)
        tune_scores = getattr(self, 'tune_scores', None)
        refit_train_scores = getattr(self, 'refit_train_scores', None)
        
        # Print original model scores
        if train_scores:
            print(f"Train (Original)     - Accuracy: {train_scores['accuracy']:.4f}, Log Loss: {train_scores['log_loss']:.4f}")
        if tune_scores:
            print(f"Val (Original)       - Accuracy: {tune_scores['accuracy']:.4f}, Log Loss: {tune_scores['log_loss']:.4f}")
        
        # Evaluate original model on holdout if available
        if X_holdout is not None and len(X_holdout) > 0:
            holdout_data_eval = X_holdout.copy()
            holdout_data_eval['y_true'] = y_holdout
            test_scores = predictor.evaluate(holdout_data_eval)
            print(f"Test (Original)      - Accuracy: {test_scores['accuracy']:.4f}, Log Loss: {test_scores['log_loss']:.4f}")
        else:
            test_scores = None
            print(f"Test (Original)      - No holdout data available")
        
        # Print refit_full scores if available
        if refit_train_scores:
            print(f"\nRefit Full Train+Val - Accuracy: {refit_train_scores['accuracy']:.4f}, Log Loss: {refit_train_scores['log_loss']:.4f}")
            
            # Evaluate refitted model on holdout if available
            if X_holdout is not None and len(X_holdout) > 0:
                holdout_data_eval = X_holdout.copy()
                holdout_data_eval['y_true'] = y_holdout
                refit_test_scores = predictor.evaluate(holdout_data_eval)
                print(f"Refit Full Test      - Accuracy: {refit_test_scores['accuracy']:.4f}, Log Loss: {refit_test_scores['log_loss']:.4f}")
            else:
                print(f"Refit Full Test      - No holdout data available")
        
        # Print ensemble composition
        try:
            best_model_info = predictor.info()['model_info'][predictor.model_best]
            if 'children_info' in best_model_info:
                if 'S1F1' in best_model_info['children_info']:
                    weighted_ensemble_info = best_model_info['children_info']['S1F1']['model_weights']
                    print(f"\nEnsemble Composition:")
                    for model, weight in weighted_ensemble_info.items():
                        print(f"  {model}: {weight:.3f}")
                else:
                    found_weights = False
                    for child_key, child_info in best_model_info['children_info'].items():
                        if 'model_weights' in child_info:
                            weighted_ensemble_info = child_info['model_weights']
                            print(f"\nEnsemble Composition ({child_key}):")
                            for model, weight in weighted_ensemble_info.items():
                                print(f"  {model}: {weight:.3f}")
                            found_weights = True
                            break
                    if not found_weights:
                        print(f"\nBest model '{predictor.model_best}' is an ensemble but weights not found.")
            else:
                print(f"\nBest Model: {predictor.model_best}")
        except (KeyError, TypeError) as e:
            print(f"\nBest Model: {predictor.model_best}")
        
        print(f"{'='*70}\n")
    
    def _evaluate_model(self, predictor: TabularPredictor,
                       X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate model performance."""
        print(f"\n=== Model Evaluation ===")
        
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1) if len(X_test) > 0 else pd.DataFrame()
        
        train_scores = predictor.evaluate(train_data)
        test_scores = predictor.evaluate(test_data) if len(test_data) > 0 else None
        
        print(f"Training - Accuracy: {train_scores['accuracy']:.4f}, Log Loss: {train_scores['log_loss']:.4f}")
        if test_scores is not None:
            print(f"Test - Accuracy: {test_scores['accuracy']:.4f}, Log Loss: {test_scores['log_loss']:.4f}")
        else:
            print(f"Test - No test data provided")
        print(f"Best Model: {predictor.model_best}")
        
        # Print ensemble composition if it's a weighted ensemble
        try:
            best_model_info = predictor.info()['model_info'][predictor.model_best]
            if 'children_info' in best_model_info:
                # Check for weighted ensemble at S1F1 (most common path)
                if 'S1F1' in best_model_info['children_info']:
                    weighted_ensemble_info = best_model_info['children_info']['S1F1']['model_weights']
                    print("\nEnsemble Composition:")
                    for model, weight in weighted_ensemble_info.items():
                        print(f"  {model}: {weight:.3f}")
                else:
                    # Try to find any weighted ensemble in children_info
                    found_weights = False
                    for child_key, child_info in best_model_info['children_info'].items():
                        if 'model_weights' in child_info:
                            weighted_ensemble_info = child_info['model_weights']
                            print(f"\nEnsemble Composition ({child_key}):")
                            for model, weight in weighted_ensemble_info.items():
                                print(f"  {model}: {weight:.3f}")
                            found_weights = True
                            break
                    if not found_weights:
                        print(f"\nBest model '{predictor.model_best}' is an ensemble but weights not found in expected location.")
            else:
                print(f"\nBest model '{predictor.model_best}' is not a weighted ensemble.")
        except (KeyError, TypeError) as e:
            print(f"\nCould not retrieve ensemble composition: {e}")
        
        # Calculate feature importance if enabled
        feature_importance = None
        if self.config.calculate_importance:
            print("\nCalculating feature importance...")
            try:
                feature_importance = predictor.feature_importance(train_data)
                print(f"Top 140 Most Important Features:")
                for i, (feature, importance_value) in enumerate(feature_importance.head(150).iterrows(), 1):
                    print(f"  {i}. {feature}: {importance_value['importance']:.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate feature importance: {e}")
                feature_importance = None
        
        return {
            'train_scores': train_scores,
            'test_scores': test_scores,
            'best_model': predictor.model_best,
            'feature_importance': feature_importance
        }
    
    def _save_results(self, results: dict, X_test: pd.DataFrame, y_test: pd.Series):
        """Save evaluation results and test predictions."""
        # Get model weights if available
        weights = None
        if hasattr(self, 'predictor') and self.predictor is not None:
            try:
                _, weights = self._get_model_weights(self.predictor)
            except Exception:
                pass
        
        # Save evaluation results
        EvaluationManager.save_evaluations(
            results,
            self.config,
            self.model_dir,
            results['best_model'],
            weights=weights,
            feature_importance=results.get('feature_importance')
        )
        
        # Save test predictions if test data exists
        if len(X_test) > 0 and hasattr(self, 'predictor') and self.predictor is not None:
            metadata_df = None
            if hasattr(self, 'holdout_data') and self.holdout_data is not None and len(self.holdout_data) > 0:
                metadata_df = self.holdout_data
            elif self.original_df is not None:
                metadata_df = self.original_df
            
            EvaluationManager.save_predictions(
                self.predictor,
                X_test,
                y_test,
                self.model_dir,
                metadata_df=metadata_df
            )
    
    def train(self):
        """
        Execute full training pipeline.
        
        Returns:
            Trained TabularPredictor or EnsemblePredictor instance
        """
        # Route to walk-forward trainer if enabled
        if self.config.split_strategy == 'walkforward':
            walkforward_trainer = WalkForwardTrainer(self.config)
            ensemble_predictor = walkforward_trainer.train()
            # Model statistics are saved in WalkForwardTrainer.train() using the ensemble
            return ensemble_predictor
        
        # Create model directory
        self.model_dir = FileManager.create_model_directory(
            self.config.model_type,
            self.config.preset
        )
        print(f"\n=== Model Training Pipeline ===")
        print(f"Model directory: {self.model_dir}")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data()
        
        # Store data for potential refit_all
        self.X_train_refit = X_train
        self.X_test_refit = X_test
        self.y_train_refit = y_train
        self.y_test_refit = y_test
        
        if self.config.split_strategy == 'timeseries_split':
            # Time-series split training mode
            predictor = self._train_timeseries(X_train, y_train)
            self.predictor = predictor
            
            # Prepare holdout data if available
            X_holdout, y_holdout = None, None
            if self.config.test_size is not None and self.holdout_data is not None and len(self.holdout_data) > 0:
                holdout_data = self.holdout_data.copy()
                
                # Prepare holdout features (apply same normalization and feature selection)
                from libs.modeling.data_utils import load_training_data
                X_holdout, y_holdout = load_training_data(holdout_data, self.selected_features)
                
                # Apply normalization if needed (use same scaler fitted on training data)
                if self.config.normalize != 'none':
                    scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                    import joblib
                    from sklearn.preprocessing import RobustScaler, StandardScaler
                    from sklearn.preprocessing import FunctionTransformer
                    
                    try:
                        scaler = joblib.load(scaler_path)
                        
                        # Use same column exclusion logic as normalization functions
                        date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
                        categorical_static_feats = ['weightclass_encoded', 'odds']
                        
                        def should_exclude_col(col_name):
                            if col_name in date_cols:
                                return True
                            for cat_feat in categorical_static_feats:
                                if cat_feat in col_name:
                                    return True
                            return False
                        
                        features_to_scale = [col for col in X_holdout.columns 
                                            if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
                        
                        # Transform only features that should be scaled
                        X_holdout_scaled = X_holdout.copy()
                        X_holdout_scaled[features_to_scale] = scaler.transform(X_holdout[features_to_scale])
                        X_holdout = X_holdout_scaled
                    except Exception as e:
                        print(f"Warning: Could not load/apply scaler for holdout data: {e}")
                        raise e
                
                # Add sample_weight if needed
                if self.config.use_recency_weights:
                    X_holdout['sample_weight'] = 1.0
            
            # Collect all evaluations and print together
            self._print_all_evaluations(predictor, X_train, y_train, X_holdout, y_holdout)
            
            # Calculate feature importance if enabled
            feature_importance = None
            if self.config.calculate_importance:
                print("\nCalculating feature importance...")
                try:
                    train_data = pd.concat([X_train, y_train], axis=1)
                    feature_importance = predictor.feature_importance(train_data)
                    print(f"Top 150 Most Important Features:")
                    for i, (feature, importance_value) in enumerate(feature_importance.head(150).iterrows(), 1):
                        print(f"  {i}. {feature}: {importance_value['importance']:.4f}")
                except Exception as e:
                    print(f"Warning: Could not calculate feature importance: {e}")
                    feature_importance = None
            
            # Save results
            if X_holdout is not None and len(X_holdout) > 0:
                holdout_data_eval = X_holdout.copy()
                holdout_data_eval['y_true'] = y_holdout
                holdout_scores = predictor.evaluate(holdout_data_eval)
            else:
                holdout_scores = None
            
            results = {
                'train_scores': getattr(self, 'train_scores', None),
                'val_scores': getattr(self, 'tune_scores', None),
                'test_scores': holdout_scores,
                'best_model': predictor.model_best,
                'feature_importance': feature_importance
            }
            self._save_results(results, X_holdout if X_holdout is not None else pd.DataFrame(), 
                             y_holdout if y_holdout is not None else pd.Series(dtype=float))
        else:
            # Standard training mode
            predictor = self._create_predictor()
            predictor = self._train_model(predictor, X_train, y_train, X_test, y_test)
            self.predictor = predictor
            
            # Evaluate on training data
            train_data = pd.concat([X_train, y_train], axis=1)
            train_scores = predictor.evaluate(train_data)
            
            # Evaluate on holdout if available
            if self.holdout_data is not None and len(self.holdout_data) > 0:
                print(f"\n=== Evaluating on Holdout Set ===")
                from libs.modeling.data_utils import load_training_data
                X_holdout, y_holdout = load_training_data(self.holdout_data, self.selected_features)
                
                # Apply normalization if needed (use same scaler fitted on training data)
                if self.config.normalize != 'none':
                    scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                    import joblib
                    from sklearn.preprocessing import RobustScaler, StandardScaler
                    
                    try:
                        scaler = joblib.load(scaler_path)
                        
                        # Use same column exclusion logic as normalization functions
                        date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
                        categorical_static_feats = ['weightclass_encoded', 'odds']
                        
                        def should_exclude_col(col_name):
                            if col_name in date_cols:
                                return True
                            for cat_feat in categorical_static_feats:
                                if cat_feat in col_name:
                                    return True
                            return False
                        
                        features_to_scale = [col for col in X_holdout.columns 
                                            if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
                        
                        # Transform only features that should be scaled
                        X_holdout_scaled = X_holdout.copy()
                        X_holdout_scaled[features_to_scale] = scaler.transform(X_holdout[features_to_scale])
                        X_holdout = X_holdout_scaled
                    except Exception as e:
                        print(f"Warning: Could not load/apply scaler for holdout data: {e}")
                        print("Proceeding without normalization for holdout data")
                
                # Add sample_weight if needed
                if self.config.use_recency_weights:
                    X_holdout['sample_weight'] = 1.0
                
                # Evaluate model on holdout
                holdout_data_eval = X_holdout.copy()
                holdout_data_eval['y_true'] = y_holdout
                holdout_scores = predictor.evaluate(holdout_data_eval)
                
                print(f"Holdout - Accuracy: {holdout_scores['accuracy']:.4f}, "
                      f"Log Loss: {holdout_scores['log_loss']:.4f}")
                
                # Evaluate model (includes feature importance if enabled)
                eval_results = self._evaluate_model(predictor, X_train, y_train, X_holdout, y_holdout)
                
                # Save results with holdout scores
                results = {
                    'train_scores': train_scores,
                    'test_scores': holdout_scores,
                    'best_model': predictor.model_best,
                    'feature_importance': eval_results['feature_importance']
                }
                self._save_results(results, X_holdout, y_holdout)
            else:
                # No holdout available, use empty test set
                # Evaluate model (includes feature importance if enabled)
                eval_results = self._evaluate_model(predictor, X_train, y_train, pd.DataFrame(), pd.Series(dtype=float))
                
                results = {
                    'train_scores': train_scores,
                    'test_scores': None,
                    'best_model': predictor.model_best,
                    'feature_importance': eval_results['feature_importance']
                }
                self._save_results(results, pd.DataFrame(), pd.Series(dtype=float))
        
        # Save features list
        if self.selected_features:
            feats_path = os.path.join(self.model_dir, 'feats.txt')
            with open(feats_path, 'w') as f:
                for feat in self.selected_features:
                    f.write(f"{feat}\n")
            print(f"Features saved to: {feats_path}")
        
        # Save model statistics (works for both normal and timeseries_split modes)
        training_data_path = os.path.join(self.model_dir, 'training_data.csv')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        
        # Check if required files exist (they should be created in both modes)
        if os.path.exists(training_data_path) and os.path.exists(scaler_path):
            utils = ModelUtils(
                model_path=self.model_dir,
                training_data_path=training_data_path,
                scaler_path=scaler_path,
                feats=self.selected_features or []
            )
            utils.save_model_stats()
            utils.plot_calibration_curve()
        else:
            print(f"Warning: Cannot generate calibration curve - missing required files")
            print(f"  training_data.csv exists: {os.path.exists(training_data_path)}")
            print(f"  scaler.pkl exists: {os.path.exists(scaler_path)}")
        
        # Refit on full data if requested (only for timeseries_split)
        if self.config.split_strategy == 'timeseries_split' and self.config.refit_all:
            refit_predictor = self._refit_all()
            return refit_predictor
        
        return predictor


def main(model_type='win', time_limit=None, preset=None, split_strategy=None, refit_full=None):
    """Main execution function. Features variable will override included_strings and excluded_strings."""
    # Configure training
    if model_type == 'win':
        features = vSeven_testing2
        #features = TEST_FEATURES        
        included_strings = None
        excluded_strings = None
        required_strings = None
        include_split_dec = True
    elif model_type == 'decision':
        included_strings = ['time_sec', 'decision', 'sub', 'ko', 'kd', 'win', 'strikes_att', 'distance_att', 'td', 'ctrl', 'weightclass_encoded']
        excluded_strings = ['total_avg']
        required_strings = None
        features = DECISION_TEST_FEATS4 # overrides the include/exclude strings
        include_split_dec = True
    config = TrainingConfig(
        model_type=model_type,  # 'win' or 'decision'
        preset=preset or 'extreme',  # 'extreme' or 'best'
        time_limit=time_limit or 3000,
        #test_size="2025-01-01",  # None, or set to date string like "2025-06-01"
        test_size=None,  # None, or set to date string like "2025-06-01"
        features=features,  # Use explicit feature list for 'win' model
        included_strings=included_strings,
        excluded_strings=excluded_strings,
        required_strings=required_strings,
        start_date='2014-01-01',
        num_fights=2,
        include_split_dec=include_split_dec,
        normalize='robust',
        use_recency_weights=True,
        decay_rate=0.15,
        #decay_rate=0,
        split_strategy=split_strategy or 'timeseries_split',  # 'standard', 'timeseries_split', or 'walkforward'
        calculate_importance=True,  # Set to True to calculate feature importance
        refit_all=False, # this only applies to timeseries_split. Walkforward already uses refit_all internally.
        refit_full=True if refit_full is None else refit_full, # this only applies to timeseries_split. Walkforward already uses refit_full internally.
        included_model_types=['TABICL', 'MITRA', 'TABM', 'GBM_PREP', 'CAT', 'GBM', 'REALTABPFN-V2'] # all models from hyperparameters (TABDPT excluded due to PyTorch Nightly attention kernel compatibility)
        #included_model_types=['GBM', 'XGB', 'CAT'] # fast for importance testing
    )
    
    # Print configuration before training
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model Type:           {config.model_type}")
    print(f"Preset:               {config.preset}")
    print(f"Time Limit:           {config.time_limit} seconds ({config.time_limit / 60:.1f} minutes)")
    print(f"\nData Split:")
    print(f"  Test Size (cutoff):  {config.test_size if config.test_size else 'None (no holdout)'}")
    print(f"  Split Strategy:     {config.split_strategy}")
    if config.split_strategy == 'walkforward':
        print(f"  Walk-Forward Windows: {config.walkforward_n_windows}")
        print(f"  Initial Year:        {config.walkforward_initial_year}")
    print(f"  Refit All:         {config.refit_all}")
    print(f"\nFeature Configuration:")
    if config.features:
        print(f"  Explicit Features:  {len(config.features)} features")
        if len(config.features) <= 10:
            print(f"    {', '.join(config.features)}")
        else:
            print(f"    {', '.join(config.features[:10])} ... (+{len(config.features) - 10} more)")
    else:
        print(f"  Explicit Features:  None (using string filters)")
    print(f"  Included Strings:   {config.included_strings}")
    print(f"  Excluded Strings:   {config.excluded_strings}")
    print(f"  Required Strings:   {config.required_strings}")
    print(f"\nData Filtering:")
    print(f"  Start Date:         {config.start_date}")
    print(f"  Min Fights:         {config.num_fights}")
    print(f"  Include Split Dec:  {config.include_split_dec}")
    print(f"\nNormalization:")
    print(f"  Method:             {config.normalize}")
    print(f"\nRecency Weighting:")
    print(f"  Enabled:            {config.use_recency_weights}")
    if config.use_recency_weights:
        print(f"  Decay Rate:         {config.decay_rate}")
    print(f"\nFeature Importance:")
    print(f"  Calculate:         {config.calculate_importance}")
    print(f"\nModel Types:")
    if config.included_model_types:
        print(f"  Included:           {', '.join(config.included_model_types)}")
    else:
        print(f"  Included:           All (default for preset)")
    print("=" * 70 + "\n")
    
    # Train model
    trainer = ModelTrainer(config)
    predictor = trainer.train()
    
    print("\n=== Training Complete ===")
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(description="Train an MMA prediction model from data/training_data*.csv.")
    parser.add_argument("--model-type", choices=["win", "decision"], default="win")
    parser.add_argument("--time-limit", type=int, default=None, help="AutoGluon training time limit in seconds.")
    parser.add_argument("--preset", choices=["extreme", "best"], default=None)
    parser.add_argument("--split-strategy", choices=["standard", "timeseries_split", "walkforward"], default=None)
    parser.add_argument("--no-refit-full", action="store_true", help="Disable AutoGluon refit_full after validation.")
    return parser.parse_args()


def cli():
    args = parse_args()
    return main(
        model_type=args.model_type,
        time_limit=args.time_limit,
        preset=args.preset,
        split_strategy=args.split_strategy,
        refit_full=not args.no_refit_full,
    )


if __name__ == "__main__":
    cli()
