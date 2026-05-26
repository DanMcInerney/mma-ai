import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, brier_score_loss
from autogluon.tabular import TabularPredictor
from sqlalchemy import create_engine, text
import joblib
from typing import Dict, Tuple
#from libs.modeling.train import split_data, load_training_data, filter_fights, apply_robust_normalization
from libs.modeling.profit import prepare_features
from libs.modeling.split_date_utils import get_test_start_date_as_timestamp
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import json
from libs.paths import database_url, models_dir

class ModelUtils:
    def __init__(self, model_path: str, training_data_path: str, scaler_path: str, feats: list, predictor=None):
        """
        Initialize ModelUtils with paths to required files. This doesn't filter certain fights like split decision (but must remove draws, and no contests) so the results will be different than from train.py evals.
        
        :param model_path: Path to the saved AutoGluon model (and folder for saving calibration curve and stats)
        :param training_data_path: Path to the training data CSV
        :param scaler_path: Path to the saved scaler
        :param feats: List of features to use.
        :param predictor: Optional predictor instance (TabularPredictor or compatible). If None, loads from model_path.
        """
        self.model_path = model_path
        self.training_data_path = training_data_path
        self.scaler_path = scaler_path
        self.feats = feats
        self.engine = create_engine(database_url())
        
        # Load model and filter data
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = TabularPredictor.load(model_path)
        orig_df = pd.read_csv(training_data_path)
        
        # Convert event_date to datetime objects immediately after loading
        orig_df['event_date'] = pd.to_datetime(orig_df['event_date'])
        
        # Sort and reset index as early as possible to ensure deterministic ordering
        # This ensures consistent filtering and evaluation across runs
        orig_df = orig_df.sort_values(by=['event_date', 'fight_id']).reset_index(drop=True)
        self.df = orig_df
        # Only load scaler if scaler_path is provided (ensemble models handle scaling internally)
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
        else:
            self.scaler = None
        
        # Check for and load calibrator if it exists
        self.calibrator = None
        calibrator_path = os.path.join(model_path, 'calibrator.pkl')
        if os.path.exists(calibrator_path):
            try:
                self.calibrator = joblib.load(calibrator_path)
                print(f"Loaded calibrator from {calibrator_path}")
            except Exception as e:
                print(f"Warning: Could not load calibrator from {calibrator_path}: {e}")
                self.calibrator = None
        
        # Prepare test data
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """Prepare train/test splits and scale features using prepare_features."""
        
        # Filter out rows where y is not 0 or 1 from the original dataframe
        # Reset index after filtering to maintain clean sequential index for alignment
        self.df = self.df[self.df['y_true'].isin([0, 1])].copy().reset_index(drop=True)
        
        # Store necessary columns before prepare_features removes them or filters rows
        y_true_original = self.df['y_true']
        event_date_original = self.df['event_date']
        
        # Filter fights using prepare_features (returns only X)
        X = prepare_features(self.df, self.feats) # Pass the already filtered df

        # Align y and event_date with the filtered X using its index
        y = y_true_original.loc[X.index]
        event_date_aligned = event_date_original.loc[X.index]

        # Use holdout fight IDs to determine test set (if available)
        holdout_fight_ids_path = os.path.join(self.model_path, 'holdout_fight_ids.txt')
        has_holdout = os.path.exists(holdout_fight_ids_path)
        
        if has_holdout:
            # Read holdout fight IDs
            with open(holdout_fight_ids_path, 'r') as f:
                holdout_fight_ids = set(int(line.strip()) for line in f if line.strip())
            
            print(f"ModelUtils: Loaded {len(holdout_fight_ids)} holdout fight IDs from {holdout_fight_ids_path}")
            
            # Get fight_id column from original dataframe (aligned with X index after prepare_features filtering)
            fight_ids_aligned = self.df.loc[X.index, 'fight_id']
            
            # Split based on fight IDs (only fights that survived prepare_features filtering)
            test_mask = fight_ids_aligned.isin(holdout_fight_ids)
            train_mask = ~test_mask
            
            # Count how many holdout fights survived filtering
            holdout_found = test_mask.sum()
            holdout_missing = len(holdout_fight_ids) - holdout_found
            
            if holdout_missing > 0:
                print(f"ModelUtils: Warning - {holdout_missing} holdout fights were filtered out by prepare_features() "
                      f"(likely due to missing odds data or NaN values in features). Using {holdout_found} remaining fights.")
            
            self.X_test = X[test_mask]
            self.y_test = y[test_mask]
            self.X_train = X[train_mask]
            self.y_train = y[train_mask]
            
            print(f"ModelUtils: Test set size: {len(self.X_test)}, Train set size: {len(self.X_train)}")
        else:
            # No holdout set - use all data as training data
            print(f"ModelUtils: No holdout set found (holdout_fight_ids.txt not found at {holdout_fight_ids_path}). "
                  f"Using all {len(X)} samples as training data.")
            self.X_test = X.iloc[0:0].copy()  # Empty DataFrame with same columns
            self.y_test = y.iloc[0:0].copy()  # Empty Series with same dtype
            self.X_train = X.copy()
            self.y_train = y.copy()
            
            print(f"ModelUtils: Test set size: 0, Train set size: {len(self.X_train)}")

        # Check if predictor is an EnsemblePredictor (handles scaling internally)
        from libs.modeling.train import EnsemblePredictor, EnsemblePredictorWrapper
        is_ensemble = isinstance(self.predictor, (EnsemblePredictor, EnsemblePredictorWrapper))
        if isinstance(self.predictor, EnsemblePredictorWrapper):
            is_ensemble = True  # Wrapper contains ensemble predictor
        
        # Scale the data (only features that were scaled during training)
        # Skip scaling if using ensemble predictor (it handles scaling internally per window)
        if is_ensemble:
            # Ensemble predictor handles scaling internally - don't scale here
            print("ModelUtils: Ensemble predictor detected - skipping scaling (ensemble handles it internally)")
            self.X_train_scaled = self.X_train.copy()
            self.X_test_scaled = self.X_test.copy()
        else:
            # Single model: scale using the provided scaler
            scaler = joblib.load(self.scaler_path)
            # Check if X_train/X_test are empty before scaling
            if not self.X_train.empty and not self.X_test.empty:
                # Identify which features should be scaled (matching training logic)
                date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
                categorical_static_feats = ['weightclass_encoded', 'odds']
                
                def should_exclude_col(col_name):
                    """Check if column should be excluded from scaling."""
                    if col_name in date_cols:
                        return True
                    # Exclude if column name contains categorical static feature strings
                    for cat_feat in categorical_static_feats:
                        if cat_feat in col_name:
                            return True
                    return False
                
                features_to_scale = [col for col in self.X_train.columns if not should_exclude_col(col)]
                
                # Scale only the features that were scaled during training
                self.X_train_scaled = self.X_train.copy()
                self.X_test_scaled = self.X_test.copy()
                
                if len(features_to_scale) > 0:
                    self.X_train_scaled[features_to_scale] = scaler.transform(self.X_train[features_to_scale])
                    self.X_test_scaled[features_to_scale] = scaler.transform(self.X_test[features_to_scale])
            else:
                # Empty train/test sets - just copy
                self.X_train_scaled = self.X_train.copy()
                self.X_test_scaled = self.X_test.copy()
    
    def _get_performance_metrics(self, y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate common performance metrics."""
        # Handle empty arrays (no test set)
        if len(y_true) == 0 or len(y_pred) == 0 or len(y_prob) == 0:
            return {}
        
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_prob),
            "brier_score": brier_score_loss(y_true, y_prob)
        }
    
    def _get_vegas_odds(self) -> pd.DataFrame:
        """
        Fetch normalized Vegas probabilities from database.
        """
        query = text("""
            SELECT o.fight_id,
                   o.fighter_id,
                   fm.fighter_name,
                   o.ip_closing_odds as implied_prob
              FROM features.odds o
              JOIN features.fighter_mapping fm ON o.fighter_id = fm.fighter_id
             WHERE o.fight_id IS NOT NULL 
               AND o.fighter_id IS NOT NULL
               AND o.ip_closing_odds IS NOT NULL
          ORDER BY o.fight_id, o.fighter_id
        """)
        return pd.read_sql(query, con=self.engine)
    
    def _rename_odds(self, odds: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Rename columns of the odds DataFrame to prepare for merging.
        
        Returns two DataFrames: one for fighter1 and one for fighter2.
        """
        odds_f1 = odds.rename(columns={
            'fighter_id': 'fighter1_id', 
            'implied_prob': 'f1_prob',
            'fighter_name': 'fighter1_name'
        })
        odds_f2 = odds.rename(columns={
            'fighter_id': 'fighter2_id', 
            'implied_prob': 'f2_prob',
            'fighter_name': 'fighter2_name'
        })
        return odds_f1, odds_f2

    def _merge_odds(self, fights: pd.DataFrame, use_index_mapping: bool = False) -> pd.DataFrame:
        """
        Merge Vegas odds with a provided fights DataFrame.
        
        :param fights: DataFrame containing fight data with columns 'fight_id' and fighter ID columns.
        :param use_index_mapping: If True, reassign the merged DataFrame index using the original df's mapping.
        :return: Merged DataFrame with odds for fighter1 and fighter2.
        """
        odds = self._get_vegas_odds()
        odds_f1, odds_f2 = self._rename_odds(odds)
        merged = pd.merge(
            fights,
            odds_f1[['fight_id', 'fighter1_id', 'fighter1_name', 'f1_prob']],
            on=['fight_id', 'fighter1_id'],
            how='left'
        )
        merged = pd.merge(
            merged,
            odds_f2[['fight_id', 'fighter2_id', 'fighter2_name', 'f2_prob']],
            on=['fight_id', 'fighter2_id'],
            how='left'
        ).dropna(subset=['f1_prob', 'f2_prob'])
        
        if use_index_mapping:
            # Map fight_id to original index from self.df.
            fight_to_index = self.df.reset_index().set_index('fight_id')['index']
            merged.index = merged['fight_id'].map(fight_to_index)
        return merged
    
    def _get_model_predictions(self, test_data, use_calibrated=None):
        """
        Get model predictions (both binary and probabilities).
        
        :param test_data: Test data DataFrame
        :param use_calibrated: If True, use calibrated predictions; if False, use original; if None, auto-detect
        :return: Tuple of (y_pred, y_prob)
        """
        # Determine whether to use calibrated predictions
        if use_calibrated is None:
            use_calibrated = self.calibrator is not None
        elif use_calibrated and self.calibrator is None:
            print("Warning: Calibrated predictions requested but no calibrator available. Using original predictions.")
            use_calibrated = False
        
        # Get original predictions
        try:
            y_pred = self.predictor.predict(test_data)
            y_prob = self.predictor.predict_proba(test_data)
        except KeyError as e:
            if 'sample_weight' in str(e):
                # Model was trained with sample_weight, add it with uniform values
                test_data = test_data.copy()
                test_data['sample_weight'] = 1.0
                y_pred = self.predictor.predict(test_data)
                y_prob = self.predictor.predict_proba(test_data)
            else:
                raise e
        
        if isinstance(y_prob, pd.DataFrame):
            # Handle both string ('0', '1') and integer (0, 1) column names
            if '1' in y_prob.columns:
                y_prob = y_prob['1'].values
            elif 1 in y_prob.columns:
                y_prob = y_prob[1].values
            else:
                # Fallback to second column if neither '1' nor 1 exists
                y_prob = y_prob.iloc[:, 1].values
        else:
            y_prob = y_prob[:, 1]
        
        # Apply calibration if requested and available
        if use_calibrated and self.calibrator is not None:
            # Preferred: probability-only calibrator
            try:
                y_prob_cal = self.calibrator.predict_proba(y_prob)
                if hasattr(y_prob_cal, 'ndim') and getattr(y_prob_cal, 'ndim', 1) == 2 and y_prob_cal.shape[1] == 2:
                    y_prob_cal = y_prob_cal[:, 1]
                y_prob = y_prob_cal
                y_pred = (y_prob > 0.5).astype(int)
            except Exception:
                # Fallback: feature-based calibrator
                try:
                    test_data_clean = test_data.drop(columns=['sample_weight'], errors='ignore')
                    y_prob_fb = self.calibrator.predict_proba(test_data_clean)
                    if isinstance(y_prob_fb, pd.DataFrame):
                        y_prob = y_prob_fb[1].values
                    else:
                        y_prob = y_prob_fb[:, 1]
                    y_pred = (y_prob > 0.5).astype(int)
                except Exception as e:
                    print(f"Warning: Could not generate calibrated predictions: {e}. Using original predictions.")
        
        return y_pred, y_prob
    
    def get_model_performance(self, use_calibrated=None) -> Dict:
        """
        Get model performance metrics on the test set.
        
        :param use_calibrated: If True, use calibrated predictions; if False, use original; if None, auto-detect
        """
        test_data = self.X_test_scaled.copy()
        y_pred, y_prob = self._get_model_predictions(test_data, use_calibrated)
        return self._get_performance_metrics(self.y_test, y_pred, y_prob)
    
    def get_vegas_performance(self) -> Dict:
        """Get Vegas odds performance metrics on the test set."""
        if self.X_test.empty:
            print("ModelUtils: No test set available (no holdout set). Cannot compute Vegas performance metrics.")
            return {}
        
        # Use fights corresponding to X_test from the filtered df.
        test_fights = self.df.loc[self.X_test.index].copy()
        merged = self._merge_odds(test_fights, use_index_mapping=True)
        y_prob_vegas = merged['f1_prob'].values
        y_pred_vegas = (merged['f1_prob'] > merged['f2_prob']).astype(int)
        y_test_filtered = self.y_test[merged.index]
        return self._get_performance_metrics(y_test_filtered, y_pred_vegas, y_prob_vegas)
    
    def get_vegas_performance_all_fights(self) -> Dict:
        """Get Vegas odds performance metrics on all fights from X_test start date onward."""
        if self.X_test.empty:
            print("ModelUtils: No test set available (no holdout set). Cannot compute Vegas performance metrics.")
            return {}
        
        earliest_test_date = self.df.loc[self.X_test.index, 'event_date'].min()
        query = text("""
            WITH fight_results AS (
                SELECT 
                    fm.fight_id,
                    fm.fighter1_id,
                    fm.fighter2_id,
                    CASE WHEN fm.result = 1 THEN 1 ELSE 0 END as fighter1_won,
                    em.event_date
                FROM features.fight_mapping fm
                JOIN features.event_mapping em ON fm.event_id = em.event_id
                WHERE em.event_date >= :start_date
            )
            SELECT * FROM fight_results
            ORDER BY event_date, fight_id
        """)
        all_fights = pd.read_sql(query, con=self.engine, params={'start_date': earliest_test_date})
        merged = self._merge_odds(all_fights)
        y_prob_vegas = merged['f1_prob'].values
        y_pred_vegas = (merged['f1_prob'] > merged['f2_prob']).astype(int)
        y_true = merged['fighter1_won'].values
        return self._get_performance_metrics(y_true, y_pred_vegas, y_prob_vegas)
    
    def plot_calibration_curve(self, n_bins=10, include_all_fights: bool = False):
        """
        Plot calibration curve comparing model and Vegas predictions against actual outcomes.
        Also includes calibrated predictions if a calibrator is available.
        Also save the figure to the model path folder.
        
        :param n_bins: Number of bins for calculating calibration curve.
        :param include_all_fights: If True, also plot the calibration curve for all fights (from X_test start date onward).
        """
        # Check if test set is available
        if self.X_test.empty:
            print("ModelUtils: No test set available (no holdout set). Skipping calibration curve plot.")
            return
        
        # Model predictions from test set.
        # Add sample_weight column if needed
        test_data = self.X_test_scaled.copy()
        try:
            y_prob_model = self.predictor.predict_proba(test_data)
        except KeyError as e:
            if 'sample_weight' in str(e):
                test_data['sample_weight'] = 1.0
                y_prob_model = self.predictor.predict_proba(test_data)
            else:
                raise e
                
        if isinstance(y_prob_model, pd.DataFrame):
            y_prob_model = y_prob_model[1].values
        else:
            y_prob_model = y_prob_model[:, 1]
        
        # Get calibrated predictions if calibrator is available
        y_prob_calibrated = None
        if self.calibrator is not None:
            # Preferred: probability-only calibrator
            try:
                y_prob_cal = self.calibrator.predict_proba(y_prob_model)
                if hasattr(y_prob_cal, 'ndim') and getattr(y_prob_cal, 'ndim', 1) == 2 and y_prob_cal.shape[1] == 2:
                    y_prob_cal = y_prob_cal[:, 1]
                y_prob_calibrated = y_prob_cal
                print("Using calibrated predictions for calibration curve")
            except Exception:
                # Fallback: feature-based calibrator
                try:
                    test_data_clean = test_data.drop(columns=['sample_weight'], errors='ignore')
                    y_prob_fb = self.calibrator.predict_proba(test_data_clean)
                    if isinstance(y_prob_fb, pd.DataFrame):
                        y_prob_calibrated = y_prob_fb[1].values
                    else:
                        y_prob_calibrated = y_prob_fb[:, 1]
                    print("Using feature-based calibrated predictions for calibration curve")
                except Exception as e:
                    print(f"Warning: Could not generate calibrated predictions: {e}")
                    y_prob_calibrated = None
        
        # Calibration for test set (using merged odds from test fights)
        if self.X_test.empty:
            print("ModelUtils: No test set available (no holdout set). Skipping calibration curve plot.")
            return
        
        test_fights = self.df.loc[self.X_test.index].copy()
        merged_test = self._merge_odds(test_fights, use_index_mapping=True)
        y_prob_vegas_test = merged_test['f1_prob'].values
        y_test_test = self.y_test.loc[merged_test.index]
        prob_true_model, prob_pred_model = calibration_curve(self.y_test, y_prob_model, n_bins=n_bins)
        prob_true_vegas, prob_pred_vegas = calibration_curve(y_test_test, y_prob_vegas_test, n_bins=n_bins)
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(prob_pred_model, prob_true_model, 's-', label='Model (Test)')
        
        # Add calibrated curve if available
        if y_prob_calibrated is not None:
            prob_true_calibrated, prob_pred_calibrated = calibration_curve(self.y_test, y_prob_calibrated, n_bins=n_bins)
            plt.plot(prob_pred_calibrated, prob_true_calibrated, '^-', label='Model Calibrated (Test)', alpha=0.8)
        
        plt.plot(prob_pred_vegas, prob_true_vegas, 'o-', label='Vegas (Test)')
        
        # Optionally, also plot calibration for all fights.
        if include_all_fights:
            query = text("""
                WITH fight_results AS (
                    SELECT 
                        fm.fight_id,
                        fm.fighter1_id,
                        fm.fighter2_id,
                        CASE WHEN fm.result = 1 THEN 1 ELSE 0 END as fighter1_won,
                        em.event_date
                    FROM features.fight_mapping fm
                    JOIN features.event_mapping em ON fm.event_id = em.event_id
                    WHERE em.event_date >= :start_date
                )
                SELECT * FROM fight_results
                ORDER BY event_date, fight_id
            """)
            earliest_test_date = self.df.loc[self.X_test.index, 'event_date'].min()
            all_fights = pd.read_sql(query, con=self.engine, params={'start_date': earliest_test_date})
            merged_all = self._merge_odds(all_fights)
            y_prob_vegas_all = merged_all['f1_prob'].values
            y_true_all = merged_all['fighter1_won'].values
            prob_true_all, prob_pred_all = calibration_curve(y_true_all, y_prob_vegas_all, n_bins=n_bins)
            plt.plot(prob_pred_all, prob_true_all, 'd-', label='Vegas (All Fights)')
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        
        # Save the figure to the model folder.
        save_path = os.path.join(self.model_path, "calibration_curve.png")
        plt.savefig(save_path)
        print(f"Calibration curve saved to {save_path}")
        
        #plt.show()
    
    def print_feature_importance(self, top_n: int = None):
        """
        Print feature importance from the AutoGluon model.
        
        :param top_n: Optional number of top features to display.
        """
        train_data = self.X_train_scaled.copy()
        train_data['y_true'] = self.y_train
        
        # Check if model needs sample_weight column
        try:
            # Try a quick prediction to see if sample_weight is needed
            self.predictor.predict(self.X_train_scaled.iloc[:1])
        except KeyError as e:
            if 'sample_weight' in str(e):
                train_data['sample_weight'] = 1.0
                
        importance = self.predictor.feature_importance(train_data)
        importance_df = importance.sort_values('importance', ascending=False)
        if top_n:
            importance_df = importance_df.head(top_n)
        print("\nFeature Importance:")
        print("------------------")
        for idx, row in importance_df.iterrows():
            print(f"{idx}: {row['importance']:.4f}")
    
    def print_performance_summary(self):
        """Print a comprehensive performance summary including calibrated results if available."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        # Check if test set is available
        if self.X_test.empty:
            print("\nNo test/holdout set available - cannot compute test performance metrics")
        else:
            # Original model performance
            model_perf = self.get_model_performance(use_calibrated=False)
            if model_perf:  # Only print if metrics were computed
                print(f"\nOriginal Model Performance:")
                print(f"  Accuracy:    {model_perf['accuracy']:.4f}")
                print(f"  Log Loss:    {model_perf['log_loss']:.4f}")
                print(f"  Brier Score: {model_perf['brier_score']:.4f}")
            
            # # Calibrated model performance if available
            # if self.calibrator is not None:
            #     model_perf_cal = self.get_model_performance(use_calibrated=True)
            #     improvement_log_loss = model_perf['log_loss'] - model_perf_cal['log_loss']
            #     improvement_brier = model_perf['brier_score'] - model_perf_cal['brier_score']
                
            #     print(f"\nCalibrated Model Performance:")
            #     print(f"  Accuracy:    {model_perf_cal['accuracy']:.4f}")
            #     print(f"  Log Loss:    {model_perf_cal['log_loss']:.4f} (Δ: {improvement_log_loss:+.4f})")
            #     print(f"  Brier Score: {model_perf_cal['brier_score']:.4f} (Δ: {improvement_brier:+.4f})")
                
            #     if improvement_log_loss > 0:
            #         print(f"  ✓ Calibration improved log loss by {improvement_log_loss:.4f}")
            #     else:
            #         print(f"  ⚠ Calibration worsened log loss by {abs(improvement_log_loss):.4f}")
            # else:
            #     print(f"\nNo calibrator available")
            
            # Vegas performance for comparison
            try:
                vegas_perf = self.get_vegas_performance()
                if vegas_perf:  # Only print if metrics were computed
                    print(f"\nVegas Odds Performance (Test Set):")
                    print(f"  Accuracy:    {vegas_perf['accuracy']:.4f}")
                    print(f"  Log Loss:    {vegas_perf['log_loss']:.4f}")
                    print(f"  Brier Score: {vegas_perf['brier_score']:.4f}")
            except Exception as e:
                print(f"\nVegas performance not available: {e}")
        
        print("="*60)
    
    def print_comparison(self):
        """Print formatted comparison of model vs Vegas performance."""
        # Check if test set is available
        if self.X_test.empty:
            print("No test/holdout set available - cannot compute test performance metrics")
            comparison = {
                "note": "No test/holdout set available - only training metrics computed"
            }
        else:
            model_perf = self.get_model_performance(use_calibrated=False)  # Original model
            vegas_perf = self.get_vegas_performance()
            vegas_perf_all = self.get_vegas_performance_all_fights()
            
            comparison = {
                "vegas_odds_performance": vegas_perf,
                "vegas_odds_performance_all_fights": vegas_perf_all,
                "mma_ai_performance": model_perf
            }
            
            # Add calibrated performance if calibrator exists
            if self.calibrator is not None:
                model_perf_cal = self.get_model_performance(use_calibrated=True)
                comparison["mma_ai_performance_calibrated"] = model_perf_cal
                print("Note: Calibrated model performance included")
        
        print(json.dumps(comparison, indent=2))
    
    def save_model_stats(self, importance=False):
        """
        Combine comparison data (model and Vegas performance) with feature importance
        and save the output to 'model_stats.txt' in the model path folder.
        """
        # Check if test set is available
        if self.X_test.empty:
            print("ModelUtils: No test set available (no holdout set). Skipping test performance metrics.")
            comparison = {
                "note": "No test/holdout set available - only training metrics computed"
            }
        else:
            # Get performance comparison.
            model_perf = self.get_model_performance(use_calibrated=False)  # Original model
            vegas_perf = self.get_vegas_performance()
            vegas_perf_all = self.get_vegas_performance_all_fights()
            
            comparison = {
                "vegas_odds_performance": vegas_perf,
                "vegas_odds_performance_all_fights": vegas_perf_all,
                "mma_ai_performance": model_perf
            }
            
            # Add calibrated performance if calibrator exists
            if self.calibrator is not None:
                model_perf_cal = self.get_model_performance(use_calibrated=True)
                comparison["mma_ai_performance_calibrated"] = model_perf_cal
        
        comparison_str = "Comparison:\n" + json.dumps(comparison, indent=2) + "\n\n"
        
        # # Get feature importance.
        # train_data = self.X_train_scaled.copy()
        # train_data['y_true'] = self.y_train
        
        # # Check if model needs sample_weight column
        # try:
        #     # Try a quick prediction to see if sample_weight is needed
        #     self.predictor.predict(self.X_train_scaled.iloc[:1])
        # except KeyError as e:
        #     if 'sample_weight' in str(e):
        #         train_data['sample_weight'] = 1.0
                
        # importance = self.predictor.feature_importance(train_data)
        # importance_df = importance.sort_values('importance', ascending=False)
        # feature_importance_str = "Feature Importance:\n"
        # for idx, row in importance_df.iterrows():
        #     feature_importance_str += f"{idx}: {row['importance']:.4f}\n"
        
        # Combine and save.
        #combined_output = comparison_str + feature_importance_str
        combined_output = comparison_str
        print(combined_output)
        file_path = os.path.join(self.model_path, "model_stats.txt")
        with open(file_path, "w") as f:
            f.write(combined_output)
        print(f"Model stats saved to {file_path}")

def main():
    model_name = os.getenv("MMA_AI_MODEL_NAME", "ag-20250610_202103")
    model_path = os.getenv("MMA_AI_MODEL_PATH", str(models_dir() / model_name))
    
    # Read features from feats.txt in the model directory
    feats_file_path = os.path.join(model_path, "feats.txt")
    with open(feats_file_path, 'r') as f:
            feats = [line.strip() for line in f if line.strip()]
        
    training_data_path = os.path.join(model_path, "training_data.csv")
    scaler_path = os.path.join(model_path, "scaler.pkl")
    
    utils = ModelUtils(model_path, training_data_path, scaler_path, feats)
    
    # Print comprehensive performance summary (includes calibrated results if available)
    utils.print_performance_summary()
    
    # Print JSON comparison for detailed analysis
    #utils.print_comparison()
    
    # Print feature importance
    #utils.print_feature_importance(top_n=50)
    
    # Plot and save calibration curve (automatically includes calibrated curve if available)
    utils.plot_calibration_curve(n_bins=10, include_all_fights=False)
    
    # Save combined stats to file (includes calibrated performance if available)
    utils.save_model_stats()

if __name__ == "__main__":
    main()
