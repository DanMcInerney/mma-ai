import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import optuna
import pandas as pd
from datetime import datetime
import joblib
from sqlalchemy import text
from autogluon.tabular import TabularPredictor
from typing import List, Dict, Any
import numpy as np

# Import your existing functions
from libs.modeling.data_preparation import DataPreparation

# Import train_model function (will be available after train.py is loaded)
def get_train_model():
    from libs.modeling.train import train_model
    return train_model

from libs.feature_store.features import FEAT_MAD2_AND_STYLES_TEST_FILTERED
from libs.paths import data_file

class HyperparameterOptimizer:
    def __init__(self, data_path: str, base_model_dir: str = 'OptimizedModels'):
        self.data_path = data_path
        self.base_model_dir = base_model_dir
        self.feats = FEAT_MAD2_AND_STYLES_TEST_FILTERED
        self.odds = False
        self.limit = 600  # Reduced time limit for faster optimization
        self.train_size = 0.75  # Train on 80%
        self.val_size = 0.15   # Validate on 10% 
        self.test_size = 0.10  # Final test on 10%
        self.fit = 'experimental'  # Fixed fit strategy
        self.hp = 'default'  # Fixed hyperparameters
        
        # Load and prepare data once
        self._prepare_data()
        
        # Store the test set separately for final unbiased evaluation
        self.final_test_data = None
        
    def _prepare_data(self):
        """Load and prepare data once for all trials."""
        print("Loading and preparing data...")
        print(f"Data path: {self.data_path}")
        print(f"Features: {len(self.feats)} selected")
        print(f"Odds enabled: {self.odds}")
        print("Data preparation will be done per trial with varying hyperparameters")
    
    def objective(self, trial):
        """Optuna objective function to maximize AutoGluon's negative log_loss."""
        
        # Suggest hyperparameters
        n_splits = trial.suggest_categorical('n_splits', [3, 4, 5, 6, 8])
        use_recency_weights = trial.suggest_categorical('use_recency_weights', [True])
        use_bag_holdout = trial.suggest_categorical('use_bag_holdout', [True])
        decay_rate = trial.suggest_categorical('decay_rate', [.7, .8, .9, .1, .11, .12, .13, .14, .15])
        num_bag_sets = trial.suggest_categorical('num_bag_sets', [1, 2, 3])
        num_stack_levels = trial.suggest_categorical('num_stack_levels', [0, 1, 2])
        normalize = trial.suggest_categorical('normalize', ['zscore', 'robust', 'none'])
        start_date = trial.suggest_categorical('start_date', ['2012-01-01', '2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01'])
        balance_fighters = trial.suggest_categorical('balance_fighters', [False])
        shuffle = trial.suggest_categorical('shuffle', [True, False])
        
        # Create unique model directory for this trial
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        trial_id = f"trial_{trial.number}_{timestamp}"
        model_dir = os.path.join(self.base_model_dir, trial_id)
        os.makedirs(model_dir, exist_ok=True)
        
        try:
            # Use DataPreparation class for consistent data handling
            data_prep = DataPreparation(
                data_path=self.data_path,
                feats=self.feats,
                odds=self.odds,
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                normalize=normalize,
                use_recency_weights=use_recency_weights,
                decay_rate=decay_rate,
                balance_fighters=balance_fighters,
                target_balance=0.5,  # Keep at 50/50 for hyperparameter optimization
                start_date=start_date,
                num_fights=2,
                include_split_dec=False
            )
            
            # Prepare data for this trial
            X_train, X_val, X_test, y_train, y_val, y_test = data_prep.prepare_data(model_dir)
            df_trial = data_prep.df
            sample_weight_col = data_prep.get_sample_weight_column()
            
            # Create predictor
            predictor = TabularPredictor(
                label='y_true',
                eval_metric='log_loss',
                problem_type='binary',
                path=model_dir,
                verbosity=0,  # Reduce verbosity for optimization
                sample_weight=sample_weight_col,
                weight_evaluation=False
            )
            
            # Train model using validation set for optimization
            train_model_func = get_train_model()
            predictor, results = train_model_func(
                X_train, y_train, X_val, y_val, predictor,
                fit=self.fit, hp=self.hp, limit=self.limit, prune=False,
                train_size=self.train_size, n_splits=n_splits,
                df=df_trial, use_recency_weights=use_recency_weights, 
                decay_rate=decay_rate, use_bag_holdout=use_bag_holdout, 
                num_stack_levels=num_stack_levels, importance=False,
                calibrate=True, shuffle=shuffle, num_bag_sets=num_bag_sets
            )
            
            # Get validation log_loss (our optimization target)
            val_log_loss = results['test_scores']['log_loss']  # Note: train_model calls it 'test_scores' but it's actually validation
            val_accuracy = results['test_scores']['accuracy']
            
            # Also evaluate on actual test set (for final comparison, not optimization)
            # Note: sample weights are already included in X_test from DataPreparation
            test_eval_data = pd.concat([X_test, y_test], axis=1)
            test_scores = predictor.evaluate(test_eval_data)
            
            # Log metrics for analysis
            trial.set_user_attr('val_log_loss', val_log_loss)
            trial.set_user_attr('val_accuracy', val_accuracy)
            trial.set_user_attr('test_log_loss', test_scores['log_loss'])
            trial.set_user_attr('test_accuracy', test_scores['accuracy'])
            trial.set_user_attr('train_log_loss', results['train_scores']['log_loss'])
            trial.set_user_attr('train_accuracy', results['train_scores']['accuracy'])
            trial.set_user_attr('model_dir', model_dir)
            
            print(f"Trial {trial.number}: Val Log Loss = {val_log_loss:.4f} (optimizing), Test Log Loss = {test_scores['log_loss']:.4f} (reference)")
            
            return val_log_loss  # Optimize based on validation performance
            
        except Exception as e:
            print(f"Trial {trial.number} failed: {str(e)}")
            # Clean up failed trial directory
            if os.path.exists(model_dir):
                import shutil
                shutil.rmtree(model_dir, ignore_errors=True)
            
            raise e
    
    def optimize(self, n_trials: int = 50, timeout: int = None):
        """Run optimization with Optuna."""
        
        # Create study
        study_name = f"mma_hyperopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction='maximize',  # Maximize AutoGluon's negative log_loss (higher is better)
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner(  # Prune unpromising trials
                n_startup_trials=5,
                n_warmup_steps=3,
                interval_steps=1
            )
        )
        
        print(f"Starting optimization with {n_trials} trials...")
        print(f"Optimization target: Maximize AutoGluon's negative log_loss (higher = better performance)")
        print(f"Data split: {self.train_size*100:.0f}% train, {self.val_size*100:.0f}% validation, {self.test_size*100:.0f}% test")
        
        # Run optimization
        study.optimize(
            self.objective, 
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Save results
        self._save_results(study)
        
        return study
    
    def _save_results(self, study):
        """Save optimization results."""
        results_dir = os.path.join(self.base_model_dir, 'optimization_results')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save study object
        study_path = os.path.join(results_dir, f'study_{timestamp}.pkl')
        joblib.dump(study, study_path)
        
        # Create results summary
        results_path = os.path.join(results_dir, f'results_{timestamp}.txt')
        with open(results_path, 'w') as f:
            f.write("=== Hyperparameter Optimization Results ===\n\n")
            
            # Best trial
            best_trial = study.best_trial
            f.write(f"Best Trial #{best_trial.number}:\n")
            f.write(f"  Validation Log Loss: {best_trial.value:.6f} (optimization target)\n")
            f.write(f"  Validation Accuracy: {best_trial.user_attrs.get('val_accuracy', 'N/A'):.4f}\n")
            f.write(f"  Test Log Loss: {best_trial.user_attrs.get('test_log_loss', 'N/A'):.6f} (unbiased estimate)\n")
            f.write(f"  Test Accuracy: {best_trial.user_attrs.get('test_accuracy', 'N/A'):.4f}\n")
            f.write(f"  Model Directory: {best_trial.user_attrs.get('model_dir', 'N/A')}\n\n")
            
            f.write("Best Parameters:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            
            f.write(f"\n=== All Trials Summary ===\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"Failed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
            
            # Top 10 trials
            f.write(f"\n=== Top 10 Trials ===\n")
            completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            sorted_trials = sorted(completed_trials, key=lambda x: x.value)[:10]  # Sort ascending for minimization
            
            for i, trial in enumerate(sorted_trials, 1):
                f.write(f"{i}. Trial #{trial.number}: Log Loss = {trial.value:.6f}\n")
                for key, value in trial.params.items():
                    f.write(f"   {key}: {value}\n")
                f.write("\n")
        
        # Save DataFrame of all trials
        df_results = study.trials_dataframe()
        csv_path = os.path.join(results_dir, f'trials_{timestamp}.csv')
        df_results.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to {results_dir}")
        print(f"Best trial: #{study.best_trial.number}")
        print(f"  Validation log_loss: {study.best_trial.value:.6f} (optimization target)")
        if 'test_log_loss' in study.best_trial.user_attrs:
            print(f"  Test log_loss: {study.best_trial.user_attrs['test_log_loss']:.6f} (unbiased estimate)")
        print(f"Best parameters: {study.best_trial.params}")
    
    def get_final_test_performance(self, study):
        """Get final unbiased test performance using the best hyperparameters."""
        print("\n" + "="*50)
        print("FINAL TEST SET EVALUATION")
        print("="*50)
        print("Evaluating best hyperparameters on unseen test data...")
        
        best_trial = study.best_trial
        best_params = best_trial.params
        
        # Use the best hyperparameters to train and evaluate on the true test set
        # This gives us an unbiased estimate of real-world performance
        
        print("Best hyperparameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        print(f"\nFinal test performance (unbiased):")
        if 'test_log_loss' in best_trial.user_attrs:
            print(f"  Test Log Loss: {best_trial.user_attrs['test_log_loss']:.6f}")
            print(f"  Test Accuracy: {best_trial.user_attrs['test_accuracy']:.4f}")
        
        return best_trial.user_attrs.get('test_log_loss', None)

def main():
    """Run hyperparameter optimization."""
    
    # Configuration
    data_path = str(data_file('training_data.csv'))
    n_trials = 100  # Adjust based on your computational budget
    timeout = None  # Set timeout in seconds if needed (e.g., 3600 for 1 hour)
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(data_path)
    
    # Run optimization
    study = optimizer.optimize(n_trials=n_trials, timeout=timeout)
    
    # Print final results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best validation log_loss: {study.best_trial.value:.6f} (optimization target)")
    if 'test_log_loss' in study.best_trial.user_attrs:
        print(f"Best test log_loss: {study.best_trial.user_attrs['test_log_loss']:.6f} (unbiased estimate)")
    print("Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Get final unbiased test performance
    optimizer.get_final_test_performance(study)

if __name__ == "__main__":
    main() 
