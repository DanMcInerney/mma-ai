import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from sqlalchemy import text
from autogluon.tabular import TabularPredictor
from typing import List
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.calibration import calibration_curve
from libs.feature_store.features import FEATS_MAD, LAYERED_TEST_FEATS, TEST_FEATS, FEATS_MAD2, FEATS_MAD3, FEATS_MAD21, FEATS_TOP100, STYLE_FEATS, FeatureSelector, FEATS_TOP_FILTERED, FEATS_STYLE, FEAT_MAD2_AND_STYLES, FEAT_MAD2_AND_STYLES_TEST, FEAT_MAD2_AND_STYLES_TEST_FILTERED
import joblib
from datetime import datetime
from libs.modeling.model_utils import ModelUtils
from libs.modeling.split_date_utils import write_test_start_date
from libs.modeling.data_preparation import DataPreparation
import numpy as np
from libs.modeling.autogluon_wrapper import AutoGluonWrapper
from libs.paths import data_file

def split_data(X, y, train_size):
    """Performs chronological split with 85/15 train/test split."""
    total_rows = len(X)
    
    # Calculate split point
    train_end = int(total_rows * train_size)
    
    # Split X and y
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    X_test = X.iloc[train_end:]
    y_test = y.iloc[train_end:]
    
    print(f"Split Percentages: Train={train_size*100:.1f}%, Test={(1-train_size)*100:.1f}%")
    print(f"Training data: {len(X_train)} fights ({X_train.index.min()} to {X_train.index.max()})")
    print(f"Test data: {len(X_test)} fights ({X_test.index.min()} to {X_test.index.max()})")
    
    return (X_train, y_train), (X_test, y_test)

def train_model(X_train, y_train, X_test, y_test, predictor, fit='medium_quality', hp='default', limit=1200, prune=False, train_size=0.9, n_splits=5, df=None, use_recency_weights=False, decay_rate=0.1, calibrate=True, use_bag_holdout=True, num_stack_levels=1, importance=True, X_val=None, y_val=None, shuffle=False, num_bag_sets=1):
    """Train model with properly separated features and target."""
    print("Training model...")

    included_model_types=['CAT', 'XGB', "GBM", 'FASTAI', 'NN_TORCH',]
    excluded_model_types=['KNN', 'TABPFNMIX']

    feature_prune_kwargs = {
        'num_features_to_prune': 20,  # Example: try removing 10 least important features at a time
        'max_num_pruning_rounds': 10,  # Example: do up to 3 rounds of pruning
        'force_prune': True           # force all models to use the pruned set if performance improves
    }

    # Variable to store sample weights for later use
    sample_weights = None

    if fit == 'time_ordered':
        # Create validation set from the same portion of training data that we took the test data off of
        train_split = int(len(X_train) * train_size)
        X_train_final = X_train.iloc[:train_split]
        y_train_final = y_train.iloc[:train_split]
        X_val = X_train.iloc[train_split:]
        y_val = y_train.iloc[train_split:]
        
        print(f"\nTime-ordered split:")
        print(f"Training data: {len(X_train_final)} fights")
        print(f"Validation data: {len(X_val)} fights")
        
        predictor.fit(
            train_data=pd.concat([X_train_final, y_train_final], axis=1),
            tuning_data=pd.concat([X_val, y_val], axis=1),
            time_limit=limit,
            hyperparameters='default',
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
            ag_args_fit={
                'shuffle': False,
                #'time_series': True  # Tell AutoGluon this is time-series data
            }
        )

    elif fit == 'time_ordered_custom':
        # Create validation set from the same portion of training data that we took the test data off of
        train_split = int(len(X_train) * train_size)
        X_train_final = X_train.iloc[:train_split]
        y_train_final = y_train.iloc[:train_split]
        X_val = X_train.iloc[train_split:]
        y_val = y_train.iloc[train_split:]
        
        print(f"\nTime-ordered split:")
        print(f"Training data: {len(X_train_final)} fights")
        print(f"Validation data: {len(X_val)} fights")
        
        predictor.fit(
            train_data=pd.concat([X_train_final, y_train_final], axis=1),
            tuning_data=pd.concat([X_val, y_val], axis=1),
            time_limit=limit,
            num_bag_folds=n_splits,
            num_bag_sets=2,
            use_bag_holdout=True,
            hyperparameters=hp,
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
            ag_args_fit={
                'shuffle': False,  # Maintain chronological order
                'time_series': True  # Tell AutoGluon this is time-series data
            }
        )
        
    elif fit == 'time_ordered_cv':
        # Time-ordered cross-validation for better temporal validation
        # This is a new preset specifically for time series data
        
        # Use expanding window validation
        val_size = int(len(X_train) * 0.1)  # 10% validation size
        
        # Dynamic stacking arguments for time series
        ds_args = {
            'detection_time_frac': 0.25,
            'validation_procedure': 'holdout',
            'holdout_frac': val_size,
            'memory_safe_fits': True,
            'clean_up_fits': True,  
            'enable_ray_logging': False
        }
        
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            time_limit=limit,
            presets=['good_quality'],  # Use good_quality as base
            hyperparameters=hp,
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
            num_bag_folds=n_splits,
            num_bag_sets=1,
            use_bag_holdout=True,
            dynamic_stacking=True,
            num_stack_levels=num_stack_levels,
            ds_args=ds_args,
            ag_args_fit={
                'shuffle': False,  # Maintain chronological order
                'time_series': True
            },
            ag_args_ensemble={
                'fold_fitting_strategy': 'sequential_local',  # Sequential for time series
                'use_orig_features': False,
                'max_base_models': 15,  # Limit total models
                'max_base_models_per_type': 3
            }
        )
        
    elif fit == 'best_quality':
        print("Fitting best_quality model...")
        
        # Feature pruning configuration for best_quality
        best_quality_prune_kwargs = {
            'num_features_to_prune': 15,  # Prune fewer features at a time
            'max_num_pruning_rounds': 5,  # More rounds for careful pruning
            'force_prune': False,  # Only prune if it improves validation score
            'stop_threshold': 0.0005  # Stop if improvement is less than 0.05%
        }
        
        # Dynamic stacking args to detect overfitting
        ds_args = {
            'detection_time_frac': 0.25,
            'validation_procedure': 'holdout',
            'holdout_frac': 0.15,  # Use 15% for overfitting detection
            'memory_safe_fits': True,
            'clean_up_fits': True
        }
        
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            presets='best_quality',
            time_limit=limit,
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
            refit_full=False,  # Don't refit to avoid overfitting on full data
            fit_strategy='parallel',
            # The best_quality preset already includes:
            # - auto_stack=True (automatically enables stacking)
            # - dynamic_stacking='auto' (will be True if not use_bag_holdout)
            # - hyperparameters='zeroshot' (~100 models from TabRepo)
            
            # Additional overfitting protection
            num_bag_folds=n_splits,  # Default for best_quality, good for stability
            num_bag_sets=1,  # Keep at 1 to avoid overfitting
            use_bag_holdout=False,  # Critical for preventing overfitting
            
            # Override dynamic stacking to ensure it's enabled
            dynamic_stacking=True,  # Force enable to detect stacking overfitting
            ds_args=ds_args,
            
            # Feature pruning to reduce overfitting
            feature_prune_kwargs=best_quality_prune_kwargs if prune else None,
            
            # Ensemble configuration to reduce overfitting
            ag_args_ensemble={
                'use_orig_features': False,  # Don't use original features in higher stack levels
                'max_base_models': 25,  # Limit total base models for L2
                'max_base_models_per_type': 5,  # Limit per model type
                'fold_fitting_strategy': 'sequential_local'  # Better for time series
            },
            
            # General model arguments
            ag_args_fit={
                'stopping_metric': 'log_loss',  # Use log_loss for early stopping
            },
            
            # Calibration for better probability estimates
            calibrate_decision_threshold=False,  # Keep False for your use case
            
            # Save disk space during training
            save_bag_folds=True,  # Keep True since refit_full=False
        )
    
    elif fit == 'experimental':
        
        # Prepare training and optional tuning (calibration) data
        train_data = pd.concat([X_train, y_train], axis=1)
        tuning_data = None
        if X_val is not None and y_val is not None:
            tuning_data = pd.concat([X_val, y_val], axis=1)
        
        predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            presets=fit,
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
            time_limit=limit,
            dynamic_stacking=False,
            # num_stack_levels=num_stack_levels,  # Must be 0 when num_bag_folds=0
            # num_bag_folds=n_splits, 
            # num_bag_sets=num_bag_sets,
            use_bag_holdout=use_bag_holdout,  # Important for preventing overfitting
            hyperparameters=hp,
            feature_prune_kwargs=None,
            fit_strategy='parallel',  # Experimental uses parallel by default
            ag_args_fit={
                'stopping_metric': 'log_loss',  # Use log_loss for early stopping
                'num_gpus': 1,  # Use 1 GPU for training
                'shuffle': shuffle, 
                #'time_series': True  # Tell AutoGluon this is time-series data
            },
            ag_args_ensemble={
                'use_orig_features': False,
                'max_base_models': 15,
                'max_base_models_per_type': 3,
                'fold_fitting_strategy': 'sequential_local', 
            },
            calibrate=calibrate  # Calibrate for better probability estimates
        )

    elif fit == 'experimental_default':
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            presets='experimental',
            time_limit=limit,
            included_model_types=included_model_types,
        )
    
    elif fit == "medium_quality":
        predictor.fit(
            train_data=pd.concat([X_train, y_train], axis=1),
            presets='medium_quality',
            time_limit=limit,
            refit_full=False,
            fit_strategy='parallel',
            included_model_types=included_model_types,
            excluded_model_types=excluded_model_types,
        )
    
    # Evaluate model
    print("\nModel Performance:")
    
    # Prepare evaluation data with sample weights if they were used during training
    train_eval_data = pd.concat([X_train, y_train], axis=1)
    if sample_weights is not None:
        train_eval_data['sample_weight'] = sample_weights
    
    train_scores = predictor.evaluate(train_eval_data)
    
    # Test data needs sample_weight column if it was used during training, but with uniform weights
    test_eval_data = pd.concat([X_test, y_test], axis=1)
    if sample_weights is not None:
        # Use uniform weights (all 1s) for test data
        test_eval_data['sample_weight'] = 1.0
        print("Added uniform sample weights to test data for evaluation")
    
    test_scores = predictor.evaluate(test_eval_data)

    # Calibration (tuning) split evaluation if available
    val_scores = None
    if X_val is not None and y_val is not None:
        val_eval_data = pd.concat([X_val, y_val], axis=1)
        if sample_weights is not None:
            val_eval_data['sample_weight'] = 1.0
            print("Added uniform sample weights to calibration data for evaluation")
        val_scores = predictor.evaluate(val_eval_data)

    print(f"Training accuracy: {train_scores['accuracy']:.4f}")
    print(f"Training log loss: {train_scores['log_loss']:.4f}")
    if val_scores is not None:
        print(f"Val accuracy: {val_scores['accuracy']:.4f}")
        print(f"Val log loss: {val_scores['log_loss']:.4f}")
    print(f"Test accuracy: {test_scores['accuracy']:.4f}")
    print(f"Test log loss: {test_scores['log_loss']:.4f}")

    print(predictor.model_best)

    best_model = predictor.model_best 
    weighted_ensemble_info = predictor.info()['model_info'][best_model]['children_info']['S1F1']['model_weights']
    print("Model weights:")
    for model, weight in weighted_ensemble_info.items():
        print(f"{model}: {weight:.3f}")
    
    # Feature importance (optional)
    feature_importance = None
    if importance:
        print("\nTop Most Important Features:")   
        
        # Use the same evaluation data for feature importance
        feature_importance = predictor.feature_importance(train_eval_data)
        
        # Print top 250 features one by one
        for i, (feature, importance_value) in enumerate(feature_importance.head(250).iterrows(), 1):
            print(f"{i}. {feature}: {importance_value['importance']:.4f}")

    # Prepare return values
    results = {
        'train_scores': train_scores,
        'test_scores': test_scores,
        'val_scores': val_scores,
        'importance': feature_importance,
        'model_weights': weighted_ensemble_info
    }
    
    return predictor, results

def main():
    # Create timestamp-based model directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    odds = False


    if odds:
        model_dir = os.path.join('AutogluonModels', f'ag-{timestamp}-odds')
    else:
        model_dir = os.path.join('AutogluonModels', f'ag-{timestamp}')
    os.makedirs(model_dir, exist_ok=True)
    
    # Define paths for all artifacts
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    feats_path = os.path.join(model_dir, 'feats.pkl')
    training_data_path = os.path.join(model_dir, 'training_data.csv')

    data_cutoff = None #'08-11-2024'
    #feats = None
    #feats = FEATS_TOP100
    #feats = FEATS_TOP_FILTERED
    #feats = FEATS_STYLE
    #feats = FEAT_MAD2_AND_STYLES
    #feats = FEAT_MAD2_AND_STYLES_TEST
    feats = FEAT_MAD2_AND_STYLES_TEST_FILTERED # v6.3
    #feats = FEATS_MAD
    #feats = TEST_FEATS
    #feats = FEATS_MAD2 # v61 819train 732test exp default hp feats_mad2 - also v62
    #feats = STYLE_FEATS
    #feats = FEATS_MAD21 # 21 features
    #feats = FEATS_MAD3
    #feats = FEATS_BALANCED_F1F2
    limit = 1200
    # Data split ratio (configurable) - Updated for calibration holdout
    train_size = 0.75     
    val_size = 0.15        
    test_size = 0.1      
    #fit = 'medium_quality'
    #fit = 'best_quality'
    #fit = 'time_ordered_custom'
    #fit = 'time_ordered_cv'
    fit = 'experimental'
    #fit = 'experimental_default'
    #fit = 'time_ordered'
    hp = 'default'
    #hp = 'zeroshot'
    prune = False
    normalize = 'robust'

    # Bagging/stacking
    n_splits = 5 # 4 is good for training, default for experimental
    num_stack_levels = 1 # 1 is good if we're stacking/bagging
    use_recency_weights = True  # Enable recency weighting
    use_bag_holdout = True # Must be true if we're using tuning_data (val split)
    num_bag_sets = 2
    decay_rate = 0.13
    shuffle = False
    start_date = '2014-01-01'
    #start_date = '2016-08-05' # 2016 rule change
    #start_date = '2017-01-01' # 2017 rule change
    calibrate = True

    # n_splits = 0 # 4 is good for training, default for experimental num_bag_folds
    # num_stack_levels = 0 # 1 is good if we're stacking/bagging
    # use_recency_weights = True  # Enable recency weighting
    # use_bag_holdout = False # Must be true if we're using tuning_data (val split)
    # decay_rate = 0.12
    # shuffle = False
    # start_date = '2014-01-01'
    # calibrate = False

    balance_fighters = False  # Enable fighter1/fighter2 balancing
    target_balance = 0.5  # Target 50/50 split
    
    # Prepare data using DataPreparation class
    data_prep = DataPreparation(
        data_path=str(data_file('training_data.csv')),
        feats=feats,
        odds=odds,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        normalize=normalize,
        use_recency_weights=use_recency_weights,
        decay_rate=decay_rate,
        balance_fighters=balance_fighters,
        target_balance=target_balance,
        start_date=start_date,
        num_fights=2,
        include_split_dec=False
    )
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.prepare_data(model_dir)
    df = data_prep.df  # Get the filtered dataframe for later use
    sample_weight_col = data_prep.get_sample_weight_column()

    # Train model with specified path and sample weight column
    predictor = TabularPredictor(
        label='y_true',
        eval_metric='log_loss',
        problem_type='binary',
        path=model_dir,
        verbosity=2,
        sample_weight=sample_weight_col,  # Tell AutoGluon which column contains sample weights
        weight_evaluation=False  # Don't use weights for evaluation metrics
    )
    
    # Train AutoGluon on training data only (not calibration set)
    predictor, results = train_model(
        X_train, y_train, X_test, y_test, predictor,
        fit=fit, hp=hp, limit=limit, prune=prune,
        train_size=train_size, n_splits=n_splits,
        df=df, use_recency_weights=use_recency_weights, decay_rate=decay_rate,
        calibrate=calibrate,
        use_bag_holdout=use_bag_holdout, num_stack_levels=num_stack_levels,
        importance=True, X_val=X_val, y_val=y_val, shuffle=shuffle, num_bag_sets=num_bag_sets
    )
    
    # Manual calibration removed; rely on AutoGluon's built-in calibration
    
    print("hp: ", hp)
    print("fit: ", fit)
    print("balance_fighters: ", balance_fighters)
    print("calibrate: ", calibrate)

    # --- Export Test Predictions CSV ---
    print("\n=== Exporting Test Predictions CSV ===")
    
    try:
        # Get test predictions (handle sample_weight if needed)
        X_test_clean = X_test.drop(columns=['sample_weight'], errors='ignore')
        test_probs = predictor.predict_proba(X_test_clean, as_pandas=False)[:, 1]
        
        # AutoGluon calibration (if enabled) is already applied inside predictor
        
        # Create predictions dataframe by mapping test indices back to original data
        test_predictions_df = df.loc[X_test.index, ['fighter1_name', 'fighter2_name', 'event_date']].copy()
        test_predictions_df['y_pred_proba'] = test_probs
        test_predictions_df['y_true'] = y_test.values
        
        # Save to CSV in the model directory
        predictions_csv_path = os.path.join(model_dir, 'test_predictions.csv')
        test_predictions_df.to_csv(predictions_csv_path, index=False)
        
        print(f"Test predictions exported to: {predictions_csv_path}")
        print(f"Exported {len(test_predictions_df)} test fight predictions")
        print(f"Columns: {list(test_predictions_df.columns)}")
        print(f"Sample rows:")
        print(test_predictions_df.head())
        
    except Exception as e:
        print(f"Error exporting test predictions CSV: {e}")
        import traceback
        traceback.print_exc()

    # --- Save Evaluation Results ---
    evals_path = os.path.join(model_dir, 'evals.txt')
    
    print(f"\nSaving evaluation results to {evals_path}...")
    
    # Use results returned from train_model
    train_scores = results['train_scores']
    test_scores = results['test_scores']
    val_scores = results.get('val_scores')
    importance = results['importance']
    model_weights_info = results['model_weights'] 
    
    # Reconstruct model weights string
    model_weights_str = ""
    if model_weights_info:
        try:
            model_weights_lines = [f"{model}: {weight:.3f}" for model, weight in model_weights_info.items()]
            model_weights_str = "\n".join(model_weights_lines)
        except Exception as e:
            model_weights_str = f"Error processing weights: {e}"
            print(f"Warning: Error processing model weights - {e}")
    else:
        model_weights_str = "Weights not available or not applicable for the best model."

    with open(evals_path, 'w') as f:
        f.write("Model Performance:\n")
        f.write(f"Training accuracy: {train_scores['accuracy']:.4f}\n")
        f.write(f"Training log loss: {train_scores['log_loss']:.4f}\n")
        f.write(f"Test accuracy: {test_scores['accuracy']:.4f}\n")
        f.write(f"Test log loss: {test_scores['log_loss']:.4f}\n")
        if val_scores is not None:
            f.write(f"Val accuracy: {val_scores['accuracy']:.4f}\n")
            f.write(f"Val log loss: {val_scores['log_loss']:.4f}\n")
        
        f.write(f"\nBest Model: {predictor.model_best}\n")
        if model_weights_str and "Error" not in model_weights_str and "not available" not in model_weights_str:
             f.write("Model weights:\n")
             f.write(model_weights_str + "\n")
        elif model_weights_str:
             f.write(model_weights_str + "\n") # Write the warning/error if present

        if importance is not None:
            f.write("\nTop Most Important Features:\n")
            # Write top 250 features one by one
            for i, (feature, importance_value) in enumerate(importance.head(250).iterrows(), 1):
                 f.write(f"{i}. {feature}: {importance_value['importance']:.4f}\n")
        else:
            f.write("\nFeature importance was not calculated.\n")

        f.write(f"\nHyperparameters: {hp}\n") # Write hp value
        f.write(f"Fit Strategy: {fit}\n") # Write fit value
        f.write(f"Balance Fighters: {balance_fighters}\n") # Write balance_fighters value
        f.write(f"Use Bag Holdout: {use_bag_holdout}\n") # Write use_bag_holdout value
        f.write(f"Num Stack Levels: {num_stack_levels}\n") # Write num_stack_levels value
        f.write(f"Num Bag Folds: {n_splits}\n") # Write n_splits value
        f.write(f"Shuffle: {shuffle}\n") # Write shuffle value
        f.write(f"Use Recency Weights: {use_recency_weights}\n") # Write use_recency_weights value
        f.write(f"Decay Rate: {decay_rate}\n") # Write decay_rate value
        f.write(f"N Splits: {n_splits}\n") # Write n_splits value
        f.write(f"Train Size: {train_size}\n") # Write train_size value
        f.write(f"Val Size: {val_size}\n") # Write val_size value
        f.write(f"Test Size: {test_size}\n") # Write test_size value
        f.write(f"Start Date: {start_date}\n") # Write start_date value
        f.write(f"Calibrate: {calibrate}\n") # Write calibrate value
        f.write(f"Normalize: {normalize}\n") # Write normalize value
        
    print("Evaluation results saved.")

    utils = ModelUtils(
        model_path=model_dir,
        training_data_path=training_data_path,
        scaler_path=scaler_path,
        feats=feats
    )
    
    print("\n=== Model Performance Analysis ===")
    utils.save_model_stats()
    utils.plot_calibration_curve()
    return predictor

if __name__ == "__main__":
    main()
