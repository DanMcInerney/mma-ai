import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import pandas as pd
from sqlalchemy import text
from autogluon.tabular import TabularPredictor
from typing import List
from sklearn.metrics import mean_squared_error # Changed from log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from libs.feature_store.features import FEATS_MAD, LAYERED_TEST_FEATS, TEST_FEATS, FEATS_MAD2
from libs.paths import data_file, models_dir
import joblib
from datetime import datetime
#from libs.modeling.model_utils import ModelUtils # Keep commented out for now

def split_data(X, y, train_size):
    """Performs chronological split with specified train/test split.""" # Updated docstring slightly
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

def load_training_data(df, feats: List[str], target_col: str): # Added target_col parameter
    """Load training data from DataFrame.""" # Updated docstring
    #df = df.sort_values(by=['event_date', 'fight_id']).reset_index(drop=True) # Already sorted in main
    
    # Select features
    if not feats:
        # Default feature selection (example, might need adjustment for odds prediction)
        keep = ['opp_dec_avg', 'dec_adjperf_dec_avg', 'vigless_ip_opening_odds'] # Added opening odds
        remove = ['age', 'reach', 'ape', 'days_since_last_fight', 'ufcage']#, '_att', 'body', 'strikes', 'slope', 'rd1']
        also_keep = ['age_diff', 'age_ratio_diff',
                     'ufcage_diff',
                    'reach_diff', 'reach_ratio_diff',
                    'days_since_last_fight_diff']
        feature_cols = [col for col in df.columns if any(k in col for k in keep)]
        feature_cols_filtered = [col for col in feature_cols if not any(k in col for k in remove)]
        feature_cols_filtered += [col for col in df.columns if any(k in col for k in also_keep)]
        # Ensure target and explicit features are not accidentally included twice or excluded
        feature_cols_filtered = [f for f in feature_cols_filtered if f != target_col]
        if 'vigless_ip_opening_odds' not in feature_cols_filtered:
             feature_cols_filtered.append('vigless_ip_opening_odds')

    else:
        # Ensure target is not in explicit feature list
        feature_cols_filtered = [f for f in feats if f != target_col]
        # Ensure opening odds is included if feats are provided explicitly
        if 'vigless_ip_opening_odds' not in feature_cols_filtered:
            feature_cols_filtered.append('vigless_ip_opening_odds')

    # Separate features and target
    X = df[feature_cols_filtered]
    y = df[target_col] # Use the specified target column

    # Drop rows where target is missing (NaN dropping happens later in filter_fights for all columns)
    # mask = y.notna()
    # X = X[mask]
    # y = y[mask]

    print(f"Loaded {len(X)} rows with {len(feature_cols_filtered)} features for target '{target_col}'")

    return X, y

def filter_fights(df, threshold, date='2014-01-01', include_split_dec=True, target_col='vigless_ip_closing_odds'): # Added target_col
    """
    Filter fights based on:
      - Both fighters must have had at least `threshold` previous fights
      - Removing unwanted fight methods (optional)
      - Fights from `date` onward
      - Removing rows with NaN values in features or target
      
    Parameters:
      df : DataFrame
          Must contain: 'fight_id', 'event_date', 'fighter1_id', 'fighter2_id',
          'method', and the `target_col`.
      threshold : int
          Minimum number of previous fights required for each fighter.
      date : str
          Start date for filtering fights (YYYY-MM-DD format).
      include_split_dec : bool
          Whether to include fights decided by split decision.
      target_col : str
            The name of the target column to check for NaNs.

    Returns:
      Filtered DataFrame.
    """
    print("\n=== Filtering Fights ===")
    orig_len_start = len(df)
    
    # Ensure required columns exist
    required_cols = ['fight_id', 'event_date', 'fighter1_id', 'fighter2_id', 'method', target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for filtering: {missing_cols}")

    # --- Step 1. Compute overall fight counts for each fighter ---
    df_f1 = df[['fight_id', 'event_date', 'fighter1_id']].rename(columns={'fighter1_id': 'fighter_id'})
    df_f2 = df[['fight_id', 'event_date', 'fighter2_id']].rename(columns={'fighter2_id': 'fighter_id'})
    df_long = pd.concat([df_f1, df_f2], ignore_index=True)
    
    df_long['event_date'] = pd.to_datetime(df_long['event_date'])
    df_long = df_long.sort_values(['fighter_id', 'event_date', 'fight_id'])
    
    df_long['fight_num'] = df_long.groupby('fighter_id').cumcount() + 1
    
    # --- Step 2. Merge fight numbers back onto the original DataFrame ---
    df_long_f1 = df_long.rename(columns={'fighter_id': 'fighter1_id', 'fight_num': 'fighter1_fight_num'})
    df = pd.merge(df, 
                  df_long_f1[['fight_id', 'fighter1_id', 'fighter1_fight_num']],
                  on=['fight_id', 'fighter1_id'],
                  how='left')
    
    df_long_f2 = df_long.rename(columns={'fighter_id': 'fighter2_id', 'fight_num': 'fighter2_fight_num'})
    df = pd.merge(df, 
                  df_long_f2[['fight_id', 'fighter2_id', 'fighter2_fight_num']],
                  on=['fight_id', 'fighter2_id'],
                  how='left')

    # Drop rows where fight_num couldn't be calculated (shouldn't happen with clean mapping tables)
    df.dropna(subset=['fighter1_fight_num', 'fighter2_fight_num'], inplace=True)
    df['fighter1_fight_num'] = df['fighter1_fight_num'].astype(int)
    df['fighter2_fight_num'] = df['fighter2_fight_num'].astype(int)

    # --- Step 3. Filter out fights where either fighter has insufficient experience ---
    before_exp = len(df)
    df = df[(df['fighter1_fight_num'] > threshold) & (df['fighter2_fight_num'] > threshold)].copy() # Use .copy()
    after_exp = len(df)
    print(f"Filtered out {before_exp - after_exp} rows due to insufficient previous fights (need current fight number > {threshold})")
    
    # --- Step 4. Remove fights with unwanted methods ---
    # original_len = len(df)
    # if include_split_dec:
    #     unwanted_methods = ['dq', 'other', 'overturned']
    # else:
    #     unwanted_methods = ['dq', 'other', 'decision - split', 'decision - majority', 'overturned']

    # Use the string accessor (.str.lower()) to lowercase each value
    # Ensure 'method' column exists and handle potential NaNs before lowercasing
    # if 'method' in df.columns:
    #     df['method_lower'] = df['method'].astype(str).str.lower() # Convert to string first
    #     method_mask = ~df['method_lower'].str.contains('|'.join(unwanted_methods), na=False)
    #     df = df[method_mask].copy() # Use .copy()
    #     print(f"Removed {original_len - len(df)} rows with unwanted methods: {unwanted_methods}")
    #     df.drop(columns=['method_lower'], inplace=True)
    # else:
    #     print("Warning: 'method' column not found, skipping method filtering.")


    # --- Step 5. Filter by event date ---
    original_len = len(df)
    df['event_date'] = pd.to_datetime(df['event_date'])
    df = df[df['event_date'] >= pd.Timestamp(date)].copy() # Use .copy()
    print(f"Filtered out {original_len - len(df)} rows before {date}")

    # --- Step 6. Filter NaN values (including target column) ---
    original_len = len(df)
    # Check NaNs in all columns used for modeling (features + target)
    # We will determine features later, so for now, just check target and key ids
    check_nan_cols = ['event_date', 'fight_id', 'fighter1_id', 'fighter2_id', target_col]
    df = df.dropna(subset=[col for col in check_nan_cols if col in df.columns])
    # Later, NaNs in feature columns will be handled by load_training_data or AutoGluon
    print(f"Filtered out {original_len - len(df)} rows with NaN values in key columns or target '{target_col}'")

    # Reset the index
    df.sort_values(by=['event_date', 'fight_id'], inplace=True)
    df = df.reset_index(drop=True)
    
    print(f"Total rows filtered: {orig_len_start - len(df)}")
    print(f"Final number of rows after all filtering: {len(df)}")
    
    return df

def apply_zscore_normalization(X_train, X_test, path):
    """Apply Z-score normalization to features only."""
    # Remove date columns before scaling
    exclude_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name'] # Add any other non-numeric identifiers
    features_to_scale = [col for col in X_train.columns if col not in exclude_cols and X_train[col].dtype in ['int64', 'float64']]


    # Fit scaler on training data
    scaler = StandardScaler()
    # Filter out non-numeric columns before fitting
    X_train_numeric = X_train[features_to_scale]
    scaler.fit(X_train_numeric)
    joblib.dump(scaler, path)

    
    # Transform the data
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[features_to_scale] = scaler.transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    return X_train_scaled, X_test_scaled

def apply_robust_normalization(X_train, X_test, path):
    """Apply Robust Scaling to features (i.e. use median and IQR)."""
    # Exclude any date or ID columns from scaling
    exclude_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name'] # Add any other non-numeric identifiers
    features_to_scale = [col for col in X_train.columns if col not in exclude_cols and X_train[col].dtype in ['int64', 'float64']]

    scaler = RobustScaler()
     # Filter out non-numeric columns before fitting
    X_train_numeric = X_train[features_to_scale]
    scaler.fit(X_train_numeric)
    # Save the scaler to disk if needed
    joblib.dump(scaler, path)

    # Transform both train and test sets
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[features_to_scale] = scaler.transform(X_train[features_to_scale])
    X_test_scaled[features_to_scale] = scaler.transform(X_test[features_to_scale])

    return X_train_scaled, X_test_scaled
    

def train_model(X_train, y_train, X_test, y_test, predictor, fit='medium_quality', hp='default', limit=1200, prune=False, train_size=0.9):
    """Train regression model.""" # Updated docstring
    print("Training model...")
    
    # Combine features and target for AutoGluon fit/evaluate
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)


    included_model_types=['CAT', 'XGB', "XT", "GBM", 'FASTAI', 'NN_TORCH', 'RF']
    excluded_model_types=['KNN', 'TABPFNMIX']

    # Feature pruning settings (optional)
    # feature_prune_kwargs = { ... } # Keep commented unless needed

    # Fit the model based on the chosen strategy
    # Note: Time-ordered splits for regression validation require careful setup.
    # AutoGluon's default random split might be sufficient unless strict chronology is vital for validation.
    # Using tuning_data implies a holdout set taken from the end of the training set if shuffle=False.
    
    fit_kwargs = {
        'time_limit': limit,
        'hyperparameters': hp,
        'included_model_types': included_model_types,
        'excluded_model_types': excluded_model_types,
        # 'refit_full': True # Often good for final model
    }

    if fit == 'time_ordered':
        # Split training data further into train/validation for time-ordered validation
        train_split_idx = int(len(train_data) * train_size)
        train_final = train_data.iloc[:train_split_idx]
        val_data = train_data.iloc[train_split_idx:]
        
        print(f"\nTime-ordered split for validation:")
        print(f"Training data: {len(train_final)} fights")
        print(f"Validation data: {len(val_data)} fights")
        
        predictor.fit(
            train_data=train_final,
            tuning_data=val_data,
             ag_args_fit={'shuffle_data': False}, # Keep data order
            **fit_kwargs
        )

    elif fit == 'time_ordered_custom':
         # Split training data further into train/validation for time-ordered validation
        train_split_idx = int(len(train_data) * train_size)
        train_final = train_data.iloc[:train_split_idx]
        val_data = train_data.iloc[train_split_idx:]
        
        print(f"\nTime-ordered split for validation:")
        print(f"Training data: {len(train_final)} fights")
        print(f"Validation data: {len(val_data)} fights")
        
        predictor.fit(
            train_data=train_final,
            tuning_data=val_data,
            # Add custom HP or bagging strategy here if needed
            num_bag_folds=10,
            num_bag_sets=2,
            use_bag_holdout=True, # Requires careful consideration with time series
            ag_args_fit={'shuffle_data': False}, # Keep data order
            **fit_kwargs
        )
        
    elif fit == 'best_quality':
        print("Fitting best_quality model...")
        predictor.fit(
            train_data=train_data,
            presets='best_quality',
            refit_full=True,
            # fit_strategy='parallel', # May use too much memory
            dynamic_stacking=False,
            num_stack_levels=0,
            num_bag_folds=8,
            num_bag_sets=2,
            **fit_kwargs
        )
    
    elif fit == 'experimental':
        predictor.fit(
            train_data=train_data,
            presets=fit, # Or specify custom HPs
             # dynamic_stacking=True, 
            # num_stack_levels=1, # sometimes 1 works
            # num_bag_folds=5,
            # num_bag_sets=1,
            # fit_strategy='parallel',
            # hyperparameters=hp,
            # refit_full=False,
            **fit_kwargs
        )
    
    elif fit == "medium_quality":
        predictor.fit(
            train_data=train_data,
            presets='medium_quality',
            refit_full=True,
            # fit_strategy='parallel',
            **fit_kwargs
        )
    else:
         predictor.fit(
            train_data=train_data, # Use standard random split validation
            **fit_kwargs
         )

    
    # Evaluate model
    print("\n--- Model Performance ---")
    # Ensure evaluation uses the correct data (train_data includes target)
    train_scores = predictor.evaluate(train_data) 
    test_scores = predictor.evaluate(test_data)

    # Print relevant regression metrics
    print(f"Training RMSE: {train_scores['root_mean_squared_error']:.4f}")
    # print(f"Training MAE: {train_scores['mean_absolute_error']:.4f}") # Example other metric
    # print(f"Training R2: {train_scores['r2']:.4f}") # Example other metric
    print(f"Test RMSE: {test_scores['root_mean_squared_error']:.4f}")
    # print(f"Test MAE: {test_scores['mean_absolute_error']:.4f}") # Example other metric
    # print(f"Test R2: {test_scores['r2']:.4f}") # Example other metric

    print(f"\nBest Model: {predictor.model_best}")

    # Print model weights if it's an ensemble
    try:
        best_model_info = predictor.info()['model_info'][predictor.model_best]
        if 'children_info' in best_model_info: # Check if it's a weighted ensemble
             # Path to weights might vary depending on stacking levels, adjust as needed
             # Common path for simple ensemble:
            if 'S1F1' in best_model_info['children_info']:
                 weighted_ensemble_info = best_model_info['children_info']['S1F1']['model_weights']
                 print("\nModel weights:")
                 for model, weight in weighted_ensemble_info.items():
                     print(f"{model}: {weight:.3f}")
            else:
                 print("\nEnsemble structure found, but weights path 'S1F1' not present.")
        else:
             print(f"\nBest model '{predictor.model_best}' is not a weighted ensemble.")
    except KeyError as e:
        print(f"\nCould not retrieve model weights, info structure might have changed: {e}")
        # print(predictor.info()) # Uncomment to debug info structure


    # Feature importance (using test data for evaluation context)
    print("\n--- Top Most Important Features ---")   
    try:
        # Provide data with features AND the label column for importance calculation
        importance = predictor.feature_importance(test_data) 
        # Print top features
        for i, (feature, importance_value) in enumerate(importance.head(50).iterrows(), 1): # Show top 50
            print(f"{i}. {feature}: {importance_value['importance']:.4f}")
    except Exception as e:
        print(f"Could not calculate feature importance: {e}")
        print("Ensure the evaluation data is correctly formatted.")

    # Display detailed evaluation metrics for all models
    print("\n--- Detailed Model Evaluations ---")
    try:
        # Get more detailed metrics for the best model
        print(f"\nDetailed Metrics for Best Model ({predictor.model_best}):")
        all_metrics = predictor.evaluate(test_data, detailed=True)
        for metric_name, value in all_metrics.items():
            print(f"{metric_name}: {value:.6f}")
    except Exception as e:
        print(f"Error getting detailed evaluations: {e}")
    
    return predictor


def main():
    # --- Configuration ---
    TARGET_COLUMN = 'vigless_ip_closing_odds'
    TRAINING_DATA_CSV = str(data_file('training_data.csv'))
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir_base = str(models_dir())
    model_run_name = f'ag_odds_{timestamp}' # Added '_odds' prefix
    model_dir = os.path.join(model_dir_base, model_run_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Define paths for artifacts within the model directory
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    # feats_path = os.path.join(model_dir, 'feats.pkl') # Not saving feats list directly anymore
    training_data_path_saved = os.path.join(model_dir, 'training_data_filtered.csv') # Save the filtered data used
    feats_txt_path = os.path.join(model_dir, 'features_used.txt') # Save list of features used

    # Feature Selection (Choose one method)
    # feats = None # Use default logic in load_training_data
    feats = FEATS_MAD # Use predefined list (ensure target and opening odds handled)
    feats = feats + ['vigless_ip_opening_odds', TARGET_COLUMN]
    
    # Preprocessing
    normalize = True # Use RobustScaler
    filter_min_prev_fights = 2 # Min number of fights EACH fighter must have HAD before this one
    filter_start_date = '2014-01-01'
    filter_include_split_dec = False # Filter out split/majority decisions

    # Training Settings
    limit = 600 # Time limit in seconds
    train_size = 0.9 # Proportion for chronological train/test split
    # fit = 'best_quality'
    fit = 'experimental'
    #fit = 'medium_quality' # Good balance
    # fit = 'experimental'
    # fit = 'time_ordered'
    #hp = 'default' # Hyperparameter tuning strategy ('default', 'light', 'very_light', 'random', 'grid')
    hp = 'zeroshot' # Experimental, may not work well

    # --- Data Loading and Preparation ---
    print(f"Loading data from: {TRAINING_DATA_CSV}")
    try:
        orig_df = pd.read_csv(TRAINING_DATA_CSV)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {TRAINING_DATA_CSV}")
        sys.exit(1)
    
    print(f"Original data loaded: {len(orig_df)} rows")
    orig_df.sort_values(by=['event_date', 'fight_id'], inplace=True)
    
    # Filter fights
    df_filtered = filter_fights(
        orig_df, 
        threshold=filter_min_prev_fights, 
        date=filter_start_date, 
        include_split_dec=filter_include_split_dec,
        target_col=TARGET_COLUMN
    )
    
    # Save the filtered data that will be used for training/testing
    df_filtered.to_csv(training_data_path_saved, index=False)
    print(f"Filtered data saved to: {training_data_path_saved}")


    # Prepare features (X) and target (y)
    # Make a copy of feats list if using a predefined one to avoid modifying the original
    current_feats = list(feats) if feats else None 

    X, y = load_training_data(df_filtered, current_feats, target_col=TARGET_COLUMN)

    # Save the final list of features used
    final_feature_list = list(X.columns)
    with open(feats_txt_path, 'w') as f:
        for feat in final_feature_list:
            f.write(f"{feat}\n")
    print(f"Final features used saved to: {feats_txt_path}")

    # Split data chronologically
    (X_train, y_train), (X_test, y_test) = split_data(X, y, train_size)

    # Print start date of test data
    if not X_test.empty:
        test_start_date = df_filtered.loc[X_test.index, 'event_date'].min()
        print(f"Test data start date: {test_start_date}")
    else:
        print("Warning: Test set is empty after splitting.")


    # Apply normalization (scaling)
    if normalize:
        print("Applying Robust Normalization...")
        try:
             X_train, X_test = apply_robust_normalization(X_train, X_test, scaler_path)
             print(f"Scaler saved to: {scaler_path}")
        except Exception as e:
             print(f"Error during normalization: {e}. Proceeding without normalization.")
             normalize = False # Turn off normalization if it failed
        # X_train, X_test = apply_zscore_normalization(X_train, X_test, scaler_path)

    # --- Model Training ---
    print("\n--- Initializing AutoGluon Predictor ---")
    predictor = TabularPredictor(
        label=TARGET_COLUMN,
        eval_metric='root_mean_squared_error', # Regression metric
        problem_type='regression',            # Regression task
        path=model_dir,                       # Save models here
        verbosity=2                           # Log level (0=silent, 1=minimal, 2=detail, 3=debug)
    )
    
    print(f"--- Starting Model Training (fit='{fit}', hp='{hp}', time_limit={limit}s) ---")
    predictor = train_model(X_train, y_train, X_test, y_test, predictor,
                          fit=fit, hp=hp, limit=limit, prune=False, train_size=train_size) # Pruning TBD
    
    print(f"\n--- Training Summary ---")
    print(f"Target Variable: {TARGET_COLUMN}")
    print(f"Problem Type: Regression")
    print(f"Evaluation Metric: RMSE")
    print(f"Normalization Applied: {normalize}")
    print(f"Features Used: {len(final_feature_list)} (see {feats_txt_path})")
    print(f"Fit Strategy: {fit}")
    print(f"Hyperparameters: {hp}")
    print(f"Model saved in: {model_dir}")
    
    # Run predictions on test data and display results
    print("\n--- Running Predictions on Test Data ---")
    test_pred_df = run_predictions_on_test_data(predictor, X_test, df_filtered.iloc[X_test.index])
    
    # Save predictions to CSV
    predictions_path = os.path.join(model_dir, 'test_predictions.csv')
    test_pred_df.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")

    # Model Utils section removed - can be added back if adapted for regression
    # utils = ModelUtils(...) 
    # print("\n=== Model Performance Analysis ===")
    # utils.save_model_stats()
    # utils.plot_calibration_curve() # Calibration plots differ for regression

    return predictor # Return predictor if running interactively

def run_predictions_on_test_data(predictor, X_test, test_data_with_meta):
    """
    Run predictions on test data and return a DataFrame with predictions and metadata.
    
    Parameters:
        predictor: Trained AutoGluon predictor
        X_test: Feature DataFrame for test data
        test_data_with_meta: Original test data with metadata columns
        
    Returns:
        DataFrame with predictions and relevant metadata
    """
    # Get predictions
    predictions = predictor.predict(X_test)
    
    # Create a DataFrame with predictions
    pred_df = pd.DataFrame({
        'predicted_closing_odds': predictions
    })
    
    # Extract relevant metadata columns 
    meta_columns = ['event_date', 'fight_id', 'fighter1_id', 'fighter2_id', 
                  'fighter1_name', 'fighter2_name', 'vigless_ip_opening_odds', 
                  'vigless_ip_closing_odds']
    
    # Only include columns that exist in the data
    available_meta_columns = [col for col in meta_columns if col in test_data_with_meta.columns]
    
    # Combine predictions with metadata
    result_df = pd.concat([test_data_with_meta[available_meta_columns].reset_index(drop=True), 
                          pred_df.reset_index(drop=True)], axis=1)
    
    # Calculate prediction error
    if 'vigless_ip_closing_odds' in result_df.columns:
        result_df['prediction_error'] = result_df['predicted_closing_odds'] - result_df['vigless_ip_closing_odds']
        result_df['abs_error'] = result_df['prediction_error'].abs()
    
    # Calculate direction prediction (higher/lower than opening odds)
    if all(col in result_df.columns for col in ['vigless_ip_opening_odds', 'vigless_ip_closing_odds', 'predicted_closing_odds']):
        # Actual direction: Did closing odds increase or decrease compared to opening?
        result_df['actual_direction'] = (result_df['vigless_ip_closing_odds'] > result_df['vigless_ip_opening_odds']).astype(int)
        
        # Predicted direction: Did model predict closing odds would increase or decrease?
        result_df['predicted_direction'] = (result_df['predicted_closing_odds'] > result_df['vigless_ip_opening_odds']).astype(int)
        
        # Direction accuracy: Did the model correctly predict the direction?
        result_df['direction_correct'] = (result_df['actual_direction'] == result_df['predicted_direction']).astype(int)
        
        # Calculate direction accuracy as percentage
        direction_accuracy = result_df['direction_correct'].mean() * 100
        
        # Add a column for the magnitude of direction change
        result_df['actual_direction_change'] = result_df['vigless_ip_closing_odds'] - result_df['vigless_ip_opening_odds']
        result_df['predicted_direction_change'] = result_df['predicted_closing_odds'] - result_df['vigless_ip_opening_odds']
    
    # Display sample of predictions
    print("\nSample predictions (first 10 rows):")
    display_cols = ['event_date', 'fighter1_name', 'fighter2_name', 
                    'vigless_ip_opening_odds', 'vigless_ip_closing_odds', 
                    'predicted_closing_odds', 'prediction_error']
    
    # Only include columns that exist
    display_cols = [col for col in display_cols if col in result_df.columns]
    
    # Print sample and summary statistics
    print(result_df[display_cols].head(10))
    
    if 'prediction_error' in result_df.columns:
        print("\nPrediction Error Statistics:")
        print(f"Mean Absolute Error: {result_df['abs_error'].mean():.4f}")
        print(f"Min Error: {result_df['prediction_error'].min():.4f}")
        print(f"Max Error: {result_df['prediction_error'].max():.4f}")
    
    # Print direction prediction accuracy
    if 'direction_correct' in result_df.columns:
        print("\nDirection Prediction Statistics:")
        print(f"Direction Prediction Accuracy: {direction_accuracy:.2f}%")
        print(f"Correct Direction Predictions: {result_df['direction_correct'].sum()} / {len(result_df)}")
        
        # Break down by prediction type
        increase_pred_count = result_df['predicted_direction'].sum()
        decrease_pred_count = len(result_df) - increase_pred_count
        print(f"Predicted Increases: {increase_pred_count} ({increase_pred_count/len(result_df)*100:.1f}%)")
        print(f"Predicted Decreases: {decrease_pred_count} ({decrease_pred_count/len(result_df)*100:.1f}%)")
        
        # Actual increases/decreases
        actual_increase_count = result_df['actual_direction'].sum()
        actual_decrease_count = len(result_df) - actual_increase_count
        print(f"Actual Increases: {actual_increase_count} ({actual_increase_count/len(result_df)*100:.1f}%)")
        print(f"Actual Decreases: {actual_decrease_count} ({actual_decrease_count/len(result_df)*100:.1f}%)")
        
        # Accuracy by direction type
        if increase_pred_count > 0:
            increase_correct = result_df[(result_df['predicted_direction']==1) & (result_df['direction_correct']==1)].shape[0]
            print(f"Accuracy when predicting increase: {increase_correct/increase_pred_count*100:.1f}%")
        
        if decrease_pred_count > 0:
            decrease_correct = result_df[(result_df['predicted_direction']==0) & (result_df['direction_correct']==1)].shape[0]
            print(f"Accuracy when predicting decrease: {decrease_correct/decrease_pred_count*100:.1f}%")
    
    return result_df

if __name__ == "__main__":
    main() 
