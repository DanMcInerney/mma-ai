import sys
import os

# --- Dynamically add project root to sys.path ---
# This allows running the script directly (e.g., via debugger) 
# without needing python -m
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of dynamic path addition ---

import pandas as pd
import joblib
from autogluon.tabular import TabularPredictor
from libs.modeling.split_date_utils import read_test_start_date
from libs.paths import data_file, models_dir
import numpy as np
from decimal import Decimal, ROUND_HALF_UP # For precise financial calculations
from itertools import combinations
from collections import defaultdict
import logging
import random # <-- Add this import
import matplotlib.pyplot as plt # <-- Add this import
import seaborn as sns # <-- Add this import for potentially nicer plots

log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profit_analysis.log')
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def configure_profit_logging(level=logging.INFO):
    """Configure profit-analysis logging only when running that workflow."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(),
        ],
        force=True,
    )

# Import AutoGluonWrapper for calibrator loading
try:
    from libs.modeling.autogluon_wrapper import AutoGluonWrapper
    logger.debug("Successfully imported AutoGluonWrapper from autogluon_wrapper.py")
except ImportError as e:
    logger.warning("Could not import AutoGluonWrapper from autogluon_wrapper.py: %s", e)
    AutoGluonWrapper = None

# Suppress matplotlib debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

# Risk-free annual rate used for Sharpe calculations (e.g., 0.02 = 2%)
RISK_FREE_ANNUAL_RATE = 0.0

def load_model(model_name):
    """
    Load an AutoGluon model from the specified path.
    Handles both single TabularPredictor models and walkforward EnsemblePredictor models.
    """
    ensemble_info_path = os.path.join(model_name, 'ensemble_info.txt')
    
    # Check if this is a walkforward ensemble model
    if os.path.exists(ensemble_info_path):
        # Walk-forward ensemble format
        logging.info(f"Detected WALK-FORWARD ENSEMBLE model at {model_name}")
        logging.info("Loading EnsemblePredictor...")
        from libs.modeling.train import EnsemblePredictor
        predictor = EnsemblePredictor.load(model_name)
        logging.info(f"Successfully loaded ensemble with {len(predictor.predictors)} models")
        return predictor
    else:
        # Single model format
        logging.info(f"Detected SINGLE model at {model_name}")
        predictor = TabularPredictor.load(model_name, require_version_match=False)
        return predictor

def load_paths():
    """Define and return necessary file paths."""
    model_name = os.getenv("MMA_AI_MODEL_NAME", "ag-20260101_203728-win-extreme")
    model_path = os.getenv("MMA_AI_MODEL_PATH", str(models_dir() / model_name))
    training_data_csv = str(data_file("training_data.csv"))
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    return model_path, training_data_csv, scaler_path

def load_assets(model_path, scaler_path, use_calibration=False):
    """
    Load the model, scaler, and calibrator (if requested and available).
    Handles both single models and walkforward ensemble models.
    """
    logging.info(f"Loading assets from model path: {model_path}")
    logging.info(f"Files in model directory: {os.listdir(model_path) if os.path.exists(model_path) else 'Directory does not exist'}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    # Load model first to determine if it's an ensemble
    model = load_model(model_path)
    
    # Check if this is an EnsemblePredictor (walkforward model)
    from libs.modeling.train import EnsemblePredictor
    is_ensemble = isinstance(model, EnsemblePredictor)
    
    # Load scaler only for single models (ensembles handle scaling internally)
    scaler = None
    if not is_ensemble:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        logging.info(f"Loaded scaler from {scaler_path}")
    else:
        logging.info("Ensemble model detected - scaling will be handled internally by each window model")
    
    # Try to load calibrator only if use_calibration is True
    calibrator = None
    if use_calibration:
        calibrator_path = os.path.join(model_path, 'calibrator.pkl')
        logging.info(f"Looking for calibrator at: {calibrator_path}")
        logging.info(f"Calibrator file exists: {os.path.exists(calibrator_path)}")
        
        if os.path.exists(calibrator_path):
            try:
                # Check if AutoGluonWrapper is available (needed for calibrator)
                if AutoGluonWrapper is None:
                    logging.error("Calibration requested but AutoGluonWrapper class is not available. Cannot load calibrator.")
                else:
                    logging.info("AutoGluonWrapper is available, attempting to load calibrator...")
                    calibrator = joblib.load(calibrator_path)
                    logging.info(f"Calibration ENABLED - Loaded calibrator from {calibrator_path}")
            except Exception as e:
                logging.warning(f"Calibration requested but failed to load calibrator: {e}")
                logging.info("Falling back to uncalibrated predictions")
        else:
            logging.warning(f"Calibration requested but no calibrator.pkl found at {calibrator_path}")
            logging.info("Falling back to uncalibrated predictions")
    else:
        logging.info("Calibration DISABLED - using uncalibrated predictions")
    
    return model, scaler, calibrator

def load_data(csv_path):
    """Load data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def prepare_features(df, features_list):
    """Prepare features: Filter by fighter experience (>=2 prior fights), drop NaNs in features+odds, return features-only df with original index."""
    logging.info("Preparing features: Applying experience and NaN filters...")
    initial_count = len(df)
    logging.info(f"Starting with {initial_count} fights")
    
    if df.empty:
        logging.warning("Input DataFrame to prepare_features is empty.")
        return pd.DataFrame() # Return empty df matching expected structure ideally

    # Ensure event_date is datetime
    try:
        df['event_date'] = pd.to_datetime(df['event_date'])
    except Exception as e:
        logging.error(f"Error converting 'event_date' to datetime in prepare_features: {e}")
        raise # Re-raise the error as this is critical

    # --- Filter by Fighter Experience (Minimum 2 Prior Fights) ---    
    logging.debug("Filtering fights based on fighter experience (min 2 prior fights)...")
    eligible_indices_experience = set()
    fighter_counts_cache = {}

    # Sort by date and fight_id to process chronologically and ensure deterministic ordering
    # This ensures consistent filtering when multiple fights occur on the same date
    df_sorted = df.sort_values(by=['event_date', 'fight_id'])

    for index, row in df_sorted.iterrows():
        current_fight_date = row['event_date']
        f1_name = row['fighter1_name']
        f2_name = row['fighter2_name']

        # Count for Fighter 1
        f1_key = (f1_name, current_fight_date)
        if f1_key not in fighter_counts_cache:
            fighter_counts_cache[f1_key] = len(df[
                (df['event_date'] < current_fight_date) & 
                ((df['fighter1_name'] == f1_name) | (df['fighter2_name'] == f1_name))
            ])
        f1_past_fights = fighter_counts_cache[f1_key]
            
        # Count for Fighter 2
        f2_key = (f2_name, current_fight_date)
        if f2_key not in fighter_counts_cache:
             fighter_counts_cache[f2_key] = len(df[
                (df['event_date'] < current_fight_date) & 
                ((df['fighter1_name'] == f2_name) | (df['fighter2_name'] == f2_name))
            ])
        f2_past_fights = fighter_counts_cache[f2_key]

        # Check Condition
        if f1_past_fights >= 2 and f2_past_fights >= 2:
            eligible_indices_experience.add(index)
        else:
            logging.debug(f"Excluding fight {index} ({f1_name} vs {f2_name}) due to experience. Counts: F1={f1_past_fights}, F2={f2_past_fights}")

    fights_filtered_by_experience = initial_count - len(eligible_indices_experience)
    logging.info(f"Experience filter complete. {len(eligible_indices_experience)} fights meet minimum experience criteria.")
    logging.info(f"Filtered out {fights_filtered_by_experience} fights due to insufficient experience (< 2 prior fights)")

    if not eligible_indices_experience:
        logging.warning("No fights meet the minimum experience criteria.")
        return pd.DataFrame() 

    # Get the subset of the DataFrame that passed the experience filter
    df_exp_filtered = df.loc[list(eligible_indices_experience)].copy()

    # --- Filter by NaNs in Features + Odds (on the experience-filtered subset) ---
    logging.debug("Filtering experienced fighters' fights based on NaNs in features + odds...")
    # Include *both* closing and seven-day odds in the NaN check
    odds_cols = ['f1_ip_closing_odds', 'f2_ip_closing_odds', 
                 'f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds'] 
                 # Add vigless if used: 'f1_sevenday_vigless_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds']
                 
    features_to_use = [feat for feat in features_list if feat in df_exp_filtered.columns]
    if not features_to_use:
        # This shouldn't happen if input df was valid, but check anyway
        raise ValueError("No features from features_list found in the experience-filtered DataFrame.")
        
    cols_to_check_nan = features_to_use + odds_cols

    missing_cols_nan = [col for col in cols_to_check_nan if col not in df_exp_filtered.columns]
    if missing_cols_nan:
        # Should have been caught earlier if missing from original df, but check subset
        # Be more informative about missing columns
        logging.warning(f"Columns required for NaN check are missing from experience-filtered DataFrame: {missing_cols_nan}. Proceeding without them for NaN check.")
        # Filter cols_to_check_nan to only include existing columns
        cols_to_check_nan = [col for col in cols_to_check_nan if col in df_exp_filtered.columns]
        if not cols_to_check_nan:
             logging.error("No columns left to check for NaNs after removing missing ones.")
             return pd.DataFrame() # Cannot proceed

    # Identify the index of rows *within the subset* that don't have NaNs in the available columns
    final_indices_to_keep = df_exp_filtered.dropna(subset=cols_to_check_nan).index
    
    rows_dropped_nan = len(df_exp_filtered) - len(final_indices_to_keep)
    logging.info(f"NaN filter complete. Dropped {rows_dropped_nan} additional rows due to NaNs. {len(final_indices_to_keep)} fights remain.")
    
    # Debug: Show examples of fights dropped due to NaN
    if rows_dropped_nan > 0:
        nan_filtered_indices = set(df_exp_filtered.index) - set(final_indices_to_keep)
        logging.debug(f"Examples of fights dropped due to NaN in features/odds:")
        for idx in list(nan_filtered_indices)[:3]:  # Show first 3
            fight = df_exp_filtered.loc[idx]
            # Check which columns have NaN
            nan_cols = []
            for col in cols_to_check_nan:
                if col in fight.index and pd.isna(fight[col]):
                    nan_cols.append(col)
            logging.debug(f"  {fight['fighter1_name']} vs {fight['fighter2_name']} - NaN in: {nan_cols[:5]}")  # Show first 5 NaN columns

    if not final_indices_to_keep.any(): # Check if index is empty
        logging.warning("No fights remain after applying NaN filter.")
        return pd.DataFrame() 

    # --- Create Final Features DataFrame --- 
    # Select only the *base feature columns* using the *final indices* from the *original df*
    df_clean_features = df.loc[final_indices_to_keep, features_to_use].copy()

    if df_clean_features.empty:
        # This check is likely redundant given the check on final_indices_to_keep, but safe
        raise ValueError("Final features DataFrame is empty after all filtering.")

    logging.info(f"prepare_features finished. Returning DataFrame with {len(df_clean_features)} rows and columns: {df_clean_features.columns.tolist()}")
    return df_clean_features

def scale_features(df, scaler):
    """
    Scale the specified features using the provided scaler, excluding categorical features.
    
    This matches the logic used during training: excludes date columns, categorical static features
    (like weightclass_encoded and odds), sample_weight, and y_true from scaling.
    """
    # Identify columns to exclude from scaling (matching train.py logic)
    date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
    categorical_static_feats = ['weightclass_encoded', 'odds']
    
    def should_exclude_col(col_name):
        if col_name in date_cols:
            return True
        for cat_feat in categorical_static_feats:
            if cat_feat in col_name:
                return True
        return False
    
    # Get features to scale (exclude categorical features, date columns, sample_weight, y_true)
    features_to_scale = [col for col in df.columns 
                         if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
    
    # Create a copy of the DataFrame
    scaled_df = df.copy()
    
    # Scale only the features that should be scaled
    if len(features_to_scale) > 0:
        scaled_df[features_to_scale] = scaler.transform(df[features_to_scale])
    
    return scaled_df

def make_predictions(model, predict_df, calibrator=None):
    """Make predictions using the model, handling sample_weight if needed, with optional calibration."""
    # Check if model was trained with sample_weight by trying a test prediction
    try:
        # Try prediction without sample_weight first
        y_pred_proba = model.predict_proba(predict_df)
    except KeyError as e:
        if 'sample_weight' in str(e):
            # Model was trained with sample_weight, add it with uniform values
            predict_df_with_weights = predict_df.copy()
            predict_df_with_weights['sample_weight'] = 1.0
            y_pred_proba = model.predict_proba(predict_df_with_weights)
            logging.info("Added sample_weight column for prediction (model was trained with weights)")
        else:
            raise e
    
    # Ensure prediction columns exist
    if 1 not in y_pred_proba.columns or 0 not in y_pred_proba.columns:
        raise ValueError("Prediction probabilities do not contain expected columns (0 and 1).")
    
    # Apply calibration if calibrator is provided
    if calibrator is not None:
        logging.info("Applying calibration to predictions...")
        try:
            # Preferred: probability-only calibrator on class-1 vector
            original_probs = y_pred_proba[1].values
            calibrated_probs = calibrator.predict_proba(original_probs)
            # Handle Nx2 outputs
            if hasattr(calibrated_probs, 'ndim') and getattr(calibrated_probs, 'ndim', 1) == 2 and calibrated_probs.shape[1] == 2:
                calibrated_probs = calibrated_probs[:, 1]
            y_pred_proba[1] = calibrated_probs
            y_pred_proba[0] = 1 - calibrated_probs
            logging.info(
                f"Calibration applied (prob-only). Original prob range: [{original_probs.min():.3f}, {original_probs.max():.3f}], "
                f"Calibrated prob range: [{calibrated_probs.min():.3f}, {calibrated_probs.max():.3f}]"
            )
        except Exception:
            # Fallback: feature-based calibrator expects features
            try:
                predict_df_clean = predict_df.drop(columns=['sample_weight'], errors='ignore')
                logging.info(
                    f"Applying feature-based calibrator to {len(predict_df_clean)} predictions with {len(predict_df_clean.columns)} features"
                )
                calibrated_probs_fb = calibrator.predict_proba(predict_df_clean)
                if isinstance(calibrated_probs_fb, pd.DataFrame):
                    calibrated_probs = calibrated_probs_fb[1].values
                else:
                    calibrated_probs = calibrated_probs_fb[:, 1]
                y_pred_proba[1] = calibrated_probs
                y_pred_proba[0] = 1 - calibrated_probs
            except Exception as e:
                logging.error(f"Error applying calibration: {e}. Using original predictions.")
                logging.error(f"Calibrator type: {type(calibrator)}")
                logging.error(
                    f"Predict dataframe shape: {predict_df_clean.shape if 'predict_df_clean' in locals() else 'Not created'}"
                )
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
    
    return y_pred_proba

def create_results_dataframe(df_original, y_pred_proba):
    """
    Create the final results DataFrame using the original df and predictions,
    aligned by index. Includes all necessary odds columns.
    """
    # Ensure y_pred_proba index is a subset of df_original index
    common_index = df_original.index.intersection(y_pred_proba.index)
    if len(common_index) != len(y_pred_proba.index):
        logging.warning("Index mismatch between original data and predictions. Using common index.")
        # Optionally handle this, e.g., raise error or filter y_pred_proba
        # Proceeding with common index
        
    # Select the relevant rows from the original DataFrame using the prediction index
    df_orig_subset = df_original.loc[common_index]
    
    # Select the corresponding predictions
    y_pred_proba_subset = y_pred_proba.loc[common_index]

    # Check if 'y_true' column exists for y_true
    y_true_col_name = 'y_true' # Use the correct column name
    if y_true_col_name not in df_orig_subset.columns:
        logging.error(f"'{y_true_col_name}' column not found in original data. Cannot determine fight outcomes.")
        raise ValueError(f"Critical column '{y_true_col_name}' is missing. Cannot proceed with profit calculations.")
    else:
        # Assuming 'y_true' is 1 if fighter1 wins, 0 otherwise
        y_true_values = df_orig_subset[y_true_col_name]

    # Define required odds columns
    required_odds_cols = [
        'f1_ip_closing_odds', 'f2_ip_closing_odds',
        'f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds',
        # Add vigless if they were included in main.py's merge
        'f1_sevenday_vigless_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds' 
    ]

    results_data = {
        'fighter1_name': df_orig_subset['fighter1_name'],
        'fighter2_name': df_orig_subset['fighter2_name'],
        'event_date': pd.to_datetime(df_orig_subset['event_date']),
        'f1_y_proba': y_pred_proba_subset[1], # Prob fighter1 wins
        'f2_y_proba': y_pred_proba_subset[0], # Prob fighter2 wins
        'y_true': y_true_values # Actual outcome (1 if f1 won, 0 otherwise)
    }

    # Add odds columns safely, using pd.NA if a column is missing
    for col in required_odds_cols:
        if col in df_orig_subset.columns:
            results_data[col] = df_orig_subset[col]
        else:
            logging.warning(f"Odds column '{col}' not found in original data subset. Filling with NA.")
            results_data[col] = pd.NA

    results_df = pd.DataFrame(results_data, index=common_index) # Ensure the results_df keeps the correct index
    
    return results_df


def create_profit_df(use_calibration=False):
    """Main function to orchestrate the data loading and prediction process."""
    # Load paths and assets
    model_path, training_data_csv, scaler_path = load_paths()
    model, scaler, calibrator = load_assets(model_path, scaler_path, use_calibration)

    # Load and prepare data
    df_orig = load_data(training_data_csv)
    
    # Ensure event_date is datetime for filtering
    df_orig['event_date'] = pd.to_datetime(df_orig['event_date'])
    
    # Check for data cutoff file and apply if present (before experience filtering)
    cutoff_path = os.path.join(model_path, 'data_cutoff.txt')
    if os.path.exists(cutoff_path):
        try:
            with open(cutoff_path, 'r') as f:
                data_cutoff = f.read().strip()
            print(f"Found data cutoff file: {data_cutoff}")
            
            # Apply cutoff filter
            before_cutoff = len(df_orig)
            df_orig = df_orig[df_orig['event_date'] <= pd.Timestamp(data_cutoff)]
            after_cutoff = len(df_orig)
            print(f"Applied data cutoff {data_cutoff}: {before_cutoff} -> {after_cutoff} rows ({before_cutoff - after_cutoff} filtered)")
        except Exception as e:
            print(f"Warning: Could not apply data cutoff from {cutoff_path}: {e}")
    else:
        print("No data cutoff file found - using all available data")
    
    # Ensure required columns exist before preparing features
    required_for_prep = ['fighter1_name', 'fighter2_name', 'event_date', 
                         'f1_ip_closing_odds', 'f2_ip_closing_odds', 
                         'f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds']
                         # Add vigless if needed: 'f1_sevenday_vigless_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds']
    missing_cols = [col for col in required_for_prep if col not in df_orig.columns]
    if missing_cols:
         raise ValueError(f"Missing required columns in {training_data_csv} for feature preparation: {missing_cols}")
         
    # Read features from feats.txt in the model directory
    feats_path = os.path.join(model_path, 'feats.txt')
    try:
        with open(feats_path, 'r') as f:
            feats = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(feats)} features from {feats_path}")
    except FileNotFoundError:
        print(f"Error: feats.txt not found at {feats_path}")
        # Handle the error appropriately, e.g., exit or use default features
        feats = [] # Or some default list, or raise an exception
        # sys.exit(1) # Uncomment to exit if file not found

    # IMPORTANT: prepare_features needs ALL historical data to count prior fights correctly
    # It filters by experience (>=2 prior fights) by looking at fights BEFORE each fight's date
    # So we must pass the full dataset, not just test data
    df_train = prepare_features(df_orig, feats) # Use the loaded feats - includes all data for experience counting
    
    if df_train.empty:
        logging.error("Feature preparation resulted in an empty DataFrame. Cannot proceed.")
        return pd.DataFrame()
    
    # NOW filter by test start date AFTER experience filtering
    # This ensures we only predict on test/holdout data that meets experience criteria
    test_start_date_path = os.path.join(model_path, 'test_start_date.txt')
    if os.path.exists(test_start_date_path):
        try:
            test_start_date_str = read_test_start_date(test_start_date_path)
            test_start_date = pd.to_datetime(test_start_date_str)
            before_date_filter = len(df_train)
            # Filter the prepared features DataFrame by date
            # Need to get event_date from original df_orig aligned by index
            df_train_with_dates = df_orig.loc[df_train.index].copy()
            df_train_with_dates['event_date'] = pd.to_datetime(df_train_with_dates['event_date'])
            test_mask = df_train_with_dates['event_date'] >= test_start_date
            df_train = df_train[test_mask]
            after_date_filter = len(df_train)
            print(f"Filtered by test start date {test_start_date_str}: {before_date_filter} -> {after_date_filter} rows ({before_date_filter - after_date_filter} filtered out)")
            logging.info(f"Using test start date {test_start_date_str} - only predicting on fights from this date onward")
        except Exception as e:
            logging.warning(f"Could not apply test start date filter from {test_start_date_path}: {e}")
            print(f"Warning: Could not apply test start date filter: {e}")
    else:
        logging.warning(f"Test start date file not found at {test_start_date_path}. Will predict on all available data.")
        print(f"Warning: Test start date file not found at {test_start_date_path}. Predicting on all available data.")
    
    if df_train.empty:
        logging.error("After date filtering, no fights remain. Cannot proceed.")
        return pd.DataFrame()

    if df_train.empty:
        logging.error("Feature preparation resulted in an empty DataFrame. Cannot proceed.")
        return pd.DataFrame()

    # Check if this is an EnsemblePredictor (walkforward model)
    from libs.modeling.train import EnsemblePredictor
    is_ensemble = isinstance(model, EnsemblePredictor)
    
    # Scale features only for single models (ensembles handle scaling internally)
    if is_ensemble:
        # For ensemble models, pass unscaled data - EnsemblePredictor will scale internally
        logging.info("Ensemble model detected - skipping scaling (ensemble handles it internally)")
        predict_df = df_train.copy()
    else:
        # For single models, scale features before prediction
        # scaled_df preserves the index from df_train
        predict_df = scale_features(df_train, scaler)

    # Make predictions
    # y_pred_proba preserves the index from predict_df
    # Note that make_predictions handles sample_weight if needed POST scaling
    y_pred_proba = make_predictions(model, predict_df, calibrator)

    # Create final results using df_orig and aligning with y_pred_proba via index
    # This results_df should now contain all necessary original columns + predictions
    results_df = create_results_dataframe(df_orig, y_pred_proba)

    # Display the head of the new DataFrame
    print("Profit Calculation DataFrame Head:")
    print(results_df.head())

    # Export simple predictions CSV
    predictions_export = results_df[['fighter1_name', 'fighter2_name', 'event_date', 'f1_y_proba', 'y_true']].copy()
    predictions_export.rename(columns={'f1_y_proba': 'y_pred_proba'}, inplace=True)
    
    # Save to model directory with calibration-specific filename
    if use_calibration:
        csv_filename = 'predictions_export_calib.csv'
        calibration_status = "calibrated"
    else:
        csv_filename = 'predictions_export.csv'
        calibration_status = "uncalibrated"
        
    predictions_csv_path = os.path.join(model_path, csv_filename)
    predictions_export.to_csv(predictions_csv_path, index=False)
    
    print(f"\nPredictions CSV ({calibration_status}) exported to: {predictions_csv_path}")
    print(f"Exported {len(predictions_export)} fight predictions")
    print("Columns: fighter1_name, fighter2_name, event_date, y_pred_proba, y_true")

    # Return results for potential further use
    return results_df

def convert_implied_to_decimal_odds(implied_proba):
    """Converts implied probability (0-1) to decimal odds (>1)."""
    if pd.isna(implied_proba) or not (0 < implied_proba < 1):
        # Handle invalid probabilities (e.g., 0, 1, NaN, or outside range)
        # Return NA which will be handled later during Decimal conversion
        return pd.NA 
    try:
        # Convert input to string first to maintain precision if it's a float
        implied_proba_str = str(implied_proba)
        # Perform the conversion 1 / p
        return Decimal(1) / Decimal(implied_proba_str)
    except Exception as e:
        logging.error(f"Error converting implied probability {implied_proba} to decimal odds: {e}")
        return pd.NA

def decimal_quantize(value):
    """Helper to quantize Decimal values to 2 decimal places for currency."""
    if pd.isna(value):
        return pd.NA # Propagate NA
    try:
        return Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except: # Handle potential conversion errors
        return pd.NA

def calculate_implied_probability(decimal_odds):
    """Calculates implied probability from decimal odds, handling invalid odds."""
    if pd.isna(decimal_odds) or decimal_odds <= 1:
        return Decimal(0) # Or handle as error/None depending on desired behavior
    return Decimal(1) / Decimal(decimal_odds)

def calculate_vegas_implied_percentage(decimal_odds):
    """Converts decimal odds to implied win percentage (0-100 scale)."""
    if pd.isna(decimal_odds) or decimal_odds <= 1:
        return Decimal(0)
    return (Decimal(1) / Decimal(decimal_odds)) * Decimal(100)

def calculate_ai_edge(ai_percentage, vegas_percentage):
    """Calculates the edge (difference) between AI and Vegas win percentages."""
    if pd.isna(ai_percentage) or pd.isna(vegas_percentage):
        return Decimal(0)
    return Decimal(ai_percentage) * Decimal(100) - Decimal(vegas_percentage)

def find_qualifying_fighters(f1_edge, f2_edge, threshold):
    """Determines which fighters qualify based on edge threshold and returns the best choice."""
    threshold_dec = Decimal(threshold)
    
    f1_qualifies = f1_edge >= threshold_dec
    f2_qualifies = f2_edge >= threshold_dec
    
    if f1_qualifies and f2_qualifies:
        # Both qualify, pick the one with higher edge
        return 'fighter1' if f1_edge >= f2_edge else 'fighter2'
    elif f1_qualifies:
        return 'fighter1'
    elif f2_qualifies:
        return 'fighter2'
    else:
        return None

def calculate_bet_outcome(bet_amount, decimal_odds, did_win):
    """Calculates the profit or loss from a single bet."""
    # Ensure inputs are valid Decimals before calculation
    if pd.isna(bet_amount) or pd.isna(decimal_odds) or pd.isna(did_win):
         logging.warning(f"Cannot calculate bet outcome due to NA values: bet_amount={bet_amount}, odds={decimal_odds}, did_win={did_win}")
         return Decimal(0) # Return 0 profit/loss if calculation is impossible

    bet_amount = Decimal(bet_amount)
    decimal_odds = Decimal(decimal_odds)
    
    if did_win:
        profit = bet_amount * (decimal_odds - Decimal(1))
    else:
        profit = -bet_amount
    return decimal_quantize(profit)

def get_ai_pick_details(row):
    """
    Determines the AI's predicted winner and their *decimal* odds 
    for both closing and seven-day odds.
    Returns: 
        ai_winner_side, ai_winner_proba, 
        ai_winner_dec_odds_closing, ai_winner_dec_odds_sevenday
    """
    if pd.isna(row['f1_y_proba']) or pd.isna(row['f2_y_proba']):
        return None, None, None, None # Cannot determine pick
        
    # Get the calculated decimal odds columns for both types
    f1_dec_odds_closing = row.get('f1_dec_odds_closing', pd.NA)
    f2_dec_odds_closing = row.get('f2_dec_odds_closing', pd.NA)
    f1_dec_odds_sevenday = row.get('f1_dec_odds_sevenday', pd.NA)
    f2_dec_odds_sevenday = row.get('f2_dec_odds_sevenday', pd.NA)
        
    if row['f1_y_proba'] > row['f2_y_proba']:
        return 'fighter1', row['f1_y_proba'], f1_dec_odds_closing, f1_dec_odds_sevenday
    elif row['f2_y_proba'] > row['f1_y_proba']:
         return 'fighter2', row['f2_y_proba'], f2_dec_odds_closing, f2_dec_odds_sevenday
    else: # Equal probability - default to fighter1
        return 'fighter1', row['f1_y_proba'], f1_dec_odds_closing, f1_dec_odds_sevenday

def get_odds_details(row):
    """
    Determines the betting favorite/underdog based on *decimal* odds 
    for both closing and seven-day odds types.
    Returns: 
        fav_side_closing, fav_odds_closing, dog_side_closing, dog_odds_closing,
        fav_side_sevenday, fav_odds_sevenday, dog_side_sevenday, dog_odds_sevenday
    """
    # Use the calculated decimal odds columns
    f1_odds_closing = row.get('f1_dec_odds_closing', pd.NA)
    f2_odds_closing = row.get('f2_dec_odds_closing', pd.NA)
    f1_odds_sevenday = row.get('f1_dec_odds_sevenday', pd.NA)
    f2_odds_sevenday = row.get('f2_dec_odds_sevenday', pd.NA)

    # Helper function to determine fav/dog for a given odds pair
    def get_fav_dog(odds1, odds2, side1_name='fighter1', side2_name='fighter2'):
        if pd.isna(odds1) or pd.isna(odds2) or odds1 <= 1 or odds2 <= 1:
            return None, None, None, None # Cannot determine
        if odds1 < odds2:
            return side1_name, odds1, side2_name, odds2 # fav, fav_odds, dog, dog_odds
        elif odds2 < odds1:
            return side2_name, odds2, side1_name, odds1 # fav, fav_odds, dog, dog_odds
        else:
            return None, None, None, None # Equal odds

    fav_c, fav_o_c, dog_c, dog_o_c = get_fav_dog(f1_odds_closing, f2_odds_closing)
    fav_s, fav_o_s, dog_s, dog_o_s = get_fav_dog(f1_odds_sevenday, f2_odds_sevenday)

    return fav_c, fav_o_c, dog_c, dog_o_c, fav_s, fav_o_s, dog_s, dog_o_s


def check_positive_ev(ai_proba, decimal_odds):
    """Checks if betting on a pick has positive Expected Value (EV)."""
    if pd.isna(ai_proba) or pd.isna(decimal_odds) or decimal_odds <= 1:
        return False
    
    # Ensure probabilities are Decimal for comparison
    ai_proba = Decimal(ai_proba) if not isinstance(ai_proba, Decimal) else ai_proba
    implied_proba = calculate_implied_probability(Decimal(decimal_odds))
    
    # Check if ai_proba is strictly greater than implied_proba
    return ai_proba > implied_proba

def evaluate_single_bet_strategies(event_fights, base_strategy, odds_type, bet_amount):
    """
    Evaluates a *single base strategy* for a given event using a specific odds type.
    
    Args:
        event_fights (pd.DataFrame): Fights for the specific event.
        base_strategy (str): The base name of the strategy (e.g., 'ai_fav_only').
        odds_type (str): 'closing' or 'sevenday'.
        bet_amount (Decimal): The amount to bet.

    Returns:
        dict: Results for this specific strategy variant (e.g., {'wagered': ..., 'profit': ...}).
    """
    event_results = {'wagered': Decimal(0), 'profit': Decimal(0), 'bets': []}
    full_strategy_name = f"{base_strategy}_{odds_type}" # e.g., ai_fav_only_closing

    for index, fight in event_fights.iterrows():
        log_prefix = f"Fight {index} ({fight['fighter1_name']} vs {fight['fighter2_name']}) [{full_strategy_name.upper()}]:"

        # --- Pre-calculate common values for the fight ---
        # These helpers now return both odds types
        ai_winner_side, ai_winner_proba, ai_winner_odds_closing, ai_winner_odds_sevenday = get_ai_pick_details(fight)
        fav_c, fav_o_c, dog_c, dog_o_c, fav_s, fav_o_s, dog_s, dog_o_s = get_odds_details(fight)

        # Explicitly check for winning conditions (y_true == 1 or y_true == 0)
        f1_actually_won = fight['y_true'] == 1
        f2_actually_won = fight['y_true'] == 0

        # Select the correct odds based on odds_type for this evaluation run
        if odds_type == 'closing':
            ai_winner_odds = ai_winner_odds_closing
            fav_side, fav_odds = fav_c, fav_o_c
            dog_side, dog_odds = dog_c, dog_o_c
            f1_dec_odds = fight.get('f1_dec_odds_closing', pd.NA)
            f2_dec_odds = fight.get('f2_dec_odds_closing', pd.NA)
        elif odds_type == 'sevenday':
            ai_winner_odds = ai_winner_odds_sevenday
            fav_side, fav_odds = fav_s, fav_o_s
            dog_side, dog_odds = dog_s, dog_o_s
            f1_dec_odds = fight.get('f1_dec_odds_sevenday', pd.NA)
            f2_dec_odds = fight.get('f2_dec_odds_sevenday', pd.NA)
        else:
            raise ValueError(f"Invalid odds_type: {odds_type}")
            
        # --- Shared AI Pick Checks ---
        if ai_winner_side is None:
            # Log only if an AI-based strategy is being run
            if base_strategy.startswith('ai_'):
                 logging.debug(f"{log_prefix} Cannot determine AI pick. Skipping.")
            continue # Skip fight if AI pick is needed but undetermined

        # Check if the AI's chosen fighter actually won
        ai_pick_actually_won = (ai_winner_side == 'fighter1' and f1_actually_won) or \
                               (ai_winner_side == 'fighter2' and f2_actually_won)

        # --- Evaluate the specific BASE strategy logic ---
        bet_on_side = None
        bet_odds = None
        place_bet = False
        reason = "" # Optional reason for logging

        if base_strategy == 'ai_picked_favorite':
            if fav_side is not None and ai_winner_side == fav_side:
                bet_on_side = fav_side
                bet_odds = fav_odds
                place_bet = True
                reason = f"AI picked favorite {fav_side}"
            elif fav_side is None:
                 logging.debug(f"{log_prefix} No favorite determined. No bet.")
            else:
                 logging.debug(f"{log_prefix} AI picked {ai_winner_side}, Favorite was {fav_side}. No bet.")

        elif base_strategy == 'ai_picked_underdog':
            if dog_side is not None and ai_winner_side == dog_side:
                bet_on_side = dog_side
                bet_odds = dog_odds
                place_bet = True
                reason = f"AI picked underdog {dog_side}"
            elif dog_side is None:
                 logging.debug(f"{log_prefix} No underdog determined. No bet.")
            else:
                 logging.debug(f"{log_prefix} AI picked {ai_winner_side}, Underdog was {dog_side}. No bet.")

        elif base_strategy == 'ai_picked_positive_ev':
             if pd.isna(ai_winner_odds):
                 logging.debug(f"{log_prefix} Odds missing for AI pick {ai_winner_side}. No bet.")
             else:
                 has_ev = check_positive_ev(ai_winner_proba, ai_winner_odds)
                 if has_ev:
                    bet_on_side = ai_winner_side
                    bet_odds = ai_winner_odds
                    place_bet = True
                    reason = f"AI pick {ai_winner_side} has +EV"
                 else:
                     implied_proba = calculate_implied_probability(ai_winner_odds)
                     logging.debug(f"{log_prefix} AI Pick {ai_winner_side} ({ai_winner_proba:.2f} vs {implied_proba:.2f} implied) not +EV. No bet.")

        elif base_strategy.startswith('ai_picked_ev_or_within_'):
            # Supports ai_picked_ev_or_within_{X}pct where X is integer percentage (e.g., 1,2,3,4,5,6,8,10,12,14,16,20)
            if pd.isna(ai_winner_odds) or pd.isna(ai_winner_proba):
                logging.debug(f"{log_prefix} Odds or AI probability missing. No bet.")
            else:
                implied_proba = calculate_implied_probability(ai_winner_odds)
                is_positive_ev = check_positive_ev(ai_winner_proba, ai_winner_odds) # ai_proba > implied_proba

                # Extract threshold percentage from strategy name
                try:
                    threshold_token = base_strategy.split('_')[-1]  # like '5pct'
                    threshold_num = float(threshold_token.replace('pct', ''))
                except Exception:
                    threshold_num = 5.0  # fallback to 5%
                threshold_dec = Decimal(str(threshold_num / 100.0))

                # Within-threshold means: AI is NOT +EV yet, but is within X percentage points
                # of crossing into +EV (i.e., below implied by <= threshold). Do not use +/-.
                ai_proba_dec = Decimal(ai_winner_proba) if not isinstance(ai_winner_proba, Decimal) else ai_winner_proba
                gap_to_positive_ev = implied_proba - ai_proba_dec
                is_within_range = (gap_to_positive_ev > 0) and (gap_to_positive_ev <= threshold_dec)

                if is_positive_ev or is_within_range:
                    bet_on_side = ai_winner_side
                    bet_odds = ai_winner_odds
                    place_bet = True
                    band_str = f"Within {threshold_num:.0f}% of +EV"
                    reason = "EV+" if is_positive_ev else band_str
                    reason = f"AI pick {ai_winner_side} {reason} (AI: {ai_winner_proba:.2f}, Implied: {implied_proba:.2f})"
                else:
                    logging.debug(f"{log_prefix} AI Pick {ai_winner_side} ({ai_winner_proba:.2f}) not +EV or within {threshold_num:.0f}% of implied ({implied_proba:.2f}). No bet.")

        elif base_strategy.startswith('bet_against_ai_if_negative_ev_'):
            # Bet the opposite side if AI's chosen side is sufficiently negative EV by X%
            # Example: bet_against_ai_if_negative_ev_5pct
            if ai_winner_side is None or pd.isna(ai_winner_odds) or pd.isna(ai_winner_proba):
                logging.debug(f"{log_prefix} Missing AI pick or odds/probability. No bet.")
            else:
                # Extract threshold percentage from strategy name
                try:
                    threshold_token = base_strategy.split('_')[-1]  # like '5pct'
                    threshold_num = float(threshold_token.replace('pct', ''))
                except Exception:
                    threshold_num = 5.0  # default to 5%
                threshold_dec = Decimal(str(threshold_num / 100.0))

                implied_proba_ai_pick = calculate_implied_probability(ai_winner_odds)
                # Negative EV magnitude for AI pick = implied - AI
                neg_ev_gap = implied_proba_ai_pick - Decimal(ai_winner_proba)

                if neg_ev_gap >= threshold_dec:
                    # Determine opposite side and its odds under current odds_type
                    if ai_winner_side == 'fighter1':
                        opposite_side = 'fighter2'
                        opposite_odds = f2_dec_odds
                    else:
                        opposite_side = 'fighter1'
                        opposite_odds = f1_dec_odds

                    if pd.notna(opposite_odds) and opposite_odds > 1:
                        bet_on_side = opposite_side
                        bet_odds = opposite_odds
                        place_bet = True
                        reason = (
                            f"AI pick {ai_winner_side} is -{neg_ev_gap * 100:.1f}% EV vs market; betting opposite {opposite_side}"
                        )
                    else:
                        logging.debug(f"{log_prefix} Opposite odds missing/invalid. No bet.")
                else:
                    logging.debug(
                        f"{log_prefix} AI negative EV gap {neg_ev_gap * 100:.1f}% < {threshold_num:.0f}% threshold. No bet."
                    )

        elif base_strategy == 'any_fighter_positive_ev':
             # Check Fighter 1 EV using the selected odds_type
             f1_proba = fight.get('f1_y_proba', pd.NA)
             has_ev_f1 = pd.notna(f1_dec_odds) and pd.notna(f1_proba) and check_positive_ev(f1_proba, f1_dec_odds)

             # Check Fighter 2 EV using the selected odds_type
             f2_proba = fight.get('f2_y_proba', pd.NA)
             has_ev_f2 = pd.notna(f2_dec_odds) and pd.notna(f2_proba) and check_positive_ev(f2_proba, f2_dec_odds)
             
             temp_bet_on_side = None
             temp_bet_odds = None
             
             if has_ev_f1 and has_ev_f2: # Both have EV
                 if ai_winner_side == 'fighter1': temp_bet_on_side, temp_bet_odds = 'fighter1', f1_dec_odds
                 elif ai_winner_side == 'fighter2': temp_bet_on_side, temp_bet_odds = 'fighter2', f2_dec_odds
                 if temp_bet_on_side: reason = f"Both +EV, betting AI pick {temp_bet_on_side}"
                 else: reason = "Both +EV, AI pick indeterminate. No bet."
             elif has_ev_f1: # Only F1 has EV
                 temp_bet_on_side, temp_bet_odds = 'fighter1', f1_dec_odds
                 reason = "Fighter 1 has +EV"
             elif has_ev_f2: # Only F2 has EV
                 temp_bet_on_side, temp_bet_odds = 'fighter2', f2_dec_odds
                 reason = "Fighter 2 has +EV"
             
             if temp_bet_on_side:
                 bet_on_side = temp_bet_on_side
                 bet_odds = temp_bet_odds
                 place_bet = True
             else:
                 # Log why no bet was placed if checks were possible
                 log_msg_f1 = f"F1 EV Check: {'+EV' if has_ev_f1 else ('Not +EV' if pd.notna(f1_dec_odds) and pd.notna(f1_proba) else 'N/A')}"
                 log_msg_f2 = f"F2 EV Check: {'+EV' if has_ev_f2 else ('Not +EV' if pd.notna(f2_dec_odds) and pd.notna(f2_proba) else 'N/A')}"
                 logging.debug(f"{log_prefix} No single +EV bet placed. {log_msg_f1}; {log_msg_f2}")
                 
        elif base_strategy == 'bet_against_ai_if_opponent_5pct_edge':
            temp_bet_on_side = None
            temp_bet_odds = None
            
            if ai_winner_side is not None:
                # Use the selected odds type for the market odds
                market_odds_ai_pick = f1_dec_odds if ai_winner_side == 'fighter1' else f2_dec_odds
                
                if ai_winner_side == 'fighter1':
                    non_ai_pick_side = 'fighter2'
                    ai_proba_non_pick = fight.get('f2_y_proba', pd.NA)
                    market_odds_non_pick = f2_dec_odds
                else: # AI picked fighter2
                    non_ai_pick_side = 'fighter1'
                    ai_proba_non_pick = fight.get('f1_y_proba', pd.NA)
                    market_odds_non_pick = f1_dec_odds
                    
                # Check if we have the necessary data for the non-AI pick
                if pd.notna(ai_proba_non_pick) and pd.notna(market_odds_non_pick) and market_odds_non_pick > 1:
                    market_implied_non_pick = calculate_implied_probability(market_odds_non_pick)

                    # *** The Core Logic: Bet against AI only if AI sees >5% edge on the non-pick vs market ***
                    if ai_proba_non_pick > (market_implied_non_pick + Decimal('0.05')):
                        temp_bet_on_side = non_ai_pick_side
                        temp_bet_odds = market_odds_non_pick
                        reason = f"Betting AGAINST AI pick {ai_winner_side}. Non-pick {non_ai_pick_side} AI proba ({ai_proba_non_pick:.2f}) > 5% above market implied ({market_implied_non_pick:.2f})"
                    else:
                        # Condition False: Bet WITH the original AI pick
                        temp_bet_on_side = ai_winner_side
                        temp_bet_odds = market_odds_ai_pick
                        reason = f"Betting WITH AI pick {ai_winner_side}. Non-pick {non_ai_pick_side} AI proba ({ai_proba_non_pick:.2f}) not > 5% above market implied ({market_implied_non_pick:.2f})"
                else:
                    # Missing data to evaluate the non-AI pick, fall back to betting on AI pick if possible
                    logging.debug(f"{log_prefix} Missing data for non-AI pick {non_ai_pick_side}. Evaluating AI pick {ai_winner_side} instead.")
                    if pd.notna(market_odds_ai_pick) and market_odds_ai_pick > 1:
                         temp_bet_on_side = ai_winner_side
                         temp_bet_odds = market_odds_ai_pick
                         reason = f"Betting on AI pick {ai_winner_side} due to missing non-AI pick data"
                    else:
                         logging.debug(f"{log_prefix} Missing odds for AI pick {ai_winner_side} as well. No bet.")

            else:
                logging.debug(f"{log_prefix} AI pick indeterminate. No bet.")

            if temp_bet_on_side is not None and pd.notna(temp_bet_odds):
                 bet_on_side = temp_bet_on_side
                 bet_odds = temp_bet_odds
                 place_bet = True
            # No need for extensive logging if no bet was placed, handled within the logic above.

        elif base_strategy == 'ai_all_picks':
            # Bet on every AI prediction where odds are available
            if ai_winner_side and pd.notna(ai_winner_odds):
                bet_on_side = ai_winner_side
                bet_odds = ai_winner_odds
                place_bet = True
                reason = f"AI pick {ai_winner_side}"
            elif ai_winner_side:
                 logging.debug(f"{log_prefix} Odds missing for AI pick {ai_winner_side}. No bet.")
            # Case where ai_winner_side is None is handled at the start

        elif base_strategy == 'highest_model_pick_favorite':
            # Event-level strategy: Skip per-fight evaluation, will handle at event level
            continue
            
        elif base_strategy == 'favorite_positive_ev':
            # Event-level strategy: Skip per-fight evaluation, will handle at event level
            continue
            
        elif base_strategy == 'highest_ev_underdog_model_pick':
            # Event-level strategy: Skip per-fight evaluation, will handle at event level
            continue
            
        elif base_strategy == 'highest_ev_any_fighter':
            # Event-level strategy: Skip per-fight evaluation, will handle at event level
            continue
            
        elif base_strategy.startswith('ai_edge_threshold'):
            # Extract threshold from strategy name (e.g., 'ai_edge_threshold_4pct' -> 4)
            try:
                threshold_str = base_strategy.split('_')[-1].replace('pct', '')
                threshold = float(threshold_str)
            except (IndexError, ValueError):
                logging.error(f"Could not extract threshold from strategy name: {base_strategy}")
                continue
            
            # Need AI winner to determine default bet
            if ai_winner_side is None:
                logging.debug(f"{log_prefix} Cannot determine AI pick. No bet.")
                continue
            
            # Calculate Vegas implied percentages
            f1_vegas_pct = calculate_vegas_implied_percentage(f1_dec_odds)
            f2_vegas_pct = calculate_vegas_implied_percentage(f2_dec_odds)
            
            # Get AI percentages (convert from 0-1 scale to 0-100 scale)
            f1_ai_pct = fight.get('f1_y_proba', pd.NA)
            f2_ai_pct = fight.get('f2_y_proba', pd.NA)
            
            if pd.notna(f1_ai_pct) and pd.notna(f2_ai_pct) and pd.notna(f1_vegas_pct) and pd.notna(f2_vegas_pct):
                # Calculate edges for both fighters
                f1_edge = calculate_ai_edge(f1_ai_pct, f1_vegas_pct)
                f2_edge = calculate_ai_edge(f2_ai_pct, f2_vegas_pct)
                
                # New logic: Only bet if AI pick has edge >= threshold, OR if opponent has edge >= threshold
                if ai_winner_side == 'fighter1':
                    # AI picked fighter1
                    if f1_edge >= threshold:
                        # AI pick has sufficient edge, bet on AI pick
                        bet_on_side = 'fighter1'
                        bet_odds = f1_dec_odds
                        place_bet = True
                        reason = f"Betting WITH AI pick: Fighter1 edge {f1_edge:.1f}% >= {threshold}% threshold"
                    elif f2_edge >= threshold:
                        # Opponent has sufficient edge, bet against AI pick
                        bet_on_side = 'fighter2'
                        bet_odds = f2_dec_odds
                        place_bet = True
                        reason = f"Betting AGAINST AI pick: Fighter2 edge {f2_edge:.1f}% >= {threshold}% threshold"
                    else:
                        # Neither fighter meets the threshold, no bet
                        place_bet = False
                        logging.debug(f"{log_prefix} No bet: F1 edge {f1_edge:.1f}%, F2 edge {f2_edge:.1f}% both < {threshold}% threshold")
                else:  # AI picked fighter2
                    # AI picked fighter2
                    if f2_edge >= threshold:
                        # AI pick has sufficient edge, bet on AI pick
                        bet_on_side = 'fighter2'
                        bet_odds = f2_dec_odds
                        place_bet = True
                        reason = f"Betting WITH AI pick: Fighter2 edge {f2_edge:.1f}% >= {threshold}% threshold"
                    elif f1_edge >= threshold:
                        # Opponent has sufficient edge, bet against AI pick
                        bet_on_side = 'fighter1'
                        bet_odds = f1_dec_odds
                        place_bet = True
                        reason = f"Betting AGAINST AI pick: Fighter1 edge {f1_edge:.1f}% >= {threshold}% threshold"
                    else:
                        # Neither fighter meets the threshold, no bet
                        place_bet = False
                        logging.debug(f"{log_prefix} No bet: F1 edge {f1_edge:.1f}%, F2 edge {f2_edge:.1f}% both < {threshold}% threshold")
                
            else:
                logging.debug(f"{log_prefix} Missing data for edge calculation. No bet.")

        # --- Place Bet and Record Results ---
        if place_bet and bet_on_side is not None and pd.notna(bet_odds):
            did_win = (bet_on_side == 'fighter1' and f1_actually_won) or \
                      (bet_on_side == 'fighter2' and f2_actually_won)
            profit = calculate_bet_outcome(bet_amount, bet_odds, did_win)

            if pd.notna(profit): # Only record if calculation was possible
                event_results['wagered'] += bet_amount
                event_results['profit'] += profit
                event_results['bets'].append({
                    'fight_id': index, 
                    'bet_on': bet_on_side, 
                    'odds': bet_odds, 
                    'won': did_win, 
                    'profit': profit
                })
                logging.debug(f"{log_prefix} Bet ${bet_amount:.2f} on {bet_on_side} @ {bet_odds:.2f} ({reason}). Outcome: {'Win' if did_win else 'Loss'} (${profit:.2f})")
            else:
                logging.warning(f"{log_prefix} Could not calculate profit for bet on {bet_on_side} @ {bet_odds}. Skipping bet record.")
        elif place_bet: # Bet should have been placed but data was missing
            logging.debug(f"{log_prefix} Bet condition met, but necessary data (side or odds) was missing. No bet placed.")

    # --- Event-Level Strategies (evaluate after processing all fights) ---
    if base_strategy == 'highest_model_pick_favorite':
        # Strategy: Per event, pick only the highest model-picked fighter who is also a betting odds favorite
        # If no fighter meets this condition, no bet is made for that event
        candidates = []  # List of (fight_index, side, proba, odds) tuples
        
        for index, fight in event_fights.iterrows():
            # Get AI pick details
            ai_winner_side, ai_winner_proba, ai_winner_odds_closing, ai_winner_odds_sevenday = get_ai_pick_details(fight)
            fav_c, fav_o_c, dog_c, dog_o_c, fav_s, fav_o_s, dog_s, dog_o_s = get_odds_details(fight)
            
            # Select the correct odds based on odds_type
            if odds_type == 'closing':
                fav_side, fav_odds = fav_c, fav_o_c
            elif odds_type == 'sevenday':
                fav_side, fav_odds = fav_s, fav_o_s
            else:
                continue
            
            # Check if AI pick exists, is a favorite, and has valid odds
            if (ai_winner_side is not None and 
                fav_side is not None and 
                ai_winner_side == fav_side and 
                pd.notna(ai_winner_proba) and 
                pd.notna(fav_odds) and 
                fav_odds > 1):
                candidates.append((index, ai_winner_side, ai_winner_proba, fav_odds, fight))
        
        # If we have candidates, pick the one with highest model probability
        if candidates:
            # Sort by model probability (descending)
            candidates.sort(key=lambda x: x[2], reverse=True)
            best_index, best_side, best_proba, best_odds, best_fight = candidates[0]
            
            # Place the bet
            f1_actually_won = best_fight['y_true'] == 1
            f2_actually_won = best_fight['y_true'] == 0
            did_win = (best_side == 'fighter1' and f1_actually_won) or \
                     (best_side == 'fighter2' and f2_actually_won)
            profit = calculate_bet_outcome(bet_amount, best_odds, did_win)
            
            if pd.notna(profit):
                event_results['wagered'] += bet_amount
                event_results['profit'] += profit
                event_results['bets'].append({
                    'fight_id': best_index,
                    'bet_on': best_side,
                    'odds': best_odds,
                    'won': did_win,
                    'profit': profit
                })
                logging.debug(f"[{full_strategy_name.upper()}] Event-level bet: ${bet_amount:.2f} on {best_fight['fighter1_name' if best_side == 'fighter1' else 'fighter2_name']} @ {best_odds:.2f} (Model proba: {best_proba:.3f}). Outcome: {'Win' if did_win else 'Loss'} (${profit:.2f})")
        else:
            logging.debug(f"[{full_strategy_name.upper()}] No fighters found who are both model picks and favorites. No bet for this event.")
    
    elif base_strategy == 'favorite_positive_ev':
        # Strategy: Per event, make only 1 bet for the fighter who is a betting favorite AND +EV
        # If no fighter meets this condition, no bet is made for that event
        candidates = []  # List of (fight_index, side, proba, odds, ev_magnitude) tuples
        
        for index, fight in event_fights.iterrows():
            # Get odds details
            fav_c, fav_o_c, dog_c, dog_o_c, fav_s, fav_o_s, dog_s, dog_o_s = get_odds_details(fight)
            
            # Select the correct odds based on odds_type
            if odds_type == 'closing':
                fav_side, fav_odds = fav_c, fav_o_c
                f1_dec_odds = fight.get('f1_dec_odds_closing', pd.NA)
                f2_dec_odds = fight.get('f2_dec_odds_closing', pd.NA)
            elif odds_type == 'sevenday':
                fav_side, fav_odds = fav_s, fav_o_s
                f1_dec_odds = fight.get('f1_dec_odds_sevenday', pd.NA)
                f2_dec_odds = fight.get('f2_dec_odds_sevenday', pd.NA)
            else:
                continue
            
            # Check if favorite exists and has valid odds
            if fav_side is None or pd.isna(fav_odds) or fav_odds <= 1:
                continue
            
            # Get the favorite's probability and check for +EV
            if fav_side == 'fighter1':
                fav_proba = fight.get('f1_y_proba', pd.NA)
            else:  # fav_side == 'fighter2'
                fav_proba = fight.get('f2_y_proba', pd.NA)
            
            # Check if favorite has +EV
            if pd.notna(fav_proba) and check_positive_ev(fav_proba, fav_odds):
                # Calculate EV magnitude for sorting (prefer higher EV)
                implied_proba = calculate_implied_probability(fav_odds)
                ev_magnitude = Decimal(fav_proba) - implied_proba
                candidates.append((index, fav_side, fav_proba, fav_odds, ev_magnitude, fight))
        
        # If we have candidates, pick the one with highest EV magnitude
        if candidates:
            # Sort by EV magnitude (descending)
            candidates.sort(key=lambda x: x[4], reverse=True)
            best_index, best_side, best_proba, best_odds, best_ev, best_fight = candidates[0]
            
            # Place the bet
            f1_actually_won = best_fight['y_true'] == 1
            f2_actually_won = best_fight['y_true'] == 0
            did_win = (best_side == 'fighter1' and f1_actually_won) or \
                     (best_side == 'fighter2' and f2_actually_won)
            profit = calculate_bet_outcome(bet_amount, best_odds, did_win)
            
            if pd.notna(profit):
                event_results['wagered'] += bet_amount
                event_results['profit'] += profit
                event_results['bets'].append({
                    'fight_id': best_index,
                    'bet_on': best_side,
                    'odds': best_odds,
                    'won': did_win,
                    'profit': profit
                })
                implied_proba_best = calculate_implied_probability(best_odds)
                logging.debug(f"[{full_strategy_name.upper()}] Event-level bet: ${bet_amount:.2f} on {best_fight['fighter1_name' if best_side == 'fighter1' else 'fighter2_name']} @ {best_odds:.2f} (Model proba: {best_proba:.3f}, Implied: {implied_proba_best:.3f}, EV: +{best_ev*100:.2f}%). Outcome: {'Win' if did_win else 'Loss'} (${profit:.2f})")
        else:
            logging.debug(f"[{full_strategy_name.upper()}] No favorites found with +EV. No bet for this event.")
    
    elif base_strategy == 'highest_ev_underdog_model_pick':
        # Strategy: Per event, pick only 1 fighter that the model picks who is the odds underdog and has the highest +EV
        # If no fighter meets this condition, no bet is made for that event
        candidates = []  # List of (fight_index, side, proba, odds, ev_magnitude) tuples
        
        for index, fight in event_fights.iterrows():
            # Get AI pick details
            ai_winner_side, ai_winner_proba, ai_winner_odds_closing, ai_winner_odds_sevenday = get_ai_pick_details(fight)
            fav_c, fav_o_c, dog_c, dog_o_c, fav_s, fav_o_s, dog_s, dog_o_s = get_odds_details(fight)
            
            # Select the correct odds based on odds_type
            if odds_type == 'closing':
                dog_side, dog_odds = dog_c, dog_o_c
                ai_winner_odds = ai_winner_odds_closing
            elif odds_type == 'sevenday':
                dog_side, dog_odds = dog_s, dog_o_s
                ai_winner_odds = ai_winner_odds_sevenday
            else:
                continue
            
            # Check if AI pick exists, is an underdog, and has valid odds
            if (ai_winner_side is not None and 
                dog_side is not None and 
                ai_winner_side == dog_side and 
                pd.notna(ai_winner_proba) and 
                pd.notna(ai_winner_odds) and 
                ai_winner_odds > 1):
                
                # Check if the underdog model pick has +EV
                if check_positive_ev(ai_winner_proba, ai_winner_odds):
                    # Calculate EV magnitude for sorting (prefer higher EV)
                    implied_proba = calculate_implied_probability(ai_winner_odds)
                    ev_magnitude = Decimal(ai_winner_proba) - implied_proba
                    candidates.append((index, ai_winner_side, ai_winner_proba, ai_winner_odds, ev_magnitude, fight))
        
        # If we have candidates, pick the one with highest EV magnitude
        if candidates:
            # Sort by EV magnitude (descending)
            candidates.sort(key=lambda x: x[4], reverse=True)
            best_index, best_side, best_proba, best_odds, best_ev, best_fight = candidates[0]
            
            # Place the bet
            f1_actually_won = best_fight['y_true'] == 1
            f2_actually_won = best_fight['y_true'] == 0
            did_win = (best_side == 'fighter1' and f1_actually_won) or \
                     (best_side == 'fighter2' and f2_actually_won)
            profit = calculate_bet_outcome(bet_amount, best_odds, did_win)
            
            if pd.notna(profit):
                event_results['wagered'] += bet_amount
                event_results['profit'] += profit
                event_results['bets'].append({
                    'fight_id': best_index,
                    'bet_on': best_side,
                    'odds': best_odds,
                    'won': did_win,
                    'profit': profit
                })
                implied_proba_best = calculate_implied_probability(best_odds)
                logging.debug(f"[{full_strategy_name.upper()}] Event-level bet: ${bet_amount:.2f} on {best_fight['fighter1_name' if best_side == 'fighter1' else 'fighter2_name']} @ {best_odds:.2f} (Model proba: {best_proba:.3f}, Implied: {implied_proba_best:.3f}, EV: +{best_ev*100:.2f}%). Outcome: {'Win' if did_win else 'Loss'} (${profit:.2f})")
        else:
            logging.debug(f"[{full_strategy_name.upper()}] No underdog model picks found with +EV. No bet for this event.")
    
    elif base_strategy == 'highest_ev_any_fighter':
        # Strategy: Per event, pick only 1 fighter with the highest +EV (any fighter, any odds)
        # If no fighter is +EV, no bet is made for that event
        candidates = []  # List of (fight_index, side, proba, odds, ev_magnitude) tuples
        
        for index, fight in event_fights.iterrows():
            # Get probabilities and odds for both fighters
            f1_proba = fight.get('f1_y_proba', pd.NA)
            f2_proba = fight.get('f2_y_proba', pd.NA)
            
            # Select the correct odds based on odds_type
            if odds_type == 'closing':
                f1_dec_odds = fight.get('f1_dec_odds_closing', pd.NA)
                f2_dec_odds = fight.get('f2_dec_odds_closing', pd.NA)
            elif odds_type == 'sevenday':
                f1_dec_odds = fight.get('f1_dec_odds_sevenday', pd.NA)
                f2_dec_odds = fight.get('f2_dec_odds_sevenday', pd.NA)
            else:
                continue
            
            # Check Fighter 1 for +EV
            if pd.notna(f1_proba) and pd.notna(f1_dec_odds) and f1_dec_odds > 1:
                if check_positive_ev(f1_proba, f1_dec_odds):
                    implied_proba = calculate_implied_probability(f1_dec_odds)
                    ev_magnitude = Decimal(f1_proba) - implied_proba
                    candidates.append((index, 'fighter1', f1_proba, f1_dec_odds, ev_magnitude, fight))
            
            # Check Fighter 2 for +EV
            if pd.notna(f2_proba) and pd.notna(f2_dec_odds) and f2_dec_odds > 1:
                if check_positive_ev(f2_proba, f2_dec_odds):
                    implied_proba = calculate_implied_probability(f2_dec_odds)
                    ev_magnitude = Decimal(f2_proba) - implied_proba
                    candidates.append((index, 'fighter2', f2_proba, f2_dec_odds, ev_magnitude, fight))
        
        # If we have candidates, pick the one with highest EV magnitude
        if candidates:
            # Sort by EV magnitude (descending)
            candidates.sort(key=lambda x: x[4], reverse=True)
            best_index, best_side, best_proba, best_odds, best_ev, best_fight = candidates[0]
            
            # Place the bet
            f1_actually_won = best_fight['y_true'] == 1
            f2_actually_won = best_fight['y_true'] == 0
            did_win = (best_side == 'fighter1' and f1_actually_won) or \
                     (best_side == 'fighter2' and f2_actually_won)
            profit = calculate_bet_outcome(bet_amount, best_odds, did_win)
            
            if pd.notna(profit):
                event_results['wagered'] += bet_amount
                event_results['profit'] += profit
                event_results['bets'].append({
                    'fight_id': best_index,
                    'bet_on': best_side,
                    'odds': best_odds,
                    'won': did_win,
                    'profit': profit
                })
                implied_proba_best = calculate_implied_probability(best_odds)
                logging.debug(f"[{full_strategy_name.upper()}] Event-level bet: ${bet_amount:.2f} on {best_fight['fighter1_name' if best_side == 'fighter1' else 'fighter2_name']} @ {best_odds:.2f} (Model proba: {best_proba:.3f}, Implied: {implied_proba_best:.3f}, EV: +{best_ev*100:.2f}%). Outcome: {'Win' if did_win else 'Loss'} (${profit:.2f})")
        else:
            logging.debug(f"[{full_strategy_name.upper()}] No fighters found with +EV. No bet for this event.")

    return event_results


def evaluate_parlay_strategies(event_fights, base_strategy, odds_type, bet_amount):
    """
    Evaluates a *single parlay base strategy* for a given event using a specific odds type.
    Limits fighter appearances to 2 per parlay *within this strategy variant*.

    Args:
        event_fights (pd.DataFrame): Fights for the specific event.
        base_strategy (str): The base name of the parlay strategy (e.g., 'parlay_2_ai_ev').
        odds_type (str): 'closing' or 'sevenday'.
        bet_amount (Decimal): The amount to bet per parlay.

    Returns:
        dict: Results for this specific parlay strategy variant.
    """
    event_results = {'wagered': Decimal(0), 'profit': Decimal(0), 'bets': []}
    full_strategy_name = f"{base_strategy}_{odds_type}"
    
    # Determine number of legs from base strategy name
    if 'parlay_2' in base_strategy:
        num_legs = 2
    elif 'parlay_3' in base_strategy:
        num_legs = 3
    else:
        logging.error(f"Could not determine number of legs for parlay strategy: {base_strategy}")
        return event_results # Cannot proceed

    # --- Identify potential legs based on the base strategy and odds type ---
    legs_list_for_parlay = []

    for index, fight in event_fights.iterrows():
        # Get data needed for all leg types
        f1_proba = fight.get('f1_y_proba', pd.NA)
        f2_proba = fight.get('f2_y_proba', pd.NA)
        f1_name = fight['fighter1_name']
        f2_name = fight['fighter2_name']
        f1_actually_won = fight['y_true'] == 1
        f2_actually_won = fight['y_true'] == 0

        # Select correct odds based on odds_type
        if odds_type == 'closing':
            f1_dec_odds = fight.get('f1_dec_odds_closing', pd.NA)
            f2_dec_odds = fight.get('f2_dec_odds_closing', pd.NA)
        elif odds_type == 'sevenday':
            f1_dec_odds = fight.get('f1_dec_odds_sevenday', pd.NA)
            f2_dec_odds = fight.get('f2_dec_odds_sevenday', pd.NA)
        else: continue # Should not happen

        # AI pick details (needed for AI EV, Top Conf, AI Random)
        ai_winner_side, ai_winner_proba, _, _ = get_ai_pick_details(fight) # We already selected odds above
        # Determine the odds for the AI's pick based on the selected odds_type
        ai_winner_odds = pd.NA
        if ai_winner_side == 'fighter1':
            ai_winner_odds = f1_dec_odds
        elif ai_winner_side == 'fighter2':
            ai_winner_odds = f2_dec_odds

        # --- Populate legs list based on the base strategy ---
        
        # AI +EV Legs (used by parlay_X_ai_ev)
        if 'ai_picked_positive_ev' in base_strategy:
            if ai_winner_side and pd.notna(ai_winner_odds) and pd.notna(ai_winner_proba):
                if check_positive_ev(ai_winner_proba, ai_winner_odds):
                    leg_won = (ai_winner_side == 'fighter1' and f1_actually_won) or \
                              (ai_winner_side == 'fighter2' and f2_actually_won)
                    legs_list_for_parlay.append({
                         'fight_id': index, 'pick_side': ai_winner_side, 'pick_odds': ai_winner_odds, 
                         'pick_won': leg_won, 'fighters': (f1_name, f2_name)
                    })

        # Odds +EV Legs (used by parlay_X_any_fighter_positive_ev)
        elif 'any_fighter_positive_ev' in base_strategy:
            if pd.notna(f1_dec_odds) and pd.notna(f1_proba) and check_positive_ev(f1_proba, f1_dec_odds):
                 legs_list_for_parlay.append({'fight_id': index, 'pick_side': 'fighter1', 'pick_odds': f1_dec_odds, 'pick_won': f1_actually_won, 'fighters': (f1_name, f2_name)})
            if pd.notna(f2_dec_odds) and pd.notna(f2_proba) and check_positive_ev(f2_proba, f2_dec_odds):
                 legs_list_for_parlay.append({'fight_id': index, 'pick_side': 'fighter2', 'pick_odds': f2_dec_odds, 'pick_won': f2_actually_won, 'fighters': (f1_name, f2_name)})

        # Top Confidence Legs (used by parlay_X_ai_top_confidence)
        elif 'ai_top_confidence' in base_strategy:
             if ai_winner_side and pd.notna(ai_winner_odds) and pd.notna(ai_winner_proba):
                 leg_won = (ai_winner_side == 'fighter1' and f1_actually_won) or \
                           (ai_winner_side == 'fighter2' and f2_actually_won)
                 legs_list_for_parlay.append({
                     'fight_id': index, 'pick_side': ai_winner_side, 'pick_odds': ai_winner_odds, 
                     'pick_won': leg_won, 'fighters': (f1_name, f2_name),
                     'confidence': Decimal(ai_winner_proba) # Keep confidence for sorting
                 })
                 
        # AI Random Legs (used by parlay_X_ai_random) - All valid AI picks
        elif 'ai_random_picks' in base_strategy:
            if ai_winner_side and pd.notna(ai_winner_odds):
                leg_won = (ai_winner_side == 'fighter1' and f1_actually_won) or \
                          (ai_winner_side == 'fighter2' and f2_actually_won)
                legs_list_for_parlay.append({
                    'fight_id': index, 'pick_side': ai_winner_side, 'pick_odds': ai_winner_odds, 
                    'pick_won': leg_won, 'fighters': (f1_name, f2_name)
                })

    # Sort/Shuffle the legs list based on strategy type
    if 'ai_top_confidence' in base_strategy:
        legs_list_for_parlay.sort(key=lambda x: x['confidence'], reverse=True)
    elif 'ai_random_picks' in base_strategy:
        random.shuffle(legs_list_for_parlay)
        
    # --- Generate and evaluate parlays for this specific strategy variant ---
    fighter_parlay_counts_strategy = defaultdict(int)
    placed_parlay_picks_set_strategy = set()
    
    if len(legs_list_for_parlay) < num_legs:
        #logging.debug(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Not enough legs ({len(legs_list_for_parlay)}) to form {num_legs}-leg parlay. Skipping.")
        return event_results # Return empty results for this variant

    #logging.debug(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Found {len(legs_list_for_parlay)} potential legs for {num_legs}-leg parlay.")

    # Use combinations to generate parlays
    for parlay_tuple in combinations(legs_list_for_parlay, num_legs):
        # Ensure unique fights in the parlay
        fight_ids_in_parlay = {leg['fight_id'] for leg in parlay_tuple}
        if len(fight_ids_in_parlay) != num_legs:
            continue # Skip if the same fight is included multiple times (can happen with odds_ev_legs)

        # --- Check Uniqueness and Limits (within this strategy variant) ---
        involved_fighters_set = set()
        picked_fighters_list = []
        valid_legs = True
        for leg in parlay_tuple:
             # Check if odds are valid before proceeding with the parlay
             if pd.isna(leg['pick_odds']):
                  logging.warning(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Skipping parlay due to missing odds in leg: {leg}")
                  valid_legs = False
                  break
             involved_fighters_set.update(leg['fighters']) # Add both fighters
             picked_fighter_name = leg['fighters'][0] if leg['pick_side'] == 'fighter1' else leg['fighters'][1]
             picked_fighters_list.append(picked_fighter_name)
        
        if not valid_legs:
            continue # Skip this combination if any leg had invalid odds

        # Create canonical signature for the picks
        picked_fighters_signature = tuple(sorted(picked_fighters_list))
        
        # Check if this exact combination of picks has already been placed *for this strategy variant*
        if picked_fighters_signature in placed_parlay_picks_set_strategy:
            #logging.debug(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Skipping parlay. Pick combination {picked_fighters_signature} already placed.")
            continue
            
        # Check if any involved fighter exceeds the limit *for this strategy variant*
        limit_exceeded = False
        fighter_triggering_limit = None
        for fighter in involved_fighters_set:
            if fighter_parlay_counts_strategy[fighter] >= 2: # Limit to 2 appearances per event per strategy variant
                limit_exceeded = True
                fighter_triggering_limit = fighter
                break
        
        if limit_exceeded:
            #logging.debug(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Skipping parlay {picked_fighters_signature}. Fighter limit (2) exceeded by: {fighter_triggering_limit} (Count: {fighter_parlay_counts_strategy[fighter_triggering_limit]}).")
            continue
            
        # --- If Checks Pass: Calculate, Place Bet, Update Tracking ---
        combined_odds = Decimal(1)
        parlay_won = True
        parlay_desc_list = []
        for leg in parlay_tuple:
            combined_odds *= leg['pick_odds'] # Assumes odds are Decimal
            if not leg['pick_won']:
                parlay_won = False
            picked_fighter_name_leg = leg['fighters'][0] if leg['pick_side'] == 'fighter1' else leg['fighters'][1]
            parlay_desc_list.append(f"{picked_fighter_name_leg} @ {leg['pick_odds']:.2f}")

        # Ensure combined_odds is valid before calculating profit
        if pd.isna(combined_odds):
             logging.warning(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Combined odds became NA for parlay [{' + '.join(parlay_desc_list)}]. Skipping bet record.")
             continue

        profit = calculate_bet_outcome(bet_amount, combined_odds, parlay_won)
        
        # Record only if profit calculation was successful
        if pd.notna(profit):
            event_results['wagered'] += bet_amount
            event_results['profit'] += profit
            parlay_desc = " + ".join(parlay_desc_list)
            event_results['bets'].append({'parlay_desc': parlay_desc, 'odds': combined_odds, 'won': parlay_won, 'profit': profit})
            logging.debug(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Parlay Bet ${bet_amount:.2f} on [{parlay_desc}] (Combined Odds: {combined_odds:.2f}). Outcome: {'Win' if parlay_won else 'Loss'} (${profit:.2f})")
            
            # Update tracking sets/dicts *for this strategy variant*
            placed_parlay_picks_set_strategy.add(picked_fighters_signature)
            for fighter in involved_fighters_set:
                fighter_parlay_counts_strategy[fighter] += 1
        else:
            logging.warning(f"Event {event_fights['event_date'].iloc[0].date()} [{full_strategy_name.upper()}]: Could not calculate profit for parlay [{' + '.join(parlay_desc_list)}]. Skipping bet record.")


    return event_results


def backtest_strategies(profit_df, start_date_str, initial_bankroll, bet_amount, model_path, edge_thresholds=None):
    """
    Performs backtesting of multiple betting strategies on historical fight data,
    evaluating each strategy using both closing and seven-day opening odds.
    
    Args:
        profit_df (pd.DataFrame): DataFrame from create_profit_df.
                                   Must include 'event_date', 'fighter1_name', 'fighter2_name',
                                   'f1_y_proba', 'f2_y_proba', 'y_true',
                                   'f1_ip_closing_odds', 'f2_ip_closing_odds',
                                   'f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds'.
                                   (Vigless sevenday odds are optional based on inclusion in create_profit_df)
        start_date_str (str): The start date for backtesting (YYYY-MM-DD).
        initial_bankroll (float): The starting bankroll for each strategy variant.
        bet_amount (float): The fixed amount to bet per single fight or parlay.
        model_path (str): The path to the model directory.
        edge_thresholds (list): List of edge thresholds to test for ai_edge_threshold strategy.

    Returns:
        dict: A dictionary containing the final results for each strategy variant.
    """
    logging.info("Starting backtest with Closing and Seven-Day Odds...")
    initial_bankroll_dec = decimal_quantize(initial_bankroll)
    bet_amount_dec = decimal_quantize(bet_amount)
    start_date = pd.to_datetime(start_date_str)

    # Default edge thresholds if not provided
    if edge_thresholds is None:
        edge_thresholds = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]

    # Define BASE strategies 
    base_single_strategies = [
        'ai_picked_favorite',         # Was 'ai_fav_only'
        'ai_picked_underdog',         # Was 'ai_dog_only'
        'ai_picked_positive_ev',      # Was 'ai_ev_only'
        # Band strategies: allow betting if AI is +EV or within X% of implied
        # We'll expand these into concrete thresholds below
        'any_fighter_positive_ev',    # Was 'any_fighter_positive_ev_prefer_ai_pick'
        'bet_against_ai_if_opponent_5pct_edge',   # Was 'ai_odds_ev_only_10'
        'ai_all_picks',               # Was 'ai_picks'
        # Event-level strategies: only one bet per event
        'highest_model_pick_favorite',  # Per event: highest model-picked fighter who is also a favorite
        'favorite_positive_ev',         # Per event: favorite who is +EV (highest EV if multiple)
        'highest_ev_underdog_model_pick', # Per event: underdog model pick with highest +EV
        'highest_ev_any_fighter'        # Per event: highest +EV fighter (any fighter, any odds)
    ]

    # Add ai_picked_ev_or_within_{X}pct strategies similar to edge thresholds
    ev_within_thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
    for t in ev_within_thresholds:
        base_single_strategies.append(f'ai_picked_ev_or_within_{t}pct')

    # Add bet_against_ai_if_negative_ev_{X}pct strategies to fade AI when its pick is sufficiently -EV
    neg_ev_thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
    for t in neg_ev_thresholds:
        base_single_strategies.append(f'bet_against_ai_if_negative_ev_{t}pct')
    
    # Add edge threshold strategies
    for threshold in edge_thresholds:
        base_single_strategies.append(f'ai_edge_threshold_{threshold}pct')
    
    logging.info(f"Testing {len(edge_thresholds)} AI edge threshold strategies with thresholds: {edge_thresholds}%")
    base_parlay_strategies = [
        'parlay_2_legs_ai_picked_positive_ev', # Was 'parlay_2_ai_ev'
        'parlay_3_legs_ai_picked_positive_ev', # Was 'parlay_3_ai_ev'
        'parlay_2_legs_any_fighter_positive_ev', # Name unchanged, logic matches new single name
        #'parlay_3_legs_any_fighter_positive_ev', # Name unchanged, logic matches new single name
        'parlay_2_legs_ai_top_confidence',    # Was 'parlay_2_top_conf'
        'parlay_3_legs_ai_top_confidence',    # Was 'parlay_3_top_conf'
        'parlay_2_legs_ai_random_picks',      # Was 'parlay_2_ai_random'
        'parlay_3_legs_ai_random_picks'       # Was 'parlay_3_ai_random'
    ]
    
    # Dynamically create the full list of strategies to test
    strategies_to_run = []
    for base in base_single_strategies + base_parlay_strategies:
        strategies_to_run.append(f"{base}_closing")
        strategies_to_run.append(f"{base}_sevenday")
    
    # Initialize results tracking for all variants
    results = {
        strategy: {
            'bankroll': initial_bankroll_dec,
            'total_wagered': Decimal(0),
            'total_profit': Decimal(0),
            'bets_placed': 0, # Track number of bets
            'bets_won': 0,     # Track number of wins
            'event_profits': [], # Profit per event where bets were placed
            'event_dates': [],   # Corresponding event dates
            'event_wagered_list': [], # Wagered per event
            'bankroll_history': [initial_bankroll_dec], # Track bankroll over time (append after each event with bets)
            'bankroll_dates': [] # Dates for bankroll history points (after each event with bets)
        } for strategy in strategies_to_run
    }

    # --- Data Preparation (Filtering by Date, Type Conversions) ---
    logging.info(f"Preparing data starting from {start_date_str}...")
    if profit_df.empty:
        logging.warning("Input profit_df for backtesting is empty.")
        return {s: data for s, data in results.items()} # Return initialized results
    
    total_fights_before_date_filter = len(profit_df)
    logging.info(f"Total fights in profit_df before date filter: {total_fights_before_date_filter}")
    
    df = profit_df.copy()
    try:
        df['event_date'] = pd.to_datetime(df['event_date'])
    except Exception as e:
        logging.error(f"Error converting 'event_date' to datetime in backtest_strategies: {e}")
        return {s: data for s, data in results.items()}
        
    df = df[df['event_date'] >= start_date].sort_values(by='event_date')
    initial_fight_count = len(df)
    
    if initial_fight_count == 0:
        logging.warning(f"No fight data found on or after the specified start date: {start_date_str}.")
        return {s: data for s, data in results.items()}
    
    logging.info(f"After date filter: {initial_fight_count} fights from {start_date_str} onwards (filtered out {total_fights_before_date_filter - initial_fight_count} fights)")
    logging.info(f"Backtesting on {initial_fight_count} fights from {start_date_str} onwards.")

    # --- Convert Implied Probabilities to Decimal Odds ---
    ip_to_dec_map = {
        'f1_ip_closing_odds': 'f1_dec_odds_closing',
        'f2_ip_closing_odds': 'f2_dec_odds_closing',
        'f1_sevenday_ip_opening_odds': 'f1_dec_odds_sevenday',
        'f2_sevenday_ip_opening_odds': 'f2_dec_odds_sevenday',
        # Add vigless conversion if needed
        # 'f1_sevenday_vigless_ip_opening_odds': 'f1_dec_odds_sevenday_vigless',
        # 'f2_sevenday_vigless_ip_opening_odds': 'f2_dec_odds_sevenday_vigless',
    }

    missing_ip_cols = []
    for ip_col, dec_col in ip_to_dec_map.items():
        if ip_col in df.columns:
            # Apply conversion and store in the new decimal odds column
            df[dec_col] = df[ip_col].apply(convert_implied_to_decimal_odds)
            # Now convert the new column to Decimal type, handling potential NAs
            df[dec_col] = df[dec_col].apply(lambda x: Decimal(x) if pd.notna(x) and x > 1 else pd.NA)
            nan_count = df[dec_col].isna().sum()
            if nan_count > 0:
                logging.warning(f"{nan_count} NaN values found or kept in {dec_col} after conversion to Decimal.")
        else:
            logging.warning(f"Implied probability column {ip_col} not found. Cannot create {dec_col}.")
            df[dec_col] = pd.NA # Create the column with NAs
            missing_ip_cols.append(ip_col)

    if missing_ip_cols:
         # Decide whether to stop or continue with warnings
         # For now, continue, but log a significant warning
         logging.error(f"CRITICAL WARNING: Implied probability columns missing: {missing_ip_cols}. Backtest results for related odds types will be inaccurate or empty.")

    # Convert AI prediction probabilities to Decimal
    proba_cols = ['f1_y_proba', 'f2_y_proba']
    for col in proba_cols:
        if col in df.columns:
             # Ensure conversion to Decimal happens correctly, handling potential NAs
             # Use str(x) to handle potential floats before Decimal conversion
             df[col] = df[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)
        else:
             logging.warning(f"AI probability column {col} not found.")
             df[col] = pd.NA

    # Convert y_true to integer/boolean if needed
    if 'y_true' in df.columns:
        # Coerce non-numeric to NaN, fill NaN with -1 (for missing/invalid), then convert to int
        df['y_true'] = pd.to_numeric(df['y_true'], errors='coerce').fillna(-1).astype(int) 
        # Check if any outcomes are still -1 (meaning original was non-numeric or missing)
        missing_outcomes = (df['y_true'] == -1).sum()
        if missing_outcomes > 0:
            logging.error(f"{missing_outcomes} fights have missing or invalid outcomes ('y_true' is -1).")
            raise ValueError(f"Cannot proceed with backtesting: {missing_outcomes} fights have missing/invalid outcomes.")
    else:
        logging.error("'y_true' column not found in profit_df. Cannot determine bet outcomes.")
        raise ValueError("Critical column 'y_true' is missing from profit_df. Cannot proceed with backtesting.")

    if df.empty: # Re-check after date filtering
        logging.warning("No fight data remains after date filtering.")
        return {s: data for s, data in results.items()}

    logging.info(f"Data prepared. Found {len(df)} fights across {df['event_date'].nunique()} events.")

    # --- Event Iteration ---
    grouped_events = df.groupby(df['event_date'].dt.date) # Group by date part only

    for event_date, event_fights in grouped_events:
        logging.info(f"\n--- Processing Event: {event_date} ---")
        
        # Debug: Show fights before and after filtering
        all_fights_on_date = profit_df[profit_df['event_date'].dt.date == event_date]
        logging.debug(f"Total fights in original data for {event_date}: {len(all_fights_on_date)}")
        if len(all_fights_on_date) > len(event_fights):
            #logging.debug(f"Fights filtered out: {len(all_fights_on_date) - len(event_fights)}")
            # Show which fights were filtered
            filtered_indices = set(all_fights_on_date.index) - set(event_fights.index)
            for idx in list(filtered_indices)[:5]:  # Show first 5 filtered fights
                fight = all_fights_on_date.loc[idx]
                #logging.debug(f"  Filtered: {fight['fighter1_name']} vs {fight['fighter2_name']}")
        
        logging.info(f"Number of fights in event after filtering: {len(event_fights)}")

        # --- Evaluate strategies for BOTH odds types ---
        for odds_type in ['closing', 'sevenday']:
            logging.debug(f"--- Evaluating strategies using {odds_type.upper()} odds ---")
            
            # Evaluate single bet strategies
            for base_strategy in base_single_strategies:
                full_strategy_name = f"{base_strategy}_{odds_type}"
                # Ensure the strategy exists in results (should always be true here)
                if full_strategy_name not in results: continue 
                
                try:
                    # Pass the DataFrame slice for the event
                    single_results_data = evaluate_single_bet_strategies(
                        event_fights.copy(), base_strategy, odds_type, bet_amount_dec
                    )
                    
                    # Update overall results for this strategy variant
                    event_profit = single_results_data['profit']
                    event_wagered = single_results_data['wagered']
                    num_bets_event = len(single_results_data['bets'])
                    num_wins_event = sum(1 for bet in single_results_data['bets'] if bet['won'])

                    results[full_strategy_name]['total_wagered'] += event_wagered
                    results[full_strategy_name]['total_profit'] += event_profit
                    results[full_strategy_name]['bankroll'] += event_profit
                    results[full_strategy_name]['bets_placed'] += num_bets_event
                    results[full_strategy_name]['bets_won'] += num_wins_event
                    # Append event profit/date to the list only if bets were placed
                    if event_wagered > Decimal(0):
                        results[full_strategy_name]['event_profits'].append(event_profit)
                        results[full_strategy_name]['event_wagered_list'].append(event_wagered)
                        results[full_strategy_name]['event_dates'].append(pd.to_datetime(event_date))
                        # Update bankroll history snapshot post-event
                        results[full_strategy_name]['bankroll_history'].append(results[full_strategy_name]['bankroll'])
                        results[full_strategy_name]['bankroll_dates'].append(pd.to_datetime(event_date))

                    # Log summary for the strategy variant for this event
                    logging.info(f"Strategy [{full_strategy_name.upper()}]: Event Bets: {num_bets_event}, Wins: {num_wins_event}, Wagered: ${event_wagered:.2f}, Profit: ${event_profit:.2f}, New Bankroll: ${results[full_strategy_name]['bankroll']:.2f}")

                except Exception as e:
                    logging.error(f"Error evaluating single strategy {full_strategy_name} for event {event_date}: {e}", exc_info=True)


            # Evaluate parlay strategies
            for base_strategy in base_parlay_strategies:
                full_strategy_name = f"{base_strategy}_{odds_type}"
                if full_strategy_name not in results: continue

                try:
                    # Pass the DataFrame slice for the event
                    parlay_results_data = evaluate_parlay_strategies(
                        event_fights.copy(), base_strategy, odds_type, bet_amount_dec
                    )

                    # Update overall results for this strategy variant
                    event_profit = parlay_results_data['profit']
                    event_wagered = parlay_results_data['wagered']
                    num_bets_event = len(parlay_results_data['bets'])
                    num_wins_event = sum(1 for bet in parlay_results_data['bets'] if bet['won'])
                    
                    results[full_strategy_name]['total_wagered'] += event_wagered
                    results[full_strategy_name]['total_profit'] += event_profit
                    results[full_strategy_name]['bankroll'] += event_profit
                    results[full_strategy_name]['bets_placed'] += num_bets_event
                    results[full_strategy_name]['bets_won'] += num_wins_event
                    # Append event profit/date to the list only if parlays were placed
                    if event_wagered > Decimal(0):
                        results[full_strategy_name]['event_profits'].append(event_profit)
                        results[full_strategy_name]['event_wagered_list'].append(event_wagered)
                        results[full_strategy_name]['event_dates'].append(pd.to_datetime(event_date))
                        # Update bankroll history snapshot post-event
                        results[full_strategy_name]['bankroll_history'].append(results[full_strategy_name]['bankroll'])
                        results[full_strategy_name]['bankroll_dates'].append(pd.to_datetime(event_date))
                    
                    # Log summary for the strategy variant for this event
                    logging.info(f"Strategy [{full_strategy_name.upper()}]: Event Parlays: {num_bets_event}, Wins: {num_wins_event}, Wagered: ${event_wagered:.2f}, Profit: ${event_profit:.2f}, New Bankroll: ${results[full_strategy_name]['bankroll']:.2f}")
                
                except Exception as e:
                     logging.error(f"Error evaluating parlay strategy {full_strategy_name} for event {event_date}: {e}", exc_info=True)


    # --- Final Summary ---
    logging.info("\n--- Backtest Finished ---")
    print("\n======= FINAL BACKTEST SUMMARY =======")
    
    summary_lines = [] # List to hold lines for the file
    
    header1 = f"Start Date: {start_date_str}"
    header2 = f"Initial Bankroll: ${initial_bankroll_dec:.2f}"
    header3 = f"Bet Amount: ${bet_amount_dec:.2f}"
    separator1 = "-" * 40
    # Increased width for strategy name from 30 to 50, Added Sharpe Ratio column
    header_cols = (
        f"{'Strategy':<50} | {'Final Bankroll':>15} | {'Total Profit':>15} | {'Total Wagered':>15} | "
        f"{'Bets':>8} | {'Wins':>8} | {'Win %':>7} | {'ROI (%)':>10} | {'Sharpe (ann.)':>13} | "
        f"{'Sortino (ann.)':>14} | {'CAGR (%)':>10} | {'Max DD (%)':>11} | {'Calmar':>8} | {'PF':>6} | {'ROI-Sharpe':>11}"
    )
    separator2 = "-" * 238
    
    # Print to console and add to list
    print(header1)
    summary_lines.append(header1)
    print(header2)
    summary_lines.append(header2)
    print(header3)
    summary_lines.append(header3)
    print(separator1)
    summary_lines.append(separator1)
    print(header_cols)
    summary_lines.append(header_cols)
    print(separator2)
    summary_lines.append(separator2)

    final_summary = {}
    
    # Custom sorting function to order ai_edge_threshold strategies numerically
    def custom_strategy_sort(strategy_name):
        if 'ai_edge_threshold_' in strategy_name:
            # Extract the numeric threshold and odds type for proper sorting
            parts = strategy_name.split('_')
            threshold_str = parts[3].replace('pct', '')  # Get threshold number
            odds_type = parts[4]  # Get 'closing' or 'sevenday'
            try:
                threshold_num = float(threshold_str)
                # Sort by threshold number first, then by odds type (closing before sevenday)
                return (0, threshold_num, 0 if odds_type == 'closing' else 1)
            except ValueError:
                return (1, strategy_name)  # Fallback to alphabetical if parsing fails
        else:
            # For non-threshold strategies, sort alphabetically but after threshold strategies
            return (2, strategy_name)
    
    # Sort results by annualized Sharpe ratio (calculate for all strategies first)
    strategy_sharpes = {}
    for strategy in results.keys():
        data = results[strategy]
        event_profits = data['event_profits']
        event_dates = data.get('event_dates', [])
        
        # Calculate Sharpe for sorting (simplified version)
        sharpe_ratio = np.nan
        if len(event_profits) > 1:
            prev_bankroll = float(initial_bankroll_dec)
            event_returns = []
            for p in event_profits:
                p_float = float(p)
                if prev_bankroll > 0:
                    event_returns.append(p_float / prev_bankroll)
                prev_bankroll = prev_bankroll + p_float

            if len(event_returns) > 1:
                events_per_year = None
                if len(event_dates) >= 2:
                    sorted_dates = sorted([pd.to_datetime(d) for d in event_dates])
                    days_delta = (sorted_dates[-1] - sorted_dates[0]).days
                    years_span = days_delta / 365.25 if days_delta > 0 else None
                    if years_span and years_span > 0:
                        events_per_year = len(event_returns) / years_span

                rf_per_event = 0.0
                if events_per_year and events_per_year > 0 and RISK_FREE_ANNUAL_RATE > 0:
                    rf_per_event = (1.0 + float(RISK_FREE_ANNUAL_RATE)) ** (1.0 / events_per_year) - 1.0

                excess_returns = np.array(event_returns, dtype=float) - rf_per_event
                mean_excess = float(np.mean(excess_returns))
                std_excess = float(np.std(excess_returns, ddof=1))

                if std_excess > 1e-12:
                    sharpe_per_event = mean_excess / std_excess
                    sharpe_ratio = sharpe_per_event * (np.sqrt(events_per_year) if events_per_year and events_per_year > 0 else 1.0)
                else:
                    if mean_excess > 0:
                        sharpe_ratio = float('inf')
                    elif mean_excess < 0:
                        sharpe_ratio = float('-inf')
                    else:
                        sharpe_ratio = 0.0
        
        strategy_sharpes[strategy] = sharpe_ratio

    # Sort strategies by Sharpe ratio (NaN at bottom)
    def _sharpe_sort_key(strategy_name):
        s = strategy_sharpes[strategy_name]
        return (0, -float(s)) if np.isfinite(s) else (1, 0.0)
    
    # Sort strategies alphabetically for final printed/text output
    sorted_strategies = sorted(results.keys())

    for strategy in sorted_strategies:
        data = results[strategy]
        total_wagered = data['total_wagered']
        total_profit = data['total_profit']
        final_bankroll = data['bankroll']
        bets_placed = data['bets_placed']
        bets_won = data['bets_won']
        event_profits = data['event_profits'] # Get event profits list
        event_wagered_list = data.get('event_wagered_list', [])
        bankroll_history_dec = data.get('bankroll_history', [])
        bankroll_dates = data.get('bankroll_dates', [])
        event_dates = data.get('event_dates', [])
        
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else Decimal(0)
        win_rate = (Decimal(bets_won) / Decimal(bets_placed) * 100) if bets_placed > 0 else Decimal(0)
        
        # --- Calculate Sharpe Ratio (annualized) using per-event returns ---
        # r_t = event_profit_t / bankroll_{t-1}; use sample std (ddof=1), subtract per-event risk-free rate
        sharpe_ratio = np.nan
        if len(event_profits) > 1:
            prev_bankroll = float(initial_bankroll_dec)
            event_returns = []
            for p in event_profits:
                p_float = float(p)
                if prev_bankroll > 0:
                    event_returns.append(p_float / prev_bankroll)
                prev_bankroll = prev_bankroll + p_float

            if len(event_returns) > 1:
                # Estimate events per year from event_dates if available
                events_per_year = None
                if len(event_dates) >= 2:
                    sorted_dates = sorted([pd.to_datetime(d) for d in event_dates])
                    days_delta = (sorted_dates[-1] - sorted_dates[0]).days
                    years_span = days_delta / 365.25 if days_delta > 0 else None
                    if years_span and years_span > 0:
                        events_per_year = len(event_returns) / years_span

                # Risk-free per event from annual rate
                if events_per_year and events_per_year > 0 and RISK_FREE_ANNUAL_RATE > 0:
                    rf_per_event = (1.0 + float(RISK_FREE_ANNUAL_RATE)) ** (1.0 / events_per_year) - 1.0
                else:
                    rf_per_event = 0.0

                excess_returns = np.array(event_returns, dtype=float) - rf_per_event
                mean_excess = float(np.mean(excess_returns))
                std_excess = float(np.std(excess_returns, ddof=1))

                if std_excess > 1e-12:
                    sharpe_per_event = mean_excess / std_excess
                    sharpe_ratio = sharpe_per_event * (np.sqrt(events_per_year) if events_per_year and events_per_year > 0 else 1.0)
                else:
                    # Zero volatility: Sharpe undefined; set sign-based value
                    if mean_excess > 0:
                        sharpe_ratio = float('inf')
                    elif mean_excess < 0:
                        sharpe_ratio = float('-inf')
                    else:
                        sharpe_ratio = 0.0
        # --- End Sharpe Ratio Calculation ---

        # --- Calculate ROI-based Sharpe (annualized) using r_t = event_profit / event_wagered ---
        roi_sharpe = np.nan
        if len(event_profits) > 1 and len(event_wagered_list) == len(event_profits):
            roi_returns = []
            for p, w in zip(event_profits, event_wagered_list):
                w_float = float(w)
                p_float = float(p)
                if w_float > 0:
                    roi_returns.append(p_float / w_float)
            if len(roi_returns) > 1:
                # Use same events_per_year estimate from above if available, else infer from event_dates
                events_per_year_roi = None
                if len(event_dates) >= 2:
                    sorted_dates_roi = sorted([pd.to_datetime(d) for d in event_dates])
                    days_delta_roi = (sorted_dates_roi[-1] - sorted_dates_roi[0]).days
                    years_span_roi = days_delta_roi / 365.25 if days_delta_roi > 0 else None
                    if years_span_roi and years_span_roi > 0:
                        events_per_year_roi = len(roi_returns) / years_span_roi
                rf_per_event_roi = ( (1.0 + float(RISK_FREE_ANNUAL_RATE)) ** (1.0 / events_per_year_roi) - 1.0 ) if (events_per_year_roi and events_per_year_roi > 0 and RISK_FREE_ANNUAL_RATE > 0) else 0.0
                excess_roi = np.array(roi_returns, dtype=float) - rf_per_event_roi
                mean_excess_roi = float(np.mean(excess_roi))
                std_excess_roi = float(np.std(excess_roi, ddof=1))
                if std_excess_roi > 1e-12:
                    roi_sharpe_per_event = mean_excess_roi / std_excess_roi
                    roi_sharpe = roi_sharpe_per_event * (np.sqrt(events_per_year_roi) if events_per_year_roi and events_per_year_roi > 0 else 1.0)
                else:
                    roi_sharpe = float('inf') if mean_excess_roi > 0 else (float('-inf') if mean_excess_roi < 0 else 0.0)

        # --- Sortino Ratio (annualized) using downside std of per-event returns ---
        sortino_ratio = np.nan
        if 'event_returns' in locals() and len(event_returns) > 1:
            downside = np.minimum(np.array(event_returns, dtype=float), 0.0)
            # Downside deviation uses std of negative returns; use ddof=1 when there are negatives
            num_neg = np.sum(downside < 0)
            if num_neg >= 2:
                downside_std = float(np.std(downside[downside < 0], ddof=1))
            elif num_neg == 1:
                downside_std = float(np.std(downside[downside < 0], ddof=0))
            else:
                downside_std = 0.0
            mean_return = float(np.mean(event_returns))
            if downside_std > 1e-12:
                sortino_per_event = mean_return / downside_std
                sortino_ratio = sortino_per_event * (np.sqrt(events_per_year) if 'events_per_year' in locals() and events_per_year and events_per_year > 0 else 1.0)
            else:
                sortino_ratio = float('inf') if mean_return > 0 else (float('-inf') if mean_return < 0 else 0.0)

        # --- CAGR, Max Drawdown, Calmar, Profit Factor ---
        cagr_percent = 0.0
        max_dd_percent = 0.0
        calmar_ratio = np.nan
        profit_factor = np.nan

        # CAGR based on bankroll history over time
        if len(bankroll_history_dec) >= 1 and len(bankroll_dates) >= 1:
            start_value = float(bankroll_history_dec[0])
            end_value = float(bankroll_history_dec[-1])
            if len(bankroll_dates) >= 2:
                days_span = (pd.to_datetime(bankroll_dates[-1]) - pd.to_datetime(bankroll_dates[0])).days
                years_span = days_span / 365.25 if days_span > 0 else None
            else:
                years_span = None
            if start_value > 0 and end_value > 0 and years_span and years_span > 0:
                cagr_percent = ( (end_value / start_value) ** (1.0 / years_span) - 1.0 ) * 100.0

        # Max drawdown from bankroll history
        if len(bankroll_history_dec) >= 2:
            equity = np.array([float(b) for b in bankroll_history_dec], dtype=float)
            running_max = np.maximum.accumulate(equity)
            drawdowns = (equity - running_max) / running_max
            max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
            max_dd_percent = abs(max_dd) * 100.0
            if cagr_percent > 0 and max_dd_percent > 0:
                calmar_ratio = (cagr_percent / 100.0) / (max_dd_percent / 100.0)

        # Profit Factor = gross wins / gross losses
        # Approximate from event profits if detailed bet-level data not available
        if len(event_profits) > 0:
            gross_wins = sum(float(p) for p in event_profits if float(p) > 0)
            gross_losses = abs(sum(float(p) for p in event_profits if float(p) < 0))
            if gross_losses > 0:
                profit_factor = gross_wins / gross_losses
            elif gross_wins > 0:
                profit_factor = float('inf')
            else:
                profit_factor = 0.0

        # Format data line for printing and file (adjusted width for strategy)
        # Increased width for header/separator for Sharpe Ratio
        data_line = (
            f"{strategy:<50} | ${final_bankroll:>14.2f} | ${total_profit:>14.2f} | ${total_wagered:>14.2f} | "
            f"{bets_placed:>8} | {bets_won:>8} | {win_rate:>7.2f}% | {roi:>10.2f}% | {sharpe_ratio:>12.2f} | "
            f"{sortino_ratio:>14.2f} | {cagr_percent:>10.2f} | {max_dd_percent:>11.2f} | {calmar_ratio:>8.2f} | {profit_factor:>6.2f} | {roi_sharpe:>11.2f}"
        )
        print(data_line) # Print to console
        summary_lines.append(data_line) # Add to list for file
        
        final_summary[strategy] = {
            'final_bankroll': float(final_bankroll),
            'total_profit': float(total_profit),
            'total_wagered': float(total_wagered),
            'bets_placed': bets_placed,
            'bets_won': bets_won,
            'win_rate_percent': float(win_rate),
            'roi_percent': float(roi),
            'sharpe_ratio': float(sharpe_ratio) if np.isfinite(sharpe_ratio) else np.nan,
            'sortino_ratio': float(sortino_ratio) if np.isfinite(sortino_ratio) else np.nan,
            'cagr_percent': float(cagr_percent),
            'max_drawdown_percent': float(max_dd_percent),
            'calmar_ratio': float(calmar_ratio) if np.isfinite(calmar_ratio) else np.nan,
            'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else np.nan,
            'roi_sharpe_ratio': float(roi_sharpe) if np.isfinite(roi_sharpe) else np.nan
        }
        
    print(separator2) # Print final separator to console
    summary_lines.append(separator2) # Add final separator for file

    # --- Save Summary to File --- 
    try:
        if USE_CALIBRATION:
            profit_file_path = os.path.join(model_path, 'profit_calibrated.txt')
        else:
            profit_file_path = os.path.join(model_path, 'profit.txt')
        with open(profit_file_path, 'w') as f:
            f.write("\n".join(summary_lines)) # Join lines with newline character
        print(f"\nSaved backtest summary to {profit_file_path}")
    except Exception as e:
        logging.error(f"Failed to save profit summary to {profit_file_path}: {e}")

    # --- Save Summary to DataFrame and CSV ---
    try:
        # Create DataFrame from final_summary
        summary_df = pd.DataFrame.from_dict(final_summary, orient='index')
        summary_df.index.name = 'strategy'
        summary_df = summary_df.reset_index()
        
        # Round numeric columns for better display
        numeric_columns = ['final_bankroll', 'total_profit', 'total_wagered', 'win_rate_percent', 
                          'roi_percent', 'sharpe_ratio', 'sortino_ratio', 'cagr_percent', 
                          'max_drawdown_percent', 'calmar_ratio', 'profit_factor', 'roi_sharpe_ratio']
        for col in numeric_columns:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(4)
        
        # Sort by strategy name for consistency with printed output
        summary_df = summary_df.sort_values('strategy').reset_index(drop=True)
        
        # Save to CSV
        if USE_CALIBRATION:
            csv_file_path = os.path.join(model_path, 'backtest_summary_calibrated.csv')
        else:
            csv_file_path = os.path.join(model_path, 'backtest_summary.csv')
            
        summary_df.to_csv(csv_file_path, index=False)
        print(f"Saved backtest summary DataFrame to {csv_file_path}")
        logging.info(f"Backtest summary DataFrame saved with {len(summary_df)} strategies and {len(summary_df.columns)} columns")
        
        # Display DataFrame info
        print(f"\nBacktest Summary DataFrame Info:")
        print(f"Shape: {summary_df.shape}")
        print(f"Columns: {list(summary_df.columns)}")

        # Additionally, save a CSV that mirrors the printed text table (profit.csv)
        try:
            column_map = {
                'strategy': 'Strategy',
                'final_bankroll': 'Final Bankroll',
                'total_profit': 'Total Profit',
                'total_wagered': 'Total Wagered',
                'bets_placed': 'Bets',
                'bets_won': 'Wins',
                'win_rate_percent': 'Win %',
                'roi_percent': 'ROI (%)',
                'sharpe_ratio': 'Sharpe (ann.)',
                'sortino_ratio': 'Sortino (ann.)',
                'cagr_percent': 'CAGR (%)',
                'max_drawdown_percent': 'Max DD (%)',
                'calmar_ratio': 'Calmar',
                'profit_factor': 'PF',
                'roi_sharpe_ratio': 'ROI-Sharpe'
            }
            ordered_cols = [
                'Strategy','Final Bankroll','Total Profit','Total Wagered','Bets','Wins','Win %',
                'ROI (%)','Sharpe (ann.)','Sortino (ann.)','CAGR (%)','Max DD (%)','Calmar','PF','ROI-Sharpe'
            ]

            display_df = summary_df.rename(columns=column_map)
            # Ensure all expected columns exist before ordering
            missing_for_display = [c for c in ordered_cols if c not in display_df.columns]
            if missing_for_display:
                logging.warning(f"Missing columns for profit.csv display: {missing_for_display}. They will be omitted.")
                present_cols = [c for c in ordered_cols if c in display_df.columns]
                display_df = display_df[present_cols]
            else:
                display_df = display_df[ordered_cols]

            profit_csv_path = os.path.join(model_path, 'profit_calibrated.csv' if USE_CALIBRATION else 'profit.csv')
            display_df.to_csv(profit_csv_path, index=False)
            print(f"Saved strategy table CSV to {profit_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save profit.csv-style table: {e}")
        
    except Exception as e:
        logging.error(f"Failed to save backtest summary DataFrame: {e}")
        summary_df = None

    return final_summary


def visualize_backtest_results(final_summary, model_path):
    """
    Generates and saves visualizations for the backtest results.

    Args:
        final_summary (dict): Dictionary containing final results per strategy.
        model_path (str): Path to the model directory to save plots.
    """
    if not final_summary:
        logging.warning("No final summary data to visualize.")
        return

    strategies = list(final_summary.keys())
    profits = [Decimal(str(data['total_profit'])) for data in final_summary.values()]
    rois = [Decimal(str(data['roi_percent'])) for data in final_summary.values()]
    win_rates = [Decimal(str(data['win_rate_percent'])) for data in final_summary.values()]
    sharpes = [data.get('sharpe_ratio', np.nan) for data in final_summary.values()]

    # Sort data by annualized Sharpe for final visualization ordering (NaN sharpes at bottom)
    def _sharpe_sort_key(idx):
        s = sharpes[idx]
        return (0, -float(s)) if np.isfinite(s) else (1, 0.0)

    sorted_indices = sorted(range(len(sharpes)), key=_sharpe_sort_key)
    sorted_strategies = [strategies[i] for i in sorted_indices]
    sorted_profits = [profits[i] for i in sorted_indices] # Keep profits sorted by ROI index for scatter plot
    sorted_rois = [rois[i] for i in sorted_indices]
    sorted_win_rates = [win_rates[i] for i in sorted_indices]

    # --- 1. Bar Chart: ROI per Strategy ---
    try:
        plt.figure(figsize=(12, 8))
        # Convert Decimal ROI to float for plotting
        bars = plt.barh(sorted_strategies, [float(r) for r in sorted_rois], color=sns.color_palette("viridis", len(sorted_strategies)))
        plt.xlabel("Return on Investment (ROI) (%)") # Updated label
        plt.ylabel("Strategy")
        plt.title("Backtest ROI by Strategy") # Updated title
        plt.gca().invert_yaxis() # Display highest ROI at the top
        plt.axvline(0, color='grey', linestyle='--', lw=1) # Add line at 0% ROI
        plt.tight_layout()

        # Add ROI values as labels on the bars
        # Calculate position for labels first, outside the loop
        max_abs_roi = max(abs(float(r)) for r in sorted_rois) if sorted_rois else 1
        # Position text slightly to the right of the longest bar (pos or neg)
        text_x_position = max_abs_roi * 1.02 
        # Adjust plot x-limits to ensure labels fit
        current_xlim = plt.xlim()
        # Ensure left limit accounts for potential negative bars, right limit for text
        plt.xlim(min(current_xlim[0] * 1.1, -max_abs_roi * 0.1), max(text_x_position * 1.1, current_xlim[1]))

        for bar in bars:
            width = bar.get_width() # This is the ROI value
            roi_text = f'{width:.2f}%' # Format as percentage

            plt.text(text_x_position, # Use the consistent x-position
                     bar.get_y() + bar.get_height()/2,
                     roi_text, 
                     va='center',
                     ha='left') # Always align left from the text_x_position

        # Updated filename
        roi_bar_plot_path = os.path.join(model_path, 'roi_by_strategy.png')
        plt.savefig(roi_bar_plot_path)
        plt.close()
        logging.info(f"Saved ROI bar chart to {roi_bar_plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate or save ROI bar chart: {e}", exc_info=True)


    # --- 2. Scatter Plot: ROI vs. Win Rate ---
    try:
        plt.figure(figsize=(10, 8))
        # Convert Decimal to float for plotting
        scatter = sns.scatterplot(x=[float(wr) for wr in sorted_win_rates], y=[float(r) for r in sorted_rois], hue=sorted_strategies, s=100, palette="viridis")
        
        plt.title('Strategy Performance: ROI vs. Win Rate')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Return on Investment (ROI) (%)')
        plt.axhline(0, color='grey', linestyle='--', lw=1) # Line for 0% ROI
        plt.axvline(50, color='grey', linestyle='--', lw=1) # Line for 50% Win Rate (optional)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Add labels next to points
        for i, strategy in enumerate(sorted_strategies):
            plt.text(float(sorted_win_rates[i]) + 0.5, float(sorted_rois[i]), strategy, fontsize=8)

        # Move legend outside the plot
        plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

        roi_plot_path = os.path.join(model_path, 'roi_vs_winrate.png')
        plt.savefig(roi_plot_path)
        plt.close() # Close the figure
        logging.info(f"Saved ROI vs Win Rate scatter plot to {roi_plot_path}")
    except Exception as e:
         logging.error(f"Failed to generate or save ROI scatter plot: {e}", exc_info=True)


    # --- 3. Bar Chart: Sharpe Ratio per Strategy --- 
    try:
        # Get Sharpe Ratios, handling potential NaNs from the summary
        sharpes = [data.get('sharpe_ratio', np.nan) for data in final_summary.values()]
        
        # Filter out strategies with NaN Sharpe ratios for plotting
        valid_indices = [i for i, s in enumerate(sharpes) if not np.isnan(s)]
        if not valid_indices:
            logging.warning("No valid Sharpe Ratios found to plot.")
            return # Skip this plot if no valid data
            
        plot_strategies = [strategies[i] for i in valid_indices]
        plot_sharpes = [sharpes[i] for i in valid_indices] # Use Decimal for precision if possible, else float

        # Sort the valid data by Sharpe Ratio for plotting
        sorted_plot_indices = sorted(range(len(plot_sharpes)), key=lambda k: plot_sharpes[k], reverse=True)
        sorted_plot_strategies = [plot_strategies[i] for i in sorted_plot_indices]
        sorted_plot_sharpes = [plot_sharpes[i] for i in sorted_plot_indices]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(sorted_plot_strategies, sorted_plot_sharpes, color=sns.color_palette("coolwarm", len(sorted_plot_strategies)))
        plt.xlabel("Sharpe Ratio (Risk-Adjusted Return)")
        plt.ylabel("Strategy")
        plt.title("Backtest Sharpe Ratio by Strategy")
        plt.gca().invert_yaxis()  # Highest Sharpe at the top
        plt.axvline(0, color='grey', linestyle='--', lw=1) # Line at 0
        plt.tight_layout()

        # Add Sharpe Ratio values as labels
        max_abs_sharpe = max(abs(s) for s in sorted_plot_sharpes) if sorted_plot_sharpes else 1
        text_x_position = max_abs_sharpe * 1.02
        current_xlim = plt.xlim()
        plt.xlim(min(current_xlim[0] * 1.1, -max_abs_sharpe * 0.1), max(text_x_position * 1.1, current_xlim[1]))

        for bar in bars:
            width = bar.get_width() # This is the Sharpe value
            sharpe_text = f'{width:.2f}' # Format Sharpe value

            plt.text(text_x_position,
                     bar.get_y() + bar.get_height()/2,
                     sharpe_text, 
                     va='center',
                     ha='left')

        sharpe_plot_path = os.path.join(model_path, 'sharpe_ratio_by_strategy.png')
        plt.savefig(sharpe_plot_path)
        plt.close()
        logging.info(f"Saved Sharpe Ratio bar chart to {sharpe_plot_path}")
    except Exception as e:
        logging.error(f"Failed to generate or save Sharpe Ratio bar chart: {e}", exc_info=True)


    # --- 4. Risk-Adjusted Scatter Plot: Top 20 Strategies ---
    try:
        # Calculate composite risk-adjusted score for each strategy
        strategy_scores = []
        
        for strategy, data in final_summary.items():
            # Extract metrics
            roi = data.get('roi_percent', 0)
            sharpe = data.get('sharpe_ratio', np.nan)
            sortino = data.get('sortino_ratio', np.nan)
            max_dd = data.get('max_drawdown_percent', 0)
            calmar = data.get('calmar_ratio', np.nan)
            profit_factor = data.get('profit_factor', np.nan)
            
            # Skip strategies with insufficient data
            if np.isnan(sharpe) or np.isnan(sortino):
                continue
                
            # Calculate composite risk-adjusted score
            # Higher values are better for: ROI, Sharpe, Sortino, Calmar, Profit Factor
            # Lower values are better for: Max Drawdown
            
            # Normalize max drawdown (invert so lower is better)
            max_dd_score = max(0, 20 - max_dd) / 20  # Cap at 20%, normalize to 0-1
            
            # Create composite score (weighted average of normalized metrics)
            risk_adjusted_score = (
                0.3 * min(max(sharpe, -3), 3) / 3 +  # Sharpe: clamp to [-3,3], normalize to [-1,1], weight 30%
                0.3 * min(max(sortino, -3), 3) / 3 + # Sortino: same treatment, weight 30%
                0.2 * max_dd_score +                  # Max DD: weight 20%
                0.1 * min(max(calmar if not np.isnan(calmar) else 0, -2), 2) / 2 + # Calmar: weight 10%
                0.1 * min(max(profit_factor if not np.isnan(profit_factor) else 1, 0), 3) / 3  # PF: weight 10%
            )
            
            strategy_scores.append({
                'strategy': strategy,
                'roi': roi,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_dd': max_dd,
                'calmar': calmar if not np.isnan(calmar) else 0,
                'profit_factor': profit_factor if not np.isnan(profit_factor) else 1,
                'risk_adjusted_score': risk_adjusted_score,
                'total_profit': data.get('total_profit', 0)
            })
        
        if not strategy_scores:
            logging.warning("No strategies with valid risk metrics found for risk-adjusted scatter plot.")
            return
        
        # Sort by risk-adjusted score and take top 20
        strategy_scores.sort(key=lambda x: x['risk_adjusted_score'], reverse=True)
        top_20 = strategy_scores[:20]
        
        # Create scatter plot
        plt.figure(figsize=(14, 10))
        
        # Extract data for plotting
        x_values = [s['roi'] for s in top_20]  # X-axis: ROI
        y_values = [s['risk_adjusted_score'] for s in top_20]  # Y-axis: Risk-adjusted score
        sizes = [max(50, min(300, abs(s['total_profit']) / 10)) for s in top_20]  # Size by profit magnitude
        colors = [s['sharpe'] for s in top_20]  # Color by Sharpe ratio
        
        # Create scatter plot
        scatter = plt.scatter(x_values, y_values, s=sizes, c=colors, 
                             cmap='RdYlGn', alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar for Sharpe ratio
        cbar = plt.colorbar(scatter)
        cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20)
        
        # Customize plot
        plt.title('Top 20 Strategies: Risk-Adjusted Performance\n(Size = Total Profit Magnitude, Color = Sharpe Ratio)', 
                  fontsize=14, pad=20)
        plt.xlabel('Return on Investment (ROI) %', fontsize=12)
        plt.ylabel('Composite Risk-Adjusted Score', fontsize=12)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add reference lines
        plt.axhline(y=np.mean(y_values), color='gray', linestyle='--', alpha=0.5, label='Mean Risk Score')
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.5, label='Break-even ROI')
        
        # Add strategy labels for top 5
        top_5 = top_20[:5]
        for i, strategy_data in enumerate(top_5):
            plt.annotate(strategy_data['strategy'], 
                        (strategy_data['roi'], strategy_data['risk_adjusted_score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Add legend for size
        sizes_legend = [50, 150, 300]
        profits_legend = [500, 1500, 3000]
        legend_elements = []
        for size, profit in zip(sizes_legend, profits_legend):
            legend_elements.append(plt.scatter([], [], s=size, c='gray', alpha=0.6, 
                                             label=f'${profit} profit'))
        
        legend1 = plt.legend(handles=legend_elements, title='Profit Magnitude', 
                           loc='upper left', bbox_to_anchor=(0.02, 0.98))
        plt.gca().add_artist(legend1)
        
        # Add metrics text box
        metrics_text = (
            "Risk-Adjusted Score Components:\n"
            "• Sharpe Ratio (30%)\n"
            "• Sortino Ratio (30%)\n"
            "• Max Drawdown (20%)\n"
            "• Calmar Ratio (10%)\n"
            "• Profit Factor (10%)"
        )
        plt.text(0.98, 0.02, metrics_text, transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='bottom', horizontalalignment='right', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        risk_plot_path = os.path.join(model_path, 'risk_adjusted_scatter_top20.png')
        plt.savefig(risk_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved risk-adjusted scatter plot to {risk_plot_path}")
        
        # Print top 5 strategies info
        print(f"\n--- TOP 5 RISK-ADJUSTED STRATEGIES ---")
        for i, strategy_data in enumerate(top_5, 1):
            print(f"{i}. {strategy_data['strategy']}")
            print(f"   ROI: {strategy_data['roi']:.2f}%, Sharpe: {strategy_data['sharpe']:.2f}")
            print(f"   Sortino: {strategy_data['sortino']:.2f}, Max DD: {strategy_data['max_dd']:.2f}%")
            print(f"   Risk Score: {strategy_data['risk_adjusted_score']:.3f}")
            print()
            
    except Exception as e:
        logging.error(f"Failed to generate or save risk-adjusted scatter plot: {e}", exc_info=True)


if __name__ == "__main__":
    configure_profit_logging()
    # === CONFIGURATION FLAGS ===
    USE_CALIBRATION = False  # Set to False to use uncalibrated predictions
    #USE_CALIBRATION = True
    
    print(f"\n🎯 PROFIT ANALYSIS - Calibration: {'ENABLED' if USE_CALIBRATION else 'DISABLED'}")
    print("=" * 60)
    
    # Load paths once at the beginning
    try:
        model_path, training_data_csv, scaler_path = load_paths()
    except Exception as e:
        logging.error(f"Failed to load paths: {e}")
        model_path, training_data_csv, scaler_path = None, None, None 
        # Consider exiting if paths are critical
        # sys.exit(1) 

    # Ensure profit_df is created with all necessary columns
    # Pass model_path and training_data_csv to create_profit_df if it needs them explicitly
    # For now, assuming create_profit_df internally calls load_paths or gets them as needed
    if model_path and training_data_csv: # Check if paths were loaded
        try:
            profit_df = create_profit_df(use_calibration=USE_CALIBRATION)
        except Exception as e:
            logging.error(f"Failed to create profit DataFrame: {e}")
            profit_df = None
    else:
        logging.error("Cannot create profit DataFrame because model_path or training_data_csv is missing.")
        profit_df = None
    
    # --- Backtesting --- 
    path = os.path.join(model_path, 'test_start_date.txt')
    start_date = read_test_start_date(path)  # Read from shared file written by train.py
    print(f"Using test start date from shared file: {start_date}")
    
    initial_bankroll = 1000.00
    bet_amount = 10.00

    # Set logging level - Use logging.DEBUG for detailed bet-by-bet info
    # Use logging.INFO for summary level
    # TEMPORARILY SET TO DEBUG TO DIAGNOSE FILTERING ISSUES
    logging.getLogger().setLevel(logging.DEBUG) # Set to DEBUG to see filtering details
    
    # Log the start of the analysis
    logging.info("="*60)
    logging.info("Starting profit analysis")
    logging.info(f"Model path: {model_path}")
    logging.info(f"Training data: {training_data_csv}")
    logging.info(f"Calibration mode: {'ENABLED' if USE_CALIBRATION else 'DISABLED'}")
    logging.info("="*60)

    # Check if profit_df and model_path are valid before backtesting
    if profit_df is not None and not profit_df.empty and model_path is not None:
         # Ensure required columns for backtesting exist in the created profit_df
         required_backtest_cols = [
             'event_date', 'fighter1_name', 'fighter2_name', 'f1_y_proba', 'f2_y_proba', 'y_true',
             'f1_ip_closing_odds', 'f2_ip_closing_odds', 
             'f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds'
             # Add vigless if needed: 'f1_sevenday_vigless_ip_opening_odds', 'f2_sevenday_vigless_ip_opening_odds'
         ]
         missing_backtest_cols = [col for col in required_backtest_cols if col not in profit_df.columns]
         if missing_backtest_cols:
             logging.error(f"Profit DataFrame is missing essential columns for backtesting: {missing_backtest_cols}. Aborting backtest.")
         else:
             try:
                 # Define edge thresholds for the new AI edge strategy
                 # These represent percentage point differences between AI and Vegas win probabilities
                 # e.g., threshold=4 means bet on fighter if AI gives them 4% higher win chance than Vegas
                 edge_thresholds = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]
                 final_results = backtest_strategies(profit_df, start_date, initial_bankroll, bet_amount, model_path, edge_thresholds) # Pass model_path and edge_thresholds
                 # You can do further analysis with final_results if needed
                 logging.info("Backtesting complete.")
                 
                 # --- Add Visualization Step ---
                 if final_results:
                     visualize_backtest_results(final_results, model_path)
                 else:
                     logging.warning("Skipping visualization as backtest results are empty.")
                 # --- End of Visualization Step ---
                 
             except Exception as e:
                 logging.error(f"Error during backtesting or visualization: {e}", exc_info=True)
    elif model_path is None:
         logging.error("Cannot run backtest because model_path was not loaded.")
    else:
        logging.error("Profit DataFrame creation failed or resulted in an empty DataFrame. Skipping backtest.")

    # Optional: Further processing or saving results 
