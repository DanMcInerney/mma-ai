import sys
import os
import pandas as pd
import joblib
import numpy as np
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
import math
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from autogluon.tabular import TabularPredictor
from sklearn.calibration import calibration_curve # For calibration plots
from libs.paths import data_file, models_dir

# --- Dynamically add project root to sys.path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of dynamic path addition ---

# --- Constants ---
DECIMAL_ZERO = Decimal('0')
DECIMAL_ONE = Decimal('1')
DECIMAL_HUNDRED = Decimal('100')
SMALL_STD_DEV_THRESHOLD = Decimal('1e-9') # Threshold for near-zero std dev

# --- Configuration ---
# Moved configuration to a dedicated section or function later.

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S' 
)
log = logging.getLogger(__name__)

# --- Utility Functions ---

def decimal_quantize(value, places=Decimal("0.01"), rounding=ROUND_HALF_UP):
    """Quantizes a Decimal value to a specified number of decimal places."""
    if pd.isna(value) or value is None:
        return pd.NA 
    try:
        # Ensure input is Decimal
        if not isinstance(value, Decimal):
            value = Decimal(str(value)) 
        return value.quantize(places, rounding=rounding)
    except (TypeError, ValueError, InvalidOperation) as e:
        log.warning(f"Could not quantize value '{value}'. Error: {e}")
        return pd.NA

def calculate_implied_probability(decimal_odds):
    """Calculates implied probability from decimal odds, handling invalid odds."""
    if pd.isna(decimal_odds) or decimal_odds <= DECIMAL_ONE:
        return pd.NA # Return NA for invalid odds to distinguish from 0% probability
    try:
        # Ensure odds are Decimal
        if not isinstance(decimal_odds, Decimal):
             decimal_odds = Decimal(str(decimal_odds))
        # Avoid division by zero just in case, although <=1 check should cover it
        if decimal_odds == DECIMAL_ZERO: return pd.NA 
        return DECIMAL_ONE / decimal_odds
    except (TypeError, ValueError, InvalidOperation) as e:
        log.warning(f"Could not calculate implied probability for odds '{decimal_odds}'. Error: {e}")
        return pd.NA

def calculate_bet_outcome(bet_amount: Decimal, decimal_odds: Decimal, did_win: bool):
    """Calculates the profit or loss from a single bet using Decimals."""
    # Input validation
    if pd.isna(bet_amount) or pd.isna(decimal_odds) or pd.isna(did_win):
         log.debug(f"Skipping bet outcome calculation due to NA input: amount={bet_amount}, odds={decimal_odds}, win={did_win}")
         return DECIMAL_ZERO # Consistent return type

    if not isinstance(bet_amount, Decimal) or not isinstance(decimal_odds, Decimal):
        log.warning(f"Non-Decimal input to calculate_bet_outcome: amount_type={type(bet_amount)}, odds_type={type(decimal_odds)}")
        # Attempt conversion, return 0 on failure
        try:
            bet_amount = Decimal(str(bet_amount))
            decimal_odds = Decimal(str(decimal_odds))
        except (TypeError, ValueError, InvalidOperation):
             return DECIMAL_ZERO

    if bet_amount < DECIMAL_ZERO or decimal_odds <= DECIMAL_ONE:
        log.warning(f"Invalid bet values: amount={bet_amount}, odds={decimal_odds}")
        return DECIMAL_ZERO

    try:
        if did_win:
            profit = bet_amount * (decimal_odds - DECIMAL_ONE)
        else:
            profit = -bet_amount
        return decimal_quantize(profit) # Quantize the final profit
    except (TypeError, ValueError, InvalidOperation) as e:
        log.error(f"Error during profit calculation: amount={bet_amount}, odds={decimal_odds}, win={did_win}. Error: {e}")
        return DECIMAL_ZERO

# --- Data Loading and Initial Preparation ---
# These functions are similar to profit.py but may be refined later

def load_paths():
    """Define and return necessary file paths."""
    model_name = os.getenv("MMA_AI_MODEL_NAME", "ag-20250416_005952 - v61 819train 732test exp default hp feats_mad2")
    model_path = os.getenv("MMA_AI_MODEL_PATH", str(models_dir() / model_name))
    training_data_csv = str(data_file("training_data.csv"))
    scaler_path = os.path.join(model_path, 'scaler.pkl')
    feats_path = os.path.join(model_path, 'feats.txt')
    output_dir = os.path.join(model_path, 'advanced_results') # New directory for advanced results
    os.makedirs(output_dir, exist_ok=True)
    return model_path, training_data_csv, scaler_path, feats_path, output_dir

def load_features_list(feats_path):
    """Loads the list of features from the feats.txt file."""
    try:
        with open(feats_path, 'r') as f:
            feats = [line.strip() for line in f if line.strip()]
        log.info(f"Loaded {len(feats)} features from {feats_path}")
        return feats
    except FileNotFoundError:
        log.error(f"Feature list file not found at {feats_path}")
        raise # Re-raise as this is critical

def load_assets(model_path, scaler_path):
    """Load the model and scaler."""
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    log.info(f"Loaded scaler from {scaler_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    model = TabularPredictor.load(model_path)
    log.info(f"Loaded AutoGluon model from {model_path}")
    return model, scaler

def load_data(csv_path):
    """Load data from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data CSV not found at {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded data from {csv_path}, shape: {df.shape}")
    return df

def convert_odds_columns(df):
    """
    Converts specified implied odds columns to decimal odds columns using Decimals.
    Handles potential errors and logs warnings. Renames original IP columns.
    Uses vig-included odds columns directly if present (e.g., f1_closing_odds).
    """
    log.info("Converting odds columns to Decimal format...")
    df_processed = df.copy()
    
    # Map: Source Column -> Target Decimal Column Name
    odds_conversion_map = {
        # Closing Odds (Prefer direct decimal if available)
        'f1_closing_odds': 'f1_dec_odds_closing', 
        'f2_closing_odds': 'f2_dec_odds_closing', 
        'f1_ip_closing_odds': 'f1_dec_odds_closing', # Fallback if direct not present
        'f2_ip_closing_odds': 'f2_dec_odds_closing', # Fallback
        
        # Seven-Day Odds (Prefer direct decimal if available)
        'f1_sevenday_opening_odds': 'f1_dec_odds_sevenday',
        'f2_sevenday_opening_odds': 'f2_dec_odds_sevenday',
        'f1_sevenday_ip_opening_odds': 'f1_dec_odds_sevenday', # Fallback
        'f2_sevenday_ip_opening_odds': 'f2_dec_odds_sevenday', # Fallback
        
        # Example: Add vigless if you have them
        # 'f1_sevenday_vigless_opening_odds': 'f1_dec_odds_sevenday_vigless', 
        # 'f2_sevenday_vigless_opening_odds': 'f2_dec_odds_sevenday_vigless',
        # 'f1_sevenday_vigless_ip_opening_odds': 'f1_dec_odds_sevenday_vigless', # Fallback
        # 'f2_sevenday_vigless_ip_opening_odds': 'f2_dec_odds_sevenday_vigless', # Fallback
    }

    processed_targets = set() # Track which target columns have been created

    for source_col, target_col in odds_conversion_map.items():
        if target_col in processed_targets: 
            continue # Already processed this target (e.g., direct decimal took precedence)
            
        if source_col in df_processed.columns:
            log.debug(f"Processing odds column: {source_col} -> {target_col}")
            
            # Check if source implies conversion (contains 'ip_') or is direct decimal
            if 'ip_' in source_col.lower(): 
                # Convert from implied probability
                df_processed[target_col] = df_processed[source_col].apply(convert_implied_to_decimal_odds)
                # Rename original IP column to avoid confusion
                df_processed.rename(columns={source_col: f"{source_col}_original_ip"}, inplace=True, errors='ignore')
                log.debug(f"Converted {source_col} (IP) to {target_col}. Renamed original.")
            else:
                # Assume it's already decimal odds, just convert type
                 df_processed[target_col] = df_processed[source_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and x > 1 else pd.NA)
                 # Optional: Rename original decimal column if desired
                 # df_processed.rename(columns={source_col: f"{source_col}_original_dec"}, inplace=True, errors='ignore')
                 log.debug(f"Converted {source_col} (Decimal) to {target_col}.")
            
            # Final check and Decimal conversion for the target column
            df_processed[target_col] = df_processed[target_col].apply(lambda x: Decimal(str(x)) if pd.notna(x) and x > 1 else pd.NA)
            nan_count = df_processed[target_col].isna().sum()
            if nan_count > 0:
                log.warning(f"{nan_count} NaN values in target column '{target_col}' after processing '{source_col}'.")
            
            processed_targets.add(target_col) # Mark target as done
            
        # else: Column not found, do nothing for this source

    # Ensure essential target columns exist, even if filled with NA
    essential_targets = ['f1_dec_odds_closing', 'f2_dec_odds_closing', 'f1_dec_odds_sevenday', 'f2_dec_odds_sevenday']
    for target in essential_targets:
        if target not in df_processed.columns:
            log.warning(f"Essential target odds column '{target}' was not created (source missing?). Filling with NA.")
            df_processed[target] = pd.NA

    return df_processed


def prepare_data_for_backtest(df_raw, features_list, scaler, model, start_date_str):
    """
    Orchestrates data preparation including feature selection, scaling, 
    prediction, date filtering, and type conversions (Odds, Probabilities, Outcome).
    """
    log.info("Starting data preparation pipeline for backtesting...")
    df = df_raw.copy()

    # 1. Prepare features (similar to profit.py - filtering, NaN checks)
    # Assuming prepare_features handles date conversion and basic filtering
    # It should return only the feature columns needed for scaling/prediction
    df_features_only = prepare_features(df, features_list) 
    if df_features_only.empty:
        log.error("Feature preparation resulted in an empty DataFrame. Stopping.")
        return pd.DataFrame()
    
    # Keep track of the index from the filtered features df
    valid_indices = df_features_only.index

    # 2. Scale Features
    scaled_features_df = scale_features(df_features_only, scaler)
    
    # 3. Make Predictions
    # Predictions will have the same index as scaled_features_df
    y_pred_proba = make_predictions(model, scaled_features_df)

    # 4. Combine Predictions and Original Data (filtered by valid_indices)
    # Select original data rows that passed feature prep
    df_filtered = df.loc[valid_indices].copy() 
    
    # Add prediction probabilities
    df_filtered['f1_y_proba'] = y_pred_proba[1]
    df_filtered['f2_y_proba'] = y_pred_proba[0]

    # 5. Filter by Start Date
    try:
        start_date = pd.to_datetime(start_date_str)
        df_filtered['event_date'] = pd.to_datetime(df_filtered['event_date']) # Ensure datetime
        df_filtered = df_filtered[df_filtered['event_date'] >= start_date].sort_values(by='event_date')
        log.info(f"Filtered data from start date {start_date_str}. {len(df_filtered)} rows remain.")
        if df_filtered.empty:
            log.warning("No data remains after date filtering.")
            return pd.DataFrame()
    except Exception as e:
        log.error(f"Error during date filtering: {e}", exc_info=True)
        return pd.DataFrame()
        
    # 6. Convert Odds Columns to Decimal Odds
    # This function now expects direct decimal odds columns preferentially
    # Pass the DataFrame that includes original odds columns and predictions
    df_final = convert_odds_columns(df_filtered)

    # 7. Convert AI Probabilities to Decimal
    proba_cols = ['f1_y_proba', 'f2_y_proba']
    for col in proba_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(lambda x: Decimal(str(x)) if pd.notna(x) else pd.NA)
        else:
            log.warning(f"AI probability column '{col}' not found for Decimal conversion.")
            df_final[col] = pd.NA

    # 8. Process Outcome Column (y_true)
    y_true_col = 'y_true'
    if y_true_col in df_final.columns:
        # Convert to numeric, coerce errors to NaN, check validity (0 or 1)
        df_final[y_true_col] = pd.to_numeric(df_final[y_true_col], errors='coerce')
        valid_outcomes_mask = df_final[y_true_col].isin([0, 1])
        invalid_outcome_count = (~valid_outcomes_mask).sum()
        if invalid_outcome_count > 0:
            log.warning(f"{invalid_outcome_count} rows have invalid or missing outcomes (not 0 or 1). Setting '{y_true_col}' to NA for these rows.")
            df_final.loc[~valid_outcomes_mask, y_true_col] = pd.NA 
        # Convert valid outcomes to integer/boolean - boolean might be slightly cleaner
        df_final[y_true_col] = df_final[y_true_col].astype('boolean') # Uses pandas nullable boolean
    else:
        log.error(f"Outcome column '{y_true_col}' not found. Cannot determine bet outcomes.")
        # Decide whether to return empty or raise error - returning empty for now
        return pd.DataFrame() 

    # 9. Log Final Data Shape and Columns
    log.info(f"Data preparation complete. Final DataFrame shape: {df_final.shape}")
    log.debug(f"Final columns: {df_final.columns.tolist()}")
    
    # 10. Check for essential columns needed by backtester
    essential_cols = [
        'event_date', 'fighter1_name', 'fighter2_name', 'y_true',
        'f1_y_proba', 'f2_y_proba',
        'f1_dec_odds_closing', 'f2_dec_odds_closing',
        'f1_dec_odds_sevenday', 'f2_dec_odds_sevenday'
        # Add others like vigless if used by strategies
    ]
    missing_essentials = [col for col in essential_cols if col not in df_final.columns or df_final[col].isnull().all()]
    if missing_essentials:
        log.error(f"Essential columns missing or entirely NA in final prepared data: {missing_essentials}. Backtesting cannot proceed reliably.")
        return pd.DataFrame() # Return empty to prevent downstream errors

    return df_final


# --- Feature Preparation, Scaling, Prediction (Adapted from profit.py) ---

def prepare_features(df, features_list):
    """Prepare features: Filter by fighter experience (>=2 prior fights), drop NaNs in features+odds, return features-only df with original index."""
    log.info("Preparing features: Applying experience and NaN filters...")
    if df.empty:
        log.warning("Input DataFrame to prepare_features is empty.")
        return pd.DataFrame()

    # Ensure event_date is datetime
    try:
        df['event_date'] = pd.to_datetime(df['event_date'])
    except Exception as e:
        log.error(f"Error converting 'event_date' to datetime in prepare_features: {e}")
        raise # Re-raise the error as this is critical

    # --- Filter by Fighter Experience (Minimum 2 Prior Fights) ---    
    log.debug("Filtering fights based on fighter experience (min 2 prior fights)...")
    eligible_indices_experience = set()
    fighter_counts_cache = {}

    # Sort by date to process chronologically for caching (though caching keys include date)
    df_sorted = df.sort_values(by='event_date')

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
        #else:
            #log.debug(f"Excluding fight {index} ({f1_name} vs {f2_name}) due to experience. Counts: F1={f1_past_fights}, F2={f2_past_fights}") # Can be verbose

    log.info(f"Experience filter complete. {len(eligible_indices_experience)} fights meet minimum experience criteria.")

    if not eligible_indices_experience:
        log.warning("No fights meet the minimum experience criteria.")
        return pd.DataFrame() 

    # Get the subset of the DataFrame that passed the experience filter
    df_exp_filtered = df.loc[list(eligible_indices_experience)].copy()

    # --- Filter by NaNs in Features + Odds (on the experience-filtered subset) ---
    log.debug("Filtering experienced fighters' fights based on NaNs in features + odds...")
    # Include *both* closing and seven-day odds in the NaN check - USE THE ORIGINAL ODDS COL NAMES
    # Assumes convert_odds_columns runs *after* prepare_data_for_backtest calls this
    # Or, adapt prepare_data_for_backtest to pass the *original* df here before conversion
    # For now, check for common original odds names from the input df
    odds_cols_to_check = []
    # Prioritize direct decimal odds column names if they exist in the original df
    if 'f1_closing_odds' in df.columns and 'f2_closing_odds' in df.columns:
        odds_cols_to_check.extend(['f1_closing_odds', 'f2_closing_odds'])
    elif 'f1_ip_closing_odds' in df.columns and 'f2_ip_closing_odds' in df.columns: # Fallback to IP
        odds_cols_to_check.extend(['f1_ip_closing_odds', 'f2_ip_closing_odds'])
    
    if 'f1_sevenday_opening_odds' in df.columns and 'f2_sevenday_opening_odds' in df.columns:
        odds_cols_to_check.extend(['f1_sevenday_opening_odds', 'f2_sevenday_opening_odds'])
    elif 'f1_sevenday_ip_opening_odds' in df.columns and 'f2_sevenday_ip_opening_odds' in df.columns: # Fallback to IP
        odds_cols_to_check.extend(['f1_sevenday_ip_opening_odds', 'f2_sevenday_ip_opening_odds'])
    
    log.debug(f"Columns used for NaN check (Odds): {odds_cols_to_check}")
                 
    features_to_use = [feat for feat in features_list if feat in df_exp_filtered.columns]
    if not features_to_use:
        raise ValueError("No features from features_list found in the experience-filtered DataFrame.")
        
    cols_to_check_nan = features_to_use + odds_cols_to_check

    # Check which columns are actually present in the experience-filtered subset
    cols_present_in_subset = [col for col in cols_to_check_nan if col in df_exp_filtered.columns]
    missing_cols_nan = list(set(cols_to_check_nan) - set(cols_present_in_subset))

    if missing_cols_nan:
        log.warning(f"Columns required for NaN check are missing from experience-filtered DataFrame subset: {missing_cols_nan}. Proceeding with available columns: {cols_present_in_subset}")
        if not cols_present_in_subset:
             log.error("No columns left in subset to check for NaNs after removing missing ones.")
             return pd.DataFrame() # Cannot proceed

    # Identify the index of rows *within the subset* that don't have NaNs in the available columns
    final_indices_to_keep = df_exp_filtered.dropna(subset=cols_present_in_subset).index
    
    rows_dropped_nan = len(df_exp_filtered) - len(final_indices_to_keep)
    log.info(f"NaN filter complete. Dropped {rows_dropped_nan} additional rows due to NaNs. {len(final_indices_to_keep)} fights remain.")

    if not final_indices_to_keep.any(): # Check if index is empty
        log.warning("No fights remain after applying NaN filter.")
        return pd.DataFrame() 

    # --- Create Final Features DataFrame --- 
    # Select only the *feature columns* using the *final indices* from the *original df*
    # The main prepare_data_for_backtest function will later align this with original data + predictions
    df_clean_features = df.loc[final_indices_to_keep, features_to_use].copy()

    if df_clean_features.empty:
        raise ValueError("Final features DataFrame is empty after all filtering.")

    log.info(f"prepare_features finished. Returning features DataFrame with {len(df_clean_features)} rows and columns: {df_clean_features.columns.tolist()}")
    return df_clean_features

def scale_features(df, scaler):
    """Scale the specified features using the provided scaler."""
    log.info(f"Scaling {len(df.columns)} features for {len(df)} rows...")
    try:
        scaled_data = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
        log.info("Feature scaling complete.")
        return scaled_df
    except Exception as e:
        log.error(f"Error during feature scaling: {e}", exc_info=True)
        raise # Re-raise error

def make_predictions(model, predict_df):
    """Make predictions using the model."""
    log.info(f"Making predictions for {len(predict_df)} rows...")
    if predict_df.empty:
        log.warning("Prediction DataFrame is empty.")
        return pd.DataFrame(columns=[0, 1], index=pd.Index([])) # Return empty with expected structure
    try:
        y_pred_proba = model.predict_proba(predict_df)
        # Ensure prediction columns exist
        if 1 not in y_pred_proba.columns or 0 not in y_pred_proba.columns:
            raise ValueError("Prediction probabilities do not contain expected columns (0 and 1).")
        log.info("Predictions complete.")
        return y_pred_proba
    except Exception as e:
        log.error(f"Error during prediction: {e}", exc_info=True)
        raise # Re-raise error


# --- Strategy Helpers ---

def check_positive_ev(ai_proba: Decimal, decimal_odds: Decimal):
    """Checks if betting on a pick has positive Expected Value (EV)."""
    # Basic validation
    if pd.isna(ai_proba) or pd.isna(decimal_odds) or decimal_odds <= DECIMAL_ONE:
        return False, None # Return False and None for EV value if data invalid
        
    try:
        # Ensure types are Decimal
        if not isinstance(ai_proba, Decimal): ai_proba = Decimal(str(ai_proba))
        if not isinstance(decimal_odds, Decimal): decimal_odds = Decimal(str(decimal_odds))
        
        # EV calculation: (Probability of Winning * Profit if Win) - (Probability of Losing * Stake)
        # Simplified EV check: Is model probability > implied probability?
        # ev = (ai_proba * (decimal_odds - DECIMAL_ONE)) - ((DECIMAL_ONE - ai_proba) * DECIMAL_ONE)
        # ev_check = ev > DECIMAL_ZERO
        
        # Direct probability comparison (more robust against floating point issues with full EV calc)
        implied_proba = calculate_implied_probability(decimal_odds)
        if pd.isna(implied_proba): # Handle case where odds were valid but IP calc failed (shouldn't happen)
            return False, None
            
        is_positive = ai_proba > implied_proba
        # Calculate the edge (difference between model proba and implied proba)
        edge = ai_proba - implied_proba 
        return is_positive, edge
        
    except (TypeError, ValueError, InvalidOperation) as e:
        log.warning(f"Error calculating EV for proba={ai_proba}, odds={decimal_odds}. Error: {e}")
        return False, None

def get_fight_details(fight_row):
    """Extracts and calculates key details for a single fight row."""
    details = {
        'f1_name': fight_row.get('fighter1_name', 'N/A'),
        'f2_name': fight_row.get('fighter2_name', 'N/A'),
        'event_date': fight_row.get('event_date', pd.NaT),
        'f1_proba': fight_row.get('f1_y_proba', pd.NA), # Already Decimal
        'f2_proba': fight_row.get('f2_y_proba', pd.NA), # Already Decimal
        'f1_odds_closing': fight_row.get('f1_dec_odds_closing', pd.NA),
        'f2_odds_closing': fight_row.get('f2_dec_odds_closing', pd.NA),
        'f1_odds_sevenday': fight_row.get('f1_dec_odds_sevenday', pd.NA),
        'f2_odds_sevenday': fight_row.get('f2_dec_odds_sevenday', pd.NA),
        'y_true': fight_row.get('y_true', pd.NA) # Already boolean/NA
    }
    
    # Determine AI pick
    details['ai_pick_side'] = None
    details['ai_pick_proba'] = pd.NA
    if pd.notna(details['f1_proba']) and pd.notna(details['f2_proba']):
        if details['f1_proba'] > details['f2_proba']:
            details['ai_pick_side'] = 'fighter1'
            details['ai_pick_proba'] = details['f1_proba']
        elif details['f2_proba'] > details['f1_proba']:
            details['ai_pick_side'] = 'fighter2'
            details['ai_pick_proba'] = details['f2_proba']
        # Handle tie? For now, defaults to None
            
    # Determine outcome
    details['f1_won'] = pd.NA
    if pd.notna(details['y_true']):
         details['f1_won'] = bool(details['y_true'] == True) # True if f1 won, False if f2 won (y_true is boolean)
         
    return details

# --- Bet Sizing Functions ---

def size_bet_fixed(params):
    """Returns a fixed bet amount defined in strategy parameters."""
    amount = params.get('amount', DECIMAL_ZERO)
    if not isinstance(amount, Decimal):
        try: amount = Decimal(str(amount))
        except: amount = DECIMAL_ZERO
    # Ensure minimum bet size (e.g., $0.01)
    min_bet = Decimal('0.01')
    quantized_amount = decimal_quantize(amount) if amount >= min_bet else DECIMAL_ZERO
    return quantized_amount

def size_bet_percentage(current_bankroll: Decimal, params):
    """Calculates bet size as a percentage of the current bankroll."""
    percentage = params.get('percentage', DECIMAL_ZERO)
    if not isinstance(percentage, Decimal):
        try: percentage = Decimal(str(percentage))
        except: percentage = DECIMAL_ZERO
        
    if not isinstance(current_bankroll, Decimal):
         log.warning("Bankroll passed to size_bet_percentage is not Decimal.")
         return DECIMAL_ZERO
         
    if not (DECIMAL_ZERO < percentage <= DECIMAL_HUNDRED):
        log.warning(f"Invalid percentage specified: {percentage}. Must be > 0 and <= 100.")
        return DECIMAL_ZERO
        
    bet_amount = current_bankroll * (percentage / DECIMAL_HUNDRED)
    # Ensure bet amount doesn't exceed bankroll (can happen with rounding)
    bet_amount = min(bet_amount, current_bankroll)
    # Ensure minimum bet size
    min_bet = Decimal('0.01')
    quantized_amount = decimal_quantize(bet_amount) if bet_amount >= min_bet else DECIMAL_ZERO
    return quantized_amount

def size_bet_kelly(current_bankroll: Decimal, ai_proba: Decimal, decimal_odds: Decimal, params):
    """Calculates bet size using the Kelly Criterion (fractional)."""
    kelly_fraction = params.get('kelly_fraction', Decimal('0.1')) # Default to 10% Kelly
    max_bet_fraction = params.get('max_bet_fraction', Decimal('0.1')) # Max portion of bankroll per bet
    
    if not isinstance(kelly_fraction, Decimal): 
        try: kelly_fraction = Decimal(str(kelly_fraction))
        except: kelly_fraction = Decimal('0.1')
    if not isinstance(max_bet_fraction, Decimal): 
        try: max_bet_fraction = Decimal(str(max_bet_fraction))
        except: max_bet_fraction = Decimal('0.1')
        
    if not (DECIMAL_ZERO < kelly_fraction <= DECIMAL_ONE):
        log.warning(f"Invalid Kelly fraction: {kelly_fraction}. Using 0.1.")
        kelly_fraction = Decimal('0.1')
    if not (DECIMAL_ZERO < max_bet_fraction <= DECIMAL_ONE):
         log.warning(f"Invalid max bet fraction: {max_bet_fraction}. Using 0.1.")
         max_bet_fraction = Decimal('0.1')
         
    # Basic validation
    if pd.isna(ai_proba) or pd.isna(decimal_odds) or decimal_odds <= DECIMAL_ONE or pd.isna(current_bankroll) or current_bankroll <= DECIMAL_ZERO:
        return DECIMAL_ZERO
        
    try:
        # Ensure types are Decimal
        if not isinstance(ai_proba, Decimal): ai_proba = Decimal(str(ai_proba))
        if not isinstance(decimal_odds, Decimal): decimal_odds = Decimal(str(decimal_odds))
        if not isinstance(current_bankroll, Decimal): current_bankroll = Decimal(str(current_bankroll))
        
        # Kelly formula: f = (bp - q) / b 
        # where b = decimal odds - 1, p = probability of winning, q = probability of losing (1-p)
        b = decimal_odds - DECIMAL_ONE
        if b <= DECIMAL_ZERO: # Odds imply no profit or loss
            return DECIMAL_ZERO 
            
        p = ai_proba
        q = DECIMAL_ONE - p
        
        kelly_value = (b * p - q) / b
        
        if kelly_value <= DECIMAL_ZERO: # No edge according to Kelly
            return DECIMAL_ZERO
            
        # Apply Kelly fraction and max bet fraction
        bet_fraction = kelly_value * kelly_fraction
        bet_fraction = min(bet_fraction, max_bet_fraction) # Cap bet size
        
        bet_amount = current_bankroll * bet_fraction
        
        # Ensure bet doesn't exceed bankroll
        bet_amount = min(bet_amount, current_bankroll)
        
        # Ensure minimum bet size
        min_bet = Decimal('0.01')
        quantized_amount = decimal_quantize(bet_amount) if bet_amount >= min_bet else DECIMAL_ZERO
        return quantized_amount
        
    except (TypeError, ValueError, InvalidOperation, ZeroDivisionError) as e:
        log.warning(f"Error calculating Kelly bet size for bankroll={current_bankroll}, proba={ai_proba}, odds={decimal_odds}. Error: {e}")
        return DECIMAL_ZERO

# --- Strategy Definitions ---
# Example structure - this should be defined in the main block or a config file
# strategy_definitions = {
#     "Fixed_Bet_AI_Pick_Closing": {
#         "type": "single",
#         "rule": "ai_pick",
#         "odds_type": "closing",
#         "params": {
#             "bet_sizing": "fixed",
#             "amount": Decimal("10.00")
#         }
#     },
#     "Kelly_Fractional_EV_Closing": {
#         "type": "single",
#         "rule": "positive_ev",
#         "odds_type": "closing",
#         "params": {
#             "min_edge_threshold": Decimal("0.01"), # Bet only if edge > 1%
#             "bet_sizing": "kelly",
#             "kelly_fraction": Decimal("0.1"), # 10% Kelly
#             "max_bet_fraction": Decimal("0.05") # Max 5% of bankroll per bet
#         }
#     },
#     "Percent_Bankroll_Confident_Pick_SevenDay": {
#         "type": "single",
#         "rule": "ai_pick_confident",
#         "odds_type": "sevenday",
#         "params": {
#              "confidence_threshold": Decimal("0.60"), # Bet only if AI proba > 60%
#              "bet_sizing": "percentage",
#              "percentage": Decimal("1.0") # Bet 1% of bankroll
#         }
#     }
#     # Add more strategies: baseline (fav/dog), EV thresholds, confidence, hybrids...
# }

# --- Backtesting Engine ---
def run_backtest_engine(df_prepared, strategy_definitions, initial_bankroll, output_dir):
    """Main engine to run backtesting simulations for defined strategies."""
    log.info(f"Starting backtest engine for {len(strategy_definitions)} strategies.")
    log.info(f"Initial bankroll: ${initial_bankroll:.2f}")

    results = {}
    all_bets_dfs = {} # Store detailed bet logs per strategy

    # Initialize results structure for each strategy
    for name, definition in strategy_definitions.items():
        results[name] = {
            'definition': definition,
            'bankroll_history': [(df_prepared['event_date'].min() - pd.Timedelta(days=1), initial_bankroll)], # (date, bankroll)
            'current_bankroll': initial_bankroll,
            'total_wagered': DECIMAL_ZERO,
            'total_profit': DECIMAL_ZERO,
            'bet_count': 0,
            'win_count': 0,
            'push_count': 0, # Optional: Track pushes if odds can be exactly 1.0
            'event_profits': [], # For Sharpe/Sortino (only append if wagered > 0)
            'daily_returns': [], # Store daily % returns for Sortino
            'drawdown_history': [], # (date, drawdown_percentage)
            'peak_bankroll': initial_bankroll,
            'bet_log': [] # Detailed log: [ {fight_details..., bet_amount, profit, bankroll_before, bankroll_after} ]
        }
        all_bets_dfs[name] = [] # List to store fight bet data for later DataFrame creation

    log.info(f"Initialized results structure for {len(results)} strategies.")

    # Group fights by event date (assuming 1 event per day for simplicity, adjust if needed)
    grouped_events = df_prepared.groupby(df_prepared['event_date'].dt.date)
    total_event_days = len(grouped_events)
    processed_event_days = 0

    log.info(f"Processing {total_event_days} event days...")

    # --- Event Loop ---
    for event_date, event_fights in grouped_events:
        processed_event_days += 1
        log.debug(f"\n--- Processing Event Date: {event_date} ({processed_event_days}/{total_event_days}) --- Fights: {len(event_fights)}")
        
        # Track daily starting bankroll for return calculation
        daily_start_bankrolls = {name: data['current_bankroll'] for name, data in results.items()}
        event_day_profits = {name: DECIMAL_ZERO for name in results}
        event_day_wagers = {name: DECIMAL_ZERO for name in results}
        
        # --- Fight Loop --- 
        for fight_index, fight_row in event_fights.iterrows():
            fight_details = get_fight_details(fight_row) # Extract common details
            # Add fight index for reference
            fight_details['fight_id'] = fight_index 

            # --- Strategy Loop --- 
            for name, data in results.items():
                strategy_def = data['definition']
                current_bankroll = data['current_bankroll']
                strategy_params = strategy_def.get('params', {})
                odds_type = strategy_def.get('odds_type', 'closing') # Default to closing

                # --- Determine Bet --- 
                # This is where the core strategy logic resides
                # It should return: bet_on_side ('fighter1'/'fighter2'/None), bet_odds (Decimal/NA), bet_reason (str), edge (Decimal/NA)
                bet_on_side, bet_odds, bet_reason, edge = determine_bet_for_strategy(
                    fight_details, strategy_def, odds_type
                )

                # --- Size Bet --- 
                bet_amount = DECIMAL_ZERO
                if bet_on_side and pd.notna(bet_odds) and bet_odds > DECIMAL_ONE:
                    # Ensure bankroll is positive before attempting to size bet
                    if current_bankroll <= DECIMAL_ZERO:
                        log.debug(f"[{name}] Bankrupt. Cannot size bet.")
                        bet_amount = DECIMAL_ZERO
                    else:
                        bet_sizing_method = strategy_params.get('bet_sizing', 'fixed')
                        ai_proba_for_bet = fight_details['f1_proba'] if bet_on_side == 'fighter1' else fight_details['f2_proba']
                        
                        if bet_sizing_method == 'fixed':
                            bet_amount = size_bet_fixed(strategy_params)
                        elif bet_sizing_method == 'percentage':
                            bet_amount = size_bet_percentage(current_bankroll, strategy_params)
                        elif bet_sizing_method == 'kelly':
                            bet_amount = size_bet_kelly(current_bankroll, ai_proba_for_bet, bet_odds, strategy_params)
                        else:
                            log.warning(f"Unknown bet sizing method '{bet_sizing_method}' for strategy {name}. Defaulting to 0.")
                        
                        # Ensure bet doesn't exceed available bankroll
                        if bet_amount > current_bankroll:
                             log.debug(f"Bet amount {bet_amount:.2f} exceeds bankroll {current_bankroll:.2f} for {name}. Capping bet.")
                             bet_amount = current_bankroll
                             # Re-quantize after capping
                             bet_amount = decimal_quantize(bet_amount) 
                             
                        # Minimum bet check already incorporated in sizing functions
                        if bet_amount == DECIMAL_ZERO:
                            log.debug(f"Sized bet is zero for {name} on fight {fight_index}. No bet placed.")

                # --- Execute Bet & Record --- 
                profit = DECIMAL_ZERO
                bet_placed_flag = False # Track if a bet was actually recorded
                if bet_amount > DECIMAL_ZERO:
                    if pd.isna(fight_details['f1_won']):
                        log.warning(f"Cannot determine outcome for fight {fight_index}. Skipping bet result for strategy {name}.")
                        # How to handle bankroll? Assume loss? Skip? For now, skip profit calc & bankroll update for this bet
                    else:
                        did_win = bool( (bet_on_side == 'fighter1' and fight_details['f1_won']) or \
                                        (bet_on_side == 'fighter2' and not fight_details['f1_won']) )
                        
                        profit = calculate_bet_outcome(bet_amount, bet_odds, did_win)
                        
                        # Store detailed bet info
                        bet_log_entry = {
                            'event_date': fight_details['event_date'].strftime('%Y-%m-%d'), # Store as string for easier DF creation
                            'fight_id': fight_index,
                            'fighter1': fight_details['f1_name'],
                            'fighter2': fight_details['f2_name'],
                            'bet_on_side': bet_on_side,
                            'ai_pick_side': fight_details['ai_pick_side'],
                            'ai_pick_proba': float(fight_details['ai_pick_proba']) if pd.notna(fight_details['ai_pick_proba']) else np.nan,
                            'f1_model_proba': float(fight_details['f1_proba']) if pd.notna(fight_details['f1_proba']) else np.nan,
                            'f2_model_proba': float(fight_details['f2_proba']) if pd.notna(fight_details['f2_proba']) else np.nan,
                            'odds_type': odds_type,
                            'odds_used': float(bet_odds) if pd.notna(bet_odds) else np.nan,
                            'bet_amount': float(bet_amount),
                            'f1_actual_win': int(fight_details['f1_won']) if pd.notna(fight_details['f1_won']) else -1, # 1=win, 0=loss, -1=unknown
                            'bet_won': int(did_win),
                            'profit': float(profit),
                            'bankroll_before': float(current_bankroll),
                            'bankroll_after': float(current_bankroll + profit),
                            'strategy_reason': bet_reason,
                            'edge_calculated': float(edge) if pd.notna(edge) else np.nan
                        }
                        all_bets_dfs[name].append(bet_log_entry)
                        # data['bet_log'].append(bet_log_entry) # Can consume memory, consider writing to file periodically
                        
                        # Update Strategy Totals
                        data['total_wagered'] += bet_amount
                        data['total_profit'] += profit
                        data['current_bankroll'] += profit
                        data['bet_count'] += 1
                        if did_win:
                            data['win_count'] += 1
                        # Accumulate event totals for Sharpe/Sortino calculation later
                        event_day_profits[name] += profit 
                        event_day_wagers[name] += bet_amount
                        bet_placed_flag = True
                            
                        log.debug(f"[{name}] Bet ${bet_amount:.2f} on {bet_on_side} @{bet_odds:.2f} ({bet_reason}). Win: {did_win}, Profit: ${profit:.2f}, New Bankroll: ${data['current_bankroll']:.2f}")
                # --- End Bet Execution ---
                
                # If bankroll hits zero, stop betting for this strategy
                if data['current_bankroll'] <= DECIMAL_ZERO:
                    log.warning(f"[{name}] Bankrupt! Bankroll at or below zero (${data['current_bankroll']:.2f}). Stopping bets for this strategy.")
                    # Optional: could break inner strategy loop here if desired
                    # For now, it will just keep sizing bets as 0
                    
            # --- End Strategy Loop ---
        # --- End Fight Loop ---
        
        # --- Post-Event Updates ---
        for name, data in results.items():
            # Record bankroll at end of day
            current_event_date = pd.to_datetime(event_date) # Ensure datetime object
            data['bankroll_history'].append((current_event_date, data['current_bankroll']))
            
            # Record event profit only if bets were placed during the event
            if event_day_wagers[name] > DECIMAL_ZERO:
                 data['event_profits'].append(event_day_profits[name]) # Append Decimal profit
                 
            # Calculate daily return for Sortino
            start_br = daily_start_bankrolls[name]
            end_br = data['current_bankroll']
            if start_br > DECIMAL_ZERO:
                daily_return = (end_br / start_br) - DECIMAL_ONE
                data['daily_returns'].append(float(daily_return)) # Store as float
            else: # Avoid division by zero if bankroll hit zero
                 data['daily_returns'].append(0.0)
                 
            # Update Drawdown
            data['peak_bankroll'] = max(data['peak_bankroll'], data['current_bankroll'])
            if data['peak_bankroll'] > DECIMAL_ZERO:
                drawdown = (data['peak_bankroll'] - data['current_bankroll']) / data['peak_bankroll']
                data['drawdown_history'].append((current_event_date, float(drawdown)))
            else:
                 data['drawdown_history'].append((current_event_date, 0.0))
                 
    # --- End Event Loop ---

    log.info("Event processing complete.")
    
    # --- Final Results Aggregation & Bet Log DataFrame Creation ---
    for name in results:
        # Convert detailed bet log list to DataFrame
        if all_bets_dfs[name]:
            df_bet_log = pd.DataFrame(all_bets_dfs[name])
            df_bet_log['event_date'] = pd.to_datetime(df_bet_log['event_date'])
            # Ensure correct types
            for col in ['ai_pick_proba', 'f1_model_proba', 'f2_model_proba', 'odds_used', 
                        'bet_amount', 'profit', 'bankroll_before', 'bankroll_after', 'edge_calculated']:
                 if col in df_bet_log.columns:
                      df_bet_log[col] = pd.to_numeric(df_bet_log[col], errors='coerce')
            for col in ['f1_actual_win', 'bet_won']:
                 if col in df_bet_log.columns:
                      df_bet_log[col] = df_bet_log[col].astype('Int64') # Use nullable integer
                      
            results[name]['bets_dataframe'] = df_bet_log.sort_values(by='event_date').reset_index(drop=True)
            # Optional: Save individual bet logs to CSV
            bet_log_path = os.path.join(output_dir, f"bet_log_{name}.csv")
            try:
                results[name]['bets_dataframe'].to_csv(bet_log_path, index=False)
                log.info(f"Saved detailed bet log for {name} to {bet_log_path}")
            except Exception as e:
                log.error(f"Failed to save bet log for {name} to {bet_log_path}: {e}")
        else:
            results[name]['bets_dataframe'] = pd.DataFrame() # Empty df if no bets
            
    log.info("Backtest engine finished.")
    return results


# --- Strategy Bet Determination Logic ---
def determine_bet_for_strategy(fight_details, strategy_def, odds_type):
    """
    Determines if a bet should be placed based on the strategy definition.
    
    Args:
        fight_details (dict): Dictionary containing pre-calculated fight info.
        strategy_def (dict): The definition of the strategy being evaluated.
        odds_type (str): 'closing' or 'sevenday'.
        
    Returns:
        tuple: (bet_on_side, bet_odds, reason, edge)
               bet_on_side: 'fighter1', 'fighter2', or None
               bet_odds: Decimal odds for the bet, or pd.NA
               reason: String explaining the bet reason
               edge: Decimal edge calculated (model_proba - implied_proba) or pd.NA
    """
    rule = strategy_def.get('rule', 'none')
    params = strategy_def.get('params', {})
    
    # Get odds based on the specified type for this strategy
    f1_odds = fight_details.get(f'f1_odds_{odds_type}', pd.NA)
    f2_odds = fight_details.get(f'f2_odds_{odds_type}', pd.NA)
    
    # --- Baseline Strategies ---
    if rule == 'bet_favourite':
        if pd.notna(f1_odds) and pd.notna(f2_odds):
             if f1_odds < f2_odds:
                 return 'fighter1', f1_odds, "Market Favourite", pd.NA
             elif f2_odds < f1_odds:
                 return 'fighter2', f2_odds, "Market Favourite", pd.NA
        return None, pd.NA, "No clear favourite or odds missing", pd.NA
        
    if rule == 'bet_underdog':
        if pd.notna(f1_odds) and pd.notna(f2_odds):
             if f1_odds > f2_odds:
                 return 'fighter1', f1_odds, "Market Underdog", pd.NA
             elif f2_odds > f1_odds:
                 return 'fighter2', f2_odds, "Market Underdog", pd.NA
        return None, pd.NA, "No clear underdog or odds missing", pd.NA
        
    # --- AI Pick Based Strategies ---
    ai_pick = fight_details['ai_pick_side']
    ai_proba = fight_details['ai_pick_proba']
    
    if ai_pick is None:
        # Most AI strategies require a pick
        if rule not in ['bet_favourite', 'bet_underdog']: # Add other non-AI rules here
             return None, pd.NA, "AI pick indeterminate", pd.NA
             
    # Get the odds corresponding to the AI's pick
    ai_pick_odds = f1_odds if ai_pick == 'fighter1' else f2_odds
    if pd.isna(ai_pick_odds):
         # Odds missing for AI pick, cannot bet if odds are needed
         if rule not in ['bet_favourite', 'bet_underdog']: # Add other non-AI rules here
              return None, pd.NA, f"Odds missing for AI pick {ai_pick}", pd.NA

    if rule == 'ai_pick':
        if ai_pick and pd.notna(ai_pick_odds):
            return ai_pick, ai_pick_odds, f"AI Pick ({ai_pick})", pd.NA
        else:
            # Reason already covered above if odds missing, or if ai_pick is None
             reason = "AI pick indeterminate" if ai_pick is None else f"Odds missing for AI pick {ai_pick}"
             return None, pd.NA, reason, pd.NA 

    if rule == 'ai_pick_confident':
        confidence_threshold = params.get('confidence_threshold', Decimal('0.55'))
        if not isinstance(confidence_threshold, Decimal): confidence_threshold = Decimal(str(confidence_threshold))
        
        if ai_pick and pd.notna(ai_pick_odds) and pd.notna(ai_proba) and ai_proba >= confidence_threshold:
            return ai_pick, ai_pick_odds, f"AI Pick ({ai_pick}, Conf: {ai_proba:.2f} >= {confidence_threshold:.2f})", pd.NA
        elif ai_pick:
            reason = f"AI Pick ({ai_pick}, Conf: {ai_proba:.2f} < {confidence_threshold:.2f})" if pd.notna(ai_proba) else f"AI Pick ({ai_pick}), Proba NA" 
            if pd.isna(ai_pick_odds): reason += " & Odds NA"
            return None, pd.NA, reason, pd.NA
        # ai_pick is None case handled earlier
        
    # --- EV Based Strategies ---
    if rule == 'positive_ev':
        min_edge = params.get('min_edge_threshold', DECIMAL_ZERO)
        if not isinstance(min_edge, Decimal): min_edge = Decimal(str(min_edge))
        
        bet_side, bet_odds, edge = (None, pd.NA, pd.NA)
        # Check F1
        is_pos_f1, edge_f1 = check_positive_ev(fight_details['f1_proba'], f1_odds)
        if is_pos_f1 and edge_f1 is not None and edge_f1 >= min_edge:
            bet_side = 'fighter1'
            bet_odds = f1_odds
            edge = edge_f1
        
        # Check F2 (only if F1 wasn't selected)
        if bet_side is None: 
             is_pos_f2, edge_f2 = check_positive_ev(fight_details['f2_proba'], f2_odds)
             if is_pos_f2 and edge_f2 is not None and edge_f2 >= min_edge:
                 bet_side = 'fighter2'
                 bet_odds = f2_odds
                 edge = edge_f2
                 
        if bet_side:
            return bet_side, bet_odds, f"EV+ (Side: {bet_side}, Edge: {edge:.3f} >= {min_edge:.3f})", edge
        else:
            return None, pd.NA, "No side meets +EV threshold", pd.NA

    # --- Hybrid Strategies --- 
    if rule == 'ai_pick_and_positive_ev':
        min_edge = params.get('min_edge_threshold', DECIMAL_ZERO)
        if not isinstance(min_edge, Decimal): min_edge = Decimal(str(min_edge))
        
        if ai_pick and pd.notna(ai_pick_odds):
            is_pos_ev, edge = check_positive_ev(ai_proba, ai_pick_odds)
            if is_pos_ev and edge is not None and edge >= min_edge:
                return ai_pick, ai_pick_odds, f"AI Pick ({ai_pick}) & EV+ (Edge: {edge:.3f} >= {min_edge:.3f})", edge
            else:
                 reason = f"AI Pick ({ai_pick}) but not EV+ (Edge: {edge:.3f} < {min_edge:.3f})" if pd.notna(edge) else f"AI Pick ({ai_pick}) but EV calc error"
                 return None, pd.NA, reason, edge
        else:
            # AI pick None or odds missing
            reason = f"AI Pick None" if ai_pick is None else f"AI Pick ({ai_pick}) but odds missing"
            return None, pd.NA, reason, pd.NA
            
    # --- Add more rules here ---
    # e.g., ai_pick_favourite, ai_pick_underdog, confidence_and_ev, etc.
    
    # --- Default: No bet if rule not matched ---
    # log.debug(f"Strategy rule '{rule}' not matched or resulted in no bet for fight {fight_details.get('fight_id', 'N/A')}")
    return None, pd.NA, "Rule not matched or no bet condition", pd.NA


# --- Metrics Calculation ---

def calculate_sortino_ratio(daily_returns, target_return=0.0):
    """Calculates the Sortino Ratio from a list of daily returns."""
    returns_array = np.array(daily_returns, dtype=float)
    if len(returns_array) <= 1: return 0.0 # Need > 1 data point

    # Calculate downside returns (returns below the target)
    downside_returns = returns_array[returns_array < target_return]
    
    if len(downside_returns) <= 1:
        # If no downside returns or only one, std dev is 0 or undefined.
        # Sortino is typically considered infinite if mean > target and 0 otherwise in this case.
        mean_return = np.mean(returns_array)
        return float('inf') if mean_return > target_return else 0.0
        
    # Calculate downside deviation (standard deviation of downside returns)
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0: # No variation in downside returns (e.g., all losses were identical)
        mean_return = np.mean(returns_array)
        if mean_return > target_return: return float('inf')
        if mean_return < target_return: return float('-inf') # Or maybe 0, debatable
        return 0.0
        
    # Calculate average return
    average_return = np.mean(returns_array)
    
    # Sortino Ratio = (Average Return - Target Return) / Downside Deviation
    sortino = (average_return - target_return) / downside_deviation
    # Return 0 if calculation leads to NaN/inf (e.g., average_return is inf)
    return sortino if np.isfinite(sortino) else 0.0 

def calculate_summary_metrics(strategy_name, results_data, initial_bankroll):
    """Calculates a comprehensive set of performance metrics for a strategy."""
    log.debug(f"Calculating summary metrics for strategy: {strategy_name}")
    metrics = {
        'strategy_name': strategy_name,
        'total_profit': results_data['total_profit'],
        'final_bankroll': results_data['current_bankroll'],
        'total_wagered': results_data['total_wagered'],
        'bet_count': results_data['bet_count'],
        'win_count': results_data['win_count'],
        'roi_percent': DECIMAL_ZERO,
        'win_rate_percent': DECIMAL_ZERO,
        'profit_factor': DECIMAL_ZERO,
        'sharpe_ratio': 0.0, # Based on event profit volatility
        'sortino_ratio': 0.0, # Based on daily return downside deviation
        'max_drawdown_percent': DECIMAL_ZERO,
        'avg_bet_size': DECIMAL_ZERO,
        'avg_profit_per_bet': DECIMAL_ZERO,
    }

    if metrics['bet_count'] > 0:
        metrics['win_rate_percent'] = (Decimal(metrics['win_count']) / Decimal(metrics['bet_count'])) * DECIMAL_HUNDRED
        metrics['avg_bet_size'] = metrics['total_wagered'] / Decimal(metrics['bet_count'])
        metrics['avg_profit_per_bet'] = metrics['total_profit'] / Decimal(metrics['bet_count'])
    
    if metrics['total_wagered'] > DECIMAL_ZERO:
        metrics['roi_percent'] = (metrics['total_profit'] / metrics['total_wagered']) * DECIMAL_HUNDRED
        
        # Profit Factor = Gross Winnings / Gross Losses
        df_bets = results_data.get('bets_dataframe')
        # Ensure DataFrame exists and profit column is numeric
        if df_bets is not None and not df_bets.empty and pd.api.types.is_numeric_dtype(df_bets['profit']):
            gross_winnings = Decimal(str(df_bets[df_bets['profit'] > 0]['profit'].sum()))
            gross_losses = abs(Decimal(str(df_bets[df_bets['profit'] < 0]['profit'].sum())))
            if gross_losses > DECIMAL_ZERO:
                metrics['profit_factor'] = gross_winnings / gross_losses
            elif gross_winnings > DECIMAL_ZERO:
                 metrics['profit_factor'] = Decimal('inf') # Infinite profit factor if no losses
            # else remains 0
        
    # Sharpe Ratio (using event profits)
    event_profits = results_data['event_profits']
    if len(event_profits) > 1:
        # Convert Decimal profits to float for numpy
        event_profits_float = [float(p) if not pd.isna(p) else 0.0 for p in event_profits]
        mean_event_profit = np.mean(event_profits_float)
        std_dev_event_profit = np.std(event_profits_float)
        
        if std_dev_event_profit > float(SMALL_STD_DEV_THRESHOLD):
            # Assuming Risk-Free Rate = 0; This is a simplified Sharpe based on per-event profit volatility
            sharpe = mean_event_profit / std_dev_event_profit
            metrics['sharpe_ratio'] = sharpe if np.isfinite(sharpe) else 0.0
        else: # Handle zero std dev
             metrics['sharpe_ratio'] = float('inf') if mean_event_profit > 0 else (float('-inf') if mean_event_profit < 0 else 0.0)

    # Sortino Ratio (using daily returns)
    daily_returns = results_data['daily_returns']
    if len(daily_returns) > 1:
         metrics['sortino_ratio'] = calculate_sortino_ratio(daily_returns)
         
    # Max Drawdown
    drawdown_hist = results_data['drawdown_history']
    if drawdown_hist:
        # Drawdown is stored as positive percentage (0 to 1), find the max
        max_dd_fraction = max(dd for _, dd in drawdown_hist)
        metrics['max_drawdown_percent'] = Decimal(str(max_dd_fraction * 100)) # Convert fraction to percentage

    # Quantize currency/percentage metrics for final display/storage
    for key in ['total_profit', 'final_bankroll', 'total_wagered', 'avg_bet_size', 'avg_profit_per_bet']:
         metrics[key] = decimal_quantize(metrics[key])
    for key in ['roi_percent', 'win_rate_percent', 'max_drawdown_percent']:
         # Use 2 decimal places for percentages
         metrics[key] = decimal_quantize(metrics[key], places=Decimal('0.01')) 
    if isinstance(metrics['profit_factor'], Decimal) and metrics['profit_factor'].is_finite():
        metrics['profit_factor'] = decimal_quantize(metrics['profit_factor'], places=Decimal('0.001'))
    # Sharpe and Sortino are ratios, typically kept as floats
    # Ensure they are float, replacing potential Decimal representations
    metrics['sharpe_ratio'] = float(metrics['sharpe_ratio'])
    metrics['sortino_ratio'] = float(metrics['sortino_ratio'])

    return metrics


# --- Analysis & Visualization ---

def plot_bankroll_history(results, output_dir):
    """Plots the bankroll history for all strategies."""
    log.info("Generating bankroll history plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 8))
    has_data_to_plot = False
    
    for name, data in results.items():
        history = data.get('bankroll_history', [])
        if len(history) > 1:
            dates, bankrolls = zip(*history)
            # Convert Decimal bankrolls to float for plotting
            bankrolls_float = [float(b) if isinstance(b, Decimal) else b for b in bankrolls]
            plt.plot(dates, bankrolls_float, label=name, linewidth=1.5)
            has_data_to_plot = True
        else:
            log.warning(f"Not enough bankroll history points to plot for strategy: {name}")
    
    if not has_data_to_plot:
        log.warning("No strategies have enough data for bankroll history plot. Skipping.")
        plt.close()
        return
            
    plt.title('Strategy Bankroll Growth Over Time')
    plt.xlabel('Date')
    plt.ylabel('Bankroll ($)')
    plt.legend(loc='upper left', fontsize='small')
    plt.yscale('linear') # Use linear scale, or 'log' if bankrolls vary widely
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'bankroll_history.png')
    try:
        plt.savefig(plot_path)
        log.info(f"Saved bankroll history plot to {plot_path}")
    except Exception as e:
        log.error(f"Failed to save bankroll history plot: {e}")
    plt.close()

def plot_metric_bars(summary_df, metric_col, title, output_filename, output_dir, lower_is_better=False):
    """Generates a horizontal bar chart for a given metric in the summary DataFrame."""
    log.info(f"Generating bar plot for metric: {metric_col}")
    if metric_col not in summary_df.columns:
        log.warning(f"Metric '{metric_col}' not found in summary DataFrame. Skipping plot: {title}")
        return
        
    # Drop strategies with NaN metric for this plot
    plot_data = summary_df[[metric_col]].dropna().copy()
    if plot_data.empty:
         log.warning(f"No valid data found for metric '{metric_col}'. Skipping plot: {title}")
         return
         
    # Convert metric column to float for plotting
    try:
        # Convert Decimal or other types to float
        plot_data[metric_col] = plot_data[metric_col].apply(lambda x: float(x) if not pd.isna(x) else np.nan).astype(float)
    except (ValueError, TypeError) as e:
         log.warning(f"Could not convert metric '{metric_col}' to numeric float. Skipping plot: {title}. Error: {e}")
         return
            
    # Handle potential infinite values (replace with NaN and drop)
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_data.dropna(subset=[metric_col], inplace=True)
    if plot_data.empty:
         log.warning(f"No finite data found for metric '{metric_col}' after handling inf. Skipping plot: {title}")
         return
         
    sort_ascending = lower_is_better
    plot_data = plot_data.sort_values(by=metric_col, ascending=sort_ascending)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    # Adjust height based on number of strategies
    fig_height = max(6, len(plot_data) * 0.4) 
    plt.figure(figsize=(10, fig_height))
    
    palette = sns.color_palette("coolwarm" if metric_col in ['sharpe_ratio', 'sortino_ratio', 'total_profit', 'roi_percent'] else "viridis", len(plot_data))
    bars = plt.barh(plot_data.index, plot_data[metric_col], color=palette)
    
    # Format xlabel nicely
    x_label = metric_col.replace('_', ' ').title()
    if 'percent' in metric_col: x_label += ' (%)'
    elif 'bankroll' in metric_col or 'profit' in metric_col or 'wagered' in metric_col or 'bet_size' in metric_col: x_label += ' ($)'
    plt.xlabel(x_label) 
    plt.ylabel("Strategy")
    plt.title(title)
    
    # Don't invert axis if lower score is better (e.g., Max Drawdown)
    # if not lower_is_better: 
    #    plt.gca().invert_yaxis()
        
    plt.axvline(0, color='grey', linestyle='--', lw=1) # Line at 0
    plt.tight_layout()

    # Add metric values as labels
    # Calculate position carefully based on min/max values
    values = plot_data[metric_col].astype(float)
    if not values.empty:
        max_abs_val = max(abs(v) for v in values) if not values.empty else 1
        min_val = min(values) if not values.empty else 0
        max_val = max(values) if not values.empty else 0
        
        # Determine label position - try to place outside bar if possible
        value_range = max(max_val - min_val, max_abs_val * 0.1) # Ensure non-zero range
        label_offset = value_range * 0.02 # Offset relative to range
        
        # Adjust plot limits slightly to make space for labels
        current_xlim = plt.xlim()
        padding = value_range * 0.05 # Padding based on range
        new_xlim_min = min(current_xlim[0], min_val - padding)
        new_xlim_max = max(current_xlim[1], max_val + padding)
        # Ensure limits accommodate labels, adjust offset calculation if needed
        label_x_min = min_val - label_offset if min_val < 0 else new_xlim_min
        label_x_max = max_val + label_offset if max_val > 0 else new_xlim_max
        plt.xlim(min(new_xlim_min, label_x_min), max(new_xlim_max, label_x_max))

        for bar in bars:
            width = bar.get_width()
            label_text = f'{width:.2f}' # Default formatting
            if 'percent' in metric_col: label_text += '%'
            
            # Position label: Right of positive bars, Left of negative bars
            x_pos = width + label_offset if width >= 0 else width - label_offset
            ha = 'left' if width >= 0 else 'right'
            
            plt.text(x_pos,
                     bar.get_y() + bar.get_height()/2,
                     label_text, 
                     va='center',
                     ha=ha,
                     fontsize=8)

    plot_path = os.path.join(output_dir, output_filename)
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        log.info(f"Saved {title} plot to {plot_path}")
    except Exception as e:
        log.error(f"Failed to save plot {output_filename}: {e}")
    plt.close()

def plot_scatter(summary_df, x_metric, y_metric, title, output_filename, output_dir):
    """Generates a scatter plot comparing two metrics."""
    log.info(f"Generating scatter plot: {title} ({y_metric} vs {x_metric})")
    if x_metric not in summary_df.columns or y_metric not in summary_df.columns:
        log.warning(f"Metrics '{x_metric}' or '{y_metric}' not found. Skipping scatter plot: {title}")
        return

    plot_data = summary_df[[x_metric, y_metric]].dropna().copy()
    if plot_data.empty:
         log.warning(f"No valid data for scatter plot '{title}'. Skipping.")
         return
         
    # Convert metrics to float for plotting
    try:
        plot_data[x_metric] = plot_data[x_metric].apply(lambda x: float(x) if not pd.isna(x) else np.nan).astype(float)
        plot_data[y_metric] = plot_data[y_metric].apply(lambda x: float(x) if not pd.isna(x) else np.nan).astype(float)
    except (ValueError, TypeError) as e:
         log.warning(f"Could not convert metrics '{x_metric}' or '{y_metric}' to numeric float. Skipping plot: {title}. Error: {e}")
         return
         
    # Handle potential infinite values
    plot_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    plot_data.dropna(subset=[x_metric, y_metric], inplace=True)
    if plot_data.empty:
         log.warning(f"No finite numeric data for scatter plot '{title}'. Skipping.")
         return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, 8))
    
    scatter = sns.scatterplot(
        data=plot_data, 
        x=x_metric, y=y_metric, 
        hue=plot_data.index, # Use strategy name from index for hue 
        s=100, 
        palette="viridis",
        legend='full' # Show legend
    )
    
    plt.title(title)
    x_label = x_metric.replace('_', ' ').title()
    if 'percent' in x_metric: x_label += ' (%)'
    y_label = y_metric.replace('_', ' ').title()
    if 'percent' in y_metric: y_label += ' (%)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(0, color='grey', linestyle='--', lw=1) # Line at Y=0
    plt.axvline(0, color='grey', linestyle='--', lw=1) # Line at X=0
    plt.grid(True, linestyle='--', alpha=0.6)

    # Add labels next to points
    for i in range(plot_data.shape[0]):
        plt.text(x=plot_data[x_metric].iloc[i] * 1.01, # Slight offset
                 y=plot_data[y_metric].iloc[i] * 1.01,
                 s=plot_data.index[i], # Get strategy name from index
                 fontsize=8,
                 ha='left')

    # Move legend outside the plot
    plt.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    plot_path = os.path.join(output_dir, output_filename)
    try:
        plt.savefig(plot_path, bbox_inches='tight')
        log.info(f"Saved scatter plot to {plot_path}")
    except Exception as e:
         log.error(f"Failed to save scatter plot {output_filename}: {e}")
    plt.close()

def plot_calibration_curve(all_results, output_dir):
    """Plots the calibration curve for model predictions across all relevant bets."""
    log.info("Generating calibration curve plot...")
    all_bets = []
    for name, data in all_results.items():
        df_bets = data.get('bets_dataframe')
        # Ensure df exists, is not empty, and has required columns
        if df_bets is not None and not df_bets.empty and all(c in df_bets.columns for c in ['bet_on_side', 'f1_model_proba', 'f2_model_proba', 'bet_won']):
            # Need to reconstruct the probability associated with the *actual bet placed*
            df_bets_copy = df_bets.copy()
            # Ensure probabilities are numeric before using np.where
            df_bets_copy['f1_model_proba'] = pd.to_numeric(df_bets_copy['f1_model_proba'], errors='coerce')
            df_bets_copy['f2_model_proba'] = pd.to_numeric(df_bets_copy['f2_model_proba'], errors='coerce')
            
            df_bets_copy['relevant_proba'] = np.where(
                df_bets_copy['bet_on_side'] == 'fighter1',
                df_bets_copy['f1_model_proba'],
                np.where(df_bets_copy['bet_on_side'] == 'fighter2',
                         df_bets_copy['f2_model_proba'],
                         np.nan # Handles cases where bet_on_side is not f1 or f2
                        )
            )
            # Keep only valid probabilities and outcomes (bet_won should be 0 or 1)
            df_bets_copy['bet_won'] = pd.to_numeric(df_bets_copy['bet_won'], errors='coerce') # Ensure numeric
            relevant_bets = df_bets_copy.dropna(subset=['relevant_proba', 'bet_won'])
            relevant_bets = relevant_bets[relevant_bets['bet_won'].isin([0, 1])]
            
            if not relevant_bets.empty:
                # Only append the necessary columns
                all_bets.append(relevant_bets[['relevant_proba', 'bet_won']])

    if not all_bets:
        log.warning("No valid bets found across all strategies to generate calibration curve.")
        return
        
    # Combine lists of DataFrames into one
    combined_bets = pd.concat(all_bets).drop_duplicates().reset_index(drop=True)
    
    if combined_bets.empty or combined_bets['relevant_proba'].isnull().all() or combined_bets['bet_won'].isnull().all():
         log.warning("Combined bets list is empty or contains only NaNs. Cannot generate calibration curve.")
         return
         
    y_true = combined_bets['bet_won'].astype(int)
    y_prob = combined_bets['relevant_proba']

    # Calculate calibration curve points
    try:
        # Ensure y_prob is within [0, 1] after potential float conversion issues
        y_prob = np.clip(y_prob, 0, 1)
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    except ValueError as e:
        log.error(f"Could not calculate calibration curve. Maybe too few samples or bins? Error: {e}")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='.', label='Model Calibration', markersize=10)
    
    plt.title('Model Calibration Curve (Across All Strategy Bets)')
    plt.xlabel('Mean Predicted Probability (per bin)')
    plt.ylabel('Fraction of Positives (Actual Win Rate per bin)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, 'calibration_curve.png')
    try:
        plt.savefig(plot_path)
        log.info(f"Saved calibration curve plot to {plot_path}")
    except Exception as e:
         log.error(f"Failed to save calibration curve plot: {e}")
    plt.close()

def analyze_and_visualize(all_results, initial_bankroll, output_dir):
    """Calculates summary metrics and generates plots for all strategies."""
    log.info("--- Starting Analysis & Visualization --- ")
    if not all_results:
        log.warning("No backtest results to analyze.")
        return

    summary_metrics_list = []
    for name, data in all_results.items():
        metrics = calculate_summary_metrics(name, data, initial_bankroll)
        summary_metrics_list.append(metrics)
        
    if not summary_metrics_list:
        log.warning("No summary metrics calculated.")
        return
        
    summary_df = pd.DataFrame(summary_metrics_list)
    # Set strategy_name as index for easier plotting lookup
    summary_df = summary_df.set_index('strategy_name', drop=False) 

    # --- Print & Save Summary Table --- 
    print("\n======= ADVANCED BACKTEST SUMMARY METRICS =======")
    # Define column order and formatting for display
    display_cols = [
        'final_bankroll', 'total_profit', 'total_wagered', 'bet_count', 'win_count',
        'win_rate_percent', 'roi_percent', 'profit_factor', 'sharpe_ratio', 
        'sortino_ratio', 'max_drawdown_percent', 'avg_bet_size', 'avg_profit_per_bet'
    ]
    # Select only columns that actually exist in the DataFrame
    display_cols = [col for col in display_cols if col in summary_df.columns]
    summary_to_print = summary_df[display_cols].copy()

    # Formatting for console output (convert Decimals to float/string for printing)
    float_format_map = {
        'final_bankroll': '{:,.2f}', 'total_profit': '{:,.2f}',
        'total_wagered': '{:,.2f}', 'avg_bet_size': '{:,.2f}', 'avg_profit_per_bet': '{:,.2f}',
        'win_rate_percent': '{:.2f}%', 'roi_percent': '{:.2f}%', 'max_drawdown_percent': '{:.2f}%',
        'sharpe_ratio': '{:.3f}', 'sortino_ratio': '{:.3f}',
        'profit_factor': '{:.3f}'
    }
    int_format_map = {'bet_count': '{:,d}', 'win_count': '{:,d}'}

    # Apply formatting carefully
    for col, fmt in float_format_map.items():
        if col in summary_to_print.columns:
            # Handle potential inf values before formatting
            summary_to_print[col] = pd.to_numeric(summary_to_print[col], errors='coerce')
            summary_to_print[col] = summary_to_print[col].apply(lambda x: fmt.format(x) if pd.notna(x) and np.isfinite(x) else ('inf' if x==np.inf else ('-inf' if x==-np.inf else 'NaN')))
    for col, fmt in int_format_map.items():
        if col in summary_to_print.columns:
            summary_to_print[col] = summary_to_print[col].apply(lambda x: fmt.format(x) if pd.notna(x) else 'NaN')

    # Print with adjusted spacing to fit typical console width
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(summary_to_print)
    print("==================================================")

    # Save the raw (unformatted) summary metrics to CSV
    summary_csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    try:
        # Convert Decimal columns to float for CSV saving consistency
        summary_df_csv = summary_df.copy()
        for col in summary_df_csv.select_dtypes(include=['object']): # Find Decimal columns (often stored as object)
            try:
                 # Attempt conversion, coerce errors to NaN
                 summary_df_csv[col] = pd.to_numeric(summary_df_csv[col], errors='coerce')
            except (TypeError, ValueError):
                 pass # Keep as object if conversion fails
        summary_df_csv.to_csv(summary_csv_path)
        log.info(f"Saved summary metrics table to {summary_csv_path}")
    except Exception as e:
        log.error(f"Failed to save summary metrics CSV: {e}")

    # --- Generate Plots --- 
    plot_bankroll_history(all_results, output_dir)
    plot_metric_bars(summary_df, 'roi_percent', 'Return on Investment (ROI) by Strategy', 'roi_by_strategy.png', output_dir)
    plot_metric_bars(summary_df, 'total_profit', 'Total Profit by Strategy', 'profit_by_strategy.png', output_dir)
    plot_metric_bars(summary_df, 'sharpe_ratio', 'Sharpe Ratio (Event Profit Volatility) by Strategy', 'sharpe_ratio_by_strategy.png', output_dir)
    plot_metric_bars(summary_df, 'sortino_ratio', 'Sortino Ratio (Daily Return Downside Deviation) by Strategy', 'sortino_ratio_by_strategy.png', output_dir)
    plot_metric_bars(summary_df, 'max_drawdown_percent', 'Maximum Drawdown by Strategy', 'max_drawdown_by_strategy.png', output_dir, lower_is_better=True)
    plot_scatter(summary_df, 'roi_percent', 'max_drawdown_percent', 'Risk vs. Reward: ROI vs. Max Drawdown', 'roi_vs_mdd.png', output_dir)
    plot_scatter(summary_df, 'win_rate_percent', 'roi_percent', 'Performance: ROI vs. Win Rate', 'roi_vs_winrate.png', output_dir)
    plot_calibration_curve(all_results, output_dir)
    
    log.info("--- Analysis & Visualization Finished --- ")


if __name__ == "__main__":
    log.info("Starting Advanced Profit Backtester")

    try:
        # 1. Load Configuration & Assets
        model_path, data_csv, scaler_path, feats_path, output_dir = load_paths()
        features = load_features_list(feats_path)
        model, scaler = load_assets(model_path, scaler_path)
        
        # Define backtest parameters
        start_date_str = "2024-01-01" # Example start date
        initial_bankroll = Decimal('1000.00')

        # 2. Load Raw Data
        df_raw = load_data(data_csv)

        # 3. Prepare Data
        # This combines scaling, prediction, filtering, and type conversions
        df_prepared = prepare_data_for_backtest(
            df_raw, features, scaler, model, start_date_str
        )

        if df_prepared.empty:
            log.error("Data preparation failed or resulted in empty DataFrame. Exiting.")
            sys.exit(1)
        
        log.info("Data preparation successful.")
        #print("\nPrepared Data Head:")
        #print(df_prepared.head())
        
        # 4. Define Strategies to Test
        # Customize this dictionary extensively!
        strategy_definitions = {
            # --- Baselines ---
            "Baseline_Fav_Closing": {
                "rule": "bet_favourite", "odds_type": "closing", 
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            "Baseline_Dog_Closing": {
                "rule": "bet_underdog", "odds_type": "closing", 
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
             "Baseline_Fav_SevenDay": {
                "rule": "bet_favourite", "odds_type": "sevenday", 
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            "Baseline_Dog_SevenDay": {
                "rule": "bet_underdog", "odds_type": "sevenday", 
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            # --- Fixed Bet Strategies ---
            "Fixed_AI_Pick_Closing": {
                "rule": "ai_pick", "odds_type": "closing",
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            "Fixed_AI_Pick_SevenDay": {
                "rule": "ai_pick", "odds_type": "sevenday",
                "params": {"bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            "Fixed_AI_Pick_EV0_Closing": { # EV > 0%
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {"min_edge_threshold": Decimal("0.00"), "bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
             "Fixed_AI_Pick_EV1_Closing": { # EV > 1%
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {"min_edge_threshold": Decimal("0.01"), "bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            "Fixed_AI_Pick_EV3_Closing": { # EV > 3%
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {"min_edge_threshold": Decimal("0.03"), "bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
             "Fixed_AI_Pick_Conf60_Closing": { # Confidence > 60%
                "rule": "ai_pick_confident", "odds_type": "closing",
                "params": {"confidence_threshold": Decimal("0.60"), "bet_sizing": "fixed", "amount": Decimal("10.00")}
            },
            # --- Percentage Bet Strategies ---
             "Percent1_AI_Pick_Closing": {
                "rule": "ai_pick", "odds_type": "closing",
                "params": {"bet_sizing": "percentage", "percentage": Decimal("1.0")}
            },
            "Percent1_AI_Pick_EV1_Closing": {
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {"min_edge_threshold": Decimal("0.01"), "bet_sizing": "percentage", "percentage": Decimal("1.0")}
            },
            # --- Kelly Bet Strategies ---
            "Kelly_10pct_EV1_Closing": {
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {
                    "min_edge_threshold": Decimal("0.01"),
                    "bet_sizing": "kelly",
                    "kelly_fraction": Decimal("0.1"), 
                    "max_bet_fraction": Decimal("0.05") # Cap bet at 5% of bankroll
                }
            },
            "Kelly_25pct_EV3_Closing": {
                "rule": "ai_pick_and_positive_ev", "odds_type": "closing",
                "params": {
                    "min_edge_threshold": Decimal("0.03"),
                    "bet_sizing": "kelly",
                    "kelly_fraction": Decimal("0.25"), 
                    "max_bet_fraction": Decimal("0.10") # Cap bet at 10% of bankroll
                }
            },
             "Kelly_10pct_EV1_SevenDay": {
                "rule": "ai_pick_and_positive_ev", "odds_type": "sevenday",
                "params": {
                    "min_edge_threshold": Decimal("0.01"),
                    "bet_sizing": "kelly",
                    "kelly_fraction": Decimal("0.1"), 
                    "max_bet_fraction": Decimal("0.05") 
                }
            },
        }

        # 5. Run Backtesting Engine
        log.info(f"Running backtest with {len(strategy_definitions)} strategies...")
        backtest_results = run_backtest_engine(
            df_prepared, 
            strategy_definitions, 
            initial_bankroll, 
            output_dir
        )

        # 6. Analyze and Visualize Results
        if backtest_results:
            analyze_and_visualize(backtest_results, initial_bankroll, output_dir)
        else:
            log.warning("Backtesting returned no results. Skipping analysis.")


    except FileNotFoundError as e:
        log.error(f"File not found: {e}. Please check paths.")
    except ValueError as e:
        log.error(f"Value error during setup or data processing: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)

    log.info("Advanced Profit Backtester Finished.")
