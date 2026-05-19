import pytest
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP
from unittest.mock import patch, MagicMock
import joblib
from autogluon.tabular import TabularPredictor # Import for mocking

# Import functions from the profit script
# Assuming profit.py is in the root or accessible via PYTHONPATH
# Adjust the import path if necessary
from libs.modeling.profit import (
    load_data, 
    prepare_features, 
    scale_features, 
    make_predictions, 
    create_results_dataframe,
    convert_implied_to_decimal_odds,
    decimal_quantize,
    calculate_implied_probability,
    calculate_bet_outcome,
    get_ai_pick_details,
    get_odds_details,
    check_positive_ev,
    evaluate_single_bet_strategies,
    evaluate_parlay_strategies,
    # We will mock load_assets and load_model directly usually
    # backtest_strategies, create_profit_df are integration-level, harder to unit test
)

# --- Fixtures --- 

@pytest.fixture
def sample_raw_df():
    """Provides a sample DataFrame similar to what's loaded from training_data.csv"""
    data = {
        'fight_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
        'fighter1_id': [101, 101, 103, 103, 101, 101, 105, 105, 103, 103, 107, 107],
        'fighter2_id': [102, 102, 104, 104, 103, 103, 106, 106, 105, 105, 108, 108],
        'fighter1_name': ['Fighter A', 'Fighter A', 'Fighter C', 'Fighter C', 'Fighter A', 'Fighter A', 'Fighter E', 'Fighter E', 'Fighter C', 'Fighter C', 'Fighter G', 'Fighter G'],
        'fighter2_name': ['Fighter B', 'Fighter B', 'Fighter D', 'Fighter D', 'Fighter C', 'Fighter C', 'Fighter F', 'Fighter F', 'Fighter E', 'Fighter E', 'Fighter H', 'Fighter H'],
        'event_date': pd.to_datetime(['2023-01-15', '2023-01-15', '2023-02-20', '2023-02-20', '2023-03-10', '2023-03-10', '2023-04-05', '2023-04-05', '2023-05-15', '2023-05-15', '2023-06-20', '2023-06-20']),
        'feature1_diff': np.random.rand(12),
        'feature2_diff': np.random.rand(12) * 10,
        'f1_ip_closing_odds': [0.6, 0.6, 0.4, 0.4, 0.7, 0.7, 0.5, 0.5, 0.3, 0.3, 0.8, 0.8],
        'f2_ip_closing_odds': [0.4, 0.4, 0.6, 0.6, 0.3, 0.3, 0.5, 0.5, 0.7, 0.7, 0.2, 0.2],
        'f1_sevenday_ip_opening_odds': [0.65, 0.65, 0.45, 0.45, 0.75, 0.75, 0.55, 0.55, 0.35, 0.35, 0.78, 0.78],
        'f2_sevenday_ip_opening_odds': [0.35, 0.35, 0.55, 0.55, 0.25, 0.25, 0.45, 0.45, 0.65, 0.65, 0.22, 0.22],
        'y_true': [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0] # Fighter 1 wins fights 1, 3, 5
    }
    # Simulate the duplicated rows structure if needed, or clean it
    df = pd.DataFrame(data)
    # In reality, training_data.csv seems to have one row per fight after cleaning? Let's assume that structure for the test input.
    df_cleaned = df.drop_duplicates(subset=['fight_id']).reset_index(drop=True)
    # Add some NaNs strategically for testing prepare_features
    df_cleaned.loc[1, 'f1_ip_closing_odds'] = np.nan # Fight 2 has NaN closing odds for F1
    df_cleaned.loc[3, 'feature1_diff'] = np.nan # Fight 4 has NaN feature
    return df_cleaned

@pytest.fixture
def sample_predictions_df(sample_raw_df): # Depend on the cleaned raw df fixture
    """Provides sample prediction probabilities aligned with sample_raw_df"""
    # Use the index from the cleaned sample_raw_df that would pass prepare_features
    # Assume fights 1, 3, 5, 6 pass experience/NaN checks for this test
    eligible_indices = sample_raw_df.index[[0, 2, 4, 5]] # Corresponds to fight_id 1, 3, 5, 6
    data = {
        0: [0.4, 0.65, 0.2, 0.3], # Probability F2 wins
        1: [0.6, 0.35, 0.8, 0.7]  # Probability F1 wins
    }
    return pd.DataFrame(data, index=eligible_indices)

@pytest.fixture
def mock_scaler():
    """Provides a mock scaler object."""
    scaler = MagicMock(spec=joblib.load) # Mock the loaded object type if known, e.g., StandardScaler
    # Mock the transform method
    scaler.transform = MagicMock(side_effect=lambda x: x * 0.9 if isinstance(x, pd.DataFrame) else x) # Simple scaling mock
    return scaler

@pytest.fixture
def mock_model():
    """Provides a mock AutoGluon predictor object."""
    model = MagicMock(spec=TabularPredictor)
    # Define a side effect for predict_proba based on input index or simple logic
    def predict_proba_side_effect(df):
        # Simple mock: return fixed probabilities based on index
        index = df.index
        proba_data = {
            0: [0.4 if i % 2 == 0 else 0.7 for i in index], # P(F2 wins)
            1: [0.6 if i % 2 == 0 else 0.3 for i in index]  # P(F1 wins)
        }
        return pd.DataFrame(proba_data, index=index)
        
    model.predict_proba = MagicMock(side_effect=predict_proba_side_effect)
    return model

# --- Test Cases --- 

# --- Helper Function Tests ---

@pytest.mark.parametrize("implied_proba, expected_odds", [
    (0.75, Decimal("1.333333333333333333333333333")),
    (0.5, Decimal("2.0")),
    (0.2, Decimal("5.0")),
    (0.9, Decimal("1.111111111111111111111111111")),
    (0.1, Decimal("10.0")),
    (0, pd.NA),          # Invalid probability
    (1, pd.NA),          # Invalid probability
    (1.1, pd.NA),        # Invalid probability
    (-0.1, pd.NA),       # Invalid probability
    (np.nan, pd.NA),
    (pd.NA, pd.NA),
])
def test_convert_implied_to_decimal_odds(implied_proba, expected_odds):
    result = convert_implied_to_decimal_odds(implied_proba)
    if pd.isna(expected_odds):
        assert pd.isna(result)
    else:
        assert isinstance(result, Decimal)
        # Use approx for potential float precision issues if inputs weren't Decimal
        assert result == expected_odds 

@pytest.mark.parametrize("value, expected", [
    (Decimal("10.123"), Decimal("10.12")),
    (Decimal("10.125"), Decimal("10.13")),
    (Decimal("10.129"), Decimal("10.13")),
    (Decimal("10.00"), Decimal("10.00")),
    (10.123, Decimal("10.12")), # Test with float input
    (np.nan, pd.NA),
    (pd.NA, pd.NA),
])
def test_decimal_quantize(value, expected):
    result = decimal_quantize(value)
    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected

@pytest.mark.parametrize("decimal_odds, expected_proba", [
    (Decimal("1.5"), Decimal(1) / Decimal("1.5")),
    (Decimal("2.0"), Decimal("0.5")),
    (Decimal("10.0"), Decimal("0.1")),
    (Decimal("1.01"), Decimal(1) / Decimal("1.01")),
    (1.5, Decimal(1) / Decimal("1.5")), # Float input
    (Decimal("1.0"), Decimal(0)),     # Invalid odds
    (Decimal("0.9"), Decimal(0)),     # Invalid odds
    (pd.NA, Decimal(0)),
    (np.nan, Decimal(0)), 
])
def test_calculate_implied_probability(decimal_odds, expected_proba):
    result = calculate_implied_probability(decimal_odds)
    # Need tolerance for float conversions
    assert result == expected_proba

@pytest.mark.parametrize("bet_amount, decimal_odds, did_win, expected_profit", [
    (Decimal(10), Decimal("2.5"), True, Decimal("15.00")),
    (Decimal(10), Decimal("1.8"), True, Decimal("8.00")),
    (Decimal(100), Decimal("3.0"), True, Decimal("200.00")),
    (Decimal(10), Decimal("2.5"), False, Decimal("-10.00")),
    (Decimal(100), Decimal("1.5"), False, Decimal("-100.00")),
    (Decimal(0), Decimal("2.0"), True, Decimal("0.00")),
    (Decimal(10), pd.NA, True, Decimal(0)),          # NA odds
    (pd.NA, Decimal("2.0"), True, Decimal(0)),          # NA amount
    (Decimal(10), Decimal("2.0"), pd.NA, Decimal(0)), # NA win status
])
def test_calculate_bet_outcome(bet_amount, decimal_odds, did_win, expected_profit):
    result = calculate_bet_outcome(bet_amount, decimal_odds, did_win)
    assert result == expected_profit

@pytest.mark.parametrize("proba1, proba2, odds1_c, odds2_c, odds1_s, odds2_s, expected", [
    # F1 picked
    (Decimal("0.6"), Decimal("0.4"), Decimal("1.8"), Decimal("2.2"), Decimal("1.7"), Decimal("2.3"), ('fighter1', Decimal("0.6"), Decimal("1.8"), Decimal("1.7"))),
    # F2 picked
    (Decimal("0.3"), Decimal("0.7"), Decimal("3.0"), Decimal("1.4"), Decimal("3.1"), Decimal("1.3"), ('fighter2', Decimal("0.7"), Decimal("1.4"), Decimal("1.3"))),
    # Equal proba (defaults to F1)
    (Decimal("0.5"), Decimal("0.5"), Decimal("2.0"), Decimal("2.0"), Decimal("1.9"), Decimal("2.1"), ('fighter1', Decimal("0.5"), Decimal("2.0"), Decimal("1.9"))),
    # NA proba
    (pd.NA, Decimal("0.5"), Decimal("2.0"), Decimal("2.0"), Decimal("1.9"), Decimal("2.1"), (None, None, None, None)),
    # NA odds (should still return pick info)
    (Decimal("0.6"), Decimal("0.4"), pd.NA, pd.NA, pd.NA, pd.NA, ('fighter1', Decimal("0.6"), pd.NA, pd.NA)),
])
def test_get_ai_pick_details(proba1, proba2, odds1_c, odds2_c, odds1_s, odds2_s, expected):
    row_data = {
        'f1_y_proba': proba1,
        'f2_y_proba': proba2,
        'f1_dec_odds_closing': odds1_c,
        'f2_dec_odds_closing': odds2_c,
        'f1_dec_odds_sevenday': odds1_s,
        'f2_dec_odds_sevenday': odds2_s,
    }
    row = pd.Series(row_data)
    result = get_ai_pick_details(row)
    
    # Handle pd.NA comparison
    assert result[0] == expected[0]
    if expected[1] is not None: assert result[1] == expected[1]
    else: assert result[1] is None
    if pd.isna(expected[2]): assert pd.isna(result[2])
    else: assert result[2] == expected[2]
    if pd.isna(expected[3]): assert pd.isna(result[3])
    else: assert result[3] == expected[3]

@pytest.mark.parametrize("odds1_c, odds2_c, odds1_s, odds2_s, expected", [
    # F1 Fav Closing, F2 Fav Sevenday
    (Decimal("1.8"), Decimal("2.2"), Decimal("2.1"), Decimal("1.9"), ('fighter1', Decimal("1.8"), 'fighter2', Decimal("2.2"), 'fighter2', Decimal("1.9"), 'fighter1', Decimal("2.1"))),
    # F2 Fav Both
    (Decimal("2.1"), Decimal("1.9"), Decimal("2.0"), Decimal("1.95"), ('fighter2', Decimal("1.9"), 'fighter1', Decimal("2.1"), 'fighter2', Decimal("1.95"), 'fighter1', Decimal("2.0"))),
    # Equal Closing Odds, F1 Fav Sevenday
    (Decimal("2.0"), Decimal("2.0"), Decimal("1.9"), Decimal("2.1"), (None, None, None, None, 'fighter1', Decimal("1.9"), 'fighter2', Decimal("2.1"))),
    # NA Closing Odds, F2 Fav Sevenday
    (pd.NA, pd.NA, Decimal("2.2"), Decimal("1.8"), (None, None, None, None, 'fighter2', Decimal("1.8"), 'fighter1', Decimal("2.2"))),
    # Invalid Odds
    (Decimal("1.0"), Decimal("5.0"), Decimal("1.8"), Decimal("2.2"), (None, None, None, None, 'fighter1', Decimal("1.8"), 'fighter2', Decimal("2.2"))),
])
def test_get_odds_details(odds1_c, odds2_c, odds1_s, odds2_s, expected):
    row_data = {
        'f1_dec_odds_closing': odds1_c,
        'f2_dec_odds_closing': odds2_c,
        'f1_dec_odds_sevenday': odds1_s,
        'f2_dec_odds_sevenday': odds2_s,
    }
    row = pd.Series(row_data)
    result = get_odds_details(row)
    # Compare tuple elements, handling Nones
    for i in range(8):
        if expected[i] is None:
            assert result[i] is None
        elif pd.isna(expected[i]):
             assert pd.isna(result[i])
        else:
            assert result[i] == expected[i]

@pytest.mark.parametrize("ai_proba, decimal_odds, expected_ev", [
    (Decimal("0.6"), Decimal("1.5"), False), # Corrected: 0.6 is NOT > (1/1.5), expected False
    (Decimal("0.7"), Decimal("1.5"), True),  # 0.7 > 0.666
    (Decimal("0.5"), Decimal("2.0"), False), # 0.5 == 1/2.0
    (Decimal("0.5"), Decimal("2.1"), True),  # 0.5 > 1/2.1 (0.476)
    (Decimal("0.4"), Decimal("2.0"), False), # 0.4 < 0.5
    (Decimal("0.1"), Decimal("10.0"), False),# 0.1 == 0.1
    (Decimal("0.11"), Decimal("10.0"), True), # 0.11 > 0.1
    (pd.NA, Decimal("2.0"), False),
    (Decimal("0.5"), pd.NA, False),
    (Decimal("0.5"), Decimal("1.0"), False), # Invalid odds
])
def test_check_positive_ev(ai_proba, decimal_odds, expected_ev):
    result = check_positive_ev(ai_proba, decimal_odds)
    assert result == expected_ev

# --- Data Processing Function Tests --- 

# Mocking pd.read_csv for load_data
@patch('pandas.read_csv')
def test_load_data_success(mock_read_csv):
    """Test loading data successfully."""
    expected_df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    mock_read_csv.return_value = expected_df
    
    # Use a dummy path, as read_csv is mocked
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = True
        df = load_data("dummy_path.csv")
        pd.testing.assert_frame_equal(df, expected_df)
        mock_read_csv.assert_called_once_with("dummy_path.csv")

@patch('os.path.exists')
def test_load_data_file_not_found(mock_exists):
    """Test loading data when file does not exist."""
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_path.csv")

def test_prepare_features(sample_raw_df):
    """Test the feature preparation logic including experience and NaN filters."""
    # Define features to select (ensure they exist in sample_raw_df)
    features_list = ['feature1_diff', 'feature2_diff']
    
    # Expected outcome based on sample_raw_df:
    # Fight 1 (A vs B): F1=0, F2=0 -> Excluded by experience
    # Fight 2 (C vs D): F1=0, F2=0 -> Excluded by experience
    # Fight 3 (A vs C): F1=1, F2=1 -> Excluded by experience
    # Fight 4 (E vs F): F1=0, F2=0 -> Excluded by experience
    # Fight 5 (C vs E): F1=2 (fights 2,3), F2=1 (fight 4) -> Excluded by experience (F2)
    # Fight 6 (G vs H): F1=0, F2=0 -> Excluded by experience
    
    # Modify sample_raw_df to have fights that *should* pass
    # Need more history for the fighters to pass the >= 2 prior fights filter.
    extra_history = pd.DataFrame({
        'fight_id': [10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15],
        'fighter1_id': [101]*2 + [102]*2 + [103]*2 + [104]*2 + [105]*2 + [106]*2, # A, B, C, D, E, F
        'fighter2_id': [901]*2 + [902]*2 + [903]*2 + [904]*2 + [905]*2 + [906]*2, # Opponents
        'fighter1_name': ['Fighter A']*2 + ['Fighter B']*2 + ['Fighter C']*2 + ['Fighter D']*2 + ['Fighter E']*2 + ['Fighter F']*2, 
        'fighter2_name': ['Opponent 1']*2 + ['Opponent 2']*2 + ['Opponent 3']*2 + ['Opponent 4']*2 + ['Opponent 5']*2 + ['Opponent 6']*2,
        'event_date': pd.to_datetime(['2022-01-01']*2 + ['2022-02-01']*2 + ['2022-03-01']*2 + ['2022-04-01']*2 + ['2022-05-01']*2 + ['2022-06-01']*2),
        'feature1_diff': np.random.rand(12),
        'feature2_diff': np.random.rand(12) * 10,
        'f1_ip_closing_odds': [0.5]*12, 'f2_ip_closing_odds': [0.5]*12,
        'f1_sevenday_ip_opening_odds': [0.5]*12, 'f2_sevenday_ip_opening_odds': [0.5]*12,
        'y_true': [1]*12
    })
    # Add even more history for Fighter C
    extra_history_c = pd.DataFrame({
        'fight_id': [16], 'fighter1_id': [103], 'fighter2_id': [907],
        'fighter1_name': ['Fighter C'], 'fighter2_name': ['Opponent 7'],
        'event_date': pd.to_datetime(['2022-07-01']), 
         'feature1_diff': [0.5], 'feature2_diff': [5.0], 
         'f1_ip_closing_odds': [0.5], 'f2_ip_closing_odds': [0.5],
         'f1_sevenday_ip_opening_odds': [0.5], 'f2_sevenday_ip_opening_odds': [0.5],
         'y_true': [1]
    })
    # Add more history for Fighter E
    extra_history_e = pd.DataFrame({
        'fight_id': [17], 'fighter1_id': [105], 'fighter2_id': [908],
        'fighter1_name': ['Fighter E'], 'fighter2_name': ['Opponent 8'],
        'event_date': pd.to_datetime(['2022-08-01']), 
         'feature1_diff': [0.5], 'feature2_diff': [5.0], 
         'f1_ip_closing_odds': [0.5], 'f2_ip_closing_odds': [0.5],
         'f1_sevenday_ip_opening_odds': [0.5], 'f2_sevenday_ip_opening_odds': [0.5],
         'y_true': [1]
    })

    df_with_history = pd.concat([extra_history, extra_history_c, extra_history_e, sample_raw_df], ignore_index=True)
    df_with_history = df_with_history.drop_duplicates(subset=['fight_id', 'fighter1_id']).reset_index(drop=True) # Ensure one row per fighter-fight initially if needed, or handle duplicates
    # Re-sort after concat
    df_with_history = df_with_history.sort_values(by='event_date').reset_index(drop=True)
    
    # Now recalculate expected outcome:
    # Fight 1 (A vs B, 2023-01-15): F1=1, F2=1 -> Excluded
    # Fight 2 (C vs D, 2023-02-20): F1=2 (fights 12, 16), F2=1 -> Excluded (F2)
    # Fight 3 (A vs C, 2023-03-10): F1=2 (fights 10, 1), F2=3 (fights 12, 16, 2) -> Included (Index corresponding to Fight 3 in df_with_history needs finding)
    # Fight 4 (E vs F, 2023-04-05): F1=2 (fights 14, 17), F2=1 -> Excluded (F2)
    # Fight 5 (C vs E, 2023-05-15): F1=4 (fights 12,16,2,3), F2=3 (fights 14, 17, 4) -> Included
    # Fight 6 (G vs H, 2023-06-20): F1=0, F2=0 -> Excluded
    
    # Fight 3 index in df_with_history: df_with_history[df_with_history['fight_id'] == 3].index[0]
    # Fight 5 index in df_with_history: df_with_history[df_with_history['fight_id'] == 5].index[0]

    # Find original indices corresponding to fight_id 3 and 5 in the sorted df_with_history
    original_index_fight_3 = df_with_history[df_with_history['fight_id'] == 3].index[0]
    original_index_fight_5 = df_with_history[df_with_history['fight_id'] == 5].index[0]
    
    # Fight 4 had NaN feature, Fight 2 had NaN odds -> these are filtered out by NaNs if they pass experience
    # Fight 3 has NaN odds after the merge in sample_raw_df -> this would be filtered by NaN check IF it passed experience first.
    # Let's modify Fight 3 odds to be non-NaN for this specific test path
    df_with_history.loc[original_index_fight_3, 'f1_ip_closing_odds'] = 0.7
    df_with_history.loc[original_index_fight_3, 'f2_ip_closing_odds'] = 0.3
    df_with_history.loc[original_index_fight_3, 'f1_sevenday_ip_opening_odds'] = 0.75
    df_with_history.loc[original_index_fight_3, 'f2_sevenday_ip_opening_odds'] = 0.25
    
    # Fight 5 should pass all checks. Find its features.
    expected_features = df_with_history.loc[[original_index_fight_3, original_index_fight_5], features_list]
    expected_indices = expected_features.index # Get the index from the filtered data

    prepared_df = prepare_features(df_with_history, features_list)
    
    assert not prepared_df.empty
    assert prepared_df.columns.tolist() == features_list
    pd.testing.assert_frame_equal(prepared_df, expected_features, check_dtype=False) # Check content
    pd.testing.assert_index_equal(prepared_df.index, expected_indices) # Check index is preserved

def test_prepare_features_empty_input():
    assert prepare_features(pd.DataFrame(), ['feat1']).empty

def test_prepare_features_no_eligible_fights(sample_raw_df): # Use the original fixture
    # This fixture naturally has no fights passing the experience criteria
    assert prepare_features(sample_raw_df, ['feature1_diff']).empty

def test_scale_features(mock_scaler):
    """Test scaling features with a mock scaler."""
    df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}, index=[10, 20, 30])
    expected_df = pd.DataFrame({'feature1': [0.9, 1.8, 2.7], 'feature2': [3.6, 4.5, 5.4]}, index=[10, 20, 30])
    
    scaled_df = scale_features(df, mock_scaler)
    
    mock_scaler.transform.assert_called_once()
    pd.testing.assert_frame_equal(scaled_df, expected_df)
    pd.testing.assert_index_equal(scaled_df.index, df.index) # Check index preservation

def test_make_predictions(mock_model):
    """Test making predictions with a mock model."""
    predict_df = pd.DataFrame({'feature1': [1, 2]}, index=[5, 15])
    # Expected based on mock_model side effect
    expected_proba = pd.DataFrame({0: [0.7, 0.7], 1: [0.3, 0.3]}, index=[5, 15])
    
    y_pred_proba = make_predictions(mock_model, predict_df)
    
    mock_model.predict_proba.assert_called_once()
    # Allow for comparison even if predict_df was passed
    pd.testing.assert_frame_equal(y_pred_proba, expected_proba)

def test_create_results_dataframe(sample_raw_df, sample_predictions_df):
    """Test creating the final results DataFrame."""
    # sample_predictions_df index: [0, 2, 4, 5] (fight_id 1, 3, 5, 6)
    # sample_raw_df needs to be filtered to the same common index
    common_index = sample_raw_df.index.intersection(sample_predictions_df.index)
    df_orig_subset = sample_raw_df.loc[common_index]
    
    results_df = create_results_dataframe(sample_raw_df, sample_predictions_df)
    
    assert len(results_df) == len(sample_predictions_df)
    assert results_df.index.equals(sample_predictions_df.index)
    assert 'fighter1_name' in results_df.columns
    assert 'fighter2_name' in results_df.columns
    assert 'event_date' in results_df.columns
    assert 'f1_y_proba' in results_df.columns
    assert 'f2_y_proba' in results_df.columns
    assert 'y_true' in results_df.columns
    assert 'f1_ip_closing_odds' in results_df.columns
    assert 'f2_ip_closing_odds' in results_df.columns
    assert 'f1_sevenday_ip_opening_odds' in results_df.columns
    assert 'f2_sevenday_ip_opening_odds' in results_df.columns
    
    # Check values for a specific row (e.g., the first one, index 0, fight_id 1)
    expected_f1_proba = sample_predictions_df.loc[0, 1]
    assert results_df.loc[0, 'f1_y_proba'] == expected_f1_proba
    assert results_df.loc[0, 'y_true'] == sample_raw_df.loc[0, 'y_true']
    assert results_df.loc[0, 'f1_ip_closing_odds'] == sample_raw_df.loc[0, 'f1_ip_closing_odds']

# --- Strategy Evaluation Tests (Simplified Example) ---
# Testing these fully requires more complex fixtures representing event slices
# and careful mocking of the helper functions if needed.

@pytest.fixture
def sample_event_df():
    """ Creates a sample DataFrame slice for a single event with pre-calculated Decimal odds. """
    data = {
        'fighter1_name': ['Fighter A', 'Fighter C'],
        'fighter2_name': ['Fighter B', 'Fighter D'],
        'event_date': pd.to_datetime(['2024-01-01', '2024-01-01']),
        'f1_y_proba': [Decimal("0.6"), Decimal("0.3")],
        'f2_y_proba': [Decimal("0.4"), Decimal("0.7")],
        'y_true': [1, 0], # F1 wins first fight, F2 wins second
        # Closing Decimal Odds
        'f1_dec_odds_closing': [Decimal("1.8"), Decimal("3.0")], 
        'f2_dec_odds_closing': [Decimal("2.2"), Decimal("1.4")],
        # Seven-day Decimal Odds
        'f1_dec_odds_sevenday': [Decimal("1.7"), Decimal("3.1")],
        'f2_dec_odds_sevenday': [Decimal("2.3"), Decimal("1.3")],
    }
    return pd.DataFrame(data)

def test_evaluate_single_bet_ai_fav_only_closing(sample_event_df):
    base_strategy = 'ai_picked_favorite'
    odds_type = 'closing'
    bet_amount = Decimal(10)
    
    # Expected behavior:
    # Fight 1: AI picks F1 (0.6 > 0.4). F1 is Fav (1.8 < 2.2). Bet on F1 @ 1.8. F1 wins. Profit = 10 * (1.8 - 1) = 8
    # Fight 2: AI picks F2 (0.7 > 0.3). F2 is Fav (1.4 < 3.0). Bet on F2 @ 1.4. F2 wins. Profit = 10 * (1.4 - 1) = 4
    # Total: Wagered=20, Profit=12
    
    results = evaluate_single_bet_strategies(sample_event_df, base_strategy, odds_type, bet_amount)
    
    assert results['wagered'] == Decimal("20.00")
    assert results['profit'] == Decimal("12.00")
    assert len(results['bets']) == 2
    assert results['bets'][0]['bet_on'] == 'fighter1'
    assert results['bets'][0]['odds'] == Decimal("1.8")
    assert results['bets'][0]['won'] == True
    assert results['bets'][0]['profit'] == Decimal("8.00")
    assert results['bets'][1]['bet_on'] == 'fighter2'
    assert results['bets'][1]['odds'] == Decimal("1.4")
    assert results['bets'][1]['won'] == True
    assert results['bets'][1]['profit'] == Decimal("4.00")

def test_evaluate_single_bet_ai_ev_only_sevenday(sample_event_df):
    base_strategy = 'ai_picked_positive_ev'
    odds_type = 'sevenday'
    bet_amount = Decimal(10)

    # Expected behavior (using sevenday odds):
    # Fight 1: AI picks F1 (0.6). Odds F1 = 1.7. Implied = 1/1.7 = 0.588. 0.6 > 0.588 -> +EV. Bet on F1 @ 1.7. F1 wins. Profit = 10 * (1.7-1) = 7
    # Fight 2: AI picks F2 (0.7). Odds F2 = 1.3. Implied = 1/1.3 = 0.769. 0.7 < 0.769 -> Not +EV. No Bet.
    # Total: Wagered=10, Profit=7
    
    results = evaluate_single_bet_strategies(sample_event_df, base_strategy, odds_type, bet_amount)

    assert results['wagered'] == Decimal("10.00")
    assert results['profit'] == Decimal("7.00")
    assert len(results['bets']) == 1
    assert results['bets'][0]['bet_on'] == 'fighter1'
    assert results['bets'][0]['odds'] == Decimal("1.7")
    assert results['bets'][0]['won'] == True
    assert results['bets'][0]['profit'] == Decimal("7.00")

# Example for Parlay - requires more setup if mocking is complex
def test_evaluate_parlay_strategies_2_legs_ai_ev_closing(sample_event_df):
    base_strategy = 'parlay_2_ai_ev'
    odds_type = 'closing'
    bet_amount = Decimal(10)
    
    # Expected Legs (Closing Odds):
    # Fight 1: AI picks F1 (0.6). Odds = 1.8. Implied = 1/1.8 = 0.555. 0.6 > 0.555 -> +EV. Leg: F1 @ 1.8 (Win)
    # Fight 2: AI picks F2 (0.7). Odds = 1.4. Implied = 1/1.4 = 0.714. 0.7 < 0.714 -> Not +EV.
    # Only 1 leg available, cannot form 2-leg parlay.
    
    results = evaluate_parlay_strategies(sample_event_df, base_strategy, odds_type, bet_amount)
    
    assert results['wagered'] == Decimal(0)
    assert results['profit'] == Decimal(0)
    assert len(results['bets']) == 0

# Add more tests for other strategies, odds types, edge cases (NAs, limits), etc.
# Particularly for parlay strategies with enough valid legs.

# Note: Testing backtest_strategies directly is complex due to its iterative nature
# and reliance on the evaluation functions. It's better tested via integration tests
# or by focusing unit tests on its sub-components (data prep, loop logic if separable). 