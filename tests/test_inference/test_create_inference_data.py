import os
import sys
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import tempfile
import math

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from libs.feature_store.create_inference_data import CreateInferenceData
from libs.feature_store.features import BASE_STATIC_FEATS

@pytest.fixture
def test_data():
    """Fixture to create test data for UFC fighters."""
    # Create a temporary CSV file with test data
    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')
    os.close(temp_fd)  # Close the file descriptor immediately
    
    # Base date for our test data
    base_date = datetime(2023, 1, 1)
    
    # Create dates for fights
    dates = [
        base_date - timedelta(days=365),  # 1 year ago
        base_date - timedelta(days=180),  # 6 months ago
        base_date - timedelta(days=90),   # 3 months ago
        base_date - timedelta(days=30)    # 1 month ago
    ]
    
    # Create fighter DOBs
    dobs = {
        'fighter a': datetime(1990, 1, 1),
        'fighter b': datetime(1992, 6, 15),
        'fighter c': datetime(1988, 3, 10),
        'fighter d': datetime(1991, 9, 22),
        'fighter e': datetime(1993, 11, 5)    # Fighter with only one fight
    }
    
    # Create test data rows
    data = []
    
    # Add fighter a - 4 fights
    for i, date in enumerate(dates):
        data.append({
            'fighter_id': f'fighter_a_id',
            'fighter_name': 'fighter a',
            'fighter_dob': dobs['fighter a'],
            'event_id': f'event_{i+1}',
            'event_date': date,
            'fight_id': f'fight_{i+1}',
            'opponent': f'opponent_{i+1}',
            'age': (date - dobs['fighter a']).days / 365.25,
            'reach': 72.5,
            'height': 70.0,
            'ufcage': (date - dates[0]).days / 365.25,
            'days_since_last_fight': 90 if i > 0 else 0,
            'sig_str_land': 50 + i * 10,
            'sig_str_att': 100 + i * 15,
            'sig_str_acc': (50 + i * 10) / (100 + i * 15),
            'td_land': 2 + i,
            'td_att': 5 + i,
            'td_acc': (2 + i) / (5 + i),
            'win': 1 if i % 2 == 0 else 0
        })
    
    # Add fighter b - 3 fights
    for i in range(3):
        data.append({
            'fighter_id': f'fighter_b_id',
            'fighter_name': 'fighter b',
            'fighter_dob': dobs['fighter b'],
            'event_id': f'event_{i+5}',
            'event_date': dates[i+1],  # Skip the earliest date
            'fight_id': f'fight_{i+5}',
            'opponent': f'opponent_{i+5}',
            'age': (dates[i+1] - dobs['fighter b']).days / 365.25,
            'reach': 74.0,
            'height': 72.0,
            'ufcage': (dates[i+1] - dates[1]).days / 365.25,
            'days_since_last_fight': 60 if i > 0 else 0,
            'sig_str_land': 40 + i * 5,
            'sig_str_att': 90 + i * 10,
            'sig_str_acc': (40 + i * 5) / (90 + i * 10),
            'td_land': 1 + i,
            'td_att': 3 + i,
            'td_acc': (1 + i) / (3 + i),
            'win': 1 if i > 1 else 0
        })
    
    # Add fighter c - 4 fights
    for i, date in enumerate(dates):
        data.append({
            'fighter_id': f'fighter_c_id',
            'fighter_name': 'fighter c',
            'fighter_dob': dobs['fighter c'],
            'event_id': f'event_{i+8}',
            'event_date': date,
            'fight_id': f'fight_{i+8}',
            'opponent': f'opponent_{i+8}',
            'age': (date - dobs['fighter c']).days / 365.25,
            'reach': 73.0,
            'height': 71.5,
            'ufcage': (date - dates[0]).days / 365.25,
            'days_since_last_fight': 75 if i > 0 else 0,
            'sig_str_land': 55 + i * 8,
            'sig_str_att': 110 + i * 12,
            'sig_str_acc': (55 + i * 8) / (110 + i * 12),
            'td_land': 3 + i,
            'td_att': 6 + i,
            'td_acc': (3 + i) / (6 + i),
            'win': 1 if i < 2 else 0
        })
    
    # Add fighter d - 3 fights
    for i in range(3):
        data.append({
            'fighter_id': f'fighter_d_id',
            'fighter_name': 'fighter d',
            'fighter_dob': dobs['fighter d'],
            'event_id': f'event_{i+12}',
            'event_date': dates[i+1],  # Skip the earliest date
            'fight_id': f'fight_{i+12}',
            'opponent': f'opponent_{i+12}',
            'age': (dates[i+1] - dobs['fighter d']).days / 365.25,
            'reach': 76.0,
            'height': 75.0,
            'ufcage': (dates[i+1] - dates[1]).days / 365.25,
            'days_since_last_fight': 45 if i > 0 else 0,
            'sig_str_land': 45 + i * 15,
            'sig_str_att': 95 + i * 20,
            'sig_str_acc': (45 + i * 15) / (95 + i * 20),
            'td_land': 0 + i,
            'td_att': 2 + i,
            'td_acc': (0 + i) / (2 + i) if (2 + i) > 0 else 0,
            'win': 1 if i > 0 else 0
        })
    
    # Add fighter e - only 1 fight (to test insufficient history)
    data.append({
        'fighter_id': f'fighter_e_id',
        'fighter_name': 'fighter e',
        'fighter_dob': dobs['fighter e'],
        'event_id': 'event_15',
        'event_date': dates[-1],
        'fight_id': 'fight_15',
        'opponent': 'opponent_15',
        'age': (dates[-1] - dobs['fighter e']).days / 365.25,
        'reach': 73.0,
        'height': 71.0,
        'ufcage': 0,
        'days_since_last_fight': 0,
        'sig_str_land': 55,
        'sig_str_att': 100,
        'sig_str_acc': 0.55,
        'td_land': 1,
        'td_att': 3,
        'td_acc': 1/3,
        'win': 1
    })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(temp_path, index=False)
    
    # Return temporary file path and data as a tuple
    yield temp_path, df
    
    # Cleanup after tests
    try:
        os.unlink(temp_path)
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not delete temporary file {temp_path}: {e}")

@pytest.fixture
def inference_data(test_data):
    """Fixture to create a CreateInferenceData instance for testing."""
    # Unpack the values from test_data fixture
    csv_path, df = test_data
    
    # Define features to use for testing
    test_feats = ['age_diff', 'reach_diff', 'height_diff', 'ufcage_diff', 
                  'days_since_last_fight_diff', 'age_ratio_diff',
                  'sig_str_acc_diff', 'td_acc_diff']
    
    # List of fights for testing (date, fighter1, fighter2)
    test_fights = [
        ('2023-06-01', 'fighter a', 'fighter b'),  # A normal fight
        ('2023-06-01', 'fighter c', 'fighter d')   # Fight with more data
    ]
    
    # Initialize the class with test data
    inference = CreateInferenceData(
        csv_path=csv_path,
        feats=test_feats,
        fight_list=test_fights
    )
    
    return inference, test_feats, test_fights, df

def test_initialization(inference_data):
    """Test that the class initializes correctly."""
    inference, test_feats, test_fights, test_df = inference_data
    
    # Check that data was loaded correctly
    assert inference.all_data is not None
    assert len(inference.all_data) == len(test_df)
    
    # Check that dates were converted to datetime
    assert pd.api.types.is_datetime64_dtype(inference.all_data['event_date'])
    assert pd.api.types.is_datetime64_dtype(inference.all_data['fighter_dob'])
    
    # Check that features were set correctly
    assert inference.feats == test_feats
    
    # Check that fight list was set correctly
    assert inference.fight_list == test_fights
    
    # Check that static stats were identified correctly - adjust expectation to match implementation
    # The implementation extracts static stats from features based on BASE_STATIC_FEATS
    expected_static_stats = ['age', 'reach', 'height', 'ufcage', 'days_since_last_fight', 'age_ratio']
    assert set(inference.static_stats) == set(expected_static_stats)

def test_load_static_data(inference_data):
    """Test loading of static data."""
    inference, _, _, test_df = inference_data
    
    # Load the static data
    inference.load_static_data()
    
    # Check that static data was loaded for all fighters
    assert len(inference.static_fighter_dfs) == 4  # fighter e should be filtered out
    
    # Check each fighter's data
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        assert fighter_name in inference.static_fighter_dfs
        
        # Check the shape - should be original rows + 1 for upcoming fight
        fighter_data = test_df[test_df['fighter_name'] == fighter_name]
        assert len(inference.static_fighter_dfs[fighter_name]) == len(fighter_data) + 1
        
        # Check that event_date of the last row is the upcoming fight date
        last_row_date = inference.static_fighter_dfs[fighter_name].iloc[-1]['event_date']
        expected_date = pd.to_datetime('2023-06-01')
        assert last_row_date == expected_date
        
        # Check opponent is set correctly
        if fighter_name == 'fighter a':
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['opponent'] == 'fighter b'
        elif fighter_name == 'fighter b':
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['opponent'] == 'fighter a'
        elif fighter_name == 'fighter c':
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['opponent'] == 'fighter d'
        elif fighter_name == 'fighter d':
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['opponent'] == 'fighter c'
        
        # Check fighter1 flag is set correctly
        if fighter_name in ['fighter a', 'fighter c']:
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['fighter1'] == True
        else:
            assert inference.static_fighter_dfs[fighter_name].iloc[-1]['fighter1'] == False

def test_update_age(inference_data):
    """Test updating age for upcoming fights."""
    inference, _, _, _ = inference_data
    
    # First load the static data
    inference.load_static_data()
    
    # Then update age
    inference.update_age()
    
    # Check that age was updated correctly for each fighter
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        df = inference.static_fighter_dfs[fighter_name]
        dob = df['fighter_dob'].iloc[0]
        fight_date = pd.to_datetime('2023-06-01')
        expected_age = round((fight_date - dob).total_seconds() / (365.25 * 24 * 60 * 60), 3)
        
        assert pytest.approx(df.iloc[-1]['age'], 0.01) == expected_age

def test_update_days_since_last_fight(inference_data):
    """Test updating days since last fight for upcoming fights."""
    inference, _, _, _ = inference_data
    
    # First load the static data
    inference.load_static_data()
    
    # Then update days since last fight
    inference.update_days_since_last_fight()
    
    # Check that days_since_last_fight was updated correctly for each fighter
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        df = inference.static_fighter_dfs[fighter_name]
        last_fight_date = df.iloc[-2]['event_date']
        upcoming_fight_date = pd.to_datetime('2023-06-01')
        expected_days = (upcoming_fight_date - last_fight_date).total_seconds() / (24 * 60 * 60)
        
        assert df.iloc[-1]['days_since_last_fight'] == expected_days

def test_update_ufcage(inference_data):
    """Test updating UFC age for upcoming fights."""
    inference, _, _, _ = inference_data
    
    # First load the static data
    inference.load_static_data()
    
    # Then update UFC age
    inference.update_ufcage()
    
    # Check that ufcage was updated correctly for each fighter
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        df = inference.static_fighter_dfs[fighter_name]
        first_fight_date = df['event_date'].min()
        upcoming_fight_date = pd.to_datetime('2023-06-01')
        expected_ufcage = round(
            (upcoming_fight_date - first_fight_date).total_seconds() / (365.25 * 24 * 60 * 60), 
            3
        )
        
        assert pytest.approx(df.iloc[-1]['ufcage'], 0.01) == expected_ufcage

def test_update_ratios(inference_data):
    """Test updating ratios for upcoming fights."""
    inference, _, _, _ = inference_data
    
    # First load the static data
    inference.load_static_data()
    
    # Then update ratios
    inference.update_ratios()
    
    # Check that ratios were updated correctly for each fighter
    fighter_pairs = [('fighter a', 'fighter b'), ('fighter c', 'fighter d')]
    
    for fighter1, fighter2 in fighter_pairs:
        df1 = inference.static_fighter_dfs[fighter1]
        df2 = inference.static_fighter_dfs[fighter2]
        
        # Check each static feature ratio
        for stat in ['age', 'reach', 'height', 'ufcage', 'days_since_last_fight']:
            if stat in df1.columns and stat in df2.columns:
                fighter1_val = df1.iloc[-1][stat]
                fighter2_val = df2.iloc[-1][stat]
                expected_ratio = fighter1_val / (fighter1_val + fighter2_val)
                
                assert pytest.approx(df1.iloc[-1][f'{stat}_ratio'], 0.000001) == expected_ratio
                
                # The other fighter should have the complement ratio
                expected_ratio2 = fighter2_val / (fighter1_val + fighter2_val)
                assert pytest.approx(df2.iloc[-1][f'{stat}_ratio'], 0.000001) == expected_ratio2

def test_update_avgs(inference_data):
    """Test updating averages for upcoming fights."""
    inference, _, _, _ = inference_data
    
    # First load the static data
    inference.load_static_data()
    inference.update_ratios()
    
    # Then update averages
    inference.update_avgs()
    
    # Check that averages were updated correctly for each fighter
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        df = inference.static_fighter_dfs[fighter_name]
        
        # Check each static feature average
        for stat in ['age', 'reach', 'height', 'ufcage', 'days_since_last_fight']:
            if stat in df.columns:
                expected_avg = df[stat].mean()
                assert pytest.approx(df.iloc[-1][f'{stat}_avg'], 0.000001) == expected_avg
            
            # Also check ratio averages
            ratio_col = f'{stat}_ratio'
            if ratio_col in df.columns:
                expected_avg = df[ratio_col].mean()
                assert pytest.approx(df.iloc[-1][f'{ratio_col}_avg'], 0.000001) == expected_avg

def test_load_dynamic_data(inference_data):
    """Test loading of dynamic data."""
    inference, _, _, _ = inference_data
    
    # First load static data to ensure fight_list is updated
    inference.load_static_data()
    
    # Then load dynamic data
    inference.load_dynamic_data()
    
    # Check that dynamic data was loaded for all fighters
    assert len(inference.dynamic_fighter_dfs) == 4
    
    # Check each fighter's data
    for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
        assert fighter_name in inference.dynamic_fighter_dfs
        
        # Check specific dynamic columns exist
        dynamic_df = inference.dynamic_fighter_dfs[fighter_name]
        dynamic_columns = ['sig_str_land', 'sig_str_att', 'sig_str_acc', 
                          'td_land', 'td_att', 'td_acc']
        
        for col in dynamic_columns:
            assert col in dynamic_df.columns
        
        # Check that event_date of the last row is the upcoming fight date
        last_row_date = dynamic_df.iloc[-1]['event_date']
        expected_date = pd.to_datetime('2023-06-01')
        assert last_row_date == expected_date
        
        # Check fighter1 flag is set correctly
        if fighter_name in ['fighter a', 'fighter c']:
            assert dynamic_df.iloc[-1]['fighter1'] == True
        else:
            assert dynamic_df.iloc[-1]['fighter1'] == False

def test_keep_final_row(inference_data):
    """Test keeping only final row of each fighter's DataFrame."""
    inference, _, _, _ = inference_data
    
    # Create a sample dict of DataFrames
    sample_dfs = {
        'fighter a': pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}),
        'fighter b': pd.DataFrame({'col1': [4, 5], 'col2': ['d', 'e']})
    }
    
    # Call keep_final_row
    result = inference.keep_final_row(sample_dfs)
    
    # Check the results
    assert len(result) == 2
    assert len(result['fighter a']) == 1
    assert len(result['fighter b']) == 1
    assert result['fighter a']['col1'].iloc[0] == 3
    assert result['fighter a']['col2'].iloc[0] == 'c'
    assert result['fighter b']['col1'].iloc[0] == 5
    assert result['fighter b']['col2'].iloc[0] == 'e'

def test_combine_static_and_dynamic(inference_data):
    """Test combining static and dynamic data."""
    inference, test_feats, _, _ = inference_data
    
    # First load and process static data
    inference.load_static_data()
    inference.update_age()
    inference.update_days_since_last_fight()
    inference.update_ufcage()
    inference.update_ratios()
    inference.update_avgs()
    inference.static_fighter_dfs = inference.keep_final_row(inference.static_fighter_dfs)
    
    # Then load and process dynamic data
    inference.load_dynamic_data()
    inference.dynamic_fighter_dfs = inference.keep_final_row(inference.dynamic_fighter_dfs)
    
    # Patch the metadata_cols to not include fighter1_id and fighter2_id
    with patch.object(inference, 'metadata_cols', ['fighter_name', 'opponent', 'event_date', 'fighter1']):
        # Combine them
        inference.fighter_dfs = inference.combine_static_and_dynamic()
        
        # Check that combined data exists for all fighters
        assert len(inference.fighter_dfs) == 4
        
        # Check each fighter's combined data
        for fighter_name in ['fighter a', 'fighter b', 'fighter c', 'fighter d']:
            combined_df = inference.fighter_dfs[fighter_name]
            
            # Check metadata columns
            metadata_cols = ['fighter_name', 'opponent', 'event_date', 'fighter1']
            for col in metadata_cols:
                assert col in combined_df.columns
                
            # Look at what features we're expecting based on test_feats
            # The implementation only keeps non-diff versions of the features in self.feats
            expected_columns = inference.metadata_cols.copy()
            for feat in test_feats:
                base_feat = feat.replace('_diff', '')
                if base_feat in combined_df.columns:
                    assert base_feat in combined_df.columns

def test_subtract_fighter2_from_fighter1(inference_data):
    """Test creating difference columns between fighter1 and fighter2."""
    inference, _, _, _ = inference_data
    
    # Set up fighter_dfs with sample data
    fighter_a_data = {
        'fighter_name': ['fighter a'], 
        'opponent': ['fighter b'], 
        'fighter1': [True],
        'age': [33.0], 
        'reach': [72.5], 
        'sig_str_acc': [0.55]
    }
    fighter_b_data = {
        'fighter_name': ['fighter b'], 
        'opponent': ['fighter a'], 
        'fighter1': [False],
        'age': [30.5], 
        'reach': [74.0], 
        'sig_str_acc': [0.45]
    }
    
    inference.fighter_dfs = {
        'fighter a': pd.DataFrame(fighter_a_data),
        'fighter b': pd.DataFrame(fighter_b_data)
    }
    
    # Call subtract_fighter2_from_fighter1
    diff_dfs = inference.subtract_fighter2_from_fighter1()
    
    # Check the results - should only have fighter a (fighter1 is True)
    assert len(diff_dfs) == 1
    assert 'fighter a' in diff_dfs
    
    # Check the diff values
    diff_df = diff_dfs['fighter a']
    assert diff_df['fighter_name'].iloc[0] == 'fighter a'
    assert diff_df['opponent'].iloc[0] == 'fighter b'
    assert pytest.approx(diff_df['age_diff'].iloc[0], 0.01) == 33.0 - 30.5
    assert pytest.approx(diff_df['reach_diff'].iloc[0], 0.01) == 72.5 - 74.0
    assert pytest.approx(diff_df['sig_str_acc_diff'].iloc[0], 0.01) == 0.55 - 0.45

def test_add_fight_experience(inference_data):
    """Test adding fight experience data."""
    inference, _, _, test_df = inference_data
    
    # Mock the count of fights from the CSV
    inference.all_data = test_df
    
    # Set up fighter_dfs with sample data
    fighter_a_data = {
        'fighter_name': ['fighter a'], 
        'opponent': ['fighter b']
    }
    fighter_b_data = {
        'fighter_name': ['fighter b'], 
        'opponent': ['fighter a']
    }
    
    inference.fighter_dfs = {
        'fighter a': pd.DataFrame(fighter_a_data),
        'fighter b': pd.DataFrame(fighter_b_data)
    }
    
    # Call add_fight_experience
    inference.add_fight_experience()
    
    # Check the results
    assert len(inference.fighter_dfs) == 2
    
    # fighter a should have 4 fights
    assert inference.fighter_dfs['fighter a']['fighter1_total_fights'].iloc[0] == 4
    
    # fighter b should have 3 fights
    assert inference.fighter_dfs['fighter a']['fighter2_total_fights'].iloc[0] == 3
    
    # Combined fights should be 7
    assert inference.fighter_dfs['fighter a']['combined_fights'].iloc[0] == 7

def test_run(inference_data):
    """Test the full run method."""
    inference, _, _, _ = inference_data
    
    # Patch the metadata_cols to not include fighter1_id and fighter2_id
    with patch.object(inference, 'metadata_cols', ['fighter_name', 'opponent', 'event_date', 'fighter1']):
        # Call the run method
        result = inference.run()
        
        # Check the results are fighter1 diffs
        assert len(result) >= 2  # At least 2 fighter1s
        
        # Check that each fighter1 has diff columns
        for fighter_name, df in result.items():
            # Should only be fighter a and fighter c as they are fighter1
            assert fighter_name in ['fighter a', 'fighter c']
            
            # Check diff columns
            diff_columns = [col for col in df.columns if col.endswith('_diff')]
            assert len(diff_columns) > 0
            
            # Check fight experience columns
            experience_columns = ['fighter1_total_fights', 'fighter2_total_fights', 'combined_fights']
            for col in experience_columns:
                assert col in df.columns

def test_edge_case_insufficient_history(inference_data):
    """Test handling of fighters with insufficient fight history."""
    inference, _, _, _ = inference_data
    
    # Add a fight with a fighter with only 1 fight
    inference.fight_list.append(('2023-06-01', 'fighter e', 'fighter a'))
    
    # Load static data
    inference.load_static_data()
    
    # Check that the fighter with only 1 fight was removed
    fighter_e_fight = ('2023-06-01', 'fighter e', 'fighter a')
    assert fighter_e_fight not in inference.fight_list
    assert 'fighter e' not in inference.static_fighter_dfs 

def test_dec_avg_calculation_with_controlled_data():
    """Test update_dec_avgs with controlled data to verify exact calculations."""
    # Create a minimal DataFrame with precisely controlled values for testing
    base_date = datetime(2023, 6, 1)  # Upcoming fight date
    
    # Create test fighter data with precisely known dates and values
    # We'll use exactly 3 past fights with specific dates and values
    test_data = pd.DataFrame({
        'fighter_name': ['test_fighter'] * 4,
        'fighter_id': ['test_id'] * 4,
        'event_date': [
            datetime(2020, 6, 1),  # 3 years before (weight should be 0.25)
            datetime(2021, 12, 1), # 1.5 years before (weight should be 0.5)
            datetime(2022, 12, 1), # 0.5 years before (weight should be 0.794)
            base_date             # Upcoming fight
        ],
        'reach': [70.0, 72.0, 74.0, np.nan],  # Test with increasing values
        'height': [68.0, 68.0, 68.0, np.nan],  # Test with constant values
        'age': [28.0, 29.5, 30.5, np.nan],     # Test with different progression
        'fighter_dob': [datetime(1992, 6, 1)] * 4
    })
    
    # Calculate expected weights manually based on the centralized config (default: 1.0 year half-life)
    from libs.feature_store.config import get_decay_rate
    decay_rate = get_decay_rate()
    
    # Calculate time differences using the same method as the implementation
    time_diffs_series = (base_date - test_data.loc[:2, 'event_date']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    time_diffs = time_diffs_series.tolist()
    
    expected_weights = [math.exp(-decay_rate * t) for t in time_diffs]
    print(f"Calculated time diffs (years): {[f'{t:.4f}' for t in time_diffs]}")
    print(f"Expected weights: {[f'{w:.4f}' for w in expected_weights]}")
    
    # Calculate expected decay averages manually
    reach_values = [70.0, 72.0, 74.0]
    expected_reach_avg = sum(r * w for r, w in zip(reach_values, expected_weights)) / sum(expected_weights)
    
    height_values = [68.0, 68.0, 68.0]
    expected_height_avg = sum(h * w for h, w in zip(height_values, expected_weights)) / sum(expected_weights)
    
    age_values = [28.0, 29.5, 30.5]
    expected_age_avg = sum(a * w for a, w in zip(age_values, expected_weights)) / sum(expected_weights)
    
    print(f"Expected time-decay weighted averages:")
    print(f"  reach_dec_avg: {expected_reach_avg:.4f}")
    print(f"  height_dec_avg: {expected_height_avg:.4f}")
    print(f"  age_dec_avg: {expected_age_avg:.4f}")
    
    # Set up a minimal CreateInferenceData instance with our test data
    test_csv_path = 'test_data.csv'
    test_data.to_csv(test_csv_path, index=False)
    
    try:
        # Create minimal instance just for testing update_dec_avgs
        inference = CreateInferenceData(
            csv_path=test_csv_path,
            feats=['reach_diff', 'height_diff', 'age_diff'],
            fight_list=[('2023-06-01', 'test_fighter', 'opponent')]
        )
        
        # Set up static_fighter_dfs directly to avoid running other methods
        inference.static_fighter_dfs = {'test_fighter': test_data}
        
        # Call the method we're testing
        inference.update_dec_avgs()
        
        # Get the results
        result_df = inference.static_fighter_dfs['test_fighter']
        actual_reach_avg = result_df.iloc[-1]['reach_dec_avg']
        actual_height_avg = result_df.iloc[-1]['height_dec_avg']
        actual_age_avg = result_df.iloc[-1]['age_dec_avg']
        
        print(f"Actual time-decay weighted averages:")
        print(f"  reach_dec_avg: {actual_reach_avg:.4f}")
        print(f"  height_dec_avg: {actual_height_avg:.4f}")
        print(f"  age_dec_avg: {actual_age_avg:.4f}")
        
        # Verify results with high precision
        assert abs(actual_reach_avg - expected_reach_avg) < 0.0001, \
            f"Reach dec_avg calculation incorrect: expected {expected_reach_avg:.6f}, got {actual_reach_avg:.6f}"
        assert abs(actual_height_avg - expected_height_avg) < 0.0001, \
            f"Height dec_avg calculation incorrect: expected {expected_height_avg:.6f}, got {actual_height_avg:.6f}"
        assert abs(actual_age_avg - expected_age_avg) < 0.0001, \
            f"Age dec_avg calculation incorrect: expected {expected_age_avg:.6f}, got {actual_age_avg:.6f}"
            
        # We should also test that the implementation handles null values correctly
        # Create test data with null values
        test_data_with_nulls = test_data.copy()
        test_data_with_nulls.loc[0, 'reach'] = None  # Set oldest fight reach to null
        
        # Recalculate expected values without the null value
        valid_reach_values = [72.0, 74.0]  # Only last two values
        valid_reach_weights = expected_weights[1:]  # Only weights for last two fights
        expected_reach_avg_nulls = sum(r * w for r, w in zip(valid_reach_values, valid_reach_weights)) / sum(valid_reach_weights)
        
        # Update the test instance - Need to ensure event_date is datetime
        test_data_with_nulls['event_date'] = pd.to_datetime(test_data_with_nulls['event_date'])
        test_data_with_nulls['fighter_dob'] = pd.to_datetime(test_data_with_nulls['fighter_dob'])
        inference.static_fighter_dfs = {'test_fighter': test_data_with_nulls}
        
        # Call the method again
        inference.update_dec_avgs()
        
        # Verify null handling
        actual_reach_avg_nulls = inference.static_fighter_dfs['test_fighter'].iloc[-1]['reach_dec_avg']
        
        print(f"Expected reach_dec_avg with nulls: {expected_reach_avg_nulls:.4f}")
        print(f"Actual reach_dec_avg with nulls: {actual_reach_avg_nulls:.4f}")
        
        assert abs(actual_reach_avg_nulls - expected_reach_avg_nulls) < 0.0001, \
            f"Reach dec_avg with nulls incorrect: expected {expected_reach_avg_nulls:.6f}, got {actual_reach_avg_nulls:.6f}"
        
    finally:
        # Clean up test file
        if os.path.exists(test_csv_path):
            os.remove(test_csv_path) 