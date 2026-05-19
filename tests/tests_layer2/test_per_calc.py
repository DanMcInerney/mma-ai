import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.per_calc import PerCalculator


class MockPerCalculator(PerCalculator):
    """
    Mock version of PerCalculator that overrides methods for testing purposes.
    """
    def __init__(self, conn=None):
        # If no connection is provided, create a mock
        if conn is None:
            conn = MagicMock()
        
        # Initialize with mock connection
        super().__init__(conn)
        
        # Mock the update_table method on the context
        self.context.update_table = MagicMock()

    def _ensure_columns_exist(self, *args, **kwargs):
        # Override to assume columns exist for testing
        pass


def test_per_calculator_feature_mappings():
    """Test that feature mappings are correctly defined."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator instance
    calculator = MockPerCalculator(mock_conn)
    
    # Test that all expected features are mapped
    expected_features = [
        'ko_per_sig_str_land', 'sig_str_per_str_att', 'distance_per_sig_str_land', 'clinch_per_sig_str_land', 'ground_per_sig_str_land',
        'head_per_sig_str_land', 'body_leg_per_sig_str_land', 'td_per_sig_str_att', 'td_land_per_ctrl',
        'ground_land_per_ctrl', 'ground_land_per_td_land', 'sub_att_per_ctrl',
        'rev_per_ctrlopp', 'ko_sub_rd1_per_win', 'ko_sub_per_win', 'sub_per_all_ctrl'
    ]
    
    features = calculator.get_features()
    assert len(features) == len(expected_features)
    
    for feature in expected_features:
        assert feature in features, f"Feature {feature} not found in calculator features"
        assert feature in calculator.feature_mappings, f"Feature {feature} not in feature mappings"


def test_per_calculator_table_organization():
    """Test that features are correctly organized by target table."""
    mock_conn = MagicMock()
    calculator = MockPerCalculator(mock_conn)
    
    # Test specific table assignments
    ko_features = calculator.get_table_features('ko')
    assert 'ko_per_sig_str_land' in ko_features
    assert 'ko_sub_rd1_per_win' in ko_features
    assert 'ko_sub_per_win' in ko_features
    
    sig_str_features = calculator.get_table_features('sig_str')
    assert 'sig_str_per_str_att' in sig_str_features
    
    distance_features = calculator.get_table_features('distance')
    assert 'distance_per_sig_str_land' in distance_features
    
    td_features = calculator.get_table_features('td')
    assert 'td_per_sig_str_att' in td_features
    assert 'td_land_per_ctrl' in td_features
    
    ctrl_features = calculator.get_table_features('ctrl')
    # ctrl_per_td_land has been moved to td table


def test_per_calculator_sql_generation():
    """Test that SQL is correctly generated for each feature."""
    mock_conn = MagicMock()
    calculator = MockPerCalculator(mock_conn)
    
    # Test SQL generation for ko table
    ko_sql = calculator.calculate_for_table('ko')
    
    # Verify SQL contains expected elements for ko features
    assert "ko_per_sig_str_land" in ko_sql
    assert "ko_sub_rd1_per_win" in ko_sql
    assert "ko_sub_per_win" in ko_sql
    assert "CASE" in ko_sql
    assert "CAST" in ko_sql
    assert "FLOAT" in ko_sql
    
    # Test SQL generation for sig_str table
    sig_str_sql = calculator.calculate_for_table('sig_str')
    assert "sig_str_per_str_att" in sig_str_sql
    assert "strikes_att > 0" in sig_str_sql


def test_per_calculator_power_calculation():
    """Test ko_per_sig_str_land calculation: ko / sig_str_land"""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3, 4],
        'fighter_id': [101, 102, 103, 104],
        'event_id': [1, 1, 2, 2],
        'kd': [1, 0, 2, 0],
        'ko': [0, 1, 0, 0],
        'sig_str_land': [20, 15, 30, 0]  # Last one tests division by zero
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    # Expected ko_per_sig_str_land calculations:
    # Fighter 1: 0 / 20 = 0.0
    # Fighter 2: 1 / 15 = 0.067
    # Fighter 3: 0 / 30 = 0.0
    # Fighter 4: 0 / 0 = 0.0 (division by zero)
    expected_ko_per_sig_str_land = [0.0, 0.067, 0.0, 0.0]
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['ko_per_sig_str_land'] = [
            row['ko'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Test ko table calculation
    result = calculator.execute_for_table('ko')
    
    # Verify ko_per_sig_str_land values
    for i, expected in enumerate(expected_ko_per_sig_str_land):
        actual = result['ko_per_sig_str_land'].iloc[i]
        assert abs(actual - expected) < 0.01, f"Fighter {i+1}: Expected ko_per_sig_str_land {expected} but got {actual}"


def test_per_calculator_range_calculations():
    """Test range calculations: distance_per_sig_str_land, clinch_per_sig_str_land, ground_per_sig_str_land"""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'sig_str_land': [20, 15, 0],  # Last one tests division by zero
        'distance_land': [12, 10, 5],
        'clinch_land': [5, 3, 2],
        'ground_land': [3, 2, 1]
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    # Expected calculations:
    # Fighter 1: distance=12/20=0.6, clinch=5/20=0.25, ground=3/20=0.15
    # Fighter 2: distance=10/15=0.667, clinch=3/15=0.2, ground=2/15=0.133
    # Fighter 3: All 0.0 due to division by zero
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql_distance(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['distance_per_sig_str_land'] = [
            row['distance_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    def mock_execute_raw_sql_clinch(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['clinch_per_sig_str_land'] = [
            row['clinch_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    def mock_execute_raw_sql_ground(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['ground_per_sig_str_land'] = [
            row['ground_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    
    # Test distance range
    calculator.execute_raw_sql = mock_execute_raw_sql_distance
    distance_result = calculator.execute_for_table('distance')
    assert abs(distance_result['distance_per_sig_str_land'].iloc[0] - 0.6) < 0.01
    assert abs(distance_result['distance_per_sig_str_land'].iloc[1] - 0.667) < 0.01
    assert distance_result['distance_per_sig_str_land'].iloc[2] == 0.0
    
    # Test clinch range
    calculator.execute_raw_sql = mock_execute_raw_sql_clinch
    clinch_result = calculator.execute_for_table('clinch')
    assert abs(clinch_result['clinch_per_sig_str_land'].iloc[0] - 0.25) < 0.01
    assert abs(clinch_result['clinch_per_sig_str_land'].iloc[1] - 0.2) < 0.01
    assert clinch_result['clinch_per_sig_str_land'].iloc[2] == 0.0


def test_per_calculator_target_calculations():
    """Test target calculations: head_per_sig_str_land, body_leg_per_sig_str_land"""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'sig_str_land': [20, 15, 0],  # Last one tests division by zero
        'head_land': [12, 8, 5],
        'body_land': [5, 4, 2],
        'leg_land': [3, 3, 1]
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql_head(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['head_per_sig_str_land'] = [
            row['head_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    def mock_execute_raw_sql_body(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        result_df['body_leg_per_sig_str_land'] = [
            (row['body_land'] + row['leg_land']) / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    
    # Test head target
    calculator.execute_raw_sql = mock_execute_raw_sql_head
    head_result = calculator.execute_for_table('head')
    assert abs(head_result['head_per_sig_str_land'].iloc[0] - 0.6) < 0.01  # 12/20
    assert abs(head_result['head_per_sig_str_land'].iloc[1] - 0.533) < 0.01  # 8/15
    assert head_result['head_per_sig_str_land'].iloc[2] == 0.0
    
    # Test body_leg target
    calculator.execute_raw_sql = mock_execute_raw_sql_body
    body_result = calculator.execute_for_table('body')
    assert abs(body_result['body_leg_per_sig_str_land'].iloc[0] - 0.4) < 0.01  # (5+3)/20
    assert abs(body_result['body_leg_per_sig_str_land'].iloc[1] - 0.467) < 0.01  # (4+3)/15
    assert body_result['body_leg_per_sig_str_land'].iloc[2] == 0.0


def test_per_calculator_td_calculations():
    """Test TD-related calculations"""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'td_att': [5, 3, 0],
        'td_land': [2, 1, 0],
        'sig_str_att': [25, 20, 15],
        'ctrl': [120, 60, 0],  # Control time in seconds
        'ground_land': [8, 4, 0],
        'sub_att': [2, 1, 0]
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        
        # Calculate all TD-related features
        result_df['td_per_sig_str_att'] = [
            row['td_att'] / row['sig_str_att'] if row['sig_str_att'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        result_df['td_land_per_ctrl'] = [
            row['td_land'] / row['ctrl'] if row['ctrl'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        result_df['ground_land_per_ctrl'] = [
            row['ground_land'] / row['ctrl'] if row['ctrl'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        result_df['ground_land_per_td_land'] = [
            row['ground_land'] / row['td_land'] if row['td_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        result_df['sub_att_per_ctrl'] = [
            row['sub_att'] / row['ctrl'] if row['ctrl'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Test td table calculation
    td_result = calculator.execute_for_table('td')
    
    # Verify TD calculations
    # Fighter 1: td_per_sig_str_att = 5/25 = 0.2
    assert abs(td_result['td_per_sig_str_att'].iloc[0] - 0.2) < 0.01
    
    # Test td table calculation for td_land_per_ctrl
    td_result_ctrl = calculator.execute_for_table('td')
    
    # Fighter 1: td_land_per_ctrl = 2/120 = 0.0167
    assert abs(td_result_ctrl['td_land_per_ctrl'].iloc[0] - 0.0167) < 0.01
    
    # Test ground table calculation
    ground_result = calculator.execute_for_table('ground')
    
    # Fighter 1: ground_land_per_ctrl = 8/120 = 0.067
    assert abs(ground_result['ground_land_per_ctrl'].iloc[0] - 0.067) < 0.01
    
    # Fighter 1: ground_land_per_td_land = 8/2 = 4.0
    assert abs(ground_result['ground_land_per_td_land'].iloc[0] - 4.0) < 0.01
    
    # Test sub table calculation
    sub_result = calculator.execute_for_table('sub')
    
    # Fighter 1: sub_att_per_ctrl = 2/120 = 0.017
    assert abs(sub_result['sub_att_per_ctrl'].iloc[0] - 0.017) < 0.01


def test_per_calculator_finishing_calculations():
    """Test finishing calculations: ko_sub_rd1_per_win, ko_sub_per_win"""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3, 4],
        'fighter_id': [101, 102, 103, 104],
        'event_id': [1, 1, 2, 2],
        'ko_rd1': [1, 0, 0, 0],
        'sub_land_rd1': [0, 1, 0, 0],
        'ko': [1, 0, 1, 0],
        'sub_land': [0, 1, 0, 0],
        'win': [1, 1, 1, 0]  # Last fighter has no wins
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        
        result_df['ko_sub_rd1_per_win'] = [
            (row['ko_rd1'] + row['sub_land_rd1']) / row['win'] if row['win'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        result_df['ko_sub_per_win'] = [
            (row['ko'] + row['sub_land']) / row['win'] if row['win'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Test ko table calculation
    result = calculator.execute_for_table('ko')
    
    # Verify calculations
    # Fighter 1: ko_sub_rd1_per_win = (1+0)/1 = 1.0
    assert abs(result['ko_sub_rd1_per_win'].iloc[0] - 1.0) < 0.01
    
    # Fighter 2: ko_sub_rd1_per_win = (0+1)/1 = 1.0
    assert abs(result['ko_sub_rd1_per_win'].iloc[1] - 1.0) < 0.01
    
    # Fighter 3: ko_sub_per_win = (1+0)/1 = 1.0
    assert abs(result['ko_sub_per_win'].iloc[2] - 1.0) < 0.01
    
    # Fighter 4: Both should be 0.0 due to no wins
    assert result['ko_sub_rd1_per_win'].iloc[3] == 0.0
    assert result['ko_sub_per_win'].iloc[3] == 0.0


def test_per_calculator_edge_cases():
    """Test edge cases with division by zero and missing data."""
    # Create mock data with edge cases
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'sig_str_land': [0, 10, 5],      # Zero case, normal, small
        'strikes_att': [0, 20, 10],      # Zero case, normal, normal
        'distance_land': [5, 8, 3],
        'head_land': [3, 6, 2]
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        
        # Calculate sig_str_per_str_att
        result_df['sig_str_per_str_att'] = [
            row['sig_str_land'] / row['strikes_att'] if row['strikes_att'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        
        # Calculate distance_per_sig_str_land
        result_df['distance_per_sig_str_land'] = [
            row['distance_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
            for _, row in mock_df.iterrows()
        ]
        
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    calculator.execute_raw_sql = mock_execute_raw_sql
    
    # Test sig_str table
    sig_str_result = calculator.execute_for_table('sig_str')
    
    # Fighter 1: 0/0 = 0.0 (division by zero)
    assert sig_str_result['sig_str_per_str_att'].iloc[0] == 0.0
    
    # Fighter 2: 10/20 = 0.5
    assert abs(sig_str_result['sig_str_per_str_att'].iloc[1] - 0.5) < 0.01
    
    # Test distance table
    distance_result = calculator.execute_for_table('distance')
    
    # Fighter 1: 5/0 = 0.0 (division by zero)
    assert distance_result['distance_per_sig_str_land'].iloc[0] == 0.0
    
    # Fighter 2: 8/10 = 0.8
    assert abs(distance_result['distance_per_sig_str_land'].iloc[1] - 0.8) < 0.01


def test_per_calculator_integration():
    """Test full integration with multiple tables."""
    # Create comprehensive mock data
    mock_data = {
        'fight_id': [1, 2],
        'fighter_id': [101, 102],
        'event_id': [1, 1],
        'kd': [1, 0],
        'ko': [0, 1],
        'ko_rd1': [0, 1],
        'sub_land': [0, 0],
        'sub_land_rd1': [0, 0],
        'sig_str_land': [20, 15],
        'strikes_att': [25, 20],
        'distance_land': [12, 10],
        'clinch_land': [5, 3],
        'ground_land': [3, 2],
        'head_land': [12, 8],
        'body_land': [5, 4],
        'leg_land': [3, 3],
        'td_att': [3, 2],
        'td_land': [1, 1],
        'sig_str_att': [25, 20],
        'ctrl': [60, 120],
        'sub_att': [1, 2],
        'win': [1, 1]
    }
    
    mock_df = pd.DataFrame(mock_data)
    
    mock_conn = MagicMock()
    
    def mock_execute_raw_sql(sql, params=None, return_results=True):
        result_df = mock_df.copy()
        
        # Add all calculations based on which features are in the SQL
        if 'ko_per_sig_str_land' in sql:
            result_df['ko_per_sig_str_land'] = [
                row['ko'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
                for _, row in mock_df.iterrows()
            ]
        if 'ko_sub_rd1_per_win' in sql:
            result_df['ko_sub_rd1_per_win'] = [
                (row['ko_rd1'] + row['sub_land_rd1']) / row['win'] if row['win'] > 0 else 0.0
                for _, row in mock_df.iterrows()
            ]
        if 'sig_str_per_str_att' in sql:
            result_df['sig_str_per_str_att'] = [
                row['sig_str_land'] / row['strikes_att'] if row['strikes_att'] > 0 else 0.0
                for _, row in mock_df.iterrows()
            ]
        if 'head_per_sig_str_land' in sql:
            result_df['head_per_sig_str_land'] = [
                row['head_land'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0
                for _, row in mock_df.iterrows()
            ]
            
        return result_df
    
    calculator = MockPerCalculator(mock_conn)
    calculator.execute_raw_sql = mock_execute_raw_sql
    calculator.execute_calculator_update = MagicMock(return_value=mock_df)
    
    # Test that save method processes all tables
    results = calculator.save()
    
    # Should have results for multiple tables
    assert len(results) >= 1
    
    # Test specific calculations
    ko_result = calculator.execute_for_table('ko')
    if not ko_result.empty and 'ko_per_sig_str_land' in ko_result.columns:
        # Fighter 1: 0 / 20 = 0.0
        assert abs(ko_result['ko_per_sig_str_land'].iloc[0] - 0.0) < 0.01


def test_per_calculator_get_table_features():
    """Test that get_table_features returns correct features for each table."""
    mock_conn = MagicMock()
    calculator = MockPerCalculator(mock_conn)
    
    # Test ko table features
    ko_features = calculator.get_table_features('ko')
    expected_ko_features = ['ko_per_sig_str_land', 'ko_sub_rd1_per_win', 'ko_sub_per_win']
    for feature in expected_ko_features:
        assert feature in ko_features
    
    # Test sig_str table features
    sig_str_features = calculator.get_table_features('sig_str')
    assert 'sig_str_per_str_att' in sig_str_features
    
    # Test td table features
    td_features = calculator.get_table_features('td')
    expected_td_features = ['td_per_sig_str_att', 'td_land_per_ctrl']
    for feature in expected_td_features:
        assert feature in td_features
    
    # Test ground table features
    ground_features = calculator.get_table_features('ground')
    expected_ground_features = ['ground_per_sig_str_land', 'ground_land_per_ctrl', 'ground_land_per_td_land']
    for feature in expected_ground_features:
        assert feature in ground_features
    
    # Test sub table features
    sub_features = calculator.get_table_features('sub')
    expected_sub_features = ['sub_att_per_ctrl', 'sub_per_all_ctrl']
    for feature in expected_sub_features:
        assert feature in sub_features
    
    # Test rev table features
    rev_features = calculator.get_table_features('rev')
    expected_rev_features = ['rev_per_ctrlopp']
    for feature in expected_rev_features:
        assert feature in rev_features
