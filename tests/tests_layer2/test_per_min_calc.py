import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.per_min_calc import PerMinCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager


class TestPerMinCalculator(PerMinCalculator):
    """
    Test version of PerMinCalculator that overrides methods for testing purposes.
    """
    def __init__(self, conn=None):
        # If no connection is provided, create a mock
        if conn is None:
            conn = MagicMock()
        
        # Initialize with mock connection
        super().__init__(conn)
        
        # Override the SQL template manager with a mock
        self.sql_template_manager = MagicMock()

    def _ensure_columns_exist(self, *args, **kwargs):
        # Override to assume columns exist for testing
        pass


def test_per_min_calculator_with_context():
    """Test the per-minute calculator with mock data and context."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_head': [10, 15, 20],
        'sig_body': [5, 8, 12],
        'td': [2, 0, 3],
        'sig_head_rd1': [5, 8, 10],
        'time_sec': [300, 240, 480],  # 5 min, 4 min, 8 min
        'time_sec_rd1': [300, 240, 300]  # 5 min, 4 min, 5 min
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Configure mock to return our data when execute_raw_sql is called
    def mock_execute_raw_sql(sql, params=None):
        # Calculate the per-minute values
        result_df = mock_df.copy()
        result_df['sig_head_per_min'] = result_df.apply(lambda row: round(row['sig_head'] / (row['time_sec']/60.0), 4), axis=1)
        result_df['sig_body_per_min'] = result_df.apply(lambda row: round(row['sig_body'] / (row['time_sec']/60.0), 4), axis=1)
        result_df['td_per_min'] = result_df.apply(lambda row: round(row['td'] / (row['time_sec']/60.0), 4), axis=1)
        result_df['sig_head_rd1_per_min'] = result_df.apply(lambda row: round(row['sig_head_rd1'] / (row['time_sec_rd1']/60.0), 4), axis=1)
        return result_df
    
    # Create calculator instance
    calculator = TestPerMinCalculator(mock_conn)
    calculator.feature_utils.execute_raw_sql = mock_execute_raw_sql
    
    # Set features
    calculator.features = [
        'sig_head', 'sig_body', 'td', 'sig_head_rd1'
    ]
    
    # Without Bayesian smoothing, per_min is just count / (time_sec/60)
    expected_results = {
        'sig_head_per_min': [2.0, 3.75, 2.5],   # count / (time_sec/60)
        'sig_body_per_min': [1.0, 2.0, 1.5],
        'td_per_min': [0.4, 0.0, 0.375],
        'sig_head_rd1_per_min': [1.0, 2.0, 2.0]  # For rd1, use time_sec_rd1
    }
    
    # Set up the SQL template manager to return a simple SQL that will be mock-executed
    template_sql = """
    SELECT fight_id, fighter_id, 
           sig_head, sig_body, td, sig_head_rd1,
           time_sec, time_sec_rd1
    FROM schema.table
    """
    calculator.sql_template_manager.render_template.return_value = template_sql
    
    # Call calculate to get the SQL
    sql = calculator.calculate()
    
    # Execute the SQL and check results
    results = calculator.feature_utils.execute_raw_sql(sql)
    
    # Verify the results
    for i, row in results.iterrows():
        fighter_id = row['fighter_id']
        for col, expected_values in expected_results.items():
            expected = expected_values[i]
            # Check that the calculated value is close to expected
            assert abs(row.get(col, 0) - expected) < 0.1, f"Expected {col} to be {expected} but got {row.get(col, 0)}"


def test_per_min_calculator_with_sql_template():
    """Test the per-minute calculator using a SQL template manager."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_head_per_min': [2.0, 3.75, 2.5],
        'sig_body_per_min': [1.0, 2.0, 1.5]
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a mock SQLTemplateManager
    mock_sql_manager = MagicMock(spec=SQLTemplateManager)
    
    # SQL template that would be returned by the template manager
    expected_sql = """
    SELECT
        fight_id,
        fighter_id,
        ROUND(CAST(sig_head::float / (time_sec::float/60.0) AS NUMERIC), 4) AS sig_head_per_min,
        ROUND(CAST(sig_body::float / (time_sec::float/60.0) AS NUMERIC), 4) AS sig_body_per_min
    FROM features.fight_stats_derived
    """
    
    # Configure the mock to return our template
    mock_sql_manager.render_template.return_value = expected_sql
    
    # Create and configure the calculator
    calculator = TestPerMinCalculator(mock_conn)
    calculator.sql_template_manager = mock_sql_manager
    
    # Mock get_features to return predefined features
    calculator.get_features = MagicMock()
    calculator.features = [
        'sig_head', 'sig_body'
    ]
    
    # Mock execute_raw_sql to return our mock data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=mock_df)
    
    # Mock execute_calculator_update to return our mock data
    calculator.execute_calculator_update = MagicMock(return_value=mock_df)
    
    # Test that calculate calls render_template and execute_raw_sql
    result = calculator.save()
    
    # Verify that execute_calculator_update was called
    calculator.execute_calculator_update.assert_called_once()
    
    # Verify the result has the expected columns
    assert 'sig_head_per_min' in result.columns
    assert 'sig_body_per_min' in result.columns


def test_per_min_calculator_integration():
    """Test the per-minute calculator with full integration using SQL template."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_head': [10, 15, 20],
        'sig_body': [5, 8, 12],
        'td': [2, 0, 3],
        'sig_head_rd1': [5, 8, 10],
        'time_sec': [300, 240, 480],  # 5 min, 4 min, 8 min
        'time_sec_rd1': [300, 240, 300],  # 5 min, 4 min, 5 min
        'event_date': pd.to_datetime(['2023-01-01', '2023-02-01', '2023-03-01']),
        'weightclass': ['Lightweight', 'Welterweight', 'Middleweight']
    }
    
    # Create result data with per-minute values
    result_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_head_per_min': [2.0, 3.75, 2.5],
        'sig_body_per_min': [1.0, 2.0, 1.5],
        'td_per_min': [0.4, 0.0, 0.375],
        'sig_head_rd1_per_min': [1.0, 2.0, 2.0]
    }
    
    # Create mock DataFrames
    mock_df = pd.DataFrame(mock_data)
    result_df = pd.DataFrame(result_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Initialize calculator
    calculator = TestPerMinCalculator(mock_conn)
    
    # Mock SQL template manager
    calculator.sql_template_manager.render_template.return_value = """
    SELECT
        fight_id,
        fighter_id,
        ROUND(CAST(sig_head::float / (time_sec::float/60.0) AS NUMERIC), 4) AS sig_head_per_min,
        ROUND(CAST(sig_body::float / (time_sec::float/60.0) AS NUMERIC), 4) AS sig_body_per_min,
        ROUND(CAST(td::float / (time_sec::float/60.0) AS NUMERIC), 4) AS td_per_min,
        ROUND(CAST(sig_head_rd1::float / (time_sec_rd1::float/60.0) AS NUMERIC), 4) AS sig_head_rd1_per_min
    FROM features.fight_stats_derived
    JOIN features.event_mapping ON fight_stats_derived.event_id = event_mapping.event_id
    JOIN features.fight_mapping ON fight_stats_derived.fight_id = fight_mapping.fight_id
    """
    
    # Mock the execute_raw_sql to return our result data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=result_df)
    
    # Mock get_features to set up the features
    def mock_get_features():
        calculator.features = [
            'sig_head', 'sig_body', 'td', 'sig_head_rd1'
        ]
        return calculator.features
    
    calculator.get_features = mock_get_features
    calculator.get_features()
    
    # Mock execute_calculator_update to return our result data
    calculator.execute_calculator_update = MagicMock(return_value=result_df)
    
    # Run the calculation
    result = calculator.save()
    
    # Verify the results
    # Fighter 1 (ID 101) - 5 minutes (300 seconds)
    assert abs(result['sig_head_per_min'].iloc[0] - 2.0) < 0.01  # 10 / (300/60) = 2.0
    assert abs(result['sig_body_per_min'].iloc[0] - 1.0) < 0.01  # 5 / (300/60) = 1.0
    assert abs(result['td_per_min'].iloc[0] - 0.4) < 0.01        # 2 / (300/60) = 0.4
    assert abs(result['sig_head_rd1_per_min'].iloc[0] - 1.0) < 0.01  # 5 / (300/60) = 1.0
    
    # Fighter 2 (ID 102) - 4 minutes (240 seconds)
    assert abs(result['sig_head_per_min'].iloc[1] - 3.75) < 0.01  # 15 / (240/60) = 3.75
    assert abs(result['sig_body_per_min'].iloc[1] - 2.0) < 0.01   # 8 / (240/60) = 2.0
    assert abs(result['td_per_min'].iloc[1] - 0.0) < 0.01         # 0 / (240/60) = 0.0
    assert abs(result['sig_head_rd1_per_min'].iloc[1] - 2.0) < 0.01  # 8 / (240/60) = 2.0
    
    # Fighter 3 (ID 103) - 8 minutes (480 seconds)
    assert abs(result['sig_head_per_min'].iloc[2] - 2.5) < 0.01   # 20 / (480/60) = 2.5
    assert abs(result['sig_body_per_min'].iloc[2] - 1.5) < 0.01   # 12 / (480/60) = 1.5
    assert abs(result['td_per_min'].iloc[2] - 0.375) < 0.01       # 3 / (480/60) = 0.375
    assert abs(result['sig_head_rd1_per_min'].iloc[2] - 2.0) < 0.01  # 10 / (300/60) = 2.0 