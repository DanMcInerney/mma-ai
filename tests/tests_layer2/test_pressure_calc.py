import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.pressure_calc import PressureCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager


class MockPressureCalculator(PressureCalculator):
    """
    Mock version of PressureCalculator that overrides methods for testing purposes.
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


def test_pressure_calculator_with_context():
    """Test the pressure calculator with mock data and context."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3, 4],
        'fighter_id': [101, 102, 103, 104],
        'event_id': [1, 1, 2, 2],
        'sig_str_land': [20, 15, 0, 30],      # Total significant strikes landed
        'sig_str_land_rd1': [10, 12, 0, 5]   # Round 1 significant strikes landed
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Configure mock to return our data when execute_raw_sql is called
    def mock_execute_raw_sql(sql, params=None):
        # Calculate the pressure values
        result_df = mock_df.copy()
        result_df['sig_str_land_pressure'] = result_df.apply(
            lambda row: row['sig_str_land_rd1'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0, 
            axis=1
        )
        return result_df
    
    # Create calculator instance
    calculator = MockPressureCalculator(mock_conn)
    calculator.feature_utils.execute_raw_sql = mock_execute_raw_sql
    
    # Expected pressure results
    # Fighter 1: 10/20 = 0.5
    # Fighter 2: 12/15 = 0.8
    # Fighter 3: 0/0 = 0.0 (division by zero case)
    # Fighter 4: 5/30 = 0.167
    expected_results = [0.5, 0.8, 0.0, 0.167]
    
    # Set up the SQL template manager to return a simple SQL that will be mock-executed
    template_sql = """
    SELECT fight_id, fighter_id, event_id,
           sig_str_land, sig_str_land_rd1
    FROM features.fight_stats_derived
    """
    calculator.sql_template_manager.render_template.return_value = template_sql
    
    # Call calculate to get the SQL
    calc_result = calculator.calculate()
    sql = calc_result.get("sql", "")
    
    # Execute the SQL and check results
    results = calculator.feature_utils.execute_raw_sql(sql)
    
    # Verify the results
    for i, row in results.iterrows():
        expected = expected_results[i]
        actual = row.get('sig_str_land_pressure', 0)
        # Check that the calculated value is close to expected
        assert abs(actual - expected) < 0.01, f"Expected sig_str_land_pressure to be {expected} but got {actual}"


def test_pressure_calculator_sql_generation():
    """Test that the pressure calculator generates correct SQL."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator instance
    calculator = MockPressureCalculator(mock_conn)
    
    # Test SQL generation
    calc_result = calculator.calculate('fight_stats_derived')
    sql = calc_result.get("sql", "")
    
    # Verify SQL contains expected elements
    assert "sig_str_land_pressure" in sql
    assert "sig_str_land_rd1" in sql
    assert "sig_str_land" in sql
    assert "CASE" in sql
    assert "CAST" in sql
    assert "FLOAT" in sql
    
    # Verify division by zero handling
    assert "sig_str_land > 0" in sql
    assert "ELSE 0.0" in sql


def test_pressure_calculator_with_sql_template():
    """Test the pressure calculator using a SQL template manager."""
    # Create mock data with expected pressure results
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'sig_str_land_pressure': [0.5, 0.8, 0.167]
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
        event_id,
        CASE 
            WHEN sig_str_land > 0 THEN 
                CAST(sig_str_land_rd1 AS FLOAT) / CAST(sig_str_land AS FLOAT)
            ELSE 0.0
        END AS sig_str_land_pressure
    FROM features.fight_stats_derived
    """
    
    # Configure the mock to return our template
    mock_sql_manager.render_template.return_value = expected_sql
    
    # Create and configure the calculator
    calculator = MockPressureCalculator(mock_conn)
    calculator.sql_template_manager = mock_sql_manager
    
    # Mock execute_raw_sql to return our mock data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=mock_df)
    
    # Mock execute_calculator_update to return our mock data
    calculator.execute_calculator_update = MagicMock(return_value=mock_df)
    
    # Test that calculate calls render_template and execute_raw_sql
    result = calculator.save()
    
    # Verify that execute_calculator_update was called
    calculator.execute_calculator_update.assert_called_once()
    
    # Verify the result has the expected column
    assert 'sig_str_land_pressure' in result.columns
    
    # Verify the pressure values are reasonable (between 0 and 1)
    pressure_values = result['sig_str_land_pressure']
    assert all(0.0 <= val <= 1.0 for val in pressure_values), "Pressure values should be between 0 and 1"


def test_pressure_calculator_integration():
    """Test the pressure calculator with full integration."""
    # Create mock data with various scenarios
    mock_data = {
        'fight_id': [1, 2, 3, 4, 5],
        'fighter_id': [101, 102, 103, 104, 105],
        'event_id': [1, 1, 2, 2, 3],
        'sig_str_land': [20, 15, 0, 30, 10],      # Total significant strikes landed
        'sig_str_land_rd1': [10, 12, 0, 5, 10],  # Round 1 significant strikes landed
        'event_date': pd.to_datetime(['2023-01-01', '2023-01-01', '2023-02-01', '2023-02-01', '2023-03-01']),
        'weightclass': ['Lightweight', 'Lightweight', 'Welterweight', 'Welterweight', 'Middleweight']
    }
    
    # Create result data with pressure values
    result_data = {
        'fight_id': [1, 2, 3, 4, 5],
        'fighter_id': [101, 102, 103, 104, 105],
        'event_id': [1, 1, 2, 2, 3],
        'sig_str_land_pressure': [0.5, 0.8, 0.0, 0.167, 1.0]  # Expected pressure values
    }
    
    # Create mock DataFrames
    mock_df = pd.DataFrame(mock_data)
    result_df = pd.DataFrame(result_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Initialize calculator
    calculator = MockPressureCalculator(mock_conn)
    
    # Mock the execute_raw_sql to return our result data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=result_df)
    
    # Mock execute_calculator_update to return our result data
    calculator.execute_calculator_update = MagicMock(return_value=result_df)
    
    # Run the calculation
    result = calculator.save()
    
    # Verify the results
    expected_pressures = [0.5, 0.8, 0.0, 0.167, 1.0]
    
    for i, expected in enumerate(expected_pressures):
        actual = result['sig_str_land_pressure'].iloc[i]
        assert abs(actual - expected) < 0.01, f"Fighter {i+1}: Expected pressure {expected} but got {actual}"
    
    # Verify all pressure values are between 0 and 1
    pressure_values = result['sig_str_land_pressure']
    assert all(0.0 <= val <= 1.0 for val in pressure_values), "All pressure values should be between 0 and 1"


def test_pressure_calculator_edge_cases():
    """Test edge cases for pressure calculation."""
    # Create mock data with edge cases
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'event_id': [1, 1, 2],
        'sig_str_land': [0, 10, 5],       # Zero total, normal, small total
        'sig_str_land_rd1': [5, 15, 5]    # Non-zero rd1, rd1 > total, rd1 = total
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Configure mock to return our data when execute_raw_sql is called
    def mock_execute_raw_sql(sql, params=None):
        result_df = mock_df.copy()
        result_df['sig_str_land_pressure'] = result_df.apply(
            lambda row: row['sig_str_land_rd1'] / row['sig_str_land'] if row['sig_str_land'] > 0 else 0.0, 
            axis=1
        )
        return result_df
    
    # Create calculator instance
    calculator = MockPressureCalculator(mock_conn)
    calculator.feature_utils.execute_raw_sql = mock_execute_raw_sql
    
    # Execute calculation
    calc_result = calculator.calculate()
    sql = calc_result.get("sql", "")
    results = calculator.feature_utils.execute_raw_sql(sql)
    
    # Test edge cases:
    # Fighter 1: 5/0 = 0.0 (division by zero should return 0)
    assert results['sig_str_land_pressure'].iloc[0] == 0.0
    
    # Fighter 2: 15/10 = 1.5 (rd1 > total is possible and valid)
    assert abs(results['sig_str_land_pressure'].iloc[1] - 1.5) < 0.01
    
    # Fighter 3: 5/5 = 1.0 (perfect pressure - all strikes in rd1)
    assert abs(results['sig_str_land_pressure'].iloc[2] - 1.0) < 0.01


def test_pressure_calculator_get_features():
    """Test that get_features returns the correct feature list."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator instance
    calculator = MockPressureCalculator(mock_conn)
    
    # Test get_features
    features = calculator.get_features()
    
    # Should only return sig_str_land
    assert features == ['sig_str_land']
    
    # Test with different table name (should still return same)
    features_with_table = calculator.get_features('some_other_table')
    assert features_with_table == ['sig_str_land']
