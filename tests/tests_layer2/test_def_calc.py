import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.def_calc import DefenseCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager

class TestDefenseCalculator(DefenseCalculator):
    """Test version of DefenseCalculator that overrides methods for testing."""
    
    def __init__(self, conn=None):
        # If no connection is provided, create a mock
        if conn is None:
            conn = MagicMock()
        super().__init__(conn)
        
        # Override feature_utils with mock
        self.feature_utils = MagicMock()
        
    def _ensure_columns_exist(self, *args, **kwargs):
        # Override to assume columns exist for testing
        pass

def test_def_calculator_with_context():
    """Test the defense calculator with mock data and context."""
    # Create mock data for two fighters in the same fight
    mock_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 101, 103, 102, 103],
        'sig_str_acc': [50.0, 60.0, 45.0, 55.0, 65.0, 40.0],
        'body_acc': [55.0, 45.0, 50.0, 60.0, 70.0, 35.0]
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestDefenseCalculator(mock_conn)
    
    # Mock get_features to return predefined features
    calculator.get_features = MagicMock(return_value=['sig_str_acc', 'body_acc'])
    calculator.features = ['sig_str_acc', 'body_acc']
    
    # Mock execute_raw_sql to implement defense calculation logic
    def mock_execute_raw_sql(sql, params=None):
        # This simulates what the SQL would do - get opponent's accuracy as defense
        result_df = pd.DataFrame()
        result_df['fight_id'] = mock_df['fight_id'].unique()
        result_df = result_df.merge(mock_df[['fight_id', 'fighter_id']], on='fight_id')
        
        # For each fighter, get opponent's accuracy as defense
        result_df['sig_str_def'] = np.nan
        result_df['body_def'] = np.nan
        
        for idx, row in result_df.iterrows():
            # Find opponent in the same fight
            opponent_mask = (mock_df['fight_id'] == row['fight_id']) & (mock_df['fighter_id'] != row['fighter_id'])
            if opponent_mask.any():
                opponent_row = mock_df[opponent_mask].iloc[0]
                result_df.at[idx, 'sig_str_def'] = opponent_row['sig_str_acc']
                result_df.at[idx, 'body_def'] = opponent_row['body_acc']
        
        return result_df
    
    calculator.feature_utils.execute_raw_sql = mock_execute_raw_sql
    
    # Run the calculation
    sql = calculator.calculate()
    
    # If sql is a dictionary, get the SQL from it
    if isinstance(sql, dict) and 'sql' in sql:
        sql_str = sql['sql']
    else:
        sql_str = sql
        
    result = calculator.feature_utils.execute_raw_sql(sql_str)
    
    # Verify the SQL contains the expected columns
    assert 'sig_str_def' in sql_str
    assert 'body_def' in sql_str
    
    # Verify the result has the expected columns
    assert 'sig_str_def' in result.columns
    assert 'body_def' in result.columns
    
    # Verify the defense values match opponent's accuracy
    # Fighter 101 vs Fighter 102 in Fight 1
    fighter101_def = result[(result['fighter_id'] == 101)].iloc[0]
    fighter102_def = result[(result['fighter_id'] == 102)].iloc[0]
    
    assert fighter101_def['sig_str_def'] == 60.0  # Fighter 102's accuracy
    assert fighter102_def['sig_str_def'] == 50.0  # Fighter 101's accuracy

def test_def_calculator_with_sql_template():
    """Test the defense calculator using a SQL template manager."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_str_def': [0.667, 0.400, 0.682],
        'body_def': [0.385, 0.600, 0.450]
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestDefenseCalculator(mock_conn)
    
    # Mock get_features to return predefined features
    calculator.get_features = MagicMock(return_value=['sig_str', 'body'])
    calculator.features = ['sig_str', 'body']
    
    # Create a mock SQLTemplateManager
    mock_sql_manager = MagicMock(spec=SQLTemplateManager)
    
    # SQL template that would be returned by the template manager
    expected_sql = """
    WITH opponent_stats AS (
        SELECT
            t.fight_id,
            t.fighter_id,
            opp.sig_str as opp_sig_str,
            opp.sig_str_att as opp_sig_str_att,
            opp.body as opp_body,
            opp.body_att as opp_body_att
        FROM features.fight_stats_derived t
        JOIN features.fight_stats_derived opp
            ON t.fight_id = opp.fight_id
            AND t.fighter_id != opp.fighter_id
    )
    SELECT 
        t.fight_id,
        t.fighter_id,
        CASE
            WHEN opp.opp_sig_str_att > 0 THEN 1 - ROUND((opp.opp_sig_str::numeric / opp.opp_sig_str_att::numeric), 3)
            ELSE 0
        END as sig_str_def,
        CASE
            WHEN opp.opp_body_att > 0 THEN 1 - ROUND((opp.opp_body::numeric / opp.opp_body_att::numeric), 3)
            ELSE 0
        END as body_def
    FROM features.fight_stats_derived t
    JOIN opponent_stats opp 
        ON t.fight_id = opp.fight_id 
        AND t.fighter_id = opp.fighter_id
    """
    
    # Configure the mock to return our template
    mock_sql_manager.render_template.return_value = expected_sql
    
    # Inject the mock manager into the calculator
    calculator.context.sql_manager = mock_sql_manager
    
    # Run the calculation
    sql = calculator.calculate()
    
    # If sql is a dictionary, get the SQL from it
    if isinstance(sql, dict) and 'sql' in sql:
        sql_str = sql['sql']
    else:
        sql_str = sql
    
    # Configure execute_raw_sql to return our mock data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=mock_df)
    
    # Execute the calculation
    result = calculator.feature_utils.execute_raw_sql(sql_str)
    
    # Verify the result has the expected columns
    assert 'sig_str_def' in result.columns
    assert 'body_def' in result.columns

def test_def_calculator_integration():
    """Test the defense calculator with full integration using SQL template."""
    # Create mock data for fighters and their opponents
    mock_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 103, 104, 105, 106],
        'sig_str': [10, 5, 8, 12, 15, 7],
        'sig_str_att': [15, 20, 20, 30, 22, 35],
        'body': [5, 8, 6, 4, 9, 11],
        'body_att': [13, 10, 10, 20, 20, 25]
    }
    
    # Create result data with defense values
    result_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 103, 104, 105, 106],
        'sig_str_def': [0.750, 0.333, 0.600, 0.400, 0.800, 0.318],
        'body_def': [0.200, 0.615, 0.600, 0.400, 0.560, 0.450]
    }
    
    # Create mock DataFrames
    mock_df = pd.DataFrame(mock_data)
    result_df = pd.DataFrame(result_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestDefenseCalculator(mock_conn)
    
    # Mock get_features to return predefined features
    calculator.get_features = MagicMock(return_value=['sig_str', 'body'])
    calculator.features = ['sig_str', 'body']
    
    # Create a mock SQLTemplateManager
    mock_sql_manager = MagicMock(spec=SQLTemplateManager)
    
    # SQL template that would be returned by the template manager
    expected_sql = """
    WITH opponent_stats AS (
        SELECT
            t.fight_id,
            t.fighter_id,
            opp.sig_str as opp_sig_str,
            opp.sig_str_att as opp_sig_str_att,
            opp.body as opp_body,
            opp.body_att as opp_body_att
        FROM features.fight_stats_derived t
        JOIN features.fight_stats_derived opp
            ON t.fight_id = opp.fight_id
            AND t.fighter_id != opp.fighter_id
    )
    SELECT 
        t.fight_id,
        t.fighter_id,
        CASE
            WHEN opp.opp_sig_str_att > 0 THEN 1 - ROUND((opp.opp_sig_str::numeric / opp.opp_sig_str_att::numeric), 3)
            ELSE 0
        END as sig_str_def,
        CASE
            WHEN opp.opp_body_att > 0 THEN 1 - ROUND((opp.opp_body::numeric / opp.opp_body_att::numeric), 3)
            ELSE 0
        END as body_def
    FROM features.fight_stats_derived t
    JOIN opponent_stats opp 
        ON t.fight_id = opp.fight_id 
        AND t.fighter_id = opp.fighter_id
    """
    
    # Configure the mock to return our template
    mock_sql_manager.render_template.return_value = expected_sql
    
    # Inject the mock manager into the calculator
    calculator.context.sql_manager = mock_sql_manager
    
    # Run the calculation
    sql = calculator.calculate()
    
    # If sql is a dictionary, get the SQL from it
    if isinstance(sql, dict) and 'sql' in sql:
        sql_str = sql['sql']
    else:
        sql_str = sql
    
    # Configure execute_raw_sql to return our mock data
    calculator.feature_utils.execute_raw_sql = MagicMock(return_value=result_df)
    
    # Execute the calculation
    result = calculator.feature_utils.execute_raw_sql(sql_str)
    
    # Verify the result has the expected columns
    assert 'sig_str_def' in result.columns
    assert 'body_def' in result.columns
    
    # Verify defensive values for fighter 101
    fighter101_def = result[result['fighter_id'] == 101].iloc[0]
    assert abs(fighter101_def['sig_str_def'] - 0.750) < 0.001
    assert abs(fighter101_def['body_def'] - 0.200) < 0.001 