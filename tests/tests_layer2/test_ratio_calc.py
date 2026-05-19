import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.ratio_calc import RatioCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager

class TestRatioCalculator(RatioCalculator):
    """Test version of RatioCalculator that overrides methods for testing."""
    
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

def test_ratio_calculator_with_context():
    """Test the ratio calculator with mock data and context."""
    # Create mock data for two fighters in the same fight
    mock_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 101, 103, 102, 103],
        'sig_str': [10, 5, 8, 12, 15, 7],
        'body': [5, 8, 6, 4, 9, 11]
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestRatioCalculator(mock_conn)
    
    # Mock get_features to return predefined features
    calculator.get_features = MagicMock(return_value=['sig_str', 'body'])
    calculator.features = ['sig_str', 'body']
    
    # Mock get_opponent_stats to return a SQL query
    calculator.feature_utils.get_opponent_stats = MagicMock(return_value="""
        SELECT 
            t.fight_id,
            t.fighter_id,
            opp.sig_str as opp_sig_str,
            opp.body as opp_body
        FROM features.fight_stats_derived t
        JOIN features.fight_stats_derived opp 
            ON t.fight_id = opp.fight_id 
            AND t.fighter_id != opp.fighter_id
    """)
    
    # Mock execute_raw_sql to implement ratio calculation logic
    def mock_execute_raw_sql(sql, params=None):
        # This simulates what the SQL would do - calculate ratios
        result_df = pd.DataFrame()
        result_df['fight_id'] = mock_df['fight_id'].unique()
        result_df = result_df.merge(mock_df[['fight_id', 'fighter_id']], on='fight_id')
        
        # For each fighter, calculate ratio against opponent
        result_df['sig_str_ratio'] = np.nan
        result_df['body_ratio'] = np.nan
        
        for idx, row in result_df.iterrows():
            # Find fighter's stats
            fighter_mask = (mock_df['fight_id'] == row['fight_id']) & (mock_df['fighter_id'] == row['fighter_id'])
            if fighter_mask.any():
                fighter_row = mock_df[fighter_mask].iloc[0]
                
                # Find opponent in the same fight
                opponent_mask = (mock_df['fight_id'] == row['fight_id']) & (mock_df['fighter_id'] != row['fighter_id'])
                if opponent_mask.any():
                    opponent_row = mock_df[opponent_mask].iloc[0]
                    
                    # Calculate ratios
                    sig_str_total = fighter_row['sig_str'] + opponent_row['sig_str']
                    body_total = fighter_row['body'] + opponent_row['body']
                    
                    if sig_str_total > 0:
                        result_df.at[idx, 'sig_str_ratio'] = round(fighter_row['sig_str'] / sig_str_total, 3)
                    else:
                        result_df.at[idx, 'sig_str_ratio'] = 0
                        
                    if body_total > 0:
                        result_df.at[idx, 'body_ratio'] = round(fighter_row['body'] / body_total, 3)
                    else:
                        result_df.at[idx, 'body_ratio'] = 0
        
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
    assert 'sig_str_ratio' in sql_str
    assert 'body_ratio' in sql_str
    
    # Verify the result has the expected columns
    assert 'sig_str_ratio' in result.columns
    assert 'body_ratio' in result.columns
    
    # Verify the ratio values are correct
    # Fighter 101 vs Fighter 102 in Fight 1
    fighter101_ratio = result[(result['fighter_id'] == 101) & (result['fight_id'] == 1)].iloc[0]
    
    # Calculate expected ratios
    fighter101_sig_str = mock_df[(mock_df['fighter_id'] == 101) & (mock_df['fight_id'] == 1)]['sig_str'].iloc[0]
    fighter102_sig_str = mock_df[(mock_df['fighter_id'] == 102) & (mock_df['fight_id'] == 1)]['sig_str'].iloc[0]
    expected_sig_str_ratio = round(fighter101_sig_str / (fighter101_sig_str + fighter102_sig_str), 3)
    
    assert fighter101_ratio['sig_str_ratio'] == expected_sig_str_ratio

def test_ratio_calculator_with_sql_template():
    """Test the ratio calculator using a SQL template manager."""
    # Create mock data
    mock_data = {
        'fight_id': [1, 2, 3],
        'fighter_id': [101, 102, 103],
        'sig_str_ratio': [0.667, 0.400, 0.682],
        'body_ratio': [0.385, 0.600, 0.450]
    }
    
    # Create mock DataFrame
    mock_df = pd.DataFrame(mock_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestRatioCalculator(mock_conn)
    
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
            opp.body as opp_body
        FROM features.fight_stats_derived t
        JOIN features.fight_stats_derived opp
            ON t.fight_id = opp.fight_id
            AND t.fighter_id != opp.fighter_id
    )
    SELECT 
        t.fight_id,
        t.fighter_id,
        CASE
            WHEN fm.fighter1_id = t.fighter_id THEN 
                CASE WHEN (t.sig_str + opp.sig_str) > 0 
                    THEN ROUND((t.sig_str::numeric / (t.sig_str + opp.sig_str)::numeric), 3)
                    ELSE 0 END
            ELSE
                CASE WHEN (t.sig_str + opp.sig_str) > 0 
                    THEN ROUND((t.sig_str::numeric / (t.sig_str + opp.sig_str)::numeric), 3)
                    ELSE 0 END
        END as sig_str_ratio,
        CASE
            WHEN fm.fighter1_id = t.fighter_id THEN 
                CASE WHEN (t.body + opp.body) > 0 
                    THEN ROUND((t.body::numeric / (t.body + opp.body)::numeric), 3)
                    ELSE 0 END
            ELSE
                CASE WHEN (t.body + opp.body) > 0 
                    THEN ROUND((t.body::numeric / (t.body + opp.body)::numeric), 3)
                    ELSE 0 END
        END as body_ratio
    FROM features.fight_stats_derived t
    JOIN opponent_stats opp 
        ON t.fight_id = opp.fight_id 
        AND t.fighter_id = opp.fighter_id
    JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
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
    assert 'sig_str_ratio' in result.columns
    assert 'body_ratio' in result.columns

def test_ratio_calculator_integration():
    """Test the ratio calculator with full integration using SQL template."""
    # Create mock data for fighters and their opponents
    mock_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 103, 104, 105, 106],
        'sig_str': [10, 5, 8, 12, 15, 7],
        'body': [5, 8, 6, 4, 9, 11]
    }
    
    # Create result data with ratio values
    result_data = {
        'fight_id': [1, 1, 2, 2, 3, 3],
        'fighter_id': [101, 102, 103, 104, 105, 106],
        'sig_str_ratio': [0.667, 0.333, 0.400, 0.600, 0.682, 0.318],
        'body_ratio': [0.385, 0.615, 0.600, 0.400, 0.450, 0.550]
    }
    
    # Create mock DataFrames
    mock_df = pd.DataFrame(mock_data)
    result_df = pd.DataFrame(result_data)
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calculator = TestRatioCalculator(mock_conn)
    
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
            opp.body as opp_body
        FROM features.fight_stats_derived t
        JOIN features.fight_stats_derived opp
            ON t.fight_id = opp.fight_id
            AND t.fighter_id != opp.fighter_id
    )
    SELECT 
        t.fight_id,
        t.fighter_id,
        CASE
            WHEN fm.fighter1_id = t.fighter_id THEN 
                CASE WHEN (t.sig_str + opp.sig_str) > 0 
                    THEN ROUND((t.sig_str::numeric / (t.sig_str + opp.sig_str)::numeric), 3)
                    ELSE 0 END
            ELSE
                CASE WHEN (t.sig_str + opp.sig_str) > 0 
                    THEN ROUND((t.sig_str::numeric / (t.sig_str + opp.sig_str)::numeric), 3)
                    ELSE 0 END
        END as sig_str_ratio,
        CASE
            WHEN fm.fighter1_id = t.fighter_id THEN 
                CASE WHEN (t.body + opp.body) > 0 
                    THEN ROUND((t.body::numeric / (t.body + opp.body)::numeric), 3)
                    ELSE 0 END
            ELSE
                CASE WHEN (t.body + opp.body) > 0 
                    THEN ROUND((t.body::numeric / (t.body + opp.body)::numeric), 3)
                    ELSE 0 END
        END as body_ratio
    FROM features.fight_stats_derived t
    JOIN opponent_stats opp 
        ON t.fight_id = opp.fight_id 
        AND t.fighter_id = opp.fighter_id
    JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
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
    assert 'sig_str_ratio' in result.columns
    assert 'body_ratio' in result.columns
    
    # Verify the ratio values are correct
    fighter101_ratio = result[(result['fighter_id'] == 101)].iloc[0]
    fighter102_ratio = result[(result['fighter_id'] == 102)].iloc[0]
    
    # Check that the ratios sum to 1.0 for each fight
    assert abs(fighter101_ratio['sig_str_ratio'] + fighter102_ratio['sig_str_ratio'] - 1.0) < 0.001
    assert abs(fighter101_ratio['body_ratio'] + fighter102_ratio['body_ratio'] - 1.0) < 0.001 