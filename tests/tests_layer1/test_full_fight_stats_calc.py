import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sqlalchemy import text
from libs.feature_store.calculators.full_fight_stats import FullFightStatsCalculator
from libs.feature_store.calculator_context import CalculatorContext


def test_full_fight_stats_calculator_with_context():
    """Test FullFightStatsCalculator with CalculatorContext and mock data"""
    # Create mock data with round-specific columns
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2],
            'sig_str_land_rd1': [10, 5, 8, 12],
            'sig_str_land_rd2': [8, 7, 6, 9],
            'sig_str_land_rd3': [5, 3, 0, 0],
            'td_land_rd1': [2, 0, 1, 3],
            'td_land_rd2': [1, 1, 0, 2],
            'td_land_rd3': [0, 0, 0, 0]
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Mock the SQL execution to return our mock data
            with patch.object(pd, 'read_sql', return_value=mock_data['fight_stats_fe']):
                # Create calculator with context
                calculator = FullFightStatsCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                calculator.calculate()
                
                # Verify features DataFrame was created correctly
                assert 'fight_id' in calculator.features.columns
                assert 'fighter_id' in calculator.features.columns
                assert 'sig_str_land' in calculator.features.columns
                assert 'td_land' in calculator.features.columns
                
                # Check values for first fighter
                fighter1 = calculator.features[calculator.features['fighter_id'] == 1]
                assert fighter1['sig_str_land'].values[0] == 23  # 10 + 8 + 5
                assert fighter1['td_land'].values[0] == 3  # 2 + 1 + 0
                
                # Check values for third fighter
                fighter3 = calculator.features[calculator.features['fighter_id'] == 3]
                assert fighter3['sig_str_land'].values[0] == 14  # 8 + 6 + 0
                assert fighter3['td_land'].values[0] == 1  # 1 + 0 + 0
                
                # Test save method with context
                with patch.object(context, 'update_table') as mock_update:
                    calculator.save()
                    
                    # Verify update_table was called with correct parameters
                    mock_update.assert_called_once()
                    table_arg = mock_update.call_args[0][0]
                    data_arg = mock_update.call_args[0][1]
                    
                    # Check table name is correct
                    assert table_arg == 'fight_stats_fe'
                    
                    # Check DataFrame was passed correctly
                    pd.testing.assert_frame_equal(data_arg, calculator.features)


def test_full_fight_stats_calculator_with_sql_template():
    """Test FullFightStatsCalculator with SQL template manager"""
    # Create mock data with round-specific columns
    mock_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102],
        'fighter_id': [1, 2, 3, 4],
        'event_id': [1, 1, 2, 2],
        'sig_str_land_rd1': [10, 5, 8, 12],
        'sig_str_land_rd2': [8, 7, 6, 9],
        'sig_str_land_rd3': [5, 3, 0, 0],
        'td_land_rd1': [2, 0, 1, 3],
        'td_land_rd2': [1, 1, 0, 2],
        'td_land_rd3': [0, 0, 0, 0]
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                SELECT * FROM features.fight_stats_fe
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = FullFightStatsCalculator(context)
            
            # Mock pd.read_sql to return our mock data
            with patch.object(pd, 'read_sql', return_value=mock_data):
                # Call get_features
                calculator.get_features()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'full_fight_stats', 'get_features', 
                    {'schema': 'features', 'table': 'fight_stats_fe'}
                )
                
                # Verify data was loaded correctly
                pd.testing.assert_frame_equal(calculator.features, mock_data)


def test_full_fight_stats_calculator_full_integration():
    """Full integration test for FullFightStatsCalculator with mocked dependencies"""
    # Create mock data with round-specific columns
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2],
            'sig_str_land_rd1': [10, 5, 8, 12],
            'sig_str_land_rd2': [8, 7, 6, 9],
            'sig_str_land_rd3': [5, 3, 0, 0],
            'td_land_rd1': [2, 0, 1, 3],
            'td_land_rd2': [1, 1, 0, 2],
            'td_land_rd3': [0, 0, 0, 0]
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create mock SQL template manager
    mock_sql_manager = MagicMock()
    mock_sql_manager.render_template.return_value = """
        SELECT * FROM mock_query
    """
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Replace the SQL manager with our mock
            context.sql_manager = mock_sql_manager
            
            # Mock the SQL execution to return our mock data
            with patch.object(pd, 'read_sql', return_value=mock_data['fight_stats_fe']):
                # Create calculator with context
                calculator = FullFightStatsCalculator(context)
                
                # Run the calculator
                calculator.run()
                
                # Verify features DataFrame was created correctly
                assert 'fight_id' in calculator.features.columns
                assert 'fighter_id' in calculator.features.columns
                assert 'sig_str_land' in calculator.features.columns
                assert 'td_land' in calculator.features.columns
                
                # Check values for first fighter
                fighter1 = calculator.features[calculator.features['fighter_id'] == 1]
                assert fighter1['sig_str_land'].values[0] == 23  # 10 + 8 + 5
                assert fighter1['td_land'].values[0] == 3  # 2 + 1 + 0
                
                # Check values for third fighter
                fighter3 = calculator.features[calculator.features['fighter_id'] == 3]
                assert fighter3['sig_str_land'].values[0] == 14  # 8 + 6 + 0
                assert fighter3['td_land'].values[0] == 1  # 1 + 0 + 0 