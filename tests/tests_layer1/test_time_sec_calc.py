import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sqlalchemy import text
from libs.feature_store.calculators.time_sec_calc import TimeSecCalculator
from libs.feature_store.calculator_context import CalculatorContext
#from tests.tests_layer1.test_utils import TestFeatureUtils

def test_time_sec_calculator_with_context():
    """Test TimeSecCalculator with CalculatorContext and mock data"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102],
            'end_round': [3, 2],
            'end_time': [120, 180],
            'time_format': ['300,300,300', '300,300']
        }),
        'fight_stats_core': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        }),
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        })
    }
    
    # Create a joined version of the data that would be returned by the SQL query
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102],
        'fighter_id': [1, 2, 3, 4],
        'end_round': [3, 3, 2, 2],
        'end_time': [120, 120, 180, 180],
        'time_format': ['300,300,300', '300,300,300', '300,300', '300,300']
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = TimeSecCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                calculator.calculate()
                
                # Verify features DataFrame was created correctly
                assert 'fight_id' in calculator.features.columns
                assert 'fighter_id' in calculator.features.columns
                assert 'time_sec_rd1' in calculator.features.columns
                assert 'time_sec_rd2' in calculator.features.columns
                assert 'time_sec_rd3' in calculator.features.columns
                
                # Check values for first fight (ended in round 3)
                fight1_fighter1 = calculator.features[(calculator.features['fight_id'] == 101) & 
                                                    (calculator.features['fighter_id'] == 1)]
                assert fight1_fighter1['time_sec_rd1'].values[0] == 300
                assert fight1_fighter1['time_sec_rd2'].values[0] == 300
                assert fight1_fighter1['time_sec_rd3'].values[0] == 120
                
                # Check values for second fight (ended in round 2)
                fight2_fighter3 = calculator.features[(calculator.features['fight_id'] == 102) & 
                                                    (calculator.features['fighter_id'] == 3)]
                assert fight2_fighter3['time_sec_rd1'].values[0] == 300
                assert fight2_fighter3['time_sec_rd2'].values[0] == 180
                assert fight2_fighter3['time_sec_rd3'].values[0] == 0
                
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

def test_time_sec_calculator_with_sql_template():
    """Test TimeSecCalculator with SQL template manager"""
    # Create mock data that would be returned by the SQL query
    mock_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102],
        'fighter_id': [1, 2, 3, 4],
        'end_round': [3, 3, 2, 2],
        'end_time': [120, 120, 180, 180],
        'time_format': ['300,300,300', '300,300,300', '300,300', '300,300']
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                SELECT 
                    fm.fight_id,
                    fs.fighter_id,
                    fm.end_round,
                    fm.end_time,
                    fm.time_format
                FROM features.fight_mapping fm
                JOIN features.fight_stats_core fs ON fm.fight_id = fs.fight_id
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = TimeSecCalculator(context)
            
            # Mock pd.read_sql to return our mock data
            with patch.object(pd, 'read_sql', return_value=mock_data):
                # Call get_features
                calculator.get_features()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'time_sec', 'get_features', {'schema': 'features'}
                )
                
                # Verify data was loaded correctly
                pd.testing.assert_frame_equal(calculator.fight_mapping, mock_data)

def test_time_sec_calculator_full_integration():
    """Full integration test for TimeSecCalculator with mocked dependencies"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102],
            'end_round': [3, 2],
            'end_time': [120, 180],
            'time_format': ['300,300,300', '300,300']
        }),
        'fight_stats_core': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        }),
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102],
            'fighter_id': [1, 2, 3, 4],
            'event_id': [1, 1, 2, 2]
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2],
            'event_date': ['2020-01-15', '2020-03-20']
        })
    }
    
    # Create a joined version of the data that would be returned by the SQL query
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102],
        'fighter_id': [1, 2, 3, 4],
        'end_round': [3, 3, 2, 2],
        'end_time': [120, 120, 180, 180],
        'time_format': ['300,300,300', '300,300,300', '300,300', '300,300']
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = TimeSecCalculator(context)
                
                # Run the calculator
                calculator.run()
                
                # Verify the mock_data was updated with the calculated values
                fight_stats_fe = mock_data['fight_stats_fe']
                
                # Check we have the expected columns
                assert 'time_sec_rd1' in fight_stats_fe.columns
                assert 'time_sec_rd2' in fight_stats_fe.columns
                assert 'time_sec_rd3' in fight_stats_fe.columns

def test_time_sec_calculator_sequential_execution():
    """Test TimeSecCalculator sequential execution"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with context
    calculator = TimeSecCalculator(mock_conn)
    
    # Since TimeSecCalculator uses the legacy pattern, we need to mock BaseCalculator.run
    # to see what it's actually doing
    with patch('libs.feature_store.base_calculator.BaseCalculator.run') as mock_run:
        # Call the calculator's run method
        calculator.run(parallel=False)
        
        # Verify BaseCalculator.run was called
        mock_run.assert_called_once_with(parallel=False)

def test_time_sec_calculator_parallel_execution():
    """Test how TimeSecCalculator handles parallel execution request"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with context
    calculator = TimeSecCalculator(mock_conn)
    
    # Since TimeSecCalculator uses the legacy pattern, we need to mock BaseCalculator.run
    # to see what it's actually doing
    with patch('libs.feature_store.base_calculator.BaseCalculator.run') as mock_run:
        # Call the calculator's run method with parallel=True
        calculator.run(parallel=True, max_workers=2)
        
        # Verify BaseCalculator.run was called with the correct parameters
        mock_run.assert_called_once_with(parallel=True, max_workers=2) 