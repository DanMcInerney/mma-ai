import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from sqlalchemy import text
from libs.feature_store.calculators.sub_land_calc import SubmissionslandCalculator
from libs.feature_store.calculator_context import CalculatorContext


def test_sub_land_calculator_initialization():
    """Test SubmissionslandCalculator initialization with different calculator types"""
    mock_conn = MagicMock()
    
    # Test default initialization
    calculator = SubmissionslandCalculator(mock_conn)
    assert calculator.calculator_type == 'single_table'
    
    # Test with explicit calculator_type
    calculator = SubmissionslandCalculator(mock_conn, calculator_type='multi_table')
    assert calculator.calculator_type == 'multi_table'
    
    # Test with context
    mock_context = MagicMock(spec=CalculatorContext)
    mock_context.connection = mock_conn
    calculator = SubmissionslandCalculator(mock_context, calculator_type='cross_table')
    assert calculator.calculator_type == 'cross_table'
    
    # Test column filter initialization
    calculator = SubmissionslandCalculator(mock_conn)
    # Should have include pattern for 'sub_land'
    assert any(pattern == 'sub_land' for pattern in calculator.include_patterns)
    
    # Test table_name is set
    assert calculator.table_name == 'fight_stats_fe'


def test_sub_land_calculator_with_context():
    """Test SubmissionslandCalculator with CalculatorContext and mock data"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'method': ['KO/TKO', 'Decision', 'Submission'],
            'result': [1, 1, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
            'fighter1_id': [1, 3, 5],
            'fighter2_id': [2, 4, 6],
            'end_round': [1, 3, 2]  # Added end_round column
        }),
        'fight_stats_core': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        }),
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        })
    }
    
    # Create a joined version of the data that would be returned by the SQL query
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102, 103, 103],
        'fighter_id': [1, 2, 3, 4, 5, 6],
        'method': ['KO/TKO', 'KO/TKO', 'Decision', 'Decision', 'Submission', 'Submission'],
        'result': [1, 1, 1, 1, 0, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [1, 1, 3, 3, 2, 2]  # Added end_round column
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
                calculator = SubmissionslandCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                result = calculator.calculate()
                
                # Check that operation returned success
                assert result["status"] == "success"
                assert result["feature_count"] == len(mock_joined_data)
                
                # Check that the features were calculated correctly
                assert 'sub_land' in calculator.features.columns
                assert 'sub_land_rd1' in calculator.features.columns
                assert 'sub_land_rd2' in calculator.features.columns
                assert 'sub_land_rd3' in calculator.features.columns
                assert 'sub_land_rd4' in calculator.features.columns
                assert 'sub_land_rd5' in calculator.features.columns
                
                # Check that fighter 6 got a submission in fight 103
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 6), 
                    'sub_land'
                ].iloc[0] == 1
                
                # Check that fighter 6 got a submission in round 2
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 6), 
                    'sub_land_rd2'
                ].iloc[0] == 1
                
                # Check that fighter 1 did not get a submission in fight 101 (KO/TKO)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 101) & 
                    (calculator.features['fighter_id'] == 1), 
                    'sub_land'
                ].iloc[0] == 0
                
                # Check that fighter 3 did not get a submission in fight 102 (Decision)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 102) & 
                    (calculator.features['fighter_id'] == 3), 
                    'sub_land'
                ].iloc[0] == 0
                
                # Test save method with context
                with patch.object(context, 'update_table') as mock_update:
                    result_df = calculator.save()
                    
                    # Verify update_table was called with correct parameters
                    mock_update.assert_called_once()
                    table_arg = mock_update.call_args[0][0]
                    data_arg = mock_update.call_args[0][1]
                    
                    # Check table name is correct
                    assert table_arg == 'fight_stats_fe'
                    
                    # Check DataFrame was passed correctly
                    pd.testing.assert_frame_equal(data_arg, calculator.features)
                    
                    # Check return value
                    pd.testing.assert_frame_equal(result_df, calculator.features)


def test_sub_land_calculator_with_sql_template():
    """Test SubmissionslandCalculator with SQL template manager"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'method': ['KO/TKO', 'Decision', 'Submission'],
            'result': [1, 1, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
            'fighter1_id': [1, 3, 5],
            'fighter2_id': [2, 4, 6],
            'end_round': [1, 3, 2]  # Added end_round column
        }),
        'fight_stats_core': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        }),
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        })
    }
    
    # Create a joined version of the data that would be returned by the SQL query
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102, 103, 103],
        'fighter_id': [1, 2, 3, 4, 5, 6],
        'method': ['KO/TKO', 'KO/TKO', 'Decision', 'Decision', 'Submission', 'Submission'],
        'result': [1, 1, 1, 1, 0, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [1, 1, 3, 3, 2, 2]  # Added end_round column
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create mock SQL template manager
    mock_sql_manager = MagicMock()
    mock_sql_manager.render_template.return_value = """
        SELECT * FROM mock_query
    """
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager', return_value=mock_sql_manager):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = SubmissionslandCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                
                # Check that the SQL template manager was called with the correct parameters
                mock_sql_manager.render_template.assert_called_once_with(
                    'sub_land', 
                    'get_features',
                    {'schema': 'features'}
                )
                
                # Verify data was loaded correctly - compare with mock_joined_data instead
                pd.testing.assert_frame_equal(calculator.fight_mapping, mock_joined_data)


def test_sub_land_calculator_sequential_execution():
    """Test SubmissionslandCalculator sequential execution"""
    # Create calculator with mocks
    mock_conn = MagicMock()
    calculator = SubmissionslandCalculator(mock_conn)
    
    # Create a simplified mock for run to avoid all the complexities
    with patch.object(calculator, 'run_sequential', return_value={"fight_stats_fe": pd.DataFrame()}) as mock_run_sequential:
        # Test sequential execution
        calculator.run(parallel=False)
        mock_run_sequential.assert_called_once()


def test_sub_land_calculator_parallel_execution():
    """Test SubmissionslandCalculator parallel execution"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create mock SQL template manager
    mock_sql_manager = MagicMock()
    mock_sql_manager.render_template.return_value = "SELECT * FROM mock_query"
    
    # Create mock data
    mock_data = pd.DataFrame({
        'fight_id': [101, 102],
        'fighter_id': [1, 2]
    })
    
    # Create context with mock connection
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager', return_value=mock_sql_manager):
            # Create mock context
            context = CalculatorContext(mock_conn)
            
            # Replace SQL manager with our mock
            context.sql_manager = mock_sql_manager
            
            # Create calculator with context
            calculator = SubmissionslandCalculator(context)
            
            # Mock the necessary methods to avoid actual execution
            with patch.object(pd, 'read_sql', return_value=mock_data):
                with patch.object(calculator, 'calculate_for_table', return_value="SELECT * FROM mock_table"):
                    with patch.object(calculator, 'execute_raw_sql', return_value=mock_data):
                        # For single_table calculators, run_sequential is called even when parallel=True
                        with patch.object(calculator, 'run_sequential', return_value={"fight_stats_fe": mock_data}) as mock_run_sequential:
                            # Test parallel execution (which will actually use sequential for single_table)
                            result = calculator.run(parallel=True, max_workers=2)
                            
                            # Verify the run_sequential method was called
                            mock_run_sequential.assert_called_once()
                            
                            # Verify the result contains the expected data
                            assert "fight_stats_fe" in result
                            assert isinstance(result["fight_stats_fe"], pd.DataFrame)


def test_sub_land_calculator_full_integration():
    """Full integration test for SubmissionslandCalculator with mocked dependencies"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'method': ['KO/TKO', 'Decision', 'Submission'],
            'result': [1, 1, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
            'fighter1_id': [1, 3, 5],
            'fighter2_id': [2, 4, 6],
            'end_round': [1, 3, 2]  # Added end_round column
        }),
        'fight_stats_core': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        }),
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 101, 102, 102, 103, 103],
            'fighter_id': [1, 2, 3, 4, 5, 6],
            'event_id': [1, 1, 2, 2, 3, 3]
        })
    }
    
    # Create a joined version of the data that would be returned by the SQL query
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102, 103, 103],
        'fighter_id': [1, 2, 3, 4, 5, 6],
        'method': ['KO/TKO', 'KO/TKO', 'Decision', 'Decision', 'Submission', 'Submission'],
        'result': [1, 1, 1, 1, 0, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [1, 1, 3, 3, 2, 2]  # Added end_round column
    })
    
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
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = SubmissionslandCalculator(context)
                
                # Set up features for testing directly
                calculator.get_features()
                calculator.calculate()
                    
                # Check that the features were calculated correctly
                assert 'sub_land' in calculator.features.columns
                assert 'sub_land_rd1' in calculator.features.columns
                
                # Check that fighter 6 got a submission in fight 103
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 6), 
                    'sub_land'
                ].iloc[0] == 1


def test_sub_land_calculator_column_filtering():
    """Test SubmissionslandCalculator column filtering functionality"""
    mock_conn = MagicMock()
    calculator = SubmissionslandCalculator(mock_conn)
    
    # Test include pattern
    assert calculator.should_process_column('sub_land')
    assert calculator.should_process_column('sub_land_rd1')
    
    # Test exclude pattern (add one for testing)
    calculator.add_exclude_pattern('sub_land_rd2')
    assert not calculator.should_process_column('sub_land_rd2')
    
    # Test with list of columns
    all_columns = ['sub_land', 'sub_land_rd1', 'sub_land_rd2', 'sub_land_rd3', 'other_column']
    filtered = calculator.filter_columns(all_columns)
    assert 'sub_land' in filtered
    assert 'sub_land_rd1' in filtered
    assert 'sub_land_rd2' not in filtered
    assert 'other_column' not in filtered 