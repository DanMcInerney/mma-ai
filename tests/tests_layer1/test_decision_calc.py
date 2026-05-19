import pytest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import numpy as np
from sqlalchemy import text
from libs.feature_store.calculators.decision_calc import DecisionCalculator
from libs.feature_store.calculator_context import CalculatorContext


def test_decision_calculator_initialization():
    """Test DecisionCalculator initialization with different calculator types"""
    mock_conn = MagicMock()
    
    # Test default initialization
    calculator = DecisionCalculator(mock_conn)
    assert calculator.calculator_type == 'single_table'
    
    # Test with explicit calculator_type
    calculator = DecisionCalculator(mock_conn, calculator_type='multi_table')
    assert calculator.calculator_type == 'multi_table'
    
    # Test with context
    mock_context = MagicMock(spec=CalculatorContext)
    mock_context.connection = mock_conn
    calculator = DecisionCalculator(mock_context, calculator_type='cross_table')
    assert calculator.calculator_type == 'cross_table'
    
    # Test column filter initialization
    calculator = DecisionCalculator(mock_conn)
    # Should have include pattern for 'decision'
    assert any(pattern == 'decision' for pattern in calculator.include_patterns)
    
    # Test table_name is set
    assert calculator.table_name == 'fight_stats_fe'


def test_decision_calculator_with_context():
    """Test DecisionCalculator with CalculatorContext and mock data"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'method': ['Unanimous Decision', 'KO/TKO', 'Split Decision'],
            'result': [1, 1, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
            'fighter1_id': [1, 3, 5],
            'fighter2_id': [2, 4, 6],
            'end_round': [3, 1, 3]  # Added end_round column
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
        'method': ['Unanimous Decision', 'Unanimous Decision', 'KO/TKO', 'KO/TKO', 'Split Decision', 'Split Decision'],
        'result': [1, 1, 1, 1, 0, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [3, 3, 1, 1, 3, 3]  # Added end_round column
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
                calculator = DecisionCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                result = calculator.calculate()
                
                # Check that operation returned success
                assert result["status"] == "success"
                assert result["feature_count"] == len(mock_joined_data)
                
                # Check that the features were calculated correctly
                assert 'decision' in calculator.features.columns
                # Note: decision_rd1 removed - decisions cannot happen in round 1
                assert 'decision_rd2' in calculator.features.columns
                assert 'decision_rd3' in calculator.features.columns
                assert 'decision_rd4' in calculator.features.columns
                assert 'decision_rd5' in calculator.features.columns
                
                # Check that fighter 1 got a decision win in fight 101
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 101) & 
                    (calculator.features['fighter_id'] == 1), 
                    'decision'
                ].iloc[0] == 1
                
                # Check that fighter 1 got a decision win in round 3
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 101) & 
                    (calculator.features['fighter_id'] == 1), 
                    'decision_rd3'
                ].iloc[0] == 1
                
                # Check that fighter 3 did not get a decision in fight 102 (KO/TKO)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 102) & 
                    (calculator.features['fighter_id'] == 3), 
                    'decision'
                ].iloc[0] == 0
                
                # Check that fighter 6 got a decision win in fight 103 (Split Decision)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 6), 
                    'decision'
                ].iloc[0] == 1
                
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


def test_decision_calculator_with_sql_template():
    """Test DecisionCalculator with SQL template manager"""
    # Create mock data
    mock_data = {
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'method': ['Unanimous Decision', 'KO/TKO', 'Majority Decision'],
            'result': [1, 1, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
            'fighter1_id': [1, 3, 5],
            'fighter2_id': [2, 4, 6],
            'end_round': [3, 1, 5]  # Added end_round column
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
        'method': ['Unanimous Decision', 'Unanimous Decision', 'KO/TKO', 'KO/TKO', 'Majority Decision', 'Majority Decision'],
        'result': [1, 1, 1, 1, 0, 0],  # fighter1 won fights 101 and 102, fighter2 won fight 103
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [3, 3, 1, 1, 5, 5]  # Added end_round column
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
                calculator = DecisionCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                
                # Check that the SQL template manager was called with the correct parameters
                mock_sql_manager.render_template.assert_called_once_with(
                    'decision', 
                    'get_features',
                    {'schema': 'features'}
                )
                
                # Verify data was loaded correctly - compare with mock_joined_data instead
                pd.testing.assert_frame_equal(calculator.fight_mapping, mock_joined_data)


def test_decision_calculator_case_insensitive():
    """Test that DecisionCalculator handles different case variations of 'decision'"""
    # Create mock data with different case variations
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102, 103, 103, 104, 104],
        'fighter_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'method': ['UNANIMOUS DECISION', 'UNANIMOUS DECISION', 'Split Decision', 'Split Decision', 
                   'majority decision', 'majority decision', 'KO/TKO', 'KO/TKO'],
        'result': [1, 1, 0, 0, 1, 1, 1, 1],  
        'fighter1_id': [1, 1, 3, 3, 5, 5, 7, 7],
        'fighter2_id': [2, 2, 4, 4, 6, 6, 8, 8],
        'end_round': [3, 3, 3, 3, 5, 5, 1, 1]
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection 
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn)
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = DecisionCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                calculator.calculate()
                
                # Check that all fighters who won by decision got decision=1
                # Fighter 1: UNANIMOUS DECISION winner
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 101) & 
                    (calculator.features['fighter_id'] == 1), 
                    'decision'
                ].iloc[0] == 1
                
                # Fighter 4: Split Decision winner (result=0 means fighter2 won)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 102) & 
                    (calculator.features['fighter_id'] == 4), 
                    'decision'
                ].iloc[0] == 1
                
                # Fighter 5: majority decision winner
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 5), 
                    'decision'
                ].iloc[0] == 1
                
                # Fighter 7: KO/TKO winner (should not have decision=1)
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 104) & 
                    (calculator.features['fighter_id'] == 7), 
                    'decision'
                ].iloc[0] == 0


def test_decision_calculator_sequential_execution():
    """Test DecisionCalculator sequential execution"""
    # Create calculator with mocks
    mock_conn = MagicMock()
    calculator = DecisionCalculator(mock_conn)
    
    # Create a simplified mock for run to avoid all the complexities
    with patch.object(calculator, 'run_sequential', return_value={"fight_stats_fe": pd.DataFrame()}) as mock_run_sequential:
        # Test sequential execution
        calculator.run(parallel=False)
        mock_run_sequential.assert_called_once()


def test_decision_calculator_column_filtering():
    """Test DecisionCalculator column filtering functionality"""
    mock_conn = MagicMock()
    calculator = DecisionCalculator(mock_conn)
    
    # Test include pattern
    assert calculator.should_process_column('decision')
    # Note: decision_rd1 removed - decisions cannot happen in round 1
    
    # Test exclude pattern (add one for testing)
    calculator.add_exclude_pattern('decision_rd2')
    assert not calculator.should_process_column('decision_rd2')
    
    # Test with list of columns
    all_columns = ['decision', 'decision_rd2', 'decision_rd3', 'other_column']
    filtered = calculator.filter_columns(all_columns)
    assert 'decision' in filtered
    assert 'decision_rd2' not in filtered
    assert 'other_column' not in filtered


def test_decision_calculator_edge_cases():
    """Test DecisionCalculator with edge cases like None/NaN methods"""
    # Create mock data with edge cases
    mock_joined_data = pd.DataFrame({
        'fight_id': [101, 101, 102, 102, 103, 103],
        'fighter_id': [1, 2, 3, 4, 5, 6],
        'method': ['Unanimous Decision', 'Unanimous Decision', None, None, pd.NA, pd.NA],
        'result': [1, 1, 1, 1, 0, 0],
        'fighter1_id': [1, 1, 3, 3, 5, 5],
        'fighter2_id': [2, 2, 4, 4, 6, 6],
        'end_round': [3, 3, 2, 2, 1, 1]
    })
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection 
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn)
            
            # Mock the SQL execution to return our joined data
            with patch.object(pd, 'read_sql', return_value=mock_joined_data):
                # Create calculator with context
                calculator = DecisionCalculator(context)
                
                # Run the calculator
                calculator.get_features()
                calculator.calculate()
                
                # Check that fighter 1 got decision=1 for valid decision
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 101) & 
                    (calculator.features['fighter_id'] == 1), 
                    'decision'
                ].iloc[0] == 1
                
                # Check that fighters with None/NaN methods got decision=0
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 102) & 
                    (calculator.features['fighter_id'] == 3), 
                    'decision'
                ].iloc[0] == 0
                
                assert calculator.features.loc[
                    (calculator.features['fight_id'] == 103) & 
                    (calculator.features['fighter_id'] == 6), 
                    'decision'
                ].iloc[0] == 0
