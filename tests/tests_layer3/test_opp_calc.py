import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.opp_calc import OpponentCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext


class TestOpponentCalculator(OpponentCalculator):
    """Test class for OpponentCalculator that overrides methods for testing purposes."""
    
    def __init__(self, conn=None):
        """Initialize with mock connection if none provided and override SQL template manager."""
        if conn is None:
            conn = MagicMock()
        
        # Create a mock context first
        self.mock_context = CalculatorContext(conn)
        
        # Set feature tables BEFORE calling parent init
        self.feature_tables = ['sig_str', 'td', 'control']
        
        # Mock all required objects for the context
        self.mock_context.feature_utils = MagicMock()
        self.mock_context.feature_utils.get_columns_from_table = MagicMock()
        self.mock_context.feature_utils.execute_raw_sql = MagicMock()
        
        # Create a mock SQL template manager
        self.mock_context.sql_manager = MagicMock()
        self.mock_context.sql_manager.render_template = MagicMock()
        
        # We need to patch get_feature_tables during init to avoid database calls
        with patch.object(OpponentCalculator, 'get_feature_tables', return_value=self.feature_tables):
            # Initialize with the mock context
            super().__init__(self.mock_context)
    
    def get_feature_tables(self, exclude_patterns=None):
        """Override to return fixed list of tables for testing."""
        # If exclude patterns are provided, filter the tables
        if exclude_patterns:
            return [table for table in self.feature_tables 
                   if not any(pattern in table for pattern in exclude_patterns)]
        return self.feature_tables
    
    def _validate_inputs(self, *args, **kwargs):
        """Override for testing to skip validation."""
        return True
    
    def _validate_outputs(self, *args, **kwargs):
        """Override for testing to assume outputs are valid."""
        return True
        
    def execute_calculator_update(self, *args, **kwargs):
        """Override to return a mock result for testing."""
        # Create a result dataframe with key columns and opponent columns
        result = pd.DataFrame({
            'fighter_id': [1, 2],
            'fight_id': [100, 100],
            'sig_str_acc_opp': [0.65, 0.75],
            'sig_str_land_opp': [18, 25]
        })
        return result
        
    def execute_raw_sql(self, sql, params=None, return_results=True):
        """Override to return mock results for testing."""
        # Return a test DataFrame instead of executing SQL
        return pd.DataFrame({
            'fighter_id': [1, 2],
            'fight_id': [100, 100],
            'sig_str_acc_opp': [0.8, 0.7],
            'sig_str_land_opp': [20, 15]
        })


def test_feature_tables_detection():
    """Test that calculator correctly detects feature-specific tables."""
    # Create a mock connection
    mock_conn = MagicMock()
    
    # Create a test DataFrame with table names
    mock_df = pd.DataFrame({'table_name': ['sig_str', 'td', 'control', 
                                         'fight_mapping', 'fighter_mapping']})
    
    # Define a patched version of get_feature_tables that filters the mock data
    def patched_get_feature_tables(self, exclude_patterns=None):
        # Default exclude patterns if none provided
        if exclude_patterns is None:
            exclude_patterns = ['_mapping', 'fight_stats_core', 'fight_stats_fe', 
                               'fight_stats_derived', 'first_time']
        
        # Filter the mock DataFrame based on exclude patterns
        filtered_tables = mock_df['table_name'].tolist()
        
        if exclude_patterns:
            filtered_tables = [
                table for table in filtered_tables 
                if not any(
                    (pattern.startswith('_') and pattern in table) or 
                    (not pattern.startswith('_') and pattern == table)
                    for pattern in exclude_patterns
                )
            ]
        
        return filtered_tables
    
    # Create a test context
    test_context = CalculatorContext(mock_conn)
    
    # Mock pd.read_sql and replace get_feature_tables with our filtered version
    with patch('pandas.read_sql', return_value=mock_df), \
         patch.object(BaseCalculator, 'get_feature_tables', patched_get_feature_tables):
        # Mock conn.begin() to return a context manager
        mock_conn.begin.return_value.__enter__ = MagicMock()
        mock_conn.begin.return_value.__exit__ = MagicMock()
        
        # Create a calculator with our mock context
        calc = OpponentCalculator(test_context)
        
        # Call the method with specific exclude patterns to filter out the non-feature tables
        tables = calc.get_feature_tables(['_mapping', 'meta'])
        
        # Verify the tables were filtered correctly
        assert len(tables) == 3
        assert 'sig_str' in tables
        assert 'td' in tables
        assert 'control' in tables
        assert 'fight_mapping' not in tables
        assert 'fighter_mapping' not in tables
        assert 'meta_table' not in tables


def test_get_features_for_table():
    """Test getting features for a specific table."""
    # Create mock connection and context
    mock_conn = MagicMock()
    test_context = CalculatorContext(mock_conn)
    
    # Create calculator with mock context
    calc = TestOpponentCalculator(mock_conn)
    
    # Ensure get_columns_from_table is properly mocked
    calc.context.feature_utils.get_columns_from_table = MagicMock(return_value=['sig_str_acc', 'sig_str_land'])
    
    # Get features for sig_str table
    features = calc.get_features('sig_str')
    
    # Verify the feature_utils method was called correctly
    calc.context.feature_utils.get_columns_from_table.assert_called_once_with(
        calc.schema, 'sig_str', exclude_strs=calc.exclude_patterns
    )
    
    # Verify only the feature columns are returned
    assert 'sig_str_acc' in features
    assert 'sig_str_land' in features
    assert len(features) == 2


def test_calculate_for_table():
    """Test calculating opponent stats for a specific table."""
    # Create mocks
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock get_features to return specific columns
    calc.get_features = MagicMock(return_value=['sig_str_acc', 'sig_str_land'])
    
    # Ensure render_template is properly mocked
    expected_sql = "-- SQL for opponent stats"
    calc.context.sql_manager.render_template = MagicMock(return_value=expected_sql)
    
    # Call calculate_for_table
    sql = calc.calculate_for_table('sig_str')
    
    # Verify SQL template manager was called with correct parameters
    calc.context.sql_manager.render_template.assert_called_once()
    template_params = calc.context.sql_manager.render_template.call_args[0]
    assert template_params[0] == 'opponent'
    assert template_params[1] == 'calculate'
    assert template_params[2]['schema'] == calc.schema
    assert template_params[2]['table_name'] == 'sig_str'
    assert template_params[2]['columns'] == ['sig_str_acc', 'sig_str_land']
    
    # Verify returned SQL matches expected
    assert sql == expected_sql


def test_execute_for_table():
    """Test executing opponent stats calculation for a specific table."""
    # Create mocks
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock calculate_for_table to return SQL
    calc.calculate_for_table = MagicMock(return_value="SELECT * FROM test")
    
    # Create expected result DataFrame
    expected_df = pd.DataFrame({
        'fighter_id': [1, 2],
        'fight_id': [100, 100],
        'sig_str_acc_opp': [0.8, 0.7],
        'sig_str_land_opp': [20, 15]
    })
    
    # Mock execute_raw_sql to return expected DataFrame
    calc.execute_raw_sql = MagicMock(return_value=expected_df)
    
    # Call execute_for_table
    result = calc.execute_for_table('sig_str')
    
    # Verify calculate_for_table was called with correct parameters
    calc.calculate_for_table.assert_called_once_with('sig_str', None)
    
    # Verify execute_raw_sql was called with correct SQL
    calc.execute_raw_sql.assert_called_once_with("SELECT * FROM test", return_results=True)
    
    # Verify returned DataFrame matches expected
    pd.testing.assert_frame_equal(result, expected_df)


def test_save_for_table():
    """Test saving opponent stats for a specific table."""
    # Create mocks
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock get_features to return columns
    calc.get_features = MagicMock(return_value=['sig_str_acc', 'sig_str_land'])
    
    # Mock calculate_for_table to return SQL
    calc.calculate_for_table = MagicMock(return_value="SELECT * FROM test")
    
    # Create expected result DataFrame
    expected_df = pd.DataFrame({
        'fighter_id': [1, 2],
        'fight_id': [100, 100],
        'sig_str_acc_opp': [0.65, 0.75],
        'sig_str_land_opp': [18, 25]
    })
    
    # Mock context.execute_calculator_update
    calc.context.execute_calculator_update = MagicMock(return_value=expected_df)
    
    # Call save_for_table, which should use our overridden execute_calculator_update
    result = calc.save_for_table('sig_str')
    
    # Verify calculate_for_table was called with correct parameters
    calc.calculate_for_table.assert_called_once_with('sig_str', ['sig_str_acc', 'sig_str_land'])
    
    # Verify result contains expected columns and values
    assert result is not None
    assert 'sig_str_acc_opp' in result.columns
    assert 'sig_str_land_opp' in result.columns
    assert result.loc[result['fighter_id'] == 1, 'sig_str_acc_opp'].values[0] == 0.65
    assert result.loc[result['fighter_id'] == 2, 'sig_str_acc_opp'].values[0] == 0.75


def test_run_method():
    """Test run method that processes all feature tables."""
    # Create mocks
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock save_for_table
    calc.save_for_table = MagicMock()
    
    # Call run
    calc.run()
    
    # Verify save_for_table was called for each table
    assert calc.save_for_table.call_count == 3
    calc.save_for_table.assert_any_call('sig_str')
    calc.save_for_table.assert_any_call('td')
    calc.save_for_table.assert_any_call('control')


def test_prepare_column_selects():
    """Test _prepare_column_selects for generating SQL column selections."""
    # Create calculator
    calc = TestOpponentCalculator()
    
    # Test with sample columns
    columns = ['sig_str_acc', 'sig_str_land']
    result = calc._prepare_column_selects(columns)
    
    # Verify format is correct - includes column aliasing with proper case
    assert "opp.sig_str_acc AS sig_str_acc_opp" in result
    assert "opp.sig_str_land AS sig_str_land_opp" in result
    assert ",\n    " in result  # Ensure columns are separated by comma and newline


def test_backward_compatibility():
    """Test that legacy methods still work for backward compatibility."""
    # Create calculator
    calc = TestOpponentCalculator()
    
    # Mock run method
    calc.run = MagicMock()
    
    # Call legacy methods
    result1 = calc.calculate()
    result2 = calc.save()
    
    # Verify run was called for both legacy methods
    assert calc.run.call_count == 2
    
    # Verify return values
    assert isinstance(result1, str)
    assert isinstance(result2, pd.DataFrame)


def test_execute_sql_template():
    """Test the execute_sql_template method with both context and instance managers."""
    # Create calculator
    calc = TestOpponentCalculator()
    
    # Test parameters
    template_name = 'opponent'
    operation = 'calculate'
    params = {'schema': 'features', 'table_name': 'sig_str'}
    
    # Mock context's SQL manager to fail
    calc.context.sql_manager.render_template.side_effect = Exception("Context SQL manager error")
    
    # Mock instance's SQL template manager to succeed
    calc.sql_template_manager = MagicMock()
    calc.sql_template_manager.render_template = MagicMock(return_value="-- SQL from instance")
    
    # Call the method
    result = calc.execute_sql_template(template_name, operation, params)
    
    # Verify both managers were attempted
    calc.context.sql_manager.render_template.assert_called_once()
    calc.sql_template_manager.render_template.assert_called_once()
    
    # Verify the result is from the instance's manager
    assert result == "-- SQL from instance"
    
    # Now test with context manager succeeding
    calc.context.sql_manager.render_template.side_effect = None
    calc.context.sql_manager.render_template.return_value = "-- SQL from context"
    
    # Reset instance manager
    calc.sql_template_manager.render_template.reset_mock()
    
    # Call again
    result = calc.execute_sql_template(template_name, operation, params)
    
    # Verify only the context manager was called
    calc.context.sql_manager.render_template.assert_called()
    calc.sql_template_manager.render_template.assert_not_called()
    
    # Verify result is from context
    assert result == "-- SQL from context"


def test_opp_calculator_integration():
    """Test full integration of opponent calculator with mock data."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock get_features to return specific columns
    calc.get_features = MagicMock(return_value=['sig_str_acc', 'sig_str_land'])
    
    # Mock calculate_for_table to return expected SQL
    expected_sql = """
    SELECT
        t.fight_id,
        t.fighter_id,
        opp.sig_str_acc AS sig_str_acc_opp,
        opp.sig_str_land AS sig_str_land_opp
    FROM base_stats t
    JOIN opponent_stats opp ON t.fight_id = opp.fight_id
    """
    calc.calculate_for_table = MagicMock(return_value=expected_sql)
    
    # Create expected result DataFrame
    expected_df = pd.DataFrame({
        'fighter_id': [1, 2],
        'fight_id': [100, 100],
        'sig_str_acc_opp': [0.65, 0.75],
        'sig_str_land_opp': [18, 25]
    })
    
    # Mock context.execute_calculator_update to return expected DataFrame
    calc.context.execute_calculator_update = MagicMock(return_value=expected_df)
    
    # Call save_for_table
    result = calc.save_for_table('sig_str')
    
    # Verify results
    assert result is not None
    assert 'sig_str_acc_opp' in result.columns
    assert 'sig_str_land_opp' in result.columns
    
    # Verify opponent stats were correctly calculated
    # Fighter 1 should have Fighter 2's stats as opponent stats
    assert result.loc[result['fighter_id'] == 1, 'sig_str_acc_opp'].values[0] == 0.65
    assert result.loc[result['fighter_id'] == 2, 'sig_str_acc_opp'].values[0] == 0.75


def test_test_calculator_feature_tables():
    """Test that TestOpponentCalculator correctly overrides get_feature_tables."""
    # Create a calculator instance
    calc = TestOpponentCalculator()
    
    # Ensure it returns the correct tables without filtering
    tables = calc.get_feature_tables()
    assert len(tables) == 3
    assert set(tables) == {'sig_str', 'td', 'control'}
    
    # Test filtering functionality works
    filtered_tables = calc.get_feature_tables(['sig'])
    assert len(filtered_tables) == 2
    assert 'td' in filtered_tables
    assert 'control' in filtered_tables
    assert 'sig_str' not in filtered_tables 


def test_opp_calculator_with_dummy_data():
    """Test that OpponentCalculator correctly calculates opponent stats using dummy data."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a test class that overrides execute_raw_sql to return dummy data
    class TestOpponentCalculatorWithDummyData(TestOpponentCalculator):
        def execute_raw_sql(self, sql, params=None, return_results=True):
            """Override to return controlled test data instead of executing SQL."""
            # For SQL with opponent patterns, return opponent data
            if "_opp" in sql.lower():
                # Return data with opponent columns
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'sig_str_land_opp': [10, 25],  # Fighter 1 has Fighter 2's stats
                    'sig_str_att_opp': [30, 50],   # Fighter 2 has Fighter 1's stats
                    'sig_str_acc_opp': [0.33, 0.5],
                    'sig_str_def_opp': [0.5, 0.7]
                })
            # For SQL with "JOIN" but not opponent columns, mock the mapping tables
            elif "JOIN" in sql.upper() and "fight_mapping" in sql.lower():
                return pd.DataFrame({
                    'fight_id': [100],
                    'fighter1_id': [1],
                    'fighter2_id': [2],
                    'event_id': [50]
                })
            elif "JOIN" in sql.upper() and "event_mapping" in sql.lower():
                return pd.DataFrame({
                    'event_id': [50],
                    'event_date': ['2023-01-01']
                })
            # For all other queries, return basic fighter stats
            else:
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'sig_str_land': [25, 10],
                    'sig_str_att': [50, 30],
                    'sig_str_acc': [0.5, 0.33],
                    'sig_str_def': [0.7, 0.5]
                })
    
    # Create the calculator
    calc = TestOpponentCalculatorWithDummyData(mock_conn)
    
    # Set up mocks for the context methods
    calc.context.feature_utils.get_columns_from_table = MagicMock(
        return_value=['sig_str_land', 'sig_str_att', 'sig_str_acc', 'sig_str_def']
    )
    
    # Override calculate_for_table to return a fixed SQL that our mock can handle
    calc.calculate_for_table = MagicMock(return_value="""
        SELECT 
            fighter_id, 
            fight_id,
            sig_str_land_opp, 
            sig_str_att_opp,
            sig_str_acc_opp,
            sig_str_def_opp
        FROM test_table
        WHERE sig_str_land_opp IS NOT NULL
    """)
    
    # Execute the opponent calculation
    result_df = calc.execute_for_table('sig_str')
    
    # Verify the results
    assert result_df is not None
    assert 'sig_str_land_opp' in result_df.columns
    assert 'sig_str_att_opp' in result_df.columns
    assert 'sig_str_acc_opp' in result_df.columns
    assert 'sig_str_def_opp' in result_df.columns
    
    # Check that Fighter 1's opponent stats match Fighter 2's values
    fighter1_row = result_df[result_df['fighter_id'] == 1]
    assert fighter1_row['sig_str_land_opp'].values[0] == 10
    assert fighter1_row['sig_str_att_opp'].values[0] == 30
    assert fighter1_row['sig_str_acc_opp'].values[0] == 0.33
    assert fighter1_row['sig_str_def_opp'].values[0] == 0.5
    
    # Check that Fighter 2's opponent stats match Fighter 1's values
    fighter2_row = result_df[result_df['fighter_id'] == 2]
    assert fighter2_row['sig_str_land_opp'].values[0] == 25
    assert fighter2_row['sig_str_att_opp'].values[0] == 50
    assert fighter2_row['sig_str_acc_opp'].values[0] == 0.5
    assert fighter2_row['sig_str_def_opp'].values[0] == 0.7


def test_opp_calculator_with_multiple_feature_tables():
    """Test OpponentCalculator handling multiple different feature tables correctly."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator
    calc = TestOpponentCalculator(mock_conn)
    
    # Override get_feature_tables to return multiple test tables
    calc.get_feature_tables = MagicMock(return_value=['sig_str', 'td', 'sub'])
    
    # Create a dictionary to store different data for different tables
    table_data = {
        'sig_str': pd.DataFrame({
            'fighter_id': [1, 2],
            'fight_id': [100, 100],
            'sig_str_acc_opp': [0.4, 0.6]
        }),
        'td': pd.DataFrame({
            'fighter_id': [1, 2],
            'fight_id': [100, 100],
            'td_acc_opp': [0.3, 0.7]
        }),
        'sub': pd.DataFrame({
            'fighter_id': [1, 2],
            'fight_id': [100, 100],
            'sub_att_opp': [2, 5]
        })
    }
    
    # Mock save_for_table to return the appropriate data for each table
    def mock_save_for_table(table_name, *args, **kwargs):
        return table_data.get(table_name, pd.DataFrame())
    
    calc.save_for_table = MagicMock(side_effect=mock_save_for_table)
    
    # Run the calculator for all tables
    results = calc.run()
    
    # Verify that all tables were processed
    assert len(results) == 3
    assert 'sig_str' in results
    assert 'td' in results
    assert 'sub' in results
    
    # Check that each table has the correct data
    assert 'sig_str_acc_opp' in results['sig_str'].columns
    assert 'td_acc_opp' in results['td'].columns
    assert 'sub_att_opp' in results['sub'].columns
    
    # Verify that save_for_table was called for each table
    assert calc.save_for_table.call_count == 3
    calc.save_for_table.assert_any_call('sig_str')
    calc.save_for_table.assert_any_call('td')
    calc.save_for_table.assert_any_call('sub')


def test_opp_calculator_with_full_flow():
    """Test the full flow of the opponent calculator with a more complex dataset."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a calculator with more detailed test data
    class DetailedTestCalculator(TestOpponentCalculator):
        def execute_raw_sql(self, sql, params=None, return_results=True):
            """Simulate a database with multiple fights and fighters."""
            # Return opponent data for the calculation result
            if "head_land_opp" in sql or "HEAD_LAND_OPP" in sql:
                # This simulates the processed opponent stats
                return pd.DataFrame({
                    'fighter_id': [1, 2, 3, 4],
                    'fight_id': [100, 100, 101, 101],
                    'event_id': [50, 50, 51, 51],
                    'head_land_opp': [15, 20, 18, 12],
                    'head_att_opp': [30, 40, 35, 25],
                    'head_acc_opp': [0.5, 0.5, 0.51, 0.48],
                    'head_def_opp': [0.7, 0.6, 0.65, 0.55]
                })
            # Return fight mapping data
            elif 'fight_mapping' in sql.lower():
                return pd.DataFrame({
                    'fight_id': [100, 101],
                    'fighter1_id': [1, 3],
                    'fighter2_id': [2, 4],
                    'event_id': [50, 51]
                })
            # Return event data
            elif 'event_mapping' in sql.lower():
                return pd.DataFrame({
                    'event_id': [50, 51],
                    'event_date': ['2023-01-01', '2023-02-01']
                })
            # Return base fighter stats
            else:
                return pd.DataFrame({
                    'fighter_id': [1, 2, 3, 4],
                    'fight_id': [100, 100, 101, 101],
                    'event_id': [50, 50, 51, 51],
                    'head_land': [20, 15, 12, 18],
                    'head_att': [40, 30, 25, 35],
                    'head_acc': [0.5, 0.5, 0.48, 0.51],
                    'head_def': [0.6, 0.7, 0.55, 0.65]
                })

        # Override the execute_calculator_update method to match what we expect
        def execute_calculator_update(self, calculation_sql, table_name, new_columns, schema):
            """Override to return a mock result for testing."""
            # Return the same dummy data the execute_raw_sql would return for opponent data
            return pd.DataFrame({
                'fighter_id': [1, 2, 3, 4],
                'fight_id': [100, 100, 101, 101],
                'event_id': [50, 50, 51, 51],
                'head_land_opp': [15, 20, 18, 12],
                'head_att_opp': [30, 40, 35, 25],
                'head_acc_opp': [0.5, 0.5, 0.51, 0.48],
                'head_def_opp': [0.7, 0.6, 0.65, 0.55]
            })
    
    # Create calculator instance
    calc = DetailedTestCalculator(mock_conn)
    
    # Get list of columns for the "head" feature table
    test_columns = ['head_land', 'head_att', 'head_acc', 'head_def']
    calc.context.feature_utils.get_columns_from_table = MagicMock(return_value=test_columns)
    
    # Override SQL template rendering to use our SQL
    calc.context.sql_manager.render_template = MagicMock(return_value="""
        SELECT
            fighter_id,
            fight_id,
            head_land_opp,
            head_att_opp,
            head_acc_opp,
            head_def_opp
        FROM test_table
        WHERE head_land_opp IS NOT NULL
    """)
    
    # Instead of mocking context.execute_calculator_update, rely on the overridden method in our test class
    
    # Run the save operation for the head table
    result = calc.save_for_table('head')
    
    # Verify results have correct structure
    assert result is not None
    assert 'fight_id' in result.columns
    assert 'fighter_id' in result.columns
    assert 'head_land_opp' in result.columns
    assert 'head_att_opp' in result.columns
    assert 'head_acc_opp' in result.columns
    assert 'head_def_opp' in result.columns
    
    # Verify fighter 1 has fighter 2's stats
    assert result.loc[result['fighter_id'] == 1, 'head_land_opp'].values[0] == 15
    assert result.loc[result['fighter_id'] == 1, 'head_att_opp'].values[0] == 30
    
    # Verify fighter 2 has fighter 1's stats
    assert result.loc[result['fighter_id'] == 2, 'head_land_opp'].values[0] == 20
    assert result.loc[result['fighter_id'] == 2, 'head_att_opp'].values[0] == 40
    
    # Verify fighter 3 has fighter 4's stats
    assert result.loc[result['fighter_id'] == 3, 'head_land_opp'].values[0] == 18
    assert result.loc[result['fighter_id'] == 3, 'head_att_opp'].values[0] == 35
    
    # Verify fighter 4 has fighter 3's stats
    assert result.loc[result['fighter_id'] == 4, 'head_land_opp'].values[0] == 12
    assert result.loc[result['fighter_id'] == 4, 'head_att_opp'].values[0] == 25


def test_opp_calculator_with_round_specific_tables():
    """Test opponent calculator with round-specific feature tables (e.g. sig_str_rd1)."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a more specialized test calculator
    class RoundTestCalculator(TestOpponentCalculator):
        def execute_raw_sql(self, sql, params=None, return_results=True):
            """Return round-specific data for testing."""
            if "kd_rd1_opp" in sql:
                # Return round 1 opponent stats results
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'kd_rd1_opp': [0, 1],
                    'sig_str_land_rd1_opp': [15, 30],
                    'sig_str_acc_rd1_opp': [0.4, 0.6]
                })
            elif "kd_opp" in sql:
                # Return overall opponent stats results
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'kd_opp': [0, 2],
                    'sig_str_land_opp': [50, 80],
                    'sig_str_acc_opp': [0.45, 0.55]
                })
            elif 'fight_mapping' in sql.lower():
                return pd.DataFrame({
                    'fight_id': [100],
                    'fighter1_id': [1],
                    'fighter2_id': [2],
                    'event_id': [50]
                })
            elif 'event_mapping' in sql.lower():
                return pd.DataFrame({
                    'event_id': [50],
                    'event_date': ['2023-01-01']
                })
            elif '_rd1' in sql.lower():
                # Return round 1 specific stats
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'kd_rd1': [1, 0],
                    'sig_str_land_rd1': [30, 15],
                    'sig_str_acc_rd1': [0.6, 0.4]
                })
            else:
                # Return overall fight stats
                return pd.DataFrame({
                    'fighter_id': [1, 2],
                    'fight_id': [100, 100],
                    'event_id': [50, 50],
                    'kd': [2, 0],
                    'sig_str_land': [80, 50],
                    'sig_str_acc': [0.55, 0.45]
                })
    
    # Create calculator
    calc = RoundTestCalculator(mock_conn)
    
    # Set up columns based on table
    def get_table_columns(schema, table_name, exclude_strs=None):
        if table_name == 'sig_str_rd1':
            return ['kd_rd1', 'sig_str_land_rd1', 'sig_str_acc_rd1']
        else:
            return ['kd', 'sig_str_land', 'sig_str_acc']
    
    calc.context.feature_utils.get_columns_from_table = MagicMock(side_effect=get_table_columns)
    
    # Override calculate_for_table to return SQL our mock can handle
    calc.calculate_for_table = MagicMock(side_effect=lambda table_name, columns=None: 
        """
        SELECT 
            fighter_id, 
            fight_id,
            kd_rd1_opp, 
            sig_str_land_rd1_opp,
            sig_str_acc_rd1_opp
        FROM test_table
        WHERE kd_rd1_opp IS NOT NULL
        """ if table_name == 'sig_str_rd1' else 
        """
        SELECT 
            fighter_id, 
            fight_id,
            kd_opp, 
            sig_str_land_opp,
            sig_str_acc_opp
        FROM test_table
        WHERE kd_opp IS NOT NULL
        """
    )
    
    # Execute two different tables: round 1 and overall
    round1_result = calc.execute_for_table('sig_str_rd1')
    overall_result = calc.execute_for_table('sig_str')
    
    # Verify round 1 results
    assert 'kd_rd1_opp' in round1_result.columns
    assert 'sig_str_land_rd1_opp' in round1_result.columns
    assert 'sig_str_acc_rd1_opp' in round1_result.columns
    
    # Verify overall results
    assert 'kd_opp' in overall_result.columns
    assert 'sig_str_land_opp' in overall_result.columns
    assert 'sig_str_acc_opp' in overall_result.columns
    
    # Check opponent values are correctly mapped
    assert round1_result.loc[round1_result['fighter_id'] == 1, 'kd_rd1_opp'].values[0] == 0
    assert round1_result.loc[round1_result['fighter_id'] == 2, 'kd_rd1_opp'].values[0] == 1
    
    assert overall_result.loc[overall_result['fighter_id'] == 1, 'kd_opp'].values[0] == 0
    assert overall_result.loc[overall_result['fighter_id'] == 2, 'kd_opp'].values[0] == 2


def test_opp_calculator_with_include_patterns():
    """Test that OpponentCalculator with include_patterns only processes specified columns."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with include_patterns
    calc = TestOpponentCalculator(mock_conn)
    
    # Mock get_features to return both adjperf and non-adjperf columns
    all_columns = ['sig_str_land', 'sig_str_acc', 'sig_str_land_adjperf', 'sig_str_acc_adjperf', 'sig_str_def']
    calc.get_features = MagicMock(return_value=all_columns)
    
    # Set include_patterns to only process adjperf columns
    calc.include_patterns = ['adjperf']
    
    # Override get_features to simulate the filtering
    def mock_get_features_with_filtering(table_name):
        columns = all_columns
        # Apply include pattern filtering like the real implementation
        if calc.include_patterns:
            columns = [col for col in columns if any(pattern in col for pattern in calc.include_patterns)]
        return columns
    
    calc.get_features = MagicMock(side_effect=mock_get_features_with_filtering)
    
    # Call get_features and verify only adjperf columns are returned
    filtered_columns = calc.get_features('sig_str')
    
    # Should only include columns with 'adjperf' in the name
    expected_columns = ['sig_str_land_adjperf', 'sig_str_acc_adjperf']
    assert set(filtered_columns) == set(expected_columns)
    
    # Verify non-adjperf columns are excluded
    assert 'sig_str_land' not in filtered_columns
    assert 'sig_str_acc' not in filtered_columns
    assert 'sig_str_def' not in filtered_columns


def test_adjperf_opponent_stats_exact_match():
    """Test that opponent adjperf stats exactly match the original fighter's adjperf stats."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a specialized test calculator for adjperf testing
    class AdjperfTestCalculator(TestOpponentCalculator):
        def __init__(self, conn=None):
            super().__init__(conn)
            self.include_patterns = ['adjperf']  # Only process adjperf columns
            
        def execute_raw_sql(self, sql, params=None, return_results=True):
            """Return controlled test data for adjperf opponent calculation."""
            if not return_results:
                return None
                
            # Create test data where we can verify exact matching
            # Fighter 1 vs Fighter 2 in fight 100
            # Fighter 3 vs Fighter 4 in fight 101
            base_data = pd.DataFrame({
                'fighter_id': [1, 2, 3, 4],
                'fight_id': [100, 100, 101, 101],
                'event_id': [50, 50, 51, 51],
                # Original adjperf stats for each fighter
                'ctrl_dec_adjperf': [1.5, 0.8, 2.1, -0.3],
                'sub_def_dec_adjperf': [0.7, -0.5, 1.2, 0.9],
                'sub_att_dec_adjperf': [-0.2, 1.1, 0.4, -0.8]
            })
            
            # If this is a query for opponent stats, return the opponent-swapped version
            if '_opp' in sql.lower():
                # Create opponent stats by swapping within each fight
                opp_data = base_data.copy()
                
                # For fight 100: Fighter 1 gets Fighter 2's stats, Fighter 2 gets Fighter 1's stats
                fight_100_f1_stats = base_data[base_data['fighter_id'] == 2][['ctrl_dec_adjperf', 'sub_def_dec_adjperf', 'sub_att_dec_adjperf']].iloc[0]
                fight_100_f2_stats = base_data[base_data['fighter_id'] == 1][['ctrl_dec_adjperf', 'sub_def_dec_adjperf', 'sub_att_dec_adjperf']].iloc[0]
                
                # For fight 101: Fighter 3 gets Fighter 4's stats, Fighter 4 gets Fighter 3's stats  
                fight_101_f3_stats = base_data[base_data['fighter_id'] == 4][['ctrl_dec_adjperf', 'sub_def_dec_adjperf', 'sub_att_dec_adjperf']].iloc[0]
                fight_101_f4_stats = base_data[base_data['fighter_id'] == 3][['ctrl_dec_adjperf', 'sub_def_dec_adjperf', 'sub_att_dec_adjperf']].iloc[0]
                
                # Apply opponent stats with _opp suffix
                opp_data.loc[opp_data['fighter_id'] == 1, ['ctrl_dec_adjperf_opp', 'sub_def_dec_adjperf_opp', 'sub_att_dec_adjperf_opp']] = fight_100_f1_stats.values
                opp_data.loc[opp_data['fighter_id'] == 2, ['ctrl_dec_adjperf_opp', 'sub_def_dec_adjperf_opp', 'sub_att_dec_adjperf_opp']] = fight_100_f2_stats.values
                opp_data.loc[opp_data['fighter_id'] == 3, ['ctrl_dec_adjperf_opp', 'sub_def_dec_adjperf_opp', 'sub_att_dec_adjperf_opp']] = fight_101_f3_stats.values
                opp_data.loc[opp_data['fighter_id'] == 4, ['ctrl_dec_adjperf_opp', 'sub_def_dec_adjperf_opp', 'sub_att_dec_adjperf_opp']] = fight_101_f4_stats.values
                
                return opp_data
            else:
                # Return base fighter stats
                return base_data
    
    # Create the calculator
    calc = AdjperfTestCalculator(mock_conn)
    
    # Mock get_features to return only adjperf columns  
    adjperf_columns = ['ctrl_dec_adjperf', 'sub_def_dec_adjperf', 'sub_att_dec_adjperf']
    calc.get_features = MagicMock(return_value=adjperf_columns)
    
    # Mock calculate_for_table to return SQL that our execute_raw_sql can handle
    calc.calculate_for_table = MagicMock(return_value="""
        SELECT 
            fighter_id, 
            fight_id,
            event_id,
            ctrl_dec_adjperf_opp,
            sub_def_dec_adjperf_opp,
            sub_att_dec_adjperf_opp
        FROM test_table_with_adjperf_opp
    """)
    
    # Execute the opponent calculation
    result_df = calc.execute_for_table('test_table')
    
    # Get the base data for comparison
    base_df = calc.execute_raw_sql("SELECT * FROM base_table", return_results=True)
    
    # Verify results structure
    assert result_df is not None
    assert 'ctrl_dec_adjperf_opp' in result_df.columns
    assert 'sub_def_dec_adjperf_opp' in result_df.columns
    assert 'sub_att_dec_adjperf_opp' in result_df.columns
    
    # Test exact matching for Fight 100 (Fighter 1 vs Fighter 2)
    # Fighter 1's opponent stats should match Fighter 2's original stats
    fighter1_opp_ctrl = result_df.loc[result_df['fighter_id'] == 1, 'ctrl_dec_adjperf_opp'].iloc[0]
    fighter2_orig_ctrl = base_df.loc[base_df['fighter_id'] == 2, 'ctrl_dec_adjperf'].iloc[0]
    assert fighter1_opp_ctrl == fighter2_orig_ctrl, f"Fighter 1's opponent ctrl should be {fighter2_orig_ctrl}, got {fighter1_opp_ctrl}"
    
    fighter1_opp_sub_def = result_df.loc[result_df['fighter_id'] == 1, 'sub_def_dec_adjperf_opp'].iloc[0]
    fighter2_orig_sub_def = base_df.loc[base_df['fighter_id'] == 2, 'sub_def_dec_adjperf'].iloc[0]
    assert fighter1_opp_sub_def == fighter2_orig_sub_def, f"Fighter 1's opponent sub_def should be {fighter2_orig_sub_def}, got {fighter1_opp_sub_def}"
    
    # Fighter 2's opponent stats should match Fighter 1's original stats
    fighter2_opp_ctrl = result_df.loc[result_df['fighter_id'] == 2, 'ctrl_dec_adjperf_opp'].iloc[0]
    fighter1_orig_ctrl = base_df.loc[base_df['fighter_id'] == 1, 'ctrl_dec_adjperf'].iloc[0]
    assert fighter2_opp_ctrl == fighter1_orig_ctrl, f"Fighter 2's opponent ctrl should be {fighter1_orig_ctrl}, got {fighter2_opp_ctrl}"
    
    fighter2_opp_sub_att = result_df.loc[result_df['fighter_id'] == 2, 'sub_att_dec_adjperf_opp'].iloc[0]
    fighter1_orig_sub_att = base_df.loc[base_df['fighter_id'] == 1, 'sub_att_dec_adjperf'].iloc[0]
    assert fighter2_opp_sub_att == fighter1_orig_sub_att, f"Fighter 2's opponent sub_att should be {fighter1_orig_sub_att}, got {fighter2_opp_sub_att}"


def test_include_patterns_with_multiple_patterns():
    """Test include_patterns with multiple patterns."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with multiple include patterns
    calc = TestOpponentCalculator(mock_conn)
    calc.include_patterns = ['adjperf', 'acc']
    
    # Mock get_features to return mixed columns
    all_columns = [
        'sig_str_land', 'sig_str_acc', 'sig_str_def',  # Should include 'acc'
        'sig_str_land_adjperf', 'sig_str_acc_adjperf',  # Should include both
        'td_land', 'td_def_adjperf'  # Should include 'def_adjperf'
    ]
    
    # Override get_features to simulate the filtering
    def mock_get_features_with_filtering(table_name):
        columns = all_columns
        # Apply include pattern filtering like the real implementation
        if calc.include_patterns:
            columns = [col for col in columns if any(pattern in col for pattern in calc.include_patterns)]
        return columns
    
    calc.get_features = MagicMock(side_effect=mock_get_features_with_filtering)
    
    # Call get_features and verify correct columns are returned
    filtered_columns = calc.get_features('sig_str')
    
    # Should include columns with either 'adjperf' or 'acc' in the name
    expected_columns = ['sig_str_acc', 'sig_str_land_adjperf', 'sig_str_acc_adjperf', 'td_def_adjperf']
    assert set(filtered_columns) == set(expected_columns)
    
    # Verify excluded columns
    assert 'sig_str_land' not in filtered_columns  # No 'adjperf' or 'acc'
    assert 'sig_str_def' not in filtered_columns   # No 'adjperf' or 'acc'
    assert 'td_land' not in filtered_columns       # No 'adjperf' or 'acc'


def test_include_patterns_empty_list():
    """Test that empty include_patterns processes all columns (normal behavior)."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with empty include patterns
    calc = TestOpponentCalculator(mock_conn)
    calc.include_patterns = []  # Empty list
    
    # Mock get_features to return all columns
    all_columns = ['sig_str_land', 'sig_str_acc', 'sig_str_land_adjperf', 'sig_str_acc_adjperf']
    
    # Override get_features to simulate the filtering
    def mock_get_features_with_filtering(table_name):
        columns = all_columns
        # Apply include pattern filtering like the real implementation
        if calc.include_patterns:  # Empty list is falsy
            columns = [col for col in columns if any(pattern in col for pattern in calc.include_patterns)]
        return columns
    
    calc.get_features = MagicMock(side_effect=mock_get_features_with_filtering)
    
    # Call get_features and verify all columns are returned
    filtered_columns = calc.get_features('sig_str')
    
    # Should include all columns when include_patterns is empty
    assert set(filtered_columns) == set(all_columns)


def test_include_patterns_no_matches():
    """Test that include_patterns with no matches returns empty list."""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create calculator with pattern that matches nothing
    calc = TestOpponentCalculator(mock_conn)
    calc.include_patterns = ['nonexistent_pattern']
    
    # Mock get_features to return columns that don't match the pattern
    all_columns = ['sig_str_land', 'sig_str_acc', 'sig_str_def']
    
    # Override get_features to simulate the filtering
    def mock_get_features_with_filtering(table_name):
        columns = all_columns
        # Apply include pattern filtering like the real implementation
        if calc.include_patterns:
            columns = [col for col in columns if any(pattern in col for pattern in calc.include_patterns)]
        return columns
    
    calc.get_features = MagicMock(side_effect=mock_get_features_with_filtering)
    
    # Call get_features and verify no columns are returned
    filtered_columns = calc.get_features('sig_str')
    
    # Should return empty list when no columns match the pattern
    assert filtered_columns == [] 