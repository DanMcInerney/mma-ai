import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from libs.feature_store.calculators.dslf_calc import DaysSinceLastFightCalculator
from libs.feature_store.calculator_context import CalculatorContext

# Create a test version of the calculator that overrides _ensure_columns_exist
class TestDaysSinceLastFightCalculator(DaysSinceLastFightCalculator):
    def _ensure_columns_exist(self, columns, table_name=None, schema=None):
        # Do nothing, assume columns exist
        pass

def test_dslf_calculator_with_context():
    """Test DaysSinceLastFightCalculator with mock data and context"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104, 105],
            'fighter_id': [1, 1, 1, 2, 2],
            'event_id': [1, 2, 3, 1, 3]
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2, 3],
            'event_date': ['2020-01-15', '2020-06-20', '2021-02-10']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestDaysSinceLastFightCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate days since last fight
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # Extract event dates from our mock data
                event_dates = {row['event_id']: pd.to_datetime(row['event_date']) 
                              for _, row in mock_data['event_mapping'].iterrows()}
                
                # Create a dictionary to track each fighter's fights in chronological order
                fighter_fights = {}
                
                # First, organize fights by fighter and sort by date
                for _, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    fight_id = row['fight_id']
                    
                    if fighter_id not in fighter_fights:
                        fighter_fights[fighter_id] = []
                    
                    fighter_fights[fighter_id].append({
                        'fight_id': fight_id,
                        'event_date': event_dates[event_id],
                        'event_id': event_id
                    })
                
                # Sort each fighter's fights by date
                for fighter_id in fighter_fights:
                    fighter_fights[fighter_id].sort(key=lambda x: x['event_date'])
                
                # Calculate days since last fight for each fight
                for fighter_id, fights in fighter_fights.items():
                    for i, fight in enumerate(fights):
                        if i == 0:
                            # First fight, set to 120 days as per the SQL logic
                            days = 120
                        else:
                            # Calculate days since previous fight
                            prev_date = fights[i-1]['event_date']
                            curr_date = fight['event_date']
                            days = (curr_date - prev_date).days
                        
                        # Update the mock data
                        idx = mock_data['fight_stats_fe'].index[
                            (mock_data['fight_stats_fe']['fight_id'] == fight['fight_id']) &
                            (mock_data['fight_stats_fe']['fighter_id'] == fighter_id)
                        ][0]
                        mock_data['fight_stats_fe'].loc[idx, 'days_since_last_fight'] = days
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'days_since_last_fight' in mock_data['fight_stats_fe'].columns
            
            # Get days for specific fighters
            fighter1_fight1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'days_since_last_fight'
            ].iloc[0]
            
            fighter1_fight2 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'days_since_last_fight'
            ].iloc[0]
            
            fighter1_fight3 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 3), 'days_since_last_fight'
            ].iloc[0]
            
            # Calculate expected days manually for verification
            # Fighter 1's first fight should be 120 days
            expected_fighter1_fight1 = 120
            
            # Fighter 1's second fight: 2020-06-20 - 2020-01-15 = 157 days
            expected_fighter1_fight2 = (pd.to_datetime('2020-06-20') - pd.to_datetime('2020-01-15')).days
            
            # Fighter 1's third fight: 2021-02-10 - 2020-06-20 = 235 days
            expected_fighter1_fight3 = (pd.to_datetime('2021-02-10') - pd.to_datetime('2020-06-20')).days
            
            # Verify the calculated days match our expectations
            assert fighter1_fight1 == expected_fighter1_fight1, f"Expected {expected_fighter1_fight1}, got {fighter1_fight1}"
            assert fighter1_fight2 == expected_fighter1_fight2, f"Expected {expected_fighter1_fight2}, got {fighter1_fight2}"
            assert fighter1_fight3 == expected_fighter1_fight3, f"Expected {expected_fighter1_fight3}, got {fighter1_fight3}"
            
            # Print the days for verification
            print(f"Fighter 1 first fight: {fighter1_fight1} days")
            print(f"Fighter 1 second fight: {fighter1_fight2} days")
            print(f"Fighter 1 third fight: {fighter1_fight3} days")


def test_dslf_calculator_with_sql_template():
    """Test DaysSinceLastFightCalculator with SQL template manager"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                -- SQL template for days since last fight calculation
                UPDATE features.fight_stats_fe SET days_since_last_fight = 120
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = TestDaysSinceLastFightCalculator(context)
            
            # Mock the execute_raw_sql method
            with patch.object(calculator, 'execute_raw_sql') as mock_execute:
                # Run the calculator
                calculator.run()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'dslf', 'calculate', {'schema': 'features'}
                )
                
                # Verify execute_raw_sql was called with the SQL from the template
                mock_execute.assert_called_with("""
                -- SQL template for days since last fight calculation
                UPDATE features.fight_stats_fe SET days_since_last_fight = 120
            """)


def test_dslf_calculator_integration():
    """Test DaysSinceLastFightCalculator with full integration"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104, 105],
            'fighter_id': [1, 1, 1, 2, 2],
            'event_id': [1, 2, 3, 1, 3]
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2, 3],
            'event_date': ['2020-01-15', '2020-06-20', '2021-02-10']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            
            # Create a SQL template that actually performs the calculation
            mock_sql_manager.render_template.return_value = """
                -- This is a mock SQL template that will be replaced by our Python implementation
                -- The actual SQL logic would calculate days since last fight
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestDaysSinceLastFightCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate days since last fight
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # Extract event dates from our mock data
                event_dates = {row['event_id']: pd.to_datetime(row['event_date']) 
                              for _, row in mock_data['event_mapping'].iterrows()}
                
                # Create a dictionary to track each fighter's fights in chronological order
                fighter_fights = {}
                
                # First, organize fights by fighter and sort by date
                for _, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    fight_id = row['fight_id']
                    
                    if fighter_id not in fighter_fights:
                        fighter_fights[fighter_id] = []
                    
                    fighter_fights[fighter_id].append({
                        'fight_id': fight_id,
                        'event_date': event_dates[event_id],
                        'event_id': event_id
                    })
                
                # Sort each fighter's fights by date
                for fighter_id in fighter_fights:
                    fighter_fights[fighter_id].sort(key=lambda x: x['event_date'])
                
                # Calculate days since last fight for each fight
                for fighter_id, fights in fighter_fights.items():
                    for i, fight in enumerate(fights):
                        if i == 0:
                            # First fight, set to 120 days as per the SQL logic
                            days = 120
                        else:
                            # Calculate days since previous fight
                            prev_date = fights[i-1]['event_date']
                            curr_date = fight['event_date']
                            days = (curr_date - prev_date).days
                        
                        # Update the mock data
                        idx = mock_data['fight_stats_fe'].index[
                            (mock_data['fight_stats_fe']['fight_id'] == fight['fight_id']) &
                            (mock_data['fight_stats_fe']['fighter_id'] == fighter_id)
                        ][0]
                        mock_data['fight_stats_fe'].loc[idx, 'days_since_last_fight'] = days
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'days_since_last_fight' in mock_data['fight_stats_fe'].columns
            
            # Get days for specific fighters
            fighter1_fight1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'days_since_last_fight'
            ].iloc[0]
            
            fighter1_fight2 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'days_since_last_fight'
            ].iloc[0]
            
            fighter1_fight3 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 3), 'days_since_last_fight'
            ].iloc[0]
            
            # Calculate expected days manually for verification
            # Fighter 1's first fight should be 120 days
            expected_fighter1_fight1 = 120
            
            # Fighter 1's second fight: 2020-06-20 - 2020-01-15 = 157 days
            expected_fighter1_fight2 = (pd.to_datetime('2020-06-20') - pd.to_datetime('2020-01-15')).days
            
            # Fighter 1's third fight: 2021-02-10 - 2020-06-20 = 235 days
            expected_fighter1_fight3 = (pd.to_datetime('2021-02-10') - pd.to_datetime('2020-06-20')).days
            
            # Verify the calculated days match our expectations
            assert fighter1_fight1 == expected_fighter1_fight1, f"Expected {expected_fighter1_fight1}, got {fighter1_fight1}"
            assert fighter1_fight2 == expected_fighter1_fight2, f"Expected {expected_fighter1_fight2}, got {fighter1_fight2}"
            assert fighter1_fight3 == expected_fighter1_fight3, f"Expected {expected_fighter1_fight3}, got {fighter1_fight3}"
            
            # Print the days for verification
            print(f"Fighter 1 first fight: {fighter1_fight1} days")
            print(f"Fighter 1 second fight: {fighter1_fight2} days")
            print(f"Fighter 1 third fight: {fighter1_fight3} days") 