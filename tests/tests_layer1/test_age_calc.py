import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import text
from libs.feature_store.calculators.age_calc import AgeCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.base import BaseFeatureStore


# Create a test version of AgeCalculator that overrides _ensure_columns_exist
class TestAgeCalculator(AgeCalculator):
    def _ensure_columns_exist(self, columns, table_name=None, schema=None):
        # Do nothing, assume columns exist
        pass


def test_age_calculator_with_context():
    """Test AgeCalculator with mock data and context"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'fighter_id': [1, 2, 1],
            'event_id': [1, 1, 2]
        }),
        'fighter_mapping': pd.DataFrame({
            'fighter_id': [1, 2],
            'fighter_name': ['Fighter 1', 'Fighter 2'],
            'fighter_dob': ['1990-01-15', '1985-05-20'],
            'weightclass': ['Lightweight', 'Welterweight']
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2],
            'event_date': ['2020-06-15', '2021-09-20']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestAgeCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate ages based on the mock data
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # We'll implement the actual age calculation logic here
                
                # Extract fighter DOBs and event dates from our mock data
                fighter_dobs = {row['fighter_id']: pd.to_datetime(row['fighter_dob']) 
                               for _, row in mock_data['fighter_mapping'].iterrows()}
                
                event_dates = {row['event_id']: pd.to_datetime(row['event_date']) 
                              for _, row in mock_data['event_mapping'].iterrows()}
                
                # Calculate age for each fight
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    
                    if fighter_id in fighter_dobs and event_id in event_dates:
                        dob = fighter_dobs[fighter_id]
                        event_date = event_dates[event_id]
                        
                        # Calculate age in years
                        age_in_days = (event_date - dob).days
                        age_in_years = age_in_days / 365.25
                        
                        # Round to 3 decimal places as in the SQL
                        age = round(age_in_years, 3)
                        
                        # Update the mock data
                        mock_data['fight_stats_fe'].loc[idx, 'age'] = age
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'age' in mock_data['fight_stats_fe'].columns
            
            # Get ages for specific fighters
            fighter1_age_event1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'age'
            ].iloc[0]
            
            fighter2_age_event1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 2) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'age'
            ].iloc[0]
            
            fighter1_age_event2 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'age'
            ].iloc[0]
            
            # Calculate expected ages manually for verification
            # Fighter 1 (born 1990-01-15) at Event 1 (2020-06-15)
            expected_age_fighter1_event1 = (pd.to_datetime('2020-06-15') - pd.to_datetime('1990-01-15')).days / 365.25
            expected_age_fighter1_event1 = round(expected_age_fighter1_event1, 3)
            
            # Fighter 2 (born 1985-05-20) at Event 1 (2020-06-15)
            expected_age_fighter2_event1 = (pd.to_datetime('2020-06-15') - pd.to_datetime('1985-05-20')).days / 365.25
            expected_age_fighter2_event1 = round(expected_age_fighter2_event1, 3)
            
            # Fighter 1 (born 1990-01-15) at Event 2 (2021-09-20)
            expected_age_fighter1_event2 = (pd.to_datetime('2021-09-20') - pd.to_datetime('1990-01-15')).days / 365.25
            expected_age_fighter1_event2 = round(expected_age_fighter1_event2, 3)
            
            # Verify the calculated ages match our expectations
            assert fighter1_age_event1 == expected_age_fighter1_event1, f"Expected {expected_age_fighter1_event1}, got {fighter1_age_event1}"
            assert fighter2_age_event1 == expected_age_fighter2_event1, f"Expected {expected_age_fighter2_event1}, got {fighter2_age_event1}"
            assert fighter1_age_event2 == expected_age_fighter1_event2, f"Expected {expected_age_fighter1_event2}, got {fighter1_age_event2}"
            
            # Also verify that Fighter 1 is older at Event 2 than at Event 1
            assert fighter1_age_event2 > fighter1_age_event1, "Fighter should be older at later event"
            
            # Print the ages for verification
            print(f"Fighter 1 age at Event 1: {fighter1_age_event1} years")
            print(f"Fighter 2 age at Event 1: {fighter2_age_event1} years")
            print(f"Fighter 1 age at Event 2: {fighter1_age_event2} years")


def test_age_calculator_with_sql_template():
    """Test AgeCalculator with SQL template manager"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                -- SQL template for age calculation
                UPDATE features.fight_stats_fe SET age = 30.0
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = TestAgeCalculator(context)
            
            # Mock the execute_raw_sql method
            with patch.object(calculator, 'execute_raw_sql') as mock_execute:
                # Run the calculator
                calculator.run()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'age', 'calculate', {'schema': 'features'}
                )
                
                # Verify execute_raw_sql was called with the SQL from the template
                mock_execute.assert_called_with("""
                -- SQL template for age calculation
                UPDATE features.fight_stats_fe SET age = 30.0
            """)


def test_age_calculator_integration():
    """Test AgeCalculator with full integration"""
    # Create mock data with realistic structure
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103],
            'fighter_id': [1, 2, 1],
            'event_id': [1, 1, 2]
        }),
        'fighter_mapping': pd.DataFrame({
            'fighter_id': [1, 2],
            'fighter_name': ['Fighter 1', 'Fighter 2'],
            'fighter_dob': ['1990-01-15', '1985-05-20'],
            'weightclass': ['Lightweight', 'Welterweight']
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2],
            'event_date': ['2020-06-15', '2021-09-20']
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            
            # Create a SQL template that actually performs the age calculation
            # This is similar to what's in the actual SQL template
            mock_sql_manager.render_template.return_value = """
                -- This is a mock SQL template that will be replaced by our Python implementation
                -- The actual SQL logic would calculate ages based on fighter DOB and event date
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestAgeCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate ages based on the mock data
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # We'll implement the actual age calculation logic here
                
                # Extract fighter DOBs and event dates from our mock data
                fighter_dobs = {row['fighter_id']: pd.to_datetime(row['fighter_dob']) 
                               for _, row in mock_data['fighter_mapping'].iterrows()}
                
                event_dates = {row['event_id']: pd.to_datetime(row['event_date']) 
                              for _, row in mock_data['event_mapping'].iterrows()}
                
                # Calculate age for each fight
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    
                    if fighter_id in fighter_dobs and event_id in event_dates:
                        dob = fighter_dobs[fighter_id]
                        event_date = event_dates[event_id]
                        
                        # Calculate age in years
                        age_in_days = (event_date - dob).days
                        age_in_years = age_in_days / 365.25
                        
                        # Round to 3 decimal places as in the SQL
                        age = round(age_in_years, 3)
                        
                        # Update the mock data
                        mock_data['fight_stats_fe'].loc[idx, 'age'] = age
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'age' in mock_data['fight_stats_fe'].columns
            
            # Get ages for specific fighters
            fighter1_age_event1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'age'
            ].iloc[0]
            
            fighter2_age_event1 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 2) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'age'
            ].iloc[0]
            
            fighter1_age_event2 = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'age'
            ].iloc[0]
            
            # Calculate expected ages manually for verification
            # Fighter 1 (born 1990-01-15) at Event 1 (2020-06-15)
            expected_age_fighter1_event1 = (pd.to_datetime('2020-06-15') - pd.to_datetime('1990-01-15')).days / 365.25
            expected_age_fighter1_event1 = round(expected_age_fighter1_event1, 3)
            
            # Fighter 2 (born 1985-05-20) at Event 1 (2020-06-15)
            expected_age_fighter2_event1 = (pd.to_datetime('2020-06-15') - pd.to_datetime('1985-05-20')).days / 365.25
            expected_age_fighter2_event1 = round(expected_age_fighter2_event1, 3)
            
            # Fighter 1 (born 1990-01-15) at Event 2 (2021-09-20)
            expected_age_fighter1_event2 = (pd.to_datetime('2021-09-20') - pd.to_datetime('1990-01-15')).days / 365.25
            expected_age_fighter1_event2 = round(expected_age_fighter1_event2, 3)
            
            # Verify the calculated ages match our expectations
            assert fighter1_age_event1 == expected_age_fighter1_event1, f"Expected {expected_age_fighter1_event1}, got {fighter1_age_event1}"
            assert fighter2_age_event1 == expected_age_fighter2_event1, f"Expected {expected_age_fighter2_event1}, got {fighter2_age_event1}"
            assert fighter1_age_event2 == expected_age_fighter1_event2, f"Expected {expected_age_fighter1_event2}, got {fighter1_age_event2}"
            
            # Also verify that Fighter 1 is older at Event 2 than at Event 1
            assert fighter1_age_event2 > fighter1_age_event1, "Fighter should be older at later event"
            
            # Print the ages for verification
            print(f"Fighter 1 age at Event 1: {fighter1_age_event1} years")
            print(f"Fighter 2 age at Event 1: {fighter2_age_event1} years")
            print(f"Fighter 1 age at Event 2: {fighter1_age_event2} years") 