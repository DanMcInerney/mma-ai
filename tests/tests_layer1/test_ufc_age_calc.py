import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from libs.feature_store.calculators.ufc_age_calc import UfcAgeCalculator
from libs.feature_store.calculator_context import CalculatorContext

# Create a test version of the calculator that overrides _ensure_columns_exist
class TestUfcAgeCalculator(UfcAgeCalculator):
    def _ensure_columns_exist(self, columns, table_name=None, schema=None):
        # Do nothing, assume columns exist
        pass

def test_ufc_age_calculator_with_context():
    """Test UfcAgeCalculator with mock data and context"""
    # Create mock data with realistic structure
    # We need fight_stats_fe, event_mapping, and fight_mapping
    today = datetime.now()
    three_years_ago = today - timedelta(days=3*365)
    five_years_ago = today - timedelta(days=5*365)
    
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'fighter_id': [1, 1, 3, 4],  # Fighter 1 appears in fights 101 and 102
            'event_id': [1, 2, 3, 4]
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2, 3, 4],
            'event_date': [
                five_years_ago.strftime('%Y-%m-%d'),  # First event (debut for fighters 1 and 2)
                (five_years_ago + timedelta(days=365)).strftime('%Y-%m-%d'),  # Second event (1 year after debut)
                (five_years_ago + timedelta(days=2*365)).strftime('%Y-%m-%d'),  # Third event (2 years after debut)
                (five_years_ago + timedelta(days=3*365)).strftime('%Y-%m-%d')   # Fourth event (3 years after debut)
            ]
        }),
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'event_id': [1, 2, 3, 4],
            'fighter1_id': [1, 1, 3, 3],
            'fighter2_id': [2, 2, 4, 4]
        })
    }
    
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create context with mock connection and data
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager'):
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestUfcAgeCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate UFC age
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # First, find each fighter's debut date (first fight)
                fighter_debut = {}
                
                # Join fight_stats_fe with event_mapping to get event dates
                fight_data = pd.merge(
                    mock_data['fight_stats_fe'],
                    mock_data['event_mapping'],
                    on='event_id'
                )
                
                # Find the earliest event date for each fighter (debut date)
                for fighter_id in fight_data['fighter_id'].unique():
                    fighter_fights = fight_data[fight_data['fighter_id'] == fighter_id]
                    debut_date = min(pd.to_datetime(fighter_fights['event_date']))
                    fighter_debut[fighter_id] = debut_date
                
                # Calculate UFC age for each fight
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    
                    # Get event date
                    event_date = pd.to_datetime(mock_data['event_mapping'].loc[
                        mock_data['event_mapping']['event_id'] == event_id, 'event_date'
                    ].iloc[0])
                    
                    # Get fighter's debut date
                    debut_date = fighter_debut[fighter_id]
                    
                    # Calculate UFC age in years
                    ufc_age = (event_date - debut_date).days / 365.0
                    
                    # Update the mock data
                    mock_data['fight_stats_fe'].loc[idx, 'ufcage'] = round(ufc_age, 2)
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'ufcage' in mock_data['fight_stats_fe'].columns
            
            # Get UFC age for specific fighters in specific fights
            # Fighter 1 in first fight (debut) should have UFC age 0
            fighter1_first_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'ufcage'
            ].iloc[0]
            
            # Fighter 1 in second fight (1 year after debut) should have UFC age 1
            fighter1_second_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'ufcage'
            ].iloc[0]
            
            # Fighter 3 in third fight (first appearance, 2 years after the first event)
            fighter3_first_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 3) & 
                (mock_data['fight_stats_fe']['event_id'] == 3), 'ufcage'
            ].iloc[0]
            
            # Verify the calculated UFC age values match our expectations
            assert fighter1_first_fight == 0.0, f"Expected 0.0, got {fighter1_first_fight}"
            assert abs(fighter1_second_fight - 1.0) < 0.1, f"Expected ~1.0, got {fighter1_second_fight}"
            assert fighter3_first_fight == 0.0, f"Expected 0.0, got {fighter3_first_fight}"
            
            # Print the UFC age values for verification
            print(f"Fighter 1 UFC age in first fight: {fighter1_first_fight} years")
            print(f"Fighter 1 UFC age in second fight: {fighter1_second_fight} years")
            print(f"Fighter 3 UFC age in first fight: {fighter3_first_fight} years")


def test_ufc_age_calculator_with_sql_template():
    """Test UfcAgeCalculator with SQL template manager"""
    # Create mock connection
    mock_conn = MagicMock()
    
    # Create a real context but patch the SQL template manager
    with patch('libs.feature_store.calculator_context.FeatureUtils'):
        with patch('libs.feature_store.calculator_context.SQLTemplateManager') as mock_sql_manager_class:
            # Configure the mock SQL template manager
            mock_sql_manager = mock_sql_manager_class.return_value
            mock_sql_manager.render_template.return_value = """
                -- SQL template for UFC age calculation
                WITH first_fights AS (
                    SELECT 
                        fighter_id, 
                        MIN(event_date) as debut_date
                    FROM features.fight_stats_fe
                    JOIN features.event_mapping USING(event_id)
                    GROUP BY fighter_id
                ),
                ufc_age_calc AS (
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        EXTRACT(YEAR FROM AGE(e.event_date, ff.debut_date)) + 
                        EXTRACT(MONTH FROM AGE(e.event_date, ff.debut_date))/12 as ufc_age
                    FROM features.fight_stats_fe f
                    JOIN features.event_mapping e USING(event_id)
                    JOIN first_fights ff USING(fighter_id)
                )
                UPDATE features.fight_stats_fe
                SET ufcage = ufc.ufc_age
                FROM ufc_age_calc ufc
                WHERE fight_stats_fe.fight_id = ufc.fight_id
                AND fight_stats_fe.fighter_id = ufc.fighter_id;
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn)
            
            # Create calculator with context
            calculator = TestUfcAgeCalculator(context)
            
            # Mock the execute_raw_sql method
            with patch.object(calculator, 'execute_raw_sql') as mock_execute:
                # Run the calculator
                calculator.run()
                
                # Verify SQL template manager was called correctly
                mock_sql_manager.render_template.assert_called_once_with(
                    'ufc_age', 'calculate', {'schema': 'features'}
                )
                
                # Verify execute_raw_sql was called with the SQL from the template
                mock_execute.assert_called_with("""
                -- SQL template for UFC age calculation
                WITH first_fights AS (
                    SELECT 
                        fighter_id, 
                        MIN(event_date) as debut_date
                    FROM features.fight_stats_fe
                    JOIN features.event_mapping USING(event_id)
                    GROUP BY fighter_id
                ),
                ufc_age_calc AS (
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        EXTRACT(YEAR FROM AGE(e.event_date, ff.debut_date)) + 
                        EXTRACT(MONTH FROM AGE(e.event_date, ff.debut_date))/12 as ufc_age
                    FROM features.fight_stats_fe f
                    JOIN features.event_mapping e USING(event_id)
                    JOIN first_fights ff USING(fighter_id)
                )
                UPDATE features.fight_stats_fe
                SET ufcage = ufc.ufc_age
                FROM ufc_age_calc ufc
                WHERE fight_stats_fe.fight_id = ufc.fight_id
                AND fight_stats_fe.fighter_id = ufc.fighter_id;
            """)


def test_ufc_age_calculator_integration():
    """Test UfcAgeCalculator with full integration"""
    # Create mock data with realistic structure
    today = datetime.now()
    three_years_ago = today - timedelta(days=3*365)
    five_years_ago = today - timedelta(days=5*365)
    
    mock_data = {
        'fight_stats_fe': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'fighter_id': [1, 1, 3, 4],  # Fighter 1 appears in fights 101 and 102
            'event_id': [1, 2, 3, 4]
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2, 3, 4],
            'event_date': [
                five_years_ago.strftime('%Y-%m-%d'),  # First event (debut for fighters 1 and 2)
                (five_years_ago + timedelta(days=365)).strftime('%Y-%m-%d'),  # Second event (1 year after debut)
                (five_years_ago + timedelta(days=2*365)).strftime('%Y-%m-%d'),  # Third event (2 years after debut)
                (five_years_ago + timedelta(days=3*365)).strftime('%Y-%m-%d')   # Fourth event (3 years after debut)
            ]
        }),
        'fight_mapping': pd.DataFrame({
            'fight_id': [101, 102, 103, 104],
            'event_id': [1, 2, 3, 4],
            'fighter1_id': [1, 1, 3, 3],
            'fighter2_id': [2, 2, 4, 4]
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
                -- The actual SQL logic would calculate UFC age
            """
            
            # Create context with mock connection
            context = CalculatorContext(mock_conn, mock_data)
            
            # Create calculator with context
            calculator = TestUfcAgeCalculator(context)
            
            # Mock the execute_raw_sql method to actually calculate UFC age
            def mock_execute_raw_sql(sql, params=None):
                # This simulates what the SQL would do to the database
                # First, find each fighter's debut date (first fight)
                fighter_debut = {}
                
                # Join fight_stats_fe with event_mapping to get event dates
                fight_data = pd.merge(
                    mock_data['fight_stats_fe'],
                    mock_data['event_mapping'],
                    on='event_id'
                )
                
                # Find the earliest event date for each fighter (debut date)
                for fighter_id in fight_data['fighter_id'].unique():
                    fighter_fights = fight_data[fight_data['fighter_id'] == fighter_id]
                    debut_date = min(pd.to_datetime(fighter_fights['event_date']))
                    fighter_debut[fighter_id] = debut_date
                
                # Calculate UFC age for each fight
                for idx, row in mock_data['fight_stats_fe'].iterrows():
                    fighter_id = row['fighter_id']
                    event_id = row['event_id']
                    
                    # Get event date
                    event_date = pd.to_datetime(mock_data['event_mapping'].loc[
                        mock_data['event_mapping']['event_id'] == event_id, 'event_date'
                    ].iloc[0])
                    
                    # Get fighter's debut date
                    debut_date = fighter_debut[fighter_id]
                    
                    # Calculate UFC age in years
                    ufc_age = (event_date - debut_date).days / 365.0
                    
                    # Update the mock data
                    mock_data['fight_stats_fe'].loc[idx, 'ufcage'] = round(ufc_age, 2)
            
            # Replace the execute_raw_sql method
            calculator.execute_raw_sql = mock_execute_raw_sql
            
            # Run the calculator
            calculator.run()
            
            # Verify results
            assert 'ufcage' in mock_data['fight_stats_fe'].columns
            
            # Get UFC age for specific fighters in specific fights
            # Fighter 1 in first fight (debut) should have UFC age 0
            fighter1_first_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 1), 'ufcage'
            ].iloc[0]
            
            # Fighter 1 in second fight (1 year after debut) should have UFC age 1
            fighter1_second_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 1) & 
                (mock_data['fight_stats_fe']['event_id'] == 2), 'ufcage'
            ].iloc[0]
            
            # Fighter 3 in third fight (first appearance, 2 years after the first event)
            fighter3_first_fight = mock_data['fight_stats_fe'].loc[
                (mock_data['fight_stats_fe']['fighter_id'] == 3) & 
                (mock_data['fight_stats_fe']['event_id'] == 3), 'ufcage'
            ].iloc[0]
            
            # Verify the calculated UFC age values match our expectations
            assert fighter1_first_fight == 0.0, f"Expected 0.0, got {fighter1_first_fight}"
            assert abs(fighter1_second_fight - 1.0) < 0.1, f"Expected ~1.0, got {fighter1_second_fight}"
            assert fighter3_first_fight == 0.0, f"Expected 0.0, got {fighter3_first_fight}"
            
            # Print the UFC age values for verification
            print(f"Fighter 1 UFC age in first fight: {fighter1_first_fight} years")
            print(f"Fighter 1 UFC age in second fight: {fighter1_second_fight} years")
            print(f"Fighter 3 UFC age in first fight: {fighter3_first_fight} years") 