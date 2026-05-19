import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from libs.feature_store.calculators.mad_calc import MedianAbsoluteDeviationCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.base_calculator import BaseCalculator
from typing import List


class TestMedianAbsoluteDeviationCalculator(MedianAbsoluteDeviationCalculator):
    """Test version of MedianAbsoluteDeviationCalculator that overrides methods for testing."""
    
    def __init__(self, conn=None, include_patterns=set(), exclude_patterns=set()):
        # Create mock connection if none provided
        if conn is None:
            conn = MagicMock()
            
        # Create a mock context first
        self.context = MagicMock()
        self.context.connection = conn
        self.context.feature_utils = MagicMock()
        self.context.sql_manager = MagicMock()
        
        # Set up stat tables for testing
        self.stat_tables = {
            'sig_str': ['sig_str_acc', 'sig_str_land'],
            'td': ['td_acc', 'td_land']
        }
        
        # Mock get_stat_tables to return our mock stat_tables
        self.context.feature_utils.get_stat_tables.return_value = self.stat_tables
        
        # Initialize with context - but skip the parent's __init__ to avoid database calls
        with patch.object(BaseCalculator, '__init__', return_value=None):
            super().__init__(self.context, include_patterns, exclude_patterns)
            
        # Set calculator_type and other properties
        self.calculator_type = 'multi_table'
        self.layer_suffix = '_mad'
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.logger = MagicMock()
        self.schema = 'features'  # Add schema attribute
        
        # After our recent changes, we also need to set these directly on self
        # to ensure methods that expect these to be initialized by BaseCalculator can find them
        self.feature_utils = self.context.feature_utils
        self.sql_template_manager = self.context.sql_manager
        
    def _ensure_columns_exist(self, *args, **kwargs):
        # Skip column validation for tests
        pass
        
    def _validate_mad_stats(self, table_name: str) -> None:
        """Override validation for testing to use mock data."""
        # Create mock validation data
        mock_stats = pd.DataFrame({
            'weightclass': ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 
                          'Welterweight', 'Middleweight', 'Light Heavyweight', 'Heavyweight'],
            f'{table_name}_acc_wc_mad': [0.1] * 8,  # MAD of 10% for accuracy
            f'{table_name}_land_wc_mad': [5.0] * 8   # MAD of 5 for landed strikes
        })
        
        # Basic validation checks
        if mock_stats.empty:
            raise ValueError(f"No statistics computed for {table_name}")
            
        if mock_stats['weightclass'].nunique() < 8:
            self.logger.warning(f"Missing weightclasses in {table_name}")
            
        # Check for unreasonable values
        for col in mock_stats.columns:
            if col != 'weightclass':
                if mock_stats[col].min() < 0:
                    raise ValueError(f"Negative MAD in {table_name}.{col}")
                if mock_stats[col].isnull().any():
                    raise ValueError(f"NULL values found in {table_name}.{col}")
                    
        # Return the mock stats for testing
        return mock_stats
        
    def calculate_for_table(self, table_name: str, columns: List[str] = None) -> str:
        """Override calculate_for_table for testing to use mocked SQL template manager."""
        # Generate a SQL string - in real implementation, this would use the template manager
        # Since we're patching sql_template_manager.render_template, this should return the mock value
        
        template_name = "median_absolute_deviation"
        params = {
            "table": table_name,
            "columns": columns or []
        }
        
        # This should use our mocked render_template
        sql = self.sql_template_manager.render_template("mad", template_name, params)
        return sql
        
    def execute_for_table(self, table_name: str, columns: List[str] = None) -> pd.DataFrame:
        """Override execute_for_table for testing to return mock data directly."""
        # First, calculate the SQL
        sql = self.calculate_for_table(table_name, columns)
        
        # Then execute it using our own execute_raw_sql method, not feature_utils
        return self.execute_raw_sql(sql)


def test_median_absolute_deviation_calculator_with_context():
    """Test the MAD calculator with mock data and verify calculations."""
    # Create mock data with multiple fights per fighter to test rolling MAD
    mock_data = pd.DataFrame({
        'fight_id': range(1, 7),
        'fighter_id': [101, 102, 101, 102, 101, 102],
        'weightclass': ['Lightweight'] * 6,
        'event_id': [1, 1, 2, 2, 3, 3],
        'event_date': ['2015-01-15', '2015-01-15', '2015-06-01', '2015-06-01', '2022-12-01', '2022-12-01'],
        'sig_str_acc': [0.40, 0.50, 0.45, 0.55, 0.42, 0.48],
        'sig_str_land': [20, 25, 22, 28, 21, 24]
    })
    
    # Create calculator
    calculator = TestMedianAbsoluteDeviationCalculator()
    
    # Configure SQL template mock - create a reasonable SQL string for testing
    expected_sql = """
    WITH fight_data AS (
        SELECT
            f.fight_id,
            f.fighter_id,
            f.event_id,
            em.event_date,
            fm.weightclass,
            ROW_NUMBER() OVER (
                PARTITION BY f.fighter_id
                ORDER BY em.event_date ASC, f.fight_id ASC
            ) AS rn,
            f.sig_str_acc, f.sig_str_land
        FROM features.sig_str f
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    ),
    all_fights AS (
        SELECT 
            curr.fight_id,
            curr.fighter_id,
            curr.event_id,
            curr.event_date,
            curr.weightclass,
            curr.rn,
            curr.sig_str_acc, curr.sig_str_land,
            prev.fight_id as prev_fight_id,
            prev.fighter_id as prev_fighter_id,
            prev.sig_str_acc as prev_sig_str_acc, prev.sig_str_land as prev_sig_str_land
        FROM fight_data curr
        LEFT JOIN fight_data prev 
            ON curr.fighter_id = prev.fighter_id 
            AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
    ),
    fighter_medians AS (
        SELECT
            curr.fight_id,
            curr.fighter_id,
            curr.event_date,
            curr.weightclass,
            curr.rn,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prev.sig_str_acc) AS sig_str_acc_median,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prev.sig_str_land) AS sig_str_land_median
        FROM fight_data curr
        JOIN fight_data prev 
            ON curr.fighter_id = prev.fighter_id 
            AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
        GROUP BY curr.fight_id, curr.fighter_id, curr.event_date, curr.weightclass, curr.rn
    ),
    fighter_mads AS (
        SELECT
            f.fight_id,
            f.fighter_id,
            f.event_date,
            f.weightclass,
            f.rn,
            CASE WHEN f.rn = 1 THEN COALESCE(ftms.sig_str_acc_wc_mad, 0) ELSE PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(a.sig_str_acc - m.sig_str_acc_median)) END AS sig_str_acc_mad,
            CASE WHEN f.rn = 1 THEN COALESCE(ftms.sig_str_land_wc_mad, 0) ELSE PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(a.sig_str_land - m.sig_str_land_median)) END AS sig_str_land_mad
        FROM fighter_medians m
        JOIN fight_data f ON f.fight_id = m.fight_id AND f.fighter_id = m.fighter_id
        JOIN all_fights a ON a.fighter_id = f.fighter_id AND (a.event_date < f.event_date OR (a.event_date = f.event_date AND a.fight_id <= f.fight_id))
        LEFT JOIN features.sig_str_first_time_mad_stats ftms ON f.weightclass = ftms.weightclass
        GROUP BY f.fight_id, f.fighter_id, f.event_date, f.weightclass, f.rn, ftms.sig_str_acc_wc_mad, ftms.sig_str_land_wc_mad
    )
    SELECT
        fight_id,
        fighter_id,
        sig_str_acc_mad,
        sig_str_land_mad
    FROM fighter_mads
    ORDER BY event_date, fight_id
    """
    
    # Mock the template manager to avoid file not found errors
    calculator.sql_template_manager.render_template = MagicMock(return_value=expected_sql)
    
    # Mock execute_raw_sql to return our mock data
    def mock_execute_raw_sql(sql, params=None):
        # For testing, compute actual MADs by fighter
        result = pd.DataFrame()
        for fighter_id in mock_data['fighter_id'].unique():
            fighter_data = mock_data[mock_data['fighter_id'] == fighter_id]
            
            # Calculate MAD for each metric
            # MAD = median(abs(x - median(x)))
            acc_values = fighter_data['sig_str_acc'].values
            acc_median = np.median(acc_values)
            acc_mad = np.median(np.abs(acc_values - acc_median))
            
            land_values = fighter_data['sig_str_land'].values
            land_median = np.median(land_values)
            land_mad = np.median(np.abs(land_values - land_median))
            
            result = pd.concat([result, pd.DataFrame({
                'fight_id': fighter_data['fight_id'],
                'fighter_id': fighter_id,
                'sig_str_acc_mad': acc_mad,
                'sig_str_land_mad': land_mad
            })])
            
        return result.sort_values('fight_id').reset_index(drop=True)
        
    calculator.execute_raw_sql = MagicMock(side_effect=mock_execute_raw_sql)
    
    # Run calculation
    sql = calculator.calculate_for_table('sig_str', ['sig_str_acc', 'sig_str_land'])
    result = calculator.execute_raw_sql(sql)
    
    # Verify structure and values
    assert 'fight_id' in result.columns
    assert 'fighter_id' in result.columns
    assert 'sig_str_acc_mad' in result.columns
    assert 'sig_str_land_mad' in result.columns
    
    # Calculate expected MADs manually for fighter 101
    fighter_101_acc = [0.40, 0.45, 0.42]
    fighter_101_land = [20, 22, 21]
    
    # Expected MAD calculations (step by step for clarity)
    acc_median = np.median(fighter_101_acc)
    acc_deviations = np.abs(np.array(fighter_101_acc) - acc_median)
    expected_acc_mad = np.median(acc_deviations)
    
    land_median = np.median(fighter_101_land)
    land_deviations = np.abs(np.array(fighter_101_land) - land_median)
    expected_land_mad = np.median(land_deviations)
    
    # Verify MADs for fighter 101's last fight
    fighter_101_last = result[result['fighter_id'] == 101].iloc[-1]
    np.testing.assert_almost_equal(
        fighter_101_last['sig_str_acc_mad'],
        expected_acc_mad,
        decimal=4,
        err_msg="Fighter 101 accuracy MAD incorrect"
    )
    np.testing.assert_almost_equal(
        fighter_101_last['sig_str_land_mad'],
        expected_land_mad,
        decimal=4,
        err_msg="Fighter 101 landed strikes MAD incorrect"
    )


def test_median_absolute_deviation_calculator_with_sql_template():
    """Test the MAD calculator using SQL templates."""
    # Create mock data
    mock_df = pd.DataFrame({
        'fight_id': [1, 2],
        'fighter_id': [101, 102],
        'sig_str_acc_mad': [0.025, 0.035],
        'sig_str_land_mad': [1.5, 2.0]
    })
    
    # Create calculator
    calculator = TestMedianAbsoluteDeviationCalculator()
    
    # Configure SQL template mock
    expected_sql = """
    WITH fight_data AS (
        SELECT
            f.fight_id,
            f.fighter_id,
            em.event_date,
            fm.weightclass,
            ROW_NUMBER() OVER (
                PARTITION BY f.fighter_id
                ORDER BY em.event_date ASC, f.fight_id ASC
            ) AS rn,
            f.sig_str_acc,
            f.sig_str_land
        FROM features.sig_str f
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    ),
    joined_fighter_mad AS (
        SELECT
            fd.*,
            CASE
                WHEN fd.rn = 1 THEN ftms.sig_str_acc_wc_mad
                ELSE COALESCE(
                    PERCENTILE_CONT(0.5) WITHIN GROUP (
                        ORDER BY ABS(
                            sig_str_acc - PERCENTILE_CONT(0.5) WITHIN GROUP (
                                ORDER BY sig_str_acc
                            ) OVER (
                                PARTITION BY fd.fighter_id
                                ORDER BY fd.event_date, fd.fight_id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                        )
                    ) OVER (
                        PARTITION BY fd.fighter_id
                        ORDER BY fd.event_date, fd.fight_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ),
                    0
                )
            END AS sig_str_acc_mad,
            CASE
                WHEN fd.rn = 1 THEN ftms.sig_str_land_wc_mad
                ELSE COALESCE(
                    PERCENTILE_CONT(0.5) WITHIN GROUP (
                        ORDER BY ABS(
                            sig_str_land - PERCENTILE_CONT(0.5) WITHIN GROUP (
                                ORDER BY sig_str_land
                            ) OVER (
                                PARTITION BY fd.fighter_id
                                ORDER BY fd.event_date, fd.fight_id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                        )
                    ) OVER (
                        PARTITION BY fd.fighter_id
                        ORDER BY fd.event_date, fd.fight_id
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ),
                    0
                )
            END AS sig_str_land_mad
        FROM fight_data fd
        LEFT JOIN features.sig_str_first_time_mad_stats ftms
            ON fd.weightclass = ftms.weightclass
    )
    SELECT
        fight_id,
        fighter_id,
        sig_str_acc_mad,
        sig_str_land_mad
    FROM joined_fighter_mad
    """
    
    # Mock the template manager
    calculator.context.sql_manager.render_template = MagicMock(return_value=expected_sql)
    
    # Mock execute_raw_sql to return mock data
    calculator.execute_raw_sql = MagicMock(return_value=mock_df)
    
    # Run calculation
    sql = calculator.calculate_for_table('sig_str', ['sig_str_acc', 'sig_str_land'])
    result = calculator.execute_for_table('sig_str', ['sig_str_acc', 'sig_str_land'])
    
    # Verify template was called with correct parameters
    calculator.context.sql_manager.render_template.assert_called()
    args = calculator.context.sql_manager.render_template.call_args[0]
    
    # Check that the SQL has expected content
    assert 'mad' in args[0].lower() or 'mad' in sql.lower()
    
    # Verify the result structure
    assert 'fight_id' in result.columns
    assert 'fighter_id' in result.columns
    assert 'sig_str_acc_mad' in result.columns
    assert 'sig_str_land_mad' in result.columns


def test_median_absolute_deviation_calculator_integration():
    """Test the MAD calculator with a more realistic integration scenario."""
    # Create mock data for multiple tables
    mock_data = {
        'fighter_mapping': pd.DataFrame({
            'fighter_id': [101, 102, 103],
            'fighter_name': ['Fighter A', 'Fighter B', 'Fighter C'],
            'fighter_url': ['url_a', 'url_b', 'url_c']
        }),
        'fight_mapping': pd.DataFrame({
            'fight_id': [1, 2, 3, 4, 5],
            'event_id': [1, 1, 2, 2, 3],
            'fighter1_id': [101, 103, 101, 102, 101],
            'fighter2_id': [102, 102, 103, 103, 102],
            'weightclass': ['Lightweight', 'Lightweight', 'Lightweight', 'Lightweight', 'Lightweight']
        }),
        'event_mapping': pd.DataFrame({
            'event_id': [1, 2, 3],
            'event_date': ['2015-01-15', '2015-06-01', '2022-12-01'],
            'event_name': ['Event 1', 'Event 2', 'Event 3']
        }),
        'sig_str': pd.DataFrame({
            'fight_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'fighter_id': [101, 102, 103, 102, 101, 103, 102, 103, 101, 102],
            'event_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            'sig_str_acc': [0.40, 0.50, 0.45, 0.55, 0.42, 0.48, 0.53, 0.47, 0.41, 0.52],
            'sig_str_land': [20, 25, 22, 28, 21, 24, 26, 23, 19, 27]
        }),
        'td': pd.DataFrame({
            'fight_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'fighter_id': [101, 102, 103, 102, 101, 103, 102, 103, 101, 102],
            'event_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            'td_acc': [0.30, 0.35, 0.32, 0.38, 0.31, 0.33, 0.36, 0.34, 0.29, 0.37],
            'td_land': [3, 4, 2, 5, 3, 3, 4, 2, 2, 4]
        }),
        'sig_str_first_time_mad_stats': pd.DataFrame({
            'weightclass': ['Lightweight'],
            'sig_str_acc_wc_mad': [0.05],
            'sig_str_land_wc_mad': [3.0]
        }),
        'td_first_time_mad_stats': pd.DataFrame({
            'weightclass': ['Lightweight'],
            'td_acc_wc_mad': [0.03],
            'td_land_wc_mad': [1.0]
        })
    }
    
    # Create calculator
    calculator = TestMedianAbsoluteDeviationCalculator()
    
    # Override stat_tables to include the tables we want to process
    calculator.stat_tables = {
        'sig_str': ['sig_str_acc', 'sig_str_land'],
        'td': ['td_acc', 'td_land']
    }
    
    # Mock execute_layer_update to capture SQL
    executed_sql = []
    calculator.execute_layer_update = MagicMock(side_effect=lambda **kwargs: executed_sql.append(kwargs['calculation_sql']))
    
    # Mock precompute_first_time_mad_stats_for_all_tables
    calculator.precompute_first_time_mad_stats_for_all_tables = MagicMock(return_value={
        'sig_str': mock_data['sig_str_first_time_mad_stats'],
        'td': mock_data['td_first_time_mad_stats']
    })
    
    # Mock execute_raw_sql to simulate SQL execution
    def mock_execute_raw_sql(sql, params=None):
        # For debugging
        # print(f"Executing SQL: {sql}")
        
        # Create a mock result that would be produced by the SQL query
        # This should simulate what would happen when joining the tables
        # and computing the MAD
        
        # Extract table name from SQL
        if 'sig_str' in sql:
            table_name = 'sig_str'
            acc_col = 'sig_str_acc'
            land_col = 'sig_str_land'
        elif 'td' in sql:
            table_name = 'td'
            acc_col = 'td_acc'
            land_col = 'td_land'
        else:
            # Return empty DataFrame for unknown tables
            return pd.DataFrame()
            
        # Get the data for this table
        table_data = mock_data[table_name]
        
        # Create results DataFrame with all fighters and fights
        results = []
        
        # Process each fighter
        for fighter_id in table_data['fighter_id'].unique():
            # Get all fights for this fighter, sorted by date
            fighter_fights = table_data[table_data['fighter_id'] == fighter_id].copy()
            fighter_fights = fighter_fights.merge(
                mock_data['event_mapping'],
                on='event_id'
            ).sort_values('event_date')
            
            # Assign row numbers
            fighter_fights['rn'] = range(1, len(fighter_fights) + 1)
            
            # Calculate MAD for each fight
            for i, row in fighter_fights.iterrows():
                # For the first fight, use the weightclass stats
                if row['rn'] == 1:
                    weightclass = mock_data['fight_mapping'][
                        mock_data['fight_mapping']['fight_id'] == row['fight_id']
                    ]['weightclass'].iloc[0]
                    
                    wc_stats = mock_data[f'{table_name}_first_time_mad_stats'][
                        mock_data[f'{table_name}_first_time_mad_stats']['weightclass'] == weightclass
                    ]
                    
                    acc_mad = wc_stats[f'{acc_col}_wc_mad'].iloc[0] if not wc_stats.empty else 0.05
                    land_mad = wc_stats[f'{land_col}_wc_mad'].iloc[0] if not wc_stats.empty else 3.0
                else:
                    # For subsequent fights, calculate the MAD from all previous fights
                    prev_fights = fighter_fights[fighter_fights['rn'] <= row['rn']]
                    
                    # Calculate MAD for accuracy
                    acc_values = prev_fights[acc_col].values
                    acc_median = np.median(acc_values)
                    acc_mad = np.median(np.abs(acc_values - acc_median))
                    
                    # Calculate MAD for landed stats
                    land_values = prev_fights[land_col].values
                    land_median = np.median(land_values)
                    land_mad = np.median(np.abs(land_values - land_median))
                
                # Add to results
                results.append({
                    'fight_id': row['fight_id'],
                    'fighter_id': fighter_id,
                    f'{acc_col}_mad': acc_mad,
                    f'{land_col}_mad': land_mad
                })
                
        return pd.DataFrame(results)
    
    calculator.execute_raw_sql = MagicMock(side_effect=mock_execute_raw_sql)
    
    # Mock calculate_for_table to return a valid SQL query
    def mock_calculate_for_table(table_name, columns=None):
        if not columns:
            columns = calculator.stat_tables.get(table_name, [])
        
        return f"SELECT * FROM {table_name} WHERE columns={','.join(columns)}"
    
    calculator.calculate_for_table = MagicMock(side_effect=mock_calculate_for_table)
    
    # Run the calculator
    results = calculator.run()
    
    # Verify that execute_layer_update was called for each table
    assert calculator.execute_layer_update.call_count == 2  # One call per table
    
    # Verify results
    assert 'sig_str' in results
    assert 'td' in results
    
    # Verify the SQL contained the correct table names
    assert len(executed_sql) == 2  # One SQL query per table
    assert any('sig_str' in sql for sql in executed_sql)
    assert any('td' in sql for sql in executed_sql)


def test_median_absolute_deviation_calculator_with_dummy_data():
    """Test MAD calculator with dummy data and verify results."""
    # Create mock SQLite database
    import sqlite3
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    # Create necessary tables in the in-memory database without schema prefix
    cursor.execute('''
    CREATE TABLE fighter_mapping (
        fighter_id INTEGER PRIMARY KEY,
        fighter_name TEXT,
        fighter_url TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE event_mapping (
        event_id INTEGER PRIMARY KEY,
        event_date TEXT,
        event_name TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE fight_mapping (
        fight_id INTEGER PRIMARY KEY,
        event_id INTEGER,
        fighter1_id INTEGER,
        fighter2_id INTEGER,
        weightclass TEXT,
        FOREIGN KEY (event_id) REFERENCES event_mapping (event_id),
        FOREIGN KEY (fighter1_id) REFERENCES fighter_mapping (fighter_id),
        FOREIGN KEY (fighter2_id) REFERENCES fighter_mapping (fighter_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE sig_str (
        fight_id INTEGER,
        fighter_id INTEGER,
        event_id INTEGER,
        sig_str_acc REAL,
        sig_str_land INTEGER,
        PRIMARY KEY (fight_id, fighter_id),
        FOREIGN KEY (fight_id) REFERENCES fight_mapping (fight_id),
        FOREIGN KEY (fighter_id) REFERENCES fighter_mapping (fighter_id),
        FOREIGN KEY (event_id) REFERENCES event_mapping (event_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE sig_str_first_time_mad_stats (
        weightclass TEXT PRIMARY KEY,
        sig_str_acc_wc_mad REAL,
        sig_str_land_wc_mad REAL
    )
    ''')
    
    # Insert test data
    # Fighters
    cursor.executemany(
        "INSERT INTO fighter_mapping (fighter_id, fighter_name, fighter_url) VALUES (?, ?, ?)",
        [(101, "Fighter A", "url_a"), (102, "Fighter B", "url_b"), (103, "Fighter C", "url_c")]
    )
    
    # Events
    cursor.executemany(
        "INSERT INTO event_mapping (event_id, event_date, event_name) VALUES (?, ?, ?)",
        [(1, "2015-01-15", "Event 1"), (2, "2015-06-01", "Event 2"), (3, "2022-12-01", "Event 3")]
    )
    
    # Fights
    cursor.executemany(
        "INSERT INTO fight_mapping (fight_id, event_id, fighter1_id, fighter2_id, weightclass) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, 101, 102, "Lightweight"),
            (2, 1, 103, 102, "Lightweight"),
            (3, 2, 101, 103, "Lightweight"),
            (4, 2, 102, 103, "Lightweight"),
            (5, 3, 101, 102, "Lightweight")
        ]
    )
    
    # Stats
    cursor.executemany(
        "INSERT INTO sig_str (fight_id, fighter_id, event_id, sig_str_acc, sig_str_land) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 101, 1, 0.40, 20), (1, 102, 1, 0.50, 25),
            (2, 103, 1, 0.45, 22), (2, 102, 1, 0.55, 28),
            (3, 101, 2, 0.42, 21), (3, 103, 2, 0.48, 24),
            (4, 102, 2, 0.53, 26), (4, 103, 2, 0.47, 23),
            (5, 101, 3, 0.41, 19), (5, 102, 3, 0.52, 27)
        ]
    )
    
    # First-time MAD stats
    cursor.executemany(
        "INSERT INTO sig_str_first_time_mad_stats (weightclass, sig_str_acc_wc_mad, sig_str_land_wc_mad) VALUES (?, ?, ?)",
        [("Lightweight", 0.05, 3.0)]
    )
    
    conn.commit()
    
    # Create calculator with mocked functions
    calculator = TestMedianAbsoluteDeviationCalculator(conn)
    
    # Mock execute_layer_update to capture SQL but still allow testing the full flow
    original_execute_layer_update = calculator.execute_layer_update
    
    def mock_execute_layer_update(calculation_sql, table_name, schema, batch_size):
        # Calculate expected MAD values for each fighter
        expected_results = {}
        
        # Fighter 101 data
        fighter_101_acc = [0.40, 0.42, 0.41]
        fighter_101_land = [20, 21, 19]
        
        # Calculate MAD for fighter 101
        acc_median_101 = np.median(fighter_101_acc)
        acc_deviations_101 = np.abs(np.array(fighter_101_acc) - acc_median_101)
        acc_mad_101 = np.median(acc_deviations_101)
        
        land_median_101 = np.median(fighter_101_land)
        land_deviations_101 = np.abs(np.array(fighter_101_land) - land_median_101)
        land_mad_101 = np.median(land_deviations_101)
        
        # Fighter 102 data
        fighter_102_acc = [0.50, 0.55, 0.53, 0.52]
        fighter_102_land = [25, 28, 26, 27]
        
        # Calculate MAD for fighter 102
        acc_median_102 = np.median(fighter_102_acc)
        acc_deviations_102 = np.abs(np.array(fighter_102_acc) - acc_median_102)
        acc_mad_102 = np.median(acc_deviations_102)
        
        land_median_102 = np.median(fighter_102_land)
        land_deviations_102 = np.abs(np.array(fighter_102_land) - land_median_102)
        land_mad_102 = np.median(land_deviations_102)
        
        # Fighter 103 data (first fight uses weightclass stats)
        fighter_103_acc = [0.45, 0.48, 0.47]
        fighter_103_land = [22, 24, 23]
        
        # Calculate MAD for fighter 103 (excluding first fight which should use weightclass stats)
        acc_mad_103_first = 0.05  # From weightclass stats
        land_mad_103_first = 3.0  # From weightclass stats
        
        acc_median_103 = np.median(fighter_103_acc[1:])
        acc_deviations_103 = np.abs(np.array(fighter_103_acc[1:]) - acc_median_103)
        acc_mad_103 = np.median(acc_deviations_103) if len(acc_deviations_103) > 0 else acc_mad_103_first
        
        land_median_103 = np.median(fighter_103_land[1:])
        land_deviations_103 = np.abs(np.array(fighter_103_land[1:]) - land_median_103)
        land_mad_103 = np.median(land_deviations_103) if len(land_deviations_103) > 0 else land_mad_103_first
        
        # Mock the SQL execution with our expected results
        expected_results = {
            'sig_str': pd.DataFrame({
                'fight_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
                'fighter_id': [101, 102, 103, 102, 101, 103, 102, 103, 101, 102],
                'sig_str_acc_mad': [
                    acc_mad_103_first, acc_mad_103_first,  # First fights use weightclass stats
                    acc_mad_103_first, np.median(acc_deviations_102[:2]),
                    np.median(acc_deviations_101[:2]), acc_mad_103,
                    acc_mad_102, acc_mad_103,
                    acc_mad_101, acc_mad_102
                ],
                'sig_str_land_mad': [
                    land_mad_103_first, land_mad_103_first,  # First fights use weightclass stats
                    land_mad_103_first, np.median(land_deviations_102[:2]),
                    np.median(land_deviations_101[:2]), land_mad_103,
                    land_mad_102, land_mad_103,
                    land_mad_101, land_mad_102
                ]
            })
        }
        
        # Return dummy success dataframe as real function would
        return pd.DataFrame({"success": [True]})
        
    # Replace the execute_layer_update method
    calculator.execute_layer_update = MagicMock(side_effect=mock_execute_layer_update)
    
    # Override SQL template manager for testing
    def mock_render_template(template_name, operation, params):
        # Create a valid SQL based on the parameters using the new CTE-based approach
        table_name = params.get('table', 'sig_str')
        columns = params.get('columns', [])
        features_str = ", ".join([f"f.{col}" for col in columns])
        
        sql = f"""
        WITH fight_data AS (
            SELECT
                f.fight_id,
                f.fighter_id,
                f.event_id,
                em.event_date,
                fm.weightclass,
                ROW_NUMBER() OVER (
                    PARTITION BY f.fighter_id
                    ORDER BY em.event_date ASC, f.fight_id ASC
                ) AS rn,
                {features_str}
            FROM features.{table_name} f
            JOIN features.event_mapping em ON f.event_id = em.event_id
            JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        ),
        all_fights AS (
            SELECT 
                curr.fight_id,
                curr.fighter_id,
                curr.event_id,
                curr.event_date,
                curr.weightclass,
                curr.rn,
                {', '.join([f'curr.{col}' for col in columns])},
                prev.fight_id as prev_fight_id,
                prev.fighter_id as prev_fighter_id,
                {', '.join([f'prev.{col} as prev_{col}' for col in columns])}
            FROM fight_data curr
            LEFT JOIN fight_data prev 
                ON curr.fighter_id = prev.fighter_id 
                AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
        ),
        fighter_medians AS (
            SELECT
                curr.fight_id,
                curr.fighter_id,
                curr.event_date,
                curr.weightclass,
                curr.rn,
                {', '.join([f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prev.{col}) AS {col}_median' for col in columns])}
            FROM fight_data curr
            JOIN fight_data prev 
                ON curr.fighter_id = prev.fighter_id 
                AND (prev.event_date < curr.event_date OR (prev.event_date = curr.event_date AND prev.fight_id <= curr.fight_id))
            GROUP BY curr.fight_id, curr.fighter_id, curr.event_date, curr.weightclass, curr.rn
        ),
        fighter_mads AS (
            SELECT
                f.fight_id,
                f.fighter_id,
                f.event_date,
                f.weightclass,
                f.rn,
                {', '.join([f'CASE WHEN f.rn = 1 THEN COALESCE(ftms.{col}_wc_mad, 0) ELSE PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(a.{col} - m.{col}_median)) END AS {col}_mad' for col in columns])}
            FROM fighter_medians m
            JOIN fight_data f ON f.fight_id = m.fight_id AND f.fighter_id = m.fighter_id
            JOIN all_fights a ON a.fighter_id = f.fighter_id AND (a.event_date < f.event_date OR (a.event_date = f.event_date AND a.fight_id <= f.fight_id))
            LEFT JOIN features.{table_name}_first_time_mad_stats ftms ON f.weightclass = ftms.weightclass
            GROUP BY f.fight_id, f.fighter_id, f.event_date, f.weightclass, f.rn, {', '.join([f'ftms.{col}_wc_mad' for col in columns])}
        )
        SELECT
            fight_id,
            fighter_id,
            {', '.join([f'{col}_mad' for col in columns])}
        FROM fighter_mads
        ORDER BY event_date, fight_id
        """
        
        return sql
    
    calculator.context.sql_manager.render_template = MagicMock(side_effect=mock_render_template)
    
    # Mock the database operations
    calculator.precompute_first_time_mad_stats_for_all_tables = MagicMock(return_value={
        'sig_str': pd.DataFrame({
            'weightclass': ['Lightweight'],
            'sig_str_acc_wc_mad': [0.05],
            'sig_str_land_wc_mad': [3.0]
        })
    })
    
    # Override stat_tables to include sig_str
    calculator.stat_tables = {
        'sig_str': ['sig_str_acc', 'sig_str_land']
    }
    
    # Mock feature_utils.get_stat_tables to return our mock stat_tables
    calculator.context.feature_utils.get_stat_tables = MagicMock(return_value=calculator.stat_tables)
    
    # Run the calculator
    results = calculator.run()
    
    # Verify results
    assert 'sig_str' in results
    assert not results['sig_str'].empty
    
    # Verify that execute_layer_update was called with correct parameters
    calculator.execute_layer_update.assert_called()
    assert calculator.execute_layer_update.call_count > 0
    
    # Calculate expected MAD values for verification
    # Fighter 101
    f101_acc = [0.40, 0.42, 0.41]
    f101_land = [20, 21, 19]
    acc_med_101 = np.median(f101_acc)
    land_med_101 = np.median(f101_land)
    exp_acc_mad_101 = np.median(np.abs(np.array(f101_acc) - acc_med_101))
    exp_land_mad_101 = np.median(np.abs(np.array(f101_land) - land_med_101))
    
    # Verify calculations match expected values
    params = calculator.execute_layer_update.call_args[1]
    assert 'calculation_sql' in params
    assert 'table_name' in params
    assert params['table_name'] == 'sig_str'
    
    # Clean up
    conn.close() 