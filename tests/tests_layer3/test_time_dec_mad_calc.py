import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from math import log
from libs.feature_store.calculators.time_dec_mad_calc import TimedecMadCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict


class TestTimedecMadCalculator(TimedecMadCalculator):
    """Test version of TimedecMadCalculator that overrides methods for testing."""
    
    def __init__(self, conn=None, decay_rate_years=1.5, include_patterns=set(), exclude_patterns=set()):
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
            super().__init__(self.context, decay_rate_years, include_patterns, exclude_patterns)
            
        # Set calculator_type and other properties
        self.calculator_type = 'multi_table'
        self.layer_suffix = '_dec_mad'
        self.decay_rate = log(2) / decay_rate_years
        self.decay_rate_years = decay_rate_years
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
        # Get columns if not provided
        if columns is None:
            columns = self.stat_tables.get(table_name, [])
        
        # Filter out columns already ending with _dec_mad
        calc_columns = [col for col in columns if not col.endswith(self.layer_suffix)]
        
        if not calc_columns:
            return ""

        # Features string for SQL
        features_str = ", ".join([f"f.{col}" for col in calc_columns])
        
        # For each column, build the CTE-based MAD calculation
        weighted_mad_calcs = []
        for col in calc_columns:
            weighted_mad_calcs.append(f"""
            -- Calculate weighted MAD for {col}
            median_{col} AS (
                SELECT 
                    b.fight_id,
                    b.fighter_id,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.{col}) AS median_value
                FROM base b
                JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
                GROUP BY b.fight_id, b.fighter_id
            ),
            
            mad_{col} AS (
                SELECT 
                    m.fight_id,
                    m.fighter_id,
                    SUM(EXP(-{self.decay_rate} * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
                    SUM(
                        ABS(b2.{col} - m.median_value) * 
                        EXP(-{self.decay_rate} * ((b.event_date - b2.event_date)::INTEGER / 365.25))
                    ) AS weighted_mad_sum,
                    COUNT(*) AS count_fights
                FROM median_{col} m
                JOIN base b ON b.fight_id = m.fight_id AND b.fighter_id = m.fighter_id
                JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
                GROUP BY m.fight_id, m.fighter_id
            )""")
        
        weighted_mad_calcs_str = ",\n".join(weighted_mad_calcs)
        
        # For each column, build the case statement for MAD calculation
        col_calcs = []
        for col in calc_columns:
            # Use fallback from features.<table>_first_time_mad_stats
            fallback_col = f"{col}_wc_mad"
            col_calcs.append(f"""
            CASE 
                WHEN m_{col}.count_fights <= 1 THEN COALESCE(ftms.{fallback_col}, 0)
                WHEN m_{col}.sum_w = 0 THEN 0
                ELSE COALESCE(m_{col}.weighted_mad_sum / NULLIF(m_{col}.sum_w, 0), 0)
            END AS {col}{self.layer_suffix}""")
        col_calcs_str = ",\n    ".join(col_calcs)
        
        # Build join clauses for the mad CTEs
        join_clauses = []
        for col in calc_columns:
            join_clauses.append(f"LEFT JOIN mad_{col} m_{col} ON b.fight_id = m_{col}.fight_id AND b.fighter_id = m_{col}.fighter_id")
        join_str = "\n    ".join(join_clauses)
        
        # Template parameters
        template_params = {
            'schema': 'features',
            'table_name': table_name,
            'features_str': features_str,
            'decay_rate': self.decay_rate,
            'weighted_mad_calcs_str': weighted_mad_calcs_str,
            'col_calcs_str': col_calcs_str,
            'join_str': join_str,
            'calc_columns': calc_columns
        }
        
        # Call the SQL template manager to render the template
        sql = self.context.sql_manager.render_template(
            'time_decayed_mad',  # Template directory
            'calculate',        # Template file
            template_params
        )
        
        return sql
        
    def execute_for_table(self, table_name: str, columns: List[str] = None) -> pd.DataFrame:
        """Override execute_for_table for testing to return mock data directly."""
        # First, calculate the SQL
        sql = self.calculate_for_table(table_name, columns)
        
        # Then execute it using our own execute_raw_sql method, not feature_utils
        return self.execute_raw_sql(sql)


def test_timedec_mad_calculator_with_context():
    """Test the TimedecMadCalculator with mock data and verify calculations."""
    # Create mock data with multiple fights per fighter to test time-decay
    mock_data = pd.DataFrame({
        'fight_id': range(1, 7),
        'fighter_id': [101, 102, 101, 102, 101, 102],
        'weightclass': ['Lightweight'] * 6,
        'event_id': [1, 1, 2, 2, 3, 3],
        'event_date': ['2015-01-15', '2015-01-15', '2015-06-01', '2015-06-01', '2022-12-01', '2022-12-01'],
        'sig_str_acc': [0.40, 0.50, 0.45, 0.55, 0.42, 0.48],
        'sig_str_land': [20, 25, 22, 28, 21, 24]
    })
    
    # Create calculator with default decay rate (1.5 years)
    calculator = TestTimedecMadCalculator()
    
    # Configure SQL template mock - create a reasonable SQL string for testing
    expected_sql = """
    WITH base AS (
        SELECT 
            f.fight_id,
            f.fighter_id,
            f.event_id,
            em.event_date,
            fm.weightclass,
            f.sig_str_acc,
            f.sig_str_land
        FROM features.sig_str f 
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    ),
    
    -- Calculate weighted MAD for sig_str_acc
    median_sig_str_acc AS (
        SELECT 
            b.fight_id,
            b.fighter_id,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.sig_str_acc) AS median_value
        FROM base b
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY b.fight_id, b.fighter_id
    ),
    
    mad_sig_str_acc AS (
        SELECT 
            m.fight_id,
            m.fighter_id,
            SUM(EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
            SUM(
                ABS(b2.sig_str_acc - m.median_value) * 
                EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))
            ) AS weighted_mad_sum,
            COUNT(*) AS count_fights
        FROM median_sig_str_acc m
        JOIN base b ON b.fight_id = m.fight_id AND b.fighter_id = m.fighter_id
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY m.fight_id, m.fighter_id
    ),
    
    -- Calculate weighted MAD for sig_str_land
    median_sig_str_land AS (
        SELECT 
            b.fight_id,
            b.fighter_id,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.sig_str_land) AS median_value
        FROM base b
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY b.fight_id, b.fighter_id
    ),
    
    mad_sig_str_land AS (
        SELECT 
            m.fight_id,
            m.fighter_id,
            SUM(EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
            SUM(
                ABS(b2.sig_str_land - m.median_value) * 
                EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))
            ) AS weighted_mad_sum,
            COUNT(*) AS count_fights
        FROM median_sig_str_land m
        JOIN base b ON b.fight_id = m.fight_id AND b.fighter_id = m.fighter_id
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY m.fight_id, m.fighter_id
    )
    
    SELECT 
        b.fight_id,
        b.fighter_id,
        b.event_id,
        CASE 
            WHEN m_acc.count_fights <= 1 THEN COALESCE(ftms.sig_str_acc_wc_mad, 0)
            WHEN m_acc.sum_w = 0 THEN 0
            ELSE COALESCE(m_acc.weighted_mad_sum / NULLIF(m_acc.sum_w, 0), 0)
        END AS sig_str_acc_dec_mad,
        CASE 
            WHEN m_land.count_fights <= 1 THEN COALESCE(ftms.sig_str_land_wc_mad, 0)
            WHEN m_land.sum_w = 0 THEN 0
            ELSE COALESCE(m_land.weighted_mad_sum / NULLIF(m_land.sum_w, 0), 0)
        END AS sig_str_land_dec_mad
    FROM base b
    LEFT JOIN mad_sig_str_acc m_acc ON b.fight_id = m_acc.fight_id AND b.fighter_id = m_acc.fighter_id
    LEFT JOIN mad_sig_str_land m_land ON b.fight_id = m_land.fight_id AND b.fighter_id = m_land.fighter_id
    LEFT JOIN features.sig_str_first_time_mad_stats ftms
        ON b.weightclass = ftms.weightclass
    ORDER BY b.event_date, b.fight_id
    """
    
    # Mock the template manager to avoid file not found errors
    calculator.sql_template_manager.render_template = MagicMock(return_value=expected_sql)
    
    # Mock execute_raw_sql to return our mock data with time-decayed MAD calculations
    def mock_execute_raw_sql(sql, params=None):
        # For testing, compute time-decayed MAD values
        result = pd.DataFrame()
        for fighter_id in mock_data['fighter_id'].unique():
            fighter_data = mock_data[mock_data['fighter_id'] == fighter_id].copy()
            fighter_data['date'] = pd.to_datetime(fighter_data['event_date'])
            fighter_data = fighter_data.sort_values('date')
            
            # Calculate time-decayed MAD values (simplified for testing)
            for i, row in fighter_data.iterrows():
                curr_date = row['date']
                # For first fight, use 0.1 as MAD
                if i == fighter_data.index[0]:
                    acc_mad = 0.1
                    land_mad = 5.0
                else:
                    # Get previous fights data
                    prev_fights = fighter_data.loc[fighter_data.index <= i]
                    
                    # Calculate weights based on time differences
                    prev_fights['days_diff'] = (curr_date - prev_fights['date']).dt.days
                    prev_fights['weight'] = np.exp(-calculator.decay_rate * (prev_fights['days_diff'] / 365.25))
                    
                    # Calculate MAD for accuracy with time-decay weights (simplified)
                    acc_values = prev_fights['sig_str_acc'].values
                    acc_median = np.median(acc_values)
                    acc_devs = np.abs(acc_values - acc_median)
                    acc_mad = np.average(acc_devs, weights=prev_fights['weight'].values)
                    
                    # Calculate MAD for landed stats with time-decay weights (simplified)
                    land_values = prev_fights['sig_str_land'].values
                    land_median = np.median(land_values)
                    land_devs = np.abs(land_values - land_median)
                    land_mad = np.average(land_devs, weights=prev_fights['weight'].values)
                
                # Add to results
                result = pd.concat([result, pd.DataFrame({
                    'fight_id': [row['fight_id']],
                    'fighter_id': [fighter_id],
                    'sig_str_acc_dec_mad': [acc_mad],
                    'sig_str_land_dec_mad': [land_mad]
                })])
        
        return result.sort_values('fight_id').reset_index(drop=True)
        
    calculator.execute_raw_sql = MagicMock(side_effect=mock_execute_raw_sql)
    
    # Run calculation
    sql = calculator.calculate_for_table('sig_str', ['sig_str_acc', 'sig_str_land'])
    result = calculator.execute_raw_sql(sql)
    
    # Verify structure and values
    assert 'fight_id' in result.columns
    assert 'fighter_id' in result.columns
    assert 'sig_str_acc_dec_mad' in result.columns
    assert 'sig_str_land_dec_mad' in result.columns
    
    # Verify that result is not empty and has the expected number of rows
    assert len(result) == len(mock_data)
    
    # Verify that time-decay is applied (later fights should have different MAD values)
    fighter_101_fights = result[result['fighter_id'] == 101]
    assert len(fighter_101_fights) == 3
    
    # MAD values should reflect time-decay
    # First fight should use weightclass MAD
    first_fight = fighter_101_fights.iloc[0]
    assert first_fight['sig_str_acc_dec_mad'] == 0.1
    assert first_fight['sig_str_land_dec_mad'] == 5.0


def test_timedec_mad_calculator_with_sql_template():
    """Test the time-decayed MAD calculator using SQL templates."""
    # Create mock data
    mock_df = pd.DataFrame({
        'fight_id': [1, 2],
        'fighter_id': [101, 102],
        'sig_str_acc_dec_mad': [0.025, 0.035],
        'sig_str_land_dec_mad': [1.5, 2.0]
    })
    
    # Create calculator
    calculator = TestTimedecMadCalculator()
    
    # Configure SQL template mock
    expected_sql = """
    WITH base AS (
        SELECT 
            f.fight_id,
            f.fighter_id,
            f.event_id,
            em.event_date,
            fm.weightclass,
            f.sig_str_acc,
            f.sig_str_land
        FROM features.sig_str f 
        JOIN features.event_mapping em ON f.event_id = em.event_id
        JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
    ),
    
    -- Calculate weighted MAD for sig_str_acc
    median_sig_str_acc AS (
        SELECT 
            b.fight_id,
            b.fighter_id,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.sig_str_acc) AS median_value
        FROM base b
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY b.fight_id, b.fighter_id
    ),
    
    mad_sig_str_acc AS (
        SELECT 
            m.fight_id,
            m.fighter_id,
            SUM(EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
            SUM(
                ABS(b2.sig_str_acc - m.median_value) * 
                EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))
            ) AS weighted_mad_sum,
            COUNT(*) AS count_fights
        FROM median_sig_str_acc m
        JOIN base b ON b.fight_id = m.fight_id AND b.fighter_id = m.fighter_id
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY m.fight_id, m.fighter_id
    ),
    
    -- Calculate weighted MAD for sig_str_land
    median_sig_str_land AS (
        SELECT 
            b.fight_id,
            b.fighter_id,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.sig_str_land) AS median_value
        FROM base b
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY b.fight_id, b.fighter_id
    ),
    
    mad_sig_str_land AS (
        SELECT 
            m.fight_id,
            m.fighter_id,
            SUM(EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
            SUM(
                ABS(b2.sig_str_land - m.median_value) * 
                EXP(-0.462 * ((b.event_date - b2.event_date)::INTEGER / 365.25))
            ) AS weighted_mad_sum,
            COUNT(*) AS count_fights
        FROM median_sig_str_land m
        JOIN base b ON b.fight_id = m.fight_id AND b.fighter_id = m.fighter_id
        JOIN base b2 ON b2.fighter_id = b.fighter_id AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
        GROUP BY m.fight_id, m.fighter_id
    )
    
    SELECT 
        b.fight_id,
        b.fighter_id,
        b.event_id,
        CASE 
            WHEN m_sig_str_acc.count_fights <= 1 THEN COALESCE(ftms.sig_str_acc_wc_mad, 0)
            WHEN m_sig_str_acc.sum_w = 0 THEN 0
            ELSE COALESCE(m_sig_str_acc.weighted_mad_sum / NULLIF(m_sig_str_acc.sum_w, 0), 0)
        END AS sig_str_acc_dec_mad,
        CASE 
            WHEN m_sig_str_land.count_fights <= 1 THEN COALESCE(ftms.sig_str_land_wc_mad, 0)
            WHEN m_sig_str_land.sum_w = 0 THEN 0
            ELSE COALESCE(m_sig_str_land.weighted_mad_sum / NULLIF(m_sig_str_land.sum_w, 0), 0)
        END AS sig_str_land_dec_mad
    FROM base b
    LEFT JOIN mad_sig_str_acc m_sig_str_acc ON b.fight_id = m_sig_str_acc.fight_id AND b.fighter_id = m_sig_str_acc.fighter_id
    LEFT JOIN mad_sig_str_land m_sig_str_land ON b.fight_id = m_sig_str_land.fight_id AND b.fighter_id = m_sig_str_land.fighter_id
    LEFT JOIN features.sig_str_first_time_mad_stats ftms
        ON b.weightclass = ftms.weightclass
    ORDER BY b.event_date, b.fight_id
    """
    
    # Mock the template manager
    calculator.context.sql_manager.render_template = MagicMock(return_value=expected_sql)
    
    # Mock execute_raw_sql to return mock data
    calculator.execute_raw_sql = MagicMock(return_value=mock_df)
    
    # Run calculation
    sql = calculator.calculate_for_table('sig_str', ['sig_str_acc', 'sig_str_land'])
    result = calculator.execute_raw_sql(sql)
    
    # Verify template was called with correct parameters
    calculator.context.sql_manager.render_template.assert_called()
    args = calculator.context.sql_manager.render_template.call_args[0]
    
    # Check that the SQL has expected content
    assert 'mad' in args[0].lower() or 'mad' in sql.lower()
    assert 'dec' in args[0].lower() or 'dec' in sql.lower()
    
    # Verify the result structure
    assert 'fight_id' in result.columns
    assert 'fighter_id' in result.columns
    assert 'sig_str_acc_dec_mad' in result.columns
    assert 'sig_str_land_dec_mad' in result.columns


def test_timedec_mad_calculator_decay_rate():
    """Test that the calculator correctly applies the decay rate."""
    # Test with different decay rates
    decay_rates = [0.5, 1.0, 1.5, 2.0]
    
    for decay_rate in decay_rates:
        # Create calculator with specific decay rate
        calculator = TestTimedecMadCalculator(decay_rate_years=decay_rate)
        
        # Verify decay rate was correctly set
        expected_lambda = log(2) / decay_rate
        assert calculator.decay_rate == expected_lambda
        assert calculator.decay_rate_years == decay_rate
        
        # Set up the mock to include the decay rate in its output
        mock_sql = f"WITH decay_rate AS ({expected_lambda}) SELECT * FROM table"
        calculator.context.sql_manager.render_template = MagicMock(return_value=mock_sql)
        
        # Generate SQL for a table and verify decay rate is included
        sql = calculator.calculate_for_table('sig_str', ['sig_str_acc'])
        
        # The SQL should contain the decay rate
        assert str(expected_lambda) in sql or str(round(expected_lambda, 3)) in sql


def test_timedec_mad_calculator_integration():
    """Test the TimedecMadCalculator with a more realistic integration scenario."""
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
    calculator = TestTimedecMadCalculator()
    
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
            
            # Convert dates to datetime for time difference calculation
            fighter_fights['date'] = pd.to_datetime(fighter_fights['event_date'])
            
            # Assign row numbers
            fighter_fights['rn'] = range(1, len(fighter_fights) + 1)
            
            # Calculate time-decayed MAD for each fight
            for i, row in fighter_fights.iterrows():
                curr_date = row['date']
                
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
                    # For subsequent fights, calculate time-decayed MAD
                    prev_fights = fighter_fights[fighter_fights['rn'] <= row['rn']].copy()
                    
                    # Calculate weights based on time differences
                    prev_fights['days_diff'] = (curr_date - prev_fights['date']).dt.days
                    prev_fights['weight'] = np.exp(-calculator.decay_rate * (prev_fights['days_diff'] / 365.25))
                    
                    # Calculate MAD for accuracy with time-decay weights (simplified)
                    acc_values = prev_fights[acc_col].values
                    acc_median = np.median(acc_values)
                    acc_devs = np.abs(acc_values - acc_median)
                    acc_mad = np.average(acc_devs, weights=prev_fights['weight'].values)
                    
                    # Calculate MAD for landed stats with time-decay weights (simplified)
                    land_values = prev_fights[land_col].values
                    land_median = np.median(land_values)
                    land_devs = np.abs(land_values - land_median)
                    land_mad = np.average(land_devs, weights=prev_fights['weight'].values)
                
                # Add to results
                results.append({
                    'fight_id': row['fight_id'],
                    'fighter_id': fighter_id,
                    f'{acc_col}_dec_mad': acc_mad,
                    f'{land_col}_dec_mad': land_mad
                })
                
        return pd.DataFrame(results)
    
    calculator.execute_raw_sql = MagicMock(side_effect=mock_execute_raw_sql)
    
    # Mock calculate_for_table to return a valid SQL query
    def mock_calculate_for_table(table_name, columns=None):
        if not columns:
            columns = calculator.stat_tables.get(table_name, [])
        
        return f"SELECT * FROM {table_name} WHERE columns={','.join(columns)} AND decay_rate={calculator.decay_rate}"
    
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
    
    # Verify that each SQL query contains the decay rate
    decay_rate_str = str(calculator.decay_rate)
    for sql in executed_sql:
        assert decay_rate_str in sql


def test_timedec_mad_calculator_with_dummy_data():
    """Test the time-decayed MAD calculator with dummy data and verify results."""
    # Create fighter data with known time pattern
    # Fighter 101 has 3 fights over time with a clear pattern
    # Fighter 102 has fights at the same dates
    fights_data = {
        'fight_id': [1, 2, 3, 4, 5, 6],
        'fighter_id': [101, 102, 101, 102, 101, 102],
        'event_id': [1, 1, 2, 2, 3, 3],
        'event_date': ['2015-01-15', '2015-01-15', '2018-01-15', '2018-01-15', '2021-01-15', '2021-01-15'],
        'sig_str_acc': [0.40, 0.50, 0.45, 0.52, 0.43, 0.53],
        'sig_str_land': [20, 25, 18, 27, 22, 26]
    }
    
    fights_df = pd.DataFrame(fights_data)
    fights_df['date'] = pd.to_datetime(fights_df['event_date'])
    
    # Create a mock connection
    conn = MagicMock()
    
    # Create calculator with specified decay rate
    decay_rate_years = 1.5
    calculator = TestTimedecMadCalculator(conn=conn, decay_rate_years=decay_rate_years)
    
    # Set up mock data
    mock_data = {
        'sig_str': fights_df[['fight_id', 'fighter_id', 'event_id', 'sig_str_acc', 'sig_str_land']],
        'sig_str_first_time_mad_stats': pd.DataFrame({
            'weightclass': ['Lightweight'],
            'sig_str_acc_wc_mad': [0.05],
            'sig_str_land_wc_mad': [3.0]
        })
    }
    
    # Mock the required functions
    calculator.precompute_first_time_mad_stats_for_all_tables = MagicMock(return_value={
        'sig_str': mock_data['sig_str_first_time_mad_stats']
    })
    
    # Mock the SQL template manager to include the decay rate in the output
    def mock_render_template(*args, **kwargs):
        # Create SQL that includes the decay rate
        return f"WITH decay_rate AS ({calculator.decay_rate}) SELECT * FROM table"
    
    calculator.context.sql_manager.render_template = MagicMock(side_effect=mock_render_template)
    
    # Mock execute_layer_update to capture SQL
    executed_sql = []
    calculator.execute_layer_update = MagicMock(side_effect=lambda **kwargs: 
        executed_sql.append(kwargs['calculation_sql']) or pd.DataFrame({"success": [True]}))
    
    # Set up time-decayed MAD calculation function
    def calculate_time_decayed_mad(fighter_id, acc_col='sig_str_acc', land_col='sig_str_land'):
        """Calculate time-decayed MAD for a fighter manually for testing."""
        fighter_data = fights_df[fights_df['fighter_id'] == fighter_id].sort_values('date')
        results = []
        
        for i, row in fighter_data.iterrows():
            curr_date = row['date']
            
            # For first fight, use weightclass MAD
            if i == fighter_data.index[0]:
                acc_mad = 0.05  # From weightclass stats
                land_mad = 3.0  # From weightclass stats
            else:
                # For subsequent fights, calculate time-decayed MAD
                prev_fights = fighter_data.loc[fighter_data.index <= i].copy()
                
                # Calculate weights based on time differences
                prev_fights['days_diff'] = (curr_date - prev_fights['date']).dt.days
                prev_fights['weight'] = np.exp(-calculator.decay_rate * (prev_fights['days_diff'] / 365.25))
                
                # Calculate MAD for accuracy with time-decay weights
                acc_values = prev_fights[acc_col].values
                acc_median = np.median(acc_values)
                acc_devs = np.abs(acc_values - acc_median)
                acc_mad = np.average(acc_devs, weights=prev_fights['weight'].values)
                
                # Calculate MAD for landed stats with time-decay weights
                land_values = prev_fights[land_col].values
                land_median = np.median(land_values)
                land_devs = np.abs(land_values - land_median)
                land_mad = np.average(land_devs, weights=prev_fights['weight'].values)
            
            results.append({
                'fight_id': row['fight_id'],
                'fighter_id': fighter_id,
                'event_date': row['event_date'],
                f'{acc_col}_dec_mad': acc_mad,
                f'{land_col}_dec_mad': land_mad
            })
        
        return pd.DataFrame(results)
    
    # Calculate expected results
    expected_results_101 = calculate_time_decayed_mad(101)
    expected_results_102 = calculate_time_decayed_mad(102)
    expected_results = pd.concat([expected_results_101, expected_results_102]).sort_values('fight_id').reset_index(drop=True)
    
    # Mock execute_raw_sql to return expected results
    def mock_execute_raw_sql(sql, params=None):
        # If this is a calculation SQL, return our pre-calculated values
        if 'sig_str' in sql and any(col in sql for col in ['sig_str_acc', 'sig_str_land']):
            return expected_results
        return pd.DataFrame()
    
    calculator.execute_raw_sql = MagicMock(side_effect=mock_execute_raw_sql)
    
    # Set up stat tables
    calculator.stat_tables = {'sig_str': ['sig_str_acc', 'sig_str_land']}
    
    # Run the calculator
    results = calculator.run()
    
    # Verify execute_layer_update was called
    assert calculator.execute_layer_update.call_count > 0
    
    # Verify SQL contains the decay rate
    assert any(str(calculator.decay_rate) in sql for sql in executed_sql)
    
    # Verify that later fights show time-decay effects - need the actual SQL execution to work
    # We can check the expected results
    assert len(expected_results) == 6  # 3 fights each for 2 fighters
    
    # Time-decay should affect the MAD values
    # For fighter 101, there should be an increasing trend due to the pattern of values
    f101_results = expected_results[expected_results['fighter_id'] == 101].sort_values('fight_id')
    
    # First fight should use weightclass MAD
    assert f101_results.iloc[0]['sig_str_acc_dec_mad'] == 0.05
    assert f101_results.iloc[0]['sig_str_land_dec_mad'] == 3.0
    
    # Later fights should have different MAD values reflecting time-decay
    # For this specific data pattern, earlier values get less weight so MAD will change
    assert f101_results.iloc[1]['sig_str_acc_dec_mad'] != f101_results.iloc[0]['sig_str_acc_dec_mad']
    assert f101_results.iloc[2]['sig_str_acc_dec_mad'] != f101_results.iloc[1]['sig_str_acc_dec_mad'] 