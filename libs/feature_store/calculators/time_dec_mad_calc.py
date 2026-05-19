import pandas as pd
import numpy as np
from typing import List, Set, Dict, Optional, Any
from math import log
import logging
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager

class TimedecMadCalculator(BaseCalculator):
    """
    Calculate a time-decayed rolling Median Absolute Deviation (MAD) for each column, 
    using an exponential decay with a configurable half-life (in years). For a fighter's 
    very first fight, fall back to the precomputed weightclass-level MAD.
    
    The decay is computed relative to the current fight's event_date:
    
        weight = EXP(-λ * ((T - t) / 365.25))
    
    where T is the current fight's event_date and t is the past fight's event_date.
    
    MAD = median(|X - median(X)|) provides a more robust measure of dispersion that is 
    less sensitive to outliers than standard deviation.
    """
    
    def __init__(self, context_or_conn, decay_rate_years: float = 1.5,
                 include_patterns: Set[str] = set(),
                 exclude_patterns: Set[str] = set()):
        """
        Initialize the time-decayed MAD calculator.
        
        Args:
            context_or_conn: CalculatorContext or database connection
            decay_rate_years: The half-life for the exponential decay in years (default: 1.5)
            include_patterns: Set of patterns to include in calculation
            exclude_patterns: Set of patterns to exclude from calculation
        """
        # Handle context initialization
        try:
            # Try to check if it's a CalculatorContext object or has the required attributes
            if hasattr(context_or_conn, 'connection') and hasattr(context_or_conn, 'feature_utils'):
                self.context = context_or_conn
                conn = self.context.connection
            else:
                conn = context_or_conn
                self.context = CalculatorContext(conn)
        except (TypeError, AttributeError):
            # Fallback for cases where context_or_conn is a connection or mock
            conn = context_or_conn
            self.context = CalculatorContext(conn)
            
        # Initialize BaseCalculator with multi_table calculator type
        super().__init__(conn, calculator_type='multi_table')
        
        # Set calculator-specific attributes
        self.layer_suffix = '_dec_mad'
        # Compute decay constant λ so that the half-life is decay_rate_years
        self.decay_rate = log(2) / decay_rate_years
        self.decay_rate_years = decay_rate_years
        
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.logger = logging.getLogger(__name__)
        
        # Set up stat tables and include/exclude patterns
        self.stat_tables = self.context.feature_utils.get_stat_tables()

        # Add include/exclude patterns
        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)
    
    def execute_sql_template(self, template_name: str, operation: str, params: Dict[str, Any]) -> str:
        """Execute a SQL template and return the results.
        
        Args:
            template_name: Name of the template to use
            operation: Operation to perform (e.g., 'calculate')
            params: Parameters to pass to the template
            
        Returns:
            Rendered SQL query string
        """
        try:
            # Use the context's SQL manager if available
            sql = self.context.sql_manager.render_template(
                template_name, operation, params
            )
                
            # Validate SQL syntax
            if not sql or 'ERROR' in sql.upper():
                self.logger.error(f"Invalid SQL template: {sql}")
                raise ValueError(f"Invalid SQL template for {template_name}.{operation}")
                
            return sql
        except Exception as e:
            self.logger.error(f"Error executing SQL template {template_name}.{operation}: {str(e)}")
            raise
    
    def _validate_mad_stats(self, table_name: str) -> None:
        """
        Validate that the first-time MAD statistics table exists and has data.
        
        Args:
            table_name: The name of the table to validate
        """
        # Check if the table exists
        check_sql = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'features' 
            AND table_name = '{table_name}_first_time_mad_stats'
        );
        """
        result = pd.read_sql(check_sql, self.context.connection)
        
        if not result.iloc[0, 0]:
            self.logger.warning(f"First-time MAD table for {table_name} does not exist")
            return
            
        # Check if the table has data
        count_sql = f"SELECT COUNT(*) FROM features.{table_name}_first_time_mad_stats;"
        count = pd.read_sql(count_sql, self.context.connection).iloc[0, 0]
        
        if count == 0:
            self.logger.warning(f"First-time MAD table for {table_name} is empty")
            return
            
        self.logger.info(f"Validated first-time MAD stats for {table_name}: {count} weightclasses")

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating time-decayed MAD for a table.
        
        Args:
            table_name: The table to calculate for
            columns: Optional list of columns to calculate for
            
        Returns:
            SQL query string for the calculation
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
                
            # Filter out columns already ending with _dec_mad to avoid re-processing
            calc_columns = [col for col in columns if not col.endswith(self.layer_suffix)]
            if not calc_columns:
                self.logger.info(f"No columns to process for {table_name}")
                return ""

            # Features string for SQL
            features_str = ", ".join([f"f.{col}" for col in calc_columns])
            
            # For each column, build the case statement for MAD calculation
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
                # Use fallback from features.<table>_first_time_mad_stats;
                # the fallback column is named <col>_wc_mad.
                fallback_col = f"{col}_wc_mad"
                col_calcs.append(f"""
                CASE 
                    WHEN m_{col}.count_fights <= 1 THEN COALESCE(ftms.{fallback_col}, 0)
                    WHEN m_{col}.sum_w = 0 THEN 0
                    ELSE COALESCE(m_{col}.weighted_mad_sum / NULLIF(m_{col}.sum_w, 0), 0)
                END AS {col}{self.layer_suffix}""")
            col_calcs_str = ",\n    ".join(col_calcs)
            
            # Build the final SQL with weighted MAD calculations
            final_select_clauses = []
            final_joins = []
            
            for col in calc_columns:
                final_select_clauses.append(f"m_{col}.weighted_mad_sum, m_{col}.sum_w, m_{col}.count_fights")
                final_joins.append(f"LEFT JOIN mad_{col} m_{col} ON b.fight_id = m_{col}.fight_id AND b.fighter_id = m_{col}.fighter_id")
            
            final_select_str = ",\n    ".join(final_select_clauses)
            final_joins_str = "\n    ".join(final_joins)
            
            # Template parameters for the new approach
            template_params = {
                'schema': 'features',
                'table_name': table_name,
                'features_str': features_str,
                'decay_rate': self.decay_rate,
                'weighted_mad_calcs_str': weighted_mad_calcs_str,
                'col_calcs_str': col_calcs_str,
                'calc_columns': calc_columns,
                'sampling': ''  # No sampling by default
            }
            
            # This is for backward compatibility if we want to use the template
            sql = self.execute_sql_template(
                'time_decayed_mad',  # Template directory
                'calculate',        # Template file
                template_params
            )
            
            # If SQL template failed or returned invalid SQL, use our direct SQL implementation
            if not sql or 'ERROR' in sql.upper():
                self.logger.warning(f"Using direct SQL for {table_name} due to template issues")
                
                # Build a direct SQL query that doesn't use nested aggregates
                mad_ctes = []
                for col in calc_columns:
                    mad_ctes.append(f"""
                    -- For {col} column
                    {col}_medians AS (
                        SELECT 
                            b.fight_id,
                            b.fighter_id,
                            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY b2.{col}) AS median_value
                        FROM base b
                        JOIN base b2 ON b2.fighter_id = b.fighter_id 
                                     AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
                        GROUP BY b.fight_id, b.fighter_id
                    ),
                    
                    {col}_mad AS (
                        SELECT 
                            med.fight_id,
                            med.fighter_id,
                            SUM(EXP(-{self.decay_rate} * ((b.event_date - b2.event_date)::INTEGER / 365.25))) AS sum_w,
                            SUM(ABS(b2.{col} - med.median_value) * 
                                EXP(-{self.decay_rate} * ((b.event_date - b2.event_date)::INTEGER / 365.25))
                            ) AS weighted_mad,
                            COUNT(*) AS count_fights
                        FROM {col}_medians med
                        JOIN base b ON b.fight_id = med.fight_id AND b.fighter_id = med.fighter_id
                        JOIN base b2 ON b2.fighter_id = b.fighter_id 
                                     AND (b2.event_date, b2.fight_id) <= (b.event_date, b.fight_id)
                        GROUP BY med.fight_id, med.fighter_id
                    )""")
                
                # Build the final SELECT with CASE statements for each MAD value
                final_case_statements = []
                for col in calc_columns:
                    final_case_statements.append(f"""
                    CASE 
                        WHEN {col}_mad.count_fights <= 1 THEN COALESCE(ftms.{col}_wc_mad, 0)
                        WHEN {col}_mad.sum_w = 0 THEN 0
                        ELSE COALESCE({col}_mad.weighted_mad / NULLIF({col}_mad.sum_w, 0), 0)
                    END AS {col}{self.layer_suffix}""")
                
                # Join clauses for each MAD CTE
                mad_joins = []
                for col in calc_columns:
                    mad_joins.append(f"LEFT JOIN {col}_mad ON b.fight_id = {col}_mad.fight_id AND b.fighter_id = {col}_mad.fighter_id")
                
                # Complete SQL query
                sql = f"""
                WITH base AS (
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        f.event_id,
                        em.event_date,
                        fm.weightclass,
                        {features_str}
                    FROM {self.schema}.{table_name} f 
                    JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
                    JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
                ),
                
                {', '.join(mad_ctes)}
                
                SELECT
                    b.fight_id,
                    b.fighter_id,
                    b.event_id,
                    {', '.join(final_case_statements)}
                FROM base b
                {' '.join(mad_joins)}
                LEFT JOIN {self.schema}.{table_name}_first_time_mad_stats ftms
                    ON b.weightclass = ftms.weightclass
                ORDER BY b.event_date, b.fight_id
                """
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Error generating SQL for {table_name}: {str(e)}")
            return ""
            
    def precompute_first_time_mad_stats_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load pre-computed MAD statistics for all feature tables.
        These statistics should be created by the FirstTimeMadCalculator.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        
        # For each stat table, load MAD if they exist
        for table_name, columns in self.stat_tables.items():
            try:
                # Check and validate the existing stats
                self._validate_mad_stats(table_name)
                
                # Store computed results if available
                try:
                    stats_df = pd.read_sql(f"SELECT * FROM features.{table_name}_first_time_mad_stats", self.context.connection)
                    results[table_name] = stats_df
                    self.logger.info(f"Loaded first-time MAD stats for {table_name}: {len(stats_df)} weightclasses")
                except Exception as e:
                    self.logger.warning(f"Could not load stats for {table_name}: {str(e)}")
                    # Continue with other tables
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                # Continue with other tables
                
        return results
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute time-decayed MAD calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate time-decayed MAD for
            columns: The list of columns to calculate time-decayed MAD for
            
        Returns:
            DataFrame with time-decayed MAD calculation results
        """
        self.logger.info(f"Executing time-decayed MAD calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate time-decayed MAD, let base class handle it
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed time-decayed MAD calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated time-decayed MAD: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing time-decayed MAD calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating time-decayed MAD: {str(e)}")
            raise

    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the time-decayed MAD calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used, kept for API compatibility)
            max_workers: Number of worker threads (not used, kept for API compatibility)
            table_pattern: Optional pattern to filter tables (e.g., 'body' for body strike tables)
            
        Returns:
            Dictionary of results by table
        """
        self.logger.info(f"Starting time-decayed MAD calculation for all tables")
        print(f"\n=== Starting time-decayed MAD calculation ===")
        
        # Load pre-computed first-time stats (created by FirstTimeMadCalculator)
        self.logger.info("Loading first-time MAD stats for all tables")
        print("Loading first-time MAD stats for all tables")
        self.precompute_first_time_mad_stats_for_all_tables()
        
        # Get tables to process
        tables = list(self.stat_tables.keys())
        
        # Apply table filtering if a pattern is provided
        if table_pattern:
            filtered_tables = [table for table in tables if table_pattern in table]
            self.logger.info(f"Filtered tables from {len(tables)} to {len(filtered_tables)} based on pattern '{table_pattern}'")
            print(f"Filtered tables from {len(tables)} to {len(filtered_tables)} based on pattern '{table_pattern}'")
            tables = filtered_tables
        
        total_tables = len(tables)
        self.logger.info(f"Found {total_tables} tables to process")
        print(f"Found {total_tables} tables to process")
        
        # Process tables sequentially
        results = {}
        
        for i, table_name in enumerate(tables, 1):
            try:
                # Filter columns based on include/exclude patterns
                columns = self.stat_tables[table_name]
                filtered_columns = [col for col in columns if self.should_process_column(col)]
                
                if not filtered_columns:
                    self.logger.info(f"No columns to process for {table_name}")
                    print(f"[{i}/{total_tables}] Skipping table {table_name}: No columns to process")
                    results[table_name] = pd.DataFrame()
                    continue
                
                # Process the table
                self.logger.info(f"Processing table {table_name} with {len(filtered_columns)} columns")
                print(f"[{i}/{total_tables}] Processing table {table_name} with {len(filtered_columns)} columns")
                
                # Generate SQL for the table
                sql = self.calculate_for_table(table_name, filtered_columns)
                if not sql:
                    print(f"[{i}/{total_tables}] ✗ Failed to generate SQL for {table_name}")
                    results[table_name] = pd.DataFrame()
                    continue
                    
                # Use execute_layer_update for high-performance updates
                # This handles the temporary table creation, indexing, and efficient data loading
                # all within the database without transferring data to Python
                self.execute_layer_update(
                    calculation_sql=sql,
                    table_name=table_name,
                    schema=self.schema,
                    batch_size=100000
                )
                
                self.logger.info(f"Completed time-decayed MAD calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
        
        successful_tables = sum(1 for df in results.values() if not df.empty)
        print(f"\n=== Completed time-decayed MAD calculation: {successful_tables}/{total_tables} tables successfully processed ===\n")
        return results
        
    def save_for_table(self, table_name: str, columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Save time-decayed MAD calculation results for a table.
        This method is maintained for backwards compatibility but is no longer used
        when execute_layer_update is used in run method.
        
        Args:
            table_name: The name of the table
            columns: The list of columns that were calculated
            df: DataFrame with calculation results
            
        Returns:
            DataFrame with saved results
        """
        if df.empty:
            self.logger.info(f"No results to save for table: {table_name}")
            print(f"  └─ No results to save for table: {table_name}")
            return df
            
        self.logger.info(f"Saving {len(df)} time-decayed MAD features for table: {table_name}")
        print(f"  └─ Saving {len(df)} time-decayed MAD features for {table_name}")
        try:
            # Get key columns for this table
            key_columns = ['fight_id', 'fighter_id']
            
            # Update the table using bulk_update_dataframe from BaseFeatureStore
            self.bulk_update_dataframe(
                df=df,
                table_name=table_name,
                schema=self.schema,
                key_columns=key_columns
            )
            
            self.logger.info(f"Successfully saved time-decayed MAD features for {table_name}")
            print(f"  └─ Successfully saved to database")
            return df
        except Exception as e:
            self.logger.error(f"Error saving time-decayed MAD features for {table_name}: {str(e)}")
            print(f"  └─ Error saving to database: {str(e)}")
            raise 