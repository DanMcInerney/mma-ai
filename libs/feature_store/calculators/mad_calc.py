import logging
import pandas as pd
from typing import Dict, List, Set, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager

class MedianAbsoluteDeviationCalculator(BaseCalculator):
    """
    Calculate a rolling Median Absolute Deviation (MAD) for each column in the given table, per fighter.
    
    For the fighter's first fight:
      Use the weightclass-specific MAD (precomputed) instead of 0.
    
    This applies to any column, including those with '_opp' in their name.
    """

    def __init__(self, context_or_conn, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        # Properly handle context initialization
        if isinstance(context_or_conn, CalculatorContext):
            self.context = context_or_conn
            conn = self.context.connection
        else:
            conn = context_or_conn
            self.context = CalculatorContext(conn)
            
        # Initialize with BaseCalculator and multi_table calculator type - pass the connection, not the context
        super().__init__(conn, calculator_type='multi_table')
            
        self.layer_suffix = '_mad'
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
        Validate that the first-time median absolute deviation statistics table exists and has data.
        
        Args:
            table_name: The table to validate
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

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """
        For each column in `columns`, compute the rolling Median Absolute Deviation and name it <col>_mad.
        
        If fighter_id is on their first fight, default the MAD to the
        weightclass-level MAD we computed in precompute_first_time_mad_stats_for_all_tables().
        
        Args:
            table: The table to calculate MAD for
            columns: Optional list of columns to process, if None, will use all columns in the table
            
        Returns:
            SQL query string for the calculations
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table)
                
            # Filter out columns already ending with _mad to avoid re-processing
            calc_columns = [col for col in columns if not col.endswith(self.layer_suffix)]
            if not calc_columns:
                self.logger.info(f"No columns to process for {table}")
                return ""
            
            # Using a direct SQL approach that avoids the issue with PERCENTILE_CONT and window functions
            features_str = ", ".join([f"f.{col}" for col in calc_columns])
            
            # Generate the SQL for each column using a recursive CTE approach that
            # calculates medians for each fighter up to each fight, then uses
            # those to calculate MAD values
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
                FROM features.{table} f
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
                    {', '.join([f'curr.{col}' for col in calc_columns])},
                    prev.fight_id as prev_fight_id,
                    prev.fighter_id as prev_fighter_id,
                    {', '.join([f'prev.{col} as prev_{col}' for col in calc_columns])}
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
                    {', '.join([f'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prev.{col}) AS {col}_median' for col in calc_columns])}
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
                    {', '.join([f'CASE WHEN f.rn = 1 THEN COALESCE(ftms.{col}_wc_mad, 0) ELSE PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(a.{col} - m.{col}_median)) END AS {col}_mad' for col in calc_columns])}
                FROM fighter_medians m
                JOIN fight_data f ON f.fight_id = m.fight_id AND f.fighter_id = m.fighter_id
                JOIN all_fights a ON a.fighter_id = f.fighter_id AND (a.event_date < f.event_date OR (a.event_date = f.event_date AND a.fight_id <= f.fight_id))
                LEFT JOIN features.{table}_first_time_mad_stats ftms ON f.weightclass = ftms.weightclass
                GROUP BY f.fight_id, f.fighter_id, f.event_date, f.weightclass, f.rn, {', '.join([f'ftms.{col}_wc_mad' for col in calc_columns])}
            )
            SELECT
                fight_id,
                fighter_id,
                {', '.join([f'{col}_mad' for col in calc_columns])}
            FROM fighter_mads
            ORDER BY event_date, fight_id
            """
            
            return sql

        except Exception as e:
            self.logger.error(f"Error calculating MAD for {table}: {str(e)}")
            return ""

    def precompute_first_time_mad_stats_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load pre-computed median absolute deviation statistics for all feature tables.
        These statistics should be created by the FirstTimeMadCalculator.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        
        # For each stat table, load MAD stats if they exist
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
        Execute MAD calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate MAD for
            columns: The list of columns to calculate MAD for
            
        Returns:
            DataFrame with MAD calculation results
        """
        self.logger.info(f"Executing MAD calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate MAD - let the base execute_raw_sql method handle it
            df = self.execute_raw_sql(sql, return_results=True)
            
            self.logger.info(f"Completed MAD calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated MAD: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing MAD calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating MAD: {str(e)}")
            raise

    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the MAD calculator for all tables.
        
        Args:
            parallel: This parameter is deprecated and ignored
            max_workers: This parameter is deprecated and ignored
            
        Returns:
            Dictionary of results by table
        """
        # Load pre-computed first-time stats (created by FirstTimeMadCalculator)
        self.logger.info("Loading first-time MAD stats for all tables")
        print("\n=== Starting median absolute deviation calculation ===")
        print("Loading first-time MAD stats for all tables")
        self.precompute_first_time_mad_stats_for_all_tables()
        
        # Then calculate rolling MAD
        results = {}
        self.logger.info("Calculating rolling median absolute deviation for all tables")
        print("Calculating rolling median absolute deviation for all tables")
        
        # Get tables to process
        tables = list(self.stat_tables.keys())
        total_tables = len(tables)
        print(f"Found {total_tables} tables to process")
        
        # Sequential processing
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
                
                # Process the table directly
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
                
                self.logger.info(f"Completed MAD calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
        
        successful_tables = sum(1 for df in results.values() if not df.empty)
        print(f"\n=== Completed median absolute deviation calculation: {successful_tables}/{total_tables} tables successfully processed ===\n")
        return results
        
    def save_for_table(self, table_name: str, columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Save MAD calculation results for the given table by creating or updating the target layer table.
        
        Args:
            table_name: The table to save MAD for
            columns: The list of columns to save MAD for
            df: The DataFrame containing the MAD calculation results
            
        Returns:
            DataFrame with MAD calculation results
        """
        self.logger.info(f"Saving MAD for {table_name}")
        print(f"  └─ Saving results for {table_name}")
        
        try:
            # Execute each SQL statement separately to handle errors gracefully
            try:
                # First drop the existing table if it exists
                drop_sql = f"DROP TABLE IF EXISTS features.{table_name}_mad;"
                self.execute_raw_sql(drop_sql, return_results=False)
            except Exception as e:
                self.logger.error(f"Error dropping existing table for {table_name}: {str(e)}")
                print(f"  └─ Error dropping table: {str(e)}")
                # Continue with the process
            
            try:
                # Then rename the temp table to the target table
                rename_sql = f"ALTER TABLE temp_{table_name}_layer RENAME TO {table_name}_mad;"
                self.execute_raw_sql(rename_sql, return_results=False)
            except Exception as e:
                self.logger.error(f"Error renaming table for {table_name}: {str(e)}")
                print(f"  └─ Error renaming table: {str(e)}")
                raise
            
            try:
                # Add a primary key to the renamed table
                pk_sql = f"ALTER TABLE features.{table_name}_mad ADD PRIMARY KEY (fight_id, fighter_id);"
                self.execute_raw_sql(pk_sql, return_results=False)
            except Exception as e:
                self.logger.warning(f"Error adding primary key for {table_name}: {str(e)}")
                print(f"  └─ Note: Could not add primary key (may already exist): {str(e)}")
                # Continue despite primary key error (may already exist)
            
            self.logger.info(f"Saved MAD for {table_name}")
            print(f"  └─ ✓ Saved MAD for {table_name}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error saving MAD for {table_name}: {str(e)}")
            print(f"  └─ ✗ Error saving MAD for {table_name}: {str(e)}")
            raise 