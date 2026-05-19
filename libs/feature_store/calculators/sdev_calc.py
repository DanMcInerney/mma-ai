import logging
import pandas as pd
from typing import Dict, List, Set, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager

class StandardDeviationCalculator(BaseCalculator):
    """
    Calculate a rolling STDDEV for each column in the given table, per fighter.
    
    For the fighter's first fight:
      Use the weightclass-specific STDDEV (precomputed) instead of 0.
    
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
            
        self.layer_suffix = '_sdev'
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

    def _validate_sdev_stats(self, table_name: str) -> None:
        """
        Validate that the first-time standard deviation statistics table exists and has data.
        
        Args:
            table_name: The table to validate
        """
        # Check if the table exists
        check_sql = f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'features' 
            AND table_name = '{table_name}_first_time_sdev_stats'
        );
        """
        result = pd.read_sql(check_sql, self.context.connection)
        
        if not result.iloc[0, 0]:
            self.logger.warning(f"First-time standard deviation table for {table_name} does not exist")
            return
            
        # Check if the table has data
        count_sql = f"SELECT COUNT(*) FROM features.{table_name}_first_time_sdev_stats;"
        count = pd.read_sql(count_sql, self.context.connection).iloc[0, 0]
        
        if count == 0:
            self.logger.warning(f"First-time standard deviation table for {table_name} is empty")
            return
            
        self.logger.info(f"Validated first-time standard deviation stats for {table_name}: {count} weightclasses")

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """
        For each column in `columns`, compute the rolling STDDEV and name it <col>_sdev.
        
        If fighter_id is on their first fight, default the sdev to the
        weightclass-level STDDEV we computed in precompute_first_time_sdev_stats_for_all_tables().
        
        Args:
            table: The table to calculate standard deviations for
            columns: Optional list of columns to process, if None, will use all columns in the table
            
        Returns:
            SQL query string for the calculations
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table)
                
            # Filter out columns already ending with _sdev to avoid re-processing
            calc_columns = [col for col in columns if not col.endswith(self.layer_suffix)]
            if not calc_columns:
                self.logger.info(f"No columns to process for {table}")
                return ""

            # Build STDDEV expressions for each column
            stddev_exprs = []
            for col in calc_columns:
                stddev_exprs.append(f"""
                    CASE 
                        WHEN fd.rn = 1 THEN ftss.{col}_wc_sdev
                        ELSE COALESCE(STDDEV({col}) OVER (
                                PARTITION BY fd.fighter_id
                                ORDER BY fd.event_date, fd.fight_id
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ), 0)
                    END AS {col}{self.layer_suffix}
                """)

            # Prepare template parameters
            template_params = {
                'schema': 'features',
                'table_name': table,
                'columns': calc_columns,
                'features_str': ", ".join(f"f.{c}" for c in calc_columns),
                'stddev_exprs_str': ",\n                ".join(stddev_exprs),
                'output_columns': ", ".join(f"{c}{self.layer_suffix}" for c in calc_columns)
            }

            # Render the SQL template
            sql = self.execute_sql_template(
                'rolling_sdev',
                'calculate',
                template_params
            )

            return sql

        except Exception as e:
            self.logger.error(f"Error calculating standard deviations for {table}: {str(e)}")
            return ""

    def precompute_first_time_sdev_stats_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Load pre-computed standard deviation statistics for all feature tables.
        These statistics should be created by the FirstTimeSdevCalculator.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        
        # For each stat table, load standard deviations if they exist
        for table_name, columns in self.stat_tables.items():
            try:
                # Check and validate the existing stats
                self._validate_sdev_stats(table_name)
                
                # Store computed results if available
                try:
                    stats_df = pd.read_sql(f"SELECT * FROM features.{table_name}_first_time_sdev_stats", self.context.connection)
                    results[table_name] = stats_df
                    self.logger.info(f"Loaded first-time standard deviation stats for {table_name}: {len(stats_df)} weightclasses")
                except Exception as e:
                    self.logger.warning(f"Could not load stats for {table_name}: {str(e)}")
                    # Continue with other tables
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                # Continue with other tables
                
        return results
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute standard deviation calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate standard deviations for
            columns: The list of columns to calculate standard deviations for
            
        Returns:
            DataFrame with standard deviation calculation results
        """
        self.logger.info(f"Executing standard deviation calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate standard deviations
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed standard deviation calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated standard deviations: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing standard deviation calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating standard deviations: {str(e)}")
            raise

    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the standard deviation calculator for all tables.
        
        Args:
            parallel: This parameter is deprecated and ignored
            max_workers: This parameter is deprecated and ignored
            
        Returns:
            Dictionary of results by table
        """
        # Load pre-computed first-time stats (created by FirstTimeSdevCalculator)
        self.logger.info("Loading first-time standard deviation stats for all tables")
        print("\n=== Starting standard deviation calculation ===")
        print("Loading first-time standard deviation stats for all tables")
        self.precompute_first_time_sdev_stats_for_all_tables()
        
        # Then calculate rolling standard deviations
        results = {}
        self.logger.info("Calculating rolling standard deviations for all tables")
        print("Calculating rolling standard deviations for all tables")
        
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
                
                # Process the table directly instead of calling _process_single_table
                self.logger.info(f"Processing table {table_name} with {len(filtered_columns)} columns")
                print(f"[{i}/{total_tables}] Processing table {table_name} with {len(filtered_columns)} columns")
                
                # Generate SQL for the table
                sql = self.calculate_for_table(table_name, filtered_columns)
                if not sql:
                    print(f"[{i}/{total_tables}] ✗ Failed to generate SQL for {table_name}")
                    results[table_name] = pd.DataFrame()
                    continue
                    
                # Use execute_layer_update for high-performance updates
                self.execute_layer_update(
                    calculation_sql=sql,
                    table_name=table_name,
                    schema=self.schema,
                    batch_size=100000
                )
                
                self.logger.info(f"Completed standard deviation calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
        
        successful_tables = sum(1 for df in results.values() if not df.empty)
        print(f"\n=== Completed standard deviation calculation: {successful_tables}/{total_tables} tables successfully processed ===\n")
        return results
        
    def save_for_table(self, table_name: str, columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Save standard deviation calculation results for a table.
        This method is maintained for backwards compatibility but is no longer used
        when execute_layer_update is used in _process_single_table.
        
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
            
        self.logger.info(f"Saving {len(df)} standard deviation features for table: {table_name}")
        print(f"  └─ Saving {len(df)} standard deviation features for {table_name}")
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
            
            self.logger.info(f"Successfully saved standard deviation features for {table_name}")
            print(f"  └─ Successfully saved to database")
            return df
        except Exception as e:
            self.logger.error(f"Error saving standard deviation features for {table_name}: {str(e)}")
            print(f"  └─ Error saving to database: {str(e)}")
            raise