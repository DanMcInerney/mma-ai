import pandas as pd
import logging
from typing import List, Set, Dict, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.sql_template_manager import SQLTemplateManager

class AverageCalculator(BaseCalculator):
    """Calculate rolling career average for all stats."""
    
    def __init__(self, context_or_conn, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize the average calculator.
        
        Args:
            context_or_conn: CalculatorContext or database connection
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
        self.layer_suffix = '_avg'
        
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
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """Calculate rolling average for each column.
        
        Args:
            table_name: The source table name
            columns: Optional list of columns to calculate for
            
        Returns:
            SQL query string for the calculations
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
                
            # Filter out columns already ending with _avg to avoid re-processing
            calc_columns = [col for col in columns if not col.endswith(self.layer_suffix)]
            if not calc_columns:
                self.logger.info(f"No columns to process for {table_name}")
                return ""
            
            # Create column calculation expressions for the template
            column_calcs = []
            for col in calc_columns:
                column_calcs.append(
                    f"AVG({col}) OVER (PARTITION BY fighter_id ORDER BY em.event_date, fight_id ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS {col}{self.layer_suffix} "
                )
            
            if not column_calcs:
                return ""
                
            # Template parameters
            template_params = {
                'schema': 'features',
                'table_name': table_name,
                'column_calcs': column_calcs,
                'sampling': ''  # Empty string for potential future use
            }
            
            # For backward compatibility, fallback to inline SQL if template rendering fails
            try:
                sql = self.execute_sql_template(
                    'average',           # Template directory
                    'calculate',         # Template file
                    template_params
                )
                return sql
            except Exception as e:
                self.logger.warning(f"Template rendering failed, using inline SQL: {str(e)}")
                
                return f"""
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        f.event_id,
                        {', '.join(column_calcs)}
                    FROM features.{table_name} f
                    JOIN features.event_mapping em ON f.event_id = em.event_id
                    ORDER BY em.event_date, fight_id
                """
        except Exception as e:
            self.logger.error(f"Error generating SQL for {table_name}: {str(e)}")
            return ""
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute average calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate averages for
            columns: The list of columns to calculate averages for
            
        Returns:
            DataFrame with average calculation results
        """
        self.logger.info(f"Executing average calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate averages
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed average calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated averages: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing average calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating averages: {str(e)}")
            raise
            
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the average calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used, kept for API compatibility)
            max_workers: Number of worker threads (not used, kept for API compatibility)
            table_pattern: Optional pattern to filter tables (e.g., 'body' for body strike tables)
            
        Returns:
            Dictionary of results by table
        """
        self.logger.info(f"Starting average calculation for all tables")
        print(f"\n=== Starting average calculation ===")
        
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
                self.execute_layer_update(
                    calculation_sql=sql,
                    table_name=table_name,
                    schema=self.schema,
                    batch_size=100000
                )
                
                self.logger.info(f"Completed average calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
        
        successful_tables = sum(1 for df in results.values() if not df.empty)
        print(f"\n=== Completed average calculation: {successful_tables}/{total_tables} tables successfully processed ===\n")
        return results

    def save_for_table(self, table_name: str, columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
        """
        Save average calculation results for a table.
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
            
        self.logger.info(f"Saving {len(df)} average features for table: {table_name}")
        print(f"  └─ Saving {len(df)} average features for {table_name}")
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
            
            self.logger.info(f"Successfully saved average features for {table_name}")
            print(f"  └─ Successfully saved to database")
            return df
        except Exception as e:
            self.logger.error(f"Error saving average features for {table_name}: {str(e)}")
            print(f"  └─ Error saving to database: {str(e)}")
            raise
