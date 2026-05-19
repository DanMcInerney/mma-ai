from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from typing import List, Set, Dict, Optional, Any
import pandas as pd
import logging

class LeagueCalculator(BaseCalculator):
    """Calculate weightclass averages for all stats from 2014 onwards."""
    
    def __init__(self, context_or_conn):
        """
        Initialize the league calculator.
        
        Args:
            context_or_conn: CalculatorContext or database connection
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
        
        self.layer_suffix = '_league'
        self.logger = logging.getLogger(__name__)
        
        # Set up stat tables
        self.stat_tables = self.context.feature_utils.get_stat_tables()
    
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """Calculate weightclass averages for each column.
        
        Args:
            table_name: The source table name
            columns: Optional list of columns to calculate averages for
            
        Returns:
            SQL query string for the calculations
        """
        # Get columns if not provided
        if columns is None:
            columns = self.get_features(table_name)
            
        column_calcs = []
        for col in columns:
            # Skip columns that are already derived stats
            if any(suffix in col for suffix in ['_league', '_sdev']):
                continue
                
            column_calcs.append(f"""
                AVG({col}) OVER (
                    PARTITION BY fm.weightclass
                    ORDER BY em.event_date, f.fight_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
                ) AS {col}_league
            """)
        
        if not column_calcs:
            self.logger.info(f"No columns to process for {table_name}")
            return ""
            
        return f"""
            SELECT 
                f.fight_id,
                f.fighter_id,
                f.event_id,
                {', '.join(column_calcs)}
            FROM features.{table_name} f
            JOIN features.event_mapping em ON f.event_id = em.event_id
            JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
            WHERE em.event_date >= '2014-01-01'
            ORDER BY em.event_date, fight_id
        """
    
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute league average calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate league averages for
            columns: The list of columns to calculate league averages for
            
        Returns:
            DataFrame with league average calculation results
        """
        self.logger.info(f"Executing league average calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate league averages
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed league average calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated league averages: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing league average calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating league averages: {str(e)}")
            raise
            
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the league calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used, kept for API compatibility)
            max_workers: Number of worker threads (not used, kept for API compatibility)
            table_pattern: Optional pattern to filter tables (e.g., 'body' for body strike tables)
            
        Returns:
            Dictionary of results by table
        """
        self.logger.info(f"Starting league average calculation for all tables")
        print(f"\n=== Starting league average calculation ===")
        
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
                # Get all columns for the table
                columns = self.stat_tables[table_name]
                
                # Process the table
                self.logger.info(f"Processing table {table_name} with {len(columns)} columns")
                print(f"[{i}/{total_tables}] Processing table {table_name} with {len(columns)} columns")
                
                # Generate SQL for the table
                sql = self.calculate_for_table(table_name, columns)
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
                
                self.logger.info(f"Completed league average calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
                
        self.logger.info(f"League average calculation for all tables completed")
        print(f"=== League average calculation completed ===\n")
        return results
