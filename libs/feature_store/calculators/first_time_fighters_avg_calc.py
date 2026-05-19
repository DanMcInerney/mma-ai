import logging
import os
import pandas as pd
from typing import Dict, List, Set, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils


class FirstTimeOpponentAverageCalculator(BaseCalculator):
    """
    Precompute a table of weightclass-level average opponent stats for first-time fighters.
    
    For each table in self.stat_table, gather all the <stat>_opp values from each first-time fighter's row data. 
    Calculate the average of all those values and alias it as <stat>_opp_wc_avg.
    Gather the data from all first-time fighters' rows, then limit it to only fights from 2014-2023 before doing 
    the average.
    
    The results are stored in a table:
    
         features.<table>_first_time_opp_avg_stats
         
    with each computed column named as <stat>_opp_wc_avg.
    """
    
    def __init__(self, context_or_conn, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize the first time average calculator.
        
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
        self.table_suffix = '_first_time_opp_avg_stats'  # Updated table suffix
        self.start_date = '2014-01-01'  # Hardcoded start date
        self.end_date = '2023-01-01'    # Hardcoded end date
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

    def _create_first_time_avg_table(self, table_name: str, columns: List[str]) -> None:
        """Create a new table for storing first-time fighter opponent stat averages.
        
        Args:
            table_name: The base table name (e.g., 'sig_str')
            columns: List of columns to compute averages for
        """
        try:
            # Drop existing table if it exists
            drop_sql = text(f"DROP TABLE IF EXISTS features.{table_name}{self.table_suffix} CASCADE;")
            self.conn.execute(drop_sql)
            self.conn.commit()
            
            # Filter for only columns ending with '_opp'
            opp_columns = [col for col in columns if col.endswith('_opp')]
            
            if not opp_columns:
                self.logger.warning(f"No opponent stats found for {table_name}")
                return

            # Build AVG expressions for each opponent stat column with the correct suffix
            avg_selects = [
                f"CAST(AVG(t.{col}) AS real) AS {col}_wc_avg"
                for col in opp_columns
            ]
            avg_sql_str = ",\n    ".join(avg_selects)

            # Skip the template approach and use inline SQL directly, which is more reliable
            self.logger.info(f"Using inline SQL for first-time fighter opponent average calculation for {table_name}")
            create_sql = f"""
            CREATE TABLE features.{table_name}{self.table_suffix} AS
            WITH first_fights AS (
                SELECT
                    t.fighter_id,
                    fm.weightclass,
                    e.event_date,
                    ROW_NUMBER() OVER (
                        PARTITION BY t.fighter_id 
                        ORDER BY e.event_date ASC
                    ) AS rn,
                    t.*
                FROM features.{table_name} t
                JOIN features.fight_mapping fm ON fm.fight_id = t.fight_id
                JOIN features.event_mapping e ON fm.event_id = e.event_id
            )
            SELECT
                weightclass,
                {avg_sql_str}
            FROM first_fights t
            WHERE rn = 1
            AND event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
            GROUP BY weightclass
            ORDER BY weightclass;
            """
            
            # Execute with parameter binding for start_date and end_date
            sql_statement = text(create_sql)
            self.conn.execute(
                sql_statement, 
                {
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            )
            self.conn.commit()

            # Create index on weightclass
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}{self.table_suffix}_wclass
            ON features.{table_name}{self.table_suffix}(weightclass);
            """
            self.conn.execute(text(index_sql))
            self.conn.commit()
            
            self.logger.info(f"Successfully created first-time opponent average table: {table_name}{self.table_suffix}")

        except Exception as e:
            self.logger.error(f"Error creating first time opponent average table for {table_name}: {str(e)}")
            self.conn.rollback()
            raise

    def _validate_avg_stats(self, table_name: str) -> None:
        """Validate the computed opponent average statistics.
        
        Args:
            table_name: Table name to validate
        """
        try:
            # Query the computed stats
            validation_sql = f"""
            SELECT * FROM features.{table_name}{self.table_suffix}
            """
            stats_df = pd.read_sql(validation_sql, self.conn)
            
            # Validation checks
            if stats_df.empty:
                self.logger.warning(f"No statistics computed for {table_name}")
                return
                
            if stats_df['weightclass'].nunique() < 8:
                self.logger.warning(f"Missing weightclasses in {table_name} (found {stats_df['weightclass'].nunique()} of 8)")
                
            # Check for negative values (shouldn't happen for most stats)
            for col in stats_df.columns:
                if col != 'weightclass':
                    if stats_df[col].min() < 0:
                        self.logger.warning(f"Negative average in {table_name}.{col}")
                    if stats_df[col].isnull().any():
                        self.logger.warning(f"NULL values found in {table_name}.{col}")
            
            self.logger.info(f"Validated {table_name}{self.table_suffix}: {len(stats_df)} rows, {len(stats_df.columns)} columns")
        except Exception as e:
            self.logger.error(f"Error validating stats for {table_name}: {str(e)}")
            raise

    def precompute_first_time_avg_stats_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Precompute opponent average statistics for all feature tables.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        tables_to_process = list(self.stat_tables.keys())
        total_tables = len(tables_to_process)
        
        self.logger.info(f"Starting first-time fighter opponent average calculation for {total_tables} tables")
        print(f"\n=== Starting first-time fighter opponent average calculation for {total_tables} tables ===")
        
        # For each stat table, precompute averages of opponent stats
        for i, table_name in enumerate(tables_to_process, 1):
            try:
                # Filter columns based on include/exclude patterns
                all_columns = self.stat_tables.get(table_name, [])
                filtered_columns = self.filter_columns(all_columns)
                
                if not filtered_columns:
                    self.logger.info(f"No columns match patterns for {table_name}")
                    print(f"  └─ Skipping {table_name}: No columns match patterns")
                    continue
                
                # Filter for opponent stat columns
                opp_columns = [col for col in filtered_columns if col.endswith('_opp')]
                
                if not opp_columns:
                    self.logger.info(f"No opponent stat columns for {table_name}")
                    print(f"  └─ Skipping {table_name}: No opponent stat columns")
                    continue
                
                self.logger.info(f"[{i}/{total_tables}] Computing first-time fighter opponent avg stats for {table_name} with {len(opp_columns)} columns")
                print(f"  └─ [{i}/{total_tables}] Processing {table_name} with {len(opp_columns)} columns")
                
                # Create the average table
                self._create_first_time_avg_table(table_name, filtered_columns)
                
                # Validate the results
                self._validate_avg_stats(table_name)
                
                # Store computed results
                try:
                    stats_df = pd.read_sql(f"SELECT * FROM features.{table_name}{self.table_suffix}", self.conn)
                    results[table_name] = stats_df
                    print(f"  └─ ✓ Completed {table_name}: {len(stats_df)} rows")
                except Exception as e:
                    self.logger.warning(f"Could not load stats for {table_name}: {str(e)}")
                    print(f"  └─ ✗ Could not load stats for {table_name}")
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"  └─ ✗ Error processing {table_name}: {str(e)}")
                # Continue with other tables
                
        print(f"=== First-time fighter opponent average calculation completed ===\n")
        self.logger.info(f"Completed first-time fighter opponent average calculation for {len(results)} tables")
        return results

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Calculate SQL for creating first-time fighter opponent averages for a specific table.
        Not typically used directly as we handle all tables in run().
        
        Args:
            table_name: The table to calculate for
            columns: Optional list of columns to process
            
        Returns:
            SQL query for calculation (empty string as this is handled elsewhere)
        """
        return ""
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table. Not typically used directly 
        as we handle all tables in run().
        
        Args:
            table_name: The table to execute for
            columns: Optional list of columns to process
            
        Returns:
            DataFrame with results
        """
        return pd.DataFrame()
        
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the first time opponent average calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used)
            max_workers: Number of workers for parallel execution (not used)
            table_pattern: Optional pattern to filter tables
            
        Returns:
            Dictionary of precomputed statistics by table
        """
        # Precompute all tables and return the results
        return self.precompute_first_time_avg_stats_for_all_tables()
