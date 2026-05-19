import logging
import pandas as pd
from typing import Set, List, Dict, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils

class MinimumSdevCalculator(BaseCalculator):
    """
    Precompute minimum standard deviation values for each stat by weightclass.
    
    For each table in self.stat_tables, compute the 5th percentile of each stat's
    existing _sdev values (or _dec_sdev if decay=True), which will serve as a minimum 
    value for adjusted performance calculations to prevent division by very small numbers.
    
    The results are stored in a table:
         features.<table>_minimum_sdev  (or _minimum_dec_sdev if decay=True)
         
    with each computed column named as <stat>_min_sdev (or _min_dec_sdev if decay=True).
    """

    def __init__(self, context_or_conn, decay: bool = False, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize the minimum standard deviation calculator.
        
        Args:
            context_or_conn: CalculatorContext or database connection
            decay: Whether to use decayed standard deviations
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
        
        self.decay = decay
        self.table_suffix = '_minimum_dec_sdev' if decay else '_minimum_sdev'
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.logger = logging.getLogger(__name__)
        
        # Date range (matching with first_time_fighters_sdev_calc.py)
        self.start_date = '2014-01-01'
        self.end_date = '2023-01-01'
        
        # Set up stat tables and include/exclude patterns
        self.stat_tables = self.context.feature_utils.get_stat_tables()

        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)

        # Removed automatic calculation during initialization to prevent double execution
        # The calculation will now only happen when run() is explicitly called

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
            if hasattr(self.context, 'sql_manager') and self.context.sql_manager is not None:
                sql = self.context.sql_manager.render_template(
                    template_name, operation, params
                )
            else:
                sql = self.sql_template_manager.render_template(
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

    def _create_minimum_sdev_table(self, table_name: str, columns: List[str]) -> None:
        """Create a new table for storing minimum standard deviation values by weightclass."""
        try:
            sdev_suffix = '_dec_sdev' if self.decay else '_sdev'
            min_suffix = '_min_dec_sdev' if self.decay else '_min_sdev'
            
            # Drop existing table if it exists
            drop_sql = text(f"DROP TABLE IF EXISTS features.{table_name}{self.table_suffix} CASCADE;")
            self.conn.execute(drop_sql)
            self.conn.commit()

            # Filter columns based on exclude patterns
            filtered_columns = [col for col in columns if self.should_process_column(col)]
            
            if not filtered_columns:
                self.logger.warning(f"No columns to process for {table_name} after filtering")
                return

            # Build percentile expressions for each stat column
            percentile_selects = []
            for col in filtered_columns:
                sdev_col = f"{col}{sdev_suffix}"
                # Calculate the 5th percentile of existing sdev values by weightclass
                percentile_selects.append(f"""
                    CAST(PERCENTILE_CONT(0.05) WITHIN GROUP (
                        ORDER BY t.{sdev_col}
                    ) AS real) AS {col}{min_suffix}
                """)
            
            if not percentile_selects:
                self.logger.warning(f"No percentile expressions generated for {table_name}")
                return
                
            percentile_sql_str = ",\n    ".join(percentile_selects)

            # Use inline SQL for reliability
            self.logger.info(f"Creating minimum {'decayed ' if self.decay else ''}sdev table for {table_name}")
            create_sql = f"""
            CREATE TABLE features.{table_name}{self.table_suffix} AS
            WITH fighter_fights AS (
                SELECT
                    t.fighter_id,
                    fm.weightclass,
                    t.*
                FROM features.{table_name} t
                JOIN features.fight_mapping fm ON fm.fight_id = t.fight_id
                JOIN features.event_mapping e ON fm.event_id = e.event_id
                WHERE e.event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
            )
            SELECT
                weightclass,
                {percentile_sql_str}
            FROM fighter_fights t
            GROUP BY weightclass
            ORDER BY weightclass;
            """
            
            # Execute the SQL
            self.conn.execute(text(create_sql))
            self.conn.commit()

            # Create index on weightclass
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}{self.table_suffix}_wclass
            ON features.{table_name}{self.table_suffix}(weightclass);
            """
            self.conn.execute(text(index_sql))
            self.conn.commit()
            
            self.logger.info(f"Successfully created minimum sdev table: {table_name}{self.table_suffix}")

        except Exception as e:
            self.logger.error(f"Error creating minimum sdev table for {table_name}: {str(e)}")
            self.conn.rollback()
            raise

    def _validate_minimum_sdev_stats(self, table_name: str) -> None:
        """Validate the computed minimum standard deviation statistics."""
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
                
            # Check for negative values (which shouldn't happen with percentiles)
            for col in stats_df.columns:
                if col != 'weightclass':
                    if stats_df[col].min() < 0:
                        self.logger.warning(f"Negative minimum sdev in {table_name}.{col}")
                    if stats_df[col].isnull().any():
                        self.logger.warning(f"NULL values found in {table_name}.{col}")
            
            self.logger.info(f"Validated {table_name}{self.table_suffix}: {len(stats_df)} rows, {len(stats_df.columns)} columns")
        except Exception as e:
            self.logger.error(f"Error validating stats for {table_name}: {str(e)}")

    def precompute_minimum_sdev_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Precompute minimum standard deviation statistics for all feature tables.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        tables_to_process = list(self.stat_tables.keys())
        total_tables = len(tables_to_process)
        
        self.logger.info(f"Starting minimum {'decayed ' if self.decay else ''}sdev calculation for {total_tables} tables")
        print(f"\n=== Starting minimum {'decayed ' if self.decay else ''}sdev calculation for {total_tables} tables ===")
        
        # For each stat table, precompute minimum standard deviations
        for i, table_name in enumerate(tables_to_process, 1):
            try:
                # Get columns for this table
                all_columns = self.stat_tables.get(table_name, [])
                
                # Filter columns based on include/exclude patterns
                filtered_columns = [col for col in all_columns if self.should_process_column(col)]
                
                if not filtered_columns:
                    self.logger.info(f"No columns match patterns for {table_name}")
                    print(f"  └─ Skipping {table_name}: No columns match patterns")
                    continue
                
                self.logger.info(f"[{i}/{total_tables}] Computing minimum {'decayed ' if self.decay else ''}sdev stats for {table_name} with {len(filtered_columns)} columns")
                print(f"  └─ [{i}/{total_tables}] Processing {table_name} with {len(filtered_columns)} columns")
                
                # Create the minimum sdev table
                self._create_minimum_sdev_table(table_name, filtered_columns)
                
                # Validate the results
                self._validate_minimum_sdev_stats(table_name)
                
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
                
        print(f"=== Minimum {'decayed ' if self.decay else ''}sdev calculation completed ===\n")
        self.logger.info(f"Completed minimum {'decayed ' if self.decay else ''}sdev calculation for {len(results)} tables")
        return results

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """Not used for this calculator as we handle all tables at once."""
        return ""
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Not used for this calculator as we handle all tables at once."""
        return pd.DataFrame()
        
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the minimum standard deviation calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used)
            max_workers: Number of workers for parallel execution (not used)
            table_pattern: Optional pattern to filter tables
            
        Returns:
            Dictionary of precomputed statistics by table
        """
        return self.precompute_minimum_sdev_for_all_tables()
