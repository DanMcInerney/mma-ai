import logging
import pandas as pd
from typing import Dict, List, Set, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils


class FirstTimeMadCalculator(BaseCalculator):
    """
    Precompute a table of weightclass-level stat median absolute deviations for first-time fighters.

    For each table in self.stat_table, gather all the <stat> values from each first-time fighter's row data. 
    Calculate the median absolute deviation of all those values and alias it as <stat>_wc_mad.
    Gather the data from all first-time fighters' rows, then limit it to only fights from 2014-2023 before doing 
    the MAD calculation.

    The results are stored in a table:
        features.<table>_first_time_mad_stats

    with each computed column named as <stat>_wc_mad
    """

    def __init__(self, conn_or_context, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        # Handle both connection and context objects for flexibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            self.conn = self.context.connection
        else:
            self.conn = conn_or_context
            self.context = CalculatorContext(self.conn)
            
        # Initialize with multi_table calculator type since this works on multiple tables
        super().__init__(self.conn, calculator_type='multi_table')
        
        self.table_suffix = '_first_time_mad_stats'  # Updated table suffix
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.start_date = '2014-01-01'  # Hardcoded start date
        self.end_date = '2023-01-01'    # Hardcoded end date
        self.logger = logging.getLogger(__name__)
        
        # Set up stat tables and include/exclude patterns
        self.stat_tables = self.feature_utils.get_stat_tables()

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

    def _create_first_time_mad_table(self, table_name: str, columns: List[str]) -> None:
        """Create a new table for storing first-time fighter stat median absolute deviations."""
        try:
            # Drop existing table if it exists
            drop_sql = text(f"DROP TABLE IF EXISTS features.{table_name}{self.table_suffix} CASCADE;")
            self.conn.execute(drop_sql)
            self.conn.commit()

            # Use a two-step approach to avoid nested aggregate functions
            # First, create CTEs for each column to calculate medians
            median_ctes = []
            for col in columns:
                median_ctes.append(f"""
                {col}_medians AS (
                    SELECT 
                        weightclass,
                        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.{col}) AS median_value
                    FROM first_fights t
                    WHERE rn = 1
                    AND event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
                    GROUP BY weightclass
                )
                """)
            
            median_ctes_str = ",\n".join(median_ctes)
            
            # Build the JOIN clauses for the median CTEs
            join_clauses = []
            for col in columns:
                join_clauses.append(f"JOIN {col}_medians m_{col} ON m_{col}.weightclass = t.weightclass")
            
            join_str = "\n        ".join(join_clauses)

            # Build MAD calculations using the pre-computed medians with proper aliases
            mad_selects = []
            for col in columns:
                mad_selects.append(f"""
                    PERCENTILE_CONT(0.5) WITHIN GROUP (
                        ORDER BY ABS(t2.{col} - m_{col}.median_value)
                    ) AS {col}_wc_mad
                """)
            
            mad_sql_str = ",\n    ".join(mad_selects)
            
            # Use inline SQL directly with the new approach
            self.logger.info(f"Using inline SQL with two-step approach for first-time fighter MAD calculation for {table_name}")
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
            ),
            
            {median_ctes_str}
            
            SELECT
                t.weightclass,
                {mad_sql_str}
            FROM (
                SELECT DISTINCT weightclass
                FROM first_fights
                WHERE rn = 1
                AND event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
            ) t
            {join_str}
            LEFT JOIN first_fights t2 ON t2.weightclass = t.weightclass 
            WHERE t2.rn = 1
            AND t2.event_date BETWEEN '{self.start_date}' AND '{self.end_date}'
            GROUP BY t.weightclass
            ORDER BY t.weightclass;
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

        except Exception as e:
            self.logger.error(f"Error creating first time MAD table for {table_name}: {str(e)}")
            self.conn.rollback()
            raise

    def _validate_mad_stats(self, table_name: str) -> None:
        """Validate the computed stat median absolute deviation statistics."""
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
                self.logger.warning(f"Missing weightclasses in {table_name}")
                
            # Check for negative values (which shouldn't happen with MAD)
            for col in stats_df.columns:
                if col != 'weightclass':
                    if stats_df[col].min() < 0:
                        self.logger.warning(f"Negative MAD in {table_name}.{col}")
                    if stats_df[col].isnull().any():
                        self.logger.warning(f"NULL values found in {table_name}.{col}")
            
        except Exception as e:
            self.logger.error(f"Error validating stats for {table_name}: {str(e)}")

    def precompute_first_time_mad_stats_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Precompute stat median absolute deviation statistics for all feature tables.
        
        Returns:
            Dictionary of precomputed statistics DataFrames by table
        """
        results = {}
        tables_to_process = list(self.stat_tables.keys())
        total_tables = len(tables_to_process)
        
        self.logger.info(f"Starting first-time fighter MAD calculation for {total_tables} tables")
        print(f"\n=== Starting first-time fighter MAD calculation for {total_tables} tables ===")
        
        # For each stat table, precompute median absolute deviations of stats
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
                
                self.logger.info(f"[{i}/{total_tables}] Computing first-time fighter MAD stats for {table_name} with {len(filtered_columns)} columns")
                print(f"  └─ [{i}/{total_tables}] Processing {table_name} with {len(filtered_columns)} columns")
                
                # Create the MAD table
                self._create_first_time_mad_table(table_name, filtered_columns)
                
                # Validate the results
                self._validate_mad_stats(table_name)
                
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
                
        print(f"=== First-time fighter MAD calculation completed ===\n")
        self.logger.info(f"Completed first-time fighter MAD calculation for {len(results)} tables")
        return results

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """Not used for this calculator as we handle all tables at once."""
        return ""
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Not used for this calculator as we handle all tables at once."""
        return pd.DataFrame()
        
    def run(self, parallel: bool = False, max_workers: int = 4) -> Dict[str, pd.DataFrame]:
        """
        Run the first-time MAD calculator for all tables.
        This method is provided for compatibility but does nothing since initialization already does the work.
        
        Args:
            parallel: This parameter is ignored
            max_workers: This parameter is ignored
            
        Returns:
            Dictionary of results by table
        """
        # The actual calculation happens in __init__ via precompute_first_time_mad_stats_for_all_tables
        # This method is kept for API compatibility
        return self.precompute_first_time_mad_stats_for_all_tables() 