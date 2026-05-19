from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from typing import List, Set, Dict, Optional, Any
from math import log
import pandas as pd
import logging
import os


class TimedecAvgCalculator(BaseCalculator):
    """Calculate time-dec rolling average with a fixed half-life in years (e.g., 1.5 years),
    ignoring NULL values so that only non-null fights contribute to the average."""

    def __init__(self, context_or_conn, decay_rate_years: float, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize calculator with a decay rate in years based on a given half-life.

        Args:
            context_or_conn: CalculatorContext or database connection
            decay_rate_years: Half-life in years (e.g. 1.0 for 1.0 year half-life)
            include_patterns: Set of patterns to include in the decayavg calculation
            exclude_patterns: Set of patterns to exclude from the decayavg calculation
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
        
        self.layer_suffix = '_dec_avg'
        self.decay_rate = log(2) / decay_rate_years  # Calculate decay rate from half-life in years
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
        """
        Calculate time-decayed average for the given columns using the specified half-life,
        ignoring fights where the column values are NULL.

        Instead of relying solely on window functions, we:
        - Extract a "base" table of all fights.
        - Use a second CTE to represent the "current" fights (the fight for which we're calculating the decayed average).
        - Self-join 'base' to 'current_fights', ensuring we only include fights that happened on or before the current fight's event_date.
        - Compute exponential decay weights using the difference between the current fight's event_date and the base fight's event_date.
        - Aggregate these values to produce a time-decayed rolling average.

        Args:
            table_name: The source table name
            columns: Optional list of columns to calculate average for

        Returns:
            SQL query string for the calculations
        """
        try:
            # Get columns if not provided
            if columns is None:
                columns = self.get_features(table_name)
                
            # Filter out columns already ending with _dec_avg to avoid re-processing
            filtered_cols = [col for col in columns if not col.endswith(self.layer_suffix)]
            if not filtered_cols:
                self.logger.info(f"No columns to process for {table_name}")
                return ""

            self.logger.debug(f"TimedecAvgCalculator processing table: {table_name}")
            self.logger.debug(f"Columns after filtering: {filtered_cols}")

            # Build the decayed average expressions for each column
            dec_cols = []
            for col in filtered_cols:
                dec_cols.append(f"""
                    SUM(
                        CASE WHEN b.{col} IS NOT NULL
                             THEN b.{col} * EXP(-1 * {self.decay_rate} * ((c.event_date - b.event_date)::INTEGER / 365.25))
                        END
                    ) 
                    /
                    NULLIF(
                        SUM(
                            CASE WHEN b.{col} IS NOT NULL
                                 THEN EXP(-1 * {self.decay_rate} * ((c.event_date - b.event_date)::INTEGER / 365.25))
                            END
                        ),
                        0
                    ) AS {col}{self.layer_suffix} 
                """)

            # Check if we have a SQL template for time_dec_avg
            template_dir = os.path.join('libs', 'feature_store', 'sql_templates', 'time_dec_avg')
            if os.path.exists(template_dir) and os.path.exists(os.path.join(template_dir, 'calculate.sql')):
                # Template parameters
                template_params = {
                    'schema': 'features',
                    'table_name': table_name,
                    'columns': filtered_cols,
                    'decay_rate': self.decay_rate,
                    'expressions_str': ",\n        ".join(dec_cols)
                }
                
                # Use template for SQL generation
                try:
                    sql = self.execute_sql_template(
                        'time_dec_avg',     # Template directory
                        'calculate',        # Template file
                        template_params
                    )
                    return sql
                except Exception as e:
                    self.logger.warning(f"Template rendering failed, using inline SQL: {str(e)}")
            else:
                self.logger.debug("No SQL template found for time_dec_avg, using inline SQL")
            
            # Fallback to inline SQL if template is not available or fails
            return f"""
                WITH base AS (
                    SELECT 
                        f.fight_id,
                        f.fighter_id,
                        f.event_id,
                        em.event_date,
                        {', '.join(filtered_cols)}
                    FROM features.{table_name} f
                    JOIN features.event_mapping em ON f.event_id = em.event_id
                ),
                current_fights AS (
                    SELECT 
                        fighter_id,
                        fight_id,
                        event_id,
                        event_date
                    FROM base
                )
                SELECT
                    c.fight_id,
                    c.fighter_id,
                    c.event_id,
                    {', '.join(dec_cols)}
                FROM current_fights c
                LEFT JOIN base b ON c.fighter_id = b.fighter_id
                                AND b.event_date <= c.event_date
                GROUP BY c.fight_id, c.fighter_id, c.event_id, c.event_date
                ORDER BY c.event_date, c.fight_id
            """
        except Exception as e:
            self.logger.error(f"Error generating SQL for {table_name}: {str(e)}")
            return ""
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute time-decayed average calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate time-decayed averages for
            columns: The list of columns to calculate time-decayed averages for
            
        Returns:
            DataFrame with time-decayed average calculation results
        """
        self.logger.info(f"Executing time-decayed average calculation for {table_name}")
        print(f"  └─ Executing calculation for {table_name}")
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                print(f"  └─ Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate time-decayed averages
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed time-decayed average calculation for {table_name} with {len(df)} rows")
            print(f"  └─ Successfully calculated time-decayed averages: {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing time-decayed average calculation for {table_name}: {str(e)}")
            print(f"  └─ Error calculating time-decayed averages: {str(e)}")
            raise
            
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the time-decayed average calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used, kept for API compatibility)
            max_workers: Number of worker threads (not used, kept for API compatibility)
            table_pattern: Optional pattern to filter tables (e.g., 'body' for body strike tables)
            
        Returns:
            Dictionary of results by table
        """
        self.logger.info(f"Starting time-decayed average calculation for all tables")
        print(f"\n=== Starting time-decayed average calculation ===")
        
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
                
                self.logger.info(f"Completed time-decayed average calculation for {table_name}")
                print(f"[{i}/{total_tables}] ✓ Completed table {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"[{i}/{total_tables}] ✗ Error processing table {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
                
        self.logger.info(f"Time-decayed average calculation for all tables completed")
        print(f"=== Time-decayed average calculation completed ===\n")
        return results
