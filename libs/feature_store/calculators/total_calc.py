from libs.feature_store.base_calculator import BaseCalculator
from typing import List, Dict, Any
import pandas as pd
import logging
from libs.feature_store.calculator_context import CalculatorContext

class TotalCalculator(BaseCalculator):
    """
    Calculates cumulative career totals for statistics within the
    `features.fight_stats_derived` table.

    Identifies numeric columns suitable for summation and calculates
    a running total for each fighter across their fights, ordered by date.
    The results are saved back into `fight_stats_derived` with a '_total' suffix.
    """

    def __init__(self, conn_or_context, calculator_type='single_table'):
        """
        Initialize the TotalCalculator.

        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext instance.
            calculator_type: Type of calculator, defaults to 'single_table'.
        """
        # Handle context initialization (similar to PerMinCalculator)
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)

        self.table_name = 'fight_stats_derived' # Target table
        self.schema = 'features'
        self.features = None # Will store columns to be totalled
        self.feature_type = 'total' # Identifier for potential SQL templates or logging
        self.layer_suffix = '_total'
        self.logger = logging.getLogger(__name__)

        # Define patterns to exclude from summation
        # Excludes IDs, time, ratios, pre-calculated derived stats, etc.
        self.exclude_patterns = {
            '_id',
            '_acc', '_def', '_ratio', '_per_min', # Ratios/rates
            '_avg', '_sdev', '_dec', '_opp', '_adjperf', '_slope', # Advanced L3 stats
            '_total', # Avoid calculating total of a total
            'age', 'days_since_last_fight', 'reach', 'height', 'ape', 'ufcage', # Static/biographic info
            'raw', # Avoid calculating total of a raw column
        }

        # Add exclude patterns to the base class filter
        for pattern in self.exclude_patterns:
            self.add_exclude_pattern(pattern)

        # Optional: Setup execution plan like PerMinCalculator if desired
        # self.execution_plan.add_operation('load_features', self.get_features)
        # self.execution_plan.add_operation('calculate_features', self.calculate)
        # self.execution_plan.add_operation('save_features', self.save)

    def get_features(self) -> List[str]:
        """
        Get columns from fight_stats_derived suitable for cumulative summation.

        Returns:
            List of feature column names to calculate totals for.
        """
        self.logger.debug(f"Getting features for Total calculation from {self.schema}.{self.table_name}")
        try:
            # Use context feature_utils if available
            if hasattr(self.context, 'feature_utils') and self.context.feature_utils:
                self.features = self.context.feature_utils.get_columns_from_table(
                    self.schema,
                    self.table_name,
                    exclude_strs=self.exclude_patterns # Use defined exclusions
                )
                self.logger.info(f"Identified {len(self.features)} features for total calculation: {self.features}")
            else:
                # Fallback if context/utils not available (e.g., basic tests)
                self.logger.warning("CalculatorContext or feature_utils not available. Cannot dynamically get features.")
                self.features = []
        except Exception as e:
            self.logger.error(f"Error getting features from {self.schema}.{self.table_name}: {e}", exc_info=True)
            self.features = []

        return self.features

    def calculate(self) -> Dict[str, Any]:
        """
        Generate the SQL query to calculate running totals for the identified features.

        Returns:
            Dictionary containing the generated SQL query under the 'sql' key,
            or indicating skipped status if no features are found.
        """
        # Ensure features are loaded
        if not self.features:
            self.get_features()

        if not self.features:
            self.logger.warning("No features identified for total calculation. Skipping SQL generation.")
            return {"status": "skipped", "message": "No features to calculate", "table_name": self.table_name}

        self.logger.debug(f"Generating total calculation SQL for {len(self.features)} features.")

        # Generate the SUM() OVER(...) expression for each feature
        # Using COALESCE to handle potential NULLs
        # Ordering by event_date ensures correct temporal order
        sum_expressions = []
        for feat in self.features:
            # Quote feature names for safety
            quoted_feat = f'"{feat}"'
            quoted_new_col = f'"{feat}{self.layer_suffix}"' # e.g., "sig_str_land_total"
            sum_expressions.append(f"""
                SUM(COALESCE(fs.{quoted_feat}, 0)) OVER (
                    PARTITION BY fs.fighter_id
                    ORDER BY e.event_date ASC, fs.fight_id ASC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS {quoted_new_col}""")

        expressions_str = ",\n                ".join(sum_expressions)

        # Construct the final SQL query
        # This query selects necessary IDs and the calculated total columns
        calculation_sql = f"""
        SELECT
            fs.fight_id,
            fs.fighter_id,
            {expressions_str}
        FROM {self.schema}.{self.table_name} fs
        JOIN {self.schema}.fight_mapping f ON fs.fight_id = f.fight_id
        JOIN {self.schema}.event_mapping e ON f.event_id = e.event_id
        WHERE fs.fighter_id IS NOT NULL AND fs.fight_id IS NOT NULL
        ORDER BY e.event_date, fs.fight_id -- Optional, but good for debugging
        """

        return {
            "status": "success",
            "sql": calculation_sql.strip(),
            "feature_count": len(self.features),
            "table_name": self.table_name
        }

    def save(self) -> pd.DataFrame:
        """
        Execute the calculation SQL and save the results (new total columns)
        back into the `fight_stats_derived` table.

        Returns:
            An empty DataFrame indicating success (as data is updated in place)
            or failure.
        """
        # Ensure features are loaded (needed to determine new column names)
        if not self.features:
            self.get_features()

        if not self.features:
            self.logger.warning("No features identified. Skipping save operation.")
            return pd.DataFrame() # Indicate nothing was done

        # Get the list of new column names that will be created/updated
        total_columns = [f"{feat}{self.layer_suffix}" for feat in self.features]

        # Generate the calculation SQL
        calc_result = self.calculate()
        sql = calc_result.get("sql", "")

        if not sql or calc_result.get("status") != "success":
            self.logger.error("Failed to generate calculation SQL. Cannot save results.")
            return pd.DataFrame() # Indicate failure

        self.logger.info(f"Executing update on {self.schema}.{self.table_name} to add/update {len(total_columns)} total columns.")

        # Execute the update using the calculator's method
        try:
            # The method is on BaseCalculator, call directly via self
            self.execute_calculator_update(
                calculation_sql=sql,
                table_name=self.table_name,
                new_columns=total_columns,
                schema=self.schema
            )
            self.logger.info(f"Successfully updated {self.schema}.{self.table_name} with total columns.")
            return pd.DataFrame({"success": [True]}) # Indicate success

        except AttributeError as e:
            # Handle cases where the method might not exist on BaseCalculator
            # (shouldn't happen if inheriting correctly)
            self.logger.error(f"BaseCalculator object missing 'execute_calculator_update' method: {e}", exc_info=True)
            raise NotImplementedError(f"BaseCalculator requires 'execute_calculator_update' method: {e}")
        except Exception as e:
            self.logger.error(f"Failed to save total columns to '{self.schema}.{self.table_name}': {e}", exc_info=True)
            return pd.DataFrame() # Indicate failure

    def run(self) -> Dict[str, Any]:
        """
        Executes the full calculation pipeline: get features, calculate SQL, save results.

        Returns:
            A dictionary summarizing the execution result.
        """
        self.logger.info(f"Running {self.__class__.__name__} on {self.schema}.{self.table_name}...")
        try:
            # Chain the operations: get features -> calculate -> save
            features = self.get_features()
            if not features:
                message = f"No features found to process in {self.schema}.{self.table_name}."
                self.logger.warning(message)
                return {"status": "skipped", "message": message}

            # Calculate and Save step (save calls calculate internally)
            result_df = self.save()

            if not result_df.empty and result_df.iloc[0].get("success"):
                status = "completed"
                message = f"{self.__class__.__name__} completed successfully for {len(features)} features."
                self.logger.info(message)
            else:
                status = "failed"
                message = f"{self.__class__.__name__} failed during save operation."
                self.logger.error(message)

            return {"status": status, "message": message, "features_processed": len(features)}

        except Exception as e:
            self.logger.error(f"{self.__class__.__name__} run failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    # execute method might be called by a runner
    def execute(self) -> Dict[str, Any]:
        """ Alias for run() for compatibility with potential runner systems. """
        return self.run()
