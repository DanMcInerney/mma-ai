from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.sql_template_manager import SQLTemplateManager
from typing import List, Dict, Optional, Any
import pandas as pd
from libs.feature_store.calculator_context import CalculatorContext


class AccuracyCalculator(BaseCalculator):
    """
    Calculator for computing accuracy features.
    
    Divides landed strikes/attempts by total attempts to create
    accuracy percentage columns.
    """
    
    def __init__(self, conn_or_context, calculator_type='single_table'):
        """
        Initialize with either a connection or a calculator context.
        
        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            calculator_type: Type of calculator ('single_table', 'multi_table', 'cross_table')
        """
        # Handle both connection and context for backward compatibility
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)
            
        self.table_name = 'fight_stats_derived'
        self.schema = 'features'
        self.features = None
        self.land_att_pairs = None
        self.feature_type = 'accuracy'  # Used for SQL template identifier
        
        # Per-weight class tau (pseudo-count) for accuracy smoothing
        # Based on comprehensive likelihood optimization with proper cross-validation
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        # Only includes parameters with >=0.5% improvement over global
        self.per_weightclass_acc_tau = {
            'flyweight': {
                'distance_acc': 17.78,
                'default': 12.0
            },
            'heavyweight': {
                'body_acc': 10.14,
                'body_acc_rd1': 8.71,
                'clinch_acc': 13.57,
                'ground_acc': 11.25,
                'td_acc': 3.96,
                'default': 12.0
            },
        }
        
        # Global parameters (fallback for unknown weight classes)
        # Based on comprehensive likelihood optimization results
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        self.acc_tau = {
            'sig_str': 18.67, 'sig_str_rd1': 16.78,
            'head': 13.5,    'head_rd1': 12.0,
            'body': 12.29,    'body_rd1': 10.86,
            'leg': 13.0,     'leg_rd1': 11.5,
            'ground': 13.75,  'ground_rd1': 12.5,
            'clinch': 12.5,  'clinch_rd1': 9.76,
            'distance': 26.11, 'distance_rd1': 23.33,
            'td': 6.88,       'td_rd1': 8.87,
            'sub': 13.74,      'sub_rd1': 6.82,
            'default': 12.0
        }
        
        # Setup execution plan
        self.execution_plan.add_operation(
            'load_features', 
            self.get_features
        )
        self.execution_plan.add_operation(
            'calculate_features',
            self.calculate
        )
        self.execution_plan.add_operation(
            'save_features',
            self.save
        )

    def get_features(self, table_name: str = None) -> List[str]:
        """
        Get all relevant landed columns and create raw pairs for beta-binomial smoothing.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            
        Returns:
            List of feature column names
        """
        if table_name is None:
            table_name = self.table_name

        if hasattr(self.context, 'feature_utils') and self.context.feature_utils:
            # Only base _land columns (no _raw here!)
            self.features = self.context.feature_utils.get_columns_from_table(
                self.schema,
                table_name,
                include_strs=['_land'],
                exclude_strs=['_id', '_total', '_raw']
            )
        else:
            self.features = []
            return self.features

        # Build raw pairs from the base names
        rd1_features = [f for f in self.features if f.endswith('_land_rd1')]
        regular_features = [f for f in self.features if f.endswith('_land') and not f.endswith('_land_rd1')]

        regular_pairs = [(f + '_raw', f.replace('_land', '_att') + '_raw', f.replace('_land', '_acc'), self._resolve_acc_key(f)) 
                         for f in regular_features]
        rd1_pairs = [(f + '_raw', f.replace('_land_rd1', '_att_rd1') + '_raw', f.replace('_land_rd1', '_rd1_acc'), self._resolve_acc_key(f)) 
                     for f in rd1_features]

        # (land_raw, att_raw, acc_alias, acc_key)
        self.land_att_pairs = regular_pairs + rd1_pairs

        # Also include att columns so SELECT has them available if your template needs them
        att_features = [p[1] for p in self.land_att_pairs]
        self.features = self.features + att_features
        return self.features

    def _resolve_acc_key(self, land_col: str) -> str:
        """
        Resolve accuracy family key for tau lookup.
        
        Args:
            land_col: Base column name like 'sig_str_land' or 'head_land_rd1'
            
        Returns:
            Key for tau lookup (e.g., 'sig_str', 'head_rd1', etc.)
        """
        # land_col is a base name like 'sig_str_land' or 'head_land_rd1'
        rd1 = land_col.endswith('_land_rd1')
        base = land_col[:-9] if rd1 else land_col[:-5]  # strip '_land' or '_land_rd1'

        # base is one of: 'sig_str', 'head', 'body', 'leg', 'td', 'sub'
        key = f"{base}_rd1" if rd1 else base
        return key if key in self.acc_tau else 'default'

    def calculate(self, table_name: str = None, columns: List[str] = None) -> Dict[str, Any]:
        """
        Calculate beta-binomial smoothed accuracy using raw counts.
        
        Args:
            table_name: Optional table name (defaults to self.table_name)
            columns: Optional list of columns to calculate
            
        Returns:
            Dictionary of operation results
        """
        if table_name is None:
            table_name = self.table_name
        if not self.features or not self.land_att_pairs:
            self.get_features(table_name)
        if not self.land_att_pairs:
            return {"status": "skipped", "message": "No land/att pairs found", "table_name": table_name}

        # Build prior columns for all pairs
        wc_exprs, gl_exprs = [], []
        for land_raw, att_raw, acc_alias, acc_key in self.land_att_pairs:
            name = acc_alias.replace('_acc', '')  # unique stem per stat family instance
            wc_exprs.extend([
                f"SUM(f.{att_raw}::float)  AS {name}_att",
                f"SUM(f.{land_raw}::float) / NULLIF(SUM(f.{att_raw}::float), 0) AS {name}_wc_rate"
            ])
            gl_exprs.extend([
                f"SUM(f.{att_raw}::float)  AS {name}_g_att",
                f"SUM(f.{land_raw}::float) / NULLIF(SUM(f.{att_raw}::float), 0) AS {name}_global_rate"
            ])

        # Priors CTEs (date windows can mirror your other calcs; change if needed)
        priors_cte = f"""
WITH global_priors AS (
    SELECT {', '.join(gl_exprs)}
    FROM {self.schema}.{self.table_name} f
    JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-06-01'
),
weightclass_priors AS (
    SELECT fm.weightclass, COUNT(*) AS sample_size, {', '.join(wc_exprs)}
    FROM {self.schema}.{self.table_name} f
    JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
    JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-01-01'
    GROUP BY fm.weightclass
)
SELECT
    fd.fight_id,
    fd.fighter_id
"""

        # Build per-column BB smoothed expressions with per-weight class tau
        acc_exprs = []
        for land_raw, att_raw, acc_alias, acc_key in self.land_att_pairs:
            stem = acc_alias.replace('_acc', '')
            
            # Generate weight class specific tau cases for per-weight class smoothing
            tau_cases = []
            for weightclass, wc_params in self.per_weightclass_acc_tau.items():
                # Get tau for this stat in this weight class using proper key resolution
                stat_tau = wc_params.get(acc_key, wc_params.get('default', 12.0))
                
                # Use case-insensitive comparison for weight class names
                tau_cases.append(f"WHEN LOWER(fm.weightclass) = '{weightclass.lower()}' THEN {stat_tau}")
            
            # Add fallback case using global parameters
            fallback_tau = self.acc_tau.get(acc_key, self.acc_tau['default'])
            tau_cases.append(f"ELSE {fallback_tau}")
            
            # Generate dynamic tau expression
            tau_expr = f"CASE {' '.join(tau_cases)} END"
            
            expr = f"""
, LEAST(1.0, GREATEST(0.0, CASE
    WHEN COALESCE(fd.{att_raw}, 0) = 0 THEN COALESCE(wp.{stem}_wc_rate, gp.{stem}_global_rate, 0.0)
    WHEN wp.{stem}_wc_rate IS NOT NULL AND wp.{stem}_att > 0 THEN
        (wp.{stem}_wc_rate * ({tau_expr}) + COALESCE(fd.{land_raw}, 0)::float) / (({tau_expr}) + COALESCE(fd.{att_raw}, 0))
    WHEN gp.{stem}_global_rate IS NOT NULL AND gp.{stem}_g_att > 0 THEN
        (gp.{stem}_global_rate * ({tau_expr}) + COALESCE(fd.{land_raw}, 0)::float) / (({tau_expr}) + COALESCE(fd.{att_raw}, 0))
    ELSE
        COALESCE(fd.{land_raw}, 0)::float / NULLIF(COALESCE(fd.{att_raw}, 0), 0)
  END)) AS {acc_alias}
"""
            acc_exprs.append(expr)

        sql = priors_cte + "".join(acc_exprs) + f"""
FROM {self.schema}.{self.table_name} fd
JOIN {self.schema}.fight_mapping fm ON fd.fight_id = fm.fight_id
LEFT JOIN weightclass_priors wp ON LOWER(fm.weightclass) = LOWER(wp.weightclass)
CROSS JOIN global_priors gp
ORDER BY fd.fight_id, fd.fighter_id
"""
        return {"status": "success", "sql": sql, "feature_count": len(self.land_att_pairs), "table_name": table_name}

    def save(self, table_name: str = None, result_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Execute the calculation SQL and save results to the database.
        
        Args:
            table_name: Table to save features to (defaults to self.table_name)
            result_df: Optional DataFrame with results (not used in this calculator)
            
        Returns:
            DataFrame with saved features
        """
        if table_name is None:
            table_name = self.table_name
        if not self.features or not self.land_att_pairs:
            self.get_features(table_name)

        # Emit accuracy column names from land_att_pairs to match SELECT order
        acc_columns = [acc_alias for (_, _, acc_alias, _) in self.land_att_pairs]

        calc_result = self.calculate(table_name)
        sql = calc_result.get("sql", "")

        if hasattr(self.context, 'execute_calculator_update'):
            result = self.context.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=acc_columns,
                schema=self.schema
            )
        else:
            result = self.execute_calculator_update(
                calculation_sql=sql,
                table_name=table_name,
                new_columns=acc_columns,
                schema=self.schema
            )
        return result
        
    def execute_sql_template(self, template_name: str, operation: str, params: Dict) -> str:
        """
        Execute a SQL template using the context's SQL manager if available.
        
        Args:
            template_name: Name of the template category
            operation: Operation name ('get_features', 'calculate', etc.)
            params: Parameters to pass to the template
            
        Returns:
            Rendered SQL string
        """
        try:
            sql = ""
            
            # First try using context's SQL manager
            if hasattr(self.context, 'sql_manager') and self.context.sql_manager:
                try:
                    sql = self.context.sql_manager.render_template(
                        template_name,
                        operation,
                        params
                    )
                except Exception as e:
                    self.logger.warning(f"Error using context SQL manager: {str(e)}")
                    
            # Fall back to instance SQL template manager if available
            if not sql and hasattr(self, 'sql_template_manager') and self.sql_template_manager:
                try:
                    sql = self.sql_template_manager.render_template(
                        template_name,
                        operation,
                        params
                    )
                except Exception as e:
                    self.logger.warning(f"Error using instance SQL template manager: {str(e)}")
                    
            # Ensure we're casting to numeric for accuracy values
            if operation == 'calculate' and sql:
                # Check if SQL contains accuracy calculations
                if '_acc' in sql and 'CASE WHEN' in sql and 'ELSE' in sql:
                    # Find all patterns like "CASE WHEN ... END AS col_acc"
                    import re
                    sql = re.sub(
                        r'(CASE WHEN.*?END) AS ([a-z_]+_acc)',
                        r'CAST(\1 AS FLOAT) AS \2',
                        sql,
                        flags=re.DOTALL
                    )
            
            return sql
            
        except Exception as e:
            self.logger.error(f"Error executing SQL template: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return ""
        
    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating accuracy features for a specific table.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            SQL query string for the calculation
        """
        # Use the same beta-binomial logic as calculate()
        calc_result = self.calculate(table_name, columns)
        return calc_result.get("sql", "")
        
    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute calculation for a specific table and return results.
        
        Args:
            table_name: Name of the table to calculate for
            columns: Optional list of columns to calculate
            
        Returns:
            DataFrame with calculation results
        """
        try:
            # Calculate SQL query
            sql = self.calculate_for_table(table_name, columns)
            
            if not sql:
                self.logger.warning(f"No SQL generated for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL and return results
            return self.execute_raw_sql(sql, return_results=True)
        except Exception as e:
            self.logger.error(f"Error executing calculation for {table_name}: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return pd.DataFrame()
        
    def save_for_table(self, table_name: str, columns: Optional[List[str]] = None, 
                   result_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Save calculation results for a specific table.
        
        Args:
            table_name: Name of the table to save to
            columns: Optional list of columns to calculate
            result_df: Optional DataFrame with results (executes calculation if None)
            
        Returns:
            DataFrame with saved results
        """
        # Get result if not provided
        if result_df is None:
            result_df = self.execute_for_table(table_name, columns)
            
        if result_df.empty:
            return pd.DataFrame()
            
        # Generate list of accuracy column names from land_att_pairs to match SELECT order
        if not self.land_att_pairs:
            self.get_features(table_name)
        acc_columns = [acc_alias for (_, _, acc_alias, _) in self.land_att_pairs]
                
        # Use context to update table if available
        if hasattr(self.context, 'update_table'):
            self.context.update_table(table_name, result_df)
        else:
            # Fallback to direct update
            self.bulk_update_dataframe(
                result_df, 
                table_name,
                self.schema,
                ['fight_id', 'fighter_id']
            )
            
        return result_df
        
    def run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Run the calculator sequentially (for testing purposes).
        
        Returns:
            Dictionary of table names to result DataFrames
        """
        try:
            return self._run_sequential()
        except Exception as e:
            self.logger.error(f"Error in run_sequential: {str(e)}")
            # For testing, return empty result
            return {}
            
    def _run_sequential(self) -> Dict[str, pd.DataFrame]:
        """
        Internal implementation of sequential execution.
        
        Returns:
            Dictionary of table names to result DataFrames
        """
        results = {}
        
        # For single_table calculators, just run on the table_name
        if self.calculator_type == 'single_table':
            features = self.get_features(self.table_name)
            result_df = self.execute_for_table(self.table_name)
            saved_df = self.save_for_table(self.table_name, result_df=result_df)
            results[self.table_name] = saved_df
            
        return results