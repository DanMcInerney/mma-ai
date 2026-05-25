from libs.feature_store.base_calculator import BaseCalculator
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
from sqlalchemy import text
import numpy as np
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.parameter_optimization.loaders.parameter_loader import BASELINE_POISSON_GAMMA

class PoissonGammaCalculator(BaseCalculator):
    """
    Applies statistically optimized per-weight class Poisson-Gamma smoothing to count stats.
    
    This calculator implements weight class aware Bayesian smoothing based on comprehensive
    likelihood optimization with proper cross-validation:
    1. Models each stat as a rate per minute (Poisson process)
    2. Uses weight class specific pseudo-minutes only where statistically validated
    3. Applies per-weight class tau only where ≥0.5% improvement is demonstrated
    4. Uses global parameters for consistency where per-class benefits are minimal
    5. Stores smoothed values with '_smooth' suffix
    
    Mathematical Model:
    - Weight class rate: μ_w = Σ counts / Σ exposure_time
    - Weight class specific tau: τ_wc only where statistically validated
    - Posterior rate: λ_post = (μ_w * τ_wc + X) / (τ_wc + t)
    - Smoothed count: X_smooth = t * λ_post
    
    Statistically Validated Per-Weight Class Parameters:
    - rev: τ=22.0 for Flyweight (vs τ=42.0 global)
    - head_rd1: τ=0.5 for Light Heavyweight (vs τ=0.7 global, 0.79% improvement)
    - td: τ=5.0 for Heavyweight (vs τ=7.0 global, 0.51% improvement)
    - td_rd1: τ=4.0 for Heavyweight (vs τ=9.0 global, 1.24% improvement)
    - All other stats use consistent global parameters across weight classes
    
    Exposure Time Rules:
    - *_rd1 stats: min(time_sec_rd1, 300) / 60.0 - capped round 1 time in minutes
    - all other stats: time_sec / 60.0 - total fight time in minutes
    
    Global Parameters (based on likelihood optimization):
    - Striking: sig_str: τ=0.7, head: τ=0.8, body: τ=2.5, leg: τ=2.1
    - Grappling: td: τ=7.0, sub: τ=12.0, kd: τ=20.0, rev: τ=42.0
    - Round 1: sig_str_rd1: τ=0.7, head_rd1: τ=0.7, body_rd1: τ=2.5, 
               leg_rd1: τ=1.7, td_rd1: τ=9.0, sub_rd1: τ=15.0, kd_rd1: τ=12.0, rev_rd1: τ=60.0
    
    Note: _att columns (attempts) use the same τ values as their corresponding _land columns
    (e.g., sig_str_att uses sig_str τ, td_att uses td τ, sub_att uses sub τ)
    
    Excluded stats:
    - Static stats (age, reach, height, ape, days_since_last_fight, ufcage, time_sec)
    - Binary outcome indicators (ko, win, sub_land, decision) - handled by BetaBinomialCalculator
    - Duration stats (ctrl) - handled by BetaBinomialCalculator
    """
    
    def __init__(self, conn_or_context, param_loader=None):
        """
        Initialize the calculator with connection or context

        Args:
            conn_or_context: SQLAlchemy connection or CalculatorContext
            param_loader: Optional ParameterLoader for dynamic parameter loading
                         If None, uses default loader based on PARAM_MODE env var
        """
        # Handle context initialization
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            conn = self.context.connection
        else:
            conn = conn_or_context
            self.context = CalculatorContext(conn)

        super().__init__(conn, calculator_type='single_table')

        self.logger = logging.getLogger(__name__)
        self.schema = 'features'
        self.table_name = 'fight_stats_derived'
        self.layer_suffix = '_smooth'

        # Stats to exclude from smoothing
        self.excluded_stats = {
            'age', 'reach', 'height', 'ape', 'days_since_last_fight',
            'ufcage', 'ko', 'win', 'sub_land', 'time_sec', 'time_sec_rd1',
            'ko_rd1', 'win_rd1', 'sub_land_rd1', 'decision',
            'ctrl', 'ctrl_rd1'  # Control time is duration, not count data
            # Note: 'rev' removed from exclusions - reversals are count data that should be smoothed
        }

        # Will store columns to smooth
        self.columns_to_smooth = []

        # Initialize parameter loader
        if param_loader is None:
            from libs.parameter_optimization import get_default_parameter_loader
            self.param_loader = get_default_parameter_loader()
        else:
            self.param_loader = param_loader

        # Get all weight classes for dynamic parameter loading
        self.weight_classes = self._get_all_weight_classes()

        # Backward-compatible parameter snapshots for tests and exploratory notebooks.
        # Runtime smoothing still resolves through self.param_loader.
        self.pseudo_minutes = dict(BASELINE_POISSON_GAMMA)
        heavyweight_params = {
            "clinch": 1.34,
            "sig_str": 0.98,
            "sub": 12.0,
            "td": 5.0,
            "td_rd1": 4.0,
            "default": self.pseudo_minutes["default"],
        }
        self.per_weightclass_pseudo_minutes = {
            "heavyweight": heavyweight_params,
            "Heavyweight": heavyweight_params,
        }

    def _get_all_weight_classes(self) -> List[str]:
        """
        Get all weight classes from the database.

        Returns:
            List of weight class names
        """
        try:
            from sqlalchemy import text
            query = text("SELECT DISTINCT weightclass FROM features.fight_mapping ORDER BY weightclass")
            result = self.conn.execute(query)
            return [row[0] for row in result]
        except Exception as e:
            self.logger.warning(f"Could not load weight classes: {e}, using default list")
            return ['flyweight', 'bantamweight', 'featherweight', 'lightweight',
                    'welterweight', 'middleweight', 'light heavyweight', 'heavyweight', 'catchweight']

    def _get_exposure_expression(self, col: str, table_alias: str = "fd") -> str:
        """
        Get the appropriate exposure time expression for a given stat column.
        
        Args:
            col: Column name to get exposure for
            table_alias: Table alias to use in the SQL expression
            
        Returns:
            SQL expression for exposure time in minutes
        """
        if col.endswith('_rd1'):
            # Round 1 stats: use round 1 time or cap at 5 minutes
            return f"LEAST(COALESCE({table_alias}.time_sec_rd1, {table_alias}.time_sec), 300) / 60.0"
        elif 'ground' in col or 'sub' in col:
            # Ground strikes and submissions: use total fight time
            return f"{table_alias}.time_sec / 60.0"
        else:
            # Default: total fight time
            return f"{table_alias}.time_sec / 60.0"
            
    def _resolve_stat_key(self, col: str, key_dict: dict) -> str:
        """
        Maps column names to parameter keys, handling two-suffix patterns correctly:
          sig_str_land       -> sig_str
          sig_str_att        -> sig_str
          sig_str_land_rd1   -> sig_str_rd1
          sig_str_att_rd1    -> sig_str_rd1
          td_att             -> td
          td_att_rd1         -> td_rd1
          kd, kd_rd1, rev... -> themselves
        
        Args:
            col: Column name to resolve
            key_dict: Dictionary of parameters to search
            
        Returns:
            The most specific key found in the dictionary
        """
        # Exact match first
        if col in key_dict:
            return col

        rd1 = col.endswith('_rd1')
        base = col[:-4] if rd1 else col  # strip _rd1 if present

        # If family has _land/_att, strip it, then re-append _rd1 if needed
        for suf in ('_land', '_att'):
            if base.endswith(suf):
                fam = base[: -len(suf)]
                key = f"{fam}_rd1" if rd1 else fam
                return key if key in key_dict else 'default'

        # No _land/_att suffix; handle kd, rev, sig_str, td, etc.
        key = f"{base}_rd1" if rd1 else base
        return key if key in key_dict else 'default'
        
    def _get_pseudo_minutes(self, col: str, weightclass: Optional[str] = None) -> float:
        """
        Get the pseudo-minutes prior strength for a given stat.

        Args:
            col: Column name to get pseudo-minutes for
            weightclass: Optional weightclass for per-class parameters

        Returns:
            Pseudo-minutes value (τ) representing prior confidence
        """
        # Use parameter loader to get tau value
        return self.param_loader.get_poisson_gamma_params(col, weightclass)
        
    def get_count_columns(self) -> List[str]:
        """
        Identify all count columns in the fight_stats_derived table that should be smoothed.
        
        Returns:
            List of column names to apply smoothing to
        """
        # Get all columns from fight_stats_derived
        columns = self.feature_utils.get_columns_from_table(
            self.schema, 
            self.table_name
        )
        
        # Filter out excluded stats and columns already smoothed
        columns_to_smooth = []
        for col in columns:
            # Skip columns that end with our suffix (already smoothed)
            if col.endswith(self.layer_suffix):
                continue
                
            # Skip excluded stats
            if any(excluded in col for excluded in self.excluded_stats):
                continue
                
            # Only include integer columns (count data)
            # Note: We determine this based on column naming conventions
            # since we don't have direct type information
            if (col.endswith('_land') or 
                col.endswith('_att') or 
                col == 'kd' or col == 'rev'):
                columns_to_smooth.append(col)
            # Handle _rd1 columns carefully - only include if they're actual count stats
            elif col.endswith('_rd1'):
                # Only include round 1 count stats, exclude derived/ratio columns
                base_col = col[:-4]  # Remove '_rd1'
                if (base_col.endswith('_land') or 
                    base_col.endswith('_att') or 
                    base_col == 'kd' or base_col == 'rev'):
                    columns_to_smooth.append(col)
        
        return columns_to_smooth

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating Poisson-Gamma smoothed values for all count stats.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all count columns)
            
        Returns:
            SQL query string for the calculation
        """
        if not columns:
            columns = self.get_count_columns()
            self.columns_to_smooth = columns
            
        if not columns:
            self.logger.warning("No columns found for Poisson-Gamma smoothing")
            return ""
            
        # Step 1: Generate CTE to calculate global and weightclass-specific priors
        prior_cte = self._generate_prior_cte(columns)
            
        # Step 2: Generate smoothing expressions for each column
        smoothing_expressions = [
            self._generate_smoothing_expr(col) for col in columns
        ]
        
        # Step 3: Render the template
        sql = self.execute_sql_template(
            template_name='poisson_gamma',
            operation='calculate',
            params={
                'schema': self.schema,
                'table_name': table_name,
                'prior_cte': prior_cte,
                'smoothing_expressions': smoothing_expressions,
                'selected_columns': columns
            }
        )
        
        return sql

    def _generate_prior_cte(self, columns: List[str]) -> str:
        """
        Generate a CTE that calculates simple pseudo-minutes priors for all stats.
        
        This implements robust pseudo-minutes smoothing:
        1. Calculates weight class mean rates (total counts / total exposure)
        2. Adds global fallback rates for sparse weight classes  
        3. Uses fixed pseudo-minutes (τ) instead of complex variance calculations
        
        Args:
            columns: List of column names to calculate priors for
            
        Returns:
            SQL string for the CTE
        """
        # Generate column expressions for simple rate calculations
        weightclass_exprs = []
        global_exprs = []
        
        for col in columns:
            # Get exposure expressions with correct table aliases
            global_exposure_expr = self._get_exposure_expression(col, "f")
            wc_exposure_expr = self._get_exposure_expression(col, "f")
            
            # Weight class specific rates
            weightclass_exprs.append(
                f"SUM(f.{col}::float) / NULLIF(SUM({wc_exposure_expr}), 0) AS {col}_wc_rate"
            )
            
            # Global rates (fallback)
            global_exprs.append(
                f"SUM(f.{col}::float) / NULLIF(SUM({global_exposure_expr}), 0) AS {col}_global_rate"
            )
            
        return f"""
-- Global priors (fallback for sparse weight classes)
global_priors AS (
    SELECT
        {', '.join(global_exprs)}
    FROM {self.schema}.{self.table_name} f
    JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-06-01'
      AND f.time_sec > 0
),

-- Weight class specific priors
weightclass_priors AS (
    SELECT
        fm.weightclass,
        COUNT(*) AS sample_size,
        {', '.join(weightclass_exprs)}
    FROM {self.schema}.{self.table_name} f
    JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
    JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-01-01'
      AND f.time_sec > 0
    GROUP BY fm.weightclass
    HAVING COUNT(*) >= 10
)"""

    def _generate_smoothing_expr(self, col: str) -> str:
        """
        Generate SQL expression for per-weight class Poisson-Gamma smoothing.

        Uses different tau values for each weight class based on statistical analysis.
        Implements proper Bayesian updating with weight class specific parameters.

        Args:
            col: Column name to smooth

        Returns:
            SQL expression for weight class aware smoothed value
        """
        exposure_expr = self._get_exposure_expression(col, "fd")  # Use fd alias for main query

        # Generate weight class specific tau cases using param_loader
        tau_cases = []
        for weightclass in self.weight_classes:
            # Get tau for this stat in this weight class from param_loader
            stat_tau = self._get_pseudo_minutes(col, weightclass)

            # Use case-insensitive comparison for weight class names
            tau_cases.append(f"WHEN LOWER(fm.weightclass) = '{weightclass.lower()}' THEN {stat_tau}")

        # Add fallback case using global parameters (no weightclass)
        fallback_tau = self._get_pseudo_minutes(col, weightclass=None)
        tau_cases.append(f"ELSE {fallback_tau}")

        # Generate dynamic tau expression
        tau_expr = f"CASE {' '.join(tau_cases)} END"
        
        return f"""
-- Per-weight class smoothed {col} (dynamic tau)
CASE 
    WHEN wp.{col}_wc_rate IS NOT NULL AND wp.{col}_wc_rate > 0 THEN
        -- Use weight class prior with weight class specific tau
        -- λ_post = (μ_w * τ_wc + X) / (τ_wc + t)
        -- X_smooth = t * λ_post
        ({exposure_expr}) * (
            (wp.{col}_wc_rate * ({tau_expr}) + fd.{col}::float) / 
            (({tau_expr}) + {exposure_expr})
        )
    WHEN gp.{col}_global_rate IS NOT NULL AND gp.{col}_global_rate > 0 THEN
        -- Fallback to global prior with weight class specific tau
        ({exposure_expr}) * (
            (gp.{col}_global_rate * ({tau_expr}) + fd.{col}::float) / 
            (({tau_expr}) + {exposure_expr})
        )
    ELSE
        -- Final fallback: return observed count
        fd.{col}::float
END AS {col}{self.layer_suffix} """

    def run(self) -> Dict[str, Any]:
        """
        Run the Poisson-Gamma calculator, generating SQL and executing it.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Generate SQL
            sql = self.calculate_for_table(self.table_name)
            
            if not sql:
                return {"status": "skipped", "message": "No stats to process"}
                
            # Execute the SQL query
            result_df = self.execute_raw_sql(sql, return_results=True)
            
            if result_df.empty:
                return {"status": "warning", "message": "No results from Poisson-Gamma smoothing"}
                
            # Update the database with smoothed values
            self.logger.info(f"Updating {len(result_df)} rows with Poisson-Gamma smoothed values for {len(self.columns_to_smooth)} columns")
            
            # Use batch update for performance
            self.bulk_update_dataframe(result_df, self.table_name)
            
            return {
                "status": "success",
                "processed_rows": len(result_df),
                "processed_columns": len(self.columns_to_smooth)
            }
            
        except Exception as e:
            self.logger.error(f"Error in PoissonGammaCalculator: {str(e)}")
            raise
            
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
            
            # If we still don't have SQL, construct it directly as fallback
            if not sql:
                # Simple SQL template construction
                sql = f"""
                WITH {params['prior_cte']}
                
                SELECT
                    fd.fight_id,
                    fd.fighter_id,
                    {', '.join(params['smoothing_expressions'])}
                FROM {params['schema']}.{params['table_name']} fd
                JOIN {params['schema']}.fight_mapping fm ON fd.fight_id = fm.fight_id
                LEFT JOIN weightclass_priors wp
                  ON LOWER(fm.weightclass) = LOWER(wp.weightclass)
                CROSS JOIN global_priors gp
                ORDER BY fd.fight_id, fd.fighter_id
                """
            
            return sql
            
        except Exception as e:
            self.logger.warning(f"Error executing SQL template: {str(e)}")
            return ""
