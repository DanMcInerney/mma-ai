from libs.feature_store.base_calculator import BaseCalculator
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
from sqlalchemy import text
import numpy as np
from libs.feature_store.calculator_context import CalculatorContext
from libs.parameter_optimization.loaders.parameter_loader import BASELINE_BETA_BINOMIAL

class BetaBinomialCalculator(BaseCalculator):
    """
    Applies statistically optimized per-weight class Beta-Binomial smoothing to binary outcome stats.
    
    This calculator implements weight class aware Bayesian smoothing based on comprehensive
    likelihood optimization with proper cross-validation:
    1. Models each binary stat as a success probability (Beta distribution)
    2. Uses weight class success rates as priors with optimized pseudo-counts
    3. Applies per-weight class tau only where statistically validated (≥0.5% improvement)
    4. Uses global parameters for consistency where per-class benefits are minimal
    5. Stores smoothed probabilities with '_smooth' suffix
    
    Mathematical Model:
    - Prior: p ~ Beta(α, β) where p = success probability
    - Likelihood: X ~ Binomial(n, p) where n = attempts, X = successes
    - Posterior: p | X ~ Beta(α + X, β + n - X)
    - Weight class specific tau: τ_wc only where statistically validated
    - Output: E[p | X] = smoothed success probability
    
    Statistically Validated Per-Weight Class Parameters:
    - sub_land: τ=3.0 for Featherweight (vs τ=9.0 global)
    - ctrl: τ=1.5 for Light Heavyweight and Heavyweight (vs τ=2.0 global)
    - All other stats use consistent global parameters across weight classes
    
    Attempts Definition:
    - ko/win/decision: fights (each fight is an opportunity)
    - sub_land: sub_att (submission attempts → successes)
    - ctrl: time_sec (control time modeled as per-second Bernoulli process)
    - ctrl_rd1: min(time_sec_rd1, 300) (capped round 1 time)
    
    Global Parameters (based on likelihood optimization):
    - ko: τ=23.0, win: τ=25.0, decision: τ=20.0, sub_land: τ=9.0, ctrl: τ=2.0
    - Round 1: ko_rd1: τ=17.0, win_rd1: τ=15.0, 
               sub_land_rd1: τ=7.0, ctrl_rd1: τ=1.0
    - Note: decision_rd1 removed - decisions cannot happen in round 1
    
    Excluded stats:
    - Count data (handled by PoissonGammaCalculator)
    - Static stats, duration stats
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

        # Binary outcome stats to smooth
        self.binary_stats = {
            'ko', 'ko_rd1',
            'sub_land', 'sub_land_rd1',
            'win', 'win_rd1',
            'decision',
            'ctrl', 'ctrl_rd1'  # NEW: Control time as time share
        }

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
        self.pseudo_counts = dict(BASELINE_BETA_BINOMIAL)
        self.per_weightclass_pseudo_counts = {
            "flyweight": {"ko": 22.44, "default": self.pseudo_counts["default"]},
        }

        # Will store columns to smooth
        self.columns_to_smooth = []

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

    def _get_attempts_expression(self, col: str) -> str:
        """
        Get the appropriate "attempts" expression for a binary outcome stat.
        
        Args:
            col: Column name to get attempts for
            
        Returns:
            SQL expression for number of attempts/opportunities
        """
        if col.startswith('ctrl'):
            # Control time attempts = fight duration (capped at 300 seconds for round 1)
            return "LEAST(COALESCE(fd.time_sec_rd1, fd.time_sec), 300)" if col.endswith('_rd1') else "fd.time_sec"
        elif col.startswith('ko'):
            # KO attempts = fights (each fight is a KO opportunity)
            return "1"
        elif col.startswith('sub_land'):
            # Submission success attempts = submission attempts (can be 0)
            base_col = col.replace('sub_land', 'sub_att')
            return f"fd.{base_col}"  # Allow 0 attempts - will fall back to prior
        elif col.startswith('win'):
            # Win attempts = fights (each fight is a win opportunity)
            return "1"
        elif col.startswith('decision'):
            # Decision attempts = fights (each fight could potentially go to decision)
            return "1"
        else:
            return "1"  # Default: each fight is one attempt
            
    def _resolve_stat_key(self, col: str, key_dict: dict) -> str:
        """
        Resolve the most specific key for a column from a parameter dictionary.
        
        Prefers exact matches, then _rd1 keys, then base keys, falling back to 'default'.
        
        Args:
            col: Column name to resolve
            key_dict: Dictionary of parameters to search
            
        Returns:
            The most specific key found in the dictionary
        """
        # Prefer exact key match
        if col in key_dict: 
            return col
            
        if col.endswith('_rd1'):
            # Try '<family>_rd1' first for round 1 stats
            fam = col[:-4]
            if f"{fam}_rd1" in key_dict:
                return f"{fam}_rd1"
                
        # Map *_land / *_att to family (e.g., 'head_land' -> 'head')
        base = (col[:-5] if col.endswith('_land') else
                col[:-4] if col.endswith('_att') else col)
        return base if base in key_dict else 'default'
    
    def _get_pseudo_count(self, col: str, weightclass: Optional[str] = None) -> float:
        """
        Get the pseudo-count confidence for a given binary stat.

        Args:
            col: Column name to get pseudo-count for
            weightclass: Optional weightclass for per-class parameters

        Returns:
            Pseudo-count value (τ) representing prior confidence
        """
        # Use parameter loader to get tau value
        return self.param_loader.get_beta_binomial_params(col, weightclass)
    
    def get_binary_columns(self) -> List[str]:
        """
        Identify all binary outcome columns that should be smoothed.
        
        Returns:
            List of column names to apply Beta-Binomial smoothing to
        """
        # Get all columns from fight_stats_derived
        columns = self.feature_utils.get_columns_from_table(
            self.schema, 
            self.table_name
        )
        
        # Filter to only binary stats that aren't already smoothed
        columns_to_smooth = []
        for col in columns:
            # Skip columns that end with our suffix (already smoothed)
            if col.endswith(self.layer_suffix):
                continue
                
            # Only include binary outcome stats
            if col in self.binary_stats:
                columns_to_smooth.append(col)
        
        return columns_to_smooth

    def _generate_prior_cte(self, columns: List[str]) -> str:
        """
        Generate a CTE that calculates Beta priors for binary outcome stats.
        
        This calculates:
        1. Weight class success rates for each binary stat
        2. Global success rates as fallback
        3. Converts to Beta parameters using pseudo-counts
        
        Args:
            columns: List of column names to calculate priors for
            
        Returns:
            SQL string for the CTE
        """
        # Generate column expressions for success rate calculations
        weightclass_exprs = []
        global_exprs = []
        
        for col in columns:
            attempts_expr = self._get_attempts_expression(col)
            
            # Weight class success rates
            weightclass_exprs.extend([
                f"SUM(fd.{col}::float) AS {col}_successes",
                f"SUM({attempts_expr}) AS {col}_attempts",
                f"SUM(fd.{col}::float) / NULLIF(SUM({attempts_expr}), 0) AS {col}_wc_rate"
            ])
            
            # Global success rates (fallback)
            global_exprs.extend([
                f"SUM(fd.{col}::float) AS {col}_global_successes", 
                f"SUM({attempts_expr}) AS {col}_global_attempts",
                f"SUM(fd.{col}::float) / NULLIF(SUM({attempts_expr}), 0) AS {col}_global_rate"
            ])
            
        return f"""
-- Global priors (fallback for sparse weight classes)
global_priors AS (
    SELECT
        {', '.join(global_exprs)}
    FROM {self.schema}.{self.table_name} fd
    JOIN {self.schema}.event_mapping em ON fd.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-01-01'
),

-- Weight class specific priors
weightclass_priors AS (
    SELECT
        fm.weightclass,
        COUNT(*) AS sample_size,
        {', '.join(weightclass_exprs)}
    FROM {self.schema}.{self.table_name} fd
    JOIN {self.schema}.fight_mapping fm ON fd.fight_id = fm.fight_id
    JOIN {self.schema}.event_mapping em ON fd.event_id = em.event_id
    WHERE em.event_date BETWEEN '2014-01-01' AND '2023-01-01'
    GROUP BY fm.weightclass
    HAVING COUNT(*) >= 10
)"""

    def _generate_smoothing_expr(self, col: str) -> str:
        """
        Generate SQL expression for per-weight class Beta-Binomial smoothing.

        Uses different tau values for each weight class based on statistical analysis.
        Implements Beta-Binomial conjugate updating with weight class specific parameters.

        Args:
            col: Column name to smooth

        Returns:
            SQL expression for weight class aware Beta-Binomial smoothed probability
        """
        attempts_expr = self._get_attempts_expression(col)

        # Generate weight class specific tau cases using param_loader
        tau_cases = []
        for weightclass in self.weight_classes:
            # Get tau for this stat in this weight class from param_loader
            stat_tau = self._get_pseudo_count(col, weightclass)

            # Use case-insensitive comparison for weight class names
            tau_cases.append(f"WHEN LOWER(fm.weightclass) = '{weightclass.lower()}' THEN {stat_tau}")

        # Add fallback case using global parameters (no weightclass)
        fallback_tau = self._get_pseudo_count(col, weightclass=None)
        tau_cases.append(f"ELSE {fallback_tau}")

        # Generate dynamic tau expression
        tau_expr = f"CASE {' '.join(tau_cases)} END"
        
        # Generate the posterior probability expression with dynamic tau
        p_post = f"""CASE 
    WHEN {attempts_expr} = 0 THEN
        COALESCE(wp.{col}_wc_rate, gp.{col}_global_rate, 0.0)
    WHEN wp.{col}_wc_rate IS NOT NULL AND wp.{col}_attempts > 0 THEN
        (wp.{col}_wc_rate * ({tau_expr}) + fd.{col}::float) / (({tau_expr}) + {attempts_expr})
    WHEN gp.{col}_global_rate IS NOT NULL AND gp.{col}_global_attempts > 0 THEN
        (gp.{col}_global_rate * ({tau_expr}) + fd.{col}::float) / (({tau_expr}) + {attempts_expr})
    ELSE
        COALESCE(fd.{col}::float / NULLIF({attempts_expr}, 0), 0.0)
END"""
        
        if col.startswith('ctrl'):
            # For control time: emit smoothed seconds only (rate handled by PerMinCalculator)
            return f"""{p_post} * {attempts_expr} AS {col}{self.layer_suffix} """
        else:
            # For other binary stats: emit smoothed probability only
            return f"""{p_post} AS {col}{self.layer_suffix} """

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for calculating Beta-Binomial smoothed values for binary outcome stats.
        
        Args:
            table_name: Name of the feature table
            columns: Optional list of columns (if None, will fetch all binary columns)
            
        Returns:
            SQL query string for the calculation
        """
        if not columns:
            columns = self.get_binary_columns()
            self.columns_to_smooth = columns
            
        if not columns:
            self.logger.warning("No binary outcome columns found for Beta-Binomial smoothing")
            return ""
            
        # Step 1: Generate CTE to calculate priors
        prior_cte = self._generate_prior_cte(columns)
            
        # Step 2: Generate smoothing expressions for each column
        smoothing_expressions = [
            self._generate_smoothing_expr(col) for col in columns
        ]
        
        # Step 3: Render the template
        sql = self.execute_sql_template(
            template_name='beta_binomial',
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

    def run(self) -> Dict[str, Any]:
        """
        Run the Beta-Binomial calculator, generating SQL and executing it.
        
        Returns:
            Dictionary with status information
        """
        try:
            # Generate SQL
            sql = self.calculate_for_table(self.table_name)
            
            if not sql:
                return {"status": "skipped", "message": "No binary outcome stats to process"}
                
            # Execute the SQL query
            result_df = self.execute_raw_sql(sql, return_results=True)
            
            if result_df.empty:
                return {"status": "warning", "message": "No results from Beta-Binomial smoothing"}
                
            # Update the database with smoothed values
            self.logger.info(f"Updating {len(result_df)} rows with Beta-Binomial smoothed values for {len(self.columns_to_smooth)} columns")
            
            # Use batch update for performance
            self.bulk_update_dataframe(result_df, self.table_name)
            
            return {
                "status": "success",
                "processed_rows": len(result_df),
                "processed_columns": len(self.columns_to_smooth)
            }
            
        except Exception as e:
            self.logger.error(f"Error in BetaBinomialCalculator: {str(e)}")
            raise
            
    def execute_sql_template(self, template_name: str, operation: str, params: Dict) -> str:
        """
        Execute a SQL template using the context's SQL manager if available.
        
        Args:
            template_name: Name of the template category
            operation: Operation name ('calculate', etc.)
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
