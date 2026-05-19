from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from config.decay import get_decay_rate_sql_constant
from typing import Set, List, Dict, Optional, Any
import pandas as pd
import logging
from sqlalchemy import text
import numpy as np

class AdjustedPerformanceCalculator(BaseCalculator):
    """
    Calculate reliability-weighted adjusted performance features using the simplified formula:

         adjperf = clip( (s - mu_shrunk) / mad_shrunk , -7, +7 )

    Where:
    - w_mean = n / (n + K_mean)
    - w_mad  = n / (n + K_mad)
    - mu_shrunk  = w_mean * opp_mean   + (1 - w_mean) * wc_mean
    - mad_shrunk = max( w_mad * opp_mad + (1 - w_mad) * wc_mad , mad_floor )

    Key features:
    1. Uses reliability-weighted (shrunk) opponent means and MADs
    2. Operates on feature-specific tables (no fight_stats_derived dependency)
    3. Uses opponent "allowed" values (what opponents achieved against F2)
    4. Implements proper Bayesian shrinkage with configurable parameters
    5. Simple, robust denominator (no exposure noise complexity)

    Mathematical Process:
    
    Step 1: Compute opponent's personal history
    - opp_mean_pers: average of what F2's opponents achieved against F2
    - opp_mad_pers: MAD of what F2's opponents achieved against F2
    - n: number of opponent history fights (Kish effective sample size if decay=True)
    
    Step 2: Get weightclass priors
    - wc_mean: weightclass average from features.<table>_wc_mean
    - wc_mad: weightclass MAD from features.<table>_wc_mad
    - mad_floor: minimum MAD from features.<table>_minimum_mad
    
    Step 3: Reliability-weighted shrinkage
    - w_mean = n / (n + K_mean), w_mad = n / (n + K_mad)
    - mu_shrunk = w_mean * opp_mean + (1 - w_mean) * wc_mean
    - mad_shrunk = max(w_mad * opp_mad + (1 - w_mad) * wc_mad, mad_floor)
    
    Step 4: Final calculation
    - adjperf = clip((observed - mu_shrunk) / mad_shrunk, -7, +7)
    
    Usage Examples:
    
    # Basic usage
    calc = AdjustedPerformanceCalculator(context)
    results = calc.run()
    
    # With time-decay
    calc = AdjustedPerformanceCalculator(context, decay=True)
    results = calc.run()
    
    # Custom shrinkage parameters
    calc = AdjustedPerformanceCalculator(context, K_mean=6.0, K_mad=4.0)
    results = calc.run()
    """

    def __init__(
        self,
        context_or_conn,
        decay: bool = False,
        K_mean: float = 4.0,
        K_mad: float = 4.0,
        include_patterns: Set[str] = set(),
        exclude_patterns: Set[str] = set()
    ):
        """
        Initialize the new adjusted performance calculator.

        Args:
            context_or_conn: CalculatorContext or database connection
            decay: Whether to use time-decay adjusted metrics
            K_mean: Shrinkage parameter for opponent means (higher = more shrinkage)
            K_mad: Shrinkage parameter for opponent MADs (higher = more shrinkage)
            include_patterns: Set of patterns to include in calculation
            exclude_patterns: Set of patterns to exclude from calculation
        """
        # Handle context initialization
        try:
            if hasattr(context_or_conn, 'connection') and hasattr(context_or_conn, 'feature_utils'):
                self.context = context_or_conn
                conn = self.context.connection
            else:
                conn = context_or_conn
                self.context = CalculatorContext(conn)
        except (TypeError, AttributeError):
            conn = context_or_conn
            self.context = CalculatorContext(conn)
            
        # Initialize BaseCalculator with multi_table calculator type
        super().__init__(conn, calculator_type='multi_table')
        
        # Set up calculator-specific attributes
        self.logger = logging.getLogger(__name__)
        self.decay = decay
        self.schema = 'features'
        self.K_mean = K_mean
        self.K_mad = K_mad

        # Set up stat tables
        self.stat_tables = self.context.feature_utils.get_stat_tables()

        # Add include/exclude patterns
        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)

        # Decide suffixes based on decay
        if self.decay:
            self.layer_suffix = '_dec_adjperf'
        else:
            self.layer_suffix = '_adjperf'

        self.logger.info(f"Initialized AdjustedPerformanceCalculator with decay={decay}, K_mean={K_mean}, K_mad={K_mad}")

    def _is_adjperf_target(self, col: str) -> bool:
        """
        Determine if a column should get adjusted performance treatment.
        
        Args:
            col: Column name to check
            
        Returns:
            True if column should get adjperf calculation
        """
        if col.endswith('_total'):
            return False
        return (
            col.endswith('_per_min') or col.endswith('_acc') or col.endswith('_def') or col.endswith('_ratio') or col.endswith('_pressure')
            or col in {
                'sub_att_per_ctrl', 'ground_land_per_ctrl', 'rev_per_ctrlopp', 'sub_per_all_ctrl',
                'ko_per_sig_str_land', 'sig_str_per_str_att', 'distance_per_sig_str_land',
                'clinch_per_sig_str_land', 'ground_per_sig_str_land', 'head_per_sig_str_land',
                'body_leg_per_sig_str_land', 'td_per_sig_str_att', 'ground_land_per_td_land',
                'td_land_per_ctrl',  # Updated from ctrl_per_td_land
                'ko_sub_per_win', 'ko_sub_rd1_per_win',  # Finishing rate features
                'win', 'decision',  # Binary outcome features that need adjperf calculation
                'time_sec'  # Time duration feature that needs adjperf calculation
            }
        )

    def _column_exists_in_table(self, table_name: str, column_name: str) -> bool:
        """
        Check if a column exists in the given table.
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to check
            
        Returns:
            True if column exists, False otherwise
        """
        try:
            table_columns = self.stat_tables.get(table_name, [])
            return column_name in table_columns
        except:
            return False





    def _compute_observed_canonical_value(self, col: str, table_name: str, alias: str = 't') -> str:
        """
        Generate SQL expression for observed value on canonical scale.
        Always use the stored feature column value (already smoothed upstream).
        
        Args:
            col: Column name
            table_name: Feature table name
            alias: Table alias to use (default 't')
            
        Returns:
            SQL expression for observed canonical value
        """
        return f"COALESCE({alias}.{col}, 0)"



    def _generate_opponent_history_cte(self, table_name: str, columns: List[str]) -> str:
        """
        Generate CTE to compute opponent's personal history using two-step MAD calculation
        and Kish effective sample size under decay.
        
        Args:
            table_name: Feature table name
            columns: List of columns to calculate for
            
        Returns:
            SQL CTE string for opponent history with proper MAD calculation
        """
        # Filter to only adjperf target columns
        target_columns = [col for col in columns if self._is_adjperf_target(col)]
        
        if not target_columns:
            return "opponent_history AS (SELECT NULL as fight_id, NULL as fighter_id, NULL as n_fights)"
        
        # Build CTEs for each column using two-step MAD calculation
        cte_blocks = []
        
        # Use first target column for n_fights calculation
        first_col = target_columns[0]
        # Use day difference as double precision to avoid INTEGER/INTERVAL operations across DATE/TIMESTAMP
        decay_rate_sql = get_decay_rate_sql_constant()
        weight_expr = (
            f"EXP(-{decay_rate_sql} * ((em_current.event_date - em_hist.event_date)::double precision) / 365.25)"
            if self.decay else "1.0"
        )
        
        # Step 1: Create rows CTE for the first column (used for n_fights)
        cte_blocks.append(f"""
        rows_0 AS (
            SELECT
                current_fight.fight_id,
                current_fight.fighter_id,
                hist_opp.{first_col} AS val,
                {weight_expr} AS w
            FROM features.{table_name} current_fight
            JOIN features.fight_mapping fm_current ON current_fight.fight_id = fm_current.fight_id
            JOIN features.event_mapping em_current ON fm_current.event_id = em_current.event_id
            CROSS JOIN LATERAL (
                SELECT CASE
                    WHEN current_fight.fighter_id = fm_current.fighter1_id THEN fm_current.fighter2_id
                    ELSE fm_current.fighter1_id
                END AS opp_id
            ) opp_info
            LEFT JOIN features.{table_name} hist_fight ON hist_fight.fighter_id = opp_info.opp_id
            LEFT JOIN features.fight_mapping fm_hist ON hist_fight.fight_id = fm_hist.fight_id
            LEFT JOIN features.event_mapping em_hist ON fm_hist.event_id = em_hist.event_id
            CROSS JOIN LATERAL (
                SELECT CASE
                    WHEN hist_fight.fighter_id = fm_hist.fighter1_id THEN fm_hist.fighter2_id
                    ELSE fm_hist.fighter1_id
                END AS hist_opp_id
            ) hist_opp_info
            LEFT JOIN features.{table_name} hist_opp
                ON hist_opp.fight_id = hist_fight.fight_id
                AND hist_opp.fighter_id = hist_opp_info.hist_opp_id
            WHERE (
                em_hist.event_date < em_current.event_date
                OR (em_hist.event_date = em_current.event_date AND fm_hist.event_id < fm_current.event_id)
                OR (em_hist.event_date = em_current.event_date AND fm_hist.event_id = fm_current.event_id AND hist_fight.fight_id < current_fight.fight_id)
            ) AND hist_opp.{first_col} IS NOT NULL
        )""")
        
        # Step 2: Compute n_fights with Kish effective sample size
        if self.decay:
            n_fights_expr = "POWER(SUM(w), 2) / NULLIF(SUM(POWER(w, 2)), 0)"
        else:
            n_fights_expr = "COUNT(*)"
            
        cte_blocks.append(f"""
        n_hist AS (
            SELECT
                fight_id, fighter_id,
                {n_fights_expr} AS n_fights
            FROM rows_0
            GROUP BY fight_id, fighter_id
        )""")
        
        # Step 3: For each target column, create stats and MAD CTEs
        stats_joins = []
        for i, col in enumerate(target_columns):
            if i > 0:  # Skip first column, already have rows_0
                cte_blocks.append(f"""
                rows_{i} AS (
                    SELECT
                        current_fight.fight_id,
                        current_fight.fighter_id,
                        hist_opp.{col} AS val,
                        {weight_expr} AS w
                    FROM features.{table_name} current_fight
                    JOIN features.fight_mapping fm_current ON current_fight.fight_id = fm_current.fight_id
                    JOIN features.event_mapping em_current ON fm_current.event_id = em_current.event_id
                    CROSS JOIN LATERAL (
                        SELECT CASE
                            WHEN current_fight.fighter_id = fm_current.fighter1_id THEN fm_current.fighter2_id
                            ELSE fm_current.fighter1_id
                        END AS opp_id
                    ) opp_info
                    LEFT JOIN features.{table_name} hist_fight ON hist_fight.fighter_id = opp_info.opp_id
                    LEFT JOIN features.fight_mapping fm_hist ON hist_fight.fight_id = fm_hist.fight_id
                    LEFT JOIN features.event_mapping em_hist ON fm_hist.event_id = em_hist.event_id
                    CROSS JOIN LATERAL (
                        SELECT CASE
                            WHEN hist_fight.fighter_id = fm_hist.fighter1_id THEN fm_hist.fighter2_id
                            ELSE fm_hist.fighter1_id
                        END AS hist_opp_id
                    ) hist_opp_info
                    LEFT JOIN features.{table_name} hist_opp
                        ON hist_opp.fight_id = hist_fight.fight_id
                        AND hist_opp.fighter_id = hist_opp_info.hist_opp_id
                    WHERE (
                        em_hist.event_date < em_current.event_date
                        OR (em_hist.event_date = em_current.event_date AND fm_hist.event_id < fm_current.event_id)
                        OR (em_hist.event_date = em_current.event_date AND fm_hist.event_id = fm_current.event_id AND hist_fight.fight_id < current_fight.fight_id)
                    ) AND hist_opp.{col} IS NOT NULL
                )""")
            
            # Median calculation (only on non-NULL rows)
            # Grouped median per (fight_id, fighter_id)
            cte_blocks.append(f"""
            med_{i} AS (
                SELECT
                    fight_id,
                    fighter_id,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY val) AS med
                FROM rows_{i}
                WHERE val IS NOT NULL
                GROUP BY fight_id, fighter_id
            )""")
            
            # Attach median to each row so we can compute |val - med|
            cte_blocks.append(f"""
            rows_with_med_{i} AS (
                SELECT
                    r.fight_id,
                    r.fighter_id,
                    r.val,
                    m.med
                FROM rows_{i} r
                JOIN med_{i} m USING (fight_id, fighter_id)
            )""")
            
            # Stats calculation (NULL-safe weighting under decay)
            if self.decay:
                mean_expr = "SUM(w * val) / NULLIF(SUM(CASE WHEN val IS NOT NULL THEN w END), 0)"
            else:
                mean_expr = "AVG(val)"  # Postgres excludes NULLs automatically
                
            cte_blocks.append(f"""
            stats_{i} AS (
                SELECT
                    fight_id, fighter_id,
                    {mean_expr} AS {col}_opp_mean_pers
                FROM rows_{i}
                GROUP BY fight_id, fighter_id
            )""")
            
            # MAD = median of |val - med| per group
            cte_blocks.append(f"""
            mad_{i} AS (
                SELECT
                    fight_id,
                    fighter_id,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(val - med)) AS {col}_opp_mad_pers
                FROM rows_with_med_{i}
                GROUP BY fight_id, fighter_id
            )""")
            
            # Add to joins for final CTE
            stats_joins.extend([
                f"LEFT JOIN stats_{i} USING (fight_id, fighter_id)",
                f"LEFT JOIN mad_{i} USING (fight_id, fighter_id)"
            ])
        
        stats_joins_str = '\n        '.join(stats_joins)
        
        # Final opponent_history CTE
        cte_blocks.append(f"""
        opponent_history AS (
            SELECT
                n_hist.fight_id,
                n_hist.fighter_id,
                n_hist.n_fights
                {', ' + ', '.join([f'stats_{i}.{col}_opp_mean_pers, mad_{i}.{col}_opp_mad_pers' for i, col in enumerate(target_columns)]) if target_columns else ''}
            FROM n_hist
            {stats_joins_str}
        )""")
        
        return ',\n        '.join(cte_blocks)

    def _generate_weightclass_priors_cte(self, table_name: str, columns: List[str]) -> str:
        """
        Generate CTE for weightclass priors using new wc_mean and wc_mad tables.
        
        Args:
            table_name: Feature table name
            columns: List of columns to calculate for
            
        Returns:
            SQL CTE string for weightclass priors
        """
        # Filter to only adjperf target columns
        target_columns = [col for col in columns if self._is_adjperf_target(col)]
        
        if not target_columns:
            return "weightclass_priors AS (SELECT NULL as weightclass)"
        
        # Use new _wc_mean and _wc_mad tables, with existing minimum_mad as floor
        mean_table = f"{table_name}_wc_mean"
        mad_table = f"{table_name}_wc_mad"
        min_mad_table = f"{table_name}_minimum_mad"
        
        prior_sql = f"""
        weightclass_priors AS (
            SELECT 
                COALESCE(mean_stats.weightclass, mad_stats.weightclass, min_mad_stats.weightclass) as weightclass,
                -- Get precomputed weightclass means and MADs for each target column
                {', '.join([f"""
                COALESCE(mean_stats.{col}_wc_mean, 0) AS {col}_wc_mean,
                COALESCE(mad_stats.{col}_wc_mad, 0.001) AS {col}_wc_mad,
                COALESCE(min_mad_stats.{col}_min_mad, 0.001) AS {col}_mad_floor""" 
                for col in target_columns])}
            FROM features.{mean_table} mean_stats
            FULL OUTER JOIN features.{mad_table} mad_stats 
                ON mean_stats.weightclass = mad_stats.weightclass
            FULL OUTER JOIN features.{min_mad_table} min_mad_stats
                ON COALESCE(mean_stats.weightclass, mad_stats.weightclass) = min_mad_stats.weightclass
        )"""
        
        return prior_sql





    def _generate_adjperf_expression(self, col: str, table_name: str) -> str:
        """
        Generate final adjusted performance SQL expression using simplified formula.
        
        Formula:
        w_mean = n / (n + K_mean)
        w_mad  = n / (n + K_mad)
        mu_shrunk  = w_mean * opp_mean   + (1 - w_mean) * wc_mean
        mad_shrunk = max( w_mad * opp_mad + (1 - w_mad) * wc_mad , mad_floor )
        adjperf = clip( (s - mu_shrunk) / mad_shrunk , -7, +7 )
        
        Args:
            col: Column name
            table_name: Feature table name
            
        Returns:
            SQL expression for final adjusted performance score
        """
        # Only process adjperf target columns
        if not self._is_adjperf_target(col):
            return "0"
        
        obs_expr = self._compute_observed_canonical_value(col, table_name, alias='t')
        
        # Shrinkage weights
        w_mean_expr = f"COALESCE(oh.n_fights, 0) / (COALESCE(oh.n_fights, 0) + {self.K_mean:.1f})"
        w_mad_expr = f"COALESCE(oh.n_fights, 0) / (COALESCE(oh.n_fights, 0) + {self.K_mad:.1f})"
        
        # Shrunk mean: w_mean * opp_mean + (1 - w_mean) * wc_mean
        mu_shrunk_expr = f"""
        ({w_mean_expr}) * COALESCE(oh.{col}_opp_mean_pers, 0) + 
        (1 - ({w_mean_expr})) * COALESCE(wp.{col}_wc_mean, 0)
        """
        
        # Shrunk MAD: max( w_mad * opp_mad + (1 - w_mad) * wc_mad , mad_floor )
        mad_shrunk_expr = f"""
        GREATEST(
            ({w_mad_expr}) * COALESCE(oh.{col}_opp_mad_pers, 0) + 
            (1 - ({w_mad_expr})) * COALESCE(wp.{col}_wc_mad, 0),
            COALESCE(wp.{col}_mad_floor, 0.001)
        )
        """

        # Use baseline winsorization limit of ±7.0 (for healthy distributions)
        # This clips extreme z-scores to prevent outlier influence
        winsor_limit = 7.0

        # Final adjusted performance: clip( (s - mu_shrunk) / mad_shrunk , -limit, +limit )
        adjperf_expr = f"""
        CASE
            WHEN ({mad_shrunk_expr}) = 0 THEN 0
            ELSE GREATEST(LEAST(
                (({obs_expr}) - ({mu_shrunk_expr})) / ({mad_shrunk_expr}),
                {winsor_limit}  -- Upper winsorization limit (baseline)
            ), -{winsor_limit})  -- Lower winsorization limit (baseline)
        END
        """
        
        return adjperf_expr

    def calculate_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> str:
        """
        Generate SQL for reliability-weighted adjusted performance calculation.
        
        Args:
            table_name: The source table name
            columns: Optional list of columns to calculate for
            
        Returns:
            SQL query string for the calculations
        """
        # Skip odds table - it should not have adjusted performance calculated
        if table_name == 'odds':
            self.logger.info(f"Skipping odds table - no adjusted performance needed")
            return ""
            
        self.logger.info(f"Building SELECT for table={table_name}, columns={columns}")
        
        # Get columns if not provided
        if columns is None:
            columns = self.get_features(table_name)
            
        if not columns:
            return ""

        # Filter to adjperf target columns and avoid re-processing
        calc_columns = [col for col in columns if not col.endswith(self.layer_suffix) and self._is_adjperf_target(col)]
        if not calc_columns:
            self.logger.info(f"No adjperf target columns to process for {table_name}")
            return ""

        # Generate CTEs
        opponent_history_cte = self._generate_opponent_history_cte(table_name, calc_columns)
        weightclass_priors_cte = self._generate_weightclass_priors_cte(table_name, calc_columns)
        
        # Generate final adjusted performance expressions
        adjperf_exprs = []
        for col in calc_columns:
            new_col = f"{col}{self.layer_suffix}"
            adjperf_expr = self._generate_adjperf_expression(col, table_name)
            adjperf_exprs.append(f"({adjperf_expr}) AS {new_col}")



        # Build the complete SQL query
        final_sql = f"""
        WITH 
        {opponent_history_cte},
        
        {weightclass_priors_cte}
        
        -- Final calculation with shrinkage and adjusted performance
        SELECT
            t.fight_id,
            t.fighter_id,
            {','.join(adjperf_exprs)}
        FROM features.{table_name} t
        JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        LEFT JOIN opponent_history oh ON t.fight_id = oh.fight_id AND t.fighter_id = oh.fighter_id
        LEFT JOIN weightclass_priors wp ON fm.weightclass = wp.weightclass
        ORDER BY t.fight_id, t.fighter_id
        """

        return final_sql

    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute adjusted performance calculation for the given table.
        
        Args:
            table_name: The name of the table to calculate for
            columns: The list of columns to calculate for
            
        Returns:
            DataFrame with calculation results
        """
        self.logger.info(f"Executing calculation for {table_name}")
        
        try:
            sql = self.calculate_for_table(table_name, columns)
            if not sql:
                self.logger.info(f"Failed to generate SQL for {table_name}")
                return pd.DataFrame()
                
            # Execute SQL to calculate values
            df = self.execute_raw_sql(sql, return_results=True)
            self.logger.info(f"Completed calculation for {table_name} with {len(df)} rows")
            return df
            
        except Exception as e:
            self.logger.error(f"Error executing calculation for {table_name}: {str(e)}")
            raise
            
    def run(self, parallel: bool = False, max_workers: int = 4, table_pattern: str = "") -> Dict[str, pd.DataFrame]:
        """
        Run the calculator for all tables.
        
        Args:
            parallel: Whether to run in parallel (not used, kept for API compatibility)
            max_workers: Number of worker threads (not used, kept for API compatibility)
            table_pattern: Optional pattern to filter tables
            
        Returns:
            Dictionary of results by table
        """
        self.logger.info(f"Starting new adjusted performance calculation for all tables")
        
        # Get tables to process
        tables = list(self.stat_tables.keys())
        
        # Apply table filtering if a pattern is provided
        if table_pattern:
            filtered_tables = [table for table in tables if table_pattern in table]
            self.logger.info(f"Filtered tables from {len(tables)} to {len(filtered_tables)} based on pattern '{table_pattern}'")
            tables = filtered_tables
        
        total_tables = len(tables)
        self.logger.info(f"Found {total_tables} tables to process")
        
        # Process tables sequentially
        results = {}
        
        for i, table_name in enumerate(tables, 1):
            try:
                # Get all columns for the table - we'll do our own filtering
                columns = self.stat_tables.get(table_name, [])
                
                if not columns:
                    self.logger.info(f"No columns to process for {table_name}")
                    results[table_name] = pd.DataFrame()
                    continue
                
                # Process the table
                self.logger.info(f"Processing table {table_name} with {len(columns)} columns [{i}/{total_tables}]")
                
                # Generate SQL for the table
                sql = self.calculate_for_table(table_name, columns)
                if not sql:
                    self.logger.info(f"Failed to generate SQL for {table_name}")
                    results[table_name] = pd.DataFrame()
                    continue
                    
                # Use execute_layer_update for high-performance updates
                self.execute_layer_update(
                    calculation_sql=sql,
                    table_name=table_name,
                    schema=self.schema,
                    batch_size=100000
                )
                
                self.logger.info(f"Completed new adjusted performance calculation for {table_name}")
                
                # Return empty dataframe since we don't need it anymore
                # (execute_layer_update handles everything)
                results[table_name] = pd.DataFrame({"success": [True]})
                
            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                results[table_name] = pd.DataFrame()
        
        return results
