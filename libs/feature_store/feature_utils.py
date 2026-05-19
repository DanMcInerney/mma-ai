"""
Utility functions for feature engineering and manipulation.
"""

from typing import List, Set, Dict, Tuple, Any
from sqlalchemy import text
import numpy as np
from datetime import datetime
from math import floor
import pandas as pd


class FeatureUtils:
    """Utility functions for feature engineering."""
    
    @staticmethod
    def calculate_ratio(value1, value2):
        """Calculate a ratio between two values."""
        total = value1 + value2
        if total > 0:
            return value1 / total
        return 0.5  # Default to 0.5 if total is 0
    
    @staticmethod
    def calculate_age(birth_date, fight_date):
        """Calculate age in years."""
        if birth_date is None or fight_date is None:
            return None
        return (fight_date - birth_date).total_seconds() / (365.25 * 24 * 60 * 60)
    
    @staticmethod
    def calculate_days_since(earlier_date, later_date):
        """Calculate days between two dates."""
        if earlier_date is None or later_date is None:
            return None
        return (later_date - earlier_date).total_seconds() / (24 * 60 * 60)

    def __init__(self, conn):
        """Initialize FeatureUtils with a database engine or CalculatorContext."""
        # Handle the case where a CalculatorContext is passed instead of a connection
        from libs.feature_store.calculator_context import CalculatorContext
        if isinstance(conn, CalculatorContext):
            self.conn = conn.connection
        else:
            self.conn = conn
            
        self.static_stat_prefixes = ['age', 'days_since_last_fight', 'reach', 'ape', 'ufcage']
        self.base_stat_prefixes = [
            'strikes', 'sig_str', 'head', 'body', 'leg', 'distance',
            'clinch', 'ground', 'td', 'sub', 'ko', 'kd', 'decision',
            'ctrl', 'win', 'rev', 'time_sec'
        ]
        self.schema = 'features'
        self.window_years = 1.5 # I don't use this anywhere yet
        
        # # Determine start_date and end_date from event_mapping
        # query = text("SELECT MIN(event_date)::date::text FROM features.event_mapping")
        # self.start_date = self.conn.execute(query).scalar() or '1990-01-01'
        
        # query = text("SELECT MAX(event_date)::date::text FROM features.event_mapping")
        # self.end_date = self.conn.execute(query).scalar() or '2023-01-01'
        
    def get_columns_from_table(self, schema: str, table: str, include_strs: List[str] = [], exclude_strs: List[str] = []) -> List[str]:
        sql = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{schema}' 
            AND table_name = '{table}'
        """
        
        cols = [row[0] for row in self.conn.execute(text(sql)).fetchall()]
        
        if include_strs:
            cols = [col for col in cols if any(inc in col for inc in include_strs)]
        if exclude_strs:
            cols = [col for col in cols if not any(exc in col for exc in exclude_strs)]
        
        return cols

    def get_opp_sdev_medians_by_weightclass(self, 
                                            tables_and_columns: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Calculates the median value for every <stat>_sdev column grouped by weightclass.

        Parameters
        ----------
        tables_and_columns: Dict[str, List[str]]
            Dictionary of table names to lists of columns. We only use the columns, 
            and specifically only those that end with '_sdev'.

        Returns
        -------
        Dict[str, Dict[str, float]]
            A dictionary mapping each weightclass to another dictionary of {column: median_value}.
        """

        print("[DEBUG] get_opp_sdev_medians_by_weightclass called.")

        # Gather all columns from the dict and filter for *_sdev
        all_columns = []
        for cols in tables_and_columns.values():
            for c in cols:
                if c.endswith('_sdev'):
                    all_columns.append(c)
        all_columns = list(set(all_columns))

        if not all_columns:
            print("[DEBUG] No _sdev columns provided. Returning empty dict.")
            return {}

        # Build the PERCENTILE_CONT expressions for each column
        # percentile_cont(0.5) WITHIN GROUP (ORDER BY col) gives median
        median_expressions = [
            f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {col}) AS {col}_median"
            for col in all_columns
        ]

        query = f"""
        SELECT
            weightclass,
            {', '.join(median_expressions)}
        FROM features.fight_stats_derived
        GROUP BY weightclass
        ORDER BY weightclass;
        """

        print("[DEBUG] Running median query...")
        result = self.conn.execute(text(query)).fetchall()

        if not result:
            print("[DEBUG] No results returned from median query. Returning empty dict.")
            return {}

        # Construct a dictionary keyed by weightclass
        # Each value is another dict mapping {column: median_value}
        weightclass_map = {}
        for row in result:
            row_dict = dict(row)
            weightclass = row_dict.pop('weightclass')
            # row_dict now contains keys like {col}_median
            # We want to map them back to col names without '_median'
            col_values = {}
            for k, v in row_dict.items():
                original_col = k.replace('_median', '')
                col_values[original_col] = v
            weightclass_map[weightclass] = col_values

        print("[DEBUG] Median dictionary constructed.")
        return weightclass_map

    def get_rolling_accuracy_snapshots(self, 
                                    table: str, 
                                    land_att_pairs: List[Tuple[str, str]],
                                    window_years: int = 2,
                                    snapshot_interval_years: int = 1) -> str:
        """
        Generate a SQL snippet (no leading WITH) for rolling 2-year window accuracy stats and alpha/beta computation.
        The caller will prepend WITH and append further CTEs and a SELECT statement.
        """
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')

        snapshot_dates = []
        current = start_dt
        while current <= end_dt:
            snapshot_dates.append(current.strftime('%Y-%m-%d'))
            current = datetime(current.year + snapshot_interval_years, current.month, current.day)

        # If no snapshot dates, return empty CTEs
        if not snapshot_dates:
            return """
            snapshot_dates_cte AS (
                SELECT NULL::date AS snapshot_date WHERE false
            ),
            snapshots AS (
                SELECT NULL::text as weightclass, NULL::date as snapshot_date WHERE false
            ),
            final_snapshots AS (
                SELECT NULL::text as weightclass, NULL::date as snapshot_date, NULL::date as valid_until WHERE false
            )
            """

        snapshot_values = ",\n".join([f"('{sd}')" for sd in snapshot_dates])
        days_window = window_years * 365

        stat_aggregates = []
        stat_selects = []
        for (land_col, att_col) in land_att_pairs:
            base = land_col.replace('_land','')
            stat_aggregates.append(f"AVG(CASE WHEN {att_col}>0 THEN {land_col}::float/{att_col}::float END) AS {base}_global_acc")
            stat_aggregates.append(f"AVG(CASE WHEN {land_col}=0 THEN 1.0 ELSE 0.0 END) AS {base}_zero_land_freq")
            stat_aggregates.append(f"AVG(CASE WHEN {att_col}>0 THEN {att_col}::float END) AS {base}_avg_att")
            stat_aggregates.append(f"COUNT(*) AS {base}_total_rows")
            stat_aggregates.append(f"AVG({land_col}::float) AS {base}_avg_land")
            stat_aggregates.append(f"AVG({att_col}::float) AS {base}_avg_att_all")

            stat_selects.extend([
                f"{base}_global_acc",
                f"{base}_zero_land_freq",
                f"{base}_avg_att",
                f"{base}_total_rows",
                f"{base}_avg_land",
                f"{base}_avg_att_all"
            ])

        # Add alpha/beta columns
        alpha_beta_cols = []
        for (land_col, att_col) in land_att_pairs:
            base = land_col.replace('_land','')
            alpha_beta_cols.append(f"(({base}_global_acc*10.0)+1.0) AS {base}_alpha")
            alpha_beta_cols.append(f"((1.0 - {base}_global_acc)*10.0+1.0) AS {base}_beta")

        # Join stat aggregates and alpha/beta columns
        all_columns = stat_selects + alpha_beta_cols
        all_columns_str = ",\n            ".join(all_columns) if all_columns else ""

        stat_aggregates_str = ",\n            ".join(stat_aggregates) if stat_aggregates else ""

        # Construct columns for snapshots:
        # Always have fm.weightclass and sd.snapshot_date
        columns_for_snapshots = ["fm.weightclass", "sd.snapshot_date"]
        if stat_aggregates_str.strip():
            columns_for_snapshots.append(stat_aggregates_str)
        columns_for_snapshots_str = ",\n            ".join(columns_for_snapshots)

        # Construct columns for final_snapshots:
        # Always have weightclass, snapshot_date, valid_until
        final_columns = [
            "weightclass",
            "snapshot_date",
            "(snapshot_date + INTERVAL '1 year') AS valid_until"
        ]
        if all_columns_str.strip():
            final_columns.append(all_columns_str)
        final_columns_str = ",\n            ".join(final_columns)

        query = f"""
        snapshot_dates_cte AS (
            SELECT snapshot_date::date AS snapshot_date
            FROM (
                VALUES
                {snapshot_values}
            ) v(snapshot_date)
        ),
        snapshots AS (
            SELECT
                {columns_for_snapshots_str}
            FROM snapshot_dates_cte sd
            JOIN {self.schema}.{table} f ON TRUE
            JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
            JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
            WHERE em.event_date < sd.snapshot_date
            AND em.event_date >= (sd.snapshot_date - INTERVAL '{days_window} days')
            GROUP BY fm.weightclass, sd.snapshot_date
        ),
        final_snapshots AS (
            SELECT
                {final_columns_str}
            FROM snapshots
        )
        """

        return query
    
    def get_rolling_per_min_snapshots(self, 
                                      table: str, 
                                      event_features: List[str],
                                      window_years: int = 2,
                                      snapshot_interval_years: int = 1) -> str:
        """
        Generate a SQL snippet for rolling 2-year window rate stats.
        For each event feature, we compute a Bayesian prior with parameters alpha/beta
        that adapt based on both the amount of data in terms of time and attempts.

        Formulas:
        avg_rate_min = (sum_events * 60 / sum_time) if sum_time>0 else 0
        total_minutes = sum_time / 60
        total_events = sum_events

        prior_strength = 10.0 / [ (1.0 + LN(1.0 + total_minutes)) * (1.0 + LN(1.0 + total_events)) ]

        alpha = (avg_rate_min * prior_strength) + 1
        beta = prior_strength + 1

        The more total_minutes and total_events we have, the larger the denominator and the smaller the prior_strength, resulting in weaker smoothing.
        """
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')

        snapshot_dates = []
        current = start_dt
        while current <= end_dt:
            snapshot_dates.append(current.strftime('%Y-%m-%d'))
            current = datetime(current.year + snapshot_interval_years, current.month, current.day)

        # If no snapshot dates, return empty CTEs
        if not snapshot_dates:
            return """
            snapshot_dates_cte AS (
                SELECT NULL::date AS snapshot_date WHERE false
            ),
            snapshots AS (
                SELECT NULL::text as weightclass, NULL::date as snapshot_date WHERE false
            ),
            final_snapshots AS (
                SELECT NULL::text as weightclass, NULL::date as snapshot_date, NULL::date as valid_until WHERE false
            )
            """

        snapshot_values = ",\n".join([f"('{sd}')" for sd in snapshot_dates])
        days_window = window_years * 365

        stat_aggregates = []
        alpha_beta_expressions = []

        for feat in event_features:
            if '_rd1' in feat:
                time_col = 'time_sec_rd1'
            else:
                time_col = 'time_sec'

            base = feat
            # sum_events, sum_time, count_rows
            stat_aggregates.append(f"SUM({feat}::float) AS {base}_sum_events")
            stat_aggregates.append(f"SUM({time_col}::float) AS {base}_sum_time")
            stat_aggregates.append(f"COUNT(*) AS {base}_count_rows")

            # prior_strength = 10.0 / ((1.0 + LN(1.0 + {base}_sum_time/60.0)) * (1.0 + LN(1.0 + {base}_sum_events)))
            # avg_rate_min = (sum_events*60/sum_time) if sum_time>0 else 0

            alpha_beta_expressions.append(f"""
            CASE WHEN {base}_sum_time > 0 THEN 
                (
                    -- Compute prior_strength
                    10.0 / (
                        (1.0 + LN(1.0 + ({base}_sum_time/60.0))) 
                        * (1.0 + LN(1.0 + {base}_sum_events))
                    )
                )
            ELSE
                10.0
            END AS {base}_prior_strength,
            CASE WHEN {base}_sum_time > 0 THEN
                (
                    (((( {base}_sum_events * 60.0 / {base}_sum_time) ) * 
                      (
                        10.0 / (
                            (1.0 + LN(1.0 + ({base}_sum_time/60.0))) 
                            * (1.0 + LN(1.0 + {base}_sum_events))
                        )
                      )
                    ) + 1.0)
                )
            ELSE
                1.0
            END AS {base}_alpha,
            CASE WHEN {base}_sum_time > 0 THEN
                (
                    (
                        10.0 / (
                            (1.0 + LN(1.0 + ({base}_sum_time/60.0))) 
                            * (1.0 + LN(1.0 + {base}_sum_events))
                        )
                    ) + 1.0
                )
            ELSE
                11.0
            END AS {base}_beta
            """)

        stat_aggregates_str = ",\n                ".join(stat_aggregates) if stat_aggregates else ""
        alpha_beta_str = ",\n                ".join(alpha_beta_expressions) if alpha_beta_expressions else ""

        columns_for_snapshots = ["fm.weightclass", "sd.snapshot_date"]
        if stat_aggregates_str.strip():
            columns_for_snapshots.append(stat_aggregates_str)

        columns_for_snapshots_str = ",\n                ".join(columns_for_snapshots)

        final_columns = [
            "weightclass",
            "snapshot_date",
            "(snapshot_date + INTERVAL '1 year') AS valid_until"
        ]
        if alpha_beta_str.strip():
            final_columns.append(alpha_beta_str)
        final_columns_str = ",\n                ".join(final_columns)

        query = f"""
        snapshot_dates_cte AS (
            SELECT snapshot_date::date AS snapshot_date
            FROM (
                VALUES
                {snapshot_values}
            ) v(snapshot_date)
        ),
        snapshots AS (
            SELECT
                {columns_for_snapshots_str}
            FROM snapshot_dates_cte sd
            JOIN {self.schema}.{table} f ON TRUE
            JOIN {self.schema}.event_mapping em ON f.event_id = em.event_id
            JOIN {self.schema}.fight_mapping fm ON f.fight_id = fm.fight_id
            WHERE em.event_date < sd.snapshot_date
              AND em.event_date >= (sd.snapshot_date - INTERVAL '{days_window} days')
            GROUP BY fm.weightclass, sd.snapshot_date
        ),
        final_snapshots AS (
            SELECT
                {final_columns_str}
            FROM snapshots
        )
        """

        return query
    
    def get_rolling_weightclass_params(self, 
                                       table: str, 
                                       columns: List[str],
                                       window_years: int = 2,
                                       snapshot_interval_years: int = 1) -> str:
        """
        Generate a SQL CTE that computes rolling 2-year window parameters per weightclass and stat.
        
        The logic:
        1. Generate snapshot dates at a given interval (e.g. yearly). For simplicity, we assume yearly snapshots on January 1 of each year in [start_date, end_date].
        2. For each snapshot date S, aggregate stats for the [S - 2 years, S) window.
        3. Store these results in a snapshot table keyed by weightclass and snapshot_date.
        4. When joining to fights, for a fight on date F, find the snapshot whose date S is the latest possible that still satisfies F <= S + 1 year.
           This means we pick the snapshot whose S is <= F+1year and is the greatest such S.
        
        This returns a CTE snippet that you can incorporate into your final query.
        
        Args:
            table: The source stats table (e.g. 'fight_stats_derived').
            columns: List of columns (stats) to aggregate.
            window_years: Size of the rolling window (default 2 years).
            snapshot_interval_years: Interval between snapshots (default 1 year).
            
        Returns:
            A SQL string containing:
             - A CTE 'snapshots' with columns: weightclass, snapshot_date, and aggregated stats per weightclass.
             - A CTE 'extended_snapshots' that adds a +1 year boundary to each snapshot_date.
             - You can then LEFT JOIN fights on conditions to find the appropriate snapshot.
        """
        
        # Convert start/end dates to datetime
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Generate snapshot dates: every `snapshot_interval_years` starting from start_date up to end_date
        snapshot_dates = []
        current = start_dt
        while current <= end_dt:
            snapshot_dates.append(current.strftime('%Y-%m-%d'))
            # Move by snapshot_interval_years
            current = datetime(current.year + snapshot_interval_years, current.month, current.day)
        
        # Build a VALUES list of snapshot dates
        snapshot_values = ",\n".join([f"('{sd}')" for sd in snapshot_dates])
        
        # Aggregations: For generality, we'll just do STDDEV and COUNT as examples.
        # You can customize as needed or add more aggregations (mean, sum, etc.)
        # We'll alias columns with _agg suffixes.
        agg_expressions = [f"STDDEV({col}) AS {col}_stddev" for col in columns]
        agg_select = ", ".join(agg_expressions)
        
        # Window length in days (approx) for filtering
        # We'll assume 365 days per year for simplicity. Adjust if you need more accuracy.
        days_window = window_years * 365
        
        # CTE Explanation:
        # 1. snapshot_dates_cte: holds all snapshot dates
        # 2. snapshots: For each snapshot_date S, aggregate stats for [S - 2y, S)
        #    We join to event_mapping/fight_mapping and filter on event_date between S - 2y and S
        # 3. extended_snapshots: Add a column valid_until = snapshot_date + 1 year
        #    This will be used to find which snapshot applies to a given fight date.
        
        # Note: For the date filtering, we'll use:
        # event_date < snapshot_date::date
        # event_date >= (snapshot_date::date - interval 'X days')
        
        query = f"""
            WITH snapshot_dates_cte AS (
                VALUES
                {snapshot_values}
            ) AS snapshot_dates(snapshot_date),
            
            snapshots AS (
                SELECT
                    fm.weightclass,
                    sd.snapshot_date::date AS snapshot_date,
                    {agg_select}
                FROM snapshot_dates sd
                JOIN features.{table} f ON TRUE
                JOIN features.event_mapping em ON f.event_id = em.event_id
                JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
                WHERE em.event_date < sd.snapshot_date::date
                  AND em.event_date >= (sd.snapshot_date::date - INTERVAL '{days_window} days')
                GROUP BY fm.weightclass, sd.snapshot_date
            ),
            
            extended_snapshots AS (
                SELECT
                    weightclass,
                    snapshot_date,
                    (snapshot_date + INTERVAL '1 year') AS valid_until,
                    {', '.join([f"{col}_stddev" for col in columns])}
                FROM snapshots
            )
            
            -- extended_snapshots now holds rolling stats per weightclass,
            -- each snapshot_date covers a 2-year window of past data,
            -- and is valid up to snapshot_date + 1 year.
            
            -- To use it:
            -- For a fight at date F, find the snapshot where:
            -- snapshot_date <= F+1year AND snapshot_date is the greatest possible that still meets this condition.
            -- One way: LEFT JOIN extended_snapshots on fm.weightclass = extended_snapshots.weightclass
            -- and extended_snapshots.snapshot_date <= (FIGHT_DATE + interval '1 year')
            -- then pick the max snapshot_date via a window function or a subselect.
        """
        
        return query

        # Example usage (not part of requested code):
        # In your StandardDeviationCalculator or AccuracyCalculator, you can do:
        # rolling_params_cte = self.feature_utils.get_rolling_weightclass_params('fight_stats_derived', ['sig_str', 'td', ...], window_years=2, snapshot_interval_years=1)
        # Then incorporate `rolling_params_cte` into your final query as a WITH clause or a subselect.


        """
        Explanation:

        - `get_rolling_weightclass_params` returns a SQL string defining several CTEs:
        - `snapshot_dates_cte`: Lists all snapshot dates at yearly intervals.
        - `snapshots`: For each snapshot_date, we aggregate stats from the preceding 2 years of data per weightclass.
        - `extended_snapshots`: Adds a `valid_until` column = snapshot_date + 1 year.

        How to Use:
        - Integrate the returned SQL into your final query by prefixing it with `WITH`.
        - After `extended_snapshots` is defined, you can LEFT JOIN your fights on `weightclass` and choose the appropriate snapshot based on `event_date`.
        For example:
        
        WITH ... (the returned CTEs),
        fight_data AS (
            SELECT
            f.fight_id,
            f.fighter_id,
            fm.weightclass,
            em.event_date
            FROM features.fight_stats_derived f
            JOIN features.event_mapping em ON f.event_id = em.event_id
            JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
        )
        SELECT
            fd.*,
            es.* -- or selected columns from extended_snapshots
        FROM fight_data fd
        LEFT JOIN LATERAL (
            SELECT * FROM extended_snapshots es
            WHERE es.weightclass = fd.weightclass
            AND es.snapshot_date <= fd.event_date + INTERVAL '1 year'
            ORDER BY es.snapshot_date DESC
            LIMIT 1
        ) es ON TRUE;

        This lateral join finds the most recent snapshot whose snapshot_date is not greater than fd.event_date + 1 year.
        The final chosen snapshot gives you the 2-year window stddevs or other aggregated parameters that you can use for first-fight logic or Bayesian priors.
        """

    def compute_global_zero_freq(self, stats_dict: Dict[str, List[str]], pprint: bool = False) -> Dict[str, float]:
        """Calculate zero frequencies for all stats in the stats dictionary.
        
        Args:
            stats_dict: Dictionary mapping base stats to their column names
            pprint: Boolean flag to enable/disable pretty printing of results
            
        Returns:
            Dictionary mapping each stat to its zero frequency
        """
        zero_freqs = {}
        
        if pprint:
            print("\nComputing global zero frequencies for each stat:")
            print("-" * 80)
        
        for table_name, columns in stats_dict.items():
            for column in columns:
                sql = f"""
                    SELECT 
                        AVG(CASE 
                            WHEN {column} = 0 THEN 1.0 
                            ELSE 0.0 
                        END) as zero_freq,
                        COUNT(*) as total_rows
                    FROM features.{table_name}
                    WHERE {column} IS NOT NULL
                """
                
                try:
                    result = self.conn.execute(text(sql)).fetchone()
                    
                    if result is None or result[1] == 0:
                        print(f"⚠️  Warning: No valid rows found for {column}")
                        zero_freqs[column] = 0.0
                        continue
                        
                    zero_freq = result[0] if result[0] is not None else 0.0
                    zero_freqs[column] = zero_freq
                    
                    if pprint:
                        print(f"\n Column: {column} (from table: {table_name})")
                        print(f"  ├─ Zero Frequency: {zero_freq:.3f}")
                        print(f"  └─ Total Rows: {result[1]:,}")
                        
                except Exception as e:
                    print(f"⚠️  Error calculating zero frequency for {column}: {str(e)}")
                    zero_freqs[column] = 0.0
        
        if pprint:
            print("\n" + "-" * 80)
        
        return zero_freqs

    def compute_global_averages(self, stats_dict: Dict[str, List[str]], exclude_zeros: bool = False, pprint: bool = False) -> Dict[str, float]:
        """Calculate average values for all stats in the stats dictionary.
        
        Args:
            stats_dict: Dictionary mapping base stats to their column names
            exclude_zeros: If True, excludes zero values from the average calculations
            pprint: Boolean flag to enable/disable pretty printing of results
            
        Returns:
            Dictionary mapping each stat to its average value
        """
        averages = {}
        
        if pprint:
            print("\nComputing global averages for each stat:")
            print("-" * 80)
        
        for table_name, columns in stats_dict.items():
            for column in columns:
                where_clause = f"WHERE {column} IS NOT NULL"
                if exclude_zeros:
                    where_clause += f" AND {column} != 0"
                
                sql = f"""
                    SELECT 
                        AVG(CAST({column} AS FLOAT)) as avg_value,
                        COUNT(*) as total_rows
                    FROM features.{table_name}
                    {where_clause}
                """
                
                try:
                    result = self.conn.execute(text(sql)).fetchone()
                    
                    if result is None or result[1] == 0:
                        print(f"⚠️  Warning: No valid rows found for {column}")
                        averages[column] = 0.0
                        continue
                    
                    avg_value = result[0] if result[0] is not None else 0.0
                    averages[column] = avg_value
                    
                    if pprint:
                        print(f"\n Column: {column} (from table: {table_name})")
                        print(f"  ├─ Average Value: {avg_value:.2f}")
                        print(f"  └─ Total Rows: {result[1]:,}")
                    
                except Exception as e:
                    print(f"⚠️  Error calculating average for {column}: {str(e)}")
                    averages[column] = 0.0
        
        if pprint:
            print("\n" + "-" * 80)
        
        return averages

    def compute_global_acc_stats(self, stats_dict: Dict[str, List[str]], pprint: bool = False) -> Dict[str, Dict[str, float]]:
        """Compute global accuracy statistics for each stat using Beta-Binomial conjugate prior without future leakage.
        
        Args:
            stats_dict: Dictionary mapping table names to their stat columns (land columns).
            cutoff_date: Optional cutoff date (YYYY-MM-DD) to avoid using future data. 
                         If provided, only fights on or before this date are included.
            pprint: Boolean to pretty print the output.

        Returns:
            global_stats: A dictionary mapping stat keys to their Bayesian prior parameters (alpha, beta).
        """
        global_stats = {}
        
        if pprint:
            print("\nComputing global accuracy statistics...")

        # Construct date filter
        date_filter = f"AND em.event_date >= '{self.start_date}'"
        if self.cutoff_date:
            date_filter += f" AND em.event_date <= '{self.cutoff_date}'"

        for table_name, base_stats in stats_dict.items():
            for base_stat in base_stats:
                land_col = base_stat
                att_col = base_stat.replace('land', 'att')
                
                sql = f"""
                    WITH base_stats AS (
                        SELECT 
                            {land_col},
                            {att_col},
                            CASE 
                                WHEN {att_col} > 0 THEN 
                                    CAST({land_col} AS FLOAT) / CAST({att_col} AS FLOAT)
                                ELSE NULL
                            END as accuracy
                        FROM {self.schema}.{table_name} t
                        JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
                        JOIN features.event_mapping em ON fm.event_id = em.event_id
                        WHERE {att_col} IS NOT NULL
                        {date_filter}
                    )
                    SELECT 
                        AVG(accuracy) as global_acc,
                        AVG(CASE WHEN {land_col} = 0 THEN 1.0 ELSE 0.0 END) as zero_land_freq,
                        AVG(CASE WHEN {att_col} > 0 THEN {att_col} ELSE NULL END) as avg_att,
                        COUNT(*) as total_rows,
                        AVG({land_col}) as avg_land,
                        AVG({att_col}) as avg_att_all
                    FROM base_stats
                """
                
                result = self.conn.execute(text(sql)).fetchone()
                
                if result is None or result[3] == 0:
                #if result is None or result['total_rows'] == 0:
                    # If no data, just skip
                    continue
                
                if pprint:
                    self._print_acc_stats(base_stat, table_name, result)
                
                params = self._calculate_acc_params(result)
                
                if pprint:
                    self._print_acc_params(params)
                
                stat_key = base_stat.replace('_land', '')
                global_stats[stat_key] = {
                    'alpha': params['alpha'],
                    'beta': params['beta'],
                    'table': table_name,
                    'land_col': land_col,
                    'att_col': att_col
                }
        
        return global_stats
    
    def _calculate_acc_params(self, result) -> Dict[str, float]:
        """Calculate Bayesian parameters for accuracy stats using adaptive priors.
        
        Key steps:
        1. Extract base statistics:
           - global_acc: Average accuracy across all fighters
           - zero_freq: How often the stat is 0 (indicates rarity)
           - avg_att: Average number of attempts (volume)
        
        2. Categorize the stat type:
           - Submissions: Very rare, high zero_freq
           - Takedowns: Moderately rare
           - Strikes: More common, varies by location
        
        3. Set base_k by stat type:
           - Lower k (0.08-0.15): Rare events like subs/TDs
           - Medium k (0.30-0.52): Common strikes
           - Higher k (0.70-0.90): Very common strikes
        
        4. Adjust for volume characteristics:
           - zero_adj: Increases k for rare events
           - volume_factor: Controls how volume affects smoothing
           - prior_weight: Final smoothing strength = k/(volume^factor)
        
        5. Calculate Beta parameters:
           α = prior_weight * global_acc + base_smooth
           β = prior_weight * (1-global_acc) + base_smooth
           
        The resulting α,β parameters create a Beta distribution that:
        - Centers around the global average
        - Has variance based on volume/rarity
        - Provides more smoothing for rare events
        - Allows more variance for common events
        """
        global_acc = float(result[0] if result[0] is not None else 0.5)
        zero_freq = float(result[1] if result[1] is not None else 0.0)
        avg_att = float(max(result[2] or 0.1, 0.1))
        avg_land = float(result[4] if result[4] is not None else 0.0)
        stat_name = str(result[-1]).lower() if result[-1] else ''
        
        # Precise event categorization
        is_submission = 'sub' in stat_name and zero_freq > 0.8
        is_takedown = 'takedown' in stat_name or 'td' in stat_name
        is_leg = 'leg' in stat_name
        is_head = 'head' in stat_name
        is_clinch = 'clinch' in stat_name
        is_ground = 'ground' in stat_name
        is_body = 'body' in stat_name
        is_distance = 'distance' in stat_name
        
        # Volume/accuracy characteristics
        is_very_rare = avg_land < 0.2 and zero_freq > 0.8
        is_rare = avg_land < 1.0 or zero_freq > 0.5
        is_high_acc = global_acc > 0.75 and avg_land > 5
        is_high_volume = avg_att > 50
        is_low_acc = global_acc < 0.40
        
        # Type-specific base_k values
        base_k = (
            0.08 if is_submission else    # Perfect: 32.0 → 29.3
            0.10 if is_takedown else     # Reduced from 0.15 to 0.10 for takedowns
            0.52 if is_leg else          # Target: 80.9 → 77.5
            0.35 if is_head else         # Perfect: 37.0 → 37.1
            0.30 if is_distance else     # Perfect: 39.0 → 39.0
            0.80 if is_clinch else       # Target: 67.4 → 65.0
            0.90 if is_ground else       # Target: 69.6 → 65.0
            0.70                         # Target: 69.3 → 67.5
        )
        
        # Volume-based adjustments - reduce zero_adj for takedowns
        zero_adj = (
            0.2 if zero_freq > 0.9 and is_very_rare and not is_takedown else
            0.15 if zero_freq > 0.8 and not is_takedown else
            0.05 if zero_freq > 0.5 and is_takedown else  # Reduced adjustment for takedowns
            0.1 if zero_freq > 0.5 else
            0.05 if zero_freq > 0.3 else
            0
        )
        k = base_k + zero_adj
        
        # Minimum attempt threshold - special case for takedowns
        if avg_att < 3:
            k *= 0.5  # More reduction for takedowns/subs
        
        # Refined volume scaling - increase denominator for takedowns
        volume_factor = (
            1/2 if is_very_rare and not is_takedown else     # Submissions
            1/2 if is_takedown else          # Increased from 1/3 to 1/2 for takedowns
            1/3 if is_rare else              # Other rare events
            1/3 if is_high_acc else          # Leg strikes
            1/8 if is_high_volume and is_low_acc else  # Head/Distance
            1/6 if is_high_volume else       # Body high volume
            1/5 if avg_land > 5 else       # Medium volume
            1/4                              # Default
        )
        prior_weight = k / (avg_att ** volume_factor)
        
        # Lower cap for takedowns
        if prior_weight > (1.0 if is_takedown or is_submission else 2.0):
            prior_weight = 1.0 if is_takedown or is_submission else 2.0
        
        # Type-specific smoothing - reduce base_smooth for takedowns
        base_smooth = (
            0.03 if is_submission else   # Very light
            0.04 if is_takedown else    # Reduced from 0.08 to 0.04 for takedowns
            0.05 if is_leg else         # Minimal for high acc
            0.03 if is_high_volume else # Very light
            0.08 if is_clinch else      # Medium
            0.015 if is_ground else     # Reduced from 0.02
            0.07                        # Light-medium
        )
        
        alpha = prior_weight * global_acc + base_smooth
        beta = prior_weight * (1 - global_acc) + base_smooth
        
        return {
            'k': k,
            'prior_weight': prior_weight,
            'alpha': alpha,
            'beta': beta
        }

    def compute_kd_power_stats(self, table, pprint: bool = False) -> Dict[str, Dict[str, float]]:
        """Compute Beta-Binomial prior parameters (alpha, beta) for KD power per weight class."""
        if pprint:
            print("\nComputing KD power statistics per weight class...")
        
        sql = f"""
            WITH base_stats AS (
                SELECT 
                    fm.weightclass,
                    fs.kd_total,
                    (fs.distance_att_total + fs.clinch_att_total) as total_attempts,
                    CASE 
                        WHEN (fs.distance_att_total + fs.clinch_att_total) > 0 THEN
                            CAST(fs.kd_total AS FLOAT) / 
                            CAST((fs.distance_att_total + fs.clinch_att_total) AS FLOAT)
                        ELSE NULL
                    END as kd_ratio
                FROM {self.schema}.{table} fs
                JOIN features.fight_mapping fm ON fs.fight_id = fm.fight_id
                WHERE fs.distance_att_total IS NOT NULL 
                AND fs.clinch_att_total IS NOT NULL
            )
            SELECT 
                weightclass,
                AVG(kd_ratio) as global_ratio,
                AVG(CASE WHEN kd_total=0 THEN 1.0 ELSE 0.0 END) as zero_freq,
                COUNT(*) as total_rows
            FROM base_stats
            WHERE kd_ratio IS NOT NULL
            GROUP BY weightclass
        """
        
        results = self.conn.execute(text(sql)).fetchall()
        power_stats = {}
        
        for row in results:
            weightclass = row[0]
            global_ratio = float(row[1] if row[1] is not None else 0.0)
            zero_freq = float(row[2] if row[2] is not None else 1.0)
            total_rows = row[3]

            # Determine prior_strength from zero_freq (linear scale)
            # Higher zero_freq -> larger prior_strength -> more shrinkage
            prior_strength = 1.0 + 59.0 * zero_freq  # min=1, max=60

            # Compute alpha, beta from global_ratio and prior_strength
            # Mean of Beta(α, β) = α/(α+β) = global_ratio
            # Let α = global_ratio * prior_strength
            # and β = (1 - global_ratio) * prior_strength
            # If global_ratio=0, just set a small α to avoid division by zero
            alpha = global_ratio * prior_strength if global_ratio > 0 else 0.1
            beta = (1.0 - global_ratio) * prior_strength if global_ratio < 1 else 0.1

            power_stats[weightclass] = {
                'alpha': alpha,
                'beta': beta
            }

            if pprint:
                print(f"Weightclass: {weightclass}, global_ratio={global_ratio:.4f}, zero_freq={zero_freq:.2f}, alpha={alpha:.2f}, beta={beta:.2f}")

        return power_stats

    def _calculate_power_params(self, result) -> Dict[str, float]:
        """Calculate Bayesian parameters for power stats."""
        weight_class = result[0]
        global_ratio = result[1] if result[1] is not None else 0.0
        zero_freq = result[2] if result[2] is not None else 1.0
        avg_attempts = max(result[3] or 0.1, 0.1)
        avg_kds = result[5] if result[5] is not None else 0.0
        
        # Scale base k with observed KD rates
        if weight_class in ['open weight', 'catchweight']:
            k = 1  # Minimal impact
        else:
            # Base k values reflect actual KD rates
            base_k = {
                'heavyweight': 4,
                'light heavyweight': 4,
                'middleweight': 3,
                'welterweight': 3,
                'lightweight': 2,
                'featherweight': 2,
                'bantamweight': 2,
                'flyweight': 1
            }.get(weight_class, 1)
            
            # Small adjustment for very high zero rates
            k = base_k + 1 if zero_freq > 0.8 else base_k
        
        # Use seventh root for gentler volume scaling
        prior_weight = k / (avg_attempts ** (1/7))
        
        return {
            'k': k,
            'prior_weight': prior_weight,
            'global_ratio': global_ratio,
            'avg_kds': avg_kds
        }

    def get_prior_strength(self, zero_freq: float, min_strength: float = 1.0, max_strength: float = 60.0) -> float:
        # Linear mapping from zero frequency [0,1] to prior_strength
        zero_freq = max(0.0, min(zero_freq, 1.0))
        return min_strength + (max_strength - min_strength) * zero_freq

    def compute_global_rate_stats(self, stats_dict: Dict[str, List[str]], pprint: bool = False) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute global rate statistics and convert them into Gamma-Poisson prior parameters without future data leakage.
        
        Args:
            stats_dict: Dictionary mapping tables to their feature columns
            pprint: Whether to print detailed statistics
            
        Returns:
            Nested dictionary of rate statistics by feature and weightclass
        """

        rate_stats = {}
        
        if pprint:
            print("\nComputing rate statistics...")

        # Build date filter to ensure no future data leakage
        date_filter = f"AND em.event_date >= '{self.start_date}'"
        if self.cutoff_date:
            date_filter += f" AND em.event_date <= '{self.cutoff_date}'"

        for table_name, features in stats_dict.items():
            for feat in features:
                # Determine time column
                time_col = (
                    'time_sec_rd1_total' if '_rd1_total' in feat else
                    'time_sec_rd1' if '_rd1' in feat else
                    'time_sec_total' if '_total' in feat else
                    'time_sec'
                )
                
                sql = f"""
                    WITH rate_stats AS (
                        SELECT 
                            fm.weightclass,
                            t.{feat},
                            t.{time_col},
                            CASE 
                                WHEN t.{time_col} > 0 THEN 
                                    CAST(t.{feat} AS FLOAT) / t.{time_col} -- events/sec
                                ELSE NULL
                            END as rate_per_sec
                        FROM {self.schema}.{table_name} t
                        JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id 
                            AND (t.fighter_id = fm.fighter1_id OR t.fighter_id = fm.fighter2_id)
                        JOIN features.event_mapping em ON fm.event_id = em.event_id
                        WHERE t.{time_col} IS NOT NULL
                        {date_filter}
                    )
                    SELECT 
                        weightclass,
                        AVG(rate_per_sec) as avg_rate_sec,
                        AVG(CASE WHEN {feat} = 0 THEN 1.0 ELSE 0.0 END) as zero_freq,
                        AVG({time_col}) as avg_time,
                        COUNT(*) as total_rows,
                        AVG({feat}) as avg_count
                    FROM rate_stats
                    GROUP BY weightclass
                """
                
                results = self.conn.execute(text(sql)).fetchall()
                
                if not results:
                    continue
                
                feat_stats = {}
                for result in results:
                    weightclass = result[0]
                    
                    if pprint:
                        self._print_rate_stats(feat, table_name, result, weightclass)
                    
                    params = self._calculate_rate_params(result)
                    
                    if pprint:
                        self._print_rate_params(params, weightclass)
                    
                    feat_stats[weightclass] = {
                        'alpha': params['alpha'],
                        'beta': params['beta']
                    }
                
                rate_stats[feat] = feat_stats
        
        return rate_stats

    def _calculate_rate_params(self, result) -> Dict[str, float]:
        """Calculate Bayesian parameters for rate statistics using Gamma-Poisson priors."""
        avg_rate_sec = float(result[1] if result[1] is not None else 0.0)
        zero_freq = float(result[2] if result[2] is not None else 1.0)
        avg_count = float(result[5] if result[5] is not None else 0.0)

        # Convert avg_rate from events/sec to events/min
        avg_rate_min = avg_rate_sec * 60.0

        # Base prior strength from zero frequency
        base_prior_strength = self.get_prior_strength(zero_freq, min_strength=1.0, max_strength=60.0)
        
        # Adjust prior strength based on volume
        volume_factor = 1.0 / (1.0 + np.log1p(avg_count))  # Decreases as volume increases
        prior_strength = base_prior_strength * volume_factor

        # Apply bounds AFTER volume adjustment
        prior_strength = max(1.0, min(60.0, prior_strength))

        # Gamma prior parameters: 
        # Mean of Gamma(alpha,beta) = alpha/beta. We want mean ≈ avg_rate_min,
        # so set alpha = avg_rate_min * prior_strength and beta = prior_strength 
        # gives mean = (avg_rate_min * prior_strength)/prior_strength = avg_rate_min.
        
        alpha = avg_rate_min * prior_strength
        beta = prior_strength

        return {
            'alpha': alpha,
            'beta': beta,
            'prior_strength': prior_strength,
            'avg_rate_min': avg_rate_min
        }

    def get_stat_tables(self) -> Dict[str, List[str]]:
        """Get all stat-specific tables and their columns."""
        tables = {}
        for stat in self.base_stat_prefixes + self.static_stat_prefixes:
            
            # Get columns for both main and rd1 tables
            for suffix in ['', '_rd1']:
                full_table = f"{stat}{suffix}"
                columns = self.get_columns_from_table(
                    self.schema, 
                    full_table,
                    exclude_strs=['fight_id', 'fighter_id', 'event_id']
                )
                if columns:
                    tables[full_table] = columns
                    
        return tables

    def get_opponent_stats(self, table: str, exclude_patterns: set = None) -> str:
        """Generate SQL to get opponent's stats for a given feature-specific table."""
        exclude_patterns = exclude_patterns or set()
        exclude_columns = {'fight_id', 'fighter_id', 'event_id', 'fighter1_id', 'fighter2_id'}

        # Get columns from the specific feature table
        columns_query = f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = 'features' 
            AND table_name = '{table}'
        """
        columns = [row[0] for row in self.conn.execute(text(columns_query)).fetchall()]

        # Filter out excluded columns and patterns
        opponent_columns = [
            col for col in columns
            if col not in exclude_columns
            and not any(pattern in col for pattern in exclude_patterns)
        ]

        # Build the core CTE with stable references
        # We pick the opponent_id using CASE based on whether the current fighter_id matches fighter1 or fighter2.
        return f"""
            WITH fighter_stats AS (
                SELECT 
                    f.fight_id,
                    f.fighter_id,
                    fm.fighter1_id,
                    fm.fighter2_id,
                    CASE 
                        WHEN f.fighter_id = fm.fighter1_id THEN fm.fighter2_id
                        ELSE fm.fighter1_id
                    END AS opponent_id,
                    {', '.join(f'f.{col}' for col in opponent_columns)}
                FROM features.{table} f
                JOIN features.fight_mapping fm ON f.fight_id = fm.fight_id
            )
            SELECT 
                fs1.fight_id,
                fs1.fighter_id,
                {', '.join(f'fs2.{col} AS {col}' for col in opponent_columns)}
            FROM fighter_stats fs1
            JOIN fighter_stats fs2
                ON fs1.fight_id = fs2.fight_id
                AND fs1.opponent_id = fs2.fighter_id
        """

    def _print_rate_stats(self, feat: str, table_name: str, result, weightclass: str) -> None:
        """Pretty print the rate statistics results."""
        print(f"\n Feature: {feat} (from table: {table_name}, weightclass: {weightclass})")
        print("  ├─ Avg Rate: {:.3f}".format(result[1] if result[1] is not None else 0))
        print("  ├─ Zero Frequency: {:.3f}".format(result[2] if result[2] is not None else 0))
        print("  ├─ Avg Time: {:.2f}".format(result[3] if result[3] is not None else 0))
        print("  ├─ Total Rows: {:,}".format(result[4]))
        print("  └─ Avg Count: {:.3f}".format(result[5] if result[5] is not None else 0))

    def _print_rate_params(self, params: Dict[str, float], weightclass: str) -> None:
        """Pretty print the rate parameters."""
        print(f"\n  Rate Parameters for {weightclass}:")
        print("  ├─ K: {}".format(params['k']))
        print("  ├─ Prior Weight: {:.3f}".format(params['prior_weight']))
        print("  └─ Prior Rate: {:.3f}".format(params['prior_rate']))

    def _print_power_stats(self, weight_class: str, result) -> None:
        """Pretty print the power statistics results."""
        print(f"\n Weight Class: {weight_class}")
        print("  ├─ Global Power Ratio: {:.6f}".format(result[1] if result[1] is not None else 0))
        print("  ├─ Zero Frequency: {:.3f}".format(result[2] if result[2] is not None else 0))
        print("  ├─ Avg Strike Attempts: {:.2f}".format(result[3] if result[3] is not None else 0))
        print("  ├─ Total Rows: {:,}".format(result[4]))
        print("  └─ Avg KDs: {:.3f}".format(result[5] if result[5] is not None else 0))

    def _print_power_params(self, weight_class: str, params: Dict[str, float]) -> None:
        """Pretty print the power parameters."""
        print(f"\n  Power Parameters for {weight_class}:")
        print("  ├─ K: {}".format(params['k']))
        print("  ├─ Prior Weight: {:.6f}".format(params['prior_weight']))
        print("  ├─ Global Ratio: {:.6f}".format(params['global_ratio']))
        print("  └─ Avg KDs: {:.3f}".format(params['avg_kds']))

    def _print_acc_stats(self, stat: str, table_name: str, result) -> None:
        """Pretty print the accuracy statistics results."""
        print(f"\n Stat: {stat} (from table: {table_name})")
        print("  ├─ Global Accuracy: {:.3f}".format(result[0] if result[0] is not None else 0))
        print("  ├─ Zero land Frequency: {:.3f}".format(result[1] if result[1] is not None else 0))
        print("  ├─ Avg Attempts: {:.2f}".format(result[2] if result[2] is not None else 0))
        print("  ├─ Total Rows: {:,}".format(result[3]))
        print("  ├─ Avg land: {:.2f}".format(result[4] if result[4] is not None else 0))
        print("  └─ Avg att: {:.2f}".format(result[5] if result[5] is not None else 0))

    def _print_acc_params(self, params: Dict[str, float]) -> None:
        """Pretty print the accuracy parameters."""
        print("\n  Accuracy Parameters:")
        print("  ├─ K: {}".format(params['k']))
        print("  ├─ Prior Weight: {:.3f}".format(params['prior_weight']))
        print("  ├─ Alpha: {:.3f}".format(params['alpha']))
        print("  └─ Beta: {:.3f}".format(params['beta']))