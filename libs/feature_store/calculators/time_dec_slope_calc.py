from libs.feature_store.base_layer_calculator import BaseLayerCalculator
from typing import List, Set
from math import log

class TimedecSlopeCalculator(BaseLayerCalculator):
    """Calculate a time-decayed average slope (change) in a fighter's stats over time.
    
    The slope is computed as a weighted average of (col - previous_col) with exponential decay 
    based on how old the fights are relative to the current fight's date. More recent fights have higher weight.
    This avoids data leakage by referencing each fight's event_date rather than a global or future date.
    """
    
    def __init__(self, conn, decay_rate_years: float, include_patterns: Set[str] = set(), exclude_patterns: Set[str] = set()):
        """
        Initialize the calculator with a decay rate defined by the half-life in years.
        
        Args:
            conn: Database connection
            decay_rate_years: Half-life in years (e.g. 1.5 means events half-weight every 1.5 years)
            include_patterns: Set of patterns to include in the calculation
            exclude_patterns: Set of patterns to exclude from the calculation
        """
        super().__init__(conn)
        self.layer_suffix = '_dec_slope'
        self.decay_rate = log(2) / decay_rate_years
        
        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)

    def calculate_for_table(self, table: str, columns: List[str]) -> str:
        """
        Calculate a time-decayed slope for the given columns.
        
        The slope is based on differences between consecutive fights: (col - prev_col).
        We apply an exponential decay factor based on how far in the past each previous fight is,
        relative to the current fight. This ensures no future knowledge is leaked.
        
        Steps:
        1. Gather base data (fights, event_date, features).
        2. Compute current_event_date (the event_date of the current fight row) using LAST_VALUE.
        3. Compute lagged values to find differences (col - prev_col).
        4. Compute weights based on decay relative to current_event_date.
        5. Compute a weighted average of the differences across all previous fights.
        
        Args:
            table: Source table name
            columns: List of columns to calculate slopes for
            
        Returns:
            SQL query string that computes the decayed slope features.
        """
        
        if not columns:
            return None
        
        # Step 1: Create a base CTE to select columns and compute LAG (previous value)
        # We'll also need current_event_date, so we do that in a separate CTE.
        base_selects = [f"f.fight_id", "f.fighter_id", "f.event_id", "em.event_date"]
        base_selects += [f"f.{col}" for col in columns]
        
        base_cte = f"""
            WITH base AS (
                SELECT
                    {', '.join(base_selects)}
                FROM features.{table} f
                JOIN features.event_mapping em ON f.event_id = em.event_id
            )
        """
        
        # Step 2: Create a CTE that adds current_event_date and previous values.
        # current_event_date is the event_date of the current fight row.
        # LAG(...) computes the previous fight's value for each column.
        
        lag_selects = []
        for col in columns:
            lag_selects.append(f"LAG({col}) OVER (PARTITION BY fighter_id ORDER BY event_date, fight_id) AS prev_{col}")
        
        cur_cte = f"""
            , cur AS (
                SELECT
                    fight_id,
                    fighter_id,
                    event_id,
                    event_date,
                    {', '.join(columns)},
                    {', '.join(lag_selects)},
                    LAST_VALUE(event_date) OVER (
                        PARTITION BY fighter_id ORDER BY event_date, fight_id
                        ROWS BETWEEN CURRENT ROW AND CURRENT ROW
                    ) AS current_event_date
                FROM base
            )
        """
        
        # Step 3: Compute differences (col - prev_col) and weights.
        # The weight for each row relative to the current row:
        # weight = exp(-decay_rate * ((current_event_date - event_date) / 365.25))
        
        diff_selects = []
        for col in columns:
            diff_selects.append(f"CASE WHEN prev_{col} IS NOT NULL THEN ({col} - prev_{col}) END AS diff_{col}")
        
        diff_cte = f"""
            , diff_calc AS (
                SELECT
                    fight_id,
                    fighter_id,
                    event_id,
                    event_date,
                    current_event_date,
                    {', '.join(columns)},
                    {', '.join([f"prev_{c}" for c in columns])},
                    {', '.join(diff_selects)},
                    EXP(-1 * {self.decay_rate} * ((current_event_date - event_date)::INTEGER / 365.25)) AS w
                FROM cur
            )
        """

        # Step 4: Compute the weighted slopes for each column.
        # sum_w = SUM(CASE WHEN prev_col IS NOT NULL THEN w ELSE 0 END)
        # sum_wdiff = SUM(CASE WHEN prev_col IS NOT NULL THEN diff_col * w ELSE 0 END)
        # slope = sum_wdiff / sum_w (if sum_w != 0 and count > 1)
        
        slope_selects = []
        for col in columns:
            sum_w = f"SUM(CASE WHEN prev_{col} IS NOT NULL THEN w ELSE 0 END) OVER (PARTITION BY fighter_id ORDER BY event_date, fight_id)"
            sum_wdiff = f"SUM(CASE WHEN prev_{col} IS NOT NULL THEN diff_{col} * w ELSE 0 END) OVER (PARTITION BY fighter_id ORDER BY event_date, fight_id)"
            count_fights = f"COUNT(*) OVER (PARTITION BY fighter_id ORDER BY event_date, fight_id)"
            
            slope_selects.append(f"""
                CASE
                    WHEN {count_fights} = 1 THEN 0.0
                    WHEN {sum_w} = 0 THEN 0.0
                    ELSE ({sum_wdiff} / {sum_w})
                END AS {col}{self.layer_suffix}
            """)

        final_query = f"""
            {base_cte}
            {cur_cte}
            {diff_cte}
            SELECT
                fight_id,
                fighter_id,
                event_id,
                {', '.join(slope_selects)}
            FROM diff_calc
            ORDER BY event_date, fight_id
        """

        return final_query
