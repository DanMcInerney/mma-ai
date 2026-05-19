"""
Statistical Quality Calculator

Computes distribution quality metrics for each stat × weightclass combination.
These metrics are used for adaptive adjperf calculation and winsorization.

Metrics computed:
- wc_mad: Cross-sectional MAD (from weightclass_mad table)
- pct_at_median: % of values within 0.001 of weightclass median
- unique_value_count: Number of distinct smoothed values
- effective_n: Effective sample size after accounting for degeneracy
- quality_tier: Classification (healthy/sparse/degenerate)
- recommended_winsor_limit: Adaptive clipping threshold

Output tables: features.<table>_quality_metrics
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.feature_store.feature_utils import FeatureUtils
from libs.feature_store.config.stat_quality_config import StatQualityConfig


class StatQualityCalculator(BaseCalculator):
    """
    Calculate distribution quality metrics for each stat by weightclass.

    This calculator analyzes the smoothed feature distributions to identify
    degenerate cases where cross-sectional variance has collapsed due to
    high rates of zero attempts and Beta-Binomial fallback to priors.

    Usage:
        context = CalculatorContext(conn)
        calc = StatQualityCalculator(context)
        results = calc.run()
    """

    def __init__(
        self,
        context_or_conn,
        include_patterns: set = set(),
        exclude_patterns: set = set()
    ):
        """
        Initialize the statistical quality calculator.

        Args:
            context_or_conn: CalculatorContext or database connection
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

        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.logger = logging.getLogger(__name__)

        # Date range (matching with training period)
        self.start_date = '2014-01-01'
        self.end_date = '2023-01-01'

        # Set up stat tables
        self.stat_tables = self.context.feature_utils.get_stat_tables()

        # Add patterns
        for pattern in include_patterns:
            self.add_include_pattern(pattern)
        for pattern in exclude_patterns:
            self.add_exclude_pattern(pattern)

    def _get_adjperf_columns(self, table_name: str) -> List[str]:
        """
        Get columns that should receive adjperf treatment.

        Based on AdjPerfCalculator._is_adjperf_target logic.

        Args:
            table_name: Feature table name

        Returns:
            List of column names that get adjperf
        """
        # Get all columns from the table
        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'features' AND table_name = '{table_name}'
        ORDER BY ordinal_position
        """
        columns = [row[0] for row in self.conn.execute(text(query)).fetchall()]

        adjperf_columns = []

        for col in columns:
            # Skip ID and total columns
            if col in ['fight_id', 'fighter_id', 'event_id']:
                continue
            if col.endswith('_total'):
                continue

            # Include stats that get adjperf
            if (col.endswith('_per_min') or
                col.endswith('_acc') or
                col.endswith('_def') or
                col.endswith('_ratio') or
                col.endswith('_pressure') or
                col in {
                    'sub_att_per_ctrl', 'ground_land_per_ctrl', 'rev_per_ctrlopp',
                    'sub_per_all_ctrl', 'ko_per_sig_str_land', 'sig_str_per_str_att',
                    'distance_per_sig_str_land', 'clinch_per_sig_str_land',
                    'ground_per_sig_str_land', 'head_per_sig_str_land',
                    'body_leg_per_sig_str_land', 'td_per_sig_str_att',
                    'ground_land_per_td_land', 'td_land_per_ctrl',
                    'ko_sub_per_win', 'ko_sub_rd1_per_win',
                    'win', 'decision', 'time_sec'
                }):

                if self.should_process_column(col):
                    adjperf_columns.append(col)

        return adjperf_columns

    def _calculate_quality_metrics_for_stat(
        self,
        table_name: str,
        stat_name: str,
        weightclass: str
    ) -> Dict[str, Any]:
        """
        Calculate quality metrics for a single stat in a single weightclass.

        Args:
            table_name: Feature table name
            stat_name: Stat column name
            weightclass: Weight class

        Returns:
            Dictionary with quality metrics or None if insufficient data
        """
        try:
            # Query to calculate quality metrics
            query = text(f"""
            WITH stat_values AS (
                SELECT
                    t.{stat_name} as value,
                    COUNT(*) OVER () as total_count
                FROM features.{table_name} t
                JOIN features.fight_mapping fm ON t.fight_id = fm.fight_id
                JOIN features.event_mapping em ON t.event_id = em.event_id
                WHERE fm.weightclass = :weightclass
                  AND em.event_date BETWEEN :start_date AND :end_date
                  AND t.{stat_name} IS NOT NULL
            ),
            percentiles AS (
                SELECT
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY value) as median_val,
                    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY value) as p05,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95
                FROM stat_values
            ),
            mad_calc AS (
                SELECT
                    PERCENTILE_CONT(0.50) WITHIN GROUP (
                        ORDER BY ABS(sv.value - p.median_val)
                    ) as mad_value
                FROM stat_values sv
                CROSS JOIN percentiles p
            ),
            mode_calc AS (
                SELECT
                    MODE() WITHIN GROUP (ORDER BY value) as mode_value
                FROM stat_values
            ),
            mode_count AS (
                SELECT
                    COUNT(*)::float as mode_freq_count
                FROM stat_values sv
                CROSS JOIN mode_calc mc
                WHERE ABS(sv.value - mc.mode_value) < 0.0001
            )
            SELECT
                COUNT(DISTINCT sv.value) as unique_values,
                COUNT(*) as total_values,
                p.median_val,
                p.p05,
                p.p95,
                m.mad_value,
                -- Calculate % of values within 0.001 of median (at prior)
                SUM(CASE WHEN ABS(sv.value - p.median_val) < 0.001 THEN 1 ELSE 0 END)::float /
                    COUNT(*) as pct_at_median,
                -- Mode information
                mc.mode_value,
                mf.mode_freq_count / COUNT(*) as mode_frequency
            FROM stat_values sv
            CROSS JOIN percentiles p
            CROSS JOIN mad_calc m
            CROSS JOIN mode_calc mc
            CROSS JOIN mode_count mf
            GROUP BY p.median_val, p.p05, p.p95, m.mad_value, mc.mode_value, mf.mode_freq_count
            """)

            result = self.conn.execute(
                query,
                {
                    'weightclass': weightclass,
                    'start_date': self.start_date,
                    'end_date': self.end_date
                }
            ).fetchone()

            if result is None or result[1] == 0:  # total_values = 0
                return None

            # Get wc_mad from existing weightclass_mad table if it exists
            try:
                wc_mad_query = text(f"""
                SELECT {stat_name}_wc_mad
                FROM features.{table_name}_wc_mad
                WHERE weightclass = :weightclass
                """)
                wc_mad_result = self.conn.execute(
                    wc_mad_query,
                    {'weightclass': weightclass}
                ).fetchone()
                wc_mad = float(wc_mad_result[0]) if wc_mad_result else None
            except Exception:
                wc_mad = None

            # Extract values
            unique_values = int(result[0])
            total_values = int(result[1])
            median_val = float(result[2]) if result[2] is not None else None
            p05 = float(result[3]) if result[3] is not None else None
            p95 = float(result[4]) if result[4] is not None else None
            mad_value = float(result[5]) if result[5] is not None else 0.0
            pct_at_median = float(result[6]) if result[6] is not None else 0.0
            mode_value = float(result[7]) if result[7] is not None else None
            mode_frequency = float(result[8]) if result[8] is not None else 0.0

            # Use wc_mad if available, otherwise use calculated mad_value
            final_mad = wc_mad if wc_mad is not None else mad_value

            # Classify quality using config
            quality_tier = StatQualityConfig.classify_distribution_quality(
                wc_mad=final_mad,
                mode_frequency=mode_frequency,
                unique_count=unique_values
            )

            # Calculate recommended winsorization limit
            recommended_winsor_limit = StatQualityConfig.calculate_winsorization_limit(
                quality_tier=quality_tier,
                wc_mad=final_mad
            )

            # Get reliability score
            reliability_score = StatQualityConfig.get_reliability_score(quality_tier)

            # Calculate effective sample size (penalize for degeneracy)
            if quality_tier == 'degenerate':
                effective_n = int(unique_values * 0.3)  # Only 30% of values are informative
            elif quality_tier == 'sparse':
                effective_n = int(unique_values * 0.6)  # 60% of values are informative
            else:
                effective_n = unique_values  # All unique values are informative

            return {
                'stat_name': stat_name,
                'weightclass': weightclass,
                'unique_values': unique_values,
                'total_values': total_values,
                'median_val': median_val,
                'p05': p05,
                'p95': p95,
                'wc_mad': final_mad,
                'pct_at_median': pct_at_median,
                'mode_value': mode_value,
                'mode_frequency': mode_frequency,
                'quality_tier': quality_tier,
                'recommended_winsor_limit': recommended_winsor_limit,
                'reliability_score': reliability_score,
                'effective_n': effective_n
            }

        except Exception as e:
            self.logger.warning(f"Error calculating quality for {table_name}.{stat_name} ({weightclass}): {e}")
            return None

    def _create_quality_metrics_table(self, table_name: str) -> None:
        """
        Create quality metrics table for a feature table.

        Args:
            table_name: Feature table name
        """
        try:
            quality_table_name = f"{table_name}_quality_metrics"

            # Drop existing table if it exists
            drop_sql = text(f"DROP TABLE IF EXISTS features.{quality_table_name} CASCADE;")
            self.conn.execute(drop_sql)
            self.conn.commit()

            # Get adjperf columns for this table
            adjperf_columns = self._get_adjperf_columns(table_name)

            if not adjperf_columns:
                self.logger.warning(f"No adjperf columns found in {table_name}")
                return

            # Get weight classes
            wc_query = text("SELECT DISTINCT weightclass FROM features.fight_mapping ORDER BY weightclass")
            weight_classes = [row[0] for row in self.conn.execute(wc_query)]

            # Collect all metrics
            all_metrics = []

            for stat_name in adjperf_columns:
                for weightclass in weight_classes:
                    metrics = self._calculate_quality_metrics_for_stat(
                        table_name, stat_name, weightclass
                    )
                    if metrics:
                        all_metrics.append(metrics)

            if not all_metrics:
                self.logger.warning(f"No quality metrics computed for {table_name}")
                return

            # Convert to DataFrame
            df = pd.DataFrame(all_metrics)

            # Write to database
            df.to_sql(
                quality_table_name,
                self.conn,
                schema='features',
                if_exists='replace',
                index=False,
                method='multi'
            )

            # Create index on (stat_name, weightclass)
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{quality_table_name}_stat_wc
            ON features.{quality_table_name}(stat_name, weightclass);
            """
            self.conn.execute(text(index_sql))
            self.conn.commit()

            # Log summary
            summary = df.groupby('quality_tier').size().to_dict()
            self.logger.info(
                f"Created {quality_table_name}: {len(df)} metrics "
                f"(degenerate={summary.get('degenerate', 0)}, "
                f"sparse={summary.get('sparse', 0)}, "
                f"healthy={summary.get('healthy', 0)})"
            )

        except Exception as e:
            self.logger.error(f"Error creating quality metrics table for {table_name}: {str(e)}")
            self.conn.rollback()
            raise

    def _validate_quality_metrics(self, table_name: str) -> None:
        """
        Validate the computed quality metrics.

        Args:
            table_name: Feature table name
        """
        try:
            quality_table_name = f"{table_name}_quality_metrics"

            # Query the computed metrics
            validation_sql = f"""
            SELECT * FROM features.{quality_table_name}
            """
            metrics_df = pd.read_sql(validation_sql, self.conn)

            # Validation checks
            if metrics_df.empty:
                self.logger.warning(f"No quality metrics computed for {table_name}")
                return

            # Check for expected weight classes
            wc_count = metrics_df['weightclass'].nunique()
            if wc_count < 8:
                self.logger.warning(
                    f"Missing weightclasses in {quality_table_name} (found {wc_count} of 9)"
                )

            # Check for NULL values in critical columns
            critical_cols = ['wc_mad', 'quality_tier', 'recommended_winsor_limit', 'reliability_score']
            for col in critical_cols:
                if metrics_df[col].isnull().any():
                    self.logger.warning(f"NULL values found in {quality_table_name}.{col}")

            # Check quality tier distribution
            tier_counts = metrics_df['quality_tier'].value_counts()
            self.logger.info(
                f"Validated {quality_table_name}: {len(metrics_df)} rows, "
                f"tiers={tier_counts.to_dict()}"
            )

        except Exception as e:
            self.logger.error(f"Error validating quality metrics for {table_name}: {str(e)}")

    def calculate_quality_metrics_for_all_tables(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate quality metrics for all feature tables.

        Returns:
            Dictionary of quality metrics DataFrames by table
        """
        results = {}
        tables_to_process = list(self.stat_tables.keys())
        total_tables = len(tables_to_process)

        self.logger.info(f"Starting quality metrics calculation for {total_tables} tables")
        print(f"\n=== Starting quality metrics calculation for {total_tables} tables ===")

        for i, table_name in enumerate(tables_to_process, 1):
            try:
                self.logger.info(f"[{i}/{total_tables}] Computing quality metrics for {table_name}")
                print(f"  └─ [{i}/{total_tables}] Processing {table_name}")

                # Create the quality metrics table
                self._create_quality_metrics_table(table_name)

                # Validate the results
                self._validate_quality_metrics(table_name)

                # Store computed results
                try:
                    quality_table_name = f"{table_name}_quality_metrics"
                    metrics_df = pd.read_sql(
                        f"SELECT * FROM features.{quality_table_name}",
                        self.conn
                    )
                    results[table_name] = metrics_df
                    print(f"  └─ ✓ Completed {table_name}: {len(metrics_df)} metrics")
                except Exception as e:
                    self.logger.warning(f"Could not load metrics for {table_name}: {str(e)}")
                    print(f"  └─ ✗ Could not load metrics for {table_name}")

            except Exception as e:
                self.logger.error(f"Error processing {table_name}: {str(e)}")
                print(f"  └─ ✗ Error processing {table_name}: {str(e)}")
                # Continue with other tables

        print(f"=== Quality metrics calculation completed ===\n")
        self.logger.info(f"Completed quality metrics calculation for {len(results)} tables")
        return results

    def calculate_for_table(self, table: str, columns: Optional[List[str]] = None) -> str:
        """Not used for this calculator as we handle all tables at once."""
        return ""

    def execute_for_table(self, table_name: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Not used for this calculator as we handle all tables at once."""
        return pd.DataFrame()

    def run(
        self,
        parallel: bool = False,
        max_workers: int = 4,
        table_pattern: str = ""
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the quality metrics calculator for all tables.

        Args:
            parallel: Whether to run in parallel (not used)
            max_workers: Number of workers for parallel execution (not used)
            table_pattern: Optional pattern to filter tables

        Returns:
            Dictionary of quality metrics by table
        """
        return self.calculate_quality_metrics_for_all_tables()
