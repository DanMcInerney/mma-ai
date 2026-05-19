import pandas as pd
import numpy as np
import logging
from typing import List, Set, Dict, Optional, Any
from sqlalchemy import text
from libs.feature_store.base_calculator import BaseCalculator
from libs.feature_store.calculator_context import CalculatorContext

class StyleCalculator(BaseCalculator):
    """
    Calculator for computing fighter style metrics based on adjusted performance statistics.
    
    This calculator computes various style-related metrics that characterize a fighter's
    fighting style, including power vs volume, grappling preferences, defensive patterns, etc.
    All metrics are derived from existing dec_adjperf_dec_avg statistics.
    """

    def __init__(self, conn_or_context, calculator_type='single_table'):
        """
        Initialize the style calculator.
        
        Args:
            conn_or_context: CalculatorContext or database connection
            calculator_type: Type of calculator ('single_table' for style table)
        """
        # Handle context initialization
        if isinstance(conn_or_context, CalculatorContext):
            self.context = conn_or_context
            super().__init__(conn_or_context.connection, calculator_type)
        else:
            self.context = CalculatorContext(conn_or_context)
            super().__init__(conn_or_context, calculator_type)
            
        self.table_name = 'style'
        self.schema = 'features'
        self.feature_type = 'style'
        self.logger = logging.getLogger(__name__)
        
        # No execution plan needed - we'll call methods directly

    def get_style_definitions(self) -> Dict[str, Dict[str, str]]:
        """
        Style metrics for clustering.

        IMPORTANT PRINCIPLES
        --------------------
        1) Never divide z-scores by z-scores.
        - Any composite / ratio should be computed from RAW decayed rates (e.g., *_dec_avg or *_per_min_dec_avg),
            then (optionally) passed through your adjperf pipeline to produce a z-scored variant for clustering.

        2) Avoid your dominance *_ratio_* fields for style (you defined ratio = (f1+f2)/f1).
        - When you need a "share", compute it from per-minute numerators over sig_str_land_per_min_dec_avg (raw).

        3) Round-1 bias must respect actual seconds fought in R1.
        - Use time-aware formulas so a 90s R1 finish doesn't look uniquely "hyper-aggressive".

        HOW TO USE
        ----------
        - Keys ending in *_raw are intended to be computed first in raw decayed space, then you can run them through
        your existing adjperf transform to obtain *_adjperf_dec_avg variants for clustering.
        - Direct adjperf features (e.g., *_dec_adjperf_dec_avg) can be used as-is.

        Assumed available columns (fighter-centric, decayed):
        - RAW decayed per-minute/total: e.g., sig_str_land_per_min_dec_avg, head_land_per_min_dec_avg, td_att_per_min_dec_avg,
            td_land_total_dec_avg, ctrl_total_dec_avg, sub_land_per_min_dec_avg, decision_per_min_dec_avg, ko_per_min_dec_avg, kd_per_min_dec_avg,
            strikes_land_rd1_per_min_dec_avg, sig_str_land_rd1_per_min_dec_avg, td_att_rd1_per_min_dec_avg,
            distance_land_per_min_dec_avg, clinch_land_per_min_dec_avg, ground_land_per_min_dec_avg,
            time_sec_dec_avg, time_sec_rd1_dec_avg (optional), scheduled_time_sec (optional), fm_rounds (3/5 fallback)
        - ADJPERF (z-scored, opponent-adjusted): e.g., sig_str_land_per_min_dec_adjperf_dec_avg, sig_str_acc_dec_adjperf_dec_avg,
            head_acc_dec_adjperf_dec_avg, ... td_acc_dec_adjperf_dec_avg, ctrl_per_min_dec_adjperf_dec_avg, ctrl_ratio_dec_adjperf_dec_avg, etc.
        - OPPONENT (what opponents do vs this fighter): e.g., sig_str_land_per_min_opp_dec_avg, sig_str_acc_opp_dec_avg,
            td_att_per_min_opp_dec_avg, td_acc_opp_dec_avg, ctrl_per_min_opp_dec_avg, kd_per_min_opp_dec_avg

        Return format:
        { metric_name: { 'expression': <SQL string>, 'description': <text> }, ... }
        """
        return {
            # =========================================================
            # A) STRIKING APPROACH (direct adjperf features)
            # =========================================================
            'style_sig_str_volume': {
                'expression': 'COALESCE(sig_str_land_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Significant striking volume per minute (adjperf z-score)'
            },
            'style_sig_str_accuracy': {
                'expression': 'COALESCE(sig_str_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Significant striking accuracy (adjperf z-score)'
            },
            'style_head_accuracy': {
                'expression': 'COALESCE(head_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Head striking accuracy (adjperf z-score)'
            },
            'style_body_accuracy': {
                'expression': 'COALESCE(body_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Body striking accuracy (adjperf z-score)'
            },
            'style_leg_accuracy': {
                'expression': 'COALESCE(leg_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Leg striking accuracy (adjperf z-score)'
            },

            # =========================================================
            # B) TARGETING SHARES (compute from RAW per-minute; then adjperf)
            # =========================================================
            'style_head_target_share_raw': {
                'expression': 'COALESCE(head_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg, 0), 0)',
                'description': 'Share of sig strikes to head (raw decayed); then adjperf'
            },
            'style_body_target_share_raw': {
                'expression': 'COALESCE(body_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg, 0), 0)',
                'description': 'Share of sig strikes to body (raw decayed); then adjperf'
            },
            'style_leg_target_share_raw': {
                'expression': 'COALESCE(leg_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg, 0), 0)',
                'description': 'Share of sig strikes to legs (raw decayed); then adjperf'
            },

            # =========================================================
            # C) POWER & DAMAGE ORIENTATION
            # =========================================================
            'style_big_hits_per_strike': {
                'expression': '(COALESCE(kd_per_min_dec_avg, 0) + (COALESCE(ko_per_min_dec_avg, 0) * 2.0)) / NULLIF(sig_str_land_per_min_dec_avg, 0)',
                'description': 'Big hits per strike (raw decayed)'
            },
            # Power vs Volume MUST be computed in raw space (no z/z)
            'style_power_vs_volume_raw': {
                'expression': 'COALESCE(strikes_land_per_min_dec_avg,0) / COALESCE(sig_str_land_per_min_dec_avg,0',
                'description': 'Power per strike volume (raw decayed); then adjperf'
            },

            # =========================================================
            # D) RANGE / PHASE PREFERENCES (true shares from RAW per-minute)
            # =========================================================
            'style_distance_share_raw': {
                'expression': 'COALESCE(distance_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg,0), 0)',
                'description': 'Share of sig strikes at distance (raw decayed); then adjperf'
            },
            'style_clinch_share_raw': {
                'expression': 'COALESCE(clinch_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg,0), 0)',
                'description': 'Share of sig strikes in clinch (raw decayed); then adjperf'
            },
            'style_ground_share_raw': {
                'expression': 'COALESCE(ground_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg,0), 0)',
                'description': 'Share of sig strikes on ground (raw decayed); then adjperf'
            },
            'style_distance_accuracy': {
                'expression': 'COALESCE(distance_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Distance striking accuracy (adjperf z-score)'
            },
            'style_clinch_accuracy': {
                'expression': 'COALESCE(clinch_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Clinch striking accuracy (adjperf z-score)'
            },
            'style_ground_accuracy': {
                'expression': 'COALESCE(ground_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Ground striking accuracy (adjperf z-score)'
            },
            # Range entropy (compute from the 3 raw shares above; then adjperf if desired)
            'style_range_entropy_raw': {
                'expression': """
                    CASE
                    WHEN COALESCE(sig_str_land_per_min_dec_avg,0) = 0 THEN 0
                    ELSE (
                        WITH shares AS (
                        SELECT
                            COALESCE(distance_land_per_min_dec_avg / NULLIF(sig_str_land_per_min_dec_avg,0),0) AS d,
                            COALESCE(clinch_land_per_min_dec_avg   / NULLIF(sig_str_land_per_min_dec_avg,0),0) AS c,
                            COALESCE(ground_land_per_min_dec_avg   / NULLIF(sig_str_land_per_min_dec_avg,0),0) AS g
                        ),
                        normed AS (
                        SELECT d, c, g, (d+c+g) AS z FROM shares
                        )
                        SELECT
                        CASE
                            WHEN z = 0 THEN 0
                            ELSE -(
                            (d/z)*LN(NULLIF(d/z,0)) +
                            (c/z)*LN(NULLIF(c/z,0)) +
                            (g/z)*LN(NULLIF(g/z,0))
                            ) / LN(3)
                        END
                        FROM normed
                    )
                    END
                """.strip(),
                'description': 'Normalized entropy of distance/clinch/ground shares (raw); then adjperf'
            },

            # =========================================================
            # E) PACE & ROUND-1 DYNAMICS (time-aware)
            # =========================================================
            'style_td_pace': {
                'expression': 'COALESCE(td_att_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Takedown attempt pace (adjperf z-score)'
            },
            # Time-aware R1 striking share (raw -> adjperf)
            'style_rd1_striking_bias_raw': {
                'expression': """
                    COALESCE(
                    (
                        -- R1 strikes ≈ per_min * actual R1 seconds / 60
                        (COALESCE(sig_str_land_rd1_per_min_dec_avg,0)
                        * LEAST(COALESCE(time_sec_rd1_dec_avg, time_sec_dec_avg, 0), 300) / 60.0)
                    )
                    /
                    NULLIF(
                        -- Total strikes ≈ per_min * total secs / 60
                        (COALESCE(sig_str_land_per_min_dec_avg,0) * COALESCE(time_sec_dec_avg,0) / 60.0),
                        0
                    ),
                    0
                    )
                """.strip(),
                'description': 'Fast-starter striking: share of total striking occurring in R1 (raw); then adjperf'
            },
            # Time-aware R1 wrestling share (raw -> adjperf)
            'style_rd1_wrestle_bias_raw': {
                'expression': """
                    COALESCE(
                    (
                        (COALESCE(td_att_rd1_per_min_dec_avg,0)
                        * LEAST(COALESCE(time_sec_rd1_dec_avg, time_sec_dec_avg, 0), 300) / 60.0)
                    )
                    /
                    NULLIF(
                        (COALESCE(td_att_per_min_dec_avg,0) * COALESCE(time_sec_dec_avg,0) / 60.0),
                        0
                    ),
                    0
                    )
                """.strip(),
                'description': 'Fast-starter wrestling: share of total TD attempts occurring in R1 (raw); then adjperf'
            },

            # =========================================================
            # F) WRESTLING / TOP CONTROL (direct + derived)
            # =========================================================
            'style_td_accuracy': {
                'expression': 'COALESCE(td_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Takedown accuracy (adjperf z-score)'
            },
            'style_td_volume': {
                'expression': 'COALESCE(td_land_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Takedown land volume per minute (adjperf z-score)'
            },
            'style_control_time': {
                'expression': 'COALESCE(ctrl_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Control time per minute (adjperf z-score)'
            },
            'style_control_share': {
                'expression': 'COALESCE(ctrl_ratio_dec_adjperf_dec_avg, 0)',
                'description': 'Control share/dominance (adjperf z-score)'
            },
            'style_ground_volume': {
                'expression': 'COALESCE(ground_land_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Ground striking volume per minute (adjperf z-score)'
            },
            'style_ground_accuracy': {
                'expression': 'COALESCE(ground_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Ground striking accuracy (adjperf z-score)'
            },
            # Control conversion quality (RAW totals -> then adjperf)
            'style_ctrl_gain_per_td_raw': {
                'expression': 'COALESCE(ctrl_dec_avg / NULLIF(td_land_dec_avg, 0), 0)',
                'description': 'Control time gained per takedown landed (raw decayed totals); then adjperf'
            },

            # =========================================================
            # 🔄 CONTROL & WRESTLING EFFICIENCY (5 derived features)
            # =========================================================
            'style_ctrl_conversion_raw': {
                'expression': 'COALESCE(ctrl_total_dec_avg / NULLIF(td_land_total_dec_avg, 0), 0)',
                'description': 'Control time gained per takedown landed (raw decayed); then adjperf'
            },
            'style_td_per_ctrl_raw': {
                'expression': 'COALESCE(td_land_per_min_dec_avg / NULLIF(ctrl_per_min_dec_avg, 0), 0)',
                'description': 'Takedown rate relative to control time (raw decayed); then adjperf'
            },
            'style_ctrl_per_sub_att_raw': {
                'expression': 'COALESCE(ctrl_per_min_dec_avg / NULLIF(sub_att_per_min_dec_avg, 0), 0)',
                'description': 'Control time vs submission attempts (raw decayed); then adjperf'
            },
            'style_sub_efficiency_raw': {
                'expression': 'COALESCE(sub_land_per_min_dec_avg / NULLIF(sub_att_per_min_dec_avg, 0), 0)',
                'description': 'Submission efficiency (raw decayed); then adjperf'
            },
            'style_ctrl_sub_conversion_raw': {
                'expression': 'COALESCE(sub_land_per_min_dec_avg / NULLIF(ctrl_per_min_dec_avg, 0), 0)',
                'description': 'Submission rate per control time (raw decayed); then adjperf'
            },

            # =========================================================
            # F2) DECISION TENDENCY 
            # =========================================================
            'style_decision_tendency': {
                'expression': 'COALESCE(decision_dec_adjperf_dec_avg, 0)',
                'description': 'Decision tendency - how often fights go to decision'
            },

            # =========================================================
            # G) BOTTOM GAME / EXITS (direct + derived)
            # =========================================================
            'style_reversal_rate': {
                'expression': 'COALESCE(rev_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Reversal rate per minute (adjperf z-score)'
            },
            'style_reversal_share': {
                'expression': 'COALESCE(rev_ratio_dec_adjperf_dec_avg, 0)',
                'description': 'Reversal share/dominance (adjperf z-score)'
            },
            'style_sub_attempts': {
                'expression': 'COALESCE(sub_att_per_min_dec_adjperf_dec_avg, 0)',
                'description': 'Submission attempt rate per minute (adjperf z-score)'
            },
            'style_sub_accuracy': {
                'expression': 'COALESCE(sub_acc_dec_adjperf_dec_avg, 0)',
                'description': 'Submission accuracy (adjperf z-score)'
            },
            # Escape activity vs opponent control (RAW -> then adjperf)
            'style_escape_activity_raw': {
                'expression': 'COALESCE(rev_per_min_dec_avg / NULLIF(ctrl_per_min_opp_dec_avg, 0), 0)',
                'description': 'Reversals per opponent control minute (raw); then adjperf'
            },

            # =========================================================
            # H) DEFENSIVE / PRESSURE HANDLING (opponent vs you)
            # =========================================================
            'style_sig_str_absorbed': {
                'expression': 'COALESCE(sig_str_land_per_min_opp_dec_avg, 0)',
                'description': 'Significant strikes absorbed per minute (opponent vs fighter)'
            },
            'style_opp_accuracy': {
                'expression': 'COALESCE(sig_str_acc_opp_dec_avg, 0)',
                'description': 'Opponent striking accuracy vs fighter'
            },
            'style_td_pressure': {
                'expression': 'COALESCE(td_att_per_min_opp_dec_avg, 0)',
                'description': 'Takedown attempts faced per minute'
            },
            'style_opp_td_accuracy': {
                'expression': 'COALESCE(td_acc_opp_dec_avg, 0)',
                'description': 'Opponent takedown accuracy vs fighter'
            },
            'style_control_absorbed': {
                'expression': 'COALESCE(ctrl_per_min_opp_dec_avg, 0)',
                'description': 'Control time absorbed per minute'
            },
            'style_kd_absorbed': {
                'expression': 'COALESCE(kd_per_min_opp_dec_avg, 0)',
                'description': 'Knockdowns absorbed per minute'
            },

            # =========================================================
            # I) FINISHING CHANNEL SIMPLEX (KO / SUB / DECISION)
            #    Compute in RAW space; then adjperf if desired
            # =========================================================
            'style_finish_total_pm_raw': {
                'expression': '''
                    COALESCE(ko_per_min_dec_avg,0)
                + COALESCE(sub_land_per_min_dec_avg,0)
                + COALESCE(decision_per_min_dec_avg,0)
                ''',
                'description': 'Total per-minute mass across KO/Sub/Decision (raw decayed)'
            },
            'style_finish_channel_ko_share_raw': {
                'expression': 'COALESCE(ko_per_min_dec_avg / NULLIF((COALESCE(ko_per_min_dec_avg,0)+COALESCE(sub_land_per_min_dec_avg,0)+COALESCE(decision_per_min_dec_avg,0)),0), 0)',
                'description': 'KO share of (KO+SUB+DEC) channels (raw); then adjperf'
            },
            'style_finish_channel_sub_share_raw': {
                'expression': 'COALESCE(sub_land_per_min_dec_avg / NULLIF((COALESCE(ko_per_min_dec_avg,0)+COALESCE(sub_land_per_min_dec_avg,0)+COALESCE(decision_per_min_dec_avg,0)),0), 0)',
                'description': 'Sub share of (KO+SUB+DEC) channels (raw); then adjperf'
            },
            'style_finish_channel_decision_share_raw': {
                'expression': 'COALESCE(decision_per_min_dec_avg / NULLIF((COALESCE(ko_per_min_dec_avg,0)+COALESCE(sub_land_per_min_dec_avg,0)+COALESCE(decision_per_min_dec_avg,0)),0), 0)',
                'description': 'Decision share of (KO+SUB+DEC) channels (raw); then adjperf'
            },
            'style_finish_channel_entropy_raw': {
                'expression': '''
                    WITH pm AS (
                    SELECT
                        COALESCE(ko_per_min_dec_avg,0)       AS ko,
                        COALESCE(sub_land_per_min_dec_avg,0) AS sub,
                        COALESCE(decision_per_min_dec_avg,0) AS deci
                    ),
                    shares AS (
                    SELECT
                        CASE WHEN (ko+sub+deci)=0 THEN 0 ELSE ko   /(ko+sub+deci) END AS pko,
                        CASE WHEN (ko+sub+deci)=0 THEN 0 ELSE sub  /(ko+sub+deci) END AS psub,
                        CASE WHEN (ko+sub+deci)=0 THEN 0 ELSE deci /(ko+sub+deci) END AS pdec
                    FROM pm
                    )
                    SELECT CASE
                            WHEN (pko+psub+pdec)=0 THEN 0
                            ELSE -( pko*LN(NULLIF(pko,0))
                                + psub*LN(NULLIF(psub,0))
                                + pdec*LN(NULLIF(pdec,0)) ) / LN(3)
                        END
                    FROM shares
                ''',
                'description': 'Entropy across KO/Sub/Decision shares (raw); then adjperf'
            },
            'style_finish_ko_vs_sub_bias_raw': {
                'expression': 'COALESCE(ko_per_min_dec_avg / NULLIF(ko_per_min_dec_avg + sub_land_per_min_dec_avg, 0), 0)',
                'description': 'KO vs SUB preference (raw); then adjperf'
            },
            'style_finisher_vs_decision_bias_raw': {
                'expression': 'COALESCE((COALESCE(ko_per_min_dec_avg,0)+COALESCE(sub_land_per_min_dec_avg,0)) / NULLIF(COALESCE(ko_per_min_dec_avg,0)+COALESCE(sub_land_per_min_dec_avg,0)+COALESCE(decision_per_min_dec_avg,0),0), 0)',
                'description': 'Finisher (KO+SUB) vs Decision preference (raw); then adjperf'
            },

            # =========================================================
            # J) EARLY-FINISH SEEKING (R1 activity; RAW -> adjperf)
            # =========================================================
            'style_early_finish_proxy_raw': {
                'expression': '''
                    COALESCE(sig_str_land_rd1_per_min_dec_avg,0)
                + COALESCE(td_att_rd1_per_min_dec_avg,0)
                + COALESCE(sub_att_rd1_per_min_dec_avg,0)
                ''',
                'description': 'R1 finish-seeking activity (raw)'
            },
        }

    def create_style_table(self):
        """Create the features.style table if it doesn't exist."""
        self.logger.info("Creating features.style table...")
        
        # Get all style metric names
        style_definitions = self.get_style_definitions()
        style_columns = ', '.join([f"{metric} FLOAT" for metric in style_definitions.keys()])
        
        # Drop existing table if it exists to ensure clean state
        drop_sql = f"DROP TABLE IF EXISTS {self.schema}.{self.table_name} CASCADE;"
        self.execute_raw_sql(drop_sql)
        
        # Create new table
        create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.{self.table_name} (
                fight_id INTEGER NOT NULL,
                fighter_id INTEGER NOT NULL,
                event_id INTEGER NOT NULL,
                {style_columns},
                PRIMARY KEY (fight_id, fighter_id),
                CONSTRAINT fk_fight FOREIGN KEY (fight_id) REFERENCES features.fight_mapping(fight_id),
                CONSTRAINT fk_fighter FOREIGN KEY (fighter_id) REFERENCES features.fighter_mapping(fighter_id),
                CONSTRAINT fk_event FOREIGN KEY (event_id) REFERENCES features.event_mapping(event_id)
            );
        """
        self.execute_raw_sql(create_table_sql)
        
        # Create indexes separately for better compatibility
        index_sqls = [
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fight ON {self.schema}.{self.table_name}(fight_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fighter ON {self.schema}.{self.table_name}(fighter_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_event ON {self.schema}.{self.table_name}(event_id);"
        ]
        for index_sql in index_sqls:
            self.execute_raw_sql(index_sql)
        self.logger.info(f"Created {self.schema}.{self.table_name} table successfully")

    def get_features(self) -> List[str]:
        """Get list of style features to calculate."""
        return list(self.get_style_definitions().keys())

    def calculate(self) -> pd.DataFrame:
        """Calculate all style metrics and return as DataFrame."""
        self.logger.info("Calculating style metrics...")
        
        # Due to complex CTE expressions, we'll calculate features in batches
        # First get base data, then calculate simple features, then complex ones
        
        try:
            # Step 1: Get base fighter-fight combinations with all needed raw data
            base_df = self._get_base_style_data()
            
            if base_df.empty:
                self.logger.warning("No base style data found")
                return pd.DataFrame()
            
            # Step 2: Calculate simple features directly
            simple_features_df = self._calculate_simple_features(base_df)
            
            # Step 3: Calculate complex features that need CTEs (pass simple features for range entropy)
            complex_features_df = self._calculate_complex_features(base_df, simple_features_df)
            
            # Step 4: Merge all features
            result_df = simple_features_df.merge(
                complex_features_df, 
                on=['fight_id', 'fighter_id', 'event_id'], 
                how='left'
            )
            
            self.logger.info(f"Calculated style metrics for {len(result_df)} records")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error calculating style metrics: {str(e)}")
            raise

    def _get_base_style_data(self) -> pd.DataFrame:
        """Get base data with all required columns for style calculations."""
        
        query = """
            WITH base_data AS (
                SELECT DISTINCT
                    fm.fight_id,
                    fm.fighter1_id as fighter_id,
                    fm.event_id
                FROM features.fight_mapping fm
                
                UNION
                
                SELECT DISTINCT
                    fm.fight_id,
                    fm.fighter2_id as fighter_id,
                    fm.event_id
                FROM features.fight_mapping fm
        )
            
                SELECT 
                    bd.fight_id,
                    bd.fighter_id,
                    bd.event_id,
                    
            -- RAW decayed features needed for calculations
            COALESCE(ss.sig_str_land_per_min_dec_avg, 0) as sig_str_land_per_min_dec_avg,
            COALESCE(ss1.sig_str_land_rd1_per_min_dec_avg, 0) as sig_str_land_rd1_per_min_dec_avg,
            COALESCE(h.head_land_per_min_dec_avg, 0) as head_land_per_min_dec_avg,
            COALESCE(b.body_land_per_min_dec_avg, 0) as body_land_per_min_dec_avg,
            COALESCE(l.leg_land_per_min_dec_avg, 0) as leg_land_per_min_dec_avg,
            COALESCE(d.distance_land_per_min_dec_avg, 0) as distance_land_per_min_dec_avg,
            COALESCE(c.clinch_land_per_min_dec_avg, 0) as clinch_land_per_min_dec_avg,
            COALESCE(g.ground_land_per_min_dec_avg, 0) as ground_land_per_min_dec_avg,
            COALESCE(kd.kd_per_min_dec_avg, 0) as kd_per_min_dec_avg,
            COALESCE(ko.ko_per_min_dec_avg, 0) as ko_per_min_dec_avg,
            COALESCE(sub.sub_land_per_min_dec_avg, 0) as sub_land_per_min_dec_avg,
            COALESCE(dec.decision_per_min_dec_avg, 0) as decision_per_min_dec_avg,
            COALESCE(td1.td_att_rd1_per_min_dec_avg, 0) as td_att_rd1_per_min_dec_avg,
            COALESCE(td.td_att_per_min_dec_avg, 0) as td_att_per_min_dec_avg,
            COALESCE(td.td_land_dec_avg, 0) as td_land_dec_avg,
            COALESCE(td.td_land_total_dec_avg, 0) as td_land_total_dec_avg,
            COALESCE(td.td_land_per_min_dec_avg, 0) as td_land_per_min_dec_avg,
            COALESCE(ctrl.ctrl_dec_avg, 0) as ctrl_dec_avg,
            COALESCE(ctrl.ctrl_total_dec_avg, 0) as ctrl_total_dec_avg,
            COALESCE(ctrl.ctrl_per_min_dec_avg, 0) as ctrl_per_min_dec_avg,
            COALESCE(sub.sub_att_per_min_dec_avg, 0) as sub_att_per_min_dec_avg,
            COALESCE(rev.rev_per_min_dec_avg, 0) as rev_per_min_dec_avg,
            COALESCE(st1.strikes_land_rd1_per_min_dec_avg, 0) as strikes_land_rd1_per_min_dec_avg,
            COALESCE(sub1.sub_att_rd1_per_min_dec_avg, 0) as sub_att_rd1_per_min_dec_avg,
            COALESCE(ts.time_sec_dec_avg, 0) as time_sec_dec_avg,
            COALESCE(ts1.time_sec_rd1_dec_avg, ts.time_sec_dec_avg, 0) as time_sec_rd1_dec_avg,
            
            -- ADJPERF features (direct use)
            COALESCE(ss.sig_str_land_per_min_dec_adjperf_dec_avg, 0) as sig_str_land_per_min_dec_adjperf_dec_avg,
                    COALESCE(ss.sig_str_acc_dec_adjperf_dec_avg, 0) as sig_str_acc_dec_adjperf_dec_avg,
            COALESCE(h.head_acc_dec_adjperf_dec_avg, 0) as head_acc_dec_adjperf_dec_avg,
            COALESCE(b.body_acc_dec_adjperf_dec_avg, 0) as body_acc_dec_adjperf_dec_avg,
            COALESCE(l.leg_acc_dec_adjperf_dec_avg, 0) as leg_acc_dec_adjperf_dec_avg,
            COALESCE(kd.kd_per_min_dec_adjperf_dec_avg, 0) as kd_per_min_dec_adjperf_dec_avg,
            COALESCE(ko.ko_per_min_dec_adjperf_dec_avg, 0) as ko_per_min_dec_adjperf_dec_avg,
            COALESCE(d.distance_acc_dec_adjperf_dec_avg, 0) as distance_acc_dec_adjperf_dec_avg,
            COALESCE(c.clinch_acc_dec_adjperf_dec_avg, 0) as clinch_acc_dec_adjperf_dec_avg,
            COALESCE(g.ground_acc_dec_adjperf_dec_avg, 0) as ground_acc_dec_adjperf_dec_avg,
                    COALESCE(td.td_att_per_min_dec_adjperf_dec_avg, 0) as td_att_per_min_dec_adjperf_dec_avg,
                    COALESCE(td.td_acc_dec_adjperf_dec_avg, 0) as td_acc_dec_adjperf_dec_avg,
            COALESCE(td.td_land_per_min_dec_adjperf_dec_avg, 0) as td_land_per_min_dec_adjperf_dec_avg,
                    COALESCE(ctrl.ctrl_per_min_dec_adjperf_dec_avg, 0) as ctrl_per_min_dec_adjperf_dec_avg,
            COALESCE(ctrl.ctrl_ratio_dec_adjperf_dec_avg, 0) as ctrl_ratio_dec_adjperf_dec_avg,
            COALESCE(g.ground_land_per_min_dec_adjperf_dec_avg, 0) as ground_land_per_min_dec_adjperf_dec_avg,
            COALESCE(rev.rev_per_min_dec_adjperf_dec_avg, 0) as rev_per_min_dec_adjperf_dec_avg,
            COALESCE(rev.rev_ratio_dec_adjperf_dec_avg, 0) as rev_ratio_dec_adjperf_dec_avg,
                    COALESCE(sub.sub_att_per_min_dec_adjperf_dec_avg, 0) as sub_att_per_min_dec_adjperf_dec_avg,
            COALESCE(sub.sub_acc_dec_adjperf_dec_avg, 0) as sub_acc_dec_adjperf_dec_avg,
                    COALESCE(sub.sub_land_per_min_dec_adjperf_dec_avg, 0) as sub_land_per_min_dec_adjperf_dec_avg,
            
            -- Decision feature
            COALESCE(dec.decision_dec_adjperf_dec_avg, 0) as decision_dec_adjperf_dec_avg,
            
            -- Additional control features for derived calculations
            COALESCE(ctrl.ctrl_total_dec_adjperf_dec_avg, 0) as ctrl_total_dec_adjperf_dec_avg,
            COALESCE(td.td_land_total_dec_adjperf_dec_avg, 0) as td_land_total_dec_adjperf_dec_avg,
            
            -- OPPONENT features
            COALESCE(ss.sig_str_land_per_min_opp_dec_avg, 0) as sig_str_land_per_min_opp_dec_avg,
            COALESCE(ss.sig_str_acc_opp_dec_avg, 0) as sig_str_acc_opp_dec_avg,
            COALESCE(td.td_att_per_min_opp_dec_avg, 0) as td_att_per_min_opp_dec_avg,
            COALESCE(td.td_acc_opp_dec_avg, 0) as td_acc_opp_dec_avg,
            COALESCE(ctrl.ctrl_per_min_opp_dec_avg, 0) as ctrl_per_min_opp_dec_avg,
            COALESCE(kd.kd_per_min_opp_dec_avg, 0) as kd_per_min_opp_dec_avg
                    
                FROM base_data bd
                LEFT JOIN features.sig_str ss ON bd.fight_id = ss.fight_id AND bd.fighter_id = ss.fighter_id
                LEFT JOIN features.sig_str_rd1 ss1 ON bd.fight_id = ss1.fight_id AND bd.fighter_id = ss1.fighter_id
                LEFT JOIN features.head h ON bd.fight_id = h.fight_id AND bd.fighter_id = h.fighter_id
                LEFT JOIN features.body b ON bd.fight_id = b.fight_id AND bd.fighter_id = b.fighter_id
                LEFT JOIN features.leg l ON bd.fight_id = l.fight_id AND bd.fighter_id = l.fighter_id
                LEFT JOIN features.distance d ON bd.fight_id = d.fight_id AND bd.fighter_id = d.fighter_id
                LEFT JOIN features.clinch c ON bd.fight_id = c.fight_id AND bd.fighter_id = c.fighter_id
                LEFT JOIN features.ground g ON bd.fight_id = g.fight_id AND bd.fighter_id = g.fighter_id
                LEFT JOIN features.td td ON bd.fight_id = td.fight_id AND bd.fighter_id = td.fighter_id
                LEFT JOIN features.td_rd1 td1 ON bd.fight_id = td1.fight_id AND bd.fighter_id = td1.fighter_id
                LEFT JOIN features.ctrl ctrl ON bd.fight_id = ctrl.fight_id AND bd.fighter_id = ctrl.fighter_id
                LEFT JOIN features.sub sub ON bd.fight_id = sub.fight_id AND bd.fighter_id = sub.fighter_id
        LEFT JOIN features.sub_rd1 sub1 ON bd.fight_id = sub1.fight_id AND bd.fighter_id = sub1.fighter_id
                LEFT JOIN features.ko ko ON bd.fight_id = ko.fight_id AND bd.fighter_id = ko.fighter_id
                LEFT JOIN features.kd kd ON bd.fight_id = kd.fight_id AND bd.fighter_id = kd.fighter_id
        LEFT JOIN features.rev rev ON bd.fight_id = rev.fight_id AND bd.fighter_id = rev.fighter_id
        LEFT JOIN features.strikes st ON bd.fight_id = st.fight_id AND bd.fighter_id = st.fighter_id
        LEFT JOIN features.strikes_rd1 st1 ON bd.fight_id = st1.fight_id AND bd.fighter_id = st1.fighter_id
        LEFT JOIN features.time_sec ts ON bd.fight_id = ts.fight_id AND bd.fighter_id = ts.fighter_id
        LEFT JOIN features.time_sec_rd1 ts1 ON bd.fight_id = ts1.fight_id AND bd.fighter_id = ts1.fighter_id
        LEFT JOIN features.decision dec ON bd.fight_id = dec.fight_id AND bd.fighter_id = dec.fighter_id
        
        ORDER BY bd.fight_id, bd.fighter_id
        """
        
        return self.execute_raw_sql(query, return_results=True)

    def _calculate_simple_features(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple features that don't need CTEs."""
        
        result_df = base_df[['fight_id', 'fighter_id', 'event_id']].copy()
        
        # Simple adjperf features (direct)
        result_df['style_sig_str_volume'] = base_df['sig_str_land_per_min_dec_adjperf_dec_avg']
        result_df['style_sig_str_accuracy'] = base_df['sig_str_acc_dec_adjperf_dec_avg']
        result_df['style_head_accuracy'] = base_df['head_acc_dec_adjperf_dec_avg']
        result_df['style_body_accuracy'] = base_df['body_acc_dec_adjperf_dec_avg']
        result_df['style_leg_accuracy'] = base_df['leg_acc_dec_adjperf_dec_avg']
        result_df['style_kd_rate'] = base_df['kd_per_min_dec_adjperf_dec_avg']
        result_df['style_ko_rate'] = base_df['ko_per_min_dec_adjperf_dec_avg']
        result_df['style_distance_accuracy'] = base_df['distance_acc_dec_adjperf_dec_avg']
        result_df['style_clinch_accuracy'] = base_df['clinch_acc_dec_adjperf_dec_avg']
        result_df['style_ground_accuracy'] = base_df['ground_acc_dec_adjperf_dec_avg']
        result_df['style_td_pace'] = base_df['td_att_per_min_dec_adjperf_dec_avg']
        result_df['style_td_accuracy'] = base_df['td_acc_dec_adjperf_dec_avg']
        result_df['style_td_volume'] = base_df['td_land_per_min_dec_adjperf_dec_avg']
        result_df['style_control_time'] = base_df['ctrl_per_min_dec_adjperf_dec_avg']
        result_df['style_control_share'] = base_df['ctrl_ratio_dec_adjperf_dec_avg']
        result_df['style_ground_volume'] = base_df['ground_land_per_min_dec_adjperf_dec_avg']
        result_df['style_reversal_rate'] = base_df['rev_per_min_dec_adjperf_dec_avg']
        result_df['style_reversal_share'] = base_df['rev_ratio_dec_adjperf_dec_avg']
        result_df['style_sub_attempts'] = base_df['sub_att_per_min_dec_adjperf_dec_avg']
        result_df['style_sub_accuracy'] = base_df['sub_acc_dec_adjperf_dec_avg']
        
        # Decision tendency
        result_df['style_decision_tendency'] = base_df['decision_dec_adjperf_dec_avg']
        
        # Additional control/wrestling efficiency features (using raw dec_avg values)
        result_df['style_ctrl_conversion_raw'] = base_df['ctrl_total_dec_avg'] / base_df['td_land_total_dec_avg'].replace(0, np.nan)
        result_df['style_td_per_ctrl_raw'] = base_df['td_land_per_min_dec_avg'] / base_df['ctrl_per_min_dec_avg'].replace(0, np.nan)
        result_df['style_ctrl_per_sub_att_raw'] = base_df['ctrl_per_min_dec_avg'] / base_df['sub_att_per_min_dec_avg'].replace(0, np.nan)
        result_df['style_sub_efficiency_raw'] = base_df['sub_land_per_min_dec_avg'] / base_df['sub_att_per_min_dec_avg'].replace(0, np.nan)
        result_df['style_ctrl_sub_conversion_raw'] = base_df['sub_land_per_min_dec_avg'] / base_df['ctrl_per_min_dec_avg'].replace(0, np.nan)
        
        # Opponent features (direct)
        result_df['style_sig_str_absorbed'] = base_df['sig_str_land_per_min_opp_dec_avg']
        result_df['style_opp_accuracy'] = base_df['sig_str_acc_opp_dec_avg']
        result_df['style_td_pressure'] = base_df['td_att_per_min_opp_dec_avg']
        result_df['style_opp_td_accuracy'] = base_df['td_acc_opp_dec_avg']
        result_df['style_control_absorbed'] = base_df['ctrl_per_min_opp_dec_avg']
        result_df['style_kd_absorbed'] = base_df['kd_per_min_opp_dec_avg']
        
        # Simple raw feature calculations
        sig_str_denom = base_df['sig_str_land_per_min_dec_avg'].replace(0, np.nan)
        result_df['style_head_target_share_raw'] = base_df['head_land_per_min_dec_avg'] / sig_str_denom
        result_df['style_body_target_share_raw'] = base_df['body_land_per_min_dec_avg'] / sig_str_denom  
        result_df['style_leg_target_share_raw'] = base_df['leg_land_per_min_dec_avg'] / sig_str_denom
        result_df['style_distance_share_raw'] = base_df['distance_land_per_min_dec_avg'] / sig_str_denom
        result_df['style_clinch_share_raw'] = base_df['clinch_land_per_min_dec_avg'] / sig_str_denom
        result_df['style_ground_share_raw'] = base_df['ground_land_per_min_dec_avg'] / sig_str_denom
        
        # Power vs volume (raw)
        result_df['style_power_vs_volume_raw'] = (base_df['kd_per_min_dec_avg'] + base_df['ko_per_min_dec_avg']) / sig_str_denom
        
        # Control conversion (raw) - use totals to reduce per-minute noise
        td_total = base_df['td_land_total_dec_avg'].replace(0, np.nan)
        result_df['style_ctrl_gain_per_td_raw'] = base_df['ctrl_total_dec_avg'] / td_total
        
        # Escape activity (raw)
        ctrl_opp_denom = base_df['ctrl_per_min_opp_dec_avg'].replace(0, np.nan)
        result_df['style_escape_activity_raw'] = base_df['rev_per_min_dec_avg'] / ctrl_opp_denom
        
        # Early finish proxy (raw) - use sig strikes for consistency
        result_df['style_early_finish_proxy_raw'] = (
            base_df['sig_str_land_rd1_per_min_dec_avg'] + 
            base_df['td_att_rd1_per_min_dec_avg'] + 
            base_df['sub_att_rd1_per_min_dec_avg']
        )
        
        # Clip all share features to [0,1] and handle infinities for ratio features
        share_columns = [
            'style_head_target_share_raw', 'style_body_target_share_raw', 'style_leg_target_share_raw',
            'style_distance_share_raw', 'style_clinch_share_raw', 'style_ground_share_raw'
        ]
        for col in share_columns:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan).clip(0, 1).fillna(0)
        
        # Handle infinities for derived efficiency features (don't clip these to [0,1] as they can exceed 1)
        efficiency_columns = [
            'style_ctrl_conversion_raw', 'style_td_per_ctrl_raw', 'style_ctrl_per_sub_att_raw',
            'style_sub_efficiency_raw', 'style_ctrl_sub_conversion_raw'
        ]
        for col in efficiency_columns:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Fill NaN values with 0
        result_df = result_df.fillna(0)
        
        return result_df

    def _calculate_complex_features(self, base_df: pd.DataFrame, simple_features_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate complex features that need special handling."""
        
        result_df = base_df[['fight_id', 'fighter_id', 'event_id']].copy()
        
        # Time-aware R1 bias calculations
        r1_secs = np.minimum(base_df['time_sec_rd1_dec_avg'], 300)
        total_secs = base_df['time_sec_dec_avg']
        
        # R1 striking bias (time-aware)
        r1_strikes = base_df['sig_str_land_rd1_per_min_dec_avg'] * r1_secs / 60.0
        total_strikes = base_df['sig_str_land_per_min_dec_avg'] * total_secs / 60.0
        result_df['style_rd1_striking_bias_raw'] = np.where(
            total_strikes > 0, r1_strikes / total_strikes, 0
        )
        
        # R1 wrestling bias (time-aware)
        r1_tds = base_df['td_att_rd1_per_min_dec_avg'] * r1_secs / 60.0
        total_tds = base_df['td_att_per_min_dec_avg'] * total_secs / 60.0
        result_df['style_rd1_wrestle_bias_raw'] = np.where(
            total_tds > 0, r1_tds / total_tds, 0
        )
        
        # Finishing channel calculations
        ko_pm = base_df['ko_per_min_dec_avg']
        sub_pm = base_df['sub_land_per_min_dec_avg']
        dec_pm = base_df['decision_per_min_dec_avg']
        total_finish = ko_pm + sub_pm + dec_pm
        
        result_df['style_finish_total_pm_raw'] = total_finish
        result_df['style_finish_channel_ko_share_raw'] = np.where(total_finish > 0, ko_pm / total_finish, 0)
        result_df['style_finish_channel_sub_share_raw'] = np.where(total_finish > 0, sub_pm / total_finish, 0)
        result_df['style_finish_channel_decision_share_raw'] = np.where(total_finish > 0, dec_pm / total_finish, 0)
        
        # KO vs Sub bias
        ko_sub_total = ko_pm + sub_pm
        result_df['style_finish_ko_vs_sub_bias_raw'] = np.where(ko_sub_total > 0, ko_pm / ko_sub_total, 0)
        
        # Finisher vs Decision bias  
        result_df['style_finisher_vs_decision_bias_raw'] = np.where(total_finish > 0, ko_sub_total / total_finish, 0)
        
        # Finishing channel entropy
        pko = result_df['style_finish_channel_ko_share_raw']
        psub = result_df['style_finish_channel_sub_share_raw']
        pdec = result_df['style_finish_channel_decision_share_raw']
        
        entropy = -(pko * np.log(np.maximum(pko, 1e-10)) + 
                   psub * np.log(np.maximum(psub, 1e-10)) + 
                   pdec * np.log(np.maximum(pdec, 1e-10))) / np.log(3)
        result_df['style_finish_channel_entropy_raw'] = np.where(total_finish > 0, entropy, 0)
        
        # Range entropy calculation - use the computed shares from simple features
        d = simple_features_df['style_distance_share_raw'].clip(lower=0)
        c = simple_features_df['style_clinch_share_raw'].clip(lower=0)
        g = simple_features_df['style_ground_share_raw'].clip(lower=0)
        z = (d + c + g).replace(0, np.nan)
        pd_norm, pc_norm, pg_norm = d/z, c/z, g/z
        range_entropy = -(pd_norm*np.log(np.maximum(pd_norm,1e-10))
                          + pc_norm*np.log(np.maximum(pc_norm,1e-10))
                          + pg_norm*np.log(np.maximum(pg_norm,1e-10))) / np.log(3)
        result_df['style_range_entropy_raw'] = range_entropy.fillna(0)
        
        # Clip all share and bias features to [0,1] and handle infinities
        share_bias_columns = [
            'style_finish_channel_ko_share_raw', 'style_finish_channel_sub_share_raw', 'style_finish_channel_decision_share_raw',
            'style_rd1_striking_bias_raw', 'style_rd1_wrestle_bias_raw',
            'style_finisher_vs_decision_bias_raw', 'style_finish_ko_vs_sub_bias_raw'
        ]
        for col in share_bias_columns:
            result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan).clip(0, 1).fillna(0)
        
        # Fill NaN values with 0 for consistent handling
        result_df = result_df.fillna(0)
        
        return result_df

    def save(self):
        """Calculate and save style metrics to the database."""
        self.logger.info("Saving style metrics to database...")
        
        # Calculate the style metrics
        style_df = self.calculate()
        
        if style_df.empty:
            self.logger.warning("No style data to save")
            return
        
        # Save to database using bulk insert
        try:
            self.bulk_insert_dataframe(
                style_df, 
                self.table_name, 
                schema=self.schema, 
                if_exists='append'
            )
            self.logger.info(f"Successfully saved {len(style_df)} style records to {self.schema}.{self.table_name}")
        except Exception as e:
            self.logger.error(f"Error saving style data: {str(e)}")
            raise

    def run(self) -> Dict[str, pd.DataFrame]:
        """Main execution method for the style calculator."""
        try:
            self.logger.info("Starting style calculation...")
            
            # Create the style table
            self.create_style_table()
            
            # Calculate and save style metrics
            self.save()
            
            self.logger.info("Style calculation completed successfully")
            return {"style": pd.DataFrame({"success": [True]})}
            
        except Exception as e:
            self.logger.error(f"Error in style calculation: {str(e)}")
            return {"style": pd.DataFrame()} 