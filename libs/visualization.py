import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict
from datetime import datetime

class FightVisualizer:
    def __init__(self, feats: List[str],
                 feature_display_names: Dict[str, str] = None,
                 group_map: Dict[str, List[str]] = None,
                 output_dir: str = None):
        # List of individual features to plot
        self.key_features = feats

        # Define default display names (can be customized)
        self.feature_display_names = {
            # Basic stats
            'age_dec_avg_diff': 'Age',
            'age_ratio_diff': 'Age Ratio',
            'reach_ratio_dec_avg_diff': 'Reach Ratio',
            'ufcage_dec_avg_diff': 'UFC Age',
            'days_since_last_fight_dec_avg_diff': 'Days Since Last Fight',

            # Significant strikes
            'sig_str_land_ratio_dec_adjperf_dec_avg_diff': 'Sig Str Ratio (Adj)',
            'sig_str_land_ratio_dec_avg_diff': 'Sig Str Ratio',

            # Strikes
            'strikes_land_rd1_dec_adjperf_dec_avg_diff': 'R1 Strikes (Adj)',

            # Head strikes
            'head_land_ratio_dec_adjperf_dec_avg_diff': 'Head Strike Ratio (Adj)',
            'head_def_dec_avg_diff': 'Head Defense',
            'head_acc_dec_adjperf_dec_avg_diff': 'Head Accuracy (Adj)',
            'head_land_dec_avg_diff': 'Head Strikes Landed',

            # Body strikes
            'body_acc_dec_adjperf_dec_avg_diff': 'Body Accuracy (Adj)',
            'body_def_dec_avg_diff': 'Body Defense',

            # Distance strikes
            'distance_land_ratio_dec_adjperf_dec_avg_diff': 'Distance Str Ratio (Adj)',
            'distance_acc_dec_adjperf_dec_avg_diff': 'Distance Accuracy (Adj)',
            'distance_def_dec_adjperf_dec_avg_diff': 'Distance Defense (Adj)',
            
            # Clinch strikes
            'clinch_land_per_min_dec_avg_diff': 'Clinch Strikes/Min',

            # Leg strikes
            'leg_land_per_min_opp_dec_avg_diff': 'Leg Strikes/Min Against',

            # Ground strikes
            'ground_def_dec_adjperf_dec_avg_diff': 'Ground Defense (Adj)',

            # Takedowns
            'td_acc_dec_avg_diff': 'TD Accuracy',
            'td_def_dec_avg_diff': 'TD Defense',

            # Control time
            'ctrl_rd1_per_min_opp_dec_avg_diff': 'R1 Ctrl Time/Min Against',
            'ctrl_rd1_dec_avg_diff': 'R1 Control Time',

            # Reversals
            'rev_dec_adjperf_dec_avg_diff': 'Reversals (Adj)',
            'rev_rd1_ratio_opp_dec_avg_diff': 'R1 Reversal Ratio Against',

            # Submissions
            'sub_att_dec_avg_diff': 'Sub Attempts',
            'sub_att_per_min_opp_dec_avg_diff': 'Sub Att/Min Against',

            # Wins
            'win_ratio_dec_avg_diff': 'Win Ratio',
            
            # Knockdowns
            'kd_opp_dec_avg_diff': 'Knockdowns Against',
            
            # Knockouts
            'ko_dec_avg_diff': 'KO Rate'
        }

        # Define a default grouping of features (you can change these as needed)
        self.group_map = {
            "Fighter Attributes": [
                'age_dec_avg_diff',
                'age_ratio_diff',
                'reach_ratio_dec_avg_diff',
                'ufcage_dec_avg_diff',
                'days_since_last_fight_dec_avg_diff',
                'win_ratio_dec_avg_diff'
            ],
            "Overall Striking": [
                'sig_str_land_ratio_dec_adjperf_dec_avg_diff',
                'sig_str_land_ratio_dec_avg_diff',
                'strikes_land_rd1_dec_adjperf_dec_avg_diff',
                'head_land_ratio_dec_adjperf_dec_avg_diff',
                'head_def_dec_avg_diff',
                'head_acc_dec_adjperf_dec_avg_diff',
                'head_land_dec_avg_diff'
            ],
            "Kickboxing": [
                'leg_land_per_min_opp_dec_avg_diff',
                'body_acc_dec_adjperf_dec_avg_diff',
                'body_def_dec_avg_diff',
                'distance_land_ratio_dec_adjperf_dec_avg_diff',
                'distance_acc_dec_adjperf_dec_avg_diff',
                'distance_def_dec_adjperf_dec_avg_diff',
                'clinch_land_per_min_dec_avg_diff',
                'kd_opp_dec_avg_diff',
                'ko_dec_avg_diff'
            ],
            "Wrestling": [
                'td_acc_dec_avg_diff',
                'td_def_dec_avg_diff',
                'ctrl_rd1_per_min_opp_dec_avg_diff',
                'ctrl_rd1_dec_avg_diff'
            ],
            "Grappling": [
                'ground_def_dec_adjperf_dec_avg_diff',
                'rev_dec_adjperf_dec_avg_diff',
                'rev_rd1_ratio_opp_dec_avg_diff',
                'sub_att_dec_avg_diff',
                'sub_att_per_min_opp_dec_avg_diff'
            ],
        }

        # Define additional features that need inversion regardless
        self.inversion_features = ['age_dec_avg_diff', 'age_ratio_diff', 'days_since_last_fight_dec_avg_diff', 'ufcage_dec_avg_diff']

        # Set up the output directory as a class variable.
        # If not provided, use a folder named "visualizations/<timestamp>"
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("visualizations", timestamp)
        else:
            self.output_dir = output_dir

    def invert_value(self, feature: str, value: float) -> float:
        """
        Invert the stat if it's a negative indicator.
        """
        if feature in self.inversion_features or ("_opp_" in feature and feature.endswith("_dec_avg_diff")):
            return -value
        return value

    def create_individual_chart(self,
                                fighter_stats: pd.DataFrame,
                                fighter1_name: str,
                                fighter2_name: str,
                                win_prob: float = None,
                                features: List[str] = None) -> go.Figure:
        """
        Create a radar chart using each individual stat.
        """
        features = features or self.key_features
        display_names = [self.feature_display_names.get(f, f) for f in features]

        # Copy stats and apply inversion where needed
        plot_stats = fighter_stats.copy()
        for feat in features:
            if feat in plot_stats.columns:
                plot_stats[feat] = self.invert_value(feat, plot_stats[feat])

        values = plot_stats[features].iloc[0].tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=display_names,
            fill='toself',
            name=fighter1_name,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[-v for v in values],
            theta=display_names,
            fill='toself',
            name=fighter2_name,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
        ))

        title = f"{fighter1_name} vs {fighter2_name} - Individual Stats"
        if win_prob is not None:
            title += f"<br>{fighter1_name} Win Probability: {win_prob:.1%}"

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2.5, 2.5],
                    tickformat=".1f"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=True,
            title=dict(text=title, x=0.5, xanchor='center'),
            width=900,
            height=900,
            margin=dict(l=200, r=200, t=200, b=200)
        )
        return fig

    def create_grouped_chart(self,
                             fighter_stats: pd.DataFrame,
                             fighter1_name: str,
                             fighter2_name: str,
                             win_prob: float = None) -> go.Figure:
        """
        Create a radar chart with grouped statistics.
        """
        group_values = {}
        for group, feats in self.group_map.items():
            values = []
            for feat in feats:
                if feat in fighter_stats.columns:
                    val = fighter_stats.at[fighter_stats.index[0], feat]
                    values.append(self.invert_value(feat, val))
            group_values[group] = np.mean(values) if values else 0.0

        groups = list(group_values.keys())
        values = list(group_values.values())

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=groups,
            fill='toself',
            name=fighter1_name,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[-v for v in values],
            theta=groups,
            fill='toself',
            name=fighter2_name,
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>"
        ))

        title = f"{fighter1_name} vs {fighter2_name} - Grouped Stats"
        if win_prob is not None:
            title += f"<br>{fighter1_name} Win Probability: {win_prob:.1%}"

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-2.5, 2.5],
                    tickformat=".1f"
                ),
                angularaxis=dict(
                    tickfont=dict(size=12),
                    rotation=90,
                    direction='clockwise'
                )
            ),
            showlegend=True,
            title=dict(text=title, x=0.5, xanchor='center'),
            width=900,
            height=900,
            margin=dict(l=200, r=200, t=200, b=200)
        )
        return fig

    def save_visualization(self,
                           fig: go.Figure,
                           fighter1_name: str,
                           fighter2_name: str,
                           grouped: bool = False,
                           output_dir: str = None) -> str:
        """
        Save the provided figure as an HTML file.
        """
        # Use provided output_dir or fallback to self.output_dir
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract first names for the filename and remove apostrophes
        fighter1_first = fighter1_name.split()[0].lower().replace("'", "")
        fighter2_first = fighter2_name.split()[0].lower().replace("'", "")
        
        # Create filename with the new format
        if grouped:
            filename = f"{output_dir}/grouped_{fighter1_first}_{fighter2_first}.html"
        else:
            filename = f"{output_dir}/ind_{fighter1_first}_{fighter2_first}.html"
            
        fig.write_html(filename)
        return filename

# Example usage:
# Assuming `df_stats` is a one-row DataFrame with your fighter1 - fighter2 stat differences,
# and FEATS_SLIM is a list of features as defined in your example.
#
# visualizer = FightVisualizer(feats=FEATS_SLIM)
# fig_individual = visualizer.create_individual_chart(df_stats, "Fighter1", "Fighter2", win_prob=0.65)
# fig_grouped = visualizer.create_grouped_chart(df_stats, "Fighter1", "Fighter2", win_prob=0.65)
# visualizer.save_visualization(fig_individual, "Fighter1", "Fighter2", grouped=False)
# visualizer.save_visualization(fig_grouped, "Fighter1", "Fighter2", grouped=True)
