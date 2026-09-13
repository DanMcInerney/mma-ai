import json
import math
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from datetime import datetime, timezone
from typing import Optional
import joblib


def _json_safe(value):
    """Convert nested NumPy/pandas values into strict JSON values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and missing:
        return None
    return value

class ShapVisualizer:
    def __init__(self, model, feats, feature_display_names=None, output_dir=None):
        """
        Initialize SHAP visualizer for binary classification with AutoGluon.
        
        Args:
            model: AutoGluon TabularPredictor or EnsemblePredictor
            feats: List of feature names used for prediction
            feature_display_names: Dictionary mapping feature names to display names
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.features = feats
        self.feature_display_names = feature_display_names or {}
        
        # Detect if this is an EnsemblePredictor
        from libs.modeling.train import EnsemblePredictor
        self.is_ensemble = isinstance(model, EnsemblePredictor)
        
        if self.is_ensemble:
            print(f"Detected EnsemblePredictor with {len(model.predictors)} models - will calculate SHAP for each and average")
            self.ensemble_predictors = model.predictors
            self.ensemble_scaler_paths = model.scaler_paths
        else:
            self.ensemble_predictors = None
            self.ensemble_scaler_paths = None
        
        # Set up output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join("visualizations", timestamp)
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize explainer to None - will be created when needed
        self.explainer = None
    
    def create_explainer(self, background_data, nsamples=500):
        """
        Create a SHAP explainer using background data.
        
        Args:
            background_data: DataFrame with background data (typically subset of training data)
            nsamples: Number of samples for SHAP approximation
            
        Returns:
            SHAP explainer
        """
        print(f"Creating SHAP explainer with {len(background_data)} background samples")
        
        # Check if model needs sample_weight column
        self.needs_sample_weight = False
        background_data_test = background_data.copy()
        try:
            # Test if model needs sample_weight
            self.model.predict_proba(background_data_test.iloc[[0]])
        except KeyError as e:
            if 'sample_weight' in str(e):
                # Model was trained with sample_weight
                self.needs_sample_weight = True
                print("Model was trained with sample weights - will add during predictions")
        
        # Create wrapper class for AutoGluon model
        class AutogluonWrapper:
            def __init__(self, predictor, feature_names, needs_sample_weight):
                self.ag_model = predictor
                self.feature_names = feature_names
                self.needs_sample_weight = needs_sample_weight
            
            def predict_binary_prob(self, X):
                """Return binary probability predictions"""
                if isinstance(X, pd.Series):
                    X = X.values.reshape(1, -1)
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=self.feature_names)
                
                # Ensure we only have the feature columns (no sample_weight from SHAP)
                X_features = X[self.feature_names] if all(col in X.columns for col in self.feature_names) else X
                
                # Get predictions - handle models trained with sample_weight
                if self.needs_sample_weight:
                    # Model was trained with sample_weight, add it
                    X_with_weights = X_features.copy()
                    X_with_weights['sample_weight'] = 1.0
                    preds = self.ag_model.predict_proba(X_with_weights)
                else:
                    # Normal prediction
                    preds = self.ag_model.predict_proba(X_features)
                
                if hasattr(preds, 'iloc'):
                    # If it's a DataFrame
                    return preds.iloc[:, 1].values
                elif isinstance(preds, np.ndarray) and preds.shape[1] >= 2:
                    # If it's a numpy array
                    return preds[:, 1]
                else:
                    return preds
        
        # Create wrapper
        ag_wrapper = AutogluonWrapper(self.model, self.features, self.needs_sample_weight)
        
        # Create explainer using background data (without sample_weight)
        explainer = shap.KernelExplainer(ag_wrapper.predict_binary_prob, background_data)
        
        # Store expected value
        expected_value = explainer.expected_value
        print(f"Baseline prediction (expected value): {expected_value}")
        
        return explainer
    
    def _scale_data_for_predictor(self, data: pd.DataFrame, scaler_path: Optional[str]) -> pd.DataFrame:
        """
        Scale data using the specified scaler (same logic as EnsemblePredictor).
        
        Args:
            data: Input data to scale
            scaler_path: Path to scaler pickle file (None if no scaling needed)
            
        Returns:
            Scaled data DataFrame
        """
        if scaler_path is None or not os.path.exists(scaler_path):
            return data.copy()
        
        try:
            scaler = joblib.load(scaler_path)
            
            # Identify columns to exclude from scaling (same as EnsemblePredictor)
            date_cols = ['event_date', 'fight_id', 'fighter_name', 'opp_name']
            categorical_static_feats = ['weightclass_encoded', 'odds']
            
            def should_exclude_col(col_name):
                if col_name in date_cols:
                    return True
                for cat_feat in categorical_static_feats:
                    if cat_feat in col_name:
                        return True
                return False
            
            # Get features to scale
            features_to_scale = [col for col in data.columns 
                                if not should_exclude_col(col) and col not in ['sample_weight', 'y_true']]
            
            # Scale only the features that should be scaled
            scaled_data = data.copy()
            if len(features_to_scale) > 0:
                scaled_data[features_to_scale] = scaler.transform(data[features_to_scale])
            
            return scaled_data
        except Exception as e:
            print(f"Warning: Could not load/apply scaler from {scaler_path}: {e}")
            return data.copy()
    
    def compute_shap_values(self, prediction_data, background_data=None, nsamples=500):
        """
        Compute SHAP values for a prediction.
        
        For EnsemblePredictor: calculates SHAP for each model separately and averages.
        For single models: calculates SHAP normally.
        
        Args:
            prediction_data: DataFrame with the prediction data (can be a single row)
            background_data: DataFrame with background data (typically subset of training data)
                             If None, will use median of prediction_data as background
            nsamples: Number of samples for SHAP approximation
            
        Returns:
            Dictionary with SHAP values and explanation data
        """
        # If only a single row provided and no background data, we need to handle specially
        is_single_row = len(prediction_data) == 1
        
        if is_single_row and background_data is None:
            print("Warning: Single row prediction without background data. Using median values as background.")
            # Create a simple background from the median of features
            # Not ideal but better than nothing
            background_data = pd.DataFrame([prediction_data.iloc[0]])
        
        # Handle ensemble models
        if self.is_ensemble:
            print(f"Computing SHAP values for ensemble with {len(self.ensemble_predictors)} models...")
            all_shap_values = []
            all_expected_values = []
            
            # Calculate SHAP for each model separately
            for i, (predictor, scaler_path) in enumerate(zip(self.ensemble_predictors, self.ensemble_scaler_paths)):
                print(f"  Computing SHAP for model {i+1}/{len(self.ensemble_predictors)}...")
                
                # Scale data using this model's scaler
                scaled_prediction_data = self._scale_data_for_predictor(prediction_data, scaler_path)
                scaled_background_data = self._scale_data_for_predictor(background_data, scaler_path)
                
                # Create explainer for this model
                explainer = self._create_explainer_for_model(predictor, scaled_background_data, nsamples)
                
                # Compute SHAP values for this model
                shap_values = explainer.shap_values(scaled_prediction_data, nsamples=nsamples)
                
                # Handle different SHAP output formats
                if isinstance(shap_values, list):
                    # Binary classification returns [shap_values_class0, shap_values_class1]
                    shap_values = shap_values[1]  # Use class 1 (positive class)
                elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                    # If 2D array, use as-is (should be [n_samples, n_features])
                    pass
                else:
                    # If 1D array for single sample, reshape to 2D
                    shap_values = shap_values.reshape(1, -1)
                
                all_shap_values.append(shap_values)
                all_expected_values.append(explainer.expected_value)
            
            # Average SHAP values across all models
            avg_shap_values = np.mean(all_shap_values, axis=0)
            avg_expected_value = np.mean(all_expected_values)
            
            print(f"Averaged SHAP values across {len(self.ensemble_predictors)} models")
            
            return {
                'shap_values': avg_shap_values,
                'expected_value': avg_expected_value,
                'feature_names': prediction_data.columns.tolist(),
                'is_single_row': is_single_row
            }
        else:
            # Single model: use existing logic
            # Create explainer if needed
            if self.explainer is None:
                self.explainer = self.create_explainer(background_data, nsamples)
            
            # Compute SHAP values
            print(f"Computing SHAP values for {len(prediction_data)} rows")
            shap_values = self.explainer.shap_values(prediction_data, nsamples=nsamples)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use class 1 (positive class)
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 1 and is_single_row:
                shap_values = shap_values.reshape(1, -1)
            
            return {
                'shap_values': shap_values,
                'expected_value': self.explainer.expected_value,
                'feature_names': prediction_data.columns.tolist(),
                'is_single_row': is_single_row
            }
    
    def _create_explainer_for_model(self, predictor, background_data, nsamples=500):
        """
        Create a SHAP explainer for a single model (used for ensemble models).
        
        Args:
            predictor: Single TabularPredictor model
            background_data: Scaled background data for this model
            nsamples: Number of samples for SHAP approximation
            
        Returns:
            SHAP explainer
        """
        # Check if model needs sample_weight column
        needs_sample_weight = False
        background_data_test = background_data.copy()
        try:
            # Test if model needs sample_weight
            predictor.predict_proba(background_data_test.iloc[[0]])
        except KeyError as e:
            if 'sample_weight' in str(e):
                needs_sample_weight = True
        
        # Create wrapper class for AutoGluon model
        class AutogluonWrapper:
            def __init__(self, ag_predictor, feature_names, needs_sample_weight):
                self.ag_model = ag_predictor
                self.feature_names = feature_names
                self.needs_sample_weight = needs_sample_weight
            
            def predict_binary_prob(self, X):
                """Return binary probability predictions"""
                if isinstance(X, pd.Series):
                    X = X.values.reshape(1, -1)
                if not isinstance(X, pd.DataFrame):
                    X = pd.DataFrame(X, columns=self.feature_names)
                
                # Ensure we only have the feature columns (no sample_weight from SHAP)
                X_features = X[self.feature_names] if all(col in X.columns for col in self.feature_names) else X
                
                # Get predictions - handle models trained with sample_weight
                if self.needs_sample_weight:
                    # Model was trained with sample_weight, add it
                    X_with_weights = X_features.copy()
                    X_with_weights['sample_weight'] = 1.0
                    preds = self.ag_model.predict_proba(X_with_weights)
                else:
                    # Normal prediction
                    preds = self.ag_model.predict_proba(X_features)
                
                if hasattr(preds, 'iloc'):
                    # If it's a DataFrame
                    return preds.iloc[:, 1].values
                elif isinstance(preds, np.ndarray) and preds.shape[1] >= 2:
                    # If it's a numpy array
                    return preds[:, 1]
                else:
                    return preds
        
        # Create wrapper
        ag_wrapper = AutogluonWrapper(predictor, self.features, needs_sample_weight)
        
        # Create explainer using background data (without sample_weight)
        explainer = shap.KernelExplainer(ag_wrapper.predict_binary_prob, background_data)
        
        return explainer
    
    def create_force_plot(self, shap_data, index=0, fighter1_name="Fighter 1", fighter2_name="Fighter 2", win_prob=None):
        """
        Create a force plot showing feature contributions.
        
        Args:
            shap_data: Dictionary from compute_shap_values
            index: Index of prediction to explain
            fighter1_name: Name of fighter 1
            fighter2_name: Name of fighter 2
            win_prob: Win probability
            
        Returns:
            Plotly figure with force plot
        """
        # Extract data for this prediction
        shap_values = shap_data['shap_values'][index]
        feature_names = shap_data['feature_names']
        expected_value = shap_data['expected_value']
        
        # Get display names for features
        display_names = [self.feature_display_names.get(f, f) for f in feature_names]
        
        # Create DataFrame for visualization
        features_df = pd.DataFrame({
            'Feature': display_names,
            'SHAP Value': shap_values,
            'Abs Value': abs(shap_values)
        })
        
        # Sort by absolute value (largest impact first)
        features_df = features_df.sort_values('Abs Value', ascending=False)
        
        # Get top features for clarity
        top_features = features_df.head(15)
        
        # Split positive and negative contributions
        pos_features = top_features[top_features['SHAP Value'] > 0].sort_values('SHAP Value')
        neg_features = top_features[top_features['SHAP Value'] < 0].sort_values('SHAP Value', ascending=False)
        
        # Create plot
        fig = go.Figure()
        
        # Add bars for negative features (favor fighter2) - Add this trace FIRST for legend order
        fig.add_trace(go.Bar(
            y=neg_features['Feature'],
            x=neg_features['SHAP Value'],
            orientation='h',
            marker_color='#B2182B',
            name=f'Favors {fighter2_name}',
            text=neg_features['SHAP Value'].round(3),
            textposition='outside',
            legendgroup='group2',
            legendrank=1  # Lower rank appears first
        ))
        
        # Add bars for positive features (favor fighter1) - Add this trace SECOND for legend order
        fig.add_trace(go.Bar(
            y=pos_features['Feature'],
            x=pos_features['SHAP Value'],
            orientation='h',
            marker_color='#2166AC',
            name=f'Favors {fighter1_name}',
            text=pos_features['SHAP Value'].round(3),
            textposition='outside',
            legendgroup='group1',
            legendrank=2  # Higher rank appears second
        ))
        
        # Title
        title = f"Feature Impact: {fighter1_name} vs {fighter2_name}"
        if win_prob is not None:
            title += f"<br>{fighter1_name} Win Probability: {win_prob:.1%}"
        
        # Layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', y=0.97),
            xaxis_title='SHAP Value (Impact on Prediction)',
            yaxis_title='Feature',
            barmode='relative',
            height=850,
            width=1100,
            margin=dict(t=140, l=380, r=80, b=80),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.9,
                xanchor="center",
                x=0.5,
                traceorder="grouped",  # Force the grouped order
                itemsizing="constant"  # Make legend items same size
            )
        )
        
        # Keep outside bar labels inside the plotting area, away from feature names.
        extent = max(float(top_features['Abs Value'].max()), 0.001)
        minimum = min(float(top_features['SHAP Value'].min()), 0.0)
        maximum = max(float(top_features['SHAP Value'].max()), 0.0)
        fig.update_xaxes(range=[minimum - extent * 0.25, maximum + extent * 0.25])
        fig.update_traces(cliponaxis=False)

        # Add zero line
        fig.add_shape(
            type="line",
            x0=0, x1=0,
            y0=-0.5, y1=len(top_features) - 0.5,
            line=dict(color="black", width=1, dash="dash")
        )
        
        # Add baseline annotation
        fig.add_annotation(
            x=0,
            y=-1,
            text=f"Baseline prediction: {expected_value:.3f}",
            showarrow=False,
            font=dict(size=12)
        )
        
        return fig
    
    def create_dependence_plot(self, shap_data, feature_name, fighter1_name, fighter2_name):
        """
        Create a dependence plot for a specific feature.
        
        Args:
            shap_data: Dictionary from compute_shap_values
            feature_name: Name of feature to create dependence plot for
            fighter1_name: Name of fighter 1
            fighter2_name: Name of fighter 2
            
        Returns:
            Plotly figure with dependence plot
        """
        # Extract data
        shap_values = shap_data['shap_values']
        feature_names = shap_data['feature_names']
        
        # Get feature index
        if feature_name not in feature_names:
            print(f"Feature {feature_name} not found. Available features: {feature_names}")
            return None
        
        feature_idx = feature_names.index(feature_name)
        
        # Create plot
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=X[feature_name],
            y=shap_values[:, feature_idx],
            mode='markers',
            marker=dict(
                size=8,
                color=shap_values[:, feature_idx],
                colorscale='RdBu',
                line=dict(width=1, color='black')
            ),
            name=feature_name
        ))
        
        # Title and layout
        fig.update_layout(
            title=f"SHAP Dependence Plot: {feature_name}",
            xaxis_title=feature_name,
            yaxis_title='SHAP Value',
            height=600,
            width=800
        )
        
        # Add horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=X[feature_name].min(),
            x1=X[feature_name].max(),
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )
        
        return fig
    
    def save_plot(self, plot, fighter1_name, fighter2_name, suffix=None):
        """Save a SHAP plot to an HTML file."""
        output_dir = self.output_dir or "."
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract first names for the filename and remove apostrophes
        fighter1_first = fighter1_name.split()[0].lower().replace("'", "")
        fighter2_first = fighter2_name.split()[0].lower().replace("'", "")
        
        # Use the simplified naming format without suffix
        filename = f"{output_dir}/shap_{fighter1_first}_{fighter2_first}.html"
        
        # Save the plot using plotly's write_html method
        plot.write_html(filename)
        return filename
    
    def explain_prediction(self, prediction_data, background_data=None, fighter1_name="Fighter 1", fighter2_name="Fighter 2", win_prob=None, nsamples=500, random_seed=42):
        """
        Generate and save SHAP explanation for a prediction.
        
        Args:
            prediction_data: DataFrame with the prediction data (typically 1 row for a future fight)
            background_data: DataFrame with background data (subset of training data)
            fighter1_name: Name of fighter 1
            fighter2_name: Name of fighter 2
            win_prob: Win probability
            nsamples: Number of samples for SHAP approximation
            
        Returns:
            Dictionary with paths to saved visualizations
        """
        # KernelExplainer is approximate and uses NumPy's global RNG. Preserve
        # the caller's state while making this explanation reproducible.
        numpy_random_state = np.random.get_state()
        np.random.seed(random_seed)
        try:
            shap_data = self.compute_shap_values(prediction_data, background_data, nsamples)
        finally:
            np.random.set_state(numpy_random_state)
        
        # Create force plot
        force_fig = self.create_force_plot(
            shap_data, 
            index=0, 
            fighter1_name=fighter1_name,
            fighter2_name=fighter2_name,
            win_prob=win_prob
        )
        
        # Save force plot with simplified naming
        force_path = self.save_plot(
            force_fig, 
            fighter1_name, 
            fighter2_name
        )

        # Keep the numeric explanation beside the HTML visualization.  The
        # HTML is useful for browsing, but it is not a stable machine-readable
        # record of the actual explanation values.
        shap_json_path = os.path.splitext(force_path)[0] + ".json"

        feature_names = list(shap_data["feature_names"])
        feature_columns = list(prediction_data.columns)
        if feature_columns != feature_names:
            raise ValueError(
                "SHAP feature/value order mismatch: "
                f"prediction columns={feature_columns}, SHAP features={feature_names}"
            )

        shap_record = {
            "fighter1": fighter1_name,
            "fighter2": fighter2_name,
            "feature_names": feature_names,
            "feature_values": _json_safe(prediction_data.to_numpy()),
            "shap_values": _json_safe(shap_data["shap_values"]),
            "base_values": _json_safe(shap_data["expected_value"]),
            "prediction_probability": _json_safe(win_prob),
            "provenance": {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_class": type(self.model).__name__,
                "target": "binary class 1 probability",
                "explainer": "shap.KernelExplainer",
                "approximation": True,
                "nsamples": int(nsamples),
                "random_seed": int(random_seed),
                "background_sample_random_state": 42,
                "background_rows": int(len(background_data)) if background_data is not None else None,
                "feature_count": int(len(shap_data["feature_names"])),
                "library_version": getattr(shap, "__version__", "unknown"),
            },
        }
        with open(shap_json_path, "w", encoding="utf-8") as shap_file:
            json.dump(shap_record, shap_file, indent=2, allow_nan=False)

        # Return paths to visualizations
        return {'force_plot': force_path, 'data': shap_json_path}

# Usage example:
# model = TabularPredictor.load('path/to/model')
# shap_viz = ShapVisualizer(model, X.columns)
# 
# # For a single fight prediction with training data as baseline
# # Take a random sample of training data as background
# background_data = X_train.sample(100, random_state=42)
# prediction_data = X_test.iloc[[0]]  # Single fight to predict
# 
# result = shap_viz.explain_prediction(
#     prediction_data, 
#     background_data, 
#     "Fighter1", 
#     "Fighter2", 
#     win_prob=0.65
# )
# print(f"SHAP visualization saved to: {result['force_plot']}")
