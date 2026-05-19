import numpy as np
import pandas as pd

class AutoGluonWrapper:
    """Wrapper for AutoGluon predictor to work with sklearn's CalibratedClassifierCV."""
    
    def __init__(self, predictor, feature_columns=None):
        self.predictor = predictor
        self.feature_columns = feature_columns
        self.classes_ = np.array([0, 1])  # Binary classification
        # Add sklearn required attributes
        self.fitted_ = True
        self._estimator_type = "classifier"
    
    def __sklearn_is_fitted__(self):
        """Custom sklearn fitted check method."""
        return True
    
    def fit(self, X, y):
        # AutoGluon is already fitted, so we just return self
        # Add fitted indicator that sklearn expects
        self.fitted_ = True
        return self
    
    def predict_proba(self, X):
        # Convert to DataFrame if needed for AutoGluon
        if hasattr(X, 'columns'):
            X_df = X
        else:
            # Use stored column names for proper DataFrame conversion
            X_df = pd.DataFrame(X, columns=self.feature_columns)
        return self.predictor.predict_proba(X_df, as_pandas=False)
    
    def predict(self, X):
        """Required for sklearn compatibility."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int) 