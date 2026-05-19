import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib

class SimplePlattCalibration:
    """
    Simple Platt scaling calibration using a dedicated holdout calibration set.
    
    Platt scaling fits a sigmoid function (logistic regression) to map uncalibrated 
    probabilities to calibrated ones. This approach:
    - Fits a LogisticRegression on out-of-sample calibration predictions
    - Avoids overfitting by using truly held-out data for calibration
    - Works well when you have sufficient calibration data
    - Produces smooth, sigmoid-shaped calibration curves
    """
    
    def __init__(self, max_iter=100, random_state=42):
        """
        Args:
            max_iter: Maximum iterations for logistic regression
            random_state: Random state for reproducible results
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.calibrator = None
        
    def fit(self, y_prob, y_true):
        """
        Fit Platt scaling on holdout calibration set.
        
        Args:
            y_prob: Out-of-sample probabilities from AutoGluon on calibration set
            y_true: True binary labels for calibration set
        """
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        
        print(f"Fitting Platt scaling calibration on {len(y_prob)} holdout predictions...")
        
        # Reshape probabilities for sklearn (needs 2D input)
        y_prob_reshaped = y_prob.reshape(-1, 1)
        
        self.calibrator = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver='lbfgs'  # Good default solver for small datasets
        )
        
        self.calibrator.fit(y_prob_reshaped, y_true)
        print("Platt scaling calibration fitted successfully")
        
    def predict_proba(self, y_prob):
        """Apply Platt scaling transformation."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_prob = np.array(y_prob)
        y_prob_reshaped = y_prob.reshape(-1, 1)
        
        # Get the calibrated probabilities (probability of class 1)
        calibrated_probs = self.calibrator.predict_proba(y_prob_reshaped)[:, 1]
        return calibrated_probs
    
    def save(self, path):
        """Save the calibrator to disk."""
        if self.calibrator is None:
            raise ValueError("Cannot save unfitted calibrator")
        joblib.dump(self, path)
        print(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load a calibrator from disk."""
        return joblib.load(path)

class SimpleIsotonicCalibration:
    """
    Simple isotonic calibration using a dedicated holdout calibration set.
    
    This approach is more suitable when you have a proper train/calibration/test split:
    - Fits a single IsotonicRegression on out-of-sample calibration predictions
    - Avoids overfitting by using truly held-out data for calibration
    - Simple, clean interface without cross-validation complexity
    """
    
    def __init__(self, y_min=0.01, y_max=0.99):
        """
        Args:
            y_min: Minimum bound for calibrated probabilities
            y_max: Maximum bound for calibrated probabilities
        """
        self.y_min = y_min
        self.y_max = y_max
        self.calibrator = None
        
    def fit(self, y_prob, y_true):
        """
        Fit isotonic calibration on holdout calibration set.
        
        Args:
            y_prob: Out-of-sample probabilities from AutoGluon on calibration set
            y_true: True binary labels for calibration set
        """
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        
        print(f"Fitting simple isotonic calibration on {len(y_prob)} holdout predictions...")
        
        self.calibrator = IsotonicRegression(
            y_min=self.y_min,
            y_max=self.y_max,
            out_of_bounds='clip'
        )
        
        self.calibrator.fit(y_prob, y_true)
        print("Isotonic calibration fitted successfully")
        
    def predict_proba(self, y_prob):
        """Apply calibration transformation."""
        if self.calibrator is None:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        y_prob = np.array(y_prob)
        return self.calibrator.transform(y_prob)
    
    def save(self, path):
        """Save the calibrator to disk."""
        if self.calibrator is None:
            raise ValueError("Cannot save unfitted calibrator")
        joblib.dump(self, path)
        print(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load a calibrator from disk."""
        return joblib.load(path)

class RobustIsotonicCalibration:
    """
    Isotonic calibration for AutoGluon models using cross-fold validation.
    
    This follows the same approach as AutoGluon's internal methodology:
    - Uses cross-fold validation to create an ensemble of calibrators
    - Avoids overfitting by fitting each calibrator on different data folds
    - Simple, focused interface for AutoGluon integration
    """
    
    def __init__(self, n_folds=5, y_min=0.01, y_max=0.99, random_state=42):
        """
        Args:
            n_folds: Number of folds for cross-fold validation (matches AutoGluon default)
            y_min: Minimum bound for calibrated probabilities
            y_max: Maximum bound for calibrated probabilities  
            random_state: Random state for reproducible CV splits
        """
        self.n_folds = n_folds
        self.y_min = y_min
        self.y_max = y_max
        self.random_state = random_state
        self.calibrators = []
        self.validation_improvement = 0.0
        
    def fit(self, y_prob, y_true):
        """
        Fit isotonic calibration ensemble using cross-fold validation.
        
        Args:
            y_prob: Uncalibrated probabilities from AutoGluon (same training data)
            y_true: True binary labels
        """
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)
        
        print(f"Fitting isotonic calibration ensemble with {self.n_folds} folds...")
        print(f"Using {len(y_prob)} training predictions for calibration")
        
        self.calibrators = []
        fold_improvements = []
        
        # Cross-fold validation for ensemble calibration (like AutoGluon)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for i, (train_idx, val_idx) in enumerate(skf.split(y_prob, y_true)):
            # Fit calibrator on this fold's training data
            calibrator = IsotonicRegression(
                y_min=self.y_min,
                y_max=self.y_max,
                out_of_bounds='clip'
            )
            
            fold_train_probs = y_prob[train_idx]
            fold_train_labels = y_true[train_idx]
            fold_val_probs = y_prob[val_idx]
            fold_val_labels = y_true[val_idx]
            
            calibrator.fit(fold_train_probs, fold_train_labels)
            self.calibrators.append(calibrator)
            
            # Evaluate this fold
            fold_val_calibrated = calibrator.transform(fold_val_probs)
            fold_improvement = log_loss(fold_val_labels, fold_val_probs) - log_loss(fold_val_labels, fold_val_calibrated)
            fold_improvements.append(fold_improvement)
            
            print(f"  Fold {i+1}: {len(fold_train_probs)} train, {len(fold_val_probs)} val, improvement: {fold_improvement:.4f}")
        
        # Calculate average improvement across folds
        self.validation_improvement = np.mean(fold_improvements)
        print(f"Average log loss improvement: {self.validation_improvement:.4f}")
        
    def predict_proba(self, y_prob):
        """Apply calibration by averaging predictions from all fold calibrators."""
        y_prob = np.array(y_prob)
        
        if not self.calibrators:
            raise ValueError("Calibrator not fitted. Call fit() first.")
        
        # Average predictions from all calibrators (ensemble approach)
        calibrated_probs = []
        for calibrator in self.calibrators:
            cal_prob = calibrator.transform(y_prob)
            calibrated_probs.append(cal_prob)
        
        return np.mean(calibrated_probs, axis=0)
    
    def should_use_calibration(self, threshold=0.001):
        """
        Determine if calibration should be used based on validation improvement.
        
        Args:
            threshold: Minimum log loss improvement required to recommend calibration
        """
        return self.validation_improvement > threshold
    
    def save(self, path):
        """Save the calibration ensemble to disk."""
        joblib.dump(self, path)
        print(f"Calibrator saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load a calibration ensemble from disk."""
        return joblib.load(path)

def evaluate_calibration(y_true, y_prob_original, y_prob_calibrated):
    """
    Comprehensive evaluation of calibration quality.
    
    Returns:
        dict: Calibration metrics including log loss, Brier score, and ECE improvements
    """
    # Calculate metrics
    log_loss_orig = log_loss(y_true, y_prob_original)
    log_loss_cal = log_loss(y_true, y_prob_calibrated)
    
    brier_orig = brier_score_loss(y_true, y_prob_original)
    brier_cal = brier_score_loss(y_true, y_prob_calibrated)
    
    # Expected Calibration Error (sample-weighted per bin)
    def _weighted_ece(y, y_prob, n_bins=10):
        # Compute standard calibration curve outputs
        frac_pos, mean_pred = calibration_curve(y, y_prob, n_bins=n_bins)
        # Recreate binning to count samples per bin
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges) - 1  # 0..n_bins-1
        # Count only non-empty bins in order (calibration_curve drops empty bins)
        counts_all = np.array([(bin_indices == i).sum() for i in range(n_bins)])
        non_empty_mask = counts_all > 0
        counts = counts_all[non_empty_mask]
        if counts.sum() == 0:
            return 0.0
        weights = counts / counts.sum()
        return float(np.sum(weights * np.abs(frac_pos - mean_pred)))

    ece_orig = _weighted_ece(y_true, y_prob_original, n_bins=10)
    ece_cal = _weighted_ece(y_true, y_prob_calibrated, n_bins=10)
    
    return {
        'log_loss_improvement': log_loss_orig - log_loss_cal,
        'brier_improvement': brier_orig - brier_cal,
        'ece_improvement': ece_orig - ece_cal,
        'log_loss_original': log_loss_orig,
        'log_loss_calibrated': log_loss_cal,
        'brier_original': brier_orig,
        'brier_calibrated': brier_cal,
        'ece_original': ece_orig,
        'ece_calibrated': ece_cal,
    }

def plot_calibration_curve(y_true, y_prob_original, y_prob_calibrated, save_path=None, method_name="Calibrated"):
    """Plot calibration curve comparing original vs calibrated probabilities."""
    fraction_pos_orig, mean_pred_orig = calibration_curve(y_true, y_prob_original, n_bins=10)
    fraction_pos_cal, mean_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax.plot(mean_pred_orig, fraction_pos_orig, 'o-', label='Original', alpha=0.7)
    ax.plot(mean_pred_cal, fraction_pos_cal, 's-', label=method_name, alpha=0.7)
    
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Calibration curve saved to {save_path}")
    
    #plt.show()
    return fig 