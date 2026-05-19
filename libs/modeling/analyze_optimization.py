import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
from typing import Optional
import numpy as np

class OptimizationAnalyzer:
    def __init__(self, study_path: str):
        """Load Optuna study from pickle file."""
        self.study = joblib.load(study_path)
        self.df = self.study.trials_dataframe()
        
        # Filter only completed trials
        self.df_completed = self.df[self.df['state'] == 'COMPLETE'].copy()
        
        print(f"Loaded study with {len(self.study.trials)} total trials")
        print(f"Completed trials: {len(self.df_completed)}")
        print(f"Failed trials: {len(self.df) - len(self.df_completed)}")
    
    def print_summary(self):
        """Print optimization summary."""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION SUMMARY")
        print("="*60)
        
        if len(self.df_completed) == 0:
            print("No completed trials found!")
            return
        
        best_trial = self.study.best_trial
        print(f"\nBest Trial #{best_trial.number}:")
        print(f"  Test Log Loss: {best_trial.value:.6f}")
        
        # Get additional metrics if available
        for attr_name in ['test_accuracy', 'train_log_loss', 'train_accuracy']:
            if attr_name in best_trial.user_attrs:
                print(f"  {attr_name.replace('_', ' ').title()}: {best_trial.user_attrs[attr_name]:.4f}")
        
        print(f"\nBest Parameters:")
        for param, value in best_trial.params.items():
            print(f"  {param}: {value}")
        
        # Statistics
        print(f"\n=== Performance Statistics ===")
        print(f"Best Log Loss: {self.df_completed['value'].min():.6f}")
        print(f"Worst Log Loss: {self.df_completed['value'].max():.6f}")
        print(f"Mean Log Loss: {self.df_completed['value'].mean():.6f}")
        print(f"Std Log Loss: {self.df_completed['value'].std():.6f}")
        
        # Top 5 trials
        print(f"\n=== Top 5 Trials ===")
        top_5 = self.df_completed.nsmallest(5, 'value')
        for idx, (_, row) in enumerate(top_5.iterrows(), 1):
            print(f"{idx}. Trial #{int(row['number'])}: {row['value']:.6f}")
    
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """Plot optimization history."""
        if len(self.df_completed) == 0:
            print("No completed trials to plot!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Optimization history
        ax1.plot(self.df_completed['number'], self.df_completed['value'], 'b-', alpha=0.7, label='Trial value')
        ax1.plot(self.df_completed['number'], self.df_completed['value'].cummin(), 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Trial')
        ax1.set_ylabel('Test Log Loss')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Value distribution
        ax2.hist(self.df_completed['value'], bins=20, alpha=0.7, edgecolor='black')
        ax2.axvline(self.df_completed['value'].min(), color='red', linestyle='--', label=f'Best: {self.df_completed["value"].min():.4f}')
        ax2.set_xlabel('Test Log Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Test Log Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        #plt.show()
    
    def plot_parameter_importance(self, save_path: Optional[str] = None):
        """Plot parameter importances."""
        if len(self.df_completed) < 10:
            print("Need at least 10 completed trials for parameter importance analysis!")
            return
        
        try:
            # Calculate parameter importances
            importance = optuna.importance.get_param_importances(self.study)
            
            if not importance:
                print("No parameter importance data available!")
                return
            
            # Create plot
            params = list(importance.keys())
            values = list(importance.values())
            
            plt.figure(figsize=(10, 6))
            bars = plt.barh(params, values)
            plt.xlabel('Importance')
            plt.title('Hyperparameter Importance')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'{value:.3f}', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Parameter importance plot saved to {save_path}")
            
            #plt.show()
            
        except Exception as e:
            print(f"Error calculating parameter importance: {e}")
    
    def plot_parameter_relationships(self, save_path: Optional[str] = None):
        """Plot relationships between parameters and objective value."""
        if len(self.df_completed) == 0:
            print("No completed trials to analyze!")
            return
        
        # Get parameter columns
        param_cols = [col for col in self.df_completed.columns if col.startswith('params_')]
        
        if len(param_cols) == 0:
            print("No parameter columns found!")
            return
        
        # Create subplots
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, param_col in enumerate(param_cols):
            ax = axes[i]
            param_name = param_col.replace('params_', '')
            
            # Check if parameter is categorical or numerical
            unique_values = self.df_completed[param_col].unique()
            
            if len(unique_values) <= 10:  # Categorical
                # Box plot for categorical parameters
                data_for_box = []
                labels = []
                for val in sorted(unique_values):
                    if pd.notna(val):
                        subset = self.df_completed[self.df_completed[param_col] == val]['value']
                        if len(subset) > 0:
                            data_for_box.append(subset)
                            labels.append(str(val))
                
                if data_for_box:
                    ax.boxplot(data_for_box, labels=labels)
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Test Log Loss')
                    ax.tick_params(axis='x', rotation=45)
            else:  # Numerical
                # Scatter plot for numerical parameters
                ax.scatter(self.df_completed[param_col], self.df_completed['value'], alpha=0.6)
                ax.set_xlabel(param_name)
                ax.set_ylabel('Test Log Loss')
            
            ax.set_title(f'{param_name} vs Test Log Loss')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter relationships plot saved to {save_path}")
        
        #plt.show()
    
    def create_comparison_table(self, top_n: int = 10):
        """Create a comparison table of top trials."""
        if len(self.df_completed) == 0:
            print("No completed trials to compare!")
            return None
        
        # Get top N trials
        top_trials = self.df_completed.nsmallest(top_n, 'value').copy()
        
        # Select relevant columns
        param_cols = [col for col in top_trials.columns if col.startswith('params_')]
        attr_cols = [col for col in top_trials.columns if col.startswith('user_attrs_')]
        
        cols_to_show = ['number', 'value'] + param_cols + attr_cols
        cols_to_show = [col for col in cols_to_show if col in top_trials.columns]
        
        comparison_df = top_trials[cols_to_show].copy()
        
        # Clean column names
        comparison_df.columns = [col.replace('params_', '').replace('user_attrs_', '') 
                               for col in comparison_df.columns]
        
        # Round numerical columns
        for col in comparison_df.select_dtypes(include=[np.number]).columns:
            if col != 'number':
                comparison_df[col] = comparison_df[col].round(4)
        
        print(f"\n=== Top {top_n} Trials Comparison ===")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def save_analysis_report(self, output_dir: str):
        """Save comprehensive analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save plots
        history_path = os.path.join(output_dir, 'optimization_history.png')
        self.plot_optimization_history(save_path=history_path)
        
        importance_path = os.path.join(output_dir, 'parameter_importance.png')
        self.plot_parameter_importance(save_path=importance_path)
        
        relationships_path = os.path.join(output_dir, 'parameter_relationships.png')
        self.plot_parameter_relationships(save_path=relationships_path)
        
        # Save comparison table
        comparison_df = self.create_comparison_table(top_n=20)
        if comparison_df is not None:
            comparison_path = os.path.join(output_dir, 'top_trials_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Top trials comparison saved to {comparison_path}")
        
        print(f"\nAnalysis report saved to {output_dir}")

def main():
    """Analyze optimization results."""
    import glob
    
    # Find the most recent study file
    study_files = glob.glob('OptimizedModels/optimization_results/study_*.pkl')
    
    if not study_files:
        print("No study files found! Run hyperparameter optimization first.")
        return
    
    # Get the most recent study file
    latest_study = max(study_files, key=os.path.getctime)
    print(f"Analyzing study: {latest_study}")
    
    # Create analyzer
    analyzer = OptimizationAnalyzer(latest_study)
    
    # Print summary
    analyzer.print_summary()
    
    # Create plots
    analyzer.plot_optimization_history()
    analyzer.plot_parameter_importance()
    analyzer.plot_parameter_relationships()
    
    # Create comparison table
    analyzer.create_comparison_table()
    
    # Save comprehensive report
    report_dir = os.path.dirname(latest_study).replace('optimization_results', 'analysis_report')
    analyzer.save_analysis_report(report_dir)

if __name__ == "__main__":
    main() 