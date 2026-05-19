"""
Run 2025 Optimization Experiments.

Compares 3 strategies to address distribution shift:
1. Baseline: Expanding Window, Decay 0.125
2. High Decay: Expanding Window, Decay 0.50
3. Rolling Window: 4-Year Window, Decay 0.125
"""

import os
import sys
import pandas as pd
import logging
from dataclasses import replace

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from libs.modeling.train import TrainingConfig
from libs.modeling.walk_forward import WalkForwardConfig, WalkForwardValidator
from libs.feature_store.features import vSeven_testing2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(name: str, wf_config: WalkForwardConfig):
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {name}")
    print(f"{'='*80}")
    
    validator = WalkForwardValidator(wf_config)
    summary_df = validator.run()
    
    # Extract recent performance (last 2 folds usually cover 2023-2024)
    recent_folds = summary_df.tail(2)
    print(f"\nRESULTS: {name}")
    print(recent_folds[['test_range', 'accuracy', 'log_loss', 'train_size']].to_string(index=False))
    print(f"Mean Accuracy (All Folds): {summary_df['accuracy'].mean():.4f}")
    
    return summary_df

def main():
    # Common Base Config
    base_config = TrainingConfig(
        model_type='win',
        preset='extreme',
        time_limit=600, # 10 mins per fold per experiment
        features=vSeven_testing2,
        use_recency_weights=True,
        decay_rate=0.125
    )
    
    # Experiment 1: Baseline (Expanding, Decay 0.125)
    config_baseline = WalkForwardConfig(
        base_config=base_config,
        n_folds=5,
        initial_train_years=6,
        output_dir='experiments/baseline'
    )
    
    # Experiment 2: High Decay (Expanding, Decay 0.50)
    base_config_high_decay = replace(base_config, decay_rate=0.50)
    config_high_decay = WalkForwardConfig(
        base_config=base_config_high_decay,
        n_folds=5,
        initial_train_years=6,
        output_dir='experiments/high_decay'
    )
    
    # Experiment 3: Rolling Window (4-Year Window, Decay 0.125)
    # 4-year window means if we test 2024, we train on 2020-2023
    config_rolling = WalkForwardConfig(
        base_config=base_config,
        n_folds=5,
        initial_train_years=4, # Start with 4 years
        rolling_window_years=4, # Maintain 4 years
        output_dir='experiments/rolling'
    )
    
    # Run Experiments
    results = {}
    results['Baseline'] = run_experiment('Baseline (Expanding, Decay 0.125)', config_baseline)
    results['High Decay'] = run_experiment('High Decay (Expanding, Decay 0.50)', config_high_decay)
    results['Rolling'] = run_experiment('Rolling Window (4 Years)', config_rolling)
    
    # Comparative Summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON (Last Fold - 2024)")
    print(f"{'='*80}")
    
    comparison_data = []
    for name, df in results.items():
        last_fold = df.iloc[-1]
        comparison_data.append({
            'Strategy': name,
            'Test Year': last_fold['test_range'],
            'Accuracy': last_fold['accuracy'],
            'Log Loss': last_fold['log_loss'],
            'Train Size': last_fold['train_size']
        })
        
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.to_string(index=False))
    
    # Save comparison
    os.makedirs('experiments', exist_ok=True)
    comp_df.to_csv('experiments/comparison_2024.csv', index=False)

if __name__ == "__main__":
    main()
