#!/usr/bin/env python3
"""
Analyze the impact of current tau values on smoothing behavior.

This script helps understand how different tau values affect smoothing strength
across different stats and scenarios.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def pseudo_minutes_smoothing(observed_count, exposure_time_min, prior_rate, tau):
    """Calculate smoothed value using pseudo-minutes formula."""
    posterior_rate = (prior_rate * tau + observed_count) / (tau + exposure_time_min)
    smoothed_count = posterior_rate * exposure_time_min
    return smoothed_count, posterior_rate

def analyze_smoothing_behavior():
    """Analyze how tau values affect smoothing behavior."""
    
    print("📊 TAU SMOOTHING BEHAVIOR ANALYSIS")
    print("=" * 50)
    
    # Current tau configuration
    current_tau = {
        # high-volume
        'sig_str': 4.0, 'strikes': 4.0,
        # targets / phases (moderate)
        'head': 4.0, 'body': 6.0, 'leg': 6.0,
        'distance': 6.0, 'clinch': 6.0, 'ground': 6.0,
        # grappling
        'td': 8.0,
        # rare events — stronger prior
        'sub': 10.0, 'rev': 12.0, 'kd': 12.0,
        'default': 6.0
    }
    
    # Test scenarios: (stat_type, observed_count, exposure_minutes, typical_prior_rate)
    scenarios = [
        # High-volume striking
        ('sig_str', 50, 15.0, 3.0),   # 50 strikes in 15min fight, prior 3.0/min
        ('sig_str', 20, 5.0, 3.0),    # 20 strikes in 5min fight, prior 3.0/min
        ('sig_str', 100, 25.0, 3.0),  # 100 strikes in 25min fight, prior 3.0/min
        
        # Moderate-volume targets
        ('head', 30, 15.0, 1.5),      # 30 head strikes in 15min, prior 1.5/min
        ('body', 8, 15.0, 0.5),       # 8 body strikes in 15min, prior 0.5/min
        
        # Grappling
        ('td', 6, 15.0, 0.3),         # 6 takedown attempts in 15min, prior 0.3/min
        ('td', 2, 5.0, 0.3),          # 2 takedown attempts in 5min, prior 0.3/min
        
        # Rare events
        ('sub', 2, 15.0, 0.1),        # 2 sub attempts in 15min, prior 0.1/min
        ('kd', 1, 15.0, 0.05),        # 1 knockdown in 15min, prior 0.05/min
        ('rev', 1, 15.0, 0.02),       # 1 reversal in 15min, prior 0.02/min
    ]
    
    print("\nSMOOTHING IMPACT BY STAT TYPE:")
    print("-" * 80)
    print(f"{'Stat':<8} {'Observed':<10} {'Duration':<8} {'Prior':<8} {'τ':<6} {'Smoothed':<10} {'Change':<8} {'Shrinkage':<10}")
    print("-" * 80)
    
    results = []
    
    for stat_type, observed, duration, prior_rate in scenarios:
        tau = current_tau.get(stat_type, current_tau['default'])
        
        smoothed, posterior_rate = pseudo_minutes_smoothing(
            observed, duration, prior_rate, tau
        )
        
        observed_rate = observed / duration
        change_pct = (smoothed - observed) / observed * 100
        
        # Shrinkage toward prior
        if abs(observed_rate - prior_rate) > 0.001:
            shrinkage = abs(observed_rate - posterior_rate) / abs(observed_rate - prior_rate) * 100
        else:
            shrinkage = 0.0
        
        print(f"{stat_type:<8} {observed:<10.0f} {duration:<8.1f} {prior_rate:<8.2f} {tau:<6.1f} "
              f"{smoothed:<10.1f} {change_pct:<8.1f}% {shrinkage:<10.1f}%")
        
        results.append({
            'stat_type': stat_type,
            'tau': tau,
            'observed': observed,
            'duration': duration,
            'prior_rate': prior_rate,
            'smoothed': smoothed,
            'change_pct': change_pct,
            'shrinkage_pct': shrinkage
        })
    
    return pd.DataFrame(results)

def compare_tau_values():
    """Compare different tau values for the same scenario."""
    
    print("\n\n🔄 TAU SENSITIVITY ANALYSIS")
    print("=" * 50)
    print("How does changing τ affect smoothing for the same scenario?")
    
    # Test scenario: moderate striking performance
    observed = 45  # strikes
    duration = 15.0  # minutes
    prior_rate = 3.0  # strikes per minute
    
    tau_values = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]
    
    print(f"\nScenario: {observed} strikes in {duration} minutes (observed rate: {observed/duration:.2f}/min)")
    print(f"Prior rate: {prior_rate}/min")
    print("-" * 60)
    print(f"{'τ':<6} {'Smoothed':<10} {'Rate':<10} {'Change':<10} {'Shrinkage':<10}")
    print("-" * 60)
    
    for tau in tau_values:
        smoothed, posterior_rate = pseudo_minutes_smoothing(
            observed, duration, prior_rate, tau
        )
        
        observed_rate = observed / duration
        change_pct = (smoothed - observed) / observed * 100 if observed != 0 else 0.0
        
        if abs(observed_rate - prior_rate) > 0.001:
            shrinkage = abs(observed_rate - posterior_rate) / abs(observed_rate - prior_rate) * 100
        else:
            shrinkage = 0.0
        
        print(f"{tau:<6.1f} {smoothed:<10.1f} {posterior_rate:<10.2f} {change_pct:<10.1f}% {shrinkage:<10.1f}%")

def tau_recommendations():
    """Provide tau tuning recommendations."""
    
    print("\n\n🎯 TAU TUNING RECOMMENDATIONS")
    print("=" * 50)
    
    print("""
GENERAL PRINCIPLES:
- Lower τ = More smoothing toward prior (less trust in observed data)
- Higher τ = Less smoothing (more trust in observed data)
- τ represents "pseudo-minutes" of prior evidence

STAT-SPECIFIC GUIDANCE:

1. HIGH-VOLUME STATS (sig_str, strikes, head):
   - Current τ=4.0 seems reasonable
   - Could try τ=2-6 range
   - These have lots of data, so moderate smoothing is appropriate
   
2. MODERATE-VOLUME STATS (body, leg, distance, clinch, ground):
   - Current τ=6.0 seems reasonable  
   - Could try τ=4-8 range
   - Balance between data and prior knowledge
   
3. GRAPPLING (td):
   - Current τ=8.0 might be slightly high
   - Could try τ=5-10 range
   - Takedowns are important but less frequent
   
4. RARE EVENTS (sub, kd, rev):
   - Current τ=10-12 is good starting point
   - Could try τ=8-15 range
   - High τ because these events are rare and noisy

TUNING STRATEGY:
1. Start with current values as baseline
2. Test τ ± 2-3 for each family
3. Measure downstream prediction performance
4. Focus on families that affect many features
5. Consider interaction effects between families

QUICK TEST RANGES:
- striking_volume (sig_str, strikes): [2, 3, 4, 5, 6]
- striking_targets (head, body, leg): [3, 4, 5, 6, 8] 
- striking_phases (distance, clinch, ground): [4, 5, 6, 8, 10]
- grappling (td): [5, 6, 8, 10, 12]
- rare_events (sub, kd, rev): [8, 10, 12, 15, 18]
""")

def main():
    """Run tau analysis."""
    
    # Analyze current behavior
    results_df = analyze_smoothing_behavior()
    
    # Compare tau sensitivity
    compare_tau_values()
    
    # Provide recommendations
    tau_recommendations()
    
    # Save results
    output_path = project_root / "data" / "tau_analysis.csv"
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Run: python scripts/quick_tau_tune.py")
    print(f"2. Review the smoothing impact on key stats")
    print(f"3. Choose the configuration with best balance")
    print(f"4. Update the tau values in PoissonGammaCalculator")

if __name__ == '__main__':
    main()
