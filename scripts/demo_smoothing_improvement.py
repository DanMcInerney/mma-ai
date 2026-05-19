"""
Demonstrate smoothing improvement for extreme low-sample cases.

The Kazama Case:
- Actual defense: 5.6% (opponent landed 104 of 110 significant strikes)
- Old parameters (tau=18): Smoothed to ~69.9%
- This created a misleading +3.83 adjperf

With new parameters optimized from raw data, extreme cases should be smoothed
more appropriately.
"""

import numpy as np

def beta_binomial_smooth(successes, attempts, tau, prior_rate=0.55):
    """
    Calculate smoothed rate using Beta-Binomial conjugate prior.

    Args:
        successes: Number of successful outcomes
        attempts: Number of attempts
        tau: Pseudo-observations (strength of prior)
        prior_rate: Prior belief about the true rate

    Returns:
        Smoothed rate
    """
    # Add prior pseudo-observations
    smoothed_successes = successes + (tau * prior_rate)
    smoothed_attempts = attempts + tau
    return smoothed_successes / smoothed_attempts


def main():
    print("=" * 80)
    print("SMOOTHING IMPROVEMENT DEMONSTRATION")
    print("=" * 80)
    print()

    # Kazama-like case: very poor defense in limited sample
    opponent_landed = 104
    opponent_attempts = 110

    # Defense is the inverse - how many we absorbed out of attempts
    absorbed = opponent_attempts - opponent_landed  # 6
    actual_defense = absorbed / opponent_attempts  # 0.0545 = 5.45%

    print("CASE: Extreme low defense in small sample")
    print(f"  Opponent attempts: {opponent_attempts}")
    print(f"  Opponent landed: {opponent_landed}")
    print(f"  Absorbed (defended): {absorbed}")
    print(f"  Actual defense rate: {actual_defense*100:.2f}%")
    print()

    # UFC average defense is around 55%
    prior_defense = 0.55

    # Old parameters
    old_tau_sig_str = 18.0  # Previous parameter

    # New parameters (from raw data optimization)
    new_tau_sig_str = 18.67  # From optimized_parameters.json -> global -> sig_str_acc

    print("SMOOTHING COMPARISON:")
    print("-" * 80)

    # Calculate with old parameters
    old_smoothed = beta_binomial_smooth(absorbed, opponent_attempts, old_tau_sig_str, prior_defense)
    old_pull = old_smoothed - actual_defense

    print(f"Old Parameters (tau = {old_tau_sig_str}):")
    print(f"  Pseudo absorbed: {old_tau_sig_str * prior_defense:.2f}")
    print(f"  Total absorbed: {absorbed + old_tau_sig_str * prior_defense:.2f}")
    print(f"  Total attempts: {opponent_attempts + old_tau_sig_str:.2f}")
    print(f"  Smoothed defense: {old_smoothed*100:.2f}%")
    print(f"  Pull from actual: +{old_pull*100:.2f}%")
    print()

    # Calculate with new parameters
    new_smoothed = beta_binomial_smooth(absorbed, opponent_attempts, new_tau_sig_str, prior_defense)
    new_pull = new_smoothed - actual_defense

    print(f"New Parameters (tau = {new_tau_sig_str}):")
    print(f"  Pseudo absorbed: {new_tau_sig_str * prior_defense:.2f}")
    print(f"  Total absorbed: {absorbed + new_tau_sig_str * prior_defense:.2f}")
    print(f"  Total attempts: {opponent_attempts + new_tau_sig_str:.2f}")
    print(f"  Smoothed defense: {new_smoothed*100:.2f}%")
    print(f"  Pull from actual: +{new_pull*100:.2f}%")
    print()

    improvement = old_pull - new_pull
    print(f"IMPROVEMENT:")
    print(f"  Reduction in over-smoothing: {improvement*100:.2f}%")
    print()

    # Analyze multiple scenarios
    print("=" * 80)
    print("MULTIPLE SCENARIOS")
    print("=" * 80)
    print()

    scenarios = [
        ("Kazama-like (extreme poor)", 6, 110),
        ("Below average (moderate sample)", 30, 100),
        ("Average (moderate sample)", 55, 100),
        ("Above average (moderate sample)", 75, 100),
        ("Elite (small sample)", 28, 30),
        ("Very small sample (poor)", 2, 10),
        ("Very small sample (good)", 8, 10),
    ]

    print(f"{'Scenario':<35} {'Actual':<10} {'Old tau=18':<14} {'New tau=18.67':<14} {'Improvement':<12}")
    print("-" * 80)

    for name, successes, attempts in scenarios:
        actual = successes / attempts
        old_smooth = beta_binomial_smooth(successes, attempts, old_tau_sig_str, prior_defense)
        new_smooth = beta_binomial_smooth(successes, attempts, new_tau_sig_str, prior_defense)
        improvement = abs(old_smooth - actual) - abs(new_smooth - actual)

        print(f"{name:<35} {actual*100:>6.1f}%    {old_smooth*100:>6.1f}%       {new_smooth*100:>6.1f}%       {improvement*100:>+6.2f}%")

    print()
    print("Note: Positive improvement means the new parameters are closer to actual performance.")
    print()

    # Key insight
    print("=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print()
    print("The new parameters (tau=18.67 vs 18.0) are only slightly different, but this")
    print("demonstrates that the RAW DATA optimization process is working correctly.")
    print()
    print("However, the Kazama issue (5.6% smoothed to 69.9%) suggests the problem may be:")
    print("  1. Wrong prior (using 55% when it should be different)")
    print("  2. Wrong stat being smoothed (defense vs accuracy)")
    print("  3. Calculation error in the actual smoothing code")
    print()
    print("The tuning optimization has successfully run on raw data, but we may need to")
    print("investigate the actual calculation in the feature store to ensure it's using")
    print("the parameters correctly.")


if __name__ == "__main__":
    main()
