"""
Apply per-weightclass parameters from latest tuning to all calculators.
This script generates the correct per-weightclass dictionaries.
"""

import json
from pathlib import Path

def load_tuning_results():
    """Load optimized parameters."""
    path = Path('data/comprehensive_tuning/optimized_parameters.json')
    with open(path) as f:
        return json.load(f)

def generate_beta_binomial_per_class(results):
    """Generate per-weightclass code for beta-binomial."""
    per_class = results['beta_binomial']['per_weightclass']

    if not per_class:
        return """        # No per-weightclass parameters (all use global)
        self.per_weightclass_pseudo_counts = {}
"""

    code = """        # Per-weight class pseudo-count parameters
        # Based on comprehensive likelihood optimization with proper cross-validation
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        # Only includes parameters with >=0.5% improvement over global
        self.per_weightclass_pseudo_counts = {
"""

    for wc, stats in per_class.items():
        code += f"            '{wc}': {{\n"
        for stat, tau in stats.items():
            code += f"                '{stat}': {tau:.2f},\n"
        code += "                'default': 15.4\n"
        code += "            },\n"

    code += "        }\n"
    return code

def generate_poisson_gamma_per_class(results):
    """Generate per-weightclass code for poisson-gamma."""
    per_class = results['poisson_gamma']['per_weightclass']

    if not per_class:
        return """        # No per-weightclass parameters (all use global)
        self.per_weightclass_pseudo_minutes = {}
"""

    code = """        # Per-weight class pseudo-minutes (statistically optimized)
        # Based on comprehensive likelihood optimization with proper cross-validation
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        # Only includes parameters with >=0.5% improvement over global
        self.per_weightclass_pseudo_minutes = {
"""

    for wc, stats in sorted(per_class.items()):
        code += f"            '{wc}': {{\n"
        for stat, tau in sorted(stats.items()):
            code += f"                '{stat}': {tau:.2f},\n"
        code += "                'default': 8.0\n"
        code += "            },\n"

    code += "        }\n"
    return code

def generate_accuracy_per_class(results):
    """Generate per-weightclass code for accuracy."""
    per_class = results['accuracy']['per_weightclass']

    if not per_class:
        return """        # No per-weightclass parameters (all use global)
        self.per_weightclass_acc_tau = {}
"""

    code = """        # Per-weight class tau (pseudo-count) for accuracy smoothing
        # Based on comprehensive likelihood optimization with proper cross-validation
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        # Only includes parameters with >=0.5% improvement over global
        self.per_weightclass_acc_tau = {
"""

    for wc, stats in sorted(per_class.items()):
        code += f"            '{wc}': {{\n"
        for stat, tau in sorted(stats.items()):
            code += f"                '{stat}': {tau:.2f},\n"
        code += "                'default': 12.0\n"
        code += "            },\n"

    code += "        }\n"
    return code

def main():
    results = load_tuning_results()

    print("=" * 80)
    print("PER-WEIGHTCLASS PARAMETER CODE GENERATION")
    print("=" * 80)
    print()

    print("1. Beta-Binomial Per-Weightclass Parameters:")
    print("=" * 80)
    print(generate_beta_binomial_per_class(results))
    print()

    print("2. Poisson-Gamma Per-Weightclass Parameters:")
    print("=" * 80)
    print(generate_poisson_gamma_per_class(results))
    print()

    print("3. Accuracy Per-Weightclass Parameters:")
    print("=" * 80)
    print(generate_accuracy_per_class(results))
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Per-weightclass parameters from tuning:")
    print(f"  Beta-Binomial: {len(results['beta_binomial']['per_weightclass'])} weight classes")
    print(f"  Poisson-Gamma: {len(results['poisson_gamma']['per_weightclass'])} weight classes")
    print(f"  Accuracy: {len(results['accuracy']['per_weightclass'])} weight classes")
    print()
    print("These should REPLACE the existing per_weightclass dictionaries in:")
    print("  - libs/feature_store/calculators/beta_binomial_calc.py")
    print("  - libs/feature_store/calculators/poisson_gamma_smoothing_calc.py")
    print("  - libs/feature_store/calculators/acc_calc.py")

if __name__ == "__main__":
    main()
