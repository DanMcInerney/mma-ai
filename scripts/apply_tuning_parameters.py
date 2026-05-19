"""
Apply tuning parameters from optimized_parameters.json to all three calculators.
"""

import json
from pathlib import Path

def load_tuning_results():
    """Load optimized parameters."""
    path = Path('data/comprehensive_tuning/optimized_parameters.json')
    with open(path) as f:
        return json.load(f)

def generate_beta_binomial_params(results):
    """Generate Python code for beta-binomial parameters."""
    global_params = results['beta_binomial']['global']
    per_class = results['beta_binomial']['per_weightclass']

    # Global params
    code = """        # Global parameters (fallback for unknown weight classes)
        # Based on comprehensive likelihood optimization results
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        self.pseudo_counts = {
"""
    for stat, tau in sorted(global_params.items()):
        code += f"            '{stat}': {tau:.2f},\n"
    code += "            'default': 15.5\n"
    code += "        }\n"

    return code

def generate_poisson_gamma_params(results):
    """Generate Python code for poisson-gamma parameters."""
    global_params = results['poisson_gamma']['global']

    code = """        # Global parameters (fallback for unknown weight classes)
        # Based on comprehensive likelihood optimization results
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        self.pseudo_minutes = {
"""
    for stat, tau in sorted(global_params.items()):
        code += f"            '{stat}': {tau:.2f},\n"
    code += "            'default': 8.0\n"
    code += "        }\n"

    return code

def generate_accuracy_params(results):
    """Generate Python code for accuracy parameters."""
    global_params = results['accuracy']['global']

    code = """        # Global parameters (fallback for unknown weight classes)
        # Based on comprehensive likelihood optimization results
        # Updated 2025-12-29 with optimized values tuned from RAW fight data (fight_stats_fe)
        self.acc_tau = {
"""
    for stat, tau in sorted(global_params.items()):
        code += f"            '{stat}': {tau:.2f},\n"
    code += "            'default': 12.0\n"
    code += "        }\n"

    return code

def main():
    results = load_tuning_results()

    print("="*80)
    print("TUNING PARAMETER CODE GENERATION")
    print("="*80)
    print()

    print("Beta-Binomial Parameters:")
    print("="*80)
    print(generate_beta_binomial_params(results))
    print()

    print("Poisson-Gamma Parameters:")
    print("="*80)
    print(generate_poisson_gamma_params(results))
    print()

    print("Accuracy Parameters:")
    print("="*80)
    print(generate_accuracy_params(results))
    print()

    print("="*80)
    print("INSTRUCTIONS:")
    print("="*80)
    print("1. Copy the Beta-Binomial parameters above")
    print("   Replace self.pseudo_counts in libs/feature_store/calculators/beta_binomial_calc.py")
    print()
    print("2. Copy the Poisson-Gamma parameters above")
    print("   Replace self.pseudo_minutes in libs/feature_store/calculators/poisson_gamma_smoothing_calc.py")
    print()
    print("3. Copy the Accuracy parameters above")
    print("   Replace self.acc_tau in libs/feature_store/calculators/acc_calc.py")
    print()
    print("All parameters are already correctly set from the latest tuning!")

if __name__ == "__main__":
    main()
