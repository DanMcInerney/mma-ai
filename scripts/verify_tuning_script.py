"""
Verify the tuning script is correctly configured.
Checks for common issues after git operations.
"""

import re
from pathlib import Path

def check_file(filepath, checks):
    """Run checks on a file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    results = []
    for check_name, pattern, should_exist in checks:
        match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
        if should_exist:
            status = "OK" if match else "FAIL"
            results.append((check_name, status, match))
        else:
            status = "OK" if not match else "FAIL"
            results.append((check_name, status, match))

    return results

def main():
    print("=" * 80)
    print("TUNING SCRIPT VERIFICATION")
    print("=" * 80)
    print()

    tuning_script = Path('tuning/comprehensive_likelihood_tuner.py')

    if not tuning_script.exists():
        print(f"ERROR: {tuning_script} not found!")
        return

    checks = [
        # Critical: Using RAW data table
        ("Uses fight_stats_fe (discover_stats)",
         r"def discover_stats.*?table: str = 'fight_stats_fe'",
         True),

        ("Uses fight_stats_fe (load_training_data)",
         r"def load_training_data.*?table: str = 'fight_stats_fe'",
         True),

        # Critical: Lowercase weight classes
        ("Lowercase weight classes",
         r"'flyweight', 'bantamweight', 'featherweight', 'lightweight'",
         True),

        # Critical: JSON serialization fix
        ("JSON bool conversion (use_per_class)",
         r"result_dict\['use_per_class'\] = bool\(result_dict\['use_per_class'\]\)",
         True),

        ("JSON bool conversion (boundary_hit)",
         r"result_dict\['boundary_hit'\] = bool\(result_dict\['boundary_hit'\]\)",
         True),

        ("JSON tuple to list conversion",
         r"result_dict\['search_range'\] = list\(result_dict\['search_range'\]\)",
         True),

        # Should NOT use wrong table
        ("NOT using fight_stats_derived",
         r"table: str = 'fight_stats_derived'",
         False),

        # Should NOT use capitalized weight classes
        ("NOT using capitalized weightclasses",
         r"'Flyweight'|'Bantamweight'",
         False),
    ]

    results = check_file(tuning_script, checks)

    print("Critical Checks:")
    print("-" * 80)

    all_passed = True
    for check_name, status, match in results:
        symbol = "[OK]" if status == "OK" else "[FAIL]"
        print(f"{symbol} {check_name}")
        if status == "FAIL":
            all_passed = False
            if match:
                print(f"     Found: {match.group(0)[:100]}")

    print()
    print("=" * 80)

    if all_passed:
        print("STATUS: ALL CHECKS PASSED")
        print()
        print("The tuning script is correctly configured:")
        print("  - Uses RAW data from fight_stats_fe")
        print("  - Uses lowercase weight class names")
        print("  - Has JSON serialization bug fixes")
        print()
        print("Ready to run tuning!")
    else:
        print("STATUS: SOME CHECKS FAILED")
        print()
        print("Please review the failed checks above.")
        print("The script may need to be updated.")

    print("=" * 80)

if __name__ == "__main__":
    main()
