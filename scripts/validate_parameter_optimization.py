"""
Manual validation script for the parameter optimization library.

Tests:
1. Parameter loader can load baseline parameters
2. Parameter loader can load optimized parameters (if available)
3. Cache manager validates cache correctly
4. Calculators can use param_loader
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from libs.parameter_optimization import (
    get_default_parameter_loader,
    get_parameter_mode,
    should_run_optimization
)
from libs.parameter_optimization.loaders import ParameterLoader
from libs.parameter_optimization.storage import JSONParameterStore
from libs.feature_store.calculators.beta_binomial_calc import BetaBinomialCalculator
from libs.feature_store.calculators.poisson_gamma_smoothing_calc import PoissonGammaCalculator
from libs.paths import database_url


def test_baseline_mode():
    """Test that baseline mode returns hardcoded parameters"""
    print("\n" + "="*60)
    print("TEST 1: Baseline Parameter Loading")
    print("="*60)

    # Force baseline mode
    os.environ['PARAM_MODE'] = 'baseline'

    store = JSONParameterStore()
    loader = ParameterLoader(store, mode='baseline')

    # Test beta-binomial parameters
    ko_tau = loader.get_beta_binomial_params('ko')
    print(f"[OK] Baseline ko tau: {ko_tau}")

    win_tau = loader.get_beta_binomial_params('win')
    print(f"[OK] Baseline win tau: {win_tau}")

    # Test poisson-gamma parameters
    sig_str_tau = loader.get_poisson_gamma_params('sig_str')
    print(f"[OK] Baseline sig_str tau: {sig_str_tau}")

    td_tau = loader.get_poisson_gamma_params('td')
    print(f"[OK] Baseline td tau: {td_tau}")

    # Test per-weightclass (should still return baseline in baseline mode)
    ko_tau_hw = loader.get_beta_binomial_params('ko', weightclass='heavyweight')
    print(f"[OK] Baseline ko tau (heavyweight): {ko_tau_hw}")

    print("\n[PASS] Baseline mode test PASSED")


def test_optimized_mode():
    """Test that optimized mode loads from JSON if available"""
    print("\n" + "="*60)
    print("TEST 2: Optimized Parameter Loading")
    print("="*60)

    # Force optimized mode
    os.environ['PARAM_MODE'] = 'optimized'

    store = JSONParameterStore()

    # Check if optimized parameters exist
    if not store.exists():
        print("[WARN] No optimized parameters found - this is OK for initial setup")
        print(f"       Expected file: {store.cache_path}")
        print("       Will fall back to baseline parameters")

        loader = ParameterLoader(store, mode='optimized')
        ko_tau = loader.get_beta_binomial_params('ko')
        print(f"[OK] Fallback ko tau: {ko_tau}")
        print("\n[PASS] Optimized mode test PASSED (using fallback)")
        return

    # Load optimized parameters
    loader = ParameterLoader(store, mode='optimized')

    # Test beta-binomial parameters
    ko_tau = loader.get_beta_binomial_params('ko')
    print(f"[OK] Optimized ko tau: {ko_tau}")

    win_tau = loader.get_beta_binomial_params('win')
    print(f"[OK] Optimized win tau: {win_tau}")

    # Test poisson-gamma parameters
    sig_str_tau = loader.get_poisson_gamma_params('sig_str')
    print(f"[OK] Optimized sig_str tau: {sig_str_tau}")

    # Test per-weightclass parameters if available
    ko_tau_hw = loader.get_beta_binomial_params('ko', weightclass='heavyweight')
    print(f"[OK] Optimized ko tau (heavyweight): {ko_tau_hw}")

    # Show metadata
    params = store.load()
    if params and 'metadata' in params:
        metadata = params['metadata']
        print(f"\n[INFO] Optimization Metadata:")
        print(f"       Training period: {metadata.get('training_period', 'N/A')}")
        print(f"       Fights: {metadata.get('n_fights', 'N/A')}")
        print(f"       Optimized at: {metadata.get('optimized_at', 'N/A')}")

    print("\n[PASS] Optimized mode test PASSED")


def test_cache_validation():
    """Test cache validation logic"""
    print("\n" + "="*60)
    print("TEST 3: Cache Validation")
    print("="*60)

    # Get database connection
    db_url = os.getenv('DATABASE_URL', database_url())
    engine = create_engine(db_url)

    with engine.connect() as conn:
        should_optimize, reason = should_run_optimization(conn)

        print(f"Should optimize: {should_optimize}")
        print(f"Reason: {reason}")

        if should_optimize:
            print("\n[WARN] Cache is invalid or missing - optimization will run on next main.py execution")
        else:
            print("\n[OK] Cache is valid - will use cached parameters")

    print("\n[PASS] Cache validation test PASSED")


def test_calculator_integration():
    """Test that calculators can use param_loader"""
    print("\n" + "="*60)
    print("TEST 4: Calculator Integration")
    print("="*60)

    # Get database connection
    db_url = os.getenv('DATABASE_URL', database_url())
    engine = create_engine(db_url)

    # Force baseline mode for testing
    os.environ['PARAM_MODE'] = 'baseline'

    with engine.connect() as conn:
        # Test BetaBinomialCalculator with param_loader
        print("Creating BetaBinomialCalculator with param_loader...")
        loader = get_default_parameter_loader()
        bb_calc = BetaBinomialCalculator(conn, param_loader=loader)
        print(f"[OK] BetaBinomialCalculator initialized")
        print(f"     Weight classes loaded: {len(bb_calc.weight_classes)}")

        # Test parameter retrieval
        test_tau = bb_calc._get_pseudo_count('ko', weightclass='heavyweight')
        print(f"[OK] Retrieved ko tau for heavyweight: {test_tau}")

        # Test PoissonGammaCalculator with param_loader
        print("\nCreating PoissonGammaCalculator with param_loader...")
        pg_calc = PoissonGammaCalculator(conn, param_loader=loader)
        print(f"[OK] PoissonGammaCalculator initialized")
        print(f"     Weight classes loaded: {len(pg_calc.weight_classes)}")

        # Test parameter retrieval
        test_tau = pg_calc._get_pseudo_minutes('sig_str', weightclass='lightweight')
        print(f"[OK] Retrieved sig_str tau for lightweight: {test_tau}")

    print("\n[PASS] Calculator integration test PASSED")


def main():
    """Run all validation tests"""
    print("="*60)
    print("PARAMETER OPTIMIZATION LIBRARY VALIDATION")
    print("="*60)

    try:
        # Test 1: Baseline mode
        test_baseline_mode()

        # Test 2: Optimized mode
        test_optimized_mode()

        # Test 3: Cache validation
        test_cache_validation()

        # Test 4: Calculator integration
        test_calculator_integration()

        # Summary
        print("\n" + "="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        print("\nParameter optimization library is working correctly!")
        print("\nUsage:")
        print("  - Set PARAM_MODE=baseline to use hardcoded parameters")
        print("  - Set PARAM_MODE=optimized to use optimized parameters (default)")
        print("  - Set FORCE_REOPTIMIZE=1 to force reoptimization")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
