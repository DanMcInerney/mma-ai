#!/usr/bin/env python3
"""
Manual validation script for StatQualityCalculator

Runs the calculator on a single table (td) to verify it works correctly
before running on all tables.
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from libs.feature_store.calculators.stat_quality_calc import StatQualityCalculator
from libs.feature_store.calculator_context import CalculatorContext
from libs.paths import database_url, no_winsor_database_url

DB_URL = no_winsor_database_url()


def main():
    """Test StatQualityCalculator on the td (takedown) table."""
    print("=" * 80)
    print("Testing StatQualityCalculator on 'td' table")
    print("=" * 80)

    # Create database connection
    if not database_exists(DB_URL):
        print(f"ERROR: Database does not exist at {DB_URL}")
        sys.exit(1)

    engine = create_engine(DB_URL, pool_size=10, max_overflow=20)

    with engine.connect() as conn:
        # Create context and calculator
        context = CalculatorContext(conn)
        calc = StatQualityCalculator(context)

        # Manually create quality metrics for just the 'td' table
        print("\n1. Creating quality metrics for 'td' table...")
        try:
            calc._create_quality_metrics_table('td')
            print("   [OK] Quality metrics table created")
        except Exception as e:
            print(f"   [ERROR] {e}")
            raise

        # Validate the results
        print("\n2. Validating quality metrics...")
        try:
            calc._validate_quality_metrics('td')
            print("   [OK] Validation complete")
        except Exception as e:
            print(f"   [ERROR] {e}")
            raise

        # Load and display results
        print("\n3. Loading results...")
        try:
            df = pd.read_sql("SELECT * FROM features.td_quality_metrics", conn)
            print(f"   [OK] Loaded {len(df)} quality metrics")

            # Show summary statistics
            print("\n" + "=" * 80)
            print("QUALITY METRICS SUMMARY")
            print("=" * 80)

            # Overall counts
            print(f"\nTotal metrics: {len(df)}")
            print(f"Unique stats: {df['stat_name'].nunique()}")
            print(f"Weight classes: {df['weightclass'].nunique()}")

            # Quality tier breakdown
            print("\nQuality Tier Distribution:")
            tier_counts = df['quality_tier'].value_counts()
            for tier, count in tier_counts.items():
                pct = 100 * count / len(df)
                print(f"  {tier:12s}: {count:3d} ({pct:5.1f}%)")

            # Show degenerate cases
            degenerate = df[df['quality_tier'] == 'degenerate'].sort_values('mode_frequency', ascending=False)
            if len(degenerate) > 0:
                print(f"\nDegenerate Cases ({len(degenerate)}):")
                print("-" * 80)
                for _, row in degenerate.iterrows():
                    print(f"  {row['stat_name']:20s} ({row['weightclass']:15s}): "
                          f"MAD={row['wc_mad']:.6f}, mode_freq={row['mode_frequency']:.1%}, "
                          f"winsor_limit=+/-{row['recommended_winsor_limit']:.1f}")

            # Show sparse cases
            sparse = df[df['quality_tier'] == 'sparse'].sort_values('wc_mad')
            if len(sparse) > 0:
                print(f"\nSparse Cases ({len(sparse)}):")
                print("-" * 80)
                for _, row in sparse.head(10).iterrows():
                    print(f"  {row['stat_name']:20s} ({row['weightclass']:15s}): "
                          f"MAD={row['wc_mad']:.6f}, "
                          f"winsor_limit=+/-{row['recommended_winsor_limit']:.2f}")

            # Verify heavyweight TD defense is marked as degenerate
            print("\n" + "=" * 80)
            print("SPECIFIC TEST: Heavyweight TD Defense")
            print("=" * 80)

            hw_td_def = df[(df['stat_name'] == 'td_def') & (df['weightclass'] == 'heavyweight')]
            if len(hw_td_def) > 0:
                row = hw_td_def.iloc[0]
                print(f"\nStat: {row['stat_name']}")
                print(f"Weightclass: {row['weightclass']}")
                print(f"wc_mad: {row['wc_mad']:.6f}")
                print(f"Mode frequency: {row['mode_frequency']:.1%}")
                print(f"Quality tier: {row['quality_tier']}")
                print(f"Recommended winsor limit: +/-{row['recommended_winsor_limit']:.1f}")
                print(f"Reliability score: {row['reliability_score']:.1f}")
                print(f"Effective N: {row['effective_n']}")

                # Verify it's classified as degenerate
                if row['quality_tier'] == 'degenerate':
                    print("\n[PASS] Heavyweight TD defense correctly classified as degenerate")
                else:
                    print(f"\n[FAIL] Expected 'degenerate', got '{row['quality_tier']}'")
            else:
                print("\n[FAIL] Heavyweight TD defense not found in results")

            # Save sample to CSV for inspection
            output_file = project_root / 'data' / 'test_td_quality_metrics.csv'
            df.to_csv(output_file, index=False)
            print(f"\n[OK] Full results saved to: {output_file}")

        except Exception as e:
            print(f"   [ERROR] {e}")
            raise

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
