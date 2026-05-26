"""
Verify that Toshiomi Kazama's striking defense smoothing is fixed with new parameters.

The original issue:
- Kazama had 5.6% actual striking defense (absorbed 104 of 110 significant strikes)
- This was smoothed to 69.9% due to over-smoothing (tau=18)
- Created misleading adjperf of +3.83

With new parameters optimized from raw data, we should see more accurate smoothing.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from sqlalchemy import create_engine, text
from libs.feature_store.calculators.acc_calc import AccuracyCalculator
from libs.paths import database_url

def main():
    # Connect to database
    engine = create_engine(database_url())

    # Query Kazama's fight data
    query = text("""
        SELECT
            f.fighter_id,
            f.fighter_name,
            fe.fight_id,
            fe.sig_str_att_opponent,
            fe.sig_str_land_opponent,
            fe.fight_time_seconds
        FROM features.fighter_metadata f
        JOIN features.fight_stats_fe fe ON f.fighter_id = fe.fighter_id
        WHERE f.fighter_name = 'Toshiomi Kazama'
        ORDER BY fe.fight_id
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        print("No data found for Toshiomi Kazama")
        return

    print(f"\n{'='*80}")
    print(f"Toshiomi Kazama Striking Defense Analysis")
    print(f"{'='*80}\n")

    # Calculate actual statistics
    total_att = df['sig_str_att_opponent'].sum()
    total_land = df['sig_str_land_opponent'].sum()
    total_time = df['fight_time_seconds'].sum()
    actual_defense = 100 * (1 - total_land / total_att) if total_att > 0 else 0

    print(f"Raw Statistics:")
    print(f"  Fights: {len(df)}")
    print(f"  Total opponent attempts: {total_att}")
    print(f"  Total opponent landed: {total_land}")
    print(f"  Actual defense: {actual_defense:.2f}%")
    print(f"  Total fight time: {total_time/60:.1f} minutes")

    # Calculate smoothed defense using new parameters
    calc = AccuracyCalculator()

    # The calculator expects certain columns, let's prepare the data properly
    # We need to provide the attempts and landed as accuracy (landed/attempted)
    if total_att > 0:
        raw_acc = total_land / total_att  # This is opponent's accuracy
        defense = 1 - raw_acc  # Our defense is 1 - opponent's accuracy

        # Get the tau for sig_str_acc from the calculator
        tau = calc.acc_tau.get('sig_str', calc.acc_tau['default'])

        # The prior for defense is typically around UFC average (~50-60% defense)
        # Let's use 0.55 (55% defense) as the prior
        prior_defense = 0.55

        # Calculate smoothed defense using beta-binomial conjugate prior
        # We're modeling "absorbed" vs "attempted"
        absorbed = total_att - total_land

        # With tau pseudo-observations, we add tau * prior successes and tau total attempts
        smoothed_absorbed = absorbed + (tau * prior_defense)
        smoothed_attempts = total_att + tau
        smoothed_defense_pct = 100 * (smoothed_absorbed / smoothed_attempts)

        print(f"\nSmoothing Parameters:")
        print(f"  Tau (pseudo-observations): {tau:.2f}")
        print(f"  Prior defense: {prior_defense*100:.1f}%")

        print(f"\nSmoothed Defense Calculation:")
        print(f"  Raw absorbed: {absorbed}")
        print(f"  Pseudo absorbed: {tau * prior_defense:.2f}")
        print(f"  Total absorbed: {smoothed_absorbed:.2f}")
        print(f"  Total attempts: {smoothed_attempts:.2f}")
        print(f"  Smoothed defense: {smoothed_defense_pct:.2f}%")

        print(f"\nComparison:")
        print(f"  Actual defense: {actual_defense:.2f}%")
        print(f"  Smoothed defense (NEW params): {smoothed_defense_pct:.2f}%")
        print(f"  Old smoothed defense (tau=18): ~69.9%")
        print(f"  Smoothing effect: {smoothed_defense_pct - actual_defense:+.2f}%")

        # Calculate what it would have been with old tau=18
        old_tau = 18.0
        old_smoothed_absorbed = absorbed + (old_tau * prior_defense)
        old_smoothed_attempts = total_att + old_tau
        old_smoothed_defense_pct = 100 * (old_smoothed_absorbed / old_smoothed_attempts)

        print(f"\n  Old smoothed defense (tau=18): {old_smoothed_defense_pct:.2f}%")
        print(f"  Old smoothing effect: {old_smoothed_defense_pct - actual_defense:+.2f}%")

        print(f"\nImprovement:")
        print(f"  Reduction in over-smoothing: {(old_smoothed_defense_pct - actual_defense) - (smoothed_defense_pct - actual_defense):.2f}%")

        # Analyze whether this is still over-smoothing
        if smoothed_defense_pct > actual_defense + 10:
            print(f"\n[WARNING] Still significant over-smoothing (>{actual_defense + 10:.1f}%)")
        elif smoothed_defense_pct > actual_defense + 5:
            print(f"\n[CAUTION] Moderate over-smoothing (>{actual_defense + 5:.1f}%)")
        else:
            print(f"\n[OK] Smoothing is reasonable (within 5% of actual)")

if __name__ == "__main__":
    main()
