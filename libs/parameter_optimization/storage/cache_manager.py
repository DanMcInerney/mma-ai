"""
Cache validation manager for parameter optimization.

Determines whether cached parameters are still valid or need recomputation.
"""

import os
import logging
from typing import Tuple
from sqlalchemy import text

from config.parameters import TRAINING_PERIOD, TRAINING_START_DATE, TRAINING_END_DATE, FIGHT_COUNT_TOLERANCE
from .json_store import JSONParameterStore

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Validates cached optimization parameters.

    Checks:
    1. Cache file exists
    2. Training period matches expected period
    3. Fight count hasn't changed significantly
    4. FORCE_REOPTIMIZE environment variable not set
    """

    def __init__(self, store: JSONParameterStore, conn):
        """
        Initialize cache manager.

        Args:
            store: JSONParameterStore instance
            conn: Database connection for validation queries
        """
        self.store = store
        self.conn = conn

    def _get_fight_count(self) -> int:
        """
        Get current count of fights in training period.

        Returns:
            Number of fights in database for training period
        """
        query = text("""
        SELECT COUNT(DISTINCT fm.fight_id)
        FROM features.fight_mapping fm
        JOIN features.event_mapping em ON fm.event_id = em.event_id
        WHERE em.event_date >= :start_date
          AND em.event_date < :end_date
        """)

        result = self.conn.execute(
            query,
            {
                'start_date': TRAINING_START_DATE,
                'end_date': TRAINING_END_DATE
            }
        ).fetchone()

        return int(result[0]) if result else 0

    def is_cache_valid(self) -> Tuple[bool, str]:
        """
        Check if cached parameters are still valid.

        Validation checks:
        1. File exists
        2. Training period matches (2014-2023)
        3. Fight count within ±100 of cached
        4. FORCE_REOPTIMIZE not set

        Returns:
            Tuple of (is_valid, reason)
            - is_valid: True if cache is valid and can be used
            - reason: String explaining the validation result
        """
        # Check for forced reoptimization
        if os.getenv('FORCE_REOPTIMIZE') == '1':
            return False, "Forced reoptimization requested (FORCE_REOPTIMIZE=1)"

        # Check if cache file exists
        if not self.store.exists():
            return False, "No cached parameters found"

        try:
            # Load metadata
            metadata = self.store.get_metadata()
            if metadata is None:
                return False, "Cache file has no metadata"

            # Check training period
            cached_period = metadata.get('training_period')
            if cached_period != TRAINING_PERIOD:
                return False, f"Training period mismatch: cached='{cached_period}', expected='{TRAINING_PERIOD}'"

            # Check fight count (simple staleness detection)
            current_count = self._get_fight_count()
            cached_count = metadata.get('n_fights', 0)

            # Allow some tolerance for minor data updates
            count_diff = abs(current_count - cached_count)
            if count_diff > FIGHT_COUNT_TOLERANCE:
                return False, f"Fight count changed significantly: {cached_count} → {current_count} (Δ={count_diff})"

            # All checks passed
            logger.info(f"Cache is valid: {current_count} fights (cached: {cached_count})")
            return True, f"Cache is valid ({current_count} fights, last optimized for {cached_period})"

        except Exception as e:
            logger.error(f"Error validating cache: {e}")
            return False, f"Cache validation error: {e}"
