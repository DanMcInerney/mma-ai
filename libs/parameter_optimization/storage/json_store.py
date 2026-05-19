"""
JSON-based parameter storage.

Reads and writes optimized parameters to/from JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from config.parameters import CACHE_PATH

logger = logging.getLogger(__name__)


class JSONParameterStore:
    """
    Stores and retrieves optimized parameters from JSON file.

    File format (matches comprehensive_likelihood_tuner.py output):
    {
        "metadata": {
            "training_period": "2014-01-01 to 2023-01-01",
            "n_stats_tuned": 49,
            "n_weight_classes": 8,
            "total_optimizations": 392
        },
        "beta_binomial": {
            "global": {"ko": 7.29, ...},
            "per_weightclass": {"flyweight": {"ko": 22.44}, ...}
        },
        "poisson_gamma": {
            "global": {"sig_str": 0.98, ...},
            "per_weightclass": {"heavyweight": {"sub": 19.17}, ...}
        }
    }
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize JSON parameter store.

        Args:
            cache_path: Path to JSON file (defaults to config.CACHE_PATH)
        """
        self.cache_path = cache_path or CACHE_PATH

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load parameters from JSON file.

        Returns:
            Dictionary with parameters, or None if file doesn't exist

        Raises:
            json.JSONDecodeError: If file contains invalid JSON
            IOError: If file cannot be read
        """
        if not self.cache_path.exists():
            logger.info(f"Parameter cache not found at {self.cache_path}")
            return None

        try:
            with open(self.cache_path, 'r') as f:
                params = json.load(f)
            logger.info(f"Loaded parameters from {self.cache_path}")
            return params
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {self.cache_path}: {e}")
            raise
        except IOError as e:
            logger.error(f"Error reading {self.cache_path}: {e}")
            raise

    def save(self, params: Dict[str, Any]) -> None:
        """
        Save parameters to JSON file.

        Args:
            params: Dictionary with parameters to save

        Raises:
            IOError: If file cannot be written
        """
        # Ensure directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.cache_path, 'w') as f:
                json.dump(params, f, indent=2)
            logger.info(f"Saved parameters to {self.cache_path}")
        except IOError as e:
            logger.error(f"Error writing to {self.cache_path}: {e}")
            raise

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata from cached parameters.

        Returns:
            Metadata dictionary, or None if file doesn't exist
        """
        params = self.load()
        if params is None:
            return None
        return params.get('metadata', {})

    def exists(self) -> bool:
        """Check if parameter file exists."""
        return self.cache_path.exists()
