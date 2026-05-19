"""Base class for feature enrichers."""

from abc import ABC, abstractmethod
import pandas as pd


class Enricher(ABC):
    """Abstract base class for enriching DataFrames with additional features."""
    
    @abstractmethod
    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add additional features to the DataFrame.
        
        Args:
            df: DataFrame to enrich
            
        Returns:
            Enriched DataFrame
        """
        pass

