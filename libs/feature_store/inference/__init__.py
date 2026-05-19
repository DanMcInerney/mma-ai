"""
Inference data creation module.

This module provides a refactored, highly organized system for creating
inference data for UFC fight prediction models.
"""

from libs.feature_store.inference.builder import InferenceDataBuilder
from libs.feature_store.inference.feature_filter import filter_features_for_model

__all__ = ['InferenceDataBuilder', 'filter_features_for_model']

