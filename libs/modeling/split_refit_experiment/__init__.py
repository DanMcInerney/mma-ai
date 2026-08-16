"""Fail-closed protocol seams for the split/refit campaign."""

from .protocol import ProtocolError, verify_split
from .registry import RegistryError, validate_registry

__all__ = ["ProtocolError", "RegistryError", "validate_registry", "verify_split"]
