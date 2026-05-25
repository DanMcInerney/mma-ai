"""Small helpers for optional dashboard LLM integrations."""

from __future__ import annotations

import os


GOOGLE_API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")


def google_api_key() -> str | None:
    for env_var in GOOGLE_API_KEY_ENV_VARS:
        value = os.getenv(env_var)
        if value:
            return value
    return None


def google_api_key_hint() -> str:
    return "Connect GEMINI_API_KEY or GOOGLE_API_KEY"
