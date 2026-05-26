"""Shared filesystem and environment defaults for local workflows."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
UFCSTATS_DATA_DIR = RAW_DATA_DIR / "ufcstats"
MODELS_DIR = PROJECT_ROOT / "AutogluonModels"
PICKS_DIR = PROJECT_ROOT / "pics" / "picks"

DEFAULT_DATABASE_URL = "postgresql://postgres@localhost:5432/mma-ai"
DEFAULT_ODDS_DATABASE_URL = "postgresql://postgres@localhost:5432/odds"
DEFAULT_NO_WINSOR_DATABASE_URL = "postgresql://postgres@localhost:5432/mma-ai-no-winsor"


def load_project_env(root: Path | None = None) -> bool:
    """Load the repo .env file for local CLI/web commands without overriding shell env."""
    env_file = (root or PROJECT_ROOT) / ".env"
    if not env_file.exists():
        return False
    return load_dotenv(env_file, override=False)


load_project_env()


def env_path(name: str, default: Path) -> Path:
    """Return an absolute path from an environment variable or repo default.

    Relative paths from the repo `.env` are resolved from the project root so
    `uv run ...` commands behave the same even when launched from another
    working directory.
    """
    raw_value = Path(os.getenv(name, str(default))).expanduser()
    if raw_value.is_absolute():
        return raw_value.resolve()
    return (PROJECT_ROOT / raw_value).resolve()


def data_dir() -> Path:
    return env_path("MMA_AI_DATA_DIR", DATA_DIR)


def raw_ufcstats_dir() -> Path:
    return env_path("MMA_AI_UFCSTATS_DIR", UFCSTATS_DATA_DIR)


def models_dir() -> Path:
    return env_path("MMA_AI_MODELS_DIR", MODELS_DIR)


def picks_dir() -> Path:
    return env_path("MMA_AI_PICKS_DIR", PICKS_DIR)


def data_file(filename: str) -> Path:
    return data_dir() / filename


def database_url(env_var: str = "DATABASE_URL", default: str = DEFAULT_DATABASE_URL) -> str:
    return os.getenv(env_var, default)


def odds_database_url() -> str:
    return database_url("ODDS_DATABASE_URL", DEFAULT_ODDS_DATABASE_URL)


def no_winsor_database_url() -> str:
    return database_url("MMA_AI_NO_WINSOR_DATABASE_URL", DEFAULT_NO_WINSOR_DATABASE_URL)
