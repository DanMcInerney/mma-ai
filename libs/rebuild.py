"""Failure-safety boundaries for destructive rebuilds."""

from sqlalchemy.engine import make_url


EXPECTED_DATABASE_NAME = "mma-ai"


def require_safe_database_target(database_url, *, allow_nonstandard=False):
    """Reject destructive work against any database other than mma-ai."""
    database_name = make_url(str(database_url)).database
    if database_name != EXPECTED_DATABASE_NAME and not allow_nonstandard:
        raise ValueError(
            f"Refusing destructive rebuild for database {database_name!r}; "
            f"expected {EXPECTED_DATABASE_NAME!r}. Use the explicit override to proceed."
        )
    return database_name
