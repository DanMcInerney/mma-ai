"""Failure-safety boundaries for destructive rebuilds."""

from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.engine import make_url


EXPECTED_DATABASE_NAME = "mma-ai"
GENERATED_SCHEMA_BACKUPS = (
    ("features", "features_rebuild_backup"),
    ("model_data", "model_data_rebuild_backup"),
)


def require_safe_database_target(database_url, *, allow_nonstandard=False):
    """Reject destructive work against any database other than mma-ai."""
    database_name = make_url(str(database_url)).database
    if database_name != EXPECTED_DATABASE_NAME and not allow_nonstandard:
        raise ValueError(
            f"Refusing destructive rebuild for database {database_name!r}; "
            f"expected {EXPECTED_DATABASE_NAME!r}. Use the explicit override to proceed."
        )
    return database_name


def _schema_exists(connection, schema_name):
    result = connection.execute(
        text(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = :schema_name
            )
            """
        ),
        {"schema_name": schema_name},
    )
    return result.scalar_one()


@contextmanager
def schema_rebuild(engine):
    """Preserve generated schemas until a rebuild has completed successfully."""
    moved_schemas = []
    with engine.begin() as connection:
        stale_backups = [
            backup
            for _, backup in GENERATED_SCHEMA_BACKUPS
            if _schema_exists(connection, backup)
        ]
        if stale_backups:
            names = ", ".join(stale_backups)
            raise RuntimeError(f"Refusing rebuild while backup schemas exist: {names}")

        for schema, backup in GENERATED_SCHEMA_BACKUPS:
            if _schema_exists(connection, schema):
                connection.execute(text(f'ALTER SCHEMA "{schema}" RENAME TO "{backup}"'))
                moved_schemas.append((schema, backup))

    try:
        yield
    except BaseException:
        with engine.begin() as connection:
            for schema, backup in GENERATED_SCHEMA_BACKUPS:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                if (schema, backup) in moved_schemas:
                    connection.execute(text(f'ALTER SCHEMA "{backup}" RENAME TO "{schema}"'))
        raise
    else:
        with engine.begin() as connection:
            for _, backup in moved_schemas:
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{backup}" CASCADE'))
