"""Failure-safety boundaries for destructive rebuilds."""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

from sqlalchemy import text
from sqlalchemy.engine import make_url


EXPECTED_DATABASE_NAME = "mma-ai"
GENERATED_SCHEMA_BACKUPS = (
    ("features", "features_rebuild_backup"),
    ("model_data", "model_data_rebuild_backup"),
)
GENERATED_CSV_NAMES = (
    "prediction_data.csv",
    "training_data.csv",
    "training_data_dec.csv",
)


class CsvRestoreError(RuntimeError):
    """Publication failed and automated restoration could not finish."""


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


def _publish_staged_csvs(staging_dir, output_dir, backup_dir, replace_file):
    backed_up = set()
    published = set()
    try:
        for name in GENERATED_CSV_NAMES:
            destination = output_dir / name
            if destination.exists():
                replace_file(destination, backup_dir / name)
                backed_up.add(name)

        for name in GENERATED_CSV_NAMES:
            replace_file(staging_dir / name, output_dir / name)
            published.add(name)
    except BaseException:
        try:
            for name in published - backed_up:
                destination = output_dir / name
                if destination.exists():
                    destination.unlink()
            for name in backed_up:
                replace_file(backup_dir / name, output_dir / name)
        except BaseException as restore_error:
            raise CsvRestoreError(
                f"CSV restoration failed; recoverable files remain in {backup_dir}"
            ) from restore_error
        raise


@contextmanager
def staged_csv_publication(output_dir, *, replace_file=os.replace):
    """Stage and atomically publish the complete generated CSV output set."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = Path(tempfile.mkdtemp(prefix=".mma-ai-rebuild-", dir=output_dir))
    staging_dir = working_dir / "staged"
    backup_dir = working_dir / "backups"
    staging_dir.mkdir()
    backup_dir.mkdir()
    preserve_working_files = False

    try:
        yield staging_dir
        missing = [name for name in GENERATED_CSV_NAMES if not (staging_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Staged rebuild is missing required CSV outputs: " + ", ".join(missing)
            )
        _publish_staged_csvs(staging_dir, output_dir, backup_dir, replace_file)
    except CsvRestoreError:
        preserve_working_files = True
        raise
    finally:
        if not preserve_working_files:
            shutil.rmtree(working_dir, ignore_errors=True)
