import os
from pathlib import Path

import pytest

import main as rebuild
from libs import rebuild as rebuild_safety


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalar_one(self):
        return self._value


class FakeConnection:
    def __init__(self, schemas):
        self.schemas = schemas
        self.statements = []

    def execute(self, statement, params=None):
        sql = " ".join(str(statement).split())
        self.statements.append((sql, params))
        if sql.startswith("SELECT EXISTS"):
            return FakeResult(params["schema_name"] in self.schemas)
        if sql.startswith("DROP SCHEMA IF EXISTS"):
            schema = sql.split()[4].strip('"')
            self.schemas.discard(schema)
            return FakeResult(None)
        if sql.startswith("ALTER SCHEMA"):
            _, _, source, _, _, destination = sql.split()
            source = source.strip('"')
            destination = destination.strip('"')
            if source not in self.schemas:
                raise AssertionError(f"Cannot rename missing schema: {source}")
            if destination in self.schemas:
                raise AssertionError(f"Cannot overwrite schema: {destination}")
            self.schemas.remove(source)
            self.schemas.add(destination)
            return FakeResult(None)
        raise AssertionError(f"Unexpected SQL: {sql}")


class FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeEngine:
    def __init__(self, url, schemas=("features", "model_data")):
        self.url = url
        self.schemas = set(schemas)
        self.connection = FakeConnection(self.schemas)

    def begin(self):
        return FakeTransaction(self.connection)


def test_database_target_rejects_nonstandard_name_without_leaking_credentials():
    engine = FakeEngine("postgresql://private-user:private-password@db.example/wrong-db")

    with pytest.raises(ValueError) as caught:
        rebuild.reset_database(engine)

    message = str(caught.value)
    assert "wrong-db" in message
    assert "private-user" not in message
    assert "private-password" not in message
    assert engine.schemas == {"features", "model_data"}
    assert engine.connection.statements == []


def test_database_target_explicit_override_allows_reset():
    engine = FakeEngine("postgresql://private-user:private-password@db.example/wrong-db")

    rebuild.reset_database(engine, allow_nonstandard_db=True)

    assert engine.schemas == set()


def test_schema_prepare_preserves_prior_schemas_and_success_removes_backups():
    engine = FakeEngine("postgresql://localhost/mma-ai")

    with rebuild_safety.schema_rebuild(engine):
        assert engine.schemas == {"features_rebuild_backup", "model_data_rebuild_backup"}
        engine.schemas.update({"features", "model_data"})

    assert engine.schemas == {"features", "model_data"}


def test_schema_handled_failure_removes_partial_schemas_and_restores_prior_ones():
    engine = FakeEngine("postgresql://localhost/mma-ai")

    with pytest.raises(RuntimeError, match="pipeline failed"):
        with rebuild_safety.schema_rebuild(engine):
            engine.schemas.update({"features", "model_data"})
            raise RuntimeError("pipeline failed")

    assert engine.schemas == {"features", "model_data"}


def test_schema_prepare_refuses_stale_backup():
    initial_schemas = {"features", "model_data", "features_rebuild_backup"}
    engine = FakeEngine("postgresql://localhost/mma-ai", schemas=initial_schemas)

    with pytest.raises(RuntimeError, match="features_rebuild_backup"):
        with rebuild_safety.schema_rebuild(engine):
            pass

    assert engine.schemas == initial_schemas


CSV_NAMES = ("prediction_data.csv", "training_data.csv", "training_data_dec.csv")


def write_csv_set(directory, prefix):
    directory.mkdir(parents=True, exist_ok=True)
    for name in CSV_NAMES:
        (directory / name).write_bytes(f"{prefix}:{name}".encode())


def read_csv_set(directory):
    return {name: (directory / name).read_bytes() for name in CSV_NAMES}


def test_csv_publication_requires_all_three_staged_outputs(tmp_path):
    output_dir = tmp_path / "data"
    write_csv_set(output_dir, "old")
    before = read_csv_set(output_dir)

    with pytest.raises(FileNotFoundError, match="training_data_dec.csv"):
        with rebuild_safety.staged_csv_publication(output_dir) as staging_dir:
            (staging_dir / "prediction_data.csv").write_bytes(b"new prediction")
            (staging_dir / "training_data.csv").write_bytes(b"new training")

    assert read_csv_set(output_dir) == before


def test_csv_publication_replaces_the_complete_output_set(tmp_path):
    output_dir = tmp_path / "data"
    write_csv_set(output_dir, "old")

    with rebuild_safety.staged_csv_publication(output_dir) as staging_dir:
        assert staging_dir.parent.parent == output_dir
        write_csv_set(staging_dir, "new")

    assert read_csv_set(output_dir) == {
        name: f"new:{name}".encode() for name in CSV_NAMES
    }


def test_csv_publication_failure_restores_every_prior_byte(tmp_path):
    output_dir = tmp_path / "data"
    write_csv_set(output_dir, "old")
    before = read_csv_set(output_dir)
    staging_path = None

    def fail_during_second_publish(source, destination):
        source = Path(source)
        destination = Path(destination)
        if (
            source.parent == staging_path
            and source.name == "training_data.csv"
            and destination.parent == output_dir
        ):
            raise OSError("injected publication failure")
        os.replace(source, destination)

    with pytest.raises(OSError, match="injected publication failure"):
        with rebuild_safety.staged_csv_publication(
            output_dir, replace_file=fail_during_second_publish
        ) as staging_dir:
            staging_path = staging_dir
            write_csv_set(staging_dir, "new")

    assert read_csv_set(output_dir) == before
