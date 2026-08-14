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
