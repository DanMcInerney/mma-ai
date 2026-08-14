import pytest

import main as rebuild


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
