from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import main as rebuild


def test_main_reaches_missing_csv_guard_before_database_access(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw-empty"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()

    def fail_database_access(*_args, **_kwargs):
        raise AssertionError("database access must follow raw CSV validation")

    monkeypatch.setattr(rebuild, "create_db_engine", fail_database_access)

    with pytest.raises(FileNotFoundError, match="Missing raw UFCStats CSVs"):
        rebuild.main(
            raw_data_dir=raw_dir,
            output_data_dir=output_dir,
            reset_db=False,
        )


def test_main_wraps_pipeline_in_rebuild_safety_contexts(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "output"
    raw_dir.mkdir()
    (raw_dir / "competitions.csv").write_text("fight_id\n", encoding="utf-8")
    (raw_dir / "individuals.csv").write_text("fighter_id\n", encoding="utf-8")
    engine = SimpleNamespace(url="postgresql://localhost/alternate")
    events = []

    monkeypatch.setattr(rebuild, "create_db_engine", lambda _url: engine)

    def fake_guard(database_url, *, allow_nonstandard=False):
        events.append(("guard", database_url, allow_nonstandard))

    @contextmanager
    def fake_schema_rebuild(received_engine):
        events.append(("schema-enter", received_engine))
        yield
        events.append(("schema-exit", received_engine))

    @contextmanager
    def fake_csv_publication(received_output_dir):
        staging_dir = received_output_dir / "staged"
        staging_dir.mkdir(parents=True)
        events.append(("csv-enter", received_output_dir))
        yield staging_dir
        events.append(("csv-exit", received_output_dir))

    def fake_pipeline(**kwargs):
        events.append(("pipeline", kwargs["engine"], kwargs["output_data_dir"]))

    monkeypatch.setattr(rebuild, "require_safe_database_target", fake_guard)
    monkeypatch.setattr(rebuild, "schema_rebuild", fake_schema_rebuild)
    monkeypatch.setattr(rebuild, "staged_csv_publication", fake_csv_publication)
    monkeypatch.setattr(rebuild, "_run_pipeline", fake_pipeline)

    rebuild.main(
        db_url=engine.url,
        raw_data_dir=raw_dir,
        output_data_dir=output_dir,
        reset_db=True,
        allow_nonstandard_db=True,
    )

    assert events == [
        ("guard", engine.url, True),
        ("schema-enter", engine),
        ("csv-enter", output_dir.resolve()),
        ("pipeline", engine, output_dir.resolve() / "staged"),
        ("csv-exit", output_dir.resolve()),
        ("schema-exit", engine),
    ]


def test_main_rejects_database_target_before_engine_creation(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "competitions.csv").write_text("fight_id\n", encoding="utf-8")
    (raw_dir / "individuals.csv").write_text("fighter_id\n", encoding="utf-8")
    unsafe_url = "postgresql://private-user:private-password@db.example/wrong-db"

    def fail_engine_creation(*_args, **_kwargs):
        raise AssertionError("target validation must precede database access")

    monkeypatch.setattr(rebuild, "create_db_engine", fail_engine_creation)

    with pytest.raises(ValueError) as caught:
        rebuild.main(
            db_url=unsafe_url,
            raw_data_dir=raw_dir,
            output_data_dir=tmp_path / "output",
            reset_db=True,
        )

    message = str(caught.value)
    assert "wrong-db" in message
    assert "private-user" not in message
    assert "private-password" not in message


def test_parse_args_exposes_nonstandard_database_override(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["mma-rebuild-db", "--reset-db", "--allow-nonstandard-db"]
    )

    args = rebuild.parse_args()

    assert args.reset_db is True
    assert args.allow_nonstandard_db is True


def test_cli_forwards_nonstandard_database_override(monkeypatch):
    args = SimpleNamespace(
        odds=False,
        odds_features=False,
        db_url="postgresql://localhost/alternate",
        raw_data_dir="raw",
        output_data_dir="output",
        scrape=False,
        reset_db=True,
        allow_nonstandard_db=True,
    )
    captured = {}
    monkeypatch.setattr(rebuild, "parse_args", lambda: args)
    monkeypatch.setattr(rebuild, "main", lambda **kwargs: captured.update(kwargs))

    rebuild.cli()

    assert captured["allow_nonstandard_db"] is True
