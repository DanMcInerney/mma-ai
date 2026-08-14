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
