import os

from libs.paths import PROJECT_ROOT, data_dir, env_path, load_project_env, models_dir, raw_ufcstats_dir


def test_load_project_env_reads_dotenv_without_overriding_shell_env(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DATABASE_URL=postgresql://from-file/mma-ai\n"
        "ODDS_DATABASE_URL=postgresql://from-file/odds\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("ODDS_DATABASE_URL", "postgresql://from-shell/odds")

    assert load_project_env(tmp_path) is True

    assert os.environ["DATABASE_URL"] == "postgresql://from-file/mma-ai"
    assert os.environ["ODDS_DATABASE_URL"] == "postgresql://from-shell/odds"


def test_load_project_env_returns_false_when_dotenv_is_absent(tmp_path):
    assert load_project_env(tmp_path) is False


def test_env_path_resolves_relative_values_from_project_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MMA_AI_DATA_DIR", "./data")
    monkeypatch.setenv("MMA_AI_MODELS_DIR", "AutogluonModels")
    monkeypatch.setenv("MMA_AI_UFCSTATS_DIR", "data/raw/ufcstats")

    assert data_dir() == (PROJECT_ROOT / "data").resolve()
    assert models_dir() == (PROJECT_ROOT / "AutogluonModels").resolve()
    assert raw_ufcstats_dir() == (PROJECT_ROOT / "data" / "raw" / "ufcstats").resolve()


def test_env_path_preserves_absolute_values(monkeypatch, tmp_path):
    absolute = tmp_path / "external-data"
    monkeypatch.setenv("MMA_AI_DATA_DIR", str(absolute))

    assert env_path("MMA_AI_DATA_DIR", PROJECT_ROOT / "data") == absolute.resolve()
