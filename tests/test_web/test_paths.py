import os

from libs.paths import load_project_env


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
