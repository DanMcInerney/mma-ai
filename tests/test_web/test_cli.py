import pytest

from libs.web import cli


def test_web_cli_help_exits_without_starting_server(monkeypatch, capsys):
    monkeypatch.setattr("libs.web.cli.uvicorn.run", lambda *_args, **_kwargs: pytest.fail("server started"))

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])

    assert excinfo.value.code == 0
    assert "Start the MMA AI web dashboard" in capsys.readouterr().out


def test_web_cli_uses_setup_selected_web_port(monkeypatch):
    captured = {}
    monkeypatch.delenv("MMA_AI_PORT", raising=False)
    monkeypatch.setenv("MMA_AI_WEB_PORT", "18000")
    monkeypatch.setattr("libs.web.cli.uvicorn.run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))

    assert cli.main([]) == 0

    assert captured == {
        "app": "libs.web.app:app",
        "host": "0.0.0.0",
        "port": 18000,
        "reload": False,
    }


def test_web_cli_arguments_override_environment(monkeypatch):
    captured = {}
    monkeypatch.setenv("MMA_AI_HOST", "0.0.0.0")
    monkeypatch.setenv("MMA_AI_PORT", "8000")
    monkeypatch.setenv("MMA_AI_WEB_PORT", "18000")
    monkeypatch.setattr("libs.web.cli.uvicorn.run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))

    assert cli.main(["--host", "127.0.0.1", "--port", "19000", "--reload"]) == 0

    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 19000
    assert captured["reload"] is True


def test_web_cli_keeps_legacy_port_env_precedence(monkeypatch):
    captured = {}
    monkeypatch.setenv("MMA_AI_PORT", "9000")
    monkeypatch.setenv("MMA_AI_WEB_PORT", "18000")
    monkeypatch.setattr("libs.web.cli.uvicorn.run", lambda app, **kwargs: captured.update({"app": app, **kwargs}))

    assert cli.main([]) == 0

    assert captured["port"] == 9000
