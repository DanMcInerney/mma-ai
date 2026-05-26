from scripts.recalculate_adjperf_no_winsor import pg_cli_parts


def test_pg_cli_parts_accepts_setup_default_without_password():
    parts = pg_cli_parts("postgresql://postgres@localhost:55432/mma-ai")

    assert parts == {
        "user": "postgres",
        "password": "",
        "host": "localhost",
        "port": "55432",
        "database": "mma-ai",
    }


def test_pg_cli_parts_accepts_password_urls_from_custom_env():
    parts = pg_cli_parts("postgresql://user:secret@db.example.test:5433/odds")

    assert parts == {
        "user": "user",
        "password": "secret",
        "host": "db.example.test",
        "port": "5433",
        "database": "odds",
    }
