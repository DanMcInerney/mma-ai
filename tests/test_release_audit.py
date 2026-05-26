from pathlib import Path

from scripts.release_audit import (
    audit_repository,
    find_forbidden_artifacts,
    find_hardcoded_local_database_urls,
    find_legacy_runtime_identifiers,
    find_missing_required_files,
    find_seed_data_issues,
    find_sensitive_text,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_audit_passes_current_tracked_repository():
    assert audit_repository() == []


def test_runtime_dependencies_do_not_include_test_tooling():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependency_block = text.split("dependencies = [", 1)[1].split("\n]", 1)[0]
    dependencies = [
        line.strip().strip('",')
        for line in dependency_block.splitlines()
        if line.strip().startswith('"')
    ]
    normalized = {
        dependency.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
        for dependency in dependencies
    }

    assert "pytest" not in normalized
    assert "pytest-mock" not in normalized
    assert "kaleido==0.2.1" in dependencies


def test_release_audit_allows_only_seed_raw_csvs_from_data():
    issues = find_forbidden_artifacts(
        [
            "data/raw/ufcstats/competitions.csv",
            "data/raw/ufcstats/individuals.csv",
            "data/prediction_data.csv",
            "AutoGluonModels/ag-test/predictor.pkl",
            "AutogluonModels/ag-test/predictor.pkl",
            ".cursor/rules/project-description.mdc",
            "pics/picks/example.png",
        ]
    )

    assert [issue.path for issue in issues] == [
        "data/prediction_data.csv",
        "AutoGluonModels/ag-test/predictor.pkl",
        "AutogluonModels/ag-test/predictor.pkl",
        ".cursor/rules/project-description.mdc",
        "pics/picks/example.png",
    ]


def test_release_audit_requires_public_entrypoints_and_seed_data():
    issues = find_missing_required_files(["README.md", "libs/web/static/index.html"], ROOT)

    missing_paths = {issue.path for issue in issues}
    assert "setup.ps1" in missing_paths
    assert "setup.sh" in missing_paths
    assert "AGENTS.md" in missing_paths
    assert "CLAUDE.md" in missing_paths
    assert "data/raw/ufcstats/competitions.csv" in missing_paths
    assert "data/raw/ufcstats/individuals.csv" in missing_paths


def test_release_audit_rejects_tiny_or_malformed_seed_csvs(tmp_path):
    competitions = tmp_path / "data" / "raw" / "ufcstats" / "competitions.csv"
    individuals = tmp_path / "data" / "raw" / "ufcstats" / "individuals.csv"
    competitions.parent.mkdir(parents=True)
    competitions.write_text("a,b,c,d,e,f\n1,2,3,4,5,6\n", encoding="utf-8")
    individuals.write_text("only_one_column\n", encoding="utf-8")

    issues = find_seed_data_issues(
        [
            "data/raw/ufcstats/competitions.csv",
            "data/raw/ufcstats/individuals.csv",
        ],
        tmp_path,
    )

    assert [issue.kind for issue in issues] == ["weak_seed_data", "weak_seed_data"]
    assert {issue.path for issue in issues} == {
        "data/raw/ufcstats/competitions.csv",
        "data/raw/ufcstats/individuals.csv",
    }


def test_release_audit_detects_realistic_secret_and_local_path(tmp_path):
    tracked_file = tmp_path / "README.md"
    tracked_file.write_text(
        "local path " + "C:" + "/Users/alice/project and token sk-" + "a" * 30,
        encoding="utf-8",
    )

    issues = find_sensitive_text(["README.md"], tmp_path)

    assert {issue.kind for issue in issues} == {"local_windows_path", "openai_api_key"}


def test_release_audit_rejects_legacy_runtime_project_names(tmp_path):
    runtime_file = tmp_path / "libs" / "scraping" / "ufcstats.py"
    docs_file = tmp_path / "README.md"
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('settings = {"BOT_NAME": "mma-ai-db"}', encoding="utf-8")
    docs_file.write_text("This public repo combines mma-ai-db with UFCScraper.", encoding="utf-8")

    issues = find_legacy_runtime_identifiers(
        [
            "libs/scraping/ufcstats.py",
            "README.md",
        ],
        tmp_path,
    )

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("legacy_mma_ai_db_name", "libs/scraping/ufcstats.py"),
    ]


def test_release_audit_rejects_hardcoded_runtime_database_urls(tmp_path):
    runtime_file = tmp_path / "scripts" / "debug.py"
    docs_file = tmp_path / "docs" / "HUGGINGFACE_DATASET.md"
    paths_file = tmp_path / "libs" / "paths.py"
    runtime_file.parent.mkdir(parents=True)
    docs_file.parent.mkdir(parents=True)
    paths_file.parent.mkdir(parents=True)
    runtime_file.write_text("DB_URL = 'postgresql://postgres@localhost:5432/mma-ai'", encoding="utf-8")
    docs_file.write_text("psql postgresql://postgres@localhost:5432/mma-ai", encoding="utf-8")
    paths_file.write_text("DEFAULT_DATABASE_URL = 'postgresql://postgres@localhost:5432/mma-ai'", encoding="utf-8")

    issues = find_hardcoded_local_database_urls(
        [
            "scripts/debug.py",
            "docs/HUGGINGFACE_DATASET.md",
            "libs/paths.py",
        ],
        tmp_path,
    )

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("hardcoded_local_postgres_url", "scripts/debug.py"),
    ]
