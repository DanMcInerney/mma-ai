from pathlib import Path

from scripts.release_audit import (
    audit_repository,
    find_forbidden_artifacts,
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
            "pics/picks/example.png",
        ]
    )

    assert [issue.path for issue in issues] == [
        "data/prediction_data.csv",
        "AutoGluonModels/ag-test/predictor.pkl",
        "AutogluonModels/ag-test/predictor.pkl",
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
