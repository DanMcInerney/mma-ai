from pathlib import Path

from scripts.release_audit import audit_repository, find_forbidden_artifacts, find_sensitive_text


def test_release_audit_passes_current_tracked_repository():
    assert audit_repository() == []


def test_release_audit_allows_only_seed_raw_csvs_from_data():
    issues = find_forbidden_artifacts(
        [
            "data/raw/ufcstats/competitions.csv",
            "data/raw/ufcstats/individuals.csv",
            "data/prediction_data.csv",
            "AutogluonModels/ag-test/predictor.pkl",
            "pics/picks/example.png",
        ]
    )

    assert [issue.path for issue in issues] == [
        "data/prediction_data.csv",
        "AutogluonModels/ag-test/predictor.pkl",
        "pics/picks/example.png",
    ]


def test_release_audit_detects_realistic_secret_and_local_path(tmp_path):
    tracked_file = tmp_path / "README.md"
    tracked_file.write_text(
        "local path " + "C:" + "/Users/alice/project and token sk-" + "a" * 30,
        encoding="utf-8",
    )

    issues = find_sensitive_text(["README.md"], tmp_path)

    assert {issue.kind for issue in issues} == {"local_windows_path", "openai_api_key"}
