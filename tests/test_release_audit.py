from pathlib import Path

from scripts.release_audit import (
    audit_repository,
    find_dockerignore_issues,
    find_compose_postgres_image_issues,
    find_compose_postgres_volume_issues,
    find_file_mode_issues,
    find_forbidden_artifacts,
    find_gitattributes_issues,
    find_hardcoded_local_database_urls,
    find_legacy_runtime_identifiers,
    find_misplaced_test_scripts,
    find_missing_required_files,
    find_non_ascii_runtime_text,
    find_package_data_issues,
    find_seed_data_issues,
    find_sensitive_text,
    find_setup_artifact_pin_issues,
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
    assert "autogluon.tabular[mitra,tabicl,tabpfn]>=1.5.0" in dependencies


def test_release_audit_allows_only_seed_raw_csvs_from_data():
    issues = find_forbidden_artifacts(
        [
            ".env",
            ".env.production",
            ".envrc",
            ".npmrc",
            ".webapp.out.log",
            "data/raw/ufcstats/competitions.csv",
            "data/raw/ufcstats/individuals.csv",
            "data/prediction_data.csv",
            "secrets/service-account.pem",
            "models/model.pkl",
            ".venv/pyvenv.cfg",
            "AutoGluonModels/ag-test/predictor.pkl",
            "AutogluonModels/ag-test/predictor.pkl",
            ".cursor/rules/project-description.mdc",
            "pics/picks/example.png",
        ]
    )

    assert [issue.path for issue in issues] == [
        ".env",
        ".env.production",
        ".envrc",
        ".npmrc",
        ".webapp.out.log",
        "data/prediction_data.csv",
        "secrets/service-account.pem",
        "models/model.pkl",
        ".venv/pyvenv.cfg",
        "AutoGluonModels/ag-test/predictor.pkl",
        "AutogluonModels/ag-test/predictor.pkl",
        ".cursor/rules/project-description.mdc",
        "pics/picks/example.png",
    ]


def test_release_audit_requires_setup_sh_to_stay_executable():
    issues = find_file_mode_issues({"setup.sh": "100644"})

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("non_executable_setup_script", "setup.sh")
    ]
    assert find_file_mode_issues({"setup.sh": "100755"}) == []


def test_release_audit_requires_dockerignore_to_protect_public_context(tmp_path):
    (tmp_path / ".dockerignore").write_text(
        "\n".join(
            [
                ".env",
                ".env.local",
                "*.log",
                "*.csv",
                "logs",
                "artifacts",
                "AutoGluonModels",
                "AutogluonModels",
                "tests",
                "data/**",
                "!data/",
                "!data/raw/",
                "!data/raw/ufcstats/",
                "!data/raw/ufcstats/competitions.csv",
                "!libs/web/static/index.html",
            ]
        ),
        encoding="utf-8",
    )

    issues = find_dockerignore_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("incomplete_dockerignore", ".dockerignore")
    ]
    assert "!data/raw/ufcstats/individuals.csv" in issues[0].detail


def test_release_audit_requires_dockerignore_to_exclude_generated_root_reports(tmp_path):
    (tmp_path / ".dockerignore").write_text(
        "\n".join(
            [
                ".env",
                ".env.local",
                "*.log",
                "logs",
                "artifacts",
                "AutoGluonModels",
                "AutogluonModels",
                "tests",
                "data/**",
                "!data/",
                "!data/raw/",
                "!data/raw/ufcstats/",
                "!data/raw/ufcstats/competitions.csv",
                "!data/raw/ufcstats/individuals.csv",
            ]
        ),
        encoding="utf-8",
    )

    issues = find_dockerignore_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("incomplete_dockerignore", ".dockerignore")
    ]
    assert "*.csv" in issues[0].detail
    assert "*.html" in issues[0].detail
    assert "!libs/web/static/index.html" in issues[0].detail


def test_release_audit_requires_git_attributes_for_cross_platform_setup_scripts(tmp_path):
    (tmp_path / ".gitattributes").write_text(
        "* text=auto\n"
        "Dockerfile text eol=lf\n",
        encoding="utf-8",
    )

    issues = find_gitattributes_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("incomplete_gitattributes", ".gitattributes")
    ]
    assert "*.sh text eol=lf" in issues[0].detail

    (tmp_path / ".gitattributes").write_text(
        "* text=auto\n"
        "*.sh text eol=lf\n"
        "Dockerfile text eol=lf\n",
        encoding="utf-8",
    )

    assert find_gitattributes_issues(tmp_path) == []


def test_release_audit_requires_dashboard_static_assets_in_package_data(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[tool.setuptools.package-data]\n"
        '"libs.web" = ["static/app.js"]\n',
        encoding="utf-8",
    )

    issues = find_package_data_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("missing_package_data", "pyproject.toml")
    ]
    assert "libs.web" in issues[0].detail
    assert "static/*" in issues[0].detail

    pyproject.write_text(
        "[tool.setuptools.package-data]\n"
        '"libs.web" = ["static/*"]\n',
        encoding="utf-8",
    )

    assert find_package_data_issues(tmp_path) == []


def test_release_audit_requires_compose_postgres_to_match_hf_dump(tmp_path):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        "  db:\n"
        "    image: postgres:17\n",
        encoding="utf-8",
    )

    issues = find_compose_postgres_image_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("compose_postgres_image_mismatch", "docker-compose.yml")
    ]
    assert "postgres:18.1" in issues[0].detail

    compose.write_text(
        "services:\n"
        "  db:\n"
        "    image: postgres:18.1\n",
        encoding="utf-8",
    )

    assert find_compose_postgres_image_issues(tmp_path) == []


def test_release_audit_requires_postgres_18_parent_volume_mount(tmp_path):
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        "services:\n"
        "  db:\n"
        "    volumes:\n"
        "      - postgres-data:/var/lib/postgresql/data\n",
        encoding="utf-8",
    )

    issues = find_compose_postgres_volume_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("compose_postgres_volume_mismatch", "docker-compose.yml")
    ]
    assert "/var/lib/postgresql/data" in issues[0].detail

    compose.write_text(
        "services:\n"
        "  db:\n"
        "    volumes:\n"
        "      - postgres-data:/var/lib/postgresql\n",
        encoding="utf-8",
    )

    assert find_compose_postgres_volume_issues(tmp_path) == []


def test_release_audit_requires_setup_artifact_pins_to_match_across_platforms(tmp_path):
    (tmp_path / "setup.sh").write_text(
        'ARTIFACTS=(\n'
        '  "manifest.json|"\n'
        '  "processed/prediction_data.csv|AAA111"\n'
        '  "models/model.tar.gz|BBB222"\n'
        ')\n',
        encoding="utf-8",
    )
    (tmp_path / "setup.ps1").write_text(
        '$Artifacts = @(\n'
        '    [pscustomobject]@{ Path = "manifest.json"; Sha256 = "" },\n'
        '    [pscustomobject]@{ Path = "processed/prediction_data.csv"; Sha256 = "AAA111" },\n'
        '    [pscustomobject]@{ Path = "models/model.tar.gz"; Sha256 = "CCC333" },\n'
        '    [pscustomobject]@{ Path = "dumps/extra.postgres-custom"; Sha256 = "DDD444" }\n'
        ')\n',
        encoding="utf-8",
    )

    issues = find_setup_artifact_pin_issues(tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("setup_artifact_pin_drift", "setup.ps1/setup.sh")
    ]
    assert "checksum mismatch: models/model.tar.gz" in issues[0].detail
    assert "only in setup.ps1: dumps/extra.postgres-custom" in issues[0].detail

    (tmp_path / "setup.ps1").write_text(
        '$Artifacts = @(\n'
        '    [pscustomobject]@{ Path = "manifest.json"; Sha256 = "" },\n'
        '    [pscustomobject]@{ Path = "processed/prediction_data.csv"; Sha256 = "AAA111" },\n'
        '    [pscustomobject]@{ Path = "models/model.tar.gz"; Sha256 = "BBB222" }\n'
        ')\n',
        encoding="utf-8",
    )

    assert find_setup_artifact_pin_issues(tmp_path) == []


def test_release_audit_requires_public_entrypoints_and_seed_data():
    issues = find_missing_required_files(["README.md", "libs/web/static/index.html"], ROOT)

    missing_paths = {issue.path for issue in issues}
    assert ".gitattributes" in missing_paths
    assert "setup.ps1" in missing_paths
    assert "setup.sh" in missing_paths
    assert "AGENTS.md" in missing_paths
    assert "CLAUDE.md" in missing_paths
    assert "data/raw/ufcstats/competitions.csv" in missing_paths
    assert "data/raw/ufcstats/individuals.csv" in missing_paths


def test_release_audit_rejects_pytest_named_scripts_outside_tests():
    issues = find_misplaced_test_scripts(
        [
            "scripts/test_parameter_optimization.py",
            "tests/test_release_audit.py",
            "scripts/validate_stat_quality_calc.py",
        ]
    )

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("misplaced_test_script", "scripts/test_parameter_optimization.py")
    ]


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


def test_release_audit_rejects_non_ascii_prediction_runtime_logs(tmp_path):
    runtime_file = tmp_path / "predict.py"
    runtime_file.write_text("print('prediction ready " + chr(0x2713) + "')\n", encoding="utf-8")

    issues = find_non_ascii_runtime_text(["predict.py"], tmp_path)

    assert [(issue.kind, issue.path) for issue in issues] == [
        ("non_ascii_runtime_text", "predict.py"),
    ]
    assert "line 1" in issues[0].detail
