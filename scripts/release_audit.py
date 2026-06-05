"""Repository release hygiene checks for the public MMA AI package."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SEED_DATA_PATHS = {
    "data/raw/ufcstats/competitions.csv",
    "data/raw/ufcstats/individuals.csv",
}
GENERATED_DATA_FILES = {
    "data/prediction_data.csv",
    "data/training_data.csv",
    "data/training_data_dec.csv",
}
REQUIRED_TRACKED_FILES = {
    ".gitattributes",
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "AGENTS.md",
    "CLAUDE.md",
    "Dockerfile",
    "README.md",
    "docker-compose.yml",
    "docker/postgres-init/01-create-odds.sql",
    "docs/HUGGINGFACE_DATASET.md",
    "docs/RELEASE_READINESS.md",
    "libs/web/static/app.js",
    "libs/web/static/icons.js",
    "libs/web/static/index.html",
    "libs/web/static/styles.css",
    "main.py",
    "predict.py",
    "pyproject.toml",
    "setup.ps1",
    "setup.sh",
    "scripts/train_dashboard.py",
    "scripts/verify_hf_manifest.sh",
    "uv.lock",
    *SEED_DATA_PATHS,
}
FORBIDDEN_PREFIXES = (
    ".cursor/",
    ".venv/",
    "venv/",
    "env/",
    "ENV/",
    "AutoGluonModels/",
    "AutogluonModels/",
    "artifacts/",
    "build/",
    "dist/",
    "mma_ai.egg-info/",
    "pics/",
    "data/predictions/",
)
FORBIDDEN_EXACT_FILES = {".env", ".env.local", ".envrc", ".netrc", ".npmrc", ".pypirc"}
FORBIDDEN_SUFFIXES = (
    ".7z",
    ".bak",
    ".backup",
    ".db",
    ".dump",
    ".gif",
    ".gz",
    ".ipynb",
    ".joblib",
    ".jpg",
    ".jpeg",
    ".kdbx",
    ".key",
    ".log",
    ".ovpn",
    ".p12",
    ".pem",
    ".pfx",
    ".pkl",
    ".png",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".zip",
)
ASCII_RUNTIME_LOG_FILES = {"predict.py"}
MIN_SEED_ROWS = 1000
MIN_SEED_COLUMNS = 6
REQUIRED_DOCKERIGNORE_LINES = {
    ".env",
    ".env.*",
    ".env.local",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "*.key",
    "*.log",
    "*.p12",
    "*.pem",
    "*.pfx",
    "*.csv",
    "*.html",
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
    "!libs/web/static/index.html",
}
REQUIRED_PACKAGE_DATA = {
    "libs.web": {"static/*"},
}
REQUIRED_GITATTRIBUTES_LINES = {
    "*.sh text eol=lf",
    "Dockerfile text eol=lf",
}
REQUIRED_COMPOSE_POSTGRES_IMAGE = "postgres:18.1"
REQUIRED_COMPOSE_POSTGRES_VOLUME = "postgres-data:/var/lib/postgresql"

SENSITIVE_PATTERNS = {
    "local_windows_path": re.compile(r"\b[A-Z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\"'`<>]+", re.IGNORECASE),
    "non_example_email": re.compile(
        r"(?<!:)\b[A-Z0-9._%+-]+@(?!example\.com\b|example\.test\b)[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    ),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_api_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "github_token": re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
}

LEGACY_RUNTIME_PATTERNS = {
    "legacy_mma_ai_db_name": re.compile(r"\bmma-ai-db\b", re.IGNORECASE),
}
LEGACY_IDENTIFIER_ALLOWED_PREFIXES = ("docs/", "tests/")
LEGACY_IDENTIFIER_ALLOWED_FILES = {"AGENTS.md", "CLAUDE.md", "README.md"}

HARDCODED_LOCAL_DB_PATTERN = re.compile(
    r"postgresql://postgres(?::[^@\s\"'`<>]+)?@localhost:5432/(?:mma-ai|odds|mma-ai-no-winsor|mma|ufc_fights)\b",
    re.IGNORECASE,
)
HARDCODED_LOCAL_DB_ALLOWED_PREFIXES = ("docs/", "tests/")
HARDCODED_LOCAL_DB_ALLOWED_FILES = {".env.example", "README.md", "libs/paths.py"}


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    path: str
    detail: str


def git_ls_files(root: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [item.decode("utf-8").replace("\\", "/") for item in result.stdout.split(b"\0") if item]


def git_ls_file_modes(root: Path = ROOT) -> dict[str, str]:
    result = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=root, check=True, capture_output=True)
    modes: dict[str, str] = {}
    for raw_item in result.stdout.split(b"\0"):
        if not raw_item:
            continue
        metadata, path = raw_item.decode("utf-8").split("\t", 1)
        modes[path.replace("\\", "/")] = metadata.split()[0]
    return modes


def find_forbidden_artifacts(paths: Iterable[str]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        filename = Path(normalized).name
        lower_filename = filename.lower()
        if normalized in SEED_DATA_PATHS:
            continue
        if (
            normalized in FORBIDDEN_EXACT_FILES
            or filename in FORBIDDEN_EXACT_FILES
            or (lower_filename.startswith(".env.") and lower_filename != ".env.example")
            or normalized in GENERATED_DATA_FILES
            or normalized.startswith(FORBIDDEN_PREFIXES)
            or normalized.lower().endswith(FORBIDDEN_SUFFIXES)
        ):
            issues.append(
                AuditIssue(
                    kind="forbidden_artifact",
                    path=normalized,
                    detail="Generated data, model, image, notebook, or runtime output is tracked.",
                )
            )
    return issues


def find_file_mode_issues(file_modes: Mapping[str, str]) -> list[AuditIssue]:
    setup_mode = file_modes.get("setup.sh")
    if setup_mode and setup_mode != "100755":
        return [
            AuditIssue(
                kind="non_executable_setup_script",
                path="setup.sh",
                detail="setup.sh must be tracked executable so the public quick start can run ./setup.sh.",
            )
        ]
    return []


def find_dockerignore_issues(root: Path = ROOT) -> list[AuditIssue]:
    path = root / ".dockerignore"
    try:
        lines = {
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as exc:
        return [AuditIssue("unreadable_dockerignore", ".dockerignore", str(exc))]

    missing = sorted(REQUIRED_DOCKERIGNORE_LINES - lines)
    if not missing:
        return []
    return [
        AuditIssue(
            kind="incomplete_dockerignore",
            path=".dockerignore",
            detail="Missing required Docker context rule(s): " + ", ".join(missing),
        )
    ]


def find_gitattributes_issues(root: Path = ROOT) -> list[AuditIssue]:
    path = root / ".gitattributes"
    try:
        lines = {
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as exc:
        return [AuditIssue("unreadable_gitattributes", ".gitattributes", str(exc))]

    missing = sorted(REQUIRED_GITATTRIBUTES_LINES - lines)
    if not missing:
        return []
    return [
        AuditIssue(
            kind="incomplete_gitattributes",
            path=".gitattributes",
            detail="Missing required Git attribute rule(s): " + ", ".join(missing),
        )
    ]


def find_package_data_issues(root: Path = ROOT) -> list[AuditIssue]:
    path = root / "pyproject.toml"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [AuditIssue("unreadable_pyproject", "pyproject.toml", str(exc))]

    match = re.search(
        r"(?ms)^\[tool\.setuptools\.package-data\]\s*(?P<body>.*?)(?=^\[|\Z)",
        text,
    )
    body = match.group("body") if match else ""
    issues = []
    for package, required_patterns in REQUIRED_PACKAGE_DATA.items():
        package_pattern = re.search(rf'(?m)^"{re.escape(package)}"\s*=\s*\[(?P<patterns>[^\]]*)\]', body)
        configured_patterns = set(re.findall(r'"([^"]+)"', package_pattern.group("patterns") if package_pattern else ""))
        missing = sorted(required_patterns - configured_patterns)
        if missing:
            issues.append(
                AuditIssue(
                    kind="missing_package_data",
                    path="pyproject.toml",
                    detail=f"{package} must include package data pattern(s): {', '.join(missing)}.",
                )
            )
    return issues


def find_compose_postgres_image_issues(root: Path = ROOT) -> list[AuditIssue]:
    """Keep local restore tooling aligned with the Hugging Face Postgres dump."""
    path = root / "docker-compose.yml"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [AuditIssue("unreadable_compose", "docker-compose.yml", str(exc))]

    if re.search(rf"(?m)^\s*image:\s*{re.escape(REQUIRED_COMPOSE_POSTGRES_IMAGE)}\s*(?:#.*)?$", text):
        return []

    found = re.findall(r"(?m)^\s*image:\s*(postgres:[^\s#]+)", text)
    configured = ", ".join(found) if found else "no postgres image"
    return [
        AuditIssue(
            kind="compose_postgres_image_mismatch",
            path="docker-compose.yml",
            detail=(
                f"Expected {REQUIRED_COMPOSE_POSTGRES_IMAGE} to match the Hugging Face dump "
                f"Postgres version; found {configured}."
            ),
        )
    ]


def find_compose_postgres_volume_issues(root: Path = ROOT) -> list[AuditIssue]:
    """Postgres 18 images require mounting the parent data directory."""
    path = root / "docker-compose.yml"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [AuditIssue("unreadable_compose", "docker-compose.yml", str(exc))]

    if re.search(rf"(?m)^\s*-\s*{re.escape(REQUIRED_COMPOSE_POSTGRES_VOLUME)}\s*(?:#.*)?$", text):
        return []

    found = re.findall(r"(?m)^\s*-\s*(postgres-data:/[^\s#]+)", text)
    configured = ", ".join(found) if found else "no postgres-data volume mount"
    return [
        AuditIssue(
            kind="compose_postgres_volume_mismatch",
            path="docker-compose.yml",
            detail=(
                f"Expected {REQUIRED_COMPOSE_POSTGRES_VOLUME}; found {configured}. "
                "Postgres 18 rejects the legacy /var/lib/postgresql/data mount."
            ),
        )
    ]


def _parse_bash_setup_artifact_pins(text: str) -> dict[str, str]:
    match = re.search(r"(?ms)^ARTIFACTS=\(\s*(?P<body>.*?)^\)", text)
    body = match.group("body") if match else ""
    return {
        path: sha.upper()
        for path, sha in re.findall(r'"([^"|]+)\|([^"]*)"', body)
    }


def _parse_powershell_setup_artifact_pins(text: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).upper()
        for match in re.finditer(
            r'\[pscustomobject\]@\{\s*Path\s*=\s*"([^"]+)"\s*;\s*Sha256\s*=\s*"([^"]*)"\s*\}',
            text,
            flags=re.DOTALL,
        )
    }


def find_setup_artifact_pin_issues(root: Path = ROOT) -> list[AuditIssue]:
    """Require Windows and Unix first-time setup scripts to restore the same artifacts."""
    try:
        powershell_text = (root / "setup.ps1").read_text(encoding="utf-8", errors="replace")
        bash_text = (root / "setup.sh").read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [AuditIssue("unreadable_setup_pins", "setup.ps1/setup.sh", str(exc))]

    powershell_pins = _parse_powershell_setup_artifact_pins(powershell_text)
    bash_pins = _parse_bash_setup_artifact_pins(bash_text)
    details = []
    bash_only = sorted(set(bash_pins) - set(powershell_pins))
    powershell_only = sorted(set(powershell_pins) - set(bash_pins))
    mismatched = sorted(
        path for path in set(bash_pins) & set(powershell_pins)
        if bash_pins[path] != powershell_pins[path]
    )
    if bash_only:
        details.append("only in setup.sh: " + ", ".join(bash_only))
    if powershell_only:
        details.append("only in setup.ps1: " + ", ".join(powershell_only))
    if mismatched:
        details.append("checksum mismatch: " + ", ".join(mismatched))
    if not bash_pins or not powershell_pins:
        details.append("could not parse artifact pins from one or both setup scripts")

    if not details:
        return []
    return [
        AuditIssue(
            kind="setup_artifact_pin_drift",
            path="setup.ps1/setup.sh",
            detail="; ".join(details),
        )
    ]


def find_misplaced_test_scripts(paths: Iterable[str]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.startswith("tests/"):
            continue
        if Path(normalized).name.startswith("test_") and normalized.endswith(".py"):
            issues.append(
                AuditIssue(
                    kind="misplaced_test_script",
                    path=normalized,
                    detail="Pytest-style test files must live under tests/ so release checks and packaging boundaries stay clear.",
                )
            )
    return issues


def find_missing_required_files(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    tracked = {path.replace("\\", "/") for path in paths}
    issues = []
    for required_path in sorted(REQUIRED_TRACKED_FILES):
        if required_path not in tracked or not (root / required_path).exists():
            issues.append(
                AuditIssue(
                    kind="missing_required_file",
                    path=required_path,
                    detail="Required public runtime file is not tracked.",
                )
            )
    return issues


def find_seed_data_issues(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    tracked = {path.replace("\\", "/") for path in paths}
    issues: list[AuditIssue] = []
    for seed_path in sorted(SEED_DATA_PATHS):
        if seed_path not in tracked:
            continue
        path = root / seed_path
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                reader = csv.reader(handle)
                header = next(reader)
                rows = sum(1 for _row in reader)
        except FileNotFoundError:
            issues.append(AuditIssue("missing_seed_data", seed_path, "Tracked seed CSV is missing from the working tree."))
            continue
        except (OSError, StopIteration, csv.Error) as exc:
            issues.append(AuditIssue("unreadable_seed_data", seed_path, str(exc)))
            continue

        if rows < MIN_SEED_ROWS or len(header) < MIN_SEED_COLUMNS:
            issues.append(
                AuditIssue(
                    kind="weak_seed_data",
                    path=seed_path,
                    detail=(
                        f"Seed CSV has {rows} data rows and {len(header)} columns; "
                        f"expected at least {MIN_SEED_ROWS} rows and {MIN_SEED_COLUMNS} columns."
                    ),
                )
            )
    return issues


def find_sensitive_text(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for relative_path in paths:
        path = root / relative_path
        try:
            raw = path.read_bytes()
        except OSError as exc:
            issues.append(AuditIssue("unreadable_file", relative_path, str(exc)))
            continue
        if b"\0" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for kind, pattern in SENSITIVE_PATTERNS.items():
            for match in pattern.finditer(text):
                issues.append(AuditIssue(kind=kind, path=relative_path, detail=_excerpt(text, match.start(), match.end())))
    return issues


def find_legacy_runtime_identifiers(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for relative_path in paths:
        normalized = relative_path.replace("\\", "/")
        if normalized in LEGACY_IDENTIFIER_ALLOWED_FILES or normalized.startswith(LEGACY_IDENTIFIER_ALLOWED_PREFIXES):
            continue

        path = root / relative_path
        try:
            raw = path.read_bytes()
        except OSError as exc:
            issues.append(AuditIssue("unreadable_file", relative_path, str(exc)))
            continue
        if b"\0" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for kind, pattern in LEGACY_RUNTIME_PATTERNS.items():
            for match in pattern.finditer(text):
                issues.append(AuditIssue(kind=kind, path=normalized, detail=_excerpt(text, match.start(), match.end())))
    return issues


def find_hardcoded_local_database_urls(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for relative_path in paths:
        normalized = relative_path.replace("\\", "/")
        if normalized in HARDCODED_LOCAL_DB_ALLOWED_FILES or normalized.startswith(HARDCODED_LOCAL_DB_ALLOWED_PREFIXES):
            continue

        path = root / relative_path
        try:
            raw = path.read_bytes()
        except OSError as exc:
            issues.append(AuditIssue("unreadable_file", relative_path, str(exc)))
            continue
        if b"\0" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="ignore")
        for match in HARDCODED_LOCAL_DB_PATTERN.finditer(text):
            issues.append(
                AuditIssue(
                    kind="hardcoded_local_postgres_url",
                    path=normalized,
                    detail=_excerpt(text, match.start(), match.end()),
                )
            )
    return issues


def find_non_ascii_runtime_text(paths: Iterable[str], root: Path = ROOT) -> list[AuditIssue]:
    """Keep dashboard-captured CLI logs portable across Windows, Docker, and CI."""
    tracked = {path.replace("\\", "/") for path in paths}
    issues: list[AuditIssue] = []
    for relative_path in sorted(ASCII_RUNTIME_LOG_FILES & tracked):
        path = root / relative_path
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            issues.append(AuditIssue("unreadable_file", relative_path, str(exc)))
            continue
        for index, char in enumerate(text):
            if ord(char) <= 127:
                continue
            line_number = text.count("\n", 0, index) + 1
            issues.append(
                AuditIssue(
                    kind="non_ascii_runtime_text",
                    path=relative_path,
                    detail=f"Non-ASCII character U+{ord(char):04X} at line {line_number}; use ASCII status text in dashboard logs.",
                )
            )
            break
    return issues


def _excerpt(text: str, start: int, end: int, radius: int = 32) -> str:
    snippet = text[max(0, start - radius) : min(len(text), end + radius)]
    return " ".join(snippet.split())


def audit_repository(root: Path = ROOT) -> list[AuditIssue]:
    tracked = git_ls_files(root)
    file_modes = git_ls_file_modes(root)
    return [
        *find_missing_required_files(tracked, root),
        *find_seed_data_issues(tracked, root),
        *find_forbidden_artifacts(tracked),
        *find_file_mode_issues(file_modes),
        *find_dockerignore_issues(root),
        *find_gitattributes_issues(root),
        *find_package_data_issues(root),
        *find_compose_postgres_image_issues(root),
        *find_compose_postgres_volume_issues(root),
        *find_setup_artifact_pin_issues(root),
        *find_misplaced_test_scripts(tracked),
        *find_sensitive_text(tracked, root),
        *find_legacy_runtime_identifiers(tracked, root),
        *find_hardcoded_local_database_urls(tracked, root),
        *find_non_ascii_runtime_text(tracked, root),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit tracked files for public release hygiene.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    issues = audit_repository(ROOT)
    if args.json:
        print(json.dumps([asdict(issue) for issue in issues], indent=2))
    elif issues:
        print("Release audit failed:")
        for issue in issues:
            print(f"- {issue.kind}: {issue.path}: {issue.detail}")
    else:
        print("Release audit passed.")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
