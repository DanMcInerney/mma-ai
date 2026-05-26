"""Repository release hygiene checks for the public MMA AI package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


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
FORBIDDEN_PREFIXES = ("AutogluonModels/", "artifacts/", "pics/", "data/predictions/")
FORBIDDEN_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".ipynb")

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


@dataclass(frozen=True)
class AuditIssue:
    kind: str
    path: str
    detail: str


def git_ls_files(root: Path = ROOT) -> list[str]:
    result = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True)
    return [item.decode("utf-8").replace("\\", "/") for item in result.stdout.split(b"\0") if item]


def find_forbidden_artifacts(paths: Iterable[str]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized in SEED_DATA_PATHS:
            continue
        if (
            normalized in GENERATED_DATA_FILES
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


def _excerpt(text: str, start: int, end: int, radius: int = 32) -> str:
    snippet = text[max(0, start - radius) : min(len(text), end + radius)]
    return " ".join(snippet.split())


def audit_repository(root: Path = ROOT) -> list[AuditIssue]:
    tracked = git_ls_files(root)
    return [*find_forbidden_artifacts(tracked), *find_sensitive_text(tracked, root)]


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
