"""Canonical hashing and order-stable artifact inventories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest().upper()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class TreeInventory:
    root: str
    files: tuple[dict[str, Any], ...]
    file_count: int
    total_bytes: int
    tree_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "files": list(self.files),
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }


def tree_inventory(root: Path, *, excluded_names: Iterable[str] = ()) -> TreeInventory:
    root = Path(root)
    excluded = set(excluded_names)
    files: list[dict[str, Any]] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    tree_hash = canonical_sha256(files)
    return TreeInventory(
        root=str(root.resolve()),
        files=tuple(files),
        file_count=len(files),
        total_bytes=sum(entry["bytes"] for entry in files),
        tree_sha256=tree_hash,
    )


def write_canonical_json(path: Path, value: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return canonical_sha256(value)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
