"""Small in-memory background job manager for long MMA AI workflows."""

from __future__ import annotations

import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable
from uuid import uuid4


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class JobRecord:
    id: str
    kind: str
    state: JobState = JobState.QUEUED
    message: str = "Queued"
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    result: dict[str, Any] | None = None
    error: str | None = None
    log_path: str | None = None
    log_tail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "state": self.state.value,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "result": self.result,
            "error": self.error,
            "log_path": self.log_path,
            "log_tail": self.log_tail,
        }


class JobLogWriter:
    """File-like object that routes stdout/stderr writes into a job log."""

    def __init__(self, manager: "JobManager", job_id: str, stream_name: str) -> None:
        self.manager = manager
        self.job_id = job_id
        self.stream_name = stream_name

    def write(self, text: str) -> int:
        if text:
            self.manager.append_log(self.job_id, text, stream_name=self.stream_name)
        return len(text)

    def flush(self) -> None:
        return None


class JobManager:
    """Run blocking workflows in daemon threads and expose their state."""

    def __init__(self, log_dir_factory: Callable[[], Path] | None = None, log_tail_chars: int = 20000) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = Lock()
        self._run_lock = Lock()
        self._log_dir_factory = log_dir_factory
        self._log_tail_chars = log_tail_chars

    def start(self, kind: str, func: Callable[[], dict[str, Any] | None]) -> JobRecord:
        job_id = uuid4().hex
        job = JobRecord(id=job_id, kind=kind, log_path=self._new_log_path(job_id))
        with self._lock:
            self._jobs[job.id] = job

        self.append_log(job.id, f"[{utcnow().isoformat()}] queued {kind} job {job.id}\n")
        thread = Thread(target=self._run, args=(job.id, func), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def append_log(self, job_id: str, text: str, stream_name: str = "system") -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            prefixed = self._prefix_log_text(text, stream_name)
            if job.log_path:
                Path(job.log_path).parent.mkdir(parents=True, exist_ok=True)
                with Path(job.log_path).open("a", encoding="utf-8", errors="replace") as handle:
                    handle.write(prefixed)
            job.log_tail = (job.log_tail + prefixed)[-self._log_tail_chars :]
            job.updated_at = utcnow()

    def read_log(self, job_id: str) -> str:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            log_path = job.log_path
            log_tail = job.log_tail
        if log_path and Path(log_path).exists():
            return Path(log_path).read_text(encoding="utf-8", errors="replace")
        return log_tail

    def _patch(self, job_id: str, **updates: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            for key, value in updates.items():
                setattr(job, key, value)
            job.updated_at = utcnow()

    def _run(self, job_id: str, func: Callable[[], dict[str, Any] | None]) -> None:
        with self._run_lock:
            self._patch(job_id, state=JobState.RUNNING, message="Running")
            self.append_log(job_id, f"[{utcnow().isoformat()}] started\n")
            stdout = JobLogWriter(self, job_id, "stdout")
            stderr = JobLogWriter(self, job_id, "stderr")
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = func() or {}
            except Exception as exc:  # pragma: no cover - exercised through API behavior
                self.append_log(job_id, traceback.format_exc(), stream_name="stderr")
                self._patch(job_id, state=JobState.FAILED, message="Failed", error=str(exc))
                self.append_log(job_id, f"[{utcnow().isoformat()}] failed: {exc}\n")
                return

            self._patch(job_id, state=JobState.SUCCEEDED, message="Succeeded", result=result)
            self.append_log(job_id, f"[{utcnow().isoformat()}] succeeded\n")

    def _new_log_path(self, job_id: str) -> str | None:
        if self._log_dir_factory is None:
            return None
        log_dir = self._log_dir_factory()
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir / f"{job_id}.log")

    @staticmethod
    def _prefix_log_text(text: str, stream_name: str) -> str:
        if stream_name == "stdout":
            return text
        if stream_name == "stderr":
            prefix = "[stderr] "
        else:
            prefix = ""
        lines = text.splitlines(keepends=True)
        return "".join(f"{prefix}{line}" if line.strip() else line for line in lines)
