import time
from threading import Event

from libs.web.jobs import JobManager, JobState


def test_job_manager_runs_successful_job():
    manager = JobManager()
    job = manager.start("unit", lambda: {"ok": True})

    for _ in range(100):
        current = manager.get(job.id)
        if current and current.state == JobState.SUCCEEDED:
            break
        time.sleep(0.01)

    current = manager.get(job.id)
    assert current is not None
    assert current.state == JobState.SUCCEEDED
    assert current.result == {"ok": True}
    assert "succeeded" in manager.read_log(job.id)


def test_job_manager_captures_failure():
    manager = JobManager()

    def fail():
        raise RuntimeError("boom")

    job = manager.start("unit", fail)
    for _ in range(100):
        current = manager.get(job.id)
        if current and current.state == JobState.FAILED:
            break
        time.sleep(0.01)

    current = manager.get(job.id)
    assert current is not None
    assert current.state == JobState.FAILED
    assert "boom" in (current.error or "")
    assert "RuntimeError: boom" in manager.read_log(job.id)


def test_job_manager_captures_stdout_and_stderr(tmp_path):
    manager = JobManager(log_dir_factory=lambda: tmp_path)

    def noisy():
        import sys

        print("hello stdout")
        print("hello stderr", file=sys.stderr)
        return {"ok": True}

    job = manager.start("unit", noisy)
    for _ in range(100):
        current = manager.get(job.id)
        if current and current.state == JobState.SUCCEEDED:
            break
        time.sleep(0.01)

    current = manager.get(job.id)
    assert current is not None
    assert current.log_path is not None
    log = manager.read_log(job.id)
    assert "hello stdout" in log
    assert "[stderr] hello stderr" in log
    assert current.log_path.endswith(f"{job.id}.log")


def test_job_manager_serializes_jobs_so_logs_do_not_overlap(tmp_path):
    manager = JobManager(log_dir_factory=lambda: tmp_path)
    first_started = Event()
    release_first = Event()
    second_started = Event()

    def first_job():
        print("first job running")
        first_started.set()
        assert release_first.wait(timeout=5)
        print("first job done")
        return {"job": "first"}

    def second_job():
        print("second job running")
        second_started.set()
        return {"job": "second"}

    first = manager.start("unit", first_job)
    assert first_started.wait(timeout=5)
    second = manager.start("unit", second_job)
    time.sleep(0.05)

    second_record = manager.get(second.id)
    assert second_record is not None
    assert second_record.state == JobState.QUEUED
    assert not second_started.is_set()

    release_first.set()
    for _ in range(200):
        records = {job.id: job for job in manager.list()}
        if records[first.id].state == JobState.SUCCEEDED and records[second.id].state == JobState.SUCCEEDED:
            break
        time.sleep(0.01)

    assert manager.get(first.id).result == {"job": "first"}
    assert manager.get(second.id).result == {"job": "second"}
    assert "second job running" not in manager.read_log(first.id)
    assert "first job running" not in manager.read_log(second.id)
