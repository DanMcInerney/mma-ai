import time

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
