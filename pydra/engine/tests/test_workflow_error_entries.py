"""``WorkflowError.entries`` names each failed job, and ``Submitter(telemetry=False)``
skips the version check"""

import pickle
import pytest
from pydra.engine import WorkflowError
from pydra.engine import job as job_module
from pydra.engine.job import Job
from pydra.engine.submitter import JobErrorEntry, Submitter
from pydra.engine.tests.test_on_error import Propagate


@pytest.mark.parametrize(
    "worker,on_error",
    [("cf", "fail_fast"), ("debug", "raise_at_end"), ("cf", "raise_at_end")],
)
def test_entries_name_the_failed_job(worker, on_error, tmp_path):
    with Submitter(worker=worker, on_error=on_error, cache_root=tmp_path) as sub:
        with pytest.raises(WorkflowError) as excinfo:
            sub(Propagate(x=[2, 1, 3], y=[10, 20]), raise_errors=True)
    error = excinfo.value
    (entry,) = error.entries
    assert isinstance(entry, JobErrorEntry)
    assert entry.job_name == "A"
    assert entry.state_index == 1
    assert entry.exception_type == "builtins.Exception"
    assert "i'm raising an exception!" in entry.message
    assert "Traceback" in entry.traceback
    assert "i'm raising an exception!" in entry.traceback
    # errors keeps its form, one string per entry
    assert len(error.errors) == 1
    assert error.errors[0].startswith("Job 'A(1)'")
    assert sub.entries == error.entries


def test_entries_survive_pickling():
    entry = JobErrorEntry("A", 3, "builtins.ValueError", "bad", "Traceback ...")
    error = pickle.loads(pickle.dumps(WorkflowError("failed", ["one"], [entry])))
    assert error.entries == [entry]
    assert (str(error), error.errors) == ("failed", ["one"])


def test_entries_default_empty():
    error = pickle.loads(pickle.dumps(WorkflowError("failed", ["one"])))
    assert error.entries == []


@pytest.fixture
def version_checks(monkeypatch):
    calls = []
    monkeypatch.setattr(job_module, "check_latest_version", lambda: calls.append(1))
    monkeypatch.setattr(Job, "_etelemetry_checked", False)
    return calls


def test_telemetry_false_skips_version_check(version_checks, tmp_path):
    sub = Submitter(telemetry=False, cache_root=tmp_path)
    assert sub.telemetry is False
    assert version_checks == []
    assert Job._etelemetry_checked is False


def test_telemetry_default_checks_version(version_checks, tmp_path):
    Submitter(cache_root=tmp_path)
    Submitter(cache_root=tmp_path)
    assert version_checks == [1]  # once per session
