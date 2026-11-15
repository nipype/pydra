"""The cheap submission path of proposal 0001 §3.2: a job's task is pickled once, and the
hashes that name results are unchanged."""

import pickle
import sys
from pathlib import Path
import cloudpickle as cp
import pytest
from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.submitter import Submitter
from pydra.utils.hash import hash_function

cp.register_pickle_by_value(sys.modules[__name__])


@python.define
def Double(x: int) -> int:
    return x * 2


@python.define
def Boom(x: int) -> int:
    raise ValueError("boom")


def make_job(tmp_path: Path, task=None, **kwargs) -> Job:
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        return Job(task or Double(x=3), submitter=sub, name="Double", **kwargs)


# --- the task is pickled once ---------------------------------------------------------


def test_job_pickle_serialises_the_task_in_one_pass(tmp_path, monkeypatch):
    job = make_job(tmp_path)
    calls = []
    real = cp.dumps

    def spy(obj, *args, **kwargs):
        calls.append(type(obj).__name__)
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(cp, "dumps", spy)
    spy(job)
    assert calls == ["Job"]


def test_job_state_holds_the_task_itself(tmp_path):
    job = make_job(tmp_path)
    assert job.__getstate__()["task"] is job.task


def test_jobs_pickled_in_one_stream_share_the_task_class(tmp_path):
    """A workflow job's ``_job.pklz`` embeds its node jobs. Their tasks share one
    cloudpickle memo, so the task class is serialised once, not once per job."""
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        jobs = [Job(Double(x=i), submitter=sub, name="Double") for i in range(50)]
        for job in jobs:
            job.checksum
        per_job = len(cp.dumps(jobs)) / len(jobs)
        alone = len(cp.dumps(jobs[0]))
    assert alone > 5000  # the class is carried by value, as a worker needs it
    assert per_job < alone / 3


def test_job_requires_cloudpickle(tmp_path):
    """The task is held as an object, so a pickler that cannot serialise task classes
    by value (the stdlib's) cannot pickle a Job"""
    job = make_job(tmp_path)
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(job)
    assert cp.loads(cp.dumps(job)).task.x == 3


def test_pickled_job_keeps_its_checksum_and_task(tmp_path):
    job = make_job(tmp_path)
    checksum = job.checksum
    loaded = cp.loads(cp.dumps(job))
    assert loaded.task.x == 3
    assert loaded._checksum == checksum  # carried, so a worker does not rehash
    assert loaded.checksum == checksum == loaded.task._checksum


def test_job_pickled_with_the_task_as_bytes_still_loads(tmp_path):
    """A ``_job.pklz`` written before the task was stored in the state"""
    job = make_job(tmp_path)
    state = job.__getstate__()
    state["task"] = cp.dumps(state["task"])
    loaded = Job.__new__(Job)
    loaded.__setstate__(state)
    assert loaded.task.x == 3


# --- hashes are unchanged -------------------------------------------------------------


def test_hash_function_values_are_pinned():
    """Values computed before the cheap submission path; a cache key must not move"""
    assert (
        hash_function([1, "a", {"b": 2.5}, (None, True)])
        == "ffb77f251aaf2c466930929c91a252fa"
    )
    assert hash_function("pydra") == "d5c29a1b00e4ac6247aff1ab89bea8c2"


def test_task_hash_is_a_function_of_the_inputs():
    assert Double(x=3)._hash == Double(x=3)._hash
    assert Double(x=3)._hash != Double(x=4)._hash
    assert Double(x=3)._checksum == f"{Double._task_type()}-{Double(x=3)._hash}"


def test_task_hash_follows_an_input_change():
    task = Double(x=3)
    first = task._hash
    task.x = 4
    assert task._hash == Double(x=4)._hash != first
    task.x = 3
    assert task._hash == first


def test_task_hash_follows_an_in_place_change_of_an_input():
    @python.define
    def Total(xs: list) -> int:
        return sum(xs)

    values = [1, 2]
    task = Total(xs=values)
    first = task._hash
    values.append(3)
    assert task._hash == Total(xs=[1, 2, 3])._hash != first


def test_checksum_is_the_same_in_a_job_and_after_pickling(tmp_path):
    task = Double(x=7)
    job = make_job(tmp_path, task)
    assert cp.loads(cp.dumps(job)).checksum == task._checksum
