"""``_job.pklz`` is written for remote workers and failed jobs, not for a job that
succeeds in process."""

import contextlib
import sys
import cloudpickle as cp
import pytest
from pydra.compose import python, workflow
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.scripts.cli import crash
from pydra.workers.base import Worker

# The installed package does not ship tests, so worker processes cannot import this
# module and must receive its classes by value
cp.register_pickle_by_value(sys.modules[__name__])


@python.define
def Inc(x: int) -> int:
    return x + 1


@python.define
def Boom(x: int) -> int:
    raise ValueError("boom")


@workflow.define
def IncTwice(x: int) -> int:
    first = workflow.add(Inc(x=x), name="first")
    second = workflow.add(Inc(x=first.out), name="second")
    return second.out


class RemoteLikeWorker(Worker):
    """Runs a job as a scheduler's worker does, which leaves ``persist_pickle`` as
    the job set it"""

    _plugin_name = "remote-like"

    def run(self, job, rerun=False):
        return job.run(rerun=rerun)


def slots(task, cache_root):
    return list(PickleStore(cache_root).slots(task._checksum))


@pytest.fixture(params=["debug", "cf"])
def in_process_worker(request):
    return request.param


def test_successful_in_process_run_leaves_no_job_pickle(in_process_worker, tmp_path):
    task = Inc(x=1)
    with Submitter(worker=in_process_worker, cache_root=tmp_path) as sub:
        result = sub(task)
    assert result.outputs.out == 2
    (slot,) = slots(task, tmp_path)
    assert (slot / "_result.pklz").exists()
    assert not (slot / "_job.pklz").exists()


def test_failed_in_process_run_leaves_job_pickle(in_process_worker, tmp_path):
    task = Boom(x=1)
    with Submitter(worker=in_process_worker, cache_root=tmp_path) as sub:
        # the debug worker raises, the cf worker returns an errored result
        with contextlib.suppress(Exception):
            sub(task)
    (slot,) = slots(task, tmp_path)
    assert (slot / "_error.pklz").exists()
    with open(slot / "_job.pklz", "rb") as f:
        assert cp.load(f).task.x == 1


def test_remote_style_worker_still_writes_job_pickle(tmp_path):
    task = Inc(x=1)
    with Submitter(worker=RemoteLikeWorker(), cache_root=tmp_path) as sub:
        sub(task)
    (slot,) = slots(task, tmp_path)
    assert (slot / "_job.pklz").exists()


def test_crash_rerun_uses_failure_pickle(cli_runner, tmp_path):
    task = Boom(x=1)
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        with pytest.raises(Exception):
            sub(task)
    (slot,) = slots(task, tmp_path)
    result = cli_runner(
        crash, [str(slot / "_error.pklz"), "--rerun", "--debugger", "pdb"]
    )
    # the rerun raises inside the job and is caught by the pdb branch, so it only
    # reaches a missing-job-file error if the pickle were absent
    assert not isinstance(result.exception, FileNotFoundError)


def test_cache_written_by_older_version_is_reused(tmp_path):
    task = Inc(x=1)
    with Submitter(worker=RemoteLikeWorker(), cache_root=tmp_path) as sub:
        sub(task)
    (slot,) = slots(task, tmp_path)
    assert (slot / "_job.pklz").exists()
    # A debug run finds the result and neither rewrites nor removes the pickle
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        result = sub(Inc(x=1))
    assert result.outputs.out == 2
    assert (slot / "_job.pklz").exists()


def test_job_pickled_by_older_version_loads(tmp_path):
    task = Inc(x=1)
    with Submitter(worker=RemoteLikeWorker(), cache_root=tmp_path) as sub:
        sub(task)
    (slot,) = slots(task, tmp_path)
    with open(slot / "_job.pklz", "rb") as f:
        job = cp.load(f)
    state = job.__getstate__()
    del state["persist_pickle"]
    older = type(job).__new__(type(job))
    older.__setstate__(state)
    assert older.persist_pickle is True


def test_workflow_job_keeps_job_pickle(in_process_worker, tmp_path):
    """A workflow job keeps its pickle whichever worker expands it, and the leaf jobs
    of the workflow do not"""
    task = IncTwice(x=1)
    with Submitter(worker=in_process_worker, cache_root=tmp_path) as sub:
        result = sub(task)
    assert result.outputs.out == 3
    (slot,) = slots(task, tmp_path)
    assert (slot / "_job.pklz").exists()
    assert result.job is not None
    leaf_pickles = [p for p in tmp_path.glob("python-*/*/_job.pklz")]
    assert leaf_pickles == []


def test_result_job_is_none_for_in_process_leaf_success(in_process_worker, tmp_path):
    with Submitter(worker=in_process_worker, cache_root=tmp_path) as sub:
        result = sub(Inc(x=1))
    assert result.job is None
