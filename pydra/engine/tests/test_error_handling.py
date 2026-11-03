import typing as ty
import sys
import time
import pytest
import cloudpickle as cp
from pathlib import Path
import glob as glob
from pydra.compose import python, workflow
from pydra.engine.submitter import Submitter
from pydra.workers import cf

no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


def test_exception_func():
    @python.define
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """checking if None is properly returned as the result"""

    @python.define
    def FunNone(x):
        return None

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out is None


def test_result_none_2():
    """checking if None is properly set for all outputs"""

    @python.define(outputs=["out1", "out2"])
    def FunNone(x) -> tuple[ty.Any, ty.Any]:
        return None  # Do we actually want this behaviour?

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out1 is None
    assert outputs.out2 is None


def test_traceback(tmpdir):
    """checking if the error raised in a function is properly returned;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def FunError(x):
        raise Exception("Error from the function")

    with pytest.raises(Exception, match="Error from the function") as exinfo:
        with Submitter(worker="cf", cache_root=tmpdir) as sub:
            sub(FunError(x=3), raise_errors=True)

    # getting error file from the error message
    error_file_match = (
        str(exinfo.value.__notes__[0]).split("here: ")[-1].split("_error.pklz")[0]
    )
    error_file = Path(error_file_match) / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    # reading error message from the pickle file
    error_tb = cp.loads(error_file.read_bytes())["error message"]
    # the error traceback should be a list and should point to a specific line in the function
    assert isinstance(error_tb, list)
    assert "in FunError" in error_tb[-2]


def test_traceback_wf(tmp_path: Path):
    """checking if the error raised in a function is properly returned by a workflow;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def FunError(x):
        raise Exception("Error from the function")

    @workflow.define
    def Workflow(x_list):
        fun_error = workflow.add(FunError().split(x=x_list), name="fun_error")
        return fun_error.out

    wf = Workflow(x_list=[3, 4])
    with pytest.raises(RuntimeError, match="Job 'fun_error.*, errored") as exinfo:
        with Submitter(worker="cf", cache_root=tmp_path) as sub:
            sub(wf, raise_errors=True)

    # getting error file from the error message
    cache_dir_match = Path(
        str(exinfo.value).split("See output directory for details: ")[-1].strip()
    )
    assert cache_dir_match.exists()
    error_file = cache_dir_match / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    assert "in FunError" in str(exinfo.value)


def test_rerun_errored(tmp_path):
    """Test rerunning a task containing errors.
    Only the errored tasks should be rerun"""

    # NB: the jobs record themselves by touching marker files rather than printing,
    # as the worker runs them in separate processes and whether their stdout reaches
    # the test depends on the multiprocessing start method
    markers = tmp_path / "markers"
    markers.mkdir()

    @python.define
    def PassOdds(x, marker_dir: str):
        from pathlib import Path
        from uuid import uuid4

        marker_path = Path(marker_dir)
        (marker_path / f"run-{x}-{uuid4().hex}").touch()
        if x % 2 == 0:
            (marker_path / f"error-{x}-{uuid4().hex}").touch()
            raise Exception("even error")
        else:
            return x

    pass_odds = PassOdds(marker_dir=str(markers)).split("x", x=[1, 2, 3, 4, 5])

    def ran(kind="run"):
        return sorted(int(p.name.split("-")[1]) for p in markers.glob(f"{kind}-*"))

    # All 5 jobs run the first time, and only the 2 that errored are rerun the second
    with pytest.raises(RuntimeError, match="even error"):
        pass_odds(cache_root=tmp_path / "cache", worker="cf", n_procs=3)
    assert ran() == [1, 2, 3, 4, 5]
    with pytest.raises(RuntimeError, match="even error"):
        pass_odds(cache_root=tmp_path / "cache", worker="cf", n_procs=3)
    assert ran() == [1, 2, 2, 3, 4, 4, 5]
    assert ran("error") == [2, 2, 4, 4]


@pytest.mark.timeout(90)
def test_worker_error_before_job_saved(tmp_path: Path):
    """If a job fails in the worker before it is able to save a result (e.g. if the
    worker is unable to run it), the workflow should fail rather than spinning
    indefinitely waiting for the job to complete"""
    from pydra.tasks.testing import UnsafeDivisionWorkflow
    from pydra.workers import cf

    class FailingWorker(cf.Worker):
        _plugin_name = "failing-cf"

        async def run(self, job, rerun=False):
            if job.name == "Divide":
                raise RuntimeError("worker was unable to run the job")
            return await super().run(job, rerun=rerun)

    wf = UnsafeDivisionWorkflow(a=10, b=5, denominator=2)
    start = time.monotonic()
    with Submitter(worker=FailingWorker(n_procs=2), cache_root=tmp_path) as sub:
        result = sub(wf)
    assert result.errored
    # NB: if the workflow spins, the pytest timeout is raised within it and captured
    # as a job error, so also check that it didn't take anywhere near that long
    assert time.monotonic() - start < 45


def test_node_update_status_errored_running_job():
    """Jobs that error while "running" should be moved to errored rather than raising
    the ValueError from Job.done (as is already the case for "queued" jobs)"""
    from datetime import datetime
    from types import SimpleNamespace
    from pydra.engine.submitter import NodeExecution

    class ErroredJob:
        state_index = 0
        errored = False

        @property
        def done(self):
            raise ValueError("Job failed")

    job = ErroredJob()
    node_exec = NodeExecution(
        node=SimpleNamespace(name="node"), submitter=None, workflow=None
    )
    node_exec.blocked = {}
    node_exec.running = {0: (job, datetime.now())}
    node_exec.update_status()
    assert node_exec.errored == {0: job}
    assert node_exec.running == {}


@python.define
def FailEvenUnlessFlagged(x: int, flag: str) -> int:
    """Raise on even ``x`` until the file ``flag`` exists"""
    from pathlib import Path

    if x % 2 == 0 and not Path(flag).exists():
        raise ValueError("even error")
    return x


@python.define
def Double(y: int) -> int:
    return 2 * y


@workflow.define
def RecoverableWorkflow(x: list[int], flag: str) -> list[int]:
    failing = workflow.add(FailEvenUnlessFlagged(flag=flag).split("x", x=x))
    doubled = workflow.add(Double(y=failing.out))
    return doubled.out


def test_rerun_errored_recovers(tmp_path, worker):
    """Elements that errored in a previous run and succeed when rerun are reported by
    that rerun, and their downstream elements run, rather than the previous run's
    errors being reported again"""
    flag = tmp_path / "flag"
    wf = RecoverableWorkflow(x=[1, 2, 3, 4, 5], flag=str(flag))
    with pytest.raises(Exception, match="even error"):
        wf(worker=worker, cache_root=tmp_path / "cache")
    flag.touch()
    outputs = wf(worker=worker, cache_root=tmp_path / "cache")
    assert outputs.out == [2, 4, 6, 8, 10]


def test_rerun_errored_slots_not_read_while_rerunning(tmp_path, monkeypatch):
    """The submitter does not read the stored result of a job it has queued until the
    job's run has finished"""
    from pydra.engine import submitter as submitter_module
    from pydra.engine.store import PickleStore
    from pydra.workers import cf

    in_flight: set[str] = set()
    reads_in_flight: list[str] = []

    release = submitter_module.NodeExecution._release

    def recording_release(self, index):
        release(self, index)
        if index in self.queued:
            in_flight.add(self.queued[index].checksum)

    run = cf.Worker.run

    async def recording_run(self, job, rerun=False):
        try:
            return await run(self, job, rerun=rerun)
        finally:
            in_flight.discard(job.checksum)

    lookup = PickleStore.lookup

    def recording_lookup(self, checksum, context, task):
        if checksum in in_flight:
            reads_in_flight.append(checksum)
        return lookup(self, checksum, context, task)

    monkeypatch.setattr(submitter_module.NodeExecution, "_release", recording_release)
    monkeypatch.setattr(cf.Worker, "run", recording_run)
    monkeypatch.setattr(PickleStore, "lookup", recording_lookup)

    flag = tmp_path / "flag"
    wf = RecoverableWorkflow(x=[1, 2, 3, 4, 5], flag=str(flag))
    with pytest.raises(RuntimeError, match="even error"):
        wf(worker="cf", cache_root=tmp_path / "cache")
    reads_in_flight.clear()
    flag.touch()
    assert wf(worker="cf", cache_root=tmp_path / "cache").out == [2, 4, 6, 8, 10]
    assert reads_in_flight == []


@python.define
def Identity(x: int) -> int:
    return x


@python.define
def Half(x: int) -> int:
    return x // 2


@workflow.define
def SameChecksumElements(x: list[int]) -> list[int]:
    first = workflow.add(Identity().split("x", x=x), name="first")
    return first.out


@workflow.define(outputs=["first", "again"])
def SameChecksumLater(x: int) -> tuple[int, int]:
    first = workflow.add(Identity(x=x), name="first")
    relay = workflow.add(Double(y=x), name="relay")
    half = workflow.add(Half(x=relay.out), name="half")
    again = workflow.add(Identity(x=half.out), name="again")
    return first.out, again.out


class CountingWorker(cf.Worker):
    """A cf worker recording the checksums of the ``Identity`` jobs it is asked to run"""

    _plugin_name = "counting-cf"
    ran: ty.ClassVar[list[str]] = []

    async def run(self, job, rerun=False):
        if job.name in ("first", "again"):
            self.ran.append(job.checksum)
        return await super().run(job, rerun=rerun)


class FailingCountingWorker(CountingWorker):
    """A counting worker that fails to run the ``Identity`` jobs"""

    _plugin_name = "failing-counting-cf"

    async def run(self, job, rerun=False):
        if job.name in ("first", "again"):
            self.ran.append(job.checksum)
            raise RuntimeError("worker was unable to run the job")
        return await super().run(job, rerun=rerun)


def test_same_checksum_elements_run_once(tmp_path, monkeypatch):
    """Elements with the same checksum are run once and all read its result"""
    monkeypatch.setattr(CountingWorker, "ran", [])
    with Submitter(worker=CountingWorker(n_procs=2), cache_root=tmp_path) as sub:
        result = sub(SameChecksumElements(x=[1, 1, 2]))
    assert not result.errored
    assert result.outputs.out == [1, 1, 2]
    assert len(CountingWorker.ran) == 2


@pytest.mark.timeout(60)
@pytest.mark.parametrize("workflow_task", ["elements", "later"])
def test_same_checksum_worker_error_before_job_saved(
    tmp_path, monkeypatch, workflow_task
):
    """If a worker fails to run a job before it saves a result, every job with the
    same checksum, including one released afterwards, errors with it"""
    monkeypatch.setattr(CountingWorker, "ran", [])
    task = (
        SameChecksumElements(x=[1, 1, 1])
        if workflow_task == "elements"
        else SameChecksumLater(x=1)
    )
    with Submitter(worker=FailingCountingWorker(n_procs=2), cache_root=tmp_path) as sub:
        result = sub(task)
    assert result.errored
    assert len(CountingWorker.ran) == 1
