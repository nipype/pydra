"""Flattening: one runner per ``Submitter`` call leases, checks, expands and commits the
workflow jobs of every level, and hands out the leaf jobs of all of them together"""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
import attrs
import pytest
from pydra.compose import python, workflow
from pydra.engine.drivers import drive_inline
from pydra.engine.executor import InlineExecutor
from pydra.engine.hooks import TaskHooks
from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.runner import GraphRunner
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter, WorkflowError

# a holder is taken over after one second without a renewal
SETTINGS = LeaseSettings(stale_after=1, timeout=30, attribute_delay=0)
# a holder that keeps renewing is waited for for as long as it runs
FOREVER = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, max_wait=0.5)


@python.define
def Span(x: int, logdir: str, hold: float = 0.4) -> int:
    """Runs for ``hold`` seconds and records when"""
    start = time.time()
    time.sleep(hold)
    (Path(logdir) / f"{os.getpid()}-{x}.span").write_text(f"{start} {time.time()}")
    return x


@workflow.define
def Inner(tag: int, logdir: str) -> list[int]:
    s = workflow.add(Span(logdir=logdir).split(x=[tag * 10 + i for i in range(4)]))
    return s.out


def make_outer(nested: int):
    @workflow.define
    def Outer(logdir: str) -> list[int]:
        outs = [
            workflow.add(Inner(tag=t, logdir=logdir), name=f"in{t}").out
            for t in range(nested)
        ]
        return outs[0]

    return Outer


def peak_overlap(logdir: Path) -> int:
    spans = [
        tuple(float(v) for v in p.read_text().split()) for p in logdir.glob("*.span")
    ]
    events = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    peak = running = 0
    for _, step in events:
        running += step
        peak = max(peak, running)
    return peak


def test_max_concurrent_caps_whole_run(tmp_path):
    logdir = tmp_path / "spans"
    logdir.mkdir()
    # the pool is larger than the cap whatever the host, so the cap is the bound
    with Submitter(
        worker="cf", cache_root=tmp_path / "cache", max_concurrent=4, n_procs=8
    ) as sub:
        sub(make_outer(8)(logdir=str(logdir)))
    assert len(list(logdir.glob("*.span"))) == 32
    assert peak_overlap(logdir) == 4


@python.define
def Twin(x: int) -> int:
    return x + 1


def make_twins(log: str):
    def note_pre_run(job):
        with open(log, "a") as f:
            f.write(f"{job.name}\n")

    @workflow.define
    def Branch(x: int, tag: str) -> int:
        t = workflow.add(Twin(x=x), hooks=TaskHooks(pre_run=note_pre_run), name="twin")
        return t.out

    @workflow.define(outputs=["first", "second"])
    def Twins(x: int):
        a = workflow.add(Branch(x=x, tag="a"), name="a")
        b = workflow.add(Branch(x=x, tag="b"), name="b")
        return a.out, b.out

    return Twins


def test_equal_checksum_across_nested_runs_once(tmp_path):
    """Two nested workflows hold a job with one checksum: it is run once, because the
    flattened round holds both and the second is folded into the first"""
    log = tmp_path / "pre_run.log"
    with Submitter(worker="cf", cache_root=tmp_path / "cache") as sub:
        outputs = sub(make_twins(str(log))(x=3)).outputs
    assert (outputs.first, outputs.second) == (4, 4)
    assert log.read_text().splitlines() == ["twin"]


# Leases of workflow jobs


@python.define
def Counted(log: str, hold: float) -> int:
    with open(log, "a") as f:
        f.write(f"{os.getpid()}\n")
    Path(log + ".started").touch()
    time.sleep(hold)
    return 7


@workflow.define
def Wrapped(log: str, hold: float) -> int:
    c = workflow.add(Counted(log=log, hold=hold))
    return c.out


@workflow.define
def AfterWrapped(log: str, hold: float) -> int:
    w = workflow.add(Wrapped(log=log, hold=hold), name="wrapped")
    after = workflow.add(Twin(x=w.out), name="after")
    return after.out


def runner_for(root: Path, task, settings=SETTINGS, **kwargs):
    store = PickleStore(root, lease_settings=settings)
    sub = Submitter(worker="debug", store=store)
    job = Job(task, submitter=sub, name="main")
    runner = GraphRunner(
        task, store, on_error="raise_at_end", submitter=sub, job=job, **kwargs
    )
    return sub, runner


def drive(sub, runner):
    drive_inline(
        runner,
        InlineExecutor(sub.worker, sub.worker_for),
        on_error="raise_at_end",
        audit=sub.audit,
    )


# Another process holds the lease of the workflow job. It is given the cache root, the
# checksum and the lock path as the first process computed them, and imports nothing of
# this module, so these tests do not depend on how the package is installed.
HOLDER = textwrap.dedent("""
    import json, sys, time
    from pathlib import Path
    from pydra.engine.lease import DirectoryLease, LeaseSettings

    spec = json.loads(sys.argv[1])
    lease = DirectoryLease(
        spec["checksum"],
        Path(spec["cache_dir"]),
        Path(spec["lockfile"]),
        settings=LeaseSettings(**spec["settings"]),
    )
    lease.__enter__()
    Path(spec["ready"]).touch()
    time.sleep(spec["hold"])
    lease.__exit__(None, None, None)
    """)


def spawn_holder(job, store, ready: Path, hold: float) -> subprocess.Popen:
    spec = {
        "checksum": job.checksum,
        "cache_dir": str(job.cache_dir),
        "lockfile": str(store.lockfile(job.checksum, job.context)),
        "ready": str(ready),
        "hold": hold,
        "settings": {"stale_after": 1, "timeout": 30, "attribute_delay": 0},
    }
    return subprocess.Popen(
        [sys.executable, "-c", HOLDER, json.dumps(spec)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for(path: Path, proc: subprocess.Popen, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if proc.poll() is not None:
            _, err = proc.communicate()
            pytest.fail(f"the holder ended before taking the lease:\n{err}")
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.02)


def finish(proc: subprocess.Popen, killed=False):
    if killed:
        proc.kill()
    _, err = proc.communicate(timeout=60)
    if not killed:
        assert proc.returncode == 0, err
    assert "Traceback" not in err, err


def count_expansions(monkeypatch):
    from pydra.engine.workflow import Workflow

    calls = []
    original = Workflow.execution_graph

    def counted(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Workflow, "execution_graph", counted)
    return calls


def log_lines(log: Path) -> list[str]:
    return log.read_text().splitlines() if log.exists() else []


@pytest.mark.timeout(120)
def test_second_submitter_waits_then_reads_stored(tmp_path, monkeypatch):
    log, root, ready = tmp_path / "runs.log", tmp_path / "cache", tmp_path / "held"
    sub0, first = runner_for(root, Wrapped(log=str(log), hold=0.0))
    with sub0:
        drive(sub0, first)
        assert first.result().outputs.out == 7
    assert len(log_lines(log)) == 1
    sub, runner = runner_for(root, Wrapped(log=str(log), hold=0.0))
    holder = spawn_holder(runner.job, sub.store, ready, hold=2.0)
    try:
        wait_for(ready, holder)
        expansions = count_expansions(monkeypatch)
        with sub:
            rnd = runner.ready()
            assert rnd.dispatch == []
            assert rnd.retry_after is not None
            assert not runner.done
            drive(sub, runner)
            assert runner.result().outputs.out == 7
        assert expansions == []
        assert not runner.errors
    finally:
        finish(holder)
    assert len(log_lines(log)) == 1


@pytest.mark.timeout(120)
def test_stale_workflow_lease_taken_over(tmp_path):
    log, root, ready = tmp_path / "runs.log", tmp_path / "cache", tmp_path / "held"
    sub, runner = runner_for(root, Wrapped(log=str(log), hold=0.0))
    holder = spawn_holder(runner.job, sub.store, ready, hold=600)
    try:
        wait_for(ready, holder)
        finish(holder, killed=True)
        with sub:
            assert runner.ready().retry_after is not None  # the lease is not yet stale
            start = time.monotonic()
            drive(sub, runner)
            assert runner.result().outputs.out == 7
        assert time.monotonic() - start < 20
        assert not runner.errors
    finally:
        holder.kill()
    assert len(log_lines(log)) == 1


@pytest.mark.timeout(120)
def test_live_holder_not_taken_over_past_stale_after(tmp_path, monkeypatch):
    """A holder that renews its lease is waited for past three times ``stale_after``
    on the storage clock, so the second submitter keeps waiting and never expands.

    The holder is a lease of this process whose own heartbeat is stopped: the test
    makes its beats between polls, and the waiter's storage clock is a fake that moves
    0.3 s per poll, so neither depends on how the OS schedules threads. The real
    heartbeat across processes is covered by
    ``test_lease.py::test_real_heartbeat_keeps_a_waiter_in_another_process_out``."""
    log, root = tmp_path / "runs.log", tmp_path / "cache"
    clock = {"t": 0.0}
    base = [0.0]

    def storage_clock() -> float:
        return base[0] + clock["t"]

    taken = []
    acquire = PickleStore.acquire

    def acquire_with_fake_clock(self, checksum, context):
        lease = attrs.evolve(acquire(self, checksum, context), clock=storage_clock)
        taken.append(lease)
        return lease

    # the runner makes its lease when it is built, so the fake clock goes in first
    monkeypatch.setattr(PickleStore, "acquire", acquire_with_fake_clock)
    sub, runner = runner_for(root, Wrapped(log=str(log), hold=0.0))
    monkeypatch.setattr(PickleStore, "acquire", acquire)
    assert len(taken) == 1
    holder = sub.store.acquire(runner.job.checksum, runner.job.context)
    holder.__enter__()
    holder._heartbeat.stopped.set()  # beats are the test's, below
    holder._heartbeat.join()
    token_dir = holder.lockfile / holder._token
    base[0] = max(
        p.lstat().st_mtime for p in [holder.lockfile, *holder.lockfile.rglob("*")]
    )
    try:
        expansions = count_expansions(monkeypatch)
        with sub:
            retries = 0
            while clock["t"] <= 3 * SETTINGS.stale_after:
                for frame in runner._frames:  # judge at every poll, not each backoff
                    frame.retry_at = None
                rnd = runner.ready()
                assert rnd.dispatch == [] and expansions == []
                if rnd.retry_after is not None:
                    retries += 1
                clock["t"] += 0.3
                # the holder renews: a mark at the new storage time
                os.utime(token_dir, (storage_clock(), storage_clock()))
            assert retries > 1
            assert expansions == []
            assert [lease.broke for lease in taken] == [[]]
            holder.__exit__(None, None, None)
            drive(sub, runner)  # the lease is free: the waiter now expands
            assert runner.result().outputs.out == 7
    finally:
        if holder.held:
            holder.__exit__(None, None, None)
    assert len(log_lines(log)) == 1


def test_lease_timeout_is_entry_readers_unrunnable(tmp_path):
    """The holder of a nested workflow's lease is another runner in this process,
    which keeps renewing: the waiter gives up at ``max_wait``"""
    log = tmp_path / "runs.log"
    root = tmp_path / "cache"
    store = PickleStore(root, lease_settings=FOREVER)
    with Submitter(worker="debug", store=store) as sub:
        holder = GraphRunner(
            AfterWrapped(log=str(log), hold=0.0),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        assert [j.name for j in holder.ready().dispatch] == ["Counted"]  # holds it
        runner = GraphRunner(
            AfterWrapped(log=str(log), hold=0.0),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        rnd = runner.ready()
        assert rnd.dispatch == [] and rnd.retry_after is not None
        assert runner.stuck() is None
        deadline = time.monotonic() + 30
        while not runner.done:
            assert time.monotonic() < deadline
            time.sleep(rnd.retry_after or 0.05)
            rnd = runner.ready()  # does not raise
        (entry,) = runner.entries
        assert entry.phase == "lease"
        assert entry.job_name == "wrapped"
        assert entry.exception_type.endswith("LeaseTimeout")
        after = runner.exec_graph["after"]
        assert after.unrunnable == {None: [runner.exec_graph["wrapped"]]}
        assert runner.failed
        holder.close()


def test_stuck_none_while_lease_waits(tmp_path):
    log = tmp_path / "runs.log"
    store = PickleStore(tmp_path / "cache", lease_settings=FOREVER)
    with Submitter(worker="debug", store=store) as sub:
        holder = GraphRunner(
            Wrapped(log=str(log), hold=0.0),
            store,
            on_error="raise_at_end",
            submitter=sub,
            job=Job(Wrapped(log=str(log), hold=0.0), submitter=sub, name="main"),
        )
        assert [j.name for j in holder.ready().dispatch] == ["Counted"]
        sub2 = Submitter(worker="debug", store=store)
        task = Wrapped(log=str(log), hold=0.0)
        runner = GraphRunner(
            task,
            store,
            on_error="raise_at_end",
            submitter=sub2,
            job=Job(task, submitter=sub2, name="main"),
        )
        rnd = runner.ready()
        assert rnd.dispatch == [] and rnd.retry_after is not None
        assert not runner.done
        assert runner.stuck() is None
        holder.close()
        sub2.close()


def break_lease(job, result):
    lockfile = job.store.acquire(job.checksum, job.context).lockfile
    shutil.rmtree(lockfile)


@workflow.define
def LeaseBroken(x: int) -> int:
    inner = workflow.add(
        Wrapped(log=os.devnull, hold=0.0),
        hooks=TaskHooks(post_run_task=break_lease),
        name="inner",
    )
    return inner.out


def test_lease_lost_on_commit_is_entry(tmp_path):
    with Submitter(
        worker="debug", store="pickle", cache_root=tmp_path, on_error="raise_at_end"
    ) as sub:
        with pytest.raises(WorkflowError) as raised:
            sub(LeaseBroken(x=1))
    (entry,) = [e for e in raised.value.entries if e.job_name == "inner"]
    assert entry.phase == "commit"
    assert entry.exception_type.endswith("LeaseLost")


def test_attempt_is_given_to_workflow_jobs(tmp_path):
    task = AfterWrapped(log=os.devnull, hold=0.0)
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            task,
            sub.store,
            on_error="raise_at_end",
            submitter=sub,
            job=Job(task, submitter=sub, name="main"),
        )
        runner.ready()
        outer = runner.job
        inner = runner.exec_graph["wrapped"].job(None)
    assert outer.attempt is not None
    assert inner.attempt is not None and inner.attempt != outer.attempt


# Empty splits and cached nested workflows


@workflow.define
def Emptied(xs: list[int], logdir: str) -> list[int]:
    s = workflow.add(Span(logdir=logdir).split(x=xs))
    return s.out


@workflow.define
def NestedEmpty(xs: list[int], logdir: str) -> list[int]:
    e = workflow.add(Emptied(xs=xs, logdir=logdir), name="emptied")
    return e.out


@workflow.define
def SplitEmpty(xs: list[int], logdir: str) -> list[list[int]]:
    e = workflow.add(Emptied(logdir=logdir).split(xs=[xs, xs]), name="emptied")
    return e.out


def test_empty_split_inside_a_nested_workflow(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        result = sub(NestedEmpty(xs=[], logdir=str(tmp_path)))
    assert result.outputs.out == []


def test_empty_split_of_a_nested_workflow(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        result = sub(SplitEmpty(xs=[], logdir=str(tmp_path)))
    assert result.outputs.out == [[], []]


@pytest.mark.parametrize("rerun", [False, True])
def test_cached_nested_workflow_rerun(tmp_path, rerun):
    """A nested workflow stored by an earlier run is not expanded again, unless the
    rerun reaches it"""
    logdir = tmp_path / "spans"
    logdir.mkdir()
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache) as sub:
        sub(make_outer(1)(logdir=str(logdir)))
    assert len(list(logdir.glob("*.span"))) == 4

    @workflow.define
    def Other(logdir: str) -> int:
        workflow.add(Inner(tag=0, logdir=logdir), name="in0")
        extra = workflow.add(Twin(x=1), name="extra")
        return extra.out

    for p in logdir.glob("*.span"):
        p.unlink()
    with Submitter(worker="debug", cache_root=cache) as sub:
        sub(Other(logdir=str(logdir)), rerun=rerun)
    assert len(list(logdir.glob("*.span"))) == (4 if rerun else 0)


# An interrupted workflow is not committed


@python.define
def Boom(x: int) -> int:
    raise ValueError("boom")


@python.define
def Sink(marker: str, gate: str) -> int:
    """Waits for ``gate`` to exist, if given, then records that it ran"""
    import time
    import uuid

    if gate:
        deadline = time.monotonic() + 60
        while not Path(gate).exists() and time.monotonic() < deadline:
            time.sleep(0.05)
    Path(marker, f"sink-{uuid.uuid4().hex}").touch()
    return 1


def make_branches(reads_sink: bool):
    @workflow.define
    def Failing(x: int) -> int:
        return workflow.add(Boom(x=x)).out

    @workflow.define
    def Sunk(marker: str, gate: str) -> int:
        out = workflow.add(Twin(x=1), name="out")
        sink = workflow.add(Sink(marker=marker, gate=gate), name="sink")
        return sink.out if reads_sink else out.out

    @workflow.define
    def Branches(marker: str, gate: str) -> int:
        workflow.add(Failing(x=1), name="a")
        b = workflow.add(Sunk(marker=marker, gate=gate), name="b")
        workflow.add(Twin(x=b.out), name="after")  # blocked on b, never handed out
        return b.out

    @workflow.define
    def Fixed(marker: str, gate: str) -> int:
        return workflow.add(Sunk(marker=marker, gate=gate), name="b").out

    return Sunk, Branches, Fixed


@pytest.mark.timeout(120)
@pytest.mark.parametrize("reads_sink", [False, True])
@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_interrupted_nested_workflow_is_not_committed(tmp_path, worker, reads_sink):
    """Branch ``a`` fails, so ``fail_fast`` stops the run with workflow ``b`` unfinished:
    ``b`` is released and stores nothing, and the next run executes its ``sink``"""
    Sunk, Branches, Fixed = make_branches(reads_sink)
    marker = tmp_path / "marker"
    marker.mkdir()
    release = tmp_path / "release"
    gate = str(release) if worker == "cf" else ""
    cache = tmp_path / "cache"
    extra = {"n_procs": 4} if worker == "cf" else {}
    sub = Submitter(worker=worker, cache_root=cache, on_error="fail_fast", **extra)
    try:
        with pytest.raises(Exception):
            sub(Branches(marker=str(marker), gate=gate), raise_errors=True)
        b = Job(Sunk(marker=str(marker), gate=gate), submitter=sub, name="b")
        assert b.result() is None
        assert not list(cache.glob("workflow-*.lock"))
    finally:
        release.touch()
        sub.close()
    sinks = len(list(marker.iterdir()))
    with Submitter(worker=worker, cache_root=cache) as sub2:
        result = sub2(Fixed(marker=str(marker), gate=gate))
    assert not result.errored
    assert result.outputs.out == (1 if reads_sink else 2)
    if worker == "debug":
        assert (sinks, len(list(marker.iterdir()))) == (0, 1)


def test_close_commits_an_errored_workflow_with_errors_and_nothing_else(tmp_path):
    from pydra.engine.runner import Failed

    Sunk, Branches, Fixed = make_branches(False)
    task = Branches(marker=str(tmp_path), gate="")
    sub, runner = runner_for(tmp_path / "cache", task)
    with sub:
        leaves = runner.ready().dispatch
        names = {j.name for j in leaves}
        assert {"Boom", "out", "sink"} <= names
        boom = next(j for j in leaves if j.name == "Boom")
        runner.complete(boom, Failed(ValueError("boom")))
        runner.close()
        graph = runner.exec_graph
        a, b = graph["a"].job(None), graph["b"].job(None)
        assert a.result().errored  # it has an error of its own
        assert b.result() is None  # it was only interrupted
        assert runner.job.result().errored
        assert not list(sub.store.lockfile(a.checksum, a.context).parent.glob("*.lock"))


def test_close_releases_the_leases(tmp_path):
    sub, runner = runner_for(tmp_path / "cache", AfterWrapped(log=os.devnull, hold=0.0))
    with sub:
        assert [j.name for j in runner.ready().dispatch] == ["Counted"]
        lockfiles = list((tmp_path / "cache").glob("*.lock"))
        assert len(lockfiles) == 2  # the outer and the nested workflow
        runner.close()
        runner.close()  # idempotent
        assert not list((tmp_path / "cache").glob("*.lock"))
        _, again = runner_for(
            tmp_path / "cache", AfterWrapped(log=os.devnull, hold=0.0)
        )
        assert again.ready().retry_after is None


# Errors while starting a workflow job are entries with the real exception


def refuse(job, *args):
    raise PermissionError("hook says no")


def test_error_starting_a_nested_workflow_is_an_entry(tmp_path):
    @workflow.define
    def Starting(x: int) -> int:
        w = workflow.add(
            Wrapped(log=os.devnull, hold=0.0),
            hooks=TaskHooks(pre_run_task=refuse),
            name="inner",
        )
        return w.out

    with Submitter(worker="debug", cache_root=tmp_path, on_error="raise_at_end") as sub:
        with pytest.raises(WorkflowError) as raised:
            sub(Starting(x=1))
    (entry,) = [e for e in raised.value.entries if e.job_name == "inner"]
    assert entry.exception_type == "builtins.PermissionError"
    assert "hook says no" in entry.message


def test_error_starting_the_outer_workflow_is_raised_as_itself(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        with pytest.raises(PermissionError, match="hook says no"):
            sub(
                Wrapped(log=os.devnull, hold=0.0),
                hooks=TaskHooks(pre_run_task=refuse),
            )


@pytest.mark.parametrize("hook", ["pre_run"])
def test_error_in_a_nested_pre_run_hook_is_an_entry(tmp_path, hook):
    @workflow.define(outputs=["first", "second"])
    def Hooked(x: int, marker: str):
        w = workflow.add(
            Wrapped(log=os.devnull, hold=0.0),
            hooks=TaskHooks(**{hook: refuse}),
            name="inner",
        )
        s = workflow.add(Sink(marker=marker, gate=""), name="other")
        return w.out, s.out

    marker = tmp_path / "marker"
    marker.mkdir()
    with Submitter(
        worker="debug", cache_root=tmp_path / "cache", on_error="raise_at_end"
    ) as sub:
        with pytest.raises(WorkflowError) as raised:
            sub(Hooked(x=1, marker=str(marker)))
    (entry,) = [e for e in raised.value.entries if e.job_name == "inner"]
    assert entry.exception_type == "builtins.PermissionError"
    assert len(list(marker.iterdir())) == 1  # the independent branch ran


# The runner places jobs in their workflows, and waits without a current task


def test_runner_sets_the_workflow_path(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        runner = GraphRunner(
            make_outer(2)(logdir=str(tmp_path)),
            sub.store,
            on_error="raise_at_end",
            submitter=sub,
        )
        leaves = runner.ready().dispatch
        assert len(leaves) == 8
        assert {j.workflow_path for j in leaves} == {("in0",), ("in1",)}
        runner.close()


@workflow.define
def Passthrough(x: int) -> int:
    return x


def test_workflow_without_nodes_on_cf(tmp_path):
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        assert sub(Passthrough(x=3)).outputs.out == 3


NEST_ASYNCIO_SCRIPT = textwrap.dedent("""
    import asyncio, sys
    import nest_asyncio
    from pydra.compose import python, workflow
    from pydra.engine.submitter import Submitter

    @python.define
    def Inc(x: int) -> int:
        return x + 1

    @workflow.define
    def Wf(x: int) -> int:
        return workflow.add(Inc(x=x)).out

    async def main(cache):
        with Submitter(worker="cf", cache_root=cache) as sub:
            return sub(Wf(x=1), raise_errors=True).outputs.out

    if __name__ == "__main__":
        nest_asyncio.apply()
        print(asyncio.run(main(sys.argv[1])))
    """)


def test_workflow_on_an_async_worker_inside_nest_asyncio(tmp_path):
    """A notebook applies ``nest_asyncio``, under which no task is current (3.14)"""
    if importlib.util.find_spec("nest_asyncio") is None:  # imported by the child only
        pytest.skip("nest_asyncio is not installed")
    script = tmp_path / "nb.py"
    script.write_text(NEST_ASYNCIO_SCRIPT)
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "cache")],
        capture_output=True,
        text=True,
        timeout=100,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "2"


# Arrays of the elements of split nodes in nested workflows


@python.define
def Shift(x: int, by: int) -> int:
    return x + by


@workflow.define
def Shifted(by: int) -> list[int]:
    return workflow.add(Shift(by=by).split(x=[1, 2, 3]), name="shift").out


@workflow.define
def TwoShifts(a: int, b: int) -> list[int]:
    first = workflow.add(Shifted(by=a), name="first")
    workflow.add(Shifted(by=b), name="second")
    return first.out


def test_nested_arrays_by_path(any_worker, tmp_path):
    """Nodes of one name in two nested workflows go to two arrays, not one"""
    with Submitter(worker=any_worker, cache_root=tmp_path / "cache") as sub:
        result = sub(TwoShifts(a=10, b=20))
    assert result.outputs.out == [11, 12, 13]
    if any_worker == "slurm":
        arrays = list((tmp_path / "cache").glob("slurm_scripts/array_*_3"))
        assert len(arrays) == 2


# The asynchronous driver waits on a workflow lease another process holds


@pytest.mark.timeout(60)
def test_async_driver_waits_on_a_held_workflow_lease(tmp_path):
    """With nothing in flight, only the timed wake brings the driver back to the lease"""
    log, root, ready = tmp_path / "runs.log", tmp_path / "cache", tmp_path / "held"
    task = Wrapped(log=str(log), hold=0.0)
    store = PickleStore(root, lease_settings=SETTINGS)
    with Submitter(worker="cf", store=store) as sub:
        holder = spawn_holder(Job(task, submitter=sub, name="main"), store, ready, 1.5)
        try:
            wait_for(ready, holder)
            start = time.monotonic()
            result = sub(task)
            waited = time.monotonic() - start
        finally:
            finish(holder)
    assert result.outputs.out == 7
    assert waited > 0.5  # it did wait for the holder
    assert len(log_lines(log)) == 1


# Each guard against committing an interrupted workflow, alone


def interrupted_runner(tmp_path):
    from pydra.engine.runner import Failed

    Sunk, Branches, Fixed = make_branches(False)
    sub, runner = runner_for(
        tmp_path / "cache", Branches(marker=str(tmp_path), gate="")
    )
    leaves = runner.ready().dispatch
    boom = next(j for j in leaves if j.name == "Boom")
    runner.complete(boom, Failed(ValueError("boom")))
    return sub, runner


def test_close_does_not_finish_a_frame_without_errors(tmp_path, monkeypatch):
    sub, runner = interrupted_runner(tmp_path)
    with sub:
        finished = []
        original = runner._finish
        monkeypatch.setattr(
            runner,
            "_finish",
            lambda f, *a: (finished.append(f.job.name), original(f, *a)),
        )
        runner.close()
        assert "b" not in finished and "a" in finished


def test_finish_stores_nothing_for_unfinished_nodes_without_errors(tmp_path):
    sub, runner = interrupted_runner(tmp_path)
    with sub:
        frame = runner._child[runner.exec_graph["b"].job(None)]
        runner._finish(frame)  # called directly: it must refuse on its own
        assert frame.job.result() is None
        assert frame.state == "finished" and frame.lease is None
        runner.close()


def test_close_names_the_stopped_elements(tmp_path):
    """An element handed out that did not complete is "stopped", once, naming the failure
    that stopped it; completed elements and blocked ones that were never handed out
    have no such entry"""
    sub, runner = interrupted_runner(tmp_path)
    with sub:
        runner.close()
        stopped = [e for e in runner.entries if e.phase == "stopped"]
        assert [(e.job_name, e.state_index) for e in stopped] == [("b", None)]
        (entry,) = stopped
        assert entry.exception_type is None
        assert entry.message.startswith("stopped by a: ")
        assert entry.message.count("stopped by") == 1
        assert "\n" not in entry.message  # the failure is named, not repeated
        # the failure is the one real entry, and the only error
        real = [e for e in runner.entries if e.phase != "stopped"]
        assert [(e.job_name, e.phase) for e in real] == [("a", "run")]
        assert len(runner.errors) == 1


@python.define
def FailsOrHolds(x: int, hold: float = 3.0) -> int:
    """Fails after a short wait when ``x`` is negative, otherwise runs for ``hold``"""
    if x < 0:
        time.sleep(0.5)
        raise ValueError("fails on a negative x")
    time.sleep(hold)
    return x


@workflow.define
def FailAlongsideSplit(xs: list[int]) -> list[int]:
    workflow.add(FailsOrHolds(x=-1), name="f")
    s = workflow.add(FailsOrHolds().split(x=xs), name="s")
    return s.out


def test_stopped_entries_only_for_elements_handed_out(tmp_path):
    """Under ``max_concurrent`` released elements held back by the cap were never handed
    out, so they get no "stopped" entry; only those handed out and unfinished do"""
    with Submitter(
        worker="cf",
        cache_root=tmp_path,
        on_error="fail_fast",
        max_concurrent=2,
        n_procs=4,
    ) as sub:
        with pytest.raises(WorkflowError) as raised:
            sub(FailAlongsideSplit(xs=[10, 11, 12, 13, 14, 15]), raise_errors=True)
    entries = raised.value.entries
    stopped = [(e.job_name, e.state_index) for e in entries if e.phase == "stopped"]
    # the cap of two leaves room for one element of ``s`` beside ``f``
    assert stopped == [("s", 0)]
    (entry,) = [e for e in entries if e.phase == "stopped"]
    assert entry.exception_type is None
    assert entry.message == "stopped by f: fails on a negative x"
    assert [(e.job_name, e.phase) for e in entries if e.phase != "stopped"] == [
        ("f", "run")
    ]


# Failures that are entries


def test_error_reading_a_nested_result_is_an_entry(tmp_path, monkeypatch):
    original = Job.result

    def result(self, *args, **kwargs):
        if self.name == "wrapped":
            raise PermissionError("cannot read")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Job, "result", result)
    with Submitter(
        worker="debug", store="pickle", cache_root=tmp_path, on_error="raise_at_end"
    ) as sub:
        with pytest.raises(WorkflowError) as raised:
            sub(AfterWrapped(log=os.devnull, hold=0.0))
    (entry,) = [e for e in raised.value.entries if e.job_name == "wrapped"]
    assert entry.exception_type == "builtins.PermissionError"


def test_error_making_the_outer_lease_is_raised_as_itself(tmp_path, monkeypatch):
    def refuse_acquire(self, checksum, context):
        raise PermissionError("no lease")

    monkeypatch.setattr(PickleStore, "acquire", refuse_acquire)
    with Submitter(worker="debug", store="pickle", cache_root=tmp_path) as sub:
        task = Wrapped(log=os.devnull, hold=0.0)
        runner = GraphRunner(  # the failure is the runner's state, not a raise
            task,
            sub.store,
            on_error="raise_at_end",
            submitter=sub,
            job=Job(task, submitter=sub, name="main"),
        )
        assert runner.done and isinstance(runner.root_error, PermissionError)
        with pytest.raises(PermissionError, match="no lease"):
            sub(Wrapped(log=os.devnull, hold=0.0))


def test_failure_at_start_runs_no_post_hook(tmp_path):
    seen = []

    def post(job, result):
        seen.append(job.name)

    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        with pytest.raises(PermissionError):
            sub(
                Wrapped(log=os.devnull, hold=0.0),
                hooks=TaskHooks(pre_run_task=refuse, post_run_task=post),
            )
    assert seen == []
