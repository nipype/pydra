"""Flattening: one runner per ``Submitter`` call leases, checks, expands and commits the
workflow jobs of every level, and hands out the leaf jobs of all of them together"""

import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
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
    with Submitter(worker="cf", cache_root=tmp_path / "cache", max_concurrent=4) as sub:
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


def wrapped(log, hold):
    """``Wrapped`` as another process imports it: pytest names this module differently,
    which would give the task another checksum"""
    import importlib

    module = importlib.import_module("pydra.engine.tests.test_runner_flatten")
    return module.Wrapped(log=str(log), hold=hold)


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


def hold_in_process(root, log, hold):
    """Run ``Wrapped`` to completion as another process does"""
    store = PickleStore(root, lease_settings=SETTINGS)
    with Submitter(worker="debug", store=store) as sub:
        sub(wrapped(log, hold))


def spawn(call: str, *args) -> subprocess.Popen:
    code = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from pydra.engine.tests.test_runner_flatten import {call}
        {call}(*[Path(a) if a.startswith("/") else float(a) for a in sys.argv[1:]])
        """)
    return subprocess.Popen(
        [sys.executable, "-c", code, *[str(a) for a in args]],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def wait_for(path: Path, timeout=30):
    deadline = time.monotonic() + timeout
    while not path.exists():
        assert time.monotonic() < deadline, f"{path} did not appear"
        time.sleep(0.02)


def count_expansions(monkeypatch):
    from pydra.engine.workflow import Workflow

    calls = []
    original = Workflow.execution_graph

    def counted(self, *args, **kwargs):
        calls.append(self.name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Workflow, "execution_graph", counted)
    return calls


@pytest.mark.timeout(120)
def test_second_submitter_waits_then_reads_stored(tmp_path, monkeypatch):
    log = tmp_path / "runs.log"
    first = spawn("hold_in_process", tmp_path / "cache", log, 2.0)
    try:
        wait_for(Path(str(log) + ".started"))
        expansions = count_expansions(monkeypatch)
        sub, runner = runner_for(tmp_path / "cache", wrapped(log, 2.0), SETTINGS)
        with sub:
            rnd = runner.ready()
            assert rnd.dispatch == []
            assert rnd.retry_after is not None and rnd.retry_after > 0
            assert not runner.done
            drive(sub, runner)
            assert runner.result().outputs.out == 7
        assert expansions == []
        assert not runner.errors
    finally:
        out, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert len(log.read_text().splitlines()) == 1


def hold_lease_forever(root, log, ready):
    """Take the lease of the ``Wrapped`` job and die with it held, as a killed
    coordinator does"""
    store = PickleStore(root, lease_settings=SETTINGS)
    sub = Submitter(worker="debug", store=store)
    job = Job(wrapped(log, 0.0), submitter=sub, name="main")
    lease = store.acquire(job.checksum, job.context)
    lease.__enter__()
    Path(ready).touch()
    time.sleep(600)


@pytest.mark.timeout(120)
def test_stale_workflow_lease_taken_over(tmp_path):
    log = tmp_path / "runs.log"
    ready = tmp_path / "held"
    holder = spawn("hold_lease_forever", tmp_path / "cache", log, ready)
    try:
        wait_for(ready)
        holder.kill()
        holder.wait()
        sub, runner = runner_for(tmp_path / "cache", wrapped(log, 0.0))
        with sub:
            assert runner.ready().retry_after is not None  # the lease is not yet stale
            start = time.monotonic()
            drive(sub, runner)
            assert runner.result().outputs.out == 7
        assert time.monotonic() - start < 20
        assert not runner.errors
    finally:
        holder.kill()
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.timeout(120)
def test_live_holder_not_taken_over_past_stale_after(tmp_path, monkeypatch):
    """The holder expands for three times ``stale_after`` with its heartbeat running,
    so the second submitter keeps waiting and never expands"""
    log = tmp_path / "runs.log"
    first = spawn("hold_in_process", tmp_path / "cache", log, 3.0)
    try:
        wait_for(Path(str(log) + ".started"))
        expansions = count_expansions(monkeypatch)
        sub, runner = runner_for(tmp_path / "cache", wrapped(log, 3.0))
        with sub:
            start = time.monotonic()
            retries = 0
            while not runner.done:
                rnd = runner.ready()
                assert rnd.dispatch == []
                if rnd.retry_after is not None:
                    retries += 1
                    time.sleep(rnd.retry_after)
            waited = time.monotonic() - start
            assert runner.result().outputs.out == 7
        assert retries > 1
        assert waited > SETTINGS.stale_after
        assert expansions == []
    finally:
        first.communicate(timeout=60)
    assert len(log.read_text().splitlines()) == 1


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
    with Submitter(worker="debug", cache_root=tmp_path, on_error="raise_at_end") as sub:
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
