"""Workflow jobs under a `PortableStore` (RFC 0003 section 13 item 6; RFC 0002 section 3.1,
"Under a `PortableStore`"): the runner takes the lease by ``try_take`` on what ``acquire``
returns, checks the store by ``find(job)``, withdraws no success record, and commits by
``publish`` of the workflow's outputs. An error in a node publishes nothing and writes
``_error.json`` to the workflow job's own work directory."""

import json
import os
import shutil
import time

import pytest
from fileformats.generic import File

from pydra.compose import python, workflow
from pydra.engine.hooks import TaskHooks
from pydra.engine.job import Job
from pydra.engine.lease import LeaseSettings
from pydra.engine.portable.reuse import Verdict
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter

from .test_portable_reuse import Scripted
from .test_portable_store import make
from .test_portable_submitter import Add1, Two, WriteFile, drive_with_run
from .test_runner_flatten import AfterWrapped, count_expansions, drive

# a holder that keeps renewing is waited for until ``max_wait``
FOREVER = LeaseSettings(
    stale_after=10, timeout=15, attribute_delay=0, max_wait=0.5, holder="anonymous"
)


@workflow.define
def Nest(x: int) -> int:
    two = workflow.add(Two(x=x), name="two")
    after = workflow.add(Add1(x=two.out), name="after")
    return after.out


@python.define
def Boom(x: int) -> int:
    raise ValueError(f"boom {x}")


@workflow.define
def Fails(x: int) -> int:
    boom = workflow.add(Boom(x=x), name="boom")
    return boom.out


BROKEN: list[str] = []


def break_own_lease(job):
    """Remove the lease the job holds (de-duplicating or resumable): it is taken over."""
    shutil.rmtree(job._acquire().lockfile)
    BROKEN.append(job.checksum)


@workflow.define
def NestFails(x: int) -> int:
    inner = workflow.add(Fails(x=x), name="inner")
    after = workflow.add(Add1(x=inner.out), name="after")
    return after.out


@workflow.define
def NestFailsLeaseLost(x: int) -> int:
    inner = workflow.add(
        Fails(x=x), hooks=TaskHooks(pre_run_task=break_own_lease), name="inner"
    )
    after = workflow.add(Add1(x=inner.out), name="after")
    return after.out


@workflow.define
def LeafLost(x: int) -> int:
    boom = workflow.add(
        Boom(x=x), hooks=TaskHooks(pre_run_task=break_own_lease), name="boom"
    )
    return boom.out


@workflow.define
def WriteWf(text: str) -> File:
    w = workflow.add(WriteFile(text=text), name="w")
    return w.out


class ByTask(Scripted):
    """A hook that gives ``verdict`` for a candidate of a task of type ``kind`` and SERVE for
    every other, counting the verdicts asked for ``kind``."""

    def __init__(self, kind, verdict):
        super().__init__()
        self.kind = kind
        self.answer = verdict
        self.asked = 0

    def verdict(self, candidate, current, task):
        if isinstance(task, self.kind):
            self.asked += 1
            return self.answer
        return Verdict.SERVE


def runner_for(store, sub, task):
    return GraphRunner(
        task,
        store,
        on_error="raise_at_end",
        submitter=sub,
        job=Job(task, submitter=sub, name="main"),
    )


def spy(monkeypatch, store, name):
    calls = []
    real = getattr(store, name)

    def wrapped(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(store, name, wrapped)
    return calls


def test_workflow_hit_not_expanded(tmp_path, monkeypatch):
    """A nested workflow job whose entry is stored is found by ``find(job)`` and completed as
    ``workflow-stored``: its nodes are not expanded, it gets no attempt and it keeps the
    `Committed` of the entry it was served from."""
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        assert sub(Two(x=1)).outputs.out == 3
    found = spy(monkeypatch, store, "find")
    expansions = count_expansions(monkeypatch)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, Nest(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 4
        two = runner.exec_graph["two"].job(None)
    settled = [s for r in rounds for s in r.settled if s.job is two]
    assert [s.how for s in settled] == ["workflow-stored"]
    assert expansions == ["Nest"]
    assert two.attempt is None
    assert two in [args[0] for args, _ in found]
    assert two.committed is not None and two.committed == settled[0].result.committed
    assert two.committed.served is True


def test_workflow_verify_expands(tmp_path):
    """A `Verify` for a nested workflow job marks it and expands it; its publish compares the
    fresh outputs with the candidate. Under a lease that excludes no other taker (no
    de-duplication) the hook is asked once for the job."""
    with Submitter(
        worker="debug", store=make(tmp_path), cache_root=tmp_path / "pc"
    ) as sub:
        assert sub(Two(x=1)).outputs.out == 3
    hook = ByTask(Two, Verdict.VERIFY)
    store = make(tmp_path, reuse=hook, dedupe=False)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, Nest(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 4
        two = runner.exec_graph["two"].job(None)
    assert not [s for r in rounds for s in r.settled if s.job is two]
    assert two.verify is not None and two.verify.rank == 0
    assert two.attempt is not None
    assert [c[1:3] for c in hook.calls("compare")] == [(0, {"out": 3})]
    assert hook.asked == 1
    assert two.committed is not None and two.committed.stored is True


def test_workflow_commit_publishes_no_lease_lost(tmp_path, monkeypatch):
    """A nested workflow job whose de-duplication lease was taken over while it ran still
    publishes, under its own checksum and execute context; no ``phase="commit"`` entry.
    """

    @workflow.define
    def LeaseBroken(x: int) -> int:
        inner = workflow.add(
            Two(x=x),
            hooks=TaskHooks(post_run_task=lambda job, result: break_own_lease(job)),
            name="inner",
        )
        return inner.out

    store = make(tmp_path, dedupe=True)
    published = spy(monkeypatch, store, "publish")
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, LeaseBroken(x=1))
        drive_with_run(runner)
        assert runner.result().outputs.out == 3
        inner = runner.exec_graph["inner"].job(None)
        context = store.identity.context(inner, "execute")
    assert runner.entries == []
    (call,) = [args for args, _ in published if args[0] == inner.checksum]
    assert call[2] == context and dict(call[3]) == {"out": 3}
    assert inner.committed is not None and inner.committed.stored is True
    assert inner.committed.won is True


def test_dedupe_workflow_lease_timeout_phase_lease(tmp_path):
    """Under a de-duplicating store a nested workflow job whose lease another runner keeps
    is waited for; past ``max_wait`` it is an entry with ``phase="lease"`` and its readers
    are unrunnable, and ``ready()`` does not raise."""
    store = make(tmp_path, dedupe=True, lease_settings=FOREVER)
    log = os.devnull
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        holder = GraphRunner(
            AfterWrapped(log=log, hold=0.0),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        assert [j.name for j in holder.ready().dispatch] == ["Counted"]  # holds it
        runner = GraphRunner(
            AfterWrapped(log=log, hold=0.0),
            store,
            on_error="raise_at_end",
            submitter=sub,
        )
        rnd = runner.ready()
        assert rnd.dispatch == [] and rnd.retry_after is not None
        assert runner.stuck() is None
        for _ in range(600):
            if runner.done:
                break
            time.sleep(rnd.retry_after or 0.05)
            rnd = runner.ready()
        assert runner.done
        (entry,) = runner.entries
        assert entry.phase == "lease" and entry.job_name == "wrapped"
        assert entry.exception_type.endswith("LeaseTimeout")
        assert runner.exec_graph["after"].unrunnable == {
            None: [runner.exec_graph["wrapped"]]
        }
        holder.close()


@pytest.mark.parametrize("lease", ["held", "lost"])
def test_node_error_writes_error_json_publishes_nothing(tmp_path, lease):
    """An error in a node of a nested workflow publishes nothing for the workflow job and
    writes ``_error.json`` to its work directory, whether or not its de-duplication lease
    is still held (the work directory is the attempt's own); the error entry is recorded.
    """
    task = NestFails(x=1) if lease == "held" else NestFailsLeaseLost(x=1)
    store = make(tmp_path, dedupe=True)
    BROKEN.clear()
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, task)
        drive(sub, runner)
        inner = runner.exec_graph["inner"].job(None)
        assert runner.failed
    assert BROKEN == ([] if lease == "held" else [inner.checksum])
    assert store.stored(inner.checksum) == []
    assert inner.committed is None
    assert inner.error_file.name == "_error.json" and inner.error_file.is_file()
    names = {(e.job_name, e.phase) for e in runner.entries}
    assert ("inner", "run") in names


def test_workflow_error_json_in_own_attempt_dir(tmp_path):
    """The ``_error.json`` of a failed nested workflow job is in the work directory of its
    own attempt, apart from its failed node's and the outer job's, and names the attempt.
    """
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, NestFails(x=1))
        drive(sub, runner)
        outer = runner.job
        inner = runner.exec_graph["inner"].job(None)
        boom = runner._child[inner].graph["boom"].job(None)
    fingerprint = store.identity.fingerprint(inner.context)
    own = store._work_path(inner.checksum, fingerprint, inner.attempt)
    assert inner.error_file == own / "_error.json"
    assert json.loads(inner.error_file.read_text())["attempt"] == inner.attempt
    assert len({inner.attempt, boom.attempt, outer.attempt}) == 3
    assert len({inner.cache_dir, boom.cache_dir, outer.cache_dir}) == 3


def test_workflow_file_outputs_copied_into_entry(tmp_path):
    """A file output of a workflow is copied into the workflow's own entry (RFC 0003 Q6's
    default), not referred to the entry of the node that wrote it."""
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, WriteWf(text="hello"))
        drive_with_run(runner)
        result = runner.result()
        w = runner.exec_graph["w"].job(None)
        node_path = w.result().outputs.out.fspath
    wf_path = result.outputs.out.fspath
    assert wf_path.read_text() == "hello"
    ref = runner.job.committed.entry
    assert wf_path.is_relative_to(os.path.join(ref.root, "results@1"))
    assert ref.fingerprint in wf_path.parts and ref.cdir in wf_path.parts
    assert wf_path != node_path
    assert not os.path.samefile(wf_path, node_path)


def test_leaf_error_json_after_lease_lost(tmp_path):
    """A leaf job whose de-duplication lease was taken over while it ran still writes
    ``_error.json`` to its attempt's own work directory, and publishes nothing."""
    store = make(tmp_path, dedupe=True)
    BROKEN.clear()
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, LeafLost(x=1))
        drive(sub, runner)
        boom = runner.exec_graph["boom"].job(None)
    assert BROKEN == [boom.checksum]
    assert store.stored(boom.checksum) == []
    assert boom.error_file.name == "_error.json" and boom.error_file.is_file()
    assert json.loads(boom.error_file.read_text())["attempt"] == boom.attempt


@pytest.mark.parametrize("kind", ["leaf", "workflow"])
def test_resumable_job_writes_no_error_json_after_lease_lost(
    tmp_path, monkeypatch, kind
):
    """A resumable job's work directory (``attempt="resume"``) is shared by its runs and
    guarded by its lease, so a job whose lease was taken over writes no ``_error.json``
    there: the new holder runs in that directory."""
    monkeypatch.setattr(Job, "_can_resume", True)
    store = make(tmp_path)
    task = LeafLost(x=1) if kind == "leaf" else NestFailsLeaseLost(x=1)
    name = "boom" if kind == "leaf" else "inner"
    BROKEN.clear()
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, task)
        drive(sub, runner)
        job = runner.exec_graph[name].job(None)
        assert runner.failed
    assert job.checksum in BROKEN
    assert job.cache_dir.name.startswith("resume-")
    assert not job.error_file.exists()


def test_dedupe_store_published_between_find_and_lease_not_expanded_again(
    tmp_path, monkeypatch
):
    """Under a de-duplicating store, another process that publishes a nested workflow job
    between this runner's ``find`` and its lease is found under the lease: the job is
    ``workflow-stored`` and expanded once in all (RFC 0003 section 5.3)."""
    store = make(tmp_path, dedupe=True)
    real_prepare = GraphRunner._prepare

    def prepare(self, frame):
        if frame.job.name == "two":
            with Submitter(
                worker="debug", store=store, cache_root=tmp_path / "other"
            ) as other:
                assert other(Two(x=1)).outputs.out == 3
        return real_prepare(self, frame)

    monkeypatch.setattr(GraphRunner, "_prepare", prepare)
    expansions = count_expansions(monkeypatch)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, Nest(x=1))
        rounds = drive_with_run(runner)
        assert runner.result().outputs.out == 4
        two = runner.exec_graph["two"].job(None)
    assert expansions.count("Two") == 1, expansions
    assert [s.how for r in rounds for s in r.settled if s.job is two] == [
        "workflow-stored"
    ]


def test_nested_workflow_waits_for_lease_then_reads_stored(tmp_path, monkeypatch):
    """A nested workflow job whose de-duplication lease another holder keeps is waited for;
    the holder's process publishes it and releases, and the waiter then finds it under the
    lease and settles it as ``workflow-stored`` without expanding it."""
    store = make(tmp_path, dedupe=True)
    expansions = count_expansions(monkeypatch)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        like = Job(Two(x=1), submitter=sub, name="two")  # the nested job's address
        holder = store.acquire(like.checksum, like.context)
        holder.__enter__()
        try:
            runner = runner_for(store, sub, Nest(x=1))
            rnd = runner.ready()
            assert rnd.dispatch == [] and rnd.retry_after is not None
            assert expansions == ["Nest"]
            # the holder's process publishes it (another store object on the root, which
            # does not wait for the lease held here)
            other = make(tmp_path, dedupe=False)
            with Submitter(
                worker="debug", store=other, cache_root=tmp_path / "other"
            ) as osub:
                assert osub(Two(x=1)).outputs.out == 3
        finally:
            holder.__exit__(None, None, None)
        assert expansions == ["Nest", "Two"]  # the holder's computation
        rounds = [rnd]
        deadline = time.monotonic() + 60
        while not runner.done:
            assert time.monotonic() < deadline
            time.sleep(min(rnd.retry_after or 0, 0.5))
            rnd = runner.ready()
            rounds.append(rnd)
            for job in rnd.dispatch:
                runner.complete(job, job.run())
        assert runner.result().outputs.out == 4
        two = runner.exec_graph["two"].job(None)
    assert expansions == ["Nest", "Two"]
    assert [s.how for r in rounds for s in r.settled if s.job is two] == [
        "workflow-stored"
    ]
