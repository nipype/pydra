"""How often a leaf job under a `PortableStore` looks up the store (RFC 0002 section 3.1, "Under a
`PortableStore`"; RFC 0003 section 5.3). A complete ``find(job)`` asked before dispatch is not
asked again under a lease that excludes nobody (`_AlwaysHeld`); under an exclusive lease
(de-duplicating, resumable or a rerun) the job looks again once the lease is taken, because
another process may have published it meanwhile."""

import cloudpickle as cp
import pytest

from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.portable.reuse import Verdict
from pydra.engine.portable.store import PortableStore
from pydra.engine.portable.trust import Root
from pydra.engine.runner import GraphRunner, NotRun
from pydra.engine.submitter import Submitter

from .test_portable_store import TRUST, make_v2 as make
from .test_portable_submitter import drive_with_run
from .test_portable_workflow_jobs import ByTask, spy

# the file each test's runs of `Counted` are appended to: read through the environment at run
# time, so it counts runs in this process and in a copy of the task a pickle round trip made
RUNS_ENV = "PYDRA_TEST_X6B_RUNS"


@python.define
def Counted(x: int) -> int:
    import os

    with open(os.environ["PYDRA_TEST_X6B_RUNS"], "a") as f:
        f.write(f"{x}\n")
    return x + 1


@workflow.define
def OneLeaf(x: int) -> int:
    a = workflow.add(Counted(x=x), name="a")
    return a.out


@pytest.fixture(autouse=True)
def _runs_file(tmp_path, monkeypatch):
    monkeypatch.setenv(RUNS_ENV, str(tmp_path / "runs.txt"))


def runs(tmp_path) -> list[int]:
    """The ``x`` of every run of `Counted` in this test so far"""
    path = tmp_path / "runs.txt"
    return [int(line) for line in path.read_text().split()] if path.exists() else []


def runner_for(store, sub, task, lookup_before_dispatch=False):
    return GraphRunner(
        task,
        store,
        on_error="raise_at_end",
        submitter=sub,
        job=Job(task, submitter=sub, name="main"),
        lookup_before_dispatch=lookup_before_dispatch,
    )


def populate(tmp_path, store):
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc0") as sub:
        assert sub(Counted(x=1)).outputs.out == 2
    assert runs(tmp_path) == [1]
    (tmp_path / "runs.txt").unlink()


def leaf_finds(calls):
    return [c for c in calls if isinstance(c[0][0].task, Counted)]


def run_one_leaf(tmp_path, store, lookup_before_dispatch):
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, OneLeaf(x=1), lookup_before_dispatch)
        drive_with_run(runner)
        assert runner.result().outputs.out == 2
        return runner.exec_graph["a"].job(None)


@pytest.mark.parametrize(
    "lookup_before_dispatch, stored",
    [(True, False), (False, True)],
    ids=["lookup-miss", "probe-present-refused"],
)
def test_leaf_miss_always_held_one_lookup(
    tmp_path, monkeypatch, lookup_before_dispatch, stored
):
    """A leaf whose complete lookup before dispatch misses (an empty store under
    ``lookup_before_dispatch``, or a present own entry the hook refuses) is not looked up again
    under its `_AlwaysHeld` lease: one ``find`` in all, and it runs once."""
    hook = ByTask(Counted, Verdict.REFUSE)
    store = make(tmp_path, dedupe=False, reuse=hook)
    if stored:
        populate(tmp_path, store)
    finds = spy(monkeypatch, store, "find")
    run_one_leaf(tmp_path, store, lookup_before_dispatch)
    assert len(leaf_finds(finds)) == 1
    assert runs(tmp_path) == [1]


def test_leaf_probe_absent_still_looks_under_the_lease(tmp_path, monkeypatch):
    """A probe that finds nothing at the job's own address read nothing else, so the lookup under
    the lease, which covers read-only roots, is kept: an entry in one is served, not computed.
    """
    donor = make(tmp_path, "donor")
    populate(tmp_path, donor)
    store = make(
        tmp_path,
        "main",
        dedupe=False,
        readonly=[Root(tmp_path / "donor", trust=TRUST)],
    )
    finds = spy(monkeypatch, store, "find")
    leaf = run_one_leaf(tmp_path, store, lookup_before_dispatch=False)
    assert runs(tmp_path) == []
    assert len(leaf_finds(finds)) == 1
    assert leaf.committed is not None
    assert leaf.committed.entry.root == str(tmp_path / "donor")


@pytest.mark.parametrize("lookup_before_dispatch", [True, False])
def test_leaf_hit_served_from_the_first_lookup(
    tmp_path, monkeypatch, lookup_before_dispatch
):
    """A leaf whose entry is stored is settled from the lookup before dispatch, keeping the
    `Committed` of that entry, and never run or looked up again."""
    store = make(tmp_path, dedupe=False)
    populate(tmp_path, store)
    finds = spy(monkeypatch, store, "find")
    leaf = run_one_leaf(tmp_path, store, lookup_before_dispatch)
    assert runs(tmp_path) == []
    assert len(leaf_finds(finds)) == 1
    assert leaf.committed is not None
    assert leaf.committed.entry.root == store.path


@pytest.mark.parametrize(
    "dedupe, asked", [(False, 1), (True, 2)], ids=["always-held", "dedupe"]
)
@pytest.mark.parametrize("lookup_before_dispatch", [True, False])
def test_leaf_verify_asked_once(tmp_path, lookup_before_dispatch, dedupe, asked):
    """The reuse hook is asked once for a leaf it answers VERIFY under an `_AlwaysHeld` lease;
    the leaf runs against the candidate of that one verdict. Under a de-duplicating lease it is
    asked again once the lease is taken (RFC 0003 section 5.3)."""
    hook = ByTask(Counted, Verdict.VERIFY)
    store = make(tmp_path, dedupe=dedupe, reuse=hook)
    populate(tmp_path, store)
    hook.asked = 0
    leaf = run_one_leaf(tmp_path, store, lookup_before_dispatch)
    assert hook.asked == asked
    assert runs(tmp_path) == [1]
    assert leaf.verify is not None


@pytest.mark.parametrize("lookup_before_dispatch", [True, False])
def test_leaf_published_between_lookup_and_lease_served(
    tmp_path, lookup_before_dispatch
):
    """Two processes, one leaf: A's lookup before dispatch misses, B publishes, then A takes
    its de-duplicating lease. A looks again under the lease and serves B's entry; the leaf is
    computed once."""
    store = make(tmp_path, dedupe=True)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, OneLeaf(x=1), lookup_before_dispatch)
        rnd = runner.ready()
        (job,) = rnd.dispatch
        assert runs(tmp_path) == []
        # process B: its own store over the same root
        other = make(tmp_path)
        with Submitter(worker="debug", store=other, cache_root=tmp_path / "pcB") as b:
            published = b(Counted(x=1))
        assert runs(tmp_path) == [1]
        runner.complete(job, job.run(rerun=job.rerun))
        drive_with_run(runner)
        assert runner.result().outputs.out == 2
    assert runs(tmp_path) == [1]
    assert job.committed is not None
    assert job.committed.entry == published.committed.entry


@pytest.mark.parametrize(
    "dedupe, asked", [(False, 1), (True, 2)], ids=["always-held", "dedupe"]
)
def test_outer_leaf_verify_asked_once(tmp_path, dedupe, asked):
    """The outer job of ``Submitter(...)(task)`` for a leaf task follows the same rule: the
    hook's VERIFY before the lease is not asked again under an `_AlwaysHeld` lease."""
    hook = ByTask(Counted, Verdict.VERIFY)
    store = make(tmp_path, dedupe=dedupe, reuse=hook)
    populate(tmp_path, store)
    hook.asked = 0
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        assert sub(Counted(x=1)).outputs.out == 2
    assert hook.asked == asked
    assert runs(tmp_path) == [1]


def test_leaf_pickled_to_a_worker_looks_again(tmp_path, monkeypatch):
    """The mark never crosses a process boundary: a leaf marked by ``ready()`` and pickled to a
    worker looks up the store under its `_AlwaysHeld` lease, so an entry published while it
    waited (in a queue, say) is served, not recomputed."""
    store = make(tmp_path, dedupe=False)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, OneLeaf(x=1), lookup_before_dispatch=True)
        (job,) = runner.ready().dispatch
        assert job.looked_up is True
        worker = cp.loads(cp.dumps(job))
        assert worker.looked_up is False
        other = make(tmp_path, dedupe=False)
        with Submitter(worker="debug", store=other, cache_root=tmp_path / "pcB") as b:
            published = b(Counted(x=1))
        assert published.committed.won is True
        assert runs(tmp_path) == [1]  # B computed it
        finds = spy(monkeypatch, worker.store, "find")
        result = worker.run(rerun=worker.rerun)
    assert len(finds) == 1
    assert runs(tmp_path) == [1]  # the worker did not
    assert result.committed.served is True  # not recomputed
    assert result.committed.entry == published.committed.entry


def test_leaf_retried_looks_again(tmp_path, monkeypatch):
    """A leaf reported `NotRun` and retried is not handed out on its old answer: the mark is
    cleared, and the run under its `_AlwaysHeld` lease looks the store up again."""
    store = make(tmp_path, dedupe=False)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = runner_for(store, sub, OneLeaf(x=1), lookup_before_dispatch=True)
        (job,) = runner.ready().dispatch
        assert job.looked_up is True
        runner.complete(job, NotRun(reason="preempted", held=False))
        runner.retry(job)
        assert job.looked_up is False
        other = make(tmp_path, dedupe=False)
        with Submitter(worker="debug", store=other, cache_root=tmp_path / "pcB") as b:
            published = b(Counted(x=1))
        assert runs(tmp_path) == [1]  # B computed it
        (again,) = runner.ready().dispatch
        assert again is job
        finds = spy(monkeypatch, store, "find")
        runner.complete(job, job.run(rerun=job.rerun))
        drive_with_run(runner)
        assert runner.result().outputs.out == 2
    assert len(leaf_finds(finds)) == 1
    assert runs(tmp_path) == [1]  # the retried run did not
    assert job.committed.served is True  # not recomputed
    assert job.committed.entry == published.committed.entry
