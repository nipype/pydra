"""`Submitter(store="portable")` and the runner seam over a `PortableStore` (RFC 0003 section 13,
P8; RFC 0002 section 3.1): the default store, `find(job)` in `ready()`, the `Committed` kept by
`complete()`, `stored=False` results and their readers, and the reads of a job that ran in another
process (through `_committed.json`, then the job's own address)."""

import asyncio
import functools
import os
import stat
import sys
import threading
import typing as ty
from pathlib import Path

import cloudpickle
import pytest
from fileformats.generic import File

from pydra.compose import python, workflow
from pydra.engine.executor import WorkerExecutor
from pydra.engine.job import Job
from pydra.engine.portable.reuse import Verdict
from pydra.engine.portable.store import PortableStore
from pydra.engine.runner import GraphRunner
from pydra.engine.submitter import Submitter
from pydra.utils.storagedir import IS_WINDOWS, StoreUnavailable
from pydra.workers import base

from .test_portable_quarantine import quarantined
from .test_portable_reuse import Scripted
from .test_portable_store import TRUST, make, posix_only

cloudpickle.register_pickle_by_value(sys.modules[__name__])


class Obj:
    """An arbitrary object: no codec encodes it, so only a pickle member can store it."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Obj) and other.n == self.n

    def __hash__(self) -> int:
        return hash(("Obj", self.n))


@python.define
def MakeObj(n: int) -> ty.Any:
    return Obj(n)


@python.define
def ReadObj(o: ty.Any) -> int:
    return o.n + 1


@workflow.define
def ObjChain(n: int) -> int:
    a = workflow.add(MakeObj(n=n), name="A")
    b = workflow.add(ReadObj(o=a.out), name="B")
    return b.out


@python.define
def Add1(x: int) -> int:
    return x + 1


@workflow.define
def Two(x: int) -> int:
    a = workflow.add(Add1(x=x), name="A")
    b = workflow.add(Add1(x=a.out), name="B")
    return b.out


@python.define
def Tick(log: str) -> int:
    """Not deterministic: the number of times it has run."""
    with open(log, "a") as f:
        f.write("ran\n")
    with open(log) as f:
        return len(f.readlines())


def work_under(base, checksum, fingerprint, attempt):
    """A caller's work directory function (sent to workers by value, with this module)."""
    return Path(base) / "work" / attempt


def nonpickling(tmp_path, **kw) -> PortableStore:
    """A root no writer pickles to: ``shared`` (POSIX), or on Windows, where a shared root is
    unsupported, ``trusted`` over a root created with ``pickle=False``."""
    work = functools.partial(work_under, tmp_path)
    if IS_WINDOWS:
        kw.update(trust="trusted", pickle=False)
    else:
        kw.update(trust="shared")
    return PortableStore(tmp_path / "nopickle", work_dir=work, **kw)


class RunsElsewhere(base.Worker):
    """Runs a copy of the job, as a process that received it pickled would, and returns
    ``True``: its result exists only in the store (the Slurm and SGE adapters)."""

    _plugin_name = "runs-elsewhere"

    def run(self, job, rerun=False):
        copy = cloudpickle.loads(cloudpickle.dumps(job))
        copy.run(rerun=rerun)
        return True


def late_committed(monkeypatch, visible: "threading.Event | None" = None):
    """Make ``_committed.json`` invisible to readers until ``visible`` is set (once, after the
    first read, without an event), as a rename by another client can be late to show on shared
    storage; returns the reads seen."""
    real = PortableStore._load_committed
    reads = []

    def load(handle, wd, attempt):
        reads.append(threading.current_thread())
        shown = visible.is_set() if visible is not None else len(reads) > 1
        return real(handle, wd, attempt) if shown else None

    monkeypatch.setattr(PortableStore, "_load_committed", staticmethod(load))
    return reads


def drive_with_run(runner, rounds=100):
    """Run every handed-out job in this process; returns every round."""
    seen = []
    for _ in range(rounds):
        if runner.done:
            return seen
        rnd = runner.ready()
        seen.append(rnd)
        for job in rnd.dispatch:
            runner.complete(job, job.run(rerun=job.rerun))
    raise AssertionError("the run did not finish")


# -- the default store ---------------------------------------------------------------------------------


def test_default_portable_root_round_trips_any_object(tmp_path, monkeypatch):
    """The default `Submitter` store is a private `PortableStore` that may
    pickle, so a task returning an arbitrary object keeps working and is served again; on a
    root that is not pickled to, the outputs are kept for the run with ``stored=False`` and
    their readers run."""
    monkeypatch.setattr(
        "pydra.engine.submitter.default_store_root", lambda: tmp_path / "default"
    )
    with Submitter(worker="debug") as sub:
        store = sub.store
        assert isinstance(store, PortableStore)
        assert store.trust == "private" and store.pickle is True
        assert store.path == str(tmp_path / "default")
        result = sub(MakeObj(n=3))
    assert result.outputs.out == Obj(3) and result.stored is True
    with Submitter(worker="debug") as sub:
        again = sub(MakeObj(n=3))
    assert again.outputs.out == Obj(3) and again.committed.served is True

    store = nonpickling(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        kept = sub(MakeObj(n=3))
        chained = sub(ObjChain(n=3))
    assert kept.outputs.out == Obj(3) and kept.stored is False
    assert not chained.errored and chained.outputs.out == 4


def test_workflow_result_is_read_from_its_entry(tmp_path):
    """The result of a workflow job is the one its publish reads back: its directory is the
    entry, not the work directory, which is emptied once the entry is published."""
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        result = sub(Two(x=1))
    assert result.outputs.out == 3 and result.cache_dir.is_dir()
    assert result.cache_dir.is_relative_to(tmp_path / "cache" / "results@1")


@posix_only
def test_submitter_roots_are_private_under_umask_002(tmp_path, monkeypatch):
    """The default root and a cache_root the submitter makes are 0o700 whatever the
    umask, so the private store opens and runs under ``umask 002``."""
    default = tmp_path / "home" / "store"
    monkeypatch.setattr("pydra.engine.submitter.default_store_root", lambda: default)
    old = os.umask(0o002)
    try:
        with Submitter(worker="debug") as sub:
            first = sub(MakeObj(n=1))
        with Submitter(worker="debug", cache_root=tmp_path / "new" / "cache") as sub:
            second = sub(Two(x=1))
    finally:
        os.umask(old)
    assert first.outputs.out == Obj(1) and first.stored
    assert second.outputs.out == 3
    for root in (default, tmp_path / "new" / "cache"):
        assert stat.S_IMODE(root.stat().st_mode) == 0o700


@posix_only
def test_a_group_writable_cache_root_names_the_way_on(tmp_path):
    root = tmp_path / "shared-by-group"
    root.mkdir()
    root.chmod(0o775)
    with pytest.raises(StoreUnavailable) as info:
        Submitter(worker="debug", cache_root=root)
    message = str(info.value)
    assert f"chmod 700 {root.resolve()}" in message and 'trust="trusted"' in message


def test_top_level_calls_leave_no_work_directories(tmp_path):
    """Every call forgets its outcome and the submitter removes the emptied work
    directories of its attempts when it closes."""
    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        for i in range(3):
            assert sub(Add1(x=i)).outputs.out == i + 1
        assert sub(Add1(x=0)).committed.served
        assert sub(Two(x=5)).outputs.out == 7
        assert sub(Add1(x=0).split(x=[7, 8])).outputs.out == [8, 9]
    assert not list(root.rglob("_committed.json"))
    assert not (root / "work").exists() or list((root / "work").iterdir()) == []


def test_top_level_hit_is_served_before_its_lease(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        sub(Add1(x=1))
    with Submitter(worker="debug", cache_root=root) as sub:
        taken = []
        acquire = sub.store.acquire
        monkeypatch.setattr(
            sub.store, "acquire", lambda *a, **k: taken.append(a) or acquire(*a, **k)
        )
        result = sub(Add1(x=1))
    assert result.outputs.out == 2 and result.committed.served and taken == []


def test_submitter_collect_and_the_store_collect_command(tmp_path, cli_runner):
    from pydra.engine.portable.store import CollectReport
    from pydra.scripts.cli import cli

    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        sub(Add1(x=1))
        assert isinstance(sub.collect(), CollectReport)
    out = cli_runner(cli, ["store", "collect", str(root)])
    assert out.exit_code == 0, out.output
    assert "removed" in out.output and "kept" in out.output
    with Submitter(worker="debug", cache_root=tmp_path / "p", store="pickle") as sub:
        with pytest.raises(TypeError, match="collect"):
            sub.collect()


def test_submitter_store_names(tmp_path):
    from pydra.engine.store import PickleStore

    with Submitter(worker="debug", cache_root=tmp_path / "p", store="pickle") as sub:
        assert isinstance(sub.store, PickleStore)
    with Submitter(worker="debug", cache_root=tmp_path / "q", store="portable") as sub:
        assert isinstance(sub.store, PortableStore)
        assert sub.store.path == str(tmp_path / "q")
    with pytest.raises(ValueError, match="store"):
        Submitter(worker="debug", cache_root=tmp_path / "r", store="other")


@python.define
def WriteFile(text: str) -> File:
    path = Path("out.txt").absolute()
    path.write_text(text)
    return File(path)


class Unlisted(File):
    """A file set class outside the fileformats packages, with no registered tag."""


@python.define
def WriteUnlisted(text: str) -> Unlisted:
    path = Path("out.txt").absolute()
    path.write_text(text)
    return Unlisted(path)


def test_file_output_stored_under_its_fileformats_name(tmp_path):
    """A fileformats file set is stored under its MIME-like name and served from the entry."""
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        first = sub(WriteFile(text="hello"))
    assert first.stored is True
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        again = sub(WriteFile(text="hello"))
    assert again.committed.served is True
    path = again.outputs.out.fspath
    assert path.read_text() == "hello"
    assert path.is_relative_to(tmp_path / "cache" / "results@1")


def test_file_set_never_pickled(tmp_path):
    """A file set with no tag is not stored, even in a root that may pickle (RFC 0003 6.4)."""
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        result = sub(WriteUnlisted(text="hello"))
    assert result.stored is False and result.outputs.out.fspath.read_text() == "hello"
    assert not list((tmp_path / "cache").rglob("*.pkl"))


def test_readonly_caches_are_private_on_windows(tmp_path, monkeypatch):
    """A shared root is unsupported on Windows, so a read-only cache given as a path is read
    there as private (accepted only under %LOCALAPPDATA%); elsewhere it is shared."""
    import pydra.engine.submitter as submitter_module
    from pydra.engine.portable.trust import Root

    ro = tmp_path / "ro"
    for windows, trust in ((False, "shared"), (True, "private")):
        monkeypatch.setattr(submitter_module, "IS_WINDOWS", windows)
        store = submitter_module._portable_store(tmp_path / "c", [ro], "consistent")
        assert store.readonly == (Root(ro.resolve(), trust=trust),)


@python.define
def WritePath(text: str) -> Path:
    path = Path("out.txt").absolute()
    path.write_text(text)
    return path


@python.define
def WriteDir(text: str) -> Path:
    path = Path("outdir").absolute()
    path.mkdir()
    (path / "a.txt").write_text(text)
    return path


@python.define
def ReadPath(p: Path) -> str:
    return Path(p).read_text()


@python.define
def Elsewhere(path: str) -> Path:
    return Path(path)


@workflow.define
def PathChain(text: str) -> str:
    a = workflow.add(WritePath(text=text), name="A")
    b = workflow.add(ReadPath(p=a.out), name="B")
    return b.out


def test_path_output_inside_the_work_directory_is_stored(tmp_path):
    """A ``Path`` output naming a file or directory the job wrote in its work directory is
    stored as a file set and read back as the entry's copy, so it outlives the emptied work
    directory, at the top level, downstream and when served again."""
    cache = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=cache) as sub:
        top = sub(WritePath(text="hello"))
        tree = sub(WriteDir(text="in a dir"))
        chained = sub(PathChain(text="downstream"))
    assert top.stored and isinstance(top.outputs.out, Path)
    assert top.outputs.out.read_text() == "hello"
    assert top.outputs.out.is_relative_to(cache / "results@1")
    assert (tree.outputs.out / "a.txt").read_text() == "in a dir"
    assert chained.outputs.out == "downstream"
    with Submitter(worker="debug", cache_root=cache) as sub:
        again = sub(WritePath(text="hello"))
    assert again.committed.served and again.outputs.out.read_text() == "hello"


def test_path_output_outside_the_work_directory_is_a_value(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("kept where it is")
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        result = sub(Elsewhere(path=str(outside)))
    assert result.stored and result.outputs.out == outside


@python.define
def Pair(a: int):
    return a, a + 1


@python.define
def Nested(a: int) -> dict:
    return {"k": (a, a), "l": [(a,)]}


def test_untyped_tuples_round_trip(tmp_path):
    """A tuple that no declared type fixes comes back a tuple (``$tuple``)."""
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        pair = sub(Pair(a=1))
        nested = sub(Nested(a=2))
    assert pair.outputs.out == (1, 2) and type(pair.outputs.out) is tuple
    assert nested.outputs.out == {"k": (2, 2), "l": [(2,)]}
    assert type(nested.outputs.out["k"]) is tuple


# -- find(job) in ready() ------------------------------------------------------------------------------


def _populate(tmp_path, store):
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        assert sub(Two(x=1)).outputs.out == 3


def _spy_find(monkeypatch, store):
    calls = []
    real = store.find

    def find(job, **kw):
        calls.append((job.name, kw))
        return real(job, **kw)

    monkeypatch.setattr(store, "find", find)
    return calls


def test_ready_find_hit_settles(tmp_path, monkeypatch):
    store = make(tmp_path)
    _populate(tmp_path, store)
    calls = _spy_find(monkeypatch, store)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        rnd = runner.ready()  # B is released once A is settled, and settled too
        assert rnd.dispatch == []
        assert [(s.job.name, s.how) for s in rnd.settled] == [
            ("A", "stored"),
            ("B", "stored"),
        ]
        assert [s.result.outputs.out for s in rnd.settled] == [2, 3]
        for settled in rnd.settled:
            assert settled.job.committed == settled.result.committed
            assert settled.job.committed.served is True
            assert settled.job.attempt is None
        assert [name for name, _ in calls] == ["A", "B"]


def test_ready_find_miss_costs_no_find(tmp_path, monkeypatch):
    """Without ``lookup_before_dispatch`` a job whose own address is empty is dispatched on one
    existence check; under it, every job is looked up with `find`."""
    store = make(tmp_path)
    calls = _spy_find(monkeypatch, store)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        (a,) = runner.ready().dispatch
        assert calls == [] or all(kw.get("probe") for _, kw in calls)
        assert a.verify is None
        runner = GraphRunner(
            Two(x=1),
            store,
            on_error="raise_at_end",
            submitter=sub,
            lookup_before_dispatch=True,
        )
        calls.clear()
        (a,) = runner.ready().dispatch
        assert calls == [("A", {})]


def test_a_hit_is_read_off_the_event_loop(tmp_path, monkeypatch):
    """Under the asynchronous driver a hit's read and verification run on a helper
    thread: a read that waits for the loop to run a coroutine three times finishes, which
    it could not if it held the loop (it gives up, bounded, and the test fails)."""
    from pydra.engine.drivers import drive
    from pydra.workers.debug import DebugWorker

    store = make(tmp_path)
    _populate(tmp_path, store)
    released = threading.Event()
    gave_up, threads = [], []
    real = store.find

    def slow_find(job, **kw):
        threads.append(threading.current_thread())
        if not released.wait(10):
            gave_up.append(job.name)
        return real(job, **kw)

    monkeypatch.setattr(store, "find", slow_find)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)

        async def main():
            ticks = 0

            async def ticker():
                nonlocal ticks
                while not released.is_set():
                    ticks += 1
                    if ticks >= 3:
                        released.set()
                    await asyncio.sleep(0)

            task = asyncio.ensure_future(ticker())
            await drive(
                runner,
                {"default": WorkerExecutor(DebugWorker())},
                lambda job: "default",
                on_error="raise_at_end",
            )
            await task
            return threading.current_thread()

        loop_thread = asyncio.run(main())
    assert gave_up == [] and loop_thread not in threads
    assert runner.done and not runner.failed
    assert runner.exec_graph["B"].job(None).result().outputs.out == 3


@workflow.define
def Fan(xs: list[int]) -> list[int]:
    a = workflow.add(Add1().split(x=xs), name="A")
    return a.out


class CountingPortable(PortableStore):
    """Counts its existence checks and finds; ``present_always`` sends every job to a read
    off the thread"""

    present_always = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.presents = self.finds_read = 0

    def present(self, job):
        self.presents += 1
        return True if self.present_always else super().present(job)

    def find(self, job, **kw):
        self.finds_read += 1
        return super().find(job, **kw)


@pytest.mark.parametrize("present_always", [False, True])
@pytest.mark.parametrize("n", [20, 80])
def test_a_job_held_back_by_the_limit_is_not_read_again_off_the_thread(
    tmp_path, n, present_always
):
    """A job read off the thread with nothing to serve (its address empty, or a Miss) is
    not read again while ``ready(limit)`` holds it back"""
    store = CountingPortable(tmp_path / "root", trust=TRUST)
    store.present_always = present_always
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            Fan(xs=list(range(n))), store, on_error="raise_at_end", submitter=sub
        )
        runner.read_off_thread()
        while not runner.done:
            for job in runner.ready(1).dispatch:
                runner.complete(job, job.run())
            for future in list(runner._reading.values()):
                future.result(timeout=30)
        runner.close()
    assert not runner.failed
    assert store.presents <= 2 * n and store.finds_read <= 2 * n


class RaisesOnTheReadThread(Scripted):
    """A hook that fails while the runner reads off its thread, and only there."""

    def verdict(self, candidate, current, task):
        if threading.current_thread().name.startswith("pydra-store-read"):
            raise RuntimeError("the hook failed")
        return super().verdict(candidate, current, task)


@pytest.mark.parametrize("hook", ["verify", "raises"])
def test_a_job_read_without_a_hit_is_handed_out_under_a_limit(tmp_path, hook):
    """Jobs whose off-thread read serves nothing (a Verify, a hook that raises) are
    handed out even when a driver's ``ready(limit)`` holds some of them back for later
    rounds."""
    from pydra.engine.executor import InlineExecutor
    from pydra.workers.debug import DebugWorker

    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        assert sub(Fan(xs=[1, 2, 3])).outputs.out == [2, 3, 4]
    reuse = Scripted({None: Verdict.VERIFY}) if hook == "verify" else None
    reuse = reuse or RaisesOnTheReadThread()
    store = PortableStore(tmp_path / "cache", trust="private", pickle=True, reuse=reuse)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "cache") as sub:
        runner = GraphRunner(
            Fan(xs=[1, 2, 3]), store, on_error="raise_at_end", submitter=sub
        )
        read = threading.Event()
        runner.read_off_thread(read.set)
        executor = InlineExecutor(DebugWorker())
        executor.bind(runner.complete)
        try:
            while not runner.done:
                rnd = runner.ready(1)
                for job in rnd.dispatch:
                    executor.submit([job])
                if not rnd.dispatch and not rnd.settled:
                    assert runner.reading, runner.stuck()
                    assert read.wait(60)  # a bound only
                    read.clear()
        finally:
            runner.close()
    assert not runner.failed
    node = runner.exec_graph["A"]
    assert [node.job(i).result().outputs.out for i in range(3)] == [2, 3, 4]


@pytest.mark.parametrize("task", ["leaf", "workflow"])
def test_a_failed_run_never_returns_an_earlier_stored_result(
    tmp_path, monkeypatch, task
):
    """With ``raise_errors=False`` a call whose run failed returns an errored result, not
    the result an earlier run stored at the same address."""
    made = Add1(x=1) if task == "leaf" else Two(x=1)
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        assert not sub(made).errored

    def broken(*args, **kwargs):
        raise RuntimeError("the driver broke")

    monkeypatch.setattr(Submitter, "submit", broken)
    monkeypatch.setattr(Submitter, "_run_workflow", broken)
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        # the run left no result of its own: the call raises, as it always has
        with pytest.raises(RuntimeError, match="the driver broke"):
            sub(made, raise_errors=False, rerun=True)


def test_a_failed_call_not_raising_returns_its_errored_result(tmp_path):
    """A run whose task fails, with raise_errors=False, returns this run's errored result."""
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        result = sub(Boom(x=1), raise_errors=False)
    assert result.errored


@python.define
def Boom(x: int) -> int:
    raise ValueError("boom")


@python.define
def PathAsText() -> str:
    path = Path("note.txt").absolute()
    path.write_text("x")
    return str(path)


def test_a_str_path_into_the_work_directory_warns(tmp_path):
    """A ``str`` output is a value; one naming a file the job wrote in its work directory
    warns at publish, naming the field and suggesting ``Path`` or ``File``."""
    with pytest.warns(UserWarning, match="out.*Path"):
        with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
            result = sub(PathAsText())
    assert result.stored and isinstance(result.outputs.out, str)


@pytest.mark.parametrize("lookup_before_dispatch", [False, True])
def test_ready_find_verify_dispatches_with_mark(
    tmp_path, monkeypatch, lookup_before_dispatch
):
    _populate(tmp_path, make(tmp_path))
    hook = Scripted({None: Verdict.VERIFY})
    store = make(tmp_path, reuse=hook)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            Two(x=1),
            store,
            on_error="raise_at_end",
            submitter=sub,
            lookup_before_dispatch=lookup_before_dispatch,
        )
        rnd = runner.ready()
        assert rnd.settled == []
        (a,) = rnd.dispatch
        assert a.verify is not None and a.verify.rank == 0
        assert a.verify.entry.fingerprint == store.identity.fingerprint(a.context)
        assert a.attempt is not None


def test_external_driver_forwards_attempt_and_verify(tmp_path):
    """RFC 0002 section 3.5: the job an external driver sends to its own process carries its
    ``attempt`` and ``verify`` across a pickle round trip, and runs against the candidate.
    """
    _populate(tmp_path, make(tmp_path))
    hook = Scripted({None: Verdict.VERIFY})
    store = make(tmp_path, reuse=hook)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        for _ in range(10):
            if runner.done:
                break
            for job in runner.ready().dispatch:
                sent = cloudpickle.loads(cloudpickle.dumps(job))
                assert sent.attempt == job.attempt and sent.verify == job.verify
                assert sent.verify is not None
                result = sent.run()
                assert [c[1] for c in sent.store.reuse.calls("compare")] == [0]
                runner.complete(job, result)
        assert runner.done and not runner.failed
        assert runner.exec_graph["B"].job(None).result().outputs.out == 3


def test_plan_counts_own_entries_without_reading_them(tmp_path, monkeypatch):
    """`plan()` over a portable store counts the jobs whose own address holds an entry, one
    existence check each, and asks the reuse hook nothing (a `served` call is a reuse).
    """
    _populate(tmp_path, make(tmp_path))
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    calls = _spy_find(monkeypatch, store)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        for lookup_before_dispatch in (False, True):
            runner = GraphRunner(
                Two(x=1),
                store,
                on_error="raise_at_end",
                submitter=sub,
                lookup_before_dispatch=lookup_before_dispatch,
            )
            plan = runner.plan()
            (a,) = [n for n in plan.nodes if n.path == "A"]
            assert a.expected_hits == 1
    assert hook.log == [] and calls == []


def test_cross_context_hit_read_back_by_its_committed(tmp_path):
    """A job served from another context's entry in one process is read back as a
    cross-context result by a process that holds only its ``Committed``."""
    _populate(tmp_path, make(tmp_path))
    store = make(tmp_path, reuse=Scripted())  # serves any context
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        runner.ready()
        a = runner.exec_graph["A"].job(None)
        a.attempt = "0" * 32
        a._context = dict(a.context, licence="other")  # another context
        ran = cloudpickle.loads(cloudpickle.dumps(a))
        served = ran.run()
        assert served.cross_context is True and served.committed.served
        reader = cloudpickle.loads(cloudpickle.dumps(a))
        reader._held = None
        read = reader.result(settled=True)
    assert read.outputs.out == 2 and read.cross_context is True
    assert read.reused_in == dict(a.context)


# -- complete() and stored=False ----------------------------------------------------------------------


def test_complete_keeps_committed(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        (a,) = runner.ready().dispatch
        ran = cloudpickle.loads(cloudpickle.dumps(a))  # another process
        result = ran.run()
        committed = result.committed
        assert committed is not None and committed.stored and committed.won
        assert a.committed is None
        runner.complete(a, result)
        assert a.committed == committed
        assert a.result() is result and a.result(settled=True).committed == committed
        assert not (store.work_dir(a) / "_committed.json").exists()  # forgotten
        reader = cloudpickle.loads(cloudpickle.dumps(a))  # read by its EntryRef
        assert reader.result(settled=True).outputs.out == 2


def test_a_directory_is_never_removed_between_a_jobs_cwd_read_and_its_hold(
    tmp_path, monkeypatch
):
    """Job B reads the working directory, which is job X's work directory because X's
    thread is in it; X then changes back and its outcome is forgotten. The forget must not
    remove the directory B has read and will change back to: the read and the hold are one
    step, and so are the check and the removal. Forced interleaving: the forget runs on
    another thread at the moment B reads the directory."""
    store = make(tmp_path)
    here = os.getcwd()
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        (x,) = runner.ready().dispatch
        x.run()
        wd = store.work_dir(x)
        b = Job(task=Add1(x=10), submitter=sub, name="b")
        real_getcwd = os.getcwd
        armed, forgets = [True], []

        def forget_x():
            os.chdir(here)  # X's thread changes back, then its outcome is forgotten
            store.forget_committed(x)

        def getcwd():
            cwd = real_getcwd()
            caller = sys._getframe(1).f_code.co_name  # no inspect: it reads the cwd too
            if armed and caller in ("run", "capture_cwd") and cwd == str(wd):
                armed.clear()
                t = threading.Thread(target=forget_x)
                forgets.append(t)
                t.start()
                t.join(2)  # done at once unless it waits for the hold to be taken
            return cwd

        os.chdir(wd)  # X's thread is in its work directory
        monkeypatch.setattr(os, "getcwd", getcwd)
        try:
            result = b.run()
        finally:
            monkeypatch.setattr(os, "getcwd", real_getcwd)
            os.chdir(here)
            for t in forgets:
                t.join(30)
    assert forgets and result.outputs.out == 11 and not result.errored


def test_a_run_that_fails_before_its_task_releases_its_hold(tmp_path, monkeypatch):
    """An exception after the working directory is captured, before the task runs,
    releases the hold and restores the directory."""
    from pydra.utils import general

    def broken(self):
        raise RuntimeError("cannot make the work directory")

    monkeypatch.setattr(Job, "_populate_filesystem", broken)
    here = os.getcwd()
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        job = Job(task=Add1(x=1), submitter=sub, name="j")
        with pytest.raises(RuntimeError, match="cannot make"):
            job.run()
    assert dict(general._held_dirs) == {} and os.getcwd() == here


def test_stored_false_is_success_not_unrunnable(tmp_path):
    store = nonpickling(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            ObjChain(n=3), store, on_error="raise_at_end", submitter=sub
        )
        rounds = drive_with_run(runner)
        a = runner.exec_graph["A"].job(None)
        b = runner.exec_graph["B"].job(None)
    assert a.result().stored is False and a.result().outputs.out == Obj(3)
    assert b in [j for rnd in rounds for j in rnd.dispatch]
    assert b.result().outputs.out == 4
    assert [u for rnd in rounds for u in rnd.unrunnable] == []
    assert runner.entries == [] and not runner.failed


def test_rerun_under_portable_submitter_replaces(tmp_path, monkeypatch):
    """Under the default store a rerun computes afresh and replaces the stored entry,
    which goes to ``quarantine/``."""
    monkeypatch.setattr(
        "pydra.engine.submitter.default_store_root", lambda: tmp_path / "default"
    )
    log = str(tmp_path / "log")
    with Submitter(worker="debug") as sub:
        assert sub(Tick(log=log)).outputs.out == 1
        assert sub(Tick(log=log)).outputs.out == 1
        rerun = sub(Tick(log=log), rerun=True)
        assert rerun.outputs.out == 2 and rerun.stored is True
        assert sub(Tick(log=log)).outputs.out == 2
        assert len(quarantined(sub.store.path)) == 1


# -- reads of a job that ran elsewhere ---------------------------------------------------------------


def test_true_returning_worker_reads_committed_json(tmp_path, monkeypatch):
    """A worker that returns ``True`` is read back through ``_committed.json`` (the result was
    not stored, so no lookup could find it), and the visibility retry for it does not hold the
    event loop: the record shows only once a callback on the loop has run."""
    store = nonpickling(tmp_path)
    store.visibility_timeout = 5.0
    visible = threading.Event()
    worker = RunsElsewhere()
    with Submitter(worker=worker, store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(
            ObjChain(n=3), store, on_error="raise_at_end", submitter=sub
        )
        (a,) = runner.ready().dispatch
        executor = WorkerExecutor(worker, visibility_timeout=5.0)
        outcomes = []

        async def run():
            loop = asyncio.get_running_loop()
            real = PortableStore._load_committed

            def load(handle, wd, attempt):
                loop.call_soon_threadsafe(
                    visible.set
                )  # runs only when the loop is free
                return real(handle, wd, attempt) if visible.is_set() else None

            monkeypatch.setattr(PortableStore, "_load_committed", staticmethod(load))
            done = asyncio.Event()
            executor.bind(lambda job, outcome: (outcomes.append(outcome), done.set()))
            executor.submit([a])
            await asyncio.wait_for(done.wait(), 30)

        asyncio.run(run())
    (outcome,) = outcomes
    assert not outcome.errored, outcome.failure
    assert outcome.stored is False and outcome.outputs.out == Obj(3)


def test_submitter_final_read_stored_false_outer_job(tmp_path, monkeypatch):
    """The final read of an outer job that ran elsewhere and was not stored goes through its
    ``_committed.json``, waiting for it to show."""
    store = nonpickling(tmp_path)
    reads = late_committed(monkeypatch)
    with Submitter(
        worker=RunsElsewhere(), store=store, cache_root=tmp_path / "pc"
    ) as sub:
        result = sub(MakeObj(n=3))
    assert len(reads) >= 2
    assert not result.errored and result.stored is False
    assert result.outputs.out == Obj(3)


def test_job_result_fallback_reads_own_address(tmp_path):
    """A job with no ``Committed`` and no ``_committed.json`` (forgotten, or a reader that has
    no attempt) reads the entry at its own address."""
    store = make(tmp_path)
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        runner = GraphRunner(Two(x=1), store, on_error="raise_at_end", submitter=sub)
        (a,) = runner.ready().dispatch
        runner.complete(a, a.run())
        assert not (store.work_dir(a) / "_committed.json").exists()
        reader = cloudpickle.loads(cloudpickle.dumps(a))
        reader.committed = None
        reader.attempt = None
        reader._held = None
        found = reader.result(settled=True)
    assert not found.errored and found.outputs.out == 2 and found.stored is True


# -- real workers ----------------------------------------------------------------------------------------


def test_portable_submitter_stored_false(any_worker, tmp_path):
    """A result that is not stored is kept for the run, read across the worker's process
    boundary, and its readers run."""
    store = nonpickling(tmp_path)
    with Submitter(worker=any_worker, store=store, cache_root=tmp_path / "pc") as sub:
        chained = sub(ObjChain(n=3))
        kept = sub(MakeObj(n=5))
    assert not chained.errored and chained.outputs.out == 4
    assert not kept.errored and kept.stored is False and kept.outputs.out == Obj(5)
