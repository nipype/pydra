"""`read_succeeded` through the reuse hook, and rerun as evict-and-replace, for a
`PortableStore` (RFC 0003 sections 5.2, 7.3 and 10, P6)."""

import errno
import os
import sys
import time

import cloudpickle
from pydra.engine.lease import LeaseSettings
import pytest

from pydra.compose import python
from pydra.engine.portable.reuse import Verdict
from pydra.engine.portable.store import StoreNotWritable
from pydra.engine.submitter import Submitter
from pydra.utils.storagedir import StorageDir

from .test_portable_quarantine import quarantined
from .test_portable_reuse import Scripted, jobfor, put
from .test_portable_store import make
from .test_portable_workdirs import job_in

cloudpickle.register_pickle_by_value(sys.modules[__name__])


@python.define
def _Tick(log: str) -> int:
    """Not deterministic: the number of times it has run."""
    with open(log, "a") as f:
        f.write("ran\n")
    with open(log) as f:
        return len(f.readlines())


class _Calls:
    """Counts the os-level metadata calls a read makes."""

    def __init__(self, monkeypatch):
        self.n = 0
        for name in ("lstat", "stat", "open", "fstat", "scandir"):
            monkeypatch.setattr(os, name, self._counting(getattr(os, name)))
        self.n = 0  # patching itself may stat

    def _counting(self, real):
        def counting(*a, **kw):
            self.n += 1
            return real(*a, **kw)

        return counting


def _store_and_job(tmp_path, **kw):
    store = make(tmp_path, **kw)
    return store, jobfor(tmp_path, store)


# -- read_succeeded --------------------------------------------------------------------------------


def test_read_succeeded_one_lstat_on_miss(tmp_path, monkeypatch):
    hook = Scripted()
    store, job = _store_and_job(tmp_path, reuse=hook)
    put(
        store, job, "other", out=7
    )  # the checksum has a directory; the job's own entry is absent
    args = (
        job.checksum,
        job.context,
        job.task,
    )  # computed lazily: not part of the read
    calls = _Calls(monkeypatch)
    got = store.read_succeeded(*args)
    n, calls.n = calls.n, 0  # read before an assertion runs, which may stat for itself
    assert (
        got is None and n == 1
    )  # one os.lstat of the joined path: no directory is opened
    assert hook.log == []  # no ranking, no verdict, no read: one existence check
    absent = jobfor(tmp_path, store, a=2)  # no directory for its checksum at all
    args = (absent.checksum, absent.context, absent.task)
    calls.n = 0
    got = store.read_succeeded(*args)
    n, calls.n = calls.n, 0
    assert got is None and n == 1
    monkeypatch.undo()
    put(store, job, None, out=5)
    got = store.read_succeeded(job.checksum, job.context, job.task)
    assert got.outputs.out == 5
    assert hook.calls("served") == [("served", None)]


def test_read_succeeded_none_on_verify_or_refuse(tmp_path):
    hook = Scripted()
    store, job = _store_and_job(tmp_path, reuse=hook)
    put(store, job, None, out=5)
    read = lambda: store.read_succeeded(
        job.checksum, job.context, job.task
    )  # noqa: E731
    for verdict in (Verdict.VERIFY, Verdict.REFUSE, Verdict.SKIP):
        hook.verdicts = {None: verdict}
        hook.log.clear()
        assert read() is None, verdict
        assert hook.calls("served") == []
    hook.verdicts = {}
    assert read().outputs.out == 5
    # an entry that does not verify is not served either
    victim = store.entry_path(job.checksum, job.context)
    victim.chmod(0o755)
    (victim / "entry.json").chmod(0o644)
    (victim / "entry.json").write_text("{")
    assert read() is None


# -- rerun -------------------------------------------------------------------------------------------


def _run(tmp_path, store, log, rerun=False):
    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        job = job_in(sub, _Tick(log=str(log)))
        return job, job.run(rerun=rerun)


def test_rerun_evicts_and_replaces(tmp_path):
    log = tmp_path / "log"
    store = make(tmp_path)
    first_job, first = _run(tmp_path, store, log)
    assert first.outputs.out == 1
    again_job, again = _run(tmp_path, store, log)  # no rerun: served
    assert again.outputs.out == 1 and log.read_text() == "ran\n"
    job, fresh = _run(tmp_path, store, log, rerun=True)
    assert fresh.outputs.out == 2 and log.read_text() == "ran\nran\n"
    assert fresh.committed.won and fresh.committed.stored and not fresh.committed.served
    assert store.read_own(job).outputs.out == 2  # the stored result is the fresh one
    assert len(quarantined(store.path)) == 1  # the replaced entry, for collect
    _, served = _run(tmp_path, store, log)
    assert served.outputs.out == 2 and log.read_text() == "ran\nran\n"


def test_rerun_loses_race_returns_fresh(tmp_path, monkeypatch):
    log = tmp_path / "log"
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    _run(tmp_path, store, log)
    real = store._evict

    def evict_then_another_commits(root, ref, expected):
        real(root, ref, expected)  # our eviction takes, then a fresh result lands
        store.publish(
            job_ref.checksum, None, dict(job_ref.context), {"out": 99}, job_ref.task,
            None,
        )  # fmt: skip

    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        job_ref = job_in(sub, _Tick(log=str(log)))
        monkeypatch.setattr(store, "_evict", evict_then_another_commits)
        fresh = job_ref.run(rerun=True)
    # the rerun never serves the occupant and never overwrites it
    assert fresh.outputs.out == 2 and log.read_text() == "ran\nran\n"
    assert fresh.stored is False and fresh.committed.won is False
    assert fresh.committed.served is False and fresh.committed.kept is not None
    assert hook.calls("served") == []
    assert (
        store.read_own(job_ref).outputs.out == 99
    )  # the concurrent fresh result stands


def test_rerun_non_owner_cannot_evict_keeps_stored_false(tmp_path, monkeypatch):
    log = tmp_path / "log"
    hook = Scripted()
    store = make(tmp_path, reuse=hook)
    _run(tmp_path, store, log)

    def refuse(root, ref, expected):
        raise StoreNotWritable(errno.EACCES, "not the owner", str(ref.cdir))

    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        job = job_in(sub, _Tick(log=str(log)))
        monkeypatch.setattr(store, "_evict", refuse)
        fresh = job.run(rerun=True)
    assert fresh.outputs.out == 2 and fresh.stored is False  # a fresh computation
    assert fresh.committed.served is False and hook.calls("served") == []
    assert store.read_own(job).outputs.out == 1  # the old entry is still the stored one
    assert quarantined(store.path) == []


# -- the own address, records of replaced entries -------------------------------------------------


def test_read_succeeded_serves_only_the_own_address(tmp_path):
    hook = Scripted()
    store, job = _store_and_job(tmp_path, reuse=hook)
    put(store, job, "other", out=7)
    put(store, job, None, out=5)
    read = lambda: store.read_succeeded(
        job.checksum, job.context, job.task
    )  # noqa: E731
    hook.verdicts = {
        None: Verdict.SKIP
    }  # the hook skips the own entry, takes the other
    hook.log.clear()
    assert read() is None
    assert hook.calls("served") == []  # nothing was recorded as served
    hook.verdicts = {}
    assert read().outputs.out == 5
    # the own entry is corrupt and the default policy would take the other context
    victim = store.entry_path(job.checksum, job.context)
    victim.chmod(0o755)
    (victim / "entry.json").chmod(0o644)
    (victim / "entry.json").write_text("{")
    hook.log.clear()
    assert read() is None
    assert hook.calls("served") == []


def test_read_succeeded_hook_exception_is_none(tmp_path):
    class Raising(Scripted):
        def verdict(self, candidate, current, task):
            raise RuntimeError("the hook is broken")

    store, job = _store_and_job(tmp_path, reuse=Raising())
    put(store, job, None, out=5)
    assert store.read_succeeded(job.checksum, job.context, job.task) is None


def test_a_record_of_a_replaced_entry_is_a_failure(tmp_path, monkeypatch):
    log = tmp_path / "log"
    monkeypatch.setattr(LeaseSettings, "visibility_timeout", property(lambda s: 0.05))
    store = make(tmp_path)
    first_job, first = _run(tmp_path, store, log)
    reader = cloudpickle.loads(cloudpickle.dumps(first_job))
    assert reader.result(settled=True).outputs.out == 1
    _, fresh = _run(tmp_path, store, log, rerun=True)
    assert fresh.outputs.out == 2
    got = reader.result(
        settled=True
    )  # a fresh Job: it reads the record of the first run
    assert got.errored and got.outputs is None
    assert "replaced" in got.failure


def test_rerun_evicts_only_the_entry_it_read(tmp_path, monkeypatch):
    log = tmp_path / "log"
    store = make(tmp_path)
    _run(tmp_path, store, log)
    real = store._evict_for_rerun

    def another_digest_first(root, ref, expected):
        store._evict(root, ref, expected)  # the entry the job read goes ...
        store.publish(  # ... and a different one is committed in its place
            job.checksum, None, dict(job.context), {"out": 99}, job.task, None
        )
        real(root, ref, expected)

    with Submitter(worker="debug", store=store, cache_root=tmp_path / "pc") as sub:
        job = job_in(sub, _Tick(log=str(log)))
        monkeypatch.setattr(store, "_evict_for_rerun", another_digest_first)
        fresh = job.run(rerun=True)
    assert store.read_own(job).outputs.out == 99  # it was not evicted
    assert fresh.stored is False and fresh.committed.kept is not None
    assert len(quarantined(store.path)) == 1  # only the one the job read


def test_an_unsettled_read_of_a_replaced_entry_never_sleeps(tmp_path, monkeypatch):
    log = tmp_path / "log"
    store = make(tmp_path)
    first_job, _ = _run(tmp_path, store, log)
    reader = cloudpickle.loads(cloudpickle.dumps(first_job))
    _run(tmp_path, store, log, rerun=True)
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    got = reader.result()  # unsettled: called from an event loop, which polls itself
    assert got.errored and "replaced" in got.failure
    assert sleeps == []
    clock = [0.0]
    sleeps.clear()

    class FakeTime:
        """The store's clock: it advances only when the store sleeps."""

        def __getattr__(self, name):
            return getattr(time, name)

        @staticmethod
        def monotonic():
            return clock[0]

        @staticmethod
        def sleep(s):
            sleeps.append(s)
            clock[0] += s

    monkeypatch.setattr("pydra.utils.retry.time", FakeTime())
    got = cloudpickle.loads(cloudpickle.dumps(first_job)).result(settled=True)
    assert sleeps  # a settled read retries before it fails
    assert got.errored and "replaced" in got.failure
