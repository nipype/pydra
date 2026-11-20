"""`read_succeeded` through the reuse hook, and rerun as evict-and-replace, for a
`PortableStore` (RFC 0003 sections 5.2, 7.3 and 10, P6)."""

import errno
import sys

import cloudpickle
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


class _Lstats:
    def __init__(self, monkeypatch):
        self.n = 0
        real = StorageDir.lstat

        def counting(handle, name):
            self.n += 1
            return real(handle, name)

        monkeypatch.setattr(StorageDir, "lstat", counting)


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
    lstats = _Lstats(monkeypatch)
    assert store.read_succeeded(job.checksum, job.context, job.task) is None
    assert lstats.n == 1
    assert hook.log == []  # no ranking, no verdict, no read: one existence check
    put(store, job, None, out=5)
    lstats.n = 0
    got = store.read_succeeded(job.checksum, job.context, job.task)
    assert got.outputs.out == 5 and lstats.n == 1
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
