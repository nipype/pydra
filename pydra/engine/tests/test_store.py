import asyncio
import errno
import os
import sys
from pathlib import Path
import cloudpickle
import pytest
from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine import store as store_module
from pydra.engine.store import (
    CacheBusyError,
    CacheWriterError,
    PickleStore,
    ResultStore,
    context_key,
    remove_from_cache,
)
from pydra.engine.submitter import Submitter
from pydra.engine.tests import utils as test_utils
from pydra.engine.tests.utils import FunAddTwo

# The installed package does not ship tests, so worker processes cannot import
# these modules and must receive their classes by value
cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(test_utils)


CTX = {"environment_checksum": "e", "platform": "p", "python_version": "v"}
KEY = context_key(CTX)


class RecordingStore(PickleStore):
    """A PickleStore that appends each call to a log file, so calls made in worker
    processes are visible to the test"""

    def __init__(self, cache_root, log):
        super().__init__(cache_root)
        self.log = Path(log)

    def _record(self, *entry):
        with open(self.log, "a") as f:
            f.write(" ".join(entry) + "\n")

    def lookup(self, checksum, context, task):
        result = super().lookup(checksum, context, task)
        self._record("lookup", checksum, "miss" if result is None else "hit")
        return result

    def acquire(self, checksum, context):
        self._record("acquire", checksum)
        return super().acquire(checksum, context)

    def commit(self, lease, result, job=None):
        self._record("commit", lease.checksum, "job" if job is not None else "nojob")
        super().commit(lease, result, job)


@workflow.define
def AddTwoWf(a: int) -> int:
    node = workflow.add(FunAddTwo(a=a))
    return node.out


def test_picklestore_is_resultstore(tmp_path):
    assert isinstance(PickleStore(tmp_path), ResultStore)


def test_picklestore_layout(tmp_path):
    store = PickleStore(tmp_path)
    assert store.cache_dir("abc", CTX) == tmp_path / "abc" / KEY
    lease = store.acquire("abc", CTX)
    assert lease.checksum == "abc"
    assert lease.cache_dir == tmp_path / "abc" / KEY
    assert lease.lockfile == tmp_path / f"abc-{KEY}.lock"
    assert not lease.lockfile.exists()


def test_picklestore_commit_lookup(tmp_path):
    store = PickleStore(tmp_path)
    assert store.lookup("abc", CTX, None) is None
    with store.acquire("abc", CTX) as lease:
        assert lease.lockfile.exists()
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not lease.lockfile.exists()
    assert (tmp_path / "abc" / KEY / "_result.pklz").exists()
    result = store.lookup("abc", CTX, None)
    assert result.cache_dir == tmp_path / "abc" / KEY
    assert not result.errored


def test_picklestore_readonly_caches(tmp_path):
    primary, ro1, ro2 = (tmp_path / n for n in ("primary", "ro1", "ro2"))
    for root in (ro1, ro2):
        with PickleStore(root).acquire("abc", CTX) as lease:
            PickleStore(root).commit(lease, Result(cache_dir=lease.cache_dir))
    store = PickleStore(primary, [ro1, ro2])
    assert store.lookup("abc", CTX, None).cache_dir == ro1 / "abc" / KEY
    with store.acquire("abc", CTX) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert lease.cache_dir == primary / "abc" / KEY
    assert store.lookup("abc", CTX, None).cache_dir == primary / "abc" / KEY


@pytest.mark.timeout(30)
def test_lease_creates_missing_lock_dir(tmp_path):
    root = tmp_path / "missing" / "cache"
    with PickleStore(root).acquire("abc", CTX) as lease:
        assert lease.lockfile.exists()


@pytest.mark.timeout(30)
def test_lease_raises_if_lock_dir_cannot_exist(tmp_path):
    (tmp_path / "file").write_text("")
    with pytest.raises(OSError):
        with PickleStore(tmp_path / "file" / "cache").acquire("abc", CTX):
            pass


def test_lease_async_waits_for_holder(tmp_path):
    store = PickleStore(tmp_path)

    async def contend():
        held = store.acquire("abc", CTX).__enter__()
        waiter = asyncio.ensure_future(store.acquire("abc", CTX).__aenter__())
        await asyncio.sleep(0.3)
        assert not waiter.done()
        held.__exit__(None, None, None)
        lease = await asyncio.wait_for(waiter, 10)
        assert lease.lockfile.exists()
        await lease.__aexit__(None, None, None)
        assert not lease.lockfile.exists()

    asyncio.run(contend())


@python.define
def LeaseUser(me: str, root: str, sync: str) -> int:
    """Take the lease on one key. ``a`` holds it until ``b`` is about to ask for it and
    has then seen ``b`` stay out for a while; ``b`` only asks once ``a`` holds it"""
    import os
    import time
    from pathlib import Path
    from pydra.engine.store import PickleStore

    sync = Path(sync)

    def wait_for(name):
        deadline = time.monotonic() + 60  # bound only
        while not (sync / name).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name} never appeared")
            time.sleep(0.01)

    store = PickleStore(Path(root))
    if me == "b":
        wait_for("held_a")
        (sync / "tried_b").touch()
    with store.acquire("key", {"environment_checksum": "e"}):
        if me == "a":
            (sync / "held_a").touch()
            wait_for("tried_b")
            end = time.monotonic() + 0.5
            while time.monotonic() < end:
                assert not (
                    sync / "entered_b"
                ).exists(), "b entered while a held the lease"
                time.sleep(0.01)
            (sync / "released_a").touch()
        else:
            assert (sync / "released_a").exists()
            (sync / "entered_b").touch()
    return os.getpid()


@workflow.define(outputs=["a", "b"])
def LeaseContention(root: str, sync: str):
    a = workflow.add(LeaseUser(me="a", root=root, sync=sync), name="a")
    b = workflow.add(LeaseUser(me="b", root=root, sync=sync), name="b")
    return a.out, b.out


@pytest.mark.timeout(120)
def test_lease_excludes_across_processes(tmp_path):
    """Two cf processes ask for the same lease at once: the second waits for the first
    to release it"""
    sync = tmp_path / "sync"
    sync.mkdir()
    with Submitter(worker="cf", n_procs=2, cache_root=tmp_path / "cache") as sub:
        outputs = sub(
            LeaseContention(root=str(tmp_path / "locks"), sync=str(sync)),
            raise_errors=True,
        ).outputs
    assert outputs.a != outputs.b
    assert (sync / "entered_b").exists()


def test_submitter_default_store(tmp_path):
    cache_root, ro = tmp_path / "cache", tmp_path / "ro"
    sub = Submitter(cache_root=cache_root, readonly_caches=[ro])
    assert isinstance(sub.store, PickleStore)
    assert sub.store.cache_root == cache_root.resolve()
    assert sub.store.readonly_caches == [ro]
    job = Job(task=FunAddTwo(a=1), submitter=sub, name="job")
    assert job.store is sub.store
    assert job.cache_dir == sub.cache_root / job.checksum / context_key(job.context)
    assert job.lockfile == sub.store.acquire(job.checksum, job.context).lockfile


def test_store_call_sites(worker, tmp_path):
    """Every read and write of a job's result goes through the submitter's store,
    whichever worker runs the job"""
    log = tmp_path / "store.log"
    store = RecordingStore(tmp_path / "cache", log)
    task = FunAddTwo(a=3)
    checksum = task._checksum

    with Submitter(worker=worker, cache_root=tmp_path / "cache", store=store) as sub:
        assert sub(task).outputs.out == 5
    calls = log.read_text().splitlines()
    assert calls.index(f"acquire {checksum}") < calls.index(f"commit {checksum} job")
    assert calls.count(f"commit {checksum} job") == 1
    assert not [c for c in calls if c.endswith("nojob")]
    assert len(list((tmp_path / "cache" / checksum).glob("*/_result.pklz"))) == 1
    assert len(list((tmp_path / "cache" / checksum).glob("*/_job.pklz"))) == 1

    log.unlink()
    with Submitter(worker=worker, cache_root=tmp_path / "cache", store=store) as sub:
        assert sub(FunAddTwo(a=3)).outputs.out == 5
    calls = log.read_text().splitlines()
    assert f"lookup {checksum} hit" in calls
    assert not [c for c in calls if c.startswith("commit")]


def test_store_call_sites_workflow(worker, tmp_path):
    """A workflow job and its node both commit through the store, and a rerun
    commits nothing"""
    log = tmp_path / "store.log"
    store = RecordingStore(tmp_path / "cache", log)

    with Submitter(worker=worker, cache_root=tmp_path / "cache", store=store) as sub:
        assert sub(AddTwoWf(a=3)).outputs.out == 5
    commits = {c for c in log.read_text().splitlines() if c.startswith("commit")}
    assert len(commits) == 2

    log.unlink()
    with Submitter(worker=worker, cache_root=tmp_path / "cache", store=store) as sub:
        assert sub(AddTwoWf(a=3)).outputs.out == 5
    assert not [c for c in log.read_text().splitlines() if c.startswith("commit")]


def test_commit_writes_result_and_job_in_one_call(tmp_path):
    sub = Submitter(cache_root=tmp_path)
    job = Job(task=FunAddTwo(a=1), submitter=sub, name="job")
    with sub.store.acquire(job.checksum, job.context) as lease:
        sub.store.commit(lease, Result(cache_dir=lease.cache_dir), job=job)
    assert (lease.cache_dir / "_result.pklz").exists()
    assert (lease.cache_dir / "_job.pklz").exists()


class ElsewhereLockStore(PickleStore):
    def lockfile(self, checksum, context):
        return self.cache_root / "locks" / checksum


def test_job_lockfile_comes_from_store(tmp_path):
    store = ElsewhereLockStore(tmp_path)
    sub = Submitter(store=store)
    job = Job(task=FunAddTwo(a=1), submitter=sub, name="job")
    assert job.lockfile == tmp_path / "locks" / job.checksum
    assert store.acquire(job.checksum, job.context).lockfile == job.lockfile


def test_store_and_cache_root_must_agree(tmp_path):
    store = PickleStore(tmp_path / "a")
    with pytest.raises(ValueError, match="cache_root"):
        Submitter(store=store, cache_root=tmp_path / "b")
    with pytest.raises(ValueError, match="readonly_caches"):
        Submitter(store=store, readonly_caches=[tmp_path / "c"])
    assert Submitter(store=store).cache_root == (tmp_path / "a").resolve()
    assert Submitter(store=store, cache_root=tmp_path / "a").store is store


def test_picklestore_slots(tmp_path):
    store = PickleStore(tmp_path)
    other = {**CTX, "platform": "q"}
    for checksum, context in (("abc", CTX), ("abc", other), ("def", CTX)):
        with store.acquire(checksum, context) as lease:
            store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert set(store.slots("abc")) == {
        store.cache_dir("abc", CTX),
        store.cache_dir("abc", other),
    }
    assert list(store.slots("def")) == [store.cache_dir("def", CTX)]
    assert len(list(store.slots())) == 3
    assert list(store.slots("missing")) == []


def _ebusy(name):
    return OSError(errno.EBUSY, "Device or resource busy", name)


def _sharing_violation(name):
    error = PermissionError(
        errno.EACCES,
        "The process cannot access the file because it is being used by another process",
        name,
        32,
    )
    if getattr(error, "winerror", None) is None:
        error.winerror = 32
    return error


@pytest.fixture
def held_open(monkeypatch):
    """Makes ``os.unlink`` of files with a given name fail as if another process held
    them open, a given number of times; records every refused attempt"""
    monkeypatch.setattr(store_module, "REMOVE_BACKOFF", 0)
    refused = []

    def hold(name, times, error=_ebusy):
        unlink = os.unlink

        matches = name if callable(name) else name.__eq__

        def held_unlink(path, *args, **kwargs):
            if matches(os.path.basename(path)) and len(refused) < times:
                refused.append(path)
                raise error(os.fspath(path))
            return unlink(path, *args, **kwargs)

        monkeypatch.setattr(os, "unlink", held_unlink)
        return refused

    return hold


def _slot_with_result(tmp_path):
    slot = tmp_path / "abc" / KEY
    slot.mkdir(parents=True)
    (slot / "_result.pklz").write_bytes(b"result")
    return slot


@pytest.mark.parametrize("error", [_ebusy, _sharing_violation])
def test_remove_from_cache_retries_a_held_file(tmp_path, held_open, error):
    slot = _slot_with_result(tmp_path)
    refused = held_open("_result.pklz", 2, error)
    remove_from_cache(slot)
    assert not slot.exists()
    assert len(refused) == 2


@pytest.mark.parametrize("error", [_ebusy, _sharing_violation])
def test_remove_from_cache_raises_cache_busy_when_held(tmp_path, held_open, error):
    slot = _slot_with_result(tmp_path)
    refused = held_open("_result.pklz", store_module.REMOVE_ATTEMPTS + 1, error)
    with pytest.raises(CacheBusyError) as excinfo:
        remove_from_cache(slot)
    assert len(refused) == store_module.REMOVE_ATTEMPTS
    assert excinfo.value.filename == str(slot)
    assert Path(excinfo.value.filename2).name == "_result.pklz"
    assert "_result.pklz held open or pending deletion" in str(excinfo.value)
    assert [Path(h).name for h in excinfo.value.held] == ["_result.pklz"]
    assert (slot / "_result.pklz").exists()
    if error is _sharing_violation:
        assert excinfo.value.winerror == 32
    unpickled = cloudpickle.loads(cloudpickle.dumps(excinfo.value))
    assert isinstance(unpickled, CacheBusyError)
    assert unpickled.held == excinfo.value.held


def test_remove_from_cache_raises_other_errors_at_once(tmp_path, held_open):
    slot = _slot_with_result(tmp_path)
    refused = held_open(
        "_result.pklz", 2, lambda n: PermissionError(errno.EACCES, "denied", n)
    )
    with pytest.raises(PermissionError) as excinfo:
        remove_from_cache(slot)
    assert not isinstance(excinfo.value, CacheBusyError)
    assert len(refused) == 1


def test_remove_from_cache_removes_a_held_lockfile(tmp_path, held_open):
    lockfile = tmp_path / "abc.lock"
    lockfile.write_text("")
    refused = held_open("abc.lock", 1)
    remove_from_cache(lockfile)
    assert not lockfile.exists()
    assert len(refused) == 1


@pytest.mark.parametrize("times", [2, store_module.REMOVE_ATTEMPTS])
def test_rerun_clears_a_held_slot(tmp_path, held_open, times):
    task = FunAddTwo(a=1)
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        assert sub(task).outputs.out == 3
    refused = held_open("_result.pklz", times)
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        if times < store_module.REMOVE_ATTEMPTS:
            assert sub(task, rerun=True).outputs.out == 3
        else:
            with pytest.raises(CacheBusyError, match="_result.pklz"):
                sub(task, rerun=True)
    assert len(refused) == times


def _not_empty(name):
    return OSError(errno.ENOTEMPTY, "Directory not empty", name)


def _rmdir_refusing(monkeypatch, directory, refusals, before_refusal):
    """Make ``os.rmdir`` of ``directory`` raise ENOTEMPTY ``refusals`` times, calling
    ``before_refusal(n)`` before the n-th; records the unlinked names"""
    monkeypatch.setattr(store_module, "REMOVE_BACKOFF", 0)
    rmdir, unlink = os.rmdir, os.unlink
    refused, unlinked = [], []

    def refusing_rmdir(path, *args, **kwargs):
        if os.path.basename(path) == directory.name and len(refused) < refusals:
            before_refusal(len(refused))
            refused.append(path)
            raise _not_empty(os.fspath(path))
        return rmdir(path, *args, **kwargs)

    def recording_unlink(path, *args, **kwargs):
        unlinked.append(os.path.basename(path))
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "rmdir", refusing_rmdir)
    monkeypatch.setattr(os, "unlink", recording_unlink)
    return refused, unlinked


def test_remove_from_cache_reports_a_concurrent_writer(tmp_path, monkeypatch):
    slot = _slot_with_result(tmp_path)
    late = slot / "late.txt"
    refused, unlinked = _rmdir_refusing(
        monkeypatch, slot, 1, lambda n: late.write_text("written during removal")
    )
    with pytest.raises(CacheWriterError) as excinfo:
        remove_from_cache(slot)
    assert len(refused) == 1
    assert excinfo.value.filename == str(slot)
    assert excinfo.value.filename2 == str(late)
    assert excinfo.value.leftovers == [str(late)]
    assert str(late) in str(excinfo.value)
    assert late.read_text() == "written during removal"
    assert "late.txt" not in unlinked


def _silly_rename(slot):
    return slot / ".nfs000000000001"


def _delete_pending(slot, monkeypatch):
    pending = slot / "pending.txt"
    stat = os.stat

    def pending_stat(path, *args, **kwargs):
        if os.fspath(path) == str(pending):
            error = PermissionError(errno.EACCES, "Access is denied", str(pending))
            error.winerror = 303
            raise error
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", pending_stat)
    return pending


@pytest.mark.parametrize("kind", ["nfs", "windows"])
def test_remove_from_cache_waits_for_pending_files(tmp_path, monkeypatch, kind):
    slot = _slot_with_result(tmp_path)
    pending = (
        _silly_rename(slot) if kind == "nfs" else _delete_pending(slot, monkeypatch)
    )

    def appear_then_go(n):
        if n == 0:
            pending.write_text("")
        elif n == 2:
            pending.unlink()

    refused, unlinked = _rmdir_refusing(monkeypatch, slot, 3, appear_then_go)
    remove_from_cache(slot)
    assert not slot.exists()
    assert len(refused) == 3
    assert unlinked.count(pending.name) == 1  # by the test, standing in for the holder


def test_remove_from_cache_raises_cache_busy_on_pending_files(tmp_path, monkeypatch):
    slot = _slot_with_result(tmp_path)
    pending = _silly_rename(slot)
    refused, unlinked = _rmdir_refusing(
        monkeypatch, slot, store_module.REMOVE_ATTEMPTS, lambda n: pending.touch()
    )
    with pytest.raises(CacheBusyError) as excinfo:
        remove_from_cache(slot)
    assert len(refused) == store_module.REMOVE_ATTEMPTS
    assert excinfo.value.filename2 == str(pending)
    assert excinfo.value.held == [str(pending)]
    assert pending.name not in unlinked


def test_remove_from_cache_counts_a_vanished_path_as_removed(tmp_path, monkeypatch):
    lockfile = tmp_path / "abc.lock"
    lockfile.write_text("")
    unlink = os.unlink

    def removed_by_another_cleaner(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        raise FileNotFoundError(errno.ENOENT, "No such file or directory", path)

    monkeypatch.setattr(os, "unlink", removed_by_another_cleaner)
    remove_from_cache(lockfile)
    assert not lockfile.exists()


def _is_info_file(name):
    return name.endswith("_info.json")


def test_held_info_file_does_not_fail_a_committed_job(
    tmp_path, held_open, caplog, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    refused = held_open(_is_info_file, store_module.REMOVE_ATTEMPTS)
    with caplog.at_level("WARNING", logger="pydra"):
        with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
            assert sub(FunAddTwo(a=1)).outputs.out == 3
    assert len(refused) == store_module.REMOVE_ATTEMPTS
    assert os.getcwd() == str(tmp_path)
    (info_file,) = (tmp_path / "cache").glob("*_info.json")
    assert f"Could not remove {info_file}" in caplog.text


@python.define
def RaisesBoom(x: int) -> int:
    raise ValueError("boom")


def test_held_info_file_keeps_the_task_error(tmp_path, held_open, monkeypatch):
    monkeypatch.chdir(tmp_path)
    held_open(_is_info_file, store_module.REMOVE_ATTEMPTS)
    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        with pytest.raises(Exception, match="boom") as excinfo:
            sub(RaisesBoom(x=1), raise_errors=True)
    assert not isinstance(excinfo.value, CacheBusyError)
    assert os.getcwd() == str(tmp_path)
