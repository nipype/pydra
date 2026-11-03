import asyncio
import sys
from pathlib import Path
import cloudpickle
import pytest
from pydra.compose import python, workflow
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.store import PickleStore, ResultStore, context_key
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

    def lookup(self, checksum, context):
        result = super().lookup(checksum, context)
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
    assert store.lookup("abc", CTX) is None
    with store.acquire("abc", CTX) as lease:
        assert lease.lockfile.exists()
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not lease.lockfile.exists()
    assert (tmp_path / "abc" / KEY / "_result.pklz").exists()
    result = store.lookup("abc", CTX)
    assert result.cache_dir == tmp_path / "abc" / KEY
    assert not result.errored


def test_picklestore_readonly_caches(tmp_path):
    primary, ro1, ro2 = (tmp_path / n for n in ("primary", "ro1", "ro2"))
    for root in (ro1, ro2):
        with PickleStore(root).acquire("abc", CTX) as lease:
            PickleStore(root).commit(lease, Result(cache_dir=lease.cache_dir))
    store = PickleStore(primary, [ro1, ro2])
    assert store.lookup("abc", CTX).cache_dir == ro1 / "abc" / KEY
    with store.acquire("abc", CTX) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert lease.cache_dir == primary / "abc" / KEY
    assert store.lookup("abc", CTX).cache_dir == primary / "abc" / KEY


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
                assert not (sync / "entered_b").exists(), "b entered while a held the lease"
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
