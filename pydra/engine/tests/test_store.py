import asyncio
from pathlib import Path
from pydra.compose import workflow
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.store import PickleStore, ResultStore
from pydra.engine.submitter import Submitter
from pydra.engine.tests.utils import FunAddTwo


class RecordingStore(PickleStore):
    """A PickleStore that appends each call to a log file, so calls made in worker
    processes are visible to the test"""

    def __init__(self, cache_root, log):
        super().__init__(cache_root)
        self.log = Path(log)

    def _record(self, *entry):
        with open(self.log, "a") as f:
            f.write(" ".join(entry) + "\n")

    def lookup(self, checksum, context=None, produced_by=None):
        result = super().lookup(checksum, context, produced_by)
        self._record("lookup", checksum, "miss" if result is None else "hit")
        return result

    def acquire(self, checksum):
        self._record("acquire", checksum)
        return super().acquire(checksum)

    def commit(self, lease, result):
        self._record("commit", lease.checksum)
        super().commit(lease, result)


@workflow.define
def AddTwoWf(a: int) -> int:
    node = workflow.add(FunAddTwo(a=a))
    return node.out


def test_picklestore_is_resultstore(tmp_path):
    assert isinstance(PickleStore(tmp_path), ResultStore)


def test_picklestore_layout(tmp_path):
    store = PickleStore(tmp_path)
    assert store.cache_dir("abc") == tmp_path / "abc"
    lease = store.acquire("abc")
    assert lease.checksum == "abc"
    assert lease.cache_dir == tmp_path / "abc"
    assert lease.lockfile == tmp_path / "abc.lock"
    assert not lease.lockfile.exists()


def test_picklestore_commit_lookup(tmp_path):
    store = PickleStore(tmp_path)
    assert store.lookup("abc") is None
    with store.acquire("abc") as lease:
        assert lease.lockfile.exists()
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not lease.lockfile.exists()
    assert (tmp_path / "abc" / "_result.pklz").exists()
    result = store.lookup("abc")
    assert result.cache_dir == tmp_path / "abc"
    assert not result.errored


def test_picklestore_readonly_caches(tmp_path):
    primary, ro1, ro2 = (tmp_path / n for n in ("primary", "ro1", "ro2"))
    for root in (ro1, ro2):
        with PickleStore(root).acquire("abc") as lease:
            PickleStore(root).commit(lease, Result(cache_dir=lease.cache_dir))
    store = PickleStore(primary, [ro1, ro2])
    assert store.lookup("abc").cache_dir == ro1 / "abc"
    with store.acquire("abc") as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert lease.cache_dir == primary / "abc"
    assert store.lookup("abc").cache_dir == primary / "abc"


def test_lease_async_waits_for_holder(tmp_path):
    store = PickleStore(tmp_path)

    async def contend():
        held = store.acquire("abc").__enter__()
        waiter = asyncio.ensure_future(store.acquire("abc").__aenter__())
        await asyncio.sleep(0.3)
        assert not waiter.done()
        held.__exit__(None, None, None)
        lease = await asyncio.wait_for(waiter, 10)
        assert lease.lockfile.exists()
        await lease.__aexit__(None, None, None)
        assert not lease.lockfile.exists()

    asyncio.run(contend())


def test_submitter_default_store(tmp_path):
    cache_root, ro = tmp_path / "cache", tmp_path / "ro"
    sub = Submitter(cache_root=cache_root, readonly_caches=[ro])
    assert isinstance(sub.store, PickleStore)
    assert sub.store.cache_root == cache_root.resolve()
    assert sub.store.readonly_caches == [ro]
    job = Job(task=FunAddTwo(a=1), submitter=sub, name="job")
    assert job.store is sub.store
    assert job.cache_dir == sub.cache_root / job.checksum
    assert job.lockfile == sub.store.acquire(job.checksum).lockfile


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
    assert calls.index(f"acquire {checksum}") < calls.index(f"commit {checksum}")
    assert calls.count(f"commit {checksum}") == 1
    assert (tmp_path / "cache" / checksum / "_result.pklz").exists()
    assert (tmp_path / "cache" / checksum / "_job.pklz").exists()

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
