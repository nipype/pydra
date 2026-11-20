"""The per-slot success record: a cached job is read without its lease and is not
dispatched (proposal 0001 §3.2, Task 44 (b))"""

import sys
import cloudpickle as cp
import pytest
from pydra.compose import python, workflow
from pydra.engine import store as store_module
from pydra.engine.job import Job
from pydra.engine.lease import LeaseLost
from pydra.engine.result import Result
from pydra.engine.store import SUCCESS_DIR, PickleStore, context_key
from pydra.engine.submitter import NodeExecution, Submitter
from pydra.workers import cf, debug

WORKERS = {"debug": debug.Worker, "cf": cf.Worker}


@pytest.fixture(autouse=True)
def _by_value():
    """Worker processes cannot import this module, so they receive its tasks by value"""
    cp.register_pickle_by_value(sys.modules[__name__])
    yield
    cp.unregister_pickle_by_value(sys.modules[__name__])


@python.define
def Triple(x: int) -> int:
    return 3 * x


@python.define
def FailEvenUnlessFlagged(x: int, flag: str) -> int:
    """Raise on even ``x`` until the file ``flag`` exists"""
    from pathlib import Path

    if x % 2 == 0 and not Path(flag).exists():
        raise ValueError("even error")
    return x


@workflow.define
def Recoverable(x: list[int], flag: str) -> list[int]:
    failing = workflow.add(FailEvenUnlessFlagged(flag=flag).split("x", x=x))
    return failing.out


@workflow.define
def Wrapped(x: int) -> int:
    return workflow.add(Triple(x=x)).out


def records(root):
    directory = root / SUCCESS_DIR
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def slot_of(root, x):
    """The slot of the ``Triple(x=x)`` job run by a submitter on ``root``"""
    with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
        job = Job(task=Triple(x=x), submitter=sub, name="t")
        return job.cache_dir, sub.store.success_record(job.checksum, job.context)


@pytest.fixture
def counted(monkeypatch):
    """The leases acquired and the jobs run by a worker in this process, by name"""
    calls = {"acquire": [], "run": []}
    acquire = PickleStore.acquire

    def counting_acquire(self, checksum, context):
        calls["acquire"].append(checksum)
        return acquire(self, checksum, context)

    monkeypatch.setattr(PickleStore, "acquire", counting_acquire)
    for name, cls in WORKERS.items():
        run = cls.run
        if name == "cf":

            async def counting_run(self, job, rerun=False, run=run):
                calls["run"].append(job.name)
                return await run(self, job, rerun=rerun)

        else:

            def counting_run(self, job, rerun=False, run=run):
                calls["run"].append(job.name)
                return run(self, job, rerun=rerun)

        monkeypatch.setattr(cls, "run", counting_run)
    return calls


def run_split(root, worker, items):
    with Submitter(worker=worker, store="pickle", cache_root=root) as sub:
        return sub(Triple().split(x=items))


def nodes(calls):
    return [name for name in calls["run"] if name != "main"]


# The submitter's fast path


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_cached_job_takes_no_lease_and_is_not_dispatched(tmp_path, worker, counted):
    root = tmp_path / "cache"
    run_split(root, worker, [1, 2, 3])
    assert len(records(root)) == 4  # three elements and the outer job
    counted["acquire"].clear()
    counted["run"].clear()
    # the elements reversed: a new outer job whose elements are all cached
    assert run_split(root, worker, [3, 2, 1]).outputs.out == [9, 6, 3]
    assert nodes(counted) == []
    assert len(counted["acquire"]) == 1  # the outer job only


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_cache_written_before_the_record_takes_the_leased_path(
    tmp_path, worker, counted
):
    root = tmp_path / "cache"
    run_split(root, worker, [1, 2, 3])
    for name in records(root):
        (
            root / SUCCESS_DIR / name
        ).unlink()  # as a cache written before the record existed
    counted["acquire"].clear()
    counted["run"].clear()
    assert run_split(root, worker, [3, 2, 1]).outputs.out == [9, 6, 3]
    assert sorted(nodes(counted)) == ["Triple"] * 3
    if worker == "debug":
        assert len(counted["acquire"]) == 4
    assert len(records(root)) == 1  # a leased hit writes no record; the new outer does


@pytest.mark.parametrize("worker", ["debug", "cf"])
@pytest.mark.parametrize("damage", ["partial", "missing", "errored", "garbage"])
def test_stale_or_partial_record_falls_back_to_the_leased_path(
    tmp_path, worker, damage, counted
):
    root = tmp_path / "cache"
    run_split(root, worker, [1, 2, 3])
    slot, record = slot_of(root, 2)
    assert record.exists()
    result_file = slot / "_result.pklz"
    if damage == "partial":
        result_file.write_bytes(result_file.read_bytes()[:20])
    elif damage == "missing":
        store_module.remove_from_cache(slot)
    elif damage == "errored":
        result = cp.loads(result_file.read_bytes())
        result.errored = True
        result_file.write_bytes(cp.dumps(result))
    else:
        result_file.write_bytes(b"not a pickle")
    counted["run"].clear()
    assert run_split(root, worker, [3, 2, 1]).outputs.out == [9, 6, 3]
    assert nodes(counted) == ["Triple"]


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_errored_slot_is_rerun_under_its_lease_and_never_read_early(
    tmp_path, worker, monkeypatch
):
    """A slot whose job errored carries no record, so its stored result is not read
    between the job being queued and its run starting"""
    flag = tmp_path / "flag"
    root = tmp_path / "cache"
    wf = Recoverable(x=[1, 2, 3, 4], flag=str(flag))
    with pytest.raises(Exception, match="even error"):
        wf(worker=worker, store="pickle", cache_root=root)
    with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
        jobs = {
            x: Job(
                task=FailEvenUnlessFlagged(x=x, flag=str(flag)), submitter=sub, name="f"
            )
            for x in (1, 2, 3, 4)
        }
        recorded = {
            x
            for x, job in jobs.items()
            if sub.store.success_record(job.checksum, job.context).exists()
        }
    errored = {jobs[x].checksum for x in (2, 4)}
    assert 1 in recorded and not recorded & {2, 4}  # debug fails fast, cf does not

    waiting: set[str] = set()
    early: list[str] = []
    ran: list[str] = []
    release = NodeExecution._release

    def recording_release(self, index):
        release(self, index)
        if index in self.queued:
            waiting.add(self.queued[index].checksum)

    load_result = store_module.load_result

    def recording_load(slot, locations, *args, **kwargs):
        if slot.split("/")[0] in waiting:
            early.append(slot)
        return load_result(slot, locations, *args, **kwargs)

    worker_cls = WORKERS[worker]
    run = worker_cls.run
    if worker == "cf":

        async def recording_run(self, job, rerun=False):
            waiting.discard(job.checksum)
            ran.append(job.task.x if job.name != "main" else None)
            return await run(self, job, rerun=rerun)

    else:

        def recording_run(self, job, rerun=False):
            waiting.discard(job.checksum)
            ran.append(job.task.x if job.name != "main" else None)
            return run(self, job, rerun=rerun)

    monkeypatch.setattr(NodeExecution, "_release", recording_release)
    monkeypatch.setattr(store_module, "load_result", recording_load)
    monkeypatch.setattr(worker_cls, "run", recording_run)

    flag.touch()
    assert wf(worker=worker, store="pickle", cache_root=root).out == [1, 2, 3, 4]
    assert not [s for s in early if s.split("/")[0] in errored]
    assert sorted(x for x in ran if x is not None) == sorted({1, 2, 3, 4} - recorded)
    assert len(records(root)) == 5  # every element now, and the outer job


# The store


def test_commit_writes_the_record_last_and_only_on_success(tmp_path, monkeypatch):
    store = PickleStore(tmp_path)
    written = []
    save = store_module.save

    def recording_save(*args, **kwargs):
        written.append(("save", store.success_record("abc", {}).exists()))
        return save(*args, **kwargs)

    monkeypatch.setattr(store_module, "save", recording_save)
    with store.acquire("abc", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert written == [("save", False)]
    assert store.success_record("abc", {}).exists()
    with store.acquire("def", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir, errored=True))
    assert not store.success_record("def", {}).exists()


def test_lost_holder_writes_no_record(tmp_path):
    store = PickleStore(tmp_path)
    with store.acquire("abc", {}) as lease:
        (token_dir,) = lease.lockfile.iterdir()
        token_dir.rename(lease.lockfile / "gone-taken")
        with pytest.raises(LeaseLost):
            store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not store.success_record("abc", {}).exists()


def test_holder_that_loses_its_lease_while_saving_writes_no_record(
    tmp_path, monkeypatch
):
    store = PickleStore(tmp_path)
    save = store_module.save

    def save_then_lose(path, **kwargs):
        save(path, **kwargs)
        (token_dir,) = lease.lockfile.iterdir()
        token_dir.rename(lease.lockfile / "gone-taken")

    monkeypatch.setattr(store_module, "save", save_then_lose)
    with store.acquire("abc", {}) as lease:
        with pytest.raises(LeaseLost):
            store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not store.success_record("abc", {}).exists()


def test_read_succeeded_needs_the_record_and_a_good_result(tmp_path):
    store = PickleStore(tmp_path)
    assert store.read_succeeded("abc", {}, None) is None
    with store.acquire("abc", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert store.read_succeeded("abc", {}, None).cache_dir == lease.cache_dir
    (lease.cache_dir / "_result.pklz").write_bytes(b"\x80")
    assert store.read_succeeded("abc", {}, None) is None
    with store.acquire("abc", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    store.success_record("abc", {}).unlink()
    assert store.read_succeeded("abc", {}, None) is None


def test_records_are_not_slots(tmp_path):
    store = PickleStore(tmp_path)
    with store.acquire("abc", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert list(store.slots()) == [lease.cache_dir]


def test_readonly_cache_is_read_under_the_lease(tmp_path):
    ro, primary = tmp_path / "ro", tmp_path / "primary"
    with PickleStore(ro).acquire("abc", {}) as lease:
        PickleStore(ro).commit(lease, Result(cache_dir=lease.cache_dir))
    store = PickleStore(primary, [ro])
    primary.mkdir()
    assert store.read_succeeded("abc", {}, None) is None
    assert store.lookup("abc", {}, None) is not None


# The job


@pytest.mark.parametrize("asynchronous", [False, True])
def test_rerun_removes_the_record_before_rewriting_the_slot(
    tmp_path, monkeypatch, asynchronous
):
    sub = Submitter(worker="debug", store="pickle", cache_root=tmp_path)
    task = Wrapped(x=1) if asynchronous else Triple(x=1)
    job = Job(task=task, submitter=sub, name="t")
    job.run()
    record = sub.store.success_record(job.checksum, job.context)
    assert record.exists()
    seen = []
    populate = Job._populate_filesystem

    def checking_populate(self):
        seen.append(record.exists())
        populate(self)

    monkeypatch.setattr(Job, "_populate_filesystem", checking_populate)
    again = Job(task=task, submitter=sub, name="t")
    if asynchronous:
        sub.loop.run_until_complete(again.run_async(rerun=True))
    else:
        again.run(rerun=True)
    assert seen[0] is False  # the workflow's own slot, before its node's
    assert record.exists()  # written again by the rerun's commit
    sub.close()


def test_errored_rerun_leaves_no_record(tmp_path):
    flag = tmp_path / "flag"
    flag.touch()
    sub = Submitter(worker="debug", store="pickle", cache_root=tmp_path / "cache")
    task = FailEvenUnlessFlagged(x=2, flag=str(flag))
    Job(task=task, submitter=sub, name="f").run()
    job = Job(task=task, submitter=sub, name="f")
    record = sub.store.success_record(job.checksum, job.context)
    assert record.exists()
    flag.unlink()
    with pytest.raises(ValueError, match="even error"):
        job.run(rerun=True)
    assert not record.exists()
    sub.close()


# Cost, failures to write, reruns and the store's hit hook


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_record_checks_grow_with_the_queued_jobs_not_the_cache(
    tmp_path, worker, monkeypatch
):
    """Each queued job costs one existence check of its own record; the record
    directory is never listed, however many records it holds"""
    import os

    root = tmp_path / "cache"
    (root / SUCCESS_DIR).mkdir(parents=True)
    for i in range(2000):  # records of earlier work in the same cache root
        (root / SUCCESS_DIR / f"python-{i:032x}-0000000000000000").touch()
    checks: list[str] = []
    listed: list[str] = []
    exists = store_module._record_exists

    def counting_exists(path):
        checks.append(str(path))
        return exists(path)

    scandir, listdir = os.scandir, os.listdir

    def counting_scandir(path="."):
        if SUCCESS_DIR in str(path):
            listed.append(str(path))
        return scandir(path)

    def counting_listdir(path="."):
        if SUCCESS_DIR in str(path):
            listed.append(str(path))
        return listdir(path)

    monkeypatch.setattr(store_module, "_record_exists", counting_exists)
    monkeypatch.setattr(os, "scandir", counting_scandir)
    monkeypatch.setattr(os, "listdir", counting_listdir)
    run_split(root, worker, [1, 2, 3, 4, 5])
    assert len(checks) == 5  # cold: one check per queued element
    checks.clear()
    assert run_split(root, worker, [5, 4, 3, 2, 1]).outputs.out == [15, 12, 9, 6, 3]
    assert len(checks) == 5  # all cached: one check per queued element
    assert listed == []


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(__import__("os"), "geteuid", lambda: 1)() == 0,
    reason="needs POSIX permissions and a non-root user",
)
@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_unwritable_record_directory_does_not_fail_a_job(
    tmp_path, worker, counted, caplog
):
    root = tmp_path / "cache"
    (root / SUCCESS_DIR).mkdir(parents=True)
    (root / SUCCESS_DIR).chmod(0o555)  # created by another user, umask 022
    try:
        result = run_split(root, worker, [1, 2])
        assert not result.errored
        assert result.outputs.out == [3, 6]
        assert records(root) == []
        counted["run"].clear()
        assert run_split(root, worker, [2, 1]).outputs.out == [6, 3]
        assert sorted(nodes(counted)) == ["Triple"] * 2  # the leased path
    finally:
        (root / SUCCESS_DIR).chmod(0o755)
    if worker == "debug":
        assert "Could not write the success record" in caplog.text


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_rerun_reruns_recorded_jobs_in_the_same_cache_root(tmp_path, worker, counted):
    root = tmp_path / "cache"
    task = Triple().split(x=[1, 2, 3])
    with Submitter(worker=worker, store="pickle", cache_root=root) as sub:
        sub(task)
    assert len(records(root)) == 4
    counted["run"].clear()
    with Submitter(worker=worker, store="pickle", cache_root=root) as sub:
        assert sub(task, rerun=True).outputs.out == [3, 6, 9]
    assert sorted(nodes(counted)) == ["Triple"] * 3


class RefusingStore(PickleStore):
    """A store whose hit hook refuses the first reuse of each stored ``Triple``
    result in a process, as a store sampling results for verification might"""

    def __init__(self, cache_root):
        super().__init__(cache_root)
        self.refused = []

    def lookup(self, checksum, context, task, stored=None):
        if type(task).__name__ == "Triple" and checksum not in self.refused:
            self.refused.append(checksum)
            return None
        return super().lookup(checksum, context, task, stored)


@pytest.mark.parametrize("worker", ["debug", "cf"])
def test_store_hit_hook_is_called_on_the_lease_free_path(tmp_path, worker, counted):
    root = tmp_path / "cache"
    run_split(root, worker, [1, 2, 3])
    assert len(records(root)) == 4
    counted["run"].clear()
    store = RefusingStore(root)
    with Submitter(worker=worker, cache_root=root, store=store) as sub:
        assert sub(Triple().split(x=[3, 2, 1])).outputs.out == [9, 6, 3]
    assert len(store.refused) == 3  # asked on the lease-free path
    assert sorted(nodes(counted)) == ["Triple"] * 3  # refused, so run under the lease
