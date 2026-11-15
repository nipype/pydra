import asyncio
import errno
import json
import shutil
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
import pytest
from pydra.compose import python
import cloudpickle as cp
from pydra.engine.lease import (
    HOLDER_FILE,
    MISSED_BEATS,
    DirectoryLease,
    Lease,
    LeaseLost,
    LeaseSettings,
    LeaseTimeout,
    LeaseUnavailable,
)
from pydra.engine.job import load_and_run
from pydra.engine.result import Result
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.utils import storage
from pydra.utils.storage import StorageUnsupported

FOREIGN_HOST = "some-other-host.example.org"
DEAD_PID = 999999

# Short bounds for local disk: beats every 0.05 s, stale after 0.5 s
FAST = LeaseSettings(stale_after=0.5, timeout=0.5, attribute_delay=0)


class Clock:
    """A storage clock the test sets."""

    def __init__(self, now: float):
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeTime:
    """A monotonic clock, a sleep and a storage clock that move together, in seconds
    from ``base`` on the storage clock. ``on_sleep`` is called after each sleep."""

    def __init__(self, base: float, on_sleep=None):
        self.base = base
        self.t = 0.0
        self.on_sleep = on_sleep

    def monotonic(self) -> float:
        return self.t

    def storage(self) -> float:
        return self.base + self.t

    def sleep(self, seconds: float) -> None:
        self.t += seconds
        if self.on_sleep is not None:
            self.on_sleep(self)

    def lease(self, tmp_path: Path, lockfile: Path, settings) -> DirectoryLease:
        return DirectoryLease(
            "abc",
            tmp_path,
            lockfile,
            settings=settings,
            clock=self.storage,
            monotonic=self.monotonic,
            sleep=self.sleep,
        )


# Beats every second, stale after 10 s, gives up 15 s after the last renewal seen
SLOW = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0)


def plant_foreign_lease(lockfile: Path) -> Path:
    """Plant the lease directory a process on another host left, return its token dir."""
    token_dir = lockfile / ("f" * 32)
    token_dir.mkdir(parents=True)
    (token_dir / HOLDER_FILE).write_text(
        json.dumps(
            {"owner": "someone", "host": FOREIGN_HOST, "pid": DEAD_PID, "started": 0}
        )
    )
    return token_dir


def plant_legacy_lock(lockfile: Path) -> None:
    """Plant the two-line file filelock.SoftFileLock writes, from another host."""
    lockfile.parent.mkdir(parents=True, exist_ok=True)
    lockfile.write_text(f"{DEAD_PID}\n{FOREIGN_HOST}\n")


def newest_mtime(lockfile: Path) -> float:
    """The newest mtime in a lease: itself, its entries and theirs."""
    return max(p.lstat().st_mtime for p in [lockfile, *lockfile.rglob("*")])


def test_directory_lease_is_a_lease(tmp_path):
    assert isinstance(DirectoryLease("abc", tmp_path, tmp_path / "abc.lock"), Lease)
    assert isinstance(PickleStore(tmp_path).acquire("abc", {}), Lease)


def test_lease_records_holder_and_releases(tmp_path):
    lease = DirectoryLease("abc", tmp_path, tmp_path / "abc.lock", settings=FAST)
    with lease:
        assert lease.held
        holder = lease.holder()
        assert holder["host"] == socket.gethostname()
        assert holder["pid"] == os.getpid()
        assert holder["owner"]
        assert holder["started"] == pytest.approx(time.time(), abs=60)
    assert not lease.held
    assert not lease.lockfile.exists()
    assert list(tmp_path.iterdir()) == []


def test_foreign_stale_lease_is_taken_over_after_stale_after(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    planted = newest_mtime(lockfile)
    clock = Clock(planted + 9.9)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert not lease._attempt()  # not yet stale
    assert (lockfile / ("f" * 32)).is_dir()
    clock.now = planted + 10.1
    with lease:
        assert lease.held
        assert lease.holder()["pid"] == os.getpid()
        assert [b["holder"]["host"] for b in lease.broke] == [FOREIGN_HOST]
        assert not (lockfile / ("f" * 32)).exists()
    assert not lockfile.exists()


def test_legacy_lock_file_is_taken_over_by_age(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_legacy_lock(lockfile)
    planted = lockfile.stat().st_mtime
    clock = Clock(planted + 1)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert not lease._attempt()
    assert lockfile.is_file()
    clock.now = planted + 11
    with lease:
        assert lockfile.is_dir()
        assert lease.broke[0]["holder"] == {"pid": DEAD_PID, "host": FOREIGN_HOST}


def test_timeout_names_foreign_holder(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    clock = Clock(newest_mtime(lockfile) + 3)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert not lease._attempt()
    err = lease._timeout(capped=False)
    assert isinstance(err, LeaseTimeout)
    assert err.holder["host"] == FOREIGN_HOST and err.holder["pid"] == DEAD_PID
    assert err.age == pytest.approx(3)
    assert f"pid {DEAD_PID} on host {FOREIGN_HOST}" in str(err)


def test_live_holder_is_never_taken_over(tmp_path):
    """A waiter behind a holder that keeps renewing waits past many stale bounds and
    timeouts, and takes the lease when the holder releases it."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)

    def holder_beats(clock):
        if clock.t < 200:
            os.utime(token_dir, (clock.storage(), clock.storage()))
        elif lockfile.exists():
            shutil.rmtree(lockfile)  # the holder finishes and releases

    clock = FakeTime(newest_mtime(lockfile), on_sleep=holder_beats)
    waiter = clock.lease(tmp_path, lockfile, SLOW)
    with waiter:
        assert clock.t >= 200  # 20 stale bounds, 13 timeouts
        assert waiter.broke == []
        assert waiter.holder()["pid"] == os.getpid()


def test_max_wait_caps_a_wait_behind_a_live_holder(tmp_path):
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)

    def holder_beats(clock):
        os.utime(token_dir, (clock.storage(), clock.storage()))

    clock = FakeTime(newest_mtime(lockfile), on_sleep=holder_beats)
    settings = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, max_wait=50)
    with pytest.raises(LeaseTimeout) as excinfo:
        clock.lease(tmp_path, lockfile, settings).__enter__()
    err = excinfo.value
    assert type(err) is LeaseTimeout
    assert 50 <= clock.t < 52
    assert "max_wait" in str(err) and FOREIGN_HOST in str(err)
    assert err.holder["pid"] == DEAD_PID and err.age < 10
    assert token_dir.is_dir()


@pytest.fixture
def takeover_refused(monkeypatch):
    """Renaming a lease's token aside and unlinking a legacy lock are refused, as on a
    shared cache where the lease belongs to another user."""
    real_rename, real_unlink = os.rename, os.unlink

    def rename(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("gone-"):
            raise PermissionError(errno.EACCES, "Permission denied", str(src))
        return real_rename(src, dst, *args, **kwargs)

    def unlink(path, *args, **kwargs):
        if str(path).endswith(".lock"):
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(os, "unlink", unlink)


def test_unrenewed_unremovable_holder_times_out(tmp_path, takeover_refused):
    """A holder that is neither renewed nor removable ends the wait after timeout."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    clock = FakeTime(newest_mtime(lockfile))
    with pytest.raises(LeaseTimeout):
        clock.lease(tmp_path, lockfile, SLOW).__enter__()
    assert 15 <= clock.t < 17
    assert token_dir.is_dir()


@pytest.mark.parametrize("legacy", [False, True])
def test_refused_takeover_raises_lease_unavailable(tmp_path, takeover_refused, legacy):
    lockfile = tmp_path / "abc.lock"
    if legacy:
        plant_legacy_lock(lockfile)
        planted = lockfile.stat().st_mtime
    else:
        plant_foreign_lease(lockfile)
        planted = newest_mtime(lockfile)
    clock = FakeTime(planted)
    with pytest.raises(LeaseUnavailable) as excinfo:
        clock.lease(tmp_path, lockfile, SLOW).__enter__()
    err = excinfo.value
    assert isinstance(err.error, PermissionError)
    assert err.lockfile == lockfile
    assert err.holder["host"] == FOREIGN_HOST and err.holder["pid"] == DEAD_PID
    message = str(err)
    assert str(lockfile) in message and FOREIGN_HOST in message
    assert "refused" in message and "Permission denied" in message


def test_holder_that_missed_a_beat_is_not_taken_over(tmp_path):
    """Under the defaults, a holder whose newest mark is one missed beat plus the
    attribute delay old still holds; it is taken over just past stale_after."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    settings = LeaseSettings()
    planted = newest_mtime(lockfile)
    clock = Clock(
        planted
        + (1 + MISSED_BEATS) * settings.renew_interval
        + settings.attribute_delay
    )
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert not lease._attempt()
    assert token_dir.is_dir()
    clock.now = planted + settings.stale_after + 0.1
    assert lease._attempt()
    lease._release()


def test_lease_taken_over_while_held_is_marked_lost(tmp_path):
    lockfile = tmp_path / "abc.lock"
    lease = DirectoryLease("abc", tmp_path, lockfile)  # no beat within the test
    with lease:
        assert lease.prove()
        (token_dir,) = lockfile.iterdir()
        token_dir.rename(lockfile / "gone-taken")
        assert not lease.prove()
        assert lease.lost is not None
        assert not lease.held
    assert lockfile.exists()  # left to whoever took it over


def test_lost_lease_does_not_commit(tmp_path):
    store = PickleStore(tmp_path)
    with store.acquire("abc", {}) as lease:
        (token_dir,) = lease.lockfile.iterdir()
        token_dir.rename(lease.lockfile / "gone-taken")
        with pytest.raises(LeaseLost):
            store.commit(lease, Result(cache_dir=lease.cache_dir))
    assert not lease.cache_dir.exists()


WAITER = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    from pydra.engine.lease import DirectoryLease, LeaseSettings, LeaseTimeout

    lockfile = Path(sys.argv[1])
    settings = LeaseSettings(
        stale_after=2, timeout=2, attribute_delay=0, max_wait=float(sys.argv[2])
    )
    waiter = DirectoryLease("abc", lockfile.parent, lockfile, settings=settings)
    try:
        with waiter:
            outcome = {"error": None}
    except LeaseTimeout as err:
        outcome = {"error": type(err).__name__, "message": str(err)}
    outcome["broke"] = waiter.broke
    print(json.dumps(outcome))
    """)


@pytest.mark.timeout(120)
def test_real_heartbeat_keeps_a_waiter_in_another_process_out(tmp_path):
    """A holder renewed by its own heartbeat thread is never taken over by a waiter
    in another process: the waiter gives up at max_wait, three stale bounds on."""
    lockfile = tmp_path / "abc.lock"
    settings = LeaseSettings(stale_after=2, timeout=2, attribute_delay=0)
    script = tmp_path / "waiter.py"
    script.write_text(WAITER)
    with DirectoryLease("abc", tmp_path, lockfile, settings=settings) as holder:
        waiter = subprocess.run(
            [sys.executable, str(script), str(lockfile), "6"],
            capture_output=True,
            text=True,
            timeout=100,
        )
        assert waiter.returncode == 0, waiter.stderr
        outcome = json.loads(waiter.stdout.strip().splitlines()[-1])
        assert holder.prove()
    assert outcome["error"] == "LeaseTimeout"
    assert "max_wait" in outcome["message"]
    assert outcome["broke"] == []
    assert holder.lost is None


@pytest.mark.timeout(60)
def test_several_waiters_on_one_dead_lease(tmp_path):
    """Of several processes judging one dead lease, one takes it over; each then
    holds the lease in turn."""
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    old = time.time() - 3600
    for path in sorted(lockfile.rglob("*"), reverse=True) + [lockfile]:
        os.utime(path, (old, old))  # the holder record, then its directories
    settings = LeaseSettings(stale_after=10, timeout=60, attribute_delay=0)
    leases = [
        DirectoryLease("abc", tmp_path, lockfile, settings=settings) for _ in range(4)
    ]
    held, start = [], threading.Barrier(len(leases))

    def take(lease):
        start.wait()
        with lease:
            held.append(lease)

    threads = [threading.Thread(target=take, args=(lease,)) for lease in leases]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(50)
    assert len(held) == len(leases)
    assert sum(len(lease.broke) for lease in leases) == 1
    assert not lockfile.exists()


@python.define
def TakenOverMidRun(root: str) -> int:
    """While it runs, another process takes its lease over, reruns it and commits a
    good result, as happens to a holder suspended for longer than stale_after."""
    import os
    import shutil
    from pathlib import Path
    from pydra.engine.lease import DirectoryLease, LeaseSettings
    from pydra.engine.result import Result
    from pydra.engine.store import PickleStore

    root, slot = Path(root), Path(os.getcwd())
    (lock,) = [p for p in root.iterdir() if p.name.endswith(".lock")]
    (token,) = list(lock.iterdir())
    token.rename(lock / "gone-taken")  # the taker claims the lease
    shutil.rmtree(lock)
    os.chdir(root)
    shutil.rmtree(slot)  # the taker's _populate_filesystem
    taker = DirectoryLease(
        "taker", slot, lock, settings=LeaseSettings(attribute_delay=0)
    )
    with taker:
        slot.mkdir()
        (slot / "taker").touch()
        PickleStore(root).commit(taker, Result(cache_dir=slot))
    raise FileNotFoundError("the working directory was removed")


@pytest.mark.timeout(60)
def test_suspended_holder_does_not_overwrite_the_takers_result(tmp_path):
    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        with pytest.raises(Exception) as excinfo:
            sub(TakenOverMidRun(root=str(root)), raise_errors=True)
    assert isinstance(excinfo.value, LeaseLost)
    assert isinstance(excinfo.value.__context__, FileNotFoundError)
    (slot,) = [s for s in PickleStore(root).slots() if (s / "taker").exists()]
    result = cp.loads((slot / "_result.pklz").read_bytes())
    assert not result.errored
    assert not (slot / "_error.pklz").exists()


@pytest.mark.timeout(60)
def test_lease_timeout_through_load_and_run_writes_nothing(tmp_path):
    """On slurm, sge and run_pickled, a job that cannot get its lease raises the typed
    error and leaves the holder's slot alone."""
    settings = LeaseSettings(
        stale_after=10, timeout=10, attribute_delay=0, max_wait=0.2
    )
    sub = Submitter(store=PickleStore(tmp_path / "cache", lease_settings=settings))
    job = Job(task=Inc(x=1), submitter=sub, name="inc")
    job_pkl = tmp_path / "job.pklz"
    job_pkl.write_bytes(cp.dumps(job))
    with sub.store.acquire(job.checksum, job.context):  # a live holder elsewhere
        with pytest.raises(LeaseTimeout):
            load_and_run(job_pkl)
    assert not (job.cache_dir / "_error.pklz").exists()
    assert not (job.cache_dir / "_result.pklz").exists()


def test_load_and_run_records_an_unloadable_job(tmp_path):
    job_pkl = tmp_path / "job.pklz"
    job_pkl.write_bytes(b"not a pickle")
    with pytest.raises(Exception) as excinfo:
        load_and_run(job_pkl)
    assert not isinstance(excinfo.value, TypeError)
    assert cp.loads((tmp_path / "_result.pklz").read_bytes()).errored


def test_slots_skip_leases_and_probes(tmp_path):
    store = PickleStore(tmp_path)
    (tmp_path / f"{storage.PROBE_PREFIX}leftover").mkdir()
    with store.acquire("abc", {}) as lease:
        store.commit(lease, Result(cache_dir=lease.cache_dir))
        assert list(store.slots()) == [lease.cache_dir]
    plant_foreign_lease(tmp_path / "dead-key.lock")
    assert list(store.slots()) == [lease.cache_dir]


def test_async_lease_waits_then_takes_over(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)

    async def take():
        async with DirectoryLease(
            "abc", tmp_path, lockfile, settings=FAST
        ) as lease:  # real storage clock
            return lease.broke

    start = time.monotonic()
    broke = asyncio.run(take())
    assert time.monotonic() - start < 5
    assert broke and broke[0]["holder"]["host"] == FOREIGN_HOST


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"stale_after": 100, "attribute_delay": 90, "timeout": 200}, "live holder"),
        ({"stale_after": 100, "attribute_delay": 0, "timeout": 50}, "dead holder"),
        ({"renew_fraction": 0.6}, "live holder"),
        ({"renew_fraction": 0}, "renew_fraction"),
        ({"stale_after": 0}, "positive"),
    ],
)
def test_settings_enforce_the_safety_inequalities(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LeaseSettings(**kwargs)


def test_default_settings_satisfy_the_inequalities():
    s = LeaseSettings()
    assert (s.stale_after, s.timeout, s.renew_interval, s.attribute_delay) == (
        300,
        600,
        30,
        60,
    )


def test_lease_refused_on_storage_without_atomic_mkdir(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_REQUIRED", {})
    real_mkdir = os.mkdir

    def mkdir_without_exclusion(path, *args, **kwargs):
        try:
            real_mkdir(path, *args, **kwargs)
        except FileExistsError:
            pass  # the second maker succeeds too

    monkeypatch.setattr(os, "mkdir", mkdir_without_exclusion)
    with pytest.raises(StorageUnsupported, match="atomic_mkdir"):
        with PickleStore(tmp_path).acquire("abc", {}):
            pass
    monkeypatch.setattr(os, "mkdir", real_mkdir)
    assert not any(p.name.startswith(storage.PROBE_PREFIX) for p in tmp_path.iterdir())


# Through the submitter


@python.define
def Inc(x: int) -> int:
    return x + 1


def planted_job_paths(root: Path, worker: str, store: PickleStore):
    sub = Submitter(worker=worker, store=store)
    job = Job(task=Inc(x=1), submitter=sub, name="job")
    return sub, job


@pytest.mark.timeout(60)
def test_issue_916_foreign_lock_file_does_not_hang(worker, tmp_path):
    """The reproduction in #916: a lock file with pid 999999 and another host's name at
    the job's lock path. pydra waited forever; it is now taken over after
    stale_after."""
    settings = LeaseSettings(stale_after=1, timeout=1, attribute_delay=0)
    store = PickleStore(tmp_path, lease_settings=settings)
    sub, job = planted_job_paths(tmp_path, worker, store)
    plant_legacy_lock(job.lockfile)
    start = time.monotonic()
    with sub:
        assert sub(Inc(x=1)).outputs.out == 2
    assert time.monotonic() - start < 30
    assert not job.lockfile.exists()


@pytest.mark.timeout(60)
def test_issue_916_old_foreign_lock_file_taken_over_at_once(tmp_path):
    """Under the default settings, a lock left over an hour ago is taken over at
    once."""
    store = PickleStore(tmp_path)
    sub, job = planted_job_paths(tmp_path, "debug", store)
    plant_legacy_lock(job.lockfile)
    old = time.time() - 3600
    os.utime(job.lockfile, (old, old))
    start = time.monotonic()
    with sub:
        assert sub(Inc(x=1)).outputs.out == 2
    assert time.monotonic() - start < 10


@pytest.mark.timeout(60)
def test_save_lock_does_not_hang(worker, tmp_path):
    """#916: a stray ``<slot>_save.lock`` from another host made save() wait forever.
    save() no longer takes a lock."""
    store = PickleStore(tmp_path, lease_settings=FAST)
    sub, job = planted_job_paths(tmp_path, worker, store)
    save_lock = job.cache_dir.parent / (job.cache_dir.name + "_save.lock")
    plant_legacy_lock(save_lock)
    with sub:
        assert sub(Inc(x=1)).outputs.out == 2


def test_clean_stale_locks_is_deprecated(tmp_path):
    with pytest.warns(DeprecationWarning, match="clean_stale_locks"):
        sub = Submitter(cache_root=tmp_path, clean_stale_locks=True)
    assert sub.clean_stale_locks is False


# Across processes

SCRIPT = textwrap.dedent("""
    import os, sys, time
    from pathlib import Path
    from pydra.compose import python
    from pydra.engine.lease import LeaseSettings
    from pydra.engine.store import PickleStore
    from pydra.engine.submitter import Submitter

    root, log, go = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

    @python.define
    def Counted(log: str) -> int:
        import os, time
        from pathlib import Path
        with open(log, "a") as f:
            f.write(f"{os.getpid()}\\n")
        Path(log + ".started").touch()
        time.sleep(float(Path(log + ".hold").read_text()))
        return 7

    if __name__ == "__main__":
        deadline = time.monotonic() + 60
        while not go.exists() and time.monotonic() < deadline:
            time.sleep(0.005)
        settings = LeaseSettings(stale_after=1, timeout=30, attribute_delay=0)
        store = PickleStore(root, lease_settings=settings)
        with Submitter(worker="debug", store=store) as sub:
            print(sub(Counted(log=log)).outputs.out)
    """)


def run_script(tmp_path: Path, hold: float, go: Path) -> subprocess.Popen:
    script = tmp_path / "counted.py"
    script.write_text(SCRIPT)
    (tmp_path / "runs.log.hold").write_text(str(hold))
    return subprocess.Popen(
        [
            sys.executable,
            str(script),
            str(tmp_path / "cache"),
            str(tmp_path / "runs.log"),
            str(go),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.timeout(120)
def test_two_processes_racing_on_one_checksum_run_the_job_once(tmp_path):
    go = tmp_path / "go"
    procs = [run_script(tmp_path, 0.5, go) for _ in range(2)]
    go.touch()
    outs = [p.communicate(timeout=100) for p in procs]
    for proc, (out, err) in zip(procs, outs):
        assert proc.returncode == 0, err
        assert out.strip() == "7"
    assert len((tmp_path / "runs.log").read_text().splitlines()) == 1


@pytest.mark.timeout(120)
def test_rerun_after_sigkill_takes_over(tmp_path):
    go = tmp_path / "go"
    go.touch()
    killed = run_script(tmp_path, 600, go)
    started = tmp_path / "runs.log.started"
    deadline = time.monotonic() + 60
    while not started.exists():
        assert time.monotonic() < deadline, killed.stderr.read()
        time.sleep(0.01)
    killed.kill()  # SIGKILL, or TerminateProcess on Windows
    killed.communicate(timeout=30)
    assert any(p.suffix == ".lock" for p in (tmp_path / "cache").iterdir())
    started.unlink()

    rerun = run_script(tmp_path, 0, go)
    out, err = rerun.communicate(timeout=100)
    assert rerun.returncode == 0, err
    assert out.strip() == "7"
    assert "Took over the lease" in err
    assert len((tmp_path / "runs.log").read_text().splitlines()) == 2
    assert not any(p.suffix == ".lock" for p in (tmp_path / "cache").iterdir())
