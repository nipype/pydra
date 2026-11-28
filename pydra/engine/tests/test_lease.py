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
from pydra.engine import lease as lease_module
import cloudpickle as cp
from pydra.engine.lease import (
    HOLDER_FILE,
    MISSED_BEATS,
    DirectoryLease,
    Lease,
    LeaseLost,
    LeaseRefused,
    LeaseSettings,
    LeaseTimeout,
    LeaseUnavailable,
    release_patience,
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
FAST = LeaseSettings(stale_after=0.5, timeout=10, attribute_delay=0)


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
        stale_after=2, timeout=10, attribute_delay=0, max_wait=float(sys.argv[2])
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
    settings = LeaseSettings(stale_after=2, timeout=10, attribute_delay=0)
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


def test_second_rename_of_a_claimed_token_does_not_count_twice(tmp_path, monkeypatch):
    """On Windows a rename opens its source and renames the open handle, so a second
    taker whose rename started before the first finished moves the directory the
    first has just claimed, and both renames succeed. Exactly one of them takes the
    lease over, and the lease is left removable at once."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    planted = newest_mtime(lockfile)
    clock = Clock(planted + 11)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    first, second = (
        DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
        for _ in range(2)
    )
    for lease in (first, second):
        lease._prepare()
    real_rename = os.rename
    calls, outcomes = [], []

    def handle_rename(src, dst, *args, **kwargs):
        src, dst = Path(src), Path(dst)
        if src != token_dir:
            return real_rename(src, dst, *args, **kwargs)
        calls.append(dst)
        if len(calls) == 1:
            # the second taker, which listed the lease before either moved it,
            # opens the token directory; meanwhile the first takes it over
            outcomes.append(first._judge())
            return  # its handle rename moved the directory, below, and succeeded
        real_rename(src, dst)  # the first taker's rename
        real_rename(dst, calls[0])  # the second's open handle moves it again

    monkeypatch.setattr(os, "rename", handle_rename)
    outcomes.append(second._judge())
    monkeypatch.setattr(os, "rename", real_rename)
    assert len(calls) == 2  # both renames succeeded
    assert len(first.broke) + len(second.broke) == 1
    assert outcomes.count(True) >= 1
    assert not lockfile.exists()
    with DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock):
        pass


def two_takers(tmp_path):
    """A dead lease and two waiters judging it on an injected clock."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    clock = Clock(newest_mtime(lockfile) + 11)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    takers = [
        DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
        for _ in range(2)
    ]
    for lease in takers:
        lease._prepare()
    return lockfile, token_dir, clock, settings, takers


def test_first_claim_before_second_rename_leaves_no_remains(tmp_path, monkeypatch):
    """The other order of the Windows handle rename: the first taker claims, then the
    second's rename moves the claimed directory to its own name. The first removes
    that remains too, so the lease is free at once, without another stale_after."""
    lockfile, token_dir, clock, settings, (first, second) = two_takers(tmp_path)
    real_rename, real_mkdir = os.rename, os.mkdir
    calls, outcomes = [], []

    def handle_rename(src, dst, *args, **kwargs):
        src, dst = Path(src), Path(dst)
        if src != token_dir:
            return real_rename(src, dst, *args, **kwargs)
        calls.append(dst)
        if len(calls) == 1:  # the second opens the token directory, then waits
            outcomes.append(first._judge())
            return
        real_rename(src, dst)  # the first taker's rename

    def claim_then_moved(path, *args, **kwargs):
        real_mkdir(path, *args, **kwargs)
        path = Path(path)
        if path.name.startswith("claimed-") and path.parent == calls[-1]:
            real_rename(path.parent, calls[0])  # the second's handle rename lands

    monkeypatch.setattr(os, "rename", handle_rename)
    monkeypatch.setattr(os, "mkdir", claim_then_moved)
    outcomes.append(second._judge())
    monkeypatch.undo()
    assert outcomes == [True, False]
    assert len(first.broke) == 1 and second.broke == []
    assert not lockfile.exists()  # the gone- remains were removed
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert lease._try_take()  # free at once
    lease._release()


def test_claim_after_the_marker_is_removed_loses(tmp_path, monkeypatch):
    """The claimant removes its marker only after the directory's other contents,
    so a second handle rename and claim landing after the marker is gone finds the
    directory emptied and loses: one takeover."""
    lockfile, token_dir, clock, settings, (first, second) = two_takers(tmp_path)
    real_rename, real_rmdir = os.rename, os.rmdir
    second_dst, results = [], {}

    def first_rename_only(src, dst, *args, **kwargs):
        return real_rename(src, dst, *args, **kwargs)

    def rmdir_then_second_claims(path, *args, **kwargs):
        real_rmdir(path, *args, **kwargs)
        path = Path(path)
        if path.name.startswith("claimed-") and not second_dst:
            # the second's open handle moves the directory, then it claims
            second_dst.append(lockfile / "gone-second")
            real_rename(path.parent, second_dst[0])
            results["second"] = second._claim(second_dst[0], path.name, True)

    monkeypatch.setattr(os, "rename", first_rename_only)
    monkeypatch.setattr(os, "rmdir", rmdir_then_second_claims)
    assert first._judge()
    monkeypatch.undo()
    assert results == {"second": False}
    assert len(first.broke) == 1 and second.broke == []
    assert not lockfile.exists()


def test_claim_that_cannot_be_made_raises(tmp_path, monkeypatch):
    """A claim marker refused for another reason than losing (the storage is full)
    raises at once instead of waiting without end."""
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    clock = FakeTime(newest_mtime(lockfile) + 11)
    real_mkdir = os.mkdir

    def full(path, *args, **kwargs):
        if Path(path).name.startswith("claimed-"):
            raise OSError(errno.ENOSPC, "No space left on device", str(path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", full)
    settings = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, max_wait=60)
    with pytest.raises(OSError) as excinfo:
        clock.lease(tmp_path, lockfile, settings).__enter__()
    assert excinfo.value.errno == errno.ENOSPC
    assert clock.t == 0  # on the first attempt


def test_token_mkdir_racing_an_rmdir_retries(tmp_path, monkeypatch):
    """APFS answers EINVAL to a mkdir in a directory being removed; the take retries."""
    lockfile = tmp_path / "abc.lock"
    real_mkdir = os.mkdir
    refused = []

    def racing(path, *args, **kwargs):
        if Path(path).parent == lockfile and not refused:
            refused.append(path)
            raise OSError(errno.EINVAL, "Invalid argument", str(path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", racing)
    clock = FakeTime(time.time())
    lease = clock.lease(tmp_path, lockfile, SLOW)
    lease._prepare()
    assert not lease._try_take()
    os.rmdir(lockfile)  # as the racing rmdir left it
    with lease:
        assert lease.held
    assert refused


def test_several_waiters_on_one_dead_lease(tmp_path, monkeypatch):
    """Several waiters all judge one dead lease stale before any of them moves it:
    one takes it over, and each then holds the lease in turn."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    clock = Clock(newest_mtime(lockfile) + 11)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    leases = [
        DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
        for _ in range(4)
    ]
    for lease in leases:
        lease._prepare()
    real_rename = os.rename
    started, outcomes = [0], {}

    def next_judges_first(src, dst, *args, **kwargs):
        # each waiter's rename waits for the next waiter to judge and rename, so
        # all four list the dead lease before any of them moves it
        if Path(src) == token_dir and started[0] < len(leases) - 1:
            started[0] += 1
            index = started[0]
            outcomes[index] = leases[index]._judge()
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", next_judges_first)
    outcomes[0] = leases[0]._judge()
    monkeypatch.undo()
    assert [outcomes[i] for i in range(4)] == [False, False, False, True]
    assert sum(len(lease.broke) for lease in leases) == 1
    assert not lockfile.exists()
    clock.now = time.time()  # the new holders' leases are fresh
    for lease in leases:
        with lease:
            assert lease.held
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
    with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
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
    clock = Clock(newest_mtime(lockfile) + 11)

    async def take():
        async with DirectoryLease(
            "abc", tmp_path, lockfile, settings=SLOW, clock=clock
        ) as lease:
            return lease.broke

    broke = asyncio.run(take())
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


def test_poll_settings_set_the_waits_behind_a_holder(tmp_path):
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    waits = []

    def holder_beats(clock):
        waits.append(clock.t - sum(waits))
        os.utime(token_dir, (clock.storage(), clock.storage()))

    clock = FakeTime(newest_mtime(lockfile), on_sleep=holder_beats)
    settings = LeaseSettings(
        stale_after=10,
        timeout=15,
        attribute_delay=0,
        max_wait=12,
        poll_min=1,
        poll_max=4,
    )
    with pytest.raises(LeaseTimeout):
        clock.lease(tmp_path, lockfile, settings).__enter__()
    assert waits[:4] == [1, 2, 4, 4]


def test_missed_beats_enters_the_live_holder_inequality():
    LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, missed_beats=8)
    with pytest.raises(ValueError, match="live holder"):
        LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, missed_beats=9)


def test_visibility_timeout_covers_the_attribute_delay():
    s = LeaseSettings(attribute_delay=30, stale_after=300, poll_max=3)
    assert s.visibility_timeout == 33


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
    settings = LeaseSettings(stale_after=1, timeout=20, attribute_delay=0)
    store = PickleStore(tmp_path, lease_settings=settings)
    sub, job = planted_job_paths(tmp_path, worker, store)
    plant_legacy_lock(job.lockfile)
    with sub:
        assert sub(Inc(x=1)).outputs.out == 2
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
    with sub:
        assert sub(Inc(x=1)).outputs.out == 2
    assert not job.lockfile.exists()


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


# try_take: one attempt at a time, for a caller that must not block


def test_try_take_returns_none_when_free(tmp_path):
    lease = DirectoryLease("abc", tmp_path, tmp_path / "abc.lock", settings=FAST)
    assert lease.try_take() is None
    assert lease.held
    lease.__exit__(None, None, None)
    assert not lease.lockfile.exists()


def test_try_take_returns_the_delay_while_held_then_takes_it(tmp_path):
    lockfile = tmp_path / "abc.lock"
    holder = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)
    waiter = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)
    with holder:
        first = waiter.try_take()
        assert first is not None and first > 0
        second = waiter.try_take()
        assert second is not None and second >= first  # backs off between attempts
        assert not waiter.held
    assert waiter.try_take() is None
    assert waiter.held
    waiter.__exit__(None, None, None)


def test_try_take_keeps_its_bounds_between_attempts(tmp_path):
    """The first call starts the wait's clock; later calls do not restart it, so the
    wait ends in ``LeaseTimeout`` however often it is retried"""
    lockfile = tmp_path / "abc.lock"
    settings = LeaseSettings(
        stale_after=10, timeout=15, attribute_delay=0, max_wait=0.2
    )
    holder = DirectoryLease("abc", tmp_path, lockfile, settings=settings)
    clock = FakeTime(0.0)  # the wait's monotonic clock, moved past max_wait by hand
    waiter = DirectoryLease(
        "abc", tmp_path, lockfile, settings=settings, monotonic=clock.monotonic
    )
    with holder:
        assert waiter.try_take() is not None
        clock.t += 0.3
        with pytest.raises(LeaseTimeout):
            waiter.try_take()
        assert not waiter.held
    # a new wait starts afresh
    assert waiter.try_take() is None
    waiter.__exit__(None, None, None)


def test_try_take_takes_over_a_stale_lease(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    settings = LeaseSettings(stale_after=0.3, timeout=10, attribute_delay=0)
    waiter = DirectoryLease("abc", tmp_path, lockfile, settings=settings)
    deadline = time.monotonic() + 60
    while (delay := waiter.try_take()) is not None:
        assert time.monotonic() < deadline
        time.sleep(delay)
    assert waiter.held
    waiter.__exit__(None, None, None)


def test_try_take_is_part_of_the_lease_protocol():
    assert "try_take" in dir(Lease)


def sharing_violation(path) -> PermissionError:
    """What Windows raises for a path another process has open: a sharing violation
    or access denied, both errno EACCES."""
    return PermissionError(
        errno.EACCES, "The process cannot access the file", str(path)
    )


def test_release_retries_a_rename_refused_while_a_waiter_reads(
    tmp_path, monkeypatch, caplog
):
    """On Windows a directory cannot be renamed while another process has it or a
    file in it open, as a waiter judging the lease does. Release retries the rename
    instead of reporting the lease taken over and leaving it behind."""
    lockfile = tmp_path / "abc.lock"
    real_rename = os.rename
    refused = []

    def rename(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("gone-") and len(refused) < 3:
            refused.append(src)
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    lease = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)  # patience 1 s
    with lease:
        monkeypatch.setattr(os, "rename", rename)
    assert len(refused) == 3
    assert lease.lost is None
    assert "taken over" not in caplog.text
    assert not lockfile.exists()


def test_release_refused_for_good_is_not_reported_as_taken_over(
    tmp_path, monkeypatch, caplog
):
    """A release whose rename is refused for good leaves a lease that ages and is taken
    over once stale; its holder was not taken over and is not reported so."""
    lockfile = tmp_path / "abc.lock"
    real_rename = os.rename

    def rename(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("gone-"):
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    clock = FakeTime(time.time())
    lease = clock.lease(tmp_path, lockfile, FAST)
    with lease:
        monkeypatch.setattr(os, "rename", rename)
    assert lease.lost is None
    assert "taken over while held" not in caplog.text
    assert "could not be released" in caplog.text


@pytest.mark.timeout(60)
def test_waiter_takes_a_lease_released_while_it_reads_the_holder(
    tmp_path, monkeypatch, caplog
):
    """Windows semantics: a directory cannot be renamed while a file in it is open. The
    holder releases while a waiter has the holder record open; the waiter then takes
    the lease at once, and the holder is not reported taken over (the failure of
    test_concurrent_submissions_run_a_job_once on Windows, where the waiter instead
    waited stale_after for a lease left behind)."""
    lockfile = tmp_path / "abc.lock"
    real_rename, real_read_text = os.rename, Path.read_text
    open_in = set()
    reading, refused = threading.Event(), threading.Event()

    def read_text(self, *args, **kwargs):
        if self.name == HOLDER_FILE and not refused.is_set():
            open_in.add(self.parent)
            reading.set()
            refused.wait(60)  # held open until the holder's rename was refused
            open_in.discard(self.parent)
        return real_read_text(self, *args, **kwargs)

    def rename(src, dst, *args, **kwargs):
        if Path(src) in open_in:
            refused.set()
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(os, "rename", rename)
    holder = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)
    holder.__enter__()
    took = []
    waiting = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)

    def wait():
        with waiting:
            took.append(True)

    waiter = threading.Thread(target=wait)
    waiter.start()
    assert reading.wait(60)
    holder.__exit__(None, None, None)
    waiter.join(60)
    assert refused.is_set()
    assert holder.lost is None
    assert "taken over while held" not in caplog.text
    # the holder's retried release went through, so the waiter took a released lease
    # rather than taking over one left behind once it was stale
    assert "could not be released" not in caplog.text
    assert took and waiting.broke == []


def test_released_remains_do_not_hold_the_lease(tmp_path):
    """What a release could not remove (on Windows a file a waiter had open) is no
    holder: the next taker removes it and takes the lease without waiting."""
    lockfile = tmp_path / "abc.lock"
    remains = lockfile / ("gone-" + "e" * 32)
    remains.mkdir(parents=True)
    (remains / HOLDER_FILE).write_text(json.dumps({"pid": DEAD_PID}))
    clock = FakeTime(newest_mtime(lockfile))
    lease = clock.lease(tmp_path, lockfile, SLOW)
    with lease:
        assert lease.held
        assert clock.t == 0
        assert not remains.exists()
        assert lease.broke == []
    assert not lockfile.exists()


def test_stall_between_the_marks_and_the_clock_is_not_the_holders_age(
    tmp_path, monkeypatch
):
    """A waiter stalled after it listed the marks is not told the holder is older for
    it: the storage clock is read before the marks, so a stall only under-estimates
    the age of a live holder."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    now = time.time()
    for path in [lockfile, token_dir, *token_dir.iterdir()]:
        os.utime(path, (now, now))  # the holder just renewed
    storage = Clock(now)
    newest_mtime = DirectoryLease._newest_mtime

    def stalled(self, entries):
        marks = newest_mtime(self, entries)
        storage.now += 3 * SLOW.stale_after  # the waiter is descheduled here
        return marks

    monkeypatch.setattr(DirectoryLease, "_newest_mtime", stalled)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW, clock=storage)
    assert not lease._judge()
    assert token_dir.is_dir()
    assert lease.broke == []


def test_beat_that_fails_is_logged(tmp_path, monkeypatch, caplog):
    """A heartbeat whose ``mkdir`` fails but is not yet lasting is retried, and says so
    in the debug log, so a failing renewal can be told from a stalled one."""
    lockfile = tmp_path / "abc.lock"
    real_mkdir = os.mkdir

    def mkdir(path, *args, **kwargs):
        if Path(path).name == "beat-1":
            raise sharing_violation(path)
        return real_mkdir(path, *args, **kwargs)

    caplog.set_level("DEBUG", logger="pydra")
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    with lease:
        monkeypatch.setattr(os, "mkdir", mkdir)
        deadline = time.monotonic() + 60
        while "Could not renew" not in caplog.text and time.monotonic() < deadline:
            time.sleep(0.01)
        monkeypatch.setattr(os, "mkdir", real_mkdir)
    assert "Could not renew" in caplog.text


def test_unreadable_heartbeat_is_not_taken_for_a_stale_lease(tmp_path, monkeypatch):
    """A holder's heartbeat marks that cannot be read for a moment (on Windows, open in
    another process) are not absent: the lease is not judged by the older marks left."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    now = time.time()
    os.utime(lockfile, (now - 100, now - 100))  # only the holder's marks are fresh
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        if Path(path) == token_dir or Path(path).parent == token_dir:
            raise sharing_violation(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW, clock=lambda: now)
    assert not lease._judge()
    assert token_dir.is_dir()
    assert lease.broke == []


@pytest.fixture
def unreadable_mark(monkeypatch):
    """Make ``lstat`` of a path always refused, as for a mark in a foreign token
    directory without search permission; returns a setter for that path."""
    real_lstat = os.lstat
    refused = []

    def lstat(path, *args, **kwargs):
        if refused and Path(path) == refused[0]:
            raise sharing_violation(path)
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", lstat)
    return refused.append


def test_live_holder_with_an_unreadable_mark_is_waited_for(tmp_path, unreadable_mark):
    """A mark that cannot ever be read does not stop the holder's renewals from keeping
    the wait alive: the waiter waits until max_wait, it does not give up as if the
    lease could not be removed."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)

    def holder_beats(clock):
        os.utime(token_dir, (clock.storage(), clock.storage()))

    clock = FakeTime(newest_mtime(lockfile), on_sleep=holder_beats)
    unreadable_mark(token_dir / HOLDER_FILE)
    settings = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, max_wait=50)
    with pytest.raises(LeaseTimeout) as excinfo:
        clock.lease(tmp_path, lockfile, settings).__enter__()
    assert type(excinfo.value) is LeaseTimeout
    assert "max_wait" in str(excinfo.value)
    assert 50 <= clock.t < 52
    assert token_dir.is_dir()


def test_dead_holder_with_an_unreadable_mark_is_taken_over(tmp_path, unreadable_mark):
    """A dead holder whose mark cannot ever be read is taken over on the storage clock
    once its readable marks have been stale for stale_after since the unreadable one
    was first seen, within the waiter's timeout."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    clock = FakeTime(newest_mtime(lockfile))
    unreadable_mark(token_dir / HOLDER_FILE)
    lease = clock.lease(tmp_path, lockfile, SLOW)
    with lease:
        assert lease.held
        assert 10 <= clock.t < 15
        assert len(lease.broke) == 1


def test_try_take_waits_for_a_live_holder_with_an_unreadable_mark(
    tmp_path, unreadable_mark
):
    """A wait driven by try_take keeps what it saw between calls: a renewing holder
    with a mark that cannot be read keeps the wait alive until max_wait, and is not
    taken over"""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)

    def holder_beats(clock):
        os.utime(token_dir, (clock.storage(), clock.storage()))

    clock = FakeTime(newest_mtime(lockfile), on_sleep=holder_beats)
    unreadable_mark(token_dir / HOLDER_FILE)
    settings = LeaseSettings(stale_after=10, timeout=15, attribute_delay=0, max_wait=50)
    waiter = clock.lease(tmp_path, lockfile, settings)
    with pytest.raises(LeaseTimeout) as excinfo:
        while (delay := waiter.try_take()) is not None:
            clock.sleep(delay)
    assert type(excinfo.value) is LeaseTimeout
    assert "max_wait" in str(excinfo.value)
    assert 50 <= clock.t < 52
    assert not waiter.held
    assert waiter.broke == []
    assert token_dir.is_dir()


def test_try_take_takes_over_a_dead_holder_with_an_unreadable_mark(
    tmp_path, unreadable_mark
):
    """try_take takes over a dead holder whose mark cannot be read once stale_after has
    passed on the storage clock since the first unreadable sighting, not before"""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    clock = FakeTime(newest_mtime(lockfile))
    unreadable_mark(token_dir / HOLDER_FILE)
    waiter = clock.lease(tmp_path, lockfile, SLOW)
    while (delay := waiter.try_take()) is not None:
        assert waiter.broke == []
        clock.sleep(delay)
    assert waiter.held
    assert 10 <= clock.t < 15
    assert len(waiter.broke) == 1
    waiter.__exit__(None, None, None)
    assert not lockfile.exists()


def test_try_take_lease_with_a_refused_release(tmp_path, monkeypatch, caplog):
    """A lease taken with try_take retries a refused release within its patience; one
    refused for good is not reported taken over, and a waiter using try_take takes
    it over only once stale"""
    lockfile = tmp_path / "abc.lock"
    real_rename = os.rename
    refusing = [3]

    def rename(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("gone-") and refusing[0] > 0:
            refusing[0] -= 1
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", rename)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=SLOW)  # patience 1 s
    assert lease.try_take() is None
    lease.__exit__(None, None, None)
    assert refusing == [0]
    assert lease.lost is None
    assert not lockfile.exists()

    clock = FakeTime(time.time())
    holder = clock.lease(tmp_path, lockfile, FAST)
    assert holder.try_take() is None
    refusing[0] = 10**9  # refused for good
    holder.__exit__(None, None, None)
    assert holder.lost is None
    assert "taken over while held" not in caplog.text
    assert "could not be released" in caplog.text
    assert lockfile.is_dir()

    refusing[0] = 0
    later = FakeTime(newest_mtime(lockfile))
    waiter = later.lease(tmp_path, lockfile, FAST)
    while (delay := waiter.try_take()) is not None:
        assert waiter.broke == []
        later.sleep(delay)
    assert later.t > FAST.stale_after
    assert len(waiter.broke) == 1
    waiter.__exit__(None, None, None)
    assert not lockfile.exists()


def test_stale_holder_beside_remains_is_taken_over(tmp_path):
    """Remains beside a holder's token do not make the lease removable at once: the
    holder is judged and taken over once stale."""
    lockfile = tmp_path / "abc.lock"
    token_dir = plant_foreign_lease(lockfile)
    (lockfile / ("gone-" + "e" * 32)).mkdir()
    clock = FakeTime(newest_mtime(lockfile))
    lease = clock.lease(tmp_path, lockfile, SLOW)
    with lease:
        assert 10 <= clock.t < 12
        assert len(lease.broke) == 1
        assert not token_dir.exists()


def test_takeover_found_only_at_release_marks_the_lease_lost(tmp_path, caplog):
    """A lease taken over while held, not found by a heartbeat or `prove`, is found at
    release: its directory is gone, which is not retried."""
    lockfile = tmp_path / "abc.lock"
    clock = FakeTime(time.time())
    lease = clock.lease(tmp_path, lockfile, SLOW)
    with lease:
        token = lease._token
        os.rename(lockfile / token, lockfile / ("gone-" + "d" * 32))
    assert lease.lost == "its holder directory was gone at release"
    assert "taken over while held" in caplog.text
    assert clock.t == 0


@pytest.mark.timeout(60)
def test_short_stale_after_release_refused_is_not_taken_over_mid_release(
    tmp_path, monkeypatch, caplog
):
    """With a short stale_after a refused release gives up before a waiter could see
    the stopped holder stale, so the holder is not taken over during its release."""
    lockfile = tmp_path / "abc.lock"
    real_rename = os.rename
    holder_thread = threading.current_thread()

    def rename(src, dst, *args, **kwargs):
        if threading.current_thread() is holder_thread and os.path.exists(src):
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    holder = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    holder.__enter__()
    took = []
    polled = threading.Event()

    def poll(seconds):
        polled.set()
        time.sleep(seconds)

    def wait():
        with DirectoryLease("abc", tmp_path, lockfile, settings=FAST, sleep=poll):
            took.append(True)

    waiter = threading.Thread(target=wait)
    waiter.start()
    assert polled.wait(60)  # the waiter found the lease held and is waiting
    monkeypatch.setattr(os, "rename", rename)
    holder.__exit__(None, None, None)
    waiter.join(30)
    assert took
    assert holder.lost is None
    assert "taken over while held" not in caplog.text


def test_async_release_does_not_block_the_event_loop(tmp_path, monkeypatch):
    """An async release retries a refused rename with asyncio.sleep, never the
    blocking sleep."""
    lockfile = tmp_path / "abc.lock"
    real_rename = os.rename
    refused = []

    def rename(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("gone-") and len(refused) < 2:
            refused.append(src)
            raise sharing_violation(src)
        return real_rename(src, dst, *args, **kwargs)

    def blocking_sleep(seconds):
        raise AssertionError("blocking sleep in an async release")

    lease = DirectoryLease(
        "abc", tmp_path, lockfile, settings=SLOW, sleep=blocking_sleep
    )

    async def run():
        async with lease:
            monkeypatch.setattr(os, "rename", rename)

    asyncio.run(run())
    assert len(refused) == 2
    assert lease.lost is None
    assert not lockfile.exists()


def delete_pending(path) -> PermissionError:
    """What Windows raises for a mkdir on, or a listing of, a name whose directory was
    removed while a handle to it was still open: ERROR_ACCESS_DENIED, errno EACCES."""
    return PermissionError(errno.EACCES, "Access is denied", str(path), 5)


@pytest.fixture
def refused(monkeypatch):
    """Patch ``os.mkdir`` and ``os.listdir`` to refuse a path as a delete-pending name as
    many times as counted in ``refused[call][path]`` (-1: always)."""
    real = {"mkdir": os.mkdir, "listdir": os.listdir}
    counts = {"mkdir": {}, "listdir": {}}

    def refusing(name):
        def call(path, *args, **kwargs):
            left = counts[name].get(Path(path), 0)
            if left:
                counts[name][Path(path)] = left - 1
                raise delete_pending(path)
            return real[name](path, *args, **kwargs)

        return call

    for name in real:
        monkeypatch.setattr(os, name, refusing(name))
    return counts


@pytest.mark.parametrize("windows", [True, False])
def test_try_take_on_a_name_pending_deletion(tmp_path, monkeypatch, refused, windows):
    """On Windows a mkdir on a lease a holder has just removed is refused while a
    handle to the removed directory is open. The waiter retries; elsewhere the
    refusal is a real permission error and raises."""
    lockfile = tmp_path / "abc.lock"
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    refused["mkdir"][lockfile] = 2  # the first attempt and its retry after judging
    monkeypatch.setattr(lease_module, "IS_WINDOWS", windows)
    if not windows:
        with pytest.raises(PermissionError):
            lease.try_take()
        return
    delay = lease.try_take()
    assert delay is not None and delay > 0
    assert refused["mkdir"][lockfile] == 0
    assert lease.try_take() is None
    lease.__exit__(None, None, None)
    assert not lockfile.exists()


def test_try_take_on_a_name_pending_deletion_still_listed(
    tmp_path, monkeypatch, refused
):
    """The name of a directory pending deletion stays in its parent until the last
    handle closes: the mkdir is refused, and so is listing it while judging. The
    waiter retries, and takes the lease once the name is gone."""
    lockfile = tmp_path / "abc.lock"
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    lockfile.mkdir()
    refused["mkdir"][lockfile] = 1
    refused["listdir"][lockfile] = 2
    monkeypatch.setattr(lease_module, "IS_WINDOWS", True)
    delay = lease.try_take()
    assert delay is not None and delay > 0
    assert refused["mkdir"][lockfile] == 0 and refused["listdir"][lockfile] == 1
    assert lease.holder() is None
    assert refused["listdir"][lockfile] == 0
    lockfile.rmdir()  # the last handle closed
    assert lease.try_take() is None
    lease.__exit__(None, None, None)
    assert not lockfile.exists()


def test_try_take_when_a_releaser_removes_the_new_lease(tmp_path, monkeypatch):
    """A releaser's trailing rmdir can remove the empty lease this process has just
    made, before its token mkdir; on Windows that mkdir is then refused, not ENOENT,
    and the taker tries again."""
    lockfile = tmp_path / "abc.lock"
    real_mkdir = os.mkdir
    refusals = []

    def mkdir(path, *args, **kwargs):
        if Path(path).parent == lockfile and not refusals:
            refusals.append(path)
            os.rmdir(lockfile)  # the releaser's rmdir
            raise delete_pending(path)
        return real_mkdir(path, *args, **kwargs)

    lease = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(lease_module, "IS_WINDOWS", True)
    assert lease.try_take() is None  # the retry after judging the lease gone
    assert len(refusals) == 1
    assert isinstance(lease._refused, PermissionError)
    lease.__exit__(None, None, None)
    assert not lockfile.exists()


@pytest.mark.parametrize("listed", [False, True])
def test_refused_for_good_on_windows_raises_lease_refused(
    tmp_path, monkeypatch, refused, listed
):
    """A refusal that outlasts the release patience is a permission problem, not a
    name pending deletion: the wait ends in LeaseRefused, which is not a LeaseTimeout
    (a pool retries those), instead of waiting out the timeout."""
    lockfile = tmp_path / "abc.lock"
    if listed:
        lockfile.mkdir()
        refused["listdir"][lockfile] = -1
    refused["mkdir"][lockfile] = -1
    monkeypatch.setattr(lease_module, "IS_WINDOWS", True)
    clock = FakeTime(time.time())
    with pytest.raises(LeaseRefused) as excinfo:
        clock.lease(tmp_path, lockfile, SLOW).__enter__()
    err = excinfo.value
    assert not isinstance(err, LeaseTimeout) and isinstance(err, PermissionError)
    assert isinstance(err.error, PermissionError) and err.lockfile == lockfile
    assert release_patience(SLOW) <= clock.t < SLOW.timeout


def test_max_wait_before_the_patience_names_the_refusal(tmp_path, monkeypatch, refused):
    """A ``max_wait`` shorter than the release patience ends the wait in LeaseTimeout
    before LeaseRefused could; the timeout names the refusal and is caused by it."""
    lockfile = tmp_path / "abc.lock"
    refused["mkdir"][lockfile] = -1
    monkeypatch.setattr(lease_module, "IS_WINDOWS", True)
    settings = LeaseSettings(
        stale_after=10, timeout=15, attribute_delay=0, max_wait=0.5
    )
    assert settings.max_wait < release_patience(settings)
    clock = FakeTime(time.time())
    with pytest.raises(LeaseTimeout) as excinfo:
        clock.lease(tmp_path, lockfile, settings).__enter__()
    assert not isinstance(excinfo.value, LeaseRefused)
    assert isinstance(excinfo.value.__cause__, PermissionError)
    assert "refusing" in str(excinfo.value)
