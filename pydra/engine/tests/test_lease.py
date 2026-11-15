import asyncio
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
import pytest
from pydra.compose import python
from pydra.engine.lease import (
    HOLDER_FILE,
    DirectoryLease,
    Lease,
    LeaseSettings,
    LeaseTimeout,
)
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
    return max(p.lstat().st_mtime for p in [lockfile, *lockfile.iterdir()])


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


def test_live_holder_is_never_taken_over(tmp_path):
    lockfile = tmp_path / "abc.lock"
    with DirectoryLease("abc", tmp_path, lockfile, settings=FAST) as held:
        waiter = DirectoryLease(
            "abc",
            tmp_path,
            lockfile,
            settings=LeaseSettings(stale_after=0.5, timeout=1.5, attribute_delay=0),
        )
        start = time.monotonic()
        with pytest.raises(LeaseTimeout) as excinfo:
            waiter.__enter__()
        assert time.monotonic() - start >= 1.5  # three stale bounds passed
        assert held.held and held.lost is None
        assert waiter.broke == []
    err = excinfo.value
    assert err.lockfile == lockfile
    assert err.holder["host"] == socket.gethostname()
    assert err.holder["pid"] == os.getpid()
    assert 0 <= err.age < 0.5
    assert str(lockfile) in str(err) and socket.gethostname() in str(err)


def test_timeout_names_foreign_holder(tmp_path):
    lockfile = tmp_path / "abc.lock"
    plant_foreign_lease(lockfile)
    clock = Clock(newest_mtime(lockfile) + 3)
    settings = LeaseSettings(stale_after=10, timeout=10, attribute_delay=0)
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=settings, clock=clock)
    lease._prepare()
    assert not lease._attempt()
    err = lease._timeout()
    assert isinstance(err, LeaseTimeout)
    assert err.holder["host"] == FOREIGN_HOST and err.holder["pid"] == DEAD_PID
    assert err.age == pytest.approx(3)
    assert f"pid {DEAD_PID} on host {FOREIGN_HOST}" in str(err)


def test_lease_taken_over_while_held_is_marked_lost(tmp_path):
    lockfile = tmp_path / "abc.lock"
    lease = DirectoryLease("abc", tmp_path, lockfile, settings=FAST)
    with lease:
        (token_dir,) = lockfile.iterdir()
        token_dir.rename(lockfile / "gone-taken")
        deadline = time.monotonic() + 5
        while lease.lost is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert lease.lost is not None
        assert not lease.held
    assert lockfile.exists()  # left to whoever took it over


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
    os.kill(killed.pid, signal.SIGKILL)
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
