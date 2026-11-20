"""The named lease and its cross-process beater (RFC 0005 §5.3, L0)."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
import pytest
from pydra.engine.lease import (
    BeaterClient,
    BeaterDead,
    DirectoryLease,
    LeaseBeater,
    LeaseSettings,
)

# Beats every 0.05 s, stale after 0.5 s: in-process tests
FAST = LeaseSettings(stale_after=0.5, timeout=10, attribute_delay=0)

# Beats every 0.2 s, stale after 2 s: an acknowledgement is due within 0.2 s, which
# a loaded CI runner meets where 0.05 s is tight
CROSS = LeaseSettings(stale_after=2, timeout=10, attribute_delay=0)

CALLBACKS = textwrap.dedent("""\
    import json, time
    from pathlib import Path


    def record_expiry(token_dir, payload):
        Path(payload["out"]).write_text(
            json.dumps({"token_dir": str(token_dir), "payload": payload})
        )


    def wedge(token_dir, payload):
        time.sleep(120)
    """)


@pytest.fixture
def callbacks(tmp_path, monkeypatch) -> str:
    """The name of a module of ``on_expire`` callbacks the beater can import: the
    tests may run against an installed wheel, which carries no test modules."""
    where = tmp_path / "callbacks"
    where.mkdir()
    (where / "l0_expiry_callbacks.py").write_text(CALLBACKS)
    path = os.environ.get("PYTHONPATH")
    monkeypatch.setenv("PYTHONPATH", str(where) + (os.pathsep + path if path else ""))
    return "l0_expiry_callbacks"


def wait_for(condition, timeout: float = 10.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(step)
    return True


def newest_mtime(lockfile: Path) -> float:
    return max(p.lstat().st_mtime for p in [lockfile, *lockfile.rglob("*")])


def heartbeat_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("pydra-lease-")]


def take(lockfile: Path, settings=CROSS) -> DirectoryLease:
    lease = DirectoryLease.at(lockfile, settings, beat="external")
    assert lease.try_take() is None
    return lease


# Settings, whole


def test_settings_differing_only_in_attribute_delay_unequal():
    a = LeaseSettings(stale_after=10, timeout=20, attribute_delay=0)
    b = LeaseSettings(stale_after=10, timeout=20, attribute_delay=1)
    assert a != b


@pytest.mark.parametrize(
    "field, value",
    [("stale_after", 11), ("timeout", 21), ("renew_fraction", 0.2)],
)
def test_settings_differing_in_a_staleness_field_unequal(field, value):
    base = dict(stale_after=10, timeout=20, attribute_delay=0, renew_fraction=0.1)
    assert LeaseSettings(**base) != LeaseSettings(**{**base, field: value})


def test_settings_differing_only_in_max_wait_equal():
    a = LeaseSettings(stale_after=10, timeout=20, attribute_delay=0)
    b = LeaseSettings(stale_after=10, timeout=20, attribute_delay=0, max_wait=5)
    assert a == b
    assert hash(a) == hash(b)


def test_settings_json_round_trip():
    settings = LeaseSettings(
        stale_after=12.5, timeout=40, renew_fraction=0.2, attribute_delay=3
    )
    record = settings.to_json()
    assert record == {
        "stale_after": 12.5,
        "renew_fraction": 0.2,
        "attribute_delay": 3,
        "timeout": 40,
    }
    assert json.loads(json.dumps(record)) == record
    assert LeaseSettings.from_json(record) == settings
    assert LeaseSettings.from_json(LeaseSettings().to_json()) == LeaseSettings()


def test_settings_json_leaves_out_max_wait():
    assert "max_wait" not in LeaseSettings(max_wait=5).to_json()


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_settings_from_json_refuses_other_keys(change):
    record = LeaseSettings().to_json()
    if change == "missing":
        del record["attribute_delay"]
    else:
        record["max_wait"] = 1
    with pytest.raises(ValueError, match="attribute_delay|max_wait"):
        LeaseSettings.from_json(record)


def test_settings_from_json_checks_the_inequalities():
    record = LeaseSettings().to_json()
    record["attribute_delay"] = 1000
    with pytest.raises(ValueError, match="live holder"):
        LeaseSettings.from_json(record)


# The named lease, beaten by its own thread


def test_named_lease_take_beat_stale_takeover(tmp_path):
    """A named lease takes, beats, goes stale once its heartbeat stops, and is taken
    over through ``try_take``."""
    from pydra.engine.tests.test_lease import FakeTime

    lockfile = tmp_path / "claims" / "ticket.lease"
    lease = DirectoryLease.at(lockfile, FAST)
    assert lease.checksum is None and lease.cache_dir is None
    assert lease.settings == FAST
    assert lease.try_take() is None
    assert lease.held and lease.token_dir.parent == lockfile
    first = newest_mtime(lockfile)
    assert wait_for(lambda: newest_mtime(lockfile) > first), "never beaten"
    assert lease.prove()

    lease._heartbeat.stopped.set()  # the holder stops beating, as if frozen
    lease._heartbeat.join(5)
    clock = FakeTime(newest_mtime(lockfile))
    taker = DirectoryLease.at(
        lockfile,
        FAST,
        clock=clock.storage,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    while (delay := taker.try_take()) is not None:
        clock.sleep(delay)
    assert taker.held
    assert clock.t > FAST.stale_after
    assert len(taker.broke) == 1
    assert not lease.prove()
    taker.__exit__(None, None, None)
    lease.__exit__(None, None, None)
    assert not lockfile.exists()


def test_named_lease_is_entered_with_with(tmp_path):
    lockfile = tmp_path / "workers" / "w" / "lease"
    with DirectoryLease.at(lockfile, FAST) as lease:
        assert lease.held
        assert (lease.token_dir / "holder.json").is_file()
    assert lease.token_dir is None
    assert not lockfile.exists()


@pytest.mark.parametrize("enter", ["try_take", "with"])
def test_external_lease_starts_no_thread(tmp_path, enter):
    lockfile = tmp_path / "claims" / "ticket.lease"
    before = heartbeat_threads()
    lease = DirectoryLease.at(lockfile, FAST, beat="external")
    if enter == "with":
        lease.__enter__()
    else:
        assert lease.try_take() is None
    assert lease.held and lease._heartbeat is None
    assert heartbeat_threads() == before
    first = newest_mtime(lockfile)
    time.sleep(3 * FAST.renew_interval)
    assert newest_mtime(lockfile) == first  # nothing beats it
    assert lease.prove()
    lease.__exit__(None, None, None)
    assert not lockfile.exists()


def test_beat_must_be_thread_or_external(tmp_path):
    with pytest.raises(ValueError, match="beat"):
        DirectoryLease.at(tmp_path / "x.lease", FAST, beat="never")


def test_on_expire_must_be_importable():
    with pytest.raises(ValueError, match="importable"):
        LeaseBeater(on_expire=lambda token_dir, payload: None)


# Cross-process beating


@pytest.mark.timeout(120)
def test_unimportable_on_expire_fails_the_start():
    with pytest.raises(BeaterDead, match="did not start"):
        BeaterClient(LeaseBeater(on_expire="pydra.engine.no_such_module:fn"))


@pytest.mark.timeout(120)
def test_lease_beaten_from_another_process(tmp_path):
    """Leases taken in this process with ``beat="external"`` and beaten by a
    `LeaseBeater` in another stay held beyond three stale bounds."""
    leases = [take(tmp_path / "claims" / f"{n}.lease") for n in range(2)]
    with BeaterClient(LeaseBeater()) as client:
        assert client.process.pid != os.getpid()
        for lease in leases:
            client.hold(lease.token_dir, CROSS, lease=lease)
        assert heartbeat_threads() == []  # no beat thread in the holder
        first = [newest_mtime(lease.lockfile) for lease in leases]
        start = time.monotonic()
        while time.monotonic() - start < 3 * CROSS.stale_after + 0.5:
            for lease in leases:
                rival = DirectoryLease.at(lease.lockfile, CROSS)
                assert rival.try_take() is not None, "taken over while beaten"
            time.sleep(0.25)
        for lease, was in zip(leases, first):
            assert newest_mtime(lease.lockfile) > was
            assert lease.prove()
            assert lease.lost is None
        assert client.lost == {}
        for lease in leases:
            client.release(lease.token_dir)
            lease.__exit__(None, None, None)
    assert not (tmp_path / "claims" / "0.lease").exists()


@pytest.mark.timeout(120)
def test_beater_reports_loss(tmp_path):
    """A token directory removed by a taker: the beater sends ``lost`` and the
    holder's lease is marked lost, so ``prove()`` answers False."""
    lease = take(tmp_path / "ticket.lease")
    seen = []
    with BeaterClient(LeaseBeater(), on_lost=lambda d, why: seen.append(d)) as client:
        client.hold(lease.token_dir, CROSS, lease=lease)
        token_dir = lease.token_dir
        gone = token_dir.parent / f"gone-{token_dir.name}"
        for _ in range(100):  # what a takeover does; Windows may refuse mid-beat
            try:
                os.rename(token_dir, gone)
                break
            except PermissionError:
                time.sleep(0.01)
        shutil.rmtree(gone)
        assert wait_for(lambda: lease.lost is not None, 5 * CROSS.renew_interval + 5)
        assert token_dir in client.lost
        assert wait_for(lambda: seen == [token_dir], 5)  # called after the mark
        assert not lease.prove()
        assert client.alive


@pytest.mark.timeout(120)
def test_release_is_not_a_loss(tmp_path):
    """The holder releases after the beater acknowledged: no ``lost`` follows."""
    lease = take(tmp_path / "ticket.lease")
    with BeaterClient(LeaseBeater()) as client:
        client.hold(lease.token_dir, CROSS, lease=lease)
        time.sleep(2 * CROSS.renew_interval)
        token_dir = lease.token_dir
        client.release(token_dir)
        lease.__exit__(None, None, None)
        assert not lease.lockfile.exists()
        time.sleep(4 * CROSS.renew_interval)
        assert client.lost == {}
        assert lease.lost is None
        assert client.alive


TAKER = textwrap.dedent("""\
    import json, sys, time
    from pathlib import Path
    from pydra.engine.lease import DirectoryLease, LeaseSettings
    settings = LeaseSettings.from_json(json.loads(sys.argv[2]))
    lease = DirectoryLease.at(Path(sys.argv[1]), settings)
    start = time.monotonic()
    while (delay := lease.try_take()) is not None:
        time.sleep(delay)
    waited = time.monotonic() - start
    broke = len(lease.broke)
    lease.__exit__(None, None, None)
    print(json.dumps({"broke": broke, "waited": waited}))
    """)


@pytest.mark.timeout(120)
def test_deadline_stops_beating(tmp_path):
    """At its deadline the beater stops beating the directory and sends
    ``expired``; a third process then takes the lease over by ``try_take``."""
    lease = take(tmp_path / "ticket.lease")
    with BeaterClient(LeaseBeater()) as client:
        client.hold(lease.token_dir, CROSS, lease=lease)
        client.deadline(lease.token_dir, time.time() + 0.5)
        assert client.wait_expired(lease.token_dir, 30)
        stopped = newest_mtime(lease.lockfile)
        script = tmp_path / "taker.py"
        script.write_text(TAKER)
        taker = subprocess.run(
            [
                sys.executable,
                str(script),
                str(lease.lockfile),
                json.dumps(CROSS.to_json()),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert taker.returncode == 0, taker.stderr
        outcome = json.loads(taker.stdout.strip().splitlines()[-1])
        assert outcome["broke"] == 1
        assert time.time() - stopped > CROSS.stale_after
        assert not lease.prove()
        assert client.lost == {}  # an expired directory is not beaten, so not lost
        client.ack_expired(lease.token_dir)
        client.release(lease.token_dir)


@pytest.mark.timeout(120)
def test_deadline_cleared_keeps_beating(tmp_path):
    lease = take(tmp_path / "ticket.lease")
    with BeaterClient(LeaseBeater()) as client:
        client.hold(lease.token_dir, CROSS, lease=lease)
        client.deadline(lease.token_dir, time.time() + 0.5)
        client.deadline(lease.token_dir, None)
        assert not client.wait_expired(lease.token_dir, 1.0)
        assert lease.prove()
        client.release(lease.token_dir)
        lease.__exit__(None, None, None)


@pytest.mark.timeout(120)
def test_deadline_payload_reaches_callback(tmp_path, callbacks):
    """The payload sent with ``deadline`` is what ``on_expire`` receives, in the
    beater process, before ``expired`` reaches the holder."""
    lease = take(tmp_path / "ticket.lease")
    out = tmp_path / "expired.json"
    payload = {"out": str(out), "jobs": ["a", "b"], "attempt": "9c41.3"}
    when_expired = []
    with BeaterClient(
        LeaseBeater(on_expire=f"{callbacks}:record_expiry"),
        on_expired=lambda d: when_expired.append(out.exists()),
    ) as client:
        client.hold(lease.token_dir, CROSS, lease=lease)
        client.deadline(lease.token_dir, time.time() + 0.3, payload)
        assert client.wait_expired(lease.token_dir, 30)
        assert when_expired == [True]
        received = json.loads(out.read_text())
        assert received == {"token_dir": str(lease.token_dir), "payload": payload}
        client.ack_expired(lease.token_dir)


@pytest.mark.timeout(120)
def test_beater_eof(tmp_path):
    """The beater killed: the holder sees end of file and takes nothing more."""
    lease = take(tmp_path / "ticket.lease")
    dead = []
    client = BeaterClient(LeaseBeater(), on_dead=lambda why: dead.append(why))
    try:
        client.hold(lease.token_dir, CROSS, lease=lease)
        client.process.kill()
        assert client.wait_dead(30)
        assert not client.alive
        assert dead and "end of file" in dead[0]
        with pytest.raises(BeaterDead):
            client.hold(lease.token_dir, CROSS)
    finally:
        client.close()
        lease.__exit__(None, None, None)


@pytest.mark.timeout(120)
def test_ack_timeout_is_beater_death(tmp_path, callbacks):
    """A beater that stops answering (here, wedged in ``on_expire``): ``hold`` times
    out after ``renew_interval`` and the holder has lost its beater."""
    first = take(tmp_path / "one.lease")
    second = take(tmp_path / "two.lease")
    client = BeaterClient(LeaseBeater(on_expire=f"{callbacks}:wedge"))
    try:
        client.hold(first.token_dir, CROSS)
        client.deadline(first.token_dir, time.time() - 1)  # the beater wedges
        start = time.monotonic()
        with pytest.raises(BeaterDead, match="acknowledge"):
            client.hold(second.token_dir, CROSS)
        waited = time.monotonic() - start
        assert CROSS.renew_interval * 0.9 <= waited < CROSS.renew_interval + 5
        assert not client.alive
        assert wait_for(lambda: client.process.poll() is not None, 30)  # stopped
        with pytest.raises(BeaterDead):
            client.release(first.token_dir)
    finally:
        client.close()
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)


HOLDER = textwrap.dedent("""\
    import json, sys, time
    from pathlib import Path
    from pydra.engine.lease import BeaterClient, DirectoryLease, LeaseBeater, LeaseSettings
    lockfile, settings, kill_after, mode = (
        Path(sys.argv[1]), LeaseSettings.from_json(json.loads(sys.argv[2])),
        json.loads(sys.argv[3]), sys.argv[4],
    )
    lease = DirectoryLease.at(lockfile, settings, beat="external")
    assert lease.try_take() is None
    with BeaterClient(LeaseBeater(kill_after=kill_after)) as client:
        client.hold(lease.token_dir, settings, lease=lease)
        client.deadline(lease.token_dir, time.time() + 0.2, {"call": 1})
        assert client.wait_expired(lease.token_dir, 30)
        print("expired", flush=True)
        if mode == "ack":
            client.ack_expired(lease.token_dir)
            print("acknowledged", flush=True)
        time.sleep(float(sys.argv[5]))
        print("survived", flush=True)
    """)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "kill_after, mode, killed",
    [(1.0, "silent", True), (None, "silent", False), (1.0, "ack", False)],
)
def test_kill_after_unacknowledged_expiry(tmp_path, kill_after, mode, killed):
    """With ``kill_after`` set, a holder that sends no ``ack_expired`` in time is
    killed by its beater; unset, or acknowledged, it never is."""
    script = tmp_path / "holder.py"
    script.write_text(HOLDER)
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            str(tmp_path / "ticket.lease"),
            json.dumps(CROSS.to_json()),
            json.dumps(kill_after),
            mode,
            "4",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert "expired" in proc.stdout, proc.stderr
    if killed:
        assert proc.returncode != 0
        assert "survived" not in proc.stdout
    else:
        assert proc.returncode == 0, proc.stderr
        assert "survived" in proc.stdout
