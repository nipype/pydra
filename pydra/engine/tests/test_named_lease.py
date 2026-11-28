"""The named lease and its cross-process beater (RFC 0005 §5.3, L0)."""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import contextlib
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

# Beats every 4 s, stale after 10 s: an acknowledgement is due within 4 s, which a
# slow Windows runner meets (1 s did not, once); (1 + 1) * 4 < 10 still holds
CROSS = LeaseSettings(stale_after=10, timeout=20, renew_fraction=0.4, attribute_delay=0)

# Beats every 0.2 s: the acknowledgement-timeout test, whose subject is the bound
SHORT = LeaseSettings(stale_after=2, timeout=10, attribute_delay=0)

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


def wait_for(condition, timeout: float = 60.0, step: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(step)
    return True


def entries(directory: Path) -> list[str]:
    """The entries of ``directory``, none if it is gone or no directory: a beater
    removes beats, and takers rename token directories, while a test reads them."""
    try:
        return os.listdir(directory)
    except OSError:
        return []


def beat_count(lockfile: Path) -> int:
    """The highest beat number in a lease's token directories, 0 if none. A beat
    makes ``beat-<n+1>`` before it removes ``beat-<n>``, so the number grows exactly
    while the lease is beaten, whatever the storage's mtime resolution, and a beat
    removed during the listing cannot lower it."""
    return max(
        (
            int(entry[5:])
            for token in entries(lockfile)
            for entry in entries(lockfile / token)
            if re.fullmatch(r"beat-\d+", entry)
        ),
        default=0,
    )


def newest_mtime(lockfile: Path) -> float:
    """The newest mtime of a lease, its entries and theirs, skipping any removed
    while it is read. Read only where nothing beats the lease."""
    paths = [lockfile]
    for token in entries(lockfile):
        paths.append(lockfile / token)
        paths.extend(lockfile / token / entry for entry in entries(lockfile / token))
    mtimes = []
    for path in paths:
        try:
            mtimes.append(path.lstat().st_mtime)
        except OSError:
            pass
    return max(mtimes)


def heartbeat_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("pydra-lease-")]


def rename_away(token_dir: Path) -> Path:
    """Rename a holder's token directory aside, as a takeover does; Windows may
    refuse while the beater makes or removes a beat in it."""
    gone = token_dir.parent / f"gone-{token_dir.name}"
    for _ in range(200):
        try:
            os.rename(token_dir, gone)
            return gone
        except PermissionError:
            time.sleep(0.01)
    raise AssertionError(f"could not rename {token_dir}")


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


def test_settings_differing_only_in_holder_equal():
    """``holder`` chooses what one holder's record names, not staleness."""
    named = LeaseSettings(stale_after=10, timeout=20, attribute_delay=0)
    anonymous = LeaseSettings(
        stale_after=10, timeout=20, attribute_delay=0, holder="anonymous"
    )
    assert named == anonymous
    assert "holder" not in anonymous.to_json()


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
    assert wait_for(lambda: beat_count(lockfile) >= 2), "never beaten"
    assert lease.prove()

    lease._heartbeat.stopped.set()  # the holder stops beating, as if frozen
    lease._heartbeat.join(60)
    assert not lease._heartbeat.is_alive()
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
    # three beats of a thread-beaten lease pass: a beat of this one would be due
    other = DirectoryLease.at(tmp_path / "claims" / "other.lease", FAST)
    with other:
        was = beat_count(other.lockfile)
        assert wait_for(lambda: beat_count(other.lockfile) >= was + 3), "never beaten"
    assert beat_count(lockfile) == 0  # nothing beats it
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
@pytest.mark.parametrize("which", ["thread", "unheld"])
def test_hold_refuses_a_lease_it_cannot_beat(tmp_path, which):
    """``hold`` takes the lease itself: one taken with ``beat="thread"`` already has
    a beater, and one not held has no token directory."""
    if which == "thread":
        lease = DirectoryLease.at(tmp_path / "x.lease", CROSS)
        assert lease.try_take() is None
    else:
        lease = DirectoryLease.at(tmp_path / "x.lease", CROSS, beat="external")
    try:
        with BeaterClient(LeaseBeater()) as client:
            with pytest.raises(ValueError, match="external|not held"):
                client.hold(lease)
            assert client.alive
    finally:
        lease.__exit__(None, None, None)


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
            client.hold(lease)
        assert heartbeat_threads() == []  # no beat thread in the holder
        first = [beat_count(lease.lockfile) for lease in leases]
        # beaten for as many beats as span three stale bounds, a rival refused all along
        beats = math.ceil(3 * CROSS.stale_after / CROSS.renew_interval)
        deadline = time.monotonic() + 100  # bound only
        while any(
            beat_count(lease.lockfile) < was + beats
            for lease, was in zip(leases, first)
        ):
            assert time.monotonic() < deadline, "not beaten often enough"
            for lease in leases:
                rival = DirectoryLease.at(lease.lockfile, CROSS)
                assert rival.try_take() is not None, "taken over while beaten"
            time.sleep(0.25)
        for lease in leases:
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
        client.hold(lease)
        token_dir = lease.token_dir
        shutil.rmtree(rename_away(token_dir))
        assert wait_for(lambda: lease.lost is not None, 60)
        assert token_dir in client.lost
        assert wait_for(lambda: seen == [token_dir], 60)  # called after the mark
        assert not lease.prove()
        assert client.alive


@pytest.mark.timeout(120)
def test_release_is_not_a_loss(tmp_path):
    """The holder releases after the beater acknowledged: no ``lost`` follows."""
    lease = take(tmp_path / "ticket.lease")
    with BeaterClient(LeaseBeater()) as client:
        client.hold(lease)
        other = take(tmp_path / "other.lease")
        client.hold(other)
        assert wait_for(lambda: beat_count(lease.lockfile) >= 1, 60), "never beaten"
        token_dir = lease.token_dir
        client.release(token_dir)
        lease.__exit__(None, None, None)
        assert not lease.lockfile.exists()
        # two beats of the other lease the beater holds: a beat after the release was due
        was = beat_count(other.lockfile)
        assert wait_for(lambda: beat_count(other.lockfile) >= was + 2), "not beaten"
        assert client.lost == {}
        assert lease.lost is None
        assert client.alive
        client.release(other.token_dir)
        other.__exit__(None, None, None)


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
        client.hold(lease)
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
        client.hold(lease)
        client.deadline(lease.token_dir, time.time() + 3)  # cleared well before
        client.deadline(lease.token_dir, None)
        assert not client.wait_expired(lease.token_dir, 4.0)
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
        client.hold(lease)
        client.deadline(lease.token_dir, time.time() + 0.3, payload)
        assert client.wait_expired(lease.token_dir, 30)
        assert wait_for(lambda: when_expired, 60)  # on the callback thread
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
        client.hold(lease)
        client.process.kill()
        assert client.wait_dead(30)
        assert not client.alive
        assert wait_for(lambda: dead, 60)  # on_dead runs on the callback thread
        assert "end of file" in dead[0]
        with pytest.raises(BeaterDead):
            client.hold(lease)
    finally:
        client.close()
        lease.__exit__(None, None, None)


@pytest.mark.timeout(120)
def test_ack_timeout_is_beater_death(tmp_path, callbacks):
    """A beater that stops answering (here, wedged in ``on_expire``): ``hold`` times
    out after ``renew_interval`` and the holder has lost its beater."""
    first = take(tmp_path / "one.lease", SHORT)
    second = take(tmp_path / "two.lease", SHORT)
    client = BeaterClient(LeaseBeater(on_expire=f"{callbacks}:wedge"))
    try:
        client.hold(first)
        client.deadline(first.token_dir, time.time() - 1)  # the beater wedges
        bounds = []
        real_wait_for = client._changed.wait_for

        def wait_for_ack(predicate, timeout=None):
            answered = real_wait_for(predicate, timeout)
            bounds.append((timeout, answered))
            return answered

        client._changed.wait_for = wait_for_ack
        bound = f"{SHORT.renew_interval:g}s"
        with pytest.raises(
            BeaterDead, match=f"did not acknowledge hold within {bound}"
        ):
            client.hold(second)
        del client._changed.wait_for
        # one wait for the acknowledgement, given renew_interval, which ran out unanswered
        assert bounds == [(SHORT.renew_interval, False)]
        assert not client.alive
        assert wait_for(lambda: client.process.poll() is not None, 30)  # stopped
        with pytest.raises(BeaterDead):
            client.release(first.token_dir)
    finally:
        client.close()
        first.__exit__(None, None, None)
        second.__exit__(None, None, None)


@pytest.mark.skipif(sys.platform == "win32", reason="stops the beater with SIGSTOP")
def test_ping_finds_a_hung_beater():
    """``ping`` answers True while the beater acknowledges. One unanswered ping (a holder whose call
    held the GIL past the timeout, or a beater slow for a moment) is a miss, not a death: the beater
    keeps running, and an answer resets the count. `PING_MISSES` consecutive misses are its death.
    """
    import signal

    from pydra.engine.lease import PING_MISSES

    client = BeaterClient()
    try:
        assert client.ping(10) is True
        os.kill(client.process.pid, signal.SIGSTOP)
        assert client.ping(0.2) is False  # one slow ping
        assert client.alive and client.process.poll() is None
        os.kill(client.process.pid, signal.SIGCONT)
        assert client.ping(10) is True  # answered: the misses are forgotten
        os.kill(client.process.pid, signal.SIGSTOP)
        for _ in range(PING_MISSES - 1):
            assert client.ping(0.2) is False
            assert client.alive
        with pytest.raises(BeaterDead, match="ping"):
            client.ping(0.2)
        assert not client.alive
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(client.process.pid, signal.SIGCONT)
        client.close()


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
        client.hold(lease)
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
    [(1.0, "silent", True), (None, "silent", False), (5.0, "ack", False)],
)
def test_kill_after_unacknowledged_expiry(tmp_path, kill_after, mode, killed):
    """With ``kill_after`` set, a holder that sends no ``ack_expired`` in time is
    killed by its beater; unset, or acknowledged, it never is. The holder sleeps
    past ``kill_after`` with a margin; the acknowledging holder has 5 s to answer,
    so only the silent one depends on the kill's timing."""
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
            str((kill_after or 4) + 5),
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


@pytest.mark.timeout(120)
def test_callbacks_may_call_the_client(tmp_path):
    """``on_lost`` and ``on_expired`` run off the thread that reads the beater, so
    a callback can release or acknowledge, and the beater stays healthy."""
    lost, expired = take(tmp_path / "lost.lease"), take(tmp_path / "expired.lease")
    kept = take(tmp_path / "kept.lease")
    done, holder = [], []

    def call(request, token_dir):
        try:
            request(token_dir)
            done.append((request.__name__, token_dir))
        except Exception as err:
            done.append((repr(err), token_dir))

    with BeaterClient(
        LeaseBeater(kill_after=30),
        on_lost=lambda d, why: call(holder[0].release, d),
        on_expired=lambda d: call(holder[0].ack_expired, d),
    ) as client:
        holder.append(client)
        for lease in (lost, expired, kept):
            client.hold(lease)
        token_dir = lost.token_dir
        rename_away(token_dir)
        client.deadline(expired.token_dir, time.time() + 0.2)
        assert wait_for(lambda: len(done) == 2, 5 * CROSS.renew_interval + 10)
        assert sorted(done) == sorted(
            [("release", token_dir), ("ack_expired", expired.token_dir)]
        )
        assert client.alive
        before = beat_count(kept.lockfile)
        assert wait_for(
            lambda: beat_count(kept.lockfile) > before, 3 * CROSS.renew_interval
        )
        assert kept.prove()
        for lease in (expired, kept):
            client.release(lease.token_dir)
    for lease in (lost, expired, kept):
        lease.__exit__(None, None, None)


ORPHANING = textwrap.dedent("""\
    import json, os, sys, time
    from pathlib import Path
    from pydra.engine.lease import BeaterClient, DirectoryLease, LeaseBeater, LeaseSettings
    settings = LeaseSettings.from_json(json.loads(sys.argv[2]))
    lease = DirectoryLease.at(Path(sys.argv[1]), settings, beat="external")
    assert lease.try_take() is None
    client = BeaterClient(LeaseBeater(kill_after=1))
    client.hold(lease)
    client.deadline(lease.token_dir, time.time() + 600)
    child = None
    if sys.argv[3] == "fork":
        child = os.fork()
        if child == 0:
            time.sleep(60)
            os._exit(0)
    print(json.dumps({"beater": client.process.pid, "child": child}), flush=True)
    time.sleep(600)
    """)


def stops_beating(lockfile: Path, quiet: float, within: float) -> bool:
    """Whether the lease's beat count stays put for ``quiet`` seconds, at some
    point within ``within`` seconds."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        before = beat_count(lockfile)
        time.sleep(quiet)
        if beat_count(lockfile) == before:
            return True
    return False


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "fork",
    [
        "no-fork",
        pytest.param(
            "fork",
            marks=pytest.mark.skipif(not hasattr(os, "fork"), reason="no fork"),
        ),
    ],
)
def test_holder_death_ends_beater(tmp_path, fork):
    """The holder SIGKILLed, with or without a forked child still holding a copy of
    its end of the pipe: the beater stops beating and exits, and the lease goes
    stale."""
    lockfile = tmp_path / "ticket.lease"
    script = tmp_path / "orphaning.py"
    script.write_text(ORPHANING)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(lockfile), json.dumps(CROSS.to_json()), fork],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    child = None
    try:
        line = proc.stdout.readline()
        assert line, proc.stderr.read()
        pids = json.loads(line)
        child = pids["child"]
        proc.kill()
        proc.wait(30)
        if os.name == "posix":
            # the beater, the only process that beats the lease, has exited
            assert wait_for(lambda: not pid_alive(pids["beater"]), 60)
        else:  # signal 0 would end the process on Windows: watch the beats instead
            assert stops_beating(lockfile, 2 * CROSS.renew_interval, 60), "still beaten"
        if child is not None:
            assert pid_alive(child)  # the child outlived its parent, as intended
    finally:
        if proc.poll() is None:
            proc.kill()
        if child is not None and pid_alive(child):
            os.kill(child, 9)


DRAINING = textwrap.dedent("""\
    import json, os, signal, sys, time
    from pathlib import Path
    from pydra.engine.lease import BeaterClient, DirectoryLease, LeaseBeater, LeaseSettings
    settings = LeaseSettings.from_json(json.loads(sys.argv[2]))
    stop = []
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    leases = [
        DirectoryLease.at(Path(sys.argv[1]) / f"{n}.lease", settings, beat="external")
        for n in range(2)
    ]
    with BeaterClient(LeaseBeater()) as client:
        for lease in leases:
            assert lease.try_take() is None
            client.hold(lease)
        print("ready", flush=True)
        deadline = time.monotonic() + 60
        while not stop and time.monotonic() < deadline:
            time.sleep(0.05)
        assert stop, "no SIGTERM"
        # the drain: the beater outlived the SIGTERM, so each lease gains two beats
        def beats(lease):
            names = os.listdir(lease.token_dir)
            return max([int(n[5:]) for n in names if n.startswith("beat-")] or [0])
        was = [beats(lease) for lease in leases]
        deadline = time.monotonic() + 60
        while any(beats(lease) < w + 2 for lease, w in zip(leases, was)):
            assert time.monotonic() < deadline, "not beaten after the SIGTERM"
            time.sleep(0.05)
        assert client.alive and all(lease.prove() for lease in leases)
        for lease in leases:
            client.release(lease.token_dir)
            lease.__exit__(None, None, None)
        print("released", flush=True)
    """)


@pytest.mark.timeout(120)
@pytest.mark.skipif(not hasattr(os, "killpg"), reason="no process groups")
def test_group_sigterm_leaves_the_beater_to_the_drain(tmp_path):
    """SIGTERM to the holder's whole process group, as Slurm and a process-group
    stop send it: the beater survives, so the holder's releases are acknowledged."""
    import signal

    script = tmp_path / "draining.py"
    script.write_text(DRAINING)
    proc = subprocess.Popen(
        [sys.executable, str(script), str(tmp_path), json.dumps(CROSS.to_json())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        assert proc.stdout.readline().strip() == "ready", proc.stderr.read()
        os.killpg(proc.pid, signal.SIGTERM)
        out, err = proc.communicate(timeout=100)
        assert proc.returncode == 0, err
        assert "released" in out
        assert not list(tmp_path.glob("*.lease"))
    finally:
        if proc.poll() is None:
            proc.kill()
