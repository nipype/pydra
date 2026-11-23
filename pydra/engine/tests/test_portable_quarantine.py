"""Fenced quarantine, `evict` and `collect` of a `PortableStore` (RFC 0003 sections 6.3, 9)."""

import errno
import os
import threading
import time
from pathlib import Path

import pytest

from pydra.engine.portable.entry import CorruptEntry
from pydra.engine.portable.store import CollectReport, PortableStore, StoreNotWritable
from pydra.utils.storagedir import IS_WINDOWS, StorageDir, StorePolicy
from pydra.utils.storagedir import StoreAges

from .test_portable_store import (
    CTX,
    OUT,
    corrupt,
    entry_path,
    make,
    publish,
    quarantined,
    recording_rename,
)

OTHER = {"a": 2}


def address(store, committed):
    """The (root, cdir_dir) handles' path pieces of a committed entry."""
    entry = entry_path(store.path, committed)
    return entry.parent.parent.name, entry.parent.name


def quarantine_now(store, committed, identity, **kw):
    shard, cdir = address(store, committed)
    with store._open() as root:
        with (
            root.child("results@1") as e,
            e.child(shard) as s,
            s.child(cdir) as d,
        ):
            return store._quarantine(
                root,
                d,
                committed.entry.fingerprint,
                committed.entry.cdir,
                identity,
                **kw,
            )


def identity_of(store, committed):
    return os.stat(entry_path(store.path, committed))


SKEW = 10_000.0


@pytest.fixture
def skew(monkeypatch):
    """The store's clock runs SKEW seconds ahead. A rename or a utime changes ``ctime`` to
    the real now, which cannot be set back, so an object is made old by moving the clock.
    """
    import pydra.engine.portable.store as mod

    monkeypatch.setattr(mod, "_now", lambda: time.time() + SKEW)


def age(path, seconds):
    """Make ``path`` ``seconds`` old on the skewed clock (at most SKEW)."""
    t = time.time() + SKEW - seconds
    if os.utime in os.supports_follow_symlinks:
        os.utime(path, (t, t), follow_symlinks=False)
    else:
        os.utime(path, (t, t))


# -- fenced quarantine -----------------------------------------------------------------------------


@pytest.mark.parametrize("zero_inode", [False, True])
def test_two_readers_quarantine_one_entry_while_writer_commits(
    tmp_path, monkeypatch, zero_inode
):
    """Reader B saw the corrupt entry that reader A moved; a writer then commits a valid
    one. B's late move must not take it, by name or, with no inode number, by the fence.
    """
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    corrupt(entry_path(store.path, c))
    seen_by_b = identity_of(store, c)
    if zero_inode:
        monkeypatch.setattr(mod, "_inode_name", lambda st: os.urandom(16).hex())
    # reader A, which found the same entry corrupt, moves it
    assert quarantine_now(store, c, identity_of(store, c)) is True
    assert len(quarantined(store.path)) == 1
    # the writer commits a valid entry at the address
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    # reader B's move, from the stale check
    late = quarantine_now(store, c, None if zero_inode else seen_by_b)
    assert late is False
    assert len(quarantined(store.path)) == 1
    assert store.read(c.entry, None).outputs == OUT


def test_non_owner_reader_q_name(tmp_path, monkeypatch):
    """A reader that cannot move across parents renames within the parent to
    ``q-<inode name>``, and the inode gives the name."""
    from pydra.engine.portable.store import _inode_name

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(store.path, c)
    corrupt(entry)
    expected = "q-" + _inode_name(os.stat(entry))

    def deny(self, name, target, new_name):
        if "quarantine" in target.path:
            raise PermissionError(errno.EACCES, "not the owner")

    recording_rename(monkeypatch, [], deny)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert [p.name for p in entry.parent.iterdir()] == [expected]
    assert quarantined(store.path) == []


def test_parent_unwritable_corrupt_entry_raises_with_owner(tmp_path, monkeypatch):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(store.path, c)
    corrupt(entry)

    def deny(self, name, target, new_name):
        raise PermissionError(errno.EACCES, "not writable")

    recording_rename(monkeypatch, [], deny)
    with pytest.raises(CorruptEntry) as info:
        store.read(c.entry, None)
    message = str(info.value)
    assert str(entry) in message and "owner:" in message
    assert entry.is_dir()  # it stays


def test_evict_moves_the_entry_with_the_expected_digest(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    store.evict(c.entry, c.own_digest)
    assert not entry_path_exists(store, c)
    assert len(quarantined(store.path)) == 1
    again = publish(store)
    assert (again.won, again.stored) == (True, True)


def entry_path_exists(store, committed):
    try:
        return entry_path(store.path, committed).exists()
    except TypeError:  # the shard is gone
        return False


def test_evict_of_an_absent_entry_is_nothing(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    store.evict(c.entry, c.own_digest)
    store.evict(c.entry, c.own_digest)
    assert len(quarantined(store.path)) == 1


def test_evict_wrong_expected_digest_restores(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    store.evict(c.entry, "0" * 64)
    assert store.read(c.entry, None).outputs == OUT
    assert quarantined(store.path) == []


def test_evict_restores_an_entry_committed_after_the_caller_read(tmp_path, monkeypatch):
    """The caller read digest D1; another writer replaced the entry with D2 between the
    evict's own read and its move: the moved copy is checked and put back."""
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    corrupt_then_replace = {"done": False}
    real = PortableStore._quarantine

    def swap_then_quarantine(self, root, parent, name, cdir, identity, expected=None):
        if not corrupt_then_replace["done"]:
            corrupt_then_replace["done"] = True
            # the entry the evict read leaves, and a valid entry with other outputs arrives
            (Path(self.path) / "quarantine").mkdir(exist_ok=True)
            os.rename(entry_path(self.path, c), Path(self.path) / "quarantine" / "old")
            publish(self, OTHER)
            assert len(quarantined(self.path)) == 1
            # a store that has no inode claims takes the fence
            monkeypatch.setattr(mod, "_inode_name", lambda st: os.urandom(16).hex())
            return real(self, root, parent, name, cdir, None, expected)
        return real(self, root, parent, name, cdir, identity, expected)

    monkeypatch.setattr(PortableStore, "_quarantine", swap_then_quarantine)
    store.evict(c.entry, c.own_digest)
    monkeypatch.setattr(PortableStore, "_quarantine", real)
    assert store.read(c.entry, None).outputs == OTHER
    assert len(quarantined(store.path)) == 1  # only the one moved by hand


def test_evict_unmovable_entry_raises_with_owner(tmp_path, monkeypatch):
    store = make(tmp_path)
    c = publish(store)

    def deny(self, name, target, new_name):
        raise PermissionError(errno.EACCES, "not writable")

    recording_rename(monkeypatch, [], deny)
    with pytest.raises(StoreNotWritable, match="owner"):
        store.evict(c.entry, c.own_digest)
    assert store.read(c.entry, None).outputs == OUT


# -- collect --------------------------------------------------------------------------------------


def aged_store(tmp_path, **ages):
    return make(tmp_path, policy=StorePolicy(ages=StoreAges(**ages)))


def test_collect_removes_aged_staging_quarantine_and_q_names(
    tmp_path, monkeypatch, skew
):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    c = publish(store)
    root = Path(store.path)
    entry = entry_path(root, c)
    parent = entry.parent
    # an aged staging directory with content, and a header temporary
    stage = root / "staging" / "dead"
    (stage / "files").mkdir(parents=True)
    (stage / "files" / "f").write_text("x")
    os.chmod(stage / "files" / "f", 0o444)
    tmp = root / ".pydra-store.{}.tmp".format("a" * 32)
    tmp.write_text("{}")
    # a quarantined entry and a q- name
    other = publish(store, OTHER, checksum="python-other")
    store.evict(other.entry, other.own_digest)
    (parent / "q-1234").mkdir()
    (parent / "q-1234" / "m").write_text("x")
    removable = [stage, tmp, parent / "q-1234", *quarantined(root)]
    for p in removable:
        age(p, 1000)
    report = store.collect()
    assert isinstance(report, CollectReport)
    assert sorted(report.removed) == sorted(str(p) for p in removable)
    for p in removable:
        assert not p.exists()
    assert store.read(c.entry, None).outputs == OUT
    assert [p.name for p in parent.iterdir()] == [entry.name]


def test_collect_spares_live_stage(tmp_path, skew):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    root = Path(store.path)
    publish(store)
    live = root / "staging" / "live"
    dead = root / "staging" / "dead"
    live.mkdir()
    dead.mkdir()
    age(live, 25)  # touched within a quarter of the age
    age(dead, 101)
    report = store.collect()
    assert live.is_dir() and not dead.exists()
    assert [str(live)] == [w for w, _ in report.kept if w == str(live)]


def test_collect_never_removes_entries(tmp_path, skew):
    store = aged_store(tmp_path, staging=1, quarantine=1)
    cs = [publish(store, OUT, checksum=f"python-e{i}") for i in range(5)]
    for p in Path(store.path).rglob("*"):
        age(p, SKEW)
    # a corrupt entry is still an entry: only a reader quarantines it
    corrupt(entry_path(store.path, cs[0]))
    age(entry_path(store.path, cs[0]), SKEW)
    store.collect()
    for c in cs[1:]:
        assert store.read(c.entry, None).outputs == OUT
    assert entry_path(store.path, cs[0]).is_dir()


def test_collect_never_touches_sidecars(tmp_path, skew):
    store = aged_store(tmp_path, staging=1, quarantine=1)
    publish(store)
    root = Path(store.path)
    side = root / "sidecars"
    (side / "run" / "q-1").mkdir(parents=True)
    (side / "run" / "q-1" / "f").write_text("x")
    (side / "note").write_text("x")
    for p in (side, side / "run", side / "run" / "q-1", side / "note"):
        age(p, SKEW)
    report = store.collect()
    assert (side / "run" / "q-1" / "f").is_file() and (side / "note").is_file()
    assert not any("sidecars" in w for w in report.removed)


def test_collect_leaves_a_claimed_object_another_collector_took(
    tmp_path, monkeypatch, skew
):
    """An object put back (or removed) between the listing and the claim is not removed."""
    store = aged_store(tmp_path, staging=100, quarantine=100)
    c = publish(store)
    other = publish(store, OTHER, checksum="python-other")
    store.evict(other.entry, other.own_digest)
    (moved,) = quarantined(store.path)
    age(moved, 1000)
    real = StorageDir.rename

    def vanish(self, name, target, new_name):
        if new_name.startswith("gc-"):  # a mover puts it back just before the claim
            real(self, name, self, "elsewhere")
        real(self, name, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", vanish)
    report = store.collect()
    assert report.removed == []
    assert (moved.parent / "elsewhere").is_dir()


def test_collect_without_a_quarantine_directory(tmp_path):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    assert store.collect() == CollectReport([], [])


@pytest.mark.skipif(IS_WINDOWS, reason="needs POSIX modes")
def test_collect_removes_a_sealed_quarantined_entry(tmp_path, skew):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    c = publish(store)
    entry = entry_path(store.path, c)
    corrupt(entry)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    (moved,) = quarantined(store.path)
    os.chmod(moved, 0o555)
    age(moved, 1000)
    try:
        report = store.collect()
    finally:
        if moved.exists():
            os.chmod(moved, 0o755)
    assert report.removed == [str(moved)] and not moved.exists()


# -- stress ---------------------------------------------------------------------------------------


def test_collect_never_removes_an_entry_published_concurrently(tmp_path):
    """Publishers (never corrupted), a churn thread that corrupts, reads and republishes other
    entries, and a collector that ages quarantined objects at 0.2 s."""
    store = aged_store(tmp_path, staging=3600, quarantine=0.2)
    n = 6
    stop = threading.Event()
    errors: list[BaseException] = []

    def guard(fn):
        def run():
            try:
                fn()
            except BaseException as err:  # noqa: BLE001
                errors.append(err)
                stop.set()

        return run

    def publisher():
        for _ in range(40):
            for i in range(n):
                c = publish(store, {"a": i}, checksum=f"python-keep{i}")
                assert c.stored

    def churn():
        j = 0
        while not stop.is_set():
            checksum = f"python-churn{j % 3}"
            c = publish(store, {"a": 1}, checksum=checksum)
            try:
                corrupt(entry_path(store.path, c))
            except (TypeError, OSError):
                pass
            try:
                store.read(c.entry, None)
            except (CorruptEntry, FileNotFoundError):
                pass
            j += 1

    def collector():
        while not stop.is_set():
            store.collect()

    pubs = [threading.Thread(target=guard(publisher)) for _ in range(3)]
    side = [threading.Thread(target=guard(f)) for f in (churn, collector)]
    for t in pubs + side:
        t.start()
    for t in pubs:
        t.join()
    stop.set()
    for t in side:
        t.join()
    assert not errors, errors
    for i in range(n):
        c = publish(store, {"a": i}, checksum=f"python-keep{i}")
        assert c.stored
        assert store.read(c.entry, None).outputs == {"a": i}


# -- live writers, vanishing stages and the fence -------------------------------------------------


class FakeClock:
    """The store's clock, advanced by the test. The writer's touch stamps its stage with this
    clock, so a stage's age is the fake time since it was last touched, with no real wait.
    """

    def __init__(self, monkeypatch):
        import pydra.engine.portable.store as mod

        self.t = time.time() + SKEW
        real_utime = os.utime

        def utime(path, times=None, **kw):
            if times is None:  # the writer's touch: "now"
                times = (self.t, self.t)
            return real_utime(path, times, **kw)

        monkeypatch.setattr(mod, "_now", lambda: self.t)
        monkeypatch.setattr(os, "utime", utime)
        self.touchers = []
        init = mod._Toucher.__init__

        def record(toucher, *a, **k):
            init(toucher, *a, **k)
            self.touchers.append(toucher)

        monkeypatch.setattr(mod._Toucher, "__init__", record)
        # the toucher's thread does nothing: the test calls ``touch`` at the moments it chooses
        monkeypatch.setattr(mod._Toucher, "_run", lambda toucher: None)


def stage_that(monkeypatch, store, during):
    """Run ``during(n)`` (n = 1, 2, ...) after each ``stage`` call finished encoding."""
    fmt = type(store.format)
    real = fmt.stage
    calls = {"n": 0}

    def stage(self, staging, *a, **k):
        d = real(self, staging, *a, **k)
        calls["n"] += 1
        during(calls["n"])
        return d

    monkeypatch.setattr(fmt, "stage", stage)
    return calls


def test_a_live_writer_is_not_collected_during_a_long_copy(tmp_path, monkeypatch):
    """A stage touched within the staging age is never collected, however long the copy
    takes in fake time. Every step moves the clock 0.9 of the staging age and runs `collect`
    before the touch, so only the touch keeps the stage young."""
    staging_age = 100.0
    store = aged_store(tmp_path, staging=staging_age, quarantine=3600)
    clock = FakeClock(monkeypatch)
    removed: list[str] = []
    steps = 30  # 27 staging ages of fake time

    def long_copy(n):
        (toucher,) = clock.touchers
        toucher.touch()  # the encoding wrote real mtimes into the stage; now it is stamped
        for _ in range(steps):
            clock.t += 0.9 * staging_age
            report = store.collect()
            removed.extend(report.removed)
            toucher.touch()

    calls = stage_that(monkeypatch, store, long_copy)
    c = publish(store, {"a": 7}, checksum="python-slow")
    assert (c.stored, c.won) == (True, True)
    assert calls["n"] == 1  # never collected, so never staged again
    assert removed == []
    assert store.read(c.entry, None).outputs == {"a": 7}


def test_a_stage_left_untouched_past_the_staging_age_is_collected(
    tmp_path, monkeypatch
):
    staging_age = 100.0
    store = aged_store(tmp_path, staging=staging_age, quarantine=3600)
    clock = FakeClock(monkeypatch)
    removed: list[str] = []

    def stalled(n):
        if n == 1:
            clock.touchers[0].touch()  # the last touch before the writer stalled
            clock.t += 1.1 * staging_age  # the writer stalled: no touch
            removed.extend(store.collect().removed)

    calls = stage_that(monkeypatch, store, stalled)
    c = publish(store, {"a": 8}, checksum="python-stalled")
    assert len(removed) == 1  # its staging directory
    assert calls["n"] == 2  # the writer found the stage gone and staged again
    assert (c.stored, c.won) == (True, True)
    assert store.read(c.entry, None).outputs == {"a": 8}


def test_the_toucher_thread_touches_every_interval():
    from pydra.engine.portable.store import _Toucher

    touches = []

    class Waits:
        def __init__(self):
            self.n = 0
            self.intervals = []

        def wait(self, interval):
            self.intervals.append(interval)
            self.n += 1
            return self.n > 3  # three intervals elapse, then the writer stops

    toucher = _Toucher.__new__(_Toucher)
    toucher._interval = 2.5
    toucher._stop = waits = Waits()
    toucher.touch = lambda: touches.append(1)
    toucher._run()
    assert len(touches) == 3
    assert set(waits.intervals) == {2.5}


def vanishing_stage(monkeypatch, store, times):
    import shutil

    fmt = type(store.format)
    real = fmt.stage
    calls = {"n": 0}

    def stage(self, staging, *a, **k):
        d = real(self, staging, *a, **k)
        calls["n"] += 1
        if calls["n"] <= times:
            for dirpath, _dirs, files in os.walk(staging.path):
                for f in files:
                    os.chmod(
                        os.path.join(dirpath, f), 0o600
                    )  # Windows: read-only files
            shutil.rmtree(staging.path)  # a collector claimed and removed it
        return d

    monkeypatch.setattr(fmt, "stage", stage)
    return calls


def test_a_stage_that_vanished_is_staged_again_once(tmp_path, monkeypatch):
    store = make(tmp_path)
    calls = vanishing_stage(monkeypatch, store, 1)
    c = publish(store)
    assert (c.won, c.stored, c.kept) == (True, True, None)
    assert calls["n"] == 2
    assert store.read(c.entry, None).outputs == OUT


def test_a_stage_that_vanishes_twice_raises_not_a_phantom_kept(tmp_path, monkeypatch):
    from pydra.utils.storagedir import StoreUnavailable

    store = make(tmp_path)
    calls = vanishing_stage(monkeypatch, store, 2)
    with pytest.raises(StoreUnavailable, match="vanished twice"):
        publish(store)
    assert calls["n"] == 2  # staged again once, not a third time


def test_a_stage_that_vanishes_mid_encode_is_staged_again_once(tmp_path, monkeypatch):
    import shutil

    store = make(tmp_path)
    fmt = type(store.format)
    real = fmt.stage
    calls = {"n": 0}

    def stage(self, staging, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            os.chmod(staging.path, 0o700)
            shutil.rmtree(staging.path)  # collected before the first member is written
        return real(self, staging, *a, **k)

    monkeypatch.setattr(fmt, "stage", stage)
    c = publish(store)
    assert (c.won, c.stored, c.kept) == (True, True, None) and calls["n"] == 2
    assert store.read(c.entry, None).outputs == OUT


@pytest.mark.skipif(IS_WINDOWS, reason="a rename changes ctime on POSIX only")
def test_an_object_moved_a_moment_ago_is_young_whatever_its_mtime(tmp_path, skew):
    """A rename changes ctime and not mtime; the age is the later of the two."""
    store = aged_store(tmp_path, staging=100, quarantine=2 * SKEW)
    c = publish(store)
    entry = entry_path(store.path, c)
    t = time.time() - 2 * 86400
    os.utime(entry, (t, t))  # commit-time mtime, two days old; ctime is now
    (Path(store.path) / "quarantine").mkdir(exist_ok=True)
    moved = Path(store.path) / "quarantine" / "x"
    os.rename(entry, moved)
    assert moved.stat().st_mtime < time.time() - 86400
    report = store.collect()  # the clock is SKEW ahead of the ctime, within the age
    assert report.removed == []
    assert str(moved) in [w for w, _ in report.kept]


def test_the_fence_restores_a_valid_entry_while_a_collector_runs(tmp_path, monkeypatch):
    """A collector between the move and the stamp does not take the moved copy."""
    store = aged_store(tmp_path, staging=3600, quarantine=3600)
    c = publish(store)
    entry = entry_path(store.path, c)
    t = time.time() - 2 * 86400
    for p in [entry, *entry.rglob("*")]:
        os.utime(p, (t, t))  # no links inside an entry, so portable
    real = PortableStore._after_move
    removed: list[str] = []

    def after_move(self, *a, **k):
        removed.extend(self.collect().removed)
        return real(self, *a, **k)

    monkeypatch.setattr(PortableStore, "_after_move", after_move)
    fmt = type(store.format)
    real_read = fmt.read
    calls = {"n": 0}

    def read(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise CorruptEntry("transient")
        return real_read(self, *a, **k)

    monkeypatch.setattr(fmt, "read", read)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert removed == []
    assert store.read(c.entry, None).outputs == OUT


@pytest.mark.parametrize(
    "staging, quarantine, staging_goes, quarantine_goes",
    [(100, 5000, True, False), (5000, 100, False, True)],
)
def test_staging_and_quarantine_have_their_own_ages(
    tmp_path, skew, staging, quarantine, staging_goes, quarantine_goes
):
    store = aged_store(tmp_path, staging=staging, quarantine=quarantine)
    c = publish(store)
    root = Path(store.path)
    parent = entry_path(root, c).parent
    (root / "staging" / "s").mkdir()
    (parent / "q-1").mkdir()
    (root / "quarantine" / "x").mkdir(parents=True, exist_ok=True)
    for p in (root / "staging" / "s", parent / "q-1", root / "quarantine" / "x"):
        age(p, 1000)
    store.collect()
    assert (not (root / "staging" / "s").exists()) is staging_goes
    assert (not (parent / "q-1").exists()) is quarantine_goes
    assert (not (root / "quarantine" / "x").exists()) is quarantine_goes


def test_wrong_digest_evict_never_renames(tmp_path, monkeypatch):
    store = make(tmp_path)
    c = publish(store)
    calls: list = []
    recording_rename(monkeypatch, calls)
    store.evict(c.entry, "0" * 64)
    assert calls == []


def test_evict_moves_only_the_entry_it_read(tmp_path, monkeypatch):
    """The entry is replaced after the evict read it: nothing of the new one is moved."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(store.path, c)
    real = PortableStore._quarantine
    calls: list = []

    def replaced(self, root, parent, name, cdir, identity, expected=None):
        assert identity is not None
        assert identity.st_ino == os.stat(entry).st_ino
        os.rename(entry, entry.parent / "elsewhere")
        publish(self, OTHER)
        recording_rename(monkeypatch, calls)
        return real(self, root, parent, name, cdir, identity, expected)

    monkeypatch.setattr(PortableStore, "_quarantine", replaced)
    store.evict(c.entry, c.own_digest)
    assert not [call for call in calls if "quarantine" in call[2]]
    assert store.read(c.entry, None).outputs == OTHER


def test_evict_of_a_wrong_kind_object_sets_it_aside(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(store.path, c)
    corrupt(entry)
    os.rename(entry, entry.parent / "elsewhere")
    entry.write_text("not a directory")
    store.evict(c.entry, c.own_digest)
    assert not entry.exists() and len(quarantined(store.path)) == 1


def test_collect_removes_a_gc_leftover_in_a_container(tmp_path, skew):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    c = publish(store)
    parent = entry_path(store.path, c).parent
    (parent / "gc-abc").mkdir()
    age(parent / "gc-abc", 1000)
    assert store.collect().removed == [str(parent / "gc-abc")]


def test_collect_removes_the_aside_names_the_format_declares(
    tmp_path, skew, monkeypatch
):
    import re

    store = aged_store(tmp_path, staging=100, quarantine=100)
    c = publish(store)
    parent = entry_path(store.path, c).parent
    monkeypatch.setattr(
        type(store.format), "aside_names", lambda self: re.compile(r"aside-.*")
    )
    (parent / "aside-1").mkdir()
    (parent / "other-1").mkdir()
    for p in (parent / "aside-1", parent / "other-1"):
        age(p, 1000)
    assert store.collect().removed == [str(parent / "aside-1")]


def test_collect_reports_what_it_cannot_remove_with_its_owner(
    tmp_path, skew, monkeypatch
):
    import pydra.engine.portable.store as mod

    store = aged_store(tmp_path, staging=100, quarantine=100)
    publish(store)
    stage = Path(store.path) / "staging" / "dead"
    stage.mkdir()
    age(stage, 1000)

    def fail(parent, name):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(mod, "_remove_tree", fail)
    report = store.collect()
    assert report.removed == []
    ((where, why),) = [k for k in report.kept if k[0] == str(stage)] or [(None, None)]
    assert where == str(stage) and "owner:" in why


def test_collect_sweeps_only_its_own_files_at_the_root(tmp_path, skew):
    store = aged_store(tmp_path, staging=100, quarantine=100)
    publish(store)
    root = Path(store.path)
    (root / "notes.txt").write_text("x")
    age(root / "notes.txt", 1000)
    age(root / "pydra-store.json", 1000)
    assert store.collect().removed == []
    assert (root / "notes.txt").is_file() and (root / "pydra-store.json").is_file()


def test_set_aside_stamps_the_moved_object(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(store.path, c)
    os.rename(entry, entry.parent / "elsewhere")
    entry.write_text("x")
    old = 1_000_000_000
    os.utime(entry, (old, old))
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    (moved,) = quarantined(store.path)
    assert moved.stat().st_mtime > old + 1000


@pytest.mark.parametrize("refusals, removed", [(3, True), (4, False)])
def test_a_sharing_violation_is_retried_on_windows(monkeypatch, refusals, removed):
    import attrs
    import pydra.engine.portable.store as mod

    monkeypatch.setattr(mod, "IS_WINDOWS", True)
    monkeypatch.setattr(
        mod, "SHARING_RETRY", attrs.evolve(mod.SHARING_RETRY, first=0, cap=0)
    )
    calls = []

    def remove(name):
        calls.append(name)
        if len(calls) <= refusals:
            raise PermissionError(13, "in use", name)

    if removed:
        mod._retrying(remove, "x")
    else:
        with pytest.raises(PermissionError):
            mod._retrying(remove, "x")
    assert len(calls) == min(refusals + 1, mod.SHARING_RETRY.attempts)
