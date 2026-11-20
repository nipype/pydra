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


def age(path, seconds):
    t = time.time() - seconds
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


def test_collect_removes_aged_staging_quarantine_q_and_aside(tmp_path, monkeypatch):
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
    # a quarantined entry, a q- name and an aside name
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


def test_collect_spares_live_stage(tmp_path):
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


def test_collect_never_removes_entries(tmp_path):
    store = aged_store(tmp_path, staging=1, quarantine=1)
    cs = [publish(store, OUT, checksum=f"python-e{i}") for i in range(5)]
    for p in Path(store.path).rglob("*"):
        age(p, 10_000)
    # a corrupt entry is still an entry: only a reader quarantines it
    corrupt(entry_path(store.path, cs[0]))
    age(entry_path(store.path, cs[0]), 10_000)
    store.collect()
    for c in cs[1:]:
        assert store.read(c.entry, None).outputs == OUT
    assert entry_path(store.path, cs[0]).is_dir()


def test_collect_never_touches_sidecars(tmp_path):
    store = aged_store(tmp_path, staging=1, quarantine=1)
    publish(store)
    root = Path(store.path)
    side = root / "sidecars"
    (side / "run" / "q-1").mkdir(parents=True)
    (side / "run" / "q-1" / "f").write_text("x")
    (side / "note").write_text("x")
    for p in (side, side / "run", side / "run" / "q-1", side / "note"):
        age(p, 10_000)
    report = store.collect()
    assert (side / "run" / "q-1" / "f").is_file() and (side / "note").is_file()
    assert not any("sidecars" in w for w in report.removed)


def test_collect_leaves_a_claimed_object_another_collector_took(tmp_path, monkeypatch):
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
def test_collect_removes_a_sealed_quarantined_entry(tmp_path):
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
