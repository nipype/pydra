"""PortableStore publish and read (RFC 0003 sections 5.1, 5.2, 6.2 and 7.2)."""

import errno
import functools
import os
import pickle
import stat
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import cloudpickle
import pytest

from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.portable.entry import (
    CorruptEntry,
    EntryNotTrusted,
    EntryRef,
    PydraEntryV1,
)
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.layout import HEADER_NAME, StoreLayoutMismatch, cdir_for
from pydra.engine.portable.reuse import PolicyReuse, Verdict
from pydra.engine.portable.store import Committed, PortableStore, StoreNotWritable
from pydra.engine.result import Result
from pydra.engine.submitter import Submitter
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
)

# The trust of the roots these tests open: on Windows `private` holds only under %LOCALAPPDATA%,
# so it is `trusted` there unless CI sets PYDRA_TEST_PORTABLE_TRUST (its temp directory is there)
TRUST = os.environ.get("PYDRA_TEST_PORTABLE_TRUST") or (
    "trusted" if IS_WINDOWS else "private"
)
posix_only = pytest.mark.skipif(
    IS_WINDOWS, reason="needs POSIX modes, owners and signals"
)
IDENT = PydraIdentityV1()
CTX = {"python": "3.12", "platform": "test", "n": 1}
CTX2 = {"python": "3.12", "platform": "test", "n": 2}
OUT = {"a": 1, "b": [1, 2], "c": "x"}


class RetryClock:
    """The retry primitive's clock: it moves only by the waits it records"""

    def __init__(self):
        self.now = 0.0
        self.waits: list[float] = []
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(time, name)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        with self._lock:
            self.waits.append(seconds)
            self.now += seconds


@pytest.fixture
def retry_clock(monkeypatch):
    clock = RetryClock()
    monkeypatch.setattr("pydra.utils.retry.time", clock)
    return clock


class Refusing(PolicyReuse):
    """A hook that serves nothing: a job that loses a race keeps its own outputs (RFC 0003
    section 7.3, third row)."""

    def __init__(self) -> None:
        from pydra.engine.store import ConsistentReuse

        super().__init__(ConsistentReuse())

    def verdict(self, candidate, current, task):
        return Verdict.REFUSE


def _work_under(base, checksum, fingerprint, attempt):
    return Path(base) / "work" / attempt


def make(tmp_path, name="root", **kw):
    kw.setdefault("trust", TRUST)
    if kw["trust"] == "shared":  # a shared root has no default work directory
        kw.setdefault("work_dir", functools.partial(_work_under, tmp_path))
    return PortableStore(tmp_path / name, **kw)


def publish(store, outputs=OUT, checksum="python-abc", context=CTX, **kw):
    return store.publish(checksum, None, context, outputs, None, None, **kw)


def entry_path(store_root, committed):
    ref = committed.entry
    shard = None
    for d in (Path(store_root) / "results@1").iterdir():
        if (d / ref.cdir).exists():
            shard = d.name
    return Path(store_root) / "results@1" / shard / ref.cdir / ref.fingerprint


def staging_names(root):
    return sorted(p.name for p in (Path(root) / "staging").iterdir())


# -- publish and read ------------------------------------------------------------------------


def test_publish_and_read_round_trip(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    assert isinstance(c, Committed)
    assert (c.won, c.stored, c.kept) == (True, True, None)
    assert c.committed_digest == c.own_digest
    assert c.entry == EntryRef(
        str(tmp_path / "root"), "python-abc", IDENT.fingerprint(CTX)
    )
    assert staging_names(tmp_path / "root") == []
    result = store.read(c.entry, None)
    assert result.outputs == OUT
    assert result.errored is False and result.stored is True
    assert result.context == CTX
    assert Path(result.cache_dir) == entry_path(tmp_path / "root", c)


@posix_only
def test_modes_come_from_the_policy_not_the_umask(tmp_path):
    old = os.umask(0o077)
    try:
        store = make(tmp_path)
        c = publish(store)
    finally:
        os.umask(old)
    root = tmp_path / "root"
    entry = entry_path(root, c)
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    assert mode(entry) == 0o755
    assert mode(entry.parent) == 0o755 and mode(entry.parent.parent) == 0o755
    assert mode(root / "results@1") == 0o755
    assert {mode(p) for p in entry.iterdir() if p.is_file()} == {0o444}
    assert mode(root / "staging") == 0o755 and mode(root / "quarantine") == 0o755


def test_unencodable_output_leaves_nothing(tmp_path):
    from pydra.engine.portable.errors import UnencodableOutput

    store = make(tmp_path)
    with pytest.raises(UnencodableOutput):
        publish(store, {"a": object()})
    assert staging_names(tmp_path / "root") == []
    assert not list((tmp_path / "root" / "results@1").iterdir())


def test_read_absent_entry_is_file_not_found(tmp_path):
    store = make(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.read(EntryRef(str(tmp_path / "root"), "python-nope", "0" * 64), None)


def test_read_refuses_another_root(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    with pytest.raises(ValueError, match="root"):
        store.read(
            EntryRef(str(tmp_path / "elsewhere"), c.entry.cdir, c.entry.fingerprint),
            None,
        )


def test_store_round_trips_through_pickle(tmp_path):
    store = make(tmp_path)
    clone = cloudpickle.loads(cloudpickle.dumps(store))
    c = publish(clone)
    assert store.read(c.entry, None).outputs == OUT


def test_header_mismatch_on_open(tmp_path):
    make(tmp_path)
    with pytest.raises(StoreLayoutMismatch):
        make(tmp_path, identity=PydraIdentityV1(recorded_only=["n"]))


def test_root_synced_once_when_created(tmp_path, monkeypatch):
    calls = []
    real = StorageDir.sync

    def sync(self):
        calls.append(self.path)
        return real(self)

    monkeypatch.setattr(StorageDir, "sync", sync)
    root = str(tmp_path / "root")
    make(tmp_path)
    assert calls.count(root) == 1
    calls.clear()
    make(tmp_path)
    assert calls.count(root) == 0


# -- races ----------------------------------------------------------------------------------


def test_race_equal_digests_both_stored(tmp_path):
    store = make(tmp_path)
    n = 6
    barrier = threading.Barrier(n)
    out: list = [None] * n

    def go(i):
        barrier.wait()
        out[i] = publish(store)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert [c.won for c in out].count(True) == 1
    assert all(c.stored and c.kept is None for c in out)
    assert len({c.committed_digest for c in out}) == 1
    assert all(c.committed_digest == c.own_digest for c in out)
    assert staging_names(tmp_path / "root") == []


def test_race_unequal_digests_loser_kept(tmp_path):
    store = make(tmp_path)
    first = publish(store, {"a": 1})
    second = publish(store, {"a": 2})
    assert second.won is False and second.stored is False
    assert second.committed_digest == first.own_digest != second.own_digest
    assert second.kept is not None and os.path.isdir(second.kept)
    assert os.path.isfile(os.path.join(second.kept, "entry.json"))
    assert store.read(second.entry, None).outputs == {"a": 1}
    assert staging_names(tmp_path / "root") == [os.path.basename(second.kept)]


def test_loser_of_an_unreadable_occupant_keeps_its_entry(tmp_path):
    if IS_WINDOWS:
        pytest.skip("a shared root is not supported on Windows")
    writer = make(tmp_path, trust="trusted", pickle=True)
    publish(writer, {"a": object()})
    reader = make(tmp_path, trust="shared")
    c = publish(reader, {"a": 1})
    assert (c.won, c.stored, c.committed_digest) == (False, False, None)
    assert c.kept is not None and os.path.isdir(c.kept)
    with pytest.raises(EntryNotTrusted):
        reader.read(c.entry, None)
    assert entry_path(tmp_path / "root", c).is_dir()  # left in place


@posix_only
def test_wrong_kind_at_the_address_is_set_aside(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    real = tmp_path / "elsewhere"
    entry.rename(real)
    entry.symlink_to(real)
    again = publish(store)
    assert again.won is True and again.stored is True
    assert not entry.is_symlink()
    assert store.read(again.entry, None).outputs == OUT
    assert len(list((tmp_path / "root" / "quarantine").iterdir())) == 1
    assert real.is_dir()  # the link was moved, not followed


def test_corrupt_occupant_is_quarantined_and_replaced(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    meta = entry / "entry.json"
    os.chmod(meta, 0o644)
    meta.write_text("not json")
    again = publish(store)
    assert again.won is True and again.stored is True
    assert store.read(again.entry, None).outputs == OUT
    assert len(list((tmp_path / "root" / "quarantine").iterdir())) == 1


def test_read_quarantines_a_corrupt_entry_and_raises(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    meta = entry / "entry.json"
    os.chmod(meta, 0o644)
    meta.write_text("not json")
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert not entry.exists()
    assert len(list((tmp_path / "root" / "quarantine").iterdir())) == 1
    with pytest.raises(FileNotFoundError):
        store.read(c.entry, None)


def test_read_does_not_quarantine_for_a_task_mismatch(tmp_path):
    store = make(tmp_path)
    c = publish(store)  # outputs a, b, c; the task below declares out
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        with pytest.raises(StoreUnavailable, match="outputs"):
            store.read(c.entry, job.task)
    assert entry_path(tmp_path / "root", c).is_dir()
    assert list((tmp_path / "root" / "quarantine").iterdir()) == []


@python.define
def _Inc(a: int) -> int:
    return a + 1


def test_read_builds_the_tasks_outputs(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        c = publish(store, {"out": 2})
        result = store.read(c.entry, job.task)
    assert result.outputs.out == 2
    assert result.task is job.task
    assert Path(result.cache_dir) == entry_path(tmp_path / "root", c)


@posix_only
def test_a_container_replaced_before_the_rename_is_walked_again(tmp_path, monkeypatch):
    """A rename onto an empty directory replaces it, so a writer can hold a container that
    another writer has replaced: its entry must not be left in the orphan."""
    store = make(tmp_path)
    real = StorageDir.rename
    done = []

    def rename(self, name, target, new_name):
        if len(new_name) == 64 and not done:
            done.append(1)
            live = Path(target.path)
            os.rename(live, live.with_name("orphan"))
            live.mkdir()
            os.chmod(live, 0o755)
        real(self, name, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)
    assert store.read(c.entry, None).outputs == OUT
    orphan = entry_path(tmp_path / "root", c).parent.with_name("orphan")
    assert list(orphan.iterdir()) == []
    assert staging_names(tmp_path / "root") == []


# -- kills -------------------------------------------------------------------------------------

SCRIPT = textwrap.dedent("""
    import os, signal, sys
    from pydra.engine.portable.store import PortableStore
    from pydra.utils.storagedir import StorageDir, StorePolicy
    root, when = sys.argv[1], sys.argv[2]
    policy = StorePolicy(sealed_mode=0o555) if when == "seal" else StorePolicy()
    store = PortableStore(root, trust="private", policy=policy)

    def die():
        os.kill(os.getpid(), signal.SIGKILL)

    real_rename, real_mode = StorageDir.rename, StorageDir.set_mode

    def rename(self, name, target, new_name):
        at_address = len(new_name) == 64
        if at_address and when == "before":
            die()
        real_rename(self, name, target, new_name)
        if at_address and when == "after":
            die()

    def set_mode(self, mode, name=None):
        if when == "seal" and name is not None and mode == 0o555:
            die()
        return real_mode(self, mode, name)

    StorageDir.rename, StorageDir.set_mode = rename, set_mode
    store.publish("python-abc", None, {"python": "3.12", "platform": "test", "n": 1},
                  {"a": 1, "b": [1, 2], "c": "x"}, None, None)
    """)


def killed(tmp_path, when):
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "root"), when],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == -9, proc.stderr
    return PortableStore(
        tmp_path / "root",
        trust=TRUST,
        policy=StorePolicy(sealed_mode=0o555) if when == "seal" else StorePolicy(),
    )


@posix_only
def test_sigkill_before_rename_leaves_no_entry(tmp_path):
    store = killed(tmp_path, "before")
    ref = EntryRef(str(tmp_path / "root"), "python-abc", IDENT.fingerprint(CTX))
    with pytest.raises(FileNotFoundError):
        store.read(ref, None)
    assert (
        len(staging_names(tmp_path / "root")) == 1
    )  # the dead writer's stage, for collect
    c = publish(store)
    assert c.won and c.stored
    assert store.read(c.entry, None).outputs == OUT


@posix_only
def test_sigkill_after_rename_entry_readable(tmp_path):
    store = killed(tmp_path, "after")
    ref = EntryRef(str(tmp_path / "root"), "python-abc", IDENT.fingerprint(CTX))
    assert store.read(ref, None).outputs == OUT
    c = publish(store)
    assert (c.won, c.stored, c.committed_digest == c.own_digest) == (False, True, True)
    assert (
        staging_names(tmp_path / "root") == []
    )  # the dead writer's stage was its entry


@posix_only
def test_sigkill_between_rename_and_seal_resealed_by_owner(tmp_path):
    store = killed(tmp_path, "seal")
    ref = EntryRef(str(tmp_path / "root"), "python-abc", IDENT.fingerprint(CTX))
    entry = entry_path(tmp_path / "root", Committed(ref, True, "", None, True, None))
    try:
        assert (
            stat.S_IMODE(entry.stat().st_mode) == 0o755
        )  # visible, readable, not sealed
        assert store.read(ref, None).outputs == OUT
        assert stat.S_IMODE(entry.stat().st_mode) == 0o555  # the reader owns it: sealed
    finally:
        os.chmod(entry, 0o755)


def test_lost_rename_reply_detected_by_token(tmp_path, monkeypatch):
    store = make(tmp_path)
    real = StorageDir.rename

    def rename(self, name, target, new_name):
        real(self, name, target, new_name)
        if len(new_name) == 64:
            raise OSError(errno.ESTALE, "the reply was lost")

    monkeypatch.setattr(StorageDir, "rename", rename)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)
    assert c.committed_digest == c.own_digest and c.kept is None
    assert staging_names(tmp_path / "root") == []
    assert store.read(c.entry, None).outputs == OUT


def test_a_failed_rename_with_no_occupant_is_raised(tmp_path, monkeypatch):
    store = make(tmp_path)
    real = StorageDir.rename

    def rename(self, name, target, new_name):
        if len(new_name) == 64:
            raise OSError(errno.EIO, "disk")
        real(self, name, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    with pytest.raises(StoreUnavailable):
        publish(store)
    assert staging_names(tmp_path / "root") == []


# -- who can write ---------------------------------------------------------------------------------


@posix_only
@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="root ignores modes")
def test_second_user_default_root_store_not_writable(tmp_path):
    store = make(tmp_path)
    publish(store)
    # another user of a default root cannot write below the creator's 0o755 directories; the
    # same denial, made by the kernel, is a staging directory without write permission
    staging = tmp_path / "root" / "staging"
    os.chmod(staging, 0o555)
    try:
        with pytest.raises(StoreNotWritable) as info:
            publish(store, {"a": 9}, checksum="python-other")
    finally:
        os.chmod(staging, 0o755)
    message = str(info.value)
    assert str(staging) in message and "0o555" in message
    assert "container" in message and "side" in message and "0o755" in message
    assert isinstance(info.value, StoreUnavailable)


@posix_only
def test_second_user_denied_at_the_container_not_taken_for_a_lost_race(
    tmp_path, monkeypatch
):
    store = make(tmp_path)
    real = StorageDir.child

    def child(self, name, *, create=False, mode=None):
        if create and self.path.endswith("staging"):
            raise PermissionError(errno.EACCES, "denied", self.path)
        return real(self, name, create=create, mode=mode)

    monkeypatch.setattr(StorageDir, "child", child)
    with pytest.raises(StoreNotWritable):
        publish(store)


@posix_only
def test_group_writable_root_second_writer(tmp_path):
    policy = StorePolicy(container_mode=0o775, side_mode=0o775)
    old = os.umask(0o077)
    try:
        first = make(tmp_path, trust="shared", policy=policy)
        second = make(tmp_path, trust="shared", policy=policy)
        a = publish(first)
        b = publish(second, context=CTX2)  # same checksum directory, another entry
        c = publish(second, checksum="python-new")  # a new shard and checksum directory
    finally:
        os.umask(old)
    root = tmp_path / "root"
    assert all(x.won and x.stored for x in (a, b, c))
    mode = lambda p: stat.S_IMODE(p.stat().st_mode)  # noqa: E731
    for entry in (a, b, c):
        e = entry_path(root, entry)
        assert mode(e) == 0o755
        assert mode(e.parent) == 0o775 and mode(e.parent.parent) == 0o775
    assert mode(root / "results@1") == 0o775
    assert mode(root / "staging") == 0o775 and mode(root / "quarantine") == 0o775
    assert entry_path(root, a).parent == entry_path(root, b).parent
    assert second.read(a.entry, None).outputs == OUT


def test_a_different_policy_is_a_different_store(tmp_path):
    make(tmp_path)
    with pytest.raises(StoreLayoutMismatch):
        make(tmp_path, policy=StorePolicy(container_mode=0o775))


@posix_only
@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="root ignores modes")
def test_read_only_root_never_written(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    root = tmp_path / "root"

    def snapshot():
        return sorted(
            (str(p), p.lstat().st_mtime_ns, stat.S_IMODE(p.lstat().st_mode))
            for p in [root, *root.rglob("*")]
        )

    dirs = [root, *(p for p in root.rglob("*") if p.is_dir())]
    for d in dirs:
        os.chmod(d, 0o555)
    try:
        before = snapshot()
        ro = make(tmp_path, policy=StorePolicy(sealed_mode=None))
        assert ro.read(c.entry, None).outputs == OUT
        with pytest.raises(StoreNotWritable):
            publish(ro, {"a": 5}, checksum="python-other")
        assert snapshot() == before
    finally:
        for d in dirs:
            os.chmod(d, 0o755)


# -- addresses ----------------------------------------------------------------------------------


def test_case_colliding_supplied_checksums(tmp_path):
    store = make(tmp_path)
    names = ["abc", "ABC", "Abc"]
    refs = {n: publish(store, {"v": n}, checksum=n) for n in names}
    cdirs = {n: c.entry.cdir for n, c in refs.items()}
    assert cdirs["abc"] == "abc"
    assert cdirs["ABC"] == cdir_for("ABC") != cdirs["Abc"] == cdir_for("Abc")
    assert cdirs["ABC"].startswith("h-") and cdirs["Abc"].startswith("h-")
    assert len({c.lower() for c in cdirs.values()}) == 3  # no two differ only in case
    assert all(c.won and c.stored for c in refs.values())
    for n, c in refs.items():
        assert store.read(c.entry, None).outputs == {"v": n}


# -- the ResultStore adapter ----------------------------------------------------------------------


class _Lost:
    """A lease that was taken over while its job ran."""

    lost = "taken over"
    cache_dir = None

    def prove(self):
        return False


def _result(job, outputs, **kw):
    job.attempt = job.attempt or uuid.uuid4().hex  # a job handed out has one
    return Result(
        outputs=outputs,
        cache_dir=job.cache_dir,
        task=job.task,
        context=job.context,
        **kw,
    )


def test_lost_lease_still_commits(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        result = _result(job, _Inc.Outputs(out=2))
        store.commit(_Lost(), result, job)  # never LeaseLost
        assert result.committed is not None and result.committed.won
        assert result.stored is True
        again = _result(job, _Inc.Outputs(out=2))
        store.commit(_Lost(), again, job)
        assert again.committed.won is False and again.stored is True
        read = store.read(result.committed.entry, job.task)
    assert read.outputs.out == 2
    assert read.context == job.context
    assert result.committed.entry.cdir == cdir_for(job.checksum)


def test_commit_of_an_unequal_result_is_not_stored(tmp_path):
    store = make(tmp_path, reuse=Refusing())
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        store.commit(_Lost(), _result(job, _Inc.Outputs(out=2)), job)
        other = _result(job, _Inc.Outputs(out=3))
        store.commit(_Lost(), other, job)
    assert other.stored is False and other.committed.kept is not None


def test_commit_of_an_errored_result_publishes_nothing(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        result = _result(job, None, errored=True)
        store.commit(_Lost(), result, job)
    assert result.committed is None
    assert not list((tmp_path / "root" / "results@1").iterdir())


def test_result_pickles_with_and_without_the_new_fields(tmp_path):
    c = Committed(EntryRef("r", "c", "f"), True, "d", "d", True, None)
    r = Result(cache_dir=tmp_path, committed=c, stored=False)
    back = pickle.loads(pickle.dumps(r))
    assert back.committed == c and back.stored is False
    state = r.__getstate__()
    del state["committed"], state["stored"]
    old = Result.__new__(Result)
    old.__setstate__(state)
    assert old.committed is None and old.stored is True


# -- a job through a worker --------------------------------------------------------------------------


@python.define
def _PublishHere(root: str, value: int) -> str:
    from pydra.engine.portable.store import PortableStore

    store = PortableStore(root, trust="private" if os.name != "nt" else "trusted")
    c = store.publish(
        "python-job", None, {"python": "x", "n": 1}, {"v": value}, None, None
    )
    return "|".join([c.entry.cdir, c.entry.fingerprint, str(c.won), str(c.stored)])


def test_publish_and_read(any_worker, tmp_path):
    root = tmp_path / "root"
    PortableStore(root, trust=TRUST)  # the root is created here, in the test process
    with Submitter(worker=any_worker, cache_root=tmp_path / "pc") as sub:
        res = sub(_PublishHere(root=str(root), value=7))
    cdir, fingerprint, won, stored = res.outputs.out.split("|")
    assert (won, stored) == ("True", "True")
    store = PortableStore(root, trust=TRUST)
    got = store.read(EntryRef(str(root), cdir, fingerprint), None)
    assert got.outputs == {"v": 7}


# -- contention, occupants and wrong kinds at the address -------------------------------------------


def corrupt(entry: Path) -> None:
    """Make the entry at ``entry`` unreadable JSON, whatever mode it is sealed in."""
    mode = stat.S_IMODE(entry.stat().st_mode)
    os.chmod(entry, 0o755)
    os.chmod(entry / "entry.json", 0o644)
    (entry / "entry.json").write_text("not json")
    os.chmod(entry, mode)


def quarantined(root) -> list:
    return list((Path(root) / "quarantine").iterdir())


def recording_rename(monkeypatch, calls, fail=None):
    real = StorageDir.rename

    def rename(self, name, target, new_name):
        calls.append((self.path, name, target.path, new_name))
        if fail is not None:
            fail(self, name, target, new_name)
        real(self, name, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)


def at_address(call) -> bool:
    return len(call[3]) == 64


def test_contention_exhaustion_keeps_the_outputs(tmp_path, monkeypatch, retry_clock):
    """An address that stays contended is not an error and does not discard the job's
    outputs; the writer waits until the visibility timeout, and no longer."""
    store = make(tmp_path)
    monkeypatch.setattr(store, "visibility_timeout", 0.3)

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)):
            raise OSError(errno.ENOENT, "the container was replaced")

    recording_rename(monkeypatch, [], fail)
    c = publish(store)
    assert (c.won, c.stored, c.committed_digest) == (False, False, None)
    assert c.kept is not None and os.path.isfile(os.path.join(c.kept, "entry.json"))
    assert len(retry_clock.waits) > 1 and sum(retry_clock.waits) == pytest.approx(0.3)


@posix_only
def test_corrupt_occupant_that_cannot_be_cleared_keeps_the_outputs(
    tmp_path, monkeypatch
):
    store = make(tmp_path)
    c = publish(store)
    corrupt(entry_path(tmp_path / "root", c))
    import pydra.engine.portable.store as mod

    monkeypatch.setattr(mod.PortableStore, "_quarantine", lambda *a, **k: None)
    again = publish(store)
    assert (again.won, again.stored, again.committed_digest) == (False, False, None)
    assert again.kept is not None and os.path.isdir(again.kept)


def test_an_occupant_that_vanishes_after_a_failed_rename_is_a_retry(
    tmp_path, monkeypatch
):
    """ENOTEMPTY, then the occupant is gone: the next attempt wins."""
    store = make(tmp_path)
    seen = []

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)) and not seen:
            seen.append(1)
            raise OSError(errno.ENOTEMPTY, "occupied, they say")

    recording_rename(monkeypatch, [], fail)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)


@posix_only
def test_clearing_an_occupant_someone_else_moved_is_a_retry(tmp_path, monkeypatch):
    """The occupant was quarantined by another party between the check and the move."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    real = PydraEntryV1.read
    moved = []

    def read(self, *a, **k):
        try:
            return real(self, *a, **k)
        except CorruptEntry:
            if not moved:
                moved.append(1)
                os.rename(entry, tmp_path / "taken")  # another reader got there first
            raise

    monkeypatch.setattr(PydraEntryV1, "read", read)
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert store.read(again.entry, None).outputs == OUT


@posix_only
def test_replaced_occupant_is_not_quarantined(tmp_path, monkeypatch):
    """The inode check: an entry that replaced the corrupt one is never moved."""
    import shutil

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    backup = tmp_path / "backup"
    shutil.copytree(entry, backup)
    corrupt(entry)
    real = PydraEntryV1.read
    swapped = []

    def read(self, *a, **k):
        try:
            return real(self, *a, **k)
        except CorruptEntry:
            if not swapped:
                swapped.append(1)
                shutil.rmtree(entry)
                shutil.copytree(backup, entry)  # a valid entry now, a new inode
            raise

    monkeypatch.setattr(PydraEntryV1, "read", read)
    calls: list = []
    recording_rename(monkeypatch, calls)
    again = publish(store)
    assert (again.won, again.stored) == (False, True)
    assert not [c for c in calls if "quarantine" in c[2]]
    assert quarantined(tmp_path / "root") == []


def test_quarantine_restores_a_valid_entry_it_moved(tmp_path, monkeypatch):
    """The fence (RFC 0003 6.3): the moved copy is re-verified and, valid, put back."""
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    with store._open() as root:
        with (
            root.child("results@1") as e,
            e.child(entry.parent.parent.name) as s,
            s.child(entry.parent.name) as d,
        ):
            assert (
                store._quarantine(root, d, c.entry.fingerprint, c.entry.cdir, None)
                is False
            )
    assert entry.is_dir() and quarantined(tmp_path / "root") == []
    assert store.read(c.entry, None).outputs == OUT


def test_quarantine_stamps_and_keeps_a_corrupt_entry(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    old = 1_000_000_000
    os.utime(entry, (old, old))
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    (moved,) = quarantined(tmp_path / "root")
    assert moved.stat().st_mtime > old + 1000  # aged from the move


@posix_only
@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="root ignores modes")
@pytest.mark.parametrize("sealed", [0o555, 0o2550])
def test_owner_quarantines_its_own_sealed_entry(tmp_path, sealed):
    policy = StorePolicy(sealed_mode=sealed)
    store = make(tmp_path, policy=policy)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    assert stat.S_IMODE(entry.stat().st_mode) & 0o200 == 0
    try:
        corrupt(entry)
        with pytest.raises(CorruptEntry):
            store.read(c.entry, None)
        assert len(quarantined(tmp_path / "root")) == 1
        again = publish(store)
        assert (again.won, again.stored) == (True, True)
        assert store.read(again.entry, None).outputs == OUT
    finally:
        for p in (tmp_path / "root").rglob("*"):
            if p.is_dir():
                os.chmod(p, 0o755)


def test_non_owner_moves_a_corrupt_entry_to_a_q_name(tmp_path, monkeypatch):
    """Cannot move across parents (no write on the entry): rename within the parent."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)

    def fail(self, name, target, new_name):
        if "quarantine" in target.path:
            raise PermissionError(errno.EACCES, "not the owner")

    recording_rename(monkeypatch, [], fail)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    names = [p.name for p in entry.parent.iterdir()]
    assert len(names) == 1 and names[0].startswith("q-")
    assert quarantined(tmp_path / "root") == []
    again = publish(store)
    assert (again.won, again.stored) == (True, True)


@pytest.mark.skipif(IS_WINDOWS, reason="a shared store is not supported")
def test_wrong_kind_regular_file_at_the_address(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    import shutil

    shutil.rmtree(entry)
    entry.write_text("not a directory")
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert len(quarantined(tmp_path / "root")) == 1  # renamed, not deleted


def test_wrong_kind_regular_file_at_the_address_any_platform(tmp_path):
    store = make(tmp_path, trust=TRUST)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    import shutil

    for p in entry.rglob("*"):
        os.chmod(p, 0o666)
    shutil.rmtree(entry)
    entry.write_text("not a directory")
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert len(quarantined(tmp_path / "root")) == 1


def test_wrong_kind_symlink_at_the_address_any_platform(tmp_path):
    store = make(tmp_path, trust=TRUST)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    real = tmp_path / "elsewhere"
    entry.rename(real)
    try:
        entry.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("no symlinks here")
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert real.is_dir() and any(
        real.iterdir()
    )  # the target is never followed or removed
    # a symlink is unlinked where the path backend cannot rename one, and moved elsewhere
    assert len(quarantined(tmp_path / "root")) == (0 if IS_WINDOWS else 1)


def test_set_aside_unlinks_only_links_on_windows(tmp_path, monkeypatch):
    """The Windows branch, run everywhere: a regular file is renamed, a link unlinked."""
    import pydra.engine.portable.store as mod

    store = make(tmp_path, trust=TRUST)
    publish(store)
    monkeypatch.setattr(mod, "IS_WINDOWS", True)
    unlinked = []
    real_unlink = StorageDir.unlink

    def unlink(self, name):
        unlinked.append(name)
        real_unlink(self, name)

    monkeypatch.setattr(StorageDir, "unlink", unlink)
    with store._open() as root:
        with root.child("staging") as stg:
            (tmp_path / "root" / "staging" / "plain").write_text("x")
            store._set_aside(root, stg, "plain")
            assert unlinked == []
            try:
                os.symlink(tmp_path, tmp_path / "root" / "staging" / "link")
            except OSError:
                pytest.skip("no symlinks here")
            store._set_aside(root, stg, "link")
            assert unlinked == ["link"]
    assert len(quarantined(tmp_path / "root")) == 1


def test_estale_on_the_entry_rename_walks_again(tmp_path, monkeypatch):
    """ESTALE is a re-walk from the root, not an error."""
    store = make(tmp_path)
    seen = []

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)) and not seen:
            seen.append(1)
            raise OSError(errno.ESTALE, "stale")

    recording_rename(monkeypatch, [], fail)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)
    assert staging_names(tmp_path / "root") == []


def test_estale_on_a_container_rename_walks_again_from_the_root(tmp_path, monkeypatch):
    store = make(tmp_path)
    calls: list = []
    seen = []

    def fail(self, name, target, new_name):
        if not at_address((None, None, None, new_name)) and not seen:
            seen.append(1)
            raise OSError(errno.ESTALE, "stale")

    recording_rename(monkeypatch, calls, fail)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)
    # the stale parent was dropped: the container was made again, not retried on the same handle
    assert len([x for x in calls if not at_address(x)]) >= 2


def test_container_denied_is_store_not_writable_naming_the_parent(
    tmp_path, monkeypatch
):
    store = make(tmp_path)

    def fail(self, name, target, new_name):
        if not at_address((None, None, None, new_name)):
            raise PermissionError(errno.EACCES, "denied")

    recording_rename(monkeypatch, [], fail)
    with pytest.raises(StoreNotWritable) as info:
        publish(store)
    assert "results@1" in str(info.value)


def test_refusal_with_no_occupant_is_raised_at_once(tmp_path, monkeypatch):
    store = make(tmp_path)
    calls: list = []

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)):
            raise PermissionError(errno.EACCES, "denied")

    recording_rename(monkeypatch, calls, fail)
    with pytest.raises(StoreNotWritable):
        publish(store)
    assert len([c for c in calls if at_address(c)]) == 1


def test_a_failed_entry_rename_is_attempted_once(tmp_path, monkeypatch):
    store = make(tmp_path)
    calls: list = []

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)):
            raise OSError(errno.EIO, "disk")

    recording_rename(monkeypatch, calls, fail)
    with pytest.raises(StoreUnavailable):
        publish(store)
    assert len([c for c in calls if at_address(c)]) == 1


@posix_only
@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="root ignores modes")
@pytest.mark.parametrize("what", ["dir", "entry.json"])
def test_unreadable_occupant_is_kept_not_raised(tmp_path, what):
    """A loser that cannot read the occupant keeps its outputs (RFC 5.1 step 8)."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    target = entry if what == "dir" else entry / "entry.json"
    os.chmod(target, 0)
    try:
        again = publish(store, {"a": 2})
    finally:
        os.chmod(target, 0o755 if what == "dir" else 0o444)
    assert (again.won, again.stored, again.committed_digest) == (False, False, None)
    assert again.kept is not None and os.path.isdir(again.kept)
    assert quarantined(tmp_path / "root") == []


def test_unreadable_occupant_identify_error_is_kept(tmp_path, monkeypatch):
    store = make(tmp_path)
    publish(store)

    def identify(self, entry):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(PydraEntryV1, "identify", identify)
    again = publish(store, {"a": 2})
    assert (again.won, again.stored, again.committed_digest) == (False, False, None)
    assert again.kept is not None


def test_loser_binds_the_occupant_to_the_address_it_lost(tmp_path):
    import json

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    meta = entry / "entry.json"
    os.chmod(meta, 0o644)
    doc = json.loads(meta.read_text())
    doc["checksum"] = "python-someone-else"
    meta.write_text(json.dumps(doc))
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert len(quarantined(tmp_path / "root")) == 1


def test_hashed_cdir_misfiling_is_refused(tmp_path):
    store = make(tmp_path)
    a = publish(store, {"v": 1}, checksum="ABC")
    other = cdir_for("Abc")
    src = entry_path(tmp_path / "root", a)
    from pydra.engine.portable.layout import shard_for

    dest = tmp_path / "root" / "results@1" / shard_for(other, "Abc", "sha256-2") / other
    dest.mkdir(parents=True)
    src.rename(dest / a.entry.fingerprint)
    from pydra.engine.portable.entry import MisfiledEntry

    with pytest.raises(MisfiledEntry):
        store.read(EntryRef(str(tmp_path / "root"), other, a.entry.fingerprint), None)
    assert len(quarantined(tmp_path / "root")) == 1


@posix_only
def test_read_of_a_wrong_kind_entry_sets_it_aside(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    real = tmp_path / "elsewhere"
    entry.rename(real)
    entry.symlink_to(real)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert not entry.exists() and real.is_dir()
    assert len(quarantined(tmp_path / "root")) == 1


@posix_only
def test_a_replaced_loser_container_missing_from_the_tree_is_walked_again(
    tmp_path, monkeypatch
):
    store = make(tmp_path)
    real = StorageDir.rename
    done = []

    def rename(self, name, target, new_name):
        if len(new_name) == 64 and not done:
            done.append(1)
            live = Path(target.path)
            os.rename(
                live, live.with_name("orphan")
            )  # gone from the tree, not replaced
        real(self, name, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)
    assert store.read(c.entry, None).outputs == OUT
    assert staging_names(tmp_path / "root") == []


@posix_only
def test_loser_reseals_a_valid_unsealed_occupant(tmp_path):
    store = killed(tmp_path, "seal")
    ref = EntryRef(str(tmp_path / "root"), "python-abc", IDENT.fingerprint(CTX))
    entry = entry_path(tmp_path / "root", Committed(ref, True, "", None, True, None))
    try:
        assert stat.S_IMODE(entry.stat().st_mode) == 0o755
        again = publish(store)
        assert (again.won, again.stored) == (False, True)
        assert stat.S_IMODE(entry.stat().st_mode) == 0o555
    finally:
        os.chmod(entry, 0o755)


@posix_only
def test_staging_mode_while_writing_and_syncs(tmp_path, monkeypatch):
    modes = []
    real_stage = PydraEntryV1.stage

    def stage(self, staging, *a, **k):
        modes.append(stat.S_IMODE(os.stat(staging.path).st_mode))
        return real_stage(self, staging, *a, **k)

    monkeypatch.setattr(PydraEntryV1, "stage", stage)
    synced: list = []
    real_sync = StorageDir.sync

    def sync(self):
        synced.append(self.path)
        return real_sync(self)

    monkeypatch.setattr(StorageDir, "sync", sync)
    store = make(tmp_path)
    synced.clear()
    c = publish(store)
    assert modes == [0o700]
    entry = entry_path(tmp_path / "root", c)
    assert str(entry.parent) in synced  # the parent, after the rename
    assert any(os.path.dirname(p) == str(tmp_path / "root" / "staging") for p in synced)

    none = make(tmp_path, "none", policy=StorePolicy(durability="none"))
    synced.clear()
    publish(none)
    assert synced == []
    syncs = []
    monkeypatch.setattr(os, "sync", lambda: syncs.append(1), raising=False)
    entry_store = make(tmp_path, "entry", policy=StorePolicy(durability="entry"))
    publish(entry_store)
    assert syncs == [1]


def test_commit_of_an_errored_result_with_outputs_publishes_nothing(tmp_path):
    store = make(tmp_path)
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        result = _result(job, _Inc.Outputs(out=2), errored=True)
        store.commit(_Lost(), result, job)
    assert result.committed is None
    assert not list((tmp_path / "root" / "results@1").iterdir())


def test_commit_uses_the_execute_context(tmp_path):
    phases = []

    class Recording(PydraIdentityV1):
        def context(self, job, phase):
            phases.append(phase)
            return super().context(job, phase)

    store = make(tmp_path, identity=Recording())
    with Submitter(worker="debug", cache_root=tmp_path / "pc") as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        store.commit(_Lost(), _result(job, _Inc.Outputs(out=2)), job)
    assert phases == ["execute"]


def test_result_defaults_to_stored(tmp_path):
    assert Result(cache_dir=tmp_path).stored is True
    assert Result(cache_dir=tmp_path).committed is None


# -- compare-and-move quarantine ----------------------------------------------------------------------


def rmtree(path) -> None:
    import shutil

    def fix(func, p, _exc):
        os.chmod(p, 0o777)
        func(p)

    shutil.rmtree(path, onerror=fix)


def inode_name(path) -> str:
    import uuid

    return uuid.UUID(int=os.stat(path).st_ino).hex


def open_parent(store, c):
    """The handles of the directory holding the entry of ``c``."""
    root = store._open()
    entries = root.child("results@1")
    shard = [
        d
        for d in (Path(store.path) / "results@1").iterdir()
        if (d / c.entry.cdir).exists()
    ]
    s = entries.child(shard[0].name)
    d = s.child(c.entry.cdir)
    return root, [entries, s, d]


def test_quarantine_target_is_named_from_the_inode(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    name = inode_name(entry)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert [p.name for p in quarantined(tmp_path / "root")] == [name]
    assert len(name) == 32


def test_q_name_is_derived_from_the_inode_too(tmp_path, monkeypatch):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    name = inode_name(entry)

    def fail(self, name, target, new_name):
        if "quarantine" in target.path:
            raise PermissionError(errno.EACCES, "not the owner")

    recording_rename(monkeypatch, [], fail)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert [p.name for p in entry.parent.iterdir()] == ["q-" + name]


def test_a_late_mover_cannot_take_a_valid_entry(tmp_path, monkeypatch):
    """The rename is the compare-and-move: a mover that verified the corrupt entry, and finds a
    valid one at the name by the time it renames, moves nothing."""
    store = make(tmp_path)
    other = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    name = inode_name(entry)
    real = StorageDir.rename
    done = []

    def rename(self, n, target, new_name):
        if "quarantine" in target.path and not done:
            done.append(1)
            real(self, n, target, new_name)  # a rival moved the corrupt entry first ...
            again = other.publish(
                "python-abc", None, CTX, OUT, None, None
            )  # ... a writer won
            assert again.won
            # and now this late mover renames: the name is taken, whatever is at the address
            with pytest.raises(OSError):
                real(self, n, target, new_name)
            raise OSError(errno.ENOTEMPTY, "taken")
        real(self, n, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is False
    assert entry.is_dir() and store.read(c.entry, None).outputs == OUT
    assert [p.name for p in quarantined(tmp_path / "root")] == [name]


def test_restore_onto_an_occupied_address_is_not_a_failure(tmp_path, monkeypatch):
    """A moved entry that was another's valid one, and an address filled again meanwhile:
    False, never None (which a publisher reports as stored=False with no digest)."""
    import shutil

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    backup, backup2 = tmp_path / "b1", tmp_path / "b2"
    shutil.copytree(entry, backup)
    shutil.copytree(entry, backup2)
    corrupt(entry)
    real_rename, real_lstat = StorageDir.rename, StorageDir.lstat
    state = {"swapped": False, "filled": False}

    def rename(self, n, target, new_name):
        if "quarantine" in target.path and not state["swapped"]:
            state["swapped"] = True
            os.rename(
                entry, tmp_path / "evicted"
            )  # an outside mover took the corrupt entry
            shutil.copytree(backup, entry)  # a valid entry now has the name
        return real_rename(self, n, target, new_name)

    def lstat(self, n):
        if state["swapped"] and not state["filled"] and "quarantine" in self.path:
            state["filled"] = True
            shutil.copytree(
                backup2, entry
            )  # the mover took a valid entry; the address is filled
        return real_lstat(self, n)

    monkeypatch.setattr(StorageDir, "rename", rename)
    monkeypatch.setattr(StorageDir, "lstat", lstat)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is False
    assert entry.is_dir()


def test_mismatch_after_the_move_restores_without_reading(tmp_path, monkeypatch):
    import shutil

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    backup = tmp_path / "b1"
    shutil.copytree(entry, backup)
    corrupt(entry)
    real = StorageDir.rename
    swapped = []

    def rename(self, n, target, new_name):
        if "quarantine" in target.path and not swapped:
            swapped.append(1)
            os.rename(entry, tmp_path / "evicted")
            shutil.copytree(backup, entry)
        return real(self, n, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    reads = []
    real_read = PydraEntryV1.read

    def read(self, *a, **k):
        reads.append(1)
        return real_read(self, *a, **k)

    monkeypatch.setattr(PydraEntryV1, "read", read)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is False
    assert reads == []  # a different inode: put back at once, nothing verified
    assert entry.is_dir() and quarantined(tmp_path / "root") == []
    assert store.read(c.entry, None).outputs == OUT


def test_entry_replaced_before_the_move_is_left_alone(tmp_path, monkeypatch):
    import shutil

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    backup = tmp_path / "b1"
    shutil.copytree(entry, backup)
    corrupt(entry)
    real = StorageDir.lstat
    swapped = []

    def lstat(self, n):
        if (
            n == c.entry.fingerprint
            and not swapped
            and self.path.endswith(c.entry.cdir)
        ):
            swapped.append(1)
            os.rename(entry, tmp_path / "evicted")
            shutil.copytree(backup, entry)
        return real(self, n)

    calls: list = []
    recording_rename(monkeypatch, calls)
    monkeypatch.setattr(StorageDir, "lstat", lstat)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is False
    assert not [x for x in calls if "quarantine" in x[2]]


def test_a_copy_that_cannot_be_judged_is_restored(tmp_path, monkeypatch):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    real_read = PydraEntryV1.read

    def read(self, *a, **k):
        raise OSError(errno.EIO, "disk")

    monkeypatch.setattr(PydraEntryV1, "read", read)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    monkeypatch.setattr(PydraEntryV1, "read", real_read)
    assert r is False and entry.is_dir() and quarantined(tmp_path / "root") == []


@posix_only
@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="root ignores modes")
def test_a_restored_entry_is_sealed_again(tmp_path):
    store = make(tmp_path, policy=StorePolicy(sealed_mode=0o555))
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    try:
        root, handles = open_parent(store, c)
        try:
            r = store._quarantine(
                root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
            )
        finally:
            for h in [root, *handles]:
                h.close()
        assert r is False  # valid: put back
        assert stat.S_IMODE(entry.stat().st_mode) == 0o555
    finally:
        os.chmod(entry, 0o755)


def test_corrupt_budget_is_bounded(tmp_path, monkeypatch):
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    corrupt(entry_path(tmp_path / "root", c))
    n = []

    def quarantine(self, *a, **k):
        n.append(1)
        assert len(n) <= 8, "the corrupt budget is not bounded"
        return True  # "cleared", but the address stays corrupt

    monkeypatch.setattr(mod.PortableStore, "_quarantine", quarantine)
    again = publish(store)
    assert (again.won, again.stored) == (False, False)
    assert again.kept is not None
    assert len(n) == 4


def test_contention_is_not_charged_to_the_corrupt_budget(tmp_path, monkeypatch):
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    corrupt(entry_path(tmp_path / "root", c))
    real = mod.PortableStore._quarantine
    n = []

    def quarantine(self, *a, **k):
        n.append(1)
        return False if len(n) <= 8 else real(self, *a, **k)

    monkeypatch.setattr(mod.PortableStore, "_quarantine", quarantine)
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert len(n) == 9


def test_read_does_not_quarantine_an_entry_that_replaced_the_corrupt_one(
    tmp_path, monkeypatch
):
    import shutil

    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    backup = tmp_path / "b1"
    shutil.copytree(entry, backup)
    corrupt(entry)
    real = PydraEntryV1.read
    swapped = []

    def read(self, *a, **k):
        try:
            return real(self, *a, **k)
        except CorruptEntry:
            if not swapped:
                swapped.append(1)
                rmtree(entry)
                shutil.copytree(backup, entry)
            raise

    monkeypatch.setattr(PydraEntryV1, "read", read)
    calls: list = []
    recording_rename(monkeypatch, calls)
    with pytest.raises(CorruptEntry):
        store.read(c.entry, None)
    assert not [x for x in calls if "quarantine" in x[2]]
    assert entry.is_dir()


def test_set_aside_leaves_a_directory_that_replaced_the_object(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    root, handles = open_parent(store, c)
    try:
        store._set_aside(root, handles[-1], c.entry.fingerprint)
    finally:
        for h in [root, *handles]:
            h.close()
    assert entry.is_dir() and quarantined(tmp_path / "root") == []


def test_set_aside_names_the_target_from_the_inode(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    rmtree(entry)
    entry.write_text("x")
    name = inode_name(entry)
    root, handles = open_parent(store, c)
    try:
        store._set_aside(root, handles[-1], c.entry.fingerprint)
    finally:
        for h in [root, *handles]:
            h.close()
    assert [p.name for p in quarantined(tmp_path / "root")] == [name]


def test_enotdir_with_no_occupant_is_contention(tmp_path, monkeypatch):
    store = make(tmp_path)
    seen = []

    def fail(self, name, target, new_name):
        if at_address((None, None, None, new_name)) and not seen:
            seen.append(1)
            raise OSError(errno.ENOTDIR, "the file that was there has gone")

    recording_rename(monkeypatch, [], fail)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)


def test_a_paused_mover_cannot_take_the_entry_a_writer_committed(tmp_path, monkeypatch):
    """A reader paused between its check and its rename (a stopped process, a hung NFS call)
    while a writer clears the corrupt entry and commits: the reader resumes and moves nothing,
    and the address is never empty."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    corrupt(entry)
    real = StorageDir.rename
    paused, resume = threading.Event(), threading.Event()
    me = {}

    def rename(self, n, target, new_name):
        if threading.current_thread() is me.get("t") and "quarantine" in target.path:
            paused.set()
            assert resume.wait(20)
        return real(self, n, target, new_name)

    monkeypatch.setattr(StorageDir, "rename", rename)
    out = {}

    def reader():
        try:
            store.read(c.entry, None)
        except CorruptEntry as err:
            out["err"] = err

    t = threading.Thread(target=reader)
    me["t"] = t
    t.start()
    assert paused.wait(20)
    won = publish(store)  # clears the corrupt entry and commits a valid one
    assert (won.won, won.stored) == (True, True)
    gone = []
    resume.set()
    while t.is_alive():
        if not entry.exists():
            gone.append(1)
    t.join()
    assert gone == []
    assert entry.is_dir() and store.read(c.entry, None).outputs == OUT
    assert len(quarantined(tmp_path / "root")) == 1


# -- set-aside, container herds and bounded clearing ------------------------------------------------


@pytest.mark.parametrize("err", [errno.ENOTDIR, errno.ENOTEMPTY, errno.EEXIST])
def test_set_aside_refusal_of_its_own_move_is_not_an_error(tmp_path, monkeypatch, err):
    """The compare-and-move refused: the object is already moved, or is not the one looked at."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    rmtree(entry)
    entry.write_text("x")

    def fail(self, name, target, new_name):
        if "quarantine" in target.path:
            raise OSError(err, "refused")

    recording_rename(monkeypatch, [], fail)
    root, handles = open_parent(store, c)
    try:
        store._set_aside(root, handles[-1], c.entry.fingerprint)  # does not raise
    finally:
        for h in [root, *handles]:
            h.close()
    assert entry.exists()


@pytest.mark.parametrize("where", ["cdir", "shard"])
def test_a_regular_file_where_a_container_belongs_is_set_aside(tmp_path, where):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    target = entry.parent if where == "cdir" else entry.parent.parent
    rmtree(target)
    target.write_text("x")
    again = publish(store)
    assert (again.won, again.stored) == (True, True)
    assert len(quarantined(tmp_path / "root")) == 1
    assert store.read(again.entry, None).outputs == OUT


@pytest.mark.parametrize("err", [errno.ENOENT, errno.ESTALE])
def test_a_container_rename_that_lost_its_parent_walks_again(
    tmp_path, monkeypatch, err
):
    store = make(tmp_path)
    seen = []

    def fail(self, name, target, new_name):
        if not at_address((None, None, None, new_name)) and not seen:
            seen.append(1)
            raise OSError(err, "the parent is gone")

    recording_rename(monkeypatch, [], fail)
    c = publish(store)
    assert (c.won, c.stored) == (True, True)


@pytest.mark.parametrize("writers", [8, 16])
def test_container_herd_over_a_regular_file(tmp_path, writers, retry_clock):
    """Writers of equal outputs meeting a regular file at the cdir: one sets it aside, one
    wins, every other stores against the winner, and none waits out the visibility timeout
    (the retry clock moves only by the writers' waits, so a writer kept for contention would
    show). Durability is off: the herd is about the race, not the cost of a flush."""
    store = make(tmp_path, policy=StorePolicy(durability="none"))
    c = publish(store)
    probe = entry_path(tmp_path / "root", c).parent
    for rnd in range(10):
        name = f"python-herd{rnd}"
        first = publish(store, checksum=name)
        victim = entry_path(tmp_path / "root", first).parent
        rmtree(victim)
        victim.write_text("x")
        barrier = threading.Barrier(writers)
        out: list = [None] * writers

        def go(i):
            barrier.wait()
            try:
                got = publish(store, checksum=name)
            except BaseException as err:  # noqa
                got = err
            out[i] = got

        ts = [threading.Thread(target=go, args=(i,)) for i in range(writers)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert not [o for o in out if isinstance(o, BaseException)], out
        assert [(o.won, o.stored) for o in out].count((True, True)) == 1, out
        assert all(o.stored and o.kept is None for o in out), out
    assert sum(retry_clock.waits) < store.visibility_timeout
    assert probe.exists()


def zero_ino(st):
    return os.stat_result((st[0], 0, *tuple(st)[2:10]))


def test_a_zero_inode_is_not_an_identity(tmp_path, monkeypatch):
    """st_ino 0 identifies nothing: every quarantine gets a random name, not one shared name."""
    store = make(tmp_path)
    cs = [publish(store, checksum=f"python-z{i}") for i in range(3)]
    for c in cs:
        corrupt(entry_path(tmp_path / "root", c))
    real_l, real_f = StorageDir.lstat, StorageDir.fstat
    monkeypatch.setattr(StorageDir, "lstat", lambda self, n: zero_ino(real_l(self, n)))
    monkeypatch.setattr(StorageDir, "fstat", lambda self: zero_ino(real_f(self)))
    for c in cs:
        with pytest.raises(CorruptEntry):
            store.read(c.entry, None)
    assert len(quarantined(tmp_path / "root")) == 3
    for c in cs:
        assert not entry_path(tmp_path / "root", c).exists()


def test_an_empty_directory_occupant_is_removed_not_moved(tmp_path):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    rmtree(entry)
    entry.mkdir()
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is True
    assert not entry.exists() and quarantined(tmp_path / "root") == []


def test_the_deadline_bounds_clearing_too(tmp_path, monkeypatch, retry_clock):
    """A cleared occupant goes through the deadline check like contention does."""
    import pydra.engine.portable.store as mod

    store = make(tmp_path)
    c = publish(store)
    corrupt(entry_path(tmp_path / "root", c))
    store.visibility_timeout = 0.2
    monkeypatch.setattr(mod, "_ATTEMPTS", 10**9)
    n = []

    def quarantine(self, *a, **k):
        n.append(1)
        assert len(n) <= 200, "clearing ignored the deadline"
        retry_clock.now += 0.01
        return True

    monkeypatch.setattr(mod.PortableStore, "_quarantine", quarantine)
    again = publish(store)
    assert (again.won, again.stored) == (False, False)
    assert again.kept is not None


@pytest.mark.parametrize(
    "err,expected",
    [
        (errno.ENOENT, True),  # removed by someone else: the address is clear
        (errno.ENOTEMPTY, False),  # committed at the name since: not ours to charge
        (errno.EEXIST, False),
        (
            errno.EACCES,
            False,
        ),  # a transient refusal (a Windows sharing violation): retry
    ],
)
def test_rmdir_of_an_empty_occupant_maps_its_failure(
    tmp_path, monkeypatch, err, expected
):
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    rmtree(entry)
    entry.mkdir()

    def rmdir(self, name):
        raise OSError(err, "refused")

    monkeypatch.setattr(StorageDir, "rmdir", rmdir)
    root, handles = open_parent(store, c)
    try:
        r = store._quarantine(
            root, handles[-1], c.entry.fingerprint, c.entry.cdir, None
        )
    finally:
        for h in [root, *handles]:
            h.close()
    assert r is expected
    assert entry.is_dir() and quarantined(tmp_path / "root") == []


@pytest.mark.parametrize("meanwhile", ["same", "replaced", "gone"])
def test_set_aside_fallback_rechecks_the_object(tmp_path, monkeypatch, meanwhile):
    """A failure of the move without a refusal errno (Windows) raises only if the object seen is
    still at the name."""
    store = make(tmp_path)
    c = publish(store)
    entry = entry_path(tmp_path / "root", c)
    rmtree(entry)
    entry.write_text("x")

    def fail(self, name, target, new_name):
        if "quarantine" in target.path:
            if meanwhile != "same":
                entry.unlink()
            if meanwhile == "replaced":
                entry.mkdir()
            raise OSError(errno.EACCES, "access denied")

    recording_rename(monkeypatch, [], fail)
    root, handles = open_parent(store, c)
    try:
        if meanwhile == "same":
            with pytest.raises(WrongKindError):
                store._set_aside(root, handles[-1], c.entry.fingerprint)
        else:
            store._set_aside(root, handles[-1], c.entry.fingerprint)
    finally:
        for h in [root, *handles]:
            h.close()
    assert entry.exists() == (meanwhile != "gone")


def test_a_read_asks_for_no_more_than_a_block(tmp_path, monkeypatch):
    """A JSON member is read in blocks, and a blob by the size the file has: ``read(n)``
    allocates ``n`` bytes up front, so a read sized by the 1 GiB cap committed 1 GiB per read on
    Windows and stalled concurrent readers for seconds."""
    store = make(tmp_path)
    c = publish(store)
    asked = []
    real = StorageDir.open_read

    class Recording:
        def __init__(self, f):
            self.f = f

        def read(self, n=-1):
            asked.append(n)
            return self.f.read(n)

        def __getattr__(self, name):
            return getattr(self.f, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.f.close()

    monkeypatch.setattr(
        StorageDir, "open_read", lambda self, name: Recording(real(self, name))
    )
    assert store.read(c.entry, None).outputs == OUT
    assert asked and all(0 < n <= 1 << 20 for n in asked), asked


def test_visibility_timeout_is_never_below_the_lease_attribute_delay(tmp_path):
    from pydra.engine.lease import LeaseSettings

    assert make(tmp_path).visibility_timeout == LeaseSettings().visibility_timeout
    assert LeaseSettings().visibility_timeout > LeaseSettings().attribute_delay


def test_a_custom_entry_format_is_bound_to_the_store(tmp_path):
    from pydra.engine.portable.entry import PydraEntryV1

    bound = []

    class Custom(PydraEntryV1):
        def bind(self, identity, pickle, trust):
            bound.append((identity, pickle, trust))
            return super().bind(identity, pickle, trust)

    store = make(tmp_path, entry_format=Custom())
    assert len(bound) == 1 and store.format.identity is bound[0][0]


def test_visibility_timeout_follows_the_store_lease_settings(tmp_path):
    from pydra.engine.lease import LeaseSettings

    local = LeaseSettings(attribute_delay=0)
    store = make(tmp_path, lease_settings=local)
    assert store.visibility_timeout == local.visibility_timeout
