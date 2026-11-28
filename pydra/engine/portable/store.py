"""`PortableStore`: publish and read (RFC 0003 sections 5.1, 5.2, 6.2 and 7.2).

A writer encodes an entry into ``staging/<uuid>/`` and renames the directory to its address,
``<entries>/<shard>/<cdir>/<fingerprint>``. The rename is the first-writer-wins primitive: an
entry exists only once it is complete. Every step below ``<root>`` goes through a
`~pydra.utils.storagedir.StorageDir`.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
import typing as ty
import uuid
from pathlib import Path

import attrs
import cloudpickle as cp

from pydra.engine.lease import (
    HOLDER_FILE,
    DirectoryLease,
    LeaseSettings,
    holder_stale_after,
    read_holder_record,
)
from pydra.engine.portable.entry import (
    CorruptEntry,
    EntryFormat,
    EntryFormatUnknown,
    EntryNotTrusted,
    EntryRef,
    PydraEntryV1,
    StoredEntry,
    read_capped,
)
from pydra.engine.portable.errors import KeyMismatch, UnencodableOutput
from pydra.engine.store import ResultStore
from pydra.engine.portable.identity import (
    Context,
    KeyRecord,
    PydraIdentityV2,
    StoreIdentity,
)
from pydra.engine.portable.layout import (
    FINGERPRINT,
    HASHED_CDIR,
    _TEMP_RE,
    HEADER_NAME,
    LayoutHeader,
    LegacyLayout,
    StoreLayoutAbsent,
    StoreLayoutMismatch,
    StoreLayoutUnknown,
    cdir_for,
    DEFAULT_DIRS,
    check_layout_names,
    open_layout,
    read_header,
    shard_for,
)
from pydra.engine.portable.reuse import (
    Candidate,
    Hit,
    Lookup,
    Miss,
    Reuse,
    Verdict,
    Verify,
    as_reuse,
)
from pydra.engine.portable.trust import TRUSTS, Root, Trust, check_private
from pydra.engine.result import Result, _loads, record_error_json, save
from pydra.utils.retry import Backoff, Tries
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorageEntry,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
    is_reparse,
    copy_hashed,
    is_within,
)

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job
    from pydra.engine.lease import Lease

__all__ = ["CollectReport", "Committed", "PortableStore", "StoreNotWritable"]

logger = logging.getLogger("pydra")

_DENIED = (errno.EACCES, errno.EPERM, errno.EROFS)
_ATTEMPTS = 4
"""Most tries of a step another writer can undo (a container made, an occupant cleared): each
retry follows another writer's change, so a few suffice."""
_STALE = getattr(errno, "ESTALE", errno.ENOENT)
_CONTENDED = (
    object()
)  # another writer's progress changed what this one saw: walk down again
_CLEARED = object()  # a corrupt occupant was set aside: place again
_REFUSED = object()  # the reuse hook gave REFUSE, which ends the lookup
_OURS = object()  # the occupant carries this writer's token
_RERANKS = 4
"""Most re-rankings of a reuse lookup whose ranked entry changed under it: each follows a
concurrent replacement, so a few suffice."""
_WORK = "work"
_LEASES = "leases"
_RESUME = "resume-"  # a resumable job's work directory, named as its lease
_AUDIT = "audit"
_COMMITTED = "_committed.json"
_ERROR = "_error.json"
_KEPT_ENTRY = "_entry"
_OUTPUTS = "_outputs.pklz"
_COMMITTED_FORMAT = "pydra-committed@1"
_ATTEMPT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
RECORD_CAP = 1 << 20
"""Most bytes of a ``_committed.json`` that are read: its fields are a fixed set of names and
digests, and the cap only bounds a hostile file. An ``_error.json`` is read whole."""
_VERSION_DIR = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*@[0-9]+")
_STRAY_WARNED: set[tuple[str, str]] = set()  # (root, name) logged by _warn_strays


@attrs.frozen
class Committed:
    """What `PortableStore.publish` returns (RFC 0003 section 5.1).

    Attributes
    ----------
    entry : EntryRef
        The entry now at the address, ours or another writer's.
    won : bool
        Our rename placed it.
    own_digest : str
        The digest of our encoded outputs.
    committed_digest : str or None
        The digest of the entry at the address; None if it cannot be read.
    stored : bool
        Our outputs, or outputs equal to them, are in the store.
    kept : str or None
        With ``stored=False``: the directory holding our staged entry, unpublished.
    served : bool
        A lost race whose occupant the reuse hook served (RFC 0003 section 7.3): the job's
        result is the committed entry's outputs, ``stored`` is true and ``kept`` is None.
    """

    entry: EntryRef
    won: bool
    own_digest: str
    committed_digest: str | None
    stored: bool
    kept: str | None
    served: bool = False


class CommitHookError(RuntimeError):
    """``on_commit`` added a member to the entry it was given. The hook keeps its records in
    sidecars or the audit directory; the members of an entry are the entry format's alone.
    """


class StoreNotWritable(StoreUnavailable):
    """This user cannot write a directory that a publish needs (RFC 0003 section 5.1). It is a
    configuration error, not a lost race."""


@attrs.frozen
class CollectReport:
    """What `PortableStore.collect` did (RFC 0003 section 9).

    Attributes
    ----------
    removed : list of str
        The paths removed.
    kept : list of (str, str)
        The paths looked at and left, each with the reason: too young, or why it could not be
        removed (with the owner, when that is the reason).
    """

    removed: list[str] = attrs.field(factory=list)
    kept: list[tuple[str, str]] = attrs.field(factory=list)


def _owner(path: str) -> str:
    try:
        uid = os.lstat(path).st_uid
    except OSError:
        return "unknown"
    try:
        import pwd

        return f"{pwd.getpwuid(uid).pw_name} (uid {uid})"
    except (ImportError, KeyError):
        return f"uid {uid}"


def _wrong_kind_error(path: str) -> WrongKindError:
    return WrongKindError(
        errno.ENOTDIR,
        f"{path!r} is a symlink or an object of another kind where the store expects a "
        f"directory, and it cannot be moved aside; owner: {_owner(path)}",
        path,
    )


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or is_reparse(st)


def _is_wrong_kind(st: os.stat_result) -> bool:
    return _is_link(st) or not stat.S_ISDIR(st.st_mode)


def _inode_name(st: os.stat_result) -> str:
    """A UUID-shaped name for a directory's inode: the target of the rename that moves it aside.
    ``st_dev`` is left out because it is client-local on network storage. An inode number of 0
    identifies nothing, so such a directory gets a random name and no claim of identity.
    """
    if st.st_ino == 0:
        return uuid.uuid4().hex
    return uuid.UUID(int=st.st_ino % (1 << 128)).hex


def _remove_tree(parent: StorageDir, name: str) -> None:
    """Remove the directory ``name`` and everything in it, never following a symlink."""
    with parent.child(name) as d:
        if not IS_WINDOWS:
            st = d.fstat()
            if st.st_uid == os.geteuid() and st.st_mode & 0o700 != 0o700:
                with contextlib.suppress(OSError):
                    d.set_mode(stat.S_IMODE(st.st_mode) | 0o700)
        for e in list(d.scandir()):
            if e.is_dir():
                _remove_tree(d, e.name)
            else:
                _unlink(d, e.name)
    _retrying(parent.rmdir, name)


SHARING_RETRY = Backoff(first=0.05, cap=0.2, attempts=4)
"""How a removal retries a Windows sharing violation (an open handle, a scanner)."""


def _retrying(fn: ty.Callable[[str], None], name: str) -> None:
    """Call ``fn(name)``; on Windows a sharing violation is retried as `SHARING_RETRY` says."""
    tries = SHARING_RETRY.tries()
    for _ in tries:
        try:
            return fn(name)
        except PermissionError:
            if not IS_WINDOWS or tries.last:
                raise


def _unlink(parent: StorageDir, name: str) -> None:
    _retrying(parent.unlink, name)


def _gc_name(st: os.stat_result) -> str:
    return "gc-" + _inode_name(st)


def _discard(parent: StorageDir, name: str) -> None:
    with contextlib.suppress(OSError):
        _remove_tree(parent, name)


class _Replaced(Exception):
    """A directory this writer holds was replaced by another writer's and is no longer in the
    tree; the walk from the root starts again."""


class _Vanished(Exception):
    """The writer's staging directory is gone (collected): stage again."""


class _Toucher:
    """Touches a staging directory every ``interval`` seconds from a thread, so that a live
    writer is never collected, whatever it is doing (RFC 0003 section 9)."""

    def __init__(self, staging: StorageDir, name: str, interval: float) -> None:
        self._staging = staging
        self._name = name
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def touch(self) -> None:
        try:
            with self._staging.child(self._name) as d:
                os.utime(d._fd if d._fd is not None else d.path, None)
        except (OSError, ValueError):
            pass  # gone: the writer finds out at the rename

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.touch()

    def __enter__(self) -> "_Toucher":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()


def _now() -> float:
    return time.time()


def _last_change(st: os.stat_result) -> float:
    """When an object was last modified or moved. A rename changes ``ctime`` and not
    ``mtime``; on Windows ``ctime`` is the creation time, so only ``mtime`` is used there.
    A mover stamps what it moves before the move, so a moved object is young there too; the
    stamp is what protects it, and one user running several processes can race the gap.
    """
    return st.st_mtime if IS_WINDOWS else max(st.st_mtime, st.st_ctime)


class _Changed(Exception):
    """The ``context.json`` of an entry is not the one that was ranked."""


class _Contended(Exception):
    """The visibility timeout passed while other writers kept changing the directories."""


CONTENTION = Backoff(first=0.004, cap=0.25, jitter=True)
"""How a writer pauses after another writer replaced a container or the occupant changed,
until the visibility timeout: jittered, so contending writers spread out."""


SETTLED_READ = Backoff(first=0.01, cap=0.25)
"""How a settled reader polls for another client's record, until the visibility timeout."""


def _pause(tries: Tries) -> None:
    if not tries.wait():
        raise _Contended


def _rebuild(args: dict[str, ty.Any]) -> "PortableStore":
    return PortableStore(**args)


def _check_legacy(root: os.PathLike, legacy: tuple[Root, ...]) -> None:
    """Refuse a ``legacy=`` root that is not a `Root`, is declared ``shared``, or holds the
    store's own root (a legacy root is never written, RFC 0003 section 8.1)."""
    own = os.path.realpath(os.fspath(root))
    for item in legacy:
        if not isinstance(item, Root):
            raise TypeError(f"a legacy root is a Root, got {item!r}")
        if item.trust == "shared":
            raise ValueError(
                f"the legacy root {os.fspath(item.path)!r} is declared shared; reading a "
                "PickleStore cache unpickles its results, so a legacy root is read only when "
                "declared private or trusted (RFC 0003 section 8.1)"
            )
        path = os.path.realpath(os.fspath(item.path))
        if is_within(own, path):
            raise ValueError(
                f"the legacy root {path!r} holds the store's root {own!r}; a legacy root is "
                "never written"
            )


def _record_of(key: KeyRecord) -> "dict[str, ty.Any] | None":
    """``Result.record`` of an entry: its key record (``pydra-identity@2``), or None for an
    identity whose key is None"""
    return dict(key) if isinstance(key, ty.Mapping) else None


class PortableStore(ResultStore):
    """A result store whose entries any process, host or user can publish and read.

    Parameters
    ----------
    root : os.PathLike
        The store's root. A new root, or an empty one, is given a layout header.
    trust : {"shared", "private", "trusted"}
        What this process asserts about who can write the root (RFC 0003 section 6.4).
    identity : StoreIdentity
        The key, context and fingerprint of an entry; default ``pydra-identity@2``.
    entry_format : EntryFormat
        What is inside an entry, bound to this store's identity, trust and the header's
        ``pickle`` flag (`EntryFormat.bind`).
    policy : StorePolicy
        Modes, durability and ages; recorded in the header, so two stores of one root must agree.
    readonly : sequence of Root
        Roots searched after ``root`` for entries to reuse (RFC 0003 section 7.4), each with
        this store's header, read under its own trust and never written.
    reuse : str, ReusePolicy or Reuse
        Which stored entries a job may use (section 7.1): a policy name or `ReusePolicy`, as
        for `PickleStore`, or a `~pydra.engine.portable.reuse.Reuse` hook. Bound to this store
        at construction and again after unpickling.
    on_commit : callable, optional
        ``on_commit(staged, checksum, key, context, digest)``, called with the staging
        directory before its rename (RFC 0003 section 7.4). It may run more than once for
        one entry, so it must be idempotent; it must not add members to ``staged``
        (`CommitHookError`) and writes through `write_sidecar` or the audit directory. If
        it raises, nothing is committed.
    legacy : sequence of Root
        `PickleStore` cache roots read in place (RFC 0003 section 8.1), each ``"private"``
        or ``"trusted"``; never written. A job none of this store's roots serves is looked
        up there, and a usable result published into ``root`` and served from it.
    dedupe : bool, optional
        Whether two submitters of one job compute it once, through a lease at
        ``<root>/leases/`` (RFC 0003 section 5.3). Default True unless ``trust`` is
        ``"shared"``. False gives a lease that is always held.
    work_dir : callable, optional
        ``work_dir(checksum, fingerprint, attempt) -> Path``, a pure function giving the
        absolute directory a job runs in (RFC 0003 section 5.4). It is required for a
        ``"shared"`` root, whose default (``<root>/work/<attempt>``) other users could read.
    pickle : bool
        Whether writers to a new root may write pickle members; recorded when the root is
        created and read from the header after.
    lease_settings : LeaseSettings, optional
        How the store's leases (de-duplication, resume and rerun) beat, age and wait; see
        `~pydra.engine.lease.LeaseSettings`, with the holder always ``"anonymous"`` (a
        lease names no user, host or pid). Default ``LeaseSettings()``; read-only after
        construction.
    shard : {"sha256-2", "prefix2"}, optional
        The shard rule (RFC 0003 section 3.1). Default: the header's for a root that has
        one, ``"sha256-2"`` for a new root.
    dirs : mapping, optional
        The header's ``dirs``: the names of the staging and quarantine directories and of
        each sidecar directory. Default: the header's for a root that has one (an adopted
        store names its own, section 8.2), ``staging`` and ``quarantine`` for a new root.
    """

    def __init__(
        self,
        root: os.PathLike,
        *,
        trust: Trust = "shared",
        identity: StoreIdentity = PydraIdentityV2(),
        entry_format: EntryFormat = PydraEntryV1(),
        policy: StorePolicy = StorePolicy(),
        readonly: ty.Sequence[Root] = (),
        legacy: ty.Sequence[Root] = (),
        dedupe: bool | None = None,
        work_dir: ty.Callable[[str, str, str], Path] | None = None,
        pickle: bool = False,
        reuse: ty.Any = "consistent",
        on_commit: ty.Callable | None = None,
        lease_settings: LeaseSettings | None = None,
        shard: ty.Literal["sha256-2", "prefix2"] | None = None,
        dirs: ty.Mapping[str, ty.Any] | None = None,
    ) -> None:
        if lease_settings is None:
            lease_settings = LeaseSettings()
        elif not isinstance(lease_settings, LeaseSettings):
            raise TypeError(
                f"lease_settings must be a LeaseSettings, got {type(lease_settings).__name__}"
            )
        # evolve checks the bounds again
        lease_settings = attrs.evolve(lease_settings, holder="anonymous")
        if trust not in TRUSTS:
            raise ValueError(f"trust must be one of {TRUSTS}, got {trust!r}")
        if trust == "shared" and work_dir is None:
            raise ValueError(
                "a shared root needs work_dir=: a work directory holds a job's pickles and crash "
                "file, which other users of the root must not read (RFC 0003 section 5.4)"
            )
        legacy = tuple(legacy)
        _check_legacy(root, legacy)
        self._args = dict(
            root=root,
            trust=trust,
            identity=identity,
            entry_format=entry_format,
            policy=policy,
            readonly=tuple(readonly),
            legacy=tuple(legacy),
            dedupe=dedupe,
            work_dir=work_dir,
            pickle=pickle,
            reuse=reuse,
            on_commit=on_commit,
            lease_settings=lease_settings,
            shard=shard,
            dirs=dirs,
        )
        missing = [
            m
            for m in (
                *StoreIdentity.__annotations__,
                *(n for n, v in vars(StoreIdentity).items() if callable(v)),
            )
            if not m.startswith("_") and not hasattr(identity, m)
        ]
        if missing:
            raise TypeError(
                f"the identity {identity!r} lacks the StoreIdentity members "
                f"{', '.join(missing)}"
            )
        self.path = os.path.abspath(os.fspath(root))
        self.trust = trust
        self.identity = identity
        self.policy = policy
        self.readonly = tuple(readonly)
        self.legacy = tuple(legacy)
        self.dedupe = trust != "shared" if dedupe is None else bool(dedupe)
        self._work_fn = work_dir
        self._attempts: set[Path] = set()
        self.reuse: Reuse = as_reuse(reuse)
        self.on_commit = on_commit
        self.visibility_timeout = lease_settings.visibility_timeout
        self._work_policy = attrs.evolve(policy, verify_files="digest")
        self._lease_settings = lease_settings
        self._make_root()
        header_present = os.path.lexists(os.path.join(self.path, HEADER_NAME))
        with self._open() as handle:
            if header_present and (shard is None or dirs is None):
                stored = read_header(handle)
                if stored is not None:
                    shard = stored.shard if shard is None else shard
                    dirs = stored.dirs if dirs is None else dirs
            shard = "sha256-2" if shard is None else shard
            check_layout_names(
                DEFAULT_DIRS if dirs is None else dirs, entry_format.entries, shard
            )
            expected = LayoutHeader.for_store(
                identity,
                policy,
                pickle=pickle,
                entries=entry_format.entries,
                entry_format=entry_format.name,
                shard=shard,
                dirs=dirs,
            )
            self.header = open_layout(handle, expected, create=True)
            self._warn_strays(handle)
            self.pickle = self.header.pickle
            self.format = entry_format.bind(identity, self.pickle, trust)
            self._expected = expected
            made = not header_present
            for name in (self.header.dirs["staging"], self.header.dirs["quarantine"]):
                made |= self._side_dir(handle, name)
            made |= self._entries_dir(handle)
            if made:
                handle.sync()
        self._checked: set[str] = set()
        for ro in self.readonly:
            self._check_readonly(os.path.abspath(os.fspath(ro.path)), ro.trust)
        self.reuse.bind(self)

    def _check_readonly(self, path: str, trust: Trust) -> bool:
        """Whether the read-only root at ``path`` carries this store's header. A root that
        cannot be reached or holds no layout is logged and skipped, to be tried again at the
        next lookup; a header that differs raises `StoreLayoutMismatch` (or the layout error
        that says why it cannot be read)."""
        if path in self._checked:
            return True
        try:
            with StorageDir.open_root(path, self.policy, trust=trust) as handle:
                open_layout(handle, self._expected, create=False)
        except StoreLayoutAbsent as err:
            logger.warning("the read-only root %s was skipped: %s", path, err)
            return False
        except (StoreLayoutMismatch, StoreLayoutUnknown, LegacyLayout):
            raise
        except OSError as err:
            logger.warning("the read-only root %s was skipped: %s", path, err)
            return False
        self._checked.add(path)
        return True

    def __reduce__(self) -> tuple:
        return (_rebuild, (self._args,))

    @property
    def lease_settings(self) -> LeaseSettings:
        """The bounds of the store's leases, holder ``"anonymous"``; fixed at construction."""
        return self._lease_settings

    def __repr__(self) -> str:
        return f"PortableStore({self.path!r}, trust={self.trust!r})"

    # -- the root ---------------------------------------------------------------------------

    def _make_root(self) -> None:
        if os.path.isdir(self.path):
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        try:
            os.mkdir(self.path, self.policy.side_mode)
        except FileExistsError:
            return
        if not IS_WINDOWS:
            os.chmod(self.path, self.policy.side_mode)

    def _open(self) -> StorageDir:
        return StorageDir.open_root(self.path, self.policy, trust=self.trust)

    def _warn_strays(self, root: StorageDir) -> None:
        """Log, once per process, each directory under the root named like a version
        directory that is not this store's: what a writer that predates an adopted store's
        header makes, and this store never reads (RFC 0003 section 8.2)."""
        dirs = self.header.dirs
        own = {self.header.entries, dirs.get("staging"), dirs.get("quarantine")}
        for relative in (dirs.get("sidecars") or {}).values():
            own.add(str(relative).split("/")[0])
        try:
            names = root.names()
            for name in names:
                if name in own or not _VERSION_DIR.fullmatch(name):
                    continue
                if (self.path, name) in _STRAY_WARNED:
                    continue
                if not stat.S_ISDIR(root.lstat(name).st_mode):
                    continue
                _STRAY_WARNED.add((self.path, name))
                logger.warning(
                    "%s holds %s, a version directory this store (%s) never reads: a "
                    "writer that predates the header is writing there (RFC 0003 section 8.2)",
                    self.path,
                    name,
                    self.header.entries,
                )
        except OSError as err:
            logger.debug("the root %s was not listed: %s", self.path, err)

    def _writable(self) -> bool:
        return os.access(self.path, os.W_OK | os.X_OK)

    def _side_dir(self, root: StorageDir, name: str) -> bool:
        """Make the directory ``name`` below the root if it is missing and the root is
        writable. True if this call made it."""
        try:
            root.lstat(name)
            return False
        except FileNotFoundError:
            pass
        try:
            root.child(name, create=True, mode=self.policy.side_mode).close()
        except PermissionError:
            return False
        return True

    def _entries_dir(self, root: StorageDir) -> bool:
        try:
            root.lstat(self.header.entries)
            return False
        except FileNotFoundError:
            pass
        try:
            root.child(
                self.header.entries, create=True, mode=self.policy.container_mode
            ).close()
        except PermissionError:
            return False
        return True

    @contextlib.contextmanager
    def _denied(self, where: str) -> ty.Iterator[None]:
        """Raise `StoreNotWritable` for a refusal to write ``where``; any other ``OSError``
        is a `StoreUnavailable`."""
        try:
            yield
        except StoreUnavailable:
            raise
        except OSError as err:
            if err.errno in _DENIED:
                try:
                    mode = f"{stat.S_IMODE(os.lstat(where).st_mode):#o}"
                except OSError:
                    mode = "unknown"
                modes = self.header.modes
                raise StoreNotWritable(
                    err.errno,
                    f"this user cannot write {where!r} (owner {_owner(where)}, mode {mode}); "
                    f"the store's header has container mode {modes['container']} and side mode "
                    f"{modes['side']}, and a root is written by its creator alone unless both "
                    "are group-writable (RFC 0003 section 9)",
                    where,
                ) from err
            raise StoreUnavailable(
                err.errno, f"{where}: {err.strerror or err}", where
            ) from err

    # -- directories --------------------------------------------------------------------------

    def _quarantine_dir(self, root: StorageDir) -> StorageDir:
        return root.child(
            self.header.dirs["quarantine"], create=True, mode=self.policy.side_mode
        )

    def _set_aside(self, root: StorageDir, parent: StorageDir, name: str) -> None:
        """Move an object of the wrong kind (a symlink, a file) out of ``parent`` to
        ``quarantine/<uuid>``. Where the path backend cannot rename a symlink (Windows) a
        symlink or reparse point is unlinked; anything else is renamed.

        Raises
        ------
        WrongKindError
            It cannot be moved.
        """
        where = os.path.join(parent.path, name)
        try:
            st = parent.lstat(name)
        except FileNotFoundError:
            return
        if stat.S_ISDIR(st.st_mode) and not _is_link(st):
            return  # a directory replaced it since the caller looked
        try:
            if IS_WINDOWS and _is_link(st):
                parent.unlink(name)
            else:
                with self._quarantine_dir(root) as q:
                    moved = _inode_name(st)
                    parent.rename(name, q, moved)
                    can_skip_links = os.utime in os.supports_follow_symlinks
                    if can_skip_links or not _is_link(st):  # never a link's target
                        with contextlib.suppress(OSError):
                            os.utime(
                                os.path.join(q.path, moved),
                                None,
                                follow_symlinks=not can_skip_links,
                            )
        except FileNotFoundError:
            return
        except OSError as err:
            if err.errno in (errno.ENOTDIR, errno.ENOTEMPTY, errno.EEXIST):
                return  # the compare-and-move refused: already moved, or not the object seen
            try:
                now = parent.lstat(name)
            except FileNotFoundError:
                return
            if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino) or (
                stat.S_ISDIR(now.st_mode) and not _is_link(now)
            ):
                return  # replaced meanwhile (the Windows refusal has no distinct errno)
            raise _wrong_kind_error(where) from err
        logger.warning(
            "%s is not an entry or a directory and was moved to quarantine", where
        )

    def _checksum_of(self, entry: StorageDir, cdir: str) -> str:
        """The checksum that ``cdir`` stands for: itself, or for a hashed name the checksum the
        entry records if that gives this directory (otherwise ``cdir``, which the read then
        finds misfiled)."""
        if HASHED_CDIR.fullmatch(cdir):
            recorded = self.format.identify(entry)[0]
            if cdir_for(recorded) == cdir:
                return recorded
        return cdir

    def _quarantine(
        self,
        root: StorageDir,
        parent: StorageDir,
        name: str,
        cdir: str,
        identity: os.stat_result | None,
        expected_digest: str | None = None,
    ) -> bool | None:
        """Move the entry ``name`` of ``parent`` aside as a compare-and-move (RFC 0003
        section 6.3) and check what was moved: a copy that verifies is put back, a corrupt
        one is stamped for ``collect``. ``identity`` is the ``fstat`` of the entry found
        corrupt; an entry with another inode is left alone. With ``expected_digest`` (an
        `evict`) a copy that verifies is kept moved when its output digest is
        ``expected_digest``.

        Returns
        -------
        True
            The address is clear (the entry was moved and is corrupt, or was already gone).
        False
            It was replaced, the moved copy verified and was restored, or an empty occupant
            could not be removed this time.
        None
            It cannot be moved.
        """
        where = os.path.join(parent.path, name)
        try:
            entry = parent.child(name)
        except FileNotFoundError:
            return True
        except WrongKindError:
            self._set_aside(root, parent, name)
            return True
        with entry:
            held = entry.fstat()
            try:
                empty = not any(True for _ in entry.scandir())
            except OSError:
                empty = False
            if identity is not None and (held.st_dev, held.st_ino) != (
                identity.st_dev,
                identity.st_ino,
            ):
                return False
            if not empty:
                with contextlib.suppress(
                    OSError
                ):  # aged from the move, before it can be listed
                    os.utime(entry._fd if entry._fd is not None else entry.path, None)
            if (
                not IS_WINDOWS
                and held.st_uid == os.geteuid()
                and not held.st_mode & 0o200
            ):
                with contextlib.suppress(OSError):
                    entry.set_mode(stat.S_IMODE(held.st_mode) | 0o200)
        try:
            now = parent.lstat(name)
        except FileNotFoundError:
            return True
        if (now.st_dev, now.st_ino) != (held.st_dev, held.st_ino):
            return False
        if empty:
            try:
                parent.rmdir(name)  # fails ENOTEMPTY on any entry: a compare-and-remove
            except FileNotFoundError:
                return True
            except OSError as err:
                if err.errno not in (errno.ENOTEMPTY, errno.EEXIST):
                    logger.warning("could not remove %s: %s", where, err)
                return False  # committed at the name since, or refused for now: retried
            return True
        moved = _inode_name(held)
        try:
            with self._quarantine_dir(root) as q:
                try:
                    parent.rename(name, q, moved)
                except PermissionError:
                    moved = "q-" + moved
                    parent.rename(name, parent, moved)
                    holder = parent
                else:
                    holder = None
                holder = holder or q
                at = holder.lstat(moved)
                if (at.st_dev, at.st_ino) != (held.st_dev, held.st_ino):
                    # another entry took the name between the check and the rename: put it
                    # back at once, before anything else, so the address is empty briefly
                    with contextlib.suppress(OSError):
                        holder.rename(moved, parent, name)
                        self._narrow(parent, name)
                    return False
                return self._after_move(
                    root, parent, name, cdir, moved, holder, expected_digest
                )
        except FileNotFoundError:
            return True
        except OSError as err:
            if err.errno in (errno.ENOTEMPTY, errno.EEXIST):
                return False  # the corrupt entry is already there: what is at the name is not it
            logger.warning("could not move %s aside: %s", where, err)
            return None

    def _after_move(
        self,
        root: StorageDir,
        parent: StorageDir,
        name: str,
        cdir: str,
        moved: str,
        holder: StorageDir,
        expected_digest: str | None = None,
    ) -> bool:
        where = os.path.join(parent.path, name)
        with holder.child(moved) as copy:
            stamp = copy._fd if copy._fd is not None else copy.path
            with contextlib.suppress(OSError):
                os.utime(stamp, None)  # aged from the move, and fresh while judged
            try:
                checksum = self._checksum_of(copy, cdir)
                stored = self.format.read(
                    copy, checksum, name, None, self.trust, root=root
                )
            except CorruptEntry as err:
                logger.warning("%s was moved to quarantine: %s", where, err)
                return True
            except Exception:  # cannot be judged: it is not shown corrupt
                stored = None
            if (
                stored is not None
                and expected_digest is not None
                and stored.digest == expected_digest
            ):
                logger.info("%s was evicted to quarantine", where)
                return True
        with contextlib.suppress(OSError):
            holder.rename(
                moved, parent, name
            )  # without replacement of a non-empty entry
            self._narrow(parent, name)
            logger.warning("%s verified after it was moved and was put back", where)
        return False

    def _container(
        self, root: StorageDir, staging: StorageDir, parent: StorageDir, name: str
    ) -> StorageDir:
        """Open the container directory ``name`` of ``parent``, making it if absent: it is
        made in ``staging`` with the container mode and renamed into place, so a kill never
        leaves it private; a loser of that rename uses the existing directory.

        Raises
        ------
        _Replaced
            ``parent`` is no longer in the tree (``ENOENT`` or ``ESTALE``): walk down again.
        """
        for _ in range(_ATTEMPTS):
            try:
                return parent.child(name)
            except FileNotFoundError:
                pass
            except WrongKindError:
                self._set_aside(root, parent, name)
                continue
            except OSError as err:
                if err.errno == _STALE:
                    raise _Replaced(parent.path) from err
                raise
            tmp = uuid.uuid4().hex
            with self._denied(parent.path):
                staging.child(tmp, create=True, mode=self.policy.container_mode).close()
                try:
                    staging.rename(tmp, parent, name)
                except OSError as err:
                    _discard(staging, tmp)
                    if self._occupied(parent, name):
                        continue  # a lost race: use the container that is there
                    if err.errno in (errno.ENOENT, _STALE):
                        raise _Replaced(parent.path) from err
                    raise
                if self.policy.durability == "member":
                    parent.sync()
        raise _Replaced(parent.path)

    def _cdir_dir(
        self,
        root: StorageDir,
        staging: StorageDir,
        shard: str,
        cdir: str,
        tries: Tries,
    ) -> list[StorageDir]:
        """The handles of ``<entries>``, ``<shard>`` and ``<cdir>``, made if absent. The caller
        closes them. A container replaced by another writer's, or gone stale, starts the walk
        again from the root after a jittered pause, while ``tries`` last."""
        while True:
            handles: list[StorageDir] = []
            try:
                handles.append(
                    self._container(root, staging, root, self.header.entries)
                )
                handles.append(self._container(root, staging, handles[0], shard))
                handles.append(self._container(root, staging, handles[1], cdir))
                return handles
            except _Replaced:
                pass
            except BaseException:
                for h in handles:
                    h.close()
                raise
            for h in handles:
                h.close()
            _pause(tries)

    @staticmethod
    def _attached(root: StorageDir, handles: list[StorageDir]) -> bool:
        """Whether each of ``handles`` is still the directory its parent names. A rename onto an
        empty directory replaces it, so a writer that opened a container another writer then
        replaced holds a directory that is no longer in the tree."""
        parent = root
        for h in handles:
            name = os.path.basename(h.path)
            try:
                now = parent.lstat(name)
            except FileNotFoundError:
                return False
            here = h.fstat()
            if (now.st_dev, now.st_ino) != (here.st_dev, here.st_ino):
                return False
            parent = h
        return True

    # -- publish ----------------------------------------------------------------------------------

    def publish(
        self,
        checksum: str,
        key: KeyRecord,
        context: Context,
        outputs: ty.Mapping[str, ty.Any],
        task: "Task | None",
        files_from: os.PathLike | None,
        *,
        verify: ty.Any = None,
        replace: bool = False,
        expected: str | None = None,
    ) -> Committed:
        """Publish ``outputs`` at the address of ``checksum`` and ``context`` (RFC 0003 section
        5.1).

        The entry is encoded into a fresh ``staging/<uuid>/`` and renamed to its address. If
        another writer was first, its entry is read and verified: with equal digests ours is
        stored and the staging directory removed, otherwise ours stays unpublished in the staging
        directory that ``Committed.kept`` names. A corrupt or misfiled occupant is quarantined and
        the commit retried. The lease is not consulted, and this never raises ``LeaseLost``.

        ``verify`` is the candidate a job marked to verify runs against (section 7.1): once the
        entry is staged, ``Reuse.compare`` and then ``on_commit`` are called before the rename; if
        either raises, nothing is committed. A candidate that can no longer be read is not compared.

        ``replace`` is a rerun (section 10): just before the rename the entry at the address is
        evicted if its output digest is ``expected``, the digest the caller read when its job
        started (None: nothing is evicted). If another writer commits in between, or the evict
        cannot move the entry, the publish loses the race and the occupant is never served: the
        result is ours, stored only when the digests are equal. The reuse hook is not asked.

        Raises
        ------
        KeyMismatch
            The store's identity refuses ``key`` for ``checksum``; nothing is written.
        StoreNotWritable
            This user cannot write a directory the publish needs.
        UnencodableOutput
            An output has no codec and this writer may not pickle it.
        StoreUnavailable
            Any other ``OSError``.
        """
        if not self.identity.check_key(checksum, key):
            raise KeyMismatch(
                f"the key record given for {checksum!r} does not digest to it under "
                f"{self.identity.name}, so no read would accept the entry"
            )
        cdir = cdir_for(checksum)
        shard = shard_for(cdir, checksum, self.header.shard)
        fingerprint = self.identity.fingerprint(context)
        writer = uuid.uuid4().hex
        name = uuid.uuid4().hex
        ref = EntryRef(self.path, cdir, fingerprint)
        with self._open() as root:
            source = None
            if files_from is not None and os.path.isdir(files_from):
                source = StorageDir.open_root(
                    files_from, StorePolicy(), trust="trusted"
                )
            try:
                with self._container_staging(root) as staging:
                    return self._publish(
                        root, staging, name, ref, shard, checksum, key, context,
                        outputs, task, source, writer, verify, replace, expected,
                    )  # fmt: skip
            finally:
                if source is not None:
                    source.close()

    def _commit_hook(
        self,
        staged: Path,
        checksum: str,
        key: KeyRecord,
        context: Context,
        digest: str,
    ) -> None:
        """Run ``on_commit``, then refuse an entry it added a member to: the member list is in
        ``entry.json`` already, so such an entry would be unreadable."""
        hook = ty.cast(ty.Callable, self.on_commit)

        def listing() -> set[str]:
            return {
                os.path.relpath(os.path.join(d, f), staged)
                for d, dirs, files in os.walk(staged)
                for f in [*dirs, *files]
            }

        before = listing()
        hook(staged, checksum, key, context, digest)
        added = sorted(listing() - before)
        if added:
            name = getattr(hook, "__qualname__", None) or repr(hook)
            raise CommitHookError(
                f"the commit hook {name} added {', '.join(added)} to the entry being "
                "committed; a hook keeps its records in a sidecar (store.write_sidecar) or "
                "the audit directory, never among the members of an entry (RFC 0003 "
                "section 7.4)"
            )

    def _compare(
        self,
        checksum: str,
        candidate: Candidate,
        outputs: ty.Mapping[str, ty.Any],
        digest: str,
        task: "Task | None",
    ) -> None:
        try:
            read = self._serve(checksum, candidate, task)
        except (_Changed, FileNotFoundError):
            read = None
        if read is None:
            logger.warning(
                "%s in %s can no longer be read, so it is not compared with the fresh outputs",
                candidate.entry.fingerprint,
                candidate.entry.root,
            )
            return
        self.reuse.compare(candidate, read[0], outputs, digest, task)

    @contextlib.contextmanager
    def _container_staging(self, root: StorageDir) -> ty.Iterator[StorageDir]:
        with self._denied(root.path):
            staging = root.child(
                self.header.dirs["staging"], create=True, mode=self.policy.side_mode
            )
        try:
            yield staging
        finally:
            staging.close()

    def _publish(
        self, root: StorageDir, staging: StorageDir, name: str, *args: ty.Any
    ) -> Committed:
        """`_publish_once`, staging again, once, if the staging directory was collected."""
        try:
            return self._publish_once(root, staging, name, *args)
        except _Vanished:
            logger.warning("the staging directory %s vanished; staging again", name)
        try:
            return self._publish_once(root, staging, uuid.uuid4().hex, *args)
        except _Vanished:
            raise StoreUnavailable(
                errno.ENOENT,
                f"the staging directory under {staging.path} vanished twice; is a collector "
                "running with a staging age shorter than a write takes?",
                staging.path,
            ) from None

    def _publish_once(
        self,
        root: StorageDir,
        staging: StorageDir,
        name: str,
        ref: EntryRef,
        shard: str,
        checksum: str,
        key: KeyRecord,
        context: Context,
        outputs: ty.Mapping[str, ty.Any],
        task: "Task | None",
        source: StorageDir | None,
        writer: str,
        verify: Candidate | None = None,
        replace: bool = False,
        expected: str | None = None,
    ) -> Committed:
        """Stage, place and clean up once. A thread touches the staging directory every
        quarter of ``ages.staging`` for as long as it is being written or placed, so `collect`
        spares it (RFC 0003 section 9). Raises `_Vanished` if the directory was collected.
        """
        policy = self.policy
        with self._denied(staging.path):
            stage = staging.child(name, create=True, mode=policy.staging_mode)
        try:
            with _Toucher(staging, name, policy.ages.staging / 4) as toucher:
                toucher.touch()
                try:
                    digest = self.format.stage(
                        stage, checksum, key, context, outputs, task, source, writer
                    )
                    if verify is not None:
                        self._compare(checksum, verify, outputs, digest, task)
                    if self.on_commit is not None:
                        self._commit_hook(
                            Path(stage.path), checksum, key, context, digest
                        )
                    stage.set_mode(policy.entry_mode)
                    if policy.durability == "member":
                        stage.sync()
                    elif policy.durability == "entry" and hasattr(os, "sync"):
                        os.sync()
                finally:
                    stage.close()
                if replace and expected is not None:
                    self._evict_for_rerun(root, ref, expected)
                committed = self._place(
                    root, staging, name, ref, shard, checksum, task, writer, digest
                )
            if committed.kept is not None and not os.path.lexists(committed.kept):
                raise _Vanished
        except _Vanished:
            raise
        except BaseException as err:
            if isinstance(err, OSError) and not self._staged(staging, name):
                raise _Vanished from err
            _discard(staging, name)
            if isinstance(err, OSError) and not isinstance(err, StoreUnavailable):
                raise StoreUnavailable(err.errno, f"{staging.path}: {err}") from err
            raise
        if committed.stored:
            _discard(staging, name)  # renamed away, or equal to the occupant
        return committed

    def _evict_for_rerun(self, root: StorageDir, ref: EntryRef, expected: str) -> None:
        """Move the entry a rerun replaces to ``quarantine/``, if it is still the one the job
        read (RFC 0003 section 10). An entry that cannot be moved stays, and the rename that
        follows loses the race. The provenance under ``audit/`` is not removed with it: the
        rerun's own attempt is recorded there, at the same address."""
        try:
            self._evict(root, ref, expected)
        except (StoreNotWritable, EntryFormatUnknown, EntryNotTrusted) as err:
            logger.warning("the rerun could not evict %s: %s", ref.fingerprint, err)
        except FileNotFoundError:
            pass  # gone already
        except OSError as err:
            logger.warning(
                "the rerun could not evict %s: %s: %s", ref.fingerprint, ref.cdir, err
            )

    @staticmethod
    def _staged(staging: StorageDir, name: str) -> bool:
        try:
            staging.lstat(name)
        except FileNotFoundError:
            return False
        return True

    def _place(
        self,
        root: StorageDir,
        staging: StorageDir,
        name: str,
        ref: EntryRef,
        shard: str,
        checksum: str,
        task: "Task | None",
        writer: str,
        digest: str,
    ) -> Committed:
        """Rename the staged entry to its address, settling a lost race (RFC 0003 section 5.1
        steps 6 to 8). Contention (another writer replaced a container, or the occupant came or
        went) is retried with jittered backoff until the visibility timeout; a corrupt occupant
        is set aside a bounded number of times within the same timeout. Exhausting either keeps
        the outputs: ``stored=False``, ``kept`` the staged entry."""
        kept = Committed(
            ref, False, digest, None, False, os.path.join(staging.path, name)
        )
        tries = CONTENTION.tries(self.visibility_timeout)
        corrupt = _ATTEMPTS
        while True:
            try:
                try:
                    result = self._attempt(
                        root, staging, name, ref, shard, checksum, task, writer, digest,
                        tries,
                    )  # fmt: skip
                except OSError as err:
                    if err.errno != _STALE:
                        raise
                    result = _CONTENDED
                if result is _CONTENDED:
                    _pause(tries)
                    continue
            except _Contended:
                return kept
            if result is _CLEARED:
                corrupt -= 1
                if corrupt <= 0 or tries.last:
                    return kept
                continue
            return ty.cast(Committed, result)

    def _attempt(
        self,
        root: StorageDir,
        staging: StorageDir,
        name: str,
        ref: EntryRef,
        shard: str,
        checksum: str,
        task: "Task | None",
        writer: str,
        digest: str,
        tries: Tries,
    ) -> "Committed | object":
        handles = self._cdir_dir(root, staging, shard, ref.cdir, tries)
        try:
            cdir_dir = handles[-1]
            try:
                with self._denied(cdir_dir.path):
                    staging.rename(name, cdir_dir, ref.fingerprint)
            except StoreUnavailable as err:
                if not self._occupied(cdir_dir, ref.fingerprint):
                    if not self._staged(staging, name):
                        raise _Vanished from err
                    if err.errno in (
                        errno.ENOENT,
                        _STALE,
                        errno.EEXIST,
                        errno.ENOTEMPTY,
                        errno.ENOTDIR,
                    ):
                        return (
                            _CONTENDED  # a container was replaced, or the occupant went
                        )
                    raise  # a refusal to write, or a failure, is not a lost race
            else:
                return self._settled(root, staging, name, handles, ref, digest)
            verdict = self._lost(root, cdir_dir, ref, checksum, task, writer)
            if verdict is _OURS:
                return self._settled(root, staging, name, handles, ref, digest)
            if verdict is _CONTENDED or verdict is _CLEARED:
                return verdict
            stored = verdict == digest
            return Committed(
                ref,
                False,
                digest,
                ty.cast("str | None", verdict),
                stored,
                None if stored else os.path.join(staging.path, name),
            )
        finally:
            for h in handles:
                h.close()

    def _settled(
        self,
        root: StorageDir,
        staging: StorageDir,
        name: str,
        handles: list[StorageDir],
        ref: EntryRef,
        digest: str,
    ) -> "Committed | object":
        """Seal the entry just placed, unless a container above it was replaced meanwhile, in
        which case it is moved back to ``staging`` and the caller places it again."""
        if not self._attached(root, handles):
            handles[-1].rename(ref.fingerprint, staging, name)
            return _CONTENDED
        self._seal(handles[-1], ref.fingerprint)
        return Committed(ref, True, digest, digest, True, None)

    @staticmethod
    def _occupied(cdir_dir: StorageDir, fingerprint: str) -> bool:
        try:
            cdir_dir.lstat(fingerprint)
        except FileNotFoundError:
            return False
        return True

    def _narrow(self, parent: StorageDir, name: str) -> None:
        if self.policy.sealed_mode is not None:
            parent.set_mode(self.policy.sealed_mode, name)

    def _seal(self, cdir_dir: StorageDir, fingerprint: str) -> None:
        """Narrow the committed entry to the sealed mode and flush its parent; a failure to
        seal leaves a readable entry that its owner's next read seals."""
        try:
            self._narrow(cdir_dir, fingerprint)
        except OSError as err:
            logger.warning(
                "could not seal %s: %s", os.path.join(cdir_dir.path, fingerprint), err
            )
        if self.policy.durability == "member":
            with contextlib.suppress(OSError):
                cdir_dir.sync()

    def _lost(
        self,
        root: StorageDir,
        cdir_dir: StorageDir,
        ref: EntryRef,
        checksum: str,
        task: "Task | None",
        writer: str,
    ) -> "str | None | object":
        """The occupant of the address after a rename that did not take: ``_OURS`` when it
        carries this writer's token (the rename succeeded and its reply was lost),
        ``_CONTENDED`` when it vanished or was replaced meanwhile, ``_CLEARED`` after a corrupt
        one was set aside, otherwise its digest, or None if this store cannot read it.
        """
        try:
            entry = cdir_dir.child(ref.fingerprint)
        except FileNotFoundError:
            return _CONTENDED
        except WrongKindError:
            try:
                self._set_aside(root, cdir_dir, ref.fingerprint)
            except WrongKindError as err:
                logger.warning("%s", err)
                return None
            return _CLEARED
        except OSError as err:
            if err.errno == _STALE:
                raise
            return None
        with entry:
            identity = entry.fstat()
            try:
                try:
                    if self.format.identify(entry)[1] == writer:
                        return _OURS
                except CorruptEntry:
                    pass
                stored = self.format.read(
                    entry, checksum, ref.fingerprint, None, self.trust, root=root
                )
            except CorruptEntry:
                pass
            except FileNotFoundError:
                return _CONTENDED
            except OSError as err:  # unreadable, not trusted, or of another format
                if err.errno == _STALE:
                    raise
                return None
            except (EntryFormatUnknown, EntryNotTrusted):
                return None
            else:
                self._reseal(entry)
                return stored.digest
        cleared = self._quarantine(root, cdir_dir, ref.fingerprint, ref.cdir, identity)
        if cleared is None:
            return None
        return _CLEARED if cleared else _CONTENDED

    def _reseal(self, entry: StorageDir) -> None:
        """Give an entry that this user owns the mode the header expects, as an entry whose
        writer was killed before the seal has the ``entry`` mode (RFC 0003 section 6.2).
        """
        if IS_WINDOWS or not self._writable():
            return
        expected = (
            self.policy.sealed_mode
            if self.policy.sealed_mode is not None
            else self.policy.entry_mode
        )
        st = entry.fstat()
        if st.st_uid != os.geteuid() or stat.S_IMODE(st.st_mode) == expected:
            return
        with contextlib.suppress(OSError):
            entry.set_mode(expected)

    # -- evict and collect --------------------------------------------------------------------------

    def evict(self, entry: EntryRef, expected_digest: str) -> None:
        """Move the entry at ``entry`` to ``quarantine/`` if its output digest is
        ``expected_digest`` (RFC 0003 sections 6.3 and 10).

        The move is the compare-and-move of a quarantine, built from the inode of the entry that
        was read, and the moved copy is checked again: an entry committed at the address after
        the caller read the old one is put back, never removed. An absent entry, or one whose
        digest differs, is left alone. A corrupt entry is quarantined.

        Raises
        ------
        StoreNotWritable
            The entry is to be moved (its digest is the expected one, or it is corrupt) and this
            user cannot move it (it is not the owner and the parent is not writable).
        EntryFormatUnknown, EntryNotTrusted
            The entry cannot be judged and is left in place.
        ValueError
            ``entry`` is in a root this store does not hold.
        """
        if os.path.abspath(entry.root) != os.path.abspath(self.path):
            raise ValueError(
                f"the entry is in the root {entry.root!r}, which this store ({self.path!r}) "
                "does not hold"
            )
        with self._open() as root:
            try:
                self._evict(root, entry, expected_digest)
                if not self._exists(root, entry):
                    self._forget_audit(root, entry)
            except FileNotFoundError:
                self._forget_audit(root, entry)
                return
            except OSError as err:
                if isinstance(err, StoreUnavailable):
                    raise
                raise StoreUnavailable(err.errno, f"{entry.cdir}: {err}") from err

    def _exists(self, root: StorageDir, ref: EntryRef) -> bool:
        try:
            with root.child(self.header.entries) as entries:
                with entries.child(self._shard_of(entries, ref.cdir)) as shard:
                    with shard.child(ref.cdir) as cdir:
                        cdir.lstat(ref.fingerprint)
        except (FileNotFoundError, WrongKindError):
            return False
        return True

    def _evict(self, root: StorageDir, ref: EntryRef, expected: str) -> None:
        with root.child(self.header.entries) as entries:
            shard = self._shard_of(entries, ref.cdir)
            with (
                entries.child(shard) as shard_dir,
                shard_dir.child(ref.cdir) as cdir_dir,
            ):
                try:
                    held = cdir_dir.child(ref.fingerprint)
                except WrongKindError:
                    self._set_aside(root, cdir_dir, ref.fingerprint)
                    return
                with held:
                    identity = held.fstat()
                    try:
                        checksum = self._checksum_of(held, ref.cdir)
                        stored = self.format.read(
                            held,
                            checksum,
                            ref.fingerprint,
                            None,
                            self.trust,
                            root=root,
                        )
                    except CorruptEntry:
                        expected = None  # nothing to protect: any corrupt copy goes
                    else:
                        if stored.digest != expected:
                            return
                if (
                    self._quarantine(
                        root,
                        cdir_dir,
                        ref.fingerprint,
                        ref.cdir,
                        identity,
                        expected,
                    )
                    is None
                ):
                    where = os.path.join(cdir_dir.path, ref.fingerprint)
                    raise StoreNotWritable(
                        errno.EACCES,
                        f"this user cannot move {where!r} aside (owner: {_owner(where)})",
                        where,
                    )

    def collect(self) -> CollectReport:
        """Remove what a store leaves behind once it is older than the policy's ages (RFC 0003
        section 9): the children of ``staging/`` and header temporaries (``ages.staging``), the
        children of ``quarantine/`` and the ``q-`` and aside names in container directories
        (``ages.quarantine``). Age is the later of the modification and change times, and a moved
        object is also stamped, so an object moved a moment ago is young. A live writer touches its
        staging directory at least once per quarter of ``ages.staging``. The walk visits every shard
        and container directory.

        An entry is never removed, and nothing outside those names is read or touched (sidecars
        included). Each object is claimed with a rename to an inode-derived ``gc-`` name before it
        is removed, so a mover that put it back first wins and two collectors do not both remove it.

        A default work directory (``<root>/work/<name>``, never one the caller's ``work_dir``
        names) is removed once it and everything directly in it, ``_committed.json`` included, is
        older than ``ages.work``, which must exceed the longest run; a resumable job's directory
        (``work/resume-<hash>``) is kept while the lease of the same hash is live. ``<root>/audit/``
        is removed by `evict` with its entry, not here.

        The aside names are the entry format's ``aside_names``, so an adopted store's own set-aside
        entries are collected (section 8.2). A directory named like a version directory that is not
        this store's is logged once, never read or removed.

        Returns
        -------
        CollectReport
            The paths removed, and the paths kept with the reason.
        """
        report = CollectReport()
        ages = self.policy.ages
        now = _now()
        aside = self.format.aside_names()
        with self._open() as root:
            self._warn_strays(root)
            for e in list(root.scandir()):
                if _TEMP_RE.fullmatch(e.name):
                    self._sweep(root, e, ages.staging, now, report)
            for side, age in (
                (self.header.dirs["staging"], ages.staging),
                (self.header.dirs["quarantine"], ages.quarantine),
            ):
                try:
                    d = root.child(side)
                except (FileNotFoundError, WrongKindError):
                    continue
                with d:
                    for e in list(d.scandir()):
                        self._sweep(d, e, age, now, report)
            self._collect_work(root, ages.work, now, report)
            try:
                entries = root.child(self.header.entries)
            except (FileNotFoundError, WrongKindError):
                return report
            with entries:
                for shard in list(entries.scandir()):
                    if not shard.is_dir():
                        continue
                    for cdir in self._children(entries, shard.name):
                        with self._opened(entries, shard.name, cdir.name) as holder:
                            if holder is None:
                                continue
                            for e in list(holder.scandir()):
                                if e.is_dir() and (
                                    e.name.startswith(("q-", "gc-"))
                                    or aside.fullmatch(e.name)
                                ):
                                    self._sweep(holder, e, ages.quarantine, now, report)
        return report

    def _collect_work(
        self, root: StorageDir, age: float, now: float, report: CollectReport
    ) -> None:
        try:
            work = root.child(_WORK)
        except (FileNotFoundError, WrongKindError):
            return
        with work:
            for e in list(work.scandir()):
                if not _ATTEMPT.fullmatch(e.name):
                    continue
                where = os.path.join(work.path, e.name)
                try:
                    newest = self._newest_in(work, e.name)
                except FileNotFoundError:
                    continue
                except OSError as err:
                    report.kept.append(
                        (
                            where,
                            f"cannot be read: {err.strerror or err} (owner: {_owner(where)})",
                        )
                    )
                    continue
                if newest is not None and now - newest < age:
                    report.kept.append((where, f"younger than {age:g} s"))
                    continue
                if e.name.startswith(_RESUME) and self._lease_live(
                    root, e.name[len(_RESUME) :], now
                ):
                    report.kept.append((where, "its lease is held"))
                    continue
                self._sweep(work, e, age, now, report)

    def _lease_live(self, root: StorageDir, name: str, now: float) -> bool:
        """Whether the lease ``<root>/leases/<name>.lock`` exists and was renewed within the
        larger of this store's ``stale_after`` and its holder's; one that cannot be read is
        taken as live."""
        try:
            with root.child(_LEASES) as leases:
                newest = self._newest_in(leases, f"{name}.lock")
                record, unreadable = self._holder_record(leases, f"{name}.lock")
        except (FileNotFoundError, WrongKindError):
            return False
        except OSError:
            return True
        if newest is None:  # not a directory: not a lease this store takes
            return False
        return now - newest < holder_stale_after(
            record, self.lease_settings, unreadable=unreadable
        )

    @staticmethod
    def _holder_record(leases: StorageDir, name: str) -> tuple[dict | None, bool]:
        """The holder record of the lease ``name`` (None if none can be read), and whether
        a record exists that could not be read."""

        def read(entry: str) -> bytes:
            try:
                with lease.child(entry) as t, t.open_read(HOLDER_FILE) as f:
                    return f.read()
            except WrongKindError as err:
                raise NotADirectoryError(str(err)) from err

        try:
            with leases.child(name) as lease:
                return read_holder_record([e.name for e in lease.scandir()], read)
        except (OSError, WrongKindError):
            return None, False

    @staticmethod
    def _newest_in(parent: StorageDir, name: str) -> float | None:
        """The latest change of the directory ``name`` or anything directly in it."""
        try:
            with parent.child(name) as d:
                newest = _last_change(d.fstat())
                for c in list(d.scandir()):
                    with contextlib.suppress(FileNotFoundError):
                        newest = max(newest, _last_change(d.lstat(c.name)))
                return newest
        except WrongKindError:
            return None

    @staticmethod
    def _children(parent: StorageDir, name: str) -> list[StorageEntry]:
        try:
            with parent.child(name) as d:
                return [e for e in d.scandir() if e.is_dir()]
        except (FileNotFoundError, WrongKindError):
            return []

    @contextlib.contextmanager
    def _opened(
        self, entries: StorageDir, shard: str, cdir: str
    ) -> ty.Iterator[StorageDir | None]:
        try:
            with entries.child(shard) as s, s.child(cdir) as c:
                yield c
        except (FileNotFoundError, WrongKindError):
            yield None

    def _sweep(
        self,
        parent: StorageDir,
        listed: StorageEntry,
        age: float,
        now: float,
        report: CollectReport,
    ) -> None:
        """Remove ``listed`` from ``parent`` if it is older than ``age``, recording the outcome."""
        where = os.path.join(parent.path, listed.name)
        try:
            # not the listing's stat, which can lack the inode (Windows)
            seen = parent.lstat(listed.name)
        except FileNotFoundError:
            return
        if _is_link(seen) or not stat.S_ISDIR(seen.st_mode):
            if now - _last_change(seen) < age:
                report.kept.append((where, f"younger than {age:g} s"))
                return
            self._finish(parent, listed.name, seen, False, where, report)
            return
        if now - _last_change(seen) < age:
            report.kept.append((where, f"younger than {age:g} s"))
            return
        try:
            with parent.child(listed.name) as h:
                held = h.fstat()
        except FileNotFoundError:
            return
        except WrongKindError:
            report.kept.append((where, "replaced by an object of another kind"))
            return
        if (held.st_dev, held.st_ino) != (seen.st_dev, seen.st_ino) or (
            now - _last_change(held) < age
        ):
            report.kept.append((where, "changed since it was listed"))
            return
        self._finish(parent, listed.name, held, True, where, report)

    def _finish(
        self,
        parent: StorageDir,
        name: str,
        st: os.stat_result,
        is_dir: bool,
        where: str,
        report: CollectReport,
    ) -> None:
        claimed = name
        try:
            if is_dir:
                if not name.startswith("gc-"):
                    claimed = _gc_name(st)
                    try:
                        parent.rename(name, parent, claimed)
                    except FileNotFoundError:
                        return  # put back, or removed by another collector
                    except OSError as err:
                        if err.errno in (errno.ENOTEMPTY, errno.EEXIST):
                            return  # another collector claimed this very object
                        raise
                _remove_tree(parent, claimed)
            else:
                now = parent.lstat(name)
                if (now.st_dev, now.st_ino) != (st.st_dev, st.st_ino):
                    report.kept.append((where, "changed since it was listed"))
                    return
                _unlink(parent, name)
        except FileNotFoundError:
            pass
        except OSError as err:
            report.kept.append(
                (
                    where,
                    f"cannot remove it: {err.strerror or err} (owner: {_owner(where)})",
                )
            )
            logger.warning(
                "collect could not remove %s (owner %s): %s", where, _owner(where), err
            )
            return
        report.removed.append(where)

    # -- read -----------------------------------------------------------------------------------

    def read(self, entry: EntryRef, task: "Task | None") -> Result:
        """Read and verify the entry at ``entry`` and return its `Result` (RFC 0003 sections
        6.2 and 7.2). The reuse hook is not consulted: this is a job's own result.

        The outputs are built as ``task``'s ``Outputs``; with no task they are a ``dict``.
        A corrupt or misfiled entry is moved to ``quarantine/`` and `CorruptEntry` raised; an
        entry this store cannot read (`EntryFormatUnknown`, `EntryNotTrusted`) is left in place.

        Raises
        ------
        FileNotFoundError
            There is no entry at the address.
        CorruptEntry, EntryFormatUnknown, EntryNotTrusted
            See `EntryFormat.read`.
        StoreUnavailable
            Any other ``OSError``, or an entry whose outputs are not the task's.
        ValueError
            ``entry`` is in a root this store does not hold.
        """
        if os.path.abspath(entry.root) != os.path.abspath(self.path):
            raise ValueError(
                f"the entry is in the root {entry.root!r}, which this store ({self.path!r}) "
                "does not hold"
            )
        with self._open() as root:
            try:
                return self._read(root, entry, task)
            except OSError as err:
                if isinstance(err, (StoreUnavailable, FileNotFoundError)):
                    raise
                raise StoreUnavailable(err.errno, f"{entry.cdir}: {err}") from err

    def _shard_of(self, entries: StorageDir, cdir: str) -> str:
        rule = self.header.shard
        shard = shard_for(cdir, cdir, rule)
        if rule == "sha256-2" and HASHED_CDIR.fullmatch(cdir):
            with contextlib.suppress(FileNotFoundError):
                with entries.child(shard) as shard_dir:
                    shard_dir.lstat(cdir)
                return shard
            for e in sorted(entries.scandir(), key=lambda e: e.name):
                if e.is_dir() and len(e.name) == 2:
                    with entries.child(e.name) as shard_dir:
                        try:
                            shard_dir.lstat(cdir)
                        except FileNotFoundError:
                            continue
                    return e.name
        return shard

    def _read(
        self,
        root: StorageDir,
        ref: EntryRef,
        task: "Task | None",
        shard: str | None = None,
        served: bool = False,
    ) -> Result:
        """The entry ``ref`` read; ``served`` gives the result the `Committed` that names
        the entry, for `record_served`."""
        stored, where = self._read_entry(root, ref, task, shard)
        result = self._result_at(stored, where, task)
        if served:
            result.committed = Committed(
                ref, False, stored.digest, stored.digest, True, None, True
            )
        return result

    def _read_entry(
        self,
        root: StorageDir,
        ref: EntryRef,
        task: "Task | None",
        shard: str | None = None,
        *,
        trust: Trust | None = None,
        readonly: bool = False,
        digest: str | None = None,
    ) -> tuple[StoredEntry, Path]:
        """Open the entry at ``ref`` below ``root`` and read and verify it (RFC 0003 section 6.2).

        ``readonly`` is a root this store never writes: a corrupt entry is not moved aside or
        resealed, only raised. ``digest``, if given, is the sha256 of the ``context.json`` that
        was ranked, which must be the bytes read through the descriptor opened here, or
        `_Changed` is raised, so a verdict applies to the members it reads.
        """
        with root.child(self.header.entries) as entries:
            if shard is None:
                shard = self._shard_of(entries, ref.cdir)
            with (
                entries.child(shard) as shard_dir,
                shard_dir.child(ref.cdir) as cdir_dir,
            ):
                try:
                    entry = cdir_dir.child(ref.fingerprint)
                except WrongKindError as err:
                    if not readonly:
                        self._set_aside(root, cdir_dir, ref.fingerprint)
                    raise CorruptEntry(str(err)) from err
                with entry:
                    identity = entry.fstat()
                    try:
                        if digest is not None:
                            raw = self.format.context(entry)[1]
                            if hashlib.sha256(raw).hexdigest() != digest:
                                raise _Changed
                        checksum = self._checksum_of(entry, ref.cdir)
                        stored = self.format.read(
                            entry,
                            checksum,
                            ref.fingerprint,
                            task,
                            self.trust if trust is None else trust,
                            root=root,
                        )
                    except CorruptEntry as err:
                        if readonly:
                            raise
                        if (
                            self._quarantine(
                                root, cdir_dir, ref.fingerprint, ref.cdir, identity
                            )
                            is None
                        ):
                            where = os.path.join(cdir_dir.path, ref.fingerprint)
                            raise CorruptEntry(
                                f"{err}; {where} cannot be moved aside "
                                f"(owner: {_owner(where)})"
                            ) from err
                        raise
                    if not readonly:
                        self._reseal(entry)
                    return stored, Path(entry.path)

    def _result(
        self, stored: StoredEntry, entry: StorageDir, task: "Task | None"
    ) -> Result:
        return self._result_at(stored, Path(entry.path), task)

    def _result_at(
        self, stored: StoredEntry, where: Path, task: "Task | None"
    ) -> Result:
        outputs = self._outputs_of(dict(stored.outputs), task, where)
        return Result(
            cache_dir=where,
            outputs=outputs,
            runtime=None,
            errored=False,
            task=task,
            context=dict(stored.context),
            stored=True,
            record=_record_of(stored.key),
        )

    @staticmethod
    def _outputs_of(outputs: ty.Any, task: "Task | None", where: Path) -> ty.Any:
        if task is None:
            return outputs
        from pydra.utils.general import get_fields

        cls = task.Outputs
        declared = {f.name for f in get_fields(cls)}
        if set(outputs) != declared:
            raise StoreUnavailable(
                f"the entry holds outputs {sorted(outputs)} and the task declares "
                f"{sorted(declared)}"
            )
        outputs = cls(**outputs)
        outputs._cache_dir = where
        return outputs

    # -- work directories --------------------------------------------------------------------

    def _work_path(self, checksum: str, fingerprint: str, attempt: str) -> Path:
        """The directory of ``attempt``: a pure function of its arguments (RFC 0003 section 5.4).
        A resumable job's, ``attempt="resume"``, is named by its address, so two resumable jobs
        never share one."""
        if self._work_fn is not None:
            path = Path(self._work_fn(checksum, fingerprint, attempt))
            if not path.is_absolute():
                raise ValueError(f"work_dir must return an absolute path, got {path}")
            return path
        address = f"{cdir_for(checksum)}/{fingerprint}".encode()
        if attempt == "resume":
            name = _RESUME + hashlib.sha256(address).hexdigest()
        elif "." in attempt and _ATTEMPT.fullmatch(attempt):
            # an attempt allocated outside the runner, such as a pool's <pool id>.<g>, is unique
            # per job only, so its directory is qualified by the job's address
            name = f"{attempt}-{hashlib.sha256(address).hexdigest()[:24]}"
        elif _ATTEMPT.fullmatch(attempt):
            name = attempt
        else:
            raise ValueError(f"attempt {attempt!r} is not usable as a directory name")
        return Path(self.path) / _WORK / name

    def work_path(self, checksum: str, fingerprint: str, attempt: str) -> Path:
        """The work directory of ``attempt`` of the job at ``(checksum, fingerprint)``: a pure
        function of the three (RFC 0003 section 5.4). An attempt allocated outside the runner,
        such as a pool's ``<pool id>.<g>``, is unique per job only; the default layout qualifies
        it by the job's address, and a ``work_dir`` function must use the address too.
        """
        return self._work_path(checksum, fingerprint, attempt)

    def work_dir(self, job: "Job") -> Path:
        """The directory ``job`` runs in and writes ``_committed.json``, ``_error.json``, its
        kept outputs and ``_job.pklz`` to (RFC 0003 section 5.4).

        Raises
        ------
        ValueError
            The job has no ``attempt`` and cannot resume.
        """
        attempt = "resume" if job.can_resume else job.attempt
        if attempt is None:
            raise ValueError(
                f"job {job.name!r} has no attempt, so it has no work directory: the runner "
                "gives each job an attempt when it hands it out"
            )
        return self._work_path(
            job.checksum, self.identity.fingerprint(job.context), attempt
        )

    def error_file(self, job: "Job") -> Path:
        """``_error.json`` in the work directory of ``job``"""
        return job.cache_dir / _ERROR

    def record_error(
        self, job: "Job", error: ty.Any, exception: BaseException | None = None
    ) -> Path:
        """Write `error_file` as JSON (read without unpickling), with the job's attempt"""
        return record_error_json(
            job.cache_dir, error=error, attempt=job.attempt, exception=exception
        )

    def audit_dir(self, job: "Job") -> Path:
        """Where the audit messages and resource logs of a run of ``job`` go:
        ``<root>/audit/<cdir>/<fingerprint>/<attempt>``, found from the entry's `EntryRef` and the
        attempt, and removed with the entry by `evict`."""
        return (
            Path(self.path)
            / _AUDIT
            / cdir_for(job.checksum)
            / self.identity.fingerprint(job.context)
            / (job.attempt or job.uid)
        )

    def make_audit_dir(self, job: "Job") -> Path:
        """Make `audit_dir`, and its ``messages`` directory, ``0o700``, by descriptor from the
        root."""
        path = self.audit_dir(job)
        parts = (*path.relative_to(self.path).parts, "messages")
        with self._denied(str(path)):
            with self._open() as root:
                handle = root
                opened = []
                try:
                    for name in parts:
                        handle = handle.child(name, create=True, mode=0o700)
                        opened.append(handle)
                finally:
                    for h in opened:
                        h.close()
        return path

    def _forget_audit(self, root: StorageDir, ref: EntryRef) -> None:
        """Remove the provenance of an entry that has gone."""
        with contextlib.suppress(OSError):
            with root.child(_AUDIT) as audit, audit.child(ref.cdir) as cdir:
                _remove_tree(cdir, ref.fingerprint)
            with root.child(_AUDIT) as audit:
                with contextlib.suppress(OSError):
                    audit.rmdir(ref.cdir)

    def _maybe_work_dir(self, job: "Job") -> Path | None:
        try:
            return self.work_dir(job)
        except ValueError:
            return None

    def _make_work_dir(self, path: Path) -> None:
        """Make ``path`` ``0o700`` if it is absent. A default one is made by descriptor below
        ``<root>/work``; one the caller's function names is made as a path, which is the
        caller's storage. An existing directory is left as it is."""
        with self._denied(str(path)):
            if self._work_fn is None:
                with (
                    self._open() as root,
                    root.child(_WORK, create=True, mode=0o700) as work,
                ):
                    work.child(path.name, create=True, mode=0o700).close()
                return
            os.makedirs(path.parent, exist_ok=True)
            try:
                os.mkdir(path, 0o700)
            except FileExistsError:
                return
            if not IS_WINDOWS:
                os.chmod(path, 0o700)

    def _open_work(self, path: Path) -> StorageDir:
        """The work directory as given: what is read from it is checked where it is opened
        (`check_private`, the entry format's trust), not here."""
        return StorageDir.open_root(path, self._work_policy, trust="trusted")

    def begin_run(self, job: "Job") -> Path:
        """Make the work directory of a run that is about to start, and remove the outcome files
        an earlier run left there (``_committed.json``, ``_error.json``): a resumable job's
        directory is stable across runs, and a stale outcome is never read as this run's.
        """
        path = self.work_dir(job)
        self._make_work_dir(path)
        with self._open_work(path) as handle:
            for name in (_COMMITTED, _ERROR):
                with contextlib.suppress(FileNotFoundError):
                    handle.unlink(name)
        return path

    def forget_committed(self, job: "Job") -> None:
        """Remove ``_committed.json`` once the runner holds the job's outcome (RFC 0003 section
        5.4), and a default work directory that is then empty, unless the process working
        directory is inside it (it is then left for the submitter's close, or `collect`).
        """
        path = self._maybe_work_dir(job)
        if path is None:
            return
        try:
            with self._open_work(path) as handle:
                with contextlib.suppress(FileNotFoundError):
                    handle.unlink(_COMMITTED)
        except FileNotFoundError:
            return
        if self._work_fn is None:  # not if held, or not empty (a failure, kept outputs)
            from pydra.utils.general import remove_if_unheld

            remove_if_unheld(path)
        if not path.exists():  # removed: nothing left for release_attempts
            self._attempts.discard(path)

    def note_attempt(self, job: "Job") -> None:
        if self._work_fn is not None:  # a directory the caller names is the caller's
            return
        try:
            self._attempts.add(Path(self.work_dir(job)))
        except ValueError:  # no attempt
            pass

    def release_attempts(self) -> None:
        from pydra.utils.general import dir_held, remove_if_unheld

        for path in list(self._attempts):
            if dir_held(path):
                continue  # tried again at the next release
            self._attempts.discard(path)
            remove_if_unheld(path)  # checked again with the removal, under one lock

    def _empty_work_dir(self, path: Path) -> None:
        """Empty a default work directory except ``_committed.json``. What cannot be removed
        (a Windows process whose working directory is inside) is left for `collect`."""
        try:
            with self._open_work(path) as handle:
                for e in list(handle.scandir()):
                    if e.name == _COMMITTED:
                        continue
                    try:
                        if e.is_dir():
                            _remove_tree(handle, e.name)
                        else:
                            _unlink(handle, e.name)
                    except FileNotFoundError:
                        pass
                    except OSError as err:
                        logger.warning(
                            "could not empty %s: %s", os.path.join(path, e.name), err
                        )
        except FileNotFoundError:
            pass

    @staticmethod
    def _write_file(handle: StorageDir, name: str, data: bytes, mode: int) -> None:
        """Write ``data`` to a temporary name in ``handle`` and rename it to ``name``, so a
        reader sees no file or a whole one."""
        tmp = f".{name}.{uuid.uuid4().hex}.tmp"
        try:
            with handle.open_new(tmp, mode) as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            try:
                handle.rename(tmp, handle, name)
            except (FileExistsError, PermissionError):
                if not IS_WINDOWS:
                    raise
                handle.unlink(name)  # Windows does not rename over a file
                handle.rename(tmp, handle, name)
        except BaseException:
            with contextlib.suppress(OSError):
                handle.unlink(tmp)
            raise

    def _write_committed(
        self, job: "Job", path: Path, committed: Committed, kind: str
    ) -> None:
        ref = committed.entry
        doc = {
            "format": _COMMITTED_FORMAT,
            "kind": kind,
            "checksum": job.checksum,
            "attempt": job.attempt,
            "entry": {
                "root": ref.root,
                "cdir": ref.cdir,
                "fingerprint": ref.fingerprint,
            },
            "won": committed.won,
            "own_digest": committed.own_digest,
            "committed_digest": committed.committed_digest,
            "stored": committed.stored,
            "served": committed.served,
        }
        self._make_work_dir(path)
        with self._open_work(path) as handle:
            self._write_file(
                handle, _COMMITTED, json.dumps(doc, sort_keys=True).encode(), 0o600
            )

    def _keep_entry(self, path: Path, committed: Committed) -> Committed:
        """Move the staged entry of a result that was not stored to ``<work dir>/_entry``
        (RFC 0003 section 5.4): by rename on one filesystem, otherwise by copy with each member
        re-hashed against ``entry.json``; the staging directory is then removed."""
        assert committed.kept is not None
        name = os.path.basename(committed.kept)
        self._make_work_dir(path)
        with self._open() as root, root.child(self.header.dirs["staging"]) as staging:
            if os.path.dirname(committed.kept) != staging.path:
                return committed
            with self._open_work(path) as handle:
                with contextlib.suppress(FileNotFoundError):
                    _remove_tree(handle, _KEPT_ENTRY)  # an earlier run's
                try:
                    staging.rename(name, handle, _KEPT_ENTRY)
                except OSError as err:
                    if err.errno != errno.EXDEV:
                        raise
                    try:
                        self._copy_kept(staging, name, handle)
                    except BaseException:
                        _discard(handle, _KEPT_ENTRY)
                        raise
                    _discard(staging, name)
        return attrs.evolve(committed, kept=str(path / _KEPT_ENTRY))

    def _copy_kept(self, staging: StorageDir, name: str, handle: StorageDir) -> None:
        with staging.child(name) as source:
            try:
                members = self.format.member_records(source)
            except ValueError as err:
                raise CorruptEntry(f"the kept entry's entry.json is malformed: {err!r}")
            with handle.child(
                _KEPT_ENTRY, create=True, mode=self.policy.entry_mode
            ) as target:
                self._copy_tree(source, target, "", members)

    def _copy_tree(
        self,
        source: StorageDir,
        target: StorageDir,
        rel: str,
        members: ty.Mapping[str, ty.Any],
    ) -> None:
        for e in list(source.scandir()):
            if e.is_dir():
                with (
                    source.child(e.name) as s,
                    target.child(e.name, create=True, mode=self.policy.entry_mode) as t,
                ):
                    self._copy_tree(s, t, f"{rel}{e.name}/", members)
                continue
            digest, size = copy_hashed(
                Path(source.path) / e.name, target, e.name, self.policy.file_mode
            )
            rec = members.get(f"{rel}{e.name}")
            if rec is None and f"{rel}{e.name}" != "entry.json":
                raise CorruptEntry(
                    f"{rel}{e.name} is in the kept entry but not in entry.json"
                )
            if rec is not None and (rec.get("sha256"), rec.get("size")) != (
                digest,
                size,
            ):
                raise CorruptEntry(
                    f"{rel}{e.name} changed while the kept entry was copied: it does not "
                    "match entry.json"
                )

    def _keep_outputs(
        self,
        job: "Job",
        path: Path,
        outputs: ty.Mapping[str, ty.Any],
        ref: EntryRef,
    ) -> Committed:
        """Pickle outputs that no codec encodes to ``<work dir>/_outputs.pklz`` (RFC 0003
        section 5.4); they are read back only after an ownership check."""
        data = cp.dumps(dict(outputs))
        self._make_work_dir(path)
        with self._open_work(path) as handle:
            self._write_file(handle, _OUTPUTS, data, 0o600)
        return Committed(
            ref,
            False,
            hashlib.sha256(data).hexdigest(),
            None,
            False,
            str(path / _OUTPUTS),
        )

    # -- reading a job's own result ------------------------------------------------------------

    has_work_dirs = True
    finds = True

    def holds(self, checksum: str, context: Context) -> bool:
        return self.entry_path(checksum, context).is_dir()

    def declared_code_reuse(
        self,
        checksum: str,
        stored: "list[tuple[Path, Context]]",
        context: Context,
        task: "Task | None",
    ) -> "Context | None":
        """The context of the first entry from another fingerprint that the reuse hook
        ranks and would serve for ``context``, or None"""
        mine = self.identity.fingerprint(context)
        for candidate in self.reuse.rank(context, self.stored(checksum), task):
            if candidate.entry.fingerprint == mine:
                continue
            verdict = self.reuse.verdict(candidate, context, task)
            if verdict is Verdict.SKIP:
                continue
            return candidate.context if verdict is Verdict.SERVE else None
        return None

    def entry_path(self, checksum: str, context: Context) -> Path:
        """Where the entry of ``checksum`` under ``context`` is, or would be. Nothing is read."""
        cdir = cdir_for(checksum)
        shard = shard_for(cdir, checksum, self.header.shard)
        return (
            Path(self.path)
            / self.header.entries
            / shard
            / cdir
            / self.identity.fingerprint(context)
        )

    def record_served(self, job: "Job", result: Result) -> None:
        """Record the entry served to ``job`` as its outcome: ``<work dir>/_committed.json``
        and ``job.committed``, so that a reader in another process reads that entry (RFC
        0003 section 7.2)."""
        committed = result.committed
        if committed is None:
            return
        self._write_committed(job, self.work_dir(job), committed, "entry")
        job.committed = committed

    def read_own(self, job: "Job") -> Result | None:
        """The result stored at the job's own address, under the lookup context's fingerprint,
        carrying the `Committed` that names it; None if there is none, or what is there is
        corrupt (moved to ``quarantine/``) or cannot be read here (left in place, not
        served)."""
        cdir = cdir_for(job.checksum)
        ref = EntryRef(self.path, cdir, self.identity.fingerprint(job.context))
        try:
            return self._read_ref(
                ref,
                job.task,
                shard_for(cdir, job.checksum, self.header.shard),
                served=True,
            )
        except FileNotFoundError:
            return None
        except CorruptEntry as err:
            logger.warning(
                "%s is corrupt and was moved aside: %s", ref.fingerprint, err
            )
            return None
        except (EntryFormatUnknown, EntryNotTrusted) as err:
            logger.warning("not served: %s", err)
            return None

    def _read_readonly(
        self, ref: EntryRef, task: "Task | None", shard: str, trust: Trust
    ) -> Result:
        """Read an entry of a read-only root under that root's trust, writing nothing."""
        try:
            with StorageDir.open_root(ref.root, self.policy, trust=trust) as root:
                stored, where = self._read_entry(
                    root, ref, task, shard, trust=trust, readonly=True
                )
        except OSError as err:
            if isinstance(err, (StoreUnavailable, FileNotFoundError)):
                raise
            raise StoreUnavailable(err.errno, f"{ref.cdir}: {err}") from err
        return self._result_at(stored, where, task)

    def _read_ref(
        self, ref: EntryRef, task: "Task | None", shard: str, served: bool = False
    ) -> Result:
        with self._open() as root:
            try:
                return self._read(root, ref, task, shard, served)
            except OSError as err:
                if isinstance(err, (StoreUnavailable, FileNotFoundError)):
                    raise
                raise StoreUnavailable(err.errno, f"{ref.cdir}: {err}") from err

    def read_result(self, job: "Job", *, settled: bool = False) -> Result | None:
        """The result of ``job`` as its readers see it (RFC 0003 section 7.2): by
        ``job.committed`` when this process holds it, else by ``<work dir>/_committed.json``
        (after the visibility retry when ``settled``), else by the entry at the job's own
        address. An entry that cannot be read is a failure, not an absence.

        Returns
        -------
        Result or None
            None when nothing is found and the job is not ``settled``; an errored `Result`
            (``failure`` names the work directory) when it failed, or when it is ``settled``
            and nothing is found, or what was found cannot be read.
        """
        wd = self._maybe_work_dir(job)
        try:
            if job.committed is not None:
                found = self._read_committed(job, wd, job.committed, settled)
            else:
                found = None
                doc = self._committed_doc(job, wd, settled) if wd is not None else None
                if doc is not None:
                    found = self._read_committed(
                        job, wd, self._committed_of(doc, wd), settled
                    )
            if found is None:
                found = self.read_own(job)
        except FileNotFoundError:
            found = self.read_own(job)
        except OSError as err:
            return self._failed(job, wd, f"its result cannot be read: {err}")
        if found is not None:
            return found
        if wd is not None and self._has_error(wd, job.attempt):
            return self._failed(job, wd, f"the job failed; see {wd / _ERROR}")
        if settled:
            return self._failed(job, wd, "no result of the job was found")
        return None

    def read_committed(self, job: "Job", committed: Committed) -> Result | None:
        """The result the process that ran ``job`` holds as ``committed``: its entry, or the
        outputs kept in the job's work directory."""
        return self._read_committed(job, self._maybe_work_dir(job), committed)

    def _read_committed(
        self,
        job: "Job",
        wd: Path | None,
        committed: Committed,
        settled: bool = False,
    ) -> Result | None:
        if committed.stored:
            ref = committed.entry
            shard = shard_for(ref.cdir, job.checksum, self.header.shard)
            try:
                trust, readonly = self._kind_of(ref.root)
            except ValueError:
                trust, readonly = self.trust, False
                ref = attrs.evolve(ref, root="")
            if ref.cdir != cdir_for(job.checksum) or not ref.root:
                raise CorruptEntry(
                    f"the committed entry {ref.cdir} is not the entry of the job"
                )
            result = self._read_recorded(
                job, ref, shard, trust, readonly, committed, settled
            )
            if committed.served and ref.fingerprint != self.identity.fingerprint(
                job.context
            ):  # served from another context, as `find` reported it
                result.cross_context = True
                result.reused_in = dict(job.context)
        elif wd is None or committed.kept is None:
            return None
        elif os.path.basename(committed.kept) == _KEPT_ENTRY:
            result = self._read_kept(job, wd, committed)
        elif os.path.basename(committed.kept) == _OUTPUTS:
            result = self._read_outputs(job, wd, committed)
        else:
            return None
        result.committed = committed
        result.stored = committed.stored
        return result

    def _read_recorded(
        self,
        job: "Job",
        ref: EntryRef,
        shard: str,
        trust: Trust,
        readonly: bool,
        committed: Committed,
        settled: bool = False,
    ) -> Result:
        """The entry a record names, which must still be the one that was recorded: its output
        digest is ``committed.committed_digest``. A rerun replaces an entry at its address, so
        an address alone no longer names its content. A mismatch is raised at once, and when
        ``settled`` retried until the visibility timeout first (a stale lookup on shared
        storage): an unsettled reader is called from an event loop and never sleeps."""
        tries = SETTLED_READ.tries(self.visibility_timeout if settled else 0.0)
        for _ in tries:
            with self._open_root_for(ref, trust, readonly) as root:
                stored, where = self._read_entry(
                    root, ref, job.task, shard, trust=trust, readonly=readonly
                )
            expected = committed.committed_digest
            if expected is None or stored.digest == expected:
                return self._result_at(stored, where, job.task)
        raise StoreUnavailable(
            errno.ESTALE,
            f"{ref.cdir}/{ref.fingerprint} was replaced since it was recorded "
            "(a rerun, or an entry committed over an evicted one)",
        )

    def _open_root_for(self, ref: EntryRef, trust: Trust, readonly: bool) -> StorageDir:
        if readonly:
            return StorageDir.open_root(ref.root, self.policy, trust=trust)
        return self._open()

    def _work_trust(self, handle: StorageDir, name: str) -> Trust:
        """The trust to read ``name`` in a work directory with: ``private`` if it is owned by
        this user and nobody else can write it, which a directory this user ran a job in is,
        otherwise the store's (a pickle member is then refused for a ``shared`` store).
        """
        if self.trust == "trusted":
            return "trusted"
        try:
            check_private(handle, [name])
        except (EntryNotTrusted, FileNotFoundError):
            return self.trust
        return "private"

    def _read_kept(self, job: "Job", wd: Path, committed: Committed) -> Result:
        with self._open_work(wd) as handle:
            trust = self._work_trust(handle, _KEPT_ENTRY)
            with handle.child(_KEPT_ENTRY) as entry:
                stored = self.format.read(
                    entry,
                    job.checksum,
                    committed.entry.fingerprint,
                    job.task,
                    trust,
                    root=handle,
                )
                if stored.digest != committed.own_digest:
                    raise CorruptEntry(
                        f"the kept entry in {wd} is not the one that was committed"
                    )
                return self._result(stored, entry, job.task)

    def _read_outputs(self, job: "Job", wd: Path, committed: Committed) -> Result:
        with self._open_work(wd) as handle:
            if self.trust != "trusted":
                check_private(handle, [_OUTPUTS])
            with handle.open_read(_OUTPUTS) as f:
                data = read_capped(f, _OUTPUTS, cap=None)
        if hashlib.sha256(data).hexdigest() != committed.own_digest:
            raise CorruptEntry(
                f"{wd / _OUTPUTS} is not the outputs that were committed"
            )
        outputs = self._outputs_of(_loads(data), job.task, wd)
        return Result(
            cache_dir=wd,
            outputs=outputs,
            runtime=None,
            errored=False,
            task=job.task,
            context=dict(job.context),
            stored=False,
            record=_record_of(self.identity.key(job)),
        )

    def _committed_doc(self, job: "Job", wd: Path, settled: bool) -> dict | None:
        """``_committed.json`` of ``wd``. When ``settled``, a missing one is waited for up to
        the visibility timeout (another client's rename can be late to show on shared storage),
        unless the job's ``_error.json`` shows."""
        tries = SETTLED_READ.tries(self.visibility_timeout if settled else 0.0)
        for _ in tries:
            try:
                with self._open_work(wd) as handle:
                    doc = self._load_committed(handle, wd, job.attempt)
                    if doc is not None:
                        return doc
                    if self._failed_here(handle, job.attempt):
                        return None
            except FileNotFoundError:
                pass
        return None

    @staticmethod
    def _load_committed(
        handle: StorageDir, wd: Path, attempt: str | None
    ) -> dict | None:
        try:
            with handle.open_read(_COMMITTED) as f:
                doc = json.loads(read_capped(f, _COMMITTED, RECORD_CAP))
        except FileNotFoundError:
            return None
        except (ValueError, WrongKindError, CorruptEntry) as err:
            logger.warning("%s is unusable: %s", wd / _COMMITTED, err)
            return None
        entry = doc.get("entry") if isinstance(doc, dict) else None
        if (
            not isinstance(doc, dict)
            or doc.get("format") != _COMMITTED_FORMAT
            or doc.get("kind") not in ("entry", "kept", "outputs")
            or not isinstance(entry, dict)
            or not all(
                isinstance(entry.get(k), str) for k in ("root", "cdir", "fingerprint")
            )
            or not isinstance(doc.get("stored"), bool)
        ):
            logger.warning("%s is not a committed record", wd / _COMMITTED)
            return None
        if (
            doc.get("attempt") != attempt
        ):  # a stable directory holds an earlier run's record
            return None
        return doc

    @staticmethod
    def _committed_of(doc: dict, wd: Path) -> Committed:
        e = doc["entry"]
        kept = {"kept": _KEPT_ENTRY, "outputs": _OUTPUTS}.get(doc["kind"])
        return Committed(
            EntryRef(e["root"], e["cdir"], e["fingerprint"]),
            bool(doc.get("won")),
            str(doc.get("own_digest")),
            doc.get("committed_digest"),
            doc["stored"],
            None if kept is None or doc["stored"] else str(wd / kept),
            bool(doc.get("served", False)),
        )

    @staticmethod
    def _failed_here(handle: StorageDir, attempt: str | None) -> bool:
        """Whether ``_error.json`` records a failure of ``attempt`` (a stable directory holds
        the last failure of an earlier one); one that cannot be parsed is a failure."""
        try:
            with handle.open_read(_ERROR) as f:
                data = read_capped(f, _ERROR, None)
        except OSError:
            return False
        try:
            doc = json.loads(data)
        except ValueError:
            return True
        return not isinstance(doc, dict) or doc.get("attempt") == attempt

    def has_error(self, job: "Job") -> bool:
        """Whether ``_error.json`` in ``job``'s work directory records its current attempt"""
        return self._has_error(job.cache_dir, job.attempt)

    def _has_error(self, wd: Path, attempt: str | None) -> bool:
        try:
            with self._open_work(wd) as handle:
                return self._failed_here(handle, attempt)
        except FileNotFoundError:
            return False

    def _failed(self, job: "Job", wd: Path | None, why: str) -> Result:
        where = (
            f"its work directory is {wd}"
            if wd is not None
            else "it has no work directory"
        )
        return Result(
            cache_dir=wd if wd is not None else Path(self.path),
            outputs=None,
            runtime=None,
            errored=True,
            task=job.task,
            failure=f"{why}; {where}",
        )

    # -- reuse ----------------------------------------------------------------------------------

    def _roots(self) -> list[tuple[str, Trust, bool]]:
        """This store's root, then each read-only root: ``(path, trust, readonly)``."""
        return [
            (self.path, self.trust, False),
            *(
                (os.path.abspath(os.fspath(r.path)), r.trust, True)
                for r in self.readonly
            ),
        ]

    def stored(self, checksum: str) -> list[tuple[EntryRef, Context, str]]:
        """Every entry stored for ``checksum`` in every root, own root first and each root in
        path order, as ``(entry, context, sha256 of its context.json)``. Only ``context.json``
        is read; an entry whose context cannot be read here is left out (the read of a chosen
        entry reports it). A read-only root that cannot be read is logged and skipped.
        """
        cdir = cdir_for(checksum)
        shard = shard_for(cdir, checksum, self.header.shard)
        found: list[tuple[EntryRef, Context, str]] = []
        for path, trust, readonly in self._roots():
            if readonly and not self._check_readonly(path, trust):
                continue
            try:
                found.extend(self._stored_in(path, trust, cdir, shard))
            except FileNotFoundError:
                continue
            except OSError as err:
                if not readonly:
                    if isinstance(err, StoreUnavailable):
                        raise
                    raise StoreUnavailable(err.errno, f"{cdir}: {err}") from err
                logger.warning("the read-only root %s was skipped: %s", path, err)
        return found

    def _stored_in(
        self, path: str, trust: Trust, cdir: str, shard: str, only: str | None = None
    ) -> list[tuple[EntryRef, Context, str]]:
        found = []
        with StorageDir.open_root(path, self.policy, trust=trust) as root:
            with contextlib.ExitStack() as stack:
                parent = root
                for name in (self.header.entries, shard, cdir):
                    parent = stack.enter_context(parent.child(name))
                names = (
                    [only]
                    if only is not None
                    else sorted(
                        e.name
                        for e in parent.scandir()
                        if FINGERPRINT.fullmatch(e.name)
                    )
                )
                for name in names:
                    try:
                        with parent.child(name) as entry:
                            context, raw = self.format.context(entry)
                    except (
                        FileNotFoundError,
                        WrongKindError,
                        CorruptEntry,
                        EntryFormatUnknown,
                    ) as err:
                        logger.debug("%s/%s is not ranked: %s", cdir, name, err)
                        continue
                    except OSError as err:
                        if err.errno == _STALE:
                            continue  # gone, or not visible yet: unknown
                        raise
                    found.append(
                        (
                            EntryRef(path, cdir, name),
                            context,
                            hashlib.sha256(raw).hexdigest(),
                        )
                    )
        return found

    def stored_contexts(self, checksum: str) -> ty.Iterator[tuple[Path, Context]]:
        """The entry and recorded context of each result stored for ``checksum``, in every
        root, in the order the default hook considers them."""
        cdir = cdir_for(checksum)
        shard = shard_for(cdir, checksum, self.header.shard)
        for ref, context, _ in self.stored(checksum):
            yield (
                Path(ref.root) / self.header.entries / shard / cdir / ref.fingerprint,
                context,
            )

    def read_succeeded(
        self, checksum: str, context: Context, task: "Task | None"
    ) -> Result | None:
        """The result stored at the rank-0 address of ``checksum`` and ``context``, read without a
        lease (RFC 0003 section 5.2), at the cost of one ``lstat`` on a miss. A present entry goes
        through the reuse hook and is returned only for a ``SERVE`` verdict; ``VERIFY``, ``REFUSE``
        and an entry that does not verify return None and the job is dispatched (a runner that must
        carry the verify mark calls `find`). Only this store's own root and the job's own
        fingerprint are probed; other contexts and read-only roots are left to `find`.
        """
        if not self._present(checksum, context):
            return None
        own = EntryRef(
            self.path, cdir_for(checksum), self.identity.fingerprint(context)
        )
        try:
            found = self._find(checksum, context, task, own)
        except Exception as err:  # a broken hook or an unreadable entry: run it leased
            logger.warning("reading %s without its lease: %r", checksum, err)
            return None
        return found.result if isinstance(found, Hit) else None

    def expected_hits(self, jobs: ty.Sequence["Job"]) -> int:
        """How many of ``jobs`` a runner's ``ready()`` may settle from this store, for
        ``GraphRunner.plan``: those whose own address holds an entry, one ``lstat`` each. The
        reuse hook is not asked (its ``served`` records a reuse), so a ``VERIFY``, ``REFUSE``
        or corrupt entry counts, and an entry of another context or a read-only root, which
        ``find`` may serve, does not."""
        return sum(self._present(job.checksum, job.context) for job in jobs)

    def present(self, job: "Job") -> bool:
        """Whether this root holds an entry at the job's own address: one ``lstat``, nothing
        read (what a runner asks of every released job before it reads one off its
        thread)."""
        return self._present(job.checksum, job.context)

    def _present(self, checksum: str, context: Context) -> bool:
        """Whether this root holds an entry at the rank-0 address of ``checksum`` and
        ``context``: one ``lstat``, and nothing else."""
        cdir = cdir_for(checksum)
        shard = shard_for(cdir, checksum, self.header.shard)
        fingerprint = self.identity.fingerprint(context)
        try:
            os.lstat(
                os.path.join(self.path, self.header.entries, shard, cdir, fingerprint)
            )
        except OSError:  # absent, or not visible from here
            return False
        return True

    def begin_rerun(self, job: "Job") -> None:
        """Mark ``job`` as a rerun that replaces the entry at its own address (RFC 0003
        section 10), under its lease: ``job.replace`` is set, and ``job.replaces`` is the output
        digest of the entry there now, which the publish evicts only if it is still the one at
        the address (None if there is none or it cannot be read). The job does not verify
        against anything: the hook is not asked."""
        job.replace = True
        job.verify = None
        job.replaces = None
        cdir = cdir_for(job.checksum)
        fingerprint = self.identity.fingerprint(job.context)
        shard = shard_for(cdir, job.checksum, self.header.shard)
        try:
            found = self._stored_in(self.path, self.trust, cdir, shard, fingerprint)
            if not found:
                return
            ref, context, context_digest = found[0]
            read = self._serve(
                job.checksum,
                Candidate(ref, context, context_digest, 0, "exact"),
                job.task,
            )
        except (_Changed, FileNotFoundError, OSError) as err:
            logger.debug("the entry a rerun replaces cannot be read: %s", err)
            return
        if read is not None:
            job.replaces = read[0].digest

    def find(self, job: "Job", *, probe: bool = False) -> Lookup:
        """What ``job`` may reuse (RFC 0003 section 7.1): a `Hit` with the result to serve, a
        `Verify` whose candidate the job runs against (the caller sets ``job.verify``), or a
        `Miss`. The hook ranks the entries stored for the job's checksum in every root and is
        asked for the verdict of the first; `Verdict.SKIP` takes the next, `REFUSE` ends the
        lookup. A served entry is read and verified (section 6.2) through a descriptor whose
        ``context.json`` must be the bytes that were ranked, else the store ranks again; a
        corrupt candidate is quarantined (skipped, in a read-only root) and the next taken.
        If ``Reuse.served`` raises, the result is not served.

        After a `Miss` that no ``REFUSE`` ended, a store with ``legacy`` roots publishes the job's
        result from one of them (`~pydra.engine.portable.migrate.publish_legacy`) and ranks again,
        once per lookup.

        Parameters
        ----------
        probe : bool
            Look only if this root holds an entry at the job's own address, which costs one
            ``lstat`` (as `read_succeeded`), and otherwise answer `Miss` with nothing read:
            what a runner's ``ready()`` asks of every released job, leaving entries of other
            contexts, read-only roots and ``legacy`` roots to the lookup under the job's lease.
        """
        if probe and not self._present(job.checksum, job.context):
            return Miss()
        return self._find(job.checksum, job.context, job.task, job=job)

    def lookup(
        self,
        checksum: str,
        context: Context,
        task: "Task | None",
        stored: ty.Any = None,
    ) -> Result | None:
        """The `ResultStore` adapter of `find`: the `Hit`'s result, or None. ``stored`` (the
        contexts `check_declared_code` read) is not used, because it carries no ``context.json``
        digest, which is what binds a verdict to the bytes it reads: the entries are ranked
        afresh. No `Job` path calls this on a `PortableStore` (it reads through `find` and
        `read_result`)."""
        found = self._find(checksum, context, task)
        return found.result if isinstance(found, Hit) else None

    def _find(
        self,
        checksum: str,
        context: Context,
        task: "Task | None",
        own: EntryRef | None = None,
        job: "Job | None" = None,
    ) -> Lookup:
        legacy = job is not None and bool(self.legacy)
        for _ in range(_RERANKS):
            ranked = self.reuse.rank(context, self.stored(checksum), task)
            outcome = self._consider(checksum, ranked, context, task, own)
            if outcome is _CONTENDED:
                continue
            if outcome is _REFUSED:
                return Miss()  # the hook ended the lookup: legacy is not read either
            if isinstance(outcome, Miss) and legacy:
                from pydra.engine.portable.migrate import LEGACY_REFUSED, publish_legacy

                legacy = False  # read the old caches once per lookup
                published = publish_legacy(self, ty.cast("Job", job))
                if published is LEGACY_REFUSED:
                    return Miss()
                if published:
                    continue
            return ty.cast(Lookup, outcome)
        logger.warning(
            "the entries of %s changed while they were read %d times; not reused",
            checksum,
            _RERANKS,
        )
        return Miss()

    def _kind_of(self, root: str) -> tuple[Trust, bool]:
        for path, trust, readonly in self._roots():
            if path == os.path.abspath(root):
                return trust, readonly
        raise ValueError(
            f"the entry is in the root {root!r}, which this store does not hold"
        )

    def _consider(
        self,
        checksum: str,
        ranked: ty.Sequence[Candidate],
        context: Context,
        task: "Task | None",
        own: EntryRef | None = None,
    ) -> "Lookup | object":
        mine = self.identity.fingerprint(context)
        for candidate in ranked:
            if own is not None and not _same_entry(candidate.entry, own):
                return Miss()  # only the own address is wanted, and it was not taken
            verdict = self.reuse.verdict(candidate, context, task)
            if verdict is Verdict.REFUSE:
                return _REFUSED
            if verdict is Verdict.SKIP:
                continue
            if verdict is Verdict.VERIFY:
                return Verify(candidate)
            if verdict is not Verdict.SERVE:
                raise ValueError(f"a reuse verdict is a Verdict, not {verdict!r}")
            try:
                read = self._serve(checksum, candidate, task)
            except (_Changed, FileNotFoundError):
                return _CONTENDED  # the entry changed after it was ranked: rank again
            if read is None:
                continue
            stored, where = read
            result = self._result_at(stored, where, task)
            result.committed = Committed(
                candidate.entry, False, stored.digest, stored.digest, True, None, True
            )
            if candidate.entry.fingerprint != mine:
                result.cross_context = True
                result.reused_in = dict(context)
            try:
                self.reuse.served(candidate, context, task, stored)
            except Exception as err:
                logger.warning(
                    "%s is not served: its reuse record failed: %r",
                    candidate.entry.fingerprint,
                    err,
                )
                return Miss()
            return Hit(result)
        return Miss()

    def _serve(
        self, checksum: str, candidate: Candidate, task: "Task | None"
    ) -> tuple[StoredEntry, Path] | None:
        """Read the candidate, or None if it cannot be served: corrupt (quarantined unless its
        root is read-only), of another format, or not readable with its root's trust. Raises
        `_Changed` when it is no longer the entry that was ranked."""
        ref = candidate.entry
        trust, readonly = self._kind_of(ref.root)
        shard = shard_for(ref.cdir, checksum, self.header.shard)
        try:
            with StorageDir.open_root(ref.root, self.policy, trust=trust) as root:
                return self._read_entry(
                    root,
                    ref,
                    task,
                    shard,
                    trust=trust,
                    readonly=readonly,
                    digest=candidate.context_digest,
                )
        except CorruptEntry as err:
            where = "skipped" if readonly else "moved aside"
            logger.warning(
                "%s in %s is corrupt and was %s: %s",
                ref.fingerprint,
                ref.root,
                where,
                err,
            )
        except (EntryFormatUnknown, EntryNotTrusted) as err:
            logger.warning("%s in %s is not served: %s", ref.fingerprint, ref.root, err)
        except FileNotFoundError:
            raise
        except OSError as err:
            if err.errno == _STALE:
                raise FileNotFoundError(err.errno, str(err)) from err
            if not readonly:
                if isinstance(err, StoreUnavailable):
                    raise
                raise StoreUnavailable(err.errno, f"{ref.cdir}: {err}") from err
            logger.warning("%s in %s was skipped: %s", ref.fingerprint, ref.root, err)
        return None

    def sidecar(self, name: str) -> StorageDir:
        """The directory of the store-level records ``name`` (RFC 0003 section 7.4):
        ``<root>/sidecars/<name>@1/``, or the directory the header's ``sidecars`` mapping gives
        for ``name``, made if it is missing. The caller closes it. `collect` never touches it.

        Raises
        ------
        StoreNotWritable
            This user cannot make it.
        """
        header = self.header.dirs.get("sidecars") or {}
        relative = header.get(name, f"sidecars/{name}@1")
        parts = [p for p in relative.split("/") if p]
        if not parts or any(p in (".", "..") for p in parts):
            raise ValueError(
                f"the sidecar directory {relative!r} is not below the root"
            )
        handles = [self._open()]
        try:
            for part in parts:
                with self._denied(os.path.join(handles[-1].path, part)):
                    parent = handles[-1]
                    made = not self._exists_in(parent, part)
                    handles.append(
                        parent.child(part, create=True, mode=self.policy.side_mode)
                    )
                    if made:
                        parent.sync()  # the new name is durable before a record is
        except BaseException:
            for h in handles:
                h.close()
            raise
        for h in handles[:-1]:
            h.close()
        return handles[-1]

    @staticmethod
    def _exists_in(parent: StorageDir, name: str) -> bool:
        try:
            parent.lstat(name)
        except FileNotFoundError:
            return False
        return True

    def write_sidecar(self, name: str, member: str, data: bytes) -> None:
        """Write ``data`` as ``member`` of the sidecar ``name`` atomically and durably: to a
        temporary name, flushed, renamed over ``member``, and the directory flushed, so a
        reader sees no file or a whole one and the record survives a crash. A hook's reuse
        record (``Reuse.served``) or comparison record is written this way.

        Raises
        ------
        ValueError
            ``member`` is not a plain file name.
        StoreNotWritable
            This user cannot write the sidecar.
        """
        if not member or member in (".", "..") or "/" in member or "\\" in member:
            raise ValueError(f"a sidecar member is a plain file name, not {member!r}")
        with self.sidecar(name) as side:
            with self._denied(os.path.join(side.path, member)):
                self._write_file(side, member, data, self.policy.file_mode)
                side.sync()

    # -- de-duplication leases ---------------------------------------------------------------

    def acquire(
        self, checksum: str, context: Context, *, resumable: bool = False
    ) -> "Lease":
        """The lease guarding the job of ``checksum`` under ``context`` (RFC 0003 section 5.3):
        a `DirectoryLease` at ``<root>/leases/<sha256 of cdir/fingerprint>.lock`` that records
        no user, host or pid, when the store de-duplicates or the job is ``resumable``;
        otherwise a lease that is always held."""
        if not (self.dedupe or resumable):
            return _AlwaysHeld(checksum)
        name = hashlib.sha256(
            f"{cdir_for(checksum)}/{self.identity.fingerprint(context)}".encode()
        ).hexdigest()
        with self._denied(self.path):
            with self._open() as root:
                root.child(_LEASES, create=True, mode=self.policy.side_mode).close()
        leases = Path(self.path) / _LEASES
        return DirectoryLease(
            checksum, leases, leases / f"{name}.lock", settings=self.lease_settings
        )

    # -- the ResultStore adapter ---------------------------------------------------------------------

    def commit(self, lease: "Lease", result: Result, job: "Job | None" = None) -> None:
        """Publish ``result`` as the outcome of ``job`` and record the `Committed` on it.

        An adapter over `publish`: the outputs are ``result.outputs``, the key and the execute
        context are the identity's for ``job`` (the result's context without a job), and the
        job's work directory is where files come from. An errored result publishes nothing,
        and the lease, held or lost, is not consulted. With a job, the outcome is also written
        to ``<work dir>/_committed.json``; a result that was not stored is kept there (RFC 0003
        section 5.4), and a stored one's default work directory is emptied. ``job`` is never
        written to an entry.
        """
        if result.errored or result.outputs is None:
            if result.errored and job is not None:
                self._keep_job(job)
            return
        wd = None
        if job is not None:
            checksum = job.checksum
            key = self.identity.key(job)
            context = self._with_pool_facts(job, self.identity.context(job, "execute"))
            wd = self.work_dir(job)
        else:
            checksum = lease.checksum
            key = None
            context = result.context
        outputs = result.outputs
        if not isinstance(outputs, ty.Mapping):
            from pydra.utils.general import get_fields

            outputs = {
                f.name: getattr(outputs, f.name) for f in get_fields(type(outputs))
            }
        files_from = wd if wd is not None else lease.cache_dir
        kind = "entry"
        replace = job is not None and job.replace
        try:
            committed = self.publish(
                checksum,
                key,
                context,
                outputs,
                result.task,
                files_from,
                verify=job.verify if job is not None else None,
                replace=replace,
                expected=job.replaces if job is not None else None,
            )
        except UnencodableOutput:
            if job is None:
                raise
            ref = EntryRef(
                self.path, cdir_for(checksum), self.identity.fingerprint(context)
            )
            committed = self._keep_outputs(job, ty.cast(Path, wd), outputs, ref)
            kind = "outputs"
        else:
            taken = (
                self._serve_occupant(checksum, committed, context, result.task)
                if not replace
                and not committed.stored
                and committed.committed_digest is not None
                else None
            )
            if taken is not None:
                committed, served = taken
                result.outputs = served.outputs
            elif job is not None and not committed.stored:
                committed = self._keep_entry(ty.cast(Path, wd), committed)
                kind = "kept"
        if job is not None:
            self._write_committed(job, ty.cast(Path, wd), committed, kind)
            if committed.stored and self._work_fn is None:
                self._empty_work_dir(ty.cast(Path, wd))
        result.committed = committed
        result.stored = committed.stored

    def _with_pool_facts(self, job: "Job", context: Context) -> Context:
        """``context`` with the ``pool`` section a pool worker computed for the call (RFC 0005
        section 8), when the identity lists ``pool`` in ``recorded_only``, so it is recorded and
        splits no slot; otherwise ``context`` unchanged."""
        facts = job.pool_facts
        if facts is None or "pool" not in self.identity.recorded_only:
            return context
        return {**context, "pool": facts}

    def _serve_occupant(
        self,
        checksum: str,
        committed: Committed,
        context: Context,
        task: "Task | None",
    ) -> tuple[Committed, Result] | None:
        """A lost race with unequal digests (RFC 0003 section 7.3): if the hook gives ``SERVE``
        for the committed entry as a rank-0 candidate, the job's result is that entry's outputs
        and ``served`` is called (a reuse); the staged entry is removed. None otherwise, or if
        the occupant cannot be read or its record cannot be written: the job keeps its own
        outputs, uncommitted."""
        ref = committed.entry
        cdir = ref.cdir
        shard = shard_for(cdir, checksum, self.header.shard)
        try:
            found = self._stored_in(self.path, self.trust, cdir, shard, ref.fingerprint)
        except OSError:
            return None
        if not found:
            return None
        _, stored_context, context_digest = found[0]
        candidate = Candidate(ref, stored_context, context_digest, 0, "exact")
        if self.reuse.verdict(candidate, context, task) is not Verdict.SERVE:
            return None
        try:
            read = self._serve(checksum, candidate, task)
        except (_Changed, FileNotFoundError):
            return None
        if read is None:
            return None
        stored, where = read
        result = self._result_at(stored, where, task)
        try:
            self.reuse.served(candidate, context, task, stored)
        except Exception as err:
            logger.warning(
                "%s is not served: its reuse record failed: %r", ref.fingerprint, err
            )
            return None
        self.discard_kept(committed)
        return attrs.evolve(committed, stored=True, kept=None, served=True), result

    def discard_kept(self, committed: Committed) -> None:
        """Remove the staged entry a lost race kept (``committed.kept``), when it is in this
        store's staging directory; nothing else is touched. For a caller, such as
        `~pydra.engine.portable.migrate.migrate`, that has no use for its own copy."""
        if committed.kept is None:
            return
        with self._open() as root, root.child(self.header.dirs["staging"]) as st:
            if os.path.dirname(committed.kept) == st.path:
                _discard(st, os.path.basename(committed.kept))

    def _keep_job(self, job: "Job") -> None:
        """Write the failed job's pickle to its work directory, for ``pydracli crash``."""
        try:
            path = self.work_dir(job)
            self._make_work_dir(path)
            save(path, job=job)
        except Exception as err:
            logger.warning("could not save the failed job %s: %s", job.name, err)


def _same_entry(a: EntryRef, b: EntryRef) -> bool:
    return (
        os.path.abspath(a.root) == os.path.abspath(b.root)
        and a.cdir == b.cdir
        and a.fingerprint == b.fingerprint
    )


class _AlwaysHeld:
    """The lease of a store that does not de-duplicate: held by every taker, so concurrent
    submitters of one job both compute it and one wins the rename (RFC 0003 section 5.3).
    """

    lost = None
    held = True
    cache_dir = None
    # it excludes no other taker, so a lookup made before taking it is still current
    exclusive = False

    def __init__(self, checksum: str) -> None:
        self.checksum = checksum

    def prove(self) -> bool:
        return True

    def try_take(self) -> None:
        """Held at once: a runner's one attempt at the lease of a workflow job succeeds."""
        return None

    def __enter__(self) -> "_AlwaysHeld":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    async def __aenter__(self) -> "_AlwaysHeld":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __repr__(self) -> str:
        return f"_AlwaysHeld({self.checksum!r})"
