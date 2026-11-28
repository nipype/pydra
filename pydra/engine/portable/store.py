"""`PortableStore`: publish and read (RFC 0003 sections 5.1, 5.2, 6.2 and 7.2).

A writer encodes an entry into ``staging/<uuid>/`` and renames the directory to its address,
``<entries>/<shard>/<cdir>/<fingerprint>``. The rename is the first-writer-wins primitive: an
entry exists only once it is complete. Every step below ``<root>`` goes through a
`~pydra.utils.storagedir.StorageDir`.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import stat
import typing as ty
import uuid
from pathlib import Path

import attrs

from pydra.engine.portable.entry import (
    CorruptEntry,
    EntryFormat,
    EntryFormatUnknown,
    EntryNotTrusted,
    EntryRef,
    PydraEntryV1,
    StoredEntry,
)
from pydra.engine.portable.identity import (
    Context,
    KeyRecord,
    PydraIdentityV1,
    StoreIdentity,
)
from pydra.engine.lease import LeaseSettings
from pydra.engine.portable.layout import (
    HASHED_CDIR,
    LayoutHeader,
    cdir_for,
    open_layout,
    shard_for,
)
from pydra.engine.portable.trust import TRUSTS, Root, Trust
from pydra.engine.result import Result
from pydra.utils.retry import Backoff, Tries
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
    is_reparse,
)

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task
    from pydra.engine.job import Job
    from pydra.engine.lease import Lease

__all__ = ["Committed", "PortableStore", "StoreNotWritable"]

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
_OURS = object()  # the occupant carries this writer's token


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
    """

    entry: EntryRef
    won: bool
    own_digest: str
    committed_digest: str | None
    stored: bool
    kept: str | None


class StoreNotWritable(StoreUnavailable):
    """This user cannot write a directory that a publish needs (RFC 0003 section 5.1). It is a
    configuration error, not a lost race."""


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
        for e in list(d.scandir()):
            if e.is_dir():
                _remove_tree(d, e.name)
            else:
                d.unlink(e.name)
    parent.rmdir(name)


def _discard(parent: StorageDir, name: str) -> None:
    with contextlib.suppress(OSError):
        _remove_tree(parent, name)


class _Replaced(Exception):
    """A directory this writer holds was replaced by another writer's and is no longer in the
    tree; the walk from the root starts again."""


class _Contended(Exception):
    """The visibility timeout passed while other writers kept changing the directories."""


CONTENTION = Backoff(first=0.004, cap=0.25, jitter=True)
"""How a writer pauses after another writer replaced a container or the occupant changed,
until the visibility timeout: jittered, so contending writers spread out."""


def _pause(tries: Tries) -> None:
    if not tries.wait():
        raise _Contended


def _rebuild(args: dict[str, ty.Any]) -> "PortableStore":
    return PortableStore(**args)


class PortableStore:
    """A result store whose entries any process, host or user can publish and read.

    Parameters
    ----------
    root : os.PathLike
        The store's root. A new root, or an empty one, is given a layout header.
    trust : {"shared", "private", "trusted"}
        What this process asserts about who can write the root (RFC 0003 section 6.4).
    identity : StoreIdentity
        The key, context and fingerprint of an entry.
    entry_format : EntryFormat
        What is inside an entry, bound to this store's identity, trust and the header's
        ``pickle`` flag (`EntryFormat.bind`).
    policy : StorePolicy
        Modes, durability and ages; recorded in the header, so two stores of one root must agree.
    readonly, legacy, reuse, on_commit
        Accepted and not yet used.
    dedupe, work_dir
        Accepted and not yet used (work directories and de-duplication leases).
    pickle : bool
        Whether writers to a new root may write pickle members; recorded when the root is
        created and read from the header after.
    """

    def __init__(
        self,
        root: os.PathLike,
        *,
        trust: Trust = "shared",
        identity: StoreIdentity = PydraIdentityV1(),
        entry_format: EntryFormat = PydraEntryV1(),
        policy: StorePolicy = StorePolicy(),
        readonly: ty.Sequence[Root] = (),
        legacy: ty.Sequence[Root] = (),
        dedupe: bool | None = None,
        work_dir: ty.Callable[[str, str, str], Path] | None = None,
        pickle: bool = False,
        reuse: ty.Any = "consistent",
        on_commit: ty.Callable | None = None,
    ) -> None:
        if trust not in TRUSTS:
            raise ValueError(f"trust must be one of {TRUSTS}, got {trust!r}")
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
        )
        self.path = os.fspath(root)
        self.trust = trust
        self.identity = identity
        self.policy = policy
        self.readonly = tuple(readonly)
        self.legacy = tuple(legacy)
        self.dedupe = dedupe
        self.work_dir = work_dir
        self.reuse = reuse
        self.on_commit = on_commit
        self.visibility_timeout = LeaseSettings().visibility_timeout
        self._make_root()
        expected = LayoutHeader.for_store(
            identity,
            policy,
            pickle=pickle,
            entries=entry_format.entries,
            entry_format=entry_format.name,
        )
        header_present = os.path.lexists(os.path.join(self.path, "pydra-store.json"))
        with self._open() as handle:
            self.header = open_layout(handle, expected, create=True)
            self.pickle = self.header.pickle
            self.format = entry_format.bind(identity, self.pickle, trust)
            made = not header_present
            for name in (self.header.dirs["staging"], self.header.dirs["quarantine"]):
                made |= self._side_dir(handle, name)
            made |= self._entries_dir(handle)
            if made:
                handle.sync()

    def __reduce__(self) -> tuple:
        return (_rebuild, (self._args,))

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
                    parent.rename(name, q, _inode_name(st))
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
    ) -> bool | None:
        """Move the entry ``name`` of ``parent`` aside as a compare-and-move (RFC 0003
        section 6.3) and check what was moved: a copy that verifies is put back, a corrupt
        one is stamped for ``collect``. ``identity`` is the ``fstat`` of the entry found
        corrupt; an entry with another inode is left alone.

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
                return self._after_move(root, parent, name, cdir, moved, holder)
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
    ) -> bool:
        where = os.path.join(parent.path, name)
        with holder.child(moved) as copy:
            try:
                checksum = self._checksum_of(copy, cdir)
                self.format.read(copy, checksum, name, None, self.trust, root=root)
            except CorruptEntry as err:
                stamp = copy._fd if copy._fd is not None else copy.path
                with contextlib.suppress(OSError):
                    os.utime(stamp, None)
                logger.warning("%s was moved to quarantine: %s", where, err)
                return True
            except Exception:  # verifies or cannot be judged: it is not shown corrupt
                pass
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
    ) -> Committed:
        """Publish ``outputs`` at the address of ``checksum`` and ``context`` (RFC 0003 section
        5.1).

        The entry is encoded into a fresh ``staging/<uuid>/`` and renamed to its address. If
        another writer was first, its entry is read and verified: with equal digests ours is
        stored and the staging directory removed, otherwise ours stays unpublished in the staging
        directory that ``Committed.kept`` names. A corrupt or misfiled occupant is quarantined and
        the commit retried. The lease is not consulted, and this never raises ``LeaseLost``.

        ``verify``, ``replace`` and the `Result` of a lost race are not acted on here.

        Raises
        ------
        StoreNotWritable
            This user cannot write a directory the publish needs.
        UnencodableOutput
            An output has no codec and this writer may not pickle it.
        StoreUnavailable
            Any other ``OSError``.
        """
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
                        outputs, task, source, writer,
                    )  # fmt: skip
            finally:
                if source is not None:
                    source.close()

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
    ) -> Committed:
        policy = self.policy
        with self._denied(staging.path):
            stage = staging.child(name, create=True, mode=policy.staging_mode)
        try:
            try:
                digest = self.format.stage(
                    stage, checksum, key, context, outputs, task, source, writer
                )
                stage.set_mode(policy.entry_mode)
                if policy.durability == "member":
                    stage.sync()
                elif policy.durability == "entry" and hasattr(os, "sync"):
                    os.sync()
            finally:
                stage.close()
            committed = self._place(
                root, staging, name, ref, shard, checksum, task, writer, digest
            )
        except BaseException as err:
            _discard(staging, name)
            if isinstance(err, OSError) and not isinstance(err, StoreUnavailable):
                raise StoreUnavailable(err.errno, f"{staging.path}: {err}") from err
            raise
        if committed.stored:
            _discard(staging, name)  # renamed away, or equal to the occupant
        return committed

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

    def _read(self, root: StorageDir, ref: EntryRef, task: "Task | None") -> Result:
        with root.child(self.header.entries) as entries:
            shard = self._shard_of(entries, ref.cdir)
            with (
                entries.child(shard) as shard_dir,
                shard_dir.child(ref.cdir) as cdir_dir,
            ):
                try:
                    entry = cdir_dir.child(ref.fingerprint)
                except WrongKindError as err:
                    self._set_aside(root, cdir_dir, ref.fingerprint)
                    raise CorruptEntry(str(err)) from err
                with entry:
                    identity = entry.fstat()
                    try:
                        checksum = self._checksum_of(entry, ref.cdir)
                        stored = self.format.read(
                            entry,
                            checksum,
                            ref.fingerprint,
                            task,
                            self.trust,
                            root=root,
                        )
                    except CorruptEntry:
                        self._quarantine(
                            root, cdir_dir, ref.fingerprint, ref.cdir, identity
                        )
                        raise
                    self._reseal(entry)
                    return self._result(stored, entry, task)

    def _result(
        self, stored: StoredEntry, entry: StorageDir, task: "Task | None"
    ) -> Result:
        outputs: ty.Any = dict(stored.outputs)
        if task is not None:
            from pydra.utils.general import get_fields

            cls = task.Outputs
            declared = {f.name for f in get_fields(cls)}
            if set(outputs) != declared:
                raise StoreUnavailable(
                    f"the entry holds outputs {sorted(outputs)} and the task declares "
                    f"{sorted(declared)}"
                )
            outputs = cls(**outputs)
            outputs._cache_dir = Path(entry.path)
        return Result(
            cache_dir=Path(entry.path),
            outputs=outputs,
            runtime=None,
            errored=False,
            task=task,
            context=dict(stored.context),
            stored=True,
        )

    # -- the ResultStore adapter ---------------------------------------------------------------------

    def commit(self, lease: "Lease", result: Result, job: "Job | None" = None) -> None:
        """Publish ``result`` as the outcome of ``job`` and record the `Committed` on it.

        An adapter over `publish`: the outputs are ``result.outputs``, the key and the execute
        context are the identity's for ``job`` (the result's context without a job), and the
        lease's directory is where files come from. An errored result publishes nothing, and
        the lease, held or lost, is not consulted. ``job`` is never written to an entry.
        """
        if result.errored or result.outputs is None:
            return
        if job is not None:
            checksum = job.checksum
            key = self.identity.key(job)
            context = self.identity.context(job, "execute")
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
        files_from = lease.cache_dir
        committed = self.publish(
            checksum, key, context, outputs, result.task, files_from
        )
        result.committed = committed
        result.stored = committed.stored
