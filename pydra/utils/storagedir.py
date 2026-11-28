"""Descriptor-relative directory handles for a portable result store.

A `StorageDir` is an open directory. Every step below it (opening a child, creating,
renaming, linking, listing, changing a mode) is made relative to its descriptor, with
``O_NOFOLLOW``, so a symlink swapped in for a directory or a file while a store is at work
is refused instead of followed. Modes are set through descriptors and never come from
the umask. A `StorePolicy` carries the modes, the durability and the ages a store applies.

Where the platform has no descriptor-relative calls (Windows) a handle works from paths and
checks, before each step, that no component below the root is a symlink or a reparse
point. That check is not race-free, so a ``shared`` root is refused there with
`PlatformUnsupported`.
"""

import errno
import hashlib
import math
import os
import stat
import sys
import typing as ty
from pathlib import Path

import attrs

from pydra.utils.storage import PlatformUnsupported

__all__ = [
    "DESCRIPTORS_SUPPORTED",
    "StorageDir",
    "StorageEntry",
    "StoreAges",
    "StorePolicy",
    "StoreUnavailable",
    "WrongKindError",
    "copy_hashed",
]

IS_WINDOWS = sys.platform == "win32"
_DARWIN = sys.platform == "darwin"

DESCRIPTORS_SUPPORTED = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and {os.open, os.mkdir, os.rename, os.link, os.stat, os.unlink, os.rmdir}
    <= os.supports_dir_fd
    and os.scandir in os.supports_fd
)
"""Whether the descriptor-relative primitives of RFC 0003 section 9 exist here."""

Trust = ty.Literal["shared", "private", "trusted"]

_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_COPY_BLOCK = 1024 * 1024
_BINARY = getattr(os, "O_BINARY", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_WRONG_KIND_ERRNOS = frozenset(
    e
    for e in (
        errno.ELOOP,
        errno.ENOTDIR,
        getattr(errno, "EFTYPE", None),
        getattr(errno, "EMLINK", None),
        getattr(errno, "EOPNOTSUPP", None),  # a socket, on macOS
        getattr(errno, "ENXIO", None),  # a socket, on Linux
    )
    if e
)


class StoreUnavailable(OSError):
    """The store cannot be used here: a directory or member it needs is unreadable,
    unwritable or of the wrong kind."""


class WrongKindError(StoreUnavailable):
    """A symlink, or an object of another kind, was found where the store expects a
    directory or a regular file. The message names the path and its owner."""


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


def _wrong_kind(path: str, expected: str) -> WrongKindError:
    return WrongKindError(
        errno.ENOTDIR if expected == "directory" else errno.EINVAL,
        f"{path!r} is not a {expected} (a symlink or an object of another kind); owner: {_owner(path)}",
        path,
    )


def _check_mode(name: str, mode: int) -> None:
    if (
        not isinstance(mode, int)
        or isinstance(mode, bool)
        or mode < 0
        or mode != mode & 0o7777
    ):
        raise ValueError(
            f"{name} must be a permission mode between 0 and 0o7777, got {mode!r}"
        )


@attrs.frozen
class StoreAges:
    """How old, in seconds, a store's leftovers must be before `collect` removes them.

    Attributes
    ----------
    staging : float
        A writer's ``staging/`` directory, and a header temporary. A live writer touches
        its staging directory (from a thread) at least once per quarter of this.
    work : float
        A default work directory. It must exceed the longest run that reads a kept result.
    quarantine : float
        A quarantined object, aged from its move.
    """

    staging: float = 24 * 3600.0
    work: float = 7 * 24 * 3600.0
    quarantine: float = 7 * 24 * 3600.0

    def __attrs_post_init__(self) -> None:
        for field in attrs.fields(type(self)):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"age {field.name} must be a number of seconds, got {value!r}"
                )
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"age {field.name} must be a positive, finite number of seconds, got {value!r}"
                )


@attrs.frozen
class StorePolicy:
    """The modes, durability and ages a store applies (RFC 0003 section 9).

    Modes are set through descriptors, never from the umask. ``file_mode``, ``entry_mode`` and
    ``sealed_mode`` must not include group or other write; ``entry_mode`` must give the owner
    read, write and search. ``container_mode``
    and ``side_mode`` may be group-writable, and must be for a root that several users write.

    Attributes
    ----------
    file_mode, entry_mode, sealed_mode, container_mode, side_mode, staging_mode : int
        The mode of a committed member, of an entry directory before its rename, of an entry
        after it (None: not narrowed), of ``<entries>``, ``<shard>`` and ``<cdir>``, of the
        side directories, and of a writer's own staging directory while it writes.
    durability : {"member", "entry", "none"}
        ``"member"``: every member, the directory and its parent are ``fsync``ed;
        ``"entry"``: one flush before the rename.
    verify_files : {"digest", "size"}
        How a read checks a file member. ``"size"`` is accepted only with
        ``durability="member"`` and, enforced by `StorageDir.open_root`, for a ``private`` root.
    ages : StoreAges
    """

    file_mode: int = 0o444
    entry_mode: int = 0o755
    sealed_mode: int | None = None
    container_mode: int = 0o755
    side_mode: int = 0o755
    staging_mode: int = 0o700
    durability: ty.Literal["member", "entry", "none"] = "member"
    verify_files: ty.Literal["digest", "size"] = "digest"
    ages: StoreAges = StoreAges()

    def __attrs_post_init__(self) -> None:
        for name in (
            "file_mode",
            "entry_mode",
            "container_mode",
            "side_mode",
            "staging_mode",
        ):
            _check_mode(name, getattr(self, name))
        if self.sealed_mode is not None:
            _check_mode("sealed_mode", self.sealed_mode)
        for name in ("file_mode", "entry_mode", "sealed_mode"):
            mode = getattr(self, name)
            if mode is not None and mode & 0o022:
                raise ValueError(
                    f"{name}={mode:#o} includes group or other write, which would let someone "
                    "but the owner change a committed entry"
                )
        if self.file_mode & 0o400 != 0o400:
            raise ValueError(f"file_mode={self.file_mode:#o} must include owner read")
        if self.entry_mode & 0o700 != 0o700:
            raise ValueError(
                f"entry_mode={self.entry_mode:#o} must give the owner read, write and search"
            )
        if self.sealed_mode is not None and self.sealed_mode & 0o500 != 0o500:
            raise ValueError(
                f"sealed_mode={self.sealed_mode:#o} must include owner read and search"
            )
        for name in ("container_mode", "side_mode"):
            if getattr(self, name) & 0o700 != 0o700:
                raise ValueError(
                    f"{name}={getattr(self, name):#o} must give the owner read, write and search"
                )
        if self.staging_mode & 0o700 != 0o700:
            raise ValueError(
                f"staging_mode={self.staging_mode:#o} must give the owner read, write and search"
            )
        if self.durability not in ("member", "entry", "none"):
            raise ValueError(
                f"durability must be 'member', 'entry' or 'none', got {self.durability!r}"
            )
        if self.verify_files not in ("digest", "size"):
            raise ValueError(
                f"verify_files must be 'digest' or 'size', got {self.verify_files!r}"
            )
        if self.verify_files == "size" and self.durability != "member":
            raise ValueError(
                "verify_files='size' is accepted only with durability='member'"
            )
        if not isinstance(self.ages, StoreAges):
            raise ValueError("ages must be a StoreAges")


_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{d}{n}" for d in ("COM", "LPT") for n in "123456789\u00b9\u00b2\u00b3"
}
_RESERVED_CHARS = frozenset('<>:"|?*')


def _windows_reserved(name: str) -> bool:
    """Whether Windows would not treat ``name`` as an ordinary file name: a device name (with or
    without an extension), a reserved character or control character, or a trailing space or dot.
    Written out, after ``ntpath.isreserved``, so that it does not need Python 3.13."""
    if name[-1] in ". " or any(c in _RESERVED_CHARS or ord(c) < 32 for c in name):
        return True
    return name.partition(".")[0].rstrip(" ").upper() in _DEVICE_NAMES


def _check_name(name: str, strict: bool = False) -> None:
    """``name`` is one path component. ``strict`` (the path backend) also refuses what Windows
    reads as a drive, an alternate data stream, a device, or a name it silently trims.
    """
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or "\0" in name
        or (os.altsep and os.altsep in name)
        or (strict and _windows_reserved(name))
    ):
        raise ValueError(f"{name!r} is not a single path component")


def is_reparse(st: os.stat_result) -> bool:
    """Whether ``st`` is of a Windows reparse point (a symlink or a junction)."""
    return bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


@attrs.frozen
class StorageEntry:
    """A name in a listing with the ``lstat`` taken when it was listed. Nothing follows a symlink."""

    name: str
    path: str
    _lstat: os.stat_result

    def stat(self) -> os.stat_result:
        """The ``lstat`` of the entry (the link itself, for a symlink)."""
        return self._lstat

    def is_dir(self) -> bool:
        return stat.S_ISDIR(self._lstat.st_mode)

    def is_file(self) -> bool:
        return stat.S_ISREG(self._lstat.st_mode)

    def is_symlink(self) -> bool:
        return stat.S_ISLNK(self._lstat.st_mode)


class StorageDir:
    """An open directory under a store root. Make one with `open_root` and `child`.

    A handle is a descriptor (POSIX) or a path with checks (the path backend, chosen once when the
    root is opened). After `close` every operation raises `ValueError`.
    """

    def __init__(
        self,
        path: str,
        fd: int | None,
        policy: StorePolicy,
        trust: Trust,
        chain: tuple[str, ...] = (),
        is_root: bool = True,
    ) -> None:
        self.path = path
        self.policy = policy
        self.trust = trust
        self._fd = fd
        self._path_mode = fd is None
        self._closed = False
        self._is_root = is_root
        self._chain = (
            chain  # path backend: the components below the root above this one
        )

    # -- opening -----------------------------------------------------------------------

    @classmethod
    def open_root(
        cls, path: os.PathLike, policy: StorePolicy, *, trust: Trust = "shared"
    ) -> "StorageDir":
        """Open ``path`` as given (a symlink at the root itself is followed).

        Parameters
        ----------
        path : os.PathLike
            The root directory; it must exist.
        policy : StorePolicy
        trust : {"shared", "private", "trusted"}
            Who can write the root (RFC 0003 section 6.4). ``"private"`` is checked here: the root's
            descriptor must be owned by the effective user with no group or other write; on Windows
            the root must be under ``%LOCALAPPDATA%``. ``"shared"`` raises `PlatformUnsupported`
            where there are no descriptor-relative calls (Windows).

        Raises
        ------
        PlatformUnsupported
            ``trust="shared"`` on a platform without descriptor-relative calls.
        StoreUnavailable
            ``trust="private"`` and the root fails the check.
        ValueError
            ``verify_files="size"`` without ``trust="private"``.
        """
        if trust not in ("shared", "private", "trusted"):
            raise ValueError(
                f"trust must be 'shared', 'private' or 'trusted', got {trust!r}"
            )
        if policy.verify_files == "size" and trust != "private":
            raise ValueError("verify_files='size' is accepted only for a private root")
        path = os.fspath(path)
        if trust == "shared" and (IS_WINDOWS or not DESCRIPTORS_SUPPORTED):
            raise PlatformUnsupported(
                errno.ENOTSUP,
                "a shared root needs descriptor-relative file operations and no flag for refusing "
                f"symlinks, which this platform ({sys.platform}) lacks; declare the root private or "
                "trusted",
                path,
            )
        if not DESCRIPTORS_SUPPORTED:
            if not os.path.isdir(path):
                raise (
                    FileNotFoundError
                    if not os.path.lexists(path)
                    else NotADirectoryError
                )(errno.ENOENT, "not a directory", path)
            root = cls(os.path.abspath(path), None, policy, trust)
            if trust == "private":
                root._check_private()
            return root
        fd = os.open(path, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
        root = cls(path, fd, policy, trust)
        if trust == "private":
            try:
                root._check_private()
            except BaseException:
                root.close()
                raise
        return root

    def _check_private(self) -> None:
        if IS_WINDOWS:
            local = os.environ.get("LOCALAPPDATA")
            if not local or not self._is_under(local):
                raise StoreUnavailable(
                    errno.EACCES,
                    "a private root on Windows must be under the user's local application data "
                    "directory; declare it trusted or shared",
                    self.path,
                )
            return
        st = self.fstat()
        if st.st_uid != os.geteuid() or st.st_mode & 0o022:
            raise StoreUnavailable(
                errno.EACCES,
                f"not a private root: its owner is {_owner(self.path)} and its mode is "
                f"{stat.S_IMODE(st.st_mode):#o}; a private root belongs to the effective user and "
                "no one else can write it",
                self.path,
            )

    def _is_under(self, base: str) -> bool:
        here = os.path.normcase(os.path.realpath(self.path))
        base = os.path.normcase(os.path.realpath(base))
        try:
            return os.path.commonpath([here, base]) == base
        except ValueError:  # another drive
            return False

    def _live(self) -> None:
        if self._closed:
            raise ValueError(f"StorageDir {self.path!r} is closed")

    def _name(self, name: str) -> None:
        _check_name(name, strict=self._path_mode)

    def _sub(self, name: str) -> str:
        return os.path.join(self.path, name)

    def _guard(self) -> None:
        """Path backend: every component below the root, down to this directory, is a plain
        directory. The root itself is as given and is not checked."""
        if not self._path_mode:
            return
        for p in self._chain:
            self._plain(p, "directory")
        if not self._is_root:
            self._plain(self.path, "directory")

    def _member(self, name: str) -> os.stat_result | None:
        """Path backend: ``lstat`` of ``name`` (None if absent), refusing a symlink or reparse
        point, which is never followed."""
        try:
            st = os.lstat(self._sub(name))
        except FileNotFoundError:
            return None
        if stat.S_ISLNK(st.st_mode) or is_reparse(st):
            raise _wrong_kind(self._sub(name), "regular file or directory")
        return st

    @staticmethod
    def _plain(path: str, expected: str) -> os.stat_result:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or is_reparse(st) or not stat.S_ISDIR(st.st_mode):
            raise _wrong_kind(path, expected)
        return st

    def _open_dir_fd(self, name: str) -> int:
        try:
            return os.open(
                name, os.O_RDONLY | _DIRECTORY | _NOFOLLOW | _CLOEXEC, dir_fd=self._fd
            )
        except OSError as err:
            if err.errno in _WRONG_KIND_ERRNOS:
                raise _wrong_kind(self._sub(name), "directory") from err
            raise

    def child(
        self, name: str, *, create: bool = False, mode: int | None = None
    ) -> "StorageDir":
        """Open the directory ``name`` below this one, without following a symlink.

        Parameters
        ----------
        name : str
            One path component.
        create : bool
            Make it if it does not exist. A directory this call makes gets ``mode`` through its
            descriptor, whatever the umask; one that exists is left as it is.
        mode : int, optional
            The mode of a directory this call creates; default the policy's ``container_mode``.

        Raises
        ------
        WrongKindError
            ``name`` is a symlink or not a directory.
        FileNotFoundError
            It does not exist and ``create`` is false.
        """
        self._live()
        self._name(name)
        mode = self.policy.container_mode if mode is None else mode
        _check_mode("mode", mode)
        sub = self._sub(name)
        if self._path_mode:
            self._guard()
            self._member(name)
            if create:
                try:
                    os.mkdir(sub, mode)
                except FileExistsError:
                    pass
                else:
                    if not IS_WINDOWS:
                        os.chmod(sub, mode)
            self._plain(sub, "directory")
            chain = self._chain if self._is_root else (*self._chain, self.path)
            return StorageDir(sub, None, self.policy, self.trust, chain, is_root=False)
        made = False
        if create:
            try:
                os.mkdir(name, mode, dir_fd=self._fd)
                made = True
            except FileExistsError:
                pass
        fd = self._open_dir_fd(name)
        if made:
            try:
                os.fchmod(fd, mode)
            except BaseException:
                os.close(fd)
                raise
        return StorageDir(sub, fd, self.policy, self.trust, is_root=False)

    # -- members -----------------------------------------------------------------------

    def open_new(self, name: str, mode: int) -> ty.BinaryIO:
        """Create the file ``name`` for writing (``O_CREAT | O_EXCL | O_NOFOLLOW``), give it
        ``mode`` through its descriptor, and return it opened for writing in binary mode.

        Raises
        ------
        FileExistsError
            ``name`` exists, as a symlink or anything else.
        """
        self._live()
        self._name(name)
        _check_mode("mode", mode)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY | _CLOEXEC
        if self._path_mode:
            self._guard()
            fd = os.open(self._sub(name), flags, mode)
        else:
            fd = os.open(name, flags, mode, dir_fd=self._fd)
        try:
            if not IS_WINDOWS:
                os.fchmod(fd, mode)
            return os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise

    def open_read(self, name: str) -> ty.BinaryIO:
        """Open the regular file ``name`` for reading, without following a symlink.

        Raises
        ------
        WrongKindError
            ``name`` is a symlink or not a regular file.
        """
        self._live()
        self._name(name)
        flags = os.O_RDONLY | _NOFOLLOW | _BINARY | _CLOEXEC | _NONBLOCK
        if self._path_mode:
            self._guard()
            st = self._member(name)
            if st is not None and not stat.S_ISREG(st.st_mode):
                raise _wrong_kind(self._sub(name), "regular file")
        try:
            if self._path_mode:
                fd = os.open(self._sub(name), flags)
            else:
                fd = os.open(name, flags, dir_fd=self._fd)
        except OSError as err:
            if err.errno in _WRONG_KIND_ERRNOS or err.errno == errno.EISDIR:
                raise _wrong_kind(self._sub(name), "regular file") from err
            raise
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise _wrong_kind(self._sub(name), "regular file")
            return os.fdopen(fd, "rb")
        except BaseException:
            os.close(fd)
            raise

    # -- entries -----------------------------------------------------------------------

    def rename(self, name: str, target: "StorageDir", new_name: str) -> None:
        """Rename ``name`` here to ``new_name`` in ``target`` (atomic within one filesystem).
        A directory is not renamed over one that is not empty; a file replaces silently on POSIX
        and fails on Windows."""
        self._live()
        target._live()
        self._name(name)
        target._name(new_name)
        if self._path_mode or target._path_mode:
            self._guard()
            target._guard()
            self._member(name)
            os.rename(self._sub(name), target._sub(new_name))
            return
        os.rename(name, new_name, src_dir_fd=self._fd, dst_dir_fd=target._fd)

    def link(self, name: str, new_name: str) -> None:
        """Hard-link ``name`` to ``new_name`` in this directory; fails if ``new_name`` exists."""
        self._live()
        self._name(name)
        self._name(new_name)
        if self._path_mode:
            self._guard()
            self._member(name)
            os.link(self._sub(name), self._sub(new_name), follow_symlinks=False)
            return
        os.link(
            name,
            new_name,
            src_dir_fd=self._fd,
            dst_dir_fd=self._fd,
            follow_symlinks=False,
        )

    def unlink(self, name: str) -> None:
        """Remove the file or symlink ``name`` from this directory; a symlink is removed, never
        followed. On Windows a read-only file is made writable first."""
        self._live()
        self._name(name)
        if self._path_mode:
            self._guard()
            path = self._sub(name)
            try:
                os.unlink(path)
            except PermissionError:
                if not IS_WINDOWS:
                    raise
                os.chmod(path, stat.S_IWRITE)
                os.unlink(path)
            return
        os.unlink(name, dir_fd=self._fd)

    def rmdir(self, name: str) -> None:
        """Remove the empty directory ``name`` from this directory; a symlink is refused, never
        followed."""
        self._live()
        self._name(name)
        if self._path_mode:
            self._guard()
            self._member(name)
            os.rmdir(self._sub(name))
            return
        try:
            os.rmdir(name, dir_fd=self._fd)
        except OSError as err:
            if err.errno in (errno.ENOTDIR, errno.ELOOP):
                raise _wrong_kind(self._sub(name), "directory") from err
            raise

    def lstat(self, name: str) -> os.stat_result:
        """``lstat`` of ``name`` in this directory."""
        self._live()
        self._name(name)
        if self._path_mode:
            self._guard()
            return os.lstat(self._sub(name))
        return os.stat(name, dir_fd=self._fd, follow_symlinks=False)

    def scandir(self) -> ty.Iterator["StorageEntry"]:
        """List this directory. Each listing opens its own directory offset on first use, so
        listings may be nested or interleaved, and closing a listing early releases it. Every
        entry carries an ``lstat`` taken as it is yielded, so it stays valid after the listing
        ends; a name removed meanwhile is skipped. The listing needs search permission on the
        directory (it opens ``.`` afresh), reports every error, including a closed handle, on its
        first advance, not here, and costs one ``fstatat`` per name."""
        self._live()
        return self._entries()

    def _entries(self) -> ty.Iterator["StorageEntry"]:
        self._live()
        if self._path_mode:
            self._guard()
            with os.scandir(self.path) as listing:
                for entry in listing:
                    self._live()
                    try:
                        st = entry.stat(follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    yield StorageEntry(entry.name, entry.path, st)
            return
        fd = os.open(".", os.O_RDONLY | _DIRECTORY | _CLOEXEC, dir_fd=self._fd)
        try:
            with os.scandir(fd) as listing:
                names = (entry.name for entry in listing)
                for name in names:
                    self._live()
                    try:
                        st = os.stat(name, dir_fd=fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    yield StorageEntry(name, self._sub(name), st)
        finally:
            os.close(fd)

    def set_mode(self, mode: int, name: str | None = None) -> None:
        """Set the mode of this directory, or of ``name`` in it, through a descriptor.

        A name that is a symlink is refused (`WrongKindError`), never followed. On Windows only
        a file's read-only attribute is applied, from the owner write bit.
        """
        self._live()
        _check_mode("mode", mode)
        if name is None:
            if self._path_mode:
                self._guard()
                if not IS_WINDOWS:
                    os.chmod(self.path, mode)
                return
            os.fchmod(self._fd, mode)
            return
        self._name(name)
        if self._path_mode:
            self._guard()
            st = self._member(name)
            if IS_WINDOWS:
                if st is not None and not stat.S_ISDIR(st.st_mode):
                    os.chmod(
                        self._sub(name), stat.S_IWRITE if mode & 0o200 else stat.S_IREAD
                    )
            else:
                os.chmod(self._sub(name), mode)
            return
        try:
            fd = os.open(
                name, os.O_RDONLY | _NOFOLLOW | _CLOEXEC | _NONBLOCK, dir_fd=self._fd
            )
        except OSError as err:
            if err.errno in _WRONG_KIND_ERRNOS:
                raise _wrong_kind(self._sub(name), "regular file or directory") from err
            raise
        try:
            os.fchmod(fd, mode)
        finally:
            os.close(fd)

    def fstat(self) -> os.stat_result:
        """``fstat`` of this directory (of the root's target, for a root that is a symlink)."""
        self._live()
        if self._path_mode:
            self._guard()
            return os.stat(self.path) if self._is_root else os.lstat(self.path)
        return os.fstat(self._fd)

    def sync(self) -> None:
        """Flush this directory's entries to storage, where the platform can."""
        self._live()
        if self._fd is not None:
            os.fsync(self._fd)

    # -- lifetime ----------------------------------------------------------------------

    def close(self) -> None:
        """Close the descriptor. Further use raises `ValueError`; closing again is allowed."""
        fd, self._fd = self._fd, None
        self._closed = True
        if fd is not None:
            os.close(fd)

    def __enter__(self) -> "StorageDir":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"StorageDir({self.path!r})"


def _libc() -> ty.Any:
    import ctypes
    import ctypes.util

    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def _reflink(src: ty.BinaryIO, dst: StorageDir, name: str) -> bool:
    """Make ``name`` in ``dst`` a copy-on-write clone of the open file ``src`` (the descriptor
    that was checked, not its path). False if the filesystem or platform cannot, in which case
    nothing was made."""
    if dst._fd is None:
        return False
    if _DARWIN:
        import ctypes

        try:
            libc = _libc()
            clone = libc.fclonefileat
        except (OSError, AttributeError, TypeError):
            return False
        clone.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        if clone(src.fileno(), dst._fd, os.fsencode(name), 0) == 0:
            return True
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(err, os.strerror(err), dst._sub(name))
        return False
    try:
        import fcntl
    except ImportError:
        return False
    out = dst.open_new(name, 0o600)
    try:
        fcntl.ioctl(out.fileno(), getattr(fcntl, "FICLONE", 0x40049409), src.fileno())
    except OSError:
        out.close()
        dst.unlink(name)
        return False
    out.close()
    return True


def _clear_flags(fd: int) -> None:
    """Remove the BSD file flags that make a clone unusable in staging, ``UF_IMMUTABLE`` and
    ``UF_APPEND``, which it carries over from its source. Every other flag stays: clearing
    ``UF_COMPRESSED`` would drop a compressed file's content."""
    flags = getattr(os.fstat(fd), "st_flags", 0)
    blocking = flags & (
        getattr(stat, "UF_IMMUTABLE", 0) | getattr(stat, "UF_APPEND", 0)
    )
    if blocking and _libc().fchflags(fd, flags & ~blocking) != 0:
        import ctypes

        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _open_source(src: Path) -> ty.BinaryIO:
    """Open ``src`` without blocking (a FIFO opens at once) and require a regular file."""
    fd = os.open(src, os.O_RDONLY | _NONBLOCK | _BINARY | _CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _wrong_kind(os.fspath(src), "regular file")
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


def copy_hashed(src: Path, dst: StorageDir, name: str, mode: int) -> tuple[str, int]:
    """Copy the file ``src`` to ``name`` in ``dst`` and return its sha256 hex digest and size.

    The copy is a reflink (``FICLONE`` on Linux, ``fclonefileat`` on macOS) where the filesystem
    supports one, else a block copy hashed as it is written; never a hard link. ``name`` is a new
    file given ``mode`` through its descriptor, so ``src`` keeps its mode and content. With
    ``durability="member"`` the member is
    ``fsync``ed and an error from that is raised.

    Raises
    ------
    FileExistsError
        ``name`` exists in ``dst``.
    WrongKindError
        ``src`` is not a regular file.
    """
    dst._live()
    dst._name(name)
    _check_mode("mode", mode)
    digest = hashlib.sha256()
    size = 0
    with _open_source(src) as source:
        expected = os.fstat(source.fileno()).st_size
        created = False
        try:
            if _reflink(source, dst, name):
                created = True
                with dst.open_read(name) as clone:
                    _clear_flags(clone.fileno())
                    if not IS_WINDOWS:
                        os.fchmod(clone.fileno(), mode)
                    for block in iter(lambda: clone.read(_COPY_BLOCK), b""):
                        digest.update(block)
                        size += len(block)
                    if dst.policy.durability == "member":
                        os.fsync(clone.fileno())
            else:
                with dst.open_new(name, mode) as out:
                    created = True
                    for block in iter(lambda: source.read(_COPY_BLOCK), b""):
                        digest.update(block)
                        out.write(block)
                        size += len(block)
                    out.flush()
                    if dst.policy.durability == "member":
                        os.fsync(out.fileno())
            if size != expected:
                raise StoreUnavailable(
                    errno.EIO,
                    f"copied {size} bytes of a source of size {expected}: the size changed "
                    "during the copy or the copy lost content",
                    os.fspath(src),
                )
        except BaseException:
            if created:
                try:
                    dst.unlink(name)
                except OSError:
                    pass
            raise
    return digest.hexdigest(), size
