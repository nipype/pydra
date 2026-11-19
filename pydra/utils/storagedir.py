"""Descriptor-relative directory handles for a portable result store.

A `StorageDir` is an open directory. Every step below it (opening a child, creating,
renaming, linking, listing, changing a mode) is made relative to its descriptor, with
``O_NOFOLLOW``, so a symlink swapped in for a directory or a file while a store is at work
is refused instead of followed. Modes are set through descriptors and never come from
the umask. A `StorePolicy` carries the modes, the durability and the ages a store applies.

Where the platform has no descriptor-relative calls (Windows) a handle works from paths and
checks, before each step, that no component below the root is a symlink or a reparse
point. That check is not race-free, which is acceptable only for a root no other user can
write, so a ``shared`` root is refused there with `PlatformUnsupported`.
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
    "StoreAges",
    "StorePolicy",
    "StoreUnavailable",
    "WrongKindError",
    "copy_hashed",
]

IS_WINDOWS = sys.platform == "win32"

DESCRIPTORS_SUPPORTED = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and {os.open, os.mkdir, os.rename, os.link, os.stat, os.chmod} <= os.supports_dir_fd
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
    )
    if e
)


class StoreUnavailable(OSError):
    """The store cannot be used here: a directory or member it needs is unreadable,
    unwritable or of the wrong kind."""


class WrongKindError(StoreUnavailable):
    """A symlink, or an object of another kind, was found where the store expects a
    directory or a regular file, and it could not be moved aside. The message names the
    path and its owner."""


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
        its staging directory at least once per quarter of this.
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
    ``sealed_mode`` must not include group or other write, which keeps a committed entry
    unchangeable by anyone but its owner; ``entry_mode`` must include owner write and search,
    because a directory without them cannot be renamed into another parent. ``container_mode``
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
        How a read checks a file member. ``"size"`` is accepted only for a ``private`` root
        with ``durability="member"``, which `StorageDir.open_root` enforces.
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
        if self.entry_mode & 0o300 != 0o300:
            raise ValueError(
                f"entry_mode={self.entry_mode:#o} must include owner write and search"
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
        if not isinstance(self.ages, StoreAges):
            raise ValueError("ages must be a StoreAges")


def _check_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or "\0" in name
        or (os.altsep and os.altsep in name)
    ):
        raise ValueError(f"{name!r} is not a single path component")


def _is_reparse(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & _FILE_ATTRIBUTE_REPARSE_POINT)


class StorageDir:
    """An open directory under a store root. Make one with `open_root` and `child`."""

    def __init__(
        self,
        path: str,
        fd: int | None,
        policy: StorePolicy,
        trust: Trust,
        chain: tuple[str, ...] = (),
    ) -> None:
        self.path = path
        self.policy = policy
        self.trust = trust
        self._fd = fd
        self._chain = (
            chain  # path mode: the paths below the root, to recheck before each step
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
            Who can write the root (RFC 0003 section 6.4). Where there are no descriptor-relative
            calls (Windows) ``"shared"`` raises `PlatformUnsupported`.

        Raises
        ------
        PlatformUnsupported
            ``trust="shared"`` on a platform without descriptor-relative calls.
        ValueError
            ``verify_files="size"`` without ``trust="private"`` and ``durability="member"``.
        """
        if trust not in ("shared", "private", "trusted"):
            raise ValueError(
                f"trust must be 'shared', 'private' or 'trusted', got {trust!r}"
            )
        if policy.verify_files == "size" and (
            trust != "private" or policy.durability != "member"
        ):
            raise ValueError(
                "verify_files='size' is accepted only for a private root with durability='member'"
            )
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
            return cls(path, None, policy, trust)
        fd = os.open(path, os.O_RDONLY | _DIRECTORY | _CLOEXEC)
        return cls(path, fd, policy, trust)

    def _sub(self, name: str) -> str:
        return os.path.join(self.path, name)

    def _guard(self, name: str | None = None) -> None:
        """Path mode: the chain from the root to here, and ``name``, are plain objects."""
        if self._fd is not None:
            return
        for p in self._chain:
            self._plain(p, "directory")
        if self._chain:
            self._plain(self.path, "directory")
        if name is not None:
            try:
                st = os.lstat(self._sub(name))
            except FileNotFoundError:
                return
            if stat.S_ISLNK(st.st_mode) or _is_reparse(st):
                raise _wrong_kind(self._sub(name), "regular file or directory")

    @staticmethod
    def _plain(path: str, expected: str) -> os.stat_result:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode) or _is_reparse(st) or not stat.S_ISDIR(st.st_mode):
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
        _check_name(name)
        mode = self.policy.container_mode if mode is None else mode
        _check_mode("mode", mode)
        sub = self._sub(name)
        if self._fd is None:
            self._guard(name)
            if create:
                try:
                    os.mkdir(sub, mode)
                except FileExistsError:
                    pass
                else:
                    if not IS_WINDOWS:
                        os.chmod(sub, mode)
            self._plain(sub, "directory")
            return StorageDir(
                sub, None, self.policy, self.trust, (*self._chain, self.path)
            )
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
        return StorageDir(sub, fd, self.policy, self.trust)

    # -- members -----------------------------------------------------------------------

    def open_new(self, name: str, mode: int) -> ty.BinaryIO:
        """Create the file ``name`` for writing (``O_CREAT | O_EXCL | O_NOFOLLOW``), give it
        ``mode`` through its descriptor, and return it opened for writing in binary mode.

        Raises
        ------
        FileExistsError
            ``name`` exists, as a symlink or anything else.
        """
        _check_name(name)
        _check_mode("mode", mode)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _BINARY | _CLOEXEC
        if self._fd is None:
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
        _check_name(name)
        flags = os.O_RDONLY | _NOFOLLOW | _BINARY | _CLOEXEC
        if self._fd is None:
            self._guard(name)
        try:
            if self._fd is None:
                fd = os.open(self._sub(name), flags | _NONBLOCK)
            else:
                fd = os.open(name, flags | _NONBLOCK, dir_fd=self._fd)
        except OSError as err:
            if err.errno in _WRONG_KIND_ERRNOS or (err.errno == errno.EISDIR):
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
        A directory is not renamed over one that is not empty."""
        _check_name(name)
        _check_name(new_name)
        if self._fd is None or target._fd is None:
            self._guard(name)
            target._guard()
            os.rename(self._sub(name), target._sub(new_name))
            return
        os.rename(name, new_name, src_dir_fd=self._fd, dst_dir_fd=target._fd)

    def link(self, name: str, new_name: str) -> None:
        """Hard-link ``name`` to ``new_name`` in this directory; fails if ``new_name`` exists."""
        _check_name(name)
        _check_name(new_name)
        if self._fd is None:
            self._guard(name)
            os.link(self._sub(name), self._sub(new_name), follow_symlinks=False)
            return
        os.link(
            name,
            new_name,
            src_dir_fd=self._fd,
            dst_dir_fd=self._fd,
            follow_symlinks=False,
        )

    def lstat(self, name: str) -> os.stat_result:
        """``lstat`` of ``name`` in this directory."""
        _check_name(name)
        if self._fd is None:
            self._guard()
            return os.lstat(self._sub(name))
        return os.stat(name, dir_fd=self._fd, follow_symlinks=False)

    def scandir(self) -> ty.Iterator[os.DirEntry]:
        """List this directory."""
        if self._fd is None:
            self._guard()
            return os.scandir(self.path)
        return os.scandir(self._fd)

    def set_mode(self, mode: int, name: str | None = None) -> None:
        """Set the mode of this directory, or of ``name`` in it, through a descriptor.

        A name that is a symlink is refused (`WrongKindError`), never followed. On Windows only
        a file's read-only attribute is applied, from the owner write bit.
        """
        _check_mode("mode", mode)
        if name is None:
            if self._fd is None:
                self._guard()
                if not IS_WINDOWS:
                    os.chmod(self.path, mode)
                return
            os.fchmod(self._fd, mode)
            return
        _check_name(name)
        if self._fd is None:
            self._guard(name)
            if IS_WINDOWS:
                if not stat.S_ISDIR(os.lstat(self._sub(name)).st_mode):
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
        """``fstat`` of this directory."""
        if self._fd is None:
            self._guard()
            return os.lstat(self.path)
        return os.fstat(self._fd)

    def sync(self) -> None:
        """Flush this directory's entries to storage, where the platform can."""
        if self._fd is not None:
            os.fsync(self._fd)

    # -- lifetime ----------------------------------------------------------------------

    def close(self) -> None:
        """Close the descriptor. A handle is not used afterwards."""
        fd, self._fd = self._fd, None
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


def _reflink(src: ty.BinaryIO, dst: StorageDir, name: str) -> bool:
    """Make ``name`` in ``dst`` a copy-on-write clone of the open file ``src``. False if the
    filesystem or platform cannot, in which case nothing was made."""
    if dst._fd is None:
        return False
    if sys.platform == "darwin":
        try:
            import ctypes
            import ctypes.util

            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            clone = libc.clonefileat
        except (OSError, AttributeError, TypeError):
            return False
        clone.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        at_fdcwd = -2
        # src's path: the clone is of the file as it is now, not of whatever the path becomes
        done = clone(at_fdcwd, os.fsencode(src.name), dst._fd, os.fsencode(name), 0)
        if done == 0:
            return True
        err = ctypes.get_errno()
        if err == errno.EEXIST:
            raise FileExistsError(err, os.strerror(err), dst._sub(name))
        return False
    try:
        import fcntl

        ficlone = 0x40049409
        out = dst.open_new(name, 0o600)
    except ImportError:
        return False
    try:
        fcntl.ioctl(out.fileno(), ficlone, src.fileno())
    except OSError:
        out.close()
        os.unlink(name, dir_fd=dst._fd)
        return False
    out.close()
    return True


def copy_hashed(src: Path, dst: StorageDir, name: str, mode: int) -> tuple[str, int]:
    """Copy the file ``src`` to ``name`` in ``dst`` and return its sha256 hex digest and size.

    The copy is a reflink (``FICLONE`` on Linux, ``clonefileat`` on macOS) where the filesystem
    supports one, else a block copy hashed as it is written; never a hard link. ``name`` is a new
    file given ``mode`` through its descriptor, so ``src``, which may itself be a hard link to a
    user's dataset file, keeps its mode and content.

    Raises
    ------
    FileExistsError
        ``name`` exists in ``dst``.
    WrongKindError
        ``src`` is not a regular file.
    """
    _check_name(name)
    _check_mode("mode", mode)
    digest = hashlib.sha256()
    size = 0
    with open(src, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise _wrong_kind(os.fspath(src), "regular file")
        if _reflink(source, dst, name):
            dst.set_mode(mode, name)
            with dst.open_read(name) as clone:
                for block in iter(lambda: clone.read(_COPY_BLOCK), b""):
                    digest.update(block)
                    size += len(block)
            if dst.policy.durability == "member":
                _fsync_member(dst, name)
            return digest.hexdigest(), size
        with dst.open_new(name, mode) as out:
            for block in iter(lambda: source.read(_COPY_BLOCK), b""):
                digest.update(block)
                out.write(block)
                size += len(block)
            out.flush()
            if dst.policy.durability == "member":
                os.fsync(out.fileno())
    return digest.hexdigest(), size


def _fsync_member(dst: StorageDir, name: str) -> None:
    with dst.open_read(name) as f:
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
