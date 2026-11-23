"""What a storage root guarantees: the filesystem primitives a lease on it relies on.

A lease on shared storage (see `pydra.engine.lease`) needs ``mkdir`` that is atomic
(of two processes making one name, exactly one succeeds), ``rename`` that is atomic,
and a clock that every host reads the same way. `probe_capabilities` tests a root for
them, and `storage_now` reads the storage server's clock.
"""

import errno
import os
import shutil
import time
import typing as ty
import uuid
from pathlib import Path
import attrs

PROBE_PREFIX = ".pydra-probe-"
"""The name prefix of the scratch directory `probe_capabilities` makes and removes."""

CLOCK_PREFIX = ".pydra-clock-"
"""The name prefix of the file `storage_now` makes and removes."""

WARM_PREFIX = ".pydra-warm-"
"""The name prefix of the scratch directory a warm container keeps in a cache root."""

UNSUPPORTED_ERRNOS = frozenset(
    e
    for e in (
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
        getattr(errno, "ENOSYS", None),
    )
    if e is not None
)
"""The errors by which a filesystem refuses an operation it does not support."""

_STAMP_AGE = 86400.0
"""Seconds into the past a probe directory's mtime is set, so that a new entry made in
it is seen to move the mtime whatever the offset between the clocks."""


class StorageUnsupported(OSError):
    """A storage root lacks a primitive a lease needs. The message names each missing
    primitive and the root."""


class PlatformUnsupported(StorageUnsupported):
    """The platform, not a storage primitive, cannot give what a root was asked for. The
    message names the platform and the root."""


@attrs.frozen
class StorageCapabilities:
    """What a root does, as measured by `probe_capabilities`.

    Attributes
    ----------
    atomic_mkdir : bool
        ``mkdir`` of a name that exists fails (checked by one process making a name
        twice), which is what lets one of two makers win.
    atomic_rename : bool
        A replace of a file replaces its target whole, a directory is renamed whole,
        and a rename onto a directory that is not empty fails.
    server_clock : bool
        A new entry moves its directory's mtime, so mtimes read from the storage are a
        clock every host shares.
    clock_offset : float | None
        Seconds by which the storage clock is ahead of this host's (negative if behind),
        None if ``server_clock`` is false.
    """

    atomic_mkdir: bool
    atomic_rename: bool
    server_clock: bool
    clock_offset: float | None

    def missing(self) -> list[str]:
        """The primitives this root lacks, by attribute name."""
        names = ("atomic_mkdir", "atomic_rename", "server_clock")
        return [n for n in names if not getattr(self, n)]

    def require(self, root: os.PathLike) -> None:
        """Raise `StorageUnsupported` naming each primitive ``root`` lacks."""
        lacking = self.missing()
        if lacking:
            raise StorageUnsupported(
                errno.ENOTSUP,
                f"cannot hold a lease here: the storage has no {', '.join(lacking)}",
                str(root),
            )


def storage_now(directory: os.PathLike) -> float:
    """The current time on the storage holding ``directory``: the mtime of a file just
    created in it, which the storage stamps. The file is removed: on Windows it is
    opened with ``os.O_TEMPORARY`` and goes on its last close."""
    probe = Path(directory) / f"{CLOCK_PREFIX}{uuid.uuid4().hex}"
    temporary = getattr(os, "O_TEMPORARY", 0)
    fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL | temporary, 0o600)
    try:
        return os.fstat(fd).st_mtime
    finally:
        os.close(fd)
        if not temporary:
            try:
                os.unlink(probe)
            except FileNotFoundError:
                pass


def _refused(call: ty.Callable[[], object], *errnos: int) -> bool:
    """Whether ``call`` raised an ``OSError`` with one of ``errnos``."""
    try:
        call()
    except OSError as err:
        return err.errno in errnos
    return False


def _clock(work: Path) -> tuple[bool, float | None]:
    """Whether a new entry moves its directory's mtime, and the storage clock's offset
    from this host's."""
    stamped = work / "stamped"
    os.mkdir(stamped)
    old = time.time() - _STAMP_AGE
    os.utime(stamped, (old, old))
    before = time.time()
    fd = os.open(stamped / "entry", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        made = os.fstat(fd).st_mtime
    finally:
        os.close(fd)
    after = time.time()
    moved = os.stat(stamped).st_mtime > old + _STAMP_AGE / 2
    if not moved:
        return False, None
    return True, made - (before + after) / 2


def probe_capabilities(root: os.PathLike) -> StorageCapabilities:
    """Test ``root`` for the primitives a lease needs.

    The probes run in a uniquely named scratch directory made under ``root``, which is
    removed afterwards. A root that does not exist yet is judged by its nearest
    existing ancestor.

    Parameters
    ----------
    root : os.PathLike
        The directory to test.

    Returns
    -------
    StorageCapabilities
        What the root does. All false if the root refuses ``mkdir`` as unsupported.

    Raises
    ------
    OSError
        The root cannot be written.
    """
    base = Path(root).absolute()
    while not base.exists() and base != base.parent:
        base = base.parent
    work = base / f"{PROBE_PREFIX}{uuid.uuid4().hex}"
    try:
        os.mkdir(work)
    except OSError as err:
        if err.errno in UNSUPPORTED_ERRNOS:
            return StorageCapabilities(False, False, False, None)
        raise
    try:
        os.mkdir(work / "made")
        atomic_mkdir = _refused(lambda: os.mkdir(work / "made"), errno.EEXIST)

        (work / "a").write_bytes(b"new")
        (work / "b").write_bytes(b"old")
        os.replace(work / "a", work / "b")
        replaced = (work / "b").read_bytes() == b"new" and not (work / "a").exists()
        os.mkdir(work / "from")
        os.rename(work / "from", work / "to")
        moved = (work / "to").is_dir() and not (work / "from").exists()
        os.mkdir(work / "full")
        (work / "full" / "x").write_bytes(b"")
        guarded = _refused(
            lambda: os.rename(work / "to", work / "full"),
            errno.ENOTEMPTY,
            errno.EEXIST,
        )
        server_clock, offset = _clock(work)
        return StorageCapabilities(
            atomic_mkdir=atomic_mkdir,
            atomic_rename=replaced and moved and guarded,
            server_clock=server_clock,
            clock_offset=offset,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


_REQUIRED: dict[Path, StorageCapabilities] = {}


def require_lease_storage(directory: os.PathLike) -> StorageCapabilities:
    """Probe ``directory`` once per process and raise `StorageUnsupported` if it cannot
    carry a lease; return what the probe found."""
    key = Path(directory).absolute()
    caps = _REQUIRED.get(key)
    if caps is None:
        caps = probe_capabilities(key)
        _REQUIRED[key] = caps
    caps.require(key)
    return caps
