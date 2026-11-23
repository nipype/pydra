"""What a pool root must do, measured (RFC 0005 section 5.3, ``pydracli pool probe``).

`probe` checks the primitives the pool protocol uses (an exclusive ``O_EXCL`` create, an atomic
``mkdir``, rename, an exclusive link (a rename that refuses an existing name, on Windows), a flush of
a directory, a storage clock) in a scratch directory under the root, and measures how late a second
process sees a directory's mtime move. `require` refuses a root that lacks a primitive or whose
measured delay exceeds ``attribute_delay``.
"""

from __future__ import annotations

import errno
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import attrs

from pydra.engine.lease import ATTRIBUTE_DELAY
from pydra.utils.storage import (
    _STAMP_AGE,
    UNSUPPORTED_ERRNOS,
    _refused,
    probe_capabilities,
)
from pydra.utils.storagedir import IS_WINDOWS

__all__ = ["ProbeResult", "probe", "refusal", "require"]

PREFIX = ".pydra-pool-probe-"

_WATCH = r"""
import os, sys, time
d, limit = sys.argv[1], float(sys.argv[2])
before = os.stat(d).st_mtime_ns
sys.stdout.write("ready\n"); sys.stdout.flush()
sys.stdin.readline()
end, missed = time.monotonic() + limit, None
while time.monotonic() < end:
    looked = time.time()
    if os.stat(d).st_mtime_ns != before:
        sys.stdout.write(repr(missed) + "\n"); sys.stdout.flush(); sys.exit(0)
    missed = looked
    time.sleep(0.005)
sys.stdout.write("none\n")
"""


@attrs.frozen
class ProbeResult:
    """What `probe` found.

    Attributes
    ----------
    oexcl_create, atomic_mkdir, atomic_rename, exclusive_link, fsync_dir, server_clock : bool
    mtime_delay : float or None
        Seconds after a create at which a second process saw its directory's mtime move (the
        probe's limit when it did not within it); None when not measured.
    """

    oexcl_create: bool
    atomic_mkdir: bool
    atomic_rename: bool
    exclusive_link: bool
    fsync_dir: bool
    server_clock: bool
    mtime_delay: float | None = None

    _PRIMITIVES = (
        "oexcl_create",
        "atomic_mkdir",
        "atomic_rename",
        "exclusive_link",
        "fsync_dir",
        "server_clock",
    )

    def missing(self) -> list[str]:
        """The primitives the root lacks."""
        return [name for name in self._PRIMITIVES if not getattr(self, name)]

    def to_json(self) -> dict:
        return attrs.asdict(self)

    @classmethod
    def from_json(cls, record: dict) -> "ProbeResult":
        return cls(**record)


def _oexcl(work: Path) -> bool:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    os.close(os.open(work / "oexcl", flags, 0o600))
    return _refused(
        lambda: os.close(os.open(work / "oexcl", flags, 0o600)), errno.EEXIST
    )


def _exclusive_link(work: Path) -> bool:
    (work / "src").write_bytes(b"a")
    (work / "dst").write_bytes(b"b")
    if IS_WINDOWS:
        return _refused(lambda: os.rename(work / "src", work / "dst"), errno.EEXIST)
    try:
        os.link(work / "src", work / "made")
    except OSError as err:
        if err.errno in UNSUPPORTED_ERRNOS or err.errno == errno.EPERM:
            return False
        raise
    refused = _refused(lambda: os.link(work / "src", work / "dst"), errno.EEXIST)
    return refused and (work / "dst").read_bytes() == b"b"


def _fsync_dir(work: Path) -> bool:
    if IS_WINDOWS:  # no directory descriptors; NTFS journals its metadata
        return True
    fd = os.open(work, os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as err:
        if err.errno in (errno.EINVAL, errno.EBADF) or err.errno in UNSUPPORTED_ERRNOS:
            return False
        raise
    finally:
        os.close(fd)
    return True


def visibility_delay(made: float, reply: str, limit: float) -> float:
    """The storage's part of how late a second process saw a change made at ``made``: the
    time from ``made`` to the last look that still missed it (the watcher's ``reply``), so
    time the watcher spent not looking (starting, descheduled) does not count; ``limit``
    when it never saw it."""
    if reply == "none" or not reply:
        return limit
    missed = None if reply == "None" else float(reply)
    return 0.0 if missed is None else max(0.0, missed - made)


def _mtime_delay(work: Path, limit: float) -> float:
    """How late a second process sees ``work``'s mtime move after an entry is made in it
    (`visibility_delay`)."""
    watched = work / "watched"
    os.mkdir(watched)
    old = time.time() - _STAMP_AGE
    os.utime(watched, (old, old))
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", _WATCH, str(watched), repr(limit)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        if proc.stdout.readline().strip() != "ready":
            return limit
        os.close(
            os.open(watched / "entry", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        )
        made = time.time()
        proc.stdin.write("go\n")
        proc.stdin.flush()
        return visibility_delay(made, proc.stdout.readline().strip(), limit)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(limit + 10)
        for stream in (proc.stdin, proc.stdout):
            stream.close()


def probe(
    root: os.PathLike,
    *,
    limit: float = ATTRIBUTE_DELAY + 1.0,
    measure_delay: bool = True,
) -> ProbeResult:
    """Probe the existing directory ``root`` in a scratch directory made and removed under it.

    Parameters
    ----------
    root : os.PathLike
    limit : float
        Seconds to wait for a second process to see a directory change; the pool passes its
        ``attribute_delay`` and one second for the watcher to start (default: the lease's
        default ``attribute_delay``).
    measure_delay : bool
    """
    work = Path(root).absolute() / f"{PREFIX}{uuid.uuid4().hex}"
    os.mkdir(work, 0o700)
    try:
        caps = probe_capabilities(work)
        return ProbeResult(
            oexcl_create=_oexcl(work),
            atomic_mkdir=caps.atomic_mkdir,
            atomic_rename=caps.atomic_rename,
            exclusive_link=_exclusive_link(work),
            fsync_dir=_fsync_dir(work),
            server_clock=caps.server_clock,
            mtime_delay=_mtime_delay(work, limit) if measure_delay else None,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


def refusal(result: ProbeResult, attribute_delay: float) -> str | None:
    """Why a pool may not use a root that probed as ``result``; None when it may."""
    reasons = []
    if missing := result.missing():
        reasons.append(f"the storage has no {', '.join(missing)}")
    if result.mtime_delay is not None and result.mtime_delay > attribute_delay:
        reasons.append(
            f"a second process saw a directory change {result.mtime_delay:.3g} s late, more than "
            f"attribute_delay {attribute_delay:g} s"
        )
    return "; ".join(reasons) or None


def require(result: ProbeResult, root: os.PathLike, attribute_delay: float) -> None:
    """Raise `PoolStorageUnsupported` when `refusal` refuses ``result``."""
    from pydra.workers.pool.storage import PoolStorageUnsupported

    why = refusal(result, attribute_delay)
    if why:
        raise PoolStorageUnsupported(
            errno.ENOTSUP, f"cannot hold a pool here: {why}", str(root)
        )
