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

from pydra.utils.storage import UNSUPPORTED_ERRNOS, probe_capabilities

__all__ = ["ProbeResult", "probe", "refusal", "require"]

IS_WINDOWS = sys.platform == "win32"
PREFIX = ".pydra-pool-probe-"
_STAMP_AGE = 86400.0
MEASURE_SLACK = 0.5
"""Seconds of a measured delay put down to starting and scheduling the watching process, not to
the storage: a delay up to ``attribute_delay + MEASURE_SLACK`` is accepted."""

_WATCH = r"""
import os, sys, time
d, limit = sys.argv[1], float(sys.argv[2])
before = os.stat(d).st_mtime_ns
sys.stdout.write("ready\n"); sys.stdout.flush()
sys.stdin.readline()
end = time.monotonic() + limit
while time.monotonic() < end:
    if os.stat(d).st_mtime_ns != before:
        sys.stdout.write(repr(time.time()) + "\n"); sys.stdout.flush(); sys.exit(0)
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


def _refused(call, *errnos: int) -> bool:
    try:
        call()
    except OSError as err:
        return err.errno in errnos
    return False


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


def _mtime_delay(work: Path, limit: float) -> float:
    """How late a second process sees ``work``'s mtime move after an entry is made in it."""
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
        seen = proc.stdout.readline().strip()
        try:
            return max(0.0, float(seen) - made)
        except ValueError:
            return limit
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(limit + 10)
        for stream in (proc.stdin, proc.stdout):
            stream.close()


def probe(
    root: os.PathLike, *, limit: float = 61.0, measure_delay: bool = True
) -> ProbeResult:
    """Probe the existing directory ``root`` in a scratch directory made and removed under it.

    Parameters
    ----------
    root : os.PathLike
    limit : float
        Seconds to wait for a second process to see a directory change.
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
    if (
        result.mtime_delay is not None
        and result.mtime_delay > attribute_delay + MEASURE_SLACK
    ):
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
