"""Trust per root and the ownership walk (RFC 0003 section 6.4).

Trust is declared by whoever opens a root and is never read from it. ``shared`` never opens a
pickle member, ``private`` opens one only below directories that belong to the current user and
that no one else can write, and ``trusted`` asserts that everyone who can write the root may
run code in this process.
"""

from __future__ import annotations

import os
import stat
import typing as ty

import attrs

from pydra.engine.portable.errors import EntryNotTrusted
from pydra.utils.storagedir import IS_WINDOWS, StorageDir, Trust, WrongKindError

__all__ = ["Trust", "Root", "EntryNotTrusted", "check_private", "TRUSTS"]

TRUSTS = ("shared", "private", "trusted")


def _trust(value: ty.Any) -> str:
    if value not in TRUSTS:
        raise ValueError(f"trust must be one of {TRUSTS}, got {value!r}")
    return value


@attrs.frozen
class Root:
    """A root the store reads or writes, with the trust its opener declares for it.

    Attributes
    ----------
    path : os.PathLike
    trust : {"shared", "private", "trusted"}
        ``shared`` is the default for every root passed explicitly.
    """

    path: os.PathLike
    trust: Trust = attrs.field(default="shared", validator=lambda _i, _a, v: _trust(v))


def _owned_and_unshared(st: os.stat_result, where: str) -> None:
    if st.st_uid != os.geteuid() or st.st_mode & 0o022:
        raise EntryNotTrusted(
            f"{where} is owned by uid {st.st_uid} with mode {stat.S_IMODE(st.st_mode):#o}; a pickle "
            "member is read from a private root only below directories and files that belong to "
            "the current user and that no one else can write"
        )


def _no_check(st: os.stat_result, where: str) -> None:
    """Windows: no owner or group and other bits to read."""


def check_private(root: StorageDir, path: ty.Sequence[str]) -> None:
    """Check that ``root``, every directory named in ``path`` below it, and its last element are
    owned by the effective user and not writable by group or others.

    ``path`` is the chain from the root to the member to be opened: for a pickle,
    ``[entries, shard, cdir, fingerprint, "<sha256>.pkl"]``. Every step is made through the
    descriptor of the one before, without following a symlink, and each descriptor is checked
    with ``fstat``. On
    Windows there is no owner or mode to read; ``private`` there means a root under the user's
    local application data, which `StorageDir.open_root` checked, and the path backend refuses
    reparse points at each step.

    Raises
    ------
    EntryNotTrusted
        A component fails the check or is a symlink or another wrong kind.
    FileNotFoundError
        A component does not exist.
    """
    check = _no_check if IS_WINDOWS else _owned_and_unshared
    check(root.fstat(), root.path)
    cur = root
    opened: list[StorageDir] = []
    try:
        for i, name in enumerate(path):
            where = os.path.join(cur.path, name)
            try:
                if stat.S_ISLNK(cur.lstat(name).st_mode):
                    raise EntryNotTrusted(f"{where} is a symlink")
                if i < len(path) - 1 or stat.S_ISDIR(cur.lstat(name).st_mode):
                    cur = cur.child(name)
                    opened.append(cur)
                    st = cur.fstat()
                else:
                    with cur.open_read(name) as f:
                        st = os.fstat(f.fileno())
            except WrongKindError as err:
                raise EntryNotTrusted(
                    f"{where} is not what it should be: {err}"
                ) from err
            check(st, where)
    finally:
        for d in opened:
            d.close()
