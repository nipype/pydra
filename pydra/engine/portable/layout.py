"""The portable store's layout: the header, ``<cdir>`` and ``<shard>`` (RFC 0003 section 3)."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import time
import typing as ty
import unicodedata
import uuid

import attrs

from pydra.engine.portable.identity import StoreIdentity
from pydra.utils.retry import Backoff
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
)

__all__ = [
    "HEADER_NAME",
    "FORMAT_MAJOR",
    "StoreLayoutUnknown",
    "StoreLayoutMismatch",
    "LegacyLayout",
    "StoreLayoutAbsent",
    "LayoutHeader",
    "open_layout",
    "read_header",
    "check_layout_names",
    "fold",
    "is_reserved",
    "cdir_for",
    "shard_for",
]

HEADER_NAME = "pydra-store.json"
FORMAT = "pydra-store@1.0"
FORMAT_MAJOR = 1
TEMP_PREFIX = ".pydra-store."
TEMP_SUFFIX = ".tmp"
MAX_HEADER_BYTES = 1 << 16
READ_RETRY = Backoff(first=0.005, cap=0.2, deadline=5.0)
"""How a read of the header retries a Windows sharing violation (a header being linked
or replaced): within the 5 s a replace takes on a loaded host."""
FINGERPRINT = re.compile(r"[0-9a-f]{64}")
"""A lowercase hex SHA-256: an entry's fingerprint, a member's content digest, the digest in a
hashed cdir."""
HASHED_CDIR = re.compile(r"h-[0-9a-f]{64}")
"""A cdir made from a checksum that is not a valid name (`cdir_for`)."""

CDIR = re.compile(r"[a-z0-9][a-z0-9_-]{0,199}")
"""A checksum that is its own cdir."""
_FORMAT_RE = re.compile(r"pydra-store@(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
_HEX2 = re.compile(r"[0-9a-f]{2}")
_TEMP_RE = re.compile(r"\.pydra-store\.[0-9a-f]{32}\.tmp")
DEFAULT_DIRS: ty.Mapping[str, ty.Any] = {
    "staging": "staging",
    "quarantine": "quarantine",
    "sidecars": {},
}


class StoreLayoutUnknown(StoreUnavailable):
    """The root's header is of a format major this code does not know, or cannot be read."""


class StoreLayoutMismatch(StoreUnavailable):
    """The root's header differs from the store's configuration in ``key``."""

    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key

    def __reduce__(self) -> tuple:
        return (type(self), (self.key, str(self)))


class LegacyLayout(StoreUnavailable):
    """The root holds names but no header: a ``PickleStore`` cache or another tool's directory.
    A new root must be an empty directory."""


class StoreLayoutAbsent(StoreUnavailable):
    """The root is empty and ``open_layout`` was told not to create the header."""


def _mode(mode: int | None) -> str | None:
    return None if mode is None else f"0o{mode:o}"


@attrs.frozen
class LayoutHeader:
    """The content of ``<root>/pydra-store.json`` (RFC 0003 section 3.2).

    Attributes
    ----------
    format : str
        ``pydra-store@<major>.<minor>``.
    entries : str
        The entry format's version directory.
    entry_format : str
    identity : dict
        ``name``, ``fingerprint`` and ``recorded_only`` (a sorted list).
    shard : {"sha256-2", "prefix2"}
    dirs : dict
        ``staging``, ``quarantine`` and ``sidecars`` (a mapping of names).
    modes : dict
        ``file``, ``entry``, ``sealed``, ``container``, ``side`` and ``staging``, each an
        ``"0o..."`` string or None.
    pickle : bool
        Whether writers to this root may write pickle members. It is recorded when the root is
        created and never compared on open (RFC 0003 sections 3.2 and 6.4).
    """

    format: str
    entries: str
    entry_format: str
    identity: dict
    shard: str
    dirs: dict
    modes: dict
    pickle: bool

    @classmethod
    def for_store(
        cls,
        identity: StoreIdentity,
        policy: StorePolicy | None = None,
        *,
        shard: ty.Literal["sha256-2", "prefix2"] = "sha256-2",
        pickle: bool = False,
        entries: str = "results@1",
        entry_format: str = "pydra-entry@1",
        dirs: ty.Mapping[str, ty.Any] | None = None,
    ) -> "LayoutHeader":
        """The header a store with ``identity`` and ``policy`` expects."""
        policy = StorePolicy() if policy is None else policy
        return cls(
            format=FORMAT,
            entries=entries,
            entry_format=entry_format,
            identity={
                "name": identity.name,
                "fingerprint": identity.fingerprint_name,
                "recorded_only": sorted(identity.recorded_only),
            },
            shard=shard,
            dirs=dict(dirs if dirs is not None else {**DEFAULT_DIRS, "sidecars": {}}),
            modes={
                "file": _mode(policy.file_mode),
                "entry": _mode(policy.entry_mode),
                "sealed": _mode(policy.sealed_mode),
                "container": _mode(policy.container_mode),
                "side": _mode(policy.side_mode),
                "staging": _mode(policy.staging_mode),
            },
            pickle=pickle,
        )

    def to_json(self) -> bytes:
        return (
            json.dumps(attrs.asdict(self), indent=2, sort_keys=False, ensure_ascii=True)
            + "\n"
        ).encode("ascii")

    @classmethod
    def from_json(cls, data: bytes) -> "LayoutHeader":
        """Parse a stored header; keys this version does not know are ignored.

        Raises
        ------
        StoreLayoutUnknown
            The document is not JSON, lacks a key, or names a format major other than 1.
        """
        try:
            doc = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as err:
            raise StoreLayoutUnknown(
                f"{HEADER_NAME} is not readable JSON: {err}"
            ) from None
        if not isinstance(doc, dict):
            raise StoreLayoutUnknown(f"{HEADER_NAME} does not hold a JSON object")
        fmt = doc.get("format")
        found = _FORMAT_RE.fullmatch(fmt) if isinstance(fmt, str) else None
        if found is None or int(found.group(1)) != FORMAT_MAJOR:
            raise StoreLayoutUnknown(
                f"{HEADER_NAME} has format {fmt!r}; this version reads pydra-store@{FORMAT_MAJOR}.x"
            )
        names = [f.name for f in attrs.fields(cls)]
        missing = [n for n in names if n not in doc]
        if missing:
            raise StoreLayoutUnknown(f"{HEADER_NAME} lacks {', '.join(missing)}")
        ident = doc["identity"]
        if not isinstance(ident, dict) or not all(
            k in ident for k in ("name", "fingerprint", "recorded_only")
        ):
            raise StoreLayoutUnknown(f"{HEADER_NAME} has a malformed identity")
        only = ident["recorded_only"]
        if not isinstance(only, list) or not all(isinstance(f, str) for f in only):
            raise StoreLayoutUnknown(
                f"{HEADER_NAME} has a malformed identity.recorded_only"
            )
        for name in ("dirs", "modes"):
            if not isinstance(doc[name], dict):
                raise StoreLayoutUnknown(f"{HEADER_NAME} has a malformed {name}")
        for name in ("entries", "entry_format", "shard"):
            if not isinstance(doc[name], str):
                raise StoreLayoutUnknown(f"{HEADER_NAME} has a malformed {name}")
        if not isinstance(doc["pickle"], bool):
            raise StoreLayoutUnknown(f"{HEADER_NAME} has a malformed pickle")
        return cls(**{n: doc[n] for n in names})


def _recorded(header: LayoutHeader) -> list:
    only = header.identity.get("recorded_only")
    return sorted(only) if isinstance(only, list) else only


def _differing_key(stored: LayoutHeader, expected: LayoutHeader) -> str | None:
    """The first header key at which ``stored`` differs from ``expected``, in the order the RFC
    lists them; ``recorded_only`` is told apart from the rest of ``identity``."""
    for key in ("entries", "entry_format"):
        if getattr(stored, key) != getattr(expected, key):
            return key
    for part in ("name", "fingerprint"):
        if stored.identity.get(part) != expected.identity.get(part):
            return "identity"
    if _recorded(stored) != _recorded(expected):
        return "recorded_only"
    for key in ("shard", "dirs", "modes"):
        if getattr(stored, key) != getattr(expected, key):
            return key
    return None


def _read(root: StorageDir) -> bytes | None:
    tries = READ_RETRY.tries()
    for _ in tries:
        try:
            f = root.open_read(HEADER_NAME)
            break
        except FileNotFoundError:
            return None
        except PermissionError as err:
            # On Windows a file being linked or replaced transiently refuses opens
            if not IS_WINDOWS or tries.last:
                raise StoreUnavailable(f"{HEADER_NAME} cannot be read: {err}") from None
    with f:
        data = f.read(MAX_HEADER_BYTES + 1)
    if len(data) > MAX_HEADER_BYTES:
        raise StoreLayoutUnknown(
            f"{HEADER_NAME} is larger than {MAX_HEADER_BYTES} bytes"
        )
    return data


def read_header(root: StorageDir) -> LayoutHeader | None:
    """The header of ``root`` as stored, or None if it has none.

    Raises
    ------
    StoreLayoutUnknown
        The header is of a format major other than 1, or cannot be read.
    """
    data = _read(root)
    return None if data is None else LayoutHeader.from_json(data)


def _check(stored: LayoutHeader, expected: LayoutHeader) -> LayoutHeader:
    key = _differing_key(stored, expected)
    if key is not None:
        raise StoreLayoutMismatch(
            key,
            f"the store at this root differs from this configuration in {key!r}: "
            f"stored {_shown(stored, key)}, configured {_shown(expected, key)}",
        )
    return stored


def _shown(header: LayoutHeader, key: str) -> str:
    if key == "recorded_only":
        return repr(_recorded(header))
    if key == "identity":
        return repr((header.identity.get("name"), header.identity.get("fingerprint")))
    return repr(getattr(header, key))


def _sweep_and_scan(root: StorageDir, sweep: bool = True) -> list[str]:
    """Remove header temporaries older than the staging age (best effort); return the first
    other names present."""
    age = root.policy.ages.staging
    now = time.time()
    others: list[str] = []
    for entry in root.scandir():
        if entry.name == HEADER_NAME:
            continue
        if _TEMP_RE.fullmatch(entry.name) and entry.is_file():
            if sweep and now - entry.stat().st_mtime > age:
                try:
                    root.unlink(entry.name)
                except OSError:
                    pass
            continue
        others.append(entry.name)
        if len(others) > 3:
            break
    return others


_NO_LINK = (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EMLINK)


def _publish(root: StorageDir, tmp: str) -> bool:
    """Make ``tmp`` the header unless one exists. True if this call made it."""
    try:
        if IS_WINDOWS:
            root.rename(tmp, root, HEADER_NAME)
        else:
            root.link(tmp, HEADER_NAME)
    except FileExistsError:
        return False
    except OSError as err:
        if IS_WINDOWS and isinstance(err, PermissionError):
            try:
                root.lstat(HEADER_NAME)
            except OSError:
                pass
            else:
                return False
        if err.errno in _NO_LINK:
            raise StoreUnavailable(
                "this filesystem cannot link a file without replacing, so "
                f"{HEADER_NAME} cannot be created: {err}"
            ) from None
        raise
    return True


def _legacy(root: StorageDir, others: list[str]) -> StoreUnavailable:
    return LegacyLayout(
        f"{root.path} has names ({', '.join(sorted(others)[:3])}"
        f"{', ...' if len(others) > 3 else ''}) but no {HEADER_NAME}: it is a PickleStore "
        "cache or another tool's directory; migrate it (RFC 0003 section 8). A new root "
        "must be an empty directory"
    )


_OPEN_PASSES = 8
"""Passes of `open_layout`. A pass that does not return lost the link to another opener,
whose header the next pass reads, so two suffice; the bound only ends a loop against a
process that keeps removing the header."""


def open_layout(
    root: StorageDir,
    expected: LayoutHeader,
    *,
    create: bool = True,
    existing: bool = False,
) -> LayoutHeader:
    """Check ``root``'s header against ``expected``, creating it if the root is empty.

    The header is written to a private temporary with the policy's ``file_mode``, ``fsync``ed,
    and linked to its name, which fails if the name exists, so a concurrent opener sees none or
    a complete one. Header temporaries older than the staging age are removed. A new root must
    be an empty directory. Every stored key but ``pickle`` must equal ``expected``'s; ``pickle``
    is recorded at creation and returned as stored.

    Parameters
    ----------
    create : bool
        With False nothing is written (a read-only root works) and an empty root raises
        `StoreLayoutAbsent`.
    existing : bool
        Write the header into a root that already holds names, which a new root may not:
        the adoption of a store already in this shape (RFC 0003 section 8.2,
        `~pydra.engine.portable.adopt.adopt`).

    Returns
    -------
    LayoutHeader
        The header as stored (a minor version above ``expected``'s is kept).

    Raises
    ------
    StoreLayoutUnknown
        The stored format major is not 1, or the header cannot be read.
    StoreLayoutMismatch
        A stored key differs from ``expected``; ``key`` names it.
    LegacyLayout
        The root holds names but no header.
    StoreLayoutAbsent
        The root is empty and ``create`` is false.
    """
    for _ in range(_OPEN_PASSES):
        data = _read(root)
        if data is not None:
            return _check(LayoutHeader.from_json(data), expected)
        others = _sweep_and_scan(root, sweep=create)
        if others and not existing:
            data = _read(root)  # another opener may have linked it since the first read
            if data is not None:
                return _check(LayoutHeader.from_json(data), expected)
            raise _legacy(root, others)
        if not create:
            raise StoreLayoutAbsent(
                f"{root.path} is empty and has no {HEADER_NAME}; it was not created"
            )
        tmp = f"{TEMP_PREFIX}{uuid.uuid4().hex}{TEMP_SUFFIX}"
        try:
            try:
                f = root.open_new(tmp, root.policy.file_mode)
            except PermissionError as err:
                raise StoreUnavailable(
                    f"{root.path} is not writable, so {HEADER_NAME} cannot be created: {err}"
                ) from None
            with f:
                f.write(expected.to_json())
                f.flush()
                os.fsync(f.fileno())
            made = _publish(root, tmp)
        finally:
            try:
                root.unlink(tmp)
            except FileNotFoundError:
                pass
        if made:
            return expected
    raise StoreUnavailable(f"could not settle the header of {root.path}")


SHARD_RULES = ("sha256-2", "prefix2")
RESERVED_NAMES = frozenset({"leases", "work", "audit", HEADER_NAME})
"""Names under the root that the store owns whatever the header says"""
SIDECAR_ROOT = "sidecars"
"""Where a sidecar the header does not map is kept; never staging or quarantine"""
_TEMP_FOLDED = re.compile(_TEMP_RE.pattern, re.IGNORECASE)


def fold(name: str) -> str:
    """``name`` as a case-insensitive, normalization-insensitive filesystem (APFS, NTFS)
    compares it: NFC, then case-folded. Names the store must keep apart are compared
    folded on every platform."""
    return unicodedata.normalize("NFC", name).casefold()


def is_reserved(name: str) -> bool:
    """Whether ``name``, folded, is one the store owns under the root whatever the header
    says: ``leases``, ``work``, ``audit``, the header or a header temporary."""
    return fold(name) in {fold(n) for n in RESERVED_NAMES} or bool(
        _TEMP_FOLDED.fullmatch(unicodedata.normalize("NFC", name))
    )


def _component(name: ty.Any) -> bool:
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\0" not in name
    )


def check_layout_names(dirs: ty.Any, entries: str, shard: ty.Any) -> None:
    """Refuse a ``dirs`` or ``shard`` under which ``collect`` or a writer could reach what is not
    theirs: ``staging``, ``quarantine`` and the first component of each sidecar directory must
    be single path components (a sidecar a relative path of them), staging and quarantine must
    differ, neither may be ``sidecars``, no sidecar may start at either, and none may be
    ``entries``, ``leases``, ``work``, ``audit``, the header or a header temporary; ``shard``
    must be a known rule. Names are compared `fold`-ed (case- and normalization-insensitive)
    on every platform. Checked wherever a store takes them from a header.

    Raises
    ------
    StoreLayoutMismatch
        ``key`` is ``"dirs"`` or ``"shard"``.
    """
    if shard not in SHARD_RULES:
        raise StoreLayoutMismatch(
            "shard", f"the shard rule {shard!r} is not one of {SHARD_RULES}"
        )

    def refuse(why: str) -> ty.NoReturn:
        raise StoreLayoutMismatch("dirs", f"the layout's dirs {dirs!r}: {why}")

    if not isinstance(dirs, ty.Mapping):
        refuse("not a mapping")
    staging, quarantine = dirs.get("staging"), dirs.get("quarantine")
    for key, name in (("staging", staging), ("quarantine", quarantine)):
        if not _component(name):
            refuse(f"{key} {name!r} is not a single path component")
    if fold(staging) == fold(quarantine):
        refuse("staging and quarantine are one directory")
    for key, name in (("staging", staging), ("quarantine", quarantine)):
        if fold(name) == fold(SIDECAR_ROOT):
            refuse(f"{key} {name!r} is where unmapped sidecars are kept")
    sidecars = dirs.get("sidecars") or {}
    if not isinstance(sidecars, ty.Mapping):
        refuse("sidecars is not a mapping")
    tops = {staging, quarantine}
    for name, relative in sidecars.items():
        parts = (
            [p for p in relative.split("/") if p] if isinstance(relative, str) else []
        )
        if not parts or not all(_component(p) for p in parts):
            refuse(f"the sidecar {name!r} at {relative!r} is not a path below the root")
        if fold(parts[0]) in (fold(staging), fold(quarantine)):
            refuse(f"the sidecar {name!r} is inside {parts[0]!r}")
        tops.add(parts[0])
    for top in tops:
        if fold(top) == fold(entries) or is_reserved(top):
            refuse(f"{top!r} is a name the store owns")


def cdir_for(checksum: str) -> str:
    """The checksum as a directory name: itself if it matches ``[a-z0-9][a-z0-9_-]{0,199}``,
    else ``h-<sha256 of the checksum>``."""
    if CDIR.fullmatch(checksum):
        return checksum
    return "h-" + hashlib.sha256(checksum.encode("utf-8", "surrogatepass")).hexdigest()


def shard_for(cdir: str, checksum: str, rule: ty.Literal["sha256-2", "prefix2"]) -> str:
    """The two-hex-character shard directory of an entry.

    ``sha256-2`` is the first two characters of the sha256 of the checksum; ``prefix2`` the
    first two of ``cdir``.

    Raises
    ------
    ValueError
        ``rule`` is unknown, or ``prefix2`` and ``cdir`` does not start with two lower-case hex
        characters.
    """
    if rule == "sha256-2":
        return hashlib.sha256(checksum.encode("utf-8", "surrogatepass")).hexdigest()[:2]
    if rule == "prefix2":
        shard = cdir[:2]
        if not _HEX2.fullmatch(shard):
            raise ValueError(
                f"shard rule 'prefix2' needs a checksum starting with two hex characters, got {cdir!r}"
            )
        return shard
    raise ValueError(f"unknown shard rule {rule!r}")
