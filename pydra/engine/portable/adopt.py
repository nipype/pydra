"""Adopting an existing store already laid out like a `PortableStore` (RFC 0003 section 8.2).

A store laid out as ``<version dir>/<shard>/<checksum>/<fingerprint>/`` with its own members
becomes a `PortableStore` without rewriting an entry: its version directory is renamed to the
entry format's ``entries`` name in one directory rename, which fences the writers that predate
the header, and the header is written naming that directory, the tree's own side directories
and the tree's own modes.
"""

from __future__ import annotations

import contextlib
import logging
import os
import stat
import typing as ty

import attrs

from pydra.engine.portable.entry import EntryFormat, PydraEntryV1
from pydra.engine.portable.errors import (
    CorruptEntry,
    EntryFormatUnknown,
    EntryNotTrusted,
)
from pydra.engine.portable.identity import StoreIdentity
from pydra.engine.portable.layout import (
    _FORMAT_RE,
    _HEX2,
    CDIR,
    FINGERPRINT,
    HASHED_CDIR,
    FORMAT_MAJOR,
    HEADER_NAME,
    LayoutHeader,
    StoreLayoutMismatch,
    _component,
    _differing_key,
    cdir_for,
    check_layout_names,
    fold,
    is_reserved,
    open_layout,
    read_header,
    shard_for,
)
from pydra.utils.storagedir import StorageDir, StorePolicy

__all__ = ["AdoptRefused", "AdoptReport", "adopt", "SAMPLE"]

logger = logging.getLogger("pydra")

SAMPLE = 16
"""The default of `adopt`'s ``sample``: enough entries to show a fault that a writer made
systematically (a mode, a shard rule), few enough to read in seconds on network storage. An
unmeasured choice; ``check="all"`` reads every entry."""

_MODE_KEYS = ("file", "entry", "sealed", "container", "side", "staging")
_UNREADABLE = (
    CorruptEntry,
    EntryFormatUnknown,
    EntryNotTrusted,
    KeyError,
    ValueError,
    IndexError,
)  # what a read back reports as misfiled


class AdoptRefused(ValueError):
    """The root cannot be adopted as asked; nothing was renamed or written."""


@attrs.frozen
class AdoptReport:
    """What `adopt` did.

    Attributes
    ----------
    renamed : tuple of str
        The old version directory and the name it was renamed to.
    header_written : bool
    checked : int
        How many entries were read back through the identity and the entry format.
    misfiled : list of str
        The entries, ``<shard>/<cdir>/<fingerprint>`` relative to the version directory, that
        failed that read; why is logged. They are left where they are; the store quarantines
        each when it reads it.
    """

    renamed: tuple[str, str]
    header_written: bool
    checked: int
    misfiled: list[str] = attrs.field(factory=list)


def _policy_of(header: LayoutHeader) -> StorePolicy:
    modes = header.modes
    missing = [k for k in _MODE_KEYS if k not in modes]
    if missing:
        raise AdoptRefused(f"the header's modes lack {', '.join(missing)}")
    try:
        parsed = {k: None if modes[k] is None else int(modes[k], 8) for k in _MODE_KEYS}
        return StorePolicy(
            file_mode=parsed["file"],
            entry_mode=parsed["entry"],
            sealed_mode=parsed["sealed"],
            container_mode=parsed["container"],
            side_mode=parsed["side"],
            staging_mode=parsed["staging"],
        )
    except (TypeError, ValueError) as err:
        raise AdoptRefused(
            f"the header's modes are not a store policy: {err}"
        ) from None


def _check_header(
    header: LayoutHeader,
    identity: StoreIdentity,
    entry_format: EntryFormat,
    old_entries: str,
) -> StorePolicy:
    found = _FORMAT_RE.fullmatch(header.format)
    if found is None or int(found.group(1)) != FORMAT_MAJOR:
        raise AdoptRefused(f"the header's format {header.format!r} is not known")
    policy = _policy_of(header)
    try:
        check_layout_names(header.dirs, header.entries, header.shard)
    except StoreLayoutMismatch as err:
        raise AdoptRefused(str(err)) from None
    if not _component(old_entries) or is_reserved(old_entries):
        raise AdoptRefused(
            f"{old_entries!r} is not the name of a version directory under the root"
        )
    if fold(old_entries) in {fold(parts[0]) for parts in _side_paths(header)}:
        raise AdoptRefused(
            f"{old_entries!r} is one of the directories the header names in dirs, not a "
            "version directory"
        )
    expected = LayoutHeader.for_store(
        identity,
        policy,
        shard=header.shard,  # type: ignore[arg-type]
        pickle=header.pickle,
        entries=entry_format.entries,
        entry_format=entry_format.name,
        dirs=header.dirs,
    )
    key = _differing_key(header, expected)
    if key is not None:
        raise AdoptRefused(
            f"the header's {key!r} is not that of the identity and entry format given: "
            f"{getattr(header, key, header.identity)!r}"
        )
    if fold(header.entries) == fold(old_entries):
        raise AdoptRefused(
            f"the header names {old_entries!r} as its entries: adoption renames the old "
            "version directory to a new name, which fences the writers that predate the "
            "header (RFC 0003 section 8.2)"
        )
    return policy


def _side_paths(header: LayoutHeader) -> list[list[str]]:
    paths = [[header.dirs["staging"]], [header.dirs["quarantine"]]]
    for relative in (header.dirs.get("sidecars") or {}).values():
        paths.append([p for p in str(relative).split("/") if p])
    return paths


def _mode_of(parent: StorageDir, name: str) -> int:
    return stat.S_IMODE(parent.lstat(name).st_mode)


def _expect(where: str, key: str, found: int, allowed: ty.Sequence[int | None]) -> None:
    if found not in allowed:
        shown = " or ".join(f"{m:#o}" for m in allowed if m is not None)
        raise AdoptRefused(
            f"{where} has mode {found:#o} and the header's {key!r} mode is {shown}: a header "
            "must state the modes the tree has, or every owner would reseal every entry on "
            "its first read (RFC 0003 section 8.2)"
        )


def _check_sides(root: StorageDir, header: LayoutHeader, policy: StorePolicy) -> None:
    for parts in _side_paths(header):
        handles: list[StorageDir] = []
        try:
            parent = root
            for part in parts:
                try:
                    mode = _mode_of(parent, part)
                except FileNotFoundError:
                    break
                _expect(
                    os.path.join(parent.path, part), "side", mode, [policy.side_mode]
                )
                parent = parent.child(part)
                handles.append(parent)
        finally:
            for h in handles:
                h.close()


def _entries_of(old: StorageDir) -> ty.Iterator[tuple[StorageDir, str, str, str]]:
    """``(cdir handle, shard, cdir, fingerprint)`` of each entry, in name order; names outside
    the layout's grammar (aside names, ``q-`` names) are passed over."""
    for shard in sorted(old.names()):
        if not _HEX2.fullmatch(shard):
            continue
        with old.child(shard) as shard_dir:
            for cdir in sorted(shard_dir.names()):
                if not (CDIR.fullmatch(cdir) or HASHED_CDIR.fullmatch(cdir)):
                    continue
                with shard_dir.child(cdir) as cdir_dir:
                    for fp in sorted(cdir_dir.names()):
                        if FINGERPRINT.fullmatch(fp):
                            yield cdir_dir, shard, cdir, fp


def _checksum_of(entry_format: EntryFormat, entry: StorageDir, cdir: str) -> str:
    if HASHED_CDIR.fullmatch(cdir):
        recorded = entry_format.identify(entry)[0]
        if cdir_for(recorded) == cdir:
            return recorded
    return cdir


def _survey(
    root: StorageDir,
    old_entries: str,
    header: LayoutHeader,
    policy: StorePolicy,
    identity: StoreIdentity,
    entry_format: EntryFormat,
    check: str,
    trust: ty.Any,
    sample: int,
) -> tuple[int, int, list[str]]:
    """Check the modes and the shard of the entries ``check`` covers (at least the first),
    and read those it covers back: ``(entries seen, entries read, misfiled)``. Nothing is
    changed."""
    _expect(
        os.path.join(root.path, old_entries),
        "container",
        _mode_of(root, old_entries),
        [policy.container_mode],
    )
    limit = {"none": 1, "sample": sample, "all": None}[check]
    sealed = [policy.sealed_mode, policy.entry_mode]
    seen = checked = 0
    misfiled: list[str] = []
    with root.child(old_entries) as old:
        checked_holders: set[tuple[str, str]] = set()
        for cdir_dir, shard, cdir, fp in _entries_of(old):
            if limit is not None and seen >= limit:
                break
            seen += 1
            rel = f"{shard}/{cdir}/{fp}"
            if (shard, cdir) not in checked_holders:
                checked_holders.add((shard, cdir))
                _expect(
                    os.path.join(old.path, shard),
                    "container",
                    _mode_of(old, shard),
                    [policy.container_mode],
                )
                _expect(
                    cdir_dir.path,
                    "container",
                    stat.S_IMODE(cdir_dir.fstat().st_mode),
                    [policy.container_mode],
                )
            with cdir_dir.child(fp) as entry:
                _expect(
                    entry.path,
                    "sealed" if policy.sealed_mode is not None else "entry",
                    stat.S_IMODE(entry.fstat().st_mode),
                    sealed,
                )
                for e in entry.scandir():
                    if e.is_symlink():
                        raise AdoptRefused(
                            f"{e.path} is a symlink, which no entry may hold"
                        )
                    if e.is_file():
                        _expect(
                            e.path,
                            "file",
                            stat.S_IMODE(e.stat().st_mode),
                            [policy.file_mode],
                        )
                try:
                    checksum = _checksum_of(entry_format, entry, cdir)
                except _UNREADABLE as err:
                    if check != "none":
                        checked += 1
                        misfiled.append(rel)
                        logger.warning("%s does not verify: %s", rel, err)
                    continue
                try:
                    expected_shard = shard_for(cdir, checksum, header.shard)  # type: ignore[arg-type]
                except ValueError as err:
                    raise AdoptRefused(f"{rel}: {err}") from None
                if expected_shard != shard:
                    raise AdoptRefused(
                        f"{rel} is in shard {shard!r} and the header's shard rule "
                        f"{header.shard!r} gives {expected_shard!r}"
                    )
                if check == "none":
                    continue
                checked += 1
                try:
                    context, _ = entry_format.context(entry)
                    if identity.fingerprint(context) != fp:
                        raise CorruptEntry(
                            "the context does not give the fingerprint it is found at"
                        )
                    stored = entry_format.read(
                        entry, checksum, fp, None, trust, root=root
                    )
                    if not identity.check_key(checksum, stored.key):
                        raise CorruptEntry(
                            "the key record does not digest to the checksum"
                        )
                except _UNREADABLE as err:
                    misfiled.append(rel)
                    logger.warning("%s does not verify: %s", rel, err)
    return seen, checked, misfiled


def adopt(
    root: os.PathLike,
    *,
    old_entries: str,
    header: LayoutHeader,
    identity: StoreIdentity,
    entry_format: EntryFormat,
    check: ty.Literal["none", "sample", "all"] = "sample",
    sample: int = SAMPLE,
    trust: ty.Literal["shared", "private", "trusted"] = "trusted",
    allow_empty: bool = False,
) -> AdoptReport:
    """Make the store at ``root`` a `PortableStore` without rewriting an entry (RFC 0003
    section 8.2). Stop the writers that predate the header first.

    The header is checked against ``identity``, ``entry_format`` and the tree's modes and
    shards; the entries ``check`` covers are also read back. ``<root>/<old_entries>`` is then
    renamed to ``header.entries`` and the header written; a failed header write undoes the
    rename. An entry that fails the read back is reported in ``misfiled``. A
    ``.pydra-identity`` marker is neither read nor written.

    Parameters
    ----------
    root : os.PathLike
        The store's root.
    old_entries : str
        The name of its version directory.
    header : LayoutHeader
        The header to write; ``dirs`` names the tree's own staging, quarantine and sidecar
        directories and ``modes`` the tree's own modes.
    identity : StoreIdentity
        The key, context and fingerprint of the tree's entries.
    entry_format : EntryFormat
        What is inside the tree's entries; a `PydraEntryV1` is given ``identity``.
    check : {"none", "sample", "all"}
        The entries checked and read back: the first entry's modes and shard only, the first
        ``sample`` in name order, or every entry.
    sample : int
        The entries ``check="sample"`` reads.
    trust : {"shared", "private", "trusted"}
        What the caller asserts about who can write the root (RFC 0003 section 6.4).
    allow_empty : bool
        Adopt a version directory that holds no entry.

    Raises
    ------
    AdoptRefused
        The root has a header, ``old_entries`` is absent or not a version directory,
        ``header.entries`` exists, the tree holds no entry, or the header is unsafe or does
        not match the identity, the entry format or the tree. Nothing was renamed or
        written.
    """
    if check not in ("none", "sample", "all"):
        raise ValueError(f"check must be 'none', 'sample' or 'all', got {check!r}")
    if sample < 1:
        raise ValueError(f"sample must be at least 1, got {sample!r}")
    if isinstance(entry_format, PydraEntryV1):
        entry_format = attrs.evolve(
            entry_format, identity=identity, pickle=header.pickle, trust=trust
        )
    policy = _check_header(header, identity, entry_format, old_entries)
    with StorageDir.open_root(root, policy, trust=trust) as handle:
        try:
            handle.lstat(HEADER_NAME)
        except FileNotFoundError:
            pass
        else:
            raise AdoptRefused(
                f"{handle.path} already has {HEADER_NAME}: it is a portable store"
            )
        try:
            if not stat.S_ISDIR(handle.lstat(old_entries).st_mode):
                raise FileNotFoundError(old_entries)
        except FileNotFoundError:
            try:
                handle.lstat(header.entries)
            except FileNotFoundError:
                pass
            else:
                raise AdoptRefused(
                    f"{handle.path} holds {header.entries!r} and no {old_entries!r} or "
                    f"{HEADER_NAME}: an adopt was interrupted after its rename; rename "
                    f"{header.entries!r} back to {old_entries!r} and adopt again"
                ) from None
            raise AdoptRefused(
                f"{handle.path} has no version directory {old_entries!r} to adopt"
            ) from None
        try:
            handle.lstat(header.entries)
        except FileNotFoundError:
            pass
        else:
            raise AdoptRefused(
                f"{handle.path} already holds {header.entries!r}, the header's entries"
            )
        _check_sides(handle, header, policy)
        seen, checked, misfiled = _survey(
            handle,
            old_entries,
            header,
            policy,
            identity,
            entry_format,
            check,
            trust,
            sample,
        )
        if not seen and not allow_empty:
            raise AdoptRefused(
                f"{old_entries!r} holds no entry named as the layout names them "
                "(<shard>/<checksum>/<fingerprint>); pass allow_empty to adopt it anyway"
            )
        handle.rename(old_entries, handle, header.entries)
        try:
            open_layout(handle, header, create=True, existing=True)
        except BaseException as err:
            published = None
            with contextlib.suppress(Exception):
                published = read_header(handle)
            if published != header:
                with contextlib.suppress(OSError):
                    handle.rename(header.entries, handle, old_entries)
                raise
            if not isinstance(err, Exception):
                raise
            logger.warning("the header was written; then: %r", err)
        handle.sync()
    return AdoptReport(
        renamed=(old_entries, header.entries),
        header_written=True,
        checked=checked,
        misfiled=misfiled,
    )
