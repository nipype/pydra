"""Moving results out of a `~pydra.engine.store.PickleStore` cache (RFC 0003 section 8.1).

`migrate` publishes every usable slot of ``<cache_root>/<checksum>/<ctxkey>/`` into a
`~pydra.engine.portable.store.PortableStore` under the slot's stored context. `publish_legacy`
does the same for one job, from the ``legacy=`` roots of a store, when its lookup finds nothing.
Neither copies a job or error pickle, a lease or a success record, and neither recomputes a job.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import typing as ty
from pathlib import Path

import attrs

from pydra.engine.lease import DirectoryLease, LeaseSettings, LeaseTimeout
from pydra.engine.portable.entry import EntryRef
from pydra.engine.portable.errors import (
    EntryNotTrusted,
    KeyMismatch,
    UnencodableOutput,
)
from pydra.engine.portable.identity import Context
from pydra.engine.portable.layout import HEADER_NAME, cdir_for
from pydra.engine.portable.trust import Trust, check_private
from pydra.engine.result import Result, _loads
from pydra.engine.store import (
    CONTEXT_FILE,
    LEASE_SUFFIX,
    _is_store_internal,
)
from pydra.utils.identity import IDENTITY_SCHEME
from pydra.utils.storagedir import StorageDir, StorePolicy, WrongKindError

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.portable.store import PortableStore

__all__ = [
    "MigratedSlot",
    "MigrationRefused",
    "MigrationReport",
    "migrate",
    "publish_legacy",
]

logger = logging.getLogger("pydra")

RESULT_FILE = "_result.pklz"
V1_SCHEME = "pydra-identity@1"
_JSON_CAP = 1 << 20

Status = ty.Literal["migrated", "skipped", "busy", "unencodable"]


class MigrationRefused(ValueError):
    """The source cannot be migrated into the target as asked; nothing was written."""


@attrs.frozen
class MigratedSlot:
    """What became of one slot ``<checksum>/<ctxkey>/`` of the source.

    Attributes
    ----------
    checksum : str
    ctxkey : str
        The slot's directory name, which the migrated entry does not keep.
    status : {"migrated", "skipped", "busy", "unencodable"}
        ``busy``: its lease was held, so a later call migrates it. ``unencodable``: an output
        has no codec in the target, so the slot stays unmigrated.
    reason : str or None
        Why it was not migrated.
    """

    checksum: str
    ctxkey: str
    status: Status
    reason: str | None = None


@attrs.frozen
class MigrationReport:
    """The outcome of `migrate`.

    Attributes
    ----------
    source : str
        The cache root.
    scheme : str or None
        What the source's ``.pydra-identity`` marker names; None without one.
    slots : tuple of MigratedSlot
        One per slot, in path order.
    """

    source: str
    scheme: str | None
    slots: tuple[MigratedSlot, ...]

    def count(self, status: Status) -> int:
        """How many slots have ``status``."""
        return sum(s.status == status for s in self.slots)


class _Skip(Exception):
    """A slot that is not migrated, and why."""


def _scheme_version(named: str) -> int | None:
    from pydra.engine.submitter import _scheme_version

    return _scheme_version(named)


def read_marker(root: StorageDir) -> str | None:
    """What the cache root's ``.pydra-identity`` names (stripped), or None without one. A
    marker that is not a regular file reads as ``""``, a scheme no pydra writes."""
    from pydra.engine.submitter import IDENTITY_MARKER

    try:
        with root.open_read(IDENTITY_MARKER) as f:
            return f.read(256).decode("utf-8", "replace").strip()
    except FileNotFoundError:
        return None
    except WrongKindError:
        return ""


def check_scheme(marker: str | None, identity: str) -> None:
    """Refuse a cache root whose slots a store of ``identity`` cannot hold.

    A root without a marker was keyed before markers, under ``pydra-identity@1``. A marker
    naming a scheme this pydra does not know, or a newer one, is refused, as is a root keyed
    under ``pydra-identity@2`` or later for a ``pydra-identity@1`` store, which never looks up
    such checksums.

    Raises
    ------
    MigrationRefused
    """
    version = 1 if marker is None else _scheme_version(marker)
    current = _scheme_version(IDENTITY_SCHEME)
    if version is None or (current is not None and version > current):
        raise MigrationRefused(
            f"the cache's .pydra-identity marker names {marker!r}, a scheme this pydra "
            f"({IDENTITY_SCHEME}) does not write, so its checksums cannot be read"
        )
    if identity == V1_SCHEME and version >= 2:
        raise MigrationRefused(
            f"the cache is keyed under {marker}, and a {V1_SCHEME} store never looks up "
            "those checksums"
        )


def _unpickle(data: bytes) -> ty.Any:
    return _loads(data)


def _context_of(cdir: StorageDir, ctxkey: str) -> tuple[dict[str, ty.Any], bytes]:
    """The slot's stored context and the bytes of its ``_context.json``."""
    try:
        with cdir.child(ctxkey) as slot, slot.open_read(CONTEXT_FILE) as f:
            raw = f.read(_JSON_CAP + 1)
    except FileNotFoundError:
        raise _Skip(f"it has no {CONTEXT_FILE}") from None
    except WrongKindError as err:
        raise _Skip(f"it is not a plain directory and file: {err}") from None
    if len(raw) > _JSON_CAP:
        raise _Skip(f"its {CONTEXT_FILE} is larger than {_JSON_CAP} bytes")
    try:
        context = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as err:
        raise _Skip(f"its {CONTEXT_FILE} is not JSON: {err}") from None
    if not isinstance(context, dict):
        raise _Skip(f"its {CONTEXT_FILE} is not a JSON object")
    if "unknown" in context:
        raise _Skip(
            "its context has 'unknown', a per-process nonce, so no other process reuses it"
        )
    return context, raw


def _result_of(
    root: StorageDir, cdir: StorageDir, checksum: str, ctxkey: str, trust: Trust
) -> Result:
    """The slot's unpickled result, which did not error. Under ``private`` the root, the
    checksum and slot directories and ``_result.pklz`` must pass the ownership check first.
    """
    try:
        with cdir.child(ctxkey) as slot:
            slot.lstat(RESULT_FILE)
            if trust == "private":
                check_private(root, [checksum, ctxkey, RESULT_FILE])
            with slot.open_read(RESULT_FILE) as f:
                data = f.read()
    except FileNotFoundError:
        raise _Skip(f"it has no {RESULT_FILE}") from None
    except (WrongKindError, EntryNotTrusted) as err:
        raise _Skip(f"its {RESULT_FILE} is not read: {err}") from None
    if not data:
        raise _Skip(f"its {RESULT_FILE} is empty")
    try:
        result = _unpickle(data)
    except Exception as err:
        raise _Skip(f"its {RESULT_FILE} cannot be unpickled: {err!r}") from None
    if not isinstance(result, Result):
        raise _Skip(f"its {RESULT_FILE} holds a {type(result).__name__}, not a Result")
    if result.errored:
        raise _Skip("its result errored")
    if result.outputs is None:
        raise _Skip("its result has no outputs")
    return result


def _outputs(result: Result) -> dict[str, ty.Any]:
    outputs = result.outputs
    if isinstance(outputs, ty.Mapping):
        return dict(outputs)
    from pydra.utils.general import get_fields

    return {f.name: getattr(outputs, f.name) for f in get_fields(type(outputs))}


def _dirs(parent: StorageDir) -> list[str]:
    """The plain directories directly below ``parent``, in path order; symlinks are left out."""
    return sorted(e.name for e in parent.scandir() if e.is_dir() and not e.is_symlink())


def migrate(
    source: os.PathLike,
    target: "PortableStore",
    *,
    trust_source: bool,
    supplied_checksums: bool = False,
    on_slot: ty.Callable[[MigratedSlot], None] | None = None,
) -> MigrationReport:
    """Publish each usable slot of the `PickleStore` cache at ``source`` into ``target``
    (RFC 0003 section 8.1).

    The cache is walked by descriptor, without following symlinks. Each slot's lease is taken
    with one attempt, so a slot being written is reported ``busy``. A slot without
    ``_context.json`` or ``_result.pklz``, whose result errored, or whose context has
    ``unknown`` is skipped. The outputs are published under the stored context, through the
    target's identity and entry format, with the slot as ``files_from``. Nothing else of the
    slot is copied, and the cache is written only by its leases.

    Parameters
    ----------
    source : os.PathLike
        The cache root, ``<cache_root>/<checksum>/<ctxkey>/``.
    target : PortableStore
    trust_source : bool
        Must be True: each ``_result.pklz`` is unpickled, which runs code of the cache.
    supplied_checksums : bool
        The caller's assertion that the slots' checksums were supplied (``checksum=``). It is
        required when ``target`` is ``pydra-identity@2``, whose automatic checksums no old slot
        carries; each slot is then published with the key ``{"scheme": "pydra-identity@2",
        "supplied": <checksum>}``. Other identities are given no key record.
    on_slot : callable, optional
        Called with each `MigratedSlot` as it is decided.

    Raises
    ------
    ValueError
        ``trust_source`` is not True.
    MigrationRefused
        ``target`` is ``pydra-identity@2`` without ``supplied_checksums``; the cache's
        ``.pydra-identity`` marker names a scheme the target cannot hold (see
        `check_scheme`); or ``source`` is a portable store.
    """
    if trust_source is not True:
        raise ValueError(
            "migrate() unpickles each slot's _result.pklz, which runs code written by "
            "whoever could write the cache: pass trust_source=True to assert you trust it"
        )
    name = target.identity.name
    v2 = name == IDENTITY_SCHEME
    if v2 and not supplied_checksums:
        raise MigrationRefused(
            f"the target's identity is {name}, whose automatic checksums are digests of "
            "records an old cache does not hold; only slots with supplied checksums can be "
            "migrated, and supplied_checksums=True asserts that they were"
        )
    path = os.path.abspath(os.fspath(source))
    slots: list[MigratedSlot] = []
    with StorageDir.open_root(path, StorePolicy(), trust="trusted") as root:
        try:
            root.lstat(HEADER_NAME)
        except FileNotFoundError:
            pass
        else:
            raise MigrationRefused(
                f"{path} holds {HEADER_NAME}: it is a portable store, not a PickleStore cache"
            )
        scheme = read_marker(root)
        check_scheme(scheme, name)
        for checksum in _dirs(root):
            if _is_store_internal(checksum):
                continue
            try:
                cdir = root.child(checksum)
            except (FileNotFoundError, WrongKindError):
                continue
            with cdir:
                for ctxkey in _dirs(cdir):
                    slot = _migrate_slot(target, root, cdir, path, checksum, ctxkey, v2)
                    slots.append(slot)
                    if on_slot is not None:
                        on_slot(slot)
    return MigrationReport(path, scheme, tuple(slots))


def _migrate_slot(
    target: "PortableStore",
    root: StorageDir,
    cdir: StorageDir,
    path: str,
    checksum: str,
    ctxkey: str,
    v2: bool,
) -> MigratedSlot:
    lease = DirectoryLease(
        checksum,
        Path(path, checksum, ctxkey),
        Path(path, f"{checksum}-{ctxkey}{LEASE_SUFFIX}"),
        settings=LeaseSettings(),
    )
    try:
        wait = lease.try_take()
    except LeaseTimeout:
        wait = 0.0
    if wait is not None:
        return MigratedSlot(checksum, ctxkey, "busy", "its lease is held")
    try:
        context, _ = _context_of(cdir, ctxkey)
        result = _result_of(root, cdir, checksum, ctxkey, "trusted")
        key = {"scheme": IDENTITY_SCHEME, "supplied": checksum} if v2 else None
        committed = target.publish(
            checksum,
            key,
            context,
            _outputs(result),
            result.task,
            Path(path, checksum, ctxkey),
        )
    except _Skip as why:
        return MigratedSlot(checksum, ctxkey, "skipped", str(why))
    except UnencodableOutput as err:
        return MigratedSlot(checksum, ctxkey, "unencodable", str(err))
    except KeyMismatch as err:
        return MigratedSlot(checksum, ctxkey, "skipped", str(err))
    finally:
        lease.__exit__(None, None, None)
    if not committed.stored:
        return MigratedSlot(
            checksum,
            ctxkey,
            "skipped",
            "the target holds another entry at its address, with other outputs",
        )
    return MigratedSlot(checksum, ctxkey, "migrated")


def publish_legacy(store: "PortableStore", job: "Job") -> bool:
    """Publish the result of ``job`` found in one of ``store.legacy`` into ``store`` under the
    stored context, with the job's key record; whether one was published.

    For each legacy root in order, ``<root>/<checksum>`` costs one ``lstat`` when absent. A
    root whose marker it cannot hold (`check_scheme`) or that cannot be opened under its trust
    is logged and skipped. The slots are ranked by ``store.reuse`` as stored entries would be,
    and the first that is usable, passes the ownership check of a ``private`` root and
    publishes is taken. A legacy root is never written.
    """
    checksum = job.checksum
    for legacy in store.legacy:
        path = os.path.abspath(os.fspath(legacy.path))
        try:
            os.lstat(os.path.join(path, checksum))
        except (OSError, ValueError):
            continue
        try:
            if _legacy_hit(store, job, path, legacy.trust):
                return True
        except MigrationRefused as err:
            logger.warning("the legacy root %s was skipped: %s", path, err)
        except (FileNotFoundError, WrongKindError):
            continue
        except OSError as err:
            logger.warning("the legacy root %s was skipped: %s", path, err)
    return False


def _legacy_hit(store: "PortableStore", job: "Job", path: str, trust: Trust) -> bool:
    checksum = job.checksum
    with StorageDir.open_root(path, StorePolicy(), trust=trust) as root:
        check_scheme(read_marker(root), store.identity.name)
        with root.child(checksum) as cdir:
            stored: list[tuple[EntryRef, Context, str]] = []
            slots: dict[str, tuple[str, dict[str, ty.Any]]] = {}
            for ctxkey in _dirs(cdir):
                try:
                    context, raw = _context_of(cdir, ctxkey)
                except _Skip as why:
                    logger.debug("%s/%s/%s: %s", path, checksum, ctxkey, why)
                    continue
                fingerprint = store.identity.fingerprint(context)
                if fingerprint in slots:
                    continue
                slots[fingerprint] = (ctxkey, context)
                ref = EntryRef(path, cdir_for(checksum), fingerprint)
                stored.append((ref, context, hashlib.sha256(raw).hexdigest()))
            if not stored:
                return False
            for candidate in store.reuse.rank(job.context, stored, job.task):
                found = slots.get(candidate.entry.fingerprint)
                if found is None or candidate.entry.root != path:
                    continue
                ctxkey, context = found
                try:
                    result = _result_of(root, cdir, checksum, ctxkey, trust)
                except _Skip as why:
                    logger.warning(
                        "%s/%s/%s is not served: %s", path, checksum, ctxkey, why
                    )
                    continue
                task = result.task if result.task is not None else job.task
                try:
                    committed = store.publish(
                        checksum,
                        store.identity.key(job),
                        context,
                        _outputs(result),
                        task,
                        Path(path, checksum, ctxkey),
                    )
                except (UnencodableOutput, KeyMismatch) as err:
                    logger.warning(
                        "%s/%s/%s is not served: %s", path, checksum, ctxkey, err
                    )
                    continue
                if committed.stored:
                    return True
    return False
