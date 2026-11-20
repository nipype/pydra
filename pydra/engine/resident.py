"""Resident state a process can hold between jobs: provided fields, providers and the
``ResidentSet`` (RFC 0005 section 3.1).

Residency is scheduler metadata. A provided field's value is the spec, which is hashed
into the checksum like any input; the loaded object never is. A run that
declares no provided field creates no ``ResidentSet`` and loads nothing; the module
itself is small (stdlib and attrs) and is imported by every job.
"""

import atexit
import functools
import hashlib
import json
import logging
import threading
import typing as ty
from collections import OrderedDict
from contextlib import ExitStack, contextmanager

import attrs

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

__all__ = [
    "Device",
    "Footprint",
    "Provided",
    "ProvidedBy",
    "Provider",
    "ResidentKey",
    "ResidentSet",
    "UnregisteredProvider",
    "provided_fields",
    "residency_digest",
]

logger = logging.getLogger("pydra.resident")

ENTRY_POINT_GROUP = "pydra.providers"


@attrs.frozen
class Device:
    """Where a provider loads its state: ``kind`` is ``"cpu"`` or ``"cuda"``"""

    kind: str = "cpu"
    index: int | None = None


@attrs.frozen
class Footprint:
    """Bytes held on the host and on the device; ``None`` is unknown"""

    host: int | None = None
    device: int | None = None


@attrs.frozen
class ProvidedBy:
    """The ``Annotated`` marker that names the provider of a field's value"""

    name: str


class Provided:
    """``Provided[T, name]`` is ``Annotated[T, ProvidedBy(name)]``"""

    def __class_getitem__(cls, item: tuple[ty.Any, str]) -> ty.Any:
        try:
            type_, name = item
        except (TypeError, ValueError):
            raise TypeError("Provided[...] takes a type and a provider name") from None
        if not isinstance(name, str):
            raise TypeError(f"The provider name must be a string, not {name!r}")
        return ty.Annotated[type_, ProvidedBy(name)]


@ty.runtime_checkable
class Provider(ty.Protocol):
    """Loads, measures and closes one kind of resident state"""

    name: ty.ClassVar[str]

    def load(self, spec: ty.Any, device: Device) -> object:
        """Load the state for ``spec`` on ``device``: an object, or a client of a process it starts"""

    def close(self, obj: object) -> None:
        """Stop what ``load`` started, and wait. A provider that starts a process
        subclasses :class:`pydra.engine.procgroup.ProcessProvider`, whose ``close``
        stops the process group and waits for its members and device memory"""

    def footprint(self, spec: ty.Any) -> Footprint | None:
        """The declared bytes ``spec`` will hold, host and device"""

    def measure(self, obj: object) -> Footprint | None:
        """The bytes the loaded state holds now"""

    def version(self) -> str:
        """The loader's code, which binds calibration"""


class UnregisteredProvider(LookupError):
    """A provided field names a provider that is neither passed to the ``Submitter``
    nor registered under the ``pydra.providers`` entry point group"""


def _canonical(value: ty.Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@attrs.frozen
class ResidentKey:
    """Names state a worker can hold between jobs

    Parameters
    ----------
    kind : str
        ``"provider"`` or ``"environment"``
    name : str
        The provider's name, or the environment's plugin name
    spec : str
        The digest (:func:`pydra.utils.digest.digest`) of the spec, or the environment's
        resolved reference
    """

    kind: ty.Literal["provider", "environment"]
    name: str
    spec: str

    def digest(self) -> str:
        """The sha256 of the canonical JSON of the three fields"""
        record = {"kind": self.kind, "name": self.name, "spec": self.spec}
        return hashlib.sha256(_canonical(record).encode()).hexdigest()


class _Entry:
    __slots__ = ("obj", "close", "footprint", "busy")

    def __init__(self, obj, close, footprint):
        self.obj = obj
        self.close = close
        self.footprint = footprint
        self.busy = 0


class ResidentSet:
    """Loaded state in LRU order under a host and a device byte budget

    State in use (taken with :meth:`get` and not yet :meth:`release`-d) is never
    evicted; eviction calls the entry's ``close``. A budget that busy entries alone
    exceed is exceeded.

    Parameters
    ----------
    host_budget, device_budget : int, optional
        Bytes; ``None`` is unbounded
    """

    def __init__(
        self, host_budget: int | None = None, device_budget: int | None = None
    ):
        self.host_budget = host_budget
        self.device_budget = device_budget
        self._entries: "OrderedDict[ResidentKey, _Entry]" = OrderedDict()
        self._lock = threading.RLock()

    def keys(self) -> list[ResidentKey]:
        """The resident keys, least recently used first"""
        with self._lock:
            return list(self._entries)

    def __contains__(self, key: ResidentKey) -> bool:
        return key in self._entries

    def _totals(self) -> tuple[int, int]:
        host = sum(e.footprint.host or 0 for e in self._entries.values())
        device = sum(e.footprint.device or 0 for e in self._entries.values())
        return host, device

    def _over(self, extra: Footprint) -> bool:
        host, device = self._totals()
        return (
            self.host_budget is not None and host + (extra.host or 0) > self.host_budget
        ) or (
            self.device_budget is not None
            and device + (extra.device or 0) > self.device_budget
        )

    def _take_evictable(self, extra: Footprint) -> list[_Entry]:
        """Remove idle entries, least recent first, until ``extra`` fits; the caller
        holds the lock and closes what is returned after releasing it"""
        taken = []
        for key in list(self._entries):
            if not self._over(extra):
                break
            entry = self._entries[key]
            if entry.busy:
                continue
            del self._entries[key]
            taken.append(entry)
        return taken

    @staticmethod
    def _close_all(entries: list[_Entry]) -> list[Exception]:
        """Close each entry, logging and returning the errors rather than raising"""
        errors = []
        for entry in entries:
            if entry.close is None:
                continue
            try:
                entry.close(entry.obj)
            except Exception as exc:
                logger.error("Closing resident state failed: %r", exc)
                errors.append(exc)
        return errors

    def get(
        self,
        key: ResidentKey,
        loader: ty.Callable[[], ty.Any],
        *,
        close: ty.Callable[[ty.Any], None] | None = None,
        footprint: Footprint | None = None,
    ) -> ty.Any:
        """The object held under ``key``, loaded by ``loader()`` on a miss, marked busy
        until :meth:`release`. Idle entries are evicted, least recent first, and closed
        (outside the lock, their errors logged) to make room for ``footprint`` before a
        load"""
        footprint = footprint or Footprint()
        while True:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    entry.busy += 1
                    return entry.obj
                doomed = self._take_evictable(footprint)
                if not doomed:
                    entry = _Entry(loader(), close, footprint)
                    self._entries[key] = entry
                    entry.busy += 1
                    return entry.obj
            self._close_all(doomed)

    def release(self, key: ResidentKey) -> None:
        """End one use of ``key``; the entry becomes evictable when none remain"""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or not entry.busy:
                return
            entry.busy -= 1
            doomed = self._take_evictable(Footprint())
        self._close_all(doomed)

    @contextmanager
    def use(self, key: ResidentKey, loader, **kwargs) -> ty.Iterator[ty.Any]:
        """:meth:`get` for the length of a ``with`` block"""
        obj = self.get(key, loader, **kwargs)
        try:
            yield obj
        finally:
            self.release(key)

    def close(self) -> None:
        """Close every entry, busy or not, even if some closes raise; the errors are
        raised together once all are closed"""
        with self._lock:
            entries, self._entries = list(self._entries.values()), OrderedDict()
        if errors := self._close_all(entries):
            raise ExceptionGroup("closing resident state failed", errors)


_PROCESS_SET: ResidentSet | None = None


def process_set() -> ResidentSet:
    """The set of this process, for workers that have no ``Submitter`` of their own"""
    global _PROCESS_SET
    if _PROCESS_SET is None:
        _PROCESS_SET = ResidentSet()
        # a pool worker leaves through multiprocessing's exit function, which skips
        # atexit handlers, so register with both
        atexit.register(_close_process_set)
        try:
            from multiprocessing.util import Finalize

            Finalize(None, _close_process_set, exitpriority=10)
        except Exception:  # pragma: no cover
            pass
    return _PROCESS_SET


def _close_process_set() -> None:
    global _PROCESS_SET
    rs, _PROCESS_SET = _PROCESS_SET, None
    if rs is not None:
        try:
            rs.close()
        except Exception as exc:
            logger.error("Closing the process's resident state failed: %r", exc)


def provided_fields(task: ty.Any) -> list[tuple[str, str]]:
    """``(field name, provider name)`` for each provided input of ``task``"""
    from pydra.utils.general import get_fields

    return [
        (f.name, f.provided)
        for f in get_fields(task)
        if getattr(f, "provided", None) is not None
    ]


def _find_provider(job: "Job", name: str) -> Provider:
    submitter = getattr(job, "submitter", None)
    for provider in getattr(submitter, "providers", ()):
        if provider.name == name:
            return provider
    from importlib.metadata import entry_points

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            loaded = ep.load()
            provider = loaded() if isinstance(loaded, type) else loaded
            if provider.name != name:
                raise UnregisteredProvider(
                    f"Entry point {name!r} loads a provider named {provider.name!r}"
                )
            return provider
    raise UnregisteredProvider(
        f"No provider {name!r}: pass it as Submitter(providers=[...]) or register it "
        f"under the {ENTRY_POINT_GROUP!r} entry point group"
    )


def _spec_key(task: ty.Any, field: str, provider: str) -> ResidentKey:
    from pydra.utils.digest import digest

    return ResidentKey("provider", provider, digest(getattr(task, field)))


def residency_digest(job: "Job") -> str | None:
    """The sha256 of the canonical list of the digests of the job's resident keys, in
    field order, or ``None`` for a job with no resident state. It is not part of the
    job's identity"""
    fields = provided_fields(job.task)
    if not fields:
        return None
    digests = [_spec_key(job.task, f, p).digest() for f, p in fields]
    return hashlib.sha256(_canonical(digests).encode()).hexdigest()


def _resident_set_of(job: "Job") -> ResidentSet:
    submitter = getattr(job, "submitter", None)
    getter = getattr(submitter, "resident_set", None)
    if getter is not None and not getattr(submitter, "_worker_copy", False):
        return getter()
    return process_set()


@functools.lru_cache(maxsize=None)
def _entry_point_names() -> frozenset[str]:
    from importlib.metadata import entry_points

    return frozenset(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))


def check_providers(job: "Job") -> None:
    """Raise :class:`UnregisteredProvider` if a provided field of ``job`` names a
    provider that is neither passed to its submitter nor registered; a check of names
    only, so no provider module is imported, made when the job is handed out"""
    check_task(job.task, getattr(job, "submitter", None))


def check_task(task: ty.Any, submitter: ty.Any) -> None:
    """:func:`check_providers` for a task definition and a submitter"""
    names = {p.name for p in getattr(submitter, "providers", ())}
    for _, name in provided_fields(task):
        if name not in names and name not in _entry_point_names():
            raise UnregisteredProvider(
                f"No provider {name!r}: pass it as Submitter(providers=[...]) or "
                f"register it under the {ENTRY_POINT_GROUP!r} entry point group"
            )


@contextmanager
def bound(job: "Job") -> ty.Iterator[dict[str, ty.Any]]:
    """Load (or find) the state of each provided field of ``job`` and yield it by field
    name; the state is busy until the block ends"""
    fields = provided_fields(job.task)
    if not fields:
        yield {}
        return
    resident = _resident_set_of(job)
    values: dict[str, ty.Any] = {}
    with ExitStack() as stack:
        for field, name in fields:
            provider = _find_provider(job, name)
            spec = getattr(job.task, field)
            key = _spec_key(job.task, field, name)
            device = Device()
            values[field] = stack.enter_context(
                resident.use(
                    key,
                    lambda p=provider, s=spec, d=device: p.load(s, d),
                    close=provider.close,
                    footprint=provider.footprint(spec),
                )
            )
        yield values
