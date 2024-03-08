"""Generic object hashing dispatch"""

import os
import struct
from datetime import datetime
import typing as ty
from pathlib import Path
from collections.abc import Mapping
from functools import singledispatch
from hashlib import blake2b
import logging
from typing import (
    Dict,
    Iterator,
    NewType,
    Sequence,
    Set,
)
from filelock import SoftFileLock
import platformdirs
import attrs.exceptions
from fileformats.core import FileSet
from pydra._version import __version__

logger = logging.getLogger("pydra")

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol  # type: ignore

try:
    from typing import runtime_checkable
except ImportError:
    from typing_extensions import runtime_checkable  # type: ignore


try:
    import numpy
except ImportError:
    HAVE_NUMPY = False
else:
    HAVE_NUMPY = True

__all__ = (
    "hash_function",
    "hash_object",
    "hash_single",
    "register_serializer",
    "Hash",
    "Cache",
    "bytes_repr_mapping_contents",
    "bytes_repr_sequence_contents",
)

Hash = NewType("Hash", bytes)
CacheKey = NewType("CacheKey", ty.Tuple[ty.Hashable, ty.Hashable])


def location_converter(path: ty.Union[Path, str, None]) -> Path:
    if path is None:
        path = PersistentCache.location_default()
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    return path


@attrs.define
class PersistentCache:
    """Persistent cache in which to store computationally expensive hashes between nodes
    and workflow/task runs. It does this in via the `get_or_calculate_hash` method, which
    takes a locally unique key (e.g. file-system path + mtime) and a function to
    calculate the hash if it isn't present in the persistent store.

    The locally unique key is hashed (cheaply) using hashlib cryptography and this
    "local hash" is use to name the entry of the (potentially expensive) hash of the
    object itself (e.g. the contents of a file). This entry is saved as a text file
    within a user-specific cache directory (see `platformdirs.user_cache_dir`), with
    the name of the file being the "local hash" of the key and the contents of the
    file being the "globally unique hash" of the object itself.

    Parameters
    ----------
    location: Path
        the directory in which to store the hashes cache
    """

    location: Path = attrs.field(converter=location_converter)  # type: ignore[misc]
    cleanup_period: int = attrs.field()
    _hashes: ty.Dict[CacheKey, Hash] = attrs.field(factory=dict)

    # Set the location of the persistent hash cache
    LOCATION_ENV_VAR = "PYDRA_HASH_CACHE"
    CLEANUP_ENV_VAR = "PYDRA_HASH_CACHE_CLEANUP_PERIOD"

    @classmethod
    def location_default(cls):
        try:
            location = os.environ[cls.LOCATION_ENV_VAR]
        except KeyError:
            location = platformdirs.user_cache_dir(
                appname="pydra",
                appauthor="nipype",
                version=__version__,
            )
        return location

    # the default needs to be an instance method
    @location.default
    def _location_default(self):
        return self.location_default()

    @location.validator
    def location_validator(self, _, location):
        if not os.path.isdir(location):
            raise ValueError(
                f"Persistent cache location '{location}' is not a directory"
            )

    @cleanup_period.default
    def cleanup_period_default(self):
        return int(os.environ.get(self.CLEANUP_ENV_VAR, 30))

    def get_or_calculate_hash(self, key: CacheKey, calculate_hash: ty.Callable) -> Hash:
        """Check whether key is present in the persistent cache store and return it if so.
        Otherwise use `calculate_hash` to generate the hash and save it in the persistent
        store.

        Parameters
        ----------
        key : CacheKey
            locally unique key (e.g. to the host) used to lookup the corresponding hash
            in the persistent store
        calculate_hash : ty.Callable
            function to calculate the hash if it isn't present in the persistent store

        Returns
        -------
        Hash
            the hash corresponding to the key, which is either retrieved from the persistent
            store or calculated using `calculate_hash` if not present
        """
        try:
            return self._hashes[key]
        except KeyError:
            pass
        key_path = self.location / blake2b(str(key).encode()).hexdigest()
        with SoftFileLock(key_path.with_suffix(".lock")):
            if key_path.exists():
                return Hash(key_path.read_bytes())
            hsh = calculate_hash()
            key_path.write_bytes(hsh)
            self._hashes[key] = Hash(hsh)
        return Hash(hsh)

    def clean_up(self):
        """Cleans up old hash caches that haven't been accessed in the last 30 days"""
        now = datetime.now()
        for path in self.location.iterdir():
            if path.name.endswith(".lock"):
                continue
            days = (now - datetime.fromtimestamp(path.lstat().st_atime)).days
            if days > self.cleanup_period:
                path.unlink()

    @classmethod
    def from_path(
        cls, path: ty.Union[Path, str, "PersistentCache", None]
    ) -> "PersistentCache":
        if isinstance(path, PersistentCache):
            return path
        return PersistentCache(path)


@attrs.define
class Cache:
    """Cache for hashing objects, used to avoid infinite recursion caused by circular
    references between objects, and to store hashes of objects that have already been
    hashed to avoid recomputation.

    This concept is extended to persistent caching of hashes for certain object types,
    for which calculating the hash is a potentially expensive operation (e.g.
    File/Directory types). For these classes the `bytes_repr` override function yields a
    "locally unique cache key" (e.g. file-system path + mtime) as the first item of its
    iterator.
    """

    persistent: ty.Optional[PersistentCache] = attrs.field(
        default=None,
        converter=PersistentCache.from_path,  # type: ignore[misc]
    )
    _hashes: ty.Dict[int, Hash] = attrs.field(factory=dict)

    def __getitem__(self, object_id: int) -> Hash:
        return self._hashes[object_id]

    def __setitem__(self, object_id: int, hsh: Hash):
        self._hashes[object_id] = hsh

    def __contains__(self, object_id):
        return object_id in self._hashes


class UnhashableError(ValueError):
    """Error for objects that cannot be hashed"""


def hash_function(obj, **kwargs):
    """Generate hash of object."""
    return hash_object(obj, **kwargs).hex()


def hash_object(
    obj: object,
    cache: ty.Optional[Cache] = None,
    persistent_cache: ty.Union[PersistentCache, Path, None] = None,
) -> Hash:
    """Hash an object

    Constructs a byte string that uniquely identifies the object,
    and returns the hash of that string.

    Base Python types are implemented, including recursive lists and
    dicts. Custom types can be registered with :func:`register_serializer`.
    """
    if cache is None:
        cache = Cache(persistent=persistent_cache)
    try:
        return hash_single(obj, cache)
    except Exception as e:
        raise UnhashableError(f"Cannot hash object {obj!r} due to '{e}'") from e


def hash_single(obj: object, cache: Cache) -> Hash:
    """Single object-scoped hash

    Uses a local cache to prevent infinite recursion. This cache is unsafe
    to reuse across multiple objects, so this function should not be used directly.
    """
    objid = id(obj)
    if objid not in cache:
        # Handle recursion by putting a dummy value in the cache
        cache[objid] = Hash(b"\x00")
        bytes_it = bytes_repr(obj, cache)
        # Pop first element from the bytes_repr iterator and check whether it is a
        # "local cache key" (e.g. file-system path + mtime tuple) or the first bytes
        # chunk

        def calc_hash(first: ty.Optional[bytes] = None) -> Hash:
            """
            Calculate the hash of the object

            Parameters
            ----------
            first : ty.Optional[bytes]
                the first bytes chunk from the bytes_repr iterator, passed if the first
                chunk wasn't a local cache key
            """
            h = blake2b(digest_size=16, person=b"pydra-hash")
            # We want to use the first chunk that was popped to check for a cache-key
            # if present
            if first is not None:
                h.update(first)
            for chunk in bytes_it:  # Note that `bytes_it` is in outer scope
                h.update(chunk)
            return Hash(h.digest())

        # Read the first item of the bytes_repr iterator and check to see whether it yields
        # a "cache-key" tuple instead of a bytes chunk for the type of the object to be cached
        # (e.g. file-system path + mtime for fileformats.core.FileSet objects). If it
        # does, use that key to check the persistent cache for a precomputed hash and
        # return it if it is, otherwise calculate the hash and store it in the persistent
        # cache with that hash of that key (not to be confused with the hash of the
        # object that is saved/retrieved).
        first = next(bytes_it)
        if isinstance(first, tuple):
            tp = type(obj)
            key = (
                tp.__module__,
                tp.__name__,
            ) + first
            hsh = cache.persistent.get_or_calculate_hash(key, calc_hash)
        else:
            # If the first item is a bytes chunk (i.e. the object type doesn't have an
            # associated 'cache-key'), then simply calculate the hash of the object,
            # passing the first chunk to the `calc_hash` function so it can be included
            # in the hash calculation
            hsh = calc_hash(first=first)
        logger.debug("Hash of %s object is %s", obj, hsh)
        cache[objid] = hsh
    return cache[objid]


@runtime_checkable
class HasBytesRepr(Protocol):
    def __bytes_repr__(self, cache: Cache) -> Iterator[bytes]:
        pass  # pragma: no cover


@singledispatch
def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
    """Default implementation of hashing for generic objects. Single dispatch is used
    to provide hooks for class-specific implementations

    Parameters
    ----------
    obj: object
        the object to hash
    cache : Cache
        a dictionary object used to store a cache of previously cached objects to
        handle circular object references

    Yields
    -------
    bytes
        unique representation of the object in a series of bytes
    """
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{{".encode()
    dct: Dict[str, ty.Any]
    if attrs.has(type(obj)):
        # Drop any attributes that aren't used in comparisons by default
        dct = attrs.asdict(obj, recurse=False, filter=lambda a, _: bool(a.eq))
    elif hasattr(obj, "__slots__"):
        dct = {attr: getattr(obj, attr) for attr in obj.__slots__}
    else:
        dct = obj.__dict__
    yield from bytes_repr_mapping_contents(dct, cache)
    yield b"}"


register_serializer = bytes_repr.register
register_serializer.__doc__ = """Register a custom serializer for a type

The generator function should yield byte strings that will be hashed
to produce the final hash. A recommended convention is to yield a
qualified type prefix (e.g. ``f"{module}.{class}"``),
followed by a colon, followed by the serialized value.

If serializing an iterable, an open and close bracket may be yielded
to identify the start and end of the iterable.

Consider using :func:`bytes_repr_mapping_contents` and
:func:`bytes_repr_sequence_contents` to serialize the contents of a mapping
or sequence. These do not include the prefix or brackets, so they can be
reused as part of a custom serializer.

As an example, the following example is the default serializer for user-defined
classes:

.. code-block:: python

    @register_serializer
    def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
        cls = obj.__class__
        yield f"{cls.__module__}.{cls.__name__}:{{".encode()
        yield from bytes_repr_mapping_contents(obj.__dict__, cache)
        yield b"}"

Serializers must accept a ``cache`` argument, which is a dictionary that
permits caching of hashes for recursive objects. If the hash of sub-objects
is used to create an object serialization, the :func:`hash_single` function
should be called with the same cache object.
"""


@register_serializer
def bytes_repr_dunder(obj: HasBytesRepr, cache: Cache) -> Iterator[bytes]:
    yield from obj.__bytes_repr__(cache)


@register_serializer(type(None))
@register_serializer(type(Ellipsis))
@register_serializer(bool)
@register_serializer(range)
def bytes_repr_builtin_repr(
    obj: object,
    cache: Cache,
) -> Iterator[bytes]:
    yield repr(obj).encode()


@register_serializer
def bytes_repr_slice(obj: slice, cache: Cache) -> Iterator[bytes]:
    yield b"slice("
    yield from bytes_repr_sequence_contents((obj.start, obj.stop, obj.step), cache)
    yield b")"


@register_serializer
def bytes_repr_pathlike(obj: os.PathLike, cache: Cache) -> Iterator[bytes]:
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{os.fspath(obj)}".encode()


@register_serializer
def bytes_repr_bytes(obj: bytes, cache: Cache) -> Iterator[bytes]:
    yield f"bytes:{len(obj)}:".encode()
    yield obj


@register_serializer
def bytes_repr_str(obj: str, cache: Cache) -> Iterator[bytes]:
    val = obj.encode()
    yield f"str:{len(val)}:".encode()
    yield val


@register_serializer
def bytes_repr_int(obj: int, cache: Cache) -> Iterator[bytes]:
    try:
        # Up to 64-bit ints
        val = struct.pack("<q", obj)
        yield b"int:"
    except struct.error:
        # Big ints (old python "long")
        val = str(obj).encode()
        yield f"long:{len(val)}:".encode()
    yield val


@register_serializer
def bytes_repr_float(obj: float, cache: Cache) -> Iterator[bytes]:
    yield b"float:"
    yield struct.pack("<d", obj)


@register_serializer
def bytes_repr_complex(obj: complex, cache: Cache) -> Iterator[bytes]:
    yield b"complex:"
    yield struct.pack("<dd", obj.real, obj.imag)


@register_serializer
def bytes_repr_dict(obj: dict, cache: Cache) -> Iterator[bytes]:
    yield b"dict:{"
    yield from bytes_repr_mapping_contents(obj, cache)
    yield b"}"


@register_serializer(ty._GenericAlias)
@register_serializer(ty._SpecialForm)
@register_serializer(type)
def bytes_repr_type(klass: type, cache: Cache) -> Iterator[bytes]:
    def type_name(tp):
        try:
            name = tp.__name__
        except AttributeError:
            name = tp._name
        return name

    yield b"type:("
    origin = ty.get_origin(klass)
    if origin:
        yield f"{origin.__module__}.{type_name(origin)}[".encode()
        for arg in ty.get_args(klass):
            if isinstance(
                arg, list
            ):  # sometimes (e.g. Callable) the args of a type is a list
                yield b"["
                yield from (b for t in arg for b in bytes_repr_type(t, cache))
                yield b"]"
            else:
                yield from bytes_repr_type(arg, cache)
        yield b"]"
    else:
        yield f"{klass.__module__}.{type_name(klass)}".encode()
    yield b")"


@register_serializer(FileSet)
def bytes_repr_fileset(
    fileset: FileSet, cache: Cache
) -> Iterator[ty.Union[CacheKey, bytes]]:
    fspaths = sorted(fileset.fspaths)
    yield CacheKey(
        tuple(repr(p) for p in fspaths)  # type: ignore[arg-type]
        + tuple(p.lstat().st_mtime_ns for p in fspaths)
    )
    yield from fileset.__bytes_repr__(cache)


@register_serializer(list)
@register_serializer(tuple)
def bytes_repr_seq(obj: Sequence, cache: Cache) -> Iterator[bytes]:
    yield f"{obj.__class__.__name__}:(".encode()
    yield from bytes_repr_sequence_contents(obj, cache)
    yield b")"


@register_serializer(set)
@register_serializer(frozenset)
def bytes_repr_set(obj: Set, cache: Cache) -> Iterator[bytes]:
    yield f"{obj.__class__.__name__}:{{".encode()
    yield from bytes_repr_sequence_contents(sorted(obj), cache)
    yield b"}"


def bytes_repr_mapping_contents(mapping: Mapping, cache: Cache) -> Iterator[bytes]:
    """Serialize the contents of a mapping

    Concatenates byte-serialized keys and hashed values.

    .. code-block:: python

        >>> from pydra.utils.hash import bytes_repr_mapping_contents, Cache
        >>> generator = bytes_repr_mapping_contents({"a": 1, "b": 2}, Cache())
        >>> b''.join(generator)
        b'str:1:a=...str:1:b=...'
    """
    for key in sorted(mapping):
        yield from bytes_repr(key, cache)
        yield b"="
        yield bytes(hash_single(mapping[key], cache))


def bytes_repr_sequence_contents(seq: Sequence, cache: Cache) -> Iterator[bytes]:
    """Serialize the contents of a sequence

    Concatenates hashed values.

    .. code-block:: python

        >>> from pydra.utils.hash import bytes_repr_sequence_contents, Cache
        >>> generator = bytes_repr_sequence_contents([1, 2], Cache())
        >>> list(generator)
        [b'\x6d...', b'\xa3...']
    """
    for val in seq:
        yield bytes(hash_single(val, cache))


if HAVE_NUMPY:

    @register_serializer(numpy.generic)
    @register_serializer(numpy.ndarray)
    def bytes_repr_numpy(obj: numpy.ndarray, cache: Cache) -> Iterator[bytes]:
        yield f"{obj.__class__.__module__}{obj.__class__.__name__}:{obj.size}:".encode()
        if obj.dtype == "object":
            yield from bytes_repr_sequence_contents(iter(obj.ravel()), cache)
        else:
            yield obj.tobytes(order="C")


NUMPY_CHUNK_LEN = 8192
