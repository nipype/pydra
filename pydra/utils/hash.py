"""Generic object hashing dispatch"""
import os

# import stat
import struct
import typing as ty
from collections.abc import Mapping
from functools import singledispatch
from hashlib import blake2b
import logging

# from pathlib import Path
from typing import (
    Dict,
    Iterator,
    NewType,
    Sequence,
    Set,
)
import attrs.exceptions

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
Cache = NewType("Cache", Dict[int, Hash])


class UnhashableError(ValueError):
    """Error for objects that cannot be hashed"""


def hash_function(obj):
    """Generate hash of object."""
    return hash_object(obj).hex()


def hash_object(obj: object) -> Hash:
    """Hash an object

    Constructs a byte string that uniquely identifies the object,
    and returns the hash of that string.

    Base Python types are implemented, including recursive lists and
    dicts. Custom types can be registered with :func:`register_serializer`.
    """
    try:
        return hash_single(obj, Cache({}))
    except Exception as e:
        raise UnhashableError(f"Cannot hash object {obj!r}") from e


def hash_single(obj: object, cache: Cache) -> Hash:
    """Single object-scoped hash

    Uses a local cache to prevent infinite recursion. This cache is unsafe
    to reuse across multiple objects, so this function should not be used directly.
    """
    objid = id(obj)
    if objid not in cache:
        # Handle recursion by putting a dummy value in the cache
        cache[objid] = Hash(b"\x00")
        h = blake2b(digest_size=16, person=b"pydra-hash")
        for chunk in bytes_repr(obj, cache):
            h.update(chunk)
        hsh = cache[objid] = Hash(h.digest())
        logger.debug("Hash of %s object is %s", obj, hsh)
    return cache[objid]


@runtime_checkable
class HasBytesRepr(Protocol):
    def __bytes_repr__(self, cache: Cache) -> Iterator[bytes]:
        ...  # pragma: no cover


@singledispatch
def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
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
        >>> generator = bytes_repr_mapping_contents({"a": 1, "b": 2}, Cache({}))
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
        >>> generator = bytes_repr_sequence_contents([1, 2], Cache({}))
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


# class MtimeCachingHash:
#     """Hashing object that stores a cache of hash values for PathLikes

#     The cache only stores values for PathLikes pointing to existing files,
#     and the mtime is checked to validate the cache. If the mtime differs,
#     the old hash is discarded and a new mtime-tagged hash is stored.

#     The cache can grow without bound; we may want to consider using an LRU
#     cache.
#     """

#     def __init__(self) -> None:
#         self.cache: ty.Dict[os.PathLike, ty.Tuple[float, Hash]] = {}

#     def __call__(self, obj: object) -> Hash:
#         if isinstance(obj, os.PathLike):
#             path = Path(obj)
#             try:
#                 stat_res = path.stat()
#                 mode, mtime = stat_res.st_mode, stat_res.st_mtime
#             except FileNotFoundError:
#                 # Only attempt to cache existing files
#                 pass
#             else:
#                 if stat.S_ISREG(mode) and obj in self.cache:
#                     # Cache (and hash) the actual object, as different pathlikes will have
#                     # different serializations
#                     save_mtime, save_hash = self.cache[obj]
#                     if mtime == save_mtime:
#                         return save_hash
#                     new_hash = hash_object(obj)
#                     self.cache[obj] = (mtime, new_hash)
#                     return new_hash
#         return hash_object(obj)
