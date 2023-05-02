"""Generic object hashing dispatch"""
import struct
from collections.abc import Iterator
from functools import singledispatch
from hashlib import blake2b
from typing import NewType, Sequence, Type, Callable, TypeVar

__all__ = (
    "hash_object",
    "hash_single",
    "register_serializer",
    "Hash",
    "Cache",
    "bytes_repr_mapping_contents",
    "bytes_repr_sequence_contents",
)

T = TypeVar("T")

Hash = NewType("Hash", bytes)
Cache = NewType("Cache", dict[int, Hash])


def hash_object(obj: object) -> Hash:
    """Hash an object

    Constructs a byte string that uniquely identifies the object,
    and returns the hash of that string.

    Base Python types are implemented, including recursive lists and
    dicts. Custom types can be registered with :func:`register_serializer`.
    """
    return hash_single(obj, Cache({}))


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
        cache[objid] = Hash(h.digest())
    return cache[objid]


def register_serializer(
    cls: Type[T], generator: Callable[[T, Cache], Iterator[bytes]]
) -> None:
    """Register a custom serializer for a type

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
    bytes_repr.register(cls)(generator)


@singledispatch
def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{{".encode()
    yield from bytes_repr_mapping_contents(obj.__dict__, cache)
    yield b"}"


@bytes_repr.register
def bytes_repr_bytes(obj: bytes, cache: Cache) -> Iterator[bytes]:
    yield f"bytes:{len(obj)}:".encode()
    yield obj


@bytes_repr.register
def bytes_repr_str(obj: str, cache: Cache) -> Iterator[bytes]:
    val = obj.encode()
    yield f"str:{len(val)}:".encode()
    yield val


@bytes_repr.register
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


@bytes_repr.register
def bytes_repr_float(obj: float, cache: Cache) -> Iterator[bytes]:
    yield b"float:"
    yield struct.pack("<d", obj)


@bytes_repr.register
def bytes_repr_dict(obj: dict, cache: Cache) -> Iterator[bytes]:
    if cache is None:
        cache = {}
    yield b"dict:{"
    yield from bytes_repr_mapping_contents(obj, cache)
    yield b"}"


@bytes_repr.register(list)
@bytes_repr.register(tuple)
def bytes_repr_seq(obj, cache: Cache) -> Iterator[bytes]:
    if cache is None:
        cache = {}
    yield f"{obj.__class__.__name__}:(".encode()
    yield from bytes_repr_sequence_contents(obj, cache)
    yield b")"


@bytes_repr.register
def bytes_repr_set(obj: set, cache: Cache) -> Iterator[bytes]:
    yield b"set:{"
    yield from bytes_repr_sequence_contents(sorted(obj), cache)
    yield b"}"


def bytes_repr_mapping_contents(mapping: dict, cache: Cache) -> Iterator[bytes]:
    for key in sorted(mapping):
        yield from bytes_repr(key, cache)
        yield b"="
        yield bytes(hash_single(mapping[key], cache))


def bytes_repr_sequence_contents(seq: Sequence, cache: Cache) -> Iterator[bytes]:
    for val in seq:
        yield bytes(hash_single(val, cache))
