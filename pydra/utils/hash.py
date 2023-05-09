"""Generic object hashing dispatch"""
import struct
from collections.abc import Mapping
from functools import singledispatch
from hashlib import blake2b
from typing import Dict, Iterator, NewType, Sequence, Set

__all__ = (
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


@singledispatch
def bytes_repr(obj: object, cache: Cache) -> Iterator[bytes]:
    if hasattr(obj, "__bytes_repr__"):
        yield from obj.__bytes_repr__(cache)
    else:
        cls = obj.__class__
        yield f"{cls.__module__}.{cls.__name__}:{{".encode()
        yield from bytes_repr_mapping_contents(obj.__dict__, cache)
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
def bytes_repr_none(obj: None, cache: Cache) -> Iterator[bytes]:
    yield b"None"


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
def bytes_repr_dict(obj: dict, cache: Cache) -> Iterator[bytes]:
    yield b"dict:{"
    yield from bytes_repr_mapping_contents(obj, cache)
    yield b"}"


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
