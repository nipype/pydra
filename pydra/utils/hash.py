"""Generic object hashing dispatch"""
import struct
from collections.abc import Iterator
from functools import singledispatch
from hashlib import blake2b
from typing import NewType, Union, Sequence

Hash = NewType("Hash", bytes)


def hash_object(obj: object) -> Hash:
    h = blake2b(digest_size=16, person=b'pydra-hash')
    for chunk in bytes_repr(obj):
        h.update(chunk)
    return Hash(h.digest())



@singledispatch
def bytes_repr(obj: object, seen: Union[set, None]=None) -> Iterator[bytes]:
    if seen is None:
        seen = set()
    cls = obj.__class__
    yield f"{cls.__module__}.{cls.__name__}:{{".encode()
    yield from bytes_repr_mapping_contents(obj.__dict__, seen)
    yield b"}"


@bytes_repr.register
def bytes_repr_bytes(obj: bytes, seen: Union[set, None]=None) -> Iterator[bytes]:
    yield f"bytes:{len(obj)}:".encode()
    yield obj


@bytes_repr.register
def bytes_repr_str(obj: str, seen: Union[set, None]=None) -> Iterator[bytes]:
    val = obj.encode()
    yield f"str:{len(val)}:".encode()
    yield val


@bytes_repr.register
def bytes_repr_int(obj: int, seen: Union[set, None]=None) -> Iterator[bytes]:
    try:
        # Up to 64-bit ints
        val = struct.pack('<q', obj)
        yield b"int:"
    except struct.error:
        # Big ints (old python "long")
        val = str(obj).encode()
        yield f"long:{len(val)}:".encode()
    yield val


@bytes_repr.register
def bytes_repr_float(obj: float, seen: Union[set, None]=None) -> Iterator[bytes]:
    yield b"float:"
    yield struct.pack('<d', obj)


@bytes_repr.register
def bytes_repr_dict(obj: dict, seen: Union[set, None]=None) -> Iterator[bytes]:
    if seen is None:
        seen = set()
    yield b"dict:{"
    yield from bytes_repr_mapping_contents(obj, seen)
    yield b"}"


@bytes_repr.register(list)
@bytes_repr.register(tuple)
def bytes_repr_seq(obj, seen: Union[set, None]=None) -> Iterator[bytes]:
    if seen is None:
        seen = set()
    yield f"{obj.__class__.__name__}:(".encode()
    yield from bytes_repr_sequence_contents(obj, seen)
    yield b")"


@bytes_repr.register
def bytes_repr_set(obj: set, seen: Union[set, None]=None) -> Iterator[bytes]:
    objid = id(obj)
    if objid in (seen := set() if seen is None else seen):
        # Unlikely to get a seen set, but sorting breaks contents
        yield b"set:{...}"
        return
    seen.add(objid)

    yield b"set:{"
    yield from bytes_repr_sequence_contents(sorted(obj), seen)
    yield b"}"


def bytes_repr_mapping_contents(mapping: dict, seen: set) -> Iterator[bytes]:
    objid = id(mapping)
    if objid in seen:
        yield b"..."
        return
    seen.add(objid)

    for key in sorted(mapping):
        yield from bytes_repr(key, seen)
        yield b"="
        yield from bytes_repr(mapping[key], seen)
        yield b";"


def bytes_repr_sequence_contents(seq: Sequence, seen: set) -> Iterator[bytes]:
    objid = id(seq)
    if objid in seen:
        yield b"..."
        return
    seen.add(objid)

    for val in seq:
        yield from bytes_repr(val, seen)
        yield b";"
