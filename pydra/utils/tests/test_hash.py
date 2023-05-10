import re
from hashlib import blake2b
from pathlib import Path

import pytest

from ..hash import Cache, UnhashableError, bytes_repr, hash_object, register_serializer


@pytest.fixture
def hasher():
    yield blake2b(digest_size=16, person=b"pydra-hash")


def join_bytes_repr(obj):
    return b"".join(bytes_repr(obj, Cache({})))


def test_bytes_repr_builtins():
    # Can't beat repr for some
    assert join_bytes_repr(None) == b"None"
    assert join_bytes_repr(Ellipsis) == b"Ellipsis"
    assert join_bytes_repr(True) == b"True"
    assert join_bytes_repr(False) == b"False"
    assert join_bytes_repr(range(1)) == b"range(0, 1)"
    assert join_bytes_repr(range(-1, 10, 2)) == b"range(-1, 10, 2)"
    # String types
    assert join_bytes_repr(b"abc") == b"bytes:3:abc"
    assert join_bytes_repr("abc") == b"str:3:abc"
    # Little-endian, 64-bit signed integer
    assert join_bytes_repr(123) == b"int:\x7b\x00\x00\x00\x00\x00\x00\x00"
    # ASCII string representation of a Python "long" integer
    assert join_bytes_repr(12345678901234567890) == b"long:20:12345678901234567890"
    # Float uses little-endian double-precision format
    assert join_bytes_repr(1.0) == b"float:\x00\x00\x00\x00\x00\x00\xf0?"
    # Complex concatenates two floats
    complex_repr = join_bytes_repr(0.0 + 0j)
    assert complex_repr == b"complex:" + bytes(16)
    # Dicts are sorted by key, and values are hashed
    dict_repr = join_bytes_repr({"b": "c", "a": 0})
    assert re.match(rb"dict:{str:1:a=.{16}str:1:b=.{16}}$", dict_repr)
    # Lists and tuples concatenate hashes of their contents
    list_repr = join_bytes_repr([1, 2, 3])
    assert re.match(rb"list:\(.{48}\)$", list_repr)
    tuple_repr = join_bytes_repr((1, 2, 3))
    assert re.match(rb"tuple:\(.{48}\)$", tuple_repr)
    # Sets sort, hash and concatenate their contents
    set_repr = join_bytes_repr({1, 2, 3})
    assert re.match(rb"set:{.{48}}$", set_repr)
    # Sets sort, hash and concatenate their contents
    fset_repr = join_bytes_repr(frozenset((1, 2, 3)))
    assert re.match(rb"frozenset:{.{48}}$", fset_repr)
    # Slice fields can be anything, so hash contents
    slice_repr = join_bytes_repr(slice(1, 2, 3))
    assert re.match(rb"slice\(.{48}\)$", slice_repr)


@pytest.mark.parametrize(
    "obj,expected",
    [
        ("abc", "bc6289a80ec21621f20dea1907cc8b9a"),
        (b"abc", "29ddaec80d4b3baba945143faa4c9e96"),
        (1, "6dc1db8d4dcdd8def573476cbb90cce0"),
        (12345678901234567890, "2b5ba668c1e8ea4902361b8d81e53074"),
        (1.0, "29492927b2e505840235e15a5be9f79a"),
        ({"b": "c", "a": 0}, "2405cd36f4e4b6318c033f32db289f7d"),
        ([1, 2, 3], "2f8902ff90f63d517bd6f6e6111e15b8"),
        ((1, 2, 3), "054a7b31c29e7875a6f83ff1dcb4841b"),
    ],
)
def test_hash_object_known_values(obj: object, expected: str):
    # Regression test to avoid accidental changes to hash_object
    # We may update this, but it will indicate that users should
    # expect cache directories to be invalidated
    assert hash_object(obj).hex() == expected


def test_pathlike_reprs(tmp_path):
    empty_file = tmp_path / "empty"
    empty_file.touch()
    one_byte = tmp_path / "zero"
    one_byte.write_bytes(b"\x00")
    # Files are raw contents, not tagged
    assert join_bytes_repr(empty_file) == b""
    assert join_bytes_repr(one_byte) == b"\x00"

    # Directories are tagged
    # Use __class__.__name__ to use PosixPath/WindowsPath based on OS
    assert (
        join_bytes_repr(tmp_path)
        == f"{tmp_path.__class__.__name__}:{tmp_path}".encode()
    )

    with pytest.raises(FileNotFoundError):
        join_bytes_repr(Path("/does/not/exist"))


def test_hash_pathlikes(tmp_path, hasher):
    empty_file = tmp_path / "empty"
    empty_file.touch()
    one_byte = tmp_path / "zero"
    one_byte.write_bytes(b"\x00")
    assert hash_object(empty_file).hex() == "b63a06566ea1caa15da1ec060066177a"
    assert hash_object(one_byte).hex() == "ebd393c59b8d3ca33426875af4bd0f22"

    # Actually hashing contents, not filename
    empty_file2 = tmp_path / "empty2"
    empty_file2.touch()
    assert hash_object(empty_file2).hex() == "b63a06566ea1caa15da1ec060066177a"

    # Hashing directories is just a path
    hasher.update(f"{tmp_path.__class__.__name__}:{tmp_path}".encode())
    assert hash_object(tmp_path) == hasher.digest()

    with pytest.raises(UnhashableError):
        hash_object(Path("/does/not/exist"))


def test_bytes_repr_custom_obj():
    class MyClass:
        def __init__(self, x):
            self.x = x

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16}}", obj_repr)


def test_recursive_object():
    a = []
    b = [a]
    a.append(b)

    obj_repr = join_bytes_repr(a)
    assert re.match(rb"list:\(.{16}\)$", obj_repr)

    # Objects are structurally equal, but not the same object
    assert hash_object(a) == hash_object(b)


def test_multi_object():
    # Including the same object multiple times in a list
    # should produce the same hash each time it is encountered
    set1 = {1, 2, 3}
    set2 = {4, 5, 6}
    listA = [set1, set2, set1]
    listB = [set1, set2, set2]

    reprA = join_bytes_repr(listA)
    reprB = join_bytes_repr(listB)
    assert re.match(rb"list:\((.{16})(.{16})\1\)$", reprA)
    assert re.match(rb"list:\((.{16})(.{16})\2\)$", reprB)


def test_magic_method():
    class MyClass:
        def __init__(self, x):
            self.x = x

        def __bytes_repr__(self, cache):
            yield b"x"

    assert join_bytes_repr(MyClass(1)) == b"x"
