import re
import os
from hashlib import blake2b
from pathlib import Path
import time
from unittest import mock
import attrs
import pytest
import typing as ty
from fileformats.application import Zip, Json
from fileformats.text import TextFile
from ..hash import (
    Cache,
    UnhashableError,
    bytes_repr,
    hash_object,
    register_serializer,
    PersistentCache,
)


@pytest.fixture
def hasher():
    yield blake2b(digest_size=16, person=b"pydra-hash")


def join_bytes_repr(obj):
    return b"".join(bytes_repr(obj, Cache()))


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
    cls = tmp_path.__class__
    prefix = f"{cls.__module__}.{cls.__name__}"
    # Directory
    assert join_bytes_repr(tmp_path) == f"{prefix}:{tmp_path}".encode()
    # Non-existent file
    empty_file = tmp_path / "empty"
    assert join_bytes_repr(empty_file) == f"{prefix}:{empty_file}".encode()
    # Existent file
    empty_file.touch()
    assert join_bytes_repr(empty_file) == f"{prefix}:{empty_file}".encode()

    class MyPathLike:
        def __fspath__(self):
            return "/tmp"

    prefix = f"{__name__}.MyPathLike"
    assert join_bytes_repr(MyPathLike()) == f"{prefix}:/tmp".encode()


def test_hash_pathlikes(tmp_path, hasher):
    cls = tmp_path.__class__
    prefix = f"{cls.__module__}.{cls.__name__}"

    # Directory
    h = hasher.copy()
    h.update(f"{prefix}:{tmp_path}".encode())
    assert hash_object(tmp_path) == h.digest()

    # Non-existent file
    empty_file = tmp_path / "empty"
    h = hasher.copy()
    h.update(f"{prefix}:{empty_file}".encode())
    assert hash_object(empty_file) == h.digest()

    # Existent file
    empty_file.touch()
    assert hash_object(empty_file) == h.digest()

    class MyPathLike:
        def __fspath__(self):
            return "/tmp"

    prefix = f"{__name__}.MyPathLike"
    h = hasher.copy()
    h.update(f"{prefix}:/tmp".encode())
    assert hash_object(MyPathLike()) == h.digest()


def test_bytes_repr_custom_obj():
    class MyClass:
        def __init__(self, x):
            self.x = x

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16}}", obj_repr)


def test_bytes_repr_slots_obj():
    class MyClass:
        __slots__ = ("x",)

        def __init__(self, x):
            self.x = x

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16}}", obj_repr)


def test_bytes_repr_attrs_slots():
    @attrs.define
    class MyClass:
        x: int

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16}}", obj_repr)


def test_bytes_repr_attrs_no_slots():
    @attrs.define(slots=False)
    class MyClass:
        x: int

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16}}", obj_repr)


def test_bytes_repr_type1():
    obj_repr = join_bytes_repr(Path)
    assert obj_repr == b"type:(pathlib.Path)"


def test_bytes_repr_type1a():
    obj_repr = join_bytes_repr(Zip[Json])
    assert obj_repr == rb"type:(fileformats.application.archive.Json__Zip)"


def test_bytes_repr_type2():
    T = ty.TypeVar("T")

    class MyClass(ty.Generic[T]):
        pass

    obj_repr = join_bytes_repr(MyClass[int])
    assert (
        obj_repr == b"type:(pydra.utils.tests.test_hash.MyClass[type:(builtins.int)])"
    )


def test_bytes_special_form1():
    obj_repr = join_bytes_repr(ty.Union[int, float])
    assert obj_repr == b"type:(typing.Union[type:(builtins.int)type:(builtins.float)])"


def test_bytes_special_form2():
    obj_repr = join_bytes_repr(ty.Any)
    assert re.match(rb"type:\(typing.Any\)", obj_repr)


def test_bytes_special_form3():
    obj_repr = join_bytes_repr(ty.Optional[Path])
    assert (
        obj_repr == b"type:(typing.Union[type:(pathlib.Path)type:(builtins.NoneType)])"
    )


def test_bytes_special_form4():
    obj_repr = join_bytes_repr(ty.Type[Path])
    assert obj_repr == b"type:(builtins.type[type:(pathlib.Path)])"


def test_bytes_special_form5():
    obj_repr = join_bytes_repr(ty.Callable[[Path, int], ty.Tuple[float, str]])
    assert obj_repr == (
        b"type:(collections.abc.Callable[[type:(pathlib.Path)type:(builtins.int)]"
        b"type:(builtins.tuple[type:(builtins.float)type:(builtins.str)])])"
    )


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


def test_registration():
    # WARNING: This test appends to a registry that cannot be restored
    # to previous state.
    class MyClass:
        def __init__(self, x):
            self.x = x

    @register_serializer
    def _(obj: MyClass, cache: Cache):
        yield b"x"

    assert join_bytes_repr(MyClass(1)) == b"x"


def test_registration_conflict():
    # Verify the order of precedence: class/superclass registration, __bytes_repr__, protocols
    #
    # WARNING: This test appends to a registry that cannot be restored
    # to previous state.
    class MyClass:
        def __init__(self, x):
            self.x = x

        def __fspath__(self):
            return "pathlike"

    assert join_bytes_repr(MyClass(1)) == f"{__name__}.MyClass:pathlike".encode()

    class MyNewClass(MyClass):
        def __bytes_repr__(self, cache: Cache):
            yield b"bytes_repr"

    assert join_bytes_repr(MyNewClass(1)) == b"bytes_repr"

    @register_serializer
    def _(obj: MyClass, cache: Cache):
        yield b"serializer"

    assert join_bytes_repr(MyClass(1)) == b"serializer"

    register_serializer(MyNewClass, _)

    assert join_bytes_repr(MyNewClass(1)) == b"serializer"


@pytest.fixture
def cache_path(tmp_path):
    cache_path = tmp_path / "hash-cache"
    cache_path.mkdir()
    return cache_path


@pytest.fixture
def text_file(tmp_path):
    text_file_path = tmp_path / "text-file.txt"
    text_file_path.write_text("foo")
    return TextFile(text_file_path)


def test_persistent_hash_cache(cache_path, text_file):
    """
    Test the persistent hash cache with a text file

    The cache is used to store the hash of the text file, and the hash is
    retrieved from the cache when the file is unchanged.
    """
    # Test hash is stable between calls
    hsh = hash_object(text_file, persistent_cache=cache_path)
    assert hsh == hash_object(text_file, persistent_cache=cache_path)

    # Test that cached hash has been used by explicitly modifying it and seeing that the
    # hash is the same as the modified hash
    cache_files = list(cache_path.iterdir())
    assert len(cache_files) == 1
    modified_hash = "modified".encode()
    cache_files[0].write_bytes(modified_hash)
    assert hash_object(text_file, persistent_cache=cache_path) == modified_hash

    # Test that changes to the text file result in new hash
    time.sleep(2)  # Need to ensure that the mtimes will be different
    text_file.fspath.write_text("bar")
    assert hash_object(text_file, persistent_cache=cache_path) != modified_hash
    assert len(list(cache_path.iterdir())) == 2


def test_persistent_hash_cache_cleanup1(cache_path, text_file):
    """
    Test the persistent hash is cleaned up after use if the periods between cleanups
    is greater than the environment variable PYDRA_HASH_CACHE_CLEANUP_PERIOD
    """
    with mock.patch.dict(
        os.environ,
        {
            "PYDRA_HASH_CACHE": str(cache_path),
            "PYDRA_HASH_CACHE_CLEANUP_PERIOD": "-100",
        },
    ):
        persistent_cache = PersistentCache()
    hash_object(text_file, persistent_cache=persistent_cache)
    assert len(list(cache_path.iterdir())) == 1
    persistent_cache.clean_up()
    assert len(list(cache_path.iterdir())) == 0


def test_persistent_hash_cache_cleanup2(cache_path, text_file):
    """
    Test the persistent hash is cleaned up after use if the periods between cleanups
    is greater than the explicitly provided cleanup_period
    """
    persistent_cache = PersistentCache(cache_path, cleanup_period=-100)
    hash_object(text_file, persistent_cache=persistent_cache)
    assert len(list(cache_path.iterdir())) == 1
    time.sleep(2)
    persistent_cache.clean_up()
    assert len(list(cache_path.iterdir())) == 0


def test_persistent_hash_cache_not_dir(text_file):
    """
    Test that an error is raised if the provided cache path is not a directory
    """
    with pytest.raises(ValueError, match="is not a directory"):
        PersistentCache(text_file.fspath)
