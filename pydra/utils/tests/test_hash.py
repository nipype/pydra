import re
import os
import sys
from hashlib import blake2b
from pathlib import Path
import time
from unittest import mock
import attrs
import pytest
import typing as ty
from fileformats.application import Zip, Json
from fileformats.text import TextFile
from pydra.utils.hash import (
    Cache,
    bytes_repr,
    hash_object,
    register_serializer,
    PersistentCache,
)
import shutil
import random
from fileformats.generic import Directory, File
from pydra.utils.hash import hash_function
from pydra.utils.tests.utils import Concatenate


def test_hash_file(tmpdir):
    outdir = Path(tmpdir)
    with open(outdir / "test.file", "w") as fp:
        fp.write("test")
    assert (
        hash_function(File(outdir / "test.file")) == "f32ab20c4a86616e32bf2504e1ac5a22"
    )


def test_hashfun_float():
    import math

    pi_50 = 3.14159265358979323846264338327950288419716939937510
    pi_15 = 3.141592653589793
    pi_10 = 3.1415926536
    # comparing for x that have the same x.as_integer_ratio()
    assert (
        math.pi.as_integer_ratio()
        == pi_50.as_integer_ratio()
        == pi_15.as_integer_ratio()
    )
    assert hash_function(math.pi) == hash_function(pi_15) == hash_function(pi_50)
    # comparing for x that have different x.as_integer_ratio()
    assert math.pi.as_integer_ratio() != pi_10.as_integer_ratio()
    assert hash_function(math.pi) != hash_function(pi_10)


def test_hash_function_dict():
    dict1 = {"a": 10, "b": 5}
    dict2 = {"b": 5, "a": 10}
    assert hash_function(dict1) == hash_function(dict2)


def test_hash_function_list_tpl():
    lst = [2, 5.6, "ala"]
    tpl = (2, 5.6, "ala")
    assert hash_function(lst) != hash_function(tpl)


def test_hash_function_list_dict():
    lst = [2, {"a": "ala", "b": 1}]
    hash_function(lst)


def test_hash_function_files(tmp_path: Path):
    file_1 = tmp_path / "file_1.txt"
    file_2 = tmp_path / "file_2.txt"
    file_1.write_text("hello")
    file_2.write_text("hello")

    assert hash_function(File(file_1)) == hash_function(File(file_2))


def test_hash_function_dir_and_files_list(tmp_path: Path):
    dir1 = tmp_path / "foo"
    dir2 = tmp_path / "bar"
    for d in (dir1, dir2):
        d.mkdir()
        for i in range(3):
            f = d / f"{i}.txt"
            f.write_text(str(i))

    assert hash_function(Directory(dir1)) == hash_function(Directory(dir2))
    file_list1: ty.List[File] = [File(f) for f in dir1.iterdir()]
    file_list2: ty.List[File] = [File(f) for f in dir2.iterdir()]
    assert hash_function(file_list1) == hash_function(file_list2)


def test_hash_function_files_mismatch(tmp_path: Path):
    file_1 = tmp_path / "file_1.txt"
    file_2 = tmp_path / "file_2.txt"
    file_1.write_text("hello")
    file_2.write_text("hi")

    assert hash_function(File(file_1)) != hash_function(File(file_2))


def test_hash_function_nested(tmp_path: Path):
    dpath = tmp_path / "dir"
    dpath.mkdir()
    hidden = dpath / ".hidden"
    nested = dpath / "nested"
    hidden.mkdir()
    nested.mkdir()
    file_1 = dpath / "file_1.txt"
    file_2 = hidden / "file_2.txt"
    file_3 = nested / ".file_3.txt"
    file_4 = nested / "file_4.txt"

    for fx in [file_1, file_2, file_3, file_4]:
        fx.write_text(str(random.randint(0, 1000)))

    nested_dir = Directory(dpath)

    orig_hash = nested_dir.hash()

    nohidden_hash = nested_dir.hash(ignore_hidden_dirs=True, ignore_hidden_files=True)
    nohiddendirs_hash = nested_dir.hash(ignore_hidden_dirs=True)
    nohiddenfiles_hash = nested_dir.hash(ignore_hidden_files=True)

    assert orig_hash != nohidden_hash
    assert orig_hash != nohiddendirs_hash
    assert orig_hash != nohiddenfiles_hash

    os.remove(file_3)
    assert nested_dir.hash() == nohiddenfiles_hash
    shutil.rmtree(hidden)
    assert nested_dir.hash() == nohidden_hash


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
    assert re.match(rb"dict:{str:1:a=.{16},str:1:b=.{16},}$", dict_repr)
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
        ({"b": "c", "a": 0}, "04e5c65ec2269775d3b9ccecaf10da38"),
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
    assert re.match(rb".*\.MyClass:{str:1:x=.{16},}", obj_repr)


def test_bytes_repr_slots_obj():
    class MyClass:
        __slots__ = ("x",)

        def __init__(self, x):
            self.x = x

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16},}", obj_repr)


def test_bytes_repr_attrs_slots():
    @attrs.define
    class MyClass:
        x: int

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16},}", obj_repr)


def test_bytes_repr_attrs_no_slots():
    @attrs.define(slots=False)
    class MyClass:
        x: int

    obj_repr = join_bytes_repr(MyClass(1))
    assert re.match(rb".*\.MyClass:{str:1:x=.{16},}", obj_repr)


def test_bytes_repr_type1():
    obj_repr = join_bytes_repr(Path)
    assert obj_repr == b"type:(pathlib.Path)"


def test_bytes_repr_type1a():
    obj_repr = join_bytes_repr(Zip[Json])
    assert obj_repr == rb"type:(mime-like:(application/json+zip))"


def test_bytes_repr_type2():
    T = ty.TypeVar("T")

    class MyClass(ty.Generic[T]):

        a: int
        b: str

        def method(self, f: float) -> float:
            return f + 1

    obj_repr = join_bytes_repr(MyClass[int])
    assert re.match(
        (
            rb"type:\(origin:\(type:\(__dict__:\(str:6:method=.{16},\),annotations:\(str:1:a=.{16},"
            rb"str:1:b=.{16},\),mro:\(type:\(typing.Generic\)\)\)\),args:\(type:\(builtins.int\)\)\)"
        ),
        obj_repr,
    )


def test_bytes_special_form1():
    obj_repr = join_bytes_repr(ty.Union[int, float])
    assert obj_repr == (
        b"type:(origin:(type:(typing.Union)),args:(type:(builtins.int)"
        b"type:(builtins.float)))"
    )


@pytest.mark.skipif(condition=sys.version_info < (3, 10), reason="requires python3.10")
def test_bytes_special_form1a():
    obj_repr = join_bytes_repr(int | float)
    assert obj_repr == (
        b"type:(origin:(type:(types.UnionType)),args:(type:(builtins.int)"
        b"type:(builtins.float)))"
    )


def test_bytes_special_form2():
    obj_repr = join_bytes_repr(ty.Any)
    assert re.match(rb"type:\(typing.Any\)", obj_repr)


def test_bytes_special_form3():
    obj_repr = join_bytes_repr(ty.Optional[Path])
    assert obj_repr == (
        b"type:(origin:(type:(typing.Union)),args:(type:(pathlib.Path)"
        b"type:(builtins.NoneType)))"
    )


@pytest.mark.skipif(condition=sys.version_info < (3, 10), reason="requires python3.10")
def test_bytes_special_form3a():
    obj_repr = join_bytes_repr(Path | None)
    assert obj_repr == (
        b"type:(origin:(type:(types.UnionType)),args:(type:(pathlib.Path)"
        b"type:(builtins.NoneType)))"
    )


def test_bytes_special_form4():
    obj_repr = join_bytes_repr(ty.Type[Path])
    assert (
        obj_repr == b"type:(origin:(type:(builtins.type)),args:(type:(pathlib.Path)))"
    )


def test_bytes_special_form5():
    obj_repr = join_bytes_repr(ty.Callable[[Path, int], ty.Tuple[float, str]])
    assert obj_repr == (
        b"type:(origin:(type:(collections.abc.Callable)),args:(list:(type:(pathlib.Path)"
        b"type:(builtins.int))type:(origin:(type:(builtins.tuple)),"
        b"args:(type:(builtins.float)type:(builtins.str)))))"
    )


def test_bytes_special_form6():
    obj_repr = join_bytes_repr(ty.Tuple[float, ...])
    assert obj_repr == (
        b"type:(origin:(type:(builtins.tuple)),args:(type:(builtins.float)type:(...)))"
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
    with pytest.raises(ValueError, match="not a directory"):
        PersistentCache(text_file.fspath)


def test_unhashable():
    """
    Test that an error is raised if an unhashable object is provided
    """

    class A:

        def __bytes_repr__(self, cache: Cache) -> ty.Generator[bytes, None, None]:
            raise TypeError("unhashable")

        def __repr__(self):
            return "A()"

    # hash_object(A())

    with pytest.raises(
        TypeError,
        match=(
            r"unhashable\nand therefore cannot hash `A\(\)` of type `.*\.test_hash\.A`"
        ),
    ):
        hash_object(A())


def test_hash_task(tmp_path):
    """
    Test that the hash of a task is consistent across runs
    """

    concatenate1 = Concatenate()
    concatenate2 = Concatenate()

    assert hash_function(concatenate1) == hash_function(concatenate2)
