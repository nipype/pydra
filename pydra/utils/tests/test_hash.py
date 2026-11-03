import ast
import importlib
import importlib.util
import linecache
import re
import os
import subprocess
import sys
import textwrap
import time
import types
import uuid
from concurrent.futures import ThreadPoolExecutor
from hashlib import blake2b
from pathlib import Path
from unittest import mock
import attrs
import cloudpickle
import numpy
import pytest
import typing as ty
from fileformats.application import Zip, Json, Yaml
from fileformats.text import TextFile
from pydra.utils import hash as hash_module
from pydra.utils.hash import (
    TIMESTAMP_RESOLUTION_NS,
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
from pydra.utils.tests.utils import (
    Concatenate,
    OtherSpecificFuncTask,
    SpecificFuncTask,
)


def test_hash_file(tmpdir):
    outdir = Path(tmpdir)
    with open(outdir / "test.file", "w") as fp:
        fp.write("test")
    assert (
        hash_function(File(outdir / "test.file")) == "a372c0b478b3f750b57a4e6a1c4042ce"
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
    assert re.match(
        rb"type:\(origin:\(type:\((types.UnionType|typing.Union)\)\),args:\(type:\(builtins.int\)"
        rb"type:\(builtins.float\)\)\)",
        obj_repr,
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
    assert re.match(
        b"type:\(origin:\(type:\((types.UnionType|typing.Union)\)\),args:\(type:\(pathlib.Path\)"
        b"type:\(builtins.NoneType\)\)\)",
        obj_repr,
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


@pytest.fixture
def settled(monkeypatch):
    """Hashes as if every file written so far was last changed long enough ago to be cached"""
    monkeypatch.setattr(
        hash_module, "time_ns", lambda: time.time_ns() + TIMESTAMP_RESOLUTION_NS + 1
    )


def _hash_when_changed(monkeypatch, fileset, cache_path):
    """Hashes ``fileset`` at the instant its file was last changed"""
    st = fileset.fspath.stat()
    monkeypatch.setattr(
        hash_module, "time_ns", lambda: max(st.st_mtime_ns, st.st_ctime_ns)
    )
    return hash_object(fileset, persistent_cache=cache_path)


def _rewrite(path, text, mtime_ns):
    path.write_text(text)
    os.utime(path, ns=(mtime_ns, mtime_ns))


def test_persistent_hash_cache_skips_recently_changed_files(
    monkeypatch, cache_path, text_file
):
    hsh = _hash_when_changed(monkeypatch, text_file, cache_path)
    assert list(cache_path.iterdir()) == []
    assert hsh == hash_object(text_file, persistent_cache=cache_path)


def test_persistent_hash_cache_rewrite_within_timestamp_resolution(
    monkeypatch, cache_path, text_file
):
    mtime_ns = text_file.fspath.stat().st_mtime_ns - 10 * TIMESTAMP_RESOLUTION_NS
    _rewrite(text_file.fspath, "hello", mtime_ns)
    first = _hash_when_changed(monkeypatch, text_file, cache_path)
    _rewrite(text_file.fspath, "world", mtime_ns)
    assert text_file.fspath.stat().st_mtime_ns == mtime_ns
    assert _hash_when_changed(monkeypatch, text_file, cache_path) != first


class _Stat:
    """A stat result with some fields replaced"""

    def __init__(self, st, **fields):
        self._st = st
        self._fields = fields

    def __getattr__(self, name):
        return self._fields.get(name, getattr(self._st, name))


def _poison_cached_hash(cache_path, fileset):
    """Replace the one cached hash, so a cache hit returns b"poisoned" """
    hash_object(fileset, persistent_cache=cache_path)
    (entry,) = cache_path.iterdir()
    entry.write_bytes(b"poisoned")
    assert hash_object(fileset, persistent_cache=cache_path) == b"poisoned"


@pytest.mark.parametrize("field", ["st_size", "st_ino", "st_mtime_ns", "st_ctime_ns"])
def test_persistent_hash_cache_key_has_each_stat_field(
    monkeypatch, settled, cache_path, text_file, field
):
    _poison_cached_hash(cache_path, text_file)
    target = os.stat(text_file.fspath)
    fstat = os.fstat

    def altered_fstat(fd):
        st = fstat(fd)
        if (st.st_dev, st.st_ino) != (target.st_dev, target.st_ino):
            return st
        return _Stat(st, **{field: getattr(st, field) - 1})

    monkeypatch.setattr(os, "fstat", altered_fstat)
    assert hash_object(text_file, persistent_cache=cache_path) != b"poisoned"
    assert len(list(cache_path.iterdir())) == 2


def test_persistent_hash_cache_ignores_stale_stat(
    monkeypatch, settled, cache_path, text_file
):
    path = text_file.fspath
    stale = os.stat(path)
    _poison_cached_hash(cache_path, text_file)
    _rewrite(path, "rewritten elsewhere", stale.st_mtime_ns)
    stat = os.stat
    monkeypatch.setattr(
        os,
        "stat",
        lambda p, *a, **kw: stale if os.fspath(p) == str(path) else stat(p, *a, **kw),
    )
    assert path.stat() is stale
    assert hash_object(text_file, persistent_cache=cache_path) != b"poisoned"


def test_persistent_hash_cache_directory_keyed_on_contents(
    settled, cache_path, tmp_path
):
    root = tmp_path / "dir"
    (root / "sub").mkdir(parents=True)
    nested = root / "sub" / "nested.txt"
    nested.write_text("foo")
    directory = Directory(root)
    root_stat = root.stat()
    _poison_cached_hash(cache_path, directory)
    _rewrite(nested, "longer", nested.stat().st_mtime_ns)
    assert root.stat().st_mtime_ns == root_stat.st_mtime_ns
    assert hash_object(directory, persistent_cache=cache_path) != b"poisoned"
    assert len(list(cache_path.iterdir())) == 2


def test_persistent_hash_cache(settled, cache_path, text_file):
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
    text_file.fspath.write_text("bar")
    os.utime(
        text_file.fspath, (0, text_file.fspath.stat().st_mtime - 10)
    )  # distinct mtime
    assert hash_object(text_file, persistent_cache=cache_path) != modified_hash
    assert len(list(cache_path.iterdir())) == 2


def test_persistent_hash_cache_cleanup1(settled, cache_path, text_file):
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


def test_persistent_hash_cache_cleanup2(settled, cache_path, text_file):
    """
    Test the persistent hash is cleaned up after use if the periods between cleanups
    is greater than the explicitly provided cleanup_period
    """
    persistent_cache = PersistentCache(cache_path, cleanup_period=-100)
    hash_object(text_file, persistent_cache=persistent_cache)
    assert len(list(cache_path.iterdir())) == 1
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


def test_bytes_repr_annotated_type():
    """The metadata of annotated types doesn't change the hash"""

    @attrs.define(frozen=True)
    class Marker:
        format: type

    annotated = ty.Annotated[ty.Union[Json, Yaml], Marker(Json)]
    assert hash_function(annotated) == hash_function(ty.Union[Json, Yaml])
    assert hash_function(annotated) == hash_function(
        ty.Annotated[ty.Union[Json, Yaml], Marker(Yaml)]
    )
    assert hash_function(ty.List[annotated]) == hash_function(
        ty.List[ty.Union[Json, Yaml]]
    )
    assert hash_function(annotated) != hash_function(ty.Union[Json, TextFile])


def _adder(k):
    def add(x):
        return x + k

    return add


def test_hash_function_closure():
    """Functions with the same source but different captured values hash differently"""
    assert hash_function(_adder(1)) == hash_function(_adder(1))
    assert hash_function(_adder(1)) != hash_function(_adder(2))
    assert hash_function(_adder(SpecificFuncTask)) != hash_function(
        _adder(OtherSpecificFuncTask)
    )


def test_hash_function_closure_values_hash_by_value():
    """A captured dict or array is hashed by its value, like any input"""
    assert hash_function(_adder({"a": 1})) == hash_function(_adder({"a": 1}))
    assert hash_function(_adder({"a": 1})) != hash_function(_adder({"a": 2}))
    assert hash_function(_adder(numpy.arange(3))) == hash_function(
        _adder(numpy.arange(3))
    )
    assert hash_function(_adder(numpy.arange(3))) != hash_function(
        _adder(numpy.arange(4))
    )


def test_hash_function_unhashable_closure():
    """A captured value that cannot be hashed fails the hash, naming the value"""

    class A:
        def __bytes_repr__(self, cache):
            raise TypeError("unhashable")

    with pytest.raises(TypeError, match=r"unhashable\nin 'k', captured by function"):
        hash_object(_adder(A()))


_HASH_IN_SUBPROCESS = (
    "import sys, cloudpickle; from pydra.utils.hash import hash_function; "
    "print(hash_function(cloudpickle.loads(sys.stdin.buffer.read())))"
)


def test_hash_function_pickled_by_value():
    """A function pickled by value, as the cf worker sends it, keeps its hash in
    another process"""
    func = _adder(3)
    module = sys.modules[func.__module__]
    cloudpickle.register_pickle_by_value(module)
    try:
        payload = cloudpickle.dumps(func)
    finally:
        cloudpickle.unregister_pickle_by_value(module)
    result = subprocess.run(
        [sys.executable, "-c", _HASH_IN_SUBPROCESS],
        input=payload,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "random"},
    )
    assert result.stdout.decode().strip() == hash_function(func)


def _define(tmp_path: Path, source: str, name: str = "f"):
    """Load `name` from `source`, written to a new module file under `tmp_path`"""
    module_name = f"_fh_{len(list(tmp_path.iterdir()))}"
    path = tmp_path / f"{module_name}.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


_BASE_SOURCE = """
def deco(func):
    return func

@deco
def f(a: int, b: int = 1) -> int:
    @deco
    def g(x):
        return x * 2
    c = a + b
    return g(c)
"""


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            _BASE_SOURCE.replace(
                "    c = a + b", "    # comment\n    c = a + b  # sum"
            ),
            id="comments",
        ),
        pytest.param(
            _BASE_SOURCE.replace("b: int = 1", "b: int=1").replace(
                "c = a + b", "c = (a +\n         b)"
            ),
            id="formatting",
        ),
        pytest.param(
            _BASE_SOURCE.replace("-> int:\n", '-> int:\n    """Doc"""\n'),
            id="docstring",
        ),
        pytest.param(
            _BASE_SOURCE.replace(
                "(a: int, b: int = 1) -> int", '(a: float, b: "str" = 1) -> "float"'
            ),
            id="annotations",
        ),
        pytest.param(
            _BASE_SOURCE.replace("def f(", "def h(") + "f = h\n",
            id="name",
        ),
        pytest.param(
            "import os\n\n\nclass A:\n    pass\n" + _BASE_SOURCE, id="location"
        ),
        pytest.param(
            _BASE_SOURCE.replace("@deco\ndef f", "def f"), id="own-decorators"
        ),
    ],
)
def test_hash_function_unchanged(tmp_path, variant):
    """Comments, formatting, docstrings, signature annotations and the function's own
    name, location and decorators do not change its hash"""
    assert hash_function(_define(tmp_path, variant)) == hash_function(
        _define(tmp_path, _BASE_SOURCE)
    )


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(_BASE_SOURCE.replace("x * 2", "x * 3"), id="body"),
        pytest.param(_BASE_SOURCE.replace("    @deco\n", ""), id="decorator"),
        pytest.param(_BASE_SOURCE.replace("b: int = 1", "b: int = 2"), id="default"),
        pytest.param(
            _BASE_SOURCE.replace("def g(x):", "def g(x) -> int:"),
            id="nested-annotation",
        ),
        pytest.param(
            _BASE_SOURCE.replace("def g(x):\n", 'def g(x):\n        """Doc"""\n'),
            id="nested-docstring",
        ),
        pytest.param(
            _BASE_SOURCE.replace("c = a", "d = a").replace("g(c)", "g(d)"),
            id="local-name",
        ),
    ],
)
def test_hash_function_changed(tmp_path, variant):
    """The body, default values, local names and the decorators, annotations and
    docstrings of the code a function defines change its hash"""
    assert hash_function(_define(tmp_path, variant)) != hash_function(
        _define(tmp_path, _BASE_SOURCE)
    )


def test_hash_function_lambdas_on_one_line(tmp_path):
    """Lambdas that share a source line are hashed by their own code"""
    source = "f, g, h = (lambda x: x + 1), (lambda x: x + 2), (lambda x: x + 1)\n"
    f, g, h = (_define(tmp_path, source, name) for name in "fgh")
    assert b"ast@1:" in b"".join(bytes_repr(g, Cache()))
    assert hash_function(f) != hash_function(g)
    assert hash_function(f) == hash_function(h)


def test_hash_function_encoding_tagged(tmp_path):
    """The code part is tagged by its encoding, the byte-code fallback by the Python
    version, and by the tier when the function reads references"""
    namespace: dict = {}
    exec("def f(a):\n    return a + 1\n", namespace)
    assert b"function:(ast@1:" in b"".join(
        bytes_repr(_define(tmp_path, "def f(a):\n    return a + 1\n"), Cache())
    )
    assert b"function:(module/ast@1:" in b"".join(
        bytes_repr(_define(tmp_path, _BASE_SOURCE), Cache(code_hash="module"))
    )
    version = f"bytecode@{sys.version_info[0]}.{sys.version_info[1]}:"
    assert f"function:({version}".encode() in b"".join(
        bytes_repr(namespace["f"], Cache())
    )


_CORPUS_SOURCE = '''
async def f(a, /, b, *args, c=None, d: int = 0, **kwargs) -> dict:
    """Docstring"""
    import os.path as osp
    from .. import os
    global_name = os.sep
    x: list[int] = [i * 2 for i in args if i]
    y = {k: v for k, v in kwargs.items()}
    z = {*x} | {1.5, 2j, b"raw", ..., True}
    w = (n async for n in b)
    if (m := len(x)) > 1 and not c or d is None:
        x[1:m:2] = f"{a!r:>{d}}" + f"{b=}" f"{x:{d}}" f"{x:>{d}.{d}}" f"" f"{{}}{a:}"
    try:
        await b
    except* ValueError as e:
        raise RuntimeError() from e
    else:
        pass
    finally:
        del y
    match a:
        case [1, *rest] if rest:
            pass
        case {"k": 1, **others}:
            pass
        case int(real=0) | None:
            pass
        case _:
            pass
    async with c as (p, q), d:
        async for i in b:
            continue
    while False:
        break
    class K(int, metaclass=type):
        """Class docstring"""
        attr = lambda self, *, u=1: u ** -self
    def g(*, e):
        nonlocal x
        yield from e
    return {"x": x, "y": y, "z": z, "w": w, "k": K, "g": g, "v": global_name}
'''


def test_hash_function_same_on_every_python():
    """The `ast@1` encoding of a function using most of the syntax of Python 3.11
    gives the same hash on every supported Python version"""
    encoding = hash_module.encode_ast(ast.parse(_CORPUS_SOURCE).body[0])
    assert blake2b(encoding.encode(), digest_size=16).hexdigest() == (
        "c3c9ce3c2e2549d95a4f97f92d16e554"
    )


def test_lru_cache_holds_most_recently_used():
    cache = hash_module._LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    cache.put("c", 3)
    assert len(cache) == 2
    assert (cache.get("a"), cache.get("b"), cache.get("c")) == (1, None, 3)


def test_hash_function_concurrently(tmp_path, monkeypatch):
    """Functions hashed from several threads while the source caches evict get the
    hashes they get serially"""
    funcs = [
        _define(tmp_path, _BASE_SOURCE.replace("x * 2", f"x * {i}")) for i in range(8)
    ]
    expected = [hash_function(f) for f in funcs]
    monkeypatch.setattr(hash_module, "_source_nodes_cache", hash_module._LRUCache(2))
    monkeypatch.setattr(hash_module, "_function_bytes_cache", hash_module._LRUCache(2))
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(hash_function, funcs * 50))
    assert results == expected * 50


@pytest.fixture
def load_modules(tmp_path, monkeypatch):
    """Write modules to a new directory and import them by name, replacing those of
    the same names loaded before"""
    loaded = set()
    count = 0

    def load(sources: dict[str, str], directory: str = "") -> list[types.ModuleType]:
        nonlocal count
        count += 1
        root = tmp_path / f"v{count}"
        for name, source in sources.items():
            path = root / directory / Path(*name.split(".")).with_suffix(".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(source))
            for package in path.parents[: name.count(".")]:
                (package / "__init__.py").touch()
        for name in loaded:
            sys.modules.pop(name, None)
        monkeypatch.syspath_prepend(str(root / directory))
        importlib.invalidate_caches()
        loaded.update(n.split(".")[0] for n in sources)
        loaded.update(sources)
        return [importlib.import_module(n) for n in sources]

    yield load
    for name in loaded:
        sys.modules.pop(name, None)


def _code_hash(func, code_hash: str) -> str:
    return hash_function(func, cache=Cache(code_hash=code_hash))


_TIERS = ("function", "module", "user")

_MAIN = """
import helper
from helper import g

K = {k}


def h(x):
    return x * {h}


def f(x):
    return h(x) + helper.u(x) + g(x) + K
"""

_HELPER = """
def u(x):
    return x - {u}


def g(x):
    return x + 1
"""


def _main_f(load_modules, k=1, h=2, u=3):
    main, _ = load_modules(
        {"main": _MAIN.format(k=k, h=h), "helper": _HELPER.format(u=u)}
    )
    return main.f


@pytest.mark.parametrize(
    "change,changed_at",
    [
        pytest.param({"k": 5}, _TIERS, id="global-value"),
        pytest.param({"h": 5}, ("module", "user"), id="same-module-helper"),
        pytest.param({"u": 5}, ("user",), id="other-module-helper"),
    ],
)
def test_code_hash_tiers(load_modules, change, changed_at):
    """A global value is hashed at every tier, a helper in the function's module from
    the "module" tier, and one in another user module at the "user" tier"""
    base = {t: _code_hash(_main_f(load_modules), t) for t in _TIERS}
    variant = {t: _code_hash(_main_f(load_modules, **change), t) for t in _TIERS}
    assert {t for t in _TIERS if base[t] != variant[t]} == set(changed_at)


def test_code_hash_tier_tagged(load_modules):
    """Hashes of one function at different tiers differ"""
    f = _main_f(load_modules)
    assert len({_code_hash(f, t) for t in _TIERS}) == 3
    assert b"function:(user/ast@1:" in b"".join(bytes_repr(f, Cache()))


def test_code_hash_default_from_env(load_modules, monkeypatch):
    f = _main_f(load_modules)
    monkeypatch.setenv(hash_module.CODE_HASH_ENV_VAR, "function")
    assert hash_function(f) == _code_hash(f, "function")
    monkeypatch.setenv(hash_module.CODE_HASH_ENV_VAR, "everything")
    with pytest.raises(ValueError, match="code_hash must be one of"):
        hash_function(f)


_LIB = """
def h(x):
    return x * {h}
"""

_USES_LIB = """
from libpkg.core import {name} as h


def f(x):
    return h(x)
"""


def _lib_f(load_modules, h=2, name="h"):
    _, main = load_modules(
        {"libpkg.core": _LIB.format(h=h), "uses_lib": _USES_LIB.format(name=name)},
        directory="site-packages",
    )
    return main.f


@pytest.mark.parametrize("code_hash", _TIERS)
def test_code_hash_library_by_name(load_modules, code_hash):
    """Code installed in site-packages is identified by its qualified name, never its
    code"""
    base = _code_hash(_lib_f(load_modules), code_hash)
    assert _code_hash(_lib_f(load_modules, h=5), code_hash) == base
    renamed = _LIB.replace("def h", "def k")
    _, main = load_modules(
        {"libpkg.core": renamed.format(h=2), "uses_lib": _USES_LIB.format(name="k")},
        directory="site-packages",
    )
    assert _code_hash(main.f, code_hash) != base


def test_code_hash_library_objects_by_name():
    """Library modules are referenced by their names, whatever is read from them"""

    def f(x):
        return numpy.add(numpy.mean(x), os.environ)

    encoded = b"".join(bytes_repr(f, Cache()))
    assert b"global:numpy=module:numpy" in encoded
    assert b"global:os=module:os" in encoded


def test_code_hash_library_instances_by_name(load_modules):
    _, main = load_modules(
        {
            "libpkg.state": "class Config:\n    pass\n\nCONFIG = Config()\n",
            "uses_state": (
                "from libpkg.state import CONFIG\nfrom os import environ\n"
                "from numpy import add\n\n\ndef f(x):\n    return add(x, CONFIG, environ)\n"
            ),
        },
        directory="site-packages",
    )
    encoded = b"".join(bytes_repr(main.f, Cache()))
    assert b"global:CONFIG=lib:libpkg.state.CONFIG" in encoded
    assert b"global:environ=lib:os.environ" in encoded
    assert b"global:add=lib:numpy.add" in encoded


@pytest.mark.parametrize(
    "setting,changed", [("", True), ("-helper", False), ("main", True)]
)
def test_code_hash_user_packages_exclude(load_modules, monkeypatch, setting, changed):
    """``PYDRA_USER_PACKAGES`` names a package ``-name`` to make it library code"""
    monkeypatch.setenv(hash_module.USER_PACKAGES_ENV_VAR, setting)
    base = _code_hash(_main_f(load_modules), "user")
    assert (_code_hash(_main_f(load_modules, u=5), "user") != base) is changed


@pytest.mark.parametrize("setting,changed", [("", False), ("libpkg", True)])
def test_code_hash_user_packages_include(load_modules, monkeypatch, setting, changed):
    """``PYDRA_USER_PACKAGES`` names an installed package to make it user code"""
    monkeypatch.setenv(hash_module.USER_PACKAGES_ENV_VAR, setting)
    base = _code_hash(_lib_f(load_modules), "user")
    assert (_code_hash(_lib_f(load_modules, h=5), "user") != base) is changed


def test_code_hash_pydra_is_library(monkeypatch):
    """pydra is library code unless named in ``PYDRA_USER_PACKAGES``"""
    from pydra.utils.general import get_fields

    assert hash_module.is_library(hash_module.hash_function)
    assert hash_module.is_library(get_fields)
    monkeypatch.setenv(hash_module.USER_PACKAGES_ENV_VAR, "pydra.utils.general")
    assert hash_module.is_library(hash_module.hash_function)
    assert not hash_module.is_library(get_fields)


def test_code_hash_mutable_global_by_value(load_modules):
    """A dict read as a global is hashed by the value it holds when hashed"""
    (main,) = load_modules(
        {"main": "CFG = {'k': 1}\n\ndef f(x):\n    return CFG['k']\n"}
    )
    before = hash_function(main.f)
    main.CFG["k"] = 2
    assert hash_function(main.f) != before
    main.CFG["k"] = 1
    assert hash_function(main.f) == before


def _notebook():
    """A namespace that runs cells as IPython does, registering their source"""
    namespace = {"__name__": "__main__"}

    def run(source: str):
        filename = f"<cell-{uuid.uuid4().hex}>"
        source = textwrap.dedent(source)
        linecache.cache[filename] = (
            len(source),
            None,
            source.splitlines(True),
            filename,
        )
        exec(compile(source, filename, "exec"), namespace)

    return namespace, run


def test_code_hash_notebook_rebinding():
    """A name a notebook cell rebinds, by assignment, ``global`` or an import fallback,
    is hashed as what it holds when hashed"""
    namespace, run = _notebook()
    run("from json import dumps as enc")
    run("def f(x):\n    return enc(x)\n")
    hashes = [hash_function(namespace["f"])]
    assert b"global:enc=lib:json.dumps" in b"".join(bytes_repr(namespace["f"], Cache()))
    run("enc = {'state': 1}")
    hashes.append(hash_function(namespace["f"]))
    run("def setup():\n    global enc\n    enc = {'state': 2}\n\nsetup()\n")
    hashes.append(hash_function(namespace["f"]))
    run(
        "try:\n    from json import nonexistent as enc\n"
        "except ImportError:\n    from pickle import dumps as enc\n"
    )
    hashes.append(hash_function(namespace["f"]))
    assert len(set(hashes)) == 4


def test_code_hash_notebook_helper():
    """A helper defined in an earlier cell is hashed by its definition"""
    hashes = []
    for body in ("x * 2", "x * 3"):
        namespace, run = _notebook()
        run(f"def h(x):\n    return {body}\n")
        run("def f(x):\n    return h(x)\n")
        hashes.append(hash_function(namespace["f"]))
    assert hashes[0] != hashes[1]


_CLASSES = """
class Base:
    def f(self, x):
        return x * {base}


class Child(Base):
    scale = 1

    def g(self, x):
        return self.f(x)


def use(x):
    return Child().g(x)
"""


@pytest.mark.parametrize("code_hash,changed", [("function", False), ("module", True)])
def test_code_hash_base_classes(load_modules, code_hash, changed):
    """The bases of a class reached are walked"""
    (a,) = load_modules({"classes": _CLASSES.format(base=2)})
    base = _code_hash(a.use, code_hash)
    (b,) = load_modules({"classes": _CLASSES.format(base=3)})
    assert (_code_hash(b.use, code_hash) != base) is changed


def test_code_hash_cycles(load_modules):
    """Mutually recursive functions and self-referencing classes terminate"""
    source = """
    def f(n):
        return g(n - 1) if n else A

    def g(n):
        return f(n - 1) if n else {body}

    class A:
        def me(self):
            return A
    """
    (a,) = load_modules({"cyc": source.format(body="0")})
    (b,) = load_modules({"cyc": source.format(body="1")})
    assert hash_function(a.f) != hash_function(b.f)


def test_code_hash_deep_chain(load_modules):
    """A long chain of calls is walked without recursion"""
    depth = 2000
    source = (
        "".join(f"def f{i}(x):\n    return f{i + 1}(x)\n\n" for i in range(depth))
        + f"def f{depth}(x):\n    return x * {{last}}\n"
    )
    (a,) = load_modules({"chain": source.format(last=2)})
    (b,) = load_modules({"chain": source.format(last=3)})
    assert hash_function(a.f0) != hash_function(b.f0)


def test_code_hash_unhashable_global(load_modules):
    """A global that cannot be hashed fails the hash, naming it"""
    (main,) = load_modules(
        {
            "unhashable": (
                "class A:\n    def __bytes_repr__(self, cache):\n"
                "        raise TypeError('unhashable')\n\n"
                "OBJ = A()\n\ndef f(x):\n    return OBJ\n"
            )
        }
    )
    with pytest.raises(TypeError, match=r"unhashable\nin 'OBJ', read by function 'f'"):
        hash_object(main.f)


def test_code_hash_helper_defaults(load_modules):
    """The defaults of a helper are hashed"""
    source = (
        "K = {k}\n\ndef h(x, k=K):\n    return x * k\n\ndef f(x):\n    return h(x)\n"
    )
    (a,) = load_modules({"defs": source.format(k=1)})
    base = hash_function(a.f)
    (b,) = load_modules({"defs": source.format(k=2)})
    b.K = 1  # the global no longer differs, only the default
    assert hash_function(b.f) != base


_WALK_HELPER = """
import json


def u(x):
    return x - 3


class Base:
    def f(self, x):
        return x * 2
"""

_WALK_MAIN = """
import os.path
import fh2helper
from fh2helper import Base

K = {"a": [1, 2]}


def h(x, k=3):
    return x * k


class Child(Base):
    def g(self, x):
        return super().f(x)


def f(x):
    y = [h(i) for i in range(x) if i not in K]
    z = (fh2helper.u(i) for i in y)

    class Local:
        v = len(y)

    lam = lambda q: q + K["a"][0]
    return os.path.join(
        str(sum(z)), str(Child().g(lam(x))), fh2helper.json.dumps(Local.v)
    )
"""


def test_code_hash_same_on_every_python(load_modules):
    """The hash of a function reading globals, attributes of modules, helpers, classes
    and their bases is the same on every supported Python version (measured on 3.11 to
    3.15)"""
    _, main = load_modules({"fh2helper": _WALK_HELPER, "fh2main": _WALK_MAIN})
    assert [_code_hash(main.f, t) for t in _TIERS] == [
        "fee118d338ef26b41b814deac6fe8b3a",
        "e5ad835c221d7657c1ebf53e177ea243",
        "d0bc4c1bc86bb5cb68d01d8b8280e059",
    ]


@pytest.mark.parametrize("code_hash,changed", [("function", False), ("user", True)])
def test_code_hash_set_by_define(load_modules, code_hash, changed):
    """``define(code_hash=...)`` sets the tier of a task's checksum"""
    from pydra.compose import python, workflow

    checksums = []
    for u in (3, 5):
        f = _main_f(load_modules, u=u)
        checksums.append(python.define(f, code_hash=code_hash)(x=1)._checksum)
    assert (checksums[0] != checksums[1]) is changed
    with pytest.raises(ValueError, match="code_hash must be one of"):
        python.define(f, code_hash="all")
    with pytest.raises(ValueError, match="code_hash must be one of"):
        workflow.define(f, code_hash="all")
