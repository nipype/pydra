"""Inputs are digested through ``pydra.utils.digest``; a ``Path`` is a value (RFC 0004
section 6.2, I2), and arrays of references are digested by content at any depth (pydra
issue #942)."""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest
from fileformats.generic import Directory, File

from pydra.compose import python
from pydra.engine.tests.test_portable_entry import (  # noqa: F401
    Pair,
    entry_path,
    publish,
    read,
    root,
    tags,
)
from pydra.utils import digest as dg
from pydra.utils import hash as hash_module
from pydra.utils.digest import digest, fileset_digest, jcs, register_digest
from pydra.utils.digest import register_fileset_tag
from pydra.utils.hash import CacheKey, PersistentCache
from pydra.utils.identity import task_record

STRING_DTYPE = getattr(getattr(np, "dtypes", None), "StringDType", None)
needs_stringdtype = pytest.mark.skipif(
    STRING_DTYPE is None, reason="numpy without StringDType"
)


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(dg, "_BY_TAG", dict(dg._BY_TAG))
    monkeypatch.setattr(dg, "_BY_TYPE", dict(dg._BY_TYPE))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@python.define
def UsePath(p: Path) -> str:
    return str(p)


@python.define
def UseFile(f: File) -> str:
    return str(f)


@python.define
def UseAny(x: object) -> str:
    return str(x)


def _input(task, name):
    return task_record(task)["inputs"][name]


# -- Path is a value ---------------------------------------------------------------------


def test_path_input_is_string_not_content(tmp_path):
    target = tmp_path / "out.txt"
    missing = _input(UsePath(p=target), "p")
    target.write_text("a")
    present = _input(UsePath(p=target), "p")
    target.write_text("a longer content")
    changed = _input(UsePath(p=target), "p")
    assert missing == present == changed
    assert missing == "json@1:" + sha(jcs({"$path": str(target)}))
    assert missing != _input(UsePath(p=tmp_path / "other.txt"), "p")


def test_path_digest_golden():
    assert (
        _input(UsePath(p=Path("data/out.txt")), "p")
        == "json@1:191649a04686e7f3b380cf014035e30a7cfc8764ecf0345a6595673841e5d552"
    )
    assert digest(Path("data/out.txt")) == _input(UsePath(p=Path("data/out.txt")), "p")


def test_path_nested_in_a_container_is_a_value(tmp_path):
    target = tmp_path / "out.txt"
    before = _input(UseAny(x=[target]), "x")
    target.write_text("now it exists")
    assert before == _input(UseAny(x=[target]), "x")
    assert before.startswith("json@1:")


# -- no digest raises, naming the field ---------------------------------------------------


def test_lone_surrogate_input_raises_naming_the_field():
    with pytest.raises(ValueError, match="'x'"):
        task_record(UseAny(x="\udcff"))
    with pytest.raises(ValueError, match="'x'"):
        task_record(UseAny(x={"k": ["\udcff"]}))
    with pytest.raises(ValueError, match="'p'"):
        task_record(UsePath(p=Path("bad\udcffname")))


# -- registered digests and numbers -------------------------------------------------------


def test_inputs_are_typed_digests():
    assert _input(UseAny(x=1), "x") != _input(UseAny(x=1.0), "x")
    assert _input(UseAny(x=[1]), "x") != _input(UseAny(x=(1,)), "x")
    assert _input(UseAny(x=2**60), "x") != _input(UseAny(x=2**60 + 1), "x")
    assert _input(UseAny(x=2**60), "x").startswith("json@1:")


def test_registered_digest_wins_and_nests_as_digest():
    class Scaled:
        def __init__(self, v):
            self.v = v

    def scaled_digest(value):
        return "scaled@1:" + sha(str(round(value.v, 1)).encode())[:32]

    register_digest(Scaled, scaled_digest)
    assert _input(UseAny(x=Scaled(1.01)), "x") == _input(UseAny(x=Scaled(1.02)), "x")
    assert _input(UseAny(x=Scaled(1.01)), "x") != _input(UseAny(x=Scaled(1.5)), "x")
    nested = _input(UseAny(x=[Scaled(1.01)]), "x")
    assert nested == _input(UseAny(x=[Scaled(1.02)]), "x")
    assert nested == "json@1:" + sha(jcs([{"$digest": scaled_digest(Scaled(1.01))}]))


# -- file sets ----------------------------------------------------------------------------


def test_file_typed_input_by_content(tmp_path):
    register_fileset_tag(File, "test.inputs.File")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    one = tmp_path / "a" / "one.txt"
    one.write_bytes(b"content")
    first = _input(UseFile(f=one), "f")
    assert first == fileset_digest("test.inputs.File", [(".", sha(b"content"))])
    renamed = tmp_path / "b" / "other-name.txt"
    shutil.copy(one, renamed)
    assert _input(UseFile(f=renamed), "f") == first
    one.write_bytes(b"different")
    assert _input(UseFile(f=one), "f") != first


def test_unregistered_file_type_keeps_its_content_hash(tmp_path):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    assert _input(UseFile(f=one), "f").startswith("py@1:")


def test_persistent_hit_populates_memory(tmp_path):
    location = tmp_path / "hashes"
    key = CacheKey(("some", "key"))
    PersistentCache(location).get_or_calculate_hash(key, lambda: b"digest-bytes")
    fresh = PersistentCache(location)

    def never():
        raise AssertionError("a hit does not calculate")

    assert fresh.get_or_calculate_hash(key, never) == b"digest-bytes"
    assert fresh._hashes[key] == b"digest-bytes"
    for path in location.iterdir():
        path.unlink()
    assert fresh.get_or_calculate_hash(key, never) == b"digest-bytes"


def test_decoded_fileset_digest_from_entry(
    root, tmp_path, tags, monkeypatch
):  # noqa: F811
    register_fileset_tag(File, "test.inputs.File")
    src = tmp_path / "one.bin"
    src.write_bytes(b"payload")
    publish(root, {"f": File(src)})
    got = read(root).outputs["f"]
    set_name = fileset_digest("test.inputs.File", [(".", sha(b"payload"))])
    assert Path(next(iter(got.fspaths))).parent.name == set_name.split(":")[1]

    def no_reading(self, *args, **kwargs):
        raise AssertionError("the digest is read from the entry, not recomputed")

    monkeypatch.setattr(File, "byte_chunks", no_reading)
    # the members were written moments ago, inside the settle window of the persistent
    # cache: only the entry can have supplied the digest
    assert digest(got) == set_name
    assert _input(UseFile(f=got), "f") == set_name


def test_decoded_pair_and_directory_recompute_agree(
    root, tmp_path, tags, monkeypatch
):  # noqa: F811
    src = tmp_path / "src"
    (src / "tree" / "sub").mkdir(parents=True)
    (src / "scan.hdr").write_bytes(b"header")
    (src / "scan.img").write_bytes(b"image-data")
    (src / "tree" / "top.txt").write_bytes(b"t")
    (src / "tree" / "sub" / "x.txt").write_bytes(b"x")
    pair = Pair([src / "scan.hdr", src / "scan.img"])
    tree = Directory(src / "tree")
    publish(root, {"pair": pair, "tree": tree})
    got = read(root).outputs
    # a process without the in-memory entry: the worker that re-hashes a downstream input
    monkeypatch.setattr(hash_module, "_fileset_memory", hash_module._LRUCache(16))
    assert digest(got["pair"]) == digest(pair)
    assert digest(got["tree"]) == digest(tree)
    assert digest(got["pair"]).startswith("file@1:")


@needs_stringdtype
def test_stringdtype_input_through_the_record():
    a = np.array(["a" * 60], dtype=STRING_DTYPE())
    b = np.array(["b" * 60], dtype=STRING_DTYPE())
    assert _input(UseAny(x=[a]), "x") != _input(UseAny(x=[b]), "x")
