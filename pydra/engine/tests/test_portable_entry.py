"""pydra-entry@1: encoding, codecs, reading and per-root trust (RFC 0003 sections 6.1 to 6.4)."""

import datetime
import hashlib
import json
import os
import pickle
import stat
import sys
import typing as ty
import uuid
from pathlib import Path

import numpy as np
import pytest
from fileformats.core.mixin import WithSeparateHeader
from fileformats.generic import Directory, File

from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.portable import codecs
from pydra.engine.portable.entry import (
    CorruptEntry,
    EntryFormatUnknown,
    EntryNotTrusted,
    EntryRef,
    MisfiledEntry,
    PydraEntryV1,
    StoredEntry,
    UnencodableOutput,
    check_encodable,
)
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.layout import cdir_for, shard_for
from pydra.engine.portable.trust import Root, check_private
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.utils import digest as dg
from pydra.utils.digest import canonical_npy, digest, fileset_digest, register_codec
from pydra.utils.digest import register_digest, register_fileset_tag
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
)

TRUST = "trusted" if IS_WINDOWS else "private"
posix_only = pytest.mark.skipif(IS_WINDOWS, reason="needs POSIX modes and owners")
CTX = {"python": "3.12", "platform": "test", "n": 1, "x": 1.5, "none": None}
IDENT = PydraIdentityV1()
FP = IDENT.fingerprint(CTX)
CHECKSUM = "python-abc123"


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(dg, "_BY_TAG", dict(dg._BY_TAG))
    monkeypatch.setattr(dg, "_BY_TYPE", dict(dg._BY_TYPE))


class Hdr(File):
    ext = ".hdr"


class Img(File):
    ext = ".img"


class Pair(WithSeparateHeader, Img):
    header_type = Hdr


class Tagged(File):
    ext = ".dat"


@pytest.fixture
def tags():
    register_fileset_tag(Pair, "test.entry.Pair")
    register_fileset_tag(Directory, "test.entry.Directory")
    register_fileset_tag(Tagged, "test.entry.Tagged")


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "root"
    d.mkdir()
    handle = StorageDir.open_root(d, StorePolicy(), trust=TRUST)
    yield handle
    handle.close()


def entry_path(root, checksum=CHECKSUM, fp=FP) -> Path:
    cdir = cdir_for(checksum)
    shard = shard_for(cdir, checksum, "sha256-2")
    return Path(root.path) / "results@1" / shard / cdir / fp


def publish(
    root,
    outputs,
    *,
    fmt=None,
    checksum=CHECKSUM,
    key=None,
    context=CTX,
    task=None,
    files_from=None,
    writer="w1",
):
    """Stage an entry in ``staging/`` and rename it to its address, as the store will."""
    fmt = fmt if fmt is not None else PydraEntryV1(pickle=False, trust=TRUST)
    fp = IDENT.fingerprint(context)
    staging_parent = root.child("staging", create=True)
    name = uuid.uuid4().hex
    with staging_parent.child(name, create=True, mode=0o700) as staging:
        digest_ = fmt.stage(
            staging, checksum, key, context, outputs, task, files_from, writer
        )
    cdir = cdir_for(checksum)
    entries = root.child(fmt.entries, create=True)
    shard = entries.child(shard_for(cdir, checksum, "sha256-2"), create=True)
    cd = shard.child(cdir, create=True)
    staging_parent.rename(name, cd, fp)
    return digest_


def read(
    root, *, fmt=None, checksum=CHECKSUM, fp=FP, task=None, trust=TRUST, policy=None
):
    fmt = fmt if fmt is not None else PydraEntryV1(pickle=False, trust=TRUST)
    root_ = root
    path = entry_path(root, checksum, fp)
    entry = root_
    for part in path.relative_to(root.path).parts:
        entry = entry.child(part)
    with entry:
        return fmt.read(entry, checksum, fp, task, trust)


def entry_json(root, checksum=CHECKSUM, fp=FP) -> dict:
    return json.loads((entry_path(root, checksum, fp) / "entry.json").read_text())


def edit_entry(root, fn, *, redigest=True, checksum=CHECKSUM, fp=FP) -> None:
    path = entry_path(root, checksum, fp) / "entry.json"
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    doc = json.loads(path.read_text())
    fn(doc)
    if redigest:
        doc["digest"] = codecs.output_digest(doc["outputs"])
    path.write_text(json.dumps(doc))


def roundtrip(root, outputs, *, task=None, **kw):
    publish(root, outputs, task=task, **kw)
    return read(root, task=task).outputs


def encoded(root, outputs, **kw) -> dict:
    publish(root, outputs, **kw)
    return entry_json(root)["outputs"]


@python.define(outputs={"a": int, "b": ty.List[float], "c": ty.Tuple[int, str]})
def Typed(x: int) -> ty.Tuple[int, ty.List[float], ty.Tuple[int, str]]:
    return x, [1.0], (1, "a")


# -- errors ------------------------------------------------------------------------------


def test_error_hierarchy():
    assert issubclass(CorruptEntry, StoreUnavailable)
    assert issubclass(MisfiledEntry, CorruptEntry)
    assert issubclass(EntryFormatUnknown, StoreUnavailable)
    assert issubclass(EntryNotTrusted, StoreUnavailable)
    assert not issubclass(EntryFormatUnknown, CorruptEntry)
    assert not issubclass(EntryNotTrusted, CorruptEntry)
    assert issubclass(UnencodableOutput, TypeError)
    err = UnencodableOutput("out", np.ma.MaskedArray)
    assert (err.field, err.type) == ("out", np.ma.MaskedArray)
    assert "out" in str(err) and "MaskedArray" in str(err)
    back = pickle.loads(pickle.dumps(err))
    assert (back.field, back.type) == ("out", np.ma.MaskedArray)


def test_root_and_ref_values():
    r = Root("/x")
    assert r.trust == "shared"
    assert Root("/x", trust="private").trust == "private"
    with pytest.raises(ValueError):
        Root("/x", trust="everyone")
    with pytest.raises(AttributeError):
        r.trust = "private"  # type: ignore[misc]
    ref = EntryRef("r", "c", "f")
    assert (ref.root, ref.cdir, ref.fingerprint) == ("r", "c", "f")
    assert StoredEntry.__name__ == "StoredEntry"


def test_format_names_and_patterns():
    fmt = PydraEntryV1()
    assert (fmt.name, fmt.entries) == ("pydra-entry@1", "results@1")
    members = fmt.members()
    sha = "ab" * 32
    for name in (
        "entry.json",
        "context.json",
        f"{sha}.npy",
        f"{sha}.bin",
        f"{sha}.pkl",
        "files",
    ):
        assert members.fullmatch(name), name
    for name in (
        "Result",
        "_job.pklz",
        "_error.pklz",
        "x.npy",
        f"{sha}.txt",
        "q-" + sha,
    ):
        assert not members.fullmatch(name), name
    assert not fmt.aside_names().fullmatch("q-" + uuid.uuid4().hex)


# -- section 6.4 encoding table ----------------------------------------------------------


def test_scalars_are_json(root):
    out = {
        "n": None,
        "t": True,
        "i": 3,
        "s": "x",
        "big": 2**70,
        "neg": -(2**64),
        "f": 1.0,
    }
    enc = encoded(root, out)
    assert enc == out and type(enc["f"]) is float
    got = read(root).outputs
    assert got == out
    assert {k: type(v) for k, v in got.items()} == {k: type(v) for k, v in out.items()}


def test_float_nonfinite_tags(root):
    enc = encoded(
        root, {"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 0.5}
    )
    assert enc == {
        "a": {"$float": "nan"},
        "b": {"$float": "inf"},
        "c": {"$float": "-inf"},
        "d": 0.5,
    }
    got = read(root).outputs
    assert (
        got["a"] != got["a"] and got["b"] == float("inf") and got["c"] == float("-inf")
    )


def test_float_negative_zero_keeps_sign(root):
    got = roundtrip(root, {"z": -0.0})
    assert str(got["z"]) == "-0.0"


def test_complex(root):
    value = {"c": complex(1.5, -2), "n": complex(float("nan"), float("inf"))}
    enc = encoded(root, value)
    assert enc["c"] == {"$complex": [{"$float": (1.5).hex()}, {"$float": (-2.0).hex()}]}
    got = read(root).outputs
    assert got["c"] == complex(1.5, -2)
    assert got["n"].real != got["n"].real and got["n"].imag == float("inf")


def test_bytes_member(root):
    data = b"\x00\x01payload"
    sha = hashlib.sha256(data).hexdigest()
    enc = encoded(root, {"b": data})
    assert enc["b"] == {"$bytes": sha}
    assert (entry_path(root) / f"{sha}.bin").read_bytes() == data
    assert read(root).outputs["b"] == data


def test_list_and_tuple_decode_to_declared_type(root):
    out = {"a": 1, "b": [1.0, 2.5], "c": (7, "q")}
    enc = encoded(root, out)
    assert enc["b"] == [1.0, 2.5] and enc["c"] == [7, "q"]
    typed = read(root, task=Typed(x=1)).outputs
    assert typed["c"] == (7, "q") and isinstance(typed["c"], tuple)
    assert typed["b"] == [1.0, 2.5]
    plain = read(root, task=None).outputs
    assert plain["c"] == [7, "q"]


def test_set_order_is_canonical(root, tmp_path):
    a = encoded(root, {"s": {3, 1, "b", "a", 2.5}})["s"]
    assert list(a) == ["$set"] and len(a["$set"]) == 5
    other = tmp_path / "o"
    other.mkdir()
    with StorageDir.open_root(other, StorePolicy(), trust=TRUST) as r2:
        b = encoded(r2, {"s": frozenset(["a", 2.5, "b", 1, 3])})["s"]
    assert a == b
    canon = [dg.jcs(codecs.typed(v)) for v in a["$set"]]
    assert canon == sorted(canon)
    got = read(root).outputs["s"]
    assert got == {3, 1, "b", "a", 2.5} and isinstance(got, set)


def test_set_of_tuples_and_frozensets_decodes(root):
    value = {(1, 2), (3, "x")}
    nested = {frozenset([1]), frozenset([2, 3])}
    got = roundtrip(root, {"t": value, "n": nested})
    assert got["t"] == value and got["n"] == nested


def test_dollar_keys_are_escaped(root):
    value = {"$set": 1, "$$x": 2, "plain": 3, "$": 4, "$codec": 5}
    enc = encoded(root, {"d": value})["d"]
    assert enc == {"$$set": 1, "$$$x": 2, "plain": 3, "$$": 4, "$$codec": 5}
    assert read(root).outputs["d"] == value


def test_dict_with_other_keys(root):
    value = {2: "b", 1: "a", (1, 2): "t", None: "n", "s": "str"}
    enc = encoded(root, {"d": value})["d"]
    assert list(enc) == ["$dict"]
    keys = [pair[0] for pair in enc["$dict"]]
    assert len(keys) == 5
    assert read(root).outputs["d"] == value


def test_dict_order_does_not_change_digest(root, tmp_path):
    d1 = publish(root, {"d": {1: "a", 2: "b"}, "e": {"x": 1, "y": 2}})
    other = tmp_path / "o"
    other.mkdir()
    with StorageDir.open_root(other, StorePolicy(), trust=TRUST) as r2:
        d2 = publish(r2, {"e": {"y": 2, "x": 1}, "d": {2: "b", 1: "a"}}, writer="other")
    assert d1 == d2


def test_naive_datetime_refused_under_shared(root):
    naive = datetime.datetime(2024, 1, 2, 3, 4, 5)
    shared = PydraEntryV1(pickle=False, trust="shared")
    with pytest.raises(UnencodableOutput) as err:
        publish(root, {"when": naive}, fmt=shared)
    assert err.value.field == "when" and err.value.type is datetime.datetime
    aware = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
    assert encoded(root, {"when": aware}, fmt=shared)["when"] == {
        "$datetime": aware.isoformat()
    }


def test_datetime_and_date_round_trip(root):
    naive = datetime.datetime(2024, 1, 2, 3, 4, 5, 6)
    aware = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=2))
    )
    day = datetime.date(2024, 2, 29)
    enc = encoded(root, {"n": naive, "a": aware, "d": day})
    assert enc["n"] == {"$datetime": naive.isoformat()}
    got = read(root).outputs
    assert got["n"] == naive and got["a"] == aware and got["d"] == day
    assert type(got["d"]) is datetime.date and type(got["n"]) is datetime.datetime
    assert got["a"].utcoffset() == aware.utcoffset()


def test_path_is_a_value_not_a_file(root, tmp_path):
    src = tmp_path / "data.txt"
    src.write_text("content")
    enc = encoded(root, {"p": src})
    assert enc["p"] == {"$path": str(src)}
    assert not (entry_path(root) / "files").exists()
    got = read(root).outputs["p"]
    assert got == src and isinstance(got, Path)


def test_array_is_a_canonical_npy_member(root):
    arr = np.arange(12, dtype=">i4").reshape(3, 4)
    sha = hashlib.sha256(canonical_npy(arr)).hexdigest()
    assert digest(arr) == f"npy@1:{sha}"
    enc = encoded(root, {"a": arr})
    assert enc["a"] == {"$array": sha}
    assert (entry_path(root) / f"{sha}.npy").read_bytes() == canonical_npy(arr)
    got = read(root).outputs["a"]
    assert type(got) is np.ndarray and got.dtype == np.dtype("<i4")
    assert np.array_equal(got, arr) and got.shape == (3, 4)


def test_fortran_and_strided_arrays_share_a_member(root):
    base = np.arange(20, dtype="<f8").reshape(4, 5)
    enc = encoded(
        root, {"a": base, "b": np.asfortranarray(base), "c": base[::2, ::2].copy()}
    )
    assert enc["a"] == enc["b"] != enc["c"]
    got = read(root).outputs
    assert np.array_equal(got["b"], base)


def _reference_arrays():
    out = [
        ("masked", np.ma.MaskedArray(np.arange(4), mask=[0, 1, 0, 0])),
        ("matrix", np.matrix(np.arange(4))),
        ("object", np.array(["a", {}], dtype=object)),
        ("structured-object", np.zeros(2, dtype=[("a", "O")])),
    ]
    if hasattr(np.dtypes, "StringDType"):
        out.append(
            ("stringdtype", np.array(["a", "bc"], dtype=np.dtypes.StringDType()))
        )
    return out


@pytest.mark.parametrize(
    "name,value", _reference_arrays(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_reference_arrays_are_unencodable_without_pickle(root, name, value):
    with pytest.raises(UnencodableOutput) as err:
        publish(root, {"v": value})
    assert err.value.field == "v" and err.value.type is type(value)


@posix_only
@pytest.mark.parametrize(
    "name,value", _reference_arrays()[:2], ids=lambda v: v if isinstance(v, str) else ""
)
def test_subclass_arrays_are_pickled_where_allowed(root, name, value):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": value}, fmt=fmt)
    got = read(root, fmt=fmt).outputs["v"]
    assert type(got) is type(value)
    assert np.array_equal(np.asarray(got), np.asarray(value))


def test_npscalar_row(root):
    values = {
        "f32": np.float32(1),
        "f64": np.float64(1),
        "i8": np.int8(-5),
        "u64": np.uint64(2**63 + 5),
        "b": np.bool_(True),
        "c": np.complex64(1.5 - 2j),
        "dt": np.datetime64("2024-01-02T03:04", "m"),
        "td": np.timedelta64(7, "s"),
        "big": np.dtype(">f4").type(2.5),
    }
    enc = encoded(root, values)
    assert enc["f32"] == {"$npscalar": ["<f4", "0x1.0000000000000p+0"]}
    assert enc["f64"] == {"$npscalar": ["<f8", "0x1.0000000000000p+0"]}
    assert enc["i8"] == {"$npscalar": ["|i1", "-5"]}
    assert enc["b"] == {"$npscalar": ["|b1", "true"]}
    assert enc["big"] == {"$npscalar": ["<f4", (2.5).hex()]}
    assert enc["u64"]["$npscalar"][1] == str(2**63 + 5)
    got = read(root).outputs
    for name, value in values.items():
        assert got[name] == value, name
        assert got[name].dtype == np.dtype(value.dtype.newbyteorder("<")), name


def test_longdouble_is_unencodable(root):
    for value in (np.longdouble(1), np.clongdouble(1)):
        with pytest.raises(UnencodableOutput):
            publish(root, {"v": value})


def test_fileset_nifti_pair_keeps_keys(root, tmp_path, tags):
    src = tmp_path / "src"
    src.mkdir()
    (src / "scan.hdr").write_bytes(b"header")
    (src / "scan.img").write_bytes(b"image-data")
    value = Pair([src / "scan.hdr", src / "scan.img"])
    enc = encoded(root, {"scan": value})["scan"]
    hdr_sha = hashlib.sha256(b"header").hexdigest()
    img_sha = hashlib.sha256(b"image-data").hexdigest()
    set_name = fileset_digest("test.entry.Pair", [(".hdr", hdr_sha), (".img", img_sha)])
    set_hex = set_name.split(":", 1)[1]
    assert enc == {
        "$fileset": "test.entry.Pair",
        "files": [
            [".hdr", f"files/{set_hex}/scan.hdr"],
            [".img", f"files/{set_hex}/scan.img"],
        ],
    }
    assert digest(value) == set_name
    got = read(root).outputs["scan"]
    assert type(got) is Pair
    assert [k for k, _ in got.byte_chunks()] == [".hdr", ".img"]
    base = entry_path(root) / "files" / set_hex
    assert (base / "scan.hdr").read_bytes() == b"header"
    assert all(Path(p).parent == base for p in got.fspaths)
    members = entry_json(root)["members"]
    assert members[f"files/{set_hex}/scan.hdr"] == {"sha256": hdr_sha, "size": 6}


def test_single_file_key_is_dot(root, tmp_path, tags):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    enc = encoded(root, {"f": Tagged(f)})["f"]
    assert [k for k, _ in enc["files"]] == ["."]
    assert enc["files"][0][1].endswith("/one.dat")
    got = read(root).outputs["f"]
    assert [k for k, _ in got.byte_chunks()] == ["."]


def test_directory_output_sits_under_its_name(root, tmp_path, tags):
    d = tmp_path / "tree"
    (d / "sub" / "deep").mkdir(parents=True)
    (d / "top.txt").write_bytes(b"t")
    (d / "sub" / "deep" / "x.txt").write_bytes(b"x")
    (d / ".hidden").write_bytes(b"h")
    value = Directory(d)
    enc = encoded(root, {"d": value})["d"]
    keys = [k for k, _ in enc["files"]]
    assert keys == sorted(keys) == [".hidden", "sub/deep/x.txt", "top.txt"]
    set_hex = digest(value).split(":", 1)[1]
    assert enc["files"][1][1] == f"files/{set_hex}/tree/sub/deep/x.txt"
    got = read(root).outputs["d"]
    assert isinstance(got, Directory)
    assert (
        Path(next(iter(got.fspaths))) == entry_path(root) / "files" / set_hex / "tree"
    )
    assert [k for k, _ in got.byte_chunks()] == keys


def test_fileset_is_copied_never_linked(root, tmp_path, tags):
    src = tmp_path / "dataset.dat"
    src.write_bytes(b"precious")
    os.chmod(src, 0o640)
    link = tmp_path / "second-name.dat"
    os.link(src, link)
    before = (src.stat().st_ino, stat.S_IMODE(src.stat().st_mode))
    publish(root, {"f": Tagged(src)})
    assert (src.stat().st_ino, stat.S_IMODE(src.stat().st_mode)) == before
    (member,) = (entry_path(root) / "files").glob("*/dataset.dat")
    assert member.stat().st_ino != src.stat().st_ino and member.stat().st_nlink == 1
    src.write_bytes(b"changed!")
    assert member.read_bytes() == b"precious"


def test_symlinked_file_output_is_copied_by_content(root, tmp_path, tags):
    real = tmp_path / "real.dat"
    real.write_bytes(b"target")
    link = tmp_path / "alias.dat"
    link.symlink_to(real)
    publish(root, {"f": Tagged(link)})
    (member,) = (entry_path(root) / "files").glob("*/alias.dat")
    assert member.read_bytes() == b"target" and not member.is_symlink()


def test_fifo_in_directory_output_is_unencodable(root, tmp_path, tags):
    d = tmp_path / "tree"
    d.mkdir()
    (d / "ok.txt").write_bytes(b"ok")
    os.mkfifo(d / "pipe")
    with pytest.raises(UnencodableOutput) as err:
        publish(root, {"d": Directory(d)})
    assert err.value.field == "d"


def test_two_outputs_with_one_set_share_a_directory(root, tmp_path, tags):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    enc = encoded(root, {"a": Tagged(f), "b": Tagged(f)})
    assert enc["a"] == enc["b"]
    assert len(list((entry_path(root) / "files").iterdir())) == 1
    read(root)


def test_unregistered_fileset_is_unencodable(root, tmp_path):
    f = tmp_path / "x.dat"
    f.write_bytes(b"1")
    with pytest.raises(UnencodableOutput) as err:
        publish(root, {"f": Tagged(f)})
    assert err.value.type is Tagged


def test_empty_directory_is_unencodable(root, tmp_path, tags):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(UnencodableOutput):
        publish(root, {"d": Directory(d)})


@pytest.mark.skip(reason="R10a registers the codec")
def test_absent_row(root):
    raise AssertionError("R10a registers the Absent codec")


class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)


def _register_point():
    register_codec(
        Point,
        "test.entry.Point",
        lambda p: {"x": p.x, "y": p.y},
        lambda d: Point(d["x"], d["y"]),
    )


def test_registered_codec_round_trips(root):
    _register_point()
    enc = encoded(root, {"p": Point(1, 2.5), "l": [Point(0, 0)]})
    assert enc["p"] == {"$codec": "test.entry.Point", "value": {"x": 1, "y": 2.5}}
    got = read(root).outputs
    assert got["p"] == Point(1, 2.5) and got["l"] == [Point(0, 0)]


def test_unregistered_codec_tag_is_format_unknown_with_no_task(root, monkeypatch):
    _register_point()
    publish(root, {"p": Point(1, 2)})
    monkeypatch.setattr(
        dg, "_BY_TAG", {k: v for k, v in dg._BY_TAG.items() if k != "test.entry.Point"}
    )
    with pytest.raises(EntryFormatUnknown):
        read(root, task=None)
    assert entry_path(root).exists()


def test_unregistered_fileset_tag_is_format_unknown(root, tmp_path, tags, monkeypatch):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    publish(root, {"f": Tagged(f)})
    monkeypatch.setattr(
        dg, "_BY_TAG", {k: v for k, v in dg._BY_TAG.items() if k != "test.entry.Tagged"}
    )
    with pytest.raises(EntryFormatUnknown):
        read(root)


def test_digest_registered_type_is_not_emitted_as_digest(root):
    class Q:
        pass

    register_digest(Q, lambda q: "json@1:" + "0" * 64)
    with pytest.raises(UnencodableOutput):
        publish(root, {"q": Q()})
    _register_point()
    register_digest(Point, lambda p: "json@1:" + "1" * 64)
    enc = encoded(root, {"p": Point(1, 1)})
    assert "$digest" not in json.dumps(enc)


def test_reserved_digest_tag_is_corrupt(root):
    publish(root, {"a": 1})
    edit_entry(root, lambda d: d["outputs"].update(a={"$digest": "json@1:" + "0" * 64}))
    with pytest.raises(CorruptEntry):
        read(root)


def test_decoding_never_imports(root, tmp_path, monkeypatch):
    mod_dir = tmp_path / "mods"
    mod_dir.mkdir()
    marker = tmp_path / "imported"
    (mod_dir / "never_imported_mod.py").write_text(
        f"open({str(marker)!r}, 'w').write('x')\nclass Evil:\n    pass\n"
    )
    monkeypatch.syspath_prepend(str(mod_dir))
    publish(root, {"a": 1})
    for tag_doc in (
        {"$codec": "never_imported_mod.Evil", "value": 1},
        {"$fileset": "never_imported_mod.Evil", "files": []},
        {"$codec": "os.system", "value": "true"},
        {"$unknowntag": 1},
    ):
        edit_entry(root, lambda d, t=tag_doc: d["outputs"].update(a=t))
        with pytest.raises(EntryFormatUnknown):
            read(root, task=None)
    assert "never_imported_mod" not in sys.modules and not marker.exists()


# -- section 6.2 binding to the address and the members ----------------------------------


def test_stage_returns_the_recorded_output_digest(root, tmp_path):
    out = {"a": np.arange(3), "b": b"x", "c": [1, 2], "d": {"k": 1.0}}
    d1 = publish(root, out, writer="w1")
    doc = entry_json(root)
    assert doc["digest"] == d1 and doc["writer"] == "w1"
    assert doc["format"] == "pydra-entry@1" and doc["checksum"] == CHECKSUM
    assert doc["fingerprint"] == FP and doc["key"] is None
    other = tmp_path / "o"
    other.mkdir()
    with StorageDir.open_root(other, StorePolicy(), trust=TRUST) as r2:
        assert publish(r2, out, writer="w2") == d1
        assert publish(r2, {**out, "c": [1, 3]}, context={**CTX, "n": 2}) != d1


def test_digest_tells_one_from_one_point_zero(root, tmp_path):
    d1 = publish(root, {"v": 1})
    other = tmp_path / "o"
    other.mkdir()
    with StorageDir.open_root(other, StorePolicy(), trust=TRUST) as r2:
        assert publish(r2, {"v": 1.0}) != d1


def test_read_returns_the_stored_entry(root):
    key = {"scheme": "test", "n": 1}
    d = publish(root, {"a": 1}, key=key, writer="tok")
    got = read(root)
    assert isinstance(got, StoredEntry)
    assert got.outputs == {"a": 1} and got.key == key and got.digest == d
    assert got.context == CTX
    assert got.ref.cdir == CHECKSUM and got.ref.fingerprint == FP
    assert got.ref.root == str(root.path)
    with pytest.raises(AttributeError):
        got.digest = "x"  # type: ignore[misc]


def test_context_is_read_with_its_bytes(root):
    publish(root, {"a": 1})
    entry = root
    for part in entry_path(root).relative_to(root.path).parts:
        entry = entry.child(part)
    with entry:
        ctx, raw = PydraEntryV1().context(entry)
    assert ctx == CTX
    assert raw == (entry_path(root) / "context.json").read_bytes()
    assert json.loads(raw)["format"] == "pydra-entry@1"
    assert IDENT.fingerprint(ctx) == FP


def test_a_real_job_context_round_trips_through_the_entry(root, tmp_path):
    @python.define
    def Inc(a: int) -> int:
        return a + 1

    with Submitter(worker="debug", cache_root=tmp_path / "cache") as sub:
        job = Job(task=Inc(a=1), name="j", submitter=sub)
        store = PickleStore(tmp_path / "other")
        store.identity = IDENT
        job.store = store
        context = job.context
        assert context == IDENT.context(job, "lookup")
    fp = IDENT.fingerprint(context)
    publish(root, {"out": 2}, context=context, checksum=job.checksum)
    got = read(root, checksum=job.checksum, fp=fp)
    assert got.outputs == {"out": 2}
    assert IDENT.fingerprint(got.context) == fp
    assert got.context == json.loads(json.dumps(context))


def test_member_modes_follow_the_policy(root):
    publish(root, {"a": np.arange(3), "b": b"x"})
    files = [p for p in entry_path(root).iterdir() if p.is_file()]
    assert len(files) == 4  # entry.json, context.json, one npy, one bin
    if not IS_WINDOWS:
        assert {stat.S_IMODE(p.stat().st_mode) for p in files} == {0o444}


def test_misfiled_by_checksum_and_by_fingerprint(root):
    publish(root, {"a": 1})
    entry = root
    for part in entry_path(root).relative_to(root.path).parts:
        entry = entry.child(part)
    fmt = PydraEntryV1()
    with entry:
        with pytest.raises(MisfiledEntry):
            fmt.read(entry, "python-other", FP, None, TRUST)
        with pytest.raises(MisfiledEntry):
            fmt.read(entry, CHECKSUM, "0" * 64, None, TRUST)
    assert issubclass(MisfiledEntry, CorruptEntry)


def test_unknown_format_is_not_corrupt(root):
    publish(root, {"a": 1})
    edit_entry(root, lambda d: d.update(format="pydra-entry@2"), redigest=False)
    with pytest.raises(EntryFormatUnknown):
        read(root)


@pytest.mark.parametrize(
    "bad", ["not json", "[]", '{"format": "pydra-entry@1"}', "\xff\xfe"]
)
def test_garbled_entry_json_is_corrupt(root, bad):
    publish(root, {"a": 1})
    path = entry_path(root) / "entry.json"
    os.chmod(path, 0o644)
    path.write_bytes(bad.encode("latin-1"))
    with pytest.raises(CorruptEntry):
        read(root)


def test_nan_in_entry_json_is_corrupt(root):
    publish(root, {"a": 1})
    path = entry_path(root) / "entry.json"
    os.chmod(path, 0o644)
    path.write_text(
        path.read_text().replace('"a": 1', '"a": NaN').replace('"a":1', '"a":NaN')
    )
    with pytest.raises(CorruptEntry):
        read(root)


def test_changed_outputs_fail_the_output_digest(root):
    publish(root, {"a": 1, "b": "x"})
    edit_entry(root, lambda d: d["outputs"].update(a=2), redigest=False)
    with pytest.raises(CorruptEntry):
        read(root)


def test_read_checks_member_digests(root):
    arr = np.arange(100, dtype="<i4")
    sha = hashlib.sha256(canonical_npy(arr)).hexdigest()
    publish(root, {"a": arr, "b": b"bytes"})
    member = entry_path(root) / f"{sha}.npy"
    original = member.read_bytes()
    os.chmod(member, 0o644)
    # the same size, one bit changed: only the digest can tell
    flipped = bytearray(original)
    flipped[-1] ^= 1
    member.write_bytes(bytes(flipped))
    with pytest.raises(CorruptEntry):
        read(root)
    member.write_bytes(original[:-4])
    with pytest.raises(CorruptEntry):
        read(root)
    member.write_bytes(original)
    assert np.array_equal(read(root).outputs["a"], arr)
    member.unlink()
    with pytest.raises(CorruptEntry):
        read(root)


def test_size_policy_checks_sizes_only(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    policy = StorePolicy(verify_files="size")
    if IS_WINDOWS:
        pytest.skip("verify_files='size' needs a private root")
    with StorageDir.open_root(d, policy, trust="private") as root:
        arr = np.arange(100, dtype="<i4")
        sha = hashlib.sha256(canonical_npy(arr)).hexdigest()
        publish(root, {"a": arr})
        member = entry_path(root) / f"{sha}.npy"
        original = member.read_bytes()
        os.chmod(member, 0o644)
        flipped = bytearray(original)
        flipped[-1] ^= 1
        member.write_bytes(bytes(flipped))
        read(root, trust="private")  # same size: accepted under "size"
        member.write_bytes(original[:-4])
        with pytest.raises(CorruptEntry):
            read(root, trust="private")


def test_unlisted_and_symlinked_members_are_corrupt(root, tmp_path):
    publish(root, {"a": 1})
    stray = entry_path(root) / ("ab" * 32 + ".bin")
    stray.write_bytes(b"smuggled")
    with pytest.raises(CorruptEntry):
        read(root)
    stray.unlink()
    read(root)
    if IS_WINDOWS:
        return
    ctx = entry_path(root) / "context.json"
    raw = ctx.read_bytes()
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(raw)
    ctx.unlink()
    ctx.symlink_to(elsewhere)
    with pytest.raises(CorruptEntry):
        read(root)


def test_unknown_member_names_are_another_versions(root):
    publish(root, {"a": 1})
    (entry_path(root) / "payload.cbor").write_bytes(b"x")
    with pytest.raises(EntryFormatUnknown):
        read(root)


def test_unreferenced_listed_member_is_corrupt(root):
    publish(root, {"a": 1})
    data = b"extra"
    sha = hashlib.sha256(data).hexdigest()
    (entry_path(root) / f"{sha}.bin").write_bytes(data)
    edit_entry(
        root, lambda d: d["members"].update({f"{sha}.bin": {"sha256": sha, "size": 5}})
    )
    with pytest.raises(CorruptEntry):
        read(root)


def test_referenced_but_unlisted_member_is_corrupt(root):
    publish(root, {"b": b"bytes"})
    edit_entry(root, lambda d: d["members"].clear())
    with pytest.raises(CorruptEntry):
        read(root)


def test_read_checks_fileset_set_name_and_keys(root, tmp_path, tags):
    src = tmp_path / "src"
    (src / "d" / "sub").mkdir(parents=True)
    (src / "d" / "a.txt").write_bytes(b"a")
    (src / "d" / "sub" / "b.txt").write_bytes(b"b")
    publish(root, {"d": Directory(src / "d")})
    good = entry_json(root)
    files_dir = entry_path(root) / "files"
    (set_name,) = [p.name for p in files_dir.iterdir()]

    # the set name must be the file@1 of the recorded tag, keys and member digests
    def rename_key(doc):
        doc["outputs"]["d"]["files"][0][0] = "renamed.txt"

    edit_entry(root, rename_key)
    with pytest.raises(CorruptEntry):
        read(root)

    def wrong_tag(doc):
        doc["outputs"]["d"] = {**good["outputs"]["d"], "$fileset": "test.entry.Tagged"}

    edit_entry(root, wrong_tag)
    with pytest.raises(CorruptEntry):
        read(root)
    edit_entry(root, lambda d: d["outputs"].update(d=good["outputs"]["d"]))
    read(root)

    # a file nobody recorded, inside the Directory's tree: the keys of the value differ
    stray = files_dir / set_name / "d" / "stray.txt"
    stray.write_bytes(b"stray")
    with pytest.raises(CorruptEntry):
        read(root)
    stray.unlink()
    read(root)

    # a set directory that is not the digest of its content
    def other_set(doc):
        name = "0" * 64
        doc["outputs"]["d"]["files"] = [
            [k, p.replace(set_name, name)] for k, p in doc["outputs"]["d"]["files"]
        ]
        doc["members"] = {
            k.replace(set_name, name): v for k, v in doc["members"].items()
        }

    os.rename(files_dir / set_name, files_dir / ("0" * 64))
    edit_entry(root, other_set)
    with pytest.raises(CorruptEntry):
        read(root)


@pytest.mark.parametrize(
    "path",
    [
        "../escape",
        "/abs/path",
        "files/zz/x",
        "files//x",
        "files/" + "0" * 64 + "/../../x",
    ],
)
def test_fileset_paths_cannot_leave_their_set(root, tmp_path, tags, path):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    publish(root, {"f": Tagged(f)})
    edit_entry(root, lambda d: d["outputs"]["f"]["files"].__setitem__(0, [".", path]))
    with pytest.raises(CorruptEntry):
        read(root)


# -- trust ------------------------------------------------------------------------------


def test_writer_pickles_only_where_allowed(root):
    value = {"v": np.ma.MaskedArray(np.arange(3), mask=[0, 1, 0])}
    for fmt in (
        PydraEntryV1(pickle=False, trust="private"),
        PydraEntryV1(pickle=False, trust="trusted"),
        PydraEntryV1(pickle=True, trust="shared"),
    ):
        with pytest.raises(UnencodableOutput):
            publish(root, value, fmt=fmt)


@posix_only
def test_shared_never_opens_pkl(root, tmp_path, monkeypatch):
    marker = tmp_path / "unpickled"

    class Payload:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"a": 1, "p": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    (pkl,) = list(entry_path(root).glob("*.pkl"))
    os.chmod(pkl, 0o644)
    pkl.write_bytes(pickle.dumps(Payload()))
    edit_entry(
        root,
        lambda d: d["members"].update(
            {
                pkl.name: {
                    "sha256": hashlib.sha256(pkl.read_bytes()).hexdigest(),
                    "size": pkl.stat().st_size,
                }
            }
        ),
    )
    opened: list[str] = []
    real = StorageDir.open_read

    def spy(self, name):
        opened.append(name)
        return real(self, name)

    monkeypatch.setattr(StorageDir, "open_read", spy)
    with pytest.raises(EntryNotTrusted):
        read(root, trust="shared")
    assert not any(n.endswith(".pkl") for n in opened) and not marker.exists()
    assert entry_path(root).exists()


@posix_only
def test_private_and_trusted_read_a_pickle_member(root):
    fmt = PydraEntryV1(pickle=True, trust="private")
    value = np.ma.MaskedArray(np.arange(4), mask=[0, 1, 0, 0])
    publish(root, {"v": value}, fmt=fmt)
    enc = entry_json(root)["outputs"]["v"]
    assert list(enc) == ["$pickle", "cloudpickle", "python"]
    assert enc["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
    for trust in ("private", "trusted"):
        got = read(root, fmt=fmt, trust=trust).outputs["v"]
        assert isinstance(got, np.ma.MaskedArray) and got.mask.tolist() == [
            False,
            True,
            False,
            False,
        ]


@posix_only
def test_pickle_from_another_interpreter_is_not_served(root):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    edit_entry(root, lambda d: d["outputs"]["v"].update(python="2.7"))
    with pytest.raises(EntryFormatUnknown):
        read(root, fmt=fmt, trust="private")


@posix_only
def test_private_ownership_walk_rejects_group_writable_shard(root, monkeypatch):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    path = entry_path(root)
    rel = list(path.relative_to(root.path).parts)
    (pkl,) = list(path.glob("*.pkl"))
    walk = [*rel, pkl.name]
    check_private(root, walk)
    read(root, fmt=fmt, trust="private")
    shard = path.parent.parent
    os.chmod(shard, 0o775)
    with pytest.raises(EntryNotTrusted) as err:
        check_private(root, walk)
    assert shard.name in str(err.value)
    with pytest.raises(EntryNotTrusted):
        read(root, fmt=fmt, trust="private")
    assert read(root, fmt=fmt, trust="trusted").outputs["v"].shape == (2,)
    os.chmod(shard, 0o755)
    os.chmod(path, 0o757)
    with pytest.raises(EntryNotTrusted):
        check_private(root, walk)
    os.chmod(path, 0o755)
    check_private(root, walk)


@posix_only
def test_private_walk_rejects_another_owner_and_symlinks(root, monkeypatch, tmp_path):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    path = entry_path(root)
    (pkl,) = list(path.glob("*.pkl"))
    walk = [*path.relative_to(root.path).parts, pkl.name]
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(root.path).st_uid + 1)
    with pytest.raises(EntryNotTrusted):
        check_private(root, walk)
    monkeypatch.undo()
    os.chmod(pkl, 0o666)
    with pytest.raises(EntryNotTrusted):
        check_private(root, walk)
    os.chmod(pkl, 0o444)
    shard = path.parent.parent
    moved = tmp_path / "moved"
    os.rename(shard, moved)
    os.symlink(moved, shard)
    with pytest.raises(EntryNotTrusted):
        check_private(root, walk)


# -- encodability at hand-out -----------------------------------------------------------


class Opaque:
    pass


def _task(out_type):
    @python.define(outputs={"out": out_type})
    def F(x: int) -> out_type:  # type: ignore[valid-type]
        return x

    return F


def test_unencodable_declared_type_at_hand_out():
    for bad in (
        np.ma.MaskedArray,
        Opaque,
        ty.List[Opaque],
        ty.Optional[np.matrix],
        ty.Dict[str, Opaque],
        ty.Tuple[int, Opaque],
        np.longdouble,
        Tagged,
    ):
        task = _task(bad)
        with pytest.raises(UnencodableOutput) as err:
            check_encodable(task, allow_pickle=False)
        assert err.value.field == "out", bad
        check_encodable(task, allow_pickle=True)


def test_encodable_declared_types_pass():
    register_codec(Opaque, "test.entry.Opaque", lambda o: 1, lambda v: Opaque())
    good = (
        int,
        float,
        str,
        bytes,
        bool,
        complex,
        type(None),
        ty.Optional[int],
        ty.Union[int, str],
        ty.List[int],
        ty.List[ty.List[float]],
        ty.Tuple[int, ...],
        ty.Set[str],
        ty.FrozenSet[int],
        ty.Dict[str, ty.List[int]],
        ty.Dict[int, str],
        ty.Any,
        object,
        Path,
        datetime.datetime,
        datetime.date,
        np.ndarray,
        np.float32,
        np.int64,
        np.bool_,
        np.datetime64,
        Opaque,
        ty.List[Opaque],
        list,
        dict,
    )
    for tp in good:
        check_encodable(_task(tp), allow_pickle=False)


def test_fileset_types_need_a_registered_tag():
    with pytest.raises(UnencodableOutput):
        check_encodable(_task(Tagged), allow_pickle=False)
    register_fileset_tag(Tagged, "test.entry.Tagged")
    check_encodable(_task(Tagged), allow_pickle=False)
    check_encodable(_task(ty.List[Tagged]), allow_pickle=False)


def test_check_encodable_accepts_a_task_instance_and_names_the_field():
    task = Typed(x=1)
    check_encodable(task, allow_pickle=False)

    @python.define(outputs={"good": int, "bad": Opaque})
    def G(x: int) -> ty.Tuple[int, Opaque]:
        return x, Opaque()

    with pytest.raises(UnencodableOutput) as err:
        check_encodable(G, allow_pickle=False)
    assert err.value.field == "bad" and err.value.type is Opaque
    assert "bad" in str(err.value)
