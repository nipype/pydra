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
from pydra.engine.absent import Absent
from pydra.engine.job import Job
from pydra.engine.portable import codecs
from pydra.engine.portable.codecs import type_is_encodable
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

# `trusted` on Windows unless CI sets PYDRA_TEST_PORTABLE_TRUST (see test_portable_store)
TRUST = os.environ.get("PYDRA_TEST_PORTABLE_TRUST") or (
    "trusted" if IS_WINDOWS else "private"
)
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
        return fmt.read(entry, checksum, fp, task, trust, root=root)


def _keys(value) -> list:
    """The ``byte_chunks`` keys of a file set, with POSIX separators."""
    return [k.replace(os.sep, "/") for k, _ in value.byte_chunks()]


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


def test_float_negative_zero_is_plain_json_and_keeps_sign(root):
    assert encoded(root, {"z": -0.0})["z"] == 0.0
    assert str(read(root).outputs["z"]) == "-0.0"
    assert "-0.0" in (entry_path(root) / "entry.json").read_text()


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
    assert enc["b"] == [1.0, 2.5] and enc["c"] == {"$tuple": [7, "q"]}
    typed = read(root, task=Typed(x=1)).outputs
    assert typed["c"] == (7, "q") and isinstance(typed["c"], tuple)
    assert typed["b"] == [1.0, 2.5]
    plain = read(root, task=None).outputs
    assert plain["c"] == (7, "q") and plain["b"] == [
        1.0,
        2.5,
    ]  # no declared type: as given


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


def test_a_fileformats_file_set_is_stored_under_its_mime_like_name(root, tmp_path):
    from fileformats.application import Json

    src = tmp_path / "data.json"
    src.write_text("{}")
    enc = encoded(root, {"j": Json(src)})
    assert enc["j"]["$fileset"] == "application/json"
    got = read(root).outputs["j"]
    assert isinstance(got, Json) and got.fspath.read_text() == "{}"
    assert got.fspath.is_relative_to(entry_path(root))


class _Unlisted(File):
    """A file set class outside the fileformats packages, with no registered tag"""


def test_a_file_set_without_a_tag_is_never_pickled(root, tmp_path):
    src = tmp_path / "data.txt"
    src.write_text("x")
    fmt = PydraEntryV1(pickle=True, trust=TRUST)
    with pytest.raises(UnencodableOutput, match="never pickled"):
        publish(root, {"f": _Unlisted(src)}, fmt=fmt)


def _work_dir(root, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    return work, StorageDir.open_root(work, StorePolicy(), trust=TRUST)


def test_a_path_inside_the_work_directory_is_stored_as_a_file_set(root, tmp_path):
    work, handle = _work_dir(root, tmp_path)
    (work / "out.txt").write_text("hello")
    (work / "outdir").mkdir()
    (work / "outdir" / "a.txt").write_text("in a dir")
    with handle:
        enc = encoded(
            root, {"f": work / "out.txt", "d": work / "outdir"}, files_from=handle
        )
    assert enc["f"]["$path"]["$fileset"] == "generic/file"
    assert enc["d"]["$path"]["$fileset"] == "generic/directory"
    got = read(root).outputs
    assert isinstance(got["f"], Path) and got["f"].read_text() == "hello"
    assert got["f"].is_relative_to(entry_path(root))
    assert (got["d"] / "a.txt").read_text() == "in a dir"


def test_a_path_outside_the_work_directory_is_a_value(root, tmp_path):
    work, handle = _work_dir(root, tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("kept where it is")
    with handle:
        enc = encoded(root, {"p": outside}, files_from=handle)
    assert enc["p"] == {"$path": str(outside)}


def test_a_string_naming_a_file_in_the_work_directory_warns_once(root, tmp_path):
    work, handle = _work_dir(root, tmp_path)
    (work / "out.txt").write_text("hello")
    with handle, pytest.warns(UserWarning, match="pathlib.Path") as caught:
        publish(
            root,
            {"s": [str(work / "out.txt"), str(work / "out.txt")]},
            files_from=handle,
        )
    assert len(caught) == 1


def test_untyped_tuples_round_trip(root):
    out = roundtrip(root, {"pair": (1, 2), "nested": {"k": (2, 2), "l": [(2,)]}})
    assert out["pair"] == (1, 2) and type(out["pair"]) is tuple
    assert out["nested"] == {"k": (2, 2), "l": [(2,)]}
    assert type(out["nested"]["k"]) is tuple


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
    assert _keys(got) == [".hdr", ".img"]
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
    assert _keys(got) == ["."]


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
    assert _keys(got) == keys


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
    if not hasattr(os, "mkfifo"):
        pytest.skip("no FIFOs on this platform")
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


def test_absent_row(root):
    outputs = {
        "a": Absent(cause="negative", source="Maybe"),
        "b": [1, Absent(cause="no-list")],
        "c": {"k": Absent(cause="not-routed", source="wf/D")},
    }
    enc = encoded(root, outputs)
    assert enc["a"] == {"$absent": {"cause": "negative", "source": "Maybe"}}
    assert enc["b"] == [1, {"$absent": {"cause": "no-list", "source": None}}]
    back = read(root).outputs
    assert back == outputs
    assert (back["a"].source, back["b"][1].source) == (
        "Maybe",
        None,
    )  # not hashed, kept
    assert type_is_encodable(Absent)


@pytest.mark.parametrize(
    "body",
    [
        {"cause": "x"},
        {"cause": "x", "source": None, "more": 1},
        {"cause": 1, "source": None},
        {"cause": "x", "source": 2},
        ["x", None],
    ],
)
def test_malformed_absent_is_corrupt(root, body):
    publish(root, {"a": Absent(cause="x")})
    edit_entry(root, lambda d: d["outputs"].update(a={"$absent": body}))
    with pytest.raises(CorruptEntry):
        read(root)


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
            fmt.read(entry, "python-other", FP, None, TRUST, root=root)
        with pytest.raises(MisfiledEntry):
            fmt.read(entry, CHECKSUM, "0" * 64, None, TRUST, root=root)
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


# -- shared contents, member modes and malformed outputs ----------------------


def _write(path: Path, data: bytes) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    path.write_bytes(data)


def _relist(root, name: str, data: bytes) -> None:
    sha = hashlib.sha256(data).hexdigest()
    edit_entry(
        root,
        lambda d: d["members"].__setitem__(name, {"sha256": sha, "size": len(data)}),
        redigest=False,
    )


def _swap_blob(root, old: str, data: bytes, suffix: str, new_doc) -> str:
    """Replace the member ``old`` with ``data`` and have the output refer to it."""
    sha = hashlib.sha256(data).hexdigest()
    os.chmod(entry_path(root) / old, stat.S_IWRITE | stat.S_IREAD)
    (entry_path(root) / old).unlink()
    (entry_path(root) / f"{sha}{suffix}").write_bytes(data)

    def edit(doc):
        del doc["members"][old]
        doc["members"][f"{sha}{suffix}"] = {"sha256": sha, "size": len(data)}
        doc["outputs"]["a"] = new_doc(sha)

    edit_entry(root, edit)
    return sha


def _dat(tmp_path, name, content=b""):
    d = tmp_path / "src"
    d.mkdir(exist_ok=True)
    (d / name).write_bytes(content)
    return Tagged(d / name)


def _names(value) -> set:
    return {Path(p).name for p in value.fspaths}


def test_equal_file_sets_under_different_names_publish(root, tmp_path, tags):
    out, err = _dat(tmp_path, "stdout.dat"), _dat(tmp_path, "stderr.dat")
    c, d = _dat(tmp_path, "c.dat", b"same"), _dat(tmp_path, "d.dat", b"same")
    publish(root, {"o": out, "e": err, "pair": [c, d]})
    got = read(root).outputs
    assert _names(got["o"]) == {"stdout.dat"} and _names(got["e"]) == {"stderr.dat"}
    assert [_names(v) for v in got["pair"]] == [{"c.dat"}, {"d.dat"}]
    assert all(Path(p).read_bytes() == b"same" for v in got["pair"] for p in v.fspaths)
    assert (
        len(list((entry_path(root) / "files").iterdir())) == 2
    )  # one tree per distinct set


def test_equal_directories_under_different_names_publish(root, tmp_path, tags):
    for name in ("a", "b"):
        (tmp_path / name / "sub").mkdir(parents=True)
        (tmp_path / name / "x.txt").write_bytes(b"x")
        (tmp_path / name / "sub" / "y.txt").write_bytes(b"y")
    publish(root, {"a": Directory(tmp_path / "a"), "b": Directory(tmp_path / "b")})
    got = read(root).outputs
    assert _names(got["a"]) == {"a"} and _names(got["b"]) == {"b"}
    assert _keys(got["a"]) == _keys(got["b"]) == ["sub/y.txt", "x.txt"]
    assert len(list((entry_path(root) / "files").iterdir())) == 1


def test_a_second_name_in_one_set_is_recorded_as_a_member(root, tmp_path, tags):
    publish(root, {"o": _dat(tmp_path, "one.dat"), "e": _dat(tmp_path, "two.dat")})
    members = entry_json(root)["members"]
    assert sorted(m.rsplit("/", 1)[1] for m in members if m.startswith("files/")) == [
        "one.dat",
        "two.dat",
    ]


def _two_pickles(root):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(
        root,
        {"a": np.ma.MaskedArray(np.arange(2)), "b": np.ma.MaskedArray(np.arange(3))},
        fmt=fmt,
    )
    return fmt, sorted(entry_path(root).glob("*.pkl"))


@posix_only
@pytest.mark.parametrize("which", [0, 1])
def test_every_pickle_member_gets_the_mode_check_through_read(root, which):
    fmt, pkls = _two_pickles(root)
    assert len(pkls) == 2
    read(root, fmt=fmt, trust="private")
    os.chmod(pkls[which], 0o666)
    with pytest.raises(EntryNotTrusted):
        read(root, fmt=fmt, trust="private")


@posix_only
def test_entry_json_gets_the_mode_check_with_pickles(root):
    fmt, _ = _two_pickles(root)
    os.chmod(entry_path(root) / "entry.json", 0o666)
    with pytest.raises(EntryNotTrusted):
        read(root, fmt=fmt, trust="private")
    assert read(root, fmt=fmt, trust="trusted").outputs["a"].shape == (2,)


@posix_only
def test_size_policy_does_not_let_a_rewritten_pickle_run(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    with StorageDir.open_root(
        d, StorePolicy(verify_files="size"), trust="private"
    ) as r:
        fmt, pkls = _two_pickles(r)
        os.chmod(pkls[1], 0o666)
        with pytest.raises(EntryNotTrusted):
            read(r, fmt=fmt, trust="private")


def _plain_entry_json(root, fn):
    path = entry_path(root) / "entry.json"
    os.chmod(path, 0o644)
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


def test_lone_surrogate_in_an_output_is_corrupt_not_a_valueerror(root):
    publish(root, {"a": "x"})
    _plain_entry_json(root, lambda d: d["outputs"].update(a="\ud800"))
    with pytest.raises(CorruptEntry):
        read(root)


@pytest.mark.parametrize("bad", [{"n": 2**60}, {"n": "\ud800"}, {"n": 1e999}])
def test_unfingerprintable_context_is_corrupt_not_a_valueerror(root, bad):
    publish(root, {"a": 1})
    raw = json.dumps({"format": "pydra-entry@1", "context": bad}).encode("ascii")
    _write(entry_path(root) / "context.json", raw)
    _relist(root, "context.json", raw)
    with pytest.raises(CorruptEntry):
        read(root)


@python.define(outputs={"f": File})
def FileOut(x: int) -> File:
    raise NotImplementedError


@pytest.mark.parametrize("value", ["/etc/hosts", Path("/etc/hosts")])
def test_a_file_field_is_never_handed_a_path_outside_the_entry(root, value):
    publish(root, {"f": value})
    with pytest.raises(CorruptEntry):
        read(root, task=FileOut(x=1))


@posix_only
def test_a_symlinked_directory_inside_a_directory_output_is_refused(
    root, tmp_path, tags
):
    d = tmp_path / "tree"
    d.mkdir()
    (d / "a.txt").write_bytes(b"a")
    publish(root, {"d": Directory(d)})
    read(root)
    (set_dir,) = (entry_path(root) / "files").iterdir()
    (set_dir / "tree" / "linkdir").symlink_to(tmp_path)
    with pytest.raises(CorruptEntry):
        read(root)


@posix_only
def test_a_symlinked_directory_in_a_directory_output_is_unencodable_at_stage(
    root, tmp_path, tags
):
    d = tmp_path / "tree"
    (tmp_path / "elsewhere").mkdir()
    d.mkdir()
    (d / "a.txt").write_bytes(b"a")
    (d / "linkdir").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(UnencodableOutput):
        publish(root, {"d": Directory(d)})


def test_an_unreferenced_directory_under_files_is_corrupt(root, tmp_path, tags):
    publish(root, {"f": _dat(tmp_path, "one.dat", b"1")})
    (entry_path(root) / "files" / ("0" * 64)).mkdir()
    with pytest.raises(CorruptEntry):
        read(root)


def test_ndarray_annotations_are_encodable():
    import numpy.typing as npt

    for tp in (
        npt.NDArray[np.float64],
        ty.Optional[npt.NDArray[np.int32]],
        ty.List[npt.NDArray[np.float32]],
    ):
        assert type_is_encodable(tp), tp
    assert not type_is_encodable(ty.List[Opaque])


@pytest.mark.parametrize(
    "bad", ["1.5", "NaN", "infinity", " 1 ", "0x1.8p+0 ", "-0.0", ""]
)
def test_float_tag_accepts_only_what_is_written(root, bad):
    publish(root, {"a": 1})
    edit_entry(root, lambda d: d["outputs"].update(a={"$float": bad}))
    with pytest.raises(CorruptEntry):
        read(root)


def test_float_tag_accepts_the_written_forms(root):
    publish(root, {"a": 1})
    for text, check in (
        ("nan", lambda v: v != v),
        ("-inf", lambda v: v == float("-inf")),
        ((1.5).hex(), lambda v: v == 1.5),
        ((-0.0).hex(), lambda v: str(v) == "-0.0"),
    ):
        edit_entry(root, lambda d, t=text: d["outputs"].update(a={"$float": t}))
        assert check(read(root).outputs["a"])


def test_read_takes_the_root_it_is_given(root):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    entry_dir = entry_path(root)
    quarantine = root.child("quarantine", create=True)
    name = uuid.uuid4().hex
    # the copy section 6.3 re-verifies sits two levels below the root, not four
    with root.child("results@1") as entries:
        with (
            entries.child(entry_dir.parent.parent.name) as sh,
            sh.child(CHECKSUM) as cd,
        ):
            cd.rename(FP, quarantine, name)
    with quarantine.child(name) as moved:
        got = fmt.read(moved, CHECKSUM, FP, None, TRUST, root=root)
    assert got.ref.root == str(root.path) and got.outputs["v"].shape == (2,)
    if not IS_WINDOWS:
        os.chmod(Path(root.path) / "quarantine" / name, 0o775)
        with quarantine.child(name) as moved:
            with pytest.raises(EntryNotTrusted):
                fmt.read(moved, CHECKSUM, FP, None, TRUST, root=root)


@posix_only
def test_private_walk_checks_the_handle_is_the_address(root):
    fmt = PydraEntryV1(pickle=True, trust="private")
    publish(root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt)
    other = fmt.identity.fingerprint({**CTX, "n": 99})
    publish(
        root, {"v": np.ma.MaskedArray(np.arange(2))}, fmt=fmt, context={**CTX, "n": 99}
    )
    real = entry_path(root)
    elsewhere = entry_path(root, fp=other)
    handle = root
    for part in elsewhere.relative_to(root.path).parts:
        handle = handle.child(part)
    with handle:
        handle.path = str(real)  # a handle that is not where its path says
        with pytest.raises(EntryNotTrusted):
            fmt.read(handle, CHECKSUM, other, None, "private", root=root)


def test_a_tampered_file_member_is_corrupt(root, tmp_path, tags):
    src = tmp_path / "scan.dat"
    src.write_bytes(b"0123456789")
    publish(root, {"f": Tagged(src)})
    (member,) = (entry_path(root) / "files").glob("*/scan.dat")
    _write(member, b"0123456780")  # the same size
    with pytest.raises(CorruptEntry):
        read(root)
    _write(member, b"012345678")  # a shorter one
    with pytest.raises(CorruptEntry):
        read(root)
    _write(member, b"0123456789")
    read(root)


def test_a_file_member_is_checked_by_size_alone_under_the_size_policy(tmp_path, tags):
    if IS_WINDOWS:
        pytest.skip("verify_files='size' needs a private root")
    d = tmp_path / "r"
    d.mkdir()
    with StorageDir.open_root(
        d, StorePolicy(verify_files="size"), trust="private"
    ) as r:
        src = tmp_path / "scan.dat"
        src.write_bytes(b"0123456789")
        publish(r, {"f": Tagged(src)})
        (member,) = (entry_path(r) / "files").glob("*/scan.dat")
        _write(member, b"0123456780")
        read(r, trust="private")
        _write(member, b"01234567")
        with pytest.raises(CorruptEntry):
            read(r, trust="private")


class _Rejecting(PydraIdentityV1):
    def check_key(self, checksum, key):
        return False


def test_check_key_is_asked_on_read(root):
    publish(root, {"a": 1}, key={"k": 1})
    fmt = PydraEntryV1(identity=_Rejecting(), pickle=False, trust=TRUST)
    with pytest.raises(MisfiledEntry):
        read(root, fmt=fmt)
    read(root)


def test_a_swapped_context_is_misfiled_even_when_relisted(root):
    publish(root, {"a": 1})
    raw = json.dumps({"format": "pydra-entry@1", "context": {**CTX, "n": 2}}).encode(
        "ascii"
    )
    _write(entry_path(root) / "context.json", raw)
    _relist(root, "context.json", raw)
    with pytest.raises(MisfiledEntry):
        read(root)


def test_an_object_array_member_is_never_unpickled(root, tmp_path):
    import io

    marker = tmp_path / "ran"

    class Payload:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    publish(root, {"a": np.arange(3)})
    (npy,) = [p.name for p in entry_path(root).glob("*.npy")]
    buf = io.BytesIO()
    np.save(buf, np.array([Payload()], dtype=object), allow_pickle=True)
    _swap_blob(root, npy, buf.getvalue(), ".npy", lambda sha: {"$array": sha})
    with pytest.raises(CorruptEntry):
        read(root)
    assert not marker.exists()


def test_a_non_canonical_npy_member_is_corrupt(root):
    import io

    from numpy.lib import format as npformat

    arr = np.arange(6, dtype="<i4")
    publish(root, {"a": arr})
    (npy,) = [p.name for p in entry_path(root).glob("*.npy")]
    for write in (
        lambda b: npformat.write_array(b, arr, version=(2, 0)),
        lambda b: npformat.write_array(b, arr.astype(">i4"), version=(1, 0)),
        lambda b: npformat.write_array(
            b, np.asfortranarray(arr.reshape(2, 3)), version=(1, 0)
        ),
    ):
        buf = io.BytesIO()
        write(buf)
        _swap_blob(root, npy, buf.getvalue(), ".npy", lambda s: {"$array": s})
        with pytest.raises(CorruptEntry):
            read(root)
        (npy,) = [p.name for p in entry_path(root).glob("*.npy")]


@pytest.mark.parametrize(
    "member",
    [
        "../x.bin",
        "files/../x",
        "/abs.bin",
        "a\\b.bin",
        "files/" + "0" * 64,
        "xx.npy",
        "entry.json",
    ],
)
def test_member_names_are_validated_in_the_member_table(root, member):
    publish(root, {"a": 1})
    rec = {"sha256": "0" * 64, "size": 0}
    edit_entry(root, lambda d: d["members"].__setitem__(member, rec))
    with pytest.raises(CorruptEntry):
        read(root)


def test_a_traversal_listed_in_members_and_outputs_is_corrupt(root, tmp_path, tags):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    publish(root, {"f": Tagged(f)})
    sha = hashlib.sha256(b"abc").hexdigest()
    bad = "files/" + "0" * 64 + "/../../one.dat"

    def edit(doc):
        doc["outputs"]["f"]["files"][0][1] = bad
        doc["members"][bad] = {"sha256": sha, "size": 3}

    edit_entry(root, edit)
    with pytest.raises(CorruptEntry):
        read(root)


def test_a_file_set_in_two_set_directories_is_corrupt(root, tmp_path, tags):
    src = tmp_path / "src"
    src.mkdir()
    (src / "scan.hdr").write_bytes(b"h")
    (src / "scan.img").write_bytes(b"i")
    publish(root, {"s": Pair([src / "scan.hdr", src / "scan.img"])})
    other = "1" * 64

    def edit(doc):
        files = doc["outputs"]["s"]["files"]
        old = files[1][1]
        new = old.replace(old.split("/")[1], other)
        files[1][1] = new
        doc["members"][new] = doc["members"].pop(old)

    edit_entry(root, edit)
    with pytest.raises(CorruptEntry):
        read(root)


@pytest.mark.parametrize(
    "record",
    [
        {"size": "x", "sha256": "0" * 64},
        {"size": True, "sha256": "0" * 64},
        {"size": 1},
        {"size": 1, "sha256": "zz"},
        [],
        None,
    ],
)
def test_member_records_are_validated(root, record):
    publish(root, {"b": b"bytes"})
    (name,) = [n for n in entry_json(root)["members"] if n.endswith(".bin")]
    edit_entry(root, lambda d: d["members"].__setitem__(name, record))
    with pytest.raises(CorruptEntry):
        read(root)


def test_context_json_must_be_listed(root):
    publish(root, {"a": 1})
    edit_entry(root, lambda d: d["members"].pop("context.json"))
    with pytest.raises(CorruptEntry):
        read(root)


def test_a_listed_member_no_output_claims_is_corrupt(root):
    publish(root, {"b": b"bytes"})
    edit_entry(root, lambda d: d["outputs"].update(b=1))
    with pytest.raises(CorruptEntry):
        read(root)


# -- declared outputs read back; reader and codec defects -----------------------


@python.define(outputs={"f": Tagged, "d": Directory})
def FileAndDir(x: int) -> ty.Tuple[Tagged, Directory]:
    raise NotImplementedError


@pytest.mark.parametrize("relative", [False, True])
def test_declared_file_and_directory_read_back_through_a_task(
    tmp_path, tags, monkeypatch, relative
):
    (tmp_path / "src" / "tree").mkdir(parents=True)
    (tmp_path / "src" / "one.dat").write_bytes(b"abc")
    (tmp_path / "src" / "tree" / "x.txt").write_bytes(b"x")
    (tmp_path / "r").mkdir()
    monkeypatch.chdir(tmp_path)
    path = "r" if relative else str(tmp_path / "r")
    with StorageDir.open_root(path, StorePolicy(), trust=TRUST) as r:
        publish(
            r,
            {
                "f": Tagged(tmp_path / "src" / "one.dat"),
                "d": Directory(tmp_path / "src" / "tree"),
            },
        )
        got = read(r, task=FileAndDir(x=1)).outputs
        base = os.path.abspath(entry_path(r))
        for key in ("f", "d"):
            for p in got[key].fspaths:
                assert os.path.abspath(p).startswith(
                    os.path.join(base, "files") + os.sep
                )
        assert _names(got["f"]) == {"one.dat"} and _keys(got["d"]) == ["x.txt"]
        # and a path outside is still refused on the same relative root
        publish(
            r,
            {"f": str(tmp_path / "src" / "one.dat"), "d": "x"},
            checksum="python-other",
        )
        with pytest.raises(CorruptEntry):
            read(r, task=FileAndDir(x=1), checksum="python-other")


def test_a_reader_defect_is_not_a_corrupt_entry(root, monkeypatch):
    from pydra.engine.portable import entry as entry_mod

    publish(root, {"a": 1})

    def boom(_encoded):
        raise AttributeError("a bug in the reader")

    monkeypatch.setattr(entry_mod, "output_digest", boom)
    with pytest.raises(AttributeError):
        read(root)
    monkeypatch.setattr(
        entry_mod, "output_digest", lambda e: (_ for _ in ()).throw(TypeError("bug"))
    )
    with pytest.raises(TypeError):
        read(root)
    monkeypatch.setattr(
        entry_mod, "output_digest", lambda e: (_ for _ in ()).throw(ValueError("bad"))
    )
    with pytest.raises(CorruptEntry):
        read(root)


def test_a_codec_defect_does_not_quarantine(root):
    class Boxed:
        pass

    register_codec(Boxed, "test.entry.Boxed", lambda b: 1, lambda v: v.nope)
    publish(root, {"b": Boxed()})
    with pytest.raises(EntryFormatUnknown):
        read(root)
    register_codec(Boxed, "test.entry.Boxed", lambda b: 1, lambda v: 1 + "x")
    with pytest.raises(EntryFormatUnknown):
        read(root)


# -- tampered entries are corrupt ------------------------------------------------


def test_data_errors_in_a_tampered_entry_are_corrupt(root, monkeypatch):
    from pydra.engine.portable import entry as entry_mod

    publish(root, {"a": 1})
    # a KeyError and an IndexError met while reading are data errors
    for exc in (KeyError("missing"), IndexError("short")):
        monkeypatch.setattr(
            entry_mod, "output_digest", lambda e, exc=exc: (_ for _ in ()).throw(exc)
        )
        with pytest.raises(CorruptEntry):
            read(root)
    monkeypatch.undo()


def test_a_required_key_removed_is_corrupt_not_a_keyerror(root, tmp_path, tags):
    f = tmp_path / "one.dat"
    f.write_bytes(b"abc")
    publish(root, {"c": complex(1, 2), "f": Tagged(f), "p": {"$set": 1}})
    for edit in (
        lambda d: d["outputs"]["c"]["$complex"].pop(),
        lambda d: d["outputs"]["c"].__setitem__("$complex", []),
        lambda d: d["outputs"]["f"]["files"].__setitem__(0, ["."]),
        lambda d: d["outputs"]["f"].pop("files"),
        lambda d: d["members"].__getitem__(next(iter(d["members"]))).pop("size"),
        lambda d: d.pop("outputs"),
        lambda d: d.pop("members"),
    ):
        publish(root, {"c": complex(1, 2), "f": Tagged(f)}, checksum="python-x")
        edit_entry(root, edit, redigest=False, checksum="python-x")
        with pytest.raises(CorruptEntry):
            read(root, checksum="python-x")
        shutil_rm(entry_path(root, "python-x"))


def shutil_rm(path: Path) -> None:
    """Remove an entry tree whose members are read-only (on Windows that blocks removal)."""
    import shutil

    def writable(func, name, _exc):
        os.chmod(name, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
        func(name)

    shutil.rmtree(path, onerror=writable)


def test_bind_gives_the_format_its_stores_identity_pickle_and_trust():
    class Other(PydraIdentityV1):
        pass

    bound = PydraEntryV1().bind(Other(), True, "private")
    assert isinstance(bound, PydraEntryV1) and isinstance(bound.identity, Other)
    assert (bound.pickle, bound.trust) == (True, "private")
