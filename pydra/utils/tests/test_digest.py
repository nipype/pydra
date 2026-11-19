import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pydra.utils import digest as dg
from pydra.utils.digest import (
    canonical_npy,
    digest,
    digest_form,
    fileset_digest,
    jcs,
    lookup_tag,
    register_codec,
    register_digest,
    register_fileset_tag,
    tag_of,
)


def npy(a):
    return digest(a)


def test_dtype_changes_npy():
    a = np.arange(4, dtype="<i4")
    assert npy(a).startswith("npy@1:")
    assert npy(a) != npy(a.astype("<i8"))
    assert npy(a) != npy(a.astype("<f4"))


def test_structured_v8_dtypes_differ():
    a = np.zeros(3, dtype=[("a", "<i4"), ("b", "<i4")])
    b = np.zeros(3, dtype=[("x", "<f4"), ("y", "<i4")])
    assert a.dtype.str == b.dtype.str == "|V8"
    assert npy(a) != npy(b)


def test_shape_changes_npy():
    a = np.arange(6, dtype="<i4")
    assert npy(a) != npy(a.reshape(2, 3))
    assert npy(a.reshape(2, 3)) != npy(a.reshape(3, 2))


def test_byte_order_equal_npy():
    a = np.arange(6, dtype="<i4")
    assert npy(a) == npy(a.astype(">i4"))
    s = np.array(["ab", "c"], dtype="<U2")
    assert npy(s) == npy(s.astype(">U2"))


def test_strided_view_equals_copy():
    base = np.arange(20, dtype="<f8").reshape(4, 5)
    view = base[::2, ::2]
    assert not view.flags.c_contiguous
    assert npy(view) == npy(view.copy())
    fortran = np.asfortranarray(base)
    assert npy(fortran) == npy(base)
    assert npy(np.array(1.0)) == npy(np.array(1.0).reshape(()).copy())


def test_subclass_and_reference_dtypes_py1():
    base = np.arange(4)
    assert digest(np.ma.MaskedArray(base, mask=[0, 1, 0, 0])).startswith("py@1:")
    assert digest(np.matrix(base)).startswith("py@1:")
    assert digest(np.array(["a", {}], dtype=object)).startswith("py@1:")
    assert digest(np.zeros(2, dtype=[("a", "O")])).startswith("py@1:")
    if hasattr(np.dtypes, "StringDType"):
        assert digest(np.array(["a", "bc"], dtype=np.dtypes.StringDType())).startswith(
            "py@1:"
        )
    assert digest(object()).startswith("py@1:")


def test_int_vs_float():
    assert digest(1) != digest(1.0)
    assert digest(True) != digest(1)
    assert digest_form(1) == {"$int": "1"}
    assert digest_form(1.0) == {"$float": (1.0).hex()}
    assert digest_form(float("nan")) == {"$float": "nan"}
    assert digest_form(float("-inf")) == {"$float": "-inf"}
    assert digest(1).startswith("json@1:")


def test_int_above_2_53_exact():
    assert digest(2**53 + 1) != digest(2**53)
    assert digest_form(2**64) == {"$int": str(2**64)}
    assert jcs(digest_form(2**53 + 1)) == b'{"$int":"9007199254740993"}'


def test_relocation_same_digest(tmp_path):
    from fileformats.generic import File

    class Tagged(File):
        ext = ".dat"

    register_fileset_tag(Tagged, "test.relocation.Tagged")
    d1 = tmp_path / "a"
    d2 = tmp_path / "b" / "c"
    d1.mkdir()
    d2.mkdir(parents=True)
    (d1 / "one.dat").write_bytes(b"hello")
    (d2 / "two.dat").write_bytes(b"hello")
    assert digest(Tagged(d1 / "one.dat")) == digest(Tagged(d2 / "two.dat"))
    assert digest(Tagged(d1 / "one.dat")).startswith("file@1:")
    (d2 / "two.dat").write_bytes(b"hellO")
    assert digest(Tagged(d1 / "one.dat")) != digest(Tagged(d2 / "two.dat"))
    assert digest(Path("/x/y")) == digest(Path("/x/y"))
    assert digest(Path("/x/y")) != digest(Path("/x/z"))
    assert digest_form(Path("/x/y")) == {"$path": str(Path("/x/y"))}


def test_list_vs_tuple():
    assert digest([1, 2]) != digest((1, 2))
    assert digest_form((1, [2])) == {"$tuple": [{"$int": "1"}, [{"$int": "2"}]]}
    assert digest({1, 2}) == digest({2, 1})
    assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
    assert digest_form({"$k": True}) == {"$$k": True}
    assert digest_form({1: "x"}) == {"$dict": [[{"$int": "1"}, "x"]]}


def test_npy_name_equals_member_sha():
    a = np.arange(12, dtype=">i4").reshape(3, 4)[:, ::2]
    sha = hashlib.sha256(canonical_npy(a)).hexdigest()
    assert digest(a) == f"npy@1:{sha}"
    assert digest_form([a]) == [{"$array": sha}]
    loaded = np.load(io.BytesIO(canonical_npy(a)), allow_pickle=False)
    assert loaded.dtype == np.dtype("<i4")
    assert (loaded == a).all()
    assert canonical_npy(a)[:8] == b"\x93NUMPY\x01\x00"
    members = [("b", "22" * 32), ("a", "11" * 32)]
    expect = hashlib.sha256(
        jcs({"$fileset": "t", "files": [["a", "11" * 32], ["b", "22" * 32]]})
    ).hexdigest()
    assert fileset_digest("t", members) == f"file@1:{expect}"


def test_lookup_tag_never_imports():
    mod = "pydra_never_imported_module_xyz"
    assert lookup_tag(f"{mod}.Thing") is None
    assert lookup_tag("os.path") is None
    assert mod not in sys.modules

    class P:
        def __init__(self, v):
            self.v = v

    register_codec(P, "test.lookup.P", lambda p: {"v": p.v}, lambda d: P(d["v"]))
    codec = lookup_tag("test.lookup.P")
    assert codec is not None and codec.type is P
    assert tag_of(P) == "test.lookup.P"
    assert tag_of(int) is None
    assert digest_form(P(1)) == {
        "$codec": "test.lookup.P",
        "value": {"v": {"$int": "1"}},
    }
    with pytest.raises(ValueError):
        register_codec(int, "test.lookup.P", str, int)
    with pytest.raises(ValueError):
        register_fileset_tag(dict, "test.lookup.P")
    register_codec(P, "test.lookup.P", lambda p: {"v": p.v}, lambda d: P(d["v"]))


def test_register_digest():
    class Q:
        pass

    register_digest(Q, lambda q: "q@1:" + "0" * 64)
    assert digest(Q()) == "q@1:" + "0" * 64
    assert digest([Q()]).startswith("py@1:")


def test_jcs_rfc8785_vectors():
    doc = json.loads(
        '{"numbers": [333333333.33333329, 1E30, 4.50, 2e-3, 0.000000000000000000000000001],'
        '"string": "\\u20ac$\\u000F\\u000aA\'\\u0042\\u0022\\u005c\\\\\\"\\/",'
        '"literals": [null, true, false]}'
    )
    assert jcs(doc) == (
        '{"literals":[null,true,false],"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27],'
        '"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'
    ).encode("utf-8")
    # Appendix B number serialisation
    vectors = {
        0.0: "0",
        -0.0: "0",
        5e-324: "5e-324",
        1.7976931348623157e308: "1.7976931348623157e+308",
        9007199254740992.0: "9007199254740992",
        -9007199254740992.0: "-9007199254740992",
        295147905179352830000.0: "295147905179352830000",
        9.999999999999997e22: "9.999999999999997e+22",
        1e21: "1e+21",
        1e23: "1e+23",
        1.2345678901234568e20: "123456789012345680000",
        0.000001: "0.000001",
        1e-7: "1e-7",
        4.5: "4.5",
        -1.5e-9: "-1.5e-9",
    }
    for value, text in vectors.items():
        assert jcs(value) == text.encode(), value
    # Appendix: keys sort by UTF-16 code units
    keys = ["€", "\r", "דּ", "1", "\U0001f600", "\u0080", "ö"]
    out = jcs({k: 0 for k in keys}).decode()
    assert [k for k in json.loads(out)] == [
        "\r",
        "1",
        "\u0080",
        "ö",
        "€",
        "\U0001f600",
        "דּ",
    ]
    assert jcs([]) == b"[]" and jcs({}) == b"{}"
    with pytest.raises(ValueError):
        jcs(float("nan"))
    with pytest.raises(ValueError):
        jcs(2**60)
    assert dg.jcs.__module__ == "pydra.utils.digest"
