import hashlib
import json
import os
import re
import subprocess
import sys
import time
import typing as ty

import attrs
import pytest

from pydra.compose import python
from pydra.engine.job import Job
from pydra.engine.portable.identity import PydraIdentityV1, StoreIdentity
from pydra.engine.portable.layout import (
    HEADER_NAME,
    LayoutHeader,
    LegacyLayout,
    StoreLayoutMismatch,
    StoreLayoutUnknown,
    cdir_for,
    open_layout,
    shard_for,
)
from pydra.engine.store import PickleStore
from pydra.engine.submitter import Submitter
from pydra.utils.digest import jcs
from pydra.utils.storagedir import (
    IS_WINDOWS,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
)

TRUST = "trusted" if IS_WINDOWS else "private"


def _open(path):
    return StorageDir.open_root(path, StorePolicy(), trust=TRUST)


@pytest.fixture
def root(tmp_path):
    d = tmp_path / "root"
    d.mkdir()
    handle = _open(d)
    yield handle
    handle.close()


@pytest.fixture
def expected():
    return LayoutHeader.for_store(PydraIdentityV1())


def _stored(root_path) -> dict:
    return json.loads((root_path / HEADER_NAME).read_text())


def test_error_hierarchy():
    for exc in (StoreLayoutUnknown, StoreLayoutMismatch, LegacyLayout):
        assert issubclass(exc, StoreUnavailable)


def test_default_header_values(expected):
    assert attrs.asdict(expected) == {
        "format": "pydra-store@1.0",
        "entries": "results@1",
        "entry_format": "pydra-entry@1",
        "identity": {
            "name": "pydra-identity@1",
            "fingerprint": "pydra-context@1",
            "recorded_only": [],
        },
        "shard": "sha256-2",
        "dirs": {"staging": "staging", "quarantine": "quarantine", "sidecars": {}},
        "modes": {
            "file": "0o444",
            "entry": "0o755",
            "sealed": None,
            "container": "0o755",
            "side": "0o755",
            "staging": "0o700",
        },
        "pickle": False,
    }


def test_open_creates_then_checks(root, expected, tmp_path):
    assert open_layout(root, expected) == expected
    assert _stored(tmp_path / "root") == attrs.asdict(expected)
    assert open_layout(root, expected) == expected
    assert [p.name for p in (tmp_path / "root").iterdir()] == [HEADER_NAME]


RACE = """
import sys
from pydra.engine.portable.identity import PydraIdentityV1
from pydra.engine.portable.layout import LayoutHeader, open_layout
from pydra.utils.storagedir import StorageDir, StorePolicy

handle = StorageDir.open_root(sys.argv[1], StorePolicy(), trust=sys.argv[2])
header = LayoutHeader.for_store(PydraIdentityV1())
for _ in range(40):
    assert open_layout(handle, header) == header
"""


def test_header_written_atomically(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    procs = [
        subprocess.Popen([sys.executable, "-c", RACE, str(path), TRUST])
        for _ in range(4)
    ]
    assert [p.wait(timeout=120) for p in procs] == [0] * 4
    assert sorted(x.name for x in path.iterdir()) == [HEADER_NAME]
    assert _stored(path)["format"] == "pydra-store@1.0"


@pytest.mark.parametrize(
    "fmt", ["pydra-store@2.0", "pydra-store@10.1", "other@1.0", "pydra-store", ""]
)
def test_unknown_format_major_raises(root, expected, tmp_path, fmt):
    open_layout(root, expected)
    doc = _stored(tmp_path / "root")
    doc["format"] = fmt
    (tmp_path / "root" / HEADER_NAME).write_text(json.dumps(doc))
    with pytest.raises(StoreLayoutUnknown):
        open_layout(root, expected)


def test_unreadable_header_is_unknown(root, expected, tmp_path):
    (tmp_path / "root" / HEADER_NAME).write_text("{not json")
    with pytest.raises(StoreLayoutUnknown):
        open_layout(root, expected)


def test_minor_version_and_unknown_keys_are_accepted(root, expected, tmp_path):
    open_layout(root, expected)
    doc = _stored(tmp_path / "root")
    doc["format"] = "pydra-store@1.7"
    doc["future"] = {"x": 1}
    (tmp_path / "root" / HEADER_NAME).write_text(json.dumps(doc))
    got = open_layout(root, expected)
    assert got.format == "pydra-store@1.7"
    assert got.entries == expected.entries


MISMATCHES: dict[str, ty.Callable[[LayoutHeader], LayoutHeader]] = {
    "entries": lambda h: attrs.evolve(h, entries="results@2"),
    "entry_format": lambda h: attrs.evolve(h, entry_format="pydra-entry@2"),
    "identity": lambda h: attrs.evolve(
        h, identity={**h.identity, "name": "pydra-identity@2"}
    ),
    "recorded_only": lambda h: attrs.evolve(
        h, identity={**h.identity, "recorded_only": ["hostname"]}
    ),
    "modes": lambda h: attrs.evolve(h, modes={**h.modes, "container": "0o2775"}),
    "shard": lambda h: attrs.evolve(h, shard="prefix2"),
    "dirs": lambda h: attrs.evolve(h, dirs={**h.dirs, "staging": "stage"}),
    "pickle": lambda h: attrs.evolve(h, pickle=True),
}


@pytest.mark.parametrize("key", sorted(MISMATCHES))
def test_mismatch_names_key(root, expected, key):
    open_layout(root, expected)
    with pytest.raises(StoreLayoutMismatch) as info:
        open_layout(root, MISMATCHES[key](expected))
    assert info.value.key == key
    assert key in str(info.value)


def test_fingerprint_name_mismatch_names_identity(root, expected):
    open_layout(root, expected)
    other = attrs.evolve(
        expected, identity={**expected.identity, "fingerprint": "pydra-context@2"}
    )
    with pytest.raises(StoreLayoutMismatch) as info:
        open_layout(root, other)
    assert info.value.key == "identity"


def test_recorded_only_order_is_irrelevant(root, expected):
    a = attrs.evolve(
        expected, identity={**expected.identity, "recorded_only": ["a", "b"]}
    )
    b = attrs.evolve(
        expected, identity={**expected.identity, "recorded_only": ["b", "a"]}
    )
    open_layout(root, a)
    open_layout(root, b)


@pytest.mark.parametrize("name", ["abc-123", "results@1", "something"])
def test_entries_without_header_is_legacy(root, expected, tmp_path, name):
    (tmp_path / "root" / name).mkdir()
    with pytest.raises(LegacyLayout) as info:
        open_layout(root, expected)
    assert "migrat" in str(info.value).lower()
    assert not (tmp_path / "root" / HEADER_NAME).exists()


def test_empty_root_is_not_legacy(root, expected):
    open_layout(root, expected)


def test_stale_temp_header_removed(root, expected, tmp_path):
    path = tmp_path / "root"
    stale = path / ".pydra-store.0123456789abcdef.tmp"
    fresh = path / ".pydra-store.fedcba9876543210.tmp"
    stale.write_text("{")
    fresh.write_text("{")
    old = time.time() - StorePolicy().ages.staging - 60
    os.utime(stale, (old, old))
    open_layout(root, expected)
    assert not stale.exists()
    assert fresh.exists()
    assert (path / HEADER_NAME).exists()


def test_temp_header_is_removed_after_create(root, expected, tmp_path):
    open_layout(root, expected)
    assert not [p for p in (tmp_path / "root").iterdir() if p.name.endswith(".tmp")]


def test_cdir_rule():
    assert cdir_for("python-0123abcd") == "python-0123abcd"
    assert cdir_for("a") == "a"
    assert cdir_for("a_b-c") == "a_b-c"
    raw200 = "a" * 200
    assert cdir_for(raw200) == raw200
    for bad in [
        "A",
        "Python-1",
        "a.b",
        "-a",
        "_a",
        "",
        "a" * 201,
        "a/b",
        "..x",
        "é",
        "a\n",
    ]:
        out = cdir_for(bad)
        assert out == "h-" + hashlib.sha256(bad.encode()).hexdigest()
        assert re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,199}", out)


def test_case_colliding_checksums_never_share_cdir():
    names = {cdir_for(c) for c in ("Task-AB", "task-ab", "TASK-AB", "tAsk-aB")}
    assert len(names) == 4
    assert len({n.lower() for n in names}) == 4


def test_shard_sha256_2():
    cs = "python-0123"
    assert (
        shard_for(cdir_for(cs), cs, "sha256-2")
        == hashlib.sha256(cs.encode()).hexdigest()[:2]
    )
    upper = "Python-0123"
    assert (
        shard_for(cdir_for(upper), upper, "sha256-2")
        == hashlib.sha256(upper.encode()).hexdigest()[:2]
    )


def test_shard_prefix2():
    cs = "ab" + "0" * 62
    assert shard_for(cdir_for(cs), cs, "prefix2") == "ab"


@pytest.mark.parametrize("cs", ["python-0123", "..x", "A" * 64, "zz" + "0" * 62, "a"])
def test_prefix2_refuses_non_hex(cs):
    with pytest.raises(ValueError):
        shard_for(cdir_for(cs), cs, "prefix2")


def test_unknown_shard_rule_refused():
    with pytest.raises(ValueError):
        shard_for("abc", "abc", "sha256-3")  # type: ignore[arg-type]


def test_v1_names():
    ident = PydraIdentityV1()
    assert ident.name == "pydra-identity@1"
    assert ident.fingerprint_name == "pydra-context@1"
    assert ident.recorded_only == frozenset()
    assert isinstance(ident, StoreIdentity)
    assert ident.key(None) is None  # type: ignore[arg-type]
    assert ident.check_key("python-abc", None)


def test_v1_fingerprint_is_full_sha256_of_jcs():
    ctx = {"b": [1, 2], "a": {"z": None, "y": "é"}, "python": "3.12.1"}
    fp = PydraIdentityV1().fingerprint(ctx)
    assert fp == hashlib.sha256(jcs(ctx)).hexdigest()
    assert len(fp) == 64 and fp == fp.lower()
    assert (
        PydraIdentityV1().fingerprint(
            {"a": {"y": "é", "z": None}, "b": [1, 2], "python": "3.12.1"}
        )
        == fp
    )


def test_v1_recorded_only_is_excluded_from_fingerprint():
    ident = PydraIdentityV1(recorded_only=frozenset({"host"}))
    a = {"python": "3", "host": "x"}
    b = {"python": "3", "host": "y"}
    assert ident.fingerprint(a) == ident.fingerprint(b)
    assert ident.fingerprint(a) == hashlib.sha256(jcs({"python": "3"})).hexdigest()
    assert ident.fingerprint({"python": "4", "host": "x"}) != ident.fingerprint(a)


def test_v1_is_picklable():
    import cloudpickle

    ident = PydraIdentityV1(recorded_only=frozenset({"host"}))
    assert cloudpickle.loads(cloudpickle.dumps(ident)) == ident


def test_header_from_identity_records_recorded_only():
    header = LayoutHeader.for_store(
        PydraIdentityV1(recorded_only=frozenset({"b", "a"}))
    )
    assert header.identity["recorded_only"] == ["a", "b"]


@python.define
def _Inc(a: int) -> int:
    return a + 1


SEEN: list = []


class _Marked(PydraIdentityV1):
    def context(self, job, phase):
        SEEN.append(phase)
        return {"marked": True, "phase": phase}


def test_job_context_through_identity(tmp_path):
    with Submitter(worker="debug", cache_root=tmp_path) as sub:
        job = Job(task=_Inc(a=1), name="j", submitter=sub)
        plain = job.context
        assert "marked" not in plain and "python" in plain
        job2 = Job(task=_Inc(a=1), name="j2", submitter=sub)
        store = PickleStore(tmp_path / "other")
        ident = _Marked()
        store.identity = ident
        job2.store = store
        assert job2.context == {"marked": True, "phase": "lookup"}
        assert job2.context is job2.context
        assert SEEN == ["lookup"]
