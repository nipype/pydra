import hashlib
import json
import errno
import os
import pickle
import re
import stat
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
    StoreLayoutAbsent,
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

# `trusted` on Windows unless CI sets PYDRA_TEST_PORTABLE_TRUST (see test_portable_store)
TRUST = os.environ.get("PYDRA_TEST_PORTABLE_TRUST") or (
    "trusted" if IS_WINDOWS else "private"
)


def _open(path):
    return StorageDir.open_root(path, StorePolicy(), trust=TRUST)


def _rewrite(root_path, doc) -> None:
    os.chmod(root_path / HEADER_NAME, stat.S_IWRITE | stat.S_IREAD)  # Windows read-only
    (root_path / HEADER_NAME).unlink()
    (root_path / HEADER_NAME).write_text(json.dumps(doc))


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
            "recorded_only": ["environment_facts", "environment_repository"],
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


def test_header_written_atomically(root, expected, tmp_path, monkeypatch):
    """Two openers race to create the header: the second opens the root while the first has
    written its temporary and not yet linked it. A reader in that window sees no header, not
    a partial one; the first then loses the link, reads the second's header, and leaves no
    temporary behind. The interleaving is forced, so no timing decides it."""
    from pydra.engine.portable import layout

    real = layout._publish
    seen_in_window = []

    def second_opener_first(r, tmp):
        monkeypatch.setattr(layout, "_publish", real)
        with StorageDir.open_root(
            tmp_path / "root", StorePolicy(), trust=TRUST
        ) as other:
            seen_in_window.append(layout._read(other))  # the first's temporary only
            assert open_layout(other, expected) == expected  # the second creates it
        return real(r, tmp)

    monkeypatch.setattr(layout, "_publish", second_opener_first)
    assert open_layout(root, expected) == expected
    assert seen_in_window == [None]
    assert sorted(x.name for x in (tmp_path / "root").iterdir()) == [HEADER_NAME]
    assert _stored(tmp_path / "root")["format"] == "pydra-store@1.0"


@pytest.mark.parametrize(
    "fmt", ["pydra-store@2.0", "pydra-store@10.1", "other@1.0", "pydra-store", ""]
)
def test_unknown_format_major_raises(root, expected, tmp_path, fmt):
    open_layout(root, expected)
    doc = _stored(tmp_path / "root")
    doc["format"] = fmt
    _rewrite(tmp_path / "root", doc)
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
    _rewrite(tmp_path / "root", doc)
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
    message = str(info.value)
    assert "empty directory" in message and "migrat" not in message.lower()
    assert not (tmp_path / "root" / HEADER_NAME).exists()


def test_empty_root_is_not_legacy(root, expected):
    open_layout(root, expected)


def test_stale_temp_header_removed(root, expected, tmp_path):
    path = tmp_path / "root"
    stale = path / ".pydra-store.0123456789abcdef0123456789abcdef.tmp"
    fresh = path / ".pydra-store.fedcba9876543210fedcba9876543210.tmp"
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
    assert ident.recorded_only == frozenset(
        {"environment_facts", "environment_repository"}
    )
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
    assert header.identity["recorded_only"] == [
        "a",
        "b",
        "environment_facts",
        "environment_repository",
    ]


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


def test_header_mode_is_policy_file_mode(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    with _open(path) as root:
        open_layout(root, LayoutHeader.for_store(PydraIdentityV1()))
    assert (path / HEADER_NAME).stat().st_mode & 0o777 == 0o444


@pytest.mark.skipif(IS_WINDOWS, reason="group modes")
def test_header_mode_under_group_policy(tmp_path):
    path = tmp_path / "root"
    path.mkdir()
    policy = StorePolicy(file_mode=0o440, container_mode=0o2770, side_mode=0o2770)
    os.chmod(path, 0o2770)
    root = StorageDir.open_root(path, policy, trust="shared")
    with root:
        open_layout(root, LayoutHeader.for_store(PydraIdentityV1(), policy))
    assert (path / HEADER_NAME).stat().st_mode & 0o777 == 0o440


@pytest.mark.parametrize("created", [False, True])
def test_pickle_is_recorded_not_compared(root, expected, created):
    first = attrs.evolve(expected, pickle=created)
    open_layout(root, first)
    got = open_layout(root, attrs.evolve(expected, pickle=not created))
    assert got.pickle is created


def test_legacy_rechecks_header_after_scan(root, expected, tmp_path, monkeypatch):
    from pydra.engine.portable import layout

    real = layout._sweep_and_scan

    def interleaved(r, *a, **kw):
        # another opener links the header and makes its first directory after our read
        (tmp_path / "root" / HEADER_NAME).write_bytes(expected.to_json())
        (tmp_path / "root" / "staging").mkdir()
        return real(r, *a, **kw)

    monkeypatch.setattr(layout, "_sweep_and_scan", interleaved)
    assert open_layout(root, expected) == expected


@pytest.mark.parametrize(
    "name", [".pydra-succeeded", "python-0123456789abcdef0123456789abcdef"]
)
def test_a_pickle_store_cache_is_named_with_the_way_on(root, expected, tmp_path, name):
    (tmp_path / "root" / name).mkdir()
    with pytest.raises(LegacyLayout) as info:
        open_layout(root, expected)
    message = str(info.value)
    assert "PickleStore cache" in message and 'store="pickle"' in message
    assert "empty directory" in message and "migrat" not in message.lower()


def test_legacy_message_says_new_root_is_empty(root, expected, tmp_path):
    (tmp_path / "root" / "x").mkdir()
    with pytest.raises(LegacyLayout, match="empty"):
        open_layout(root, expected)


def test_exceptions_pickle(expected):
    import cloudpickle

    exc = StoreLayoutMismatch("shard", "differs in shard")
    for dumps, loads in (
        (pickle.dumps, pickle.loads),
        (cloudpickle.dumps, pickle.loads),
    ):
        back = loads(dumps(exc))
        assert type(back) is StoreLayoutMismatch
        assert back.key == "shard" and str(back) == "differs in shard"
    for cls in (StoreLayoutUnknown, LegacyLayout, StoreLayoutAbsent):
        assert type(pickle.loads(pickle.dumps(cls("m")))) is cls


@pytest.mark.parametrize(
    "bad",
    [[1, "a"], [{"a": 1}, {"b": 2}], "host", None, 3],
)
def test_malformed_recorded_only_is_unknown(root, expected, tmp_path, bad):
    open_layout(root, expected)
    doc = _stored(tmp_path / "root")
    doc["identity"]["recorded_only"] = bad
    _rewrite(tmp_path / "root", doc)
    with pytest.raises(StoreLayoutUnknown):
        open_layout(root, expected)


@pytest.mark.parametrize(
    "key,bad", [("entries", 1), ("entry_format", None), ("shard", 2), ("pickle", "yes")]
)
def test_malformed_scalar_keys_are_unknown(root, expected, tmp_path, key, bad):
    open_layout(root, expected)
    doc = _stored(tmp_path / "root")
    doc[key] = bad
    _rewrite(tmp_path / "root", doc)
    with pytest.raises(StoreLayoutUnknown):
        open_layout(root, expected)


def test_recorded_only_string_refused():
    with pytest.raises(TypeError):
        PydraIdentityV1(recorded_only="host")  # type: ignore[arg-type]


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX link errors")
@pytest.mark.parametrize("code", [errno.EPERM, errno.ENOTSUP])
def test_link_failure_is_store_unavailable(root, expected, monkeypatch, code):
    def refuse(name, new_name):
        raise (PermissionError if code == errno.EPERM else OSError)(code, "no links")

    monkeypatch.setattr(root, "link", refuse)
    with pytest.raises(StoreUnavailable) as info:
        open_layout(root, expected)
    assert not isinstance(info.value, PermissionError)


def test_no_create_never_writes(root, expected, tmp_path):
    with pytest.raises(StoreLayoutAbsent):
        open_layout(root, expected, create=False)
    assert list((tmp_path / "root").iterdir()) == []


def test_no_create_checks_existing(root, expected):
    open_layout(root, expected)
    assert open_layout(root, expected, create=False) == expected
    with pytest.raises(StoreLayoutMismatch):
        open_layout(root, MISMATCHES["shard"](expected), create=False)


def test_no_create_on_legacy_root(root, expected, tmp_path):
    (tmp_path / "root" / "x").mkdir()
    with pytest.raises(LegacyLayout):
        open_layout(root, expected, create=False)


@pytest.mark.skipif(
    IS_WINDOWS or os.geteuid() == 0, reason="needs a read-only POSIX root"
)
def test_no_create_on_read_only_root(tmp_path, expected):
    path = tmp_path / "ro"
    path.mkdir()
    with _open(path) as writable:
        open_layout(writable, expected)
    empty = tmp_path / "empty"
    empty.mkdir()
    os.chmod(path, 0o555)
    os.chmod(empty, 0o555)
    try:
        with _open(path) as r:
            assert open_layout(r, expected, create=False) == expected
        with _open(empty) as r:
            with pytest.raises(StoreLayoutAbsent):
                open_layout(r, expected, create=False)
    finally:
        os.chmod(path, 0o755)
        os.chmod(empty, 0o755)


def _stale(path):
    stale = path / ".pydra-store.0123456789abcdef0123456789abcdef.tmp"
    stale.write_text("{")
    old = time.time() - StorePolicy().ages.staging - 60
    os.utime(stale, (old, old))
    return stale


def test_no_create_does_not_sweep(root, expected, tmp_path):
    stale = _stale(tmp_path / "root")
    with pytest.raises(StoreLayoutAbsent):
        open_layout(root, expected, create=False)
    assert stale.exists()


def test_sweep_is_best_effort(root, expected, tmp_path, monkeypatch):
    stale = _stale(tmp_path / "root")
    real = root.unlink

    def refuse(name):
        if name == stale.name:
            raise PermissionError(errno.EPERM, "not yours")
        real(name)

    monkeypatch.setattr(root, "unlink", refuse)
    assert open_layout(root, expected) == expected
    assert stale.exists()


@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="needs POSIX permissions")
def test_unreadable_header_is_store_unavailable(root, expected, tmp_path):
    open_layout(root, expected)
    os.chmod(tmp_path / "root" / HEADER_NAME, 0)
    try:
        with pytest.raises(StoreUnavailable) as info:
            open_layout(root, expected)
        assert not isinstance(info.value, PermissionError)
    finally:
        os.chmod(tmp_path / "root" / HEADER_NAME, 0o444)


@pytest.mark.skipif(IS_WINDOWS or os.geteuid() == 0, reason="needs POSIX permissions")
def test_empty_read_only_root_create_is_store_unavailable(tmp_path, expected):
    path = tmp_path / "empty"
    path.mkdir()
    os.chmod(path, 0o555)
    try:
        with _open(path) as r:
            with pytest.raises(StoreUnavailable) as info:
                open_layout(r, expected)
        assert not isinstance(info.value, PermissionError)
    finally:
        os.chmod(path, 0o755)


def test_recorded_only_adds_to_the_default():
    ident = PydraIdentityV1(recorded_only=frozenset({"host"}))
    assert ident.recorded_only == frozenset(
        {"host", "environment_facts", "environment_repository"}
    )


def test_transient_permission_error_is_retried_on_windows(root, expected, monkeypatch):
    from pydra.engine.portable import layout

    open_layout(root, expected)
    monkeypatch.setattr(layout, "IS_WINDOWS", True)
    monkeypatch.setattr(layout.time, "sleep", lambda s: None)
    real = root.open_read
    calls = []

    def flaky(name):
        calls.append(name)
        if len(calls) < 3:
            raise PermissionError(errno.EACCES, "sharing violation")
        return real(name)

    monkeypatch.setattr(root, "open_read", flaky)
    assert open_layout(root, expected) == expected
    assert len(calls) == 3


def test_persistent_permission_error_on_windows_is_store_unavailable(
    root, expected, monkeypatch
):
    from pydra.engine.portable import layout

    monkeypatch.setattr(layout, "IS_WINDOWS", True)
    monkeypatch.setattr(
        layout, "READ_RETRY", attrs.evolve(layout.READ_RETRY, deadline=0.05)
    )

    def deny(name):
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(root, "open_read", deny)
    with pytest.raises(StoreUnavailable):
        open_layout(root, expected)
