"""``pydra-identity@2`` as the default identity: the ``.pydra-identity`` marker of a
`PickleStore` root, the layout header of a `PortableStore` root, and the ``absent``
triples of a key whose checksum carries an absence suffix (RFC 0004 sections 3.1, 9.1
and 10 I8)."""

import json
import logging

import pytest
from click.testing import CliRunner

from pydra.compose import python, workflow
from pydra.engine import submitter as submitter_module
from pydra.engine.absent import Absent, absence_suffix
from pydra.engine.portable.identity import PydraIdentityV1, PydraIdentityV2
from pydra.engine.portable.layout import StoreLayoutMismatch
from pydra.engine.portable.store import PortableStore
from pydra.engine.submitter import IDENTITY_MARKER, Submitter
from pydra.scripts.cli import cli
from pydra.utils.identity import IDENTITY_SCHEME, record_checksum

from .test_absent_store import Spy, entry_docs
from .test_gates import Check, Download
from .test_portable_store import TRUST


@python.define
def Add(a: int, b: int = 2) -> int:
    return a + b


@workflow.define
def Twice(a: int) -> int:
    one = workflow.add(Add(a=a), name="one")
    two = workflow.add(Add(a=one.out), name="two")
    return two.out


def by_length(task, inputs):
    """A supplied checksum built from how many values a gather reads, not from them"""
    return f"gather-n{len(inputs['vals'])}"


@python.define(
    checksum=by_length, inputs={"vals": python.arg(type=list[str], absent="accept")}
)
def KeyedGather(vals) -> list[str]:
    return [f"absent:{v.cause}" if isinstance(v, Absent) else v for v in vals]


@workflow.define
def GatedKeyed(xs: list[int]) -> list[str]:
    c = workflow.add(Check().split(x=xs), name="C")
    d = workflow.add(Download(url="x").combine("C.x"), name="D", when=c.out)
    g = workflow.add(KeyedGather(vals=d.out), name="G")
    return g.out


@workflow.define
def GatedKeyedSibling(xs: list[int], tag: str = "b") -> list[str]:
    """`GatedKeyed`'s nodes under another workflow key, so nothing is served whole"""
    c = workflow.add(Check().split(x=xs), name="C")
    d = workflow.add(Download(url="x").combine("C.x"), name="D", when=c.out)
    g = workflow.add(KeyedGather(vals=d.out), name="G")
    return g.out


def _messages(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "pydra.submitter"]


def _store(root, **kw):
    return PortableStore(root, trust=TRUST, pickle=True, **kw)


# -- the PickleStore marker --------------------------------------------------------------------


def test_marker_written_on_first_run(tmp_path, caplog):
    root = tmp_path / "cache"
    with caplog.at_level(logging.WARNING, logger="pydra.submitter"):
        with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
            assert sub(Add(a=1)).outputs.out == 3
    assert (root / IDENTITY_MARKER).read_text().strip() == IDENTITY_SCHEME
    assert _messages(caplog) == []  # a new root holds nothing to recompute


def test_old_scheme_logged_once(tmp_path, caplog):
    root = tmp_path / "cache"
    root.mkdir()
    (root / IDENTITY_MARKER).write_text("pydra-identity@1\n")
    with caplog.at_level(logging.WARNING, logger="pydra.submitter"):
        for a in (1, 2):
            with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
                sub(Add(a=a))
    (line,) = _messages(caplog)
    assert "pydra-identity@1" in line and IDENTITY_SCHEME in line
    assert (root / IDENTITY_MARKER).read_text().strip() == IDENTITY_SCHEME


def test_missing_marker_with_slots_logged_once(tmp_path, caplog):
    root = tmp_path / "cache"
    (root / "python-0123").mkdir(parents=True)  # a slot written before the marker
    with caplog.at_level(logging.WARNING, logger="pydra.submitter"):
        for a in (1, 2):
            with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
                sub(Add(a=a))
    (line,) = _messages(caplog)
    assert "keyed before" in line and IDENTITY_SCHEME in line


@pytest.mark.parametrize("named", ["pydra-identity@3", "someone-else@1"])
def test_newer_marker_left_alone_logged_once(tmp_path, caplog, named):
    """A marker naming a newer (or unknown) scheme is never overwritten, so two installs
    sharing a cache do not flip it back and forth"""
    root = tmp_path / "cache"
    root.mkdir()
    (root / IDENTITY_MARKER).write_text(named + "\n")
    with caplog.at_level(logging.WARNING, logger="pydra.submitter"):
        for a in (1, 2):
            with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
                sub(Add(a=a))
    (line,) = _messages(caplog)
    assert named in line
    assert (root / IDENTITY_MARKER).read_text().strip() == named


def test_marker_one_stat_per_run(tmp_path, monkeypatch):
    calls = []
    real = submitter_module.check_identity_marker

    def counting(root):
        calls.append(root)
        return real(root)

    monkeypatch.setattr(submitter_module, "check_identity_marker", counting)
    opened = []
    real_open = open

    def counting_open(path, *args, **kw):
        if str(path).endswith(IDENTITY_MARKER):
            opened.append(path)
        return real_open(path, *args, **kw)

    monkeypatch.setattr("builtins.open", counting_open)
    root = tmp_path / "cache"
    with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
        sub(Twice(a=1))  # three jobs, one run
        assert len(calls) == 1
        sub(Twice(a=2))
    assert len(calls) == 2
    # the first run finds no marker and writes it; the second reads it once
    assert len(opened) == 2


def test_portable_root_has_no_marker(tmp_path):
    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        sub(Add(a=1))
    assert not (root / IDENTITY_MARKER).exists()


# -- the PortableStore header ------------------------------------------------------------------


def test_new_root_header_names_v2(tmp_path):
    root = tmp_path / "cache"
    with Submitter(worker="debug", cache_root=root) as sub:
        result = sub(Add(a=1))
    header = json.loads((root / "pydra-store.json").read_text())
    assert header["identity"]["name"] == IDENTITY_SCHEME
    assert isinstance(PortableStore(root, trust=TRUST).identity, PydraIdentityV2)
    # the default store's results carry their key record
    assert result.record is not None
    assert record_checksum(result.record) == Add(a=1)._checksum


@pytest.mark.parametrize(
    "task", [Add(a=1), KeyedGather(vals=["a"])], ids=["computed", "supplied"]
)
def test_v1_root_refused_under_v2(tmp_path, task):
    """A root written under ``pydra-identity@1`` is refused, never read, under the default;
    a root holding only supplied-checksum entries too, although their checksums agree"""
    root = tmp_path / "cache"
    v1 = _store(root, identity=PydraIdentityV1())
    with Submitter(worker="debug", store=v1) as sub:
        assert not sub(task).errored
    with pytest.raises(StoreLayoutMismatch, match="identity"):
        _store(root)


# -- absence triples ---------------------------------------------------------------------------


def _suffixed(spy) -> str:
    (checksum,) = {
        c for what, c in spy.calls if what == "publish" and c.startswith("gather-n")
    }
    return checksum


def test_v2_check_key_accepts_absence_suffix(tmp_path, monkeypatch):
    """A supplied-key gather with one accepted ``Absent`` from a gate: run 1 publishes under
    the suffixed checksum with the triples in its key; run 2, a sibling workflow with the
    same nodes, looks the gather up under that checksum and its key is accepted, with no
    quarantine and no recompute"""
    spy = Spy(monkeypatch)
    checked = []
    real_check = PydraIdentityV2.check_key

    def check_key(self, checksum, key):
        accepted = real_check(self, checksum, key)
        checked.append((checksum, accepted))
        return accepted

    monkeypatch.setattr(PydraIdentityV2, "check_key", check_key)
    root = tmp_path / "cache"
    with Submitter(worker="debug", store=_store(root)) as sub:
        first = sub(GatedKeyed(xs=[1, -1]))
    assert first.outputs.out == ["X", "absent:not-routed"]
    checksum = _suffixed(spy)
    assert checksum.startswith("gather-n2-absent-")
    (doc,) = [d for d in entry_docs(root) if d["key"].get("supplied") == "gather-n2"]
    assert doc["key"]["absent"] == [["vals", {"$int": "1"}, "not-routed"]]
    assert checksum == "gather-n2" + absence_suffix([("vals", 1, "not-routed")])
    spy.calls.clear()
    checked.clear()
    quarantined = []
    real = PortableStore._quarantine

    def quarantine(self, *args, **kw):
        quarantined.append(args)
        return real(self, *args, **kw)

    monkeypatch.setattr(PortableStore, "_quarantine", quarantine)
    with Submitter(worker="debug", store=_store(root)) as sub:
        second = sub(GatedKeyedSibling(xs=[1, -1]))
    assert second.outputs.out == ["X", "absent:not-routed"]
    assert "find" in spy.of(checksum)  # the gather is looked up under its suffix
    assert (checksum, True) in checked and (checksum, False) not in checked
    # only the new outer workflow is published: the gather was served, not run again
    assert [c for c in spy.of_kind("publish") if not c.startswith("workflow-")] == []
    assert quarantined == []


def test_v2_suffixed_checksum_without_absent_is_misfiled():
    ident = PydraIdentityV2()
    key = {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    suffix = absence_suffix([("vals", 1, "not-routed")])
    assert ident.check_key("gather-n2", key)
    assert not ident.check_key("gather-n2" + suffix, key)
    assert ident.check_key(
        "gather-n2" + suffix, {**key, "absent": [["vals", {"$int": "1"}, "not-routed"]]}
    )


def test_v2_check_key_never_parses_suffix():
    """A record whose triples differ from the checksum's suffix is refused, whatever the
    checksum's text says"""
    ident = PydraIdentityV2()
    base = {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    checksum = "gather-n2" + absence_suffix([("vals", 1, "not-routed")])
    for absent in (
        [["vals", {"$int": "0"}, "not-routed"]],  # another index
        [["vals", {"$int": "1"}, "other"]],  # another cause
        [["vals", 1, "not-routed"]],  # not the canonical form
        [],  # no triples, but a suffix
        "-absent-",  # not a list
    ):
        assert not ident.check_key(checksum, {**base, "absent": absent}), absent
    # a key whose supplied string holds the suffix itself is not accepted as triples
    assert ident.check_key(checksum, {**base, "supplied": checksum})
    assert not ident.check_key(
        checksum,
        {
            **base,
            "supplied": checksum,
            "absent": [["vals", {"$int": "1"}, "not-routed"]],
        },
    )


def test_explain_shows_absent_triples(tmp_path):
    first = {
        "scheme": IDENTITY_SCHEME,
        "supplied": "gather-n2",
        "absent": [["vals", {"$int": "1"}, "not-routed"]],
    }
    second = {**first, "absent": [["vals", {"$int": "0"}, "not-routed"]]}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(first))
    b.write_text(json.dumps(second))
    result = CliRunner().invoke(cli, ["explain", str(a), str(b)])
    assert result.exit_code == 1
    assert 'absent: ["vals[1]: not-routed"] -> ["vals[0]: not-routed"]' in result.output


def test_explain_reads_a_portable_entry(tmp_path):
    root = tmp_path / "cache"
    for a in (1, 2):
        with Submitter(worker="debug", store=_store(root)) as sub:
            sub(Add(a=a))
    entries = sorted(p.parent for p in root.glob("results@1/*/*/*/entry.json"))
    assert len(entries) == 2
    result = CliRunner().invoke(cli, ["explain", str(entries[0]), str(entries[1])])
    assert result.exit_code == 1
    assert "inputs.a" in result.output


def test_publish_refuses_a_key_its_identity_refuses(tmp_path):
    """An entry its own identity would refuse on every read is never written: the publish
    raises where the key was made, not at a later read"""
    from pydra.engine.portable.errors import KeyMismatch

    root = tmp_path / "cache"
    store = _store(root)
    suffix = absence_suffix([("vals", 1, "not-routed")])
    supplied = {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    for checksum, key in [
        ("python-abc", None),  # no key under pydra-identity@2
        ("gather-n2" + suffix, supplied),  # a suffix with no triples
        ("gather-n3", supplied),  # another checksum
    ]:
        with pytest.raises(KeyMismatch, match="does not digest"):
            store.publish(checksum, key, {"n": 1}, {"v": 1}, None, None)
    assert entry_docs(root) == []
    committed = store.publish("gather-n2", supplied, {"n": 1}, {"v": 1}, None, None)
    assert store.read(committed.entry, None).outputs == {"v": 1}


@pytest.mark.parametrize("garbled", ["", "\x00\x00", "not a scheme"])
def test_an_empty_or_garbled_marker_is_rewritten(tmp_path, garbled):
    root = tmp_path / "cache"
    root.mkdir()
    (root / IDENTITY_MARKER).write_text(garbled)
    with Submitter(worker="debug", store="pickle", cache_root=root) as sub:
        sub(Add(a=1))
    assert (root / IDENTITY_MARKER).read_text().strip() == IDENTITY_SCHEME
