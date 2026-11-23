"""Migrating a `PickleStore` cache into a `PortableStore`, and reading one in place through
``legacy=`` roots (RFC 0003 section 8.1)."""

import json
import os
import types
import typing as ty
from pathlib import Path

import cloudpickle as cp
import pytest
from click.testing import CliRunner

from pydra.compose import python
from pydra.engine.lease import DirectoryLease, LeaseSettings
from pydra.engine.portable import migrate as migrate_mod
from pydra.engine.portable.entry import EntryRef
from pydra.engine.portable.errors import KeyMismatch
from pydra.engine.portable.layout import HEADER_NAME, cdir_for
from pydra.engine.portable.migrate import (
    MigratedSlot,
    MigrationRefused,
    MigrationReport,
    migrate,
)
from pydra.engine.portable.reuse import Hit, Miss, PolicyReuse, Verdict
from pydra.engine.portable.store import PortableStore
from pydra.engine.portable.trust import Root
from pydra.engine.result import Result
from pydra.engine.store import LEASE_SUFFIX, SUCCESS_DIR, context_key
from pydra.engine.submitter import IDENTITY_MARKER
from pydra.scripts.cli import cli
from pydra.utils.identity import IDENTITY_SCHEME
from pydra.utils.storagedir import IS_WINDOWS

from .test_portable_store import IDENT, TRUST, corrupt, make, make_v2, posix_only

CTX = {"python": "3.12", "platform": "test", "n": 1}
CTX_NOW = {"python": "3.12", "platform": "test", "n": 2}
SUM = "python-0123456789abcdef"


@python.define
def Add(a: int, b: int = 2) -> int:
    return a + b


class Opaque:
    """A value no codec encodes"""


@python.define
def MakeOpaque(a: int) -> ty.Any:
    return Opaque()


def write_slot(
    cache: Path,
    checksum: str = SUM,
    context: dict | None = CTX,
    *,
    out: ty.Any = 3,
    errored: bool = False,
    result: bool = True,
    task: ty.Any = None,
    ctxkey: str | None = None,
) -> Path:
    """A `PickleStore` slot as `PickleStore.commit` and a job's run leave it: the context,
    the pickled result, the job's pickle and an error pickle"""
    ctx = CTX if context is None else context
    slot = Path(cache) / checksum / (ctxkey or context_key(ctx))
    slot.mkdir(parents=True)
    task = Add(a=1) if task is None else task
    if context is not None:
        (slot / "_context.json").write_text(json.dumps(context, sort_keys=True))
    if result:
        outputs = type(task).Outputs(out=out)
        res = Result(
            cache_dir=slot, outputs=outputs, errored=errored, task=task, context=ctx
        )
        (slot / "_result.pklz").write_bytes(cp.dumps(res))
    (slot / "_job.pklz").write_bytes(cp.dumps({"a job": 1}))
    (slot / "_error.pklz").write_bytes(cp.dumps({"an error": 1}))
    return slot


def mark(cache: Path, scheme: str) -> None:
    (Path(cache) / IDENTITY_MARKER).write_text(scheme + "\n")


def tree(top: Path) -> dict[str, bytes | None]:
    out = {}
    for d, dirs, files in os.walk(top):
        for name in dirs:
            out[os.path.relpath(os.path.join(d, name), top)] = None
        for name in files:
            p = os.path.join(d, name)
            with open(p, "rb") as f:
                out[os.path.relpath(p, top)] = f.read()
    return out


def outputs_of(result: Result) -> dict:
    outputs = result.outputs
    if isinstance(outputs, ty.Mapping):
        return dict(outputs)
    return {"out": outputs.out}


def entry_ref(store: PortableStore, checksum: str, context) -> EntryRef:
    return EntryRef(store.path, cdir_for(checksum), store.identity.fingerprint(context))


def job(checksum=SUM, context=CTX_NOW, **kw):
    return types.SimpleNamespace(checksum=checksum, context=context, task=None, **kw)


# -- migrate() ---------------------------------------------------------------------------------


def test_migrate_requires_trust_source(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    before = tree(cache)
    target = make(tmp_path)
    with pytest.raises(ValueError, match="trust_source"):
        migrate(cache, target, trust_source=False)
    assert tree(cache) == before  # not even a lease was taken
    assert not target.entry_path(SUM, CTX).exists()


def test_outputs_published_under_stored_context(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, out=7)
    target = make(tmp_path)
    report = migrate(cache, target, trust_source=True)
    assert isinstance(report, MigrationReport)
    (slot,) = report.slots
    assert slot == MigratedSlot(SUM, context_key(CTX), "migrated", None)
    assert report.count("migrated") == 1
    assert target.entry_path(SUM, CTX).is_dir()
    result = target.read(entry_ref(target, SUM, CTX), None)
    assert result.context == CTX
    assert outputs_of(result) == {"out": 7}


def test_ctxkey_discarded(tmp_path):
    """The slot's directory name plays no part: the address is the stored context's"""
    cache = tmp_path / "cache"
    write_slot(cache, ctxkey="0" * 16)
    target = make(tmp_path)
    assert migrate(cache, target, trust_source=True).count("migrated") == 1
    assert target.entry_path(SUM, CTX).is_dir()


def test_no_job_or_error_pickles_copied(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    (cache / SUCCESS_DIR).mkdir()
    (cache / SUCCESS_DIR / f"{SUM}-{context_key(CTX)}").touch()
    target = make(tmp_path)
    migrate(cache, target, trust_source=True)
    names = {os.path.basename(p) for p in tree(Path(target.path))}
    assert not names & {"_job.pklz", "_error.pklz", "_result.pklz", "_context.json"}
    assert SUCCESS_DIR not in names
    assert not any(n.endswith(LEASE_SUFFIX) for n in names)
    assert set(os.listdir(target.entry_path(SUM, CTX))) == {
        "entry.json",
        "context.json",
    }


def test_busy_slot_skipped_and_reported(tmp_path):
    cache = tmp_path / "cache"
    slot = write_slot(cache)
    target = make(tmp_path)
    seen: list[MigratedSlot] = []
    holder = DirectoryLease(
        SUM,
        slot,
        cache / f"{SUM}-{context_key(CTX)}{LEASE_SUFFIX}",
        settings=LeaseSettings(),
    )
    with holder:
        report = migrate(cache, target, trust_source=True, on_slot=seen.append)
    assert [s.status for s in report.slots] == ["busy"]
    assert seen == list(report.slots)
    assert not target.entry_path(SUM, CTX).exists()
    report = migrate(cache, target, trust_source=True)  # a later call migrates it
    assert [s.status for s in report.slots] == ["migrated"]
    assert not (cache / f"{SUM}-{context_key(CTX)}{LEASE_SUFFIX}").exists()


def test_errored_or_contextless_skipped(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, "python-errored", errored=True)
    write_slot(cache, "python-noresult", result=False)
    write_slot(cache, "python-nocontext", context=None)
    write_slot(cache, "python-good")
    target = make(tmp_path)
    report = migrate(cache, target, trust_source=True)
    by = {s.checksum: s for s in report.slots}
    assert by["python-good"].status == "migrated"
    for name in ("python-errored", "python-noresult", "python-nocontext"):
        assert by[name].status == "skipped", by[name]
        assert by[name].reason
        assert not target.entry_path(name, CTX).exists()
    assert "errored" in by["python-errored"].reason
    assert "_result.pklz" in by["python-noresult"].reason
    assert "_context.json" in by["python-nocontext"].reason


def test_unknown_context_skipped(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    write_slot(cache, context={**CTX, "unknown": "a1b2c3"})
    unpickled = []
    real = migrate_mod._unpickle
    monkeypatch.setattr(
        migrate_mod, "_unpickle", lambda *a: unpickled.append(1) or real(*a)
    )
    target = make(tmp_path)
    (slot,) = migrate(cache, target, trust_source=True).slots
    assert slot.status == "skipped" and "unknown" in slot.reason
    assert unpickled == []  # its result is never unpickled


def test_unencodable_output_reported(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, task=MakeOpaque(a=1), out=Opaque())
    target = make(tmp_path)
    (slot,) = migrate(cache, target, trust_source=True).slots
    assert slot.status == "unencodable" and "out" in slot.reason
    assert not target.entry_path(SUM, CTX).exists()


@posix_only
def test_symlinked_slot_not_followed(tmp_path):
    cache = tmp_path / "cache"
    elsewhere = write_slot(tmp_path / "elsewhere")
    (cache / SUM).mkdir(parents=True)
    os.symlink(elsewhere, cache / SUM / context_key(CTX))
    os.symlink(elsewhere.parent, cache / "python-linked")
    target = make(tmp_path)
    report = migrate(cache, target, trust_source=True)
    assert report.count("migrated") == 0
    assert not target.entry_path(SUM, CTX).exists()
    assert not target.entry_path("python-linked", CTX).exists()


def test_v2_target_refused_without_supplied_checksums(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    before = tree(cache)
    target = make_v2(tmp_path)
    with pytest.raises(MigrationRefused, match=IDENTITY_SCHEME):
        migrate(cache, target, trust_source=True)
    assert tree(cache) == before


def test_v2_supplied_key_record(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, "gather-n2", out=5)
    target = make_v2(tmp_path)
    report = migrate(cache, target, trust_source=True, supplied_checksums=True)
    assert report.count("migrated") == 1
    ref = entry_ref(target, "gather-n2", CTX)
    doc = json.loads((target.entry_path("gather-n2", CTX) / "entry.json").read_text())
    assert doc["key"] == {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    result = target.read(ref, None)
    assert result.record == {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    assert outputs_of(result) == {"out": 5}
    # a job of that supplied checksum finds it
    supplied = job(
        "gather-n2", CTX, record={"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    )
    assert isinstance(target.find(supplied), Hit)


def test_identity_refusing_key_skips_slot(tmp_path):
    """A downstream identity is asked through `publish`, which raises `KeyMismatch`"""

    class Strict(type(IDENT)):
        name = "strict-identity@1"

        def check_key(self, checksum, key):
            return False

    cache = tmp_path / "cache"
    write_slot(cache)
    target = make(tmp_path, identity=Strict())
    with pytest.raises(KeyMismatch):
        target.publish(SUM, None, CTX, {"out": 1}, None, None)
    (slot,) = migrate(cache, target, trust_source=True).slots
    assert slot.status == "skipped" and "key" in slot.reason


def test_marker_naming_v2_refused_by_v1_target(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, IDENTITY_SCHEME)
    with pytest.raises(MigrationRefused, match=IDENTITY_SCHEME):
        migrate(cache, make(tmp_path), trust_source=True)


@pytest.mark.parametrize("scheme", ["pydra-identity@9", "someone-else@1"])
def test_marker_naming_unknown_scheme_refused(tmp_path, scheme):
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, scheme)
    with pytest.raises(MigrationRefused, match="marker|keyed"):
        migrate(cache, make_v2(tmp_path), trust_source=True, supplied_checksums=True)


@pytest.mark.parametrize("text", ["", "pydra-identi", "\x00\x00"])
def test_an_empty_or_garbled_marker_reads_as_no_marker(tmp_path, text):
    """As the submitter reads it: a marker write cut short means a root keyed before
    markers, under pydra-identity@1"""
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, text)
    report = migrate(cache, make(tmp_path), trust_source=True)
    assert report.count("migrated") == 1


def test_marker_naming_v1_migrates(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, "pydra-identity@1")
    report = migrate(cache, make(tmp_path), trust_source=True)
    assert report.scheme == "pydra-identity@1"
    assert report.count("migrated") == 1


def test_portable_root_refused_as_source(tmp_path):
    source = make(tmp_path, "other")
    with pytest.raises(MigrationRefused, match=HEADER_NAME):
        migrate(source.path, make(tmp_path), trust_source=True)


def _submitter_cache(tmp_path) -> Path:
    from pydra.engine.submitter import Submitter

    cache = tmp_path / "cache"
    with Submitter(worker="debug", store="pickle", cache_root=cache) as sub:
        assert sub(Add(a=4)).outputs.out == 6
    assert (cache / IDENTITY_MARKER).read_text().strip() == IDENTITY_SCHEME
    return cache


def test_v2_marked_cache_refuses_supplied_checksums(tmp_path):
    """Its checksums are automatic @2 digests: a supplied-key record would misstate them"""
    cache = _submitter_cache(tmp_path)
    before = tree(cache)
    target = PortableStore(tmp_path / "root", trust=TRUST, pickle=True)
    with pytest.raises(MigrationRefused, match="legacy="):
        migrate(cache, target, trust_source=True, supplied_checksums=True)
    assert tree(cache) == before


def test_real_v2_cache_served_through_legacy_with_its_record(tmp_path):
    """A cache a `Submitter` wrote (marked pydra-identity@2) is read in place through
    ``legacy=``; the entry it publishes carries the job's own record, not a supplied one
    """
    from pydra.engine.submitter import Submitter

    cache = _submitter_cache(tmp_path)
    before = tree(cache)
    target = PortableStore(
        tmp_path / "root",
        trust=TRUST,
        pickle=True,
        legacy=[Root(cache, trust="private")],
    )
    with Submitter(worker="debug", store=target) as sub:
        result = sub(Add(a=4))
    assert result.outputs.out == 6
    assert result.committed.served
    assert result.record is not None and "supplied" not in result.record
    assert result.record["scheme"] == IDENTITY_SCHEME
    assert tree(cache) == before


def test_migrate_twice_reports_present(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    target = make(tmp_path)
    migrate(cache, target, trust_source=True)
    report = migrate(cache, target, trust_source=True)
    assert [s.status for s in report.slots] == ["present"]
    assert report.count("present") == 1 and report.count("migrated") == 0


def staging_of(store) -> list:
    return os.listdir(Path(store.path) / "staging")


def test_conflicting_slot_leaves_no_staged_copy(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, out=3)
    target = make(tmp_path)
    target.publish(SUM, None, CTX, {"out": 99}, None, None)
    for _ in range(2):
        (slot,) = migrate(cache, target, trust_source=True).slots
        assert slot.status == "skipped" and "other outputs" in slot.reason
        assert staging_of(target) == []
    assert outputs_of(target.read(entry_ref(target, SUM, CTX), None)) == {"out": 99}


def test_target_inside_source_refused(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    with pytest.raises(MigrationRefused, match="inside"):
        migrate(cache, make(tmp_path, "cache/portable"), trust_source=True)


@posix_only
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes anywhere"
)
def test_read_only_cache_refused_clearly(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    target = make(tmp_path)
    os.chmod(cache, 0o555)
    try:
        with pytest.raises(MigrationRefused, match="lease"):
            migrate(cache, target, trust_source=True)
    finally:
        os.chmod(cache, 0o755)


# -- legacy= roots -----------------------------------------------------------------------------


def legacy_store(tmp_path, cache, trust="private", **kw):
    kw.setdefault("reuse", "any")
    return make(tmp_path, legacy=[Root(cache, trust=trust)], **kw)


def test_legacy_shared_refused(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    with pytest.raises(ValueError, match="legacy"):
        legacy_store(tmp_path, cache, trust="shared")


def test_legacy_root_never_the_store_root(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    with pytest.raises(ValueError, match="legacy"):
        make(tmp_path, "cache", legacy=[Root(cache, trust="private")])
    with pytest.raises(ValueError, match="legacy"):
        make(tmp_path, "cache/portable", legacy=[Root(cache, trust="private")])


def test_legacy_hit_published_under_stored_fingerprint(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, out=9)
    store = legacy_store(tmp_path, cache)
    found = store.find(job(context=CTX_NOW))
    assert isinstance(found, Hit)
    assert outputs_of(found.result) == {"out": 9}
    assert found.result.cross_context and found.result.reused_in == CTX_NOW
    assert found.result.committed.entry == entry_ref(store, SUM, CTX)
    assert store.entry_path(SUM, CTX).is_dir()
    assert not store.entry_path(SUM, CTX_NOW).exists()


def test_legacy_private_hit_published_once(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    write_slot(cache)
    before = tree(cache)
    store = legacy_store(tmp_path, cache)
    unpickled = []
    real = migrate_mod._unpickle
    monkeypatch.setattr(
        migrate_mod, "_unpickle", lambda *a: unpickled.append(1) or real(*a)
    )
    assert isinstance(store.find(job()), Hit)
    assert isinstance(store.find(job()), Hit)
    assert unpickled == [1]  # the old cache is read once per result
    assert tree(cache) == before  # and never written


def test_legacy_miss_costs_nothing(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    store = legacy_store(tmp_path, cache)
    assert isinstance(store.find(job("python-other")), Miss)
    assert isinstance(store.find(job(), probe=True), Miss)  # left to the leased lookup


def test_legacy_respects_reuse_policy(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    store = legacy_store(tmp_path, cache, reuse="exact")
    assert isinstance(store.find(job(context=CTX_NOW)), Miss)
    assert not store.entry_path(SUM, CTX).exists()
    assert isinstance(store.find(job(context=CTX)), Hit)


def test_legacy_skips_unusable_slots(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, errored=True)
    write_slot(cache, "python-unknown", context={**CTX, "unknown": "x"})
    store = legacy_store(tmp_path, cache)
    assert isinstance(store.find(job()), Miss)
    assert isinstance(store.find(job("python-unknown")), Miss)


@posix_only
def test_legacy_private_refuses_group_writable_slot(tmp_path):
    cache = tmp_path / "cache"
    slot = write_slot(cache)
    os.chmod(slot, 0o775)
    store = legacy_store(tmp_path, cache)
    assert isinstance(store.find(job()), Miss)
    store = legacy_store(tmp_path, cache, trust="trusted")
    assert isinstance(store.find(job()), Hit)


def test_legacy_v2_supplied_job(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, "gather-n2")
    store = make_v2(tmp_path, legacy=[Root(cache, trust="private")], reuse="any")
    record = {"scheme": IDENTITY_SCHEME, "supplied": "gather-n2"}
    found = store.find(job("gather-n2", record=record, absent=()))
    assert isinstance(found, Hit)
    doc = json.loads((store.entry_path("gather-n2", CTX) / "entry.json").read_text())
    assert doc["key"] == record


def test_legacy_survives_pickling(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    store = cp.loads(cp.dumps(legacy_store(tmp_path, cache)))
    assert store.legacy == (Root(cache, trust="private"),)
    assert isinstance(store.find(job()), Hit)


def test_legacy_read_after_a_corrupt_own_entry(tmp_path):
    """The own entry at the job's address is quarantined: the same lookup reads legacy"""
    cache = tmp_path / "cache"
    write_slot(cache, out=4)
    store = legacy_store(tmp_path, cache)
    store.publish(SUM, None, CTX, {"out": 4}, None, None)
    corrupt(store.entry_path(SUM, CTX))
    found = store.find(job(context=CTX))
    assert isinstance(found, Hit)
    assert outputs_of(found.result) == {"out": 4}


class SkipOwn(PolicyReuse):
    """Skips every entry of this store's own root that was there before legacy was read"""

    def __init__(self, skipped):
        from pydra.engine.store import AnyReuse

        super().__init__(AnyReuse())
        self.skipped = skipped

    def verdict(self, candidate, current, task):
        if candidate.entry.fingerprint in self.skipped:
            return Verdict.SKIP
        return Verdict.SERVE


class RefuseAll(PolicyReuse):
    def __init__(self):
        from pydra.engine.store import AnyReuse

        super().__init__(AnyReuse())

    def verdict(self, candidate, current, task):
        return Verdict.REFUSE


def test_legacy_read_when_every_candidate_is_skipped(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache, out=5)
    store = legacy_store(tmp_path, cache, reuse=SkipOwn({IDENT.fingerprint(CTX_NOW)}))
    store.publish(SUM, None, CTX_NOW, {"out": 1}, None, None)
    found = store.find(job(context=CTX_NOW))
    assert isinstance(found, Hit)
    assert outputs_of(found.result) == {"out": 5}
    assert found.result.committed.entry == entry_ref(store, SUM, CTX)


def test_legacy_not_read_after_a_refuse(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    write_slot(cache)
    store = legacy_store(tmp_path, cache, reuse=RefuseAll())
    store.publish(SUM, None, CTX_NOW, {"out": 1}, None, None)
    called = []
    monkeypatch.setattr(
        migrate_mod, "publish_legacy", lambda *a: called.append(a) or False
    )
    assert isinstance(store.find(job(context=CTX_NOW)), Miss)
    assert called == []
    assert not store.entry_path(SUM, CTX).exists()


def test_legacy_read_once_per_lookup(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    write_slot(cache, errored=True)
    store = legacy_store(tmp_path, cache)
    called = []
    real = migrate_mod.publish_legacy
    monkeypatch.setattr(
        migrate_mod, "publish_legacy", lambda *a: called.append(1) or real(*a)
    )
    assert isinstance(store.find(job()), Miss)
    assert called == [1]


def test_legacy_v2_marked_cache_skipped_by_v1_store(tmp_path, caplog):
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, IDENTITY_SCHEME)
    store = legacy_store(tmp_path, cache)
    with caplog.at_level("WARNING", logger="pydra"):
        assert isinstance(store.find(job()), Miss)
    assert any("legacy root" in r.getMessage() for r in caplog.records)
    assert not store.entry_path(SUM, CTX).exists()


def test_legacy_first_slot_of_a_fingerprint_taken(tmp_path):
    """Two slots of one stored context (directories named differently): the first in path
    order is the one ranked"""
    cache = tmp_path / "cache"
    write_slot(cache, out=6, ctxkey="0" * 16)
    write_slot(cache, errored=True, ctxkey="f" * 16)
    store = legacy_store(tmp_path, cache)
    found = store.find(job(context=CTX))
    assert isinstance(found, Hit)
    assert outputs_of(found.result) == {"out": 6}


# -- pydracli store ----------------------------------------------------------------------------


def test_cli_store_migrate(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    root = tmp_path / "root"
    runner = CliRunner()
    args = ["store", "migrate", str(cache), str(root)]
    result = runner.invoke(cli, args)
    assert result.exit_code != 0 and "--trust-source" in result.output
    result = runner.invoke(cli, [*args, "--trust-source", "--trust", TRUST])
    assert result.exit_code != 0 and IDENTITY_SCHEME in result.output
    result = runner.invoke(
        cli, [*args, "--trust-source", "--supplied-checksums", "--trust", TRUST]
    )
    assert result.exit_code == 0, result.output
    assert "1 migrated" in result.output
    v1 = tmp_path / "root-v1"
    result = runner.invoke(
        cli,
        [
            "store",
            "migrate",
            str(cache),
            str(v1),
            "--trust-source",
            "--trust",
            TRUST,
            "--identity",
            "pydra.engine.portable.identity:PydraIdentityV1",
        ],
    )
    assert result.exit_code == 0, result.output
    store = PortableStore(v1, trust=TRUST, identity=IDENT)
    assert store.entry_path(SUM, CTX).is_dir()


@pytest.mark.skipif(IS_WINDOWS, reason="the adopted fixture's modes are POSIX")
def test_cli_store_collect_adopted_root(tmp_path):
    from pydra.engine.tests import test_portable_adopt as adopted

    adopted.tree.build_tree(tmp_path, n=1)
    adopted.do_adopt(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli, ["store", "collect", str(tmp_path), "--trust", "trusted"]
    )
    assert result.exit_code != 0  # the default identity, format and policy do not match
    module = adopted.__name__
    result = runner.invoke(
        cli,
        [
            "store",
            "collect",
            str(tmp_path),
            "--trust",
            "trusted",
            "--identity",
            f"{module}:ToyIdentity",
            "--entry-format",
            f"{module}:ToyFormat",
            "--policy",
            f"{module}:POLICY",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "removed" in result.output and "kept" in result.output


def test_cli_store_help_lists_migrate():
    result = CliRunner().invoke(cli, ["store", "--help"])
    assert result.exit_code == 0
    for name in ("adopt", "collect", "migrate"):
        assert name in result.output
    result = CliRunner().invoke(cli, ["store", "migrate", "--help"])
    assert "created" in result.output


class OnLegacy(PolicyReuse):
    """Answers ``verdict`` for a candidate in a legacy root, serves the rest"""

    def __init__(self, legacy_root, verdict):
        from pydra.engine.store import AnyReuse

        super().__init__(AnyReuse())
        self.legacy_root, self.answer = str(legacy_root), verdict

    def verdict(self, candidate, current, task):
        if os.path.abspath(candidate.entry.root) == self.legacy_root:
            return self.answer
        return Verdict.SERVE


def _counting_unpickle(monkeypatch):
    unpickled = []
    real = migrate_mod._unpickle
    monkeypatch.setattr(
        migrate_mod, "_unpickle", lambda *a: unpickled.append(1) or real(*a)
    )
    return unpickled


@pytest.mark.parametrize("verdict", [Verdict.SKIP, Verdict.REFUSE])
def test_a_legacy_candidate_the_hook_rejects_is_never_read_or_published(
    tmp_path, monkeypatch, verdict
):
    cache = tmp_path / "cache"
    write_slot(cache, out=5)
    store = legacy_store(tmp_path, cache, reuse=OnLegacy(cache.resolve(), verdict))
    unpickled = _counting_unpickle(monkeypatch)
    for _ in range(3):
        assert isinstance(store.find(job(context=CTX_NOW)), Miss)
    assert unpickled == []
    assert not store.entry_path(SUM, CTX).exists()


class RefusesFirstRoot(PolicyReuse):
    """Refuses a candidate in ``first``, serves any other"""

    def __init__(self, first):
        from pydra.engine.store import AnyReuse

        super().__init__(AnyReuse())
        self.first = str(first)

    def verdict(self, candidate, current, task):
        if os.path.abspath(candidate.entry.root) == self.first:
            return Verdict.REFUSE
        return Verdict.SERVE


def test_refusing_the_first_legacy_candidate_ends_the_lookup(tmp_path, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    write_slot(first, out=5)
    write_slot(second, out=6)
    store = make(
        tmp_path,
        legacy=[Root(first, trust="private"), Root(second, trust="private")],
        reuse=RefusesFirstRoot(first.resolve()),
    )
    unpickled = _counting_unpickle(monkeypatch)
    assert isinstance(store.find(job(context=CTX_NOW)), Miss)
    assert unpickled == []
    assert not store.entry_path(SUM, CTX).exists()


class PreMarkerIdentity:
    """An identity written to the protocol before ``scheme`` and ``supplied_key``"""

    name = "client-identity@1"
    fingerprint_name = "client-context@1"
    recorded_only: frozenset = frozenset()

    def key(self, job):
        return None

    def check_key(self, checksum, key):
        return True

    def context(self, job, phase):
        return job.context

    def fingerprint(self, context):
        return IDENT.fingerprint(context)


def test_an_identity_missing_protocol_members_is_refused_when_the_store_opens(
    tmp_path,
):
    with pytest.raises(TypeError, match="scheme.*supplied_key"):
        PortableStore(tmp_path / "root", trust=TRUST, identity=PreMarkerIdentity())


def test_cli_store_migrate_into_the_cache_writes_nothing(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    root = cache / "newstore"
    result = CliRunner().invoke(
        cli,
        [
            "store",
            "migrate",
            str(cache),
            str(root),
            "--trust-source",
            "--trust",
            TRUST,
            "--identity",
            "pydra.engine.portable.identity:PydraIdentityV1",
        ],
    )
    assert result.exit_code != 0 and "inside the cache" in result.output
    assert not root.exists()


def test_a_published_legacy_result_the_store_does_not_serve_is_not_read_again(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    write_slot(cache, out=5)
    # serves the legacy slot, skips what it becomes in the store's own root
    store = legacy_store(tmp_path, cache, reuse=OnLegacy(cache.resolve(), None))
    store.reuse.verdict = lambda candidate, current, task: (
        Verdict.SERVE
        if os.path.abspath(candidate.entry.root) == str(cache.resolve())
        else Verdict.SKIP
    )
    unpickled = _counting_unpickle(monkeypatch)
    for _ in range(3):
        store.find(job(context=CTX_NOW))
    assert store.entry_path(SUM, CTX).exists()
    assert unpickled == [1]


def test_the_scheme_check_asks_the_identity_not_its_name():
    """A store keying under pydra-identity@1 by any name never holds @2 checksums, and one
    keying under @2 by any name records supplied keys"""
    from pydra.engine.portable.identity import PydraIdentityV1, PydraIdentityV2
    from pydra.engine.portable.migrate import check_scheme

    class RenamedV1(PydraIdentityV1):
        name = "site-identity@1"

    class RenamedV2(PydraIdentityV2):
        name = "site-identity@2"

    with pytest.raises(MigrationRefused, match="never looks up"):
        check_scheme("pydra-identity@2", RenamedV1())
    check_scheme("pydra-identity@2", RenamedV2())
    assert RenamedV2().supplied_key("python-x") == {
        "scheme": "pydra-identity@2",  # the record format the checksum digests
        "supplied": "python-x",
    }
