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
from pydra.engine.portable.reuse import Hit, Miss
from pydra.engine.portable.store import PortableStore
from pydra.engine.portable.trust import Root
from pydra.engine.result import Result
from pydra.engine.store import LEASE_SUFFIX, SUCCESS_DIR, context_key
from pydra.engine.submitter import IDENTITY_MARKER
from pydra.scripts.cli import cli
from pydra.utils.identity import IDENTITY_SCHEME
from pydra.utils.storagedir import IS_WINDOWS

from .test_portable_store import IDENT, TRUST, make, make_v2, posix_only

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


@pytest.mark.parametrize("scheme", ["pydra-identity@9", "someone-else@1", ""])
def test_marker_naming_unknown_scheme_refused(tmp_path, scheme):
    cache = tmp_path / "cache"
    write_slot(cache)
    mark(cache, scheme)
    with pytest.raises(MigrationRefused, match="marker|keyed"):
        migrate(cache, make_v2(tmp_path), trust_source=True, supplied_checksums=True)


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


def test_real_pickle_cache_served_without_recompute(tmp_path):
    """A cache a `Submitter` wrote (marked pydra-identity@2) serves its jobs from the
    portable store after migration"""
    from pydra.engine.submitter import Submitter

    cache = tmp_path / "cache"
    with Submitter(worker="debug", store="pickle", cache_root=cache) as sub:
        assert sub(Add(a=4)).outputs.out == 6
    assert (cache / IDENTITY_MARKER).read_text().strip() == IDENTITY_SCHEME
    target = PortableStore(tmp_path / "root", trust=TRUST, pickle=True)
    report = migrate(cache, target, trust_source=True, supplied_checksums=True)
    assert report.count("migrated") == 1, report
    with Submitter(worker="debug", store=target) as sub:
        result = sub(Add(a=4))
    assert result.outputs.out == 6
    assert result.committed.served


def test_migrate_twice_is_idempotent(tmp_path):
    cache = tmp_path / "cache"
    write_slot(cache)
    target = make(tmp_path)
    migrate(cache, target, trust_source=True)
    assert [s.status for s in migrate(cache, target, trust_source=True).slots] == [
        "migrated"
    ]


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
