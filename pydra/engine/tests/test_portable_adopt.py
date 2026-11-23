"""Adopting an existing store laid out like a PortableStore (RFC 0003 section 8.2)."""

import hashlib
import io
import json
import logging
import os
import re
import stat
import time
import types
import typing as ty

import attrs
import numpy as np
import pytest
from click.testing import CliRunner

from pydra.engine.portable.adopt import AdoptRefused, adopt
from pydra.engine.portable.entry import EntryRef, StoredEntry
from pydra.engine.portable.errors import (
    CorruptEntry,
    EntryFormatUnknown,
    MisfiledEntry,
)
from pydra.engine.portable.layout import HEADER_NAME, LayoutHeader, cdir_for
from pydra.engine.portable.reuse import Hit, Miss
from pydra.engine.portable.store import PortableStore
from pydra.engine.tests.data_tests import adopt_tree as tree
from pydra.scripts.cli import cli
from pydra.utils.digest import jcs
from pydra.utils.storagedir import IS_WINDOWS, StoreAges, StorePolicy

pytestmark = pytest.mark.skipif(
    IS_WINDOWS, reason="the fixture's group modes and set-group-ID bits are POSIX"
)

NEW = "store@7"


@attrs.frozen
class ToyIdentity:
    """The key is the digest of ``result.json``'s key record; the fingerprint is the sha256
    of ``context.json``, the canonical JSON of the context."""

    recorded_only: frozenset = frozenset()
    name: ty.ClassVar[str] = "toy-identity@6"
    fingerprint_name: ty.ClassVar[str] = "toy-context@1"

    def key(self, job):
        return None

    def check_key(self, checksum, key):
        return isinstance(key, dict) and tree.key_digest(key) == checksum

    def context(self, job, phase):
        return job.context

    def fingerprint(self, context):
        return hashlib.sha256(jcs(dict(context))).hexdigest()


@attrs.frozen
class LaxIdentity(ToyIdentity):
    """Accepts any key: a format built on it checks nothing of the key."""

    def check_key(self, checksum, key):
        return True


_MEMBERS = re.compile(r"result\.json|context\.json|[0-9a-f]{64}\.npy")
_ASIDE = re.compile(r"[0-9a-f]{64}\.aside-[0-9]+")


@attrs.frozen
class ToyFormat:
    """The tool's members, read through ``identity``."""

    identity: ty.Any = attrs.field(factory=ToyIdentity)
    name: ty.ClassVar[str] = "toy-entry@6"
    entries: ty.ClassVar[str] = NEW

    def members(self):
        return _MEMBERS

    def aside_names(self):
        return _ASIDE

    def stage(self, *args, **kw):
        raise NotImplementedError("the toy format only reads")

    @staticmethod
    def _bytes(entry, name):
        try:
            with entry.open_read(name) as f:
                return f.read()
        except FileNotFoundError as err:
            raise CorruptEntry(f"{name} is missing") from err

    def context(self, entry):
        raw = self._bytes(entry, "context.json")
        try:
            return json.loads(raw), raw
        except ValueError as err:
            raise CorruptEntry(f"context.json: {err}") from err

    def identify(self, entry):
        doc = json.loads(self._bytes(entry, "result.json"))
        return tree.key_digest(doc["key"]), ""

    def read(self, entry, checksum, fingerprint, task, trust, *, root):
        foreign = [n for n in entry.names() if not _MEMBERS.fullmatch(n)]
        if foreign:
            raise EntryFormatUnknown(f"{foreign[0]} is not a member")
        context, _ = self.context(entry)
        if self.identity.fingerprint(context) != fingerprint:
            raise MisfiledEntry("the context does not give the fingerprint")
        raw = self._bytes(entry, "result.json")
        doc = json.loads(raw)
        if not self.identity.check_key(checksum, doc["key"]):
            raise MisfiledEntry("the key does not digest to the checksum")
        outputs = {}
        for name, value in doc["outputs"].items():
            if isinstance(value, dict):
                blob = self._bytes(entry, value["npy"] + ".npy")
                if hashlib.sha256(blob).hexdigest() != value["npy"]:
                    raise CorruptEntry(f"{value['npy']}.npy is not its digest")
                value = np.load(io.BytesIO(blob), allow_pickle=False)
            outputs[name] = value
        return StoredEntry(
            outputs=outputs,
            context=context,
            key=doc["key"],
            digest=hashlib.sha256(raw).hexdigest(),
            ref=EntryRef(str(root.path), cdir_for(checksum), fingerprint),
        )


POLICY = StorePolicy(
    file_mode=tree.FILE_MODE,
    entry_mode=tree.ENTRY_MODE,
    sealed_mode=tree.SEALED_MODE,
    container_mode=tree.CONTAINER_MODE,
    side_mode=tree.SIDE_MODE,
)
DIRS = {
    "staging": tree.STAGING,
    "quarantine": tree.QUARANTINE,
    "sidecars": {"provenance": tree.SIDECAR},
}


def header(policy=POLICY, shard="prefix2", identity=None) -> LayoutHeader:
    return LayoutHeader.for_store(
        identity or ToyIdentity(),
        policy,
        shard=shard,
        entries=NEW,
        entry_format=ToyFormat.name,
        dirs=DIRS,
    )


def do_adopt(root, **kw):
    kw.setdefault("header", header())
    kw.setdefault("identity", ToyIdentity())
    kw.setdefault("entry_format", ToyFormat())
    return adopt(root, old_entries=tree.VERSION, **kw)


def open_store(root, **kw):
    kw.setdefault("policy", POLICY)
    return PortableStore(
        root, trust="trusted", identity=ToyIdentity(), entry_format=ToyFormat(), **kw
    )


def job_of(entry):
    return types.SimpleNamespace(
        checksum=entry.digest, context=entry.context, task=None
    )


def snapshot(top):
    out = {}
    for dirpath, dirnames, filenames in os.walk(top):
        for name in [".", *dirnames, *filenames]:
            path = os.path.normpath(os.path.join(dirpath, name))
            st = os.lstat(path)
            out[os.path.relpath(path, top)] = (
                st.st_ino,
                stat.S_IMODE(st.st_mode),
                st.st_mtime_ns,
            )
    return out


def test_adopt_renames_version_dir_and_writes_header(tmp_path):
    entries = tree.build_tree(tmp_path)
    before = snapshot(tmp_path / tree.VERSION)
    report = do_adopt(tmp_path, check="all")
    assert report.renamed == (tree.VERSION, NEW)
    assert report.header_written
    assert report.checked == len(entries)
    assert report.misfiled == []
    assert not (tmp_path / tree.VERSION).exists()
    stored = LayoutHeader.from_json((tmp_path / HEADER_NAME).read_bytes())
    assert stored == header()
    assert stored.identity["name"] == ToyIdentity.name
    assert snapshot(tmp_path / NEW) == before


def test_adopted_entries_served_without_reseal(tmp_path):
    entries = tree.build_tree(tmp_path)
    do_adopt(tmp_path)
    store = open_store(tmp_path)
    for entry in entries:
        found = store.find(job_of(entry))
        assert isinstance(found, Hit)
        assert found.result.outputs["score"] == entry.score
        np.testing.assert_array_equal(found.result.outputs["vec"], entry.vec)
        path = entry.path(tmp_path, NEW)
        assert stat.S_IMODE(os.lstat(path).st_mode) == tree.SEALED_MODE
        for name in os.listdir(path):
            mode = stat.S_IMODE(os.lstat(os.path.join(path, name)).st_mode)
            assert mode == tree.FILE_MODE
    # the tool's own side directories are used; none is made beside them
    assert sorted(os.listdir(tmp_path)) == sorted(
        [HEADER_NAME, NEW, tree.STAGING, tree.QUARANTINE, tree.SIDECAR]
    )


def test_adopted_entry_checked_against_address_key_and_context(tmp_path):
    good = tree.build_tree(tmp_path, n=1)[0]
    bad_context = tree.write_entry(tmp_path, 1, bad_context=True)
    bad_key = tree.write_entry(tmp_path, 2, bad_key=True)
    # the format's own identity accepts any key: the supplied identity still refuses it
    report = do_adopt(tmp_path, check="all", entry_format=ToyFormat(LaxIdentity()))
    assert report.checked == 3
    rel = {os.path.relpath(e.path(tmp_path, NEW), tmp_path / NEW): e for e in (good,)}
    assert sorted(report.misfiled) == sorted(
        os.path.relpath(e.path(tmp_path, NEW), tmp_path / NEW)
        for e in (bad_context, bad_key)
    )
    assert not set(report.misfiled) & set(rel)
    # nothing was moved by adopt
    for e in (good, bad_context, bad_key):
        assert os.path.isdir(e.path(tmp_path, NEW))
    store = open_store(tmp_path)
    ref = EntryRef(str(tmp_path), bad_context.digest, bad_context.fingerprint)
    with pytest.raises(MisfiledEntry):
        store.read(ref, None)
    assert store.read(EntryRef(str(tmp_path), good.digest, good.fingerprint), None)


def test_mistranslated_entry_quarantined(tmp_path, caplog):
    tree.prepare_root(tmp_path)
    bad = tree.write_entry(tmp_path, 0, bad_key=True)
    report = do_adopt(tmp_path)
    assert report.misfiled == [os.path.relpath(bad.path(tmp_path, NEW), tmp_path / NEW)]
    store = open_store(tmp_path)
    with caplog.at_level(logging.WARNING, logger="pydra"):
        assert isinstance(store.find(job_of(bad)), Miss)
    assert not os.path.lexists(bad.path(tmp_path, NEW))
    moved = os.listdir(tmp_path / tree.QUARANTINE)
    assert len(moved) == 1
    assert "corrupt" in caplog.text
    assert isinstance(store.find(job_of(bad)), Miss)


def test_old_writer_after_adoption_creates_unread_tree(tmp_path, caplog):
    tree.build_tree(tmp_path, n=1)
    do_adopt(tmp_path)
    late = tree.write_entry(tmp_path, 5)  # a writer that predates the header
    assert late.version == tree.VERSION
    with caplog.at_level(logging.WARNING, logger="pydra"):
        store = open_store(tmp_path)
        assert isinstance(store.find(job_of(late)), Miss)
        open_store(tmp_path).collect()
    warned = [r for r in caplog.records if tree.VERSION in r.getMessage()]
    assert len(warned) == 1
    assert "8.2" in warned[0].getMessage()
    assert os.path.isdir(late.path(tmp_path))  # never read, never moved


def test_header_mode_mismatch_refused(tmp_path):
    tree.build_tree(tmp_path)
    wrong = header(attrs.evolve(POLICY, sealed_mode=0o2500))
    with pytest.raises(AdoptRefused, match="sealed"):
        do_adopt(tmp_path, header=wrong)
    assert (tmp_path / tree.VERSION).is_dir()
    assert not (tmp_path / NEW).exists()
    assert not (tmp_path / HEADER_NAME).exists()
    wrong = header(attrs.evolve(POLICY, container_mode=0o755))
    with pytest.raises(AdoptRefused, match="container"):
        do_adopt(tmp_path, header=wrong)
    assert (tmp_path / tree.VERSION).is_dir()


def test_aside_names_collected_as_own(tmp_path):
    entry = tree.build_tree(tmp_path, n=1)[0]
    holder = os.path.join(tmp_path, tree.VERSION, entry.shard, entry.digest)
    aside = os.path.join(holder, f"{entry.fingerprint}.aside-1")
    other = os.path.join(holder, "not-ours")
    for path in (aside, other):
        os.mkdir(path)
        with open(os.path.join(path, "result.json"), "w") as f:
            f.write("{}")
    do_adopt(tmp_path)
    time.sleep(0.05)
    policy = attrs.evolve(POLICY, ages=StoreAges(quarantine=0.01))
    report = open_store(tmp_path, policy=policy).collect()
    new_holder = os.path.join(tmp_path, NEW, entry.shard, entry.digest)
    assert os.path.join(new_holder, f"{entry.fingerprint}.aside-1") in report.removed
    assert not os.path.lexists(os.path.join(new_holder, f"{entry.fingerprint}.aside-1"))
    assert os.path.isdir(os.path.join(new_holder, "not-ours"))
    assert os.path.isdir(entry.path(tmp_path, NEW))


def test_adopt_refuses_existing_header(tmp_path):
    tree.build_tree(tmp_path)
    do_adopt(tmp_path)
    with pytest.raises(AdoptRefused, match=HEADER_NAME):
        do_adopt(tmp_path)
    other = tmp_path / "other"
    tree.prepare_root(other)
    with pytest.raises(AdoptRefused, match=tree.VERSION):
        do_adopt(other)
    assert not (other / HEADER_NAME).exists()


def test_prefix2_shard_with_hex_digests(tmp_path):
    entries = tree.build_tree(tmp_path)
    for e in entries:
        assert re.fullmatch(r"[0-9a-f]{64}", e.digest)
        assert os.path.isdir(os.path.join(tmp_path, tree.VERSION, e.digest[:2]))
    with pytest.raises(AdoptRefused, match="shard"):
        do_adopt(tmp_path, header=header(shard="sha256-2"))
    assert (tmp_path / tree.VERSION).is_dir()
    do_adopt(tmp_path)
    store = open_store(tmp_path)
    assert store.header.shard == "prefix2"
    assert all(isinstance(store.find(job_of(e)), Hit) for e in entries)


def test_adopt_leaves_identity_marker(tmp_path):
    entry = tree.build_tree(tmp_path, n=1)[0]
    marker = tmp_path / ".pydra-identity"
    marker.write_text("pydra-identity@2\n")
    do_adopt(tmp_path)
    assert marker.read_text() == "pydra-identity@2\n"
    stored = LayoutHeader.from_json((tmp_path / HEADER_NAME).read_bytes())
    assert stored.identity["name"] == ToyIdentity.name
    assert isinstance(open_store(tmp_path).find(job_of(entry)), Hit)


def test_cli_store_adopt(tmp_path):
    entry = tree.build_tree(tmp_path, n=1)[0]
    spec = tmp_path.parent / f"{tmp_path.name}-header.json"
    spec.write_bytes(header().to_json())
    runner = CliRunner()
    result = runner.invoke(cli, ["store", "--help"])
    assert result.exit_code == 0
    assert "adopt" in result.output and "collect" in result.output
    module = __name__
    result = runner.invoke(
        cli,
        [
            "store",
            "adopt",
            str(tmp_path),
            "--from",
            tree.VERSION,
            "--header",
            str(spec),
            "--identity",
            f"{module}.ToyIdentity",
            "--entry-format",
            f"{module}.ToyFormat",
            "--check",
            "all",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / HEADER_NAME).exists()
    assert "1 checked" in result.output
    assert isinstance(open_store(tmp_path).find(job_of(entry)), Hit)
