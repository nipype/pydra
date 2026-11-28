"""Inputs are digested through ``pydra.utils.digest``; a ``Path`` is a value (RFC 0004
section 6.2), and arrays of references are digested by content at any depth."""

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
import typing as ty
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import pytest
import fileformats.application.misc as misc_formats
from fileformats.application import Pdf
from fileformats.audio import Wav
from fileformats.generic import Directory, DirectoryOf, File
from fileformats.text import Plain

from pydra.compose import python
from pydra.engine.portable import codecs
from pydra.engine.portable.entry import (
    EntryFormatUnknown,
    PydraEntryV1,
    UnencodableOutput,
)
from pydra.engine.tests.test_portable_entry import (  # noqa: F401
    Pair,
    entry_path,
    CHECKSUM,
    FP,
    TRUST,
    publish,
    read,
    root,
    tags,
)
from pydra.utils import digest as dg
from pydra.utils import hash as hash_module
from pydra.utils.digest import digest, fileset_digest, jcs, register_digest
from pydra.utils.digest import register_fileset_tag
from pydra.utils.hash import Cache, CacheKey, PersistentCache
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
    # the pure forms give the same string on every OS, so every OS compares both
    posix = digest(PurePosixPath("data/out.txt"))
    windows = digest(PureWindowsPath("data/out.txt"))
    assert (
        posix
        == "json@1:191649a04686e7f3b380cf014035e30a7cfc8764ecf0345a6595673841e5d552"
    )
    assert (
        windows
        == "json@1:edea6a6010173384280015cd79edf81ce4891efbda9d65fbf20728e1453f249a"
    )
    native = PureWindowsPath if os.name == "nt" else PurePosixPath
    assert digest(Path("data/out.txt")) == digest(native("data/out.txt"))
    assert _input(UsePath(p=Path("data/out.txt")), "p") == digest(
        native("data/out.txt")
    )


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


def test_default_file_digest_goldens(tmp_path):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "a.txt").write_bytes(b"a")
    (tree / "sub" / "b.txt").write_bytes(b"b")
    assert (
        digest(File(one))
        == "file@1:cb60bd61ad40972b19425f994664fc4710d5f2b73a2831e47ef51640f896bc2e"
    )
    # a fileformats class nobody registered in the test
    assert (
        digest(Plain(one))
        == "file@1:d7cb60faf2be2a3be24ce9429512c2170ce739dc31390654681248e39068e876"
    )
    assert (
        digest(Directory(tree))
        == "file@1:9970cdd49249537e111d29779221cf1b542ce6444207ae90745e5b7458260a2c"
    )
    assert _input(UseFile(f=one), "f") == digest(File(one))


def test_user_registration_wins_over_the_default(tmp_path):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    default = digest(File(one))
    register_fileset_tag(File, "test.inputs.Mine")
    assert digest(File(one)) != default
    assert digest(File(one)) == fileset_digest(
        "test.inputs.Mine", [(".", sha(b"content"))]
    )
    assert digest(Plain(one)).startswith("file@1:")


def test_file_set_subclass_outside_fileformats_keeps_its_content_hash(tmp_path):
    class Foreign(File):
        ext = ".txt"

    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    assert digest(Foreign(one)).startswith("py@1:")


def test_file_set_settle_window(tmp_path, monkeypatch):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    monkeypatch.setattr(hash_module, "_fileset_memory", hash_module._LRUCache(16))
    cache = Cache()
    location = cache.persistent.location
    # written moments ago: computed, neither held in memory nor written to the cache
    assert digest(File(one), cache) == digest(File(one))
    assert len(hash_module._fileset_memory) == 0
    assert not any(location.iterdir())
    # settled: the members are kept in both
    monkeypatch.setattr(
        hash_module, "time_ns", lambda: __import__("time").time_ns() + 10**10
    )
    settled = digest(File(one), cache)
    assert len(hash_module._fileset_memory) == 1
    assert any(location.iterdir())
    assert settled == digest(File(one))


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


# -- registered digests ---------------------------------------------------------------------

BUILT_IN_SCHEMES = ["json", "npy", "file", "py", "class", "code", "type"]


class Thing:
    pass


@pytest.mark.parametrize("scheme", BUILT_IN_SCHEMES)
def test_registered_digest_cannot_use_a_built_in_scheme(scheme):
    register_digest(Thing, lambda value: f"{scheme}@1:" + "ab" * 16, replace=True)
    with pytest.raises(ValueError, match=scheme):
        digest(Thing())
    with pytest.raises(ValueError, match=scheme):
        digest([Thing()])


def test_registered_digest_wins_over_npy_and_file(tmp_path):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    register_digest(File, lambda value: "mine@1:" + "cd" * 16)
    assert digest(File(one)) == "mine@1:" + "cd" * 16
    assert digest([File(one)]) == "json@1:" + sha(
        jcs([{"$digest": "mine@1:" + "cd" * 16}])
    )
    register_digest(np.ndarray, lambda value: "mine@1:" + "ef" * 16)
    assert digest(np.arange(3)) == "mine@1:" + "ef" * 16
    assert digest({"k": np.arange(3)}) == "json@1:" + sha(
        jcs({"k": {"$digest": "mine@1:" + "ef" * 16}})
    )


# -- default tags: any FileSet class, tag from mime_like, resolved on demand -----------------

WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\0" * 20


def test_default_tag_for_any_fileformats_class(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(WAV_BYTES)
    assert (
        digest(Wav(wav))
        == "file@1:25d722acb56193e38b3dab7dfa968edaf0791dcc597fd71d74b88775052f759d"
    )
    assert dg.tag_of(Wav) == "audio/wav"
    assert dg.tag_of(DirectoryOf[Plain]) == "text/plain+directory-of"


def test_no_registry_walk_for_a_class_without_a_mime_like(tmp_path):
    class Foreign(File):
        ext = ".txt"

    assert dg.tag_of(Foreign) is None


# the child imports nothing from a ``tests`` package: the wheel is built without them
FRESH_READER = textwrap.dedent("""
    import sys
    from pathlib import Path
    from pydra.engine.portable.entry import PydraEntryV1
    from pydra.engine.portable.identity import PydraIdentityV1
    from pydra.utils.storagedir import StorageDir, StorePolicy

    path, trust, checksum, fingerprint, entry_dir = sys.argv[1:]
    handle = StorageDir.open_root(Path(path), StorePolicy(), trust=trust)
    entry = handle
    for part in Path(entry_dir).relative_to(path).parts:
        entry = entry.child(part)
    with entry:
        out = PydraEntryV1(identity=PydraIdentityV1(), pickle=False, trust=trust).read(
            entry, checksum, fingerprint, None, trust, root=handle
        ).outputs
    for name in sorted(out):
        t = type(out[name])
        print(name, t.__module__, t.__qualname__, t.mime_like)
    """)


def test_default_tagged_sets_decode_in_a_fresh_process(root, tmp_path):  # noqa: F811
    src = tmp_path / "src"
    (src / "tree").mkdir(parents=True)
    (src / "one.bin").write_bytes(b"payload")
    (src / "tree" / "a.txt").write_bytes(b"a")
    (src / "a.wav").write_bytes(WAV_BYTES)
    publish(
        root,
        {
            "f": File(src / "one.bin"),
            "d": DirectoryOf[Plain](src / "tree"),
            "w": Wav(src / "a.wav"),
        },
    )
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            FRESH_READER,
            str(root.path),
            TRUST,
            CHECKSUM,
            FP,
            str(entry_path(root)),
        ],
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    lines = {ln.split()[0]: ln.split() for ln in done.stdout.splitlines()}
    assert lines["f"][1:3] == ["fileformats.generic.file", "File"]
    assert lines["d"][3] == "text/plain+directory-of"
    assert lines["w"][1:3] == ["fileformats.audio", "Wav"]


@python.define(outputs={"d": misc_formats.Pdf})
def MiscPdfOut(x: int) -> misc_formats.Pdf:
    raise NotImplementedError


def _published_pdf(root, tmp_path):  # noqa: F811
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4\nbody")
    publish(root, {"d": misc_formats.Pdf(src)})


def test_shared_tag_decodes_to_the_declared_class(root, tmp_path):  # noqa: F811
    assert misc_formats.Pdf.mime_like == Pdf.mime_like
    assert misc_formats.Pdf is not Pdf
    _published_pdf(root, tmp_path)
    got = read(root, task=MiscPdfOut(x=1)).outputs["d"]
    assert type(got) is misc_formats.Pdf


def test_shared_tag_decodes_to_the_resolvers_class_when_undeclared(
    root, tmp_path
):  # noqa: F811
    from fileformats.core import from_mime

    _published_pdf(root, tmp_path)
    got = read(root).outputs["d"]
    assert type(got) is from_mime("application/pdf")


def test_user_registration_of_a_default_tag_wins_and_two_user_ones_clash():
    register_fileset_tag(misc_formats.Pdf, "application/pdf")
    assert dg.lookup_tag("application/pdf").type is misc_formats.Pdf
    with pytest.raises(ValueError, match="already registered"):
        register_fileset_tag(Pdf, "application/pdf")


def test_registered_digest_survives_a_later_tag_registration(tmp_path):
    one = tmp_path / "one.txt"
    one.write_bytes(b"content")
    register_digest(File, lambda value: "mine@1:" + "cd" * 16)
    register_fileset_tag(File, "test.inputs.Later")
    assert digest(File(one)) == "mine@1:" + "cd" * 16
    assert dg.tag_of(File) == "test.inputs.Later"


def test_mock_file_set_is_not_read_as_files(tmp_path):
    mock = File.mock(tmp_path / "never-written.txt")
    assert dg.tag_of(type(mock)) is None
    assert digest(mock).startswith("py@1:")


# -- decoding order, declared types and unresolvable tags ----------------------------------


@python.define(outputs={"d": list[misc_formats.Pdf]})
def MiscPdfList(x: int) -> list[misc_formats.Pdf]:
    raise NotImplementedError


@python.define(outputs={"d": ty.Optional[misc_formats.Pdf]})
def MiscPdfOptional(x: int) -> ty.Optional[misc_formats.Pdf]:
    raise NotImplementedError


def test_declared_type_arguments_are_searched(root, tmp_path):  # noqa: F811
    src = tmp_path / "doc.pdf"
    src.write_bytes(b"%PDF-1.4\nbody")
    publish(root, {"d": [misc_formats.Pdf(src)]})
    got = read(root, task=MiscPdfList(x=1)).outputs["d"]
    assert [type(g) for g in got] == [misc_formats.Pdf]
    codec = dg.lookup_fileset_tag(
        "application/pdf", codecs._fileset_classes(ty.Optional[misc_formats.Pdf])
    )
    assert codec.type is misc_formats.Pdf


def test_a_class_given_its_own_tag_still_reads_its_default_tag():
    register_fileset_tag(Plain, "my/plain")
    assert dg.tag_of(Plain) == "my/plain"
    assert dg.lookup_fileset_tag("text/plain").type is Plain
    assert dg.lookup_fileset_tag("text/plain", [Plain]).type is Plain
    assert dg.lookup_fileset_tag("my/plain").type is Plain


def test_lookup_order_is_declared_then_user_then_resolver():
    register_fileset_tag(Directory, "generic/file")
    # the declared type, whose own mime_like is the stored tag, wins over a registration
    assert dg.lookup_fileset_tag("generic/file", [File]).type is File
    # without one, the user registration wins over the resolver
    assert dg.lookup_fileset_tag("generic/file").type is Directory
    assert dg.lookup_fileset_tag("generic/directory").type is Directory


def test_a_default_tag_is_not_entered_in_the_registry(tmp_path):
    register_digest(File, lambda value: "mine@1:" + "cd" * 16)
    assert "generic/file" not in dg._BY_TAG
    assert dg.lookup_fileset_tag("generic/file").type is File


def test_unresolvable_tag_is_py_and_never_written_as_a_file_set(
    root, tmp_path  # noqa: F811
):
    from fileformats.image.base import Image

    one = tmp_path / "pic.bin"
    one.write_bytes(b"not really an image")
    assert Image.mime_like == "image/image"
    assert dg.lookup_fileset_tag("image/image") is None
    assert dg.tag_of(Image) is None
    assert digest(Image(one)).startswith("py@1:")
    with pytest.raises(UnencodableOutput, match="Image") as err:
        publish(root, {"i": Image(one)})
    assert "image/image" in str(err.value)
    # a file set is refused where pickles are allowed too, and the message says why
    with pytest.raises(UnencodableOutput, match="never pickled") as err:
        publish(root, {"i": Image(one)}, fmt=PydraEntryV1(pickle=True, trust=TRUST))
    assert "cannot pickle" not in str(err.value)
    assert not entry_path(root).exists()
    register_fileset_tag(Image, "my/image")  # a user registration is written
    publish(root, {"i": Image(one)})


@python.define(outputs={"w": File})
def FileOut(x: int) -> File:
    raise NotImplementedError


def test_a_declared_parent_class_reads_the_subclass_entry(root, tmp_path):  # noqa: F811
    wav = tmp_path / "a.wav"
    wav.write_bytes(WAV_BYTES)
    publish(root, {"w": Wav(wav)})
    got = read(root, task=FileOut(x=1)).outputs["w"]
    assert type(got) is Wav


def test_a_declared_class_given_its_own_tag_reads_its_default_tag():
    register_fileset_tag(misc_formats.Pdf, "my/pdf")
    codec = dg.lookup_fileset_tag("application/pdf", [misc_formats.Pdf])
    assert codec.type is misc_formats.Pdf


def test_a_format_mismatch_after_verified_members_is_unknown_not_corrupt(
    root, tmp_path  # noqa: F811
):
    one = tmp_path / "one.bin"
    one.write_bytes(b"payload")
    publish(root, {"f": File(one)})  # as another process, with the default tags
    register_fileset_tag(Directory, "generic/file")
    with pytest.raises(EntryFormatUnknown, match="Directory"):
        read(root)
    assert entry_path(root).exists()
    assert type(read(root, task=FileOutF(x=1)).outputs["f"]) is File


@python.define(outputs={"f": File})
def FileOutF(x: int) -> File:
    raise NotImplementedError


BROKEN_PLUGIN = textwrap.dedent("""
    import sys
    from pathlib import Path
    from fileformats.generic import File
    from fileformats.image import Png
    from fileformats.text import Plain
    from pydra.utils import digest as dg

    d = Path(sys.argv[1])
    for cls, name in ((File, "one.bin"), (Plain, "a.txt"), (Png, "a.png")):
        try:
            print(cls.__name__, dg.tag_of(cls), dg.digest(cls(d / name)).split(":")[0])
        except ImportError as err:
            print(cls.__name__, "ImportError", err)
    """)


def test_a_broken_fileformats_plugin_never_changes_a_digest(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "one.bin").write_bytes(b"payload")
    (data / "a.txt").write_text("hi")
    (data / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 16)
    plugin = tmp_path / "plugins" / "fileformats" / "broken"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text("raise ImportError('broken plugin')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(tmp_path / "plugins")] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    done = subprocess.run(
        [sys.executable, "-c", BROKEN_PLUGIN, str(data)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert done.returncode == 0, done.stderr
    rows = [ln.split(None, 2) for ln in done.stdout.splitlines()]
    assert [r[0] for r in rows] == ["File", "Plain", "Png"]
    for row in rows:
        # never a silent fall back to py@1: the digest is file@1 under the class's tag,
        # or the plugin's import error reaches the caller
        if row[1] == "ImportError":
            assert "broken plugin" in row[2]
        else:
            assert row[2] == "file@1", row
    expected = {"File": "generic/file", "Plain": "text/plain", "Png": "image/png"}
    assert all(r[1] in (expected[r[0]], "ImportError") for r in rows), rows
