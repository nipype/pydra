import dataclasses
import importlib.metadata
import json
import os
import shutil
import subprocess
import platform
import sys
import uuid
from pathlib import Path
import cloudpickle
import numpy as np
import pytest
from fileformats.generic import File
from pydra.compose import python
from pydra.engine import context
from pydra.engine.job import Job
from pydra.engine.result import Result
from pydra.engine.store import (
    CONTEXT_FILE,
    DeclaredVersionWarning,
    PickleStore,
    context_key,
)
from pydra.engine.submitter import Submitter
from pydra.environments import native
import pydra.compose.python
from pydra.utils.general import STDLIB_MODULES
from pydra.utils.hash import (
    CODE_HASH_TIERS,
    USER_PACKAGES_ENV_VAR,
    code_digest,
    hash_function,
    library_modules,
    type_module,
)

cloudpickle.register_pickle_by_value(sys.modules[__name__])


class FakeDist:
    """An installed distribution, owning `files` under `base`"""

    def __init__(
        self, name, version, direct_url=None, base=None, files=(), requires=()
    ):
        self.metadata = {"Name": name}
        self.version = version
        self.direct_url = direct_url
        self.base = Path(base or "/nowhere")
        self.files = list(files)
        self.requires = list(requires)

    def locate_file(self, path):
        return self.base / path

    def read_text(self, name):
        if name == "direct_url.json" and self.direct_url is not None:
            return json.dumps(self.direct_url)
        if name == "RECORD":
            return "\n".join(self.files)
        return None


PYDRA_ROOT = Path(pydra.compose.python.__file__).parents[2]
PYDRA_FILES = [str(p.relative_to(PYDRA_ROOT)) for p in PYDRA_ROOT.glob("pydra/**/*.py")]


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """Installs ``fakelib`` (requiring ``base``) as library code, and sets the
    installed distributions, the Python version and the C library pydra reads, at the
    importlib.metadata and platform boundary. Each call stands for a new process."""
    site = tmp_path / "site-packages"
    (site / "fakelib").mkdir(parents=True)
    (site / "fakelib" / "__init__.py").write_text("def scale(a):\n    return 2 * a\n")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "fakelib", raising=False)
    requires = ["base>=1", "py2only; python_version < '3'", "extra; extra == 'x'"]

    def install(python="3.13.1", libc=("glibc", "2.35"), **versions):
        specs = {
            "pydra": ("1.0", None, PYDRA_ROOT, PYDRA_FILES),
            "fakelib": ("1.0", None, site, ["fakelib/__init__.py"], requires),
            "orphan-owner": ("1.0", None, site, []),
            "base": "1.0",
            "other": "1.0",
            **versions,
        }
        if isinstance(specs["fakelib"], str):
            specs["fakelib"] = _fakelib(specs["fakelib"])
        version, url, base, files, reqs = specs["fakelib"]
        specs["fakelib"] = (version, url, base or site, files, reqs)
        dists = [
            FakeDist(n, *v) if isinstance(v, tuple) else FakeDist(n, v)
            for n, v in specs.items()
        ]
        monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {n: [n] for n in specs},
        )
        monkeypatch.setattr(platform, "python_version", lambda: python)
        monkeypatch.setattr(platform, "libc_ver", lambda: libc)
        context.clear_caches()

    install.site = site
    install()
    yield install
    monkeypatch.undo()
    context.clear_caches()


def _fakelib(version, direct_url=None, files=("fakelib/__init__.py",), base=None):
    return (version, direct_url, base, list(files), ["base>=1"])


def _record_run(log):
    Path(log).mkdir(exist_ok=True)
    (Path(log) / uuid.uuid4().hex).touch()


def _scaled_task():
    import fakelib

    def scaled(a: int, log: str) -> int:
        _record_run(log)
        return fakelib.scale(a)

    scaled.__globals__["fakelib"] = fakelib
    return python.define(scaled)


def _runs(log):
    return len(list(Path(log).iterdir()))


def _run_twice(tmp_path, installed, change, reuse, worker="debug"):
    """Run the task, change the installation, run it again under `reuse`; return the
    number of runs and the second result"""
    task = _scaled_task()
    log = tmp_path / "log"
    with Submitter(worker=worker, cache_root=tmp_path / "cache", reuse=reuse) as sub:
        first = sub(task(a=1, log=str(log)))
    installed(**change)
    with Submitter(worker=worker, cache_root=tmp_path / "cache", reuse=reuse) as sub:
        second = sub(task(a=1, log=str(log)))
    assert first.outputs.out == second.outputs.out == 2
    return _runs(log), first, second


def test_record_layout(installed):
    task = _scaled_task()(a=1, log="x")
    record = context.job_context(task, native.Native())
    assert record == {
        "record": context.RECORD,
        "python": "3.13.1",
        "implementation": sys.implementation.name,
        "platform": f"{sys.platform}-{platform.machine()}-glibc2.35",
        "environment": native.Native().checksum(),
        "packages": {"base": "1.0", "fakelib": "1.0", "pydra": "1.0"},
        "installed": record["installed"],
    }
    installed(other="2.0")
    changed = context.job_context(task, native.Native())
    assert changed["installed"] != record["installed"]
    assert {k: v for k, v in changed.items() if k != "installed"} == {
        k: v for k, v in record.items() if k != "installed"
    }


def test_native_checksum_ignores_versions(installed):
    checksum = native.Native().checksum()
    installed(pydra="2.0", python="3.14.0")
    assert native.Native().checksum() == checksum


@pytest.mark.parametrize(
    "change, reuse, runs",
    [
        ({"fakelib": "2.0"}, "consistent", 2),
        ({"fakelib": "2.0"}, "any", 1),
        ({"other": "2.0"}, "consistent", 1),
        ({"other": "2.0"}, "exact", 2),
        ({"python": "3.14.0"}, "consistent", 2),
        ({"python": "3.13.2"}, "consistent", 1),
        ({"python": "3.13.2"}, "exact", 2),
        ({"libc": ("glibc", "2.39")}, "consistent", 2),
        ({"base": "2.0"}, "consistent", 2),
        ({"base": "2.0"}, "any", 1),
        ({"fakelib": _fakelib("1.0", {"url": "file:///x", "dir_info": {}})}, "any", 1),
    ],
)
def test_reuse_policies(installed, tmp_path, change, reuse, runs):
    n, first, second = _run_twice(tmp_path, installed, change, reuse)
    assert n == runs
    assert second.cross_context == (runs == 1)
    if runs == 1:
        assert second.context == first.context
        assert second.reused_in != first.context
        assert second.reused_in["environment"] == first.context["environment"]
    else:
        assert second.reused_in is None
        assert second.context != first.context


def test_any_records_both_contexts(installed, tmp_path):
    _, first, second = _run_twice(tmp_path, installed, {"fakelib": "2.0"}, "any")
    assert second.context["packages"]["fakelib"] == "1.0"
    assert second.reused_in["packages"]["fakelib"] == "2.0"


@pytest.mark.parametrize("reuse", ["consistent", "exact"])
def test_unknown_is_reused_by_no_other_process(installed, tmp_path, reuse, caplog):
    """An unknown version (an editable source that is gone) gives an equal record in
    a new process, which still recomputes; within one process the result is found"""
    unknown = _fakelib("1.0", {"url": "file:///gone", "dir_info": {"editable": True}})
    installed(fakelib=unknown)
    n, first, second = _run_twice(tmp_path, installed, {"fakelib": unknown}, reuse)
    assert first.context["packages"]["fakelib"] is None
    assert {k: v for k, v in first.context.items() if k != "unknown"} == {
        k: v for k, v in second.context.items() if k != "unknown"
    }
    assert n == 2
    assert "'fakelib' is unknown, because its editable source" in caplog.text
    task = _scaled_task()
    with Submitter(cache_root=tmp_path / "cache", reuse=reuse) as sub:
        assert sub(task(a=1, log=str(tmp_path / "log"))).outputs.out == 2
    assert _runs(tmp_path / "log") == 2


def _editable(installed, monkeypatch, source):
    """Install ``fakelib`` editable from `source`, as library code"""
    monkeypatch.setenv(USER_PACKAGES_ENV_VAR, "-fakelib")
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.delitem(sys.modules, "fakelib", raising=False)
    shutil.rmtree(installed.site / "fakelib")
    url = {"url": source.as_uri(), "dir_info": {"editable": True}}
    return _fakelib("1.0", url, files=["fakelib.pth"])


@pytest.mark.parametrize("git", [False, True])
def test_editable_source_is_identified_by_content(
    installed, tmp_path, monkeypatch, git
):
    source = tmp_path / "src"
    (source / "fakelib").mkdir(parents=True)
    module = source / "fakelib" / "__init__.py"
    module.write_text("def scale(a):\n    return 2 * a\n")
    if git:
        if shutil.which("git") is None:
            pytest.skip("no git")
        for cmd in (["init", "-q"], ["add", "."]):
            subprocess.run(["git", "-C", str(source), *cmd], check=True)
        subprocess.run(
            ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@t"]
            + ["commit", "-qm", "x"],
            check=True,
        )
    fakelib = _editable(installed, monkeypatch, source)
    installed(fakelib=fakelib)
    unchanged, first, second = _run_twice(
        tmp_path, installed, {"fakelib": fakelib}, "consistent"
    )
    identity = first.context["packages"]["fakelib"]
    assert identity.startswith("1.0@git:" if git else "1.0@dir:")
    assert unchanged == 1 and second.context == first.context
    module.write_text("def scale(a):\n    return 2 * a  # edited\n")
    with Submitter(cache_root=tmp_path / "cache") as sub:
        installed(fakelib=fakelib)  # a new process
        result = sub(_scaled_task()(a=1, log=str(tmp_path / "log")))
    assert result.context["packages"]["fakelib"] not in (None, identity)
    assert _runs(tmp_path / "log") == 2


def test_identity_of_direct_installs(installed):
    installed(
        fakelib=_fakelib("1.0", {"url": "git+x", "vcs_info": {"commit_id": "abc"}}),
        other=("1.0", {"url": "x.whl", "archive_info": {"hashes": {"sha256": "f"}}}),
    )
    assert context.package_versions({"fakelib", "absent"}) == {
        "absent": None,
        "base": "1.0",
        "fakelib": "1.0@abc",
    }
    assert context.package_versions({"other"}) == {"other": "1.0@sha256=f"}
    local = _fakelib("1.0", {"url": "file:///x", "dir_info": {}})
    installed(fakelib=local)
    first = context.package_versions({"fakelib"})["fakelib"]
    assert first.startswith("1.0@record:")
    installed(fakelib=_fakelib("1.0", {"url": "file:///x", "dir_info": {}}, ["a"]))
    assert context.package_versions({"fakelib"})["fakelib"] != first


def test_installed_digest_uses_reached_identities_only(installed):
    def digest(**versions):
        installed(**versions)
        task = _scaled_task()(a=1, log="x")
        return context.job_context(task, native.Native())["installed"]

    def vcs(commit):
        return {"url": "git+x", "vcs_info": {"commit_id": commit}}

    reached = digest(fakelib=_fakelib("1.0", vcs("abc")))
    assert digest(fakelib=_fakelib("1.0", vcs("def"))) != reached
    unreached = digest(other=("1.0", vcs("abc")))
    assert digest(other=("1.0", vcs("def"))) == unreached


def _large_editable(tmp_path, name, files):
    source = tmp_path / name
    (source / name).mkdir(parents=True)
    for i in range(files):
        (source / name / f"m{i}.py").write_text(f"x = {i}\n")
    return ("1.0", {"url": source.as_uri(), "dir_info": {"editable": True}})


def test_unreached_editable_is_never_read(installed, tmp_path, monkeypatch):
    read = []
    monkeypatch.setattr(
        context, "_identify", lambda d: read.append(d.name) or ("1.0", None)
    )
    installed(bigdist=_large_editable(tmp_path, "bigdist", 3))
    task = _scaled_task()(a=1, log="x")
    record = context.job_context(task, native.Native())
    assert "bigdist" not in record["packages"]
    assert sorted(read) == ["base", "fakelib", "pydra"]


def test_reached_editable_over_the_cap_is_unknown(
    installed, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(context, "IDENTITY_MAX_FILES", 2)
    source = tmp_path / "src"
    _large_editable(tmp_path, "src", 0)
    (source / "fakelib").mkdir()
    for i in range(3):
        (source / "fakelib" / f"m{i}.py").write_text("")
    (source / "fakelib" / "__init__.py").write_text("def scale(a):\n    return 2 * a\n")
    installed(fakelib=_editable(installed, monkeypatch, source))
    record = context.job_context(_scaled_task()(a=1, log="x"), native.Native())
    assert record["packages"]["fakelib"] is None and "unknown" in record
    assert "it has more than 2 files" in caplog.text


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_context_file_gets_the_umask_mode(tmp_path):
    store = PickleStore(tmp_path)
    old = os.umask(0o027)
    try:
        with store.acquire("abc", {"a": 1}) as lease:
            store.commit(lease, Result(cache_dir=lease.cache_dir, context={"a": 1}))
    finally:
        os.umask(old)
    slot = store.cache_dir("abc", {"a": 1})
    assert (slot / CONTEXT_FILE).stat().st_mode & 0o777 == 0o640
    assert [p.name for p in slot.iterdir() if p.name.startswith(CONTEXT_FILE)] == [
        CONTEXT_FILE
    ]


def test_namespace_package_records_its_owner_only(installed, monkeypatch):
    for part in ("one", "two"):
        (installed.site / "nsp" / part).mkdir(parents=True)
        (installed.site / "nsp" / part / "__init__.py").write_text("")
    monkeypatch.delitem(sys.modules, "nsp", raising=False)
    import nsp.one  # noqa: F401

    files = {p: [f"nsp/{p}/__init__.py"] for p in ("one", "two")}
    installed(
        **{
            "nsp-one": ("1.0", None, installed.site, files["one"]),
            "nsp-two": ("1.0", None, installed.site, files["two"]),
        }
    )
    monkeypatch.setattr(
        importlib.metadata,
        "packages_distributions",
        lambda: {"nsp": ["nsp-one", "nsp-two"], "fakelib": ["fakelib"]},
    )
    context.clear_caches()
    assert context.package_versions({"nsp.one"}) == {"nsp-one": "1.0"}
    assert context.package_versions({"nsp"}) == {"nsp-one": "1.0", "nsp-two": "1.0"}
    (installed.site / "orphan.py").write_text("")
    monkeypatch.delitem(sys.modules, "orphan", raising=False)
    import orphan  # noqa: F401

    assert context.package_versions({"orphan"}) == {"orphan": None}


def test_worker_records_the_submitters_packages(installed, tmp_path):
    """A cf worker, which sees the real installation, records the context the job
    carried from the submitting process"""
    task = _scaled_task()
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        job = Job(task=task(a=1, log=str(tmp_path / "log")), name="j", submitter=sub)
        expected = job.context
        result = sub(task(a=1, log=str(tmp_path / "log")))
    assert expected["packages"] == {"base": "1.0", "fakelib": "1.0", "pydra": "1.0"}
    assert result.context == expected
    stored = json.loads((result.cache_dir / CONTEXT_FILE).read_text())
    assert stored == expected


GLOBAL_ARRAY_SCALE = np.float64(2)


def _uses_numpy_value(a):
    return a * GLOBAL_ARRAY_SCALE


def _uses_numpy_module(a):
    return np.sum(a)


def _helper(a):
    return _uses_numpy_module(a)


def _calls_helper(a):
    return _helper(a)


def _tops(modules):
    return {m.partition(".")[0] for m in modules}


def test_library_modules_of_the_walk():
    assert _tops(library_modules(_uses_numpy_module)) == {"numpy"}
    # a library-typed value, which is serialized again on every walk, memoised or not
    assert _tops(library_modules(_uses_numpy_value)) == {"numpy"}
    assert _tops(library_modules(_uses_numpy_value)) == {"numpy"}
    # through user code, whatever the task's own tier
    assert _tops(library_modules(_calls_helper)) == {"numpy"}
    assert library_modules(json.dumps) == {"json"}


def test_checksum_collects_the_modules(installed, monkeypatch):
    """The record takes the modules from the checksum's walk, not a second one"""

    def fail(func):
        raise AssertionError("walked again")

    task = python.define(_calls_helper)(a=1)
    task._checksum
    assert "numpy" in _tops(task._code_modules)
    monkeypatch.setattr(context, "library_modules", fail)
    assert "numpy" in _tops(context.reached_modules(task))


@dataclasses.dataclass
class Holder:
    value: object


def test_reached_modules_include_class_and_input_types(installed, tmp_path):
    @python.define
    def Reads(f: File, xs: list, held: object) -> int:
        return 1

    (tmp_path / "a.txt").write_text("a")
    task = Reads(f=File(tmp_path / "a.txt"), xs=[1], held=Holder([np.float64(1)]))
    task._checksum
    assert _tops(context.reached_modules(task)) == {"fileformats", "numpy", "pydra"}


def test_extension_type_reporting_builtins(installed):
    import fakelib

    fakelib.Ext = type("Ext", (), {"__module__": "builtins"})
    assert type_module(fakelib.Ext) == "fakelib"
    # an interpreter type is attributed to a standard library module, whichever
    assert type_module(type(None)).partition(".")[0] in STDLIB_MODULES
    assert type_module(int) == "builtins"


def test_unhashable_walk_leaves_packages_unknown(installed, monkeypatch, caplog):
    task = python.define(_calls_helper, version="1")(a=1)

    def fail(func):
        raise TypeError("unhashable")

    monkeypatch.setattr(context, "library_modules", fail)
    record = context.job_context(task, native.Native())
    assert record["packages"] is None and "unknown" in record
    assert "cannot be determined" in caplog.text and "unhashable" in caplog.text


def test_mismatched_slot_is_not_loaded(tmp_path):
    """A slot whose context is rejected is never unpickled, so a result written by
    other versions is a miss, not an error; a slot without a context is skipped"""
    store = PickleStore(tmp_path, reuse="consistent")
    current = {"record": context.RECORD, "python": "3.13.1"}
    for name, ctx in (("other", {"python": "2.7"}), ("legacy", None)):
        slot = tmp_path / "abc" / name
        slot.mkdir(parents=True)
        (slot / "_result.pklz").write_bytes(b"not a pickle")
        if ctx is not None:
            (slot / CONTEXT_FILE).write_text(json.dumps(ctx))
    assert store.lookup("abc", current, None) is None
    assert not (tmp_path / "abc" / context_key(current)).exists()


def test_file_urls_of_windows_paths(monkeypatch):
    import nturl2path

    monkeypatch.setattr(context, "url2pathname", nturl2path.url2pathname)
    assert context._file_url_path("file:///C:/src/my%20lib") == "C:\\src\\my lib"
    assert context._file_url_path("https://x/y") is None


def test_editable_identity_ignores_loaded_modules(installed, tmp_path, monkeypatch):
    """The packages an editable install provides come from its metadata, so its
    identity is the same whichever of its modules a process has imported"""
    source = tmp_path / "src"
    (source / "fakelib").mkdir(parents=True)
    (source / "fakelib" / "__init__.py").write_text("def scale(a):\n    return 2 * a\n")
    (source / "scratch.py").write_text("x = 1\n")
    fakelib = _editable(installed, monkeypatch, source)
    installed(fakelib=fakelib)
    before = context.package_versions({"fakelib"})["fakelib"]
    monkeypatch.delitem(sys.modules, "scratch", raising=False)
    import scratch  # noqa: F401

    installed(fakelib=fakelib)
    assert context.package_versions({"fakelib"})["fakelib"] == before
    (source / "scratch.py").write_text("x = 2\n")
    installed(fakelib=fakelib)
    assert context.package_versions({"fakelib"})["fakelib"] == before


def _versioned(a: int, log: str) -> int:
    _record_run(log)
    return a + 1


def _versioned_edit(a: int, log: str) -> int:
    _record_run(log)
    return a + 2


def _make_versioned(k):
    def added(a: int, log: str) -> int:
        _record_run(log)
        return a + k

    return added


@pytest.fixture
def edited(monkeypatch):
    """`_versioned_edit` standing for an edit of `_versioned`: same module and name"""
    monkeypatch.setattr(_versioned_edit, "__qualname__", _versioned.__qualname__)
    monkeypatch.setattr(_versioned_edit, "__name__", _versioned.__name__)
    return _versioned_edit


def _run_each(tmp_path, tasks, reuse="consistent", worker="debug"):
    """Run each task in turn on the same cache; return the number of runs and results"""
    log = tmp_path / "log"
    results = []
    for task in tasks:
        with Submitter(
            worker=worker, cache_root=tmp_path / "cache", reuse=reuse
        ) as sub:
            results.append(sub(task(a=1, log=str(log))))
    return _runs(log), results


@pytest.mark.parametrize("code_hash", CODE_HASH_TIERS)
def test_version_keeps_the_values_read(code_hash):
    with pytest.warns(UserWarning, match="version='1'"):
        two = python.define(_make_versioned(2), version="1", code_hash=code_hash)
        three = python.define(_make_versioned(3), version="1", code_hash=code_hash)
        again = python.define(_make_versioned(2), version="1", code_hash=code_hash)
    assert two(a=1, log="x")._checksum != three(a=1, log="x")._checksum
    assert two(a=1, log="x")._checksum == again(a=1, log="x")._checksum
    # the code is the same, only the values differ
    assert code_digest(two) == code_digest(three)


@pytest.mark.parametrize("code_hash", CODE_HASH_TIERS)
def test_version_replaces_the_code(code_hash, edited):
    first = python.define(_versioned, version="1", code_hash=code_hash)
    edit = python.define(edited, version="1", code_hash=code_hash)
    bumped = python.define(edited, version="2", code_hash=code_hash)
    plain = python.define(_versioned, code_hash=code_hash)
    checksums = [t(a=1, log="x")._checksum for t in (first, edit, bumped, plain)]
    assert checksums[0] == checksums[1]
    assert len(set(checksums[1:])) == 3
    assert code_digest(first) != code_digest(edit)
    assert code_digest(edit) == code_digest(bumped) == code_digest(edited, code_hash)
    # the code part is tagged, so a versioned task never shares a code-hashed one's key
    assert hash_function(first(a=1, log="x")) != hash_function(plain(a=1, log="x"))


def test_version_records_the_code_digest(installed):
    task = python.define(_versioned, version="1")(a=1, log="x")
    task._checksum
    record = context.job_context(task, native.Native())
    assert record["code"] == task._code_digest == code_digest(type(task))
    assert "code" not in context.job_context(
        python.define(_versioned)(a=1, log="x"), native.Native()
    )
    # computed again if the checksum was not
    fresh = python.define(_versioned, version="1")(a=1, log="x")
    assert context.job_context(fresh, native.Native())["code"] == record["code"]


@pytest.mark.parametrize("worker", ["debug", "cf"])
@pytest.mark.parametrize(
    "reuse,runs,outs",
    [("consistent", 1, [2, 2]), ("any", 1, [2, 2]), ("exact", 2, [2, 3])],
)
def test_version_guard(installed, tmp_path, edited, worker, reuse, runs, outs):
    """The same version over other code warns under every policy; exact recomputes"""
    first = python.define(_versioned, version="1")
    edit = python.define(edited, version="1")
    with pytest.warns(DeclaredVersionWarning) as record:
        n, results = _run_each(tmp_path, [first, edit], reuse, worker)
    assert (n, [r.outputs.out for r in results]) == (runs, outs)
    assert results[1].cross_context == (runs == 1)
    (message,) = {
        str(w.message) for w in record if issubclass(w.category, DeclaredVersionWarning)
    }
    assert "_versioned declares version='1'" in message
    assert code_digest(first) in message and code_digest(edit) in message
    assert ("is reused" if runs == 1 else "is not reused") in message
    assert message.endswith("bump version= or remove it")


@pytest.mark.parametrize("worker", ["debug", "cf"])
@pytest.mark.parametrize("reuse", ["consistent", "any", "exact"])
def test_version_guard_is_quiet(installed, tmp_path, edited, worker, reuse, recwarn):
    """Bumping the version, keeping the code, or versioning a code-hashed task
    neither reuses another's result nor warns"""
    first = python.define(_versioned, version="1")
    tasks = [
        python.define(_versioned),
        first,
        first,
        python.define(edited, version="2"),
    ]
    n, results = _run_each(tmp_path, tasks, reuse, worker)
    assert (n, [r.outputs.out for r in results]) == (3, [2, 2, 2, 3])
    assert not results[2].cross_context
    assert not [w for w in recwarn if issubclass(w.category, DeclaredVersionWarning)]
