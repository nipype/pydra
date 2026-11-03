import importlib.metadata
import json
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
from pydra.engine.store import CONTEXT_FILE, PickleStore, context_key
from pydra.engine.submitter import Submitter
from pydra.environments import native
from pydra.utils.hash import library_packages

cloudpickle.register_pickle_by_value(sys.modules[__name__])


class FakeDist:
    def __init__(self, name, version, direct_url=None):
        self.metadata = {"Name": name}
        self.version = version
        self.direct_url = direct_url

    def read_text(self, name):
        if name == "direct_url.json" and self.direct_url is not None:
            return json.dumps(self.direct_url)
        return None


BASE = {"pydra": "1.0", "fakelib": "1.0", "other": "1.0"}


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """Installs ``fakelib`` as library code and sets the versions of the installed
    distributions, the Python version and the C library pydra reads, at the
    importlib.metadata and platform boundary"""
    site = tmp_path / "site-packages"
    (site / "fakelib").mkdir(parents=True)
    (site / "fakelib" / "__init__.py").write_text("def scale(a):\n    return 2 * a\n")
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.delitem(sys.modules, "fakelib", raising=False)
    state = {}

    def install(python="3.13.1", libc=("glibc", "2.35"), **versions):
        dists = [
            FakeDist(n, *v) if isinstance(v, tuple) else FakeDist(n, v)
            for n, v in {**BASE, **versions}.items()
        ]
        monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: {n: [n] for n in BASE},
        )
        monkeypatch.setattr(platform, "python_version", lambda: python)
        monkeypatch.setattr(platform, "libc_ver", lambda: libc)
        context.clear_caches()
        state.update(python=python)

    install()
    yield install
    monkeypatch.undo()
    context.clear_caches()


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
        "packages": {"fakelib": "1.0", "pydra": "1.0"},
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
        ({"fakelib": ("1.0", {"url": "file:///x", "dir_info": {}})}, "consistent", 2),
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


def test_unknown_versions_are_different(installed, tmp_path):
    """Unknown on both sides, differing only in an unreached distribution"""
    unknown = ("1.0", {"url": "file:///x", "dir_info": {"editable": True}})
    installed(fakelib=unknown)
    change = {"fakelib": unknown, "other": "2.0"}
    n, first, second = _run_twice(tmp_path, installed, change, "consistent")
    assert first.context["packages"]["fakelib"] is None
    assert n == 2


def test_identity_of_direct_installs(installed):
    installed(
        fakelib=("1.0", {"url": "git+x", "vcs_info": {"commit_id": "abc"}}),
        other=("1.0", {"url": "x.whl", "archive_info": {"hashes": {"sha256": "f"}}}),
    )
    assert context.package_versions({"fakelib", "other", "absent"}) == {
        "absent": None,
        "fakelib": "1.0@abc",
        "other": "1.0@sha256=f",
    }


def test_worker_records_the_submitters_packages(installed, tmp_path):
    """A cf worker, which sees the real installation, records the context the job
    carried from the submitting process"""
    task = _scaled_task()
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        job = Job(task=task(a=1, log=str(tmp_path / "log")), name="j", submitter=sub)
        expected = job.context
        result = sub(task(a=1, log=str(tmp_path / "log")))
    assert expected["packages"] == {"fakelib": "1.0", "pydra": "1.0"}
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


def test_library_packages_of_the_walk():
    assert library_packages(_uses_numpy_module) == {"numpy"}
    # a library-typed value, which is serialized again on every walk, memoised or not
    assert library_packages(_uses_numpy_value) == {"numpy"}
    assert library_packages(_uses_numpy_value) == {"numpy"}
    # through user code, whatever the task's own tier
    assert library_packages(_calls_helper) == {"numpy"}
    assert library_packages(np.sum) == {"numpy"}
    assert library_packages(json.dumps) == {"json"}


def test_reached_packages_include_class_and_input_types(installed, tmp_path):
    @python.define
    def Reads(f: File, xs: list) -> int:
        return 1

    (tmp_path / "a.txt").write_text("a")
    task = Reads(f=File(tmp_path / "a.txt"), xs=[np.float64(1)])
    assert context.reached_packages(task) == {"fileformats", "numpy", "pydra"}


def test_unhashable_walk_leaves_packages_unknown(installed, monkeypatch):
    task = _scaled_task()(a=1, log="x")

    def fail(func):
        raise TypeError("unhashable")

    monkeypatch.setattr(context, "library_packages", fail)
    assert context.job_context(task, native.Native())["packages"] is None


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
