"""Library references keyed by distribution identity (RFC 0004 section 4.1, I3). The
installed distributions are fake metadata, set at the ``importlib.metadata`` boundary,
so no test needs the network or installs anything."""

import importlib
import importlib.metadata
import json
import sys
import types
from pathlib import Path

import numpy
import pytest
from packaging.version import Version

from pydra.compose import python
from pydra.engine import context
from pydra.utils import hash as hash_module
from pydra.utils.hash import Cache, bytes_repr, hash_function, owner_of
from pydra.utils.identity import task_record

FAKELIB = "def scale(a):\n    return 2 * a\n\n\nclass Thing:\n    pass\n"
USER = (
    "from fakelib import scale\nimport fakelib\nimport os\nfrom json import dumps\nfrom os import environ\n"
    "from numpy import add\n\n\n"
    "def f(x):\n    return scale(x), os, dumps(x), fakelib, environ\n\n\n"
    "def h(x):\n    return add(x, 1)\n\n\n"
    "def g(x):\n    return x\n"
)


class FakeDist:
    def __init__(self, name, version, direct=None, base="/", files=(), requires=()):
        self.metadata = {"Name": name}
        self.version = version
        self.direct = direct
        self.base = Path(base)
        self.files = list(files)
        self.requires = list(requires)

    def locate_file(self, path):
        return self.base / path

    def read_text(self, name):
        if name == "direct_url.json" and self.direct is not None:
            return json.dumps(self.direct)
        if name == "RECORD":
            return "\n".join(self.files)
        return None


@pytest.fixture
def libs(monkeypatch, tmp_path):
    """``fakelib`` is installed in a ``site-packages``; ``libs(**kwargs)`` replaces the
    installed distributions, standing for a new process"""
    site = tmp_path / "site-packages"
    (site / "fakelib").mkdir(parents=True)
    (site / "fakelib" / "__init__.py").write_text(FAKELIB)
    (tmp_path / "uses_fakelib.py").write_text(USER)
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("fakelib", "uses_fakelib"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    state = types.SimpleNamespace(site=site)

    def install(
        version="1.0",
        direct=None,
        files=("fakelib/__init__.py",),
        requires=("base>=1",),
        others=None,
        top_level=None,
    ):
        dists = [
            FakeDist("fakelib", version, direct, site, files, requires),
            FakeDist("base", "1.0"),
        ]
        dists += [FakeDist(n, *v) for n, v in (others or {}).items()]
        monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(dists))
        monkeypatch.setattr(
            importlib.metadata,
            "packages_distributions",
            lambda: (
                {"fakelib": ["fakelib"], "base": ["base"]}
                if top_level is None
                else top_level
            ),
        )
        context.clear_caches()

    install()
    state.install = install
    state.user = importlib.import_module("uses_fakelib")
    yield state
    monkeypatch.undo()
    context.clear_caches()
    for name in ("fakelib", "uses_fakelib"):
        sys.modules.pop(name, None)


def _encoded(func) -> bytes:
    return b"".join(bytes_repr(func, Cache()))


def _hash(func) -> str:
    return hash_function(func, cache=Cache())


def test_library_ref_carries_version(libs):
    encoded = _encoded(libs.user.f)
    assert b"global:scale=lib:fakelib==1.0:fakelib.scale" in encoded
    assert b"global:fakelib=module:fakelib==1.0:fakelib" in encoded
    before = _hash(libs.user.f)
    libs.install(version="2.0")
    assert b"lib:fakelib==2.0:fakelib.scale" in _encoded(libs.user.f)
    assert _hash(libs.user.f) != before


def test_local_segment_not_keyed(libs):
    libs.install(version="1.0+cu121")
    cuda = _hash(libs.user.f)
    assert b"lib:fakelib==1.0:fakelib.scale" in _encoded(libs.user.f)
    # the local label is recorded, not keyed
    assert context._distributions()["fakelib"].identity == "1.0+cu121"
    assert context.package_versions({"fakelib"})["fakelib"] == "1.0+cu121"
    libs.install(version="1.0+cpu")
    assert _hash(libs.user.f) == cuda
    libs.install(version="1.1+cpu")
    assert _hash(libs.user.f) != cuda


def test_vcs_archive_localdir_identities(libs):
    def ref(**kwargs):
        libs.install(**kwargs)
        return _encoded(libs.user.f)

    vcs = {"url": "git+x", "vcs_info": {"commit_id": "abc"}}
    assert b"lib:fakelib==1.0@abc:fakelib.scale" in ref(direct=vcs)
    vcs_local = ref(version="1.0+g123", direct=vcs)
    # a VCS install keeps the whole version: it is the commit that identifies it
    assert b"lib:fakelib==1.0+g123@abc:fakelib.scale" in vcs_local
    archive = {"url": "https://x/f.whl", "archive_info": {"hash": "sha256=ff"}}
    assert b"lib:fakelib==1.0@sha256=ff:fakelib.scale" in ref(direct=archive)
    directory = {"url": "file:///src", "dir_info": {}}
    first = ref(direct=directory)
    assert b"lib:fakelib==1.0@record:" in first
    second = ref(direct=directory, files=("fakelib/__init__.py", "fakelib/extra.py"))
    assert second != first and b"lib:fakelib==1.0@record:" in second


def test_stdlib_unversioned(libs):
    encoded = _encoded(libs.user.f)
    assert b"global:os=module:os" in encoded
    assert b"global:environ=lib:os.environ" in encoded
    assert b"global:dumps=lib:json.dumps" in encoded
    assert b"lib:?:" not in encoded
    assert owner_of("os") is None


def test_ufunc_owner_from_file(libs):
    """numpy.add is owned by the distribution whose RECORD lists the file of numpy"""
    path = numpy.__file__
    libs.install(
        others={"numpy": ("9.9", None, "/", [path.lstrip("/")])},
        top_level={"numpy": ["numpy"]},
    )
    assert owner_of("numpy") == ("numpy", "9.9")
    assert b"global:add=lib:numpy==9.9:numpy.add" in _encoded(libs.user.h)


def test_unowned_extension_lib_question(libs, monkeypatch):
    """An extension module no distribution owns is referenced as ``lib:?:`` and the
    context cannot name its version; a pure-Python one is walked as user code"""
    ext = types.ModuleType("extmod")
    ext.__file__ = str(libs.site / "extmod.cpython-313-x86_64-linux-gnu.so")
    Kind = type("Kind", (), {"__module__": "extmod"})
    ext.THING = Kind()
    monkeypatch.setitem(sys.modules, "extmod", ext)
    namespace = {"THING": ext.THING, "__name__": "uses_ext"}
    exec("def f(x):\n    return THING\n", namespace)
    assert b"global:THING=lib:?:extmod.THING" in _encoded(namespace["f"])
    assert context.package_versions({"extmod"}) == {"extmod": None}
    # fakelib not owned: walked as user code, not referenced by name
    libs.install(top_level={}, files=())
    encoded = _encoded(libs.user.f)
    assert b"lib:fakelib" not in encoded and b"fakelib.scale" not in encoded
    assert b"return(2*a)" in encoded.replace(b" ", b"") or b"BinOp" in encoded


def test_packages_distributions_fallback(libs):
    """A module whose file no RECORD lists is owned by the one distribution
    ``packages_distributions`` names for its top-level package"""
    libs.install(files=(), top_level={"fakelib": ["fakelib"]})
    assert owner_of("fakelib") == ("fakelib", "1.0")
    assert b"lib:fakelib==1.0:fakelib.scale" in _encoded(libs.user.f)
    assert context.package_versions({"fakelib"}) == {"base": "1.0", "fakelib": "1.0"}
    libs.install(
        files=(),
        others={"fakelib-fork": ("1.0",)},
        top_level={"fakelib": ["fakelib", "fakelib-fork"]},
    )
    assert owner_of("fakelib") is None


def test_input_value_type_owner_keyed(libs):
    """A function that only calls methods of its input names no library, so the owner of
    the input's type is keyed"""
    import fakelib

    task = python.define(libs.user.g)(x=fakelib.Thing())
    record = task_record(task)
    assert record["task"]["values"] == ["dist:fakelib==1.0"]
    before = task._checksum
    libs.install(version="2.0")
    after = python.define(libs.user.g)(x=fakelib.Thing())
    assert task_record(after)["task"]["values"] == ["dist:fakelib==2.0"]
    assert after._checksum != before
    plain = python.define(libs.user.g)(x=1)
    assert task_record(plain)["task"]["values"] == []


def test_transitive_not_keyed(libs):
    """fakelib requires base; base is in the context's closure, not in the key"""
    before = _hash(libs.user.f)
    libs.install()
    context._distributions()["base"].version  # read
    assert b"base" not in _encoded(libs.user.f)
    assert set(context.package_versions({"fakelib"})) == {"base", "fakelib"}
    # a new version of base, which the record does not name, does not re-key
    libs.install(others={"base": ("2.0",)})
    assert _hash(libs.user.f) == before


def test_real_installed_numpy_owner():
    """The venv's numpy, owned through the files of its ``RECORD``"""
    dist = importlib.metadata.distribution("numpy")
    files = {str(dist.locate_file(f)) for f in dist.files or ()}
    assert numpy.__file__ in files or Path(numpy.__file__).resolve() in {
        Path(f).resolve() for f in files
    }
    assert owner_of("numpy") == ("numpy", Version(numpy.__version__).public)
    assert hash_module.library_owner("numpy.linalg").startswith("numpy==")
