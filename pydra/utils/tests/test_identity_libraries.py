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
    "from numpy import add, zeros\nfrom pathlib import Path\nimport numpy\n\n"
    "SUM = numpy.ndarray.sum\n\n\n"
    "def f(x):\n    return scale(x), os, dumps(x), fakelib, environ\n\n\n"
    "def h(x):\n    return add(x, 1), zeros, SUM\n\n\n"
    "def k(x):\n    return fakelib.scale(x)\n\n\n"
    "class Local:\n    pass\n\n\n"
    "def g(x):\n    return x\n"
)


class FakeDist:
    def __init__(self, name, version, direct=None, base="/", files=(), requires=()):
        self.metadata = {"Name": name}
        self.version = version
        self.direct = direct
        self.base = Path(base)
        # "path#hash" gives a file the hash its RECORD row carries, "#-" none
        self.hashes = dict(f.partition("#")[::2] for f in files)
        self.files = [f.partition("#")[0] for f in files]
        self.requires = list(requires)

    def _row_hash(self, path):
        hash_ = self.hashes[path]
        return "" if hash_ == "-" else f"sha256={hash_ or path}"

    def locate_file(self, path):
        return self.base / path

    def read_text(self, name):
        if name == "direct_url.json" and self.direct is not None:
            return (
                self.direct if isinstance(self.direct, str) else json.dumps(self.direct)
            )
        if name == "RECORD":
            return "\n".join(f"{f},{self._row_hash(f)},1" for f in self.files)
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
        base="1.0",
    ):
        dists = [
            FakeDist("fakelib", version, direct, site, files, requires),
            FakeDist("base", base),
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
    assert b"lib:fakelib==dir:" in first
    second = ref(direct=directory, files=("fakelib/__init__.py", "fakelib/extra.py"))
    assert second != first and b"lib:fakelib==dir:" in second


def test_stdlib_unversioned(libs):
    encoded = _encoded(libs.user.f)
    assert b"global:os=module:os" in encoded
    assert b"global:environ=lib:os.environ" in encoded
    assert b"global:dumps=lib:json.dumps" in encoded
    assert b"lib:?:" not in encoded
    assert owner_of("os") is None


def test_ufunc_owner_from_file(libs):
    """numpy.add is owned by the distribution whose RECORD lists the file of numpy; a
    builtin function and a method descriptor of numpy are owned the same way"""
    path = numpy.__file__
    libs.install(
        others={"numpy": ("9.9", None, "/", [path.lstrip("/")])},
        top_level={"numpy": ["numpy"]},
    )
    encoded = _encoded(libs.user.h)
    assert b"global:add=lib:numpy==9.9:numpy.add" in encoded
    assert b"global:zeros=lib:numpy==9.9:numpy.zeros" in encoded
    assert b"global:SUM=lib:numpy==9.9:numpy.ndarray.sum" in encoded
    assert owner_of("numpy") == ("numpy", "9.9")


def test_unowned_extension_is_referenced_not_walked(libs, monkeypatch):
    """A function or type of an extension module no distribution owns is referenced as
    ``lib:?:`` and never walked, and the context cannot name its version"""
    ext = types.ModuleType("extmod")
    ext.__file__ = str(libs.site / "extmod.so")

    namespace = {"__name__": "extmod"}
    exec(compile("def func(x):\n    return x\n", ext.__file__, "exec"), namespace)
    func = namespace["func"]
    Kind = type("Kind", (), {"__module__": "extmod"})
    ext.func, ext.Kind = func, Kind
    monkeypatch.setitem(sys.modules, "extmod", ext)
    namespace = {"func": func, "Kind": Kind, "__name__": "uses_ext"}
    exec("def f(x):\n    return func, Kind\n", namespace)
    encoded = _encoded(namespace["f"])
    assert b"global:Kind=lib:?:extmod.Kind" in encoded
    assert b"global:func=lib:?:extmod.func" in encoded
    assert b"#1" not in encoded
    assert context.package_versions({"extmod"}) == {"extmod": None}
    assert "extmod" in hash_module.library_modules(namespace["f"])


def test_unowned_python_is_walked_and_keyed_by_code(libs):
    """A pure-Python module no distribution owns is walked as user code, whether a
    name or the module is read; the context does not call it unknown"""
    libs.install(top_level={}, files=())
    for func in (libs.user.f, libs.user.k):
        encoded = _encoded(func)
        assert b"lib:fakelib" not in encoded and b"module:?:" not in encoded
    assert b"global:fakelib.scale=#" in _encoded(libs.user.k)
    before = _hash(libs.user.k)
    source = libs.site / "fakelib" / "__init__.py"
    source.write_text(FAKELIB.replace("2 * a", "3 * a"))
    importlib.reload(sys.modules["fakelib"])
    context.clear_caches()
    assert _hash(libs.user.k) != before
    assert "fakelib" not in hash_module.library_modules(libs.user.k)


def test_unidentifiable_distribution_is_keyed_as_unknown(libs):
    libs.install(direct="{not json")
    assert b"lib:fakelib==?:fakelib.scale" in _encoded(libs.user.f)
    assert context.package_versions({"fakelib"})["fakelib"] is None


def test_engine_modules_carry_no_identity(libs, monkeypatch):
    """The engine is referenced by qualified name, as the standard library is, never in
    ``task.values``; ``pydra.tasks`` is keyed by the distribution that owns it"""
    from pydra.compose import workflow
    from pydra.utils.typing import StateArray

    assert hash_module.library_owner("pydra.compose.workflow") == ""
    assert hash_module.library_owner("pydra.engine.state") == ""
    assert hash_module.library_owner("pydra.utils.hash") == ""
    assert hash_module.library_owner("pydra") == ""
    tasks = types.ModuleType("pydra.tasks.faketasks")
    tasks.__file__ = str(libs.site / "pydra" / "tasks" / "faketasks.py")
    monkeypatch.setitem(sys.modules, "pydra.tasks.faketasks", tasks)
    libs.install(
        others={
            "pydra": ("1.0",),
            "pydra-tasks-fake": (
                "3.0",
                None,
                libs.site,
                ["pydra/tasks/faketasks.py"],
            ),
        },
        top_level={"pydra": ["pydra", "pydra-tasks-fake"]},
    )
    assert (
        hash_module.library_owner("pydra.tasks.faketasks") == "pydra-tasks-fake==3.0:"
    )
    task = python.define(libs.user.g)(x=StateArray([1, 2]))
    assert task_record(task)["task"]["values"] == []
    assert "pydra" not in {
        r.partition("==")[0][4:] for r in task_record(task)["task"]["refs"]
    }
    assert workflow.define is not None


def test_namespace_owner_needs_one_distribution(libs):
    """A module two distributions claim is owned by the one whose RECORD lists its
    file, and by none if no RECORD does"""
    libs.install(
        files=("fakelib/__init__.py",),
        others={"fakelib-fork": ("2.0",)},
        top_level={"fakelib": ["fakelib", "fakelib-fork"]},
    )
    assert owner_of("fakelib") == ("fakelib", "1.0")
    libs.install(
        files=(),
        others={"fakelib-fork": ("2.0",)},
        top_level={"fakelib": ["fakelib", "fakelib-fork"]},
    )
    assert owner_of("fakelib") is None


def test_editable_root_owns_a_module(libs, tmp_path):
    """A module of an editable install is owned through its source directory"""
    source = tmp_path / "src"
    (source / "fakelib").mkdir(parents=True)
    (source / "fakelib" / "__init__.py").write_text(FAKELIB)
    direct = {"url": source.as_uri(), "dir_info": {"editable": True}}
    monkey = sys.modules["fakelib"]
    monkey.__file__ = str(source / "fakelib" / "__init__.py")
    libs.install(direct=direct, files=("fakelib.pth",), top_level={})
    owner = owner_of("fakelib")
    assert owner is not None and owner[0] == "fakelib"
    assert owner[1].startswith("dir:")


def test_refs_in_record(libs):
    """``task.refs`` lists the references to library code with their identities, the
    standard library left out, and again from the memoised walk"""
    task = python.define(libs.user.f)(x=1)
    refs = task_record(task)["task"]["refs"]
    assert "lib:fakelib==1.0:fakelib.scale" in refs
    assert "module:fakelib==1.0:fakelib" in refs
    assert not any(r.startswith(("lib:os", "lib:json", "module:os")) for r in refs)
    assert refs == sorted(refs)
    assert task_record(python.define(libs.user.f)(x=1))["task"]["refs"] == refs
    libs.install(version="2.0")
    after = task_record(python.define(libs.user.f)(x=1))["task"]["refs"]
    assert "lib:fakelib==2.0:fakelib.scale" in after
    assert "lib:fakelib==1.0:fakelib.scale" not in after


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
    # the standard library, user code and no owner are left out
    for value in (Path("/x"), libs.user.Local()):
        record = task_record(python.define(libs.user.g)(x=value))
        assert record["task"]["values"] == []
    # the executor field is not a value
    assert record["task"]["values"] == []


def test_transitive_not_keyed(libs):
    """fakelib requires base; base is in the context's closure, not in the key"""
    before = _hash(libs.user.f)
    assert b"base" not in _encoded(libs.user.f)
    assert context.package_versions({"fakelib"})["base"] == "1.0"
    libs.install(base="2.0")
    assert context.package_versions({"fakelib"})["base"] == "2.0"
    assert _hash(libs.user.f) == before


def _local_ref(libs, *files):
    libs.install(direct={"url": "file:///src", "dir_info": {}}, files=files)
    return _encoded(libs.user.f)


def test_record_digest_ignores_install_metadata(libs):
    """Two installs of the same files from different places are one key: the
    ``direct_url.json``, ``INSTALLER`` and console-script rows, which say how and where
    it was installed, are not in the digest"""
    init = "fakelib/__init__.py"
    first = _local_ref(libs, init, "fakelib-1.0.dist-info/direct_url.json#a")
    assert _local_ref(libs, init, "fakelib-1.0.dist-info/direct_url.json#b") == first
    assert _local_ref(libs, init, "fakelib-1.0.dist-info/INSTALLER#c") == first
    assert _local_ref(libs, init, "../../bin/tool#a") == _local_ref(
        libs, init, "../../bin/tool#b"
    )
    assert _local_ref(libs, init, "../../bin/tool#a") == first


def test_record_digest_covers_every_installed_file(libs):
    """An edited installed file, whatever its type, and an added one change the key: a
    task that reads ``fakelib/factor.txt`` must not reuse a result computed from
    another one"""
    init = "fakelib/__init__.py"
    data = _local_ref(libs, init, "fakelib/factor.txt#2")
    assert _local_ref(libs, init, "fakelib/factor.txt#2") == data
    assert _local_ref(libs, init, "fakelib/factor.txt#3") != data
    assert _local_ref(libs, init, "fakelib/factor.txt#2", "fakelib/more.dat#1") != data
    assert _local_ref(libs, "fakelib/__init__.py#x") != _local_ref(
        libs, "fakelib/__init__.py#y"
    )
    # a row without a hash (byte-code) is not part of it
    assert _local_ref(libs, init, "fakelib/factor.txt#2", "fakelib/x.pyc#-") == data


def test_editable_library_digest_covers_package_data(libs, tmp_path, monkeypatch):
    """An editable install named library code is keyed by every file of its package,
    data included, and not by notes beside it"""
    source = tmp_path / "src"
    (source / "fakelib").mkdir(parents=True)
    (source / "fakelib" / "__init__.py").write_text(FAKELIB)
    factor = source / "fakelib" / "factor.txt"
    factor.write_text("2")
    sys.modules["fakelib"].__file__ = str(source / "fakelib" / "__init__.py")
    monkeypatch.setenv("PYDRA_USER_PACKAGES", "-fakelib")
    direct = {"url": source.as_uri(), "dir_info": {"editable": True}}

    def ref():
        libs.install(direct=direct, files=("fakelib.pth",), top_level={})
        return _encoded(libs.user.f)

    first = ref()
    assert b"lib:fakelib==dir:" in first
    (source / "NOTES.txt").write_text("a note")
    assert ref() == first
    factor.write_text("3")
    assert ref() != first


def test_real_installed_numpy_owner():
    """The venv's numpy, owned through the files of its ``RECORD``"""
    dist = importlib.metadata.distribution("numpy")
    files = {Path(str(dist.locate_file(f))).resolve() for f in dist.files or ()}
    assert Path(numpy.__file__).resolve() in files
    assert owner_of("numpy") == ("numpy", Version(dist.version).public)
    assert hash_module.library_owner("numpy.linalg").startswith("numpy==")
