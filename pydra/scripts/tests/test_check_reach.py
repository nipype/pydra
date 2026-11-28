import importlib
import itertools
import json
from pathlib import Path
import sys
import textwrap
import typing as ty

from click.testing import CliRunner
import cloudpickle as cp
import pytest

import pydra.engine
from pydra.scripts.cli import cli
from pydra.utils.hash import is_user_code, reached_definitions
from pydra.utils.reach import check_reach, reach_report

HELPERS = """
import collections
import dataclasses

def hidden(x):
    return x + 1

def direct(x):
    return x * 2

def zzz(x):
    return x

def aaa(x):
    return x

@dataclasses.dataclass
class DC:
    a: int

NT = collections.namedtuple("NT", "a b")

class Obj:
    def method(self, x):
        return hidden(x)
"""

TASKS = """
import threading
import typing as ty
import helpers_{n} as h
from fileformats.generic import File
from pydra.compose import python

@python.define
def Dispatch(x: int) -> int:
    return getattr(h, "hidden")(x)

@python.define
def Direct(x: int) -> int:
    return h.direct(x)

@python.define
def Imported(x: int) -> int:
    import importlib
    return importlib.import_module("helpers_{n}").hidden(x)

@python.define
def Two(x: int) -> int:
    getattr(h, "zzz")(x)
    return getattr(h, "aaa")(x)

@python.define
def Generated(x: int) -> int:
    return h.DC(x).a + h.NT(x, 1).a

@python.define
def CallsInput(x: int, f: ty.Any) -> int:
    return f(x)

@python.define
def MethodOnInput(x: int, o: ty.Any) -> int:
    return o.method(x)

@python.define
def Threaded(x: int) -> int:
    t = threading.Thread(target=lambda: getattr(h, "hidden")(x))
    t.start()
    t.join()
    return x

@python.define
def Comprehension(x: int) -> int:
    return sum([h.direct(i) for i in range(x)])

@python.define
def Lambda(x: int) -> int:
    return (lambda v: h.direct(v))(x)

@python.define
def Nested(x: int) -> int:
    def inner(v):
        return h.direct(v)
    return inner(x)

@python.define
def ClassUse(x: int) -> int:
    return h.Obj().method(x)

def local(v):
    return v

@python.define
def SameModule(x: int) -> int:
    return local(x)

@python.define
def Raises(x: int) -> int:
    raise ValueError("boom")

@python.define
def Size(in_file: File) -> int:
    return in_file.fspath.stat().st_size
"""

UNOWNED = """
def hidden(x):
    return x
"""

_counter = itertools.count()


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A user project of a helpers module and the tasks of TASKS, importable"""
    n = str(next(_counter))
    (tmp_path / f"helpers_{n}.py").write_text(textwrap.dedent(HELPERS))
    (tmp_path / f"tasks_{n}.py").write_text(textwrap.dedent(TASKS).replace("{n}", n))
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(f"tasks_{n}")
    module.helpers = importlib.import_module(f"helpers_{n}")
    yield module
    for name in (f"helpers_{n}", f"tasks_{n}"):
        sys.modules.pop(name, None)


def names(lines):
    return [line.split(" (")[0].rpartition(".")[2] for line in lines]


def test_getattr_dispatch_reported(project):
    reached = check_reach(project.Dispatch, {"x": 1})
    assert len(reached) == 1
    assert "hidden" in reached[0]


def test_importlib_reported(project):
    reached = check_reach(project.Imported, {"x": 1})
    assert len(reached) == 1
    assert "hidden" in reached[0]


def test_direct_helper_not_reported(project):
    assert check_reach(project.Direct, {"x": 1}) == []


def test_reports_are_sorted(project):
    assert names(check_reach(project.Two, {"x": 1})) == ["aaa", "zzz"]


def test_generated_code_not_reported(project):
    """dataclass, attrs and namedtuple code generated with a user module's globals"""
    assert check_reach(project.Generated, {"x": 1}) == []


def test_comprehension_lambda_and_nested_function_not_reported(project):
    for task in (project.Comprehension, project.Lambda, project.Nested):
        assert check_reach(task, {"x": 2}) == [], task


def test_class_methods_are_reached(project):
    assert check_reach(project.ClassUse, {"x": 1}) == []


def test_callable_input_is_reached(project):
    """The override RFC 0004 section 4.2 recommends for a dispatch"""
    h = project.helpers
    assert check_reach(project.CallsInput, {"x": 1, "f": h.hidden}) == []


def test_user_class_instance_input_is_reached(project):
    h = project.helpers
    assert check_reach(project.MethodOnInput, {"x": 1, "o": h.Obj()}) == []


@pytest.fixture
def by_value(project):
    """The project's modules pickled by value, as a script's are"""
    modules = [project, project.helpers]
    for module in modules:
        cp.register_pickle_by_value(module)
    yield lambda obj: cp.loads(cp.dumps(obj))
    for module in modules:
        cp.unregister_pickle_by_value(module)


@pytest.mark.parametrize(
    "task", ["Direct", "ClassUse", "Nested", "SameModule", "Generated"]
)
def test_task_pickled_by_value_reports_no_false_miss(project, by_value, task):
    """A task whose function and helpers were unpickled by value (RFC 0004 T1, T2)
    runs only code the walk reached or code it does not read as user code"""
    assert reach_report(by_value(getattr(project, task)), {"x": 2}) == ([], [])


def test_inputs_pickled_by_value_report_no_false_miss(project, by_value):
    h = project.helpers
    assert reach_report(project.CallsInput, {"x": 1, "f": by_value(h.direct)}) == (
        [],
        [],
    )
    for obj in (by_value(h.Obj()), by_value(h.Obj)()):
        assert reach_report(project.MethodOnInput, {"x": 1, "o": obj}) == ([], [])


def test_helper_in_a_thread_reported(project):
    assert names(check_reach(project.Threaded, {"x": 1})) == ["hidden"]


def test_profiler_restored(project):
    def sentinel(frame, event, arg):
        pass

    sys.setprofile(sentinel)
    try:
        check_reach(project.Direct, {"x": 1})
        assert sys.getprofile() is sentinel
        with pytest.raises(ValueError):
            check_reach(project.Raises, {"x": 1})
        assert sys.getprofile() is sentinel
    finally:
        sys.setprofile(None)


def test_inputs_are_validated_by_the_task(project):
    with pytest.raises(TypeError):
        check_reach(project.Direct, {"y": 1})


def test_not_a_python_task():
    from pydra.compose import shell

    Echo = shell.define("echo")
    with pytest.raises(TypeError):
        check_reach(Echo, {})


def test_function_tier_reports_out_of_tier_helpers_by_name(project, monkeypatch):
    monkeypatch.setenv("PYDRA_CODE_HASH", "function")
    missed, by_name = reach_report(project.Direct, {"x": 1})
    assert missed == []
    assert names(by_name) == ["direct"]
    missed, by_name = reach_report(project.Dispatch, {"x": 1})
    assert names(missed) == ["hidden"]
    assert by_name == []


def test_module_tier_keeps_in_tier_helpers_reached(project, monkeypatch):
    monkeypatch.setenv("PYDRA_CODE_HASH", "module")
    assert reach_report(project.SameModule, {"x": 1}) == ([], [])
    missed, by_name = reach_report(project.Direct, {"x": 1})
    assert missed == []
    assert names(by_name) == ["direct"]
    missed, _ = reach_report(project.Dispatch, {"x": 1})
    assert names(missed) == ["hidden"]


def test_unowned_python_module_is_user_code(tmp_path, monkeypatch, project):
    """A pure-Python module in site-packages that no distribution owns is walked as
    user code, so a dispatch into it is a miss"""
    site = tmp_path / "site-packages"
    site.mkdir()
    name = f"unowned_{next(_counter)}"
    (site / f"{name}.py").write_text(UNOWNED)
    monkeypatch.syspath_prepend(str(site))
    module = importlib.import_module(name)
    try:
        assert (
            names(check_reach(project.CallsInput, {"x": 1, "f": module.hidden})) == []
        )
        mod_task = _dispatch_task(tmp_path, monkeypatch, name)
        assert names(check_reach(mod_task, {"x": 1})) == ["hidden"]
    finally:
        sys.modules.pop(name, None)


def _dispatch_task(tmp_path, monkeypatch, module_name):
    source = (
        "from pydra.compose import python\n"
        "import " + module_name + " as m\n\n"
        "@python.define\n"
        "def Go(x: int) -> int:\n"
        "    return getattr(m, 'hidden')(x)\n"
    )
    name = f"go_{next(_counter)}"
    (tmp_path / f"{name}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    return importlib.import_module(name).Go


def test_user_packages_setting_makes_a_module_library(project, monkeypatch):
    """``-name`` in PYDRA_USER_PACKAGES is library code, which is keyed by name"""
    helpers = project.helpers.__name__
    monkeypatch.setenv("PYDRA_USER_PACKAGES", f"-{helpers}")
    assert check_reach(project.Dispatch, {"x": 1}) == []


def test_is_user_code_rules(project):
    h = project.helpers
    assert is_user_code(h.direct.__code__, vars(h))
    assert not is_user_code(compile("1", "<string>", "eval"), vars(h))
    json = importlib.import_module("json")
    assert not is_user_code(json.dumps.__code__, vars(json))


def test_reached_definitions_include_inputs(project):
    h = project.helpers
    keyed, _ = reached_definitions(project.Direct(x=1).function, {})
    assert h.direct.__code__ in keyed
    assert h.hidden.__code__ not in keyed
    keyed, _ = reached_definitions(
        project.CallsInput(x=1, f=1).function, {"f": h.Obj()}
    )
    assert h.Obj.method.__code__ in keyed and h.hidden.__code__ in keyed


def test_cli_lists_and_exits_1(project):
    result = CliRunner().invoke(
        cli, ["check-reach", f"{project.__name__}:Dispatch", '{"x": 1}']
    )
    assert result.exit_code == 1
    assert "hidden" in result.output
    clean = CliRunner().invoke(
        cli, ["check-reach", f"{project.__name__}:Direct", '{"x": 1}']
    )
    assert clean.exit_code == 0


@pytest.mark.parametrize("folder", ["plain", "back\\slash"])
def test_cli_file_input_from_a_path(project, tmp_path, folder):
    """A path holding backslashes, as every Windows path does, given as a JSON string"""
    if folder != "plain" and sys.platform == "win32":
        folder = "back"  # a separator here: the tmp_path itself holds backslashes
    data = tmp_path / folder / "a.txt"
    data.parent.mkdir()
    data.write_text("abc")
    result = CliRunner().invoke(
        cli,
        ["check-reach", f"{project.__name__}:Size", json.dumps({"in_file": str(data)})],
    )
    assert result.exit_code == 0, result.output


def test_cli_misspelt_input_is_a_click_error(project):
    result = CliRunner().invoke(
        cli, ["check-reach", f"{project.__name__}:Direct", '{"y": 1}']
    )
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_never_runs_in_workflow():
    engine = Path(pydra.engine.__file__).parent
    for path in engine.rglob("*.py"):
        if "tests" in path.parts:
            continue
        assert "utils.reach" not in path.read_text(), path
        assert "import reach" not in path.read_text(), path
