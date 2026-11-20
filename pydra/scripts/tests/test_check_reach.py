import importlib
from pathlib import Path
import sys
import textwrap

from click.testing import CliRunner
import pytest

import pydra.engine
from pydra.scripts.cli import cli
from pydra.utils.reach import check_reach

HELPERS = """
def hidden(x):
    return x + 1

def direct(x):
    return x * 2
"""

GETATTR_TASK = """
import helpers_{n} as helpers
from pydra.compose import python

@python.define
def Dispatch(x: int) -> int:
    return getattr(helpers, "hidden")(x)

@python.define
def Direct(x: int) -> int:
    return helpers.direct(x)

@python.define
def Imported(x: int) -> int:
    import importlib
    return importlib.import_module("helpers_{n}").hidden(x)
"""


@pytest.fixture
def project(tmp_path, monkeypatch, request):
    """A user project of a helpers module and the tasks of GETATTR_TASK, importable"""
    n = request.node.name
    (tmp_path / f"helpers_{n}.py").write_text(textwrap.dedent(HELPERS))
    (tmp_path / f"tasks_{n}.py").write_text(
        textwrap.dedent(GETATTR_TASK).replace("{n}", n)
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module(f"tasks_{n}")
    for name in (f"helpers_{n}", f"tasks_{n}"):
        sys.modules.pop(name, None)


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


def test_never_runs_in_workflow():
    engine = Path(pydra.engine.__file__).parent
    for path in engine.rglob("*.py"):
        if "tests" in path.parts:
            continue
        assert "utils.reach" not in path.read_text(), path
        assert "import reach" not in path.read_text(), path
