import hashlib
import os
import stat
import sys
import time
from pathlib import Path
import pytest
from pydra.compose import python, shell
from pydra.engine import context
from pydra.engine.context import job_context
from pydra.engine.store import ConsistentReuse
from pydra.engine.submitter import Submitter
from pydra.environments import native
from pydra.engine.tests.test_context import installed  # noqa: F401

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")


def _script(path: Path, body: str = "echo hi") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.fixture
def bindir(tmp_path, monkeypatch):
    """A directory first on PATH, so lookups find what the test puts there"""
    bin = tmp_path / "bin"
    bin.mkdir()
    monkeypatch.setenv("PATH", f"{bin}{os.pathsep}{os.environ['PATH']}")
    context.clear_caches()
    yield bin
    context.clear_caches()


def _tools(task):
    return job_context(task, native.Native())["tools"]


def test_tools_path_sha_owner(bindir, tmp_path):
    real = _script(tmp_path / "Cellar" / "mytool" / "1.2" / "bin" / "mytool")
    (bindir / "mytool").symlink_to(real)
    tools = _tools(shell.define("mytool")())
    assert tools == [
        {
            "executable": "mytool",
            "found": True,
            "path": os.path.realpath(real),
            "sha256": hashlib.sha256(real.read_bytes()).hexdigest(),
            "owner": {"manager": "homebrew", "name": "mytool", "version": "1.2"},
        }
    ]
    listed = shell.define("mytool")()
    listed.executable = ["mytool", "-x"]
    assert _tools(listed) == tools


def test_tools_not_found_stable(bindir):
    task = shell.define("no-such-tool-here")()
    expected = [{"executable": "no-such-tool-here", "found": False}]
    assert _tools(task) == expected
    assert _tools(task) == expected


def test_tools_of_python_task_are_empty(bindir):
    @python.define
    def f(a: int) -> int:
        return a

    assert _tools(f(a=1)) == []


def test_owner_lookup_time_limited(bindir, tmp_path, monkeypatch):
    _script(bindir / "dpkg", "exec sleep 30")
    _script(bindir / "rpm", "exec sleep 30")
    tool = _script(tmp_path / "opt" / "slowowner")
    (bindir / "slowowner").symlink_to(tool)
    monkeypatch.setattr(context, "OWNER_TIMEOUT", 0.2)
    start = time.monotonic()
    tools = _tools(shell.define("slowowner")())
    assert time.monotonic() - start < 10
    assert tools[0]["found"] is True
    assert tools[0]["owner"] is None


def test_consistent_reuse_refuses_tools_change(installed, bindir):  # noqa: F811
    tool = _script(bindir / "mytool", "echo one")
    task = shell.define("mytool")()
    stored = job_context(task, native.Native())
    policy = ConsistentReuse()
    assert policy.accept(stored, job_context(task, native.Native()), task)
    tool.write_text("#!/bin/sh\necho two\n")
    context.clear_caches()
    changed = job_context(task, native.Native())
    assert changed["tools"] != stored["tools"]
    assert not policy.accept(stored, changed, task)


@pytest.mark.parametrize("worker", ["debug", "cf"], ids=["debug", "cf"])
def test_tools_built_in_submitter_only(bindir, tmp_path, worker):
    """The worker uses the context carried with the job: a tool that removes itself
    while running does not change the tools the result records"""
    tool = _script(bindir / "selfdestruct", f'rm -f "{bindir}/selfdestruct"')
    # by absolute path: a pooled worker started earlier does not see this test's PATH
    task = shell.define(str(tool))()
    expected = _tools(task)
    assert expected[0]["found"] is True
    with Submitter(worker=worker, cache_root=tmp_path / "cache") as sub:
        result = sub(task)
    assert not (bindir / "selfdestruct").exists()
    assert result.context["tools"] == expected
