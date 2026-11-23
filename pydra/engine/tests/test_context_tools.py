import hashlib
import os
import stat
import subprocess
import sys
from pathlib import Path
import pytest
from pydra.compose import python, shell
from pydra.engine import context
from pydra.engine.context import job_context, job_tools
from pydra.engine.job import Job
from pydra.engine.store import ConsistentReuse
from pydra.engine.submitter import Submitter
from pydra.environments import native
from pydra.environments.docker import Docker
from pydra.engine.tests.test_context import installed  # noqa: F401

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell scripts")


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


@posix
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


@posix
def test_owner_lookup_time_limited(bindir, tmp_path, monkeypatch):
    _script(bindir / "dpkg", "exec sleep 30")
    _script(bindir / "rpm", "exec sleep 30")
    tool = _script(tmp_path / "opt" / "slowowner")
    (bindir / "slowowner").symlink_to(tool)
    monkeypatch.setattr(context, "OWNER_TIMEOUT", 0.2)
    real_run = context.subprocess.run
    queries = []

    def recording(command, *args, **kwargs):
        try:
            return real_run(command, *args, **kwargs)
        except subprocess.TimeoutExpired:
            queries.append((Path(command[0]).name, kwargs.get("timeout"), "timed out"))
            raise

    monkeypatch.setattr(context.subprocess, "run", recording)
    tools = _tools(shell.define("slowowner")())
    assert tools[0]["found"] is True
    assert tools[0]["owner"] is None
    # each package-manager query was cut off at OWNER_TIMEOUT, not waited for
    assert queries == [("dpkg", 0.2, "timed out"), ("rpm", 0.2, "timed out")]


@posix
def test_consistent_reuse_refuses_tools_change(installed, bindir):  # noqa: F811
    tool = _script(bindir / "mytool", "echo one")
    task = shell.define("mytool")()
    stored = job_context(task, native.Native())
    policy = ConsistentReuse()
    assert policy.accept(stored, job_context(task, native.Native()), task)
    checksum = Job(
        task, name="j", submitter=Submitter(cache_root=bindir.parent / "cache")
    ).checksum
    tool.write_text("#!/bin/sh\necho two\n")
    changed = job_context(task, native.Native())
    assert changed["tools"] != stored["tools"]
    assert not policy.accept(stored, changed, task)
    # tools are recorded, never keyed: the job's checksum does not follow them
    again = Job(task, name="j", submitter=Submitter(cache_root=bindir.parent / "cache"))
    assert again.checksum == checksum


@posix
def test_tools_built_in_submitter_only(bindir, tmp_path, any_worker):
    """The worker uses the context carried with the job: a tool that removes itself
    while running does not change the tools the result records"""
    tool = _script(bindir / "selfdestruct", f'rm -f "{bindir}/selfdestruct"')
    # by absolute path: a pooled worker started earlier does not see this test's PATH
    task = shell.define(str(tool))()
    expected = _tools(task)
    assert expected[0]["found"] is True
    with Submitter(worker=any_worker, cache_root=tmp_path / "cache") as sub:
        result = sub(task)
    assert not (bindir / "selfdestruct").exists()
    assert result.context["tools"] == expected


@posix
def test_owner_lookup_memoised(bindir, tmp_path, monkeypatch):
    _script(bindir / "dpkg", "exit 1")
    _script(bindir / "rpm", "exit 1")
    tool = _script(tmp_path / "opt" / "memoowner")
    (bindir / "memoowner").symlink_to(tool)
    queries = []
    query = context._query
    monkeypatch.setattr(context, "_query", lambda c: queries.append(c) or query(c))
    task = shell.define("memoowner")()
    for _ in range(5):
        assert _tools(task)[0]["owner"] is None
    assert [q[0] for q in queries] == ["dpkg", "rpm"]


@posix
def test_conda_owner(bindir, tmp_path):
    env = tmp_path / "env"
    tool = _script(env / "bin" / "ctool")
    (env / "conda-meta").mkdir()
    (env / "conda-meta" / "ctool-1.0-0.json").write_text(
        '{"name": "ctool", "version": "1.0", "files": ["bin/ctool"]}'
    )
    (bindir / "ctool").symlink_to(tool)
    assert _tools(shell.define("ctool")())[0]["owner"] == {
        "manager": "conda",
        "name": "ctool",
        "version": "1.0",
    }


@posix
def test_conda_owner_scan_time_limited(bindir, tmp_path, monkeypatch):
    env = tmp_path / "env"
    tool = _script(env / "bin" / "ctool")
    (env / "conda-meta").mkdir()
    for i in range(20):
        (env / "conda-meta" / f"p{i}-1.0-0.json").write_text(
            '{"name": "p", "version": "1", "files": ["bin/ctool"]}'
        )
    (bindir / "ctool").symlink_to(tool)
    monkeypatch.setattr(context, "OWNER_TIMEOUT", 0.0)
    assert _tools(shell.define("ctool")())[0]["owner"] is None


@posix
def test_tool_rewritten_in_place_gets_new_sha(bindir):
    tool = _script(bindir / "mytool", "echo one")
    task = shell.define("mytool")()
    before = _tools(task)[0]["sha256"]
    stat_before = tool.stat()
    tool.write_text("#!/bin/sh\necho two\n")  # the same size
    os.utime(tool, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    after = _tools(task)[0]["sha256"]
    assert after != before
    assert after == hashlib.sha256(tool.read_bytes()).hexdigest()


@posix
def test_tool_replaced_by_rename_gets_new_sha(bindir):
    tool = _script(bindir / "mytool", "echo one")
    task = shell.define("mytool")()
    before = _tools(task)[0]["sha256"]
    stat_before = tool.stat()
    other = _script(bindir / "other", "echo two")
    os.utime(other, ns=(stat_before.st_atime_ns, stat_before.st_mtime_ns))
    os.replace(other, tool)
    after = _tools(task)[0]["sha256"]
    assert after != before


@posix
def test_version_command_program_is_a_tool(bindir):
    _script(bindir / "vtool")
    task = shell.define(
        "mytool", version_command=["vtool", "--version"], version_pattern=r"(\d+)"
    )()
    tools = _tools(task)
    assert [t["executable"] for t in tools] == ["mytool", "vtool"]
    assert [t["found"] for t in tools] == [False, True]


def test_container_environment_records_no_tools(bindir):
    _script(bindir / "mytool")
    task = shell.define("mytool")()
    assert job_tools(task, Docker(image="ubuntu:22.04")) == []


def _tool(**changes):
    record = {
        "executable": "echo",
        "found": True,
        "path": "/usr/bin/echo",
        "sha256": "a" * 64,
        "owner": {"manager": "dpkg", "name": "coreutils"},
    }
    return {**record, **changes}


@pytest.mark.parametrize(
    "other,accepted",
    [
        (_tool(owner=None), True),  # an owner lookup cut short on a loaded host
        (_tool(path="/bin/echo"), True),
        (_tool(sha256="b" * 64), False),
        (_tool(found=False), False),
    ],
)
def test_consistent_reuse_compares_what_a_tool_is_not_who_owns_it(other, accepted):
    """``ConsistentReuse`` compares a tool's executable, whether it was found and its
    digest; the path and owner are recorded only, so a timed-out owner lookup never
    refuses a result"""
    from pydra.engine.store import ConsistentReuse

    stored = job_context(shell.define("echo <x:str>")(x="1"), _native())
    current = dict(stored)
    stored = {**stored, "tools": [_tool()]}
    current["tools"] = [other]
    assert ConsistentReuse().accept(stored, current, None) is accepted


def _native():
    from pydra.environments.native import Native

    return Native()
