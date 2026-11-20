"""The shell task's identity: the command template, the tool version and the job record
(RFC 0004 sections 3.1, 5.1 and 5.2, I5)."""

import shutil
import sys
import warnings
from pathlib import Path

import pytest

from pydra.compose import python, shell
from pydra.compose.shell import VersionCommandError
from pydra.compose.shell import version as shell_version
from pydra.engine.job import Job
from pydra.engine.portable.identity import PydraIdentityV2
from pydra.engine.submitter import Submitter
from pydra.environments import docker
from pydra.utils.identity import job_record, record_checksum, task_record


def _job(task, environment=None):
    return Job(name="t", task=task, submitter=Submitter(), environment=environment)


def _ls(flag: str, help: str = "list", position: int | None = None):
    return shell.define(
        "ls",
        inputs={
            "flag": shell.arg(
                type=bool, argstr=flag, help=help, default=False, position=position
            )
        },
    )


def test_argstr_change_changes_checksum():
    one = shell.define("tool", inputs={"x": shell.arg(type=int, argstr="--a")})
    other = shell.define("tool", inputs={"x": shell.arg(type=int, argstr="--b")})
    assert one(x=1)._checksum != other(x=1)._checksum
    assert _job(one(x=1)).checksum != _job(other(x=1)).checksum
    assert one(x=1)._checksum == one(x=1)._checksum
    sep_a = shell.define("tool", inputs={"x": shell.arg(type=list[int], sep=",")})
    sep_b = shell.define("tool", inputs={"x": shell.arg(type=list[int], sep=":")})
    assert sep_a(x=[1])._checksum != sep_b(x=[1])._checksum
    pos_a = shell.define("tool", inputs={"x": shell.arg(type=int, position=1)})
    pos_b = shell.define("tool", inputs={"x": shell.arg(type=int, position=2)})
    assert pos_a(x=1)._checksum != pos_b(x=1)._checksum


def test_ls_l_vs_ls_a():
    # one checksum for both before the template was keyed (RFC 0004 section 1)
    long, all_ = _ls("-l"), _ls("-a")
    assert long(flag=True)._checksum != all_(flag=True)._checksum
    assert task_record(long(flag=True))["task"]["template"] != (
        task_record(all_(flag=True))["task"]["template"]
    )
    assert _ls("-l")(flag=True)._checksum == long(flag=True)._checksum


def test_help_change_no_change():
    assert _ls("-l", help="one")(flag=True)._checksum == (
        _ls("-l", help="two")(flag=True)._checksum
    )

    def with_output_help(text: str):
        return shell.define(
            "tool <x:int>",
            outputs={"o": shell.out(type=int, help=text, callable=len)},
        )

    assert with_output_help("a")(x=1)._checksum == with_output_help("b")(x=1)._checksum
    # a field that shapes the command line is not documentation
    assert _ls("-l")(flag=True)._checksum != _ls("-l", position=3)(flag=True)._checksum


def test_executable_and_version_are_in_the_task_record():
    record = task_record(
        shell.define("fslmaths")(executable=["fslmaths", "-dt", "float"])
    )
    assert record["task"]["template"]["executable"] == ["fslmaths", "-dt", "float"]
    assert record["task"]["version"] is None
    assert "executable" not in record["inputs"]
    declared = shell.define("tool <x:int>", version="6.0.7")
    assert task_record(declared(x=1))["task"]["version"] == "6.0.7"
    assert declared(x=1)._checksum != shell.define("tool <x:int>")(x=1)._checksum
    assert declared(x=1)._checksum != (
        shell.define("tool <x:int>", version="6.0.8")(x=1)._checksum
    )


def test_version_arguments_are_checked():
    with pytest.raises(ValueError, match="version_pattern"):
        shell.define("tool", version_command=["tool", "--version"])
    with pytest.raises(ValueError, match="version_command"):
        shell.define("tool", version_pattern=r"(\d+)")
    with pytest.raises(ValueError, match="capture group"):
        shell.define("tool", version_command=["tool", "-V"], version_pattern=r"\d+")
    with pytest.raises(ValueError, match="one of"):
        shell.define(
            "tool",
            version="1",
            version_command=["tool", "-V"],
            version_pattern=r"(\d+)",
        )


def _script(tmp_path: Path, body: str) -> list[str]:
    path = tmp_path / "tool.py"
    path.write_text(body)
    return [sys.executable, str(path)]


@pytest.fixture
def fresh_memo(monkeypatch):
    monkeypatch.setattr(shell_version, "_MEMO", {})
    monkeypatch.setattr(shell_version, "_WORKER_MEMO", {})


def test_version_pattern_first_group_only(tmp_path, fresh_memo):
    command = _script(
        tmp_path,
        "print('toolkit version 3.4.5 (Apple Git-155)')\n"
        "print('configuration: --prefix=/opt/x --enable-y 9.9.9')\n",
    )
    task = shell.define(
        "tool", version_command=command, version_pattern=r"version (\S+) \((\w+)"
    )()
    record = _job(task).record
    assert record["job"]["tool_version"] == "3.4.5"
    # the task's own record has no job field and its checksum is environment-free
    assert "job" not in task_record(task)


def test_version_command_no_match_raises(tmp_path, fresh_memo):
    command = _script(tmp_path, "print('no number here')\n")
    task = shell.define(
        "tool", version_command=command, version_pattern=r"version (\d+)"
    )()
    with pytest.raises(VersionCommandError, match="version"):
        _job(task).checksum
    assert issubclass(VersionCommandError, RuntimeError)
    # the task-level checksum does not run the command
    assert task._checksum.startswith("shell-")


def test_version_command_memoised_in_process(tmp_path, fresh_memo):
    counter = tmp_path / "calls"
    command = _script(
        tmp_path,
        f"open({str(counter)!r}, 'a').write('x')\nprint('v 1.0')\n",
    )

    def make():
        return shell.define(
            "tool", version_command=command, version_pattern=r"v (\S+)"
        )()

    assert _job(make()).record["job"]["tool_version"] == "1.0"
    assert _job(make()).record["job"]["tool_version"] == "1.0"
    assert counter.read_text() == "x"


def test_version_command_not_run_under_a_container(tmp_path, fresh_memo):
    counter = tmp_path / "calls"
    command = _script(
        tmp_path, f"open({str(counter)!r}, 'a').write('x')\nprint('v 1.0')\n"
    )
    task = shell.define("tool", version_command=command, version_pattern=r"v (\S+)")()
    env = docker.Docker(image="repo/tool", identity="sha256:aaaa")
    record = _job(task, env).record
    assert record["job"]["tool_version"] is None
    assert not counter.exists()


def test_executable_not_found_unresolved(fresh_memo):
    missing = ["no-such-tool-for-pydra-tests", "--version"]
    task = shell.define("tool", version_command=missing, version_pattern=r"(\d+)")()
    assert shutil.which(missing[0]) is None
    with pytest.warns(UserWarning, match="version="):
        job = _job(task)
        record = job.record
    assert record["job"]["tool_version"] == {
        "command": missing,
        "state": "unresolved",
    }
    # stable, so the result is found again within the process
    with pytest.warns(UserWarning, match="version="):
        assert _job(task).checksum == job.checksum
    # and used by this process only: the context carries a value unique to it
    assert "unknown" in job.context
    resolved = _job(shell.define("echo <t:str>", version="1")(t="a"))
    assert "unknown" not in resolved.context
    # a declared version looks nothing up and the key does not depend on the host
    declared = shell.define("no-such-tool-for-pydra-tests", version="1.0")()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _job(declared).record["job"]["tool_version"] is None


def test_task_checksum_vs_job_checksum():
    task = shell.define("echo <text:str>")(text="hi")
    job = _job(task)
    assert job.checksum != task._checksum
    assert job.checksum == record_checksum(job.record)
    assert task._checksum == record_checksum(task_record(task))
    assert job.record == {
        **task_record(task),
        "job": {"environment": job.environment.checksum(), "tool_version": None},
    }
    assert job_record(job) == job.record
    # the environment is the job's, and keys it
    one = _job(task, docker.Docker(image="repo/tool", identity="sha256:aaaa"))
    other = _job(task, docker.Docker(image="repo/tool", identity="sha256:bbbb"))
    assert len({job.checksum, one.checksum, other.checksum}) == 3
    assert one.record["job"]["environment"] != other.record["job"]["environment"]
    # the store identity's key is the carried record, and it checks out
    ident = PydraIdentityV2()
    assert ident.key(one) == one.record
    assert ident.check_key(one.checksum, ident.key(one))
    assert not ident.check_key(task._checksum, ident.key(one))


def _add(a: int) -> int:
    return a + 1


def test_python_and_supplied_jobs_keep_the_task_checksum():
    py = python.define(_add)(a=1)
    assert _job(py).checksum == py._checksum
    assert "job" not in _job(py).record

    def keyed(task, inputs):
        return "key-1"

    supplied = shell.define("echo <text:str>", checksum=keyed)(text="hi")
    assert _job(supplied).checksum == supplied._checksum == "key-1"
    assert "job" not in _job(supplied).record


def _counting_script(tmp_path: Path) -> list[str]:
    counter = tmp_path / "invocations"
    return _script(
        tmp_path,
        f"p = {str(counter)!r}\n"
        "import os\n"
        "n = (os.path.getsize(p) if os.path.exists(p) else 0) + 1\n"
        "open(p, 'a').write('x')\n"
        "print(f'tool {n}.0')\n",
    )


def test_worker_version_mismatch_fails_job(any_worker, tmp_path, fresh_memo):
    """Every call of the tool reports a new version, so the worker's check differs
    from the version the submitting process keyed"""
    task = shell.define(
        "echo <text:str>",
        version_command=_counting_script(tmp_path),
        version_pattern=r"tool (\S+)",
    )(text="hi")
    with pytest.raises(Exception, match=r"1\.0.*2\.0|2\.0.*1\.0"):
        task(cache_root=tmp_path / "cache", worker=any_worker)


def test_worker_version_match_runs(any_worker, tmp_path, fresh_memo):
    task = shell.define(
        "echo <text:str>",
        version_command=_script(tmp_path, "print('tool 7.1')\n"),
        version_pattern=r"tool (\S+)",
    )(text="hi")
    outputs = task(cache_root=tmp_path / "cache", worker=any_worker)
    assert outputs.stdout.strip() == "hi"


def test_every_field_of_a_template_has_its_own_digest():
    """Fields that differ only in their ``argstr`` do not share a digest, however the
    allocator reuses the memory of the specs hashed before them"""
    inputs = {
        f"f{i}": shell.arg(type=int, argstr=f"--flag-{i}", position=i + 1)
        for i in range(40)
    }
    for _ in range(5):
        record = task_record(shell.define("tool", inputs=inputs)(**{"f0": 1}))
        fields = record["task"]["template"]["fields"]
        assert len(set(fields.values())) == len(fields)


def _consume(t: object) -> int:
    return 1


def test_task_value_is_keyed_by_template_and_version():
    """A task passed as an input value is a value: its checksum is the task's
    environment-free one, so its command and declared version are in it (RFC 0004 3.1)
    """
    consumer = python.define(_consume)
    assert consumer(t=_ls("-l")(flag=True))._checksum != (
        consumer(t=_ls("-a")(flag=True))._checksum
    )
    one = shell.define("tool <x:int>", version="1")
    two = shell.define("tool <x:int>", version="2")
    assert consumer(t=one(x=1))._checksum != consumer(t=two(x=1))._checksum
    assert consumer(t=one(x=1))._checksum == consumer(t=one(x=1))._checksum
    # a split task value keeps its splitter in the encoding
    assert consumer(t=one().split(x=[1, 2]))._checksum != (
        consumer(t=one(x=1))._checksum
    )


def test_version_command_elements_are_strings():
    with pytest.raises(ValueError, match="list of strings"):
        shell.define("tool", version_command=["tool", 1], version_pattern=r"(\d+)")


def test_no_match_with_empty_stdout_shows_stderr_and_return_code(tmp_path, fresh_memo):
    command = _script(
        tmp_path,
        "import sys\nprint('openjdk 17.0.1', file=sys.stderr)\nsys.exit(3)\n",
    )
    task = shell.define("tool", version_command=command, version_pattern=r"jdk (\S+)")()
    with pytest.raises(VersionCommandError, match=r"return code 3.*openjdk 17"):
        _job(task).checksum
