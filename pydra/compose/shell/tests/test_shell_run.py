import typing as ty
import os
import sys
import pytest
from pathlib import Path
import re
import stat
import attrs
from pydra.engine.submitter import Submitter
from pydra.compose import shell, workflow, python
from fileformats.generic import (
    File,
    Directory,
)
from pydra.utils.typing import (
    MultiOutputFile,
    MultiInputObj,
    StateArray,
)
from pydra.engine.tests.utils import (
    run_no_submitter,
    run_submitter,
    no_win,
    get_output_names,
)

if sys.platform.startswith("win"):
    pytest.skip("SLURM not available in windows", allow_module_level=True)


@pytest.mark.flaky(reruns=2)  # when dask
@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_1(worker, results_function, tmp_path):
    """simple command, no arguments"""
    cmd = ["pwd"]
    shelly = shell.define(cmd)()
    assert shelly.cmdline == " ".join(cmd)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert Path(outputs.stdout.rstrip()).parent == tmp_path
    assert outputs.return_code == 0
    assert outputs.stderr == ""


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_1_strip(worker, results_function, tmp_path):
    """simple command, no arguments
    strip option to remove \n at the end os stdout
    """
    cmd = ["pwd"]
    shelly = shell.define(cmd)()

    assert shelly.cmdline == " ".join(cmd)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert Path(outputs.stdout).parent == tmp_path
    assert outputs.return_code == 0
    assert outputs.stderr == ""


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_2(worker, results_function, tmp_path):
    """a command with arguments, cmd and args given as executable"""
    cmd = ["echo", "hail", "pydra"]
    shelly = shell.define(cmd)()

    assert shelly.cmdline == " ".join(cmd)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout.strip() == " ".join(cmd[1:])
    assert outputs.return_code == 0
    assert outputs.stderr == ""


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_2a(worker, results_function, tmp_path):
    """a command with arguments, using executable and args"""
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = shell.define(cmd_exec)(append_args=cmd_args)

    assert shelly.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout.strip() == " ".join(cmd_args)
    assert outputs.return_code == 0
    assert outputs.stderr == ""


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_2b(worker, results_function, tmp_path):
    """a command with arguments, using  strings executable and args"""
    cmd_exec = "echo"
    cmd_args = ["pydra"]
    # separate command into exec + args
    shelly = shell.define(cmd_exec)(append_args=cmd_args)

    assert shelly.executable == "echo"
    assert shelly.cmdline == "echo pydra"

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "pydra\n"
    assert outputs.return_code == 0
    assert outputs.stderr == ""


# tests with State


@pytest.mark.flaky(reruns=2)
def test_shell_cmd_3(worker, tmp_path):
    """commands without arguments
    splitter = executable
    """
    cmd = ["pwd", "whoami"]

    # all args given as executable
    shelly = shell.define("shelly")().split(executable=cmd)

    # assert shelly.cmdline == ["pwd", "whoami"]
    outputs = shelly(worker=worker, cache_root=tmp_path)
    assert Path(outputs.stdout[0].rstrip()).parent == tmp_path

    if "USER" in os.environ:
        assert outputs.stdout[1] == f"{os.environ['USER']}\n"
    else:
        assert outputs.stdout[1]
    assert outputs.return_code[0] == outputs.return_code[1] == 0
    assert outputs.stderr[0] == outputs.stderr[1] == ""


def test_shell_cmd_4(worker, tmp_path):
    """a command with arguments, using executable and args
    splitter=args
    """
    cmd_exec = "echo"
    cmd_args = [["nipype"], ["pydra"]]
    # separate command into exec + args
    shelly = shell.define(cmd_exec)().split(append_args=cmd_args)

    assert shelly.executable == "echo"
    assert shelly.append_args == StateArray([["nipype"], ["pydra"]])
    # assert shelly.cmdline == ["echo nipype", "echo pydra"]
    outputs = shelly(worker=worker)

    assert outputs.stdout[0] == "nipype\n"
    assert outputs.stdout[1] == "pydra\n"

    assert outputs.return_code[0] == outputs.return_code[1] == 0
    assert outputs.stderr[0] == outputs.stderr[1] == ""


def test_shell_cmd_5(worker, tmp_path):
    """a command with arguments
    using splitter and combiner for args
    """
    cmd_exec = "echo"
    cmd_args = [["nipype"], ["pydra"]]
    # separate command into exec + args
    shelly = shell.define(cmd_exec)().split(append_args=cmd_args).combine("append_args")

    assert shelly.executable == "echo"
    assert shelly.append_args == StateArray([["nipype"], ["pydra"]])
    # assert shelly.cmdline == ["echo nipype", "echo pydra"]
    outputs = shelly(worker=worker)

    assert outputs.stdout[0] == "nipype\n"
    assert outputs.stdout[1] == "pydra\n"


def test_shell_cmd_6(worker, tmp_path):
    """a command with arguments,
    outer splitter for executable and args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = [["nipype"], ["pydra"]]
    # separate command into exec + args
    shelly = shell.define("shelly")().split(
        ["executable", "append_args"], executable=cmd_exec, append_args=cmd_args
    )

    assert shelly.executable == ["echo", ["echo", "-n"]]
    assert shelly.append_args == StateArray([["nipype"], ["pydra"]])
    outputs = shelly(cache_root=tmp_path, worker=worker)

    assert outputs.stdout == ["nipype\n", "pydra\n", "nipype", "pydra"]

    assert (
        outputs.return_code[0]
        == outputs.return_code[1]
        == outputs.return_code[2]
        == outputs.return_code[3]
        == 0
    )
    assert (
        outputs.stderr[0]
        == outputs.stderr[1]
        == outputs.stderr[2]
        == outputs.stderr[3]
        == ""
    )


def test_shell_cmd_7(worker, tmp_path):
    """a command with arguments,
    outer splitter for executable and args, and combiner=args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = [["nipype"], ["pydra"]]
    # separate command into exec + args
    shelly = (
        shell.define("shelly")()
        .split(
            ["executable", "append_args"],
            executable=cmd_exec,
            append_args=cmd_args,
        )
        .combine("append_args")
    )

    assert shelly.executable == ["echo", ["echo", "-n"]]
    assert shelly.append_args == StateArray([["nipype"], ["pydra"]])

    outputs = shelly(worker=worker)

    assert outputs.stdout == [["nipype\n", "pydra\n"], ["nipype", "pydra"]]


# tests with workflows


def test_wf_shell_cmd_1(worker, tmp_path):
    """a workflow with two connected commands"""

    @workflow.define
    def Workflow(cmd1, cmd2):
        shelly_pwd = workflow.add(shell.define(cmd1)())

        @python.define
        def StripAndListify(x: str) -> list[str]:
            return [x.strip()]

        listify = workflow.add(StripAndListify(x=shelly_pwd.stdout))
        shelly_ls = workflow.add(shell.define(cmd2)(append_args=listify.out))
        return shelly_ls.stdout

    wf = Workflow(cmd1="pwd", cmd2="ls")

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert "_result.pklz" in res.outputs.out
    assert "_job.pklz" in res.outputs.out


# customised input task


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_1(worker, results_function, tmp_path):
    """a command with executable, args and one command opt,
    using a customized input_spec to add the opt to the command
    in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = True
    cmd_args = ["hello from pydra"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_n: bool = shell.arg(
            position=1,
            argstr="-n",
            help="option",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(append_args=cmd_args, opt_n=cmd_opt)
    assert shelly.executable == cmd_exec
    assert shelly.append_args == cmd_args
    assert shelly.cmdline == "echo -n 'hello from pydra'"

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "hello from pydra"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_2(worker, results_function, tmp_path):
    """a command with executable, args and two command options,
    using a customized input_spec to add the opt to the command
    in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = True
    cmd_opt_hello = "HELLO"
    cmd_args = ["from pydra"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_hello: str = shell.arg(
            position=3,
            help="todo",
            argstr="",
        )
        opt_n: bool = shell.arg(
            position=1,
            help="todo",
            argstr="-n",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(append_args=cmd_args, opt_n=cmd_opt, opt_hello=cmd_opt_hello)
    assert shelly.executable == cmd_exec
    assert shelly.append_args == cmd_args
    assert shelly.cmdline == "echo -n HELLO 'from pydra'"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "HELLO from pydra"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_3(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    hello = "HELLO"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(text=hello)
    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_3a(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided
    using shorter syntax for input (no attr.ib)
    """
    cmd_exec = "echo"
    hello = "HELLO"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(position=1, help="text", argstr="")

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(text=hello)
    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_3b(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided after init"""
    cmd_exec = "echo"
    hello = "HELLO"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(executable=cmd_exec)
    shelly.text = hello

    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "HELLO\n"


def test_shell_cmd_inputspec_3c_exception(worker, tmp_path):
    """mandatory field added to fields, value is not provided, so exception is raised"""
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(executable=cmd_exec)

    with pytest.raises(ValueError, match="Mandatory field 'text' is not set"):
        shelly(cache_root=tmp_path)


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_3c(worker, results_function, tmp_path):
    """mandatory=False, so tasks runs fine even without the value"""
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: ty.Optional[str] = shell.arg(
            default=None,
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(executable=cmd_exec)

    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_4(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            default="Hello",
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(executable=cmd_exec)

    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_4a(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided
    using shorter syntax for input (no attr.ib)
    """
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(default="Hello", position=1, help="text", argstr="")

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(executable=cmd_exec)

    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_4b(worker, results_function, tmp_path):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            default="Hi",
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(executable=cmd_exec)

    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "echo Hi"

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "Hi\n"


def test_shell_cmd_inputspec_4d_exception(worker):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"

    # separate command into exec + args
    with pytest.raises(
        ValueError,
        match=r"path_template \('exception'\) can only be provided when there is no default",
    ):

        @shell.define
        class Shelly(shell.Task["Shelly.Outputs"]):
            executable = cmd_exec
            text: File = shell.outarg(
                default="Hello",
                position=1,
                help="text",
                path_template="exception",
                argstr="",
            )

            class Outputs(shell.Outputs):
                pass


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_5_nosubm(worker, results_function, tmp_path):
    """checking xor in metadata: task should work fine, since only one option is True"""
    cmd_exec = "ls"
    cmd_t = True

    @shell.define(xor=["opt_S", "opt_t"])
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_t: bool = shell.arg(
            position=1,
            help="opt t",
            argstr="-t",
        )
        opt_S: bool = shell.arg(
            default=False,
            position=2,
            help="opt S",
            argstr="-S",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(opt_t=cmd_t)
    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "ls -t"
    results_function(shelly, worker=worker, cache_root=tmp_path)


def test_shell_cmd_inputspec_5a_exception(worker, tmp_path):
    """checking xor in metadata: both options are True, so the task raises exception"""
    cmd_exec = "ls"
    cmd_t = True
    cmd_S = True

    @shell.define(xor=["opt_S", "opt_t"])
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_t: bool = shell.arg(
            position=1,
            help="opt t",
            argstr="-t",
        )
        opt_S: bool = shell.arg(
            position=2,
            help="opt S",
            argstr="-S",
        )

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(opt_t=cmd_t, opt_S=cmd_S)
    with pytest.raises(ValueError, match="Mutually exclusive fields"):
        shelly(cache_root=tmp_path)


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_6(worker, results_function, tmp_path):
    """checking requires in metadata:
    the required field is set in the init, so the task works fine
    """
    cmd_exec = "ls"
    cmd_l = True
    cmd_t = True

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_t: bool = shell.arg(
            position=2,
            help="opt t",
            argstr="-t",
            requires=["opt_l"],
        )
        opt_l: bool = shell.arg(
            position=1,
            help="opt l",
            argstr="-l",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(opt_t=cmd_t, opt_l=cmd_l)
    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    results_function(shelly, worker=worker, cache_root=tmp_path)


def test_shell_cmd_inputspec_6a_exception(worker, tmp_path):
    """checking requires in metadata:
    the required field is None, so the task works raises exception
    """
    cmd_exec = "ls"
    cmd_t = True

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_t: bool = shell.arg(
            position=2,
            help="opt t",
            argstr="-t",
            requires=["opt_l"],
        )
        opt_l: bool = shell.arg(
            default=False,
            position=1,
            help="opt l",
            argstr="-l",
        )

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(executable=cmd_exec, opt_t=cmd_t)

    with pytest.raises(ValueError, match=r"'opt_t' requires \['opt_l'\]"):
        shelly()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_6b(worker, results_function, tmp_path):
    """checking requires in metadata:
    the required field set after the init
    """
    cmd_exec = "ls"
    cmd_l = True
    cmd_t = True

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        opt_t: bool = shell.arg(
            position=2,
            help="opt t",
            argstr="-t",
            requires=["opt_l"],
        )
        opt_l: bool = shell.arg(
            position=1,
            help="opt l",
            argstr="-l",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly(
        opt_t=cmd_t
        # opt_l=cmd_l,
    )
    shelly.opt_l = cmd_l
    assert shelly.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    results_function(shelly, worker=worker, cache_root=tmp_path)


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_7(worker, results_function, tmp_path):
    """
    providing output name using input_spec,
    using name_tamplate in metadata
    """
    cmd = "touch"
    arg = tmp_path / "newfile_tmp.txt"
    cache_root = tmp_path / "cache"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(executable=cmd, arg=arg)

    outputs = results_function(shelly, worker=worker, cache_root=cache_root)
    assert outputs.stdout == ""
    out1 = outputs.out1.fspath
    assert out1.exists()
    # checking if the file is created in a good place
    assert out1.parent.parent == cache_root
    assert out1.name == "newfile_tmp.txt"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_7b(worker, results_function, tmp_path):
    """
    providing new file and output name using input_spec,
    using name_template in metadata
    """
    cmd = "touch"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        newfile: str = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{newfile}",
                help="output file",
            )

    shelly = Shelly(executable=cmd, newfile=File.mock("newfile_tmp.txt"))

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out1.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_7c(worker, results_function, tmp_path):
    """
    providing output name using input_spec,
    using name_tamplate with txt extension (extension from args should be removed
    """
    cmd = "touch"
    arg = File.mock("newfile_tmp.txt")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd

        arg = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{arg}.txt",
                help="output file",
            )

    shelly = Shelly(executable=cmd, arg=arg)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    # checking if the file is created in a good place
    assert outputs.out1.fspath.parent.parent == tmp_path
    assert outputs.out1.fspath.name == "newfile_tmp.txt"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_8(worker, results_function, tmp_path):
    """
    providing new file and output name using input_spec,
    adding additional string input field with argstr
    """
    cmd = "touch"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        newfile: str = shell.arg(
            position=2,
            help="new file",
            argstr="",
        )
        time: str = shell.arg(
            position=1,
            argstr="-t",
            help="time of modif.",
        )

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{newfile}",
                help="output file",
            )

    shelly = Shelly(
        executable=cmd,
        newfile=tmp_path / "newfile_tmp.txt",
        time="02121010",
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out1.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_8a(worker, results_function, tmp_path):
    """
    providing new file and output name using input_spec,
    adding additional string input field with argstr (argstr uses string formatting)
    """
    cmd = "touch"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        newfile: str = shell.arg(
            position=2,
            help="new file",
            argstr="",
        )
        time: str = shell.arg(
            position=1,
            argstr="-t {time}",
            help="time of modif.",
        )

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{newfile}",
                help="output file",
            )

    shelly = Shelly(
        executable=cmd,
        newfile=tmp_path / "newfile_tmp.txt",
        time="02121010",
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out1.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_9(tmp_path, worker, results_function):
    """
    providing output name using input_spec (path_template in metadata),
    the template has a suffix, the extension of the file will be moved to the end
    """
    cmd = "cp"
    ddir = tmp_path / "data_inp"
    ddir.mkdir()
    file = ddir / ("file.txt")
    file.write_text("content\n")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template="{file_orig}_copy",
                help="output file",
                argstr="",
            )

    shelly = Shelly(
        executable=cmd,
        file_orig=file,
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.file_copy.fspath.exists()
    assert outputs.file_copy.fspath.name == "file_copy.txt"
    # checking if it's created in a good place
    assert outputs.file_copy.fspath.parent.parent == tmp_path


@pytest.mark.parametrize("results_function", [run_no_submitter])
def test_shell_cmd_inputspec_9a(tmp_path, worker, results_function):
    """
    providing output name using input_spec (path_template in metadata),
    the template has a suffix, the extension of the file will be moved to the end
    the change: input file has directory with a dot
    """
    cmd = "cp"
    file = tmp_path / "data.inp" / "file.txt"
    file.parent.mkdir()
    file.write_text("content\n")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template="{file_orig}_copy",
                help="output file",
                argstr="",
            )

    shelly = Shelly(executable=cmd, file_orig=file)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.file_copy.fspath.exists()
    assert outputs.file_copy.fspath.name == "file_copy.txt"
    # checking if it's created in a good place
    assert outputs.file_copy.fspath.parent.parent == tmp_path


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_9b(tmp_path, worker, results_function):
    """
    providing output name using input_spec (path_template in metadata)
    and the keep_extension is set to False, so the extension is removed completely.
    """
    cmd = "cp"
    file = tmp_path / "file.txt"
    file.write_text("content\n")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template="{file_orig}_copy",
                keep_extension=False,
                help="output file",
                argstr="",
            )

    shelly = Shelly(
        executable=cmd,
        file_orig=file,
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.file_copy.fspath.exists()
    assert outputs.file_copy.fspath.name == "file_copy"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_9c(tmp_path, worker, results_function):
    """
    providing output name using input_spec (path_template in metadata)
    and the keep_extension is set to False, so the extension is removed completely,
    no suffix in the template.
    """
    cmd = "cp"
    file = tmp_path / "file.txt"
    file.write_text("content\n")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template="{file_orig}",
                keep_extension=False,
                help="output file",
                argstr="",
            )

    shelly = Shelly(executable=cmd, file_orig=file)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.file_copy.fspath.exists()
    assert outputs.file_copy.fspath.name == "file"
    assert outputs.file_copy.fspath.parent.parent == tmp_path


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_9d(tmp_path, worker, results_function):
    """
    providing output name explicitly by manually setting value in input_spec
    (instead of using default provided bypath_template in metadata)
    """
    cmd = "cp"
    ddir = tmp_path / "data_inp"
    ddir.mkdir()
    file = ddir / ("file.txt")
    file.write_text("content\n")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=1,
            help="new file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template="{file_orig}_copy",
                help="output file",
                argstr="",
            )

    shelly = Shelly(
        executable=cmd,
        file_orig=file,
        file_copy="my_file_copy.txt",
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.file_copy.fspath.exists()
    assert outputs.file_copy.fspath.name == "my_file_copy.txt"
    # checking if it's created in a good place
    assert outputs.file_copy.fspath.parent.parent == tmp_path


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_10(worker, results_function, tmp_path):
    """using input_spec, providing list of files as an input"""

    file_1 = tmp_path / "file_1.txt"
    file_2 = tmp_path / "file_2.txt"
    with open(file_1, "w") as f:
        f.write("hello ")
    with open(file_2, "w") as f:
        f.write("from boston")

    cmd_exec = "cat"
    files_list = [file_1, file_2]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        files: ty.List[File] = shell.arg(
            position=1,
            argstr="...",
            sep=" ",
            help="list of files",
        )

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(
        files=files_list,
    )

    assert shelly.executable == cmd_exec
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == "hello from boston"


def test_shell_cmd_inputspec_10_err(tmp_path):
    """checking if the proper error is raised when broken symlink is provided
    as a input field with File as a type
    """

    file_1 = tmp_path / "file_1.txt"
    with open(file_1, "w") as f:
        f.write("hello")
    file_2 = tmp_path / "file_2.txt"

    # creating symlink and removing the original file
    os.symlink(file_1, file_2)
    os.remove(file_1)

    cmd_exec = "cat"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        files: File = shell.arg(
            position=1,
            argstr="",
            help="a file",
        )

        class Outputs(shell.Outputs):
            pass

    with pytest.raises(FileNotFoundError):
        Shelly(executable=cmd_exec, files=file_2)


def test_shell_cmd_inputspec_11(tmp_path):

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        inputFiles: MultiInputObj[str] = shell.arg(
            argstr="...",
            help="The list of input image files to be segmented.",
        )

        executable = "touch"

        class Outputs(shell.Outputs):
            outputFiles: MultiOutputFile = shell.outarg(
                help="""Corrected Output Images: should specify the same number of
                images as inputVolume, if only one element is given, then it is used as
                a file pattern where %s is replaced by the imageVolumeType,
                and %d by the index list location.""",
                path_template="{inputFiles}",
            )

    @workflow.define
    def Workflow(inputFiles):

        echoMultiple = workflow.add(Shelly(inputFiles=inputFiles))
        return echoMultiple.outputFiles

    wf = Workflow(
        inputFiles=[File.mock(tmp_path / "test1"), File.mock(tmp_path / "test2")]
    )

    # XXX: Figure out why this fails with "cf". Occurs in CI when using Ubuntu + Python >= 3.10
    #      (but not when using macOS + Python >= 3.10). Same error occurs in test_shell_cmd_outputspec_7a
    #      see https://github.com/nipype/pydra/issues/671
    with Submitter(worker="debug") as sub:
        result = sub(wf)

    for out_file in result.outputs.out:
        assert out_file.fspath.name == "test1" or out_file.fspath.name == "test2"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_12(tmp_path: Path, worker, results_function):
    """
    providing output name using input_spec
    path_template is provided as a function that returns
    various templates depending on the values of inputs fields
    """
    cmd = "cp"
    ddir = tmp_path / "data_inp"
    ddir.mkdir()
    file = ddir / "file.txt"
    file.write_text("content\n")

    def template_function(inputs):
        if inputs.number % 2 == 0:
            return "{file_orig}_even"
        else:
            return "{file_orig}_odd"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file_orig: File = shell.arg(
            position=2,
            help="new file",
            argstr="",
        )
        number: int = shell.arg(
            help="a number",
            argstr=None,
        )

        class Outputs(shell.Outputs):
            file_copy: File = shell.outarg(
                path_template=template_function,
                help="output file",
                argstr="",
            )

    shelly = Shelly(
        executable=cmd,
        file_orig=file,
        number=2,
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    fspath = outputs.file_copy.fspath
    assert fspath.exists()
    assert fspath.name == "file_even.txt"
    # checking if it's created in a good place
    assert fspath.parent.parent == tmp_path


def test_shell_cmd_inputspec_with_iterable():
    """Test formatting of argstr with different iterable types."""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "test"
        iterable_1: list[int] = shell.arg(
            help="iterable input 1",
            argstr="--in1",
            sep=" ",
        )
        iterable_2: set[str] = shell.arg(
            help="iterable input 2",
            argstr="--in2",
            sep=" ",
        )
        iterable_3: tuple[float, ...] = shell.arg(
            help="iterable input 3",
            argstr="--in3...",
        )

        class Outputs(shell.Outputs):
            pass

    task = Shelly()

    for iterable_type in (list, tuple):
        task.iterable_1 = iterable_type(range(3))
        task.iterable_2 = iterable_type(["foo"])
        task.iterable_3 = iterable_type([1, 0])
        assert task.cmdline == "test --in1 0 1 2 --in2 foo --in3 1.0 --in3 0.0"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_copyfile_1(worker, results_function, tmp_path):
    """shelltask changes a file in place,
    adding copy_mode="copy" to the file-input from input_spec
    hardlink or copy in the cache_dir should be created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        orig_file: File = shell.arg(
            help="orig file",
            copy_mode="copy",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.out(
                help="output file",
                callable=lambda orig_file: orig_file,
            )

    shelly = Shelly(executable=cmd, orig_file=str(file))

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out_file.fspath.exists()
    # the file is  copied, and than it is changed in place
    assert outputs.out_file.fspath.parent.parent == tmp_path
    with open(outputs.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_copyfile_1a(worker, results_function, tmp_path):
    """shelltask changes a file in place,
    adding copyfile=False to the File-input from input_spec
    hardlink or softlink in the cache_dir is created
    """
    file = tmp_path / "file_pydra.txt"
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        orig_file: File = shell.arg(
            position=1,
            argstr="",
            help="orig file",
            copy_mode="hardlink",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.out(
                callable=lambda orig_file: orig_file,
                help="output file",
            )

    shelly = Shelly(executable=cmd, orig_file=str(file))

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out_file.fspath.exists()
    # the file is uses a soft link, but it creates and an extra copy before modifying
    assert outputs.out_file.fspath.parent.parent == tmp_path

    assert outputs.out_file.fspath.parent.joinpath(
        outputs.out_file.fspath.name + "s"
    ).exists()
    with open(outputs.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the file is uses a soft link, but it creates and an extra copy
    # it might depend on the OS
    linked_file_copy = outputs.out_file.fspath.parent.joinpath(
        outputs.out_file.fspath.name + "s"
    )
    if linked_file_copy.exists():
        with open(linked_file_copy) as f:
            assert "hello from pydra\n" == f.read()

    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_state_1(worker, results_function, tmp_path):
    """adding state to the input from input_spec"""
    cmd_exec = "echo"
    hello = ["HELLO", "hi"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly().split("text", text=hello)
    assert shelly.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout[0] == "HELLO\n"
    assert outputs.stdout[1] == "hi\n"


def test_shell_cmd_inputspec_typeval_1(tmp_path):
    """customized input_spec with a type that doesn't match the value
    - raise an exception
    """
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: int = shell.arg(
            position=1,
            argstr="",
            help="text",
        )

        class Outputs(shell.Outputs):
            pass

    with pytest.raises(TypeError):
        Shelly(text="hello")


def test_shell_cmd_inputspec_typeval_2(tmp_path):
    """customized input_spec (shorter syntax) with a type that doesn't match the value
    - raise an exception
    """
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec

        text: int = shell.arg(position=1, argstr="", help="text")

        class Outputs(shell.Outputs):
            pass

    with pytest.raises(TypeError):
        Shelly(text="hello")


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_state_1a(worker, results_function, tmp_path):
    """adding state to the input from input_spec
    using shorter syntax for input_spec (without default)
    """
    cmd_exec = "echo"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        text: str = shell.arg(
            position=1,
            help="text",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    # separate command into exec + args
    shelly = Shelly().split(text=["HELLO", "hi"])
    assert shelly.executable == cmd_exec

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout[0] == "HELLO\n"
    assert outputs.stdout[1] == "hi\n"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_state_2(worker, results_function, tmp_path):
    """
    adding splitter to input that is used in the output_file_tamplate
    """
    cmd = "touch"
    args = ["newfile_1.txt", "newfile_2.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(executable=cmd).split(arg=args)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    for i in range(len(args)):
        assert outputs.stdout[i] == ""
        assert outputs.out1[i].fspath.exists()
        assert outputs.out1[i].fspath.parent.parent == tmp_path


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_state_3(worker, results_function, tmp_path):
    """adding state to the File-input from input_spec"""

    file_1 = tmp_path / "file_pydra.txt"
    file_2 = tmp_path / "file_nice.txt"
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd_exec = "cat"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd_exec
        file: File = shell.arg(
            position=1,
            help="files",
            argstr="",
        )

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly().split(file=[file_1, file_2])

    assert shelly.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ["hello from pydra", "have a nice one"]


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_inputspec_copyfile_state_1(worker, results_function, tmp_path):
    """adding state to the File-input from input_spec"""

    file1 = tmp_path / "file1.txt"
    with open(file1, "w") as f:
        f.write("hello from pydra\n")

    file2 = tmp_path / "file2.txt"
    with open(file2, "w") as f:
        f.write("hello world\n")

    files = [str(file1), str(file2)]
    cmd = ["sed", "-is", "s/hello/hi/"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        orig_file: File = shell.arg(
            position=1,
            argstr="",
            help="orig file",
            copy_mode="copy",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.out(
                callable=lambda orig_file: orig_file,
                help="output file",
            )

    shelly = Shelly(
        executable=cmd,
    ).split("orig_file", orig_file=files)

    txt_l = ["from pydra", "world"]
    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    for i in range(len(files)):
        assert outputs.stdout[i] == ""
        assert outputs.out_file[i].fspath.exists()
        # the file is  copied, and than it is changed in place
        assert outputs.out_file[i].fspath.parent.parent == tmp_path
        with open(outputs.out_file[i]) as f:
            assert f"hi {txt_l[i]}\n" == f.read()
        # the original file is unchanged
        with open(files[i]) as f:
            assert f"hello {txt_l[i]}\n" == f.read()


# customised input_spec in Workflow


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_shell_cmd_2(worker, tmp_path):
    """a workflow with input with defined path_template (str)
    that requires wf.lzin
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "touch"

        arg: str = shell.arg()

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @workflow.define(outputs=["out_f", "stdout"])
    def Workflow(cmd, arg):

        shelly = workflow.add(
            Shelly(
                executable=cmd,
                arg=arg,
            )
        )

        return shelly.out1, shelly.stdout

    wf = Workflow(cmd="touch", arg="newfile.txt")

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert res.outputs.stdout == ""
    assert res.outputs.out_f.fspath.exists()
    assert res.outputs.out_f.fspath.parent.parent == tmp_path


def test_wf_shell_cmd_2a(worker, tmp_path):
    """a workflow with input with defined path_template (tuple)
    that requires wf.lzin
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "shelly"

        arg: str = shell.arg()

        class Outputs(shell.Outputs):
            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @workflow.define(outputs=["out_f", "out"])
    def Workflow(cmd, arg):

        shelly = workflow.add(
            Shelly(
                executable=cmd,
                arg=arg,
            )
        )

        return shelly.out1, shelly.stdout

    wf = Workflow(cmd="touch", arg="newfile.txt")

    with Submitter(worker=worker) as sub:
        res = sub(wf)

    assert res.outputs.out == ""
    assert res.outputs.out_f.fspath.exists()


def test_wf_shell_cmd_3(worker, tmp_path):
    """a workflow with 2 tasks,
    first one has input with path_template (str, uses wf.lzin),
    that is passed to the second task
    """

    @shell.define
    class Shelly1(shell.Task["Shelly1.Outputs"]):
        executable = "shelly"

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            file: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @shell.define
    class Shelly2(shell.Task["Shelly2.Outputs"]):

        executable = "shelly2"

        orig_file: File = shell.arg(
            position=1,
            help="output file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                position=2,
                argstr="",
                path_template="{orig_file}_copy",
                help="output file",
            )

    @workflow.define(outputs=["touch_file", "out1", "cp_file", "out2"])
    def Workflow(cmd1, cmd2, arg):

        shelly1 = workflow.add(
            Shelly1(
                executable=cmd1,
                arg=arg,
            )
        )
        shelly2 = workflow.add(
            Shelly2(
                executable=cmd2,
                orig_file=shelly1.file,
            )
        )

        return shelly1.file, shelly1.stdout, shelly2.out_file, shelly2.stdout

    wf = Workflow(cmd1="touch", cmd2="cp", arg="newfile.txt")

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert res.outputs.out1 == ""
    assert res.outputs.touch_file.fspath.exists()
    assert res.outputs.touch_file.fspath.parent.parent == tmp_path
    assert res.outputs.out2 == ""
    assert res.outputs.cp_file.fspath.exists()
    assert res.outputs.cp_file.fspath.parent.parent == tmp_path


def test_wf_shell_cmd_3a(worker, tmp_path):
    """a workflow with 2 tasks,
    first one has input with path_template (str, uses wf.lzin),
    that is passed to the second task
    """

    @shell.define
    class Shelly1(shell.Task["Shelly1.Outputs"]):
        executable = "shelly"
        arg: str = shell.outarg(argstr=None)

        class Outputs(shell.Outputs):

            file: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @shell.define
    class Shelly2(shell.Task["Shelly2.Outputs"]):
        executable = "shelly2"
        orig_file: str = shell.arg(
            position=1,
            help="output file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                position=2,
                argstr="",
                path_template="{orig_file}_cp",
                help="output file",
            )

    @workflow.define(outputs=["touch_file", "out1", "cp_file", "out2"])
    def Workflow(cmd1, cmd2, arg):

        shelly1 = workflow.add(
            Shelly1(
                executable=cmd1,
                arg=arg,
            )
        )
        shelly2 = workflow.add(
            Shelly2(
                executable=cmd2,
                orig_file=shelly1.file,
            )
        )

        return shelly1.file, shelly1.stdout, shelly2.out_file, shelly2.stdout

    wf = Workflow(cmd1="touch", cmd2="cp", arg="newfile.txt")

    with Submitter(worker=worker) as sub:
        res = sub(wf)

    assert res.outputs.out1 == ""
    assert res.outputs.touch_file.fspath.exists()
    assert res.outputs.out2 == ""
    assert res.outputs.cp_file.fspath.exists()


def test_wf_shell_cmd_state_1(worker, tmp_path):
    """a workflow with 2 tasks and splitter on the wf level,
    first one has input with path_template (str, uses wf.lzin),
    that is passed to the second task
    """

    @shell.define
    class Shelly1(shell.Task["Shelly1.Outputs"]):
        executable = "shelly1"

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            file: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @shell.define
    class Shelly2(shell.Task["Shelly2.Outputs"]):
        executable = "shelly2"
        orig_file: str = shell.arg(
            position=1,
            help="output file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                position=2,
                argstr="",
                path_template="{orig_file}_copy",
                help="output file",
            )

    @workflow.define(outputs=["touch_file", "out1", "cp_file", "out2"])
    def Workflow(cmd1, cmd2, arg):

        shelly1 = workflow.add(
            Shelly1(
                executable=cmd1,
                arg=arg,
            )
        )
        shelly2 = workflow.add(
            Shelly2(
                executable=cmd2,
                orig_file=shelly1.file,
            )
        )

        return shelly1.file, shelly1.stdout, shelly2.out_file, shelly2.stdout

    wf = Workflow(cmd1="touch", cmd2="cp").split(arg=["newfile_1.txt", "newfile_2.txt"])

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    for i in range(2):
        assert res.outputs.out1[i] == ""
        assert res.outputs.touch_file[i].fspath.exists()
        assert res.outputs.touch_file[i].fspath.parent.parent == tmp_path
        assert res.outputs.out2[i] == ""
        assert res.outputs.cp_file[i].fspath.exists()
        assert res.outputs.cp_file[i].fspath.parent.parent == tmp_path


def test_wf_shell_cmd_ndst_1(worker, tmp_path):
    """a workflow with 2 tasks and a splitter on the node level,
    first one has input with path_template (str, uses wf.lzin),
    that is passed to the second task
    """

    @shell.define
    class Shelly1(shell.Task["Shelly1.Outputs"]):
        executable = "shelly"

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):
            file: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    @shell.define
    class Shelly2(shell.Task["Shelly2.Outputs"]):
        executable = "shelly2"

        orig_file: str = shell.arg(
            position=1,
            help="output file",
            argstr="",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                position=2,
                argstr="",
                path_template="{orig_file}_copy",
                help="output file",
            )

    @workflow.define(outputs=["touch_file", "out1", "cp_file", "out2"])
    def Workflow(cmd1, cmd2, args):

        shelly1 = workflow.add(
            Shelly1(
                executable=cmd1,
            ).split("arg", arg=args)
        )
        shelly2 = workflow.add(
            Shelly2(
                executable=cmd2,
                orig_file=shelly1.file,
            )
        )

        return shelly1.file, shelly1.stdout, shelly2.out_file, shelly2.stdout

    wf = Workflow(
        cmd1="touch",
        cmd2="cp",
        args=["newfile_1.txt", "newfile_2.txt"],
    )

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert res.outputs.out1 == ["", ""]
    assert all([file.fspath.exists() for file in res.outputs.touch_file])
    assert res.outputs.out2 == ["", ""]
    assert all([file.fspath.exists() for file in res.outputs.cp_file])


# customised output task


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_1(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    Shelly = shell.define(
        cmd,
        outputs=[
            shell.out(name="newfile", type=File, callable=lambda: "newfile_tmp.txt")
        ],
    )
    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_1a(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: File = shell.outarg(path_template="newfile_tmp.txt")

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile.fspath.exists()


def test_shell_cmd_outputspec_1b_exception(worker, tmp_path):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: File = shell.out(callable=lambda: "newfile_tmp_.txt")

    shelly = Shelly()

    with pytest.raises(Exception) as exinfo:
        shelly(worker=worker)
    assert "does not exist" in str(exinfo.value)


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_2(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: File = shell.outarg(path_template="newfile_*.txt")

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile.fspath.exists()


def test_shell_cmd_outputspec_2a_exception(worker, tmp_path):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: File = shell.out(default="newfile_*K.txt")

    shelly = Shelly()

    with pytest.raises(FileNotFoundError):
        shelly(cache_root=tmp_path, worker="debug")


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_3(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default, should collect two files
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: MultiOutputFile = "newfile_*.txt"

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    # newfile is a list
    assert len(outputs.newfile) == 2
    assert all([file.fspath.exists() for file in outputs.newfile])


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_5(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    and uses cache_dir and the glob function
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(field, cache_dir):
        if field.name == "newfile":
            return list(Path(cache_dir).expanduser().glob("newfile*.txt"))

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: MultiOutputFile = shell.out(callable=gather_output)

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    # newfile is a list
    assert len(outputs.newfile) == 2
    assert all([file.fspath.exists() for file in outputs.newfile])
    assert get_output_names(shelly) == ["newfile", "return_code", "stderr", "stdout"]


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_5a(worker, results_function, tmp_path):
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    and uses cache_dir and inputs element
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(executable, cache_dir):
        files = executable[1:]
        return [Path(cache_dir) / file for file in files]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):

            newfile: MultiOutputFile = shell.out(callable=gather_output)

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    # newfile is a list
    assert len(outputs.newfile) == 2
    assert all([file.fspath.exists() for file in outputs.newfile])


def test_shell_cmd_outputspec_5b_error(tmp_path):
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    with an argument that is not part of the inputs - error is raised
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(executable, cache_dir, ble):
        files = executable[1:]
        return [Path(cache_dir) / file for file in files]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):
            newfile: File = shell.out(callable=gather_output)

    shelly = Shelly()
    with pytest.raises(KeyError, match="ble"):
        shelly(cache_root=tmp_path)


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_5c(worker, results_function, tmp_path):
    """
    Customised output defined as a class,
    using a static function to collect output files.
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

        class Outputs(shell.Outputs):

            @staticmethod
            def gather_output(executable, cache_dir):
                files = executable[1:]
                return [Path(cache_dir) / file for file in files]

            newfile: MultiOutputFile = shell.out(callable=gather_output)

    shelly = Shelly()

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    # newfile is a list
    assert len(outputs.newfile) == 2
    assert all([file.exists() for file in outputs.newfile])


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_6(worker, results_function, tmp_path):
    """
    providing output name by providing path_template
    (similar to the previous example, but not touching input_spec)
    """
    cmd = "touch"
    arg = "newfile_tmp.txt"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        arg: str = shell.arg()

        class Outputs(shell.Outputs):

            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(
        executable=cmd,
        arg=arg,
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out1.fspath.exists()


def test_shell_cmd_outputspec_6a(tmp_path):
    """
    providing output name by providing path_template
    (using shorter syntax)
    """
    cmd = "touch"
    arg = "newfile_tmp.txt"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        arg: str = shell.arg(argstr=None)

        class Outputs(shell.Outputs):

            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(arg=arg)

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.out1.fspath.exists()


@pytest.mark.xfail(
    sys.version_info >= (3, 11),
    reason=(
        "Fails on Python 3.11 in some cases (presumably a typing thing with that specific "
        "version of Python)"
    ),
)
@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_7(tmp_path, worker, results_function):
    """
    providing output with output_file_name and using MultiOutputFile as a type.
    the input field used in the template is a MultiInputObj, so it can be and is a list
    """
    file = tmp_path / "script.sh"
    file.write_text('for var in "$@"; do touch file"$var".txt; done')

    cmd = "bash"
    new_files_id = ["1", "2", "3"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        script: File = shell.arg(
            help="script file",
            position=1,
            argstr="",
        )
        files_id: MultiInputObj = shell.arg(
            position=2,
            argstr="...",
            sep=" ",
            help="list of name indices",
        )

        class Outputs(shell.Outputs):

            new_files: MultiOutputFile = shell.outarg(
                argstr=None,
                path_template="file{files_id}.txt",
                help="output file",
            )

    shelly = Shelly(
        script=file,
        files_id=new_files_id,
    )

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    assert outputs.stdout == ""
    for file in outputs.new_files:
        assert file.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_7a(tmp_path, worker, results_function):
    """
    providing output with output_file_name and using MultiOutputFile as a type.
    the input field used in the template is a MultiInputObj, but a single element is used
    """
    file = tmp_path / "script.sh"
    file.write_text('for var in "$@"; do touch file"$var".txt; done')

    cmd = "bash"
    new_files_id = "1"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        script: File = shell.arg(
            help="script file",
            position=1,
            argstr="",
        )
        files_id: MultiInputObj = shell.arg(
            position=2,
            argstr="...",
            sep=" ",
            help="list of name indices",
        )

        class Outputs(shell.Outputs):

            new_files: MultiOutputFile = shell.outarg(
                argstr=None,
                path_template="file{files_id}.txt",
                help="output file",
            )

    shelly = Shelly(
        script=file,
        files_id=new_files_id,
    )

    # XXX: Figure out why this fails with "cf". Occurs in CI when using Ubuntu + Python >= 3.10
    #      (but not when using macOS + Python >= 3.10). Same error occurs in test_shell_cmd_inputspec_11
    #      see https://github.com/nipype/pydra/issues/671
    outputs = results_function(shelly, cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.new_files.fspath.exists()


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_8a(tmp_path, worker, results_function):
    """
    customised output_spec, adding int and str to the output,
    requiring two callables with parameters stdout and stderr
    """
    cmd = "echo"
    args = [["newfile_1.txt"], ["newfile_2.txt"]]

    def get_file_index(stdout):
        stdout = re.sub(r".*_", "", stdout)
        stdout = re.sub(r".txt", "", stdout)
        print(stdout)
        return int(stdout)

    def get_stderr(stderr):
        return f"stderr: {stderr}"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        class Outputs(shell.Outputs):

            out_file_index: int = shell.out(
                help="output file",
                callable=get_file_index,
            )
            stderr_field: str = shell.out(
                help="The standard error output",
                callable=get_stderr,
            )

    shelly = Shelly().split(append_args=args)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    for index in range(2):
        assert outputs.out_file_index[index] == index + 1
        assert outputs.stderr_field[index] == f"stderr: {outputs.stderr[index]}"


def test_shell_cmd_outputspec_8b_error(tmp_path):
    """
    customised output_spec, adding Int to the output,
    requiring a function to collect output
    """

    with pytest.raises(
        ValueError,
        match="A shell output field must have either a callable or a path_template",
    ):

        @shell.define
        class Shelly(shell.Task["Shelly.Outputs"]):
            executable = "echo"

            class Outputs(shell.Outputs):
                out: int = shell.out(help="output file")


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_8c(tmp_path, worker, results_function):
    """
    customised output_spec, adding Directory to the output named by args
    """

    def get_lowest_directory(directory_path):
        return str(directory_path).replace(str(Path(directory_path).parents[0]), "")

    cmd = "mkdir"
    args = [f"{tmp_path}/dir1", f"{tmp_path}/dir2"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        arg: str = shell.arg()

        class Outputs(shell.Outputs):

            resultsDir: Directory = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(resultsDir="outdir").split(arg=args)

    results_function(shelly, worker=worker, cache_root=tmp_path)
    for index, arg_dir in enumerate(args):
        assert Path(Path(tmp_path) / Path(arg_dir)).exists()
        assert get_lowest_directory(arg_dir) == f"/dir{index+1}"


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_outputspec_8d(tmp_path, worker, results_function):
    """
    customised output_spec, adding Directory to the output named by input task
    """

    # For /tmp/some_dict/test this function returns "/test"
    def get_lowest_directory(directory_path):
        return str(directory_path).replace(str(Path(directory_path).parents[0]), "")

    cmd = "mkdir"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        resultsDir: str = shell.arg(
            position=1,
            help="new directory",
            argstr="",
        )

        class Outputs(shell.Outputs):

            resultsDir: Directory = shell.outarg(
                path_template="{resultsDir}",
                help="output file",
            )

    shelly = Shelly(resultsDir="test")
    assert get_output_names(shelly) == ["resultsDir", "return_code", "stderr", "stdout"]
    cache_root = tmp_path / "cache"
    outputs = results_function(shelly, worker=worker, cache_root=cache_root)
    cache_dir = next(p for p in cache_root.iterdir() if p.name.startswith("shell-"))
    assert (cache_dir / Path("test")).exists()
    assert get_lowest_directory(outputs.resultsDir) == get_lowest_directory(
        cache_dir / Path("test")
    )


@pytest.mark.parametrize("results_function", [run_no_submitter, run_submitter])
def test_shell_cmd_state_outputspec_1(worker, results_function, tmp_path):
    """
    providing output name by providing path_template
    splitter for a field that is used in the template
    """
    cmd = "touch"
    args = ["newfile_1.txt", "newfile_2.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = cmd

        arg: str = shell.arg()

        class Outputs(shell.Outputs):

            out1: File = shell.outarg(
                path_template="{arg}",
                help="output file",
            )

    shelly = Shelly(executable=cmd).split(arg=args)

    outputs = results_function(shelly, worker=worker, cache_root=tmp_path)
    for i in range(len(args)):
        assert outputs.stdout[i] == ""
        assert outputs.out1[i].fspath.exists()


# customised output_spec for tasks in workflows


def test_shell_cmd_outputspec_wf_1(worker, tmp_path):
    """
    customised output_spec for tasks within a Workflow,
    adding files to the output, providing specific pathname
    """

    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = "shelly"

        class Outputs(shell.Outputs):
            newfile: File = shell.outarg(path_template="newfile_tmp.txt")

    @workflow.define(outputs=["stdout", "newfile"])
    def Workflow(cmd):
        shelly = workflow.add(Shelly(executable=cmd))
        return shelly.stdout, shelly.newfile

    wf = Workflow(cmd=cmd)

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        res = sub(wf)

    assert res.outputs.stdout == ""
    assert res.outputs.newfile.fspath.exists()
    # checking if the file was copied to the wf dir
    assert res.outputs.newfile.fspath.parent.parent == tmp_path


def test_shell_cmd_inputspec_outputspec_1(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in templates
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        file1: File = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: File = shell.arg(help="2nd creadted file", argstr="", position=2)

        class Outputs(shell.Outputs):
            newfile1: File = shell.outarg(path_template="{file1}", help="newfile 1")
            newfile2: File = shell.outarg(path_template="{file2}", help="newfile 2")

        executable = cmd

    shelly = Shelly(
        file1=File.mock(tmp_path / "new_file_1.txt"),
        file2=File.mock(tmp_path / "new_file_2.txt"),
    )

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    assert outputs.newfile2.fspath.exists()


def test_shell_cmd_inputspec_outputspec_1a(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in templates,
    file2 is used in a template for newfile2, but it is not provided, so newfile2 is set to NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: str | None = shell.arg(
            default=None, help="2nd creadted file", argstr="", position=2
        )

        class Outputs(shell.Outputs):

            newfile1: File = shell.out(callable=lambda file1: file1, help="newfile 1")
            newfile2: File | None = shell.out(
                callable=lambda file2: file2, help="newfile 2"
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = File.mock(tmp_path / "new_file_1.txt")

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    # newfile2 is not created, since file2 is not provided
    assert outputs.newfile2 is None


def test_shell_cmd_inputspec_outputspec_2(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: str = shell.arg(help="2nd creadted file", argstr="", position=2)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                requires=["file1"],
            )
            newfile2: File | None = shell.out(
                callable=lambda file2: file2,
                help="newfile 1",
                requires=["file1", "file2"],
            )

    shelly = Shelly(file1="new_file_1.txt", file2="new_file_2.txt")
    assert get_output_names(shelly) == [
        "newfile1",
        "newfile2",
        "return_code",
        "stderr",
        "stdout",
    ]

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    assert outputs.newfile2.fspath.exists()


def test_shell_cmd_inputspec_outputspec_2a(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: str | None = shell.arg(
            default=None, help="2nd creadted file", argstr="", position=2
        )

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                requires=["file1"],
            )
            newfile2: File | None = shell.out(
                callable=lambda file2: file2,
                help="newfile 1",
                requires=["file1", "file2"],
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = tmp_path / "new_file_1.txt"
    assert get_output_names(shelly) == [
        "newfile1",
        "newfile2",
        "return_code",
        "stderr",
        "stdout",
    ]

    outputs = shelly(cache_root=tmp_path)

    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    assert outputs.newfile2 is None


def test_shell_cmd_inputspec_outputspec_3(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input that is not in the template, but in the requires field,
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: str = shell.arg(help="2nd creadted file", argstr="", position=2)
        additional_inp: int = shell.arg(help="additional inp")

        class Outputs(shell.Outputs):

            newfile1: File = shell.out(callable=lambda file1: file1, help="newfile 1")
            newfile2: File | None = shell.out(
                callable=lambda file2: file2,
                help="newfile 1",
                requires=["file1", "additional_inp"],
            )

    shelly = Shelly(executable=cmd)
    shelly.file1 = tmp_path / "new_file_1.txt"
    shelly.file2 = tmp_path / "new_file_2.txt"
    shelly.additional_inp = 2

    outputs = shelly(cache_root=tmp_path / "cache")
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    assert outputs.newfile2.fspath.exists()


def test_shell_cmd_inputspec_outputspec_3a(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input that is not in the template, but in the requires field,
    the additional input not provided, so the output is NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        file2: str | None = shell.arg(help="2nd creadted file", argstr="", position=2)
        additional_inp: str | None = shell.arg(default=None, help="additional inp")

        class Outputs(shell.Outputs):

            newfile1: File = shell.out(callable=lambda file1: file1, help="newfile 1")
            newfile2: File | None = shell.out(
                callable=lambda file2: file2,
                help="newfile 1",
                requires=["file1", "additional_inp"],
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = "new_file_1.txt"
    shelly.file2 = "new_file_2.txt"
    assert get_output_names(shelly) == [
        "newfile1",
        "newfile2",
        "return_code",
        "stderr",
        "stdout",
    ]

    shelly.file2 = None
    outputs = shelly(cache_root=tmp_path)

    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()
    # additional input not provided so no newfile2 set (even if the file was created)
    assert outputs.newfile2 is None


def test_shell_cmd_inputspec_outputspec_4(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input to the requires together with a list of the allowed values,
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        additional_inp: int | None = shell.arg(help="additional inp", default=None)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                requires=["file1", ("additional_inp", [2, 3])],
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = File.mock("new_file_1.txt")
    shelly.additional_inp = 2

    outputs = shelly(cache_root=tmp_path)
    assert get_output_names(shelly) == [
        "newfile1",
        "return_code",
        "stderr",
        "stdout",
    ]

    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()


def test_shell_cmd_inputspec_outputspec_4a(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input to the requires together with a list of the allowed values,
    the input is set to a value that is not in the list, so output is None
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        additional_inp: int | None = shell.arg(help="additional inp", default=None)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                requires=("file1", ("additional_inp", [2, 3])),
            )

    shelly = Shelly(executable=cmd)
    shelly.file1 = File.mock("new_file_1.txt")
    # the value is not in the list from requires
    shelly.additional_inp = 1

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1 is None


def test_shell_cmd_inputspec_outputspec_5(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    the first element of the requires list has all the fields set
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        additional_inp_A: int | None = shell.arg(help="additional inp A", default=None)
        additional_inp_B: str | None = shell.arg(help="additional inp B", default=None)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                # requires is a list of list so it's treated as el[0] OR el[1] OR...
                requires=[
                    ["file1", "additional_inp_A"],
                    ["file1", "additional_inp_B"],
                ],
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = File.mock("new_file_1.txt")
    shelly.additional_inp_A = 2

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()


def test_shell_cmd_inputspec_outputspec_5a(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    the second element of the requires list (i.e. additional_inp_B) has all the fields set
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        additional_inp_A: str | None = shell.arg(help="additional inp A", default=None)
        additional_inp_B: int | None = shell.arg(help="additional inp B", default=None)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                # requires is a list of list so it's treated as el[0] OR el[1] OR...
                requires=[
                    ["file1", "additional_inp_A"],
                    ["file1", "additional_inp_B"],
                ],
            )

    shelly = Shelly(
        executable=cmd,
    )
    shelly.file1 = File.mock("new_file_1.txt")
    shelly.additional_inp_B = 2

    outputs = shelly(cache_root=tmp_path)
    assert outputs.stdout == ""
    assert outputs.newfile1.fspath.exists()


def test_shell_cmd_inputspec_outputspec_5b(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    neither of the list from requirements has all the fields set, so the output is NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = cmd
        file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
        additional_inp_A: str | None = shell.arg(help="additional inp A", default=None)
        additional_inp_B: str | None = shell.arg(help="additional inp B", default=None)

        class Outputs(shell.Outputs):

            newfile1: File | None = shell.out(
                callable=lambda file1: file1,
                help="newfile 1",
                # requires is a list of list so it's treated as el[0] OR el[1] OR...
                requires=[
                    ["file1", "additional_inp_A"],
                    ["file1", "additional_inp_B"],
                ],
            )

    shelly = Shelly(executable=cmd)
    shelly.file1 = tmp_path / "new_file_1.txt"

    outputs = shelly(cache_root=tmp_path / "cache")
    assert outputs.stdout == ""
    # neither additional_inp_A nor additional_inp_B is set, so newfile1 is None
    assert outputs.newfile1 is None


@pytest.mark.xfail(
    reason="I'm not sure why this requirements specification should fail"
)
def test_shell_cmd_inputspec_outputspec_6_except(tmp_path):
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires has invalid syntax - exception is raised
    """

    with pytest.raises(Exception, match="requires field can be"):

        @shell.define
        class Shelly(shell.Task["Shelly.Outputs"]):
            executable = "touch"
            file1: str = shell.arg(help="1st creadted file", argstr="", position=1)
            additional_inp_A: str | None = shell.arg(
                default=None, help="additional inp A"
            )

            class Outputs(shell.Outputs):
                newfile1: File | None = shell.out(
                    callable=lambda file1: file1,
                    help="newfile 1",
                    # requires has invalid syntax
                    requires=[["file1", "additional_inp_A"], "file1"],
                )


def no_fsl():
    if "FSLDIR" not in os.environ:
        return True


@pytest.mark.skipif(no_fsl(), reason="fsl is not installed")
def test_fsl(data_tests_dir, tmp_path):
    """mandatory field added to fields, value provided"""

    def change_name(file):
        name, ext = os.path.splitext(file)
        return f"{name}_brain.{ext}"

    @shell.define(
        xor=[
            "functional",
            "reduce_bias",
            "robust",
            "padding",
            "remove_eyes",
            "surfaces",
            "t2_guided",
            None,
        ]
    )
    class Bet(shell.Task["Bet.Outputs"]):
        executable = "bet"
        in_file: File = shell.arg(
            help="input file to skull strip",
            position=1,
            argstr="",
        )

        outline: bool = shell.arg(
            default=False,
            help="create surface outline image",
            argstr="-o",
        )
        mask: bool = shell.arg(
            default=False,
            help="create binary mask image",
            argstr="-m",
        )
        skull: bool = shell.arg(
            default=False,
            help="create skull image",
            argstr="-s",
        )
        no_output: bool = shell.arg(
            default=False,
            help="Don't generate segmented output",
            argstr="-n",
        )
        frac: float | None = shell.arg(
            default=None,
            help="fractional intensity threshold",
            argstr="-f",
        )
        vertical_gradient: float | None = shell.arg(
            default=None,
            help="vertical gradient in fractional intensity threshold (-1, 1)",
            argstr="-g",
            allowed_values={"min_val": -1, "max_val": 1},
        )
        radius: int | None = shell.arg(default=None, argstr="-r", help="head radius")
        center: ty.List[int] | None = shell.arg(
            default=None,
            help="center of gravity in voxels",
            argstr="-c",
            allowed_values={"min_value": 0, "max_value": 3},
        )
        threshold: bool = shell.arg(
            default=False,
            argstr="-t",
            help="apply thresholding to segmented brain image and mask",
        )
        mesh: bool = shell.arg(
            default=False,
            argstr="-e",
            help="generate a vtk mesh brain surface",
        )
        robust: bool = shell.arg(
            default=False,
            help="robust brain centre estimation (iterates BET several times)",
            argstr="-R",
        )
        padding: bool = shell.arg(
            default=False,
            help="improve BET if FOV is very small in Z (by temporarily padding end slices",
            argstr="-Z",
        )
        remove_eyes: bool = shell.arg(
            default=False,
            help="eye & optic nerve cleanup (can be useful in SIENA)",
            argstr="-S",
        )
        surfaces: bool = shell.arg(
            default=False,
            help="run bet2 and then betsurf to get additional skull and scalp surfaces (includes registrations)",
            argstr="-A",
        )
        t2_guided: File | str | None = shell.arg(
            default=None,
            help="as with creating surfaces, when also feeding in non-brain-extracted T2 (includes registrations)",
            argstr="-A2",
        )
        functional: bool = shell.arg(
            default=False,
            argstr="-F",
            help="apply to 4D fMRI data",
        )
        reduce_bias: bool = shell.arg(
            default=False,
            argstr="-B",
            help="bias field and neck cleanup",
        )

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                help="name of output skull stripped image",
                position=2,
                argstr="",
                path_template="{in_file}_brain",
            )

        # ("number_classes", int, attr.ib(metadata={help='number of tissue-type classes', argstr='-n',
        #                                            allowed_values={"min_val": 1, max_val=10}})),
        # ("output_biasfield", bool,
        #  attr.ib(metadata={help='output estimated bias field', argstr='-b'})),
        # ("output_biascorrected", bool,
        #  attr.ib(metadata={help='output restored image (bias-corrected image)', argstr='-B'})),

    # TODO: not sure why this has to be string
    in_file = data_tests_dir / "test.nii.gz"

    # separate command into exec + args
    shelly = Bet(in_file=File.mock("/path/to/nifti.nii.gz"))
    assert shelly.executable == "bet"
    try:
        orig_dir = os.getcwd()
        os.chdir(tmp_path)
        assert (
            shelly.cmdline == f"bet /path/to/nifti.nii.gz {tmp_path}/nifti_brain.nii.gz"
        )
    finally:
        os.chdir(orig_dir)
    shelly = Bet(in_file=in_file)
    outputs = shelly(cache_root=tmp_path)
    assert outputs.out_file.name == "test_brain.nii.gz"


def test_shell_cmd_optional_output_file1(tmp_path):
    """
    Test to see that 'unused' doesn't complain about not having an output passed to it
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        input: File = shell.arg(argstr="", help="input file")

        executable = "cp"

        class Outputs(shell.Outputs):
            output: File = shell.outarg(
                argstr="",
                path_template="out.txt",
                help="output file",
            )
            unused: File | None = shell.outarg(
                default=None,
                argstr="--not-used",
                path_template="out.txt",
                help="dummy output",
            )

    file1 = tmp_path / "file1.txt"
    file1.write_text("foo")
    my_cp = Shelly(input=file1, unused=False)
    outputs = my_cp(cache_root=tmp_path)
    assert outputs.output.fspath.read_text() == "foo"


def test_shell_cmd_optional_output_file2(tmp_path):
    """
    Test to see that 'unused' doesn't complain about not having an output passed to it
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cp"

        input: File = shell.arg(argstr="", help="input file")

        class Outputs(shell.Outputs):
            output: File | None = shell.outarg(
                argstr="",
                path_template="out.txt",
                help="dummy output",
            )

    file1 = tmp_path / "file1.txt"
    file1.write_text("foo")
    my_cp = Shelly(input=file1, output=True)
    outputs = my_cp(cache_root=tmp_path)
    assert outputs.output.fspath.read_text() == "foo"

    file2 = tmp_path / "file2.txt"
    file2.write_text("bar")
    my_cp2 = Shelly(input=file2, output=False)
    with pytest.raises(RuntimeError):
        my_cp2()


def test_shell_cmd_non_existing_outputs_1(tmp_path):
    """Checking that non existing output files do not return a phantom path,
    but return None instead"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "echo"
        out_name: str = shell.arg(
            help="""
            base name of the pretend outputs.
            """,
        )

        class Outputs(shell.Outputs):
            out_1: File | None = shell.out(
                help="fictional output #1",
                callable=lambda: "out_1.nii",
            )
            out_2: File | None = shell.out(
                help="fictional output #2",
                callable=lambda: "out_2.nii",
            )

    shelly = Shelly(out_name="test")
    outputs = shelly(cache_root=tmp_path)
    assert outputs.out_1 is None
    assert outputs.out_2 is None


def test_shell_cmd_non_existing_outputs_2(tmp_path):
    """Checking that non existing output files do not return a phantom path,
    but return None instead. This test has one existing and one non existing output file.
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "touch"
        out_name: str = shell.arg(
            help="""
            base name of the pretend outputs.
            """,
            argstr="{out_name}_1.nii",
        )

        class Outputs(shell.Outputs):
            out_1: File = shell.outarg(
                help="fictional output #1",
                path_template="{out_name}_1.nii",
            )
            out_2: File | None = shell.outarg(
                help="fictional output #2",
                path_template="{out_name}_2.nii",
            )

    shelly = Shelly(out_name="test")
    cache_root = tmp_path / "cache"
    outputs = shelly(cache_root=cache_root)
    # the first output file is created
    assert (
        outputs.out_1.fspath
        == next(p for p in cache_root.iterdir() if p.name.startswith("shell-"))
        / "test_1.nii"
    )
    assert outputs.out_1.fspath.exists()
    # the second output file is not created
    assert outputs.out_2 is None


def test_shell_cmd_non_existing_outputs_3(tmp_path):
    """Checking that non existing output files do not return a phantom path,
    but return None instead. This test has an existing mandatory output and another
    non existing output file.
    """

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "touch"
        out_name: str = shell.arg(
            help="""
            base name of the pretend outputs.
            """,
            argstr=None,
        )

        class Outputs(shell.Outputs):
            out_1: File = shell.outarg(
                help="real output #1",
                path_template="{out_name}_1.nii",
            )
            out_2: File | None = shell.out(
                help="fictional output #2",
                callable=lambda out_name: f"{out_name}_2.nii",
            )

    shelly = Shelly(out_name="test")

    cache_root = tmp_path / "cache"
    outputs = shelly(cache_root=cache_root)
    # the first output file is created
    assert (
        outputs.out_1.fspath
        == next(p for p in cache_root.iterdir() if p.name.startswith("shell-"))
        / "test_1.nii"
    )
    assert outputs.out_1.fspath.exists()
    # the second output file is not created
    assert outputs.out_2 is None


def test_shell_cmd_non_existing_outputs_4(tmp_path):
    """Checking that non existing output files do not return a phantom path,
    but return None instead. This test has an existing mandatory output and another non existing
    mandatory output file."""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "touch"
        out_name: str = shell.arg(
            help="""base name of the pretend outputs.""",
            argstr="{out_name}_1.nii",
        )

        class Outputs(shell.Outputs):
            out_1: File = shell.out(
                help="real output #1",
                callable=lambda out_name: f"{out_name}_1.nii",
            )
            out_2: File = shell.out(
                help="fictional output #2",
                callable=lambda out_name: f"{out_name}_2.nii",
            )

    shelly = Shelly(out_name="test")
    # An exception should be raised because the second mandatory output does not exist
    cache_root = tmp_path / "cache"
    with pytest.raises(
        ValueError,
        match=r"file system path\(s\) provided to mandatory field .* does not exist",
    ):
        shelly(cache_root=cache_root)
    # checking if the first output was created
    assert (
        next(p for p in cache_root.iterdir() if p.name.startswith("shell-"))
        / "test_1.nii"
    ).exists()


def test_shell_cmd_non_existing_outputs_multi_1(tmp_path):
    """This test looks if non existing files of an multiOuputFile are also set to NOTHING"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "echo"
        out_name: MultiInputObj = shell.arg(
            help="""
                        base name of the pretend outputs.
                        """,
            argstr="...",
        )

        class Outputs(shell.Outputs):
            out_list: MultiOutputFile | None = shell.out(
                help="fictional output #1",
                callable=lambda out_name: out_name,
            )

    shelly = Shelly(out_name=["test_1.nii", "test_2.nii"])

    # with pytest.raises(ValueError):
    outputs = shelly(cache_root=tmp_path)
    assert outputs.out_list == None


def test_shell_cmd_non_existing_outputs_multi_2(tmp_path):
    """This test looks if non existing files of an multiOutputFile are also set to NOTHING.
    It checks that it also works if one file of the multiOutputFile actually exists."""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "touch"
        out_name: MultiInputObj = shell.arg(
            help="""base name of the pretend outputs.""",
            sep=" test_1_real.nii",  # hacky way of creating an extra file with that name
            argstr="...",
        )

        class Outputs(shell.Outputs):
            out_list: MultiOutputFile | None = shell.out(
                help="fictional output #1",
                callable=lambda out_name: f"{out_name}_real.nii",
            )

    shelly = Shelly(out_name=["test_1", "test_2"])

    outputs = shelly(cache_root=tmp_path)
    # checking if the outputs is None
    assert outputs.out_list is None


def test_shellspec_formatter_1(tmp_path):
    """test the input callable 'formatter'."""

    def formatter_1(inputs):
        print("FORMATTER:", inputs)
        return f"-t [{inputs['in1']}, {inputs['in2']}]"

    def make_shelly(formatter):
        @shell.define
        class Shelly(shell.Task["Shelly.Outputs"]):
            executable = "exec"
            in1: str = shell.arg(
                argstr=None,
                help="""just a dummy name""",
            )
            in2: str = shell.arg(
                argstr=None,
                help="""just a dummy name""",
            )

            together: ty.List = shell.arg(
                default=attrs.Factory(list),
                help="""combines in1 and in2 into a list""",
                # When providing a formatter all other metadata options are discarded.
                formatter=formatter,
            )

            class Outputs(shell.Outputs):
                pass

        return Shelly

    Shelly = make_shelly(formatter=formatter_1)
    shelly = Shelly(in1="i1", in2="i2")
    assert shelly.cmdline == "exec -t [i1, i2]"

    # testing that the formatter can overwrite a provided value for together.
    shelly = Shelly(in1="i1", in2="i2", together=[1])
    assert shelly.cmdline == "exec -t [i1, i2]"

    # asking for specific inputs
    def formatter_2(in1, in2):
        print("FORMATTER:", in1, in2)
        return f"-t [{in1}, {in2}]"

    Shelly = make_shelly(formatter_2)

    shelly = Shelly(in1="i1", in2="i2")
    assert shelly.cmdline == "exec -t [i1, i2]"

    def formatter_3(in1, in3):
        print("FORMATTER:", in1, in3)
        return f"-t [{in1}, {in3}]"

    Shelly = make_shelly(formatter_3)

    shelly = Shelly(in1="i1", in2="i2")
    with pytest.raises(Exception) as excinfo:
        shelly.cmdline
    assert (
        "arguments of the formatter function from together has to be in inputs or be field, but in3 is used"
        == str(excinfo.value)
    )

    # checking if field value is accessible when None
    def formatter_4(field):
        assert isinstance(field, shell.arg)
        # formatter must return a string
        return "-t test"

    Shelly = make_shelly(formatter_4)

    shelly = Shelly(
        in1="i1",
        in2="i2",
        # together="-t test",
    )
    assert shelly.cmdline == "exec -t test"


def test_shellspec_formatter_splitter_2(tmp_path):
    """test the input callable 'formatter' when a splitter is used on an argument of the formatter."""

    # asking for specific inputs
    def formatter_1(in1, in2):
        return f"-t [{in1} {in2}]"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "executable"
        in1: str = shell.arg(help="in1")
        in2: str = shell.arg(help="in2")
        together: ty.List = shell.arg(
            help="""
            uses in1
            """,
            # When providing a formatter all other metadata options are discarded.
            formatter=formatter_1,
            sep=" ",
        )

        class Outputs(shell.Outputs):
            pass

    in1 = ["in11", "in12"]
    shelly = Shelly(in2="in2").split("in1", in1=in1)
    assert shelly is not None

    # results = shelly.cmdline
    # assert len(results) == 2
    # com_results = ["executable -t [in11 in2]", "executable -t [in12 in2]"]
    # for i, cr in enumerate(com_results):
    #     assert results[i] == cr


@no_win
def test_shellcommand_error_msg(tmp_path):
    script_path = Path(tmp_path) / "script.sh"

    with open(script_path, "w") as f:
        f.write(
            """#!/bin/bash
                echo "first line is ok, it prints '$1'"
                /command-that-doesnt-exist"""
        )

    os.chmod(
        script_path,
        mode=(
            stat.S_IRUSR
            | stat.S_IWUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IWGRP
            | stat.S_IROTH
        ),
    )

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):

        executable = str(script_path)

        in1: str = shell.arg(help="a dummy string", argstr="")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in1="hello")

    with pytest.raises(RuntimeError) as excinfo:
        shelly(cache_root=tmp_path)

    path_str = str(script_path)

    assert (
        str(excinfo.value)
        == f"""Error running 'main' job with ['{path_str}', 'hello']:

stderr:
{path_str}: line 3: /command-that-doesnt-exist: No such file or directory


stdout:
first line is ok, it prints 'hello'
"""
    )


@no_win
def test_shell_cmd(tmpdir):
    cmd = ["echo", "hail", "pydra"]

    # all args given as executable
    Shelly = shell.define(" ".join(cmd))
    shelly = Shelly()
    assert shelly.cmdline == " ".join(cmd)
    outputs = shelly()
    assert outputs.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    Shelly = shell.define(
        cmd[0], inputs=[shell.arg(name=a, default=a) for a in cmd[1:]]
    )
    shelly = Shelly()
    assert shelly.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    outputs = shelly()
    assert outputs.return_code == 0
    assert outputs.stdout == " ".join(cmd[1:]) + "\n"
