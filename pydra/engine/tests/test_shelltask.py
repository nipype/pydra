import attr
import typing as ty
import os, sys
import pytest
from pathlib import Path
import re
import stat

from ..task import ShellCommandTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import (
    ShellOutSpec,
    ShellSpec,
    SpecInfo,
    File,
    Directory,
    MultiInputFile,
    MultiOutputFile,
    MultiInputObj,
)
from .utils import result_no_submitter, result_submitter, use_validator, no_win

if sys.platform.startswith("win"):
    pytest.skip("SLURM not available in windows", allow_module_level=True)


@pytest.mark.flaky(reruns=2)  # when dask
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_1(plugin_dask_opt, results_function, tmpdir):
    """simple command, no arguments"""
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, cache_dir=tmpdir)
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin=plugin_dask_opt)
    assert Path(res.output.stdout.rstrip()) == shelly.output_dir
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_1_strip(plugin, results_function, tmpdir):
    """simple command, no arguments
    strip option to remove \n at the end os stdout
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, strip=True)
    shelly.cache_dir = tmpdir
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin)
    assert Path(res.output.stdout) == Path(shelly.output_dir)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2(plugin, results_function, tmpdir):
    """a command with arguments, cmd and args given as executable"""
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    shelly.cache_dir = tmpdir
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin)
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2a(plugin, results_function, tmpdir):
    """a command with arguments, using executable and args"""
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    shelly.cache_dir = tmpdir
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)

    res = results_function(shelly, plugin)
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2b(plugin, results_function, tmpdir):
    """a command with arguments, using  strings executable and args"""
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    shelly.cache_dir = tmpdir
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


# tests with State


@pytest.mark.flaky(reruns=2)
def test_shell_cmd_3(plugin_dask_opt, tmpdir):
    """commands without arguments
    splitter = executable
    """
    cmd = ["pwd", "whoami"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd).split("executable")
    shelly.cache_dir = tmpdir

    # assert shelly.cmdline == ["pwd", "whoami"]
    res = shelly(plugin=plugin_dask_opt)
    assert Path(res[0].output.stdout.rstrip()) == shelly.output_dir[0]

    if "USER" in os.environ:
        assert res[1].output.stdout == f"{os.environ['USER']}\n"
    else:
        assert res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


def test_shell_cmd_4(plugin, tmpdir):
    """a command with arguments, using executable and args
    splitter=args
    """
    cmd_exec = "echo"
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter="args"
    )
    shelly.cache_dir = tmpdir

    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    # assert shelly.cmdline == ["echo nipype", "echo pydra"]
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"

    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


def test_shell_cmd_5(plugin, tmpdir):
    """a command with arguments
    using splitter and combiner for args
    """
    cmd_exec = "echo"
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = (
        ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
        .split(splitter="args")
        .combine("args")
    )
    shelly.cache_dir = tmpdir

    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    # assert shelly.cmdline == ["echo nipype", "echo pydra"]
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"


def test_shell_cmd_6(plugin, tmpdir):
    """a command with arguments,
    outer splitter for executable and args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter=["executable", "args"]
    )
    shelly.cache_dir = tmpdir

    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]
    # assert shelly.cmdline == [
    #     "echo nipype",
    #     "echo pydra",
    #     "echo -n nipype",
    #     "echo -n pydra",
    # ]
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"
    assert res[2].output.stdout == "nipype"
    assert res[3].output.stdout == "pydra"

    assert (
        res[0].output.return_code
        == res[1].output.return_code
        == res[2].output.return_code
        == res[3].output.return_code
        == 0
    )
    assert (
        res[0].output.stderr
        == res[1].output.stderr
        == res[2].output.stderr
        == res[3].output.stderr
        == ""
    )


def test_shell_cmd_7(plugin, tmpdir):
    """a command with arguments,
    outer splitter for executable and args, and combiner=args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = (
        ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
        .split(splitter=["executable", "args"])
        .combine("args")
    )
    shelly.cache_dir = tmpdir

    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]

    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"

    assert res[1][0].output.stdout == "nipype"
    assert res[1][1].output.stdout == "pydra"


# tests with workflows


def test_wf_shell_cmd_1(plugin, tmpdir):
    """a workflow with two connected commands"""
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2"])
    wf.inputs.cmd1 = "pwd"
    wf.inputs.cmd2 = "ls"
    wf.add(ShellCommandTask(name="shelly_pwd", executable=wf.lzin.cmd1, strip=True))
    wf.add(
        ShellCommandTask(
            name="shelly_ls", executable=wf.lzin.cmd2, args=wf.shelly_pwd.lzout.stdout
        )
    )

    wf.set_output([("out", wf.shelly_ls.lzout.stdout)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert "_result.pklz" in res.output.out
    assert "_task.pklz" in res.output.out


# customised input spec


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_1(plugin, results_function, use_validator, tmpdir):
    """a command with executable, args and one command opt,
    using a customized input_spec to add the opt to the command
    in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = True
    cmd_args = "hello from pydra"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_n",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "argstr": "-n", "help_string": "option"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        args=cmd_args,
        opt_n=cmd_opt,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n 'hello from pydra'"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "hello from pydra"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_2(plugin, results_function, use_validator, tmpdir):
    """a command with executable, args and two command options,
    using a customized input_spec to add the opt to the command
    in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = True
    cmd_opt_hello = "HELLO"
    cmd_args = "from pydra"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_hello",
                attr.ib(
                    type=str,
                    metadata={"position": 3, "help_string": "todo", "argstr": ""},
                ),
            ),
            (
                "opt_n",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "help_string": "todo", "argstr": "-n"},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        args=cmd_args,
        opt_n=cmd_opt,
        opt_hello=cmd_opt_hello,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n HELLO 'from pydra'"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO from pydra"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        text=hello,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3a(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided
    using shorter syntax for input spec (no attr.ib)
    """
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                str,
                {"position": 1, "help_string": "text", "mandatory": True, "argstr": ""},
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        text=hello,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3b(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided after init"""
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )
    shelly.inputs.text = hello

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


def test_shell_cmd_inputspec_3c_exception(plugin, tmpdir):
    """mandatory field added to fields, value is not provided, so exception is raised"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )

    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "mandatory" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3c(plugin, results_function, tmpdir):
    """mandatory=False, so tasks runs fine even without the value"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default=None,
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": False,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hello",
                    metadata={"position": 1, "help_string": "text", "argstr": ""},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4a(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided
    using shorter syntax for input spec (no attr.ib)
    """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("text", str, "Hello", {"position": 1, "help_string": "text", "argstr": ""})
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4b(plugin, results_function, tmpdir):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hi",
                    metadata={"position": 1, "help_string": "text", "argstr": ""},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec, cache_dir=tmpdir
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hi"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hi\n"


def test_shell_cmd_inputspec_4c_exception(plugin):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hello",
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    with pytest.raises(Exception) as excinfo:
        shelly = ShellCommandTask(
            name="shelly", executable=cmd_exec, input_spec=my_input_spec
        )
    assert (
        str(excinfo.value)
        == "default value should not be set when the field is mandatory"
    )


def test_shell_cmd_inputspec_4d_exception(plugin):
    """mandatory field added to fields, value provided"""
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hello",
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "output_file_template": "exception",
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    with pytest.raises(Exception) as excinfo:
        shelly = ShellCommandTask(
            name="shelly", executable=cmd_exec, input_spec=my_input_spec
        )
    assert (
        str(excinfo.value)
        == "default value should not be set together with output_file_template"
    )


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_5_nosubm(plugin, results_function, tmpdir):
    """checking xor in metadata: task should work fine, since only one option is True"""
    cmd_exec = "ls"
    cmd_t = True
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_t",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 1,
                        "help_string": "opt t",
                        "argstr": "-t",
                        "xor": ["opt_S"],
                    },
                ),
            ),
            (
                "opt_S",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 2,
                        "help_string": "opt S",
                        "argstr": "-S",
                        "xor": ["opt_t"],
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        opt_t=cmd_t,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -t"
    res = results_function(shelly, plugin)


def test_shell_cmd_inputspec_5a_exception(plugin, tmpdir):
    """checking xor in metadata: both options are True, so the task raises exception"""
    cmd_exec = "ls"
    cmd_t = True
    cmd_S = True
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_t",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 1,
                        "help_string": "opt t",
                        "argstr": "-t",
                        "xor": ["opt_S"],
                    },
                ),
            ),
            (
                "opt_S",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 2,
                        "help_string": "opt S",
                        "argstr": "-S",
                        "xor": ["opt_t"],
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        opt_t=cmd_t,
        opt_S=cmd_S,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "is mutually exclusive" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_6(plugin, results_function, tmpdir):
    """checking requires in metadata:
    the required field is set in the init, so the task works fine
    """
    cmd_exec = "ls"
    cmd_l = True
    cmd_t = True
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_t",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 2,
                        "help_string": "opt t",
                        "argstr": "-t",
                        "requires": ["opt_l"],
                    },
                ),
            ),
            (
                "opt_l",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "help_string": "opt l", "argstr": "-l"},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        opt_t=cmd_t,
        opt_l=cmd_l,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    res = results_function(shelly, plugin)


def test_shell_cmd_inputspec_6a_exception(plugin):
    """checking requires in metadata:
    the required field is None, so the task works raises exception
    """
    cmd_exec = "ls"
    cmd_t = True
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_t",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 2,
                        "help_string": "opt t",
                        "argstr": "-t",
                        "requires": ["opt_l"],
                    },
                ),
            ),
            (
                "opt_l",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "help_string": "opt l", "argstr": "-l"},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, opt_t=cmd_t, input_spec=my_input_spec
    )
    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "requires" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_6b(plugin, results_function, tmpdir):
    """checking requires in metadata:
    the required field set after the init
    """
    cmd_exec = "ls"
    cmd_l = True
    cmd_t = True
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "opt_t",
                attr.ib(
                    type=bool,
                    metadata={
                        "position": 2,
                        "help_string": "opt t",
                        "argstr": "-t",
                        "requires": ["opt_l"],
                    },
                ),
            ),
            (
                "opt_l",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "help_string": "opt l", "argstr": "-l"},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        opt_t=cmd_t,
        # opt_l=cmd_l,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )
    shelly.inputs.opt_l = cmd_l
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    res = results_function(shelly, plugin)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7(plugin, results_function, tmpdir):
    """
    providing output name using input_spec,
    using name_tamplate in metadata
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()
    # checking if the file is created in a good place
    assert shelly.output_dir == res.output.out1.parent
    assert res.output.out1.name == "newfile_tmp.txt"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7a(plugin, results_function, tmpdir):
    """
    providing output name using input_spec,
    using name_tamplate in metadata
    and changing the output name for output_spec using output_field_name
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "output_field_name": "out1_changed",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1_changed.exists()
    # checking if the file is created in a good place
    assert shelly.output_dir == res.output.out1_changed.parent
    assert res.output.out1_changed.name == "newfile_tmp.txt"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7b(plugin, results_function, tmpdir):
    """
    providing new file and output name using input_spec,
    using name_template in metadata
    """
    cmd = "touch"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "newfile",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{newfile}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        newfile="newfile_tmp.txt",
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7c(plugin, results_function, tmpdir):
    """
    providing output name using input_spec,
    using name_tamplate with txt extension (extension from args should be removed
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}.txt",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()
    # checking if the file is created in a good place
    assert shelly.output_dir == res.output.out1.parent
    assert res.output.out1.name == "newfile_tmp.txt"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_8(plugin, results_function, tmpdir):
    """
    providing new file and output name using input_spec,
    adding additional string input field with argstr
    """
    cmd = "touch"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "newfile",
                attr.ib(
                    type=str,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "time",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "argstr": "-t",
                        "help_string": "time of modif.",
                    },
                ),
            ),
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{newfile}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        newfile="newfile_tmp.txt",
        time="02121010",
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_8a(plugin, results_function, tmpdir):
    """
    providing new file and output name using input_spec,
    adding additional string input field with argstr (argstr uses string formatting)
    """
    cmd = "touch"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "newfile",
                attr.ib(
                    type=str,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "time",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "argstr": "-t {time}",
                        "help_string": "time of modif.",
                    },
                ),
            ),
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{newfile}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        newfile="newfile_tmp.txt",
        time="02121010",
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_9(tmpdir, plugin, results_function):
    """
    providing output name using input_spec (output_file_template in metadata),
    the template has a suffix, the extension of the file will be moved to the end
    """
    cmd = "cp"
    file = tmpdir.mkdir("data_inp").join("file.txt")
    file.write("content\n")

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}_copy",
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        file_orig=file,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "file_copy.txt"
    # checking if it's created in a good place
    assert shelly.output_dir == res.output.file_copy.parent


@pytest.mark.parametrize("results_function", [result_no_submitter])
def test_shell_cmd_inputspec_9a(tmpdir, plugin, results_function):
    """
    providing output name using input_spec (output_file_template in metadata),
    the template has a suffix, the extension of the file will be moved to the end
    the change: input file has directory with a dot
    """
    cmd = "cp"
    file = tmpdir.mkdir("data.inp").join("file.txt")
    file.write("content\n")

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}_copy",
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, input_spec=my_input_spec, file_orig=file
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "file_copy.txt"
    # checking if it's created in a good place
    assert shelly.output_dir == res.output.file_copy.parent


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_9b(tmpdir, plugin, results_function):
    """
    providing output name using input_spec (output_file_template in metadata)
    and the keep_extension is set to False, so the extension is removed completely.
    """
    cmd = "cp"
    file = tmpdir.join("file.txt")
    file.write("content\n")

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}_copy",
                        "keep_extension": False,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        file_orig=file,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "file_copy"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_9c(tmpdir, plugin, results_function):
    """
    providing output name using input_spec (output_file_template in metadata)
    and the keep_extension is set to False, so the extension is removed completely,
    no suffix in the template.
    """
    cmd = "cp"
    file = tmpdir.join("file.txt")
    file.write("content\n")

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}",
                        "keep_extension": False,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        file_orig=file,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "file"
    assert res.output.file_copy.parent == shelly.output_dir


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_9d(tmpdir, plugin, results_function):
    """
    providing output name explicitly by manually setting value in input_spec
    (instead of using default provided byoutput_file_template in metadata)
    """
    cmd = "cp"
    file = tmpdir.mkdir("data_inp").join("file.txt")
    file.write("content\n")

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{file_orig}_copy",
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        file_orig=file,
        file_copy="my_file_copy.txt",
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "my_file_copy.txt"
    # checking if it's created in a good place
    assert shelly.output_dir == res.output.file_copy.parent


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_10(plugin, results_function, tmpdir):
    """using input_spec, providing list of files as an input"""

    file_1 = tmpdir.join("file_1.txt")
    file_2 = tmpdir.join("file_2.txt")
    with open(file_1, "w") as f:
        f.write("hello ")
    with open(file_2, "w") as f:
        f.write("from boston")

    cmd_exec = "cat"
    files_list = [file_1, file_2]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "files",
                attr.ib(
                    type=ty.List[File],
                    metadata={
                        "position": 1,
                        "argstr": "...",
                        "sep": " ",
                        "help_string": "list of files",
                        "mandatory": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        files=files_list,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    )

    assert shelly.inputs.executable == cmd_exec
    res = results_function(shelly, plugin)
    assert res.output.stdout == "hello from boston"


def test_shell_cmd_inputspec_10_err(tmpdir):
    """checking if the proper error is raised when broken symlink is provided
    as a input field with File as a type
    """

    file_1 = tmpdir.join("file_1.txt")
    with open(file_1, "w") as f:
        f.write("hello")
    file_2 = tmpdir.join("file_2.txt")

    # creating symlink and removing the original file
    os.symlink(file_1, file_2)
    os.remove(file_1)

    cmd_exec = "cat"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "files",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help_string": "a file",
                        "mandatory": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, files=file_2, input_spec=my_input_spec
    )
    shelly.cache_dir = tmpdir

    with pytest.raises(FileNotFoundError):
        res = shelly()


def test_shell_cmd_inputsspec_11():
    input_fields = [
        (
            "inputFiles",
            attr.ib(
                type=MultiInputFile,
                metadata={
                    "argstr": "...",
                    "help_string": "The list of input image files to be segmented.",
                },
            ),
        )
    ]

    output_fields = [
        (
            "outputFiles",
            attr.ib(
                type=MultiOutputFile,
                metadata={
                    "help_string": "Corrected Output Images: should specify the same number of images as inputVolume, if only one element is given, then it is used as a file pattern where %s is replaced by the imageVolumeType, and %d by the index list location.",
                    "output_file_template": "{inputFiles}",
                },
            ),
        )
    ]

    input_spec = SpecInfo(name="Input", fields=input_fields, bases=(ShellSpec,))
    output_spec = SpecInfo(name="Output", fields=output_fields, bases=(ShellOutSpec,))

    task = ShellCommandTask(
        name="echoMultiple",
        executable="touch",
        input_spec=input_spec,
        output_spec=output_spec,
    )
    wf = Workflow(name="wf", input_spec=["inputFiles"], inputFiles=["test1", "test2"])

    task.inputs.inputFiles = wf.lzin.inputFiles

    wf.add(task)
    wf.set_output([("out", wf.echoMultiple.lzout.outputFiles)])

    with Submitter(plugin="cf") as sub:
        sub(wf)
    result = wf.result()

    for out_file in result.output.out:
        assert out_file.name == "test1" or out_file.name == "test2"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_12(tmpdir, plugin, results_function):
    """
    providing output name using input_spec
    output_file_template is provided as a function that returns
    various templates depending on the values of inputs fields
    """
    cmd = "cp"
    file = tmpdir.mkdir("data_inp").join("file.txt")
    file.write("content\n")

    def template_function(inputs):
        if inputs.number % 2 == 0:
            return "{file_orig}_even"
        else:
            return "{file_orig}_odd"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file_orig",
                attr.ib(
                    type=File,
                    metadata={"position": 2, "help_string": "new file", "argstr": ""},
                ),
            ),
            (
                "number",
                attr.ib(
                    type=int,
                    metadata={"help_string": "a number", "mandatory": True},
                ),
            ),
            (
                "file_copy",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": template_function,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        file_orig=file,
        number=2,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.file_copy.exists()
    assert res.output.file_copy.name == "file_even.txt"
    # checking if it's created in a good place
    assert shelly.output_dir == res.output.file_copy.parent


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1(plugin, results_function, tmpdir):
    """shelltask changes a file in place,
    adding copyfile=True to the file-input from input_spec
    hardlink or copy in the output_dir should be created
    """
    file = tmpdir.join("file_pydra.txt")
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help_string": "orig file",
                        "mandatory": True,
                        "copyfile": True,
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  copied, and than it is changed in place
    assert res.output.out_file.parent == shelly.output_dir
    with open(res.output.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1a(plugin, results_function, tmpdir):
    """shelltask changes a file in place,
    adding copyfile=False to the File-input from input_spec
    hardlink or softlink in the output_dir is created
    """
    file = tmpdir.join("file_pydra.txt")
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help_string": "orig file",
                        "mandatory": True,
                        "copyfile": False,
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is uses a soft link, but it creates and an extra copy before modifying
    assert res.output.out_file.parent == shelly.output_dir

    assert res.output.out_file.parent.joinpath(res.output.out_file.name + "s").exists()
    with open(res.output.out_file) as f:
        assert "hi from pydra\n" == f.read()
    # the file is uses a soft link, but it creates and an extra copy
    # it might depend on the OS
    linked_file_copy = res.output.out_file.parent.joinpath(
        res.output.out_file.name + "s"
    )
    if linked_file_copy.exists():
        with open(linked_file_copy) as f:
            assert "hello from pydra\n" == f.read()

    # the original file is unchanged
    with open(file) as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.xfail(
    reason="not sure if we want to support input overwrite,"
    "if we allow for this orig_file is changing, so does checksum,"
    " and the results can't be found"
)
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1b(plugin, results_function, tmpdir):
    """shelltask changes a file in place,
    copyfile is None for the file-input, so original filed is changed
    """
    file = tmpdir.join("file_pydra.txt")
    with open(file, "w") as f:
        f.write("hello from pydra\n")

    cmd = ["sed", "-is", "s/hello/hi/"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help_string": "orig file",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=str(file),
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  not copied, it is changed in place
    assert res.output.out_file == file
    with open(res.output.out_file) as f:
        assert "hi from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_1(plugin, results_function, tmpdir):
    """adding state to the input from input_spec"""
    cmd_exec = "echo"
    hello = ["HELLO", "hi"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "text",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        text=hello,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    ).split("text")
    assert shelly.inputs.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "HELLO\n"
    assert res[1].output.stdout == "hi\n"


def test_shell_cmd_inputspec_typeval_1(use_validator):
    """customized input_spec with a type that doesn't match the value
    - raise an exception
    """
    cmd_exec = "echo"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=int,
                    metadata={"position": 1, "argstr": "", "help_string": "text"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    with pytest.raises(TypeError):
        shelly = ShellCommandTask(
            executable=cmd_exec, text="hello", input_spec=my_input_spec
        )


def test_shell_cmd_inputspec_typeval_2(use_validator):
    """customized input_spec (shorter syntax) with a type that doesn't match the value
    - raise an exception
    """
    cmd_exec = "echo"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("text", int, {"position": 1, "argstr": "", "help_string": "text"})],
        bases=(ShellSpec,),
    )

    with pytest.raises(TypeError):
        shelly = ShellCommandTask(
            executable=cmd_exec, text="hello", input_spec=my_input_spec
        )


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_1a(plugin, results_function, tmpdir):
    """adding state to the input from input_spec
    using shorter syntax for input_spec (without default)
    """
    cmd_exec = "echo"
    hello = ["HELLO", "hi"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                str,
                {"position": 1, "help_string": "text", "mandatory": True, "argstr": ""},
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        text=hello,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    ).split("text")
    assert shelly.inputs.executable == cmd_exec

    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "HELLO\n"
    assert res[1].output.stdout == "hi\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_2(plugin, results_function, tmpdir):
    """
    adding splitter to input that is used in the output_file_tamplate
    """
    cmd = "touch"
    args = ["newfile_1.txt", "newfile_2.txt"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    ).split("args")

    res = results_function(shelly, plugin)
    for i in range(len(args)):
        assert res[i].output.stdout == ""
        assert res[i].output.out1.exists()
        assert res[i].output.out1.parent == shelly.output_dir[i]


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_3(plugin, results_function, tmpdir):
    """adding state to the File-input from input_spec"""

    file_1 = tmpdir.join("file_pydra.txt")
    file_2 = tmpdir.join("file_nice.txt")
    with open(file_1, "w") as f:
        f.write("hello from pydra")
    with open(file_2, "w") as f:
        f.write("have a nice one")

    cmd_exec = "cat"
    files = [file_1, file_2]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "files",
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        file=files,
        input_spec=my_input_spec,
        cache_dir=tmpdir,
    ).split("file")

    assert shelly.inputs.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_state_1(plugin, results_function, tmpdir):
    """adding state to the File-input from input_spec"""

    file1 = tmpdir.join("file1.txt")
    with open(file1, "w") as f:
        f.write("hello from pydra\n")

    file2 = tmpdir.join("file2.txt")
    with open(file2, "w") as f:
        f.write("hello world\n")

    files = [str(file1), str(file2)]
    cmd = ["sed", "-is", "s/hello/hi/"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help_string": "orig file",
                        "mandatory": True,
                        "copyfile": True,
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{orig_file}",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        orig_file=files,
        cache_dir=tmpdir,
    ).split("orig_file")

    txt_l = ["from pydra", "world"]
    res_l = results_function(shelly, plugin)
    for i, res in enumerate(res_l):
        assert res.output.stdout == ""
        assert res.output.out_file.exists()
        # the file is  copied, and than it is changed in place
        assert res.output.out_file.parent == shelly.output_dir[i]
        with open(res.output.out_file) as f:
            assert f"hi {txt_l[i]}\n" == f.read()
        # the original file is unchanged
        with open(files[i]) as f:
            assert f"hello {txt_l[i]}\n" == f.read()


# customised input_spec in Workflow


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_shell_cmd_2(plugin_dask_opt, tmpdir):
    """a workflow with input with defined output_file_template (str)
    that requires wf.lzin
    """
    wf = Workflow(name="wf", input_spec=["cmd", "args"])

    wf.inputs.cmd = "touch"
    wf.inputs.args = "newfile.txt"
    wf.cache_dir = tmpdir

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly",
            input_spec=my_input_spec,
            executable=wf.lzin.cmd,
            args=wf.lzin.args,
        )
    )

    wf.set_output([("out_f", wf.shelly.lzout.out1), ("out", wf.shelly.lzout.stdout)])

    with Submitter(plugin=plugin_dask_opt) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == ""
    assert res.output.out_f.exists()
    assert res.output.out_f.parent == wf.output_dir


def test_wf_shell_cmd_2a(plugin, tmpdir):
    """a workflow with input with defined output_file_template (tuple)
    that requires wf.lzin
    """
    wf = Workflow(name="wf", input_spec=["cmd", "args"])

    wf.inputs.cmd = "touch"
    wf.inputs.args = "newfile.txt"
    wf.cache_dir = tmpdir

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly",
            input_spec=my_input_spec,
            executable=wf.lzin.cmd,
            args=wf.lzin.args,
        )
    )

    wf.set_output([("out_f", wf.shelly.lzout.out1), ("out", wf.shelly.lzout.stdout)])

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out == ""
    assert res.output.out_f.exists()


def test_wf_shell_cmd_3(plugin, tmpdir):
    """a workflow with 2 tasks,
    first one has input with output_file_template (str, uses wf.lzin),
    that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = "newfile.txt"
    wf.cache_dir = tmpdir

    my_input_spec1 = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    my_input_spec2 = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "output_file_template": "{orig_file}_copy",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly1",
            input_spec=my_input_spec1,
            executable=wf.lzin.cmd1,
            args=wf.lzin.args,
        )
    )
    wf.add(
        ShellCommandTask(
            name="shelly2",
            input_spec=my_input_spec2,
            executable=wf.lzin.cmd2,
            orig_file=wf.shelly1.lzout.file,
        )
    )

    wf.set_output(
        [
            ("touch_file", wf.shelly1.lzout.file),
            ("out1", wf.shelly1.lzout.stdout),
            ("cp_file", wf.shelly2.lzout.out_file),
            ("out2", wf.shelly2.lzout.stdout),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out1 == ""
    assert res.output.touch_file.exists()
    assert res.output.touch_file.parent == wf.output_dir
    assert res.output.out2 == ""
    assert res.output.cp_file.exists()
    assert res.output.cp_file.parent == wf.output_dir


def test_wf_shell_cmd_3a(plugin, tmpdir):
    """a workflow with 2 tasks,
    first one has input with output_file_template (str, uses wf.lzin),
    that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = "newfile.txt"
    wf.cache_dir = tmpdir

    my_input_spec1 = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    my_input_spec2 = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "output_file_template": "{orig_file}_cp",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly1",
            input_spec=my_input_spec1,
            executable=wf.lzin.cmd1,
            args=wf.lzin.args,
        )
    )
    wf.add(
        ShellCommandTask(
            name="shelly2",
            input_spec=my_input_spec2,
            executable=wf.lzin.cmd2,
            orig_file=wf.shelly1.lzout.file,
        )
    )

    wf.set_output(
        [
            ("touch_file", wf.shelly1.lzout.file),
            ("out1", wf.shelly1.lzout.stdout),
            ("cp_file", wf.shelly2.lzout.out_file),
            ("out2", wf.shelly2.lzout.stdout),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out1 == ""
    assert res.output.touch_file.exists()
    assert res.output.out2 == ""
    assert res.output.cp_file.exists()


def test_wf_shell_cmd_state_1(plugin):
    """a workflow with 2 tasks and splitter on the wf level,
    first one has input with output_file_template (str, uses wf.lzin),
    that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"]).split("args")

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = ["newfile_1.txt", "newfile_2.txt"]

    my_input_spec1 = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    my_input_spec2 = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "output_file_template": "{orig_file}_copy",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly1",
            input_spec=my_input_spec1,
            executable=wf.lzin.cmd1,
            args=wf.lzin.args,
        )
    )
    wf.add(
        ShellCommandTask(
            name="shelly2",
            input_spec=my_input_spec2,
            executable=wf.lzin.cmd2,
            orig_file=wf.shelly1.lzout.file,
        )
    )

    wf.set_output(
        [
            ("touch_file", wf.shelly1.lzout.file),
            ("out1", wf.shelly1.lzout.stdout),
            ("cp_file", wf.shelly2.lzout.out_file),
            ("out2", wf.shelly2.lzout.stdout),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res_l = wf.result()
    for i, res in enumerate(res_l):
        assert res.output.out1 == ""
        assert res.output.touch_file.exists()
        assert res.output.touch_file.parent == wf.output_dir[i]
        assert res.output.out2 == ""
        assert res.output.cp_file.exists()
        assert res.output.cp_file.parent == wf.output_dir[i]


def test_wf_shell_cmd_ndst_1(plugin, tmpdir):
    """a workflow with 2 tasks and a splitter on the node level,
    first one has input with output_file_template (str, uses wf.lzin),
    that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = ["newfile_1.txt", "newfile_2.txt"]
    wf.cache_dir = tmpdir

    my_input_spec1 = SpecInfo(
        name="Input",
        fields=[
            (
                "file",
                attr.ib(
                    type=str,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    my_input_spec2 = SpecInfo(
        name="Input",
        fields=[
            (
                "orig_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "output file",
                        "argstr": "",
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "output_file_template": "{orig_file}_copy",
                        "help_string": "output file",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    wf.add(
        ShellCommandTask(
            name="shelly1",
            input_spec=my_input_spec1,
            executable=wf.lzin.cmd1,
            args=wf.lzin.args,
        ).split("args")
    )
    wf.add(
        ShellCommandTask(
            name="shelly2",
            input_spec=my_input_spec2,
            executable=wf.lzin.cmd2,
            orig_file=wf.shelly1.lzout.file,
        )
    )

    wf.set_output(
        [
            ("touch_file", wf.shelly1.lzout.file),
            ("out1", wf.shelly1.lzout.stdout),
            ("cp_file", wf.shelly2.lzout.out_file),
            ("out2", wf.shelly2.lzout.stdout),
        ]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.out1 == ["", ""]
    assert all([file.exists() for file in res.output.touch_file])
    assert res.output.out2 == ["", ""]
    assert all([file.exists() for file in res.output.cp_file])


# customised output spec


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_1(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_1a(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", attr.ib(type=File, default="newfile_tmp.txt"))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


def test_shell_cmd_outputspec_1b_exception(plugin, tmpdir):
    """
    customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp_.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    with pytest.raises(Exception) as exinfo:
        with Submitter(plugin=plugin) as sub:
            shelly(submitter=sub)
    assert "does not exist" in str(exinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_2(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_*.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


def test_shell_cmd_outputspec_2a_exception(plugin, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_*K.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            shelly(submitter=sub)
    assert "no file matches" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_3(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default, should collect two files
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_*.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_4(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default (in the directory name)
    """
    cmd = ["mkdir", "tmp1", ";", "touch", "tmp1/newfile.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "tmp*/newfile.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_4a(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a wildcard in default (in the directory name), should collect two files
    """
    cmd = [
        "mkdir",
        "tmp1",
        "tmp2",
        ";",
        "touch",
        "tmp1/newfile.txt",
        "tmp2/newfile.txt",
    ]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "tmp*/newfile.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_5(plugin, results_function, tmpdir):
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    and uses output_dir and the glob function
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(field, output_dir):
        if field.name == "newfile":
            return list(Path(output_dir).expanduser().glob("newfile*.txt"))

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile",
                attr.ib(type=MultiOutputFile, metadata={"callable": gather_output}),
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, output_spec=my_output_spec, cache_dir=tmpdir
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])
    assert (
        shelly.output_names
        == shelly.generated_output_names
        == ["return_code", "stdout", "stderr", "newfile"]
    )


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_5a(plugin, results_function):
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    and uses output_dir and inputs element
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(executable, output_dir):
        files = executable[1:]
        return [Path(output_dir) / file for file in files]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", attr.ib(type=File, metadata={"callable": gather_output}))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


def test_shell_cmd_outputspec_5b_error():
    """
    customised output_spec, adding files to the output,
    using a function to collect output, the function is saved in the field metadata
    with an argument that is not part of the inputs - error is raised
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(executable, output_dir, ble):
        files = executable[1:]
        return [Path(output_dir) / file for file in files]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", attr.ib(type=File, metadata={"callable": gather_output}))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)
    with pytest.raises(AttributeError, match="ble"):
        shelly()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_6(plugin, results_function, tmpdir):
    """
    providing output name by providing output_file_template
    (similar to the previous example, but not touching input_spec)
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out1",
                attr.ib(
                    type=File,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        output_spec=my_output_spec,
        cache_dir=tmpdir,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


def test_shell_cmd_outputspec_6a():
    """
    providing output name by providing output_file_template
    (using shorter syntax)
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out1",
                File,
                {"output_file_template": "{args}", "help_string": "output file"},
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_7(tmpdir, plugin, results_function):
    """
    providing output with output_file_name and using MultiOutputFile as a type.
    the input field used in the template is a MultiInputObj, so it can be and is a list
    """
    file = tmpdir.join("script.sh")
    file.write(f'for var in "$@"; do touch file"$var".txt; done')

    cmd = "bash"
    new_files_id = ["1", "2", "3"]

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "script",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "script file",
                        "mandatory": True,
                        "position": 1,
                        "argstr": "",
                    },
                ),
            ),
            (
                "files_id",
                attr.ib(
                    type=MultiInputObj,
                    metadata={
                        "position": 2,
                        "argstr": "...",
                        "sep": " ",
                        "help_string": "list of name indices",
                        "mandatory": True,
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "new_files",
                attr.ib(
                    type=MultiOutputFile,
                    metadata={
                        "output_file_template": "file{files_id}.txt",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
        script=file,
        files_id=new_files_id,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    for file in res.output.new_files:
        assert file.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_7a(tmpdir, plugin, results_function):
    """
    providing output with output_file_name and using MultiOutputFile as a type.
    the input field used in the template is a MultiInputObj, but a single element is used
    """
    file = tmpdir.join("script.sh")
    file.write(f'for var in "$@"; do touch file"$var".txt; done')

    cmd = "bash"
    new_files_id = "1"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "script",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "script file",
                        "mandatory": True,
                        "position": 1,
                        "argstr": "",
                    },
                ),
            ),
            (
                "files_id",
                attr.ib(
                    type=MultiInputObj,
                    metadata={
                        "position": 2,
                        "argstr": "...",
                        "sep": " ",
                        "help_string": "list of name indices",
                        "mandatory": True,
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "new_files",
                attr.ib(
                    type=MultiOutputFile,
                    metadata={
                        "output_file_template": "file{files_id}.txt",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
        script=file,
        files_id=new_files_id,
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.new_files.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_8a(tmpdir, plugin, results_function):
    """
    customised output_spec, adding int and str to the output,
    requiring two callables with parameters stdout and stderr
    """
    cmd = "echo"
    args = ["newfile_1.txt", "newfile_2.txt"]

    def get_file_index(stdout):
        stdout = re.sub(r".*_", "", stdout)
        stdout = re.sub(r".txt", "", stdout)
        print(stdout)
        return int(stdout)

    def get_stderr(stderr):
        return f"stderr: {stderr}"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out1",
                attr.ib(
                    type=File,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            ),
            (
                "out_file_index",
                attr.ib(
                    type=int,
                    metadata={"help_string": "output file", "callable": get_file_index},
                ),
            ),
            (
                "stderr_field",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "The standard error output",
                        "callable": get_stderr,
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    ).split("args")

    results = results_function(shelly, plugin)
    for index, res in enumerate(results):
        assert res.output.out_file_index == index + 1
        assert res.output.stderr_field == f"stderr: {res.output.stderr}"


def test_shell_cmd_outputspec_8b_error():
    """
    customised output_spec, adding Int to the output,
    requiring a function to collect output
    """
    cmd = "echo"
    args = ["newfile_1.txt", "newfile_2.txt"]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out",
                attr.ib(
                    type=int, metadata={"help_string": "output file", "value": "val"}
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    ).split("args")
    with pytest.raises(Exception) as e:
        shelly()
    assert "has to have a callable" in str(e.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_8c(tmpdir, plugin, results_function):
    """
    customised output_spec, adding Directory to the output named by args
    """

    def get_lowest_directory(directory_path):
        return str(directory_path).replace(str(Path(directory_path).parents[0]), "")

    cmd = "mkdir"
    args = [f"{tmpdir}/dir1", f"{tmpdir}/dir2"]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "resultsDir",
                attr.ib(
                    type=Directory,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        output_spec=my_output_spec,
        resultsDir="outdir",
    ).split("args")

    res = results_function(shelly, plugin)
    for index, arg_dir in enumerate(args):
        assert Path(Path(tmpdir) / Path(arg_dir)).exists() == True
        assert get_lowest_directory(arg_dir) == f"/dir{index+1}"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_8d(tmpdir, plugin, results_function):
    """
    customised output_spec, adding Directory to the output named by input spec
    """
    # For /tmp/some_dict/test this function returns "/test"
    def get_lowest_directory(directory_path):
        return str(directory_path).replace(str(Path(directory_path).parents[0]), "")

    cmd = "mkdir"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "resultsDir",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "new directory",
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "resultsDir",
                attr.ib(
                    type=Directory,
                    metadata={
                        "output_file_template": "{resultsDir}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name=cmd,
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
        cache_dir=tmpdir,
        resultsDir="test",  # Path(tmpdir) / "test" TODO: Not working without absolute path support
    )
    assert (
        shelly.output_names
        == shelly.generated_output_names
        == ["return_code", "stdout", "stderr", "resultsDir"]
    )
    res = results_function(shelly, plugin)
    print("Cache_dirr:", shelly.cache_dir)
    assert (shelly.output_dir / Path("test")).exists() == True
    assert get_lowest_directory(res.output.resultsDir) == get_lowest_directory(
        shelly.output_dir / Path("test")
    )


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_state_outputspec_1(plugin, results_function, tmpdir):
    """
    providing output name by providing output_file_template
    splitter for a field that is used in the template
    """
    cmd = "touch"
    args = ["newfile_1.txt", "newfile_2.txt"]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out1",
                attr.ib(
                    type=File,
                    metadata={
                        "output_file_template": "{args}",
                        "help_string": "output file",
                    },
                ),
            )
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        args=args,
        output_spec=my_output_spec,
        cache_dir=tmpdir,
    ).split("args")

    res = results_function(shelly, plugin)
    for i in range(len(args)):
        assert res[i].output.stdout == ""
        assert res[i].output.out1.exists()


# customised output_spec for tasks in workflows


def test_shell_cmd_outputspec_wf_1(plugin, tmpdir):
    """
    customised output_spec for tasks within a Workflow,
    adding files to the output, providing specific pathname
    """

    cmd = ["touch", "newfile_tmp.txt"]
    wf = Workflow(name="wf", input_spec=["cmd"])
    wf.inputs.cmd = cmd
    wf.cache_dir = tmpdir

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    wf.add(
        ShellCommandTask(
            name="shelly", executable=wf.lzin.cmd, output_spec=my_output_spec
        )
    )
    wf.set_output(
        [("stdout", wf.shelly.lzout.stdout), ("newfile", wf.shelly.lzout.newfile)]
    )

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()
    # checking if the file was copied to the wf dir
    assert res.output.newfile.parent == wf.output_dir


def test_shell_cmd_inputspec_outputspec_1():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in templates
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {"output_file_template": "{file1}", "help_string": "newfile 1"},
            ),
            (
                "newfile2",
                File,
                {"output_file_template": "{file2}", "help_string": "newfile 2"},
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.file2 = "new_file_2.txt"

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    assert res.output.newfile2.exists()


def test_shell_cmd_inputspec_outputspec_1a():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in templates,
    file2 is used in a template for newfile2, but it is not provided, so newfile2 is set to NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {"output_file_template": "{file1}", "help_string": "newfile 1"},
            ),
            (
                "newfile2",
                File,
                {"output_file_template": "{file2}", "help_string": "newfile 2"},
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    # newfile2 is not created, since file2 is not provided
    assert res.output.newfile2 is attr.NOTHING


def test_shell_cmd_inputspec_outputspec_2():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    "requires": ["file1"],
                },
            ),
            (
                "newfile2",
                File,
                {
                    "output_file_template": "{file2}",
                    "help_string": "newfile 1",
                    "requires": ["file1", "file2"],
                },
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.file2 = "new_file_2.txt"
    # all fields from output_spec should be in output_names and generated_output_names
    assert (
        shelly.output_names
        == shelly.generated_output_names
        == ["return_code", "stdout", "stderr", "newfile1", "newfile2"]
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    assert res.output.newfile2.exists()


def test_shell_cmd_inputspec_outputspec_2a():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    "requires": ["file1"],
                },
            ),
            (
                "newfile2",
                File,
                {
                    "output_file_template": "{file2}",
                    "help_string": "newfile 1",
                    "requires": ["file1", "file2"],
                },
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    # generated_output_names should know that newfile2 will not be generated
    assert shelly.output_names == [
        "return_code",
        "stdout",
        "stderr",
        "newfile1",
        "newfile2",
    ]
    assert shelly.generated_output_names == [
        "return_code",
        "stdout",
        "stderr",
        "newfile1",
    ]

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    assert res.output.newfile2 is attr.NOTHING


def test_shell_cmd_inputspec_outputspec_3():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input that is not in the template, but in the requires field,
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
            ("additional_inp", str, {"help_string": "additional inp"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {"output_file_template": "{file1}", "help_string": "newfile 1"},
            ),
            (
                "newfile2",
                File,
                {
                    "output_file_template": "{file2}",
                    "help_string": "newfile 1",
                    "requires": ["file1", "additional_inp"],
                },
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.file2 = "new_file_2.txt"
    shelly.inputs.additional_inp = 2

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    assert res.output.newfile2.exists()


def test_shell_cmd_inputspec_outputspec_3a():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input that is not in the template, but in the requires field,
    the additional input not provided, so the output is NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            (
                "file2",
                str,
                {"help_string": "2nd creadted file", "argstr": "", "position": 2},
            ),
            ("additional_inp", str, {"help_string": "additional inp"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {"output_file_template": "{file1}", "help_string": "newfile 1"},
            ),
            (
                "newfile2",
                File,
                {
                    "output_file_template": "{file2}",
                    "help_string": "newfile 1",
                    "requires": ["file1", "additional_inp"],
                },
            ),
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.file2 = "new_file_2.txt"
    # generated_output_names should know that newfile2 will not be generated
    assert shelly.output_names == [
        "return_code",
        "stdout",
        "stderr",
        "newfile1",
        "newfile2",
    ]
    assert shelly.generated_output_names == [
        "return_code",
        "stdout",
        "stderr",
        "newfile1",
    ]

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()
    # additional input not provided so no newfile2 set (even if the file was created)
    assert res.output.newfile2 is attr.NOTHING


def test_shell_cmd_inputspec_outputspec_4():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input to the requires together with a list of the allowed values,
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp", str, {"help_string": "additional inp"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    "requires": ["file1", ("additional_inp", [2, 3])],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.additional_inp = 2
    # generated_output_names should be the same as output_names
    assert (
        shelly.output_names
        == shelly.generated_output_names
        == ["return_code", "stdout", "stderr", "newfile1"]
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()


def test_shell_cmd_inputspec_outputspec_4a():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires filed
    adding one additional input to the requires together with a list of the allowed values,
    the input is set to a value that is not in the list, so output is NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp", str, {"help_string": "additional inp"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    "requires": ["file1", ("additional_inp", [2, 3])],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    # the value is not in the list from requires
    shelly.inputs.additional_inp = 1

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1 is attr.NOTHING


def test_shell_cmd_inputspec_outputspec_5():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    the firs element of the requires list has all the fields set
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp_A", str, {"help_string": "additional inp A"}),
            ("additional_inp_B", str, {"help_string": "additional inp B"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    # requires is a list of list so it's treated as el[0] OR el[1] OR...
                    "requires": [
                        ["file1", "additional_inp_A"],
                        ["file1", "additional_inp_B"],
                    ],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.additional_inp_A = 2

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()


def test_shell_cmd_inputspec_outputspec_5a():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    the second element of the requires list (i.e. additional_inp_B) has all the fields set
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp_A", str, {"help_string": "additional inp A"}),
            ("additional_inp_B", str, {"help_string": "additional inp B"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    # requires is a list of list so it's treated as el[0] OR el[1] OR...
                    "requires": [
                        ["file1", "additional_inp_A"],
                        ["file1", "additional_inp_B"],
                    ],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"
    shelly.inputs.additional_inp_B = 2

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile1.exists()


def test_shell_cmd_inputspec_outputspec_5b():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires is a list of list so it is treated as OR list (i.e. el[0] OR el[1] OR...)
    neither of the list from requirements has all the fields set, so the output is NOTHING
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp_A", str, {"help_string": "additional inp A"}),
            ("additional_inp_B", str, {"help_string": "additional inp B"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    # requires is a list of list so it's treated as el[0] OR el[1] OR...
                    "requires": [
                        ["file1", "additional_inp_A"],
                        ["file1", "additional_inp_B"],
                    ],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"

    res = shelly()
    assert res.output.stdout == ""
    # neither additional_inp_A nor additional_inp_B is set, so newfile1 is NOTHING
    assert res.output.newfile1 is attr.NOTHING


def test_shell_cmd_inputspec_outputspec_6_except():
    """
    customised input_spec and output_spec, output_spec uses input_spec fields in the requires
    requires has invalid syntax - exception is raised
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "file1",
                str,
                {"help_string": "1st creadted file", "argstr": "", "position": 1},
            ),
            ("additional_inp_A", str, {"help_string": "additional inp A"}),
        ],
        bases=(ShellSpec,),
    )

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "newfile1",
                File,
                {
                    "output_file_template": "{file1}",
                    "help_string": "newfile 1",
                    # requires has invalid syntax
                    "requires": [["file1", "additional_inp_A"], "file1"],
                },
            )
        ],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd,
        input_spec=my_input_spec,
        output_spec=my_output_spec,
    )
    shelly.inputs.file1 = "new_file_1.txt"

    with pytest.raises(Exception, match="requires field can be"):
        res = shelly()


def no_fsl():
    if "FSLDIR" not in os.environ:
        return True


@pytest.mark.skipif(no_fsl(), reason="fsl is not installed")
def test_fsl():
    """mandatory field added to fields, value provided"""

    _xor_inputs = [
        "functional",
        "reduce_bias",
        "robust",
        "padding",
        "remove_eyes",
        "surfaces",
        "t2_guided",
    ]

    def change_name(file):
        name, ext = os.path.splitext(file)
        return f"{name}_brain.{ext}"

    bet_input_spec = SpecInfo(
        name="Input",
        # TODO: change the position??
        fields=[
            (
                "in_file",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "input file to skull strip",
                        "position": 1,
                        "mandatory": True,
                        "argstr": "",
                    },
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "name of output skull stripped image",
                        "position": 2,
                        "argstr": "",
                        "output_file_template": "{in_file}_brain",
                    },
                ),
            ),
            (
                "outline",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "create surface outline image",
                        "argstr": "-o",
                    },
                ),
            ),
            (
                "mask",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "create binary mask image",
                        "argstr": "-m",
                    },
                ),
            ),
            (
                "skull",
                attr.ib(
                    type=bool,
                    metadata={"help_string": "create skull image", "argstr": "-s"},
                ),
            ),
            (
                "no_output",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "Don't generate segmented output",
                        "argstr": "-n",
                    },
                ),
            ),
            (
                "frac",
                attr.ib(
                    type=float,
                    metadata={
                        "help_string": "fractional intensity threshold",
                        "argstr": "-f",
                    },
                ),
            ),
            (
                "vertical_gradient",
                attr.ib(
                    type=float,
                    metadata={
                        "help_string": "vertical gradient in fractional intensity threshold (-1, 1)",
                        "argstr": "-g",
                        "allowed_values": {"min_val": -1, "max_val": 1},
                    },
                ),
            ),
            (
                "radius",
                attr.ib(
                    type=int, metadata={"argstr": "-r", "help_string": "head radius"}
                ),
            ),
            (
                "center",
                attr.ib(
                    type=ty.List[int],
                    metadata={
                        "help_string": "center of gravity in voxels",
                        "argstr": "-c",
                        "allowed_values": {"min_value": 0, "max_value": 3},
                    },
                ),
            ),
            (
                "threshold",
                attr.ib(
                    type=bool,
                    metadata={
                        "argstr": "-t",
                        "help_string": "apply thresholding to segmented brain image and mask",
                    },
                ),
            ),
            (
                "mesh",
                attr.ib(
                    type=bool,
                    metadata={
                        "argstr": "-e",
                        "help_string": "generate a vtk mesh brain surface",
                    },
                ),
            ),
            (
                "robust",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "robust brain centre estimation (iterates BET several times)",
                        "argstr": "-R",
                        "xor": _xor_inputs,
                    },
                ),
            ),
            (
                "padding",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "improve BET if FOV is very small in Z (by temporarily padding end slices",
                        "argstr": "-Z",
                        "xor": _xor_inputs,
                    },
                ),
            ),
            (
                "remove_eyes",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "eye & optic nerve cleanup (can be useful in SIENA)",
                        "argstr": "-S",
                        "xor": _xor_inputs,
                    },
                ),
            ),
            (
                "surfaces",
                attr.ib(
                    type=bool,
                    metadata={
                        "help_string": "run bet2 and then betsurf to get additional skull and scalp surfaces (includes registrations)",
                        "argstr": "-A",
                        "xor": _xor_inputs,
                    },
                ),
            ),
            (
                "t2_guided",
                attr.ib(
                    type=ty.Union[File, str],
                    metadata={
                        "help_string": "as with creating surfaces, when also feeding in non-brain-extracted T2 (includes registrations)",
                        "argstr": "-A2",
                        "xor": _xor_inputs,
                    },
                ),
            ),
            (
                "functional",
                attr.ib(
                    type=bool,
                    metadata={
                        "argstr": "-F",
                        "xor": _xor_inputs,
                        "help_string": "apply to 4D fMRI data",
                    },
                ),
            ),
            (
                "reduce_bias",
                attr.ib(
                    type=bool,
                    metadata={
                        "argstr": "-B",
                        "xor": _xor_inputs,
                        "help_string": "bias field and neck cleanup",
                    },
                ),
            )
            # ("number_classes", int, attr.ib(metadata={"help_string": 'number of tissue-type classes', "argstr": '-n',
            #                                            "allowed_values": {"min_val": 1, "max_val": 10}})),
            # ("output_biasfield", bool,
            #  attr.ib(metadata={"help_string": 'output estimated bias field', "argstr": '-b'})),
            # ("output_biascorrected", bool,
            #  attr.ib(metadata={"help_string": 'output restored image (bias-corrected image)', "argstr": '-B'})),
        ],
        bases=(ShellSpec,),
    )

    # TODO: not sure why this has to be string
    in_file = Path(__file__).parent / "data_tests" / "test.nii.gz"

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="bet_task", executable="bet", in_file=in_file, input_spec=bet_input_spec
    )
    out_file = shelly.output_dir / "test_brain.nii.gz"
    assert shelly.inputs.executable == "bet"
    assert shelly.cmdline == f"bet {in_file} {out_file}"
    # res = shelly(plugin="cf")


def test_shell_cmd_non_existing_outputs_1(tmpdir):
    """Checking that non existing output files do not return a phantom path,
    but return NOTHING instead"""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "mandatory": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}_1.nii",
                    },
                ),
            ),
            (
                "out_2",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #2",
                        "output_file_template": "{out_name}_2.nii",
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="echo",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name="test",
    )
    shelly()
    res = shelly.result()
    assert res.output.out_1 == attr.NOTHING and res.output.out_2 == attr.NOTHING


def test_shell_cmd_non_existing_outputs_2(tmpdir):
    """Checking that non existing output files do not return a phantom path,
    but return NOTHING instead. This test has one existing and one non existing output file."""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "mandatory": True,
                        "argstr": "{out_name}_1.nii",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}_1.nii",
                    },
                ),
            ),
            (
                "out_2",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #2",
                        "output_file_template": "{out_name}_2.nii",
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="touch",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name="test",
    )
    shelly()
    res = shelly.result()
    # the first output file is created
    assert res.output.out_1 == Path(shelly.output_dir) / Path("test_1.nii")
    assert res.output.out_1.exists()
    # the second output file is not created
    assert res.output.out_2 == attr.NOTHING


def test_shell_cmd_non_existing_outputs_3(tmpdir):
    """Checking that non existing output files do not return a phantom path,
    but return NOTHING instead. This test has an existing mandatory output and another non existing output file."""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "mandatory": True,
                        "argstr": "{out_name}_1.nii",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}_1.nii",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "out_2",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #2",
                        "output_file_template": "{out_name}_2.nii",
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="touch",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name="test",
    )
    shelly()
    res = shelly.result()
    # the first output file is created
    assert res.output.out_1 == Path(shelly.output_dir) / Path("test_1.nii")
    assert res.output.out_1.exists()
    # the second output file is not created
    assert res.output.out_2 == attr.NOTHING


def test_shell_cmd_non_existing_outputs_4(tmpdir):
    """Checking that non existing output files do not return a phantom path,
    but return NOTHING instead. This test has an existing mandatory output and another non existing
    mandatory output file."""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "mandatory": True,
                        "argstr": "{out_name}_1.nii",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}_1.nii",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "out_2",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "fictional output #2",
                        "output_file_template": "{out_name}_2.nii",
                        "mandatory": True,
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="touch",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name="test",
    )
    # An exception should be raised because the second mandatory output does not exist
    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "mandatory output for variable out_2 does not exist" == str(excinfo.value)
    # checking if the first output was created
    assert (Path(shelly.output_dir) / Path("test_1.nii")).exists()


def test_shell_cmd_non_existing_outputs_multi_1(tmpdir):
    """This test looks if non existing files of an multiOuputFile are also set to NOTHING"""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=MultiInputObj,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "mandatory": True,
                        "argstr": "...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_list",
                attr.ib(
                    type=MultiOutputFile,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}",
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="echo",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name=["test_1.nii", "test_2.nii"],
    )
    shelly()
    res = shelly.result()
    # checking if the outputs are Nothing
    assert res.output.out_list[0] == attr.NOTHING
    assert res.output.out_list[1] == attr.NOTHING


def test_shell_cmd_non_existing_outputs_multi_2(tmpdir):
    """This test looks if non existing files of an multiOutputFile are also set to NOTHING.
    It checks that it also works if one file of the multiOutputFile actually exists."""
    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out_name",
                attr.ib(
                    type=MultiInputObj,
                    metadata={
                        "help_string": """
                        base name of the pretend outputs.
                        """,
                        "sep": " test_1_real.nii",  # hacky way of creating an extra file with that name
                        "mandatory": True,
                        "argstr": "...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )
    out_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_list",
                attr.ib(
                    type=MultiOutputFile,
                    metadata={
                        "help_string": "fictional output #1",
                        "output_file_template": "{out_name}_real.nii",
                    },
                ),
            ),
        ],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        cache_dir=tmpdir,
        executable="touch",
        input_spec=input_spec,
        output_spec=out_spec,
        out_name=["test_1", "test_2"],
    )
    shelly()
    res = shelly.result()
    # checking if the outputs are Nothing
    assert res.output.out_list[0] == Path(shelly.output_dir) / "test_1_real.nii"
    assert res.output.out_list[1] == attr.NOTHING


def test_shellspec_formatter_1(tmpdir):
    """test the input callable 'formatter'."""

    def spec_info(formatter):
        return SpecInfo(
            name="Input",
            fields=[
                (
                    "in1",
                    attr.ib(
                        type=str,
                        metadata={
                            "help_string": """
                            just a dummy name
                            """,
                            "mandatory": True,
                        },
                    ),
                ),
                (
                    "in2",
                    attr.ib(
                        type=str,
                        metadata={
                            "help_string": """
                                just a dummy name
                                """,
                            "mandatory": True,
                        },
                    ),
                ),
                (
                    "together",
                    attr.ib(
                        type=ty.List,
                        metadata={
                            "help_string": """
                                combines in1 and in2 into a list
                                """,
                            # When providing a formatter all other metadata options are discarded.
                            "formatter": formatter,
                        },
                    ),
                ),
            ],
            bases=(ShellSpec,),
        )

    def formatter_1(inputs):
        print("FORMATTER:", inputs)
        return f"-t [{inputs['in1']}, {inputs['in2']}]"

    input_spec = spec_info(formatter_1)
    shelly = ShellCommandTask(
        executable="exec", input_spec=input_spec, in1="i1", in2="i2"
    )
    assert shelly.cmdline == "exec -t [i1, i2]"

    # testing that the formatter can overwrite a provided value for together.
    shelly = ShellCommandTask(
        executable="exec",
        input_spec=input_spec,
        in1="i1",
        in2="i2",
        together=[1],
    )
    assert shelly.cmdline == "exec -t [i1, i2]"

    # asking for specific inputs
    def formatter_2(in1, in2):
        print("FORMATTER:", in1, in2)
        return f"-t [{in1}, {in2}]"

    input_spec = spec_info(formatter_2)

    shelly = ShellCommandTask(
        executable="exec", input_spec=input_spec, in1="i1", in2="i2"
    )
    assert shelly.cmdline == "exec -t [i1, i2]"

    def formatter_3(in1, in3):
        print("FORMATTER:", in1, in3)
        return f"-t [{in1}, {in3}]"

    input_spec = spec_info(formatter_3)

    shelly = ShellCommandTask(
        executable="exec", input_spec=input_spec, in1="i1", in2="i2"
    )
    with pytest.raises(Exception) as excinfo:
        shelly.cmdline
    assert (
        "arguments of the formatter function from together has to be in inputs or be field or output_dir, but in3 is used"
        == str(excinfo.value)
    )

    # chcking if field value is accessible when None
    def formatter_5(field):
        assert field == "-t test"
        # formatter must return a string
        return field

    input_spec = spec_info(formatter_5)

    shelly = ShellCommandTask(
        executable="exec",
        input_spec=input_spec,
        in1="i1",
        in2="i2",
        together="-t test",
    )
    assert shelly.cmdline == "exec -t test"

    # chcking if field value is accessible when None
    def formatter_4(field):
        assert field == None
        # formatter must return a string
        return ""

    input_spec = spec_info(formatter_4)

    shelly = ShellCommandTask(
        executable="exec", input_spec=input_spec, in1="i1", in2="i2"
    )
    assert shelly.cmdline == "exec"


def test_shellspec_formatter_splitter_2(tmpdir):
    """test the input callable 'formatter' when a splitter is used on an argument of the formatter."""

    def spec_info(formatter):
        return SpecInfo(
            name="Input",
            fields=[
                (
                    "in1",
                    attr.ib(
                        type=str,
                        metadata={
                            "help_string": "in1",
                        },
                    ),
                ),
                (
                    "in2",
                    attr.ib(
                        type=str,
                        metadata={
                            "help_string": "in2",
                        },
                    ),
                ),
                (
                    "together",
                    attr.ib(
                        type=ty.List,
                        metadata={
                            "help_string": """
                            uses in1
                            """,
                            # When providing a formatter all other metadata options are discarded.
                            "formatter": formatter,
                        },
                    ),
                ),
            ],
            bases=(ShellSpec,),
        )

    # asking for specific inputs
    def formatter_1(in1, in2):
        return f"-t [{in1} {in2}]"

    input_spec = spec_info(formatter_1)
    in1 = ["in11", "in12"]
    shelly = ShellCommandTask(
        name="f", executable="executable", input_spec=input_spec, in1=in1, in2="in2"
    ).split("in1")
    assert shelly != None

    # results = shelly.cmdline
    # assert len(results) == 2
    # com_results = ["executable -t [in11 in2]", "executable -t [in12 in2]"]
    # for i, cr in enumerate(com_results):
    #     assert results[i] == cr


@no_win
def test_shellcommand_error_msg(tmpdir):

    script_path = Path(tmpdir) / "script.sh"

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

    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "in1",
                str,
                {"help_string": "a dummy string", "argstr": "", "mandatory": True},
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="err_msg", executable=str(script_path), input_spec=input_spec, in1="hello"
    )

    with pytest.raises(RuntimeError) as excinfo:
        shelly()

    path_str = str(script_path)

    assert (
        str(excinfo.value)
        == f"""Error running 'err_msg' task with ['{path_str}', 'hello']:

stderr:
{path_str}: line 3: /command-that-doesnt-exist: No such file or directory


stdout:
first line is ok, it prints 'hello'
"""
    )
