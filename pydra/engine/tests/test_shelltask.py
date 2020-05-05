# -*- coding: utf-8 -*-

import attr
import typing as ty
import os, sys, shutil
import pytest
from pathlib import Path


from ..task import ShellCommandTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, ShellSpec, SpecInfo, File
from .utils import result_no_submitter, result_submitter


if sys.platform.startswith("win"):
    pytest.skip("SLURM not available in windows", allow_module_level=True)


@pytest.mark.flaky(reruns=2)  # when dask
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_1(plugin_dask_opt, results_function):
    """ simple command, no arguments """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin=plugin_dask_opt)
    assert Path(res.output.stdout.rstrip()) == shelly.output_dir
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_1_strip(plugin, results_function):
    """ simple command, no arguments
        strip option to remove \n at the end os stdout
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, strip=True)
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin)
    assert Path(res.output.stdout) == Path(shelly.output_dir)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2(plugin, results_function):
    """ a command with arguments, cmd and args given as executable """
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)

    res = results_function(shelly, plugin)
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2a(plugin, results_function):
    """ a command with arguments, using executable and args """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)

    res = results_function(shelly, plugin)
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_2b(plugin, results_function):
    """ a command with arguments, using  strings executable and args """
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


# tests with State


@pytest.mark.flaky(reruns=2)
def test_shell_cmd_3(plugin_dask_opt):
    """ commands without arguments
        splitter = executable
    """
    cmd = ["pwd", "whoami"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd).split("executable")
    assert shelly.cmdline == ["pwd", "whoami"]
    res = shelly(plugin=plugin_dask_opt)
    assert Path(res[0].output.stdout.rstrip()) == shelly.output_dir[0]

    if "USER" in os.environ:
        assert res[1].output.stdout == f"{os.environ['USER']}\n"
    else:
        assert res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


def test_shell_cmd_4(plugin):
    """ a command with arguments, using executable and args
        splitter=args
    """
    cmd_exec = "echo"
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter="args"
    )
    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    assert shelly.cmdline == ["echo nipype", "echo pydra"]
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"

    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


def test_shell_cmd_5(plugin):
    """ a command with arguments
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
    assert shelly.inputs.executable == "echo"
    assert shelly.inputs.args == ["nipype", "pydra"]
    assert shelly.cmdline == ["echo nipype", "echo pydra"]
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"


def test_shell_cmd_6(plugin):
    """ a command with arguments,
        outer splitter for executable and args
    """
    cmd_exec = ["echo", ["echo", "-n"]]
    cmd_args = ["nipype", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args).split(
        splitter=["executable", "args"]
    )
    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]
    assert shelly.cmdline == [
        "echo nipype",
        "echo pydra",
        "echo -n nipype",
        "echo -n pydra",
    ]
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


def test_shell_cmd_7(plugin):
    """ a command with arguments,
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
    assert shelly.inputs.executable == ["echo", ["echo", "-n"]]
    assert shelly.inputs.args == ["nipype", "pydra"]

    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"

    assert res[1][0].output.stdout == "nipype"
    assert res[1][1].output.stdout == "pydra"


# tests with workflows


def test_wf_shell_cmd_1(plugin):
    """ a workflow with two connected commands"""
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

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    res = wf.result()
    assert "_result.pklz" in res.output.out
    assert "_task.pklz" in res.output.out


# customised input spec


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_1(plugin, results_function):
    """ a command with executable, args and one command opt,
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
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n hello from pydra"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "hello from pydra"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_2(plugin, results_function):
    """ a command with executable, args and two command options,
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
                attr.ib(type=str, metadata={"position": 3, "help_string": "todo"}),
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
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n HELLO from pydra"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO from pydra"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3(plugin, results_function):
    """  mandatory field added to fields, value provided """
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "text", "mandatory": True},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, text=hello, input_spec=my_input_spec
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3a(plugin, results_function):
    """  mandatory field added to fields, value provided
        using shorter syntax for input spec (no attr.ib)
    """
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("text", str, {"position": 1, "help_string": "text", "mandatory": True})
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, text=hello, input_spec=my_input_spec
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3b(plugin, results_function):
    """  mandatory field added to fields, value provided after init"""
    cmd_exec = "echo"
    hello = "HELLO"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "text", "mandatory": True},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )
    shelly.inputs.text = hello
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "HELLO\n"


def test_shell_cmd_inputspec_3c_exception(plugin):
    """  mandatory field added to fields, value is not provided, so exception is raised """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "text", "mandatory": True},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )
    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "mandatory" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_3c(plugin, results_function):
    """  mandatory=False, so tasks runs fine even without the value """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default=None,
                    metadata={"position": 1, "help_string": "text", "mandatory": False},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo"
    res = results_function(shelly, plugin)
    assert res.output.stdout == "\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4(plugin, results_function):
    """  mandatory field added to fields, value provided """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hello",
                    metadata={"position": 1, "help_string": "text"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4a(plugin, results_function):
    """  mandatory field added to fields, value provided
        using shorter syntax for input spec (no attr.ib)
    """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[("text", str, "Hello", {"position": 1, "help_string": "text"})],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hello"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hello\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_4b(plugin, results_function):
    """  mandatory field added to fields, value provided """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hi",
                    metadata={"position": 1, "help_string": "text"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, input_spec=my_input_spec
    )

    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "echo Hi"

    res = results_function(shelly, plugin)
    assert res.output.stdout == "Hi\n"


def test_shell_cmd_inputspec_4c_exception(plugin):
    """  mandatory field added to fields, value provided """
    cmd_exec = "echo"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    default="Hello",
                    metadata={"position": 1, "help_string": "text", "mandatory": True},
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
    """  mandatory field added to fields, value provided """
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
def test_shell_cmd_inputspec_5_nosubm(plugin, results_function):
    """ checking xor in metadata: task should work fine, since only one option is True"""
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
        name="shelly", executable=cmd_exec, opt_t=cmd_t, input_spec=my_input_spec
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -t"
    res = results_function(shelly, plugin)


def test_shell_cmd_inputspec_5a_exception(plugin):
    """ checking xor in metadata: both options are True, so the task raises exception"""
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
    )
    with pytest.raises(Exception) as excinfo:
        shelly()
    assert "is mutually exclusive" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_6(plugin, results_function):
    """ checking requires in metadata:
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
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    res = results_function(shelly, plugin)


def test_shell_cmd_inputspec_6a_exception(plugin):
    """ checking requires in metadata:
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
def test_shell_cmd_inputspec_6b(plugin, results_function):
    """ checking requires in metadata:
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
    )
    shelly.inputs.opt_l = cmd_l
    assert shelly.inputs.executable == cmd_exec
    assert shelly.cmdline == "ls -l -t"
    res = results_function(shelly, plugin)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7(plugin, results_function):
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
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_7a(plugin, results_function):
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
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1_changed.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_8(plugin, results_function, tmpdir):
    """ using input_spec, providing list of files as an input """

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
                        "help_string": "list of files",
                        "mandatory": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, files=files_list, input_spec=my_input_spec
    )

    assert shelly.inputs.executable == cmd_exec
    res = results_function(shelly, plugin)
    assert res.output.stdout == "hello from boston"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1(plugin, results_function, tmpdir):
    """ shelltask changes a file in place,
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
        name="shelly", executable=cmd, input_spec=my_input_spec, orig_file=str(file)
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  copied, and than it is changed in place
    assert res.output.out_file.parent == shelly.output_dir
    with open(res.output.out_file, "r") as f:
        assert "hi from pydra\n" == f.read()
    # the original file is unchanged
    with open(file, "r") as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1a(plugin, results_function, tmpdir):
    """ shelltask changes a file in place,
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
        name="shelly", executable=cmd, input_spec=my_input_spec, orig_file=str(file)
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is uses a soft link, but it creates and an extra copy before modifying
    assert res.output.out_file.parent == shelly.output_dir

    assert res.output.out_file.parent.joinpath(res.output.out_file.name + "s").exists()
    with open(res.output.out_file, "r") as f:
        assert "hi from pydra\n" == f.read()
    # the file is uses a soft link, but it creates and an extra copy
    # it might depend on the OS
    linked_file_copy = res.output.out_file.parent.joinpath(
        res.output.out_file.name + "s"
    )
    if linked_file_copy.exists():
        with open(linked_file_copy, "r") as f:
            assert "hello from pydra\n" == f.read()

    # the original file is unchanged
    with open(file, "r") as f:
        assert "hello from pydra\n" == f.read()


@pytest.mark.xfail(
    reason="not sure if we want to support input overwrite,"
    "if we allow for this orig_file is changing, so does checksum,"
    " and the results cant be found"
)
@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_1b(plugin, results_function, tmpdir):
    """ shelltask changes a file in place,
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
        name="shelly", executable=cmd, input_spec=my_input_spec, orig_file=str(file)
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out_file.exists()
    # the file is  not copied, it is changed in place
    assert res.output.out_file == file
    with open(res.output.out_file, "r") as f:
        assert "hi from pydra\n" == f.read()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_1(plugin, results_function):
    """  adding state to the input from input_spec """
    cmd_exec = "echo"
    hello = ["HELLO", "hi"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "text",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "text", "mandatory": True},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, text=hello, input_spec=my_input_spec
    ).split("text")
    assert shelly.inputs.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "HELLO\n"
    assert res[1].output.stdout == "hi\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_1a(plugin, results_function):
    """  adding state to the input from input_spec
        using shorter syntax for input_spec (without default)
    """
    cmd_exec = "echo"
    hello = ["HELLO", "hi"]
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("text", str, {"position": 1, "help_string": "text", "mandatory": True})
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, text=hello, input_spec=my_input_spec
    ).split("text")
    assert shelly.inputs.executable == cmd_exec

    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "HELLO\n"
    assert res[1].output.stdout == "hi\n"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_2(plugin, results_function):
    """
        adding splitter to input tha is used in the output_file_tamplate
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
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    ).split("args")

    res = results_function(shelly, plugin)
    for i in range(len(args)):
        assert res[i].output.stdout == ""
        assert res[i].output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_state_3(plugin, results_function, tmpdir):
    """  adding state to the File-input from input_spec """

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
                    metadata={"position": 1, "help_string": "files", "mandatory": True},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd_exec, file=files, input_spec=my_input_spec
    ).split("file")

    assert shelly.inputs.executable == cmd_exec
    # todo: this doesn't work when state
    # assert shelly.cmdline == "echo HELLO"
    res = results_function(shelly, plugin)
    assert res[0].output.stdout == "hello from pydra"
    assert res[1].output.stdout == "have a nice one"


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_inputspec_copyfile_state_1(plugin, results_function, tmpdir):
    """  adding state to the File-input from input_spec """

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
        name="shelly", executable=cmd, input_spec=my_input_spec, orig_file=files
    ).split("orig_file")

    txt_l = ["from pydra", "world"]
    res_l = results_function(shelly, plugin)
    for i, res in enumerate(res_l):
        assert res.output.stdout == ""
        assert res.output.out_file.exists()
        # the file is  copied, and than it is changed in place
        assert res.output.out_file.parent == shelly.output_dir[i]
        with open(res.output.out_file, "r") as f:
            assert f"hi {txt_l[i]}\n" == f.read()
        # the original file is unchanged
        with open(files[i], "r") as f:
            assert f"hello {txt_l[i]}\n" == f.read()


# customised input_spec in Workflow


@pytest.mark.flaky(reruns=2)  # when dask
def test_wf_shell_cmd_2(plugin_dask_opt):
    """ a workflow with input with defined output_file_template (str)
        that requires wf.lzin
    """
    wf = Workflow(name="wf", input_spec=["cmd", "args"])

    wf.inputs.cmd = "touch"
    wf.inputs.args = "newfile.txt"

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


def test_wf_shell_cmd_2a(plugin):
    """ a workflow with input with defined output_file_template (tuple)
        that requires wf.lzin
    """
    wf = Workflow(name="wf", input_spec=["cmd", "args"])

    wf.inputs.cmd = "touch"
    wf.inputs.args = "newfile.txt"

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


def test_wf_shell_cmd_3(plugin):
    """ a workflow with 2 tasks,
        first one has input with output_file_template (str, uses wf.lzin),
        that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = "newfile.txt"

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
                    type=File, metadata={"position": 1, "help_string": "output file"}
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
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
    assert res.output.out2 == ""
    assert res.output.cp_file.exists()


def test_wf_shell_cmd_3a(plugin):
    """ a workflow with 2 tasks,
        first one has input with output_file_template (str, uses wf.lzin),
        that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

    wf.inputs.cmd1 = "touch"
    wf.inputs.cmd2 = "cp"
    wf.inputs.args = "newfile.txt"

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
                    type=str, metadata={"position": 1, "help_string": "output file"}
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
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
    """ a workflow with 2 tasks and splitter on the wf level,
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
                    type=str, metadata={"position": 1, "help_string": "output file"}
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
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
    for res in res_l:
        assert res.output.out1 == ""
        assert res.output.touch_file.exists()
        assert res.output.out2 == ""
        assert res.output.cp_file.exists()


def test_wf_shell_cmd_ndst_1(plugin):
    """ a workflow with 2 tasks and a splitter on the node level,
        first one has input with output_file_template (str, uses wf.lzin),
        that is passed to the second task
    """
    wf = Workflow(name="wf", input_spec=["cmd1", "cmd2", "args"])

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
                    type=str, metadata={"position": 1, "help_string": "output file"}
                ),
            ),
            (
                "out_file",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
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
def test_shell_cmd_outputspec_1(plugin, results_function):
    """
        customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_1a(plugin, results_function):
    """
        customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", attr.ib(type=File, default="newfile_tmp.txt"))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


def test_shell_cmd_outputspec_1b_exception(plugin):
    """
        customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp_.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    with pytest.raises(Exception) as exinfo:
        with Submitter(plugin=plugin) as sub:
            shelly(submitter=sub)
    assert "does not exist" in str(exinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_2(plugin, results_function):
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
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


def test_shell_cmd_outputspec_2a_exception(plugin):
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
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            shelly(submitter=sub)
    assert "no file matches" in str(excinfo.value)


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_3(plugin, results_function):
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
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_4(plugin, results_function):
    """
        customised output_spec, adding files to the output,
        using a function to collect output, the function is saved in the field metadata
    """
    cmd = ["touch", "newfile_tmp1.txt", "newfile_tmp2.txt"]

    def gather_output(keyname, output_dir):
        if keyname == "newfile":
            return list(Path(output_dir).expanduser().glob("newfile*.txt"))

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


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_outputspec_5(plugin, results_function):
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
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    )

    res = results_function(shelly, plugin)
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("results_function", [result_no_submitter, result_submitter])
def test_shell_cmd_state_outputspec_1(plugin, results_function):
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
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    ).split("args")

    res = results_function(shelly, plugin)
    for i in range(len(args)):
        assert res[i].output.stdout == ""
        assert res[i].output.out1.exists()


# customised output_spec for tasks in workflows


def test_shell_cmd_outputspec_wf_1(plugin):
    """
        customised output_spec for tasks within a Workflow,
        adding files to the output, providing specific pathname
    """

    cmd = ["touch", "newfile_tmp.txt"]
    wf = Workflow(name="wf", input_spec=["cmd"])
    wf.inputs.cmd = cmd

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


def no_fsl():
    if "FSLDIR" not in os.environ:
        return True


@pytest.mark.skipif(no_fsl(), reason="fsl is not installed")
def test_fsl():
    """  mandatory field added to fields, value provided """

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
    in_file = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "foo.nii"

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="bet_task", executable="bet", in_file=in_file, input_spec=bet_input_spec
    )
    assert shelly.inputs.executable == "bet"
    assert shelly.cmdline == f"bet {in_file} {in_file}_brain"
    # res = shelly(plugin="cf")
