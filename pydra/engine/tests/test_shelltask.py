# -*- coding: utf-8 -*-

import typing as ty
import os, shutil
import dataclasses as dc
import pytest
from pathlib import Path


from ..task import ShellCommandTask
from ..submitter import Submitter
from ..core import Workflow
from ..specs import ShellOutSpec, ShellSpec, SpecInfo, File

if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]


def test_shell_cmd_1_nosubm(tmpdir):
    """ simple command, no arguments
        no submitter
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_1(plugin):
    """ simple command, no arguments
        using submitter
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == str(shelly.output_dir) + "\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_1_strip(plugin):
    """ simple command, no arguments
        strip option to remove \n at the end os stdout
    """
    cmd = ["pwd"]
    shelly = ShellCommandTask(name="shelly", executable=cmd, strip=True)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == str(shelly.output_dir)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2_nosubm():
    """ a command with arguments, cmd and args given as executable
        no submitter
    """
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2(plugin):
    """ a command with arguments, cmd and args given as executable
        using submitter
    """
    cmd = ["echo", "hail", "pydra"]
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout.strip() == " ".join(cmd[1:])
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2a_nosubm():
    """ a command with arguments, using executable and args
        no submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)
    res = shelly()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2a(plugin):
    """ a command with arguments, using executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = ["hail", "pydra"]
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo " + " ".join(cmd_args)
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout.strip() == " ".join(cmd_args)
    assert res.output.return_code == 0
    assert res.output.stderr == ""


def test_shell_cmd_2b_nosubm():
    """ a command with arguments, using strings for executable and args
        no submitter
    """
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"
    res = shelly()
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_2b(plugin):
    """ a command with arguments, using  strings executable and args
        using submitter
    """
    cmd_exec = "echo"
    cmd_args = "pydra"
    # separate command into exec + args
    shelly = ShellCommandTask(name="shelly", executable=cmd_exec, args=cmd_args)
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == "echo pydra"
    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)
    res = shelly.result()
    assert res.output.stdout == "pydra\n"
    assert res.output.return_code == 0
    assert res.output.stderr == ""


# tests with State


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_3(plugin):
    """ commands without arguments
        splitter = executable
    """
    cmd = ["pwd", "whoami"]

    # all args given as executable
    shelly = ShellCommandTask(name="shelly", executable=cmd).split("executable")
    # TODO: doesnt make sense for tasks with splitter
    #    assert shelly.cmdline == " ".join(cmd)
    res = shelly(plugin=plugin)
    assert res[0].output.stdout == f"{str(shelly.output_dir[0])}\n"
    if "USER" in os.environ:
        assert res[1].output.stdout == f"{os.environ['USER']}\n"
    else:
        assert res[1].output.stdout
    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
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
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0].output.stdout == "nipype\n"
    assert res[1].output.stdout == "pydra\n"

    assert res[0].output.return_code == res[1].output.return_code == 0
    assert res[0].output.stderr == res[1].output.stderr == ""


@pytest.mark.parametrize("plugin", Plugins)
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
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"


@pytest.mark.parametrize("plugin", Plugins)
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
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
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


@pytest.mark.parametrize("plugin", Plugins)
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
    # this doesnt work, cmdline gives echo nipype pydra
    # assert shelly.cmdline == "echo pydra"
    res = shelly(plugin=plugin)

    assert res[0][0].output.stdout == "nipype\n"
    assert res[0][1].output.stdout == "pydra\n"

    assert res[1][0].output.stdout == "nipype"
    assert res[1][1].output.stdout == "pydra"


# tests with workflows


@pytest.mark.parametrize("plugin", Plugins)
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


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_inputspec_1_nosubm(plugin):
    """ a command with executable, args and one command opt,
        using a customized input_spec to add the opt to the command
        in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = "-n"
    cmd_args = "hello from pydra"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[("opt", str, dc.field(metadata={"cmd_pos": 1}))],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        args=cmd_args,
        opt=cmd_opt,
        input_spec=my_input_spec,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n hello from pydra"
    res = shelly(plugin=plugin)

    assert res.output.stdout == "hello from pydra"


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_inputspec_1(plugin):
    """ a command with executable, args and one command opt,
        using a customized input_spec to add the opt to the command
        in the right place that is specified in metadata["cmd_pos"]
        using submitter
    """
    cmd_exec = "echo"
    cmd_opt = "-n"
    cmd_args = "hello from pydra"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[("opt", str, dc.field(metadata={"cmd_pos": 1}))],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        args=cmd_args,
        opt=cmd_opt,
        input_spec=my_input_spec,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n hello from pydra"

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == "hello from pydra"


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_inputspec_2_nosubm(plugin):
    """ a command with executable, args and two command options,
        using a customized input_spec to add the opt to the command
        in the right place that is specified in metadata["cmd_pos"]
    """
    cmd_exec = "echo"
    cmd_opt = "-n"
    cmd_opt_hello = "HELLO"
    cmd_args = "from pydra"
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("opt_hello", str, dc.field(metadata={"cmd_pos": 3})),
            ("opt", str, dc.field(metadata={"cmd_pos": 1})),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        name="shelly",
        executable=cmd_exec,
        args=cmd_args,
        opt=cmd_opt,
        opt_hello=cmd_opt_hello,
        input_spec=my_input_spec,
    )
    assert shelly.inputs.executable == cmd_exec
    assert shelly.inputs.args == cmd_args
    assert shelly.cmdline == "echo -n HELLO from pydra"
    res = shelly(plugin=plugin)

    assert res.output.stdout == "HELLO from pydra"


# customised output spec


def test_shell_cmd_outputspec_1_nosubm():
    """
        customised output_spec, adding files to the output, providing specific pathname
        no submitter
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_tmp.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_1(plugin):
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

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_1a(plugin):
    """
        customised output_spec, adding files to the output, providing specific pathname
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, dc.field(default="newfile_tmp.txt"))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("plugin", Plugins)
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


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_2_nosubm(plugin):
    """
        customised output_spec, adding files to the output,
        using a wildcard in default
        no submitter
    """
    cmd = ["touch", "newfile_tmp.txt"]
    my_output_spec = SpecInfo(
        name="Output",
        fields=[("newfile", File, "newfile_*.txt")],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_2(plugin):
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

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.newfile.exists()


@pytest.mark.parametrize("plugin", Plugins)
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


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_3(plugin):
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

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_4(plugin):
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
        fields=[("newfile", File, dc.field(metadata={"callable": gather_output}))],
        bases=(ShellOutSpec,),
    )
    shelly = ShellCommandTask(name="shelly", executable=cmd, output_spec=my_output_spec)

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    # newfile is a list
    assert len(res.output.newfile) == 2
    assert all([file.exists for file in res.output.newfile])


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_5_nosubm(plugin):
    """
        providing output name using input_spec,
        using name_tamplate in metadata
        no submitter
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("out1", str, dc.field(metadata={"output_file_template": "{args}"}))],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_5(plugin):
    """
        providing output name using input_spec,
        using name_tamplate in metadata
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("out1", str, dc.field(metadata={"output_file_template": "{args}"}))],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    )

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_5a_nosubm(plugin):
    """
        providing output name using input_spec,
        using name_tamplate in metadata
        and changing the output name for output_spec using output_field_name
        no submitter
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "out1",
                str,
                dc.field(
                    metadata={
                        "output_file_template": "{args}",
                        "output_field_name": "out1_changed",
                    }
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, input_spec=my_input_spec
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.out1_changed.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_6_nosubm(plugin):
    """
        providing output name by providing output_file_template
        (similar to the previous example, but not touching input_spec)
        no submitter
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", File, dc.field(metadata={"output_file_template": "{args}"}))],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    )

    res = shelly()
    assert res.output.stdout == ""
    assert res.output.out1.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_shell_cmd_outputspec_6(plugin):
    """
        providing output name by providing output_file_template
        (similar to the previous example, but not touching input_spec)
    """
    cmd = "touch"
    args = "newfile_tmp.txt"

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", File, dc.field(metadata={"output_file_template": "{args}"}))],
        bases=(ShellOutSpec,),
    )

    shelly = ShellCommandTask(
        name="shelly", executable=cmd, args=args, output_spec=my_output_spec
    )

    with Submitter(plugin=plugin) as sub:
        shelly(submitter=sub)

    res = shelly.result()
    assert res.output.stdout == ""
    assert res.output.out1.exists()


# customised output_spec for tasks in workflows


@pytest.mark.parametrize("plugin", Plugins)
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
