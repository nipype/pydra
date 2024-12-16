import typing as ty
import pytest
from pathlib import Path
from pydra.engine.specs import ShellOutputs, ShellSpec
from fileformats.generic import File
from pydra.design import shell


def find_txt(output_dir: Path) -> File:
    files = list(output_dir.glob("*.txt"))
    assert len(files) == 1
    return files[0]


interf_inputs = [shell.arg(name="test", type=ty.Any, help_string="test")]
interf_outputs = [shell.out(name="test_out", type=File, callable=find_txt)]


Interf_1 = shell.define(inputs=interf_inputs, outputs=interf_outputs)
Interf_2 = shell.define("testing command", inputs=interf_inputs, outputs=interf_outputs)


@shell.define
class Interf_3(ShellSpec["Interf_3.Outputs"]):
    """class with customized input and executables"""

    executable = ["testing", "command"]

    in_file: str = shell.arg(help_string="in_file", argstr="{in_file}")

    @shell.outputs
    class Outputs(ShellOutputs):
        pass


@shell.define
class TouchInterf(ShellSpec["TouchInterf.Outputs"]):
    """class with customized input and executables"""

    new_file: str = shell.outarg(
        help_string="new_file", argstr="", path_template="{new_file}"
    )
    executable = "touch"

    @shell.outputs
    class Outputs(ShellOutputs):
        pass


def test_interface_specs_1():
    """testing if class input/output spec are set properly"""
    task_spec = Interf_1(executable="ls")
    assert task.Outputs == Interf_1.Outputs


def test_interface_specs_2():
    """testing if class input/output spec are overwritten properly by the user's specs"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[("my_inp", ty.Any, {"help_string": "my inp"})],
        bases=(ShellSpec,),
    )
    my_output_spec = SpecInfo(
        name="Output", fields=[("my_out", File, "*.txt")], bases=(ShellOutputs,)
    )
    task = Interf_1(input_spec=my_input_spec, output_spec=my_output_spec)
    assert task.input_spec == my_input_spec
    assert task.output_spec == my_output_spec


def test_interface_executable_1():
    """testing if the class executable is properly set and used in the command line"""
    task = Interf_2()
    assert task.executable == "testing command"
    assert task.spec.executable == "testing command"
    assert task.cmdline == "testing command"


def test_interface_executable_2():
    """testing if the class executable is overwritten by the user's input (and if the warning is raised)"""
    # warning that the user changes the executable from the one that is set as a class attribute
    with pytest.warns(UserWarning, match="changing the executable"):
        task = Interf_2(executable="i want a different command")
        assert task.executable == "testing command"
        # task.executable stays the same, but input.executable is changed, so the cmd is changed
        assert task.spec.executable == "i want a different command"
        assert task.cmdline == "i want a different command"


def test_interface_cmdline_with_spaces():
    task = Interf_3(in_file="/path/to/file/with spaces")
    assert task.executable == "testing command"
    assert task.spec.executable == "testing command"
    assert task.cmdline == "testing command '/path/to/file/with spaces'"


def test_interface_run_1():
    """testing execution of a simple interf with customized input and executable"""
    task = TouchInterf(new_file="hello.txt")
    assert task.cmdline == "touch hello.txt"
    res = task()
    assert res.output.new_file.fspath.exists()
