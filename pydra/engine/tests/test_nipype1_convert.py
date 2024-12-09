import typing as ty
import pytest


from ..task import ShellTask
from ..specs import ShellOutputs, ShellSpec, SpecInfo, File

interf_input_spec = SpecInfo(
    name="Input", fields=[("test", ty.Any, {"help_string": "test"})], bases=(ShellSpec,)
)
interf_output_spec = SpecInfo(
    name="Output", fields=[("test_out", File, "*.txt")], bases=(ShellOutputs,)
)


class Interf_1(ShellTask):
    """class with customized input/output specs"""

    input_spec = interf_input_spec
    output_spec = interf_output_spec


class Interf_2(ShellTask):
    """class with customized input/output specs and executables"""

    input_spec = interf_input_spec
    output_spec = interf_output_spec
    executable = "testing command"


class Interf_3(ShellTask):
    """class with customized input and executables"""

    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "in_file",
                str,
                {"help_string": "in_file", "argstr": "'{in_file}'"},
            )
        ],
        bases=(ShellSpec,),
    )
    executable = "testing command"


class TouchInterf(ShellTask):
    """class with customized input and executables"""

    input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "new_file",
                str,
                {
                    "help_string": "new_file",
                    "argstr": "",
                    "output_file_template": "{new_file}",
                },
            )
        ],
        bases=(ShellSpec,),
    )
    executable = "touch"


def test_interface_specs_1():
    """testing if class input/output spec are set properly"""
    task = Interf_1(executable="ls")
    assert task.input_spec == interf_input_spec
    assert task.output_spec == interf_output_spec


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
    assert task.inputs.executable == "testing command"
    assert task.cmdline == "testing command"


def test_interface_executable_2():
    """testing if the class executable is overwritten by the user's input (and if the warning is raised)"""
    # warning that the user changes the executable from the one that is set as a class attribute
    with pytest.warns(UserWarning, match="changing the executable"):
        task = Interf_2(executable="i want a different command")
        assert task.executable == "testing command"
        # task.executable stays the same, but input.executable is changed, so the cmd is changed
        assert task.inputs.executable == "i want a different command"
        assert task.cmdline == "i want a different command"


def test_interface_cmdline_with_spaces():
    task = Interf_3(in_file="/path/to/file/with spaces")
    assert task.executable == "testing command"
    assert task.inputs.executable == "testing command"
    assert task.cmdline == "testing command '/path/to/file/with spaces'"


def test_interface_run_1():
    """testing execution of a simple interf with customized input and executable"""
    task = TouchInterf(new_file="hello.txt")
    assert task.cmdline == "touch hello.txt"
    res = task()
    assert res.output.new_file.fspath.exists()
