import attr
import typing as ty
import pytest

from ..task import ShellCommandTask
from ..specs import ShellOutSpec, ShellSpec, SpecInfo, File
from .utils import use_validator


def test_shell_cmd_execargs_1():
    # separate command into exec + args
    shelly = ShellCommandTask(executable="executable", args="arg")
    assert shelly.cmdline == "executable arg"


def test_shell_cmd_execargs_2():
    # separate command into exec + args
    shelly = ShellCommandTask(executable=["cmd_1", "cmd_2"], args="arg")
    assert shelly.cmdline == "cmd_1 cmd_2 arg"


def test_shell_cmd_inputs_1():
    """ additional input with provided position """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inp1", "argstr": ""},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", args="arg", inpA="inp1", input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable inp1 arg"


def test_shell_cmd_inputs_1a():
    """ additional input without provided position """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("inpA", attr.ib(type=str, metadata={"help_string": "inpA", "argstr": ""}))
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", args="arg", inpA="inpNone1", input_spec=my_input_spec
    )
    # inp1 should be firt one after executable
    assert shelly.cmdline == "executable inpNone1 arg"


def test_shell_cmd_inputs_1b():
    """ additional input with negative position """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": -1, "help_string": "inpA", "argstr": ""},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        executable="executable", args="arg", inpA="inp-1", input_spec=my_input_spec
    )
    # inp1 should be last before arg
    assert shelly.cmdline == "executable inp-1 arg"


def test_shell_cmd_inputs_2():
    """ additional inputs with provided positions """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": 2, "help_string": "inpA", "argstr": ""},
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpN", "argstr": ""},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        executable="executable", inpB="inp1", inpA="inp2", input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable inp1 inp2"


def test_shell_cmd_inputs_2a():
    """ additional inputs without provided positions """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("inpA", attr.ib(type=str, metadata={"help_string": "inpA", "argstr": ""})),
            ("inpB", attr.ib(type=str, metadata={"help_string": "inpB", "argstr": ""})),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        executable="executable",
        inpA="inpNone1",
        inpB="inpNone2",
        input_spec=my_input_spec,
    )
    # position taken from the order in input spec
    assert shelly.cmdline == "executable inpNone1 inpNone2"


def test_shell_cmd_inputs_2_err():
    """ additional inputs with provided positions (exception due to the duplication)"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpA", "argstr": ""},
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpB", "argstr": ""},
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA="inp1", inpB="inp1", input_spec=my_input_spec
    )
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "1 is already used" in str(e.value)


def test_shell_cmd_inputs_3():
    """ additional inputs: positive pos, negative pos and  no pos """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpA", "argstr": ""},
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={"position": -1, "help_string": "inpB", "argstr": ""},
                ),
            ),
            ("inpC", attr.ib(type=str, metadata={"help_string": "inpC", "argstr": ""})),
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        executable="executable",
        inpA="inp1",
        inpB="inp-1",
        inpC="inpNone",
        input_spec=my_input_spec,
    )
    # input without position shoild be between positive an negative positions
    assert shelly.cmdline == "executable inp1 inpNone inp-1"


def test_shell_cmd_inputs_argstr_1():
    """ additional string inputs with argstr """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpA", "argstr": "-v"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA="inp1", input_spec=my_input_spec
    )
    # flag used before inp1
    assert shelly.cmdline == "executable -v inp1"


def test_shell_cmd_inputs_argstr_2():
    """ additional bool inputs with argstr """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=bool,
                    metadata={"position": 1, "help_string": "inpA", "argstr": "-v"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    # separate command into exec + args
    shelly = ShellCommandTask(
        executable="executable", args="arg", inpA=True, input_spec=my_input_spec
    )
    # a flag is used without any additional argument
    assert shelly.cmdline == "executable -v arg"


def test_shell_cmd_inputs_list_1():
    """ providing list as an additional input, no sep, no argstr """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=ty.List[str],
                    metadata={"position": 2, "help_string": "inpA", "argstr": ""},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["el_1", "el_2", "el_3"], input_spec=my_input_spec
    )
    # multiple elements
    assert shelly.cmdline == "executable el_1 el_2 el_3"


def test_shell_cmd_inputs_list_2():
    """ providing list as an additional input, no sep, but argstr """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=ty.List[str],
                    metadata={"position": 2, "help_string": "inpA", "argstr": "-v"},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["el_1", "el_2", "el_3"], input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v el_1 el_2 el_3"


def test_shell_cmd_inputs_list_3():
    """ providing list as an additional input, no sep, argstr with ..."""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=ty.List[str],
                    metadata={"position": 2, "help_string": "inpA", "argstr": "-v..."},
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["el_1", "el_2", "el_3"], input_spec=my_input_spec
    )
    # a flag is repeated
    assert shelly.cmdline == "executable -v el_1 -v el_2 -v el_3"


def test_shell_cmd_inputs_list_sep_1():
    """ providing list as an additional input:, sep, no argstr"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "sep": ",",
                        "argstr": "",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["aaa", "bbb", "ccc"], input_spec=my_input_spec
    )
    # separated by commas
    assert shelly.cmdline == "executable aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_2():
    """ providing list as an additional input:, sep, and argstr"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "sep": ",",
                        "argstr": "-v",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["aaa", "bbb", "ccc"], input_spec=my_input_spec
    )
    # a flag is used once
    assert shelly.cmdline == "executable -v aaa,bbb,ccc"


def test_shell_cmd_inputs_sep_3():
    """ providing list as an additional input:, sep, argstr with ..."""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "sep": ",",
                        "argstr": "-v...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["aaa", "bbb", "ccc"], input_spec=my_input_spec
    )
    # a flag is repeated
    assert shelly.cmdline == "executable -v aaa, -v bbb, -v ccc"


def test_shell_cmd_inputs_sep_4():
    """ providing 1-el list as an additional input:, sep, argstr with ..., """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "sep": ",",
                        "argstr": "-v...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["aaa"], input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_sep_4a():
    """ providing str instead of list as an additional input:, sep, argstr with ..."""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "sep": ",",
                        "argstr": "-v...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA="aaa", input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_format_1():
    """ additional inputs with argstr that has string formatting"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "-v {inpA}",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA="aaa", input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_format_2():
    """ additional inputs with argstr that has string formatting and ..."""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "-v {inpA}...",
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", inpA=["el_1", "el_2"], input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v el_1 -v el_2"


def test_shell_cmd_inputs_mandatory_1():
    """ additional inputs with mandatory=True"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(executable="executable", input_spec=my_input_spec)
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value)


def test_shell_cmd_inputs_template_1():
    """ additional inputs, one uses output_file_template (and argstr)"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "-o",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA"
    )
    # outA has argstr in the metadata fields, so it's a part of the command line
    # the full path will be use din the command line
    assert shelly.cmdline == f"executable inpA -o {str(shelly.output_dir / 'inpA_out')}"
    # checking if outA in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA"]


def test_shell_cmd_inputs_template_1a():
    """ additional inputs, one uses output_file_template (without argstr)"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outA",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA"
    )
    # outA has no argstr in metadata, so it's not a part of the command line
    assert shelly.cmdline == f"executable inpA"


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_2():
    """ additional inputs, one uses output_file_template (and argstr, but input not provided)"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={"position": 1, "help_string": "inpB", "argstr": ""},
                ),
            ),
            (
                "outB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outB",
                        "argstr": "-o",
                        "output_file_template": "{inpB}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(executable="executable", input_spec=my_input_spec)
    # inpB not in the inputs, so no outB in the command line
    assert shelly.cmdline == "executable"
    # checking if outB in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outB"]


def test_shell_cmd_inputs_template_3():
    """ additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "inpB",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outA",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
            (
                "outB",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outB",
                        "output_file_template": "{inpB}_out",
                    },
                ),
            ),
            (
                "outAB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": -1,
                        "help_string": "outAB",
                        "argstr": "-o {outA} {outB}",
                        "readonly": True,
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", inpB="inpB"
    )
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable inpA inpB -o {str(shelly.output_dir / 'inpA_out')} {str(shelly.output_dir / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


def test_shell_cmd_inputs_template_3a():
    """ additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    testing a different order within the input spec
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "inpB",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outAB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": -1,
                        "help_string": "outAB",
                        "argstr": "-o {outA} {outB}",
                        "readonly": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outA",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
            (
                "outB",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outB",
                        "output_file_template": "{inpB}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", inpB="inpB"
    )
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable inpA inpB -o {str(shelly.output_dir / 'inpA_out')} {str(shelly.output_dir / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_4():
    """ additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    one output_file_template can't be resolved - no inpB is provided
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "inpB",
                attr.ib(
                    type=str,
                    metadata={"position": 2, "help_string": "inpB", "argstr": ""},
                ),
            ),
            (
                "outAB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": -1,
                        "help_string": "outAB",
                        "argstr": "-o {outA} {outB}",
                        "readonly": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outA",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
            (
                "outB",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "outB",
                        "output_file_template": "{inpB}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA"
    )
    # inpB is not provided so outB not in the command line
    assert shelly.cmdline == f"executable inpA -o {str(shelly.output_dir / 'inpA_out')}"
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


def test_shell_cmd_inputs_template_5_ex():
    """ checking if the exception is raised for read-only fields when input is set"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "outAB",
                attr.ib(
                    type=str,
                    metadata={
                        "position": -1,
                        "help_string": "outAB",
                        "argstr": "-o",
                        "readonly": True,
                    },
                ),
            )
        ],
        bases=(ShellSpec,),
    )

    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, outAB="outAB"
    )
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "read only" in str(e.value)


def test_shell_cmd_inputs_template_6():
    """ additional inputs with output_file_template that has type ty.Union[str, bool]
        no default is set, so if nothing is provided as an input, the output  is used
        whenever the template can be formatted
        (the same way as for templates that has type=str)
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=ty.Union[str, bool],
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "-o",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # no input for outA (and no default value), so the output is created whenever the
    # template can be formatted (the same way as for templates that has type=str)
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA"
    )
    assert shelly.cmdline == f"executable inpA -o {str(shelly.output_dir / 'inpA_out')}"

    # a string is provided for outA, so this should be used as the outA value
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA="outA"
    )
    assert shelly.cmdline == "executable inpA -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA=True
    )
    assert shelly.cmdline == f"executable inpA -o {str(shelly.output_dir / 'inpA_out')}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA=False
    )
    assert shelly.cmdline == "executable inpA"


def test_shell_cmd_inputs_template_6a():
    """ additional inputs with output_file_template that has type ty.Union[str, bool]
        and default is set to False,
        so if nothing is provided as an input, the output is not used
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=ty.Union[str, bool],
                    default=False,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "-o",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    # no input for outA, but default is False, so the outA shouldn't be used
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA"
    )
    assert shelly.cmdline == "executable inpA"

    # a string is provided for outA, so this should be used as the outA value
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA="outA"
    )
    assert shelly.cmdline == "executable inpA -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA=True
    )
    assert shelly.cmdline == f"executable inpA -o {str(shelly.output_dir / 'inpA_out')}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA="inpA", outA=False
    )
    assert shelly.cmdline == "executable inpA"


def test_shell_cmd_inputs_template_7(tmpdir):
    """ additional inputs uses output_file_template with a suffix (no extension)
    no keep_extension is used
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "",
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    inpA_file = tmpdir.join("a_file.txt")
    inpA_file.write("content")
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA=inpA_file
    )

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmpdir.join('a_file.txt')} {str(shelly.output_dir / 'a_file_out.txt')}"
    )


def test_shell_cmd_inputs_template_7a(tmpdir):
    """ additional inputs uses output_file_template with a suffix (no extension)
        keep_extension is True (as default)
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "",
                        "keep_extension": True,
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    inpA_file = tmpdir.join("a_file.txt")
    inpA_file.write("content")
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA=inpA_file
    )

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmpdir.join('a_file.txt')} {str(shelly.output_dir / 'a_file_out.txt')}"
    )


def test_shell_cmd_inputs_template_7b(tmpdir):
    """ additional inputs uses output_file_template with a suffix (no extension)
    keep extension is False (so the extension is removed when creating the output)
    """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "",
                        "keep_extension": False,
                        "output_file_template": "{inpA}_out",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    inpA_file = tmpdir.join("a_file.txt")
    inpA_file.write("content")
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA=inpA_file
    )

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmpdir.join('a_file.txt')} {str(shelly.output_dir / 'a_file_out')}"
    )


def test_shell_cmd_inputs_template_8(tmpdir):
    """additional inputs uses output_file_template with a suffix and an extension"""
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "inpA",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "help_string": "inpA",
                        "argstr": "",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "outA",
                attr.ib(
                    type=str,
                    metadata={
                        "position": 2,
                        "help_string": "outA",
                        "argstr": "",
                        "output_file_template": "{inpA}_out.txt",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    inpA_file = tmpdir.join("a_file.t")
    inpA_file.write("content")
    shelly = ShellCommandTask(
        executable="executable", input_spec=my_input_spec, inpA=inpA_file
    )

    # outA should be formatted in a way that inpA extension is removed and the template extension is used
    assert (
        shelly.cmdline
        == f"executable {tmpdir.join('a_file.t')} {str(shelly.output_dir / 'a_file_out.txt')}"
    )


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_di(tmpdir, use_validator):
    """ example from #279 """
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "image_dimensionality",
                attr.ib(
                    type=int,
                    metadata={
                        "help_string": """
                    2/3/4
                    This option forces the image to be treated as a specified-dimensional image.
                    If not specified, the program tries to infer the dimensionality from
                    the input image.
                    """,
                        "allowed_values": [2, 3, 4],
                        "argstr": "-d",
                    },
                ),
            ),
            (
                "inputImageFilename",
                attr.ib(
                    type=File,
                    metadata={
                        "help_string": "A scalar image is expected as input for noise correction.",
                        "argstr": "-i",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "noise_model",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                    Rician/(Gaussian)
                    Employ a Rician or Gaussian noise model.
                    """,
                        "allowed_values": ["Rician", "Gaussian"],
                        "argstr": "-n",
                    },
                ),
            ),
            (
                "maskImageFilename",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "If a mask image is specified, denoising is only performed in the mask region.",
                        "argstr": "-x",
                    },
                ),
            ),
            (
                "shrink_factor",
                attr.ib(
                    type=int,
                    default=1,
                    metadata={
                        "help_string": """
                    (1)/2/3/...
                    Running noise correction on large images can be time consuming.
                    To lessen computation time, the input image can be resampled.
                    The shrink factor, specified as a single integer, describes this
                    resampling. Shrink factor = 1 is the default.
                    """,
                        "argstr": "-s",
                    },
                ),
            ),
            (
                "patch_radius",
                attr.ib(
                    type=int,
                    default=1,
                    metadata={
                        "help_string": "Patch radius. Default = 1x1x1",
                        "argstr": "-p",
                    },
                ),
            ),
            (
                "search_radius",
                attr.ib(
                    type=int,
                    default=2,
                    metadata={
                        "help_string": "Search radius. Default = 2x2x2.",
                        "argstr": "-r",
                    },
                ),
            ),
            (
                "correctedImage",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": """
                    The output consists of the noise corrected version of the input image.
                    Optionally, one can also output the estimated noise image.
                    """,
                        "output_file_template": "{inputImageFilename}_out",
                    },
                ),
            ),
            (
                "noiseImage",
                attr.ib(
                    type=ty.Union[str, bool],
                    default=False,
                    metadata={
                        "help_string": """
                    The output consists of the noise corrected version of the input image.
                    Optionally, one can also output the estimated noise image.
                    """,
                        "output_file_template": "{inputImageFilename}_noise",
                    },
                ),
            ),
            (
                "output",
                attr.ib(
                    type=str,
                    metadata={
                        "help_string": "Combined output",
                        "argstr": "-o [{correctedImage}, {noiseImage}]",
                        "position": -1,
                        "readonly": True,
                    },
                ),
            ),
            (
                "version",
                attr.ib(
                    type=bool,
                    default=False,
                    metadata={
                        "help_string": "Get Version Information.",
                        "argstr": "--version",
                    },
                ),
            ),
            (
                "verbose",
                attr.ib(
                    type=int,
                    default=0,
                    metadata={"help_string": "(0)/1. Verbose output. ", "argstr": "-v"},
                ),
            ),
            (
                "help_short",
                attr.ib(
                    type=bool,
                    default=False,
                    metadata={
                        "help_string": "Print the help menu (short version)",
                        "argstr": "-h",
                    },
                ),
            ),
            (
                "help",
                attr.ib(
                    type=int,
                    metadata={
                        "help_string": "Print the help menu.",
                        "argstr": "--help",
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )

    my_input_file = tmpdir.join("a_file.ext")
    my_input_file.write("content")

    # no input provided
    shelly = ShellCommandTask(executable="DenoiseImage", input_spec=my_input_spec)
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value)

    # input file name, noiseImage is not set, so using default value False
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        input_spec=my_input_spec,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -i {tmpdir.join('a_file.ext')} -s 1 -p 1 -r 2 -o [{str(shelly.output_dir / 'a_file_out.ext')}]"
    )

    # input file name, noiseImage is set to True, so template is used in the utput
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        input_spec=my_input_spec,
        noiseImage=True,
    )
    assert (
        shelly.cmdline == f"DenoiseImage -i {tmpdir.join('a_file.ext')} -s 1 -p 1 -r 2 "
        f"-o [{str(shelly.output_dir / 'a_file_out.ext')}, {str(shelly.output_dir / 'a_file_noise.ext')}]"
    )

    # input file name and help_short
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        help_short=True,
        input_spec=my_input_spec,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -i {tmpdir.join('a_file.ext')} -s 1 -p 1 -r 2 -h -o [{str(shelly.output_dir / 'a_file_out.ext')}]"
    )

    assert shelly.output_names == [
        "return_code",
        "stdout",
        "stderr",
        "correctedImage",
        "noiseImage",
    ]

    # adding image_dimensionality that has allowed_values [2, 3, 4]
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        input_spec=my_input_spec,
        image_dimensionality=2,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -d 2 -i {tmpdir.join('a_file.ext')} -s 1 -p 1 -r 2 -o [{str(shelly.output_dir / 'a_file_out.ext')}]"
    )

    # adding image_dimensionality that has allowed_values [2, 3, 4] and providing 5 - exception should be raised
    with pytest.raises(ValueError) as excinfo:
        shelly = ShellCommandTask(
            executable="DenoiseImage",
            inputImageFilename=my_input_file,
            input_spec=my_input_spec,
            image_dimensionality=5,
        )
    assert "value of image_dimensionality" in str(excinfo.value)
