import typing as ty
from pathlib import Path
import attr
import pytest
from pydra.engine.specs import ShellOutputs, ShellDef
from fileformats.generic import File
from pydra.design import shell


def test_shell_cmd_execargs_1():
    # separate command into exec + args
    shelly = shell.define(["executable", "arg"])
    assert shelly.cmdline == "executable arg"
    assert shelly.name == "ShellTask_noname"


def test_shell_cmd_execargs_2():
    # separate command into exec + args
    shelly = shell.define(["cmd_1", "cmd_2", "arg"])
    assert shelly.cmdline == "cmd_1 cmd_2 arg"


def test_shell_cmd_inputs_1():
    """additional input with provided position"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"
        inpA: str = shell.arg(position=1, help="inp1", argstr="")

    shelly = ShellDef(
        executable="executable",
        additional_args=["arg"],
        inpA="inp1",
    )
    assert shelly.cmdline == "executable inp1 arg"


def test_shell_cmd_inputs_1a():
    """additional input without provided position"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"
        inpA: str = shell.arg(help="inpA", argstr="")

    shelly = ShellDef(
        executable="executable",
        additional_args=["arg"],
        inpA="inpNone1",
    )
    # inp1 should be the first one after executable
    assert shelly.cmdline == "executable inpNone1 arg"


def test_shell_cmd_inputs_1b():
    """additional input with negative position"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=-1, help="inpA", argstr="")

    # separate command into exec + args
    shelly = ShellDef(
        executable="executable",
        additional_args=["arg"],
        inpA="inp-1",
    )
    # inp1 should be last before arg
    assert shelly.cmdline == "executable inp-1 arg"


def test_shell_cmd_inputs_1_st():
    """additional input with provided position, checking cmdline when splitter"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=1, help="inp1", argstr="")

    ShellDef(
        name="shelly",
        executable="executable",
        additional_args=["arg"],
    ).split("inpA", inpA=["inp1", "inp2"])
    # cmdline should be a list
    # assert shelly.cmdline[0] == "executable inp1 arg"
    # assert shelly.cmdline[1] == "executable inp2 arg"


def test_shell_cmd_inputs_2():
    """additional inputs with provided positions"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=2, help="inpA", argstr="")
        inpB: str = shell.arg(position=1, help="inpN", argstr="")

    # separate command into exec + args
    shelly = ShellDef(
        executable="executable",
        inpB="inp1",
        inpA="inp2",
    )
    assert shelly.cmdline == "executable inp1 inp2"


def test_shell_cmd_inputs_2a():
    """additional inputs without provided positions"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(help="inpA", argstr="")
        inpB: str = shell.arg(help="inpB", argstr="")

    # separate command into exec + args
    shelly = ShellDef(
        executable="executable",
        inpA="inpNone1",
        inpB="inpNone2",
    )
    # position taken from the order in input definition
    assert shelly.cmdline == "executable inpNone1 inpNone2"


def test_shell_cmd_inputs_2_err():
    """additional inputs with provided positions (exception due to the duplication)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=1, help="inpA", argstr="")
        inpB: str = shell.arg(position=1, help="inpB", argstr="")

    shelly = ShellDef(
        executable="executable",
        inpA="inp1",
        inpB="inp2",
    )
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "1 is already used" in str(e.value)


def test_shell_cmd_inputs_2_noerr():
    """additional inputs with provided positions
    (duplication of the position doesn't lead to error, since only one field has value)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=1, help="inpA", argstr="")
        inpB: str = shell.arg(position=1, help="inpB", argstr="")

    shelly = ShellDef(
        executable="executable",
        inpA="inp1",
    )
    shelly.cmdline


def test_shell_cmd_inputs_3():
    """additional inputs: positive pos, negative pos and  no pos"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=1, help="inpA", argstr="")
        inpB: str = shell.arg(position=-1, help="inpB", argstr="")
        inpC: str = shell.arg(help="inpC", argstr="")

    # separate command into exec + args
    shelly = ShellDef(
        executable="executable",
        inpA="inp1",
        inpB="inp-1",
        inpC="inpNone",
    )
    # input without position should be between positive an negative positions
    assert shelly.cmdline == "executable inp1 inpNone inp-1"


def test_shell_cmd_inputs_argstr_1():
    """additional string inputs with argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(position=1, help="inpA", argstr="-v")

    shelly = ShellDef(
        executable="executable",
        inpA="inp1",
    )
    # flag used before inp1
    assert shelly.cmdline == "executable -v inp1"


def test_shell_cmd_inputs_argstr_2():
    """additional bool inputs with argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: bool = shell.arg(position=1, help="inpA", argstr="-v")

    # separate command into exec + args
    shelly = ShellDef(
        executable="executable",
        additional_args=["arg"],
        inpA=True,
    )
    # a flag is used without any additional argument
    assert shelly.cmdline == "executable -v arg"


def test_shell_cmd_inputs_list_1():
    """providing list as an additional input, no sep, no argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="")

    shelly = ShellDef(
        executable="executable",
        inpA=["el_1", "el_2", "el_3"],
    )
    # multiple elements
    assert shelly.cmdline == "executable el_1 el_2 el_3"


def test_shell_cmd_inputs_list_2():
    """providing list as an additional input, no sep, but argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="-v")

    shelly = ShellDef(
        executable="executable",
        inpA=["el_1", "el_2", "el_3"],
    )
    assert shelly.cmdline == "executable -v el_1 el_2 el_3"


def test_shell_cmd_inputs_list_3():
    """providing list as an additional input, no sep, argstr with ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="-v...")

    shelly = ShellDef(
        executable="executable",
        inpA=["el_1", "el_2", "el_3"],
    )
    # a flag is repeated
    assert shelly.cmdline == "executable -v el_1 -v el_2 -v el_3"


def test_shell_cmd_inputs_list_sep_1():
    """providing list as an additional input:, sep, no argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa", "bbb", "ccc"],
    )
    # separated by commas
    assert shelly.cmdline == "executable aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_2():
    """providing list as an additional input:, sep, and argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa", "bbb", "ccc"],
    )
    # a flag is used once
    assert shelly.cmdline == "executable -v aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_2a():
    """providing list as an additional input:, sep, and argstr with f-string"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v {inpA}",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa", "bbb", "ccc"],
    )
    # a flag is used once
    assert shelly.cmdline == "executable -v aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_3():
    """providing list as an additional input:, sep, argstr with ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v...",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa", "bbb", "ccc"],
    )
    # a flag is repeated
    assert shelly.cmdline == "executable -v aaa, -v bbb, -v ccc"


def test_shell_cmd_inputs_list_sep_3a():
    """providing list as an additional input:, sep, argstr with ... and f-string"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v {inpA}...",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa", "bbb", "ccc"],
    )
    # a flag is repeated
    assert shelly.cmdline == "executable -v aaa, -v bbb, -v ccc"


def test_shell_cmd_inputs_sep_4():
    """providing 1-el list as an additional input:, sep, argstr with ...,"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v...",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["aaa"],
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_sep_4a():
    """providing str instead of list as an additional input:, sep, argstr with ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v...",
        )

    shelly = ShellDef(
        executable="executable",
        inpA="aaa",
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_format_1():
    """additional inputs with argstr that has string formatting"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="-v {inpA}",
        )

    shelly = ShellDef(
        executable="executable",
        inpA="aaa",
    )
    assert shelly.cmdline == "executable -v aaa"


def test_shell_cmd_inputs_format_2():
    """additional inputs with argstr that has string formatting and ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: MultiInputObj[str] = shell.arg(
            position=1,
            help="inpA",
            argstr="-v {inpA}...",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=["el_1", "el_2"],
    )
    assert shelly.cmdline == "executable -v el_1 -v el_2"


def test_shell_cmd_inputs_format_3():
    """adding float formatting for argstr with input field"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: float = shell.arg(
            position=1,
            help="inpA",
            argstr="-v {inpA:.5f}",
        )

    shelly = ShellDef(
        executable="executable",
        inpA=0.007,
    )
    assert shelly.cmdline == "executable -v 0.00700"


def test_shell_cmd_inputs_mandatory_1():
    """additional inputs with mandatory=True"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
            mandatory=True,
        )

    shelly = ShellDef(
        executable="executable",
    )
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value)


def test_shell_cmd_inputs_not_given_1():
    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        arg1: MultiInputObj = (
            shell.arg(
                argstr="--arg1",
                help="Command line argument 1",
            ),
        )
        arg2: MultiInputObj = (
            shell.arg(
                argstr="--arg2",
                help="Command line argument 2",
            ),
        )
        arg3: File = (
            shell.arg(
                argstr="--arg3",
                help="Command line argument 3",
            ),
        )

    shelly = ShellDef(
        name="shelly",
        executable="executable",
    )

    shelly.definition.arg2 = "argument2"

    assert shelly.cmdline == "executable --arg2 argument2"


def test_shell_cmd_inputs_template_1():
    """additional inputs, one uses output_file_template (and argstr)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_out",
            ),
        )

    shelly = ShellDef(executable="executable", inpA="inpA")
    # outA has argstr in the metadata fields, so it's a part of the command line
    # the full path will be use din the command line
    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"
    # checking if outA in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA"]


def test_shell_cmd_inputs_template_1a():
    """additional inputs, one uses output_file_template (without argstr)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                help="outA",
                output_file_template="{inpA}_out",
            ),
        )

    shelly = ShellDef(executable="executable", inpA="inpA")
    # outA has no argstr in metadata, so it's not a part of the command line
    assert shelly.cmdline == "executable inpA"


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_2():
    """additional inputs, one uses output_file_template (and argstr, but input not provided)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpB: str = shell.arg(position=1, help="inpB", argstr="")
        outB: str = (
            shell.arg(
                position=2,
                help="outB",
                argstr="-o",
                output_file_template="{inpB}_out",
            ),
        )

    shelly = ShellDef(
        executable="executable",
    )
    # inpB not in the inputs, so no outB in the command line
    assert shelly.cmdline == "executable"
    # checking if outB in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outB"]


def test_shell_cmd_inputs_template_3(tmp_path):
    """additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    """
    inpA = tmp_path / "inpA"
    inpB = tmp_path / "inpB"
    Path.touch(inpA)
    Path.touch(inpB)

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpB: str = (
            shell.arg(
                position=2,
                help="inpB",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                help="outA",
                output_file_template="{inpA}_out",
            ),
        )
        outB: str = (
            shell.arg(
                help="outB",
                output_file_template="{inpB}_out",
            ),
        )
        outAB: str = (
            shell.arg(
                position=-1,
                help="outAB",
                argstr="-o {outA} {outB}",
                readonly=True,
            ),
        )

    shelly = ShellDef(executable="executable", inpA=inpA, inpB=inpB)
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA'} {tmp_path / 'inpB'} -o {shelly.output_dir / 'inpA_out'} {str(shelly.output_dir / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


def test_shell_cmd_inputs_template_3a():
    """additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    testing a different order within the input definition
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpB: str = (
            shell.arg(
                position=2,
                help="inpB",
                argstr="",
                mandatory=True,
            ),
        )
        outAB: str = (
            shell.arg(
                position=-1,
                help="outAB",
                argstr="-o {outA} {outB}",
                readonly=True,
            ),
        )
        outA: str = (
            shell.arg(
                help="outA",
                output_file_template="{inpA}_out",
            ),
        )
        outB: str = (
            shell.arg(
                help="outB",
                output_file_template="{inpB}_out",
            ),
        )

    shelly = ShellDef(executable="executable", inpA="inpA", inpB="inpB")
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable inpA inpB -o {shelly.output_dir / 'inpA_out'} {str(shelly.output_dir / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_4():
    """additional inputs with output_file_template and an additional
    read-only fields that combine two outputs together in the command line
    one output_file_template can't be resolved - no inpB is provided
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpB: str = shell.arg(position=2, help="inpB", argstr="")
        outAB: str = (
            shell.arg(
                position=-1,
                help="outAB",
                argstr="-o {outA} {outB}",
                readonly=True,
            ),
        )
        outA: str = (
            shell.arg(
                help="outA",
                output_file_template="{inpA}_out",
            ),
        )
        outB: str = (
            shell.arg(
                help="outB",
                output_file_template="{inpB}_out",
            ),
        )

    shelly = ShellDef(executable="executable", inpA="inpA")
    # inpB is not provided so outB not in the command line
    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA", "outB"]


def test_shell_cmd_inputs_template_5_ex():
    """checking if the exception is raised for read-only fields when input is set"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        outAB: str = shell.arg(
            position=-1,
            help="outAB",
            argstr="-o",
            readonly=True,
        )

    shelly = ShellDef(executable="executable", outAB="outAB")
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "read only" in str(e.value)


def test_shell_cmd_inputs_template_6():
    """additional inputs with output_file_template that has type ty.Union[str, bool]
    no default is set, so if nothing is provided as an input, the output  is used
    whenever the template can be formatted
    (the same way as for templates that has type=str)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: ty.Union[str, bool] = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_out",
            ),
        )

    # no input for outA (and no default value), so the output is created whenever the
    # template can be formatted (the same way as for templates that has type=str)
    shelly = ShellDef(executable="executable", inpA="inpA")
    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"

    # a string is provided for outA, so this should be used as the outA value
    shelly = ShellDef(executable="executable", inpA="inpA", outA="outA")
    assert shelly.cmdline == "executable inpA -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = ShellDef(executable="executable", inpA="inpA", outA=True)
    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = ShellDef(executable="executable", inpA="inpA", outA=False)
    assert shelly.cmdline == "executable inpA"


def test_shell_cmd_inputs_template_6a():
    """additional inputs with output_file_template that has type ty.Union[str, bool]
    and default is set to False,
    so if nothing is provided as an input, the output is not used
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: ty.Union[str, bool] = (
            shell.arg(
                default=False,
                position=2,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_out",
            ),
        )

    # no input for outA, but default is False, so the outA shouldn't be used
    shelly = ShellDef(executable="executable", inpA="inpA")
    assert shelly.cmdline == "executable inpA"

    # a string is provided for outA, so this should be used as the outA value
    shelly = ShellDef(executable="executable", inpA="inpA", outA="outA")
    assert shelly.cmdline == "executable inpA -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = ShellDef(executable="executable", inpA="inpA", outA=True)
    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = ShellDef(executable="executable", inpA="inpA", outA=False)
    assert shelly.cmdline == "executable inpA"


def test_shell_cmd_inputs_template_7(tmp_path: Path):
    """additional inputs uses output_file_template with a suffix (no extension)
    no keep_extension is used
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="",
                output_file_template="{inpA}_out",
            ),
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = ShellDef(executable="executable", inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {shelly.output_dir / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_7a(tmp_path: Path):
    """additional inputs uses output_file_template with a suffix (no extension)
    keep_extension is True (as default)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="",
                keep_extension=True,
                output_file_template="{inpA}_out",
            ),
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = ShellDef(executable="executable", inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {shelly.output_dir / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_7b(tmp_path: Path):
    """additional inputs uses output_file_template with a suffix (no extension)
    keep extension is False (so the extension is removed when creating the output)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="",
                keep_extension=False,
                output_file_template="{inpA}_out",
            ),
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = ShellDef(executable="executable", inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {shelly.output_dir / 'a_file_out'}"
    )


def test_shell_cmd_inputs_template_8(tmp_path: Path):
    """additional inputs uses output_file_template with a suffix and an extension"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="",
                output_file_template="{inpA}_out.txt",
            ),
        )

    inpA_file = tmp_path / "a_file.t"
    inpA_file.write_text("content")
    shelly = ShellDef(executable="executable", inpA=inpA_file)

    # outA should be formatted in a way that inpA extension is removed and the template extension is used
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.t'} {shelly.output_dir / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_9(tmp_path: Path):
    """additional inputs, one uses output_file_template with two fields:
    one File and one ints - the output should be recreated from the template
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpInt: int = (
            shell.arg(
                position=2,
                help="inp int",
                argstr="-i",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=3,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_{inpInt}_out.txt",
            ),
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = ShellDef(executable="executable", inpA=inpA_file, inpInt=3)

    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA.t'} -i 3 -o {shelly.output_dir / 'inpA_3_out.txt'}"
    )
    # checking if outA in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA"]


def test_shell_cmd_inputs_template_9a(tmp_path: Path):
    """additional inputs, one uses output_file_template with two fields:
    one file and one string without extension - should be fine
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpStr: str = (
            shell.arg(
                position=2,
                help="inp str",
                argstr="-i",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=3,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_{inpStr}_out.txt",
            ),
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = ShellDef(executable="executable", inpA=inpA_file, inpStr="hola")

    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA.t'} -i hola -o {shelly.output_dir / 'inpA_hola_out.txt'}"
    )
    # checking if outA in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA"]


def test_shell_cmd_inputs_template_9b_err(tmp_path: Path):
    """output_file_template with two fields that are both Files,
    an exception should be raised
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpFile: File = (
            shell.arg(
                position=2,
                help="inp file",
                argstr="-i",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=3,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_{inpFile}_out.txt",
            ),
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    inpFile_file = tmp_path / "inpFile.t"
    inpFile_file.write_text("content")

    shelly = ShellDef(
        executable="executable",
        inpA=inpA_file,
        inpFile=inpFile_file,
    )
    # the template has two files so the exception should be raised
    with pytest.raises(Exception, match="can't have multiple paths"):
        shelly.cmdline


def test_shell_cmd_inputs_template_9c_err(tmp_path: Path):
    """output_file_template with two fields: a file and a string with extension,
    that should be used as an additional file and the exception should be raised
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: File = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpStr: str = (
            shell.arg(
                position=2,
                help="inp str with extension",
                argstr="-i",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=3,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_{inpStr}_out.txt",
            ),
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = ShellDef(
        executable="executable",
        inpA=inpA_file,
        inpStr="hola.txt",
    )
    # inptStr has an extension so should be treated as a second file in the template formatting
    # and the exception should be raised
    with pytest.raises(Exception, match="can't have multiple paths"):
        shelly.cmdline


def test_shell_cmd_inputs_template_10():
    """output_file_template uses a float field with formatting"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: float = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="{inpA:.1f}",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template="file_{inpA:.1f}_out",
            ),
        )

    shelly = ShellDef(executable="executable", inpA=3.3456)
    # outA has argstr in the metadata fields, so it's a part of the command line
    # the full path will be use din the command line
    assert shelly.cmdline == f"executable 3.3 -o {shelly.output_dir / 'file_3.3_out'}"
    # checking if outA in the output fields
    assert shelly.output_names == ["return_code", "stdout", "stderr", "outA"]


def test_shell_cmd_inputs_template_requires_1():
    """Given an input definition with a templated output file subject to required fields,
    ensure the field is set only when all requirements are met."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        in_file: str = (
            shell.arg(
                help="input file",
                mandatory=True,
                argstr="",
            ),
        )
        with_tpl: bool = shell.arg(help="enable template")
        out_file: str = (
            shell.arg(
                help="output file",
                argstr="--tpl",
                output_file_template="tpl.{in_file}",
                requires={"with_tpl"},
            ),
        )

    # When requirements are not met.
    shelly = ShellDef(executable="cmd", in_file="in.file")
    assert "--tpl" not in shelly.cmdline

    # When requirements are met.
    shelly.definition.with_tpl = True
    assert "tpl.in.file" in shelly.cmdline


def test_shell_cmd_inputs_template_function_1():
    """one input field uses output_file_template that is a simple function
    this can be easily done by simple template as in test_shell_cmd_inputs_template_1
    """

    # a function that return an output template
    def template_fun(inputs):
        return "{inpA}_out"

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template=template_fun,
            ),
        )

    shelly = ShellDef(executable="executable", inpA="inpA")

    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_out'}"


def test_shell_cmd_inputs_template_function_2():
    """one input field uses output_file_template that is a function,
    depending on a value of an input it returns different template
    """

    # a function that return an output template that depends on value of the input
    def template_fun(inputs):
        if inputs.inpB % 2 == 0:
            return "{inpA}_even"
        else:
            return "{inpA}_odd"

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        inpB: int = (
            shell.arg(
                help="inpB",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template=template_fun,
            ),
        )

    shelly = ShellDef(
        executable="executable",
        inpA="inpA",
        inpB=1,
    )

    assert shelly.cmdline == f"executable inpA -o {shelly.output_dir / 'inpA_odd'}"


def test_shell_cmd_inputs_template_1_st():
    """additional inputs, one uses output_file_template (and argstr)
    testing cmdline when splitter defined
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: str = (
            shell.arg(
                position=1,
                help="inpA",
                argstr="",
                mandatory=True,
            ),
        )
        outA: str = (
            shell.arg(
                position=2,
                help="outA",
                argstr="-o",
                output_file_template="{inpA}_out",
            ),
        )

    inpA = ["inpA_1", "inpA_2"]
    ShellDef(
        name="f",
        executable="executable",
    ).split("inpA", inpA=inpA)

    # cmdline_list = shelly.cmdline
    # assert len(cmdline_list) == 2
    # for i in range(2):
    #     path_out = Path(shelly.output_dir[i]) / f"{inpA[i]}_out"
    #     assert cmdline_list[i] == f"executable {inpA[i]} -o {path_out}"


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_denoise_image(
    tmp_path,
):
    """example from #279"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        image_dimensionality: int = shell.arg(
            help="""
                    2/3/4
                    This option forces the image to be treated as a specified-dimensional image.
                    If not specified, the program tries to infer the dimensionality from
                    the input image.
                    """,
            allowed_values=[2, 3, 4],
            argstr="-d",
        )
        inputImageFilename: File = shell.arg(
            help="A scalar image is expected as input for noise correction.",
            argstr="-i",
            mandatory=True,
        )
        noise_model: str = shell.arg(
            help=""" Rician/(Gaussian) Employ a Rician or Gaussian noise model. """,
            allowed_values=["Rician", "Gaussian"],
            argstr="-n",
        )
        maskImageFilename: str = shell.arg(
            help="If a mask image is specified, denoising is only performed in the mask region.",
            argstr="-x",
        )
        shrink_factor: int = shell.arg(
            default=1,
            help=""" (1)/2/3/... Running noise correction on large images can be time consuming. To lessen computation time, the input image can be resampled. The shrink factor, specified as a single integer, describes this resampling. Shrink factor = 1 is the default. """,
            argstr="-s",
        )
        patch_radius: int = shell.arg(
            default=1,
            help="Patch radius. Default = 1x1x1",
            argstr="-p",
        )
        search_radius: int = shell.arg(
            default=2,
            help="Search radius. Default = 2x2x2.",
            argstr="-r",
        )
        correctedImage: str = shell.arg(
            help=""" The output consists of the noise corrected version of the input image. Optionally, one can also output the estimated noise image. """,
            output_file_template="{inputImageFilename}_out",
        )
        noiseImage: ty.Union[str, bool] = shell.arg(
            default=False,
            help=""" The output consists of the noise corrected version of the input image. Optionally, one can also output the estimated noise image. """,
            output_file_template="{inputImageFilename}_noise",
        )
        output: str = shell.arg(
            help="Combined output",
            argstr="-o [{correctedImage}, {noiseImage}]",
            position=-1,
            readonly=True,
        )
        version: bool = shell.arg(
            default=False,
            help="Get Version Information.",
            argstr="--version",
        )
        verbose: int = shell.arg(default=0, help="(0)/1. Verbose output. ", argstr="-v")
        help_short: bool = shell.arg(
            default=False,
            help="Print the help menu (short version)",
            argstr="-h",
        )
        help: int = shell.arg(
            help="Print the help menu.",
            argstr="--help",
        )

    my_input_file = tmp_path / "a_file.ext"
    my_input_file.write_text("content")

    # no input provided
    shelly = ShellDef(
        executable="DenoiseImage",
    )
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value)

    # input file name, noiseImage is not set, so using default value False
    shelly = ShellDef(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -o [{shelly.output_dir / 'a_file_out.ext'}]"
    )

    # input file name, noiseImage is set to True, so template is used in the output
    shelly = ShellDef(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        noiseImage=True,
    )
    assert (
        shelly.cmdline == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 "
        f"-o [{shelly.output_dir / 'a_file_out.ext'}, {str(shelly.output_dir / 'a_file_noise.ext')}]"
    )

    # input file name and help_short
    shelly = ShellDef(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        help_short=True,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -h -o [{shelly.output_dir / 'a_file_out.ext'}]"
    )

    assert shelly.output_names == [
        "return_code",
        "stdout",
        "stderr",
        "correctedImage",
        "noiseImage",
    ]

    # adding image_dimensionality that has allowed_values [2, 3, 4]
    shelly = ShellDef(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        image_dimensionality=2,
    )
    assert (
        shelly.cmdline
        == f"DenoiseImage -d 2 -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -o [{shelly.output_dir / 'a_file_out.ext'}]"
    )

    # adding image_dimensionality that has allowed_values [2, 3, 4] and providing 5 - exception should be raised
    with pytest.raises(ValueError) as excinfo:
        shelly = ShellDef(
            executable="DenoiseImage",
            inputImageFilename=my_input_file,
            image_dimensionality=5,
        )
    assert "value of image_dimensionality" in str(excinfo.value)


# tests with XOR in input metadata


@shell.define
class SimpleTaskXor(ShellDef["SimpleTaskXor.Outputs"]):

    input_1: str = shell.arg(
        help="help",
        xor=("input_1", "input_2", "input_3"),
    )
    input_2: bool = shell.arg(
        help="help",
        argstr="--i2",
        xor=("input_1", "input_2", "input_3"),
    )
    input_3: bool = shell.arg(
        help="help",
        xor=("input_1", "input_2", "input_3"),
    )

    @shell.outputs
    class Outputs(ShellOutputs):
        pass

    executable = "cmd"


def test_task_inputs_mandatory_with_xOR_one_mandatory_is_OK():
    """input definition with mandatory inputs"""
    task = SimpleTaskXor()
    task.definition.input_1 = "Input1"
    task.definition.input_2 = attr.NOTHING
    task.definition.check_fields_input_spec()


def test_task_inputs_mandatory_with_xOR_one_mandatory_out_3_is_OK():
    """input definition with mandatory inputs"""
    task = SimpleTaskXor()
    task.definition.input_1 = attr.NOTHING
    task.definition.input_2 = attr.NOTHING
    task.definition.input_3 = True
    task.definition.check_fields_input_spec()


def test_task_inputs_mandatory_with_xOR_zero_mandatory_raises_error():
    """input definition with mandatory inputs"""
    task = SimpleTaskXor()
    task.definition.input_1 = attr.NOTHING
    task.definition.input_2 = attr.NOTHING
    with pytest.raises(Exception) as excinfo:
        task.definition.check_fields_input_spec()
    assert "input_1 is mandatory" in str(excinfo.value)
    assert "no alternative provided by ['input_2', 'input_3']" in str(excinfo.value)
    assert excinfo.type is AttributeError


def test_task_inputs_mandatory_with_xOR_two_mandatories_raises_error():
    """input definition with mandatory inputs"""
    task = SimpleTaskXor()
    task.definition.input_1 = "Input1"
    task.definition.input_2 = True

    with pytest.raises(Exception) as excinfo:
        task.definition.check_fields_input_spec()
    assert "input_1 is mutually exclusive with ['input_2']" in str(excinfo.value)
    assert excinfo.type is AttributeError


def test_task_inputs_mandatory_with_xOR_3_mandatories_raises_error():
    """input definition with mandatory inputs"""
    task = SimpleTaskXor()
    task.definition.input_1 = "Input1"
    task.definition.input_2 = True
    task.definition.input_3 = False

    with pytest.raises(Exception) as excinfo:
        task.definition.check_fields_input_spec()
    assert "input_1 is mutually exclusive with ['input_2', 'input_3']" in str(
        excinfo.value
    )
    assert excinfo.type is AttributeError
