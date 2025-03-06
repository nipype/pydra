import typing as ty
from pathlib import Path
import attrs
import pytest
from pydra.engine.specs import ShellOutputs, ShellDef
from fileformats.generic import File
from pydra.design import shell
from pydra.utils.typing import MultiInputObj
from .utils import get_output_names


def test_shell_cmd_execargs_1():
    # separate command into exec + args
    Shelly = shell.define(["executable", "arg"])
    shelly = Shelly()
    assert shelly.cmdline == "executable arg"


def test_shell_cmd_execargs_2():
    # separate command into exec + args
    Shelly = shell.define(["cmd_1", "cmd_2", "arg"])
    shelly = Shelly()
    assert shelly.cmdline == "cmd_1 cmd_2 arg"


def test_shell_cmd_inputs_1():
    """additional input with provided position"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"
        inpA: str = shell.arg(position=1, help="inp1", argstr="")

    shelly = Shelly(
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

    shelly = Shelly(
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
    shelly = Shelly(
        additional_args=["arg"],
        inpA="inp-1",
    )
    # inp1 should be last before arg
    assert shelly.cmdline == "executable inp-1 arg"


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
    shelly = Shelly(
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
    shelly = Shelly(
        inpA="inpNone1",
        inpB="inpNone2",
    )
    # position taken from the order in input definition
    assert shelly.cmdline == "executable inpNone1 inpNone2"


def test_shell_cmd_inputs_2_err():
    """additional inputs with provided positions (exception due to the duplication)"""

    with pytest.raises(Exception) as e:

        @shell.define
        class Shelly(ShellDef["Shelly.Outputs"]):
            class Outputs(ShellOutputs):
                pass

            executable = "executable"

            inpA: str = shell.arg(position=1, help="inpA", argstr="")
            inpB: str = shell.arg(position=1, help="inpB", argstr="")

    assert "Multiple fields have the overlapping positions" in str(e.value)


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
    shelly = Shelly(
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

    shelly = Shelly(inpA="inp1")
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
    shelly = Shelly(additional_args=["arg"], inpA=True)
    # a flag is used without any additional argument
    assert shelly.cmdline == "executable -v arg"


def test_shell_cmd_inputs_list_1():
    """providing list as an additional input, no sep, no argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="", sep=" ")

    shelly = Shelly(inpA=["el_1", "el_2", "el_3"])
    # multiple elements
    assert shelly.cmdline == "executable el_1 el_2 el_3"


def test_shell_cmd_inputs_list_2():
    """providing list as an additional input, no sep, but argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="-v", sep=" ")

    shelly = Shelly(inpA=["el_1", "el_2", "el_3"])
    assert shelly.cmdline == "executable -v el_1 el_2 el_3"


def test_shell_cmd_inputs_list_3():
    """providing list as an additional input, no sep, argstr with ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: ty.List[str] = shell.arg(position=2, help="inpA", argstr="-v...", sep=" ")

    shelly = Shelly(inpA=["el_1", "el_2", "el_3"])
    # a flag is repeated
    assert shelly.cmdline == "executable -v el_1 -v el_2 -v el_3"


def test_shell_cmd_inputs_list_sep_1():
    """providing list as an additional input:, sep, no argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: list[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="",
        )

    shelly = Shelly(inpA=["aaa", "bbb", "ccc"])
    # separated by commas
    assert shelly.cmdline == "executable aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_2():
    """providing list as an additional input:, sep, and argstr"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: list[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v",
        )

    shelly = Shelly(inpA=["aaa", "bbb", "ccc"])
    # a flag is used once
    assert shelly.cmdline == "executable -v aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_2a():
    """providing list as an additional input:, sep, and argstr with f-string"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: list[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v {inpA}",
        )

    shelly = Shelly(inpA=["aaa", "bbb", "ccc"])
    # a flag is used once
    assert shelly.cmdline == "executable -v aaa,bbb,ccc"


def test_shell_cmd_inputs_list_sep_3():
    """providing list as an additional input:, sep, argstr with ..."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: list[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v...",
        )

    shelly = Shelly(inpA=["aaa", "bbb", "ccc"])
    # a flag is repeated
    assert shelly.cmdline == "executable -v aaa, -v bbb, -v ccc"


def test_shell_cmd_inputs_list_sep_3a():
    """providing list as an additional input:, sep, argstr with ... and f-string"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        inpA: list[str] = shell.arg(
            position=1,
            help="inpA",
            sep=",",
            argstr="-v {inpA}...",
        )

    shelly = Shelly(inpA=["aaa", "bbb", "ccc"])
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
            argstr="-v...",
        )

    shelly = Shelly(inpA=["aaa"])
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
            argstr="-v...",
        )

    shelly = Shelly(inpA="aaa")
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

    shelly = Shelly(inpA="aaa")
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

    shelly = Shelly(inpA=["el_1", "el_2"])
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

    shelly = Shelly(inpA=0.007)
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
        )

    shelly = Shelly()
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value).lower()


def test_shell_cmd_inputs_not_given_1():
    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            pass

        executable = "executable"

        arg1: MultiInputObj = shell.arg(
            argstr="--arg1",
            default=attrs.Factory(list),
            help="Command line argument 1",
        )
        arg2: MultiInputObj = shell.arg(
            argstr="--arg2",
            help="Command line argument 2",
        )
        arg3: File | None = shell.arg(
            argstr="--arg3",
            default=None,
            help="Command line argument 3",
        )

    shelly = Shelly()

    shelly.arg2 = "argument2"

    assert shelly.cmdline == "executable --arg2 argument2"


def test_shell_cmd_inputs_template_1():
    """additional inputs, one uses path_template (and argstr)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    shelly = Shelly(inpA="inpA")
    # outA has argstr in the metadata fields, so it's a part of the command line
    # the full path will be use din the command line
    assert shelly.cmdline == f"executable inpA -o {Path.cwd() / 'inpA_out'}"
    # checking if outA in the output fields
    assert get_output_names(shelly) == ["outA", "return_code", "stderr", "stdout"]


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_2():
    """additional inputs, one uses path_template (and argstr, but input not provided)"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outB: File | None = shell.outarg(
                position=2,
                help="outB",
                argstr="-o",
                path_template="{inpB}_out",
            )

        executable = "executable"

        inpB: File | None = shell.arg(position=1, help="inpB", argstr="", default=None)

    shelly = Shelly()
    # inpB not in the inputs, so no outB in the command line
    assert shelly.cmdline == "executable"
    # checking if outB in the output fields
    assert get_output_names(shelly) == ["outB", "return_code", "stderr", "stdout"]


def test_shell_cmd_inputs_template_3(tmp_path):
    """additional inputs with path_template and an additional
    read-only fields that combine two outputs together in the command line
    """
    inpA = tmp_path / "inpA"
    inpB = tmp_path / "inpB"
    Path.touch(inpA)
    Path.touch(inpB)

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                help="outA",
                argstr=None,
                path_template="{inpA}_out",
            )
            outB: File = shell.outarg(
                help="outB",
                argstr=None,
                path_template="{inpB}_out",
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpB: str = shell.arg(
            position=2,
            help="inpB",
            argstr="",
        )
        outAB: str = shell.arg(
            position=-1,
            help="outAB",
            argstr="-o {outA} {outB}",
            readonly=True,
        )

    shelly = Shelly(inpA=inpA, inpB=inpB)
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA'} {tmp_path / 'inpB'} -o {Path.cwd() / 'inpA_out'} {str(Path.cwd() / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert get_output_names(shelly) == [
        "outA",
        "outB",
        "return_code",
        "stderr",
        "stdout",
    ]


def test_shell_cmd_inputs_template_3a():
    """additional inputs with path_template and an additional
    read-only fields that combine two outputs together in the command line
    testing a different order within the input definition
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                argstr=None,
                help="outA",
                path_template="{inpA}_out",
            )
            outB: File = shell.outarg(
                argstr=None,
                help="outB",
                path_template="{inpB}_out",
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpB: str = shell.arg(
            position=2,
            help="inpB",
            argstr="",
        )
        outAB: str = shell.arg(
            position=-1,
            help="outAB",
            argstr="-o {outA} {outB}",
            readonly=True,
        )

    shelly = Shelly(inpA="inpA", inpB="inpB")
    # using syntax from the outAB field
    assert (
        shelly.cmdline
        == f"executable inpA inpB -o {Path.cwd() / 'inpA_out'} {str(Path.cwd() / 'inpB_out')}"
    )
    # checking if outA and outB in the output fields (outAB should not be)
    assert get_output_names(shelly) == [
        "outA",
        "outB",
        "return_code",
        "stderr",
        "stdout",
    ]


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_template_4():
    """additional inputs with path_template and an additional
    read-only fields that combine two outputs together in the command line
    one path_template can't be resolved - no inpB is provided
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                argstr=None,
                help="outA",
                path_template="{inpA}_out",
            )
            outB: File | None = shell.outarg(
                argstr=None,
                help="outB",
                path_template="{inpB}_out",
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpB: str | None = shell.arg(position=2, help="inpB", argstr="", default=None)
        outAB: str = shell.arg(
            position=-1,
            help="outAB",
            argstr="-o {outA} {outB}",
            readonly=True,
        )

    shelly = Shelly(inpA="inpA")
    # inpB is not provided so outB not in the command line
    assert shelly.cmdline == f"executable inpA -o {Path.cwd() / 'inpA_out'}"
    assert get_output_names(shelly) == [
        "outA",
        "outB",
        "return_code",
        "stderr",
        "stdout",
    ]


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

    shelly = Shelly(outAB="outAB")
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "read only" in str(e.value)


def test_shell_cmd_inputs_template_6():
    """additional inputs with path_template that has type ty.Union[str, bool]
    no default is set, so if nothing is provided as an input, the output  is used
    whenever the template can be formatted
    (the same way as for templates that has type=str)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    # no input for outA (and no default value), so the output is created whenever the
    # template can be formatted (the same way as for templates that has type=str)
    inpA = File.mock("inpA")
    shelly = Shelly(inpA=inpA)

    inpA_path = Path.cwd() / "inpA"
    outA_path = Path.cwd() / "inpA_out"
    assert shelly.cmdline == f"executable {inpA_path} -o {outA_path}"

    # a string is provided for outA, so this should be used as the outA value
    shelly = Shelly(inpA=inpA, outA="outA")
    assert shelly.cmdline == f"executable {inpA_path} -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = Shelly(inpA=inpA, outA=True)
    assert shelly.cmdline == f"executable {inpA_path} -o {outA_path}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = Shelly(inpA=inpA, outA=False)
    assert shelly.cmdline == f"executable {inpA_path}"


def test_shell_cmd_inputs_template_6a():
    """additional inputs with path_template that has type ty.Union[str, bool]
    and default is set to False,
    so if nothing is provided as an input, the output is not used
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File | None = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    # no input for outA, but default is False, so the outA shouldn't be used
    shelly = Shelly(inpA="inpA")
    assert shelly.cmdline == "executable inpA"

    # a string is provided for outA, so this should be used as the outA value
    shelly = Shelly(inpA="inpA", outA="outA")
    assert shelly.cmdline == "executable inpA -o outA"

    # True is provided for outA, so the formatted template should be used as outA value
    shelly = Shelly(inpA="inpA", outA=True)
    assert shelly.cmdline == f"executable inpA -o {Path.cwd() / 'inpA_out'}"

    # False is provided for outA, so the outA shouldn't be used
    shelly = Shelly(inpA="inpA", outA=False)
    assert shelly.cmdline == "executable inpA"


def test_shell_cmd_inputs_template_7(tmp_path: Path):
    """additional inputs uses path_template with a suffix (no extension)
    no keep_extension is used
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="",
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = Shelly(inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {Path.cwd() / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_7a(tmp_path: Path):
    """additional inputs uses path_template with a suffix (no extension)
    keep_extension is True (as default)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="",
                keep_extension=True,
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = Shelly(inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {Path.cwd() / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_7b(tmp_path: Path):
    """additional inputs uses path_template with a suffix (no extension)
    keep extension is False (so the extension is removed when creating the output)
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="",
                keep_extension=False,
                path_template="{inpA}_out",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    inpA_file = tmp_path / "a_file.txt"
    inpA_file.write_text("content")
    shelly = Shelly(inpA=inpA_file)

    # outA should be formatted in a way that that .txt goes to the end
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.txt'} {Path.cwd() / 'a_file_out'}"
    )


def test_shell_cmd_inputs_template_8(tmp_path: Path):
    """additional inputs uses path_template with a suffix and an extension"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="",
                path_template="{inpA}_out.txt",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    inpA_file = tmp_path / "a_file.t"
    inpA_file.write_text("content")
    shelly = Shelly(inpA=inpA_file)

    # outA should be formatted in a way that inpA extension is removed and the template extension is used
    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'a_file.t'} {Path.cwd() / 'a_file_out.txt'}"
    )


def test_shell_cmd_inputs_template_9(tmp_path: Path):
    """additional inputs, one uses path_template with two fields:
    one File and one ints - the output should be recreated from the template
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=3,
                help="outA",
                argstr="-o",
                path_template="{inpA}_{inpInt}_out.txt",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpInt: int = shell.arg(
            position=2,
            help="inp int",
            argstr="-i",
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = Shelly(inpA=inpA_file, inpInt=3)

    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA.t'} -i 3 -o {Path.cwd() / 'inpA_3_out.txt'}"
    )
    # checking if outA in the output fields
    assert get_output_names(shelly) == ["outA", "return_code", "stderr", "stdout"]


def test_shell_cmd_inputs_template_9a(tmp_path: Path):
    """additional inputs, one uses path_template with two fields:
    one file and one string without extension - should be fine
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):
            outA: File = shell.outarg(
                position=3,
                help="outA",
                argstr="-o",
                path_template="{inpA}_{inpStr}_out.txt",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpStr: str = shell.arg(
            position=2,
            help="inp str",
            argstr="-i",
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = Shelly(inpA=inpA_file, inpStr="hola")

    assert (
        shelly.cmdline
        == f"executable {tmp_path / 'inpA.t'} -i hola -o {Path.cwd() / 'inpA_hola_out.txt'}"
    )
    # checking if outA in the output fields
    assert get_output_names(shelly) == ["outA", "return_code", "stderr", "stdout"]


def test_shell_cmd_inputs_template_9b_err(tmp_path: Path):
    """path_template with two fields that are both Files,
    an exception should be raised
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                position=3,
                help="outA",
                argstr="-o",
                path_template="{inpA}_{inpFile}_out.txt",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpFile: File = shell.arg(
            position=2,
            help="inp file",
            argstr="-i",
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    inpFile_file = tmp_path / "inpFile.t"
    inpFile_file.write_text("content")

    shelly = Shelly(
        inpA=inpA_file,
        inpFile=inpFile_file,
    )
    # the template has two files so the exception should be raised
    with pytest.raises(Exception, match="can't have multiple paths"):
        shelly.cmdline


def test_shell_cmd_inputs_template_9c_err(tmp_path: Path):
    """path_template with two fields: a file and a string with extension,
    that should be used as an additional file and the exception should be raised
    """

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                position=3,
                help="outA",
                argstr="-o",
                path_template="{inpA}_{inpStr}_out.txt",
            )

        executable = "executable"

        inpA: File = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpStr: str = shell.arg(
            position=2,
            help="inp str with extension",
            argstr="-i",
        )

    inpA_file = tmp_path / "inpA.t"
    inpA_file.write_text("content")

    shelly = Shelly(
        inpA=inpA_file,
        inpStr="hola.txt",
    )
    # inptStr has an extension so should be treated as a second file in the template formatting
    # and the exception should be raised
    with pytest.raises(Exception, match="can't have multiple paths"):
        shelly.cmdline


def test_shell_cmd_inputs_template_10():
    """path_template uses a float field with formatting"""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template="file_{inpA:.1f}_out",
            )

        executable = "executable"

        inpA: float = shell.arg(
            position=1,
            help="inpA",
            argstr="{inpA:.1f}",
        )

    shelly = Shelly(inpA=3.3456)
    # outA has argstr in the metadata fields, so it's a part of the command line
    # the full path will be use din the command line
    assert shelly.cmdline == f"executable 3.3 -o {Path.cwd() / 'file_3.3_out'}"
    # checking if outA in the output fields
    assert get_output_names(shelly) == ["outA", "return_code", "stderr", "stdout"]


def test_shell_cmd_inputs_template_requires_1():
    """Given an input definition with a templated output file subject to required fields,
    ensure the field is set only when all requirements are met."""

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            out_file: File | None = shell.outarg(
                help="output file",
                argstr="--tpl",
                path_template="tpl.{in_file}",
                requires={"with_tpl"},
            )

        executable = "executable"

        in_file: str = shell.arg(
            help="input file",
            argstr="",
        )
        with_tpl: bool = shell.arg(help="enable template", default=False)

    # When requirements are not met.
    shelly = Shelly(executable="cmd", in_file="in.file")
    assert "--tpl" not in shelly.cmdline

    # When requirements are met.
    shelly.with_tpl = True
    assert "tpl.in.file" in shelly.cmdline


def test_shell_cmd_inputs_template_function_1():
    """one input field uses path_template that is a simple function
    this can be easily done by simple template as in test_shell_cmd_inputs_template_1
    """

    # a function that return an output template
    def template_fun(inputs):
        return "{inpA}_out"

    @shell.define
    class Shelly(ShellDef["Shelly.Outputs"]):
        class Outputs(ShellOutputs):

            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template=template_fun,
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )

    shelly = Shelly(inpA="inpA")

    assert shelly.cmdline == f"executable inpA -o {Path.cwd() / 'inpA_out'}"


def test_shell_cmd_inputs_template_function_2():
    """one input field uses path_template that is a function,
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

            outA: File = shell.outarg(
                position=2,
                help="outA",
                argstr="-o",
                path_template=template_fun,
            )

        executable = "executable"

        inpA: str = shell.arg(
            position=1,
            help="inpA",
            argstr="",
        )
        inpB: int = shell.arg(
            help="inpB",
            argstr=None,
        )

    shelly = Shelly(
        inpA="inpA",
        inpB=1,
    )

    assert shelly.cmdline == f"executable inpA -o {Path.cwd() / 'inpA_odd'}"


# TODO: after deciding how we use requires/templates
def test_shell_cmd_inputs_denoise_image(
    tmp_path,
):
    """example from #279"""

    @shell.define
    class DenoiseImage(ShellDef["DenoiseImage.Outputs"]):
        class Outputs(ShellOutputs):

            correctedImage: File = shell.outarg(
                help="""
                    The output consists of the noise corrected version of the input image.
                    Optionally, one can also output the estimated noise image. """,
                path_template="{inputImageFilename}_out",
                argstr=None,
            )
            noiseImage: File | None = shell.outarg(
                help="""
                    The output consists of the noise corrected version of the input image.
                    Optionally, one can also output the estimated noise image. """,
                path_template="{inputImageFilename}_noise",
                argstr=None,
            )

        executable = "executable"

        image_dimensionality: int | None = shell.arg(
            help="""
                    2/3/4
                    This option forces the image to be treated as a specified-dimensional image.
                    If not specified, the program tries to infer the dimensionality from
                    the input image.
                    """,
            allowed_values=[2, 3, 4, None],
            default=None,
            argstr="-d",
        )
        inputImageFilename: File = shell.arg(
            help="A scalar image is expected as input for noise correction.",
            argstr="-i",
        )
        noise_model: str | None = shell.arg(
            default=None,
            help=""" Rician/(Gaussian) Employ a Rician or Gaussian noise model. """,
            allowed_values=["Rician", "Gaussian"],
            argstr="-n",
        )
        maskImageFilename: str | None = shell.arg(
            default=None,
            help="If a mask image is specified, denoising is only performed in the mask region.",
            argstr="-x",
        )
        shrink_factor: int = shell.arg(
            default=1,
            help="""
                (1)/2/3/...
                Running noise correction on large images can be time consuming.
                To lessen computation time, the input image can be resampled.
                The shrink factor, specified as a single integer, describes this
                resampling. Shrink factor = 1 is the default. """,
            argstr="-s",
        )
        patch_radius: int = shell.arg(
            default=1, help="Patch radius. Default = 1x1x1", argstr="-p", position=2
        )
        search_radius: int = shell.arg(
            default=2, help="Search radius. Default = 2x2x2.", argstr="-r", position=3
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
        help: int | None = shell.arg(
            default=None,
            help="Print the help menu.",
            argstr="--help",
        )

    my_input_file = tmp_path / "a_file.ext"
    my_input_file.write_text("content")

    # no input provided
    denoise_image = DenoiseImage(
        executable="DenoiseImage",
    )
    with pytest.raises(Exception) as e:
        denoise_image.cmdline
    assert "mandatory" in str(e.value).lower()

    # input file name, noiseImage is not set, so using default value False
    denoise_image = DenoiseImage(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
    )
    assert (
        denoise_image.cmdline
        == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -o [{Path.cwd() / 'a_file_out.ext'}]"
    )

    # input file name, noiseImage is set to True, so template is used in the output
    denoise_image = DenoiseImage(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        noiseImage=True,
    )
    assert (
        denoise_image.cmdline
        == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 "
        f"-o [{Path.cwd() / 'a_file_out.ext'}, {str(Path.cwd() / 'a_file_noise.ext')}]"
    )

    # input file name and help_short
    denoise_image = DenoiseImage(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        help_short=True,
    )
    assert (
        denoise_image.cmdline
        == f"DenoiseImage -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -h -o [{Path.cwd() / 'a_file_out.ext'}]"
    )

    assert get_output_names(denoise_image) == [
        "return_code",
        "stderr",
        "stdout",
        "correctedImage",
        "noiseImage",
    ]

    # adding image_dimensionality that has allowed_values [2, 3, 4]
    denoise_image = DenoiseImage(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        image_dimensionality=2,
    )
    assert (
        denoise_image.cmdline
        == f"DenoiseImage -d 2 -i {tmp_path / 'a_file.ext'} -s 1 -p 1 -r 2 -o [{Path.cwd() / 'a_file_out.ext'}]"
    )

    # adding image_dimensionality that has allowed_values [2, 3, 4] and providing 5 - exception should be raised
    with pytest.raises(ValueError) as excinfo:
        denoise_image = DenoiseImage(
            executable="DenoiseImage",
            inputImageFilename=my_input_file,
            image_dimensionality=5,
        )
    assert "value of image_dimensionality" in str(excinfo.value)


# tests with XOR in input metadata


@shell.define
class SimpleXor(ShellDef["SimpleTaskXor.Outputs"]):

    input_1: str | None = shell.arg(
        default=None,
        help="help",
        xor=("input_1", "input_2", "input_3"),
    )
    input_2: bool | None = shell.arg(
        default=None,
        help="help",
        argstr="--i2",
        xor=("input_1", "input_2", "input_3"),
    )
    input_3: bool | None = shell.arg(
        default=None,
        help="help",
        xor=("input_1", "input_2", "input_3"),
    )

    @shell.outputs
    class Outputs(ShellOutputs):
        pass

    executable = "cmd"


def test_task_inputs_mandatory_with_xOR_one_mandatory_is_OK():
    """input definition with mandatory inputs"""
    simple_xor = SimpleXor()
    simple_xor.input_1 = "Input1"
    simple_xor._check_rules()


def test_task_inputs_mandatory_with_xOR_one_mandatory_out_3_is_OK():
    """input definition with mandatory inputs"""
    simple_xor = SimpleXor()
    simple_xor.input_3 = True
    simple_xor._check_rules()


def test_task_inputs_mandatory_with_xOR_zero_mandatory_raises_error():
    """input definition with mandatory inputs"""
    simple_xor = SimpleXor()
    simple_xor.input_2 = False
    with pytest.raises(
        ValueError, match="At least one of the mutually exclusive fields should be set:"
    ):
        simple_xor._check_rules()


def test_task_inputs_mandatory_with_xOR_two_mandatories_raises_error():
    """input definition with mandatory inputs"""
    simple_xor = SimpleXor()
    simple_xor.input_1 = "Input1"
    simple_xor.input_2 = True

    with pytest.raises(
        ValueError, match="Mutually exclusive fields .* are set together"
    ):
        simple_xor._check_rules()


def test_task_inputs_mandatory_with_xOR_3_mandatories_raises_error():
    """input definition with mandatory inputs"""
    simple_xor = SimpleXor()
    simple_xor.input_1 = "Input1"
    simple_xor.input_2 = True
    simple_xor.input_3 = False

    with pytest.raises(
        ValueError,
        match=r".*Mutually exclusive fields \(input_1='Input1', input_2=True\) are set together",
    ):
        simple_xor._check_rules()
