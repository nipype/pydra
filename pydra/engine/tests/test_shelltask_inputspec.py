# -*- coding: utf-8 -*-

import attr
import typing as ty
import os, sys
import pytest
from pathlib import Path


from ..task import ShellCommandTask
from ..specs import ShellOutSpec, ShellSpec, SpecInfo, File


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
        executable="executable", inpA="inpA", input_spec=my_input_spec
    )
    assert shelly.cmdline == "executable -v inpA"


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


def test_shell_cmd_inputs_di(tmpdir):
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
                    type=File,
                    metadata={
                        "help_string": "If a mask image is specified, denoising is only performed in the mask region.",
                        "argstr": "-x",
                    },
                ),
            ),
            (
                "noise_model",
                attr.ib(
                    type=int,
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
                    type=ty.Any,
                    metadata={
                        "help_string": """
                        The output consists of the noise corrected version of the input image.
                        Optionally, one can also output the estimated noise image.
                        """,
                        "argstr": "-o",
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
                    type=bool,
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

    my_input_file = tmpdir.join("a_file.txt")
    my_input_file.write("content")

    # no input provided
    shelly = ShellCommandTask(executable="DenoiseImage", input_spec=my_input_spec)
    with pytest.raises(Exception) as e:
        shelly.cmdline
    assert "mandatory" in str(e.value)

    # input file name
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        input_spec=my_input_spec,
    )
    assert shelly.cmdline == f"DenoiseImage -i {my_input_file} -s 1 -p 1 -r 2"

    # input file name and help_short
    shelly = ShellCommandTask(
        executable="DenoiseImage",
        inputImageFilename=my_input_file,
        help_short=True,
        input_spec=my_input_spec,
    )
    assert shelly.cmdline == f"DenoiseImage -i {my_input_file} -s 1 -p 1 -r 2 -h"
