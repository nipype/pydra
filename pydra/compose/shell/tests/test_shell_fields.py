import os
import typing as ty
from pathlib import Path
import attrs
import pytest
import cloudpickle as cp
from pydra.compose import shell
from pydra.utils.general import task_fields, task_help, wrap_text
from pydra.compose.shell.builder import _InputPassThrough
from fileformats.generic import File, Directory, FsObject
from fileformats import text, image
from pydra.utils.typing import MultiInputObj


def test_interface_template():

    Cp = shell.define("cp <in_path> <out|out_path>")

    assert issubclass(Cp, shell.Task)
    output = shell.outarg(
        name="out_path",
        path_template="out_path",
        type=FsObject,
        position=2,
    )
    assert sorted_fields(Cp) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(name="in_path", type=FsObject, position=1),
        output,
        shell.Task.append_args,
    ]
    assert sorted_fields(Cp.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]
    intf = Cp(in_path=File.mock("in-path.txt"))
    assert intf.executable == "cp"
    Cp(in_path=File.mock("in-path.txt"), out_path=Path("./out-path.txt"))
    Cp.Outputs(out_path=File.mock("in-path.txt"))


def test_executable_arg_fail():

    with pytest.raises(ValueError, match="The argument 'executable' is reserved"):

        shell.define("my-cmd <executable>")


def test_interface_template_w_types_and_path_template_ext():

    TrimPng = shell.define("trim-png <in_image:image/png> <out|out_image:image/png>")

    assert issubclass(TrimPng, shell.Task)
    output = shell.outarg(
        name="out_image",
        path_template="out_image.png",
        type=image.Png,
        position=2,
    )
    assert sorted_fields(TrimPng) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="trim-png",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(name="in_image", type=image.Png, position=1),
        output,
        shell.Task.append_args,
    ]
    assert sorted_fields(TrimPng.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]
    TrimPng(in_image=image.Png.mock())
    TrimPng(in_image=image.Png.mock(), out_image=Path("./new_image.png"))
    TrimPng.Outputs(out_image=image.Png.mock())


def test_interface_template_w_modify():

    TrimPng = shell.define("trim-png <modify|image:image/png>")

    assert issubclass(TrimPng, shell.Task)
    assert sorted_fields(TrimPng) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="trim-png",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(
            name="image", type=image.Png, position=1, copy_mode=File.CopyMode.copy
        ),
        shell.Task.append_args,
    ]
    assert sorted_fields(TrimPng.Outputs) == [
        shell.out(
            name="image",
            type=image.Png,
            callable=_InputPassThrough("image"),
        ),
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]
    TrimPng(image=image.Png.mock())
    TrimPng.Outputs(image=image.Png.mock())


def test_interface_template_more_complex():

    Cp = shell.define(
        (
            "cp <in_fs_objects:fs-object+> <out|out_dir:directory> "
            "-R<recursive> "
            "--text-arg <text_arg?> "
            "--int-arg <int_arg:int?> "
            "--tuple-arg <tuple_arg:int,str?> "
        ),
    )

    assert issubclass(Cp, shell.Task)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=2,
    )
    assert sorted_fields(Cp) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(
            name="in_fs_objects",
            type=MultiInputObj[FsObject],
            position=1,
        ),
        output,
        shell.arg(name="recursive", argstr="-R", type=bool, default=False, position=3),
        shell.arg(
            name="text_arg",
            argstr="--text-arg",
            type=str | None,
            default=None,
            position=4,
        ),
        shell.arg(
            name="int_arg",
            argstr="--int-arg",
            type=int | None,
            default=None,
            position=5,
        ),
        shell.arg(
            name="tuple_arg",
            argstr="--tuple-arg",
            type=tuple[int, str] | None,
            sep=" ",
            default=None,
            position=6,
        ),
        shell.Task.append_args,
    ]
    assert sorted_fields(Cp.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]
    Cp(in_fs_objects=[File.sample(), File.sample(seed=1)])
    Cp.Outputs(out_dir=Directory.sample())


def test_interface_template_with_overrides_and_optionals():

    RECURSIVE_HELP = (
        "If source_file designates a directory, cp copies the directory and the entire "
        "subtree connected at that point."
    )

    Cp = shell.define(
        (
            "cp <in_fs_objects:fs-object+> <out|out_dir:directory> <out|out_file:file?> "
            "-R<recursive> "
            "--text-arg <text_arg> "
            "--int-arg <int_arg:int?> "
            "--tuple-arg <tuple_arg:int,str> "
        ),
        inputs={"recursive": shell.arg(help=RECURSIVE_HELP)},
        outputs={
            "out_dir": shell.outarg(position=-2),
            "out_file": shell.outarg(position=-1),
        },
    )

    assert issubclass(Cp, shell.Task)
    outargs = [
        shell.outarg(
            name="out_dir",
            type=Directory,
            path_template="out_dir",
            position=-2,
        ),
        shell.outarg(
            name="out_file",
            type=File | None,
            default=None,
            path_template="out_file",
            position=-1,
        ),
    ]
    assert sorted_fields(Cp) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(name="in_fs_objects", type=MultiInputObj[FsObject], position=1),
        shell.arg(
            name="recursive",
            argstr="-R",
            type=bool,
            default=False,
            help=RECURSIVE_HELP,
            position=2,
        ),
        shell.arg(name="text_arg", argstr="--text-arg", type=str, position=3),
        shell.arg(
            name="int_arg",
            argstr="--int-arg",
            type=int | None,
            default=None,
            position=4,
        ),
        shell.arg(
            name="tuple_arg",
            argstr="--tuple-arg",
            type=tuple[int, str],
            sep=" ",
            position=5,
        ),
    ] + outargs + [shell.Task.append_args]
    assert sorted_fields(Cp.Outputs) == outargs + [
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]


def test_interface_template_with_defaults():

    Cp = shell.define(
        (
            "cp <in_fs_objects:fs-object+> <out|out_dir:directory> "
            "-R<recursive=True> "
            "--text-arg <text_arg='foo'> "
            "--int-arg <int_arg:int=99> "
            "--tuple-arg <tuple_arg:int,str=(1,'bar')> "
        ),
    )

    assert issubclass(Cp, shell.Task)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=2,
    )
    assert sorted_fields(Cp) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(name="in_fs_objects", type=MultiInputObj[FsObject], position=1),
        output,
        shell.arg(name="recursive", argstr="-R", type=bool, default=True, position=3),
        shell.arg(
            name="text_arg", argstr="--text-arg", type=str, position=4, default="foo"
        ),
        shell.arg(name="int_arg", argstr="--int-arg", type=int, position=5, default=99),
        shell.arg(
            name="tuple_arg",
            argstr="--tuple-arg",
            type=tuple[int, str],
            default=(1, "bar"),
            position=6,
            sep=" ",
        ),
        shell.Task.append_args,
    ]
    assert sorted_fields(Cp.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]
    Cp(in_fs_objects=[File.sample(), File.sample(seed=1)])
    Cp.Outputs(out_dir=Directory.sample())


def test_interface_template_with_type_overrides():

    Cp = shell.define(
        (
            "cp <in_fs_objects:fs-object+> <out|out_dir:directory> "
            "-R<recursive> "
            "--text-arg <text_arg> "
            "--int-arg <int_arg> "
            "--tuple-arg <tuple_arg:int,str> "
        ),
        inputs={"text_arg": str, "int_arg": int | None},
    )

    assert issubclass(Cp, shell.Task)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=2,
    )
    assert sorted_fields(Cp) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(name="in_fs_objects", type=MultiInputObj[FsObject], position=1),
        output,
        shell.arg(name="recursive", argstr="-R", type=bool, default=False, position=3),
        shell.arg(name="text_arg", argstr="--text-arg", type=str, position=4),
        shell.arg(
            name="int_arg",
            argstr="--int-arg",
            type=int | None,
            position=5,
        ),
        shell.arg(
            name="tuple_arg",
            argstr="--tuple-arg",
            type=tuple[int, str],
            position=6,
            sep=" ",
        ),
        shell.Task.append_args,
    ]
    assert sorted_fields(Cp.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]


@pytest.fixture(params=["static", "dynamic"])
def Ls(request):
    if request.param == "static":

        @shell.define(xor=["complete_date", "date_format_str", None])
        class Ls(shell.Task["Ls.Outputs"]):
            executable = "ls"

            directory: Directory = shell.arg(
                help="the directory to list the contents of",
                argstr="",
                position=-1,
            )
            hidden: bool = shell.arg(
                help=("display hidden FS objects"),
                argstr="-a",
                default=False,
            )
            long_format: bool = shell.arg(
                help=(
                    "display properties of FS object, such as permissions, size and "
                    "timestamps "
                ),
                default=False,
                argstr="-l",
            )
            human_readable: bool = shell.arg(
                help="display file sizes in human readable form",
                argstr="-h",
                default=False,
                requires=["long_format"],
            )
            complete_date: bool = shell.arg(
                help="Show complete date in long format",
                argstr="-T",
                default=False,
                requires=["long_format"],
            )
            date_format_str: str | None = shell.arg(
                help="format string for ",
                argstr="-D",
                default=None,
                requires=["long_format"],
            )

            class Outputs(shell.Outputs):
                entries: list = shell.out(
                    help="list of entries returned by ls command",
                    callable=list_entries,
                )

    elif request.param == "dynamic":
        Ls = shell.define(
            "ls",
            inputs={
                "directory": shell.arg(
                    type=Directory,
                    help="the directory to list the contents of",
                    argstr="",
                    position=-1,
                ),
                "hidden": shell.arg(
                    type=bool,
                    help="display hidden FS objects",
                    default=False,
                    argstr="-a",
                ),
                "long_format": {  # Mix it up with a full dictionary based definition
                    "type": bool,
                    "default": False,
                    "help": (
                        "display properties of FS object, such as permissions, size and "
                        "timestamps "
                    ),
                    "argstr": "-l",
                },
                "human_readable": shell.arg(
                    type=bool,
                    help="display file sizes in human readable form",
                    default=False,
                    argstr="-h",
                    requires=["long_format"],
                ),
                "complete_date": shell.arg(
                    type=bool,
                    help="Show complete date in long format",
                    argstr="-T",
                    default=False,
                    requires=["long_format"],
                ),
                "date_format_str": shell.arg(
                    type=str | None,
                    help="format string for ",
                    default=None,
                    argstr="-D",
                    requires=["long_format"],
                ),
            },
            outputs={
                "entries": shell.out(
                    type=list,
                    help="list of entries returned by ls command",
                    callable=list_entries,
                )
            },
            xor=["complete_date", "date_format_str", None],
            name="Ls",
        )

    else:
        assert False

    return Ls


def test_shell_fields(Ls):
    assert sorted([a.name for a in sorted_fields(Ls)]) == sorted(
        [
            "append_args",
            "executable",
            "directory",
            "hidden",
            "long_format",
            "human_readable",
            "complete_date",
            "date_format_str",
        ]
    )

    assert [a.name for a in sorted_fields(Ls.Outputs)] == sorted(
        [
            "entries",
            "stdout",
            "stderr",
            "return_code",
        ]
    )


def test_shell_pickle_roundtrip(Ls, tmp_path):
    pkl_file = tmp_path / "ls.pkl"
    with open(pkl_file, "wb") as f:
        cp.dump(Ls, f)

    with open(pkl_file, "rb") as f:
        RereadLs = cp.load(f)

    assert RereadLs is Ls


# @pytest.mark.xfail(reason="Still need to update tasks to use new shell interface")
def test_shell_run(Ls, tmp_path):
    Path.touch(tmp_path / "a")
    Path.touch(tmp_path / "b")
    Path.touch(tmp_path / "c")

    ls = Ls(directory=tmp_path, long_format=True)

    # Test cmdline
    assert ls.directory == Directory(tmp_path)
    assert not ls.hidden
    assert ls.long_format
    assert ls.cmdline == f"ls -l {tmp_path}"

    # Drop Long format flag to make output simpler
    ls = Ls(directory=tmp_path)
    outputs = ls()

    assert sorted(outputs.entries) == ["a", "b", "c"]


@pytest.fixture(params=["static", "dynamic"])
def A(request):
    if request.param == "static":

        @shell.define
        class A(shell.Task["A.Outputs"]):
            """An example shell interface described in a class

            Parameters
            ----------
            x : File
                an input file
            """

            executable = "cp"

            x: File = shell.arg(argstr="", position=1)

            class Outputs(shell.Outputs):
                """The outputs of the example shell interface

                Parameters
                ----------
                y : File
                    path of output file"""

                y: File = shell.outarg(path_template="{x}_out", position=-1)

    elif request.param == "dynamic":
        A = shell.define(
            "cp",
            inputs={
                "x": shell.arg(
                    type=File,
                    help="an input file",
                    argstr="",
                    position=1,
                ),
            },
            outputs={
                "y": shell.outarg(
                    type=File,
                    help="path of output file",
                    argstr="",
                    path_template="{x}_out",
                ),
            },
            name="A",
        )
    else:
        assert False

    return A


def test_shell_output_path_template(A):
    assert "y" in [a.name for a in attrs.fields(A.Outputs)]


def test_shell_output_field_name_static():
    @shell.define
    class A(shell.Task["A.Outputs"]):
        """Copy a file"""

        executable = "cp"

        x: File = shell.arg(help="an input file", argstr="", position=1)

        class Outputs(shell.Outputs):
            y: File = shell.outarg(
                help="the output file",
                path_template="{x}_out",
                argstr="",
                position=-1,
            )

    assert sorted([a.name for a in attrs.fields(A) if not a.name.startswith("_")]) == [
        "append_args",
        "executable",
        "x",
        "y",
    ]
    assert sorted(
        a.name for a in attrs.fields(A.Outputs) if not a.name.startswith("_")
    ) == [
        "return_code",
        "stderr",
        "stdout",
        "y",
    ]
    output = shell.outarg(
        name="y",
        type=File,
        help="the output file",
        path_template="{x}_out",
        argstr="",
        position=-1,
    )
    assert sorted_fields(A) == [
        shell.arg(
            name="executable",
            validator=attrs.validators.min_len(1),
            default="cp",
            type=str | ty.Sequence[str],
            argstr="",
            position=0,
            help=shell.Task.EXECUTABLE_HELP,
        ),
        shell.arg(
            name="x",
            type=File,
            help="an input file",
            argstr="",
            position=1,
        ),
        output,
        shell.Task.append_args,
    ]
    assert sorted_fields(A.Outputs) == [
        output,
        shell.out(
            name="return_code",
            type=int,
            help=shell.Outputs.RETURN_CODE_HELP,
        ),
        shell.out(
            name="stderr",
            type=str,
            help=shell.Outputs.STDERR_HELP,
        ),
        shell.out(
            name="stdout",
            type=str,
            help=shell.Outputs.STDOUT_HELP,
        ),
    ]


def test_shell_output_field_name_dynamic():
    A = shell.define(
        "cp",
        name="A",
        inputs={
            "x": shell.arg(
                type=File,
                help="an input file",
                argstr="",
                position=1,
            ),
        },
        outputs={
            "y": shell.outarg(
                type=File,
                help="path of output file",
                argstr="",
                path_template="{x}_out",
            ),
        },
    )

    assert "y" in [a.name for a in attrs.fields(A.Outputs)]


def get_file_size(y: Path):
    result = os.stat(y)
    return result.st_size


def test_shell_bases_dynamic(A, tmp_path):
    B = shell.define(
        name="B",
        inputs={"y": shell.arg(type=File, help="output file", argstr="", position=-1)},
        outputs={
            "out_file_size": {
                "type": int,
                "help": "size of the output directory",
                "callable": get_file_size,
            }
        },
        bases=[A],
    )

    xpath = tmp_path / "x.txt"
    ypath = tmp_path / "y.txt"
    Path.touch(xpath)
    Path.touch(ypath)

    b = B(x=xpath, y=ypath)

    assert b.x == File(xpath)
    assert b.y == File(ypath)

    # outputs = b()
    # assert outputs.y == str(ypath)


def test_shell_bases_static(A, tmp_path):
    @shell.define
    class B(A):

        y: text.Plain = shell.arg()  # Override the output arg in A

        class Outputs(shell.Outputs):
            """
            Args:
                out_file_size: size of the output directory
            """

            out_file_size: int = shell.out(callable=get_file_size)

    xpath = tmp_path / "x.txt"
    ypath = tmp_path / "y.txt"
    Path.touch(xpath)
    ypath.write_text("Hello, World!")

    a = A(x=xpath, y=ypath)
    assert a.x == File(xpath)
    assert a.y == ypath

    b = B(x=xpath, y=str(ypath))
    assert b.x == File(xpath)
    # We have overridden the type of y from an output arg with a path_template so it
    # gets coerced to a text.Plain object
    assert b.y == text.Plain(ypath)

    # outputs = b()
    # assert outputs.y == str(ypath)


def test_shell_inputs_outputs_bases_dynamic(tmp_path):
    A = shell.define(
        "ls",
        name="A",
        inputs={
            "directory": shell.arg(
                type=Directory,
                help="input directory",
                argstr="",
                position=-1,
            )
        },
        outputs={
            "entries": shell.out(
                type=list,
                help="list of entries returned by ls command",
                callable=list_entries,
            )
        },
    )
    B = shell.define(
        "ls",
        name="B",
        inputs={
            "hidden": shell.arg(
                type=bool,
                argstr="-a",
                help="show hidden files",
                default=False,
            )
        },
        bases=[A],
    )

    b = B(directory=tmp_path, hidden=True)

    assert b.directory == Directory(tmp_path)
    assert b.hidden

    # File.sample(tmp_path, stem=".hidden-file")
    # outputs = b()
    # assert result.runner.cmdline == f"ls -a {tmp_path}"
    # assert outputs.entries == [".", "..", ".hidden-file"]


def test_shell_inputs_outputs_bases_static(tmp_path):
    @shell.define
    class A(shell.Task["A.Outputs"]):
        executable = "ls"

        directory: Directory = shell.arg(help="input directory", argstr="", position=-1)

        class Outputs(shell.Outputs):
            entries: list = shell.out(
                help="list of entries returned by ls command",
                callable=list_entries,
            )

    @shell.define
    class B(A):
        hidden: bool = shell.arg(
            help="show hidden files",
            argstr="-a",
            default=False,
        )

    Path.touch(tmp_path / ".hidden")

    b = B(directory=tmp_path, hidden=True)

    assert b.directory == Directory(tmp_path)
    assert b.hidden

    # outputs = b()
    # assert outputs.entries == [".", "..", ".hidden"]


def test_shell_missing_executable_static():
    with pytest.raises(AttributeError, match="must have an `executable` attribute"):

        @shell.define
        class A:
            directory: Directory = shell.arg(
                help="input directory", argstr="", position=-1
            )

            class Outputs:
                entries: list = shell.out(
                    help="list of entries returned by ls command",
                    callable=list_entries,
                )


def test_shell_missing_executable_dynamic():
    with pytest.raises(
        ValueError,
        match=r"name \('A'\) can only be provided when creating a class dynamically",
    ):
        shell.define(
            name="A",
            inputs={
                "directory": shell.arg(
                    type=Directory,
                    help="input directory",
                    argstr="",
                    position=-1,
                ),
            },
            outputs={
                "entries": shell.out(
                    type=list,
                    help="list of entries returned by ls command",
                    callable=list_entries,
                )
            },
        )


def test_shell_help1():

    Shelly = shell.define(
        "shelly <in_file:generic/file> <out|out_file:generic/file> --arg1 <arg1:int> "
        "--arg2 <arg2:float?> --opt-out <out|opt_out:File?>"
    )

    assert task_help(Shelly) == [
        "----------------------------",
        "Help for Shell task 'shelly'",
        "----------------------------",
        "",
        "Inputs:",
        "- executable: str | Sequence[str]; default = 'shelly'",
        "    the first part of the command, can be a string, e.g. 'ls', or a list, e.g.",
        "    ['ls', '-l', 'dirname']",
        "- in_file: generic/file",
        "- out_file: Path | bool; default = True",
    ] + wrap_text(shell.outarg.PATH_TEMPLATE_HELP).split("\n") + [
        "- arg1: int ('--arg1')",
        "- arg2: float | None; default = None ('--arg2')",
        "- opt_out: Path | bool | None; default = None ('--opt-out')",
    ] + wrap_text(
        shell.outarg.OPTIONAL_PATH_TEMPLATE_HELP
    ).split(
        "\n"
    ) + [
        "- append_args: list[str | generic/file]; default-factory = list()",
        "    Additional free-form arguments to append to the end of the command.",
        "",
        "Outputs:",
        "- out_file: generic/file",
        "- opt_out: generic/file | None; default = None",
        "- return_code: int",
        "    " + shell.Outputs.RETURN_CODE_HELP,
        "- stdout: str",
        "    " + shell.Outputs.STDOUT_HELP,
        "- stderr: str",
        "    " + shell.Outputs.STDERR_HELP,
        "",
    ]


def list_entries(stdout):
    return stdout.split("\n")[:-1]


def sorted_fields(interface):
    fields = task_fields(interface)
    length = len(fields) - 1

    def pos_key(out: shell.out) -> int:
        if out.name == "append_args":
            return (length + 1, out.name)
        try:
            pos = out.position
        except AttributeError:
            return (length, out.name)
        if pos < 0:
            key = length + pos
        else:
            key = pos
        return (key, out.name)

    return sorted(fields, key=pos_key)
