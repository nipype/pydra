import os
from pathlib import Path
import attrs
import pytest
import cloudpickle as cp
from pydra.design import shell, Interface, list_fields
from fileformats.generic import File, Directory, FsObject, SetOf
from fileformats import field, text, image


def test_interface_template():

    SampleInterface = shell.interface("cp <in_path> <out|out_path>")

    assert issubclass(SampleInterface, Interface)
    output = shell.outarg(
        name="out_path",
        path_template="out_path",
        default=True,
        type=FsObject,
        position=2,
    )
    assert list_fields(SampleInterface) == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_path", type=FsObject, position=1),
        output,
    ]
    assert list_fields(SampleInterface.Outputs) == [output]
    intf = SampleInterface(in_path=File.mock("in-path.txt"))
    assert intf.executable == "cp"
    SampleInterface(in_path=File.mock("in-path.txt"), out_path=Path("./out-path.txt"))
    SampleInterface.Outputs(out_path=File.mock("in-path.txt"))


def test_interface_template_w_types_and_path_template_ext():

    SampleInterface = shell.interface(
        "trim-png <in_image:image/png> <out|out_image.png:image/png>"
    )

    assert issubclass(SampleInterface, Interface)
    output = shell.outarg(
        name="out_image",
        path_template="out_image.png",
        default=True,
        type=image.Png,
        position=2,
    )
    assert list_fields(SampleInterface) == [
        shell.arg(name="executable", default="trim-png", type=str, position=0),
        shell.arg(name="in_image", type=image.Png, position=1),
        output,
    ]
    assert list_fields(SampleInterface.Outputs) == [output]
    SampleInterface(in_image=image.Png.mock())
    SampleInterface(in_image=image.Png.mock(), out_image=Path("./new_image.png"))
    SampleInterface.Outputs(out_image=image.Png.mock())


def test_interface_template_more_complex():

    SampleInterface = shell.interface(
        (
            "cp <in_fs_objects:fs-object+set-of> <out|out_dir:directory> "
            "-R<recursive> "
            "--text-arg <text_arg> --int-arg <int_arg:integer> "
            "--tuple-arg <tuple_arg:integer,text> "
        ),
    )

    assert issubclass(SampleInterface, Interface)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=2,
        default=True,
    )
    assert sorted(list_fields(SampleInterface), key=pos_key) == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_fs_objects", type=SetOf[FsObject], position=1, sep=" "),
        output,
        shell.arg(name="recursive", arg_str="-R", type=bool, position=3),
        shell.arg(
            name="text_arg", arg_str="--text-arg", type=field.Text | None, position=5
        ),
        shell.arg(
            name="int_arg", arg_str="--int-arg", type=field.Integer | None, position=6
        ),
        shell.arg(
            name="tuple_arg",
            arg_str="--tuple-arg",
            type=tuple[field.Integer, field.Text] | None,
            position=6,
        ),
    ]
    assert list_fields(SampleInterface.Outputs) == [output]
    SampleInterface(in_fs_objects=[File.sample(), File.sample(seed=1)])
    SampleInterface.Outputs(out_path=File.sample())


def test_interface_template_with_overrides():

    RECURSIVE_HELP = (
        "If source_file designates a directory, cp copies the directory and the entire "
        "subtree connected at that point."
    )

    SampleInterface = shell.interface(
        (
            "cp <in_fs_objects:fs-object+set-of> <out|out_dir:directory> "
            "-R<recursive> "
            "--text-arg <text_arg> --int-arg <int_arg:integer> "
            "--tuple-arg <tuple_arg:integer> <tuple_arg:text> "
        ),
        inputs={"recursive": shell.arg(help_string=RECURSIVE_HELP)},
        outputs={"out_dir": shell.outarg(position=-1)},
    )

    assert issubclass(SampleInterface, Interface)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=-1,
        default=True,
    )
    assert list_fields(SampleInterface) == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_fs_objects", type=SetOf[FsObject], position=1, sep=" "),
        shell.arg(
            name="recursive",
            arg_str="-R",
            type=bool,
            help_string=RECURSIVE_HELP,
            position=2,
        ),
        shell.arg(
            name="text_arg", argstr="--text-arg", type=field.Text | None, position=3
        ),
        shell.arg(
            name="int_arg", argstr="--int-arg", type=field.Integer | None, position=4
        ),
        shell.arg(
            name="tuple_arg",
            argstr="--tuple-arg",
            type=tuple[field.Integer, field.Text] | None,
            position=5,
        ),
        output,
    ]
    assert list_fields(SampleInterface.Outputs) == [output]


def test_interface_template_with_type_overrides():

    SampleInterface = shell.interface(
        (
            "cp <in_fs_objects:fs-object+set-of> <out|out_dir:directory> "
            "-R<recursive> "
            "--text-arg <text_arg> --int-arg <int_arg> "
            "--tuple-arg <tuple_arg:integer> <tuple_arg:text> "
        ),
        inputs={"text_arg": str | None, "int_arg": int | None},
    )

    assert issubclass(SampleInterface, Interface)
    output = shell.outarg(
        name="out_dir",
        type=Directory,
        path_template="out_dir",
        position=-1,
        default=True,
    )
    assert list_fields(SampleInterface) == [
        shell.arg(name="in_fs_objects", type=SetOf[FsObject], position=1, sep=" "),
        shell.arg(name="recursive", argstr="-R", type=bool, position=2),
        shell.arg(name="text_arg", argstr="--text-arg", type=str | None, position=4),
        shell.arg(name="int_arg", argstr="--text-arg", type=int | None, position=5),
        shell.arg(
            name="tuple_arg",
            targstr="--text-arg",
            ype=tuple[field.Integer, field.Text] | None,
            position=6,
        ),
        output,
    ]
    assert list_fields(SampleInterface.Outputs) == [output]


@pytest.fixture(params=["static", "dynamic"])
def Ls(request):
    if request.param == "static":

        @shell.interface
        class Ls(Interface["Ls.Outputs"]):
            executable = "ls"

            directory: Directory = shell.arg(
                help_string="the directory to list the contents of",
                argstr="",
                position=-1,
            )
            hidden: bool = shell.arg(
                help_string=("display hidden FS objects"),
                argstr="-a",
                default=False,
            )
            long_format: bool = shell.arg(
                help_string=(
                    "display properties of FS object, such as permissions, size and "
                    "timestamps "
                ),
                default=False,
                argstr="-l",
            )
            human_readable: bool = shell.arg(
                help_string="display file sizes in human readable form",
                argstr="-h",
                default=False,
                requires=["long_format"],
            )
            complete_date: bool = shell.arg(
                help_string="Show complete date in long format",
                argstr="-T",
                default=False,
                requires=["long_format"],
                xor=["date_format_str"],
            )
            date_format_str: str | None = shell.arg(
                help_string="format string for ",
                argstr="-D",
                default=None,
                requires=["long_format"],
                xor=["complete_date"],
            )

            class Outputs:
                entries: list = shell.out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )

    elif request.param == "dynamic":
        Ls = shell.interface(
            "ls",
            inputs={
                "directory": shell.arg(
                    type=Directory,
                    help_string="the directory to list the contents of",
                    argstr="",
                    position=-1,
                ),
                "hidden": shell.arg(
                    type=bool,
                    help_string="display hidden FS objects",
                    argstr="-a",
                ),
                "long_format": {  # Mix it up with a full dictionary based definition
                    "type": bool,
                    "help_string": (
                        "display properties of FS object, such as permissions, size and "
                        "timestamps "
                    ),
                    "argstr": "-l",
                },
                "human_readable": shell.arg(
                    type=bool,
                    help_string="display file sizes in human readable form",
                    argstr="-h",
                    requires=["long_format"],
                ),
                "complete_date": shell.arg(
                    type=bool,
                    help_string="Show complete date in long format",
                    argstr="-T",
                    requires=["long_format"],
                    xor=["date_format_str"],
                ),
                "date_format_str": shell.arg(
                    type=str | None,
                    help_string="format string for ",
                    argstr="-D",
                    requires=["long_format"],
                    xor=["complete_date"],
                ),
            },
            outputs={
                "entries": shell.out(
                    type=list,
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )
            },
            name="Ls",
        )

    else:
        assert False

    return Ls


def test_shell_fields(Ls):
    assert sorted([a.name for a in list_fields(Ls)]) == sorted(
        [
            "executable",
            "directory",
            "hidden",
            "long_format",
            "human_readable",
            "complete_date",
            "date_format_str",
        ]
    )

    assert [a.name for a in list_fields(Ls.Outputs)] == ["entries"]


def test_shell_pickle_roundtrip(Ls, tmp_path):
    pkl_file = tmp_path / "ls.pkl"
    with open(pkl_file, "wb") as f:
        cp.dump(Ls, f)

    with open(pkl_file, "rb") as f:
        RereadLs = cp.load(f)

    assert RereadLs is Ls


@pytest.mark.xfail(reason="Still need to update tasks to use new shell interface")
def test_shell_run(Ls, tmp_path):
    Path.touch(tmp_path / "a")
    Path.touch(tmp_path / "b")
    Path.touch(tmp_path / "c")

    ls = Ls(directory=tmp_path, long_format=True)

    # Test cmdline
    assert ls.inputs.directory == tmp_path
    assert not ls.inputs.hidden
    assert ls.inputs.long_format
    assert ls.cmdline == f"ls -l {tmp_path}"

    # Drop Long format flag to make output simpler
    ls = Ls(directory=tmp_path)
    result = ls()

    assert result.output.entries == ["a", "b", "c"]


@pytest.fixture(params=["static", "dynamic"])
def A(request):
    if request.param == "static":

        @shell.interface
        class A:
            """An example shell interface described in a class

            Parameters
            ----------
            x : File
                an input file
            """

            executable = "cp"

            x: File = shell.arg(argstr="", position=1)

            class Outputs:
                """The outputs of the example shell interface

                Parameters
                ----------
                y : File
                    path of output file"""

                y: File = shell.outarg(path_template="{x}_out", position=-1)

    elif request.param == "dynamic":
        A = shell.interface(
            "cp",
            inputs={
                "x": shell.arg(
                    type=File,
                    help_string="an input file",
                    argstr="",
                    position=1,
                ),
            },
            outputs={
                "y": shell.outarg(
                    type=File,
                    help_string="path of output file",
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
    @shell.interface
    class A:
        """Copy a file"""

        executable = "cp"

        x: File = shell.arg(help_string="an input file", argstr="", position=1)

        class Outputs:
            y: File = shell.outarg(
                help_string="the output file",
                path_template="{x}_out",
                argstr="",
                position=-1,
            )

    assert sorted([a.name for a in attrs.fields(A)]) == ["executable", "x", "y"]
    assert [a.name for a in attrs.fields(A.Outputs)] == ["y"]
    output = shell.outarg(
        name="y",
        type=File,
        help_string="the output file",
        path_template="{x}_out",
        default=True,
        argstr="",
        position=-1,
    )
    assert sorted(list_fields(A), key=pos_key) == [
        shell.arg(
            name="executable",
            default="cp",
            type=str,
            argstr="",
            position=0,
        ),
        shell.arg(
            name="x",
            type=File,
            help_string="an input file",
            argstr="",
            position=1,
        ),
        output,
    ]
    assert list_fields(A.Outputs) == [output]


def test_shell_output_field_name_dynamic():
    A = shell.interface(
        "cp",
        name="A",
        inputs={
            "x": shell.arg(
                type=File,
                help_string="an input file",
                argstr="",
                position=1,
            ),
        },
        outputs={
            "y": shell.outarg(
                type=File,
                help_string="path of output file",
                argstr="",
                path_template="{x}_out",
                default=True,
            ),
        },
    )

    assert "y" in [a.name for a in attrs.fields(A.Outputs)]


def get_file_size(y: Path):
    result = os.stat(y)
    return result.st_size


def test_shell_bases_dynamic(A, tmp_path):
    B = shell.interface(
        name="B",
        inputs={
            "y": shell.arg(type=File, help_string="output file", argstr="", position=-1)
        },
        outputs={
            "out_file_size": {
                "type": int,
                "help_string": "size of the output directory",
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

    # result = b()
    # assert result.output.y == str(ypath)


def test_shell_bases_static(A, tmp_path):
    @shell.interface
    class B(A):

        y: text.Plain = shell.arg()  # Override the output arg in A

        class Outputs:
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

    # result = b()
    # assert result.output.y == str(ypath)


def test_shell_inputs_outputs_bases_dynamic(tmp_path):
    A = shell.interface(
        "ls",
        name="A",
        inputs={
            "directory": shell.arg(
                type=Directory,
                help_string="input directory",
                argstr="",
                position=-1,
            )
        },
        outputs={
            "entries": shell.out(
                type=list,
                help_string="list of entries returned by ls command",
                callable=list_entries,
            )
        },
    )
    B = shell.interface(
        "ls",
        name="B",
        inputs={
            "hidden": shell.arg(
                type=bool,
                argstr="-a",
                help_string="show hidden files",
                default=False,
            )
        },
        bases=[A],
    )

    b = B(directory=tmp_path, hidden=True)

    assert b.directory == Directory(tmp_path)
    assert b.hidden

    # File.sample(tmp_path, stem=".hidden-file")
    # result = b()
    # assert result.runner.cmdline == f"ls -a {tmp_path}"
    # assert result.output.entries == [".", "..", ".hidden-file"]


def test_shell_inputs_outputs_bases_static(tmp_path):
    @shell.interface
    class A:
        executable = "ls"

        directory: Directory = shell.arg(
            help_string="input directory", argstr="", position=-1
        )

        class Outputs:
            entries: list = shell.out(
                help_string="list of entries returned by ls command",
                callable=list_entries,
            )

    @shell.interface
    class B(A):
        hidden: bool = shell.arg(
            help_string="show hidden files",
            argstr="-a",
            default=False,
        )

    Path.touch(tmp_path / ".hidden")

    b = B(directory=tmp_path, hidden=True)

    assert b.directory == Directory(tmp_path)
    assert b.hidden

    # result = b()
    # assert result.output.entries == [".", "..", ".hidden"]


def test_shell_missing_executable_static():
    with pytest.raises(AttributeError, match="must have an `executable` attribute"):

        @shell.interface
        class A:
            directory: Directory = shell.arg(
                help_string="input directory", argstr="", position=-1
            )

            class Outputs:
                entries: list = shell.out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )


def test_shell_missing_executable_dynamic():
    with pytest.raises(
        ValueError,
        match=r"name \('A'\) can only be provided when creating a class dynamically",
    ):
        shell.interface(
            name="A",
            inputs={
                "directory": shell.arg(
                    type=Directory,
                    help_string="input directory",
                    argstr="",
                    position=-1,
                ),
            },
            outputs={
                "entries": shell.out(
                    type=list,
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )
            },
        )


def list_entries(stdout):
    return stdout.split("\n")[:-1]


def pos_key(out: shell.out) -> int:
    LARGE_NUMBER = 1000000
    try:
        pos = out.position
    except AttributeError:
        pos = LARGE_NUMBER
    else:
        if pos < 0:
            pos = LARGE_NUMBER + pos
    return pos
