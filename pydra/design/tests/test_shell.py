import os
from operator import attrgetter
from pathlib import Path
import attrs
import pytest
import cloudpickle as cp
from pydra.design import shell, Interface, list_fields
from fileformats.generic import File, Directory, FsObject, SetOf
from fileformats import field


def test_interface_template():

    SampleInterface = shell.interface("cp <in_path> <out|out_path>")

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=inp_sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=out_sort_key)
    assert inputs == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_path", type=FsObject, position=1),
        shell.outarg(name="out_path", type=FsObject, position=2),
    ]
    assert outputs == [
        shell.outarg(name="out_path", type=FsObject, position=2),
    ]


def test_interface_template_more_complex():

    SampleInterface = shell.interface(
        (
            "cp <in_paths:fs-object+set-of> <out|out_path> -R<recursive> -v<verbose> "
            "--text-arg <text_arg> --int-arg <int_arg:integer> "
            # "--tuple-arg <tuple_arg:tuple[integer,text]> "
        ),
    )

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=inp_sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=out_sort_key)
    assert inputs == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_paths", type=SetOf[FsObject], position=1, sep=" "),
        shell.outarg(name="out_path", type=FsObject, position=2),
        shell.arg(name="recursive", type=bool, position=3),
        shell.arg(name="verbose", type=bool, position=4),
        shell.arg(name="text_arg", type=field.Text, position=5),
        shell.arg(name="int_arg", type=field.Integer, position=6),
        # shell.arg(name="tuple_arg", type=tuple[field.Integer,field.Text], position=6),
    ]
    assert outputs == [
        shell.outarg(name="out_path", type=FsObject, position=2),
    ]


def test_interface_template_with_overrides():

    RECURSIVE_HELP = (
        "If source_file designates a directory, cp copies the directory and the entire "
        "subtree connected at that point."
    )

    SampleInterface = shell.interface(
        (
            "cp <in_paths:fs-object+set-of> <out|out_path> -R<recursive> -v<verbose> "
            "--text-arg <text_arg> --int-arg <int_arg:integer> "
            # "--tuple-arg <tuple_arg:tuple[integer,text]> "
        ),
        inputs={"recursive": shell.arg(help_string=RECURSIVE_HELP)},
        outputs={"out_path": shell.outarg(position=-1)},
    )

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=inp_sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=out_sort_key)
    assert inputs == [
        shell.arg(name="executable", default="cp", type=str, position=0),
        shell.arg(name="in_paths", type=SetOf[FsObject], position=1, sep=" "),
        shell.arg(name="recursive", type=bool, help_string=RECURSIVE_HELP, position=2),
        shell.arg(name="verbose", type=bool, position=3),
        shell.arg(name="text_arg", type=field.Text, position=4),
        shell.arg(name="int_arg", type=field.Integer, position=5),
        # shell.arg(name="tuple_arg", type=tuple[field.Integer,field.Text], position=6),
        shell.outarg(name="out_path", type=FsObject, position=-1),
    ]
    assert outputs == [
        shell.outarg(name="out_path", type=FsObject, position=-1),
    ]


def test_interface_template_with_type_overrides():

    SampleInterface = shell.interface(
        (
            "cp <in_paths:fs-object+set-of> <out|out_path> -R<recursive> -v<verbose> "
            "--text-arg <text_arg> --int-arg <int_arg> "
            # "--tuple-arg <tuple_arg:tuple[integer,text]> "
        ),
        inputs={"text_arg": str, "int_arg": int},
    )

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=inp_sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=out_sort_key)
    assert inputs == [
        shell.arg(name="in_paths", type=SetOf[FsObject], position=1, sep=" "),
        shell.arg(name="recursive", type=bool, position=2),
        shell.arg(name="verbose", type=bool, position=3),
        shell.arg(name="text_arg", type=str, position=4),
        shell.arg(name="int_arg", type=int, position=5),
        # shell.arg(name="tuple_arg", type=tuple[field.Integer,field.Text], position=6),
        shell.outarg(name="out_path", type=FsObject, position=-1),
    ]
    assert outputs == [
        shell.outarg(name="out_path", type=FsObject, position=-1),
    ]


@pytest.fixture(params=["static", "dynamic"])
def Ls(request):
    if request.param == "static":

        @shell.interface
        class Ls(Interface["Ls.Outputs"]):
            executable = "ls"

            directory: Directory = shell.arg(
                help_string="the directory to list the contents of",
                argstr="",
                mandatory=True,
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
            date_format_str: str = shell.arg(
                help_string="format string for ",
                argstr="-D",
                default=attrs.NOTHING,
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
                    mandatory=True,
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
                    type=str,
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

                y: File = shell.outarg(file_template="{x}_out", position=-1)

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
                    file_template="{x}_out",
                ),
            },
            name="A",
        )
    else:
        assert False

    return A


def test_shell_output_file_template(A):
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
                file_template="{x}_out",
                argstr="",
                position=-1,
            )

    assert sorted([a.name for a in attrs.fields(A)]) == ["executable", "x", "y"]
    assert [a.name for a in attrs.fields(A.Outputs)] == ["y"]
    inputs = sorted(list_fields(A), key=attrgetter("name"))
    outputs = sorted(list_fields(A.Outputs), key=attrgetter("name"))
    assert inputs == [
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
        shell.outarg(
            name="y",
            type=File,
            help_string="the output file",
            file_template="{x}_out",
            argstr="",
            position=-1,
        ),
    ]
    assert outputs == [
        shell.outarg(
            name="y",
            type=File,
            help_string="the output file",
            file_template="{x}_out",
            argstr="",
            position=-1,
        )
    ]


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
                template_field="y_out",
                file_template="{x}_out",
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

        y: File

        class Outputs:
            out_file_size: int = shell.out(
                help_string="size of the output directory", callable=get_file_size
            )

    xpath = tmp_path / "x.txt"
    ypath = tmp_path / "y.txt"
    Path.touch(xpath)
    Path.touch(ypath)

    b = B(x=xpath, y=str(ypath))

    assert b.x == File(xpath)
    assert b.y == File(ypath)

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

    hidden = File.sample(tmp_path, stem=".hidden")

    b = B(directory=tmp_path, hidden=True)

    assert b.directory == Directory(tmp_path)
    assert b.hidden

    # result = b()
    # assert result.runner.cmdline == f"ls -a {tmp_path}"
    # assert result.output.entries == [".", "..", ".hidden"]


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


inp_sort_key = attrgetter("position")


def out_sort_key(out: shell.out) -> int:
    LARGE_NUMBER = 1000000
    try:
        pos = out.position
    except AttributeError:
        pos = LARGE_NUMBER
    else:
        if pos < 0:
            pos = LARGE_NUMBER + pos
    return pos
