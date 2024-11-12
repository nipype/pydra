import os
import tempfile
from operator import attrgetter
from pathlib import Path
import attrs
import pytest
import cloudpickle as cp
from pydra.design import shell, Interface, list_fields
from fileformats.generic import File, Directory


def list_entries(stdout):
    return stdout.split("\n")[:-1]


@pytest.fixture
def tmpdir():
    return Path(tempfile.mkdtemp())


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


def test_shell_pickle_roundtrip(Ls, tmpdir):
    pkl_file = tmpdir / "ls.pkl"
    with open(pkl_file, "wb") as f:
        cp.dump(Ls, f)

    with open(pkl_file, "rb") as f:
        RereadLs = cp.load(f)

    assert RereadLs is Ls


def test_shell_run(Ls, tmpdir):
    Path.touch(tmpdir / "a")
    Path.touch(tmpdir / "b")
    Path.touch(tmpdir / "c")

    ls = Ls(directory=tmpdir, long_format=True)

    # Test cmdline
    assert ls.inputs.directory == tmpdir
    assert not ls.inputs.hidden
    assert ls.inputs.long_format
    assert ls.cmdline == f"ls -l {tmpdir}"

    # Drop Long format flag to make output simpler
    ls = Ls(directory=tmpdir)
    result = ls()

    assert result.output.entries == ["a", "b", "c"]


@pytest.fixture(params=["static", "dynamic"])
def A(request):
    if request.param == "static":

        @shell.interface
        class A:
            executable = "cp"

            x: File = shell.arg(help_string="an input file", argstr="", position=0)

            class Outputs:
                y: File = shell.outarg(
                    help_string="path of output file",
                    file_template="{x}_out",
                )

    elif request.param == "dynamic":
        A = shell.interface(
            "A",
            executable="cp",
            inputs={
                "x": shell.arg(
                    type=File,
                    help_string="an input file",
                    argstr="",
                    position=0,
                ),
            },
            outputs={
                "y": shell.outarg(
                    type=File,
                    help_string="path of output file",
                    argstr="",
                    output_file_template="{x}_out",
                ),
            },
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


def test_shell_bases_dynamic(A, tmpdir):
    B = shell.interface(
        "B",
        outputs={
            "out_file_size": {
                "type": int,
                "help_string": "size of the output directory",
                "callable": get_file_size,
            }
        },
        bases=[A],
    )

    xpath = tmpdir / "x.txt"
    ypath = tmpdir / "y.txt"
    Path.touch(xpath)

    b = B(x=xpath, y=str(ypath))

    result = b()

    assert b.inputs.x == xpath
    assert result.output.y == str(ypath)


def test_shell_bases_static(A, tmpdir):
    @shell.interface
    class B(A):
        class Outputs:
            out_file_size: int = shell.out(
                help_string="size of the output directory", callable=get_file_size
            )

    xpath = tmpdir / "x.txt"
    ypath = tmpdir / "y.txt"
    Path.touch(xpath)

    b = B(x=xpath, y=str(ypath))

    result = b()

    assert b.inputs.x == xpath
    assert result.output.y == str(ypath)


def test_shell_inputs_outputs_bases_dynamic(tmpdir):
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

    hidden = File.sample(tmpdir, stem=".hidden")

    b = B(directory=tmpdir, hidden=True)

    assert b.directory == Directory(tmpdir)
    assert b.hidden

    # result = b()
    # assert result.runner.cmdline == f"ls -a {tmpdir}"
    # assert result.output.entries == [".", "..", ".hidden"]


def test_shell_inputs_outputs_bases_static(tmpdir):
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

    Path.touch(tmpdir / ".hidden")

    b = B(directory=tmpdir, hidden=True)

    assert b.inputs.directory == tmpdir
    assert b.inputs.hidden

    # result = b()
    # assert result.output.entries == [".", "..", ".hidden"]


def test_shell_missing_executable_static():
    with pytest.raises(RuntimeError, match="should contain an `executable`"):

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
    with pytest.raises(RuntimeError, match="should contain an `executable`"):
        shell.interface(
            "A",
            executable=None,
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


def test_shell_missing_inputs_static():
    with pytest.raises(RuntimeError, match="should contain an `Inputs`"):

        @shell.interface
        class A:
            executable = "ls"

            class Outputs:
                entries: list = shell.out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )


def test_shell_decorator_misuse(A):
    with pytest.raises(
        RuntimeError,
        match=("`shell.interface` should not be provided any other arguments"),
    ):
        shell.interface(A, executable="cp")
