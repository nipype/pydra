import os
import tempfile
import attrs
from pathlib import Path
import pytest
import cloudpickle as cp
from pydra.mark import shell_task, shell_arg, shell_out


def list_entries(stdout):
    return stdout.split("\n")[:-1]


@pytest.fixture
def tmpdir():
    return Path(tempfile.mkdtemp())


@pytest.fixture(params=["static", "dynamic"])
def Ls(request):
    if request.param == "static":

        @shell_task
        class Ls:
            executable = "ls"

            class Inputs:
                directory: os.PathLike = shell_arg(
                    help_string="the directory to list the contents of",
                    argstr="",
                    mandatory=True,
                    position=-1,
                )
                hidden: bool = shell_arg(
                    help_string=("display hidden FS objects"),
                    argstr="-a",
                    default=False,
                )
                long_format: bool = shell_arg(
                    help_string=(
                        "display properties of FS object, such as permissions, size and "
                        "timestamps "
                    ),
                    default=False,
                    argstr="-l",
                )
                human_readable: bool = shell_arg(
                    help_string="display file sizes in human readable form",
                    argstr="-h",
                    default=False,
                    requires=["long_format"],
                )
                complete_date: bool = shell_arg(
                    help_string="Show complete date in long format",
                    argstr="-T",
                    default=False,
                    requires=["long_format"],
                    xor=["date_format_str"],
                )
                date_format_str: str = shell_arg(
                    help_string="format string for ",
                    argstr="-D",
                    default=attrs.NOTHING,
                    requires=["long_format"],
                    xor=["complete_date"],
                )

            class Outputs:
                entries: list = shell_out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )

    elif request.param == "dynamic":
        Ls = shell_task(
            "Ls",
            executable="ls",
            input_fields={
                "directory": {
                    "type": os.PathLike,
                    "help_string": "the directory to list the contents of",
                    "argstr": "",
                    "mandatory": True,
                    "position": -1,
                },
                "hidden": {
                    "type": bool,
                    "help_string": "display hidden FS objects",
                    "argstr": "-a",
                },
                "long_format": {
                    "type": bool,
                    "help_string": (
                        "display properties of FS object, such as permissions, size and "
                        "timestamps "
                    ),
                    "argstr": "-l",
                },
                "human_readable": {
                    "type": bool,
                    "help_string": "display file sizes in human readable form",
                    "argstr": "-h",
                    "requires": ["long_format"],
                },
                "complete_date": {
                    "type": bool,
                    "help_string": "Show complete date in long format",
                    "argstr": "-T",
                    "requires": ["long_format"],
                    "xor": ["date_format_str"],
                },
                "date_format_str": {
                    "type": str,
                    "help_string": "format string for ",
                    "argstr": "-D",
                    "requires": ["long_format"],
                    "xor": ["complete_date"],
                },
            },
            output_fields={
                "entries": {
                    "type": list,
                    "help_string": "list of entries returned by ls command",
                    "callable": list_entries,
                }
            },
        )

    else:
        assert False

    return Ls


def test_shell_fields(Ls):
    assert [a.name for a in attrs.fields(Ls.Inputs)] == [
        "executable",
        "args",
        "directory",
        "hidden",
        "long_format",
        "human_readable",
        "complete_date",
        "date_format_str",
    ]

    assert [a.name for a in attrs.fields(Ls.Outputs)] == [
        "return_code",
        "stdout",
        "stderr",
        "entries",
    ]


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

        @shell_task
        class A:
            executable = "cp"

            class Inputs:
                x: os.PathLike = shell_arg(
                    help_string="an input file", argstr="", position=0
                )
                y: str = shell_arg(
                    help_string="path of output file",
                    output_file_template="{x}_out",
                    argstr="",
                )

    elif request.param == "dynamic":
        A = shell_task(
            "A",
            executable="cp",
            input_fields={
                "x": {
                    "type": os.PathLike,
                    "help_string": "an input file",
                    "argstr": "",
                    "position": 0,
                },
                "y": {
                    "type": str,
                    "help_string": "path of output file",
                    "argstr": "",
                    "output_file_template": "{x}_out",
                },
            },
        )
    else:
        assert False

    return A


def test_shell_output_file_template(A):
    assert "y" in [a.name for a in attrs.fields(A.Outputs)]


def test_shell_output_field_name_static():
    @shell_task
    class A:
        executable = "cp"

        class Inputs:
            x: os.PathLike = shell_arg(
                help_string="an input file", argstr="", position=0
            )
            y: str = shell_arg(
                help_string="path of output file",
                output_file_template="{x}_out",
                output_field_name="y_out",
                argstr="",
            )

    assert "y_out" in [a.name for a in attrs.fields(A.Outputs)]


def test_shell_output_field_name_dynamic():
    A = shell_task(
        "A",
        executable="cp",
        input_fields={
            "x": {
                "type": os.PathLike,
                "help_string": "an input file",
                "argstr": "",
                "position": 0,
            },
            "y": {
                "type": str,
                "help_string": "path of output file",
                "argstr": "",
                "output_field_name": "y_out",
                "output_file_template": "{x}_out",
            },
        },
    )

    assert "y_out" in [a.name for a in attrs.fields(A.Outputs)]


def get_file_size(y: Path):
    result = os.stat(y)
    return result.st_size


def test_shell_bases_dynamic(A, tmpdir):
    B = shell_task(
        "B",
        output_fields={
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
    @shell_task
    class B(A):
        class Outputs:
            out_file_size: int = shell_out(
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
    A = shell_task(
        "A",
        "ls",
        input_fields={
            "directory": {
                "type": os.PathLike,
                "help_string": "input directory",
                "argstr": "",
                "position": -1,
            }
        },
        output_fields={
            "entries": {
                "type": list,
                "help_string": "list of entries returned by ls command",
                "callable": list_entries,
            }
        },
    )
    B = shell_task(
        "B",
        "ls",
        input_fields={
            "hidden": {
                "type": bool,
                "argstr": "-a",
                "help_string": "show hidden files",
                "default": False,
            }
        },
        bases=[A],
        inputs_bases=[A.Inputs],
    )

    Path.touch(tmpdir / ".hidden")

    b = B(directory=tmpdir, hidden=True)

    assert b.inputs.directory == tmpdir
    assert b.inputs.hidden
    assert b.cmdline == f"ls -a {tmpdir}"

    result = b()
    assert result.output.entries == [".", "..", ".hidden"]


def test_shell_inputs_outputs_bases_static(tmpdir):
    @shell_task
    class A:
        executable = "ls"

        class Inputs:
            directory: os.PathLike = shell_arg(
                help_string="input directory", argstr="", position=-1
            )

        class Outputs:
            entries: list = shell_out(
                help_string="list of entries returned by ls command",
                callable=list_entries,
            )

    @shell_task
    class B(A):
        class Inputs(A.Inputs):
            hidden: bool = shell_arg(
                help_string="show hidden files",
                argstr="-a",
                default=False,
            )

    Path.touch(tmpdir / ".hidden")

    b = B(directory=tmpdir, hidden=True)

    assert b.inputs.directory == tmpdir
    assert b.inputs.hidden

    result = b()
    assert result.output.entries == [".", "..", ".hidden"]


def test_shell_missing_executable_static():
    with pytest.raises(RuntimeError, match="should contain an `executable`"):

        @shell_task
        class A:
            class Inputs:
                directory: os.PathLike = shell_arg(
                    help_string="input directory", argstr="", position=-1
                )

            class Outputs:
                entries: list = shell_out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )


def test_shell_missing_executable_dynamic():
    with pytest.raises(RuntimeError, match="should contain an `executable`"):
        A = shell_task(
            "A",
            executable=None,
            input_fields={
                "directory": {
                    "type": os.PathLike,
                    "help_string": "input directory",
                    "argstr": "",
                    "position": -1,
                }
            },
            output_fields={
                "entries": {
                    "type": list,
                    "help_string": "list of entries returned by ls command",
                    "callable": list_entries,
                }
            },
        )


def test_shell_missing_inputs_static():
    with pytest.raises(RuntimeError, match="should contain an `Inputs`"):

        @shell_task
        class A:
            executable = "ls"

            class Outputs:
                entries: list = shell_out(
                    help_string="list of entries returned by ls command",
                    callable=list_entries,
                )


def test_shell_decorator_misuse(A):
    with pytest.raises(
        RuntimeError, match=("`shell_task` should not be provided any other arguments")
    ):
        shell_task(A, executable="cp")
