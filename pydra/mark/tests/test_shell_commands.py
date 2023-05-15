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
                    default=None,
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


def test_shell_task_fields(Ls):
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


def test_shell_task_pickle_roundtrip(Ls, tmpdir):
    pkl_file = tmpdir / "ls.pkl"
    with open(pkl_file, "wb") as f:
        cp.dump(Ls, f)

    with open(pkl_file, "rb") as f:
        RereadLs = cp.load(f)

    assert RereadLs is Ls


@pytest.mark.xfail(
    reason=(
        "Need to change relationship between Inputs/Outputs and input_spec/output_spec "
        "for the task to run"
    )
)
def test_shell_task_init(Ls, tmpdir):
    inputs = Ls.Inputs(directory=tmpdir)
    assert inputs.directory == tmpdir
    assert not inputs.hidden
    outputs = Ls.Outputs(entries=[])
    assert outputs.entries == []


@pytest.mark.xfail(
    reason=(
        "Need to change relationship between Inputs/Outputs and input_spec/output_spec "
        "for the task to run"
    )
)
def test_shell_task_run(Ls, tmpdir):
    Path.touch(tmpdir / "a")
    Path.touch(tmpdir / "b")
    Path.touch(tmpdir / "c")

    ls = Ls(directory=tmpdir)

    result = ls()

    assert result.output.entries == ["a", "b", "c"]
