"""Tests that file paths containing spaces are passed to shell commands as single
arguments (i.e. not split on the whitespace) both when generating the command args
and when actually running the command."""

import shlex
from pathlib import Path
import pytest
from fileformats.generic import File, Directory
from fileformats.text import TextFile
from pydra.compose import shell
from pydra.engine.submitter import Submitter
from pydra.environments import docker
from pydra.utils.general import attrs_values
from pydra.utils.typing import MultiInputObj
from pydra.engine.tests.utils import no_win, need_docker


@pytest.fixture
def spaced_dir(tmp_path: Path) -> Path:
    dpath = tmp_path / "dir with spaces"
    dpath.mkdir()
    return dpath


@pytest.fixture
def spaced_file(spaced_dir: Path) -> TextFile:
    fspath = spaced_dir / "file with spaces.txt"
    fspath.write_text("hello")
    return TextFile(fspath)


def _args(task: shell.Task) -> list[str]:
    """The arguments that would be passed to subprocess"""
    return task._command_args(values=attrs_values(task))


@pytest.mark.parametrize(
    "argstr,expected_prefix",
    [
        ("", []),
        ("-i", ["-i"]),
    ],
)
def test_shell_spaces_file_arg(
    spaced_file: TextFile, argstr: str, expected_prefix: list[str]
):
    """a single file input with spaces in its path should be a single argument"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_file: File = shell.arg(argstr=argstr, position=1, help="input file")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_file=spaced_file)
    assert _args(shelly) == ["cat"] + expected_prefix + [str(spaced_file)]


def test_shell_spaces_file_arg_template(spaced_file: TextFile):
    """a file input formatted with an argstr template, e.g. '--in={in_file}'"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        in_file: File = shell.arg(argstr="--in={in_file}", help="input file")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_file=spaced_file)
    assert _args(shelly) == ["cmd", f"--in={spaced_file}"]


def test_shell_spaces_file_arg_inline_template(spaced_file: TextFile):
    """a file input defined inline in the executable string"""
    Shelly = shell.define("cat <in_file:generic/file>")
    shelly = Shelly(in_file=spaced_file)
    assert _args(shelly) == ["cat", str(spaced_file)]


def test_shell_spaces_multi_file_arg(spaced_dir: Path):
    """a list of file inputs, each containing spaces"""
    fspaths = []
    for i in range(3):
        fspath = spaced_dir / f"file {i}.txt"
        fspath.write_text(str(i))
        fspaths.append(fspath)

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_files: MultiInputObj[File] = shell.arg(argstr="", help="input files")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_files=fspaths)
    assert _args(shelly) == ["cat"] + [str(p) for p in fspaths]


def test_shell_spaces_list_file_arg_ellipsis(spaced_dir: Path):
    """a list of file inputs with a repeated argstr, e.g. '-i file1 -i file2'"""
    fspaths = []
    for i in range(2):
        fspath = spaced_dir / f"file {i}.txt"
        fspath.write_text(str(i))
        fspaths.append(fspath)

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        in_files: list[File] = shell.arg(argstr="-i...", help="input files")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_files=fspaths)
    assert _args(shelly) == ["cmd", "-i", str(fspaths[0]), "-i", str(fspaths[1])]


def test_shell_spaces_str_arg():
    """a plain string input containing spaces (e.g. a label) should not be split"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "echo"
        label: str = shell.arg(argstr="--label", help="a label")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(label="a label with spaces")
    assert _args(shelly) == ["echo", "--label", "a label with spaces"]


def test_shell_spaces_cmdline_roundtrip(spaced_file: TextFile):
    """the 'cmdline' string should be correctly quoted so that it can be split back
    into the original arguments"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_file: File = shell.arg(argstr="", position=1, help="input file")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_file=spaced_file)
    assert shlex.split(shelly.cmdline) == ["cat", str(spaced_file)]


def test_shell_spaces_cmdline_roundtrip_quote_in_path(spaced_dir: Path):
    """paths containing both spaces and single quotes should also roundtrip"""
    fspath = spaced_dir / "it's a file.txt"
    fspath.write_text("hello")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_file: File = shell.arg(argstr="", position=1, help="input file")

        class Outputs(shell.Outputs):
            pass

    shelly = Shelly(in_file=fspath)
    assert _args(shelly) == ["cat", str(fspath)]
    assert shlex.split(shelly.cmdline) == ["cat", str(fspath)]


@pytest.mark.parametrize(
    "sep,expected",
    [
        (None, ["cmd", "--in", "a b", "c d"]),
        (",", ["cmd", "--in", "a b,c d"]),
        (", ", ["cmd", "--in", "a b, c d"]),
    ],
)
@pytest.mark.parametrize("argstr", ["--in", "--in {items}"])
def test_shell_spaces_list_sep(argstr: str, sep: str | None, expected: list[str]):
    """sep=None passes the items as separate args, otherwise they are joined into a
    single arg (regardless of whether the items or separator contain spaces)"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        items: list[str] = shell.arg(argstr=argstr, sep=sep, help="items")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(items=["a b", "c d"])) == expected


def test_shell_whitespace_sep_deprecated():
    """sep=" " used to (implicitly) mean separate args, it is now treated as
    sep=None with a deprecation warning"""
    with pytest.warns(DeprecationWarning, match="sep=' '"):

        @shell.define
        class Shelly(shell.Task["Shelly.Outputs"]):
            executable = "cmd"
            items: list[str] = shell.arg(argstr="--in", sep=" ", help="items")

            class Outputs(shell.Outputs):
                pass

    assert _args(Shelly(items=["a b", "c d"])) == ["cmd", "--in", "a b", "c d"]


def test_shell_spaces_bracketed_template(spaced_dir: Path):
    """values containing spaces within a bracketed template (e.g. ANTs style
    '-o [{a},{b}]'), including when one of the values is not set"""
    a = spaced_dir / "a file.txt"
    b = spaced_dir / "b file.txt"

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        a: str = shell.arg(argstr=None, help="a")
        b: str | None = shell.arg(argstr=None, default=None, help="b")
        out: str = shell.arg(argstr="-o [{a},{b}]", readonly=True, help="combined")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(a=str(a), b=str(b))) == ["cmd", "-o", f"[{a},{b}]"]
    assert _args(Shelly(a=str(a))) == ["cmd", "-o", f"[{a}]"]


def test_shell_spaces_template_attribute(spaced_file: TextFile):
    """format-string lookups on values (e.g. '{in_file.stem}') with spaces"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        in_file: File = shell.arg(argstr="--prefix {in_file.stem}", help="input")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(in_file=spaced_file)) == [
        "cmd",
        "--prefix",
        "file with spaces",
    ]


def test_shell_spaces_value_with_braces(spaced_dir: Path):
    """braces in values must not be interpreted as format fields"""
    fspath = spaced_dir / "file {a}.txt"
    fspath.write_text("hello")

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        in_file: File = shell.arg(argstr="--in={in_file}", help="input")
        a: str = shell.arg(argstr=None, default="injected", help="a")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(in_file=fspath)) == ["cmd", f"--in={fspath}"]


def test_shell_spaces_formatter_list(spaced_file: TextFile):
    """formatters can return a list of args to avoid them being split"""

    def formatter(in_file):
        return ["--in", str(in_file)]

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        in_file: File = shell.arg(formatter=formatter, help="input")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(in_file=spaced_file)) == ["cmd", "--in", str(spaced_file)]


@pytest.mark.parametrize(
    "formatted,expected",
    [
        ("  --in   a    b  ", ["--in", "a", "b"]),
        ("--in 'my  file.txt'", ["--in", "my  file.txt"]),
        ("   ", []),
    ],
)
def test_shell_formatter_str_whitespace(formatted: str, expected: list[str]):
    """runs of whitespace separate args, but whitespace within quotes is preserved"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        x: str = shell.arg(formatter=lambda x: formatted, help="x")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(x="dummy")) == ["cmd"] + expected


def test_shell_literal_ellipsis_in_argstr():
    """only a trailing '...' marks a repeated argstr, others are left as is"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        lo: int = shell.arg(argstr=None, help="lower")
        hi: int = shell.arg(argstr="--range {lo}...{hi}", help="upper")
        items: MultiInputObj[str] = shell.arg(argstr="-i...", help="items")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(lo=1, hi=5, items=["a"])) == [
        "cmd",
        "--range",
        "1...5",
        "-i",
        "a",
    ]


def test_shell_zero_value_repeated_args():
    """zero values are not dropped from '...' sequences"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cmd"
        values: list[float] = shell.arg(argstr="-x...", help="values")

        class Outputs(shell.Outputs):
            pass

    assert _args(Shelly(values=[1.0, 0.0])) == ["cmd", "-x", "1.0", "-x", "0.0"]


@no_win
def test_shell_spaces_run_input(spaced_file: TextFile, tmp_path: Path):
    """actually run a command on an input file with spaces in its path"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_file: File = shell.arg(argstr="", position=1, help="input file")

        class Outputs(shell.Outputs):
            pass

    outputs = Shelly(in_file=spaced_file)(cache_root=tmp_path / "cache")
    assert outputs.stdout == "hello"


@no_win
def test_shell_spaces_run_cache_root(spaced_file: TextFile, spaced_dir: Path):
    """run a command where the cache root (and therefore the location of the
    generated output files) contains spaces"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cp"
        in_file: File = shell.arg(argstr="", position=1, help="input file")

        class Outputs(shell.Outputs):
            out_file: File = shell.outarg(
                argstr="",
                position=2,
                path_template="{in_file}_copy",
                help="output file",
            )

    outputs = Shelly(in_file=spaced_file)(cache_root=spaced_dir / "cache root")
    assert " " in str(outputs.out_file)
    assert Path(outputs.out_file).read_text() == "hello"


@no_win
def test_shell_spaces_run_output_dir(spaced_dir: Path):
    """an output directory whose name contains spaces"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "mkdir"

        class Outputs(shell.Outputs):
            out_dir: Directory = shell.outarg(
                argstr="",
                position=1,
                path_template="out dir",
                help="output directory",
            )

    outputs = Shelly()(cache_root=spaced_dir / "cache")
    assert Path(outputs.out_dir).name == "out dir"
    assert Path(outputs.out_dir).is_dir()


@no_win
@need_docker
def test_shell_spaces_run_docker(spaced_file: TextFile, spaced_dir: Path):
    """input files with spaces in their paths need to be both mounted correctly
    and passed as single arguments in the container"""

    @shell.define
    class Shelly(shell.Task["Shelly.Outputs"]):
        executable = "cat"
        in_file: File = shell.arg(argstr="", position=1, help="input file")

        class Outputs(shell.Outputs):
            pass

    with Submitter(
        cache_root=spaced_dir / "cache root",
        environment=docker.Environment(image="busybox"),
    ) as sub:
        res = sub(Shelly(in_file=spaced_file))
    assert not res.errored, res.errors
    assert res.outputs.stdout == "hello"
