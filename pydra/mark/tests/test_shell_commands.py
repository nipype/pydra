import os
import tempfile
from pathlib import Path
import attrs
import pydra.engine
from pydra.mark import cmd_arg, cmd_out


def test_shell_cmd():
    @attrs.define(kw_only=True, slots=False)
    class LsInputSpec(pydra.specs.ShellSpec):
        directory: os.PathLike = cmd_arg(
            help_string="the directory to list the contents of",
            argstr="",
            mandatory=True,
        )
        hidden: bool = cmd_arg(help_string=("display hidden FS objects"), argstr="-a")
        long_format: bool = cmd_arg(
            help_string=(
                "display properties of FS object, such as permissions, size and timestamps "
            ),
            argstr="-l",
        )
        human_readable: bool = cmd_arg(
            help_string="display file sizes in human readable form",
            argstr="-h",
            requires=["long_format"],
        )
        complete_date: bool = cmd_arg(
            help_string="Show complete date in long format",
            argstr="-T",
            requires=["long_format"],
            xor=["date_format_str"],
        )
        date_format_str: str = cmd_arg(
            help_string="format string for ",
            argstr="-D",
            requires=["long_format"],
            xor=["complete_date"],
        )

    def list_outputs(stdout):
        return stdout.split("\n")[:-1]

    @attrs.define(kw_only=True, slots=False)
    class LsOutputSpec(pydra.specs.ShellOutSpec):
        entries: list = cmd_out(
            help_string="list of entries returned by ls command", callable=list_outputs
        )

    class Ls(pydra.engine.ShellCommandTask):
        """Task definition for the `ls` command line tool"""

        executable = "ls"

        input_spec = pydra.specs.SpecInfo(
            name="LsInput",
            bases=(LsInputSpec,),
        )

        output_spec = pydra.specs.SpecInfo(
            name="LsOutput",
            bases=(LsOutputSpec,),
        )

    tmpdir = Path(tempfile.mkdtemp())
    Path.touch(tmpdir / "a")
    Path.touch(tmpdir / "b")
    Path.touch(tmpdir / "c")

    ls = Ls(directory=tmpdir)

    result = ls()

    assert result.output.entries == ["a", "b", "c"]
