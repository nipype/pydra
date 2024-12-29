from fileformats.generic import File
from fileformats.core.mixin import WithSeparateHeader, WithMagicNumber
from pydra.design import python
from pydra.engine.task import ShellTask
from pydra.engine import specs
from pydra.design import shell, python


class MyFormat(WithMagicNumber, File):
    ext = ".my"
    magic_number = b"MYFORMAT"


class MyHeader(File):
    ext = ".hdr"


class MyFormatX(WithSeparateHeader, MyFormat):
    header_type = MyHeader


class MyOtherFormatX(WithMagicNumber, WithSeparateHeader, File):
    magic_number = b"MYFORMAT"
    ext = ".my"
    header_type = MyHeader


@python.define
def generic_func_task(in_file: File) -> File:
    return in_file


@shell.define
class GenericShellTask(specs.ShellDef["GenericShellTask.Outputs"]):
    """class with customized input and executables"""

    in_file: File = shell.arg(
        help_string="the input file",
        argstr="",
        copy_mode="copy",
    )

    class Outputs(specs.ShellOutputs):
        out: File = shell.outarg(
            help_string="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",
        )

    executable = "echo"


@python.define
def specific_func_task(in_file: MyFormatX) -> MyFormatX:
    return in_file


@shell.define
class SpecificShellTask(specs.ShellDef["SpecificShellTask.Outputs"]):
    executable = "echo"

    in_file: MyFormatX = shell.arg(
        help_string="the input file",
        argstr="",
        copy_mode="copy",
        sep=" ",
    )

    class Outputs(specs.ShellOutputs):
        out: MyFormatX = shell.outarg(
            help_string="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",  # Pass through un-altered
        )


@python.define
def other_specific_func_task(in_file: MyOtherFormatX) -> MyOtherFormatX:
    return in_file


class OtherSpecificShellTask(ShellTask):

    in_file: MyOtherFormatX = shell.arg(
        help_string="the input file",
        argstr="",
        copy_mode="copy",
        sep=" ",
    )

    class Outputs(specs.ShellOutputs):
        out: MyOtherFormatX = shell.outarg(
            help_string="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",  # Pass through un-altered
        )

    executable = "echo"
