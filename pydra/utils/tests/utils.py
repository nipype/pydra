from fileformats.generic import File, BinaryFile
from fileformats.core.mixin import WithSeparateHeader, WithMagicNumber
from pydra.compose import shell, python


class MyFormat(WithMagicNumber, BinaryFile):
    ext = ".my"
    magic_number = b"MYFORMAT"


class MyHeader(File):
    ext = ".hdr"


class MyFormatX(WithSeparateHeader, MyFormat):
    header_type = MyHeader


class MyOtherFormatX(WithMagicNumber, WithSeparateHeader, BinaryFile):
    magic_number = b"MYFORMAT"
    ext = ".my"
    header_type = MyHeader


@python.define
def GenericFuncTask(in_file: File) -> File:
    return in_file


@shell.define
class GenericShellTask(shell.Task["GenericShellTask.Outputs"]):
    """class with customized input and executables"""

    in_file: File = shell.arg(
        help="the input file",
        argstr="",
        copy_mode="copy",
    )

    class Outputs(shell.Outputs):
        out: File = shell.outarg(
            help="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",
        )

    executable = "echo"


@python.define
def SpecificFuncTask(in_file: MyFormatX) -> MyFormatX:
    return in_file


@shell.define
class SpecificShellTask(shell.Task["SpecificShellTask.Outputs"]):
    executable = "echo"

    in_file: MyFormatX = shell.arg(
        help="the input file",
        argstr="",
        copy_mode="copy",
    )

    class Outputs(shell.Outputs):
        out: MyFormatX = shell.outarg(
            help="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",  # Pass through un-altered
        )


@python.define
def OtherSpecificFuncTask(in_file: MyOtherFormatX) -> MyOtherFormatX:
    return in_file


class OtherSpecificShellTask(shell.Task):

    in_file: MyOtherFormatX = shell.arg(
        help="the input file",
        argstr="",
        copy_mode="copy",
    )

    class Outputs(shell.Outputs):
        out: MyOtherFormatX = shell.outarg(
            help="output file name",
            argstr="",
            position=-1,
            path_template="{in_file}",  # Pass through un-altered
        )

    executable = "echo"
