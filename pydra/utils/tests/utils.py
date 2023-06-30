from fileformats.generic import File
from fileformats.core.mixin import WithSeparateHeader
from pydra import mark
from pydra.engine.task import ShellCommandTask
from pydra.engine import specs


class MyFormat(File):
    ext = ".my"


class MyHeader(File):
    ext = ".hdr"


class MyFormatX(WithSeparateHeader, MyFormat):
    header_type = MyHeader


@mark.task
def generic_func_task(in_file: File) -> File:
    return in_file


generic_shell_input_fields = [
    (
        "in_file",
        File,
        {
            "help_string": "the input file",
            "argstr": "",
            "copyfile": "copy",
        },
    ),
    (
        "out",
        str,
        {
            "help_string": "output file name",
            "argstr": "",
            "position": -1,
            "output_file_template": "{in_file}",
        },
    ),
]

generic_shell_input_spec = specs.SpecInfo(
    name="Input", fields=generic_shell_input_fields, bases=(specs.ShellSpec,)
)

generic_shell_output_fields = [
    (
        "out",
        File,
        {
            "help_string": "output file",
        },
    ),
]
generic_shelloutput_spec = specs.SpecInfo(
    name="Output", fields=generic_shell_output_fields, bases=(specs.ShellOutSpec,)
)


class GenericShellTask(ShellCommandTask):
    input_spec = generic_shell_input_spec
    output_spec = generic_shelloutput_spec
    executable = "echo"


@mark.task
def specific_func_task(in_file: MyFormatX) -> MyFormatX:
    return in_file


specific_shell_input_fields = [
    (
        "in_file",
        MyFormatX,
        {
            "help_string": "the input file",
            "argstr": "",
            "copyfile": "copy",
            "sep": " ",
        },
    ),
    (
        "out",
        str,
        {
            "help_string": "output file name",
            "argstr": "",
            "position": -1,
            "output_file_template": "{in_file}",  # Pass through un-altered
        },
    ),
]

specific_shell_input_spec = specs.SpecInfo(
    name="Input", fields=specific_shell_input_fields, bases=(specs.ShellSpec,)
)

specific_shell_output_fields = [
    (
        "out",
        MyFormatX,
        {
            "help_string": "output file",
        },
    ),
]
specific_shelloutput_spec = specs.SpecInfo(
    name="Output", fields=specific_shell_output_fields, bases=(specs.ShellOutSpec,)
)


class SpecificShellTask(ShellCommandTask):
    input_spec = specific_shell_input_spec
    output_spec = specific_shelloutput_spec
    executable = "echo"
