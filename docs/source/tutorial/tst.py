import os
from pathlib import Path
from fileformats.generic import File
from pydra.compose import shell
from pydra.utils import print_help


# Arguments to the callable function can be one of
def get_file_size(out_file: Path) -> int:
    """Calculate the file size"""
    result = os.stat(out_file)
    return result.st_size


ACommand = shell.define(
    "a-command",
    inputs={
        "in_file": shell.arg(type=File, help="output file", argstr="", position=-2)
    },
    outputs={
        "out_file": shell.outarg(type=File, help="output file", argstr="", position=-1),
        "out_file_size": {
            "type": int,
            "help": "size of the output directory",
            "callable": get_file_size,
        },
    },
)

print_help(ACommand)
