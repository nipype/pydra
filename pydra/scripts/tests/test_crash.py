import pytest
from pydra.scripts.cli import crash
from pydra.tasks.testing import Divide
from traceback import format_exception
import typing as ty


# @pytest.mark.xfail(reason="Need to fix a couple of things after syntax changes")
def test_crash_cli(cli_runner, tmp_path):
    divide = Divide(x=15, y=0)
    with pytest.raises(ZeroDivisionError):
        divide(cache_root=tmp_path)

    result = cli_runner(
        crash,
        [
            f"{tmp_path}/{divide._checksum}/_error.pklz",
            "--rerun",
            "--debugger",
            "pdb",
        ],
    )
    assert result.exit_code == 0, show_cli_trace(result)


def show_cli_trace(result: ty.Any) -> str:
    "Used in testing to show traceback of CLI output"
    return "".join(format_exception(*result.exc_info))
