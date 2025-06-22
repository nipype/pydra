from pydra.scripts.cli import crash
from pydra.utils.general import default_run_cache_root
from pydra.tasks.testing import Divide
from traceback import format_exception
import typing as ty


# @pytest.mark.xfail(reason="Need to fix a couple of things after syntax changes")
def test_crash_cli(cli_runner):
    result = cli_runner(
        crash,
        [
            f"{default_run_cache_root}/{Divide(x=15, y=0)._checksum}/_error.pklz",
            "--rerun",
            "--debugger",
            "pdb",
        ],
    )
    assert result.exit_code == 0, show_cli_trace(result)


def show_cli_trace(result: ty.Any) -> str:
    "Used in testing to show traceback of CLI output"
    return "".join(format_exception(*result.exc_info))
