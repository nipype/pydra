import shutil
import os
import pytest
import typing as ty
from click.testing import CliRunner, Result as CliResult

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--with-dask", action="store_true", help="run all combinations")
    parser.addoption(
        "--only-worker",
        help="only run tests with provided worker",
    )


@pytest.fixture(scope="session", params=["debug", "cf"])
def worker(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "any_worker" in metafunc.fixturenames:
        try:
            with_dask = metafunc.config.getoption("with_dask")
        except ValueError:
            with_dask = False
        try:
            only_worker = metafunc.config.getoption("only_worker")
        except ValueError:
            only_worker = None
        if only_worker is None:
            available_workers = ["debug", "cf"]
            if with_dask:
                available_workers.append("dask")
            if bool(shutil.which("sbatch")):
                available_workers.append("slurm")
        else:
            available_workers = [only_worker]
        # Set the available workers as a parameter to the
        # test function
        metafunc.parametrize("any_worker", available_workers)


@pytest.fixture
def cli_runner(catch_cli_exceptions: bool) -> ty.Callable[..., ty.Any]:
    def invoke(
        *args: ty.Any, catch_exceptions: bool = catch_cli_exceptions, **kwargs: ty.Any
    ) -> CliResult:
        runner = CliRunner()
        result = runner.invoke(*args, catch_exceptions=catch_exceptions, **kwargs)  # type: ignore[misc]
        return result

    return invoke


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo[ty.Any]) -> None:
        if call.excinfo is not None:
            raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
        raise excinfo.value

    CATCH_CLI_EXCEPTIONS = False
else:
    CATCH_CLI_EXCEPTIONS = True


@pytest.fixture
def catch_cli_exceptions() -> bool:
    return CATCH_CLI_EXCEPTIONS


# Example VSCode launch configuration for debugging unittests
# {
#     "name": "Test Config",
#     "type": "python",
#     "request": "launch",
#     "purpose": ["debug-test"],
#     "justMyCode": false,
#     "console": "internalConsole",
#     "env": {
#         "_PYTEST_RAISE": "1"
#     },
# }
