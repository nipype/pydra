import shutil
import os
import pytest

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--with-dask", action="store_true", help="run all combinations")
    parser.addoption(
        "--with-psij",
        action="store_true",
        help="run with psij workers in test matrix",
    )
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
            with_psij = metafunc.config.getoption("with_psij")
        except ValueError:
            with_psij = False
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
                if with_psij:
                    available_workers.append("psij-slurm")
            if with_psij:
                available_workers.append("psij-local")
        else:
            available_workers = [only_worker]
        # Set the available workers as a parameter to the
        # test function
        metafunc.parametrize("any_worker", available_workers)


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call):
        raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo):
        raise excinfo.value


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
