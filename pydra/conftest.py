import shutil
import os
import pytest

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--dask", action="store_true", help="run all combinations")
    parser.addoption(
        "--with-psij",
        action="store_true",
        help="run with psij workers in test matrix",
    )
    parser.addoption(
        "--only-slurm",
        action="store_true",
        help="only run tests with slurm workers",
    )


@pytest.fixture(scope="session", params=["debug", "cf"])
def worker(request):
    return request.param


def pytest_generate_tests(metafunc):
    if "any_worker" in metafunc.fixturenames:
        try:
            use_dask = metafunc.config.getoption("dask")
        except ValueError:
            use_dask = False
        try:
            use_psij = metafunc.config.getoption("with_psij")
        except ValueError:
            use_psij = False
        try:
            only_slurm = metafunc.config.getoption("only_slurm")
        except ValueError:
            only_slurm = False
        available_workers = ["debug", "cf"] if not only_slurm else []
        if use_dask:
            available_workers.append("dask")
        if bool(shutil.which("sbatch")):
            available_workers.append("slurm")
            if use_psij:
                available_workers.append("psij-slurm")
        if use_psij and not only_slurm:
            available_workers.append("psij-local")
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
