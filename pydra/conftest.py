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


def pytest_generate_tests(metafunc):
    if "worker" in metafunc.fixturenames:
        metafunc.parametrize("worker", ["debug", "cf"])

    if "any_worker" in metafunc.fixturenames:
        available_workers = ["debug", "cf"]
        try:
            use_dask = metafunc.config.getoption("dask")
        except ValueError:
            use_dask = False
        try:
            use_psij = metafunc.config.getoption("with-psij")
        except ValueError:
            use_psij = False
        if use_dask:
            available_workers.append("dask")
        if bool(shutil.which("sbatch")):
            available_workers.append("slurm")
            if use_psij:
                available_workers.append("psij-slurm")
        if use_psij:
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
