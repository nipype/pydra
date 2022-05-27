import shutil
import os
import pytest

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--dask", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "plugin_dask_opt" in metafunc.fixturenames:
        if bool(shutil.which("sbatch")):
            Plugins = ["slurm"]
        else:
            Plugins = ["cf"]
        if metafunc.config.getoption("dask"):
            Plugins.append("dask")
        metafunc.parametrize("plugin_dask_opt", Plugins)

    if "plugin" in metafunc.fixturenames:
        if metafunc.config.getoption("dask"):
            Plugins = []
        elif bool(shutil.which("sbatch")):
            Plugins = ["slurm"]
        else:
            Plugins = ["cf"]
        metafunc.parametrize("plugin", Plugins)


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
