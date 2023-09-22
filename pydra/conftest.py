import shutil
import os
import pytest

os.environ["NO_ET"] = "true"


def pytest_addoption(parser):
    parser.addoption("--dask", action="store_true", help="run all combinations")
    parser.addoption(
        "--psij",
        action="store",
        help="run with psij subtype plugin",
        choices=["local", "slurm"],
    )


def pytest_generate_tests(metafunc):
    if "plugin_dask_opt" in metafunc.fixturenames:
        if bool(shutil.which("sbatch")):
            Plugins = ["slurm"]
        else:
            Plugins = ["cf"]
        try:
            if metafunc.config.getoption("dask"):
                Plugins.append("dask")
        except ValueError:
            # Called as --pyargs, so --dask isn't available
            pass
        try:
            if metafunc.config.getoption("psij"):
                Plugins.append("psij-" + metafunc.config.getoption("psij"))
                if (
                    bool(shutil.which("sbatch"))
                    and metafunc.config.getoption("psij") == "slurm"
                ):
                    Plugins.remove("slurm")
        except ValueError:
            pass
        metafunc.parametrize("plugin_dask_opt", Plugins)

    if "plugin" in metafunc.fixturenames:
        use_dask = False
        try:
            use_dask = metafunc.config.getoption("dask")
        except ValueError:
            pass
        if use_dask:
            Plugins = []
        elif bool(shutil.which("sbatch")):
            Plugins = ["slurm"]
        else:
            Plugins = ["cf"]
        try:
            if metafunc.config.getoption("psij"):
                Plugins.append("psij-" + metafunc.config.getoption("psij"))
                if (
                    bool(shutil.which("sbatch"))
                    and metafunc.config.getoption("psij") == "slurm"
                ):
                    Plugins.remove("slurm")
        except ValueError:
            pass
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
