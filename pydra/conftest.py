import shutil


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
