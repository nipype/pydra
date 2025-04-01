from pathlib import Path
import pytest


try:
    import importlib_resources
except ImportError:
    import importlib.resources as importlib_resources


@pytest.fixture(scope="package")
def data_tests_dir() -> Path:
    data_dir = importlib_resources.files("pydra").joinpath(
        "engine", "tests", "data_tests"
    )
    with importlib_resources.as_file(data_dir) as path:
        yield path
