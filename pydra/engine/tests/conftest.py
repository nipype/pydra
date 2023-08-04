import pytest


try:
    import importlib_resources
except ImportError:
    import importlib.resources as importlib_resources


@pytest.fixture(scope="package")
def data_tests_dir():
    test_nii = importlib_resources.files("pydra").joinpath(
        "engine", "tests", "data_tests"
    )
    with importlib_resources.as_file(test_nii) as path:
        yield path
