"""Tests for `pydra.engine.utils`."""
import pytest

from pydra.engine import utils

def test__check_dependencies():
    @utils._check_dependencies("42", "hunter3")
    def foo():
        return True

    with pytest.raises(ImportError):
        foo()

    @utils._check_dependencies("os")
    def bar():
        return True

    assert bar()
