import pytest
from pydra.engine.submitter import Submitter
from pydra.tasks.testing import SafeDivisionWorkflow, UnsafeDivisionWorkflow


def test_safe_division_workflow():
    wf = SafeDivisionWorkflow(a=10, b=5).split(denominator=[3, 2, 0])
    with Submitter(worker="cf") as sub:
        result = sub(wf)

    assert not result.errored, "\n".join(result.errors["error message"])


def test_unsafe_division_workflow():
    wf = UnsafeDivisionWorkflow(a=10, b=5).split(denominator=[3, 2, 0])

    with pytest.raises(ZeroDivisionError):
        wf(worker="debug")
