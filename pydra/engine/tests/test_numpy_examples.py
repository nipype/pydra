import numpy as np
import typing as ty
import importlib
import pytest

from ..submitter import Submitter
from ..core import Workflow
from ...mark import task, annotate

if importlib.util.find_spec("numpy") is None:
    pytest.skip("can't find numpy library", allow_module_level=True)


@task
@annotate({"return": {"b": ty.Any}})
def arrayout(val):
    return np.array([val, val])


def test_multiout(plugin, tmpdir):
    """ testing a simple function that returns a numpy array"""
    wf = Workflow("wf", input_spec=["val"], val=2)
    wf.add(arrayout(name="mo", val=wf.lzin.val))

    wf.set_output([("array", wf.mo.lzout.b)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin, n_procs=2) as sub:
        sub(runnable=wf)

    results = wf.result(return_inputs=True)

    assert results[0] == {"wf.val": 2}
    assert np.array_equal(results[1].output.array, np.array([2, 2]))


def test_multiout_st(plugin, tmpdir):
    """ testing a simple function that returns a numpy array, adding splitter"""
    wf = Workflow("wf", input_spec=["val"], val=[0, 1, 2])
    wf.add(arrayout(name="mo", val=wf.lzin.val))
    wf.mo.split("val").combine("val")

    wf.set_output([("array", wf.mo.lzout.b)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin, n_procs=2) as sub:
        sub(runnable=wf)

    results = wf.result(return_inputs=True)

    assert results[0] == {"wf.val": [0, 1, 2]}
    for el in range(3):
        assert np.array_equal(results[1].output.array[el], np.array([el, el]))
