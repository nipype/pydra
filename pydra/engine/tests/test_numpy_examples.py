import numpy as np
import typing as ty
import os
import pytest

from ..submitter import Submitter
from ..core import Workflow
from ...mark import task, annotate


@task
@annotate({"return": {"b": ty.Any}})
def arrayout(val):
    return np.array([val, val])


def test_multiout(plugin):
    """ testing a simple function that returns a numpy array"""
    cache_dir = os.path.join(os.getcwd(), "cache3")
    wf = Workflow("wf", input_spec=["val"], val=[0, 1, 2], cache_dir=cache_dir)
    wf.add(arrayout(name="mo", val=wf.lzin.val, cache_dir=cache_dir))
    wf.mo.split("val").combine("val")

    wf.set_output([("b", wf.mo.lzout.b)])

    with Submitter(plugin=plugin, n_procs=2) as sub:
        sub(runnable=wf)

    results = wf.result(return_inputs=True)

    assert results[0] == {"wf.val": [0, 1, 2]}
    for el in range(3):
        assert np.array_equal(results[1].output.b[el], np.array([el, el]))
