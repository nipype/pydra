import numpy as np
import typing as ty
import importlib
import pytest
import pickle as pk

from ..submitter import Submitter
from ..core import Workflow
from ...mark import task, annotate
from .utils import identity
from ..helpers import hash_value

if importlib.util.find_spec("numpy") is None:
    pytest.skip("can't find numpy library", allow_module_level=True)


@task
@annotate({"return": {"b": ty.Any}})
def arrayout(val):
    return np.array([val, val])


def test_multiout(tmpdir):
    """testing a simple function that returns a numpy array"""
    wf = Workflow("wf", input_spec=["val"], val=2)
    wf.add(arrayout(name="mo", val=wf.lzin.val))

    wf.set_output([("array", wf.mo.lzout.b)])
    wf.cache_dir = tmpdir

    with Submitter(plugin="cf", n_procs=2) as sub:
        sub(runnable=wf)

    results = wf.result(return_inputs=True)

    assert results[0] == {"wf.val": 2}
    assert np.array_equal(results[1].output.array, np.array([2, 2]))


def test_multiout_st(tmpdir):
    """testing a simple function that returns a numpy array, adding splitter"""
    wf = Workflow("wf", input_spec=["val"], val=[0, 1, 2])
    wf.add(arrayout(name="mo", val=wf.lzin.val))
    wf.mo.split("val").combine("val")

    wf.set_output([("array", wf.mo.lzout.b)])
    wf.cache_dir = tmpdir

    with Submitter(plugin="cf", n_procs=2) as sub:
        sub(runnable=wf)

    results = wf.result(return_inputs=True)

    assert results[0] == {"wf.val": [0, 1, 2]}
    for el in range(3):
        assert np.array_equal(results[1].output.array[el], np.array([el, el]))


def test_numpy_hash_1():
    """hashing check for numeric numpy array"""
    A = np.array([1, 2])
    A_pk = pk.loads(pk.dumps(A))
    assert (A == A_pk).all()
    assert hash_value(A) == hash_value(A_pk)


def test_numpy_hash_2():
    """hashing check for numpy array of type object"""
    A = np.array([["NDAR"]], dtype=object)
    A_pk = pk.loads(pk.dumps(A))
    assert (A == A_pk).all()
    assert hash_value(A) == hash_value(A_pk)


def test_task_numpyinput_1(tmpdir):
    """task with numeric numpy array as an input"""
    nn = identity(name="NA", x=[np.array([1, 2]), np.array([3, 4])])
    nn.cache_dir = tmpdir
    nn.split("x")
    # checking the results
    results = nn()
    assert (results[0].output.out == np.array([1, 2])).all()
    assert (results[1].output.out == np.array([3, 4])).all()


def test_task_numpyinput_2(tmpdir):
    """task with numpy array of type object as an input"""
    nn = identity(
        name="NA",
        x=[np.array(["VAL1"], dtype=object), np.array(["VAL2"], dtype=object)],
    )
    nn.cache_dir = tmpdir
    nn.split("x")
    # checking the results
    results = nn()
    assert (results[0].output.out == np.array(["VAL1"], dtype=object)).all()
