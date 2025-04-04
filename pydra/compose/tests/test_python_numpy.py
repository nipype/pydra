import typing as ty
import importlib
from pathlib import Path
import pickle as pk
import numpy as np
import pytest


from pydra.engine.submitter import Submitter
from pydra.compose import python, workflow
from pydra.engine.tests.utils import Identity
from pydra.utils.hash import hash_function

if importlib.util.find_spec("numpy") is None:
    pytest.skip("can't find numpy library", allow_module_level=True)


@python.define(outputs=["b"])
def ArrayOut(val):
    return np.array([val, val])


def test_multiout(tmpdir):
    """testing a simple function that returns a numpy array"""

    @workflow.define(outputs=["array"])
    def Workflow(val):
        mo = workflow.add(ArrayOut(val=val))
        return mo.b

    wf = Workflow(val=2)

    with Submitter(worker="cf", cache_root=tmpdir, n_procs=2) as sub:
        results = sub(wf)

    assert np.array_equal(results.outputs.array, np.array([2, 2]))


def test_multiout_st(tmpdir):
    """testing a simple function that returns a numpy array, adding splitter"""

    @workflow.define(outputs=["array"])
    def Workflow(values):
        mo = workflow.add(ArrayOut().split(val=values).combine("val"))
        return mo.b

    wf = Workflow(values=[0, 1, 2])

    with Submitter(worker="cf", cache_root=tmpdir, n_procs=2) as sub:
        results = sub(wf)

    for el in range(3):
        assert np.array_equal(results.outputs.array[el], np.array([el, el]))


def test_numpy_hash_1():
    """hashing check for numeric numpy array"""
    A = np.array([1, 2])
    A_pk = pk.loads(pk.dumps(A))
    assert (A == A_pk).all()
    assert hash_function(A) == hash_function(A_pk)


def test_numpy_hash_2():
    """hashing check for numpy array of type object"""
    A = np.array([["NDAR"]], dtype=object)
    A_pk = pk.loads(pk.dumps(A))
    assert (A == A_pk).all()
    assert hash_function(A) == hash_function(A_pk)


def test_numpy_hash_3():
    """hashing check for numeric numpy array"""
    A = np.array([1, 2])
    B = np.array([3, 4])
    assert hash_function(A) != hash_function(B)


def test_task_numpyinput_1(tmp_path: Path):
    """task with numeric numpy array as an input"""
    nn = Identity().split(x=[np.array([1, 2]), np.array([3, 4])])
    # checking the results
    outputs = nn(cache_root=tmp_path)
    assert (np.array(outputs.out) == np.array([[1, 2], [3, 4]])).all()


def test_task_numpyinput_2(tmp_path: Path):
    """task with numpy array of type object as an input"""
    nn = Identity().split(
        x=[np.array(["VAL1"], dtype=object), np.array(["VAL2"], dtype=object)]
    )
    # checking the results
    outputs = nn(cache_root=tmp_path)
    assert outputs.out[0] == np.array(["VAL1"], dtype=object)
    assert outputs.out[1] == np.array(["VAL2"], dtype=object)


def test_numpy_fft():
    """checking if mark.task works for numpy functions"""
    np = pytest.importorskip("numpy")
    FFT = python.define(inputs={"a": np.ndarray}, outputs={"out": np.ndarray})(
        np.fft.fft
    )

    arr = np.array([[1, 10], [2, 20]])
    fft = FFT(a=arr)
    outputs = fft()
    assert np.allclose(np.fft.fft(arr), outputs.out)
