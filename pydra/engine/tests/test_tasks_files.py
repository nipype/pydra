import os, shutil
import numpy as np
import pytest

from ..submitter import Submitter
from ..task import to_task


@to_task
def file_add2(file):
    array_inp = np.load(file)
    array_out = array_inp + 2
    cwd = os.getcwd()
    # providing a full path
    file_out = os.path.join(cwd, "arr_out.npy")
    # providing relative path
    file_out_default = os.path.join("arr_out.npy")
    np.save(file_out, array_out)
    return file_out, file_out_default


@to_task
def file_mult(file1, file2):
    array_inp1 = np.load(file1)
    array_inp2 = np.load(file2)
    array_out = array_inp1 * array_inp2
    cwd = os.getcwd()
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


def test_1(tmpdir):
    """ using full path in the function"""
    os.chdir(tmpdir)
    arr = np.array([2])
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = file_add2(name="add2", file=file)
    # assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin="cf") as sub:
        sub(nn)

    # checking the results
    results = nn.result()

    res = np.load(results.output.out[0])
    assert res == np.array([4])


@pytest.mark.xfail(reason="doesn't work with relative path (can't find the file), TODO")
def test_default_out_1(tmpdir):
    """ using relative path in the function"""
    os.chdir(tmpdir)
    arr = np.array([2])
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = file_add2(name="add2", file=file)
    # assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin="cf") as sub:
        sub(nn)

    # checking the results
    results = nn.result()

    res = np.load(results.output.out[1])
    assert res == np.array([4])
