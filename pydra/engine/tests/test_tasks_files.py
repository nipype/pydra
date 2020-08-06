import os
from pathlib import Path
import numpy as np
import pytest
import typing as ty

from ..submitter import Submitter
from ..core import Workflow
from ... import mark
from ..specs import File, Directory


@mark.task
def dir_count_file(dirpath: Directory):
    return len(os.listdir(dirpath))


@mark.task
def file_append_text(file: File):
    with open(file, "a") as f:
        f.write("!")
    return f


@mark.task
def file_np_add2(file):
    array_inp = np.load(file)
    array_out = array_inp + 2
    cwd = os.getcwd()
    # providing a full path
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@mark.task
def file_np_mult(file):
    array_inp = np.load(file)
    array_out = 10 * array_inp
    cwd = os.getcwd()
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@mark.task
def file_np_add2_annot(file: File) -> ty.NamedTuple("Output", [("out", File)]):
    array_inp = np.load(file)
    array_out = array_inp + 2
    cwd = os.getcwd()
    # providing a full path
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@mark.task
def file_np_mult_annot(file: File) -> ty.NamedTuple("Output", [("out", File)]):
    array_inp = np.load(file)
    array_out = 10 * array_inp
    cwd = os.getcwd()
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


def test_text_task(tmpdir):
    file = tmpdir.join("file.txt")
    with open(file, "w") as f:
        f.write("hello")
    nn = file_append_text(name="add_text", file=file)

    with Submitter(plugin="cf") as sub:
        sub(nn)

    results = nn.result()
    with open(file, "r") as text:
        text.read()
    assert text == "hello!"


def test_broken_task(tmpdir):
    """ task that takes file as an input"""
    os.chdir(tmpdir)
    file = os.path.join(os.getcwd(), "file.txt")
    nn = file_append_text(name="add_text", file=file)

    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn)
    assert "File doesn't exist" in str(e.value)


def test_task_np_1(tmpdir):
    """ task that takes file as an input"""
    os.chdir(tmpdir)
    arr = np.array([2])
    # creating abs path
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = file_np_add2(name="add2", file=file)

    with Submitter(plugin="cf") as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    res = np.load(results.output.out)
    assert res == np.array([4])


def test_wf_np_1(tmpdir):
    """ workflow with 2 tasks that take file as an input and give file as an aoutput"""
    wf = Workflow(name="wf_1", input_spec=["file_orig"])
    wf.add(file_np_add2(name="add2", file=wf.lzin.file_orig))
    wf.add(file_np_mult(name="mult", file=wf.add2.lzout.out))
    wf.set_output([("out", wf.mult.lzout.out)])

    os.chdir(tmpdir)
    arr = np.array([2, 3])
    # creating abs path
    file_orig = os.path.join(os.getcwd(), "arr_orig.npy")
    np.save(file_orig, arr)
    wf.inputs.file_orig = file_orig

    with Submitter(plugin="cf") as sub:
        sub(wf)

    assert wf.output_dir.exists()
    file_output = wf.result().output.out
    assert Path(file_output).exists()
    # loading results
    array_out = np.load(file_output)
    assert np.array_equal(array_out, [40, 50])


def test_file_annotation_np_1(tmpdir):
    """ task that takes file as an input"""
    os.chdir(tmpdir)
    arr = np.array([2])
    # creating abs path
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = file_np_add2_annot(name="add2", file=file)

    with Submitter(plugin="cf") as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    res = np.load(results.output.out)
    assert res == np.array([4])
