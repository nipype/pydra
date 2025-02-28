import os
from pathlib import Path
import numpy as np
import pytest
import typing as ty

from ..submitter import Submitter
from pydra.design import python, workflow
from fileformats.generic import File, Directory


@python.define
def DirCountFile(dirpath: Directory) -> int:
    return len(os.listdir(dirpath))


@python.define
def DirCountFileAnnot(dirpath: Directory) -> int:
    return len(os.listdir(dirpath))


@python.define
def FileAdd2(file: File) -> File:
    array_inp = np.load(file)
    array_out = array_inp + 2
    cwd = os.getcwd()
    # providing a full path
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@python.define
def FileMult(file: File) -> File:
    array_inp = np.load(file)
    array_out = 10 * array_inp
    cwd = os.getcwd()
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@python.define
def FileAdd2Annot(file: File) -> File:
    array_inp = np.load(file)
    array_out = array_inp + 2
    cwd = os.getcwd()
    # providing a full path
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


@python.define
def FileMultAnnot(file: File) -> File:
    array_inp = np.load(file)
    array_out = 10 * array_inp
    cwd = os.getcwd()
    file_out = os.path.join(cwd, "arr_out.npy")
    np.save(file_out, array_out)
    return file_out


def test_task_1(tmpdir):
    """task that takes file as an input"""
    os.chdir(tmpdir)
    arr = np.array([2])
    # creating abs path
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = FileAdd2(file=file)

    with Submitter(worker="cf") as sub:
        res = sub(nn)

    # checking the results

    result = np.load(res.outputs.out)
    assert result == np.array([4])


def test_wf_1(tmpdir):
    """workflow with 2 tasks that take file as an input and give file as an aoutput"""

    @workflow.define
    def Workflow(file_orig: File):
        add2 = workflow.add(FileAdd2(file=file_orig))
        mult = workflow.add(FileMult(file=add2.out))
        return mult.out

    os.chdir(tmpdir)
    arr = np.array([2, 3])
    # creating abs path
    file_orig = os.path.join(os.getcwd(), "arr_orig.npy")
    np.save(file_orig, arr)
    wf = Workflow(file_orig=file_orig)

    with Submitter(worker="cf") as sub:
        res = sub(wf)

    assert res.output_dir.exists()
    file_output = res.outputs.out
    assert Path(file_output).exists()
    # loading results
    array_out = np.load(file_output)
    assert np.array_equal(array_out, [40, 50])


def test_file_annotation_1(tmpdir):
    """task that takes file as an input"""
    os.chdir(tmpdir)
    arr = np.array([2])
    # creating abs path
    file = os.path.join(os.getcwd(), "arr1.npy")
    np.save(file, arr)
    nn = FileAdd2Annot(file=file)

    with Submitter(worker="cf") as sub:
        res = sub(nn)

    # checking the results
    assert res.errored is False, " ".join(res.errors["error message"])
    arr = np.load(res.outputs.out)
    assert arr == np.array([4])


def test_broken_file(tmpdir):
    """task that takes file as an input"""
    os.chdir(tmpdir)
    file = os.path.join(os.getcwd(), "non_existent.npy")

    with pytest.raises(FileNotFoundError):
        with Submitter(worker="cf") as sub:
            sub(FileAdd2(file=file))

    with pytest.raises(FileNotFoundError, match="do not exist"):
        FileAdd2Annot(file=file)


def test_broken_file_link(tmpdir):
    """
    Test how broken symlinks are handled during hashing
    """
    os.chdir(tmpdir)
    file = os.path.join(os.getcwd(), "arr.npy")
    arr = np.array([2])
    np.save(file, arr)

    file_link = os.path.join(os.getcwd(), "link_to_arr.npy")
    os.symlink(file, file_link)
    os.remove(file)

    # raises error inside task
    # unless variable is defined as a File pydra will treat it as a string
    with pytest.raises(FileNotFoundError):
        with Submitter(worker="cf") as sub:
            sub(FileAdd2(file=file_link))

    with pytest.raises(FileNotFoundError, match="do not exist"):
        FileAdd2Annot(file=file_link)


def test_broken_dir():
    """Test how broken directories are handled during hashing"""

    # unless variable is defined as a File pydra will treat it as a string
    with pytest.raises(FileNotFoundError):
        with Submitter(worker="cf") as sub:
            sub(DirCountFile(dirpath="/broken_dir_path/"))

    # raises error before task is run
    with pytest.raises(FileNotFoundError):
        DirCountFileAnnot(dirpath="/broken_dir_path/")


def test_broken_dir_link1(tmpdir):
    """
    Test how broken symlinks are hashed in hash_dir
    """
    # broken symlink to dir path
    dir1 = tmpdir.join("dir1")
    os.mkdir(dir1)
    dir1_link = tmpdir.join("dir1_link")
    os.symlink(dir1, dir1_link)
    os.rmdir(dir1)

    # raises error while running task
    with pytest.raises(FileNotFoundError):
        with Submitter(worker="cf") as sub:
            sub(DirCountFile(dirpath=Path(dir1)))

    with pytest.raises(FileNotFoundError):
        DirCountFileAnnot(dirpath=Path(dir1))
