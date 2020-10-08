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
def dir_count_file(dirpath):
    return len(os.listdir(dirpath))


@mark.task
def dir_count_file_annot(dirpath: Directory):
    return len(os.listdir(dirpath))


@mark.task
def file_append_text(file):
    with open(file, "a") as f:
        f.write("!")
    return f


@mark.task
def file_append_text_annot(file: File):
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
    nn = file_append_text_annot(name="add_text", file=file)

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
    nn = file_append_text_annot(name="add_text", file=file)

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


def test_broken_file(tmpdir):
    """ Test how broken paths are handled during file hashing"""
    # file path doesn't exist
    file = os.path.join(tmpdir, "A.txt")

    nn2 = file_append_text_annot(name="add_text", file=file)
    with pytest.raises(AttributeError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn2)


def test_broken_file_link(tmpdir):
    """
    Test how broken symlinks are handled during hashing
    """
    os.chdir(tmpdir)
    file = tmpdir.join("file1")
    file.open("w+").close()

    file_link = tmpdir.join("file_link")
    os.symlink(file, file_link)
    os.remove(file)

    nn = file_append_text(name="add_text", file=file_link)
    # raises error inside task
    # unless variable is defined as a File pydra will treat it as a string
    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn)

    # raises error before task is run
    # TODO typecheck at add???
    nn2 = file_append_text(name="add_text", file=file)
    with pytest.raises(AttributeError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn2)


@pytest.mark.skip(reason="pickling error with type hinting - how to improve?")
def test_broken_dir():
    """ Test how broken directories are handled during hashing"""

    # dirpath doesn't exist
    nn = dir_count_file(name="listdir", dirpath="/broken_dir_path/")
    # raises error inside task
    # unless variable is defined as a File pydra will treat it as a string
    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn)

    # raises error before task is run
    nn2 = dir_count_file_annot(name="listdir", dirpath="/broken_dir_path/")
    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn2)


def test_broken_dir_link(tmpdir):
    """
    Test how broken symlinks are hashed in hash_dir
    """
    # broken symlink to dir path
    dir1 = tmpdir.join("dir1")
    os.mkdir(dir1)
    dir1_link = tmpdir.join("dir1_link")
    os.symlink(dir1, dir1_link)
    os.rmdir(dir1)

    nn = dir_count_file(name="listdir", dirpath=dir1)
    # raises error while running task
    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn)

    nn2 = dir_count_file_annot(name="listdir", dirpath=dir1)
    # raises error before task is run
    with pytest.raises(FileNotFoundError) as e:
        with Submitter(plugin="cf") as sub:
            sub(nn2)


def test_broken_dir_link2(tmpdir):
    # valid dirs with broken symlink(s) are hashed
    dir2 = tmpdir.join("dir2")
    os.mkdir(dir2)
    file1 = dir2.join("file1")
    file2 = dir2.join("file2")
    file1.open("w+").close()
    file2.open("w+").close()

    file1_link = dir2.join("file1_link")
    os.symlink(file1, file1_link)
    os.remove(file1)  # file1_link is broken

    nn = dir_count_file(name="listdir", dirpath=dir2)
    # does not raises error because pydra treats dirpath as a string
    with Submitter(plugin="cf") as sub:
        sub(nn)

    nn2 = dir_count_file_annot(name="listdir", dirpath=str(dir2))
    with Submitter(plugin="cf") as sub:
        sub(nn2)
