import os
from pathlib import Path
import numpy as np
import pytest
from pydra.engine.submitter import Submitter
from pydra.compose import python, workflow
from fileformats.generic import File, Directory
import time
from pydra.engine.tests.utils import (
    FileOrIntIdentity,
    FileAndIntIdentity,
    ListOfListOfFileOrIntIdentity,
    ListOfDictOfFileOrIntIdentity,
)


@pytest.mark.flaky(reruns=5)
def test_input_file_hash_1(tmp_path):
    """input definition with File types, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    # checking specific hash value
    hash1 = FileOrIntIdentity(in_file=file)._hash
    # assert hash1 == "eba2fafb8df4bae94a7aa42bb159b778"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = FileOrIntIdentity(in_file=file_diffname)._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = FileOrIntIdentity(in_file=file_diffcontent)._hash
    assert hash1 != hash3


@pytest.mark.flaky(reruns=5)
def test_input_file_hash_2(tmp_path):
    """input definition with ty.Union[File, ...] type, checking when the checksum changes"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    # checking specific hash value
    hash1 = FileOrIntIdentity(in_file=file)._hash
    # assert hash1 == "eba2fafb8df4bae94a7aa42bb159b778"

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = FileOrIntIdentity(in_file=file_diffname)._hash
    assert hash1 == hash2

    # checking if string is also accepted
    hash3 = FileOrIntIdentity(in_file=str(file))._hash
    assert hash3 == hash1

    # checking if different content (the same name) affects the hash
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash4 = FileOrIntIdentity(in_file=file_diffcontent)._hash
    assert hash1 != hash4


@pytest.mark.flaky(reruns=5)
def test_input_file_hash_3(tmp_path):
    """input definition with File types, checking when the hash and file_hash change"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    a = FileAndIntIdentity(in_file=file, in_int=3)
    # original hash and files_hash (dictionary contains info about files)
    hash1 = a._hash
    # files_hash1 = deepcopy(my_inp.files_hash)
    # file name should be in files_hash1[in_file]
    # filename = str(Path(file))
    # assert filename in files_hash1["in_file"]

    # changing int input
    a.in_int = 5
    hash2 = a._hash
    # files_hash2 = deepcopy(my_inp.files_hash)
    # hash should be different
    assert hash1 != hash2
    # files_hash should be the same, and the tuple for filename shouldn't be recomputed
    # assert files_hash1 == files_hash2
    # assert id(files_hash1["in_file"][filename]) == id(files_hash2["in_file"][filename])

    # recreating the file
    time.sleep(2)  # ensure mtime is different
    with open(file, "w") as f:
        f.write("hello")

    hash3 = a._hash
    # files_hash3 = deepcopy(my_inp.files_hash)
    # hash should be the same,
    # but the entry for in_file in files_hash should be different (modification time)
    assert hash3 == hash2
    # assert files_hash3["in_file"][filename] != files_hash2["in_file"][filename]
    # different timestamp
    # assert files_hash3["in_file"][filename][0] != files_hash2["in_file"][filename][0]
    # the same content hash
    # assert files_hash3["in_file"][filename][1] == files_hash2["in_file"][filename][1]

    # setting the in_file again
    a.in_file = file
    # filename should be removed from files_hash
    # assert my_inp.files_hash["in_file"] == {}
    # will be saved again when hash is calculated
    assert a._hash == hash3
    # assert filename in my_inp.files_hash["in_file"]


@pytest.mark.flaky(reruns=5)
def test_input_file_hash_4(tmp_path):
    """input definition with nested list, that contain ints and Files,
    checking changes in checksums
    """
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    # checking specific hash value
    hash1 = ListOfListOfFileOrIntIdentity(in_file=[[file, 3]])._hash
    # assert hash1 == "2c35c94089b00a7a399d3d4faf208fee"

    # the same file, but int field changes
    hash1a = ListOfListOfFileOrIntIdentity(in_file=[[file, 5]])._hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = ListOfListOfFileOrIntIdentity(in_file=[[file_diffname, 3]])._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # need the mtime to be different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = ListOfListOfFileOrIntIdentity(in_file=[[file_diffcontent, 3]])._hash
    assert hash1 != hash3


@pytest.mark.flaky(reruns=5)
def test_input_file_hash_5(tmp_path):
    """input definition with File in nested containers, checking changes in checksums"""
    file = tmp_path / "in_file_1.txt"
    with open(file, "w") as f:
        f.write("hello")

    # checking specific hash value
    hash1 = ListOfDictOfFileOrIntIdentity(in_file=[{"file": file, "int": 3}])._hash
    # assert hash1 == "7692ffe0b3323c13ecbd642b494f1f53"

    # the same file, but int field changes
    hash1a = ListOfDictOfFileOrIntIdentity(in_file=[{"file": file, "int": 5}])._hash
    assert hash1 != hash1a

    # checking if different name doesn't affect the hash
    file_diffname = tmp_path / "in_file_2.txt"
    with open(file_diffname, "w") as f:
        f.write("hello")
    hash2 = ListOfDictOfFileOrIntIdentity(
        in_file=[{"file": file_diffname, "int": 3}]
    )._hash
    assert hash1 == hash2

    # checking if different content (the same name) affects the hash
    time.sleep(2)  # ensure mtime is different
    file_diffcontent = tmp_path / "in_file_1.txt"
    with open(file_diffcontent, "w") as f:
        f.write("hi")
    hash3 = ListOfDictOfFileOrIntIdentity(
        in_file=[{"file": file_diffcontent, "int": 3}]
    )._hash
    assert hash1 != hash3


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
