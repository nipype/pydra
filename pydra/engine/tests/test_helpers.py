import os
import shutil
from pathlib import Path
import random
import platform
import pytest
import cloudpickle as cp
from unittest.mock import Mock
from fileformats.generic import Directory, File
from fileformats.core import FileSet
from .utils import multiply, raise_xeq1
from ..helpers import (
    get_available_cpus,
    save,
    load_and_run,
    position_sort,
    parse_copyfile,
)
from ...utils.hash import hash_function
from .. import helpers_file
from ..core import Workflow


def test_save(tmpdir):
    outdir = Path(tmpdir)
    with pytest.raises(ValueError):
        save(tmpdir)
    foo = multiply(name="mult", x=1, y=2)
    # save task
    save(outdir, task=foo)
    del foo
    # load saved task
    task_pkl = outdir / "_task.pklz"
    foo = cp.loads(task_pkl.read_bytes())
    assert foo.name == "mult"
    assert foo.inputs.x == 1 and foo.inputs.y == 2
    # execute task and save result
    res = foo()
    assert res.output.out == 2
    save(outdir, result=res)
    del res
    # load saved result
    res_pkl = outdir / "_result.pklz"
    res = cp.loads(res_pkl.read_bytes())
    assert res.output.out == 2


def test_hash_file(tmpdir):
    outdir = Path(tmpdir)
    with open(outdir / "test.file", "w") as fp:
        fp.write("test")
    assert (
        hash_function(File(outdir / "test.file")) == "37fcc546dce7e59585f3217bb4c30299"
    )


def test_hashfun_float():
    import math

    pi_50 = 3.14159265358979323846264338327950288419716939937510
    pi_15 = 3.141592653589793
    pi_10 = 3.1415926536
    # comparing for x that have the same x.as_integer_ratio()
    assert (
        math.pi.as_integer_ratio()
        == pi_50.as_integer_ratio()
        == pi_15.as_integer_ratio()
    )
    assert hash_function(math.pi) == hash_function(pi_15) == hash_function(pi_50)
    # comparing for x that have different x.as_integer_ratio()
    assert math.pi.as_integer_ratio() != pi_10.as_integer_ratio()
    assert hash_function(math.pi) != hash_function(pi_10)


def test_hash_function_dict():
    dict1 = {"a": 10, "b": 5}
    dict2 = {"b": 5, "a": 10}
    assert hash_function(dict1) == hash_function(dict2)


def test_hash_function_list_tpl():
    lst = [2, 5.6, "ala"]
    tpl = (2, 5.6, "ala")
    assert hash_function(lst) != hash_function(tpl)


def test_hash_function_list_dict():
    lst = [2, {"a": "ala", "b": 1}]
    hash_function(lst)


def test_hash_function_files(tmp_path: Path):
    file_1 = tmp_path / "file_1.txt"
    file_2 = tmp_path / "file_2.txt"
    file_1.write_text("hello")
    file_2.write_text("hello")

    assert hash_function(File(file_1)) == hash_function(File(file_2))


def test_hash_function_dir_and_files_list(tmp_path: Path):
    dir1 = tmp_path / "foo"
    dir2 = tmp_path / "bar"
    for d in (dir1, dir2):
        d.mkdir()
        for i in range(3):
            f = d / f"{i}.txt"
            f.write_text(str(i))

    assert hash_function(Directory(dir1)) == hash_function(Directory(dir2))
    file_list1: ty.List[File] = [File(f) for f in dir1.iterdir()]
    file_list2: ty.List[File] = [File(f) for f in dir2.iterdir()]
    assert hash_function(file_list1) == hash_function(file_list2)


def test_hash_function_files_mismatch(tmp_path: Path):
    file_1 = tmp_path / "file_1.txt"
    file_2 = tmp_path / "file_2.txt"
    file_1.write_text("hello")
    file_2.write_text("hi")

    assert hash_function(File(file_1)) != hash_function(File(file_2))


def test_hash_function_nested(tmp_path: Path):
    dpath = tmp_path / "dir"
    dpath.mkdir()
    hidden = dpath / ".hidden"
    nested = dpath / "nested"
    hidden.mkdir()
    nested.mkdir()
    file_1 = dpath / "file_1.txt"
    file_2 = hidden / "file_2.txt"
    file_3 = nested / ".file_3.txt"
    file_4 = nested / "file_4.txt"

    for fx in [file_1, file_2, file_3, file_4]:
        fx.write_text(str(random.randint(0, 1000)))

    nested_dir = Directory(dpath)

    orig_hash = nested_dir.hash()

    nohidden_hash = nested_dir.hash(ignore_hidden_dirs=True, ignore_hidden_files=True)
    nohiddendirs_hash = nested_dir.hash(ignore_hidden_dirs=True)
    nohiddenfiles_hash = nested_dir.hash(ignore_hidden_files=True)

    assert orig_hash != nohidden_hash
    assert orig_hash != nohiddendirs_hash
    assert orig_hash != nohiddenfiles_hash

    os.remove(file_3)
    assert nested_dir.hash() == nohiddenfiles_hash
    shutil.rmtree(hidden)
    assert nested_dir.hash() == nohidden_hash


def test_get_available_cpus():
    assert get_available_cpus() > 0
    try:
        import psutil

        has_psutil = True
    except ImportError:
        has_psutil = False

    if hasattr(os, "sched_getaffinity"):
        assert get_available_cpus() == len(os.sched_getaffinity(0))

    if has_psutil and platform.system().lower() != "darwin":
        assert get_available_cpus() == len(psutil.Process().cpu_affinity())

    if platform.system().lower() == "darwin":
        assert get_available_cpus() == os.cpu_count()


def test_load_and_run(tmpdir):
    """testing load_and_run for pickled task"""
    task_pkl = Path(tmpdir.join("task_main.pkl"))

    task = multiply(name="mult", y=10).split(x=[1, 2])
    task.state.prepare_states(inputs=task.inputs)
    task.state.prepare_inputs()
    with task_pkl.open("wb") as fp:
        cp.dump(task, fp)

    resultfile_0 = load_and_run(task_pkl=task_pkl, ind=0)
    resultfile_1 = load_and_run(task_pkl=task_pkl, ind=1)
    # checking the result files
    result_0 = cp.loads(resultfile_0.read_bytes())
    result_1 = cp.loads(resultfile_1.read_bytes())
    assert result_0.output.out == 10
    assert result_1.output.out == 20


def test_load_and_run_exception_load(tmpdir):
    """testing raising exception and saving info in crashfile when when load_and_run"""
    task_pkl = Path(tmpdir.join("task_main.pkl"))
    raise_xeq1(name="raise").split("x", x=[1, 2])
    with pytest.raises(FileNotFoundError):
        load_and_run(task_pkl=task_pkl, ind=0)


def test_load_and_run_exception_run(tmpdir):
    """testing raising exception and saving info in crashfile when when load_and_run"""
    task_pkl = Path(tmpdir.join("task_main.pkl"))

    task = raise_xeq1(name="raise").split("x", x=[1, 2])
    task.state.prepare_states(inputs=task.inputs)
    task.state.prepare_inputs()

    with task_pkl.open("wb") as fp:
        cp.dump(task, fp)

    with pytest.raises(Exception) as excinfo:
        load_and_run(task_pkl=task_pkl, ind=0)
    assert "i'm raising an exception!" in str(excinfo.value)
    # checking if the crashfile has been created
    assert "crash" in str(excinfo.value)
    errorfile = Path(str(excinfo.value).split("here: ")[1][:-2])
    assert errorfile.exists()

    resultfile = errorfile.parent / "_result.pklz"
    assert resultfile.exists()
    # checking the content
    result_exception = cp.loads(resultfile.read_bytes())
    assert result_exception.errored is True

    # the second task should be fine
    resultfile = load_and_run(task_pkl=task_pkl, ind=1)
    result_1 = cp.loads(resultfile.read_bytes())
    assert result_1.output.out == 2


def test_load_and_run_wf(tmpdir):
    """testing load_and_run for pickled task"""
    wf_pkl = Path(tmpdir.join("wf_main.pkl"))

    wf = Workflow(name="wf", input_spec=["x", "y"], y=10)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.split("x", x=[1, 2])

    wf.set_output([("out", wf.mult.lzout.out)])

    # task = multiply(name="mult", x=[1, 2], y=10).split("x")
    wf.state.prepare_states(inputs=wf.inputs)
    wf.state.prepare_inputs()
    wf.plugin = "cf"

    with wf_pkl.open("wb") as fp:
        cp.dump(wf, fp)

    resultfile_0 = load_and_run(ind=0, task_pkl=wf_pkl)
    resultfile_1 = load_and_run(ind=1, task_pkl=wf_pkl)
    # checking the result files
    result_0 = cp.loads(resultfile_0.read_bytes())
    result_1 = cp.loads(resultfile_1.read_bytes())
    assert result_0.output.out == 10
    assert result_1.output.out == 20


@pytest.mark.parametrize(
    "pos_args",
    [
        [(2, "b"), (1, "a"), (3, "c")],
        [(-2, "b"), (1, "a"), (-1, "c")],
        [(None, "b"), (1, "a"), (-1, "c")],
        [(-3, "b"), (None, "a"), (-1, "c")],
        [(None, "b"), (1, "a"), (None, "c")],
    ],
)
def test_position_sort(pos_args):
    final_args = position_sort(pos_args)
    assert final_args == ["a", "b", "c"]


def test_parse_copyfile():
    Mode = FileSet.CopyMode
    Collation = FileSet.CopyCollation

    def mock_field(copyfile):
        mock = Mock(["metadata"])
        mock.metadata = {"copyfile": copyfile}
        return mock

    assert parse_copyfile(mock_field((Mode.any, Collation.any))) == (
        Mode.any,
        Collation.any,
    )
    assert parse_copyfile(mock_field("copy"), default_collation=Collation.siblings) == (
        Mode.copy,
        Collation.siblings,
    )
    assert parse_copyfile(mock_field("link,adjacent")) == (
        Mode.link,
        Collation.adjacent,
    )
    assert parse_copyfile(mock_field(True)) == (
        Mode.copy,
        Collation.any,
    )
    assert parse_copyfile(mock_field(False)) == (
        Mode.link,
        Collation.any,
    )
    assert parse_copyfile(mock_field(None)) == (
        Mode.any,
        Collation.any,
    )
    with pytest.raises(TypeError, match="Unrecognised type for mode copyfile"):
        parse_copyfile(mock_field((1, 2)))
    with pytest.raises(TypeError, match="Unrecognised type for collation copyfile"):
        parse_copyfile(mock_field((Mode.copy, 2)))
