import os
import shutil
from pathlib import Path
import random
import platform
import typing as ty
import pytest
import cloudpickle as cp
from pydra.engine.submitter import Submitter
from pydra.engine.result import Result
from pydra.engine.job import Job
from pydra.compose import workflow
from fileformats.generic import Directory, File
from pydra.engine.tests.utils import Multiply, RaiseXeq1
from pydra.utils.general import position_sort
from pydra.compose.shell.templating import parse_format_string
from pydra.engine.job import save, load_and_run
from pydra.workers.cf import get_available_cpus
from pydra.utils.hash import hash_function


def test_save(tmpdir):
    outdir = Path(tmpdir)
    with pytest.raises(ValueError):
        save(tmpdir)
    foo = Job(name="mult", task=Multiply(x=1, y=2), submitter=Submitter())
    # save job
    save(outdir, job=foo)
    del foo
    # load saved job
    job_pkl = outdir / "_job.pklz"
    foo: Job = cp.loads(job_pkl.read_bytes())
    assert foo.name == "mult"
    assert foo.inputs["x"] == 1 and foo.inputs["y"] == 2
    # execute job and save result
    res: Result = foo.run()
    assert res.outputs.out == 2
    save(outdir, result=res)
    del res
    # load saved result
    res_pkl = outdir / "_result.pklz"
    res: Result = cp.loads(res_pkl.read_bytes())
    assert res.outputs.out == 2


def test_hash_file(tmpdir):
    outdir = Path(tmpdir)
    with open(outdir / "test.file", "w") as fp:
        fp.write("test")
    assert (
        hash_function(File(outdir / "test.file")) == "f32ab20c4a86616e32bf2504e1ac5a22"
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
    """testing load_and_run for pickled job"""
    job_pkl = Path(tmpdir.join("task_main.pkl"))
    # Note that tasks now don't have state arrays and indices, just a single resolved
    # set of parameters that are ready to run
    job = Job(name="mult", task=Multiply(x=2, y=10), submitter=Submitter())
    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)
    resultfile = load_and_run(job_pkl=job_pkl)
    # checking the result files
    result = cp.loads(resultfile.read_bytes())
    assert result.outputs.out == 20


def test_load_and_run_exception_run(tmpdir):
    """testing raising exception and saving info in crashfile when when load_and_run"""
    job_pkl = Path(tmpdir.join("task_main.pkl"))
    cache_root = Path(tmpdir.join("cache"))
    cache_root.mkdir()

    job = Job(
        task=RaiseXeq1(x=1),
        name="raise",
        submitter=Submitter(worker="cf", cache_dir=cache_root),
    )

    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)

    with pytest.raises(Exception) as excinfo:
        load_and_run(job_pkl=job_pkl)
    exc_msg = excinfo.value.args[0]
    assert "i'm raising an exception!" in exc_msg
    # checking if the crashfile has been created
    assert "crash" in excinfo.value.__notes__[0]
    errorfile = Path(excinfo.value.__notes__[0].split("here: ")[1])
    assert errorfile.exists()

    resultfile = errorfile.parent / "_result.pklz"
    assert resultfile.exists()
    # checking the content
    result_exception = cp.loads(resultfile.read_bytes())
    assert result_exception.errored is True

    job = Job(task=RaiseXeq1(x=2), name="wont_raise", submitter=Submitter())

    with job_pkl.open("wb") as fp:
        cp.dump(job, fp)

    # the second job should be fine
    resultfile = load_and_run(job_pkl=job_pkl)
    result_1 = cp.loads(resultfile.read_bytes())
    assert result_1.outputs.out == 2


def test_load_and_run_wf(tmpdir, worker):
    """testing load_and_run for pickled job"""
    wf_pkl = Path(tmpdir.join("wf_main.pkl"))

    @workflow.define
    def Workflow(x, y=10):
        multiply = workflow.add(Multiply(x=x, y=y))
        return multiply.out

    job = Job(
        name="mult",
        task=Workflow(x=2),
        submitter=Submitter(cache_dir=tmpdir, worker=worker),
    )

    with wf_pkl.open("wb") as fp:
        cp.dump(job, fp)

    resultfile = load_and_run(job_pkl=wf_pkl)
    # checking the result files
    result = cp.loads(resultfile.read_bytes())
    assert result.outputs.out == 20


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


def test_parse_format_string1():
    assert parse_format_string("{a}") == {"a"}


def test_parse_format_string2():
    assert parse_format_string("{abc}") == {"abc"}


def test_parse_format_string3():
    assert parse_format_string("{a:{b}}") == {"a", "b"}


def test_parse_format_string4():
    assert parse_format_string("{a:{b[2]}}") == {"a", "b"}


def test_parse_format_string5():
    assert parse_format_string("{a.xyz[somekey].abc:{b[a][b].d[0]}}") == {"a", "b"}


def test_parse_format_string6():
    assert parse_format_string("{a:05{b[a 2][b].e}}") == {"a", "b"}


def test_parse_format_string7():
    assert parse_format_string(
        "{a1_field} {b2_field:02f} -test {c3_field[c]} -me {d4_field[0]}"
    ) == {"a1_field", "b2_field", "c3_field", "d4_field"}
