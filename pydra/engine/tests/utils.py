# Tasks for testing
import time
import sys, shutil
import typing as ty
from pathlib import Path
import functools
import operator
import subprocess as sp
import pytest
from fileformats.generic import File

from ..core import Workflow
from ..submitter import Submitter
from ... import mark


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None or sp.call(["docker", "info"]),
    reason="no docker within the container",
)
need_singularity = pytest.mark.skipif(
    shutil.which("singularity") is None, reason="no singularity available"
)
no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker command not adjusted for windows docker",
)
need_slurm = pytest.mark.skipif(
    not (bool(shutil.which("sbatch")) and bool(shutil.which("sacct"))),
    reason="slurm not available",
)
need_sge = pytest.mark.skipif(
    not (bool(shutil.which("qsub")) and bool(shutil.which("qacct"))),
    reason="sge not available",
)


def result_no_submitter(shell_task, plugin=None):
    """helper function to return result when running without submitter"""
    return shell_task()


def result_submitter(shell_task, plugin):
    """helper function to return result when running with submitter
    with specific plugin
    """
    with Submitter(plugin=plugin) as sub:
        shell_task(submitter=sub)
    return shell_task.result()


dot_check = sp.run(["which", "dot"], stdout=sp.PIPE, stderr=sp.PIPE)
if dot_check.stdout:
    DOT_FLAG = True
else:
    DOT_FLAG = False


@mark.task
def op_4var(a, b, c, d) -> str:
    return f"{a} {b} {c} {d}"


@mark.task
def fun_addtwo(a: int) -> int:
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@mark.task
def fun_addtwo_notype(a):
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@mark.task
def fun_addtwo_with_threadcount(a: int, sgeThreads: int = 1) -> int:
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@mark.task
def fun_addvar(
    a: ty.Union[int, float], b: ty.Union[int, float]
) -> ty.Union[int, float]:
    return a + b


@mark.task
def fun_addvar_notype(a, b):
    return a + b


@mark.task
@mark.annotate({"return": {"sum": float, "sub": float}})
def fun_addsubvar(a: float, b: float):
    return a + b, a - b


@mark.task
def fun_addvar_none(a: int, b: ty.Optional[int]) -> int:
    if b is None:
        return a
    else:
        return a + b


@mark.task
def fun_addvar_default(a: int, b: int = 1) -> int:
    return a + b


@mark.task
def fun_addvar_default_notype(a, b=1):
    return a + b


@mark.task
def fun_addvar3(a: int, b: int, c: int) -> int:
    return a + b + c


@mark.task
def fun_addvar4(a: int, b: int, c: int, d: int) -> int:
    return a + b + c + d


@mark.task
def moment(lst: ty.List[float], n: float) -> float:
    return sum([i**n for i in lst]) / len(lst)


@mark.task
def fun_div(a: ty.Union[int, float], b: ty.Union[int, float]) -> float:
    return a / b


@mark.task
def multiply(x: int, y: int) -> int:
    return x * y


@mark.task
def multiply_list(x: list, y: int) -> list:
    return x * y


@mark.task
def multiply_mixed(x: list, y: int) -> list:
    return x * y


@mark.task
def add2(x: int) -> int:
    if x == 1 or x == 12:
        time.sleep(1)
    return x + 2


@mark.task
def raise_xeq1(x: int) -> int:
    if x == 1:
        raise Exception("x is 1, so i'm raising an exception!")
    return x


@mark.task
@mark.annotate({"return": {"out_add": float, "out_sub": float}})
def add2_sub2_res(res):
    """function that takes entire output as an input"""
    return res["out"] + 2, res["out"] - 2


@mark.task
@mark.annotate({"return": {"out_add": ty.List[float], "out_sub": ty.List[float]}})
def add2_sub2_res_list(res):
    """function that takes entire output as an input"""
    return [r["out"] + 2 for r in res], [r["out"] - 2 for r in res]


@mark.task
def power(a: int, b: int) -> int:
    return a**b


@mark.task
def identity(x):
    return x


@mark.task
def identity_2flds(
    x1, x2
) -> ty.NamedTuple("Output", [("out1", ty.Any), ("out2", ty.Any)]):
    return x1, x2


@mark.task
def ten(x) -> int:
    return 10


@mark.task
def add2_wait(x: int) -> int:
    time.sleep(2)
    return x + 2


@mark.task
def list_output(x: int) -> ty.List[int]:
    return [x, 2 * x, 3 * x]


@mark.task
def list_sum(x: ty.Sequence[ty.Union[int, float]]) -> ty.Union[int, float]:
    return sum(x)


@mark.task
def fun_dict(d: dict) -> str:
    kv_list = [f"{k}:{v}" for (k, v) in d.items()]
    return "_".join(kv_list)


@mark.task
def fun_write_file(filename: Path, text="hello") -> File:
    with open(filename, "w") as f:
        f.write(text)
    return File(filename)


@mark.task
def fun_write_file_list(
    filename_list: ty.List[ty.Union[str, File, Path]], text="hi"
) -> ty.List[File]:
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
    filename_list = [Path(filename).absolute() for filename in filename_list]
    return filename_list


@mark.task
def fun_write_file_list2dict(
    filename_list: ty.List[ty.Union[str, File, Path]], text="hi"
) -> ty.Dict[str, ty.Union[File, int]]:
    filename_dict = {}
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
        filename_dict[f"file_{ii}"] = Path(filename).absolute()
    # adding an additional field with int
    filename_dict["random_int"] = 20
    return filename_dict


@mark.task
def fun_file(filename: File):
    with open(filename) as f:
        txt = f.read()
    return txt


@mark.task
def fun_file_list(filename_list: ty.List[File]):
    txt_list = []
    for filename in filename_list:
        with open(filename) as f:
            txt_list.append(f.read())
    return " ".join(txt_list)


def gen_basic_wf(name="basic-wf"):
    """
    Generates `Workflow` of two tasks

    Task Input
    ----------
    x : int (5)

    Task Output
    -----------
    out : int (9)
    """
    wf = Workflow(name=name, input_spec=["x"])
    wf.inputs.x = 5
    wf.add(fun_addtwo(name="task1", a=wf.lzin.x, b=0))
    wf.add(fun_addvar(name="task2", a=wf.task1.lzout.out, b=2))
    wf.set_output([("out", wf.task2.lzout.out)])
    return wf


def gen_basic_wf_with_threadcount(name="basic-wf-with-threadcount"):
    """
    Generates `Workflow` of two tasks

    Task Input
    ----------
    x : int (5)

    Task Output
    -----------
    out : int (9)
    """
    wf = Workflow(name=name, input_spec=["x"])
    wf.inputs.x = 5
    wf.add(fun_addtwo_with_threadcount(name="task1", a=wf.lzin.x, sgeThreads=4))
    wf.add(fun_addvar(name="task2", a=wf.task1.lzout.out, b=2))
    wf.set_output([("out", wf.task2.lzout.out)])
    return wf


def gen_basic_wf_with_threadcount_concurrent(name="basic-wf-with-threadcount"):
    """
    Generates `Workflow` of two tasks

    Task Input
    ----------
    x : int (5)

    Task Output
    -----------
    out : int (9)
    """
    wf = Workflow(name=name, input_spec=["x"])
    wf.inputs.x = 5
    wf.add(fun_addtwo_with_threadcount(name="task1_1", a=wf.lzin.x, sgeThreads=4))
    wf.add(fun_addtwo_with_threadcount(name="task1_2", a=wf.lzin.x, sgeThreads=2))
    wf.add(fun_addvar(name="task2", a=wf.task1_1.lzout.out, b=2))
    wf.set_output([("out1", wf.task2.lzout.out), ("out2", wf.task1_2.lzout.out)])
    return wf


@mark.task
@mark.annotate({"return": {"sum": int, "products": ty.List[int]}})
def list_mult_sum(scalar: int, in_list: ty.List[int]) -> ty.Tuple[int, ty.List[int]]:
    products = [scalar * x for x in in_list]
    return functools.reduce(operator.add, products, 0), products


@mark.task
@mark.annotate({"return": {"x": str, "y": int, "z": float}})
def foo(a: str, b: int, c: float) -> ty.Tuple[str, int, float]:
    return a, b, c
