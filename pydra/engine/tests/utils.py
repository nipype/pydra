# Tasks for testing
import time
import sys, shutil
import typing as ty
from pathlib import Path
import subprocess as sp
import pytest

from ..core import Workflow
from ..submitter import Submitter
from ... import mark
from ..specs import File
from ... import set_input_validator


need_docker = pytest.mark.skipif(
    shutil.which("docker") is None or sp.call(["docker", "info"]),
    reason="no docker within the container",
)
no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker command not adjusted for windows docker",
)


def result_no_submitter(shell_task, plugin=None):
    """ helper function to return result when running without submitter """
    return shell_task()


def result_submitter(shell_task, plugin):
    """ helper function to return result when running with submitter
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
def fun_addtwo(a):
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@mark.task
def fun_addvar(a, b):
    return a + b


@mark.task
@mark.annotate({"return": {"sum": float, "sub": float}})
def fun_addsubvar(a, b):
    return a + b, a - b


@mark.task
def fun_addvar_none(a, b):
    if b is None:
        return a
    else:
        return a + b


@mark.task
def fun_addvar_default(a, b=1):
    return a + b


@mark.task
def fun_addvar3(a, b, c):
    return a + b + c


@mark.task
def fun_addvar4(a, b, c, d):
    return a + b + c + d


@mark.task
def moment(lst, n):
    return sum([i ** n for i in lst]) / len(lst)


@mark.task
def fun_div(a, b):
    return a / b


@mark.task
def multiply(x, y):
    return x * y


@mark.task
def add2(x):
    if x == 1 or x == 12:
        time.sleep(1)
    return x + 2


@mark.task
def raise_xeq1(x):
    if x == 1:
        raise Exception("x is 1, so i'm raising an exception!")
    return x


@mark.task
@mark.annotate({"return": {"out_add": float, "out_sub": float}})
def add2_sub2_res(res):
    """function that takes entire output as an input"""
    if isinstance(res, list):
        return [r["out"] + 2 for r in res], [r["out"] - 2 for r in res]
    return res["out"] + 2, res["out"] - 2


@mark.task
def power(a, b):
    return a ** b


@mark.task
def identity(x):
    return x


@mark.task
def ten(x):
    return 10


@mark.task
def add2_wait(x):
    time.sleep(2)
    return x + 2


@mark.task
def list_output(x):
    return [x, 2 * x, 3 * x]


@mark.task
def list_sum(x):
    return sum(x)


@mark.task
def fun_dict(d):
    kv_list = [f"{k}:{v}" for (k, v) in d.items()]
    return "_".join(kv_list)


@mark.task
def fun_write_file(filename: ty.Union[str, File, Path], text="hello") -> File:
    with open(filename, "w") as f:
        f.write(text)
    return Path(filename).absolute()


@mark.task
def fun_write_file_list(filename_list: ty.List[ty.Union[str, File, Path]], text="hi"):
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
    filename_list = [Path(filename).absolute() for filename in filename_list]
    return filename_list


@mark.task
def fun_write_file_list2dict(
    filename_list: ty.List[ty.Union[str, File, Path]], text="hi"
):
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


@pytest.fixture(scope="function")
def use_validator(request):
    set_input_validator(flag=True)

    def fin():
        set_input_validator(flag=False)

    request.addfinalizer(fin)
