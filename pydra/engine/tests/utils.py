# Tasks for testing
import time
import typing as tp
from pathlib import Path

from ..core import Workflow
from ... import mark
from ..specs import File


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
    time.sleep(3)
    return x + 2


@mark.task
def list_output(x):
    return [x, 2 * x, 3 * x]


@mark.task
def fun_dict(d):
    kv_list = [f"{k}:{v}" for (k, v) in d.items()]
    return "_".join(kv_list)


@mark.task
def fun_write_file(filename: tp.Union[str, File, Path], text="hello"):
    with open(filename, "w") as f:
        f.write(text)
    return Path(filename).absolute()


@mark.task
def fun_write_file_list(filename_list: tp.List[tp.Union[str, File, Path]], text="hi"):
    for ii, filename in enumerate(filename_list):
        with open(filename, "w") as f:
            f.write(f"from file {ii}: {text}")
    filename_list = [Path(filename).absolute() for filename in filename_list]
    return filename_list


@mark.task
def fun_write_file_list2dict(
    filename_list: tp.List[tp.Union[str, File, Path]], text="hi"
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
def fun_file_list(filename_list: File):
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
