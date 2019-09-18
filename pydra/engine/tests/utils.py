# Tasks for testing
import time

from ..core import Workflow
from ... import mark


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
def power(a, b):
    return a ** b


@mark.task
def identity(x):
    return x


@mark.task
def add2_wait(x):
    time.sleep(3)
    return x + 2


@mark.task
def list_output(x):
    return [x, 2 * x, 3 * x]


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
