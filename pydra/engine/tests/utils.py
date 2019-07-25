# Tasks for testing
import time

from ..core import Workflow
from ..task import to_task


@to_task
def fun_addtwo(a):
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@to_task
def fun_addvar(a, b):
    return a + b


@to_task
def fun_addvar4(a, b, c, d):
    return a + b + c + d


@to_task
def moment(lst, n):
    return sum([i ** n for i in lst]) / len(lst)


@to_task
def fun_div(a, b):
    return a / b


@to_task
def multiply(x, y):
    return x * y


@to_task
def add2(x):
    if x == 1 or x == 12:
        time.sleep(1)
    return x + 2


@to_task
def add2_wait(x):
    time.sleep(3)
    return x + 2


def gen_basic_wf():
    """Generates `Workflow` of two tasks"""
    wf = Workflow(name="basic-wf", input_spec=["x"])
    wf.inputs.x = 5
    wf.add(fun_addtwo(name="task1", a=wf.lzin.x))
    wf.add(fun_addvar(name="task2", a=wf.task1.lzout.out, b=2))
    wf.set_output([("out", wf.task2.lzout.out)])
    return wf
