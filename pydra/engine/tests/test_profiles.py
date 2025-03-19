from pydra.engine.helpers import load_task
from pydra.design import python, workflow
from pydra.engine.core import Task
from pydra.engine.submitter import Submitter

import numpy as np
from pympler import asizeof
from pytest import approx


def generate_list(n):
    return np.arange(n).tolist()


@python.define
def ShowVar(a):
    return a


def create_wf(size):
    @workflow.define
    def Workflow(x):
        show = workflow.add(ShowVar(a=x))
        return show.out

    return Workflow().split(x=generate_list(size))


def test_wf_memory():
    """creating two workflow with relatively big splitter: 1000, 2000 and 4000 elements
    testings if the size of workflow grows linearly
    """

    wf_10000 = create_wf(size=10000)
    wf_10000_mem = asizeof.asizeof(wf_10000)

    wf_20000 = create_wf(size=20000)
    wf_20000_mem = asizeof.asizeof(wf_20000)

    wf_40000 = create_wf(size=40000)
    wf_40000_mem = asizeof.asizeof(wf_40000)
    # checking if it's linear with the size of the splitter
    # check print(asizeof.asized(wf_4000, detail=2).format()) in case of problems
    assert wf_40000_mem / wf_20000_mem == approx(2, 0.05)
    assert wf_20000_mem / wf_10000_mem == approx(2, 0.05)
