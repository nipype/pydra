from ..core import Workflow
from ..helpers import load_task
from ... import mark

import numpy as np
from pympler import asizeof
from pytest import approx


def generate_list(l):
    return np.arange(l).tolist()


@mark.task
def show_var(a):
    return a


def create_wf(size):
    wf = Workflow(name="wf", input_spec=["x"])
    wf.split("x", x=generate_list(size))
    wf.add(show_var(name="show", a=wf.lzin.x))
    wf.set_output([("out", wf.show.lzout.out)])
    wf.state.prepare_states(wf.inputs)
    wf.state.prepare_inputs()
    return wf


def test_wf_memory():
    """creating two workflow with relatively big splitter: 1000, 2000 and 4000 elements
        testings if the size of workflow grows linearly
    """

    wf_1000 = create_wf(size=1000)
    wf_1000_mem = asizeof.asizeof(wf_1000)

    wf_2000 = create_wf(size=2000)
    wf_2000_mem = asizeof.asizeof(wf_2000)

    wf_4000 = create_wf(size=4000)
    wf_4000_mem = asizeof.asizeof(wf_4000)
    # checking if it's linear with the size of the splitter
    # check print(asizeof.asized(wf_4000, detail=2).format()) in case of problems
    assert wf_4000_mem / wf_2000_mem == approx(2, 0.05)
    assert wf_2000_mem / wf_1000_mem == approx(2, 0.05)


def test_load_task_memory():
    """creating two workflow with relatively big splitter: 1000 and 4000 elements
        testings if load_task for a single element returns tasks of a similar size
    """

    wf_1000 = create_wf(size=1000)
    wf_1000_pkl = wf_1000.pickle_task()
    wf_1000_loaded = load_task(task_pkl=wf_1000_pkl, ind=1)
    wf_1000_single_mem = asizeof.asizeof(wf_1000_loaded)

    wf_4000 = create_wf(size=4000)
    wf_4000_pkl = wf_4000.pickle_task()
    wf_4000_loaded = load_task(task_pkl=wf_4000_pkl, ind=1)
    wf_4000_single_mem = asizeof.asizeof(wf_4000_loaded)

    # checking if it doesn't change with size of the splitter
    # check print(asizeof.asized(wf_4000_loaded, detail=2).format()) in case of problems
    assert wf_1000_single_mem / wf_4000_single_mem == approx(1, 0.05)
