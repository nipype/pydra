from ..core import Workflow
from ..helpers import load_task
from ... import mark

import numpy as np
from pympler import asizeof


def test_load_task_memory():
    """creating two workflow with relatively big splitter: 1000 and 4000 elements
        testings if load_task for a single element returns tasks of a similar size
    """

    def generate_list(l):
        return np.arange(l).tolist()

    @mark.task
    def show_var(a):
        return a

    def create_wf_pkl(size):
        wf = Workflow(name="wf", input_spec=["x"])
        wf.split("x", x=generate_list(size))
        wf.add(show_var(name="show", a=wf.lzin.x))
        wf.set_output([("out", wf.show.lzout.out)])
        wf.state.prepare_states(wf.inputs)
        wf.state.prepare_inputs()
        wf_pkl = wf.pickle_task()
        return wf_pkl

    wf_1000_pkl = create_wf_pkl(size=1000)
    wf_1000_loaded = load_task(task_pkl=wf_1000_pkl, ind=1)
    wf_1000_single_mem = asizeof.asizeof(wf_1000_loaded)

    wf_4000_pkl = create_wf_pkl(size=4000)
    wf_4000_loaded = load_task(task_pkl=wf_4000_pkl, ind=1)
    wf_4000_single_mem = asizeof.asizeof(wf_4000_loaded)

    assert abs(wf_1000_single_mem - wf_4000_single_mem) / wf_1000_single_mem < 0.1
