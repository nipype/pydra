from ..core import Workflow
from ..task import to_task

import time


@to_task
def sleep_add_one(x):
    time.sleep(1)
    return x + 1


def test_concurrent_wf():
    # concurrent workflow
    # A --> C
    # B --> D
    wf = Workflow('new_wf', input_spec=['x', 'y'])
    wf.inputs.x = 5
    wf.inputs.y = 10
    wf.add(sleep_add_one(name="taska", x=wf.lzin.x))
    wf.add(sleep_add_one(name="taskb", x=wf.lzin.y))
    wf.add(sleep_add_one(name="taskc", x=wf.taska.lzout.out))
    wf.add(sleep_add_one(name="taskd", x=wf.taskb.lzout.out))
    wf.set_output([("out1", wf.taskc.lzout.out),
                   ("out2", wf.taskd.lzout.out)])
    wf.plugin = 'cf'
    res = wf.run()
    assert res


def test_wf_in_wf():
    """WF(A --> SUBWF(A --> B) --> B)"""
    wf = Workflow(name='wf_in_wf', input_spec=['x'])
    wf.inputs.x = 3
    wf.add(sleep_add_one(name="wf_a", x=wf.lzin.x))

    # workflow task
    subwf = Workflow(name='sub_wf', input_spec=['x'])
    subwf.add(sleep_add_one(name="sub_a", x=subwf.lzin.x))
    subwf.add(sleep_add_one(name="sub_b", x=subwf.sub_a.lzout.out))
    subwf.set_output([("out", subwf.sub_b.lzout.out)])
    # connect, then add
    subwf.inputs.x = wf.wf_a.lzout.out
    wf.add(subwf)

    wf.add(sleep_add_one(name="wf_b", x=wf.sub_wf.lzout.out))
    wf.set_output([("out", wf.wf_b.lzout.out)])

    wf.plugin = 'cf'
    res = wf.run()
    assert res
