from ..node import Workflow
from ..task import to_task
from ..submitter import Submitter

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

    with Submitter('cf') as sub:
        procs = sub.worker.nr_proc
        start = time.time()
        sub.run(wf)

    diff = time.time() - start

    if procs >= 2:
        assert diff < 4
