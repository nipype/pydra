import pytest, pdb
from time import sleep

from ..submitter import Submitter
from ..task import to_task
from ..node import Workflow


Plugins = ["serial", "cf"]

@to_task
def double(x):
    return x * 2

@to_task
def multiply(x, y):
    return x * y

@to_task
def add2(x):
    return x + 2


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1(plugin):
    """ workflow with one task and no splitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    assert 4 == results.output.out


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_2(plugin):
    """ workflow with 2 tasks, no splitter"""
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    assert 8 == results.output.out


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_1(plugin):
    """ Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split(("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_1(plugin):
    """ workflow with one task, a splitter on the task level"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]



@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_2(plugin):
    """ workflow with one task, splitters and combiner for workflow"""
    wf = Workflow(name="wf_st_2", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split(("x")).combine(combiner="x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [[({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]]
    assert results[0][0].output.out == 3
    assert results[0][1].output.out == 4


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_2(plugin):
    """ workflow with one task, splitters and combiner on the task level"""
    wf = Workflow(name="wf_ndst_2", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x").combine(combiner="x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [[({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]]
    assert results.output.out[0] == [3, 4]



@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_3(plugin):
    """ workflow with 2 tasks, splitter on wf level"""
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.split(("x", "y"))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results[0].output.out == 13
    assert results[1].output.out == 26


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_3(plugin):
    """Test workflow with 2 tasks, splitter on a task level"""
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(("x", "y")))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results.output.out == [13, 26]


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_4(plugin):
    """ workflow with two tasks, scalar splitter and combiner for the workflow"""
    wf = Workflow(name="wf_st_4", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split(("x", "y"), x=[1, 2], y=[11, 12])
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [
    #     [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    # ]
    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 26


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_4(plugin):
    """ workflow with two tasks, scalar splitter and combiner on tasks level"""
    wf = Workflow(name="wf_ndst_4", input_spec=["a", "b"])
    wf.add(multiply(name="mult", x=wf.lzin.a, y=wf.lzin.b).split(("x", "y")))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))

    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin
    wf.inputs.a = [1, 2]
    wf.inputs.b = [11, 12]
    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    # expected: [
    #     [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    # ]
    assert results.output.out[0] == [13, 26]


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_5(plugin):
    """ workflow with two tasks, outer splitter and combiner for the workflow"""
    wf = Workflow(name="wf_st_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split(["x", "y"], x=[1, 2], y=[11, 12])
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 24
    assert results[1][0].output.out == 14
    assert results[1][1].output.out == 26


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_5(plugin):
    """ workflow with two tasks, outer splitter and combiner on tasks level"""
    wf = Workflow(name="wf_ndst_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert results.output.out[0] == [13, 24]
    assert results.output.out[1] == [14, 26]


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_6(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter on the workflow level
    """
    wf = Workflow(name="wf_st_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12])

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert len(results) == 6
    assert results[0].output.out == 39
    assert results[1].output.out == 42
    assert results[5].output.out == 70


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_6(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter on the tasks levels
    """
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert len(results.output.out) == 6
    assert results.output.out == [39, 42, 52, 56, 65, 70]


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_7(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner on the workflow level
    """
    wf = Workflow(name="wf_st_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert len(results) == 2
    assert results[0][0].output.out == 39
    assert results[0][1].output.out == 52
    assert results[0][2].output.out == 65
    assert results[1][0].output.out == 42
    assert results[1][1].output.out == 56
    assert results[1][2].output.out == 70


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_7(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner on the tasks levels
    """
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out).
           combine("add2x.x"))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()

    assert len(results.output.out) == 2
    assert results.output.out[0] == [39, 52, 65]
    assert results.output.out[1] == [42, 56, 70]


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_1(plugin):
    """ workflow as a node
        workflow-node with one task and no splitter
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.inputs.x = 2

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    while not wf.done:
        sleep(1)
    results = wf.result()
    assert 4 == results.output.out
