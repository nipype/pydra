import pytest
import shutil, os
import time
import platform

from .utils import add2, add2_wait, multiply, power, identity, list_output, fun_addvar3
from ..submitter import Submitter
from ..core import Workflow
from ... import mark


if bool(shutil.which("sbatch")):
    Plugins = ["cf", "slurm"]
else:
    Plugins = ["cf"]


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1(plugin):
    """ workflow with one task and no splitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1a_outpastuple(plugin):
    """ workflow with one task and no splitter
        set_output takes a tuple
    """
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output(("out", wf.add2.lzout.out))
    wf.inputs.x = 2
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1_call_subm(plugin):
    """using wf.__call_ with submitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1_call_plug(plugin):
    """using wf.__call_ with plugin"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin

    wf(plugin=plugin)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_1_call_exception(plugin):
    """using wf.__call_ with plugin and submitter - should raise an exception"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        with pytest.raises(Exception) as e:
            wf(submitter=sub, plugin=plugin)
        assert "Specify submitter OR plugin" in str(e.value)


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
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_2a(plugin):
    """ workflow with 2 tasks, no splitter
        creating add2_task first (before calling add method),
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    add2_task = add2(name="add2")
    add2_task.inputs.x = wf.mult.lzout.out
    wf.add(add2_task)
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 8 == results.output.out
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_2b(plugin):
    """ workflow with 2 tasks, no splitter
        creating add2_task first (before calling add method),
        adding inputs.x after add method
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    add2_task = add2(name="add2")
    wf.add(add2_task)
    add2_task.inputs.x = wf.mult.lzout.out
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 8 == results.output.out

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_2c_multoutp(plugin):
    """ workflow with 2 tasks, no splitter
        setting multiple outputs for the workflow
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    add2_task = add2(name="add2")
    add2_task.inputs.x = wf.mult.lzout.out
    wf.add(add2_task)
    # setting multiple output (from both nodes)
    wf.set_output([("out_add2", wf.add2.lzout.out), ("out_mult", wf.mult.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking outputs from both nodes
    assert 6 == results.output.out_mult
    assert 8 == results.output.out_add2
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_2d_outpasdict(plugin):
    """ workflow with 2 tasks, no splitter
        setting multiple outputs using a dictionary
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    add2_task = add2(name="add2")
    add2_task.inputs.x = wf.mult.lzout.out
    wf.add(add2_task)
    # setting multiple output (from both nodes)
    wf.set_output({"out_add2": wf.add2.lzout.out, "out_mult": wf.mult.lzout.out})
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking outputs from both nodes
    assert 6 == results.output.out_mult
    assert 8 == results.output.out_add2
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_1(plugin):
    """ Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split(("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_1_call_subm(plugin):
    """ Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split(("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_1_call_plug(plugin):
    """ Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split(("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    wf(plugin=plugin)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_1(plugin):
    """ workflow with one task, a splitter on the task level"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_updatespl_1(plugin):
    """ workflow with one task,
        a splitter on the task level is added *after* calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin
    wf.add2.split("x")

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_updatespl_1a(plugin):
    """ workflow with one task (initialize before calling add),
        a splitter on the task level is added *after* calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    task_add2 = add2(name="add2", x=wf.lzin.x)
    wf.add(task_add2)
    task_add2.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_updateinp_1(plugin):
    """ workflow with one task,
        a splitter on the task level,
        updating input of the task after calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin
    wf.add2.split("x")
    wf.add2.inputs.x = wf.lzin.y

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [13, 14]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


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
        sub(wf)

    results = wf.result()
    # expected: [[({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]]
    assert results[0][0].output.out == 3
    assert results[0][1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_2(plugin):
    """ workflow with one task, splitters and combiner on the task level"""
    wf = Workflow(name="wf_ndst_2", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x").combine(combiner="x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [[({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]]
    assert results.output.out[0] == [3, 4]
    assert wf.output_dir.exists()


# workflows with structures A -> B


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
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results[0].output.out == 13
    assert results[1].output.out == 26
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


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
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results.output.out == [13, 26]
    # checking the output directory
    assert wf.output_dir.exists()


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
        sub(wf)

    results = wf.result()
    # expected: [
    #     [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    # ]
    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 26
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


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
        sub(wf)

    results = wf.result()
    # expected: [
    #     [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    # ]
    assert results.output.out[0] == [13, 26]
    # checking the output directory
    assert wf.output_dir.exists()


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
        sub(wf)

    results = wf.result()
    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 24
    assert results[1][0].output.out == 14
    assert results[1][1].output.out == 26
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


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
        sub(wf)

    results = wf.result()
    assert results.output.out[0] == [13, 24]
    assert results.output.out[1] == [14, 26]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures A -> C, B -> C


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
        sub(wf)

    results = wf.result()
    assert len(results) == 6
    assert results[0].output.out == 39
    assert results[1].output.out == 42
    assert results[5].output.out == 70
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


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
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 6
    assert results.output.out == [39, 42, 52, 56, 65, 70]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_7(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner on the workflow level
    """
    wf = Workflow(name="wf_st_7", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 2
    assert results[0][0].output.out == 39
    assert results[0][1].output.out == 52
    assert results[0][2].output.out == 65
    assert results[1][0].output.out == 42
    assert results[1][1].output.out == 56
    assert results[1][2].output.out == 70
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_7(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner on the tasks levels
    """
    wf = Workflow(name="wf_ndst_7", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out).combine(
            "add2x.x"
        )
    )
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 2
    assert results.output.out[0] == [39, 52, 65]
    assert results.output.out[1] == [42, 56, 70]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_8(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner (from the second task) on the workflow level
    """
    wf = Workflow(name="wf_st_8", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("y")

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 3
    assert results[0][0].output.out == 39
    assert results[0][1].output.out == 42
    assert results[1][0].output.out == 52
    assert results[1][1].output.out == 56
    assert results[2][0].output.out == 65
    assert results[2][1].output.out == 70
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_8(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and partial combiner (from the second task) on the tasks levels
    """
    wf = Workflow(name="wf_ndst_8", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out).combine(
            "add2y.x"
        )
    )
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 3
    assert results.output.out[0] == [39, 42]
    assert results.output.out[1] == [52, 56]
    assert results.output.out[2] == [65, 70]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_9(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and full combiner on the workflow level
    """
    wf = Workflow(name="wf_st_9", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine(["x", "y"])
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 1
    assert results[0][0].output.out == 39
    assert results[0][1].output.out == 42
    assert results[0][2].output.out == 52
    assert results[0][3].output.out == 56
    assert results[0][4].output.out == 65
    assert results[0][5].output.out == 70
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_9(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter and full combiner on the tasks levels
    """
    wf = Workflow(name="wf_ndst_9", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out).combine(
            ["add2x.x", "add2y.x"]
        )
    )
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()

    assert len(results.output.out) == 1
    assert results.output.out == [[39, 42, 52, 56, 65, 70]]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with Left and Right part in splitters A -> B (L&R parts of the splitter)


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstLR_1(plugin):
    """ Test workflow with 2 tasks, splitters on tasks levels
        The second task has its own simple splitter
        and the  Left part from the first task should be added
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(multiply(name="mult", x=wf.add2.lzout.out, y=wf.lzin.y).split("y"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # checking if the splitter is created properly
    assert wf.mult.state.splitter == ["_add2", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["add2.x", "mult.y", "*"]

    results = wf.result()
    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]
    assert results.output.out == [33, 36, 44, 48]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstLR_1a(plugin):
    """ Test workflow with 2 tasks, splitters on tasks levels
        The second task has splitter that has Left part (from previous state)
        and the Right part (it's onw splitter)
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2.lzout.out, y=wf.lzin.y).split(["_add2", "y"])
    )
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # checking if the splitter is created properly
    assert wf.mult.state.splitter == ["_add2", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["add2.x", "mult.y", "*"]

    results = wf.result()
    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]
    assert results.output.out == [33, 36, 44, 48]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstLR_2(plugin):
    """ Test workflow with 2 tasks, splitters on tasks levels
        The second task has its own outer splitter
        and the  Left part from the first task should be added
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y", "z"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(
        fun_addvar3(name="addvar", a=wf.add2.lzout.out, b=wf.lzin.y, c=wf.lzin.z).split(
            ["b", "c"]
        )
    )
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [10, 20]
    wf.inputs.z = [100, 200]
    wf.set_output([("out", wf.addvar.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # checking if the splitter is created properly
    assert wf.addvar.state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf.addvar.state.splitter_rpn == ["add2.x", "addvar.b", "addvar.c", "*", "*"]

    results = wf.result()
    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert results.output.out == [
        113,
        213,
        123,
        223,
        114,
        214,
        124,
        224,
        115,
        215,
        125,
        225,
    ]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstLR_2a(plugin):
    """ Test workflow with 2 tasks, splitters on tasks levels
        The second task has splitter that has Left part (from previous state)
        and the Right part (it's onw outer splitter)
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y", "z"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(
        fun_addvar3(name="addvar", a=wf.add2.lzout.out, b=wf.lzin.y, c=wf.lzin.z).split(
            ["_add2", ["b", "c"]]
        )
    )
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [10, 20]
    wf.inputs.z = [100, 200]
    wf.set_output([("out", wf.addvar.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # checking if the splitter is created properly
    assert wf.addvar.state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf.addvar.state.splitter_rpn == ["add2.x", "addvar.b", "addvar.c", "*", "*"]

    results = wf.result()
    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert results.output.out == [
        113,
        213,
        123,
        223,
        114,
        214,
        124,
        224,
        115,
        215,
        125,
        225,
    ]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with inner splitters A -> B (inner spl)


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstinner_1(plugin):
    """ workflow with 2 tasks,
        the second task has inner splitter
    """
    wf = Workflow(name="wf_st_3", input_spec=["x"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(add2(name="add2", x=wf.list.lzout.out).split("x"))
    wf.inputs.x = 1
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.add2.state.splitter == "add2.x"
    assert wf.add2.state.splitter_rpn == ["add2.x"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [3, 4, 5]

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstinner_2(plugin):
    """ workflow with 2 tasks,
        the second task has two inputs and inner splitter from one of the input
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split("x"))
    wf.inputs.x = 1
    wf.inputs.y = 10
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == "mult.x"
    assert wf.mult.state.splitter_rpn == ["mult.x"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [10, 20, 30]

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstinner_3(plugin):
    """ workflow with 2 tasks,
        the second task has two inputs and outer splitter that includes an inner field
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split(["x", "y"]))
    wf.inputs.x = 1
    wf.inputs.y = [10, 100]
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == ["mult.x", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["mult.x", "mult.y", "*"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [10, 100, 20, 200, 30, 300]

    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstinner_4(plugin):
    """ workflow with 3 tasks,
        the second task has two inputs and inner splitter from one of the input,
        the third task has no its own splitter
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split("x"))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = 1
    wf.inputs.y = 10
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == "mult.x"
    assert wf.mult.state.splitter_rpn == ["mult.x"]
    assert wf.add2.state.splitter == "_mult"
    assert wf.add2.state.splitter_rpn == ["mult.x"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [12, 22, 32]

    assert wf.output_dir.exists()


# workflow that have some single values as the input


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_singl_1(plugin):
    """ workflow with two tasks, only one input is in the splitter and combiner"""
    wf = Workflow(name="wf_st_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split("x", x=[1, 2], y=11)
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 24
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_singl_1(plugin):
    """ workflow with two tasks, outer splitter and combiner on tasks level;
        only one input is part of the splitter, the other is a single value
    """
    wf = Workflow(name="wf_ndst_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split("x"))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = 11
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out[0] == [13, 24]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_st_singl_2(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter on the workflow level
        only one input is part of the splitter, the other is a single value
    """
    wf = Workflow(name="wf_st_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split("x", x=[1, 2, 3], y=11)

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 3
    assert results[0].output.out == 39
    assert results[1].output.out == 52
    assert results[2].output.out == 65
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndst_singl_2(plugin):
    """ workflow with three tasks, third one connected to two previous tasks,
        splitter on the tasks levels
        only one input is part of the splitter, the other is a single value
    """
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = 11
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 3
    assert results.output.out == [39, 52, 65]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures wf(A)


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
        sub(wf)

    results = wf.result()
    assert results.output.out == 4
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfinp_1(plugin):
    """ workflow as a node
        workflow-node with one task and no splitter
        input set for the main workflow
    """
    wf = Workflow(name="wf", input_spec=["x"])
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])

    wf.add(wfnd)
    wf.inputs.x = 2
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert results.output.out == 4
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfndupdate(plugin):
    """ workflow as a node
        workflow-node with one task and no splitter
        wfasnode input is updated to use the main workflow input
    """

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=2)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])

    wf = Workflow(name="wf", input_spec=["x"], x=3)
    wfnd.inputs.x = wf.lzin.x
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == 5
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfndupdate_rerun(plugin):
    """ workflow as a node
        workflow-node with one task and no splitter
        wfasnode is run first and later is
        updated to use the main workflow input
    """

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=2)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    with Submitter(plugin=plugin) as sub:
        sub(wfnd)

    wf = Workflow(name="wf", input_spec=["x"], x=3)
    # trying to set before
    wfnd.inputs.x = wf.lzin.x
    wf.add(wfnd)
    # trying to set after add...
    wf.wfnd.inputs.x = wf.lzin.x
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == 5
    assert wf.output_dir.exists()

    # adding another layer of workflow
    wf_o = Workflow(name="wf_o", input_spec=["x"], x=4)
    wf.inputs.x = wf_o.lzin.x
    wf_o.add(wf)
    wf_o.set_output([("out", wf_o.wf.lzout.out)])
    wf_o.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf_o)

    results = wf_o.result()
    assert results.output.out == 6
    assert wf_o.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_st_1(plugin):
    """ workflow as a node
        workflow-node with one task,
        splitter for wfnd
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.split("x")
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_st_updatespl_1(plugin):
    """ workflow as a node
        workflow-node with one task,
        splitter for wfnd is set after add
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wfnd.split("x")
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_ndst_1(plugin):
    """ workflow as a node
        workflow-node with one task,
        splitter for node
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x).split("x"))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    # TODO: without this the test is failing
    wfnd.plugin = plugin
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_ndst_updatespl_1(plugin):
    """ workflow as a node
        workflow-node with one task,
        splitter for node added after add
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    # TODO: without this the test is failing
    wfnd.plugin = plugin
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wfnd.add2.split("x")
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfst_1(plugin):
    """ workflow as a node
        workflow-node with one task,
        splitter for the main workflow
    """
    wf = Workflow(name="wf", input_spec=["x"])
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])

    wf.add(wfnd)
    wf.split("x")
    wf.inputs.x = [2, 4]
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results[0].output.out == 4
    assert results[1].output.out == 6
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


# workflows with structures wf(A) -> B


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_st_2(plugin):
    """ workflow as a node,
        the main workflow has two tasks,
        splitter for wfnd
    """
    wfnd = Workflow(name="wfnd", input_spec=["x", "y"])
    wfnd.add(multiply(name="mult", x=wfnd.lzin.x, y=wfnd.lzin.y))
    wfnd.set_output([("out", wfnd.mult.lzout.out)])
    wfnd.split(("x", "y"))
    wfnd.inputs.x = [2, 4]
    wfnd.inputs.y = [1, 10]

    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(wfnd)
    wf.add(add2(name="add2", x=wf.wfnd.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out == [4, 42]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfst_2(plugin):
    """ workflow as a node,
        the main workflow has two tasks,
        splitter for the main workflow
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wfnd = Workflow(name="wfnd", input_spec=["x", "y"], x=wf.lzin.x, y=wf.lzin.y)
    wfnd.add(multiply(name="mult", x=wfnd.lzin.x, y=wfnd.lzin.y))
    wfnd.set_output([("out", wfnd.mult.lzout.out)])

    wf.add(wfnd)
    wf.add(add2(name="add2", x=wf.wfnd.lzout.out))
    wf.split(("x", "y"))
    wf.inputs.x = [2, 4]
    wf.inputs.y = [1, 10]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results[0].output.out == 4
    assert results[1].output.out == 42
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


# workflows with structures A -> wf(B)


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_ndst_3(plugin):
    """ workflow as the second node,
        the main workflow has two tasks,
        splitter for the first task
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(("x", "y")))
    wf.inputs.x = [2, 4]
    wf.inputs.y = [1, 10]

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.mult.lzout.out)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)

    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out == [4, 42]
    # checking the output directory
    assert wf.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wfasnd_wfst_3(plugin):
    """ workflow as the second node,
        the main workflow has two tasks,
        splitter for the main workflow
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.inputs.x = [2, 4]
    wf.inputs.y = [1, 10]
    wf.split(("x", "y"))

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.mult.lzout.out)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)

    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results[0].output.out == 4
    assert results[1].output.out == 42
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


# Testing caching


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_cachedir(plugin, tmpdir):
    """ wf with provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_wf_cache_1")

    wf = Workflow(name="wf_2", input_spec=["x", "y"], cache_dir=cache_dir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out

    shutil.rmtree(cache_dir)


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_cachedir_relativepath(tmpdir, plugin):
    """ wf with provided cache_dir as relative path"""
    tmpdir.chdir()
    cache_dir = "test_wf_cache_2"

    wf = Workflow(name="wf_2", input_spec=["x", "y"], cache_dir=cache_dir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out

    shutil.rmtree(cache_dir)


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_cachelocations(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir;
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out

    # checking execution time
    assert t1 > 3
    assert t2 < 0.5

    # checking if the second wf didn't run again
    assert wf1.output_dir.exists()
    assert not wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_state_cachelocations(plugin, tmpdir):
    """
    Two identical wfs (with states) with provided cache_dir;
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 4]
    wf1.split(splitter=("x", "y"))
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert results1[0].output.out == 8
    assert results1[1].output.out == 82

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.split(splitter=("x", "y"))
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2[0].output.out == 8
    assert results2[1].output.out == 82

    # checking execution time
    assert t1 > 3
    assert t2 < 0.5

    # checking all directories
    assert wf1.output_dir
    for odir in wf1.output_dir:
        assert odir.exists()
    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir
    for odir in wf2.output_dir:
        assert not odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_state_cachelocations_updateinp(plugin, tmpdir):
    """
    Two identical wfs (with states) with provided cache_dir;
    the second wf has cache_locations and should not recompute the results
    (the lazy input of the node is updated to the correct one,
    i.e. the same as in wf1, after adding the node to the wf)
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 4]
    wf1.split(splitter=("x", "y"))
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert results1[0].output.out == 8
    assert results1[1].output.out == 82

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.x))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.split(splitter=("x", "y"))
    wf2.plugin = plugin
    wf2.mult.inputs.y = wf2.lzin.y

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2[0].output.out == 8
    assert results2[1].output.out == 82

    # checking execution time
    assert t1 > 3
    assert t2 < 0.5

    # checking all directories
    assert wf1.output_dir
    for odir in wf1.output_dir:
        assert odir.exists()
    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir
    for odir in wf2.output_dir:
        assert not odir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_state_n_nostate_cachelocations(plugin, tmpdir):
    """
    Two wfs with provided cache_dir, the first one has no state, the second has;
    the second wf has cache_locations and should not recompute only one element
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert results1.output.out == 8

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.split(splitter=("x", "y"))
    wf2.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf2)

    results2 = wf2.result()
    assert results2[0].output.out == 8
    assert results2[1].output.out == 82

    # checking the directory from the first wf
    assert wf1.output_dir.exists()
    # checking directories from the second wf, only second element should be recomputed
    assert not wf2.output_dir[0].exists()
    assert wf2.output_dir[1].exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_cachelocations_updated(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir;
    the second wf has cache_locations in init,
     that is later overwritten in Submitter.__call__;
    the cache_locations from call doesn't exist so the second task should run again
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir1_empty = tmpdir.mkdir("test_wf_cache3_empty")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    # changing cache_locations to non-existing dir
    with Submitter(plugin=plugin) as sub:
        sub(wf2, cache_locations=cache_dir1_empty)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out

    # checking execution time
    assert t1 > 3
    assert t2 > 3

    # checking if both wf run
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_cachelocations_recompute(plugin, tmpdir):
    """
    Two wfs with the same inputs but slightly different graph;
    the second wf should recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    # different argument assigment
    wf2.add(multiply(name="mult", x=wf2.lzin.y, y=wf2.lzin.x))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out

    # checking execution time
    assert t1 > 3
    assert t2 > 3

    # checking if both dir exists
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstate_cachelocations(plugin, tmpdir):
    """
    Two wfs with identical inputs and node states;
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(
        multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y).split(splitter=("x", "y"))
    )
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 4]
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert results1.output.out == [8, 82]

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(
        multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y).split(splitter=("x", "y"))
    )
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2.output.out == [8, 82]

    # checking execution time
    assert t1 > 3
    assert t2 < 0.5

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert not wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstate_cachelocations_updatespl(plugin, tmpdir):
    """
    Two wfs with identical inputs and node state (that is set after adding the node!);
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(
        multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y).split(splitter=("x", "y"))
    )
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 4]
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert results1.output.out == [8, 82]

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))

    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.mult.split(splitter=("x", "y"))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2.output.out == [8, 82]

    # checking execution time
    assert t1 > 3
    assert t2 < 0.5

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert not wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_ndstate_cachelocations_recompute(plugin, tmpdir):
    """
    Two wfs (with nodes with states) with provided cache_dir;
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(
        multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y).split(splitter=("x", "y"))
    )
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 4]
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert results1.output.out == [8, 82]

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(
        multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y).split(splitter=["x", "y"])
    )
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = [2, 20]
    wf2.inputs.y = [3, 4]
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2.output.out == [8, 10, 62, 82]

    # checking execution time
    assert t1 > 3
    assert t2 > 3

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_nostate_runtwice_usecache(plugin, tmpdir):
    """
    running worflow (without state) twice,
    the second run should use the results from the first one
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 8 == results1.output.out
    # checkoing output_dir after the first run
    assert wf1.output_dir.exists()

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 8 == results1.output.out
    # checking if no new directory is not created
    assert cache_dir_content == os.listdir(wf1.cache_dir)


@pytest.mark.parametrize("plugin", Plugins)
def test_wf_state_runtwice_usecache(plugin, tmpdir):
    """
    running worflow with a state twice,
    the second run should use the results from the first one
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.split(splitter=("x", "y"))
    wf1.inputs.x = [2, 20]
    wf1.inputs.y = [3, 30]
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 8 == results1[0].output.out
    assert 602 == results1[1].output.out

    # checkoing output_dir after the first run
    assert [odir.exists() for odir in wf1.output_dir]

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 8 == results1[0].output.out
    assert 602 == results1[1].output.out
    # checking if no new directory is not created
    assert cache_dir_content == os.listdir(wf1.cache_dir)


@pytest.fixture
def create_tasks():
    wf = Workflow(name="wf", input_spec=["x"])
    wf.inputs.x = 1
    wf.add(add2(name="t1", x=wf.lzin.x))
    wf.add(multiply(name="t2", x=wf.t1.lzout.out, y=2))
    wf.set_output([("out", wf.t2.lzout.out)])
    t1 = wf.name2obj["t1"]
    t2 = wf.name2obj["t2"]
    return wf, t1, t2


def test_cache_propagation1(tmpdir, create_tasks):
    """No cache set, all independent"""
    wf, t1, t2 = create_tasks
    wf(plugin="cf")
    assert wf.cache_dir == t1.cache_dir == t2.cache_dir
    wf.cache_dir = (tmpdir / "shared").strpath
    wf(plugin="cf")
    assert wf.cache_dir == t1.cache_dir == t2.cache_dir


def test_cache_propagation2(tmpdir, create_tasks):
    """Task explicitly states no inheriting"""
    wf, t1, t2 = create_tasks
    wf.cache_dir = (tmpdir / "shared").strpath
    t2.allow_cache_override = False
    wf(plugin="cf")
    assert wf.cache_dir == t1.cache_dir != t2.cache_dir


def test_cache_propagation3(tmpdir, create_tasks):
    """Shared cache_dir with state"""
    wf, t1, t2 = create_tasks
    wf.inputs.x = [1, 2]
    wf.split("x")
    wf.cache_dir = (tmpdir / "shared").strpath
    wf(plugin="cf")
    assert wf.cache_dir == t1.cache_dir == t2.cache_dir


def test_workflow_combine1(tmpdir):
    wf1 = Workflow(name="wf1", input_spec=["a", "b"], a=[1, 2], b=[2, 3])
    wf1.add(power(name="power", a=wf1.lzin.a, b=wf1.lzin.b).split(["a", "b"]))
    wf1.add(identity(name="identity1", x=wf1.power.lzout.out).combine("power.a"))
    wf1.add(identity(name="identity2", x=wf1.identity1.lzout.out).combine("power.b"))
    wf1.set_output(
        {
            "out_pow": wf1.power.lzout.out,
            "out_iden1": wf1.identity1.lzout.out,
            "out_iden2": wf1.identity2.lzout.out,
        }
    )
    wf1.cache_dir = tmpdir
    result = wf1(plugin="cf")

    assert result.output.out_pow == [1, 1, 4, 8]
    assert result.output.out_iden1 == [[1, 4], [1, 8]]
    assert result.output.out_iden2 == [[[1, 4], [1, 8]]]


def test_workflow_combine2(tmpdir):
    wf1 = Workflow(name="wf1", input_spec=["a", "b"], a=[1, 2], b=[2, 3])
    wf1.add(
        power(name="power", a=wf1.lzin.a, b=wf1.lzin.b).split(["a", "b"]).combine("a")
    )
    wf1.add(identity(name="identity", x=wf1.power.lzout.out).combine("power.b"))
    wf1.set_output({"out_pow": wf1.power.lzout.out, "out_iden": wf1.identity.lzout.out})
    wf1.cache_dir = tmpdir
    result = wf1(plugin="cf")

    assert result.output.out_pow == [[1, 4], [1, 8]]
    assert result.output.out_iden == [[[1, 4], [1, 8]]]
