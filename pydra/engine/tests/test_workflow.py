import pytest
import shutil, os, sys
import time
import attr
from pathlib import Path
import logging

from .utils import (
    add2,
    add2_wait,
    multiply,
    power,
    ten,
    identity,
    identity_2flds,
    list_output,
    fun_addsubvar,
    fun_addvar3,
    fun_addvar,
    fun_addtwo,
    add2_sub2_res,
    fun_addvar_none,
    fun_addvar_default,
    fun_write_file,
    fun_write_file_list,
    fun_write_file_list2dict,
    list_sum,
    DOT_FLAG,
)
from ..submitter import Submitter
from ..core import Workflow
from ... import mark
from ..specs import SpecInfo, BaseSpec, ShellSpec


def test_wf_no_input_spec():
    with pytest.raises(ValueError, match="Empty input_spec"):
        Workflow(name="workflow")


def test_wf_specinfo_input_spec():
    input_spec = SpecInfo(
        name="Input",
        fields=[
            ("a", str, "", {"mandatory": True}),
            ("b", dict, {"foo": 1, "bar": False}, {"mandatory": False}),
        ],
        bases=(BaseSpec,),
    )
    wf = Workflow(
        name="workflow",
        input_spec=input_spec,
    )
    for x in ["a", "b", "_graph_checksums"]:
        assert hasattr(wf.inputs, x)
    assert wf.inputs.a == ""
    assert wf.inputs.b == {"foo": 1, "bar": False}
    bad_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("a", str, {"mandatory": True}),
        ],
        bases=(ShellSpec,),
    )
    with pytest.raises(
        ValueError, match="Provided SpecInfo must have BaseSpec as it's base."
    ):
        Workflow(name="workflow", input_spec=bad_input_spec)


def test_wf_name_conflict1():
    """raise error when workflow name conflicts with a class attribute or method"""
    with pytest.raises(ValueError) as excinfo1:
        wf = Workflow(name="result", input_spec=["x"])
    assert "Cannot use names of attributes or methods" in str(excinfo1.value)
    with pytest.raises(ValueError) as excinfo2:
        wf = Workflow(name="done", input_spec=["x"])
    assert "Cannot use names of attributes or methods" in str(excinfo2.value)


def test_wf_name_conflict2():
    """raise error when a task with the same name is already added to workflow"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="task_name", x=wf.lzin.x))
    with pytest.raises(ValueError) as excinfo:
        wf.add(identity(name="task_name", x=3))
    assert "Another task named task_name is already added" in str(excinfo.value)


def test_wf_no_output(plugin, tmpdir):
    """Raise error when output isn't set with set_output"""
    wf = Workflow(name="wf_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = 2

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "Workflow output cannot be None" in str(excinfo.value)


def test_wf_1(plugin, tmpdir):
    """workflow with one task and no splitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


def test_wf_1a_outpastuple(plugin, tmpdir):
    """workflow with one task and no splitter
    set_output takes a tuple
    """
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output(("out", wf.add2.lzout.out))
    wf.inputs.x = 2
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


def test_wf_1_call_subm(plugin, tmpdir):
    """using wf.__call_ with submitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        wf(submitter=sub)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


def test_wf_1_call_plug(plugin, tmpdir):
    """using wf.__call_ with plugin"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    wf(plugin=plugin)

    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


def test_wf_1_call_noplug_nosubm(plugin, tmpdir):
    """using wf.__call_ without plugin or submitter"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.cache_dir = tmpdir

    wf()
    results = wf.result()
    assert 4 == results.output.out
    assert wf.output_dir.exists()


def test_wf_1_call_exception(plugin, tmpdir):
    """using wf.__call_ with plugin and submitter - should raise an exception"""
    wf = Workflow(name="wf_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        with pytest.raises(Exception) as e:
            wf(submitter=sub, plugin=plugin)
        assert "Specify submitter OR plugin" in str(e.value)


def test_wf_1_inp_in_call(tmpdir):
    """Defining input in __call__"""
    wf = Workflow(name="wf_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 1
    results = wf(x=2)
    assert 4 == results.output.out


def test_wf_1_upd_in_run(tmpdir):
    """Updating input in __call__"""
    wf = Workflow(name="wf_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 1
    results = wf(x=2)
    assert 4 == results.output.out


def test_wf_2(plugin, tmpdir):
    """workflow with 2 tasks, no splitter"""
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out


def test_wf_2a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 8 == results.output.out
    assert wf.output_dir.exists()


def test_wf_2b(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 8 == results.output.out

    assert wf.output_dir.exists()


def test_wf_2c_multoutp(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking outputs from both nodes
    assert 6 == results.output.out_mult
    assert 8 == results.output.out_add2
    assert wf.output_dir.exists()


def test_wf_2d_outpasdict(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking outputs from both nodes
    assert 6 == results.output.out_mult
    assert 8 == results.output.out_add2
    assert wf.output_dir.exists()


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3(plugin_dask_opt, tmpdir):
    """testing None value for an input"""
    wf = Workflow(name="wf_3", input_spec=["x", "y"])
    wf.add(fun_addvar_none(name="addvar", a=wf.lzin.x, b=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.addvar.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = None
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 4 == results.output.out


@pytest.mark.xfail(reason="the task error doesn't propagate")
def test_wf_3a_exception(plugin, tmpdir):
    """testinh wf without set input, attr.NOTHING should be set
    and the function should raise an exception
    """
    wf = Workflow(name="wf_3", input_spec=["x", "y"])
    wf.add(fun_addvar_none(name="addvar", a=wf.lzin.x, b=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.addvar.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = attr.NOTHING
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with pytest.raises(TypeError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "unsupported" in str(excinfo.value)


def test_wf_4(plugin, tmpdir):
    """wf with a task that doesn't set one input and use the function default value"""
    wf = Workflow(name="wf_4", input_spec=["x", "y"])
    wf.add(fun_addvar_default(name="addvar", a=wf.lzin.x))
    wf.add(add2(name="add2", x=wf.addvar.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 5 == results.output.out


def test_wf_4a(plugin, tmpdir):
    """wf with a task that doesn't set one input,
    the unset input is send to the task input,
    so the task should use the function default value
    """
    wf = Workflow(name="wf_4a", input_spec=["x", "y"])
    wf.add(fun_addvar_default(name="addvar", a=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.addvar.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 5 == results.output.out


def test_wf_5(plugin, tmpdir):
    """wf with two outputs connected to the task outputs
    one set_output
    """
    wf = Workflow(name="wf_5", input_spec=["x", "y"], x=3, y=2)
    wf.add(fun_addsubvar(name="addsub", a=wf.lzin.x, b=wf.lzin.y))
    wf.set_output([("out_sum", wf.addsub.lzout.sum), ("out_sub", wf.addsub.lzout.sub)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 5 == results.output.out_sum
    assert 1 == results.output.out_sub


def test_wf_5a(plugin, tmpdir):
    """wf with two outputs connected to the task outputs,
    set_output set twice
    """
    wf = Workflow(name="wf_5", input_spec=["x", "y"], x=3, y=2)
    wf.add(fun_addsubvar(name="addsub", a=wf.lzin.x, b=wf.lzin.y))
    wf.set_output([("out_sum", wf.addsub.lzout.sum)])
    wf.set_output([("out_sub", wf.addsub.lzout.sub)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert 5 == results.output.out_sum
    assert 1 == results.output.out_sub


def test_wf_5b_exception(tmpdir):
    """set_output used twice with the same name - exception should be raised"""
    wf = Workflow(name="wf_5", input_spec=["x", "y"], x=3, y=2)
    wf.add(fun_addsubvar(name="addsub", a=wf.lzin.x, b=wf.lzin.y))
    wf.set_output([("out", wf.addsub.lzout.sum)])
    wf.cache_dir = tmpdir

    with pytest.raises(Exception) as excinfo:
        wf.set_output([("out", wf.addsub.lzout.sub)])
    assert "is already set" in str(excinfo.value)


def test_wf_6(plugin, tmpdir):
    """wf with two tasks and two outputs connected to both tasks,
    one set_output
    """
    wf = Workflow(name="wf_6", input_spec=["x", "y"], x=2, y=3)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out1", wf.mult.lzout.out), ("out2", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 6 == results.output.out1
    assert 8 == results.output.out2


def test_wf_6a(plugin, tmpdir):
    """wf with two tasks and two outputs connected to both tasks,
    set_output used twice
    """
    wf = Workflow(name="wf_6", input_spec=["x", "y"], x=2, y=3)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out1", wf.mult.lzout.out)])
    wf.set_output([("out2", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 6 == results.output.out1
    assert 8 == results.output.out2


def test_wf_st_1(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_st_1_call_subm(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_st_1_call_plug(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__(plugin)
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    wf(plugin=plugin)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_st_1_call_selfplug(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__() and using self.plugin
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    wf()
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_st_1_call_noplug_nosubm(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__()  without plugin and submitter
    (a submitter should be created within the __call__ function)
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    wf()
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_st_1_inp_in_call(tmpdir):
    """Defining input in __call__"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"], cache_dir=tmpdir).split("x")
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.set_output([("out", wf.add2.lzout.out)])
    results = wf(x=[1, 2])
    assert results[0].output.out == 3
    assert results[1].output.out == 4


def test_wf_st_1_upd_inp_call(tmpdir):
    """Updating input in __call___"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"], cache_dir=tmpdir).split("x")
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = [11, 22]
    wf.set_output([("out", wf.add2.lzout.out)])
    results = wf(x=[1, 2])
    assert results[0].output.out == 3
    assert results[1].output.out == 4


def test_wf_st_noinput_1(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x")
    wf.inputs.x = []
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert results == []
    # checking all directories
    assert wf.output_dir == []


def test_wf_ndst_1(plugin, tmpdir):
    """workflow with one task, a splitter on the task level"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()


def test_wf_ndst_updatespl_1(plugin, tmpdir):
    """workflow with one task,
    a splitter on the task level is added *after* calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir
    wf.add2.split("x")

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


def test_wf_ndst_updatespl_1a(plugin, tmpdir):
    """workflow with one task (initialize before calling add),
    a splitter on the task level is added *after* calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    task_add2 = add2(name="add2", x=wf.lzin.x)
    wf.add(task_add2)
    task_add2.split("x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


def test_wf_ndst_updateinp_1(plugin, tmpdir):
    """workflow with one task,
    a splitter on the task level,
    updating input of the task after calling add
    """
    wf = Workflow(name="wf_spl_1", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.add2.split("x")
    wf.add2.inputs.x = wf.lzin.y
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [13, 14]
    assert wf.output_dir.exists()

    assert wf.output_dir.exists()


def test_wf_ndst_noinput_1(plugin, tmpdir):
    """workflow with one task, a splitter on the task level"""
    wf = Workflow(name="wf_spl_1", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.inputs.x = []
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()

    assert results.output.out == []
    assert wf.output_dir.exists()


def test_wf_st_2(plugin, tmpdir):
    """workflow with one task, splitters and combiner for workflow"""
    wf = Workflow(name="wf_st_2", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x))

    wf.split("x").combine(combiner="x")
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results[0].output.out == 3
    assert results[1].output.out == 4
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_2(plugin, tmpdir):
    """workflow with one task, splitters and combiner on the task level"""
    wf = Workflow(name="wf_ndst_2", input_spec=["x"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x").combine(combiner="x"))
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.output.out == [3, 4]
    assert wf.output_dir.exists()


# workflows with structures A -> B


def test_wf_st_3(plugin, tmpdir):
    """workflow with 2 tasks, splitter on wf level"""
    wf = Workflow(name="wfst_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.split(("x", "y"))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    expected = [
        ({"wfst_3.x": 1, "wfst_3.y": 11}, 13),
        ({"wfst_3.x": 2, "wfst_3.y": 12}, 26),
    ]
    expected_ind = [
        ({"wfst_3.x": 0, "wfst_3.y": 0}, 13),
        ({"wfst_3.x": 1, "wfst_3.y": 1}, 26),
    ]

    results = wf.result()
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]

    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = wf.result(return_inputs=True)
    results_verb_val = wf.result(return_inputs="val")
    for i, res in enumerate(expected):
        assert (results_verb[i][0], results_verb[i][1].output.out) == res
        assert (results_verb_val[i][0], results_verb_val[i][1].output.out) == res

    # checking the return_inputs option return_inputs="ind"
    # it should give indices of inputs (instead of values) for each element
    results_verb_ind = wf.result(return_inputs="ind")
    for i, res in enumerate(expected_ind):
        assert (results_verb_ind[i][0], results_verb_ind[i][1].output.out) == res

    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_3(plugin, tmpdir):
    """Test workflow with 2 tasks, splitter on a task level"""
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(("x", "y")))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results.output.out == [13, 26]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_st_4(plugin, tmpdir):
    """workflow with two tasks, scalar splitter and combiner for the workflow"""
    wf = Workflow(name="wf_st_4", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split(("x", "y"), x=[1, 2], y=[11, 12])
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert results[0].output.out == 13
    assert results[1].output.out == 26
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_4(plugin, tmpdir):
    """workflow with two tasks, scalar splitter and combiner on tasks level"""
    wf = Workflow(name="wf_ndst_4", input_spec=["a", "b"])
    wf.add(multiply(name="mult", x=wf.lzin.a, y=wf.lzin.b).split(("x", "y")))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))

    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir
    wf.inputs.a = [1, 2]
    wf.inputs.b = [11, 12]

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert results.output.out == [13, 26]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_st_5(plugin, tmpdir):
    """workflow with two tasks, outer splitter and no combiner"""
    wf = Workflow(name="wf_st_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split(["x", "y"], x=[1, 2], y=[11, 12])
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results[0].output.out == 13
    assert results[1].output.out == 14
    assert results[2].output.out == 24
    assert results[3].output.out == 26
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_5(plugin, tmpdir):
    """workflow with two tasks, outer splitter on tasks level and no combiner"""
    wf = Workflow(name="wf_ndst_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out[0] == 13
    assert results.output.out[1] == 14
    assert results.output.out[2] == 24
    assert results.output.out[3] == 26
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_st_6(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner for the workflow"""
    wf = Workflow(name="wf_st_6", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12])
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results[0][0].output.out == 13
    assert results[0][1].output.out == 24
    assert results[0][2].output.out == 35
    assert results[1][0].output.out == 14
    assert results[1][1].output.out == 26
    assert results[1][2].output.out == 38
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_6(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner on tasks level"""
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out[0] == [13, 24, 35]
    assert results.output.out[1] == [14, 26, 38]

    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_ndst_7(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (full) combiner for first node only"""
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split("x").combine("x"))
    wf.add(identity(name="iden", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = 11
    wf.set_output([("out", wf.iden.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [11, 22, 33]

    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_ndst_8(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (partial) combiner for first task only"""
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(
        multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]).combine("x")
    )
    wf.add(identity(name="iden", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.iden.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out[0] == [11, 22, 33]
    assert results.output.out[1] == [12, 24, 36]

    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_ndst_9(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (full) combiner for first task only"""
    wf = Workflow(name="wf_ndst_6", input_spec=["x", "y"])
    wf.add(
        multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y)
        .split(["x", "y"])
        .combine(["x", "y"])
    )
    wf.add(identity(name="iden", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.iden.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [11, 12, 22, 24, 33, 36]

    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures A ->  B -> C


def test_wf_3sernd_ndst_1(plugin, tmpdir):
    """workflow with three "serial" tasks, checking if the splitter is propagating"""
    wf = Workflow(name="wf_3sernd_ndst_1", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2(name="add2_1st", x=wf.mult.lzout.out))
    wf.add(add2(name="add2_2nd", x=wf.add2_1st.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2_2nd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # splitter from the first task should propagate to all tasks,
    # splitter_rpn should be the same in all tasks
    assert wf.mult.state.splitter == ["mult.x", "mult.y"]
    assert wf.add2_1st.state.splitter == "_mult"
    assert wf.add2_2nd.state.splitter == "_add2_1st"
    assert (
        ["mult.x", "mult.y", "*"]
        == wf.mult.state.splitter_rpn
        == wf.add2_1st.state.splitter_rpn
        == wf.add2_2nd.state.splitter_rpn
    )

    results = wf.result()
    assert results.output.out[0] == 15
    assert results.output.out[1] == 16
    assert results.output.out[2] == 26
    assert results.output.out[3] == 28
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3sernd_ndst_1a(plugin, tmpdir):
    """
    workflow with three "serial" tasks, checking if the splitter is propagating
    first task has a splitter that propagates to the 2nd task,
    and the 2nd task is adding one more input to the splitter
    """
    wf = Workflow(name="wf_3sernd_ndst_1", input_spec=["x", "y"])
    wf.add(add2(name="add2_1st", x=wf.lzin.x).split("x"))
    wf.add(multiply(name="mult", x=wf.add2_1st.lzout.out, y=wf.lzin.y).split("y"))
    wf.add(add2(name="add2_2nd", x=wf.mult.lzout.out))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.add2_2nd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    # splitter from the 1st task should propagate and the 2nd task should add one more
    # splitter_rpn for the 2nd and the 3rd task should be the same
    assert wf.add2_1st.state.splitter == "add2_1st.x"
    assert wf.mult.state.splitter == ["_add2_1st", "mult.y"]
    assert wf.add2_2nd.state.splitter == "_mult"
    assert (
        ["add2_1st.x", "mult.y", "*"]
        == wf.mult.state.splitter_rpn
        == wf.add2_2nd.state.splitter_rpn
    )

    results = wf.result()
    assert results.output.out[0] == 35
    assert results.output.out[1] == 38
    assert results.output.out[2] == 46
    assert results.output.out[3] == 50
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures A -> C, B -> C


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_st_1(plugin_dask_opt, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    """
    wf = Workflow(name="wf_st_7", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12])

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
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


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_ndst_1(plugin_dask_opt, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the tasks levels
    """
    wf = Workflow(name="wf_ndst_7", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.inputs.x = [1, 2, 3]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 6
    assert results.output.out == [39, 42, 52, 56, 65, 70]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3nd_st_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the workflow level
    """
    wf = Workflow(name="wf_st_8", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_3nd_ndst_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the tasks levels
    """
    wf = Workflow(name="wf_ndst_8", input_spec=["x", "y"])
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 2
    assert results.output.out[0] == [39, 52, 65]
    assert results.output.out[1] == [42, 56, 70]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3nd_st_3(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the workflow level
    """
    wf = Workflow(name="wf_st_9", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("y")

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_3nd_ndst_3(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the tasks levels
    """
    wf = Workflow(name="wf_ndst_9", input_spec=["x", "y"])
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 3
    assert results.output.out[0] == [39, 42]
    assert results.output.out[1] == [52, 56]
    assert results.output.out[2] == [65, 70]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3nd_st_4(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the workflow level
    """
    wf = Workflow(name="wf_st_10", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine(["x", "y"])
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 6
    assert results[0].output.out == 39
    assert results[1].output.out == 42
    assert results[2].output.out == 52
    assert results[3].output.out == 56
    assert results[4].output.out == 65
    assert results[5].output.out == 70
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_3nd_ndst_4(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the tasks levels
    """
    wf = Workflow(name="wf_ndst_10", input_spec=["x", "y"])
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()

    assert len(results.output.out) == 6
    assert results.output.out == [39, 42, 52, 56, 65, 70]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3nd_st_5(plugin, tmpdir):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    splitter and partial combiner (from the second task) on the workflow level
    """
    wf = Workflow(name="wf_st_9", input_spec=["x", "y", "z"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(
        fun_addvar3(
            name="addvar", a=wf.add2x.lzout.out, b=wf.add2y.lzout.out, c=wf.lzin.z
        )
    )
    wf.split(["x", "y", "z"], x=[2, 3], y=[11, 12], z=[10, 100]).combine("y")

    wf.set_output([("out", wf.addvar.lzout.out)])
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results) == 4
    assert results[0][0].output.out == 27
    assert results[0][1].output.out == 28
    assert results[1][0].output.out == 117
    assert results[1][1].output.out == 118
    assert results[2][0].output.out == 28
    assert results[2][1].output.out == 29
    assert results[3][0].output.out == 118
    assert results[3][1].output.out == 119

    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_3nd_ndst_5(plugin, tmpdir):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    all tasks have splitters and the last one has a partial combiner (from the 2nd)
    """
    wf = Workflow(name="wf_st_9", input_spec=["x", "y", "z"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(
        fun_addvar3(
            name="addvar", a=wf.add2x.lzout.out, b=wf.add2y.lzout.out, c=wf.lzin.z
        )
        .split("c")
        .combine("add2x.x")
    )
    wf.inputs.x = [2, 3]
    wf.inputs.y = [11, 12]
    wf.inputs.z = [10, 100]

    wf.set_output([("out", wf.addvar.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 4
    assert results.output.out[0] == [27, 28]
    assert results.output.out[1] == [117, 118]
    assert results.output.out[2] == [28, 29]
    assert results.output.out[3] == [118, 119]

    # checking all directories
    assert wf.output_dir.exists()


def test_wf_3nd_ndst_6(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones and a combiner
    """
    wf = Workflow(name="wf_ndst_9", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.y).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out)
        .split(("_add2x", "_add2y"))
        .combine("add2y.x")
    )
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [39, 56]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_3nd_ndst_7(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones
    """
    wf = Workflow(name="wf_ndst_9", input_spec=["x"])
    wf.add(add2(name="add2x", x=wf.lzin.x).split("x"))
    wf.add(add2(name="add2y", x=wf.lzin.x).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out).split(
            ("_add2x", "_add2y")
        )
    )
    wf.inputs.x = [1, 2]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [9, 16]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures A -> B -> C with multiple connections


def test_wf_3nd_8(tmpdir):
    """workflow with three tasks A->B->C vs two tasks A->C with multiple connections"""
    wf = Workflow(name="wf", input_spec=["zip"], cache_dir=tmpdir)
    wf.inputs.zip = [["test1", "test3", "test5"], ["test2", "test4", "test6"]]

    wf.add(identity_2flds(name="iden2flds_1", x1=wf.lzin.zip, x2="Hoi").split("x1"))

    wf.add(identity(name="identity", x=wf.iden2flds_1.lzout.out1))

    wf.add(
        identity_2flds(
            name="iden2flds_2", x1=wf.identity.lzout.out, x2=wf.iden2flds_1.lzout.out2
        )
    )

    wf.add(
        identity_2flds(
            name="iden2flds_2a",
            x1=wf.iden2flds_1.lzout.out1,
            x2=wf.iden2flds_1.lzout.out2,
        )
    )

    wf.set_output(
        [
            ("out1", wf.iden2flds_2.lzout.out1),
            ("out2", wf.iden2flds_2.lzout.out2),
            ("out1a", wf.iden2flds_2a.lzout.out1),
            ("out2a", wf.iden2flds_2a.lzout.out2),
        ]
    )

    with Submitter(plugin="cf") as sub:
        sub(wf)

    res = wf.result()

    assert (
        res.output.out1
        == res.output.out1a
        == [["test1", "test3", "test5"], ["test2", "test4", "test6"]]
    )
    assert res.output.out2 == res.output.out2a == ["Hoi", "Hoi"]


# workflows with Left and Right part in splitters A -> B (L&R parts of the splitter)


def test_wf_ndstLR_1(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has its own simple splitter
    and the  Left part from the first task should be added
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(multiply(name="mult", x=wf.add2.lzout.out, y=wf.lzin.y).split("y"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_ndstLR_1a(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own splitter)
    """
    wf = Workflow(name="wf_ndst_3", input_spec=["x", "y"])
    wf.add(add2(name="add2", x=wf.lzin.x).split("x"))
    wf.add(
        multiply(name="mult", x=wf.add2.lzout.out, y=wf.lzin.y).split(["_add2", "y"])
    )
    wf.inputs.x = [1, 2]
    wf.inputs.y = [11, 12]
    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_ndstLR_2(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
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
    wf.cache_dir = tmpdir

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


def test_wf_ndstLR_2a(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own outer splitter)
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
    wf.cache_dir = tmpdir

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


def test_wf_ndstinner_1(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has inner splitter
    """
    wf = Workflow(name="wf_st_3", input_spec=["x"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(add2(name="add2", x=wf.list.lzout.out).split("x"))
    wf.inputs.x = 1
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.add2.state.splitter == "add2.x"
    assert wf.add2.state.splitter_rpn == ["add2.x"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [3, 4, 5]

    assert wf.output_dir.exists()


def test_wf_ndstinner_2(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has two inputs and inner splitter from one of the input
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split("x"))
    wf.inputs.x = 1
    wf.inputs.y = 10
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == "mult.x"
    assert wf.mult.state.splitter_rpn == ["mult.x"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [10, 20, 30]

    assert wf.output_dir.exists()


def test_wf_ndstinner_3(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has two inputs and outer splitter that includes an inner field
    """
    wf = Workflow(name="wf_st_3", input_spec=["x", "y"])
    wf.add(list_output(name="list", x=wf.lzin.x))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split(["x", "y"]))
    wf.inputs.x = 1
    wf.inputs.y = [10, 100]
    wf.set_output([("out_list", wf.list.lzout.out), ("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == ["mult.x", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["mult.x", "mult.y", "*"]

    results = wf.result()
    assert results.output.out_list == [1, 2, 3]
    assert results.output.out == [10, 100, 20, 200, 30, 300]

    assert wf.output_dir.exists()


def test_wf_ndstinner_4(plugin, tmpdir):
    """workflow with 3 tasks,
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
    wf.cache_dir = tmpdir

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


def test_wf_ndstinner_5(plugin, tmpdir):
    """workflow with 3 tasks,
    the second task has two inputs and inner splitter from one of the input,
    (inner input come from the first task that has its own splitter,
    there is a inner_cont_dim)
    the third task has no new splitter
    """
    wf = Workflow(name="wf_5", input_spec=["x", "y", "b"])
    wf.add(list_output(name="list", x=wf.lzin.x).split("x"))
    wf.add(multiply(name="mult", x=wf.list.lzout.out, y=wf.lzin.y).split(["y", "x"]))
    wf.add(fun_addvar(name="addvar", a=wf.mult.lzout.out, b=wf.lzin.b).split("b"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = [10, 100]
    wf.inputs.b = [3, 5]

    wf.set_output(
        [
            ("out_list", wf.list.lzout.out),
            ("out_mult", wf.mult.lzout.out),
            ("out_add", wf.addvar.lzout.out),
        ]
    )
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.mult.state.splitter == ["_list", ["mult.y", "mult.x"]]
    assert wf.mult.state.splitter_rpn == ["list.x", "mult.y", "mult.x", "*", "*"]
    assert wf.addvar.state.splitter == ["_mult", "addvar.b"]
    assert wf.addvar.state.splitter_rpn == [
        "list.x",
        "mult.y",
        "mult.x",
        "*",
        "*",
        "addvar.b",
        "*",
    ]

    results = wf.result()
    assert results.output.out_list == [[1, 2, 3], [2, 4, 6]]
    assert results.output.out_mult == [
        10,
        20,
        30,
        20,
        40,
        60,
        100,
        200,
        300,
        200,
        400,
        600,
    ]
    assert results.output.out_add == [
        13,
        15,
        23,
        25,
        33,
        35,
        23,
        25,
        43,
        45,
        63,
        65,
        103,
        105,
        203,
        205,
        303,
        305,
        203,
        205,
        403,
        405,
        603,
        605,
    ]

    assert wf.output_dir.exists()


# workflow that have some single values as the input


def test_wf_st_singl_1(plugin, tmpdir):
    """workflow with two tasks, only one input is in the splitter and combiner"""
    wf = Workflow(name="wf_st_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))

    wf.split("x", x=[1, 2], y=11)
    wf.combine("x")
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results[0].output.out == 13
    assert results[1].output.out == 24
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


def test_wf_ndst_singl_1(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner on tasks level;
    only one input is part of the splitter, the other is a single value
    """
    wf = Workflow(name="wf_ndst_5", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split("x"))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))
    wf.inputs.x = [1, 2]
    wf.inputs.y = 11
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [13, 24]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wf_st_singl_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    only one input is part of the splitter, the other is a single value
    """
    wf = Workflow(name="wf_st_6", input_spec=["x", "y"])
    wf.add(add2(name="add2x", x=wf.lzin.x))
    wf.add(add2(name="add2y", x=wf.lzin.y))
    wf.add(multiply(name="mult", x=wf.add2x.lzout.out, y=wf.add2y.lzout.out))
    wf.split("x", x=[1, 2, 3], y=11)

    wf.set_output([("out", wf.mult.lzout.out)])
    wf.cache_dir = tmpdir

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


def test_wf_ndst_singl_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert len(results.output.out) == 3
    assert results.output.out == [39, 52, 65]
    # checking the output directory
    assert wf.output_dir.exists()


# workflows with structures wf(A)


def test_wfasnd_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.inputs.x = 2

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == 4
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfinp_1(plugin, tmpdir):
    """workflow as a node
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
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert results.output.out == 4
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfndupdate(plugin, tmpdir):
    """workflow as a node
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == 5
    assert wf.output_dir.exists()


def test_wfasnd_wfndupdate_rerun(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    wfasnode is run first and later is
    updated to use the main workflow input
    """

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=2)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wfnd.cache_dir = tmpdir
    with Submitter(plugin=plugin) as sub:
        sub(wfnd)

    wf = Workflow(name="wf", input_spec=["x"], x=3)
    # trying to set before
    wfnd.inputs.x = wf.lzin.x
    wf.add(wfnd)
    # trying to set after add...
    wf.wfnd.inputs.x = wf.lzin.x
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

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
    wf_o.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf_o)

    results = wf_o.result()
    assert results.output.out == 6
    assert wf_o.output_dir.exists()


def test_wfasnd_st_1(plugin, tmpdir):
    """workflow as a node
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
    wf.cache_dir = tmpdir

    checksum_before = wf.checksum
    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.checksum == checksum_before
    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_st_updatespl_1(plugin, tmpdir):
    """workflow as a node
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_ndst_1(plugin, tmpdir):
    """workflow as a node
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_ndst_updatespl_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for node added after add
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    # TODO: without this the test is failing
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wfnd.add2.split("x")
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [4, 6]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfst_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for the main workflow
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])

    wf.add(wfnd)
    wf.split("x")
    wf.inputs.x = [2, 4]
    wf.set_output([("out", wf.wfnd.lzout.out)])

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


def test_wfasnd_st_2(plugin, tmpdir):
    """workflow as a node,
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out == [4, 42]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfst_2(plugin, tmpdir):
    """workflow as a node,
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
    wf.cache_dir = tmpdir

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


def test_wfasnd_ndst_3(plugin, tmpdir):
    """workflow as the second node,
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
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out == [4, 42]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfst_3(plugin, tmpdir):
    """workflow as the second node,
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
    wf.cache_dir = tmpdir

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


# workflows with structures wfns(A->B)


def test_wfasnd_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks and no splitter
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2_1st", x=wfnd.lzin.x))
    wfnd.add(add2(name="add2_2nd", x=wfnd.add2_1st.lzout.out))
    wfnd.set_output([("out", wfnd.add2_2nd.lzout.out)])
    wfnd.inputs.x = 2

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == 6
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_ndst_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks,
    splitter for node
    """
    wfnd = Workflow(name="wfnd", input_spec=["x"])
    wfnd.add(add2(name="add2_1st", x=wfnd.lzin.x).split("x"))
    wfnd.add(add2(name="add2_2nd", x=wfnd.add2_1st.lzout.out))
    wfnd.set_output([("out", wfnd.add2_2nd.lzout.out)])
    wfnd.inputs.x = [2, 4]

    wf = Workflow(name="wf", input_spec=["x"])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    assert results.output.out == [6, 8]
    # checking the output directory
    assert wf.output_dir.exists()


def test_wfasnd_wfst_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks,
    splitter for the main workflow
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2_1st", x=wfnd.lzin.x))
    wfnd.add(add2(name="add2_2nd", x=wfnd.add2_1st.lzout.out))
    wfnd.set_output([("out", wfnd.add2_2nd.lzout.out)])

    wf.add(wfnd)
    wf.split("x")
    wf.inputs.x = [2, 4]
    wf.set_output([("out", wf.wfnd.lzout.out)])

    with Submitter(plugin=plugin) as sub:
        sub(wf)
    # assert wf.output_dir.exists()
    results = wf.result()
    assert results[0].output.out == 6
    assert results[1].output.out == 8
    # checking all directories
    assert wf.output_dir
    for odir in wf.output_dir:
        assert odir.exists()


# Testing caching


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir(plugin, tmpdir):
    """wf with provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_wf_cache_1")

    wf = Workflow(name="wf_2", input_spec=["x", "y"], cache_dir=cache_dir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir_relativepath(tmpdir, plugin):
    """wf with provided cache_dir as relative path"""
    tmpdir.chdir()
    cache_dir = "test_wf_cache_2"
    tmpdir.mkdir(cache_dir)

    wf = Workflow(name="wf_2", input_spec=["x", "y"], cache_dir=cache_dir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2(name="add2", x=wf.mult.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.inputs.x = 2
    wf.inputs.y = 3

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=3)
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

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out

    # checking execution time (for unix and cf)
    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking if the second wf didn't run again
    assert wf1.output_dir.exists()
    assert not wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_a(plugin, tmpdir):
    """
    the same as previous test, but workflows names differ;
    the task should not be run and it should be fast,
    but the wf itself is triggered and the new output dir is created
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf1", input_spec=["x", "y"], cache_dir=cache_dir1)
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
        name="wf2",
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time (second one should be quick)
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # checking if both wf.output_dir are created
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_b(plugin, tmpdir):
    """
    the same as previous test, but the 2nd workflows has two outputs
    (connected to the same task output);
    the task should not be run and it should be fast,
    but the wf itself is triggered and the new output dir is created
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
    # additional output
    wf2.set_output([("out_pr", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out == results2.output.out_pr

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # execution time for second run should be much shorter
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking if the second wf didn't run again
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_setoutputchange(plugin, tmpdir):
    """
    the same as previous test, but wf output names differ,
    the tasks should not be run and it should be fast,
    but the wf itself is triggered and the new output dir is created
    (the second wf has updated name in its Output)
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out1", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out1

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out2", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time (the second wf should be fast, nodes do not have to rerun)
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # both wf output_dirs should be created
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_setoutputchange_a(plugin, tmpdir):
    """
    the same as previous test, but wf names and output names differ,
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf1", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2_wait(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out1", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out1

    wf2 = Workflow(
        name="wf2",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out2", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf2)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # both wf output_dirs should be created
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_forcererun(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir;
    the second wf has cache_locations,
    but submitter is called with rerun=True, so should recompute
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
        sub(wf2, rerun=True)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert 8 == results2.output.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time
        assert t1 > 2
        assert t2 > 2

    # checking if the second wf didn't run again
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_wftaskrerun_propagateTrue(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir and cache_locations for the second one;
    submitter doesn't have rerun, but the second wf has rerun=True,
    propagate_rerun is True as default, so everything should be rerun
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
        rerun=True,  # wh has to be rerun (default for propagate_rerun is True)
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

    # checking if the second wf runs again
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()

    # everything has to be recomputed
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # runtime for recomputed workflows should be about the same
        assert abs(t1 - t2) < t1 / 2


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_wftaskrerun_propagateFalse(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir and cache_locations for the second one;
    submitter doesn't have rerun, but the second wf has rerun=True,
    propagate_rerun is set to False, so wf will be triggered,
    but tasks will not have rerun, so will use the previous results
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
        rerun=True,  # wh has to be rerun
        propagate_rerun=False,  # but rerun doesn't propagate to the tasks
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

    # checking if the second wf runs again
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # tasks should not be recomputed
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 0


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_taskrerun_wfrerun_propagateFalse(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir, and cache_locations for the second wf;
    submitter doesn't have rerun, but wf has rerun=True,
    since propagate_rerun=False, only tasks that have rerun=True will be rerun
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
        rerun=True,
        propagate_rerun=False,  # rerun will not be propagated to each task
    )
    wf2.add(multiply(name="mult", x=wf2.lzin.x, y=wf2.lzin.y))
    # rerun on the task level needed (wf.propagate_rerun is False)
    wf2.add(add2_wait(name="add2", x=wf2.mult.lzout.out, rerun=True))
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

    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()
    # the second task should be recomputed
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 1

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_nodecachelocations(plugin, tmpdir):
    """
    Two wfs with different input, but the second node has the same input;
    the second wf has cache_locations and should recompute the wf,
    but without recomputing the second node
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x"], cache_dir=cache_dir1)
    wf1.add(ten(name="ten", x=wf1.lzin.x))
    wf1.add(add2(name="add2", x=wf1.ten.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 3
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 12 == results1.output.out

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    wf2.add(ten(name="ten", x=wf2.lzin.x))
    wf2.add(add2(name="add2", x=wf2.ten.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf2)

    results2 = wf2.result()
    assert 12 == results2.output.out

    # checking if the second wf runs again, but runs only one task
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()
    # the second wf should rerun one task
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 1


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_nodecachelocations_upd(plugin, tmpdir):
    """
    Two wfs with different input, but the second node has the same input;
    the second wf has cache_locations (set after adding tasks) and should recompute,
    but without recomputing the second node
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x"], cache_dir=cache_dir1)
    wf1.add(ten(name="ten", x=wf1.lzin.x))
    wf1.add(add2(name="add2", x=wf1.ten.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 3
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 12 == results1.output.out

    wf2 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir2)
    wf2.add(ten(name="ten", x=wf2.lzin.x))
    wf2.add(add2(name="add2", x=wf2.ten.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.plugin = plugin
    # updating cache_locations after adding the tasks
    wf2.cache_locations = cache_dir1

    with Submitter(plugin=plugin) as sub:
        sub(wf2)

    results2 = wf2.result()
    assert 12 == results2.output.out

    # checking if the second wf runs again, but runs only one task
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()
    # the second wf should have only one task run
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 1


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories
    assert wf1.output_dir
    for odir in wf1.output_dir:
        assert odir.exists()
    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir
    for odir in wf2.output_dir:
        assert not odir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_state_cachelocations_forcererun(plugin, tmpdir):
    """
    Two identical wfs (with states) with provided cache_dir;
    the second wf has cache_locations,
    but submitter is called with rerun=True, so should recompute
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
        sub(wf2, rerun=True)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2[0].output.out == 8
    assert results2[1].output.out == 82

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories
    assert wf1.output_dir
    for odir in wf1.output_dir:
        assert odir.exists()
    # checking if the second wf run again
    # checking all directories
    assert wf2.output_dir
    for odir in wf2.output_dir:
        assert odir.exists()


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories
    assert wf1.output_dir
    for odir in wf1.output_dir:
        assert odir.exists()
    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir
    for odir in wf2.output_dir:
        assert not odir.exists()


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking if both wf run
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_recompute(plugin, tmpdir):
    """
    Two wfs with the same inputs but slightly different graph;
    the second wf should recompute the results,
    but the second node should use the results from the first wf (has the same input)
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    wf1 = Workflow(name="wf", input_spec=["x", "y"], cache_dir=cache_dir1)
    wf1.add(multiply(name="mult", x=wf1.lzin.x, y=wf1.lzin.y))
    wf1.add(add2(name="add2", x=wf1.mult.lzout.out))
    wf1.set_output([("out", wf1.add2.lzout.out)])
    wf1.inputs.x = 2
    wf1.inputs.y = 3
    wf1.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf1)

    results1 = wf1.result()
    assert 8 == results1.output.out

    wf2 = Workflow(
        name="wf",
        input_spec=["x", "y"],
        cache_dir=cache_dir2,
        cache_locations=cache_dir1,
    )
    # different argument assignment
    wf2.add(multiply(name="mult", x=wf2.lzin.y, y=wf2.lzin.x))
    wf2.add(add2(name="add2", x=wf2.mult.lzout.out))
    wf2.set_output([("out", wf2.add2.lzout.out)])
    wf2.inputs.x = 2
    wf2.inputs.y = 3
    wf2.plugin = plugin

    with Submitter(plugin=plugin) as sub:
        sub(wf2)

    results2 = wf2.result()
    assert 8 == results2.output.out

    # checking if both dir exists
    assert wf1.output_dir.exists()
    assert wf2.output_dir.exists()

    # the second wf should have only one task run
    assert len(list(Path(cache_dir1).glob("F*"))) == 2
    assert len(list(Path(cache_dir2).glob("F*"))) == 1


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert not wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_ndstate_cachelocations_forcererun(plugin, tmpdir):
    """
    Two wfs with identical inputs and node states;
    the second wf has cache_locations,
    but submitter is called with rerun=True, so should recompute
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
        sub(wf2, rerun=True)
    t2 = time.time() - t0

    results2 = wf2.result()
    assert results2.output.out == [8, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf run again
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert not wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
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

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories
    assert wf1.output_dir.exists()

    # checking if the second wf didn't run again
    # checking all directories
    assert wf2.output_dir.exists()


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_runtwice_usecache(plugin, tmpdir):
    """
    running workflow (without state) twice,
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

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out
    # checkoing output_dir after the first run
    assert wf1.output_dir.exists()

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t2 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1.output.out
    # checking if no new directory is created
    assert cache_dir_content == os.listdir(wf1.cache_dir)

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


def test_wf_state_runtwice_usecache(plugin, tmpdir):
    """
    running workflow with a state twice,
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

    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t1 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1[0].output.out
    assert 602 == results1[1].output.out

    # checkoing output_dir after the first run
    assert [odir.exists() for odir in wf1.output_dir]

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    t0 = time.time()
    with Submitter(plugin=plugin) as sub:
        sub(wf1)
    t2 = time.time() - t0

    results1 = wf1.result()
    assert 8 == results1[0].output.out
    assert 602 == results1[1].output.out
    # checking if no new directory is created
    assert cache_dir_content == os.listdir(wf1.cache_dir)
    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


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
    result = wf1()

    assert result.output.out_pow == [1, 1, 4, 8]
    assert result.output.out_iden1 == [[1, 4], [1, 8]]
    assert result.output.out_iden2 == [[1, 4], [1, 8]]


def test_workflow_combine2(tmpdir):
    wf1 = Workflow(name="wf1", input_spec=["a", "b"], a=[1, 2], b=[2, 3])
    wf1.add(
        power(name="power", a=wf1.lzin.a, b=wf1.lzin.b).split(["a", "b"]).combine("a")
    )
    wf1.add(identity(name="identity", x=wf1.power.lzout.out).combine("power.b"))
    wf1.set_output({"out_pow": wf1.power.lzout.out, "out_iden": wf1.identity.lzout.out})
    wf1.cache_dir = tmpdir
    result = wf1()

    assert result.output.out_pow == [[1, 4], [1, 8]]
    assert result.output.out_iden == [[1, 4], [1, 8]]


# testing lzout.all to collect all of the results and let FunctionTask deal with it


def test_wf_lzoutall_1(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_sub2_res function
    by using lzout.all syntax
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out", wf.add_sub.lzout.out_add)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert 8 == results.output.out


def test_wf_lzoutall_1a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    by using lzout.all syntax in the node connections and for wf output
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out_all", wf.add_sub.lzout.all_)])
    wf.inputs.x = 2
    wf.inputs.y = 3
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out_all == {"out_add": 8, "out_sub": 4}


def test_wf_lzoutall_st_1(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    by using lzout.all syntax
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out_add", wf.add_sub.lzout.out_add)])
    wf.inputs.x = [2, 20]
    wf.inputs.y = [3, 30]
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out_add == [8, 62, 62, 602]


def test_wf_lzoutall_st_1a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    by using lzout.all syntax
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]))
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out_all", wf.add_sub.lzout.all_)])
    wf.inputs.x = [2, 20]
    wf.inputs.y = [3, 30]
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out_all == [
        {"out_add": 8, "out_sub": 4},
        {"out_add": 62, "out_sub": 58},
        {"out_add": 62, "out_sub": 58},
        {"out_add": 602, "out_sub": 598},
    ]


def test_wf_lzoutall_st_2(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    by using lzout.all syntax
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(
        multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]).combine("x")
    )
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out_add", wf.add_sub.lzout.out_add)])
    wf.inputs.x = [2, 20]
    wf.inputs.y = [3, 30]
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out_add[0] == [8, 62]
    assert results.output.out_add[1] == [62, 602]


def test_wf_lzoutall_st_2a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    by using lzout.all syntax
    """
    wf = Workflow(name="wf_2", input_spec=["x", "y"])
    wf.add(
        multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split(["x", "y"]).combine("x")
    )
    wf.add(add2_sub2_res(name="add_sub", res=wf.mult.lzout.all_))
    wf.set_output([("out_all", wf.add_sub.lzout.all_)])
    wf.inputs.x = [2, 20]
    wf.inputs.y = [3, 30]
    wf.plugin = plugin
    wf.cache_dir = tmpdir

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    assert wf.output_dir.exists()
    results = wf.result()
    assert results.output.out_all == [
        {"out_add": [8, 62], "out_sub": [4, 58]},
        {"out_add": [62, 602], "out_sub": [58, 598]},
    ]


# workflows that have files in the result, the files should be copied to the wf dir


def test_wf_resultfile_1(plugin, tmpdir):
    """workflow with a file in the result, file should be copied to the wf dir"""
    wf = Workflow(name="wf_file_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_write_file(name="writefile", filename=wf.lzin.x))
    wf.inputs.x = "file_1.txt"
    wf.plugin = plugin
    wf.set_output([("wf_out", wf.writefile.lzout.out)])

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking if the file exists and if it is in the Workflow directory
    assert results.output.wf_out.exists()
    assert results.output.wf_out == wf.output_dir / "file_1.txt"


def test_wf_resultfile_2(plugin, tmpdir):
    """workflow with a list of files in the wf result,
    all files should be copied to the wf dir
    """
    wf = Workflow(name="wf_file_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_write_file_list(name="writefile", filename_list=wf.lzin.x))
    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    wf.inputs.x = file_list
    wf.plugin = plugin
    wf.set_output([("wf_out", wf.writefile.lzout.out)])

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking if the file exists and if it is in the Workflow directory
    for ii, file in enumerate(results.output.wf_out):
        assert file.exists()
        assert file == wf.output_dir / file_list[ii]


def test_wf_resultfile_3(plugin, tmpdir):
    """workflow with a dictionaries of files in the wf result,
    all files should be copied to the wf dir
    """
    wf = Workflow(name="wf_file_1", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_write_file_list2dict(name="writefile", filename_list=wf.lzin.x))
    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    wf.inputs.x = file_list
    wf.plugin = plugin
    wf.set_output([("wf_out", wf.writefile.lzout.out)])

    with Submitter(plugin=plugin) as sub:
        sub(wf)

    results = wf.result()
    # checking if the file exists and if it is in the Workflow directory
    for key, val in results.output.wf_out.items():
        if key == "random_int":
            assert val == 20
        else:
            assert val.exists()
            ii = int(key.split("_")[1])
            assert val == wf.output_dir / file_list[ii]


def test_wf_upstream_error1(plugin, tmpdir):
    """workflow with two tasks, task2 dependent on an task1 which raised an error"""
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.set_output([("out", wf.addvar2.lzout.out)])

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error2(plugin, tmpdir):
    """task2 dependent on task1, task1 errors, workflow-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = [1, "hi"]  # TypeError for adding str and int
    wf.split("x")  # workflow-level split
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.set_output([("out", wf.addvar2.lzout.out)])

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


@pytest.mark.flaky(reruns=2)  # when slurm
def test_wf_upstream_error3(plugin, tmpdir):
    """task2 dependent on task1, task1 errors, task-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = [1, "hi"]  # TypeError for adding str and int
    wf.addvar1.split("a")  # task-level split
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.set_output([("out", wf.addvar2.lzout.out)])

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error4(plugin, tmpdir):
    """workflow with one task, which raises an error"""
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.set_output([("out", wf.addvar1.lzout.out)])

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "raised an error" in str(excinfo.value)
    assert "addvar1" in str(excinfo.value)


def test_wf_upstream_error5(plugin, tmpdir):
    """nested workflow with one task, which raises an error"""
    wf_main = Workflow(name="wf_main", input_spec=["x"], cache_dir=tmpdir)
    wf = Workflow(name="wf", input_spec=["x"], x=wf_main.lzin.x)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.plugin = plugin
    wf.set_output([("wf_out", wf.addvar1.lzout.out)])

    wf_main.add(wf)
    wf_main.inputs.x = "hi"  # TypeError for adding str and int
    wf_main.set_output([("out", wf_main.wf.lzout.wf_out)])

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf_main)

    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error6(plugin, tmpdir):
    """nested workflow with two tasks, the first one raises an error"""
    wf_main = Workflow(name="wf_main", input_spec=["x"], cache_dir=tmpdir)
    wf = Workflow(name="wf", input_spec=["x"], x=wf_main.lzin.x)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.plugin = plugin
    wf.set_output([("wf_out", wf.addvar2.lzout.out)])

    wf_main.add(wf)
    wf_main.inputs.x = "hi"  # TypeError for adding str and int
    wf_main.set_output([("out", wf_main.wf.lzout.wf_out)])

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf_main)

    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error7(plugin, tmpdir):
    """
    workflow with three sequential tasks, the first task raises an error
    the last task is set as the workflow output
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar3", a=wf.addvar2.lzout.out))
    wf.set_output([("out", wf.addvar3.lzout.out)])

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.addvar1._errored is True
    assert wf.addvar2._errored == wf.addvar3._errored == ["addvar1"]


def test_wf_upstream_error7a(plugin, tmpdir):
    """
    workflow with three sequential tasks, the first task raises an error
    the second task is set as the workflow output
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar3", a=wf.addvar2.lzout.out))
    wf.set_output([("out", wf.addvar2.lzout.out)])

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.addvar1._errored is True
    assert wf.addvar2._errored == wf.addvar3._errored == ["addvar1"]


def test_wf_upstream_error7b(plugin, tmpdir):
    """
    workflow with three sequential tasks, the first task raises an error
    the second and the third tasks are set as the workflow output
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar3", a=wf.addvar2.lzout.out))
    wf.set_output([("out1", wf.addvar2.lzout.out), ("out2", wf.addvar3.lzout.out)])

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.addvar1._errored is True
    assert wf.addvar2._errored == wf.addvar3._errored == ["addvar1"]


def test_wf_upstream_error8(plugin, tmpdir):
    """workflow with three tasks, the first one raises an error, so 2 others are removed"""
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = "hi"  # TypeError for adding str and int
    wf.plugin = plugin
    wf.add(fun_addvar_default(name="addvar2", a=wf.addvar1.lzout.out))
    wf.add(fun_addtwo(name="addtwo", a=wf.addvar1.lzout.out))
    wf.set_output([("out1", wf.addvar2.lzout.out), ("out2", wf.addtwo.lzout.out)])

    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)

    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.addvar1._errored is True
    assert wf.addvar2._errored == wf.addtwo._errored == ["addvar1"]


def test_wf_upstream_error9(plugin, tmpdir):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    the errored branch is connected to the workflow output
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = 2
    wf.add(fun_addvar(name="err", a=wf.addvar1.lzout.out, b="hi"))
    wf.add(fun_addvar_default(name="follow_err", a=wf.err.lzout.out))

    wf.add(fun_addtwo(name="addtwo", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar2", a=wf.addtwo.lzout.out))
    wf.set_output([("out1", wf.follow_err.lzout.out)])

    wf.plugin = plugin
    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "err" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.err._errored is True
    assert wf.follow_err._errored == ["err"]


def test_wf_upstream_error9a(plugin, tmpdir):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    the branch without error is connected to the workflow output
    so the workflow finished clean
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = 2
    wf.add(fun_addvar(name="err", a=wf.addvar1.lzout.out, b="hi"))
    wf.add(fun_addvar_default(name="follow_err", a=wf.err.lzout.out))

    wf.add(fun_addtwo(name="addtwo", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar2", a=wf.addtwo.lzout.out))
    wf.set_output([("out1", wf.addvar2.lzout.out)])  # , ("out2", wf.addtwo.lzout.out)])

    wf.plugin = plugin
    with Submitter(plugin=plugin) as sub:
        sub(wf)
    assert wf.err._errored is True
    assert wf.follow_err._errored == ["err"]


def test_wf_upstream_error9b(plugin, tmpdir):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    both branches are connected to the workflow output
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(fun_addvar_default(name="addvar1", a=wf.lzin.x))
    wf.inputs.x = 2
    wf.add(fun_addvar(name="err", a=wf.addvar1.lzout.out, b="hi"))
    wf.add(fun_addvar_default(name="follow_err", a=wf.err.lzout.out))

    wf.add(fun_addtwo(name="addtwo", a=wf.addvar1.lzout.out))
    wf.add(fun_addvar_default(name="addvar2", a=wf.addtwo.lzout.out))
    wf.set_output([("out1", wf.follow_err.lzout.out), ("out2", wf.addtwo.lzout.out)])

    wf.plugin = plugin
    with pytest.raises(ValueError) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(wf)
    assert "err" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.err._errored is True
    assert wf.follow_err._errored == ["err"]


def exporting_graphs(wf, name):
    """helper function to run dot to create png/pdf files from dotfiles"""
    # exporting the simple graph
    dotfile_pr, formatted_dot = wf.create_dotfile(export=True, name=name)
    assert len(formatted_dot) == 1
    assert formatted_dot[0] == dotfile_pr.with_suffix(".png")
    assert formatted_dot[0].exists()
    print("\n png of a simple graph in: ", formatted_dot[0])
    # exporting nested graph
    dotfile_pr, formatted_dot = wf.create_dotfile(
        type="nested", export=["pdf", "png"], name=f"{name}_nest"
    )
    assert len(formatted_dot) == 2
    assert formatted_dot[0] == dotfile_pr.with_suffix(".pdf")
    assert formatted_dot[0].exists()
    print("\n pdf of the nested graph in: ", formatted_dot[0])
    # detailed graph
    dotfile_pr, formatted_dot = wf.create_dotfile(
        type="detailed", export="pdf", name=f"{name}_det"
    )
    assert len(formatted_dot) == 1
    assert formatted_dot[0] == dotfile_pr.with_suffix(".pdf")
    assert formatted_dot[0].exists()
    print("\n pdf of the detailed graph in: ", formatted_dot[0])


@pytest.mark.parametrize("splitter", [None, "x"])
def test_graph_1(tmpdir, splitter):
    """creating a set of graphs, wf with two nodes"""
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult_1", x=wf.lzin.x, y=wf.lzin.y))
    wf.add(multiply(name="mult_2", x=wf.lzin.x, y=wf.lzin.x))
    wf.add(add2(name="add2", x=wf.mult_1.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])
    wf.split(splitter)

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult_1" in dotstr_s_lines
    assert "mult_2" in dotstr_s_lines
    assert "add2" in dotstr_s_lines
    assert "mult_1 -> add2" in dotstr_s_lines

    # nested graph (should have the same elements)
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult_1" in dotstr_n_lines
    assert "mult_2" in dotstr_n_lines
    assert "add2" in dotstr_n_lines
    assert "mult_1 -> add2" in dotstr_n_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult_1:out -> struct_add2:x;" in dotstr_d_lines

    # exporting graphs if dot available
    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_1st(tmpdir):
    """creating a set of graphs, wf with two nodes
    some nodes have splitters, should be marked with blue color
    """
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult_1", x=wf.lzin.x, y=wf.lzin.y).split("x"))
    wf.add(multiply(name="mult_2", x=wf.lzin.x, y=wf.lzin.x))
    wf.add(add2(name="add2", x=wf.mult_1.lzout.out))
    wf.set_output([("out", wf.add2.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult_1 [color=blue]" in dotstr_s_lines
    assert "mult_2" in dotstr_s_lines
    assert "add2 [color=blue]" in dotstr_s_lines
    assert "mult_1 -> add2 [color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult_1 [color=blue]" in dotstr_n_lines
    assert "mult_2" in dotstr_n_lines
    assert "add2 [color=blue]" in dotstr_n_lines
    assert "mult_1 -> add2 [color=blue]" in dotstr_n_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult_1:out -> struct_add2:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_1st_cmb(tmpdir):
    """creating a set of graphs, wf with three nodes
    the first one has a splitter, the second has a combiner, so the third one is stateless
    first two nodes should be blue and the arrow between them should be blue
    """
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split("x"))
    wf.add(add2(name="add2", x=wf.mult.lzout.out).combine("mult.x"))
    wf.add(list_sum(name="sum", x=wf.add2.lzout.out))
    wf.set_output([("out", wf.sum.lzout.out)])
    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_s_lines
    assert "add2 [color=blue]" in dotstr_s_lines
    assert "sum" in dotstr_s_lines
    assert "mult -> add2 [color=blue]" in dotstr_s_lines
    assert "add2 -> sum" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_n_lines
    assert "add2 [color=blue]" in dotstr_n_lines
    assert "sum" in dotstr_n_lines
    assert "mult -> add2 [color=blue]" in dotstr_n_lines
    assert "add2 -> sum" in dotstr_n_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_add2:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_2(tmpdir):
    """creating a graph, wf with one workflow as a node"""
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "wfnd [shape=box]" in dotstr_s_lines

    # nested graph
    dotfile = wf.create_dotfile(type="nested")
    dotstr_lines = dotfile.read_text().split("\n")
    assert "subgraph cluster_wfnd {" in dotstr_lines
    assert "add2" in dotstr_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x}}"];' in dotstr_d_lines
    )

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_2st(tmpdir):
    """creating a set of graphs, wf with one workflow as a node
    the inner workflow has a state, so should be blue
    """
    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x).split("x")
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "wfnd [shape=box, color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_s = wf.create_dotfile(type="nested")
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "subgraph cluster_wfnd {" in dotstr_s_lines
    assert "color=blue" in dotstr_s_lines
    assert "add2" in dotstr_s_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x}}"];' in dotstr_d_lines
    )
    assert "struct_wfnd:out -> struct_wf_out:out;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_3(tmpdir):
    """creating a set of graphs, wf with two nodes (one node is a workflow)"""
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.mult.lzout.out)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "mult -> wfnd" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult" in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2" in dotstr_n_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_3st(tmpdir):
    """creating a set of graphs, wf with two nodes (one node is a workflow)
    the first node has a state and it should be passed to the second node
    (blue node and a wfasnd, and blue arrow from the node to the wfasnd)
    """
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y).split("x"))

    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.mult.lzout.out)
    wfnd.add(add2(name="add2", x=wfnd.lzin.x))
    wfnd.set_output([("out", wfnd.add2.lzout.out)])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_s_lines
    assert "wfnd [shape=box, color=blue]" in dotstr_s_lines
    assert "mult -> wfnd [color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2" in dotstr_n_lines

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_4(tmpdir):
    """creating a set of graphs, wf with two nodes (one node is a workflow with two nodes
    inside). Connection from the node to the inner workflow.
    """
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wf.add(multiply(name="mult", x=wf.lzin.x, y=wf.lzin.y))
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.mult.lzout.out)
    wfnd.add(add2(name="add2_a", x=wfnd.lzin.x))
    wfnd.add(add2(name="add2_b", x=wfnd.add2_a.lzout.out))
    wfnd.set_output([("out", wfnd.add2_b.lzout.out)])
    wf.add(wfnd)
    wf.set_output([("out", wf.wfnd.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "mult -> wfnd" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    for el in ["mult", "add2_a", "add2_b"]:
        assert el in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2_a -> add2_b" in dotstr_n_lines
    assert "mult -> add2_a [lhead=cluster_wfnd]"

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_wf:y -> struct_mult:y;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


def test_graph_5(tmpdir):
    """creating a set of graphs, wf with two nodes (one node is a workflow with two nodes
    inside). Connection from the inner workflow to the node.
    """
    wf = Workflow(name="wf", input_spec=["x", "y"], cache_dir=tmpdir)
    wfnd = Workflow(name="wfnd", input_spec=["x"], x=wf.lzin.x)
    wfnd.add(add2(name="add2_a", x=wfnd.lzin.x))
    wfnd.add(add2(name="add2_b", x=wfnd.add2_a.lzout.out))
    wfnd.set_output([("out", wfnd.add2_b.lzout.out)])
    wf.add(wfnd)
    wf.add(multiply(name="mult", x=wf.wfnd.lzout.out, y=wf.lzin.y))
    wf.set_output([("out", wf.mult.lzout.out)])

    # simple graph
    dotfile_s = wf.create_dotfile()
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "wfnd -> mult" in dotstr_s_lines

    # nested graph
    dotfile_n = wf.create_dotfile(type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    for el in ["mult", "add2_a", "add2_b"]:
        assert el in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2_a -> add2_b" in dotstr_n_lines
    assert "add2_b -> mult [ltail=cluster_wfnd]"

    # detailed graph
    dotfile_d = wf.create_dotfile(type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_wf [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_wf:x -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(wf=wf, name=name)


@pytest.mark.timeout(20)
def test_duplicate_input_on_split_wf(tmpdir):
    """checking if the workflow gets stuck if it has to run two tasks with equal checksum;
    This can occur when splitting on a list containing duplicate values.
    """
    text = ["test"] * 2

    @mark.task
    def printer(a):
        return a

    wf = Workflow(name="wf", input_spec=["text"], cache_dir=tmpdir)
    wf.split(("text"), text=text)

    wf.add(printer(name="printer1", a=wf.lzin.text))

    wf.set_output([("out1", wf.printer1.lzout.out)])

    with Submitter(plugin="cf", n_procs=6) as sub:
        sub(wf)

    res = wf.result()

    assert res[0].output.out1 == "test" and res[1].output.out1 == "test"


@pytest.mark.timeout(40)
def test_inner_outer_wf_duplicate(tmpdir):
    """checking if the execution gets stuck if there is an inner and outer workflows
    thar run two nodes with the exact same inputs.
    """
    task_list = ["First", "Second"]
    start_list = [3]

    @mark.task
    def one_arg(start_number):
        for k in range(10):
            start_number += 1
        return start_number

    @mark.task
    def one_arg_inner(start_number):
        for k in range(10):
            start_number += 1
        return start_number

    # Outer workflow
    test_outer = Workflow(
        name="test_outer", input_spec=["start_number", "task_name"], cache_dir=tmpdir
    )
    # Splitting on both arguments
    test_outer.split(
        ["start_number", "task_name"], start_number=start_list, task_name=task_list
    )

    # Inner Workflow
    test_inner = Workflow(name="test_inner", input_spec=["start_number1"])
    test_inner.add(
        one_arg_inner(name="Ilevel1", start_number=test_inner.lzin.start_number1)
    )
    test_inner.set_output([("res", test_inner.Ilevel1.lzout.out)])

    # Outer workflow has two nodes plus the inner workflow
    test_outer.add(one_arg(name="level1", start_number=test_outer.lzin.start_number))
    test_outer.add(test_inner)
    test_inner.inputs.start_number1 = test_outer.level1.lzout.out

    test_outer.set_output([("res2", test_outer.test_inner.lzout.res)])

    with Submitter(plugin="cf") as sub:
        sub(test_outer)

    res = test_outer.result()
    assert res[0].output.res2 == 23 and res[1].output.res2 == 23


def test_rerun_errored(tmpdir, capfd):
    """Test rerunning a workflow containing errors.
    Only the errored tasks and workflow should be rerun"""

    @mark.task
    def pass_odds(x):
        if x % 2 == 0:
            print(f"x%2 = {x % 2} (error)\n")
            raise Exception("even error")
        else:
            print(f"x%2 = {x % 2}\n")
            return x

    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir)
    wf.add(pass_odds(name="pass_odds", x=[1, 2, 3, 4, 5]).split("x"))
    wf.set_output([("out", wf.pass_odds.lzout.out)])

    with pytest.raises(Exception):
        wf()
    with pytest.raises(Exception):
        wf()

    out, err = capfd.readouterr()
    stdout_lines = out.splitlines()

    tasks_run = 0
    errors_found = 0

    for line in stdout_lines:
        if "x%2" in line:
            tasks_run += 1
        if "(error)" in line:
            errors_found += 1

    # There should have been 5 messages of the form "x%2 = XXX" after calling task() the first time
    # and another 2 messagers after calling the second time
    assert tasks_run == 7
    assert errors_found == 4
