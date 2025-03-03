import pytest
import shutil
import os
import sys
import time
import typing as ty
import attr
from pathlib import Path
from .utils import (
    Add2,
    Add2Wait,
    Multiply,
    # MultiplyList,
    # MultiplyMixed,
    Power,
    Ten,
    Identity,
    Identity2Flds,
    ListOutput,
    FunAddSubVar,
    FunAddVar3,
    FunAddVar,
    FunAddTwo,
    Add2Sub2Res,
    Add2Sub2ResList,
    FunAddVarNone,
    FunAddVarDefault,
    FunAddVarDefaultNoType,
    FunAddVarNoType,
    FunAddTwoNoType,
    FunWriteFile,
    FunWriteFileList,
    FunWriteFileList2Dict,
    ListSum,
    ListMultSum,
    DOT_FLAG,
)
from pydra.engine.submitter import Submitter
from pydra.design import python, workflow
from pydra.utils import exc_info_matches


def test_wf_no_output(plugin, tmpdir):
    """Raise error when output isn't set with set_output"""

    @workflow.define
    def Workflow(x):
        workflow.add(Add2(x=x))

    wf = Workflow(x=2)

    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "Workflow output cannot be None" in str(excinfo.value)


def test_wf_1(plugin, tmpdir):
    """workflow with one task and no splitter"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    assert 4 == results.outputs.out


def test_wf_1a_outpastuple(plugin, tmpdir):
    """workflow with one task and no splitter
    set_output takes a tuple
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 4 == results.outputs.out


def test_wf_1_call_subm(plugin, tmpdir):
    """using wf.__call_ with submitter"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 4 == results.outputs.out


def test_wf_1_call_plug(plugin, tmpdir):
    """using wf.__call_ with plugin"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    outputs = wf(plugin=plugin)

    assert 4 == outputs.out


def test_wf_1_call_noplug_nosubm(plugin, tmpdir):
    """using wf.__call_ without plugin or submitter"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    outputs = wf()

    assert 4 == outputs.out


def test_wf_1_call_exception(plugin, tmpdir):
    """using wf.__call_ with plugin and submitter - should raise an exception"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        with pytest.raises(Exception) as e:
            wf(submitter=sub, plugin=plugin)
        assert "Defify submitter OR plugin" in str(e.value)


def test_wf_1_inp_in_call(tmpdir):
    """Defining input in __call__"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=1)
    results = wf(x=2)
    assert 4 == results.outputs.out


def test_wf_1_upd_in_run(tmpdir):
    """Updating input in __call__"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow(x=1)
    results = wf(x=2)
    assert 4 == results.outputs.out


def test_wf_2(plugin, tmpdir):
    """workflow with 2 tasks, no splitter"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out


def test_wf_2a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    creating add2_task first (before calling add method),
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out


def test_wf_2b(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    creating add2_task first (before calling add method),
    adding inputs.x after add method
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out


def test_wf_2c_multoutp(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    setting multiple outputs for the workflow
    """

    @workflow.define(outputs=["out_add2", "out_mult"])
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out, mult.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking outputs from both nodes
    assert 6 == results.outputs.out_mult
    assert 8 == results.outputs.out_add2


def test_wf_2d_outpasdict(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    setting multiple outputs using a dictionary
    """

    @workflow.define(outputs=["out_add2", "out_mult"])
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out, mult.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking outputs from both nodes
    assert 6 == results.outputs.out_mult
    assert 8 == results.outputs.out_add2


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3(plugin_dask_opt, tmpdir):
    """testing None value for an input"""

    @workflow.define
    def Workflow(x, y):
        addvar = workflow.add(FunAddVarNone(a=x, b=y))
        add2 = workflow.add(Add2(x=addvar.out))
        return add2.out

    wf = Workflow(x=2, y=None)

    with Submitter(worker=plugin_dask_opt) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 4 == results.outputs.out


@pytest.mark.xfail(reason="the task error doesn't propagate")
def test_wf_3a_exception(plugin, tmpdir):
    """testinh wf without set input, attr.NOTHING should be set
    and the function should raise an exception
    """

    @workflow.define
    def Workflow(x, y):
        addvar = workflow.add(FunAddVarNone(a=x, b=y))
        add2 = workflow.add(Add2(x=addvar.out))
        return add2.out

    wf = Workflow(x=2, y=attr.NOTHING)

    with pytest.raises(TypeError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "unsupported" in str(excinfo.value)


def test_wf_4(plugin, tmpdir):
    """wf with a task that doesn't set one input and use the function default value"""

    @workflow.define
    def Workflow(x, y):
        addvar = workflow.add(FunAddVarDefault(a=x))
        add2 = workflow.add(Add2(x=addvar.out))
        return add2.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 5 == results.outputs.out


def test_wf_4a(plugin, tmpdir):
    """wf with a task that doesn't set one input,
    the unset input is send to the task input,
    so the task should use the function default value
    """

    @workflow.define
    def Workflow(x, y):
        addvar = workflow.add(FunAddVarDefault(a=x, y=y))
        add2 = workflow.add(Add2(x=addvar.out))
        return add2.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 5 == results.outputs.out


def test_wf_5(plugin, tmpdir):
    """wf with two outputs connected to the task outputs
    one set_output
    """

    @workflow.define(outputs=["out_sum", "out_sub"])
    def Workflow(x, y):
        addsub = workflow.add(FunAddSubVar(a=x, b=y))
        return addsub.sum, addsub.sub

    wf = Workflow(x=3, y=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 5 == results.outputs.out_sum
    assert 1 == results.outputs.out_sub


def test_wf_5a(plugin, tmpdir):
    """wf with two outputs connected to the task outputs,
    set_output set twice
    """

    @workflow.define
    def Workflow(x, y):
        addsub = workflow.add(FunAddSubVar(a=x, b=y))
        return addsub.sum  # out_sum
        return addsub.sub  # out_sub

    wf = Workflow(x=3, y=2)
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 5 == results.outputs.out_sum
    assert 1 == results.outputs.out_sub


def test_wf_6(plugin, tmpdir):
    """wf with two tasks and two outputs connected to both tasks,
    one set_output
    """

    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return mult.out, add2.out  #

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 6 == results.outputs.out1
    assert 8 == results.outputs.out2


def test_wf_6a(plugin, tmpdir):
    """wf with two tasks and two outputs connected to both tasks,
    set_output used twice
    """

    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return mult.out, add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 6 == results.outputs.out1
    assert 8 == results.outputs.out2


def test_wf_st_1(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))

        return add2.out

    wf = Workflow(x=[1, 2])

    checksum_before = wf._checksum
    with Submitter(cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out[0] == 3
    assert results.outputs.out[1] == 4


def test_wf_st_1_call_subm(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))

        return add2.out

    wf = Workflow(x=[1, 2])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out[0] == 3
    assert results.outputs.out[1] == 4


def test_wf_st_1_call_plug(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__(plugin)
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))

        return add2.out

    wf = Workflow(x=[1, 2])

    outputs = wf(plugin=plugin)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_selfplug(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__() and using self.plugin
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))
        return add2.out

    wf = Workflow(x=[1, 2])

    outputs = wf()

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_noplug_nosubm(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow
    using Workflow.__call__()  without plugin and submitter
    (a submitter should be created within the __call__ function)
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))
        return add2.out

    wf = Workflow(x=[1, 2])

    outputs = wf()

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_inp_in_call(tmpdir):
    """Defining input in __call__"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow().split("x", x=[1, 2])
    results = wf()
    assert results.outputs.out[0] == 3
    assert results.outputs.out[1] == 4


def test_wf_st_1_upd_inp_call(tmpdir):
    """Updating input in __call___"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wf = Workflow().split("x", x=[11, 22])
    results = wf(x=[1, 2])
    assert results.outputs.out[0] == 3
    assert results.outputs.out[1] == 4


def test_wf_st_noinput_1(plugin, tmpdir):
    """Workflow with one task, a splitter for the workflow"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x).split("x", x=x))
        return add2.out

    wf = Workflow(x=[])

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    assert results == []


def test_wf_ndst_1(plugin, tmpdir):
    """workflow with one task, a splitter on the task level"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2().split("x", x=x))
        return add2.out

    wf = Workflow(x=[1, 2])

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out == [3, 4]


def test_wf_ndst_updatespl_1(plugin, tmpdir):
    """workflow with one task,
    a splitter on the task level is added *after* calling add
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(name="add2").split("x", x=x))
        return add2.out

    wf = Workflow(x=[1, 2])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out == [3, 4]


def test_wf_ndst_updatespl_1a(plugin, tmpdir):
    """workflow with one task (initialize before calling add),
    a splitter on the task level is added *after* calling add
    """

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2().split("x", x=x))
        return add2.out

    wf = Workflow(x=[1, 2])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out == [3, 4]


def test_wf_ndst_updateinp_1(plugin, tmpdir):
    """workflow with one task,
    a splitter on the task level,
    updating input of the task after calling add
    """

    @workflow.define
    def Workflow(x, y):
        add2 = workflow.add(Add2(x=x).split("x", x=y))
        return add2.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [13, 14]


def test_wf_ndst_noinput_1(plugin, tmpdir):
    """workflow with one task, a splitter on the task level"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2().split("x", x=x))
        return add2.out

    wf = Workflow(x=[])

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    assert results.outputs.out == []


def test_wf_st_2(plugin, tmpdir):
    """workflow with one task, splitters and combiner for workflow"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2(x=x))

        return add2.out

    wf = Workflow().split("x", x=[1, 2]).combine(combiner="x")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out[0] == 3
    assert results.outputs.out[1] == 4


def test_wf_ndst_2(plugin, tmpdir):
    """workflow with one task, splitters and combiner on the task level"""

    @workflow.define
    def Workflow(x):
        add2 = workflow.add(Add2().split("x", x=x).combine(combiner="x"))
        return add2.out

    wf = Workflow(x=[1, 2])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert results.outputs.out == [3, 4]


# workflows with structures A -> B


def test_wf_st_3(plugin, tmpdir):
    """workflow with 2 tasks, splitter on wf level"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))

        return add2.out

    wf = Workflow().split(("x", "y"), x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    expected = [
        ({"wfst_3.x": 1, "wfst_3.y": 11}, 13),
        ({"wfst_3.x": 2, "wfst_3.y": 12}, 26),
    ]
    expected_ind = [
        ({"wfst_3.x": 0, "wfst_3.y": 0}, 13),
        ({"wfst_3.x": 1, "wfst_3.y": 1}, 26),
    ]

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


def test_wf_ndst_3(plugin, tmpdir):
    """Test workflow with 2 tasks, splitter on a task level"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert results.outputs.out == [13, 26]


def test_wf_st_4(plugin, tmpdir):
    """workflow with two tasks, scalar splitter and combiner for the workflow"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))

        return add2.out

    wf = Workflow().split(("x", "y"), x=[1, 2], y=[11, 12]).combine("x")
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert results.outputs.out[0] == 13
    assert results.outputs.out[1] == 26


def test_wf_ndst_4(plugin, tmpdir):
    """workflow with two tasks, scalar splitter and combiner on tasks level"""

    @workflow.define
    def Workflow(a, b):
        mult = workflow.add(Multiply().split(("x", "y"), x=a, y=b))
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"))

        return add2.out

    wf = Workflow(a=[1, 2], b=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert results.outputs.out == [13, 26]


def test_wf_st_5(plugin, tmpdir):
    """workflow with two tasks, outer splitter and no combiner"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out).split(["x", "y"], x=x, y=y))

        return add2.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0] == 13
    assert results.outputs.out[1] == 14
    assert results.outputs.out[2] == 24
    assert results.outputs.out[3] == 26


def test_wf_ndst_5(plugin, tmpdir):
    """workflow with two tasks, outer splitter on tasks level and no combiner"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0] == 13
    assert results.outputs.out[1] == 14
    assert results.outputs.out[2] == 24
    assert results.outputs.out[3] == 26


def test_wf_st_6(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner for the workflow"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))

        return add2.out

    wf = Workflow().split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0][0] == 13
    assert results.outputs.out[0][1] == 24
    assert results.outputs.out[0][2] == 35
    assert results.outputs.out[1][0] == 14
    assert results.outputs.out[1][1] == 26
    assert results.outputs.out[1][2] == 38


def test_wf_ndst_6(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner on tasks level"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"))
        return add2.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0] == [13, 24, 35]
    assert results.outputs.out[1] == [14, 26, 38]


def test_wf_ndst_7(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (full) combiner for first node only"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split("x", x=x, y=y).combine("x"))
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    wf = Workflow(x=[1, 2, 3], y=11)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [11, 22, 33]


def test_wf_ndst_8(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (partial) combiner for first task only"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y).combine("x"))
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0] == [11, 22, 33]
    assert results.outputs.out[1] == [12, 24, 36]


def test_wf_ndst_9(plugin, tmpdir):
    """workflow with two tasks, outer splitter and (full) combiner for first task only"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y).combine(["x", "y"]))
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [11, 12, 22, 24, 33, 36]


# workflows with structures A ->  B -> C


def test_wf_3sernd_ndst_1(plugin, tmpdir):
    """workflow with three "serial" tasks, checking if the splitter is propagating"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y))
        add2_1st = workflow.add(Add2(x=mult.out), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

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

    assert results.outputs.out[0] == 15
    assert results.outputs.out[1] == 16
    assert results.outputs.out[2] == 26
    assert results.outputs.out[3] == 28


def test_wf_3sernd_ndst_1a(plugin, tmpdir):
    """
    workflow with three "serial" tasks, checking if the splitter is propagating
    first task has a splitter that propagates to the 2nd task,
    and the 2nd task is adding one more input to the splitter
    """

    @workflow.define
    def Workflow(x, y):
        add2_1st = workflow.add(Add2().split("x", x=x), name="add2_1st")
        mult = workflow.add(Multiply(x=add2_1st.out).split("y", y=y))
        add2_2nd = workflow.add(Add2(x=mult.out), name="add2_2nd")
        return add2_2nd.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

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

    assert results.outputs.out[0] == 35
    assert results.outputs.out[1] == 38
    assert results.outputs.out[2] == 46
    assert results.outputs.out[3] == 50


# workflows with structures A -> C, B -> C


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_st_1(plugin_dask_opt, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))

        return mult.out

    wf = Workflow().split(["x", "y"], x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin_dask_opt) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 6
    assert results.outputs.out[0] == 39
    assert results.outputs.out[1] == 42
    assert results.outputs.out[5] == 70


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_ndst_1(plugin_dask_opt, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the tasks levels
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin_dask_opt) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results.outputs.out) == 6
    assert results.outputs.out == [39, 42, 52, 56, 65, 70]


def test_wf_3nd_st_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the workflow level
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 2
    assert results.outputs.out[0][0] == 39
    assert results.outputs.out[0][1] == 52
    assert results.outputs.out[0][2] == 65
    assert results.outputs.out[1][0] == 42
    assert results.outputs.out[1][1] == 56
    assert results.outputs.out[1][2] == 70


def test_wf_3nd_ndst_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the tasks levels
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x))
        add2y = workflow.add(Add2().split("x", x=y))
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out).combine("add2x.x"))
        return mult.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results.outputs.out) == 2
    assert results.outputs.out[0] == [39, 52, 65]
    assert results.outputs.out[1] == [42, 56, 70]


def test_wf_3nd_st_3(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the workflow level
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("y")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 3
    assert results.outputs.out[0][0] == 39
    assert results.outputs.out[0][1] == 42
    assert results.outputs.out[1][0] == 52
    assert results.outputs.out[1][1] == 56
    assert results.outputs.out[2][0] == 65
    assert results.outputs.out[2][1] == 70


def test_wf_3nd_ndst_3(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the tasks levels
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out).combine("add2y.x"))
        return mult.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results.outputs.out) == 3
    assert results.outputs.out[0] == [39, 42]
    assert results.outputs.out[1] == [52, 56]
    assert results.outputs.out[2] == [65, 70]


def test_wf_3nd_st_4(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the workflow level
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow.split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine(["x", "y"])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 6
    assert results.outputs.out[0] == 39
    assert results.outputs.out[1] == 42
    assert results.outputs.out[2] == 52
    assert results.outputs.out[3] == 56
    assert results.outputs.out[4] == 65
    assert results.outputs.out[5] == 70


def test_wf_3nd_ndst_4(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the tasks levels
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).combine(["add2x.x", "add2y.x"])
        )
        return mult.out

    wf = Workflow(x=[1, 2, 3], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert len(results.outputs.out) == 6
    assert results.outputs.out == [39, 42, 52, 56, 65, 70]


def test_wf_3nd_st_5(plugin, tmpdir):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    splitter and partial combiner (from the second task) on the workflow level
    """

    @workflow.define
    def Workflow(x, y, z):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        addvar = workflow.add(FunAddVar3(a=add2x.out, b=add2y.out, c=z))
        return addvar.out

    wf = Workflow.split(["x", "y", "z"], x=[2, 3], y=[11, 12], z=[10, 100]).combine("y")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 4
    assert results.outputs.out[0][0] == 27
    assert results.outputs.out[0][1] == 28
    assert results.outputs.out[1][0] == 117
    assert results.outputs.out[1][1] == 118
    assert results.outputs.out[2][0] == 28
    assert results.outputs.out[2][1] == 29
    assert results.outputs.out[3][0] == 118
    assert results.outputs.out[3][1] == 119


def test_wf_3nd_ndst_5(plugin, tmpdir):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    all tasks have splitters and the last one has a partial combiner (from the 2nd)
    """

    @workflow.define
    def Workflow(x, y, z):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        addvar = workflow.add(
            FunAddVar3(a=add2x.out, b=add2y.out).split("c", c=z).combine("add2x.x")
        )

        return addvar.out

    wf = Workflow(x=[2, 3], y=[11, 12], z=[10, 100])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results.outputs.out) == 4
    assert results.outputs.out[0] == [27, 28]
    assert results.outputs.out[1] == [117, 118]
    assert results.outputs.out[2] == [28, 29]
    assert results.outputs.out[3] == [118, 119]

    # checking all directories


def test_wf_3nd_ndst_6(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones and a combiner
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out)
            .split(("_add2x", "_add2y"))
            .combine("add2y.x")
        )
        return mult.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [39, 56]


def test_wf_3nd_ndst_7(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones
    """

    @workflow.define
    def Workflow(x):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=x), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).split(("_add2x", "_add2y"))
        )
        return mult.out

    wf = Workflow(x=[1, 2])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [9, 16]


# workflows with structures A -> B -> C with multiple connections


def test_wf_3nd_8(tmpdir):
    """workflow with three tasks A->B->C vs two tasks A->C with multiple connections"""

    @workflow.define(outputs=["out1", "out2", "out1a", "out2a"])
    def Workflow(zip):

        iden2flds_1 = workflow.add(
            Identity2Flds(x2="Hoi").split("x1", x1=zip), name="iden2flds_1"
        )

        identity = workflow.add(Identity(x=iden2flds_1.out1))

        iden2flds_2 = workflow.add(
            Identity2Flds(x1=identity.out, x2=iden2flds_1.out2), name="iden2flds_2"
        )

        iden2flds_2a = workflow.add(
            Identity2Flds(
                x1=iden2flds_1.out1,
                x2=iden2flds_1.out2,
            )
        )

        return iden2flds_2.out1, iden2flds_2.out2, iden2flds_2a.out1, iden2flds_2a.out2

    wf = Workflow(zip=[["test1", "test3", "test5"], ["test2", "test4", "test6"]])

    with Submitter(worker="cf") as sub:
        res = sub(wf)

    assert (
        res.outputs.out1
        == res.outputs.out1a
        == [["test1", "test3", "test5"], ["test2", "test4", "test6"]]
    )
    assert res.outputs.out2 == res.output.out2a == ["Hoi", "Hoi"]


# workflows with Left and Right part in splitters A -> B (L&R parts of the splitter)


def test_wf_ndstLR_1(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has its own simple splitter
    and the  Left part from the first task should be added
    """

    @workflow.define
    def Workflow(x, y):
        add2 = workflow.add(Add2().split("x", x=x))
        mult = workflow.add(Multiply(x=add2.out).split("y", y=y))
        return mult.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the splitter is created properly
    assert wf.mult.state.splitter == ["_add2", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["add2.x", "mult.y", "*"]

    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]
    assert results.outputs.out == [33, 36, 44, 48]


def test_wf_ndstLR_1a(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own splitter)
    """

    @workflow.define
    def Workflow(x, y):
        add2 = workflow.add(Add2().split("x", x=x))
        mult = workflow.add(Multiply().split(["_add2", "y"], x=add2.out, y=y))
        return mult.out

    wf = Workflow(x=[1, 2], y=[11, 12])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the splitter is created properly
    assert wf.mult.state.splitter == ["_add2", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["add2.x", "mult.y", "*"]

    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]
    assert results.outputs.out == [33, 36, 44, 48]


def test_wf_ndstLR_2(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has its own outer splitter
    and the  Left part from the first task should be added
    """

    @workflow.define
    def Workflow(x, y, z):
        add2 = workflow.add(Add2().split("x", x=x))
        addvar = workflow.add(FunAddVar3(a=add2.out).split(["b", "c"], b=y, c=z))
        return addvar.out

    wf = Workflow(x=[1, 2, 3], y=[10, 20], z=[100, 200])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the splitter is created properly
    assert wf.addvar.state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf.addvar.state.splitter_rpn == ["add2.x", "addvar.b", "addvar.c", "*", "*"]

    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert results.outputs.out == [
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


def test_wf_ndstLR_2a(plugin, tmpdir):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own outer splitter)
    """

    @workflow.define
    def Workflow(x, y, z):
        add2 = workflow.add(Add2().split("x", x=x))
        addvar = workflow.add(
            FunAddVar3(a=add2.out).split(["_add2", ["b", "c"]], b=y, c=z)
        )

        return addvar.out

    wf = Workflow(x=[1, 2, 3], y=[10, 20], z=[100, 200])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the splitter is created properly
    assert wf.addvar.state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf.addvar.state.splitter_rpn == ["add2.x", "addvar.b", "addvar.c", "*", "*"]

    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert results.outputs.out == [
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


# workflows with inner splitters A -> B (inner spl)


def test_wf_ndstinner_1(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has inner splitter
    """

    @workflow.define(outputs=["out_list", "out"])
    def Workflow(x: int):
        list = workflow.add(ListOutput(x=x))
        add2 = workflow.add(Add2().split("x", x=list.out))
        return list.out, add2.out

    wf = Workflow(x=1)  #

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf.add2.state.splitter == "add2.x"
    assert wf.add2.state.splitter_rpn == ["add2.x"]

    assert results.outputs.out_list == [1, 2, 3]
    assert results.outputs.out == [3, 4, 5]


def test_wf_ndstinner_2(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has two inputs and inner splitter from one of the input
    """

    @workflow.define(outputs=["out_list", "out"])
    def Workflow(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply(y=y).split("x", x=list.out))
        return list.out, mult.out

    wf = Workflow(x=1, y=10)  #

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf.mult.state.splitter == "mult.x"
    assert wf.mult.state.splitter_rpn == ["mult.x"]

    assert results.outputs.out_list == [1, 2, 3]
    assert results.outputs.out == [10, 20, 30]


def test_wf_ndstinner_3(plugin, tmpdir):
    """workflow with 2 tasks,
    the second task has two inputs and outer splitter that includes an inner field
    """

    @workflow.define(outputs=["out_list", "out"])
    def Workflow(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply().split(["x", "y"], x=list.out, y=y))
        return list.out, mult.out

    wf = Workflow(x=1, y=[10, 100])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf.mult.state.splitter == ["mult.x", "mult.y"]
    assert wf.mult.state.splitter_rpn == ["mult.x", "mult.y", "*"]

    assert results.outputs.out_list == [1, 2, 3]
    assert results.outputs.out == [10, 100, 20, 200, 30, 300]


def test_wf_ndstinner_4(plugin, tmpdir):
    """workflow with 3 tasks,
    the second task has two inputs and inner splitter from one of the input,
    the third task has no its own splitter
    """

    @workflow.define(outputs=["out_list", "out"])
    def Workflow(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply(y=y).split("x", x=list.out))
        add2 = workflow.add(Add2(x=mult.out))
        return list.out, add2.out

    wf = Workflow(x=1, y=10)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf.mult.state.splitter == "mult.x"
    assert wf.mult.state.splitter_rpn == ["mult.x"]
    assert wf.add2.state.splitter == "_mult"
    assert wf.add2.state.splitter_rpn == ["mult.x"]

    assert results.outputs.out_list == [1, 2, 3]
    assert results.outputs.out == [12, 22, 32]


def test_wf_ndstinner_5(plugin, tmpdir):
    """workflow with 3 tasks,
    the second task has two inputs and inner splitter from one of the input,
    (inner input come from the first task that has its own splitter,
    there is a inner_cont_dim)
    the third task has no new splitter
    """

    @workflow.define(outputs=["out_list", "out_mult", "out_add"])
    def Workflow(x, y, b):
        list = workflow.add(ListOutput().split("x", x=x))
        mult = workflow.add(Multiply().split(["y", "x"], x=list.out, y=y))
        addvar = workflow.add(FunAddVar(a=mult.out).split("b", b=b))
        return list.out, mult.out, addvar.out

    wf = Workflow(x=[1, 2], y=[10, 100], b=[3, 5])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

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

    assert results.outputs.out_list == [[1, 2, 3], [2, 4, 6]]
    assert results.outputs.out_mult == [
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
    assert results.outputs.out_add == [
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


# workflow that have some single values as the input


def test_wf_st_singl_1(plugin, tmpdir):
    """workflow with two tasks, only one input is in the splitter and combiner"""

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))

        return add2.out

    wf = Workflow().split("x", x=[1, 2], y=11).combine("x")

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out[0] == 13
    assert results.outputs.out[1] == 24


def test_wf_ndst_singl_1(plugin, tmpdir):
    """workflow with two tasks, outer splitter and combiner on tasks level;
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x))
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"))
        return add2.out

    wf = Workflow(x=[1, 2], y=11)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [13, 24]


def test_wf_st_singl_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow.split("x", x=[1, 2, 3], y=11)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results) == 3
    assert results.outputs.out[0] == 39
    assert results.outputs.out[1] == 52
    assert results.outputs.out[2] == 65


def test_wf_ndst_singl_2(plugin, tmpdir):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the tasks levels
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Workflow(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out))
        return mult.out

    wf = Workflow(x=[1, 2, 3], y=11)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert len(results.outputs.out) == 3
    assert results.outputs.out == [39, 52, 65]


# workflows with structures wf(A)


def test_wfasnd_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x))
        return wfnd.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == 4


def test_wfasnd_wfinp_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    input set for the main workflow
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=2)

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    assert results.outputs.out == 4


def test_wfasnd_wfndupdate(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    wfasnode input is updated to use the main workflow input
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x))
        return wfnd.out

    wf = Workflow(x=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == 5


def test_wfasnd_wfndupdate_rerun(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task and no splitter
    wfasnode is run first and later is
    updated to use the main workflow input
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    wfnd = Wfnd(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        sub(wfnd)

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x))
        return wfnd.out

    wf = Workflow(x=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == 5

    # adding another layer of workflow
    @workflow.define
    def WorkflowO(x):
        wf = workflow.add(Workflow(x=3))
        return wf.out

    wf_o = WorkflowO(x=4)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf_o)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == 6


def test_wfasnd_st_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for wfnd
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x).split(x=x))
        return wfnd.out

    wf = Workflow(x=[2, 4])

    checksum_before = wf._checksum
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert wf._checksum == checksum_before

    assert results.outputs.out == [4, 6]


def test_wfasnd_st_updatespl_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for wfnd is set after add
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x).split(x=x))
        return wfnd.out

    wf = Workflow(x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [4, 6]


def test_wfasnd_ndst_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for node
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2().split("x", x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [4, 6]


def test_wfasnd_ndst_updatespl_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for node added after add
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2().split("x", x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [4, 6]


def test_wfasnd_wfst_1(plugin, tmpdir):
    """workflow as a node
    workflow-node with one task,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow().split("x", x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out[0] == 4
    assert results.outputs.out[1] == 6


# workflows with structures wf(A) -> B


def test_wfasnd_st_2(plugin, tmpdir):
    """workflow as a node,
    the main workflow has two tasks,
    splitter for wfnd
    """

    @workflow.define
    def Wfnd(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y))
        return mult.out

    @workflow.define
    def Workflow(x, y):
        wfnd = workflow.add(Wfnd(x=x, y=y))
        add2 = workflow.add(Add2(x=wfnd.out))
        return add2.out

    wf = Workflow(x=[2, 4], y=[1, 10])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out == [4, 42]


def test_wfasnd_wfst_2(plugin, tmpdir):
    """workflow as a node,
    the main workflow has two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        return mult.out

    @workflow.define
    def Workflow(x, y):
        wfnd = workflow.add(Wfnd(x=x, y=y))
        add2 = workflow.add(Add2(x=wfnd.out))
        return add2.out

    wf = Workflow().split(("x", "y"), x=[2, 4], y=[1, 10])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out[0] == 4
    assert results.outputs.out[1] == 42


# workflows with structures A -> wf(B)


def test_wfasnd_ndst_3(plugin, tmpdir):
    """workflow as the second node,
    the main workflow has two tasks,
    splitter for the first task
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y))
        wfnd = workflow.add(Wfnd(mult.out))
        return wfnd.out

    wf = Workflow(x=[2, 4], y=[1, 10])

    with Submitter(cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out == [4, 42]


def test_wfasnd_wfst_3(plugin, tmpdir):
    """workflow as the second node,
    the main workflow has two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x))
        return add2.out

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))

        wfnd = workflow.add(Wfnd(mult.out))

        return wfnd.out

    wf = Workflow().split(("x", "y"), x=[2, 4], y=[1, 10])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out[0] == 4
    assert results.outputs.out[1] == 42


# workflows with structures wfns(A->B)


def test_wfasnd_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks and no splitter
    """

    @workflow.define
    def Wfnd(x):
        add2_1st = workflow.add(Add2(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=2))
        return wfnd.out

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == 6


def test_wfasnd_ndst_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks,
    splitter for node
    """

    @workflow.define
    def Wfnd(x):
        add2_1st = workflow.add(Add2().split(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow(x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [6, 8]


def test_wfasnd_wfst_4(plugin, tmpdir):
    """workflow as a node
    workflow-node with two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x):
        add2_1st = workflow.add(Add2(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    wf = Workflow().split("x", x=[2, 4])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])
    # assert wf.output_dir.exists()

    assert results.outputs.out[0] == 6
    assert results.outputs.out[1] == 8


# Testing caching


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir(plugin, tmpdir):
    """wf with provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_wf_cache_1")

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir_relativepath(tmpdir, plugin):
    """wf with provided cache_dir as relative path"""
    tmpdir.chdir()
    cache_dir = "test_wf_cache_2"
    tmpdir.mkdir(cache_dir)

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir;
    the second wf has cache_locations and should not recompute the results
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time (second one should be quick)
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # checking if both wf.output_dir are created
    assert results1.output_dir != results2.output_dir


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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define("out_pr")
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out == results2.outputs.out_pr

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # execution time for second run should be much shorter
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking if the second wf didn't run again
    assert results1.output_dir != results2.output_dir


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

    @workflow.define(outputs=["out1"])
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out  # out1

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out1

    @workflow.define(outputs=["out2"])
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out  # out2

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time (the second wf should be fast, nodes do not have to rerun)
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # both wf output_dirs should be created
    assert results1.output_dir != results2.output_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_setoutputchange_a(plugin, tmpdir):
    """
    the same as previous test, but wf names and output names differ,
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    @workflow.define(outputs=["out1"])
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out  # out1

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out1

    @workflow.define(outputs=["out2"])
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create wf itself)
        assert t2 < max(1, t1 - 1)

    # both wf output_dirs should be created
    assert results1.output_dir != results2.output_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_forcererun(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir;
    the second wf has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir2) as sub:
        results2 = sub(wf2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking execution time
        assert t1 > 2
        assert t2 > 2

    # checking if the second wf didn't run again
    assert results1.output_dir != results2.output_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_wftaskrerun_propagateTrue(plugin, tmpdir):
    """
    Two identical wfs with provided cache_dir and cache_locations for the second one;
    submitter doesn't have rerun, but the second wf has rerun=True,
    propagate_rerun is True as default, so everything should be rerun
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # checking if the second wf runs again
    assert results1.output_dir != results2.output_dir

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2, rerun=True, propagate_rerun=False)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # checking if the second wf runs again
    assert results1.output_dir != results2.output_dir

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        # rerun on the task level needed (wf.propagate_rerun is False)
        add2 = workflow.add(Add2Wait(x=mult.out, rerun=True))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(
            wf2, rerun=True, propagate_rerun=False
        )  # rerun will not be propagated to each task)
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    assert results1.output_dir != results2.output_dir
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

    @workflow.define
    def Workflow1(x):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out))
        return add2.out

    wf1 = Workflow1(x=3)

    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 12 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out))
        return add2.out

    wf2 = Workflow2(x=2)

    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 12 == results2.outputs.out

    # checking if the second wf runs again, but runs only one task
    assert results1.output_dir != results2.output_dir
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

    @workflow.define
    def Workflow1(x):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out))
        return add2.out

    wf1 = Workflow1(x=3)

    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 12 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out))
        return add2.out

    wf2 = Workflow2(x=2)

    # updating cache_locations after adding the tasks
    wf2.cache_locations = cache_dir1

    with Submitter(worker=plugin, cache_dir=cache_dir2) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 12 == results2.outputs.out

    # checking if the second wf runs again, but runs only one task
    assert results1.output_dir != results2.output_dir
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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir2) as sub:
        results2 = sub(wf2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Workflow2(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert results1.outputs.out == 8

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2().split(splitter=("x", "y"), x=[2, 20], y=[3, 4])

    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    t0 = time.time()
    # changing cache_locations to non-existing dir
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1_empty
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking if both wf run
    assert results1.output_dir != results2.output_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_recompute(plugin, tmpdir):
    """
    Two wfs with the same inputs but slightly different graph;
    the second wf should recompute the results,
    but the second node should use the results from the first wf (has the same input)
    """
    cache_dir1 = tmpdir.mkdir("test_wf_cache3")
    cache_dir2 = tmpdir.mkdir("test_wf_cache4")

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 8 == results1.outputs.out

    @workflow.define
    def Workflow2(x, y):

        # different argument assignment
        mult = workflow.add(Multiply(x=y, y=x))
        add2 = workflow.add(Add2(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=2, y=3)

    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 8 == results2.outputs.out

    # checking if both dir exists
    assert results1.output_dir != results2.output_dir

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir2) as sub:
        results2 = sub(wf2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply().split(splitter=("x", "y"), x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Workflow2(x, y):

        mult = workflow.add(Multiply().split(splitter=["x", "y"], x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf2 = Workflow2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(wf2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 10, 62, 82]

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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out
    # checkoing output_dir after the first run
    assert wf1.output_dir.exists()

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results1.outputs.out
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

    @workflow.define
    def Workflow1(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add2 = workflow.add(Add2Wait(x=mult.out))
        return add2.out

    wf1 = Workflow1().split(splitter=("x", "y"), x=[2, 20], y=[3, 30])

    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out[0]
    assert 602 == results1.outputs.out[1]

    # checkoing output_dir after the first run
    assert [odir.exists() for odir in wf1.output_dir]

    # saving the content of the cache dit after the first run
    cache_dir_content = os.listdir(wf1.cache_dir)

    # running workflow the second time
    t0 = time.time()
    with Submitter(worker=plugin, cache_dir=cache_dir1) as sub:
        results1 = sub(wf1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results1.outputs.out[0]
    assert 602 == results1.outputs.out[1]
    # checking if no new directory is created
    assert cache_dir_content == os.listdir(wf1.cache_dir)
    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and plugin == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


@pytest.fixture
def create_tasks():
    @workflow.define
    def Workflow(x):
        t1 = workflow.add(Add2(x=x))
        t2 = workflow.add(Multiply(x=t1.out, y=2))
        return t2.out

    wf = Workflow(x=1)
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
    wf = wf.split("x", x=[1, 2])
    wf.cache_dir = (tmpdir / "shared").strpath
    wf(plugin="cf")
    assert wf.cache_dir == t1.cache_dir == t2.cache_dir


def test_workflow_combine1(tmpdir):
    @workflow.define(outputs=["out_pow", "out_iden1", "out_iden2"])
    def Workflow1(a, b):
        power = workflow.add(Power().split(["a", "b"], a=a, b=b))
        identity1 = workflow.add(
            Identity(x=power.out).combine("power.a"), name="identity1"
        )
        identity2 = workflow.add(
            Identity(x=identity1.out).combine("power.b"), name="identity2"
        )
        return power.out, identity1.out, identity2.out

    wf1 = Workflow1(a=[1, 2], b=[2, 3])
    outputs = wf1()

    assert outputs.out_pow == [1, 1, 4, 8]
    assert outputs.out_iden1 == [[1, 4], [1, 8]]
    assert outputs.out_iden2 == [[1, 4], [1, 8]]


def test_workflow_combine2(tmpdir):
    @workflow.define(outputs=["out_pow", "out_iden"])
    def Workflow1(a, b):
        power = workflow.add(Power().split(["a", "b"], a=a, b=b).combine("a"))
        identity = workflow.add(Identity(x=power.out).combine("power.b"))
        return power.out, identity.out

    wf1 = Workflow1(a=[1, 2], b=[2, 3])
    outputs = wf1(cache_dir=tmpdir)

    assert outputs.out_pow == [[1, 4], [1, 8]]
    assert outputs.out_iden == [[1, 4], [1, 8]]


# g.all to collect all of the results and let PythonTask deal with it


def test_wf_lzoutall_1(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_sub2_res function
    using.all syntax
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add_sub = workflow.add(Add2Sub2Res(res=mult.all_))
        return add_sub.out_add

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert 8 == results.outputs.out


def test_wf_lzoutall_1a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    using.all syntax in the node connections and for wf output
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y))
        add_sub = workflow.add(Add2Sub2Res(res=mult.all_))
        return add_sub.all_  # out_all

    wf = Workflow(x=2, y=3)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out_all == {"out_add": 8, "out_sub": 4}


def test_wf_lzoutall_st_1(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    using.all syntax
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y))
        add_sub = workflow.add(Add2Sub2Res(res=mult.all_))
        return add_sub.out_add  # out_add

    wf = Workflow(x=[2, 20], y=[3, 30])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out_add == [8, 62, 62, 602]


def test_wf_lzoutall_st_1a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    using.all syntax
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y))
        add_sub = workflow.add(Add2Sub2Res(res=mult.all_))
        return add_sub.all_  # out_all

    wf = Workflow(x=[2, 20], y=[3, 30])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out_all == [
        {"out_add": 8, "out_sub": 4},
        {"out_add": 62, "out_sub": 58},
        {"out_add": 62, "out_sub": 58},
        {"out_add": 602, "out_sub": 598},
    ]


def test_wf_lzoutall_st_2(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    using.all syntax
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y).combine("x"))
        add_sub = workflow.add(Add2Sub2ResList(res=mult.all_))
        return add_sub.out_add  # out_add

    wf = Workflow(x=[2, 20], y=[3, 30])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out_add[0] == [8, 62]
    assert results.outputs.out_add[1] == [62, 602]


@pytest.mark.xfail(
    condition=bool(shutil.which("sbatch")),  # using SLURM
    reason=(
        "Not passing on SLURM image for some reason, hoping upgrade of image/Python "
        "version fixes it"
    ),
)
def test_wf_lzoutall_st_2a(plugin, tmpdir):
    """workflow with 2 tasks, no splitter
    passing entire result object to add2_res function
    using.all syntax
    """

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y).combine("x"))
        add_sub = workflow.add(Add2Sub2ResList(res=mult.all_))
        return add_sub.all_  # out_all

    wf = Workflow(x=[2, 20], y=[3, 30])

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out_all == [
        {"out_add": [8, 62], "out_sub": [4, 58]},
        {"out_add": [62, 602], "out_sub": [58, 598]},
    ]


# workflows that have files in the result, the files should be copied to the wf dir


def test_wf_resultfile_1(plugin, tmpdir):
    """workflow with a file in the result, file should be copied to the wf dir"""

    @workflow.define
    def Workflow(x):
        writefile = workflow.add(FunWriteFile(filename=x))

        return writefile.out  # wf_out

    wf = Workflow(x="file_1.txt")
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the file exists and if it is in the Workflow directory
    wf_out = results.outputs.wf_out.fspath
    wf_out.exists()
    assert wf_out == wf.output_dir / "file_1.txt"


def test_wf_resultfile_2(plugin, tmpdir):
    """workflow with a list of files in the wf result,
    all files should be copied to the wf dir
    """

    @workflow.define
    def Workflow(x):
        writefile = workflow.add(FunWriteFileList(filename_list=x))

        return writefile.out  # wf_out

    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    wf = Workflow(x=file_list)
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the file exists and if it is in the Workflow directory
    for ii, file in enumerate(results.outputs.wf_out):
        assert file.fspath.exists()
        assert file.fspath == wf.output_dir / file_list[ii]


def test_wf_resultfile_3(plugin, tmpdir):
    """workflow with a dictionaries of files in the wf result,
    all files should be copied to the wf dir
    """

    @workflow.define
    def Workflow(x):
        writefile = workflow.add(FunWriteFileList2Dict(filename_list=x))

        return writefile.out  # wf_out

    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    wf = Workflow(x=file_list)
    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    # checking if the file exists and if it is in the Workflow directory
    for key, val in results.outputs.wf_out.items():
        if key == "random_int":
            assert val == 20
        else:
            assert val.fspath.exists()
            ii = int(key.split("_")[1])
            assert val.fspath == wf.output_dir / file_list[ii]


def test_wf_upstream_error1(plugin, tmpdir):
    """workflow with two tasks, task2 dependent on an task1 which raised an error"""

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    wf = Workflow(x="hi")  # TypeError for adding str and int

    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error2(plugin, tmpdir):
    """task2 dependent on task1, task1 errors, workflow-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    wf = Workflow().split(
        "x", x=[1, "hi"]
    )  # workflow-level split TypeError for adding str and int

    with pytest.raises(Exception) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


@pytest.mark.flaky(reruns=2)  # when slurm
def test_wf_upstream_error3(plugin, tmpdir):
    """task2 dependent on task1, task1 errors, task-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType().split("a", a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    wf = Workflow(x=[1, "hi"])  # TypeError for adding str and int
    with pytest.raises(Exception) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error4(plugin, tmpdir):
    """workflow with one task, which raises an error"""

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x))

        return addvar1.out

    wf = Workflow(x="hi")  # TypeError for adding str and int
    with pytest.raises(Exception) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "raised an error" in str(excinfo.value)
    assert "addvar1" in str(excinfo.value)


def test_wf_upstream_error5(plugin, tmpdir):
    """nested workflow with one task, which raises an error"""

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x))
        return addvar1.out  # wf_out

    @workflow.define
    def WfMain(x):
        wf = workflow.add(Workflow(x=x))
        return wf.out

    wf_main = WfMain(x="hi")  # TypeError for adding str and int

    with pytest.raises(Exception) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf_main)

    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error6(plugin, tmpdir):
    """nested workflow with two tasks, the first one raises an error"""

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")

        return addvar2.out  # wf_out

    @workflow.define
    def WfMain(x):
        wf = workflow.add(Workflow(x=x))
        return wf.out

    wf_main = WfMain(x="hi")  # TypeError for adding str and int

    with pytest.raises(Exception) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf_main)

    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)


def test_wf_upstream_error7(plugin, tmpdir):
    """
    workflow with three sequential tasks, the first task raises an error
    the last task is set as the workflow output
    """

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar3.out

    wf = Workflow(x="hi")  # TypeError for adding str and int

    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
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

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar3.out

    wf = Workflow(x="hi")  # TypeError for adding str and int
    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
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

    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar2.out, addvar3.out  #

    wf = Workflow(x="hi")  # TypeError for adding str and int
    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
            sub(wf)
    assert "addvar1" in str(excinfo.value)
    assert "raised an error" in str(excinfo.value)
    assert wf.addvar1._errored is True
    assert wf.addvar2._errored == wf.addvar3._errored == ["addvar1"]


def test_wf_upstream_error8(plugin, tmpdir):
    """workflow with three tasks, the first one raises an error, so 2 others are removed"""

    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addtwo = workflow.add(FunAddTwo(a=addvar1.out))
        return addvar2.out, addtwo.out  #

    wf = Workflow(x="hi")  # TypeError for adding str and int
    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
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

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        follow_err = workflow.add(FunAddVarDefaultNoType(a=err.out), name="follow_err")

        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        workflow.add(FunAddVarDefaultNoType(a=addtwo.out))
        return follow_err.out  # out1

    wf = Workflow(x=2)
    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
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

    @workflow.define
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefault(a=x), name="addvar1")

        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        workflow.add(FunAddVarDefault(a=err.out))

        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        addvar2 = workflow.add(FunAddVarDefault(a=addtwo.out), name="addvar2")
        return addvar2.out  # out1  # , ("out2", addtwo.out)])

    wf = Workflow(x=2)

    with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
        sub(wf)
    assert wf.err._errored is True
    assert wf.follow_err._errored == ["err"]


def test_wf_upstream_error9b(plugin, tmpdir):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    both branches are connected to the workflow output
    """

    @workflow.define(outputs=["out1", "out2"])
    def Workflow(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        follow_err = workflow.add(FunAddVarDefaultNoType(a=err.out), name="follow_err")

        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addtwo.out), name="addvar2")
        return follow_err.out, addvar2.out

    wf = Workflow(x=2)
    with pytest.raises(ValueError) as excinfo:
        with Submitter(worker=plugin, cache_dir=tmpdir) as sub:
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

    @workflow.define
    def Workflow(x, y):
        mult_1 = workflow.add(Multiply(x=x, y=y), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    wf = Workflow().split(splitter, x=[1, 2])

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

    @workflow.define
    def Workflow(x, y):
        mult_1 = workflow.add(Multiply(y=y).split("x", x=x), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    wf = Workflow(x=[1, 2], y=2)

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

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x), name="mult")
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"), name="add2")
        sum = workflow.add(ListSum(x=add2.out), name="sum")
        return sum.out

    wf = Workflow(x=[1, 2], y=2)
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

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x), name="wfnd")
        return wfnd.out

    wf = Workflow(x=2)

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

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Workflow(x):
        wfnd = workflow.add(Wfnd(x=x).split("x", x=x), name="wfnd")
        return wfnd.out

    wf = Workflow(x=[1, 2])

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

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    wf = Workflow(x=2)

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

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    wf = Workflow(x=[1, 2], y=2)

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

    @workflow.define
    def Wfnd(x):
        add2_a = workflow.add(Add2(x=x), name="add2_a")
        add2_b = workflow.add(Add2(x=add2_a.out), name="add2_b")
        return add2_b.out

    @workflow.define
    def Workflow(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    wf = Workflow(x=2, y=3)

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

    @workflow.define
    def Wfnd(x):
        add2_a = workflow.add(Add2(x=x), name="add2_a")
        add2_b = workflow.add(Add2(x=add2_a.out), name="add2_b")
        return add2_b.out

    @workflow.define
    def Workflow(x, y):
        wfnd = workflow.add(Wfnd(x=x), name="wfnd")
        mult = workflow.add(Multiply(x=wfnd.out, y=y), name="mult")
        return mult.out

    wf = Workflow(x=2, y=3)

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

    @python.define
    def printer(a):
        return a

    @workflow.define
    def Workflow(text):
        printer1 = workflow.add(printer(a=text))
        return printer1.out  # out1

    wf = Workflow().split(text=text)

    with Submitter(worker="cf", n_procs=6) as sub:
        results = sub(wf)

    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.output.out1[0] == "test" and results.output.out1[0] == "test"


@pytest.mark.timeout(40)
def test_inner_outer_wf_duplicate(tmpdir):
    """checking if the execution gets stuck if there is an inner and outer workflows
    that run two nodes with the exact same inputs.
    """
    task_list = ["First", "Second"]
    start_list = [3, 4]

    @python.define
    def OneArg(start_number):
        for k in range(10):
            start_number += 1
        return start_number

    @python.define
    def OneArgInner(start_number):
        for k in range(10):
            start_number += 1
        return start_number

    # Inner Workflow
    @workflow.define(outputs=["res"])
    def InnerWf(start_number1):
        inner_level1 = workflow.add(OneArgInner(start_number=start_number1))
        return inner_level1.out

        # Outer workflow has two nodes plus the inner workflow

    # Outer workflow
    @workflow.define(outputs=["res2"])
    def OuterWf(start_number, task_name, dummy):
        level1 = workflow.add(OneArg(start_number=start_number))
        inner = workflow.add(InnerWf(start_number1=level1.out))
        return inner.res

    test_outer = OuterWf(dummy=1).split(
        ["start_number", "task_name"], start_number=start_list, task_name=task_list
    )

    with Submitter(worker="cf") as sub:
        res = sub(test_outer)

    assert res.output.res2[0] == 23 and res.output.res2[1] == 23


def test_rerun_errored(tmpdir, capfd):
    """Test rerunning a workflow containing errors.
    Only the errored tasks and workflow should be rerun"""

    @python.define
    def PassOdds(x):
        if x % 2 == 0:
            print(f"x%2 = {x % 2} (error)\n")
            raise Exception("even error")
        else:
            print(f"x%2 = {x % 2}\n")
            return x

    @workflow.define
    def Workflow(x):
        pass_odds = workflow.add(PassOdds().split("x", x=x))
        return pass_odds.out

    wf = Workflow(x=[1, 2, 3, 4, 5])

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


def test_wf_state_arrays():
    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def Workflow(x: ty.List[int], y: int):

        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                in_list=x,
            ).split(scalar=x)
        )

        B = workflow.add(  # Workflow is still split over "x", combined over "x" on out
            ListMultSum(
                scalar=A.sum,
                in_list=A.products,
            ).combine("A.scalar")
        )

        C = workflow.add(  # Workflow "
            ListMultSum(
                scalar=y,
                in_list=B.sum,
            )
        )

        D = workflow.add(  # Workflow is split again, this time over C.products
            ListMultSum(
                in_list=x,
            )
            .split(scalar=C.products)
            .combine("scalar")
        )

        E = workflow.add(  # Workflow is finally combined again into a single node
            ListMultSum(scalar=y, in_list=D.sum)
        )

        return E.sum, E.products

    wf = Workflow(x=[1, 2, 3, 4], y=10)

    results = wf()
    assert results.outputs.alpha == 3000000
    assert results.outputs.beta == [100000, 400000, 900000, 1600000]


def test_wf_input_output_typing():

    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def MismatchInputWf(x: int, y: ty.List[int]):
        ListMultSum(
            scalar=y,
            in_list=y,
            name="A",
        )

    with pytest.raises(TypeError) as exc_info:
        MismatchInputWf(x=1, y=[1, 2, 3])
    exc_info_matches(exc_info, "Cannot coerce <class 'list'> into <class 'int'>")

    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def MismatchOutputWf(x: int, y: ty.List[int]):
        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                scalar=x,
                in_list=y,
            )
        )
        return A.products, A.products

    with pytest.raises(TypeError, match="don't match their declared types"):
        MismatchOutputWf(x=1, y=[1, 2, 3])

    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def Workflow(x: int, y: ty.List[int]):
        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                scalar=x,
                in_list=y,
            )
        )
        return A.sum, A.products

    outputs = Workflow(x=10, y=[1, 2, 3, 4])()
    assert outputs.sum == 10
    assert outputs.products == [10, 20, 30, 40]
