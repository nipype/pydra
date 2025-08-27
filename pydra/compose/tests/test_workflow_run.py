import pytest
import shutil
import os
import sys
import time
import typing as ty
import attr
from pathlib import Path
from pydra.engine.tests.utils import (
    Add2,
    Add2Wait,
    Multiply,
    Divide,
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
from pydra.compose import python, workflow
from pydra.engine.workflow import Workflow
from pydra.utils.general import plot_workflow


def test_wf_no_output(worker: str, tmp_path: Path):
    """Raise error when output isn't set with set_output"""

    @workflow.define
    def Worky(x):
        workflow.add(Add2(x=x))

    with pytest.raises(ValueError, match="returned None"):
        Workflow.construct(Worky(x=2))


def test_wf_1(worker: str, tmp_path: Path):
    """workflow with one task and no splitter"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=2)

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    Workflow.construct(worky)
    assert worky._hash == checksum_before

    assert 4 == outputs.out


def test_wf_1a_outpastuple(worker: str, tmp_path: Path):
    """workflow with one task and no splitter
    set_output takes a tuple
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 4 == outputs.out


def test_wf_1_call_subm(worker: str, tmp_path: Path):
    """using wf["__call_"] with submitter"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 4 == outputs.out


def test_wf_1_call_plug(worker: str, tmp_path: Path):
    """using wf["__call_"] with worker"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky(worker=worker)

    assert 4 == outputs.out


def test_wf_1_call_noplug_nosubm(worker: str, tmp_path: Path):
    """using wf["__call_"] without worker or submitter"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky()

    assert 4 == outputs.out


def test_wf_1_upd_in_run(tmp_path, worker):
    """Updating input in __call__"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky(x=1)
    worky.x = 2
    outputs = worky(cache_root=tmp_path, worker=worker)
    assert 4 == outputs.out


def test_wf_2(worker: str, tmp_path: Path):
    """workflow with 2 tasks, no splitter"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 8 == outputs.out


def test_wf_2a(worker: str, tmp_path: Path):
    """workflow with 2 tasks, no splitter
    creating add2_task first (before calling add method),
    """

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 8 == outputs.out


def test_wf_2b(worker: str, tmp_path: Path):
    """workflow with 2 tasks, no splitter
    creating add2_task first (before calling add method),
    adding inputs.x after add method
    """

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 8 == outputs.out


def test_wf_2c_multoutp(worker: str, tmp_path: Path):
    """workflow with 2 tasks, no splitter
    setting multiple outputs for the workflow
    """

    @workflow.define(outputs=["out_add2", "out_mult"])
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out, mult.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking outputs from both nodes
    assert 6 == outputs.out_mult
    assert 8 == outputs.out_add2


def test_wf_2d_outpasdict(worker: str, tmp_path: Path):
    """workflow with 2 tasks, no splitter
    setting multiple outputs using a dictionary
    """

    @workflow.define(outputs=["out_add2", "out_mult"])
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out, mult.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking outputs from both nodes
    assert 6 == outputs.out_mult
    assert 8 == outputs.out_add2


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3(worker, tmp_path: Path):
    """testing None value for an input"""

    @workflow.define
    def Worky(x, y):
        addvar = workflow.add(FunAddVarNone(a=x, b=y))
        add2 = workflow.add(Add2(x=addvar.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=None)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 4 == outputs.out


@pytest.mark.xfail(reason="the task error doesn't propagate")
def test_wf_3a_exception(worker: str, tmp_path: Path):
    """testinh worky without set input, attr.NOTHING should be set
    and the function should raise an exception
    """

    @workflow.define
    def Worky(x, y):
        addvar = workflow.add(FunAddVarNone(a=x, b=y))
        add2 = workflow.add(Add2(x=addvar.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=attr.NOTHING)

    with pytest.raises(TypeError, match="unsupported"):
        worky(worker=worker, cache_root=tmp_path)


def test_wf_4(worker: str, tmp_path: Path):
    """worky with a task that doesn't set one input and use the function default value"""

    @workflow.define
    def Worky(x, y=None):
        addvar = workflow.add(FunAddVarDefault(a=x))
        add2 = workflow.add(Add2(x=addvar.out), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 5 == outputs.out


def test_wf_4a(worker: str, tmp_path: Path):
    """worky with a task that doesn't set one input,
    the unset input is send to the task input,
    so the task should use the function default value
    """

    @workflow.define
    def Worky(x):
        addvar = workflow.add(FunAddVarDefault(a=x))
        add2 = workflow.add(Add2(x=addvar.out), name="add2")
        return add2.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 5 == outputs.out


def test_wf_5(worker: str, tmp_path: Path):
    """worky with two outputs connected to the task outputs
    one set_output
    """

    @workflow.define(outputs=["out_sum", "out_sub"])
    def Worky(x, y):
        addsub = workflow.add(FunAddSubVar(a=x, b=y))
        return addsub.sum, addsub.sub

    worky = Worky(x=3, y=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 5 == outputs.out_sum
    assert 1 == outputs.out_sub


def test_wf_5a(worker: str, tmp_path: Path):
    """worky with two outputs connected to the task outputs,
    set_output set twice
    """

    @workflow.define(outputs=["out_sum", "out_sub"])
    def Worky(x, y):
        addsub = workflow.add(FunAddSubVar(a=x, b=y))
        return addsub.sum, addsub.sub

    worky = Worky(x=3, y=2)
    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 5 == outputs.out_sum
    assert 1 == outputs.out_sub


def test_wf_6(worker: str, tmp_path: Path):
    """worky with two tasks and two outputs connected to both tasks,
    one set_output
    """

    @workflow.define(outputs=["out1", "out2"])
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return mult.out, add2.out  #

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 6 == outputs.out1
    assert 8 == outputs.out2


def test_wf_6a(worker: str, tmp_path: Path):
    """worky with two tasks and two outputs connected to both tasks,
    set_output used twice
    """

    @workflow.define(outputs=["out1", "out2"])
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return mult.out, add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 6 == outputs.out1
    assert 8 == outputs.out2


def test_wf_st_1(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")

        return add2.out

    worky = Worky(x=[1, 2])

    checksum_before = worky._hash
    outputs = worky(cache_root=tmp_path, worker=worker)

    Workflow.construct(worky)
    assert worky._hash == checksum_before

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_subm(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")

        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_plug(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow
    using Worky.__call__(worker)
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")

        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_selfplug(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow
    using Worky.__call__() and using self.worker
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky()

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_call_noplug_nosubm(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow
    using Worky.__call__()  without worker and submitter
    (a submitter should be created within the __call__ function)
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky()

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_inp_in_call(tmp_path, worker):
    """Defining input in __call__"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky().split("x", x=[1, 2])
    outputs = worky(cache_root=tmp_path, worker=worker)  #
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_st_1_upd_inp_call(tmp_path, worker):
    """Updating input in __call___"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    worky = Worky().split("x", x=[1, 2])
    outputs = worky(cache_root=tmp_path, worker=worker)
    assert outputs.out == [3, 4]


def test_wf_st_noinput_1(worker: str, tmp_path: Path):
    """Worky with one task, a splitter for the workflow"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[])

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    assert worky._hash == checksum_before

    assert outputs.out == []


def test_wf_ndst_1(worker: str, tmp_path: Path):
    """workflow with one task, a splitter on the task level"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert worky._hash == checksum_before

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out == [3, 4]


def test_wf_ndst_updatespl_1(worker: str, tmp_path: Path):
    """workflow with one task,
    a splitter on the task level is added *after* calling add
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out == [3, 4]


def test_wf_ndst_updatespl_1a(worker: str, tmp_path: Path):
    """workflow with one task (initialize before calling add),
    a splitter on the task level is added *after* calling add
    """

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out == [3, 4]


def test_wf_ndst_updateinp_1(worker: str, tmp_path: Path):
    """workflow with one task,
    a splitter on the task level,
    updating input of the task after calling add
    """

    @workflow.define
    def Worky(x, y):
        add2 = workflow.add(Add2().split("x", x=y), name="add2")
        return add2.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [13, 14]


def test_wf_ndst_noinput_1(worker: str, tmp_path: Path):
    """workflow with one task, a splitter on the task level"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    worky = Worky(x=[])

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert worky._hash == checksum_before

    assert outputs.out == []


def test_wf_st_2(worker: str, tmp_path: Path):
    """workflow with one task, splitters and combiner for workflow"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2(x=x), name="add2")

        return add2.out

    worky = Worky().split("x", x=[1, 2]).combine(combiner="x")

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out[0] == 3
    assert outputs.out[1] == 4


def test_wf_ndst_2(worker: str, tmp_path: Path):
    """workflow with one task, splitters and combiner on the task level"""

    @workflow.define
    def Worky(x):
        add2 = workflow.add(Add2().split("x", x=x).combine(combiner="x"), name="add2")
        return add2.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1}, 3), ({"test7.x": 2}, 4)]
    assert outputs.out == [3, 4]


# workflows with structures A -> B


def test_wf_st_3(worker: str, tmp_path: Path):
    """workflow with 2 tasks, splitter on worky level"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")

        return add2.out

    worky = Worky().split(("x", "y"), x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    expected = [
        ({"wfst_3.x": 1, "wfst_3.y": 11}, 13),
        ({"wfst_3.x": 2, "wfst_3.y": 12}, 26),
    ]
    expected_ind = [
        ({"wfst_3.x": 0, "wfst_3.y": 0}, 13),
        ({"wfst_3.x": 1, "wfst_3.y": 1}, 26),
    ]

    for i, res in enumerate(expected):
        assert outputs.out[i] == res[1]


def test_wf_ndst_3(worker: str, tmp_path: Path):
    """Test workflow with 2 tasks, splitter on a task level"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)]
    assert outputs.out == [13, 26]


def test_wf_st_4(worker: str, tmp_path: Path):
    """workflow with two tasks, scalar splitter and combiner for the workflow"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")

        return add2.out

    worky = Worky().split(("x", "y"), x=[1, 2], y=[11, 12]).combine("x")
    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert outputs.out[0] == 13
    assert outputs.out[1] == 26


def test_wf_ndst_4(worker: str, tmp_path: Path):
    """workflow with two tasks, scalar splitter and combiner on tasks level"""

    @workflow.define
    def Worky(a, b):
        mult = workflow.add(Multiply().split(("x", "y"), x=a, y=b), name="mult")
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"), name="add2")

        return add2.out

    worky = Worky(a=[1, 2], b=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # expected: [
    #     ({"test7.x": 1, "test7.y": 11}, 13), ({"test7.x": 2, "test.y": 12}, 26)
    # ]
    assert outputs.out == [13, 26]


def test_wf_st_5(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and no combiner"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")

        return add2.out

    worky = Worky().split(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [13, 14, 24, 26]


def test_wf_ndst_5(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter on tasks level and no combiner"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0] == 13
    assert outputs.out[1] == 14
    assert outputs.out[2] == 24
    assert outputs.out[3] == 26


def test_wf_st_6(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and combiner for the workflow"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")

        return add2.out

    worky = Worky().split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0][0] == 13
    assert outputs.out[0][1] == 24
    assert outputs.out[0][2] == 35
    assert outputs.out[1][0] == 14
    assert outputs.out[1][1] == 26
    assert outputs.out[1][2] == 38


def test_wf_ndst_6(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and combiner on tasks level"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"), name="add2")
        return add2.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [[13, 24, 35], [14, 26, 38]]


def test_wf_ndst_7(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and (full) combiner for first node only"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(y=y).split(x=x).combine("x"), name="mult")
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    worky = Worky(x=[1, 2, 3], y=11)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [11, 22, 33]


def test_wf_ndst_8(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and (partial) combiner for first task only"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(
            Multiply().split(["x", "y"], x=x, y=y).combine("x"), name="mult"
        )
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [[11, 22, 33], [12, 24, 36]]


def test_wf_ndst_9(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and (full) combiner for first task only"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(
            Multiply().split(["x", "y"], x=x, y=y).combine(["x", "y"]), name="mult"
        )
        iden = workflow.add(Identity(x=mult.out))
        return iden.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [11, 12, 22, 24, 33, 36]


# workflows with structures A ->  B -> C


def test_wf_3sernd_ndst_1(worker: str, tmp_path: Path):
    """workflow with three "serial" tasks, checking if the splitter is propagating"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y), name="mult")
        add2_1st = workflow.add(Add2(x=mult.out), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # splitter from the first task should propagate to all tasks,
    # splitter_rpn should be the same in all tasks
    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == ["mult.x", "mult.y"]
    assert wf["add2_1st"].state.splitter == "_mult"
    assert wf["add2_2nd"].state.splitter == "_add2_1st"
    assert (
        ["mult.x", "mult.y", "*"]
        == wf["mult"].state.splitter_rpn
        == wf["add2_1st"].state.splitter_rpn
        == wf["add2_2nd"].state.splitter_rpn
    )

    assert outputs.out == [15, 16, 26, 28]


def test_wf_3sernd_ndst_1a(worker: str, tmp_path: Path):
    """
    workflow with three "serial" tasks, checking if the splitter is propagating
    first task has a splitter that propagates to the 2nd task,
    and the 2nd task is adding one more input to the splitter
    """

    @workflow.define
    def Worky(x, y):
        add2_1st = workflow.add(Add2().split("x", x=x), name="add2_1st")
        mult = workflow.add(Multiply(x=add2_1st.out).split("y", y=y), name="mult")
        add2_2nd = workflow.add(Add2(x=mult.out), name="add2_2nd")
        return add2_2nd.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # splitter from the 1st task should propagate and the 2nd task should add one more
    # splitter_rpn for the 2nd and the 3rd task should be the same
    wf = Workflow.construct(worky)
    assert wf["add2_1st"].state.splitter == "add2_1st.x"
    assert wf["mult"].state.splitter == ["_add2_1st", "mult.y"]
    assert wf["add2_2nd"].state.splitter == "_mult"
    assert (
        ["add2_1st.x", "mult.y", "*"]
        == wf["mult"].state.splitter_rpn
        == wf["add2_2nd"].state.splitter_rpn
    )

    assert outputs.out == [35, 38, 46, 50]


# workflows with structures A -> C, B -> C


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_st_1(worker, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")

        return mult.out

    worky = Worky().split(["x", "y"], x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0] == 39
    assert outputs.out[1] == 42
    assert outputs.out[5] == 70


@pytest.mark.flaky(reruns=3)  # when dask
def test_wf_3nd_ndst_1(worker, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the tasks levels
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert len(outputs.out) == 6
    assert outputs.out == [39, 42, 52, 56, 65, 70]


def test_wf_3nd_st_2(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the workflow level
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky().split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("x")

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0][0] == 39
    assert outputs.out[0][1] == 52
    assert outputs.out[0][2] == 65
    assert outputs.out[1][0] == 42
    assert outputs.out[1][1] == 56
    assert outputs.out[1][2] == 70


def test_wf_3nd_ndst_2(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner on the tasks levels
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).combine("add2x.x"), name="mult"
        )
        return mult.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert len(outputs.out) == 2
    assert outputs.out == [[39, 52, 65], [42, 56, 70]]


def test_wf_3nd_st_3(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the workflow level
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky().split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine("y")

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0][0] == 39
    assert outputs.out[0][1] == 42
    assert outputs.out[1][0] == 52
    assert outputs.out[1][1] == 56
    assert outputs.out[2][0] == 65
    assert outputs.out[2][1] == 70


def test_wf_3nd_ndst_3(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and partial combiner (from the second task) on the tasks levels
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).combine("add2y.x"), name="mult"
        )
        return mult.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert len(outputs.out) == 3
    assert outputs.out[0] == [39, 42]
    assert outputs.out[1] == [52, 56]
    assert outputs.out[2] == [65, 70]


def test_wf_3nd_st_4(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the workflow level
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky().split(["x", "y"], x=[1, 2, 3], y=[11, 12]).combine(["x", "y"])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0] == 39
    assert outputs.out[1] == 42
    assert outputs.out[2] == 52
    assert outputs.out[3] == 56
    assert outputs.out[4] == 65
    assert outputs.out[5] == 70


def test_wf_3nd_ndst_4(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter and full combiner on the tasks levels
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).combine(["add2x.x", "add2y.x"])
        )
        return mult.out

    worky = Worky(x=[1, 2, 3], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert len(outputs.out) == 6
    assert outputs.out == [39, 42, 52, 56, 65, 70]


def test_wf_3nd_st_5(worker: str, tmp_path: Path):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    splitter and partial combiner (from the second task) on the workflow level
    """

    @workflow.define
    def Worky(x, y, z):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        addvar = workflow.add(FunAddVar3(a=add2x.out, b=add2y.out, c=z))
        return addvar.out

    worky = (
        Worky().split(["x", "y", "z"], x=[2, 3], y=[11, 12], z=[10, 100]).combine("y")
    )

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out[0][0] == 27
    assert outputs.out[0][1] == 28
    assert outputs.out[1][0] == 117
    assert outputs.out[1][1] == 118
    assert outputs.out[2][0] == 28
    assert outputs.out[2][1] == 29
    assert outputs.out[3][0] == 118
    assert outputs.out[3][1] == 119


def test_wf_3nd_ndst_5(worker: str, tmp_path: Path):
    """workflow with three tasks (A->C, B->C) and three fields in the splitter,
    all tasks have splitters and the last one has a partial combiner (from the 2nd)
    """

    @workflow.define
    def Worky(x, y, z):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        addvar = workflow.add(
            FunAddVar3(a=add2x.out, b=add2y.out).split("c", c=z).combine("add2x.x")
        )

        return addvar.out

    worky = Worky(x=[2, 3], y=[11, 12], z=[10, 100])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert len(outputs.out) == 4
    assert outputs.out[0] == [27, 28]
    assert outputs.out[1] == [117, 118]
    assert outputs.out[2] == [28, 29]
    assert outputs.out[3] == [118, 119]

    # checking all directories


def test_wf_3nd_ndst_6(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones and a combiner
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=y), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out)
            .split(("_add2x", "_add2y"))
            .combine("add2y.x")
        )
        return mult.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [39, 56]


def test_wf_3nd_ndst_7(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    the third one uses scalar splitter from the previous ones
    """

    @workflow.define
    def Worky(x):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2().split("x", x=x), name="add2y")
        mult = workflow.add(
            Multiply(x=add2x.out, y=add2y.out).split(("_add2x", "_add2y"))
        )
        return mult.out

    worky = Worky(x=[1, 2])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [9, 16]


# workflows with structures A -> B -> C with multiple connections


def test_wf_3nd_8(tmp_path: Path):
    """workflow with three tasks A->B->C vs two tasks A->C with multiple connections"""

    @workflow.define(outputs=["out1", "out2", "out1a", "out2a"])
    def Worky(zip):

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

    worky = Worky(zip=[["test1", "test3", "test5"], ["test2", "test4", "test6"]])

    with Submitter(worker="cf") as sub:
        res = sub(worky)

    assert (
        res.outputs.out1
        == res.outputs.out1a
        == [["test1", "test3", "test5"], ["test2", "test4", "test6"]]
    )
    assert res.outputs.out2 == res.outputs.out2a == ["Hoi", "Hoi"]


# workflows with Left and Right part in splitters A -> B (L&R parts of the splitter)


def test_wf_ndstLR_1(worker: str, tmp_path: Path):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has its own simple splitter
    and the  Left part from the first task should be added
    """

    @workflow.define
    def Worky(x, y):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        mult = workflow.add(Multiply(x=add2.out).split("y", y=y), name="mult")
        return mult.out

    worky = Worky(x=[1, 2], y=[11, 12])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the splitter is created properly
    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == ["_add2", "mult.y"]
    assert wf["mult"].state.splitter_rpn == ["add2.x", "mult.y", "*"]

    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]
    assert outputs.out == [33, 36, 44, 48]


def test_wf_ndstLR_1a(worker: str, tmp_path: Path):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own splitter)
    """

    @workflow.define
    def Worky(x, y):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        mult = workflow.add(
            Multiply(x=add2.out).split(["_add2", "y"], y=y), name="mult"
        )
        return mult.out

    worky = Worky(x=[1, 2], y=[11, 12])

    # checking if the splitter is created properly
    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == ["_add2", "mult.y"]
    assert wf["mult"].state.splitter_rpn == ["add2.x", "mult.y", "*"]

    # expected: [({"add2.x": 1, "mult.y": 11}, 33), ({"add2.x": 1, "mult.y": 12}, 36),
    #            ({"add2.x": 2, "mult.y": 11}, 44), ({"add2.x": 2, "mult.y": 12}, 48)]

    outputs = worky(worker=worker, cache_root=tmp_path)
    assert outputs.out == [33, 36, 44, 48]


def test_wf_ndstLR_2(worker: str, tmp_path: Path):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has its own outer splitter
    and the  Left part from the first task should be added
    """

    @workflow.define
    def Worky(x, y, z):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        addvar = workflow.add(
            FunAddVar3(a=add2.out).split(["b", "c"], b=y, c=z), name="addvar"
        )
        return addvar.out

    worky = Worky(x=[1, 2, 3], y=[10, 20], z=[100, 200])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the splitter is created properly
    wf = Workflow.construct(worky)
    assert wf["addvar"].state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf["addvar"].state.splitter_rpn == [
        "add2.x",
        "addvar.b",
        "addvar.c",
        "*",
        "*",
    ]

    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert outputs.out == [
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


def test_wf_ndstLR_2a(worker: str, tmp_path: Path):
    """Test workflow with 2 tasks, splitters on tasks levels
    The second task has splitter that has Left part (from previous state)
    and the Right part (it's own outer splitter)
    """

    @workflow.define
    def Worky(x, y, z):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        addvar = workflow.add(
            FunAddVar3(a=add2.out).split(["_add2", ["b", "c"]], b=y, c=z), name="addvar"
        )

        return addvar.out

    worky = Worky(x=[1, 2, 3], y=[10, 20], z=[100, 200])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the splitter is created properly
    wf = Workflow.construct(worky)
    assert wf["addvar"].state.splitter == ["_add2", ["addvar.b", "addvar.c"]]
    assert wf["addvar"].state.splitter_rpn == [
        "add2.x",
        "addvar.b",
        "addvar.c",
        "*",
        "*",
    ]

    # expected: [({"add2.x": 1, "mult.b": 10, "mult.c": 100}, 113),
    #            ({"add2.x": 1, "mult.b": 10, "mult.c": 200}, 213),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 100}, 123),
    #            ({"add2.x": 1, "mult.b": 20, "mult.c": 200}, 223),
    #            ...]
    assert outputs.out == [
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


def test_wf_ndstinner_1(worker: str, tmp_path: Path):
    """workflow with 2 tasks,
    the second task has inner splitter
    """

    @workflow.define(outputs=["out_list", "out"])
    def Worky(x: int):
        list = workflow.add(ListOutput(x=x))
        add2 = workflow.add(Add2().split("x", x=list.out), name="add2")
        return list.out, add2.out

    worky = Worky(x=1)

    wf = Workflow.construct(worky)
    assert wf["add2"].state.splitter == "add2.x"
    assert wf["add2"].state.splitter_rpn == ["add2.x"]

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out_list == [1, 2, 3]
    assert outputs.out == [3, 4, 5]


def test_wf_ndstinner_2(worker: str, tmp_path: Path):
    """workflow with 2 tasks,
    the second task has two inputs and inner splitter from one of the input
    """

    @workflow.define(outputs=["out_list", "out"])
    def Worky(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply(y=y).split("x", x=list.out), name="mult")
        return list.out, mult.out

    worky = Worky(x=1, y=10)  #

    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == "mult.x"
    assert wf["mult"].state.splitter_rpn == ["mult.x"]

    assert outputs.out_list == [1, 2, 3]
    assert outputs.out == [10, 20, 30]


def test_wf_ndstinner_3(worker: str, tmp_path: Path):
    """workflow with 2 tasks,
    the second task has two inputs and outer splitter that includes an inner field
    """

    @workflow.define(outputs=["out_list", "out"])
    def Worky(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply().split(["x", "y"], x=list.out, y=y), name="mult")
        return list.out, mult.out

    worky = Worky(x=1, y=[10, 100])

    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == ["mult.x", "mult.y"]
    assert wf["mult"].state.splitter_rpn == ["mult.x", "mult.y", "*"]

    assert outputs.out_list == [1, 2, 3]
    assert outputs.out == [10, 100, 20, 200, 30, 300]


def test_wf_ndstinner_4(worker: str, tmp_path: Path):
    """workflow with 3 tasks,
    the second task has two inputs and inner splitter from one of the input,
    the third task has no its own splitter
    """

    @workflow.define(outputs=["out_list", "out"])
    def Worky(x, y):
        list = workflow.add(ListOutput(x=x))
        mult = workflow.add(Multiply(y=y).split("x", x=list.out), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return list.out, add2.out

    worky = Worky(x=1, y=10)

    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == "mult.x"
    assert wf["mult"].state.splitter_rpn == ["mult.x"]
    assert wf["add2"].state.splitter == "_mult"
    assert wf["add2"].state.splitter_rpn == ["mult.x"]

    assert outputs.out_list == [1, 2, 3]
    assert outputs.out == [12, 22, 32]


@pytest.mark.flaky(reruns=3)
def test_wf_ndstinner_5(worker: str, tmp_path: Path):
    """workflow with 3 tasks,
    the second task has two inputs and inner splitter from one of the input,
    (inner input come from the first task that has its own splitter,
    there is a inner_container_ndim)
    the third task has no new splitter
    """

    @workflow.define(outputs=["out_list", "out_mult", "out_add"])
    def Worky(x, y, b):
        list = workflow.add(ListOutput().split("x", x=x), name="list")
        mult = workflow.add(Multiply().split(["y", "x"], x=list.out, y=y), name="mult")
        addvar = workflow.add(FunAddVar(a=mult.out).split("b", b=b), name="addvar")
        return list.out, mult.out, addvar.out

    worky = Worky(x=[1, 2], y=[10, 100], b=[3, 5])

    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert wf["mult"].state.splitter == ["_list", ["mult.y", "mult.x"]]
    assert wf["mult"].state.splitter_rpn == ["list.x", "mult.y", "mult.x", "*", "*"]
    assert wf["addvar"].state.splitter == ["_mult", "addvar.b"]
    assert wf["addvar"].state.splitter_rpn == [
        "list.x",
        "mult.y",
        "mult.x",
        "*",
        "*",
        "addvar.b",
        "*",
    ]

    assert outputs.out_list == [[1, 2, 3], [2, 4, 6]]
    assert outputs.out_mult == [
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
    assert outputs.out_add == [
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


def test_wf_st_singl_1(worker: str, tmp_path: Path):
    """workflow with two tasks, only one input is in the splitter and combiner"""

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")

        return add2.out

    worky = Worky(y=11).split("x", x=[1, 2]).combine("x")

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [13, 24]


def test_wf_ndst_singl_1(worker: str, tmp_path: Path):
    """workflow with two tasks, outer splitter and combiner on tasks level;
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x), name="mult")
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"), name="add2")
        return add2.out

    worky = Worky(x=[1, 2], y=11)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [13, 24]


def test_wf_st_singl_2(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the workflow level
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2(x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky(y=11).split("x", x=[1, 2, 3])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [39, 52, 65]


def test_wf_ndst_singl_2(worker: str, tmp_path: Path):
    """workflow with three tasks, third one connected to two previous tasks,
    splitter on the tasks levels
    only one input is part of the splitter, the other is a single value
    """

    @workflow.define
    def Worky(x, y):
        add2x = workflow.add(Add2().split("x", x=x), name="add2x")
        add2y = workflow.add(Add2(x=y), name="add2y")
        mult = workflow.add(Multiply(x=add2x.out, y=add2y.out), name="mult")
        return mult.out

    worky = Worky(x=[1, 2, 3], y=11)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert len(outputs.out) == 3
    assert outputs.out == [39, 52, 65]


# workflows with structures worky(A)


def test_wfasnd_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task and no splitter
    """

    @workflow.define
    def Wfnd1(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd1(x=x))
        return wfnd.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == 4


def test_wfasnd_wfinp_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task and no splitter
    input set for the main workflow
    """

    @workflow.define
    def Wfnd1A(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd1A(x=x))
        return wfnd.out

    worky = Worky(x=2)

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert worky._hash == checksum_before

    assert outputs.out == 4


def test_wfasnd_wfndupdate(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task and no splitter
    wfasnode input is updated to use the main workflow input
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    worky = Worky(x=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == 5


def test_wfasnd_wfndupdate_rerun(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task and no splitter
    wfasnode is run first and later is
    updated to use the main workflow input
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    wfnd = Wfnd(x=2)

    with Submitter(worker=worker, cache_root=tmp_path) as sub:
        sub(wfnd)

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    worky = Worky(x=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == 5

    # adding another layer of workflow
    @workflow.define
    def WorkyO(x):
        worky = workflow.add(Worky(x=x))
        return worky.out

    wf_o = WorkyO(x=4)

    outputs = wf_o(worker=worker, cache_root=tmp_path)

    assert outputs.out == 6


def test_wfasnd_st_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task,
    splitter for wfnd
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x).split(x=x))
        return wfnd.out

    worky = Worky(x=[2, 4])

    checksum_before = worky._hash
    outputs = worky(worker=worker, cache_root=tmp_path)

    wf = Workflow.construct(worky)
    assert worky._hash == checksum_before

    assert outputs.out == [4, 6]


def test_wfasnd_st_updatespl_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task,
    splitter for wfnd is set after add
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x).split(x=x))
        return wfnd.out

    worky = Worky(x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [4, 6]


def test_wfasnd_ndst_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task,
    splitter for node
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    worky = Worky(x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [4, 6]


def test_wfasnd_ndst_updatespl_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task,
    splitter for node added after add
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2().split("x", x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x))
        return wfnd.out

    worky = Worky(x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [4, 6]


def test_wfasnd_wfst_1(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with one task,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd1B(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd1B(x=x))
        return wfnd.out

    worky = Worky().split("x", x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert outputs.out[0] == 4
    assert outputs.out[1] == 6


# workflows with structures worky(A) -> B


def test_wfasnd_st_2(worker: str, tmp_path: Path):
    """workflow as a node,
    the main workflow has two tasks,
    splitter for wfnd
    """

    @workflow.define
    def Wfnd(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        return mult.out

    @workflow.define
    def Worky(x, y):
        wfnd = workflow.add(Wfnd(x=x, y=y))
        add2 = workflow.add(Add2().split(x=wfnd.out), name="add2")
        return add2.out

    worky = Worky(x=[2, 4], y=[1, 10])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert outputs.out == [4, 42]


def test_wfasnd_wfst_2(worker: str, tmp_path: Path):
    """workflow as a node,
    the main workflow has two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        return mult.out

    @workflow.define
    def Worky(x, y):
        wfnd = workflow.add(Wfnd(x=x, y=y))
        add2 = workflow.add(Add2(x=wfnd.out), name="add2")
        return add2.out

    worky = Worky().split(("x", "y"), x=[2, 4], y=[1, 10])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert outputs.out[0] == 4
    assert outputs.out[1] == 42


# workflows with structures A -> worky(B)


def test_wfasnd_ndst_3(worker: str, tmp_path: Path):
    """workflow as the second node,
    the main workflow has two tasks,
    splitter for the first task
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out))
        return wfnd.out

    worky = Worky(x=[2, 4], y=[1, 10])

    outputs = worky(cache_root=tmp_path, worker=worker)

    # assert outputs._cache_dir.exists()

    assert outputs.out == [4, 42]


def test_wfasnd_wfst_3(worker: str, tmp_path: Path):
    """workflow as the second node,
    the main workflow has two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")

        wfnd = workflow.add(Wfnd(x=mult.out))

        return wfnd.out

    worky = Worky().split(("x", "y"), x=[2, 4], y=[1, 10])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert outputs.out[0] == 4
    assert outputs.out[1] == 42


# workflows with structures wfns(A->B)


def test_wfasnd_4(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with two tasks and no splitter
    """

    @workflow.define
    def Wfnd(x):
        add2_1st = workflow.add(Add2(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=2))
        return wfnd.out

    worky = Worky(x=2)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == 6


def test_wfasnd_ndst_4(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with two tasks,
    splitter for node
    """

    @workflow.define
    def Wfnd4(x):
        add2_1st = workflow.add(Add2().split(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd4(x=x))
        return wfnd.out

    worky = Worky(x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert outputs.out == [6, 8]


def test_wfasnd_wfst_4(worker: str, tmp_path: Path):
    """workflow as a node
    workflow-node with two tasks,
    splitter for the main workflow
    """

    @workflow.define
    def Wfnd4A(x):
        add2_1st = workflow.add(Add2(x=x), name="add2_1st")
        add2_2nd = workflow.add(Add2(x=add2_1st.out), name="add2_2nd")
        return add2_2nd.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd4A(x=x))
        return wfnd.out

    worky = Worky().split("x", x=[2, 4])

    outputs = worky(worker=worker, cache_root=tmp_path)

    # assert outputs._cache_dir.exists()

    assert outputs.out == [6, 8]


# Testing caching


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir(worker: str, tmp_path: Path):
    """worky with provided cache_root using pytest tmp_path"""
    cache_root = tmp_path / "test_wf_cache_1"
    cache_root.mkdir()

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 8 == outputs.out

    shutil.rmtree(cache_root)


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachedir_relativepath(tmp_path, worker):
    """worky with provided cache_root as relative path"""
    cache_root = tmp_path / "test_wf_cache_2"
    cache_root.mkdir()

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky = Worky(x=2, y=3)

    outputs = worky(worker=worker, cache_root=tmp_path)

    assert 8 == outputs.out

    shutil.rmtree(cache_root)


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations(worker: str, tmp_path: Path):
    """
    Two identical wfs with provided cache_root;
    the second worky has readonly_caches and should not recompute the results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # checking execution time (for unix and cf)
    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking if the second worky didn't run again


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_a(worker: str, tmp_path: Path):
    """
    the same as previous test, but workflows differ
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):
        mult = workflow.add(Divide(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 2 == results2.outputs.out

    # checking if both cache_dirs are created
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_b(worker: str, tmp_path: Path):
    """
    the same as previous test, but the 2nd workflows has two outputs
    (connected to the same task output);
    the task should not be run and it should be fast,
    but the worky itself is triggered and the new output dir is created
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define(outputs=["out", "out_pr"])
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out, add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out == results2.outputs.out_pr

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # execution time for second run should be much shorter
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking if the second worky didn't run again
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_setoutputchange(worker: str, tmp_path: Path):
    """
    the same as previous test, but worky output names differ,
    the tasks should not be run and it should be fast,
    but the worky itself is triggered and the new output dir is created
    (the second worky has updated name in its Output)
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define(outputs=["out1"])
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out  # out1

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out1

    @workflow.define(outputs=["out2"])
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out  # out2

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking execution time (the second worky should be fast, nodes do not have to rerun)
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create worky itself)
        assert t2 < max(1, t1 - 1)

    # both worky cache_dirs should be created
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_setoutputchange_a(worker: str, tmp_path: Path):
    """
    the same as previous test, but worky names and output names differ,
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define(outputs=["out1"])
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out  # out1

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out1

    @workflow.define(outputs=["out2"])
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        assert t1 > 2
        # testing relative values (windows or slurm takes much longer to create worky itself)
        assert t2 < max(1, t1 - 1)

    # both worky cache_dirs should be created
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_forcererun(worker: str, tmp_path: Path):
    """
    Two identical wfs with provided cache_root;
    the second worky has readonly_caches,
    but submitter is called with rerun=True, so should recompute
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root2) as sub:
        results2 = sub(worky2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking execution time
        assert t1 > 2
        assert t2 > 2

    # checking if the second worky didn't run again
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_wftaskrerun_propagateTrue(
    worker: str, tmp_path: Path
):
    """
    Two identical wfs with provided cache_root and readonly_caches for the second one;
    submitter doesn't have rerun, but the second worky has rerun=True,
    propagate_rerun is True as default, so everything should be rerun
    """
    cache_root1 = tmp_path / "test_wf_cache1"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache2"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # checking if the second worky runs again
    assert results1.cache_dir != results2.cache_dir

    # everything has to be recomputed
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 2

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # runtime for recomputed workflows should be about the same
        assert abs(t1 - t2) < t1 / 2


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_wftaskrerun_propagateFalse(
    worker: str, tmp_path: Path
):
    """
    Two identical wfs with provided cache_root and readonly_caches for the second one;
    submitter doesn't have rerun, but the second worky has rerun=True,
    propagate_rerun is set to False, so worky will be triggered,
    but tasks will not have rerun, so will use the previous results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker,
        cache_root=cache_root2,
        readonly_caches=cache_root1,
        propagate_rerun=False,
    ) as sub:
        results2 = sub(worky2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # checking if the second worky runs again
    assert results1.cache_dir != results2.cache_dir

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # tasks should not be recomputed
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 0


@pytest.mark.xfail(
    reason=(
        "Cannot specify tasks within a workflow to be rerun, maybe rerun could take a "
        "list of task names instead"
    )
)
@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_taskrerun_wfrerun_propagateFalse(
    worker: str, tmp_path: Path
):
    """
    Two identical wfs with provided cache_root, and readonly_caches for the second worky;
    submitter doesn't have rerun, but worky has rerun=True,
    since propagate_rerun=False, only tasks that have rerun=True will be rerun
    """
    cache_root1 = tmp_path / "cache1"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "cache2"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        # rerun on the task level needed (wf["propagate_rerun"] is False)
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    with Submitter(
        worker=worker,
        cache_root=cache_root2,
        readonly_caches=cache_root1,
        propagate_rerun=False,
    ) as sub:
        results2 = sub(worky2, rerun=True)  # rerun will not be propagated to each task)
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    assert results1.cache_dir != results2.cache_dir
    # the second task should be recomputed
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 1

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_nodecachelocations(worker: str, tmp_path: Path):
    """
    Two wfs with different input, but the second node has the same input;
    the second worky has readonly_caches and should recompute the worky,
    but without recomputing the second node
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out), name="add2")
        return add2.out

    worky1 = Worky1(x=3)

    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 12 == results1.outputs.out

    @workflow.define
    def Worky2(x, y=None):

        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2)

    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 12 == results2.outputs.out

    # checking if the second worky runs again, but runs only one task
    assert results1.cache_dir != results2.cache_dir
    # the second worky should rerun one task
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 1


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_nodecachelocations_upd(worker: str, tmp_path: Path):
    """
    Two wfs with different input, but the second node has the same input;
    the second worky has readonly_caches (set after adding tasks) and should recompute,
    but without recomputing the second node
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out), name="add2")
        return add2.out

    worky1 = Worky1(x=3)

    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 12 == results1.outputs.out

    @workflow.define
    def Worky2(x, y=None):
        ten = workflow.add(Ten(x=x))
        add2 = workflow.add(Add2(x=ten.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2)

    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 12 == results2.outputs.out

    # checking if the second worky runs again, but runs only one task
    assert results1.cache_dir != results2.cache_dir
    # the second worky should have only one task run
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 1


@pytest.mark.flaky(reruns=3)
def test_wf_state_cachelocations(worker: str, tmp_path: Path):
    """
    Two identical wfs (with states) with provided cache_root;
    the second worky has readonly_caches and should not recompute the results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories

    # checking if the second worky didn't run again
    # checking all directories

    assert results1.cache_dir == results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_state_cachelocations_forcererun(worker: str, tmp_path: Path):
    """
    Two identical wfs (with states) with provided cache_root;
    the second worky has readonly_caches,
    but submitter is called with rerun=True, so should recompute
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root2) as sub:
        results2 = sub(worky2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories

    # checking if the second worky run again
    # checking all directories


@pytest.mark.flaky(reruns=3)
def test_wf_state_cachelocations_updateinp(worker: str, tmp_path: Path):
    """
    Two identical wfs (with states) with provided cache_root;
    the second worky has readonly_caches and should not recompute the results
    (the lazy input of the node is updated to the correct one,
    i.e. the same as in worky1, after adding the node to the worky)
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out[0] == 8
    assert results1.outputs.out[1] == 82

    @workflow.define
    def Worky2(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2().split(("x", "y"), x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories

    # checking if the second worky didn't run again
    # checking all directories


@pytest.mark.flaky(reruns=3)
def test_wf_state_n_nostate_cachelocations(worker: str, tmp_path: Path):
    """
    Two wfs with provided cache_root, the first one has no state, the second has;
    the second worky has readonly_caches and should not recompute only one element
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert results1.outputs.out == 8

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2().split(("x", "y"), x=[2, 20], y=[3, 4])

    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert results2.outputs.out[0] == 8
    assert results2.outputs.out[1] == 82


def test_wf_nostate_cachelocations_updated(worker: str, tmp_path: Path):
    """
    Two identical wfs with provided cache_root;
    the second worky has readonly_caches in init,
     that is later overwritten in Submitter.__call__;
    the readonly_caches from call doesn't exist so the second task should run again
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root1_empty = tmp_path / "test_wf_cache3_empty"
    cache_root1_empty.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    t0 = time.time()
    # changing readonly_caches to non-existing dir
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1_empty
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results2.outputs.out

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking if both worky run
    assert results1.cache_dir != results2.cache_dir


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_cachelocations_recompute(worker: str, tmp_path: Path):
    """
    Two wfs with the same inputs but slightly different graph;
    the second worky should recompute the results,
    but the second node should use the results from the first worky (has the same input)
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])

    assert 8 == results1.outputs.out

    @workflow.define
    def Worky2(x, y):

        # different argument assignment
        mult = workflow.add(Multiply(x=y, y=x), name="mult")
        add2 = workflow.add(Add2(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=2, y=3)

    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])

    assert 8 == results2.outputs.out

    # checking if both dir exists
    assert results1.cache_dir != results2.cache_dir

    # the second worky should have only one task run
    assert len(list(Path(cache_root1).glob("python-*"))) == 2
    assert len(list(Path(cache_root2).glob("python-*"))) == 1


@pytest.mark.flaky(reruns=3)
def test_wf_ndstate_cachelocations(worker: str, tmp_path: Path):
    """
    Two wfs with identical inputs and node states;
    the second worky has readonly_caches and should not recompute the results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


@pytest.mark.flaky(reruns=3)
def test_wf_ndstate_cachelocations_forcererun(worker: str, tmp_path: Path):
    """
    Two wfs with identical inputs and node states;
    the second worky has readonly_caches,
    but submitter is called with rerun=True, so should recompute
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root2) as sub:
        results2 = sub(worky2, rerun=True)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories

    # checking if the second worky run again


@pytest.mark.flaky(reruns=3)
def test_wf_ndstate_cachelocations_updatespl(worker: str, tmp_path: Path):
    """
    Two wfs with identical inputs and node state (that is set after adding the node!);
    the second worky has readonly_caches and should not recompute the results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)

    # checking all directories

    # checking if the second worky didn't run again
    # checking all directories


@pytest.mark.flaky(reruns=3)
def test_wf_ndstate_cachelocations_recompute(worker: str, tmp_path: Path):
    """
    Two wfs (with nodes with states) with provided cache_root;
    the second worky has readonly_caches and should not recompute the results
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()
    cache_root2 = tmp_path / "test_wf_cache4"
    cache_root2.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply().split(("x", "y"), x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert results1.outputs.out == [8, 82]

    @workflow.define
    def Worky2(x, y):

        mult = workflow.add(Multiply().split(["x", "y"], x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky2 = Worky2(x=[2, 20], y=[3, 4])

    t0 = time.time()
    with Submitter(
        worker=worker, cache_root=cache_root2, readonly_caches=cache_root1
    ) as sub:
        results2 = sub(worky2)

    assert not results2.errored, "\n".join(results2.errors["error message"])
    t2 = time.time() - t0

    assert results2.outputs.out == [8, 10, 62, 82]

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 > 2

    # checking all directories

    # checking if the second worky didn't run again
    # checking all directories


@pytest.mark.flaky(reruns=3)
def test_wf_nostate_runtwice_usecache(worker: str, tmp_path: Path):
    """
    running workflow (without state) twice,
    the second run should use the results from the first one
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1(x=2, y=3)

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out
    # checkoing cache_dir after the first run

    # saving the content of the cache dit after the first run
    cache_root_content = os.listdir(cache_root1)

    # running workflow the second time
    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results1.outputs.out
    # checking if no new directory is created
    assert cache_root_content == os.listdir(cache_root1)

    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


def test_wf_state_runtwice_usecache(worker: str, tmp_path: Path):
    """
    running workflow with a state twice,
    the second run should use the results from the first one
    """
    cache_root1 = tmp_path / "test_wf_cache3"
    cache_root1.mkdir()

    @workflow.define
    def Worky1(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        add2 = workflow.add(Add2Wait(x=mult.out), name="add2")
        return add2.out

    worky1 = Worky1().split(("x", "y"), x=[2, 20], y=[3, 30])

    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t1 = time.time() - t0

    assert 8 == results1.outputs.out[0]
    assert 602 == results1.outputs.out[1]

    # checkoing cache_dir after the first run
    assert results1.cache_dir.exists()

    # saving the content of the cache dit after the first run
    cache_root_content = os.listdir(results1.job.cache_root)

    # running workflow the second time
    t0 = time.time()
    with Submitter(worker=worker, cache_root=cache_root1) as sub:
        results1 = sub(worky1)

    assert not results1.errored, "\n".join(results1.errors["error message"])
    t2 = time.time() - t0

    assert 8 == results1.outputs.out[0]
    assert 602 == results1.outputs.out[1]
    # checking if no new directory is created
    assert cache_root_content == os.listdir(results1.job.cache_root)
    # for win and dask/slurm the time for dir creation etc. might take much longer
    if not sys.platform.startswith("win") and worker == "cf":
        # checking the execution time
        assert t1 > 2
        assert t2 < max(1, t1 - 1)


@pytest.fixture
def create_tasks():
    @workflow.define
    def Worky(x):
        t1 = workflow.add(Add2(x=x), name="t1")
        t2 = workflow.add(Multiply(x=t1.out, y=2), name="t2")
        return t2.out

    worky = Worky(x=1)
    workflow_obj = Workflow.construct(worky)
    t1 = workflow_obj["t1"]
    t2 = workflow_obj["t2"]
    return worky, t1, t2


def test_workflow_combine1(tmp_path: Path):
    @workflow.define(outputs=["out_pow", "out_iden1", "out_iden2"])
    def Worky1(a, b):
        power = workflow.add(Power().split(["a", "b"], a=a, b=b), name="power")
        identity1 = workflow.add(
            Identity(x=power.out).combine("power.a"), name="identity1"
        )
        identity2 = workflow.add(
            Identity(x=identity1.out).combine("power.b"), name="identity2"
        )
        return power.out, identity1.out, identity2.out

    worky1 = Worky1(a=[1, 2], b=[2, 3])
    outputs = worky1()

    assert outputs.out_pow == [1, 1, 4, 8]
    assert outputs.out_iden1 == [[1, 4], [1, 8]]
    assert outputs.out_iden2 == [[1, 4], [1, 8]]


def test_workflow_combine2(tmp_path: Path):
    @workflow.define(outputs=["out_pow", "out_iden"])
    def Worky1(a, b):
        power = workflow.add(
            Power().split(["a", "b"], a=a, b=b).combine("a"), name="power"
        )
        identity = workflow.add(Identity(x=power.out).combine("power.b"))
        return power.out, identity.out

    worky1 = Worky1(a=[1, 2], b=[2, 3])
    outputs = worky1(cache_root=tmp_path)

    assert outputs.out_pow == [[1, 4], [1, 8]]
    assert outputs.out_iden == [[1, 4], [1, 8]]


def test_wf_resultfile_1(worker: str, tmp_path: Path):
    """workflow with a file in the result, file should be copied to the worky dir"""

    @workflow.define(outputs=["wf_out"])
    def Worky(x):
        writefile = workflow.add(FunWriteFile(filename=x))

        return writefile.out  #

    worky = Worky(x="file_1.txt")
    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the file exists and if it is in the Worky directory
    wf_out = outputs.wf_out.fspath
    assert wf_out.exists()
    assert wf_out == outputs._cache_dir / "file_1.txt"


def test_wf_resultfile_2(worker: str, tmp_path: Path):
    """workflow with a list of files in the worky result,
    all files should be copied to the worky dir
    """

    @workflow.define(outputs=["wf_out"])
    def Worky(x):
        writefile = workflow.add(FunWriteFileList(filename_list=x))

        return writefile.out  #

    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    worky = Worky(x=file_list)
    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the file exists and if it is in the Worky directory
    for ii, file in enumerate(outputs.wf_out):
        assert file.fspath.exists()
        assert file.fspath == outputs._cache_dir / file_list[ii]


def test_wf_resultfile_3(worker: str, tmp_path: Path):
    """workflow with a dictionaries of files in the worky result,
    all files should be copied to the worky dir
    """

    @workflow.define(outputs=["wf_out"])
    def Worky(x):
        writefile = workflow.add(FunWriteFileList2Dict(filename_list=x))

        return writefile.out  #

    file_list = ["file_1.txt", "file_2.txt", "file_3.txt"]
    worky = Worky(x=file_list)
    outputs = worky(worker=worker, cache_root=tmp_path)

    # checking if the file exists and if it is in the Worky directory
    for key, val in outputs.wf_out.items():
        if key == "random_int":
            assert val == 20
        else:
            assert val.fspath.exists()
            ii = int(key.split("_")[1])
            assert val.fspath == outputs._cache_dir / file_list[ii]


def test_wf_upstream_error1(tmp_path: Path):
    """workflow with two tasks, task2 dependent on an task1 which raised an error"""

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    worky = Worky(x="hi")  # TypeError for adding str and int

    with pytest.raises(RuntimeError) as excinfo:
        worky(worker="cf", cache_root=tmp_path)
    assert "addvar1" in str(excinfo.value)
    assert "failed with errors" in str(excinfo.value)


def test_wf_upstream_error2(tmp_path: Path):
    """task2 dependent on task1, task1 errors, workflow-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    worky = Worky().split(
        "x", x=[1, "hi"]
    )  # workflow-level split TypeError for adding str and int

    with pytest.raises(Exception) as excinfo:
        worky(worker="cf", cache_root=tmp_path)
    assert "addvar1" in str(excinfo.value)
    assert "failed with errors" in str(excinfo.value)


@pytest.mark.flaky(reruns=2)  # when slurm
def test_wf_upstream_error3(tmp_path: Path):
    """task2 dependent on task1, task1 errors, task-level split on task 1
    goal - workflow finish running, one output errors but the other doesn't
    """

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType().split("a", a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        return addvar2.out

    worky = Worky(x=[1, "hi"])  # TypeError for adding str and int
    with pytest.raises(RuntimeError) as excinfo:
        worky(worker="cf", cache_root=tmp_path)
    assert "addvar1" in str(excinfo.value)
    assert "failed with errors" in str(excinfo.value)


def test_wf_upstream_error4(tmp_path: Path):
    """workflow with one task, which raises an error"""

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        return addvar1.out

    worky = Worky(x="hi")  # TypeError for adding str and int
    with pytest.raises(Exception) as excinfo:
        worky(worker="cf", cache_root=tmp_path)
    assert "failed with errors" in str(excinfo.value)
    assert "addvar1" in str(excinfo.value)


def test_wf_upstream_error5(tmp_path: Path):
    """nested workflow with one task, which raises an error"""

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        return addvar1.out  # wf_out

    @workflow.define
    def WfMain(x):
        worky = workflow.add(Worky(x=x))
        return worky.out

    wf_main = WfMain(x="hi")  # TypeError for adding str and int

    with pytest.raises(Exception) as excinfo:
        wf_main(worker="cf", cache_root=tmp_path)

    assert "addvar1" in str(excinfo.value)
    assert "failed with errors" in str(excinfo.value)


def test_wf_upstream_error6(tmp_path: Path):
    """nested workflow with two tasks, the first one raises an error"""

    @workflow.define(outputs=["wf_out"])
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")

        return addvar2.out  #

    @workflow.define
    def WfMain(x):
        worky = workflow.add(Worky(x=x))
        return worky.wf_out

    wf_main = WfMain(x="hi")  # TypeError for adding str and int

    with pytest.raises(RuntimeError) as excinfo:
        wf_main(worker="cf", cache_root=tmp_path)

    assert "addvar1" in str(excinfo.value)
    assert "failed with errors" in str(excinfo.value)


def test_wf_upstream_error7(tmp_path: Path):
    """
    workflow with three sequential tasks, the first task raises an error
    the last task is set as the workflow output
    """

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar3.out

    worky = Worky(x="hi")  # TypeError for adding str and int

    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "addvar1" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["addvar1"].has_errored is True
    assert list(graph["addvar2"].unrunnable.values()) == [[graph["addvar1"]]]
    assert list(graph["addvar3"].unrunnable.values()) == [[graph["addvar2"]]]


def test_wf_upstream_error7a(tmp_path: Path):
    """
    workflow with three sequential tasks, the first task raises an error
    the second task is set as the workflow output
    """

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar3.out

    worky = Worky(x="hi")  # TypeError for adding str and int
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "addvar1" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["addvar1"].has_errored is True
    assert list(graph["addvar2"].unrunnable.values()) == [[graph["addvar1"]]]
    assert list(graph["addvar3"].unrunnable.values()) == [[graph["addvar2"]]]


def test_wf_upstream_error7b(tmp_path: Path):
    """
    workflow with three sequential tasks, the first task raises an error
    the second and the third tasks are set as the workflow output
    """

    @workflow.define(outputs=["out1", "out2"])
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addvar3 = workflow.add(FunAddVarDefaultNoType(a=addvar2.out), name="addvar3")
        return addvar2.out, addvar3.out  #

    worky = Worky(x="hi")  # TypeError for adding str and int
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "addvar1" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["addvar1"].has_errored is True
    assert list(graph["addvar2"].unrunnable.values()) == [[graph["addvar1"]]]
    assert list(graph["addvar3"].unrunnable.values()) == [[graph["addvar2"]]]


def test_wf_upstream_error8(tmp_path: Path):
    """workflow with three tasks, the first one raises an error, so 2 others are removed"""

    @workflow.define(outputs=["out1", "out2"])
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")

        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addvar1.out), name="addvar2")
        addtwo = workflow.add(FunAddTwo(a=addvar1.out), name="addtwo")
        return addvar2.out, addtwo.out  #

    worky = Worky(x="hi")  # TypeError for adding str and int
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "addvar1" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["addvar1"].has_errored is True

    assert list(graph["addvar2"].unrunnable.values()) == [[graph["addvar1"]]]
    assert list(graph["addtwo"].unrunnable.values()) == [[graph["addvar1"]]]


def test_wf_upstream_error9(tmp_path: Path):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    the errored branch is connected to the workflow output
    """

    @workflow.define
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        follow_err = workflow.add(FunAddVarDefaultNoType(a=err.out), name="follow_err")
        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        workflow.add(FunAddVarDefaultNoType(a=addtwo.out))
        return follow_err.out  # out1

    worky = Worky(x=2)
    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "err" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["err"].has_errored is True
    assert list(graph["follow_err"].unrunnable.values()) == [[graph["err"]]]


def test_wf_upstream_error9a(tmp_path: Path):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    the branch without error is connected to the workflow output
    so the workflow finished clean
    """

    @workflow.define(outputs=["out1"])
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefault(a=x), name="addvar1")
        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        workflow.add(FunAddVarDefault(a=err.out), name="follow_err")
        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        addvar2 = workflow.add(FunAddVarDefault(a=addtwo.out), name="addvar2")
        return addvar2.out

    worky = Worky(x=2)

    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "err" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["err"].has_errored is True
    assert list(graph["follow_err"].unrunnable.values()) == [[graph["err"]]]


def test_wf_upstream_error9b(tmp_path: Path):
    """
    workflow with five tasks with two "branches",
    one branch has an error, the second is fine
    both branches are connected to the workflow output
    """

    @workflow.define(outputs=["out1", "out2"])
    def Worky(x):
        addvar1 = workflow.add(FunAddVarDefaultNoType(a=x), name="addvar1")
        err = workflow.add(FunAddVarNoType(a=addvar1.out, b="hi"), name="err")
        follow_err = workflow.add(FunAddVarDefaultNoType(a=err.out), name="follow_err")
        addtwo = workflow.add(FunAddTwoNoType(a=addvar1.out), name="addtwo")
        addvar2 = workflow.add(FunAddVarDefaultNoType(a=addtwo.out), name="addvar2")
        return follow_err.out, addvar2.out

    worky = Worky(x=2)

    with Submitter(worker="cf", cache_root=tmp_path) as sub:
        results = sub(worky)
    error_message = "".join(results.errors["error message"])
    assert "err" in error_message
    assert "failed with errors" in error_message

    graph = results.job.return_values["exec_graph"]
    assert graph["err"].has_errored is True
    assert list(graph["follow_err"].unrunnable.values()) == [[graph["err"]]]


def exporting_graphs(worky, name, out_dir):
    """helper function to run dot to create png/pdf files from dotfiles"""
    # exporting the simple graph
    dotfile_pr, formatted_dot = plot_workflow(worky, out_dir, export=True, name=name)
    assert len(formatted_dot) == 1
    assert formatted_dot[0] == dotfile_pr.with_suffix(".png")
    assert formatted_dot[0].exists()
    print("\n png of a simple graph in: ", formatted_dot[0])
    # exporting nested graph
    dotfile_pr, formatted_dot = plot_workflow(
        worky, out_dir, plot_type="nested", export=["pdf", "png"], name=f"{name}_nest"
    )
    assert len(formatted_dot) == 2
    assert formatted_dot[0] == dotfile_pr.with_suffix(".pdf")
    assert formatted_dot[0].exists()
    print("\n pdf of the nested graph in: ", formatted_dot[0])
    # detailed graph
    dotfile_pr, formatted_dot = plot_workflow(
        worky, out_dir, plot_type="detailed", export="pdf", name=f"{name}_det"
    )
    assert len(formatted_dot) == 1
    assert formatted_dot[0] == dotfile_pr.with_suffix(".pdf")
    assert formatted_dot[0].exists()
    print("\n pdf of the detailed graph in: ", formatted_dot[0])


@pytest.mark.parametrize("splitter", [None, "x"])
def test_graph_simple(tmp_path, splitter):
    """creating a set of graphs, worky with two nodes"""

    @workflow.define
    def Worky(x=1, y=2):
        mult_1 = workflow.add(Multiply(x=x, y=y), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    worky = Worky().split(splitter, x=[1, 2])

    # simple graph
    dotfile_s = plot_workflow(worky, tmp_path, name="simple")
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult_1" in dotstr_s_lines
    assert "mult_2" in dotstr_s_lines
    assert "add2" in dotstr_s_lines
    assert "mult_1 -> add2" in dotstr_s_lines


@pytest.mark.parametrize("splitter", [None, "x"])
def test_graph_nested(tmp_path, splitter):
    """creating a set of graphs, worky with two nodes"""

    @workflow.define
    def Worky(x=1, y=2):
        mult_1 = workflow.add(Multiply(x=x, y=y), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    worky = Worky().split(splitter, x=[1, 2])

    # nested graph (should have the same elements)
    dotfile_n = plot_workflow(
        worky, tmp_path, lazy=["x", "y"], plot_type="nested", name="nested"
    )
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult_1" in dotstr_n_lines
    assert "mult_2" in dotstr_n_lines
    assert "add2" in dotstr_n_lines
    assert "mult_1 -> add2" in dotstr_n_lines


@pytest.mark.parametrize("splitter", [None, "x"])
def test_graph_detailed(tmp_path, splitter):
    """creating a set of graphs, worky with two nodes"""

    @workflow.define
    def Worky(x=1, y=2):
        mult_1 = workflow.add(Multiply(x=x, y=y), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    worky = Worky().split(splitter, x=[1, 2])

    # detailed graph
    dotfile_d = plot_workflow(
        worky, tmp_path, plot_type="detailed", lazy=["x", "y"], name="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult_1:out -> struct_add2:x;" in dotstr_d_lines


@pytest.mark.skipif(not DOT_FLAG, reason="dot not available")
@pytest.mark.parametrize("splitter", [None, "x"])
def test_graph_export_dot(tmp_path, splitter):
    """creating a set of graphs, worky with two nodes"""

    @workflow.define
    def Worky(x=1, y=2):
        mult_1 = workflow.add(Multiply(x=x, y=y), name="mult_1")
        workflow.add(Multiply(x=x, y=x), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    worky = Worky().split(splitter, x=[1, 2])

    name = f"graph_{sys._getframe().f_code.co_name}"
    exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_1st(tmp_path: Path):
    """creating a set of graphs, worky with two nodes
    some nodes have splitters, should be marked with blue color
    """

    @workflow.define
    def Worky(x, y):
        mult_1 = workflow.add(Multiply(y=y).split("x", x=x), name="mult_1")
        workflow.add(Multiply(x=y, y=y), name="mult_2")
        add2 = workflow.add(Add2(x=mult_1.out), name="add2")
        return add2.out

    worky = Worky(x=[1, 2], y=2)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult_1 [color=blue]" in dotstr_s_lines
    assert "mult_2" in dotstr_s_lines
    assert "add2 [color=blue]" in dotstr_s_lines
    assert "mult_1 -> add2 [color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult_1 [color=blue]" in dotstr_n_lines
    assert "mult_2" in dotstr_n_lines
    assert "add2 [color=blue]" in dotstr_n_lines
    assert "mult_1 -> add2 [color=blue]" in dotstr_n_lines

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult_1:out -> struct_add2:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_1st_cmb(tmp_path: Path):
    """creating a set of graphs, worky with three nodes
    the first one has a splitter, the second has a combiner, so the third one is stateless
    first two nodes should be blue and the arrow between them should be blue
    """

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x), name="mult")
        add2 = workflow.add(Add2(x=mult.out).combine("mult.x"), name="add2")
        sum = workflow.add(ListSum(x=add2.out), name="sum")
        return sum.out

    worky = Worky(x=[1, 2], y=2)
    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_s_lines
    assert "add2 [color=blue]" in dotstr_s_lines
    assert "sum" in dotstr_s_lines
    assert "mult -> add2 [color=blue]" in dotstr_s_lines
    assert "add2 -> sum" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_n_lines
    assert "add2 [color=blue]" in dotstr_n_lines
    assert "sum" in dotstr_n_lines
    assert "mult -> add2 [color=blue]" in dotstr_n_lines
    assert "add2 -> sum" in dotstr_n_lines

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_add2:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_2(tmp_path: Path):
    """creating a graph, worky with one workflow as a node"""

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x), name="wfnd")
        return wfnd.out

    worky = Worky(x=2)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "wfnd [shape=box]" in dotstr_s_lines

    # nested graph
    dotfile = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_lines = dotfile.read_text().split("\n")
    assert "subgraph cluster_wfnd {" in dotstr_lines
    assert "add2" in dotstr_lines

    # detailed graph
    dotfile_d = plot_workflow(worky, out_dir=tmp_path, lazy=["x"], plot_type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x}}"];'
        in dotstr_d_lines
    )

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_2st(tmp_path: Path):
    """creating a set of graphs, worky with one workflow as a node
    the inner workflow has a state, so should be blue
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x):
        wfnd = workflow.add(Wfnd(x=x).split("x", x=x), name="wfnd")
        return wfnd.out

    worky = Worky(x=[1, 2])

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "wfnd [shape=box, color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "subgraph cluster_wfnd {" in dotstr_s_lines
    assert "color=blue" in dotstr_s_lines
    assert "add2" in dotstr_s_lines

    # detailed graph
    dotfile_d = plot_workflow(worky, out_dir=tmp_path, lazy=["x"], plot_type="detailed")
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x}}"];'
        in dotstr_d_lines
    )
    assert "struct_wfnd:out -> struct_Worky_out:out;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_3(tmp_path: Path):
    """creating a set of graphs, worky with two nodes (one node is a workflow)"""

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x, y=1):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    worky = Worky(x=2)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "mult -> wfnd" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult" in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2" in dotstr_n_lines

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_3st(tmp_path: Path):
    """creating a set of graphs, worky with two nodes (one node is a workflow)
    the first node has a state and it should be passed to the second node
    (blue node and a wfasnd, and blue arrow from the node to the wfasnd)
    """

    @workflow.define
    def Wfnd(x):
        add2 = workflow.add(Add2(x=x), name="add2")
        return add2.out

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(y=y).split("x", x=x), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    worky = Worky(x=[1, 2], y=2)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_s_lines
    assert "wfnd [shape=box, color=blue]" in dotstr_s_lines
    assert "mult -> wfnd [color=blue]" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    assert "mult [color=blue]" in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2" in dotstr_n_lines

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_mult:out -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_4(tmp_path: Path):
    """creating a set of graphs, worky with two nodes (one node is a workflow with two nodes
    inside). Connection from the node to the inner workflow.
    """

    @workflow.define
    def Wfnd(x):
        add2_a = workflow.add(Add2(x=x), name="add2_a")
        add2_b = workflow.add(Add2(x=add2_a.out), name="add2_b")
        return add2_b.out

    @workflow.define
    def Worky(x, y):
        mult = workflow.add(Multiply(x=x, y=y), name="mult")
        wfnd = workflow.add(Wfnd(x=mult.out), name="wfnd")
        return wfnd.out

    worky = Worky(x=2, y=3)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "mult -> wfnd" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    for el in ["mult", "add2_a", "add2_b"]:
        assert el in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2_a -> add2_b" in dotstr_n_lines
    assert "mult -> add2_a [lhead=cluster_wfnd]"

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_Worky:y -> struct_mult:y;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


def test_graph_5(tmp_path: Path):
    """creating a set of graphs, worky with two nodes (one node is a workflow with two nodes
    inside). Connection from the inner workflow to the node.
    """

    @workflow.define
    def Wfnd(x):
        add2_a = workflow.add(Add2(x=x), name="add2_a")
        add2_b = workflow.add(Add2(x=add2_a.out), name="add2_b")
        return add2_b.out

    @workflow.define
    def Worky(x, y):
        wfnd = workflow.add(Wfnd(x=x), name="wfnd")
        mult = workflow.add(Multiply(x=wfnd.out, y=y), name="mult")
        return mult.out

    worky = Worky(x=2, y=3)

    # simple graph
    dotfile_s = plot_workflow(worky, out_dir=tmp_path)
    dotstr_s_lines = dotfile_s.read_text().split("\n")
    assert "mult" in dotstr_s_lines
    assert "wfnd [shape=box]" in dotstr_s_lines
    assert "wfnd -> mult" in dotstr_s_lines

    # nested graph
    dotfile_n = plot_workflow(worky, out_dir=tmp_path, plot_type="nested")
    dotstr_n_lines = dotfile_n.read_text().split("\n")
    for el in ["mult", "add2_a", "add2_b"]:
        assert el in dotstr_n_lines
    assert "subgraph cluster_wfnd {" in dotstr_n_lines
    assert "add2_a -> add2_b" in dotstr_n_lines
    assert "add2_b -> mult [ltail=cluster_wfnd]"

    # detailed graph
    dotfile_d = plot_workflow(
        worky, out_dir=tmp_path, lazy=["x", "y"], plot_type="detailed"
    )
    dotstr_d_lines = dotfile_d.read_text().split("\n")
    assert (
        'struct_Worky [color=red, label="{WORKFLOW INPUT: | {<x> x | <y> y}}"];'
        in dotstr_d_lines
    )
    assert "struct_Worky:x -> struct_wfnd:x;" in dotstr_d_lines

    if DOT_FLAG:
        name = f"graph_{sys._getframe().f_code.co_name}"
        exporting_graphs(worky=worky, name=name, out_dir=tmp_path)


@pytest.mark.timeout(20)
def test_duplicate_input_on_split_wf(tmp_path: Path):
    """checking if the workflow gets stuck if it has to run two tasks with equal checksum;
    This can occur when splitting on a list containing duplicate values.
    """
    text = ["test"] * 2

    @python.define
    def printer(a):
        return a

    @workflow.define(outputs=["out1"])
    def Worky(text):
        printer1 = workflow.add(printer(a=text))
        return printer1.out  #

    worky = Worky().split(text=text)

    outputs = worky(worker="cf")

    assert outputs.out1[0] == "test" and outputs.out1[0] == "test"


@pytest.mark.timeout(40)
def test_inner_outer_wf_duplicate(tmp_path: Path):
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

    # Inner Worky
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

    assert res.outputs.res2[0] == 23 and res.outputs.res2[1] == 23


@pytest.mark.flaky(reruns=3)
def test_rerun_errored(tmp_path, capfd):
    """Test rerunning a workflow containing errors.
    Only the errored tasks and workflow should be rerun"""

    @python.define
    def PassOdds(x):
        if x % 2 == 0:
            print(f"x={x}, running x%2 = {x % 2} (even error)\n")
            raise ValueError("even error")
        else:
            print(f"x={x}, running x%2 = {x % 2}\n")
            return x

    @workflow.define
    def WorkyPassOdds(x):
        pass_odds = workflow.add(PassOdds().split("x", x=x))
        return pass_odds.out

    worky = WorkyPassOdds(x=[1, 2, 3, 4, 5])

    print("Starting run 1")
    with pytest.raises(RuntimeError):
        # Must be cf to get the error from all tasks, otherwise will only get the first error
        worky(worker="cf", cache_root=tmp_path)

    print("Starting run 2")
    with pytest.raises(RuntimeError):
        worky(worker="cf", cache_root=tmp_path)

    out, err = capfd.readouterr()
    stdout_lines = out.splitlines()

    tasks_run = 0
    errors_found = 0

    for line in stdout_lines:
        if "running x%2" in line:
            tasks_run += 1
        if "(even error)" in line:
            errors_found += 1

    # There should have been 5 messages of the form "x%2 = XXX" after calling task() the first time
    # and another 2 messagers after calling the second time
    assert tasks_run == 7
    assert errors_found == 4


def test_wf_state_arrays(tmp_path, worker):
    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def Worky(x: ty.List[int], y: int):

        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                in_list=x,
            ).split(scalar=x),
            name="A",
        )

        B = workflow.add(  # Worky is still split over "x", combined over "x" on out
            ListMultSum(
                scalar=A.sum,
                in_list=A.products,
            ).combine("A.scalar"),
            name="B",
        )

        C = workflow.add(  # Worky "
            ListMultSum(
                scalar=y,
                in_list=B.sum,
            ),
            name="C",
        )

        D = workflow.add(  # Worky is split again, this time over C.products
            ListMultSum(
                in_list=x,
            )
            .split(scalar=C.products)
            .combine("scalar"),
            name="D",
        )

        E = workflow.add(  # Worky is finally combined again into a single node
            ListMultSum(scalar=y, in_list=D.sum),
            name="E",
        )

        return E.sum, E.products

    worky = Worky(x=[1, 2, 3, 4], y=10)

    outputs = worky(cache_root=tmp_path, worker=worker)
    assert outputs.alpha == 3000000
    assert outputs.beta == [100000, 400000, 900000, 1600000]


def test_wf_input_typing_fail():

    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def MismatchInputWf(x: int, y: int):
        ListMultSum(
            scalar=y,
            in_list=y,
            name="A",
        )

    with pytest.raises(TypeError, match="Incorrect type for field in 'y'"):
        MismatchInputWf(x=1, y=[1, 2, 3])


def test_wf_output_typing_fail(tmp_path: Path):

    @workflow.define(outputs={"alpha": int, "beta": ty.List[int]})
    def MismatchOutputWf(x: int, y: ty.List[int]):
        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                scalar=x,
                in_list=y,
            )
        )
        return A.products, A.products

    worky = MismatchOutputWf(x=1, y=[1, 2, 3])

    with pytest.raises(
        TypeError,
        match="Incorrect type for lazy field in 'alpha' field of MismatchOutputWf.Outputs interface",
    ):
        worky(cache_root=tmp_path)


def test_wf_input_output_typing(tmp_path: Path):
    @workflow.define(outputs={"sum": int, "products": ty.List[int]})
    def Worky(x: int, y: ty.List[int]):
        A = workflow.add(  # Split over workflow input "x" on "scalar" input
            ListMultSum(
                scalar=x,
                in_list=y,
            )
        )
        return A.sum, A.products

    outputs = Worky(x=10, y=[1, 2, 3, 4])(cache_root=tmp_path)
    assert outputs.sum == 100
    assert outputs.products == [10, 20, 30, 40]


def test_plot_exec_workflow(tmp_path: Path):

    # Example python tasks
    @python.define
    def Add(a, b):
        return a + b

    @python.define
    def Mul(a, b):
        return a * b

    @workflow.define
    def BasicWorkflow(a, b):
        add = workflow.add(Add(a=a, b=b))
        mul = workflow.add(Mul(a=add.out, b=b))
        return mul.out

    wf = BasicWorkflow(a=2, b=3)

    plot_workflow(BasicWorkflow, tmp_path / "plot-out")

    outputs = wf(cache_root=tmp_path / "cache")
    assert outputs.out == 15
