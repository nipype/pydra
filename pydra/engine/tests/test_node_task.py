import os
import shutil
import attrs
import typing as ty
import numpy as np
import time
from pathlib import Path
import pytest
from fileformats.generic import File
from pydra.design import python, workflow

from .utils import (
    FunAddTwo,
    FunAddVar,
    FunAddVarNone,
    FunAddVarDefault,
    Moment,
    FunDiv,
    FunDict,
    FunFile,
    FunFileList,
    Op4Var,
)

from pydra.engine.core import Task
from pydra.engine.specs import TaskDef
from pydra.engine.state import State
from pydra.utils.typing import StateArray
from pydra.engine.submitter import Submitter
from pydra.engine.core import Workflow


@workflow.define
def IdentityWorkflow(a: int) -> int:

    @python.define
    def Identity(a):
        return a

    a = workflow.add(Identity(a=a))
    return a.out


def get_state(task: TaskDef, name="NA") -> State:
    """helper function to get the state of the task once it has been added to workflow"""
    identity_workflow = IdentityWorkflow(a=1)
    wf = Workflow.construct(identity_workflow, dont_cache=True)
    wf.add(task, name=name)
    node = wf[name]
    if node.state:
        node.state.prepare_states()
        node.state.prepare_inputs()
    return node.state


def num_python_cache_dirs(cache_path: Path) -> int:
    return len(list(cache_path.glob("python-*")))


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_outputs")
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


def test_task_init_1a():
    with pytest.raises(TypeError):
        FunAddTwo("NA")


def test_task_init_2():
    """task with a name and inputs"""
    nn = FunAddTwo(a=3)
    # adding NA to the name of the variable
    assert nn.a == 3
    state = get_state(nn)
    assert state is None


@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, states_ind, states_val",
    [("a", "NA.a", ["NA.a"], [{"NA.a": 0}, {"NA.a": 1}], [{"NA.a": 3}, {"NA.a": 5}])],
)
@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_init_3(
    splitter, state_splitter, state_rpn, states_ind, states_val, input_type
):
    """task with inputs and splitter"""
    a_in = [3, 5]
    if input_type == "array":
        a_in = np.array(a_in)

    nn = FunAddTwo().split(splitter, a=a_in)

    assert np.allclose(nn.a, [3, 5])
    state = get_state(nn)
    assert state.splitter == state_splitter
    assert state.splitter_rpn == state_rpn

    assert state.states_ind == states_ind
    assert state.states_val == states_val


@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, states_ind, states_val",
    [
        (
            ("a", "b"),
            ("NA.a", "NA.b"),
            ["NA.a", "NA.b", "."],
            [{"NA.a": 0, "NA.b": 0}, {"NA.a": 1, "NA.b": 1}],
            [{"NA.a": 3, "NA.b": 10}, {"NA.a": 5, "NA.b": 20}],
        ),
        (
            ["a", "b"],
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b", "*"],
            [
                {"NA.a": 0, "NA.b": 0},
                {"NA.a": 0, "NA.b": 1},
                {"NA.a": 1, "NA.b": 0},
                {"NA.a": 1, "NA.b": 1},
            ],
            [
                {"NA.a": 3, "NA.b": 10},
                {"NA.a": 3, "NA.b": 20},
                {"NA.a": 5, "NA.b": 10},
                {"NA.a": 5, "NA.b": 20},
            ],
        ),
    ],
)
@pytest.mark.parametrize("input_type", ["list", "array", "mixed"])
def test_task_init_3a(
    splitter, state_splitter, state_rpn, states_ind, states_val, input_type
):
    """task with inputs and splitter"""
    a_in, b_in = [3, 5], [10, 20]
    if input_type == "array":
        a_in, b_in = np.array(a_in), np.array(b_in)
    elif input_type == "mixed":
        a_in = np.array(a_in)
    nn = FunAddVar().split(splitter, a=a_in, b=b_in)
    state = get_state(nn)

    assert np.allclose(nn.a, [3, 5])
    assert np.allclose(nn.b, [10, 20])
    assert state.splitter == state_splitter
    assert state.splitter_rpn == state_rpn

    assert state.states_ind == states_ind
    assert state.states_val == states_val


def test_task_init_4():
    """task with interface splitter and inputs set in the split method"""
    nn = FunAddTwo()
    nn = nn.split("a", a=[3, 5])
    state = get_state(nn)
    assert np.allclose(nn.a, [3, 5])

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]

    assert state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert state.states_val == [{"NA.a": 3}, {"NA.a": 5}]


def test_task_init_4b():
    """updating splitter using overwrite=True"""
    nn = FunAddTwo()
    nn = nn.split("a", a=[1, 2])
    nn = nn.split("a", a=[3, 5], overwrite=True)
    state = get_state(nn)
    assert np.allclose(nn.a, [3, 5])

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]

    assert state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert state.states_val == [{"NA.a": 3}, {"NA.a": 5}]


def test_task_init_4c():
    """trying to set splitter twice without using overwrite"""
    nn = FunAddVar().split("b", b=[1, 2])
    state = get_state(nn)
    with pytest.raises(Exception) as excinfo:
        nn.split("a", a=[3, 5])
    assert "Cannot overwrite existing splitter" in str(excinfo.value)

    assert state.splitter == "NA.b"


def test_task_init_4d():
    """trying to set the same splitter twice without using overwrite
    if the splitter is the same, the exception shouldn't be raised
    """
    nn = FunAddTwo().split("a", a=[3, 5])
    nn = nn.split("a", a=[3, 5], overwrite=True)
    state = get_state(nn)
    assert state.splitter == "NA.a"


def test_task_init_5():
    """task with inputs, splitter and combiner"""
    nn = FunAddVar().split(["a", "b"], a=[3, 5], b=[1, 2]).combine("b")
    state = get_state(nn)

    assert state.splitter == ["NA.a", "NA.b"]
    assert state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert state.combiner == ["NA.b"]

    assert state.splitter_final == "NA.a"
    assert state.splitter_rpn_final == ["NA.a"]

    assert state.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert state.states_val == [
        {"NA.a": 3, "NA.b": 1},
        {"NA.a": 3, "NA.b": 2},
        {"NA.a": 5, "NA.b": 1},
        {"NA.a": 5, "NA.b": 2},
    ]

    assert state.final_combined_ind_mapping == {0: [0, 1], 1: [2, 3]}


def test_task_init_5a():
    """updating combiner using overwrite=True"""
    nn = FunAddVar().split(["a", "b"], a=[3, 5], b=[1, 2]).combine("b")
    nn = nn.combine("a", overwrite=True)
    state = get_state(nn)

    assert state.splitter == ["NA.a", "NA.b"]
    assert state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert state.combiner == ["NA.a"]

    assert state.splitter_final == "NA.b"
    assert state.splitter_rpn_final == ["NA.b"]

    assert state.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert state.states_val == [
        {"NA.a": 3, "NA.b": 1},
        {"NA.a": 3, "NA.b": 2},
        {"NA.a": 5, "NA.b": 1},
        {"NA.a": 5, "NA.b": 2},
    ]

    assert state.final_combined_ind_mapping == {0: [0, 2], 1: [1, 3]}


def test_task_init_5b():
    """updating combiner without using overwrite"""
    nn = FunAddVar().split(["a", "b"], a=[3, 5], b=[1, 2]).combine("b")
    state = get_state(nn)
    with pytest.raises(Exception) as excinfo:
        nn.combine("a")
    assert "Attempting to overwrite existing combiner" in str(excinfo.value)

    assert state.combiner == ["NA.b"]


def test_task_init_5c():
    """trying to set the same combiner twice without using overwrite
    if the combiner is the same, the exception shouldn't be raised
    """
    nn = FunAddVar().split(["a", "b"], a=[3, 5], b=[1, 2]).combine("b")
    state = get_state(nn)
    nn = nn.combine("b", overwrite=True)

    assert state.splitter == ["NA.a", "NA.b"]
    assert state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert state.combiner == ["NA.b"]

    assert state.splitter_final == "NA.a"
    assert state.splitter_rpn_final == ["NA.a"]


def test_task_init_6():
    """task with splitter, but the input is an empty list"""
    nn = FunAddTwo()
    nn = nn.split("a", a=[])
    state = get_state(nn)
    assert nn.a == []

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]

    assert state.states_ind == []
    assert state.states_val == []


def test_task_init_7(tmp_path):
    """task with a dictionary of files as an input, checking checksum"""
    file1 = tmp_path / "file1.txt"
    with open(file1, "w") as f:
        f.write("hello")

    file2 = tmp_path / "file2.txt"
    with open(file2, "w") as f:
        f.write("from pydra\n")

    nn1 = FunFileList(filename_list=[file1, file2])
    hash1 = nn1._hash

    # changing the content of the file
    time.sleep(2)  # need the mtime to be different
    file2 = tmp_path / "file2.txt"
    with open(file2, "w") as f:
        f.write("from pydra")

    nn2 = FunFileList(filename_list=[file1, file2])
    hash2 = nn2._hash

    # the checksum should be different - content of file2 is different
    assert hash1 != hash2


def test_task_init_8():
    """task without setting the input, the value should be set to attrs.NOTHING"""
    nn = FunAddTwo()
    assert nn.a is attrs.NOTHING


def test_task_init_9():
    """task without setting the input, but using the default avlue from function"""
    nn1 = FunAddVarDefault(a=2)
    assert nn1.b == 1

    nn2 = FunAddVarDefault(a=2, b=1)
    assert nn2.b == 1
    # both tasks should have the same checksum
    assert nn1._hash == nn2._hash


def test_task_error(tmp_path):
    func = FunDiv(a=1, b=0)
    with pytest.raises(ZeroDivisionError):
        func(cache_dir=tmp_path)
    assert (next(tmp_path.iterdir()) / "_error.pklz").exists()


# Tests for tasks without state (i.e. no splitter)


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1(plugin_dask_opt, tmp_path):
    """task without splitter"""
    nn = FunAddTwo(a=3)

    assert np.allclose(nn.a, [3])
    state = get_state(nn)
    assert state is None

    with Submitter(worker=plugin_dask_opt, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results
    assert results.outputs.out == 5

    # checking the output_dir
    assert results.output_dir.exists()


def test_task_nostate_1_call(tmp_path):
    """task without splitter"""
    nn = FunAddTwo(a=3)
    with Submitter(cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])
    # checking the results

    assert results.outputs.out == 5
    # checking the output_dir
    assert results.output_dir.exists()


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1_call_subm(plugin_dask_opt, tmp_path):
    """task without splitter"""
    nn = FunAddTwo(a=3)

    assert np.allclose(nn.a, [3])
    state = get_state(nn)
    assert state is None

    with Submitter(worker=plugin_dask_opt, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == 5
    # checking the output_dir
    assert results.output_dir.exists()


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1_call_plug(plugin_dask_opt, tmp_path):
    """task without splitter"""
    nn = FunAddTwo(a=3)

    assert np.allclose(nn.a, [3])
    state = get_state(nn)
    assert state is None

    with Submitter(cache_dir=tmp_path, worker=plugin_dask_opt) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == 5
    # checking the output_dir
    assert results.output_dir.exists()


def test_task_nostate_2(plugin, tmp_path):
    """task with a list as an input, but no splitter"""
    nn = Moment(n=3, lst=[2, 3, 4])

    assert np.allclose(nn.n, [3])
    assert np.allclose(nn.lst, [2, 3, 4])
    state = get_state(nn)
    assert state is None

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == 33
    # checking the output_dir
    assert results.output_dir.exists()


def test_task_nostate_3(plugin, tmp_path):
    """task with a dictionary as an input"""
    nn = FunDict(d={"a": "ala", "b": "bala"})

    assert nn.d == {"a": "ala", "b": "bala"}

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == "a:ala_b:bala"
    # checking the output_dir
    assert results.output_dir.exists()


def test_task_nostate_4(plugin, tmp_path):
    """task with a dictionary as an input"""
    file1 = tmp_path / "file.txt"
    with open(file1, "w") as f:
        f.write("hello from pydra\n")

    nn = FunFile(filename=file1)

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == "hello from pydra\n"
    # checking the output_dir
    assert results.output_dir.exists()


def test_task_nostate_5(tmp_path):
    """task with a dictionary of files as an input"""
    file1 = tmp_path / "file1.txt"
    with open(file1, "w") as f:
        f.write("hello")

    file2 = tmp_path / "file2.txt"
    with open(file2, "w") as f:
        f.write("from pydra\n")

    nn = FunFileList(filename_list=[file1, file2])

    outputs = nn()

    # checking the results

    assert outputs.out == "hello from pydra\n"


def test_task_nostate_6():
    """checking if the function gets the None value"""
    nn = FunAddVarNone(a=2, b=None)
    assert nn.b is None
    outputs = nn()
    assert outputs.out == 2


def test_task_nostate_6a_exception():
    """checking if the function gets the attrs.Nothing value"""
    nn = FunAddVarNone(a=2)
    assert nn.b is attrs.NOTHING
    with pytest.raises(ValueError) as excinfo:
        nn()
    assert "Mandatory field 'b' is not set" in str(excinfo.value)


def test_task_nostate_7():
    """using the default value from the function for b input"""
    nn = FunAddVarDefault(a=2)
    assert nn.b == 1
    outputs = nn()
    assert outputs.out == 3


# Testing caching for tasks without states


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachedir(plugin_dask_opt, tmp_path):
    """task with provided cache_dir using pytest tmp_path"""
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    nn = FunAddTwo(a=3)
    state = get_state(nn)
    assert np.allclose(nn.a, [3])
    assert state is None

    with Submitter(worker=plugin_dask_opt, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == 5


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachedir_relativepath(tmp_path, plugin_dask_opt):
    """task with provided cache_dir as relative path"""
    os.chdir(tmp_path)
    cache_dir = "test_task_nostate"
    (tmp_path / cache_dir).mkdir()

    nn = FunAddTwo(a=3)
    assert np.allclose(nn.a, [3])
    state = get_state(nn)
    assert state is None

    with Submitter(worker=plugin_dask_opt, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results.outputs.out == 5

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachelocations(plugin_dask_opt, tmp_path):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3)
    with Submitter(worker=plugin_dask_opt, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    nn2 = FunAddTwo(a=3)
    with Submitter(
        worker=plugin_dask_opt, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results2.outputs.out == 5

    # checking if the second task didn't run the interface again
    assert results.output_dir == results2.output_dir


def test_task_nostate_cachelocations_forcererun(plugin, tmp_path):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3)
    with Submitter(worker=plugin, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    nn2 = FunAddTwo(a=3)
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results2 = sub(nn2, rerun=True)

    # checking the results

    assert results2.outputs.out == 5

    # checking if the second task rerun the interface
    assert results.output_dir.exists()
    assert results2.output_dir.exists()


def test_task_nostate_cachelocations_nosubmitter(tmp_path):
    """
    Two identical tasks (that are run without submitter!) with provided cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3)
    nn(cache_dir=cache_dir)

    nn2 = FunAddTwo(a=3)
    outputs2 = nn2(cache_dir=cache_dir2, cache_locations=cache_dir)

    # checking the results

    assert outputs2.out == 5

    # checking if the second task didn't run the interface again
    assert num_python_cache_dirs(cache_dir) == 1
    assert not num_python_cache_dirs(cache_dir2)


def test_task_nostate_cachelocations_nosubmitter_forcererun(tmp_path):
    """
    Two identical tasks (that are run without submitter!) with provided cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3)
    nn(cache_dir=cache_dir)

    nn2 = FunAddTwo(a=3)
    outputs2 = nn2(rerun=True, cache_dir=cache_dir2, cache_locations=cache_dir)

    # checking the results

    assert outputs2.out == 5

    # checking if the second task run the interface again
    assert num_python_cache_dirs(cache_dir) == 1
    assert num_python_cache_dirs(cache_dir2)


def test_task_nostate_cachelocations_updated(plugin, tmp_path):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations in init,
     that is later overwritten in Submitter.__call__;
    the cache_locations passed to call doesn't exist so the second task should run again
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir1 = tmp_path / "test_task_nostate1"
    cache_dir1.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3)
    with Submitter(worker=plugin, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    nn2 = FunAddTwo(a=3)
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results1 = sub(nn2)
    assert not results1.errored, "\n".join(results.errors["error message"])

    # updating cache location to non-existing dir
    with Submitter(
        worker=plugin, cache_locations=cache_dir1, cache_dir=tmp_path
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results2.outputs.out == 5

    # checking if both tasks run interface
    assert results.output_dir == results1.output_dir
    assert results.output_dir != results2.output_dir


# Tests for tasks with states (i.e. with splitter)


@pytest.mark.flaky(reruns=2)  # when dask
@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_state_1(plugin_dask_opt, input_type, tmp_path):
    """task with the simplest splitter"""
    a_in = [3, 5]
    if input_type == "array":
        a_in = np.array(a_in)

    nn = FunAddTwo().split("a", a=a_in)
    state = get_state(nn)

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert (nn.a == np.array([3, 5])).all()

    with Submitter(worker=plugin_dask_opt, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


def test_task_state_1a(plugin, tmp_path):
    """task with the simplest splitter (inputs set separately)"""
    nn = FunAddTwo()
    nn = nn.split("a", a=[1, 2])
    nn.a = StateArray([3, 5])

    state = get_state(nn)

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert (nn.a == np.array([3, 5])).all()

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


def test_task_state_singl_1(plugin, tmp_path):
    """Tasks with two inputs and a splitter (no combiner)
    one input is a single value, the other is in the splitter and combiner
    """
    nn = FunAddVar(b=10).split("a", a=[3, 5])
    state = get_state(nn)

    assert nn.a == [3, 5]
    assert nn.b == 10
    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert state.splitter_final == "NA.a"
    assert state.splitter_rpn_final == ["NA.a"]

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results
    expected = [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 10}, 15)]

    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]
    # checking the output_dir
    assert results.output_dir.exists()


@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, expected, expected_ind",
    [
        (
            ("a", "b"),
            ("NA.a", "NA.b"),
            ["NA.a", "NA.b", "."],
            [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
            [({"NA.a": 0, "NA.b": 0}, 13), ({"NA.a": 1, "NA.b": 1}, 25)],
        ),
        (
            ["a", "b"],
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b", "*"],
            [
                ({"NA.a": 3, "NA.b": 10}, 13),
                ({"NA.a": 3, "NA.b": 20}, 23),
                ({"NA.a": 5, "NA.b": 10}, 15),
                ({"NA.a": 5, "NA.b": 20}, 25),
            ],
            [
                ({"NA.a": 0, "NA.b": 0}, 13),
                ({"NA.a": 0, "NA.b": 1}, 23),
                ({"NA.a": 1, "NA.b": 0}, 15),
                ({"NA.a": 1, "NA.b": 1}, 25),
            ],
        ),
    ],
)
@pytest.mark.parametrize("input_type", ["list", "array", "mixed"])
def test_task_state_2(
    plugin,
    splitter,
    state_splitter,
    state_rpn,
    expected,
    expected_ind,
    input_type,
    tmp_path,
):
    """Tasks with two inputs and a splitter (no combiner)"""
    a_in, b_in = [3, 5], [10, 20]
    if input_type == "array":
        a_in, b_in = np.array(a_in), np.array(b_in)
    elif input_type == "mixed":
        a_in = np.array(a_in)
    nn = FunAddVar().split(splitter, a=a_in, b=b_in)
    state = get_state(nn)

    assert (nn.a == np.array([3, 5])).all()
    assert (nn.b == np.array([10, 20])).all()
    assert state.splitter == state_splitter
    assert state.splitter_rpn == state_rpn
    assert state.splitter_final == state_splitter
    assert state.splitter_rpn_final == state_rpn

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


def test_task_state_3(plugin, tmp_path):
    """task with the simplest splitter, the input is an empty list"""
    nn = FunAddTwo().split("a", a=[])
    state = get_state(nn)

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert nn.a == []

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = []
    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_state_4(plugin, input_type, tmp_path):
    """task with a list as an input, and a simple splitter"""
    lst_in = [[2, 3, 4], [1, 2, 3]]
    if input_type == "array":
        lst_in = np.array(lst_in, dtype=int)
    nn = Moment(n=3).split("lst", lst=lst_in)
    state = get_state(nn)

    assert np.allclose(nn.n, 3)
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])
    assert state.splitter == "NA.lst"

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking that split is done across dim 0
    el_0 = state.states_val[0]["NA.lst"]
    if input_type == "list":
        assert el_0 == [2, 3, 4]
    elif input_type == "array":
        assert el_0 == [2, 3, 4]

    # checking the results

    for i, expected in enumerate([33, 12]):
        assert results.outputs.out[i] == expected


def test_task_state_4a(plugin, tmp_path):
    """task with a tuple as an input, and a simple splitter"""
    nn = Moment(n=3).split("lst", lst=[(2, 3, 4), (1, 2, 3)])
    state = get_state(nn)

    assert np.allclose(nn.n, 3)
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])
    assert state.splitter == "NA.lst"

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    for i, expected in enumerate([33, 12]):
        assert results.outputs.out[i] == expected


def test_task_state_5(plugin, tmp_path):
    """task with a list as an input, and the variable is part of the scalar splitter"""
    nn = Moment().split(("n", "lst"), n=[1, 3], lst=[[2, 3, 4], [1, 2, 3]])
    state = get_state(nn)

    assert np.allclose(nn.n, [1, 3])
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])
    assert state.splitter == ("NA.n", "NA.lst")

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    for i, expected in enumerate([3, 12]):
        assert results.outputs.out[i] == expected


def test_task_state_5_exception(plugin, tmp_path):
    """task with a list as an input, and the variable is part of the scalar splitter
    the shapes are not matching, so exception should be raised
    """
    nn = Moment().split(("n", "lst"), n=[1, 3, 3], lst=[[2, 3, 4], [1, 2, 3]])

    assert np.allclose(nn.n, [1, 3, 3])
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])

    with pytest.raises(Exception) as excinfo:
        get_state(nn)

    assert "shape" in str(excinfo.value)


def test_task_state_6(plugin, tmp_path):
    """ask with a list as an input, and the variable is part of the outer splitter"""
    nn = Moment().split(["n", "lst"], n=[1, 3], lst=[[2, 3, 4], [1, 2, 3]])
    state = get_state(nn)

    assert np.allclose(nn.n, [1, 3])
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])
    assert state.splitter == ["NA.n", "NA.lst"]

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    for i, expected in enumerate([3, 2, 33, 12]):
        assert results.outputs.out[i] == expected


def test_task_state_6a(plugin, tmp_path):
    """ask with a tuple as an input, and the variable is part of the outer splitter"""
    nn = Moment().split(["n", "lst"], n=[1, 3], lst=[(2, 3, 4), (1, 2, 3)])
    state = get_state(nn)

    assert np.allclose(nn.n, [1, 3])
    assert np.allclose(nn.lst, [[2, 3, 4], [1, 2, 3]])
    assert state.splitter == ["NA.n", "NA.lst"]

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    for i, expected in enumerate([3, 2, 33, 12]):
        assert results.outputs.out[i] == expected


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_state_comb_1(plugin_dask_opt, tmp_path):
    """task with the simplest splitter and combiner"""
    nn = FunAddTwo().split(a=[3, 5]).combine(combiner="a")
    state = get_state(nn)

    assert (nn.a == np.array([3, 5])).all()

    assert state.splitter == ["NA.a"]
    assert state.splitter_rpn == ["NA.a"]
    assert state.combiner == ["NA.a"]
    assert state.splitter_final is None
    assert state.splitter_rpn_final == []

    with Submitter(worker=plugin_dask_opt, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    assert state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert state.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    # checking the results

    # fully combined (no nested list)
    assert results.outputs.out == [5, 7]


@pytest.mark.parametrize(
    "splitter, combiner, state_splitter, state_rpn, state_combiner, state_combiner_all, "
    "state_splitter_final, state_rpn_final, expected",  # , expected_val",
    [
        (
            ("a", "b"),
            "a",
            ("NA.a", "NA.b"),
            ["NA.a", "NA.b", "."],
            ["NA.a"],
            ["NA.a", "NA.b"],
            None,
            [],
            [13, 25],
            # [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
        ),
        (
            ("a", "b"),
            "b",
            ("NA.a", "NA.b"),
            ["NA.a", "NA.b", "."],
            ["NA.b"],
            ["NA.a", "NA.b"],
            None,
            [],
            [13, 25],
            # [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
        ),
        (
            ["a", "b"],
            "a",
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b", "*"],
            ["NA.a"],
            ["NA.a"],
            "NA.b",
            ["NA.b"],
            [[13, 15], [23, 25]],
            # [
            #     [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 10}, 15)],
            #     [({"NA.a": 3, "NA.b": 20}, 23), ({"NA.a": 5, "NA.b": 20}, 25)],
            # ],
        ),
        (
            ["a", "b"],
            "b",
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b", "*"],
            ["NA.b"],
            ["NA.b"],
            "NA.a",
            ["NA.a"],
            [[13, 23], [15, 25]],
            # [
            #     [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 3, "NA.b": 20}, 23)],
            #     [({"NA.a": 5, "NA.b": 10}, 15), ({"NA.a": 5, "NA.b": 20}, 25)],
            # ],
        ),
        (
            ["a", "b"],
            ["a", "b"],
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b", "*"],
            ["NA.a", "NA.b"],
            ["NA.a", "NA.b"],
            None,
            [],
            [13, 23, 15, 25],
            # [
            #     ({"NA.a": 3, "NA.b": 10}, 13),
            #     ({"NA.a": 3, "NA.b": 20}, 23),
            #     ({"NA.a": 5, "NA.b": 10}, 15),
            #     ({"NA.a": 5, "NA.b": 20}, 25),
            # ],
        ),
    ],
)
def test_task_state_comb_2(
    plugin,
    splitter,
    combiner,
    state_splitter,
    state_rpn,
    state_combiner,
    state_combiner_all,
    state_splitter_final,
    state_rpn_final,
    expected,
    # expected_val,
    tmp_path,
):
    """Tasks with scalar and outer splitters and  partial or full combiners"""
    nn = FunAddVar().split(splitter, a=[3, 5], b=[10, 20]).combine(combiner=combiner)
    state = get_state(nn)

    assert (nn.a == np.array([3, 5])).all()

    assert state.splitter == state_splitter
    assert state.splitter_rpn == state_rpn
    assert state.combiner == state_combiner

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    assert state.splitter_final == state_splitter_final
    assert state.splitter_rpn_final == state_rpn_final
    assert set(state.current_combiner_all) == set(state_combiner_all)

    # checking the results

    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    # results_verb = nn.result(return_inputs=True)

    if state.splitter_rpn_final:
        for i, res in enumerate(expected):
            assert results.outputs.out == res
        # results_verb
        # for i, res_l in enumerate(expected_val):
        #     for j, res in enumerate(res_l):
        #         assert (results_verb[i][j][0], results_verb[i][j][1].output.out) == res
    # if the combiner is full expected is "a flat list"
    else:
        assert results.outputs.out == expected
        # for i, res in enumerate(expected_val):
        #     assert (results_verb[i][0], results_verb[i][1].output.out) == res


def test_task_state_comb_singl_1(plugin, tmp_path):
    """Tasks with two inputs;
    one input is a single value, the other is in the splitter and combiner
    """
    nn = FunAddVar(b=10).split("a", a=[3, 5]).combine(combiner="a")
    state = get_state(nn)

    assert nn.a == [3, 5]
    assert nn.b == 10
    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert state.combiner == ["NA.a"]
    assert state.splitter_final is None
    assert state.splitter_rpn_final == []

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    assert results.outputs.out == [13, 15]


def test_task_state_comb_3(plugin, tmp_path):
    """task with the simplest splitter, the input is an empty list"""
    nn = FunAddTwo().split("a", a=[]).combine(combiner=["a"])
    state = get_state(nn)

    assert state.splitter == "NA.a"
    assert state.splitter_rpn == ["NA.a"]
    assert nn.a == []

    with Submitter(worker=plugin, cache_dir=tmp_path) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = []
    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


def test_task_state_comb_order(tmp_path):
    """tasks with an outer splitter and various combiner;
    showing the order of results
    """

    # single combiner "a" - will create two lists, first one for b=3, second for b=5
    nn_a = FunAddVar().split(["a", "b"], a=[10, 20], b=[3, 5]).combine(combiner="a")
    state_a = get_state(nn_a)
    assert state_a.combiner == ["NA.a"]

    outputs = nn_a(cache_dir=tmp_path / "cache")
    # combined_results_a = [[res.output.out for res in res_l] for res_l in results_a]
    assert outputs.out == [[13, 23], [15, 25]]

    # single combiner "b" - will create two lists, first one for a=10, second for a=20
    nn_b = FunAddVar().split(["a", "b"], a=[10, 20], b=[3, 5]).combine(combiner="b")
    state_b = get_state(nn_b)
    assert state_b.combiner == ["NA.b"]

    outputs_b = nn_b(cache_dir=tmp_path / "cache_b")
    # combined_results_b = [[res.output.out for res in res_l] for res_l in results_b]
    assert outputs_b.out == [[13, 15], [23, 25]]

    # combiner with both fields ["a", "b"] - will create one list
    nn_ab = (
        FunAddVar().split(["a", "b"], a=[10, 20], b=[3, 5]).combine(combiner=["a", "b"])
    )
    state_ab = get_state(nn_ab)
    assert state_ab.combiner == ["NA.a", "NA.b"]

    outputs_ab = nn_ab(cache_dir=tmp_path / "cache_ab")
    assert outputs_ab.out == [13, 15, 23, 25]

    # combiner with both fields ["b", "a"] - will create the same list as nn_ab
    # no difference in the order for setting combiner
    nn_ba = (
        FunAddVar().split(["a", "b"], a=[10, 20], b=[3, 5]).combine(combiner=["b", "a"])
    )
    state_ba = get_state(nn_ba)
    assert state_ba.combiner == ["NA.b", "NA.a"]

    outputs_ba = nn_ba(cache_dir=tmp_path / "cache_ba")
    assert outputs_ba.out == [13, 15, 23, 25]


# Testing with container dimensions for the input


def test_task_state_contdim_1(tmp_path):
    """task with a spliter and container dimension for one of the value"""
    task_4var = Op4Var(
        a="a1",
    ).split(
        ("b", ["c", "d"]),
        b=[["b1", "b2"], ["b3", "b4"]],
        c=["c1", "c2"],
        d=["d1", "d2"],
        cont_dim={"b": 2},
    )
    outputs = task_4var(cache_dir=tmp_path)
    assert len(outputs.out) == 4
    assert outputs.out[3] == "a1 b4 c2 d2"


def test_task_state_contdim_2(tmp_path):
    """task with a splitter and container dimension for one of the value"""
    task_4var = Op4Var().split(
        ["a", ("b", ["c", "d"])],
        cont_dim={"b": 2},
        a=["a1", "a2"],
        b=[["b1", "b2"], ["b3", "b4"]],
        c=["c1", "c2"],
        d=["d1", "d2"],
    )
    outputs = task_4var(cache_dir=tmp_path)
    assert len(outputs.out) == 8
    assert outputs.out[7] == "a2 b4 c2 d2"


def test_task_state_comb_contdim_1(tmp_path):
    """task with a splitter-combiner, and container dimension for one of the value"""
    task_4var = (
        Op4Var(a="a1")
        .split(
            ("b", ["c", "d"]),
            cont_dim={"b": 2},
            b=[["b1", "b2"], ["b3", "b4"]],
            c=["c1", "c2"],
            d=["d1", "d2"],
        )
        .combine("b")
    )
    outputs = task_4var(cache_dir=tmp_path)
    assert len(outputs.out) == 4
    assert outputs.out[3] == "a1 b4 c2 d2"


def test_task_state_comb_contdim_2(tmp_path):
    """task with a splitter-combiner, and container dimension for one of the value"""
    task_4var = (
        Op4Var()
        .split(
            ["a", ("b", ["c", "d"])],
            a=["a1", "a2"],
            b=[["b1", "b2"], ["b3", "b4"]],
            c=["c1", "c2"],
            d=["d1", "d2"],
            cont_dim={"b": 2},
        )
        .combine("a")
    )
    outputs = task_4var(cache_dir=tmp_path)
    assert len(outputs.out) == 4
    assert outputs.out[3][1] == "a2 b4 c2 d2"


# Testing caching for tasks with states


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_state_cachedir(plugin_dask_opt, tmp_path):
    """task with a state and provided cache_dir using pytest tmp_path"""
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    nn = FunAddTwo().split("a", a=[3, 5])
    state = get_state(nn)

    assert state.splitter == "NA.a"
    assert (nn.a == np.array([3, 5])).all()

    with Submitter(worker=plugin_dask_opt, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results.outputs.out[i] == res[1]


def test_task_state_cachelocations(plugin, tmp_path):
    """
    Two identical tasks with a state and cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(worker=plugin, cache_dir=cache_dir) as sub:
        sub(nn)

    nn2 = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # Would ideally check for all nodes of the workflows
    assert num_python_cache_dirs(cache_dir) == 2
    assert not num_python_cache_dirs(cache_dir2)


def test_task_state_cachelocations_forcererun(plugin, tmp_path):
    """
    Two identical tasks with a state and cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(worker=plugin, cache_dir=cache_dir) as sub:
        sub(nn)

    nn2 = FunAddTwo(a=3).split("a", a=[3, 5])
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results2 = sub(nn2, rerun=True)

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # both workflows should be run
    assert num_python_cache_dirs(cache_dir) == 2
    assert num_python_cache_dirs(cache_dir2) == 2


def test_task_state_cachelocations_updated(plugin, tmp_path):
    """
    Two identical tasks with states and cache_dir;
    the second task has cache_locations in init,
     that is later overwritten in Submitter.__call__;
    the cache_locations from call doesn't exist so the second task should run again
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir1 = tmp_path / "test_task_nostate1"
    cache_dir1.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()

    nn = FunAddTwo().split("a", a=[3, 5])
    with Submitter(worker=plugin, cache_dir=cache_dir) as sub:
        sub(nn)

    nn2 = FunAddTwo().split("a", a=[3, 5])
    with Submitter(
        worker=plugin, cache_dir=cache_dir2, cache_locations=cache_dir1
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2.outputs.out[i] == res[1]

    # both workflows should be run
    assert num_python_cache_dirs(cache_dir) == 2
    assert num_python_cache_dirs(cache_dir2) == 2


def test_task_files_cachelocations(plugin_dask_opt, tmp_path):
    """
    Two identical tasks with provided cache_dir that use file as an input;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmp_path / "test_task_nostate"
    cache_dir.mkdir()
    cache_dir2 = tmp_path / "test_task_nostate2"
    cache_dir2.mkdir()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    input1 = input_dir / "input1.txt"
    input1.write_text("test")
    input2 = input_dir / "input2.txt"
    input2.write_text("test")

    nn = FunFile(filename=input1)
    with Submitter(worker=plugin_dask_opt, cache_dir=cache_dir) as sub:
        results = sub(nn)
    assert not results.errored, "\n".join(results.errors["error message"])

    nn2 = FunFile(filename=input2)
    with Submitter(
        worker=plugin_dask_opt, cache_dir=cache_dir2, cache_locations=cache_dir
    ) as sub:
        results2 = sub(nn2)
    assert not results2.errored, "\n".join(results.errors["error message"])

    # checking the results

    assert results2.outputs.out == "test"

    # checking if the second task didn't run the interface again
    assert results.output_dir == results2.output_dir


class OverriddenContentsFile(File):
    """A class for testing purposes, to that enables you to override the contents
    of the file to allow you to check whether the persistent cache is used."""

    def __init__(
        self,
        fspaths: ty.Iterator[Path],
        contents: ty.Optional[bytes] = None,
        metadata: ty.Dict[str, ty.Any] = None,
    ):
        super().__init__(fspaths, metadata=metadata)
        self._contents = contents

    def byte_chunks(self, **kwargs) -> ty.Generator[ty.Tuple[str, bytes], None, None]:
        if self._contents is not None:
            yield (str(self.fspath), iter([self._contents]))
        else:
            yield from super().byte_chunks(**kwargs)

    @property
    def raw_contents(self):
        if self._contents is not None:
            return self._contents
        return super().raw_contents


def test_task_files_persistentcache(tmp_path):
    """
    Two identical tasks with provided cache_dir that use file as an input;
    the second task has cache_locations and should not recompute the results
    """
    test_file_path = tmp_path / "test_file.txt"
    test_file_path.write_bytes(b"foo")
    cache_dir = tmp_path / "cache-dir"
    cache_dir.mkdir()
    test_file = OverriddenContentsFile(test_file_path)

    @python.define
    def read_contents(x: OverriddenContentsFile) -> bytes:
        return x.raw_contents

    assert read_contents(x=test_file)(cache_dir=cache_dir).out == b"foo"
    test_file._contents = b"bar"
    # should return result from the first run using the persistent cache
    assert read_contents(x=test_file)(cache_dir=cache_dir).out == b"foo"
    time.sleep(2)  # Windows has a 2-second resolution for mtime
    test_file_path.touch()  # update the mtime to invalidate the persistent cache value
    assert (
        read_contents(x=test_file)(cache_dir=cache_dir).out == b"bar"
    )  # returns the overridden value
