import os
import shutil
import attr
import numpy as np
import pytest

from .utils import (
    fun_addtwo,
    fun_addvar,
    fun_addvar_none,
    fun_addvar_default,
    moment,
    fun_div,
    fun_dict,
    fun_file,
    fun_file_list,
)

from ..core import TaskBase
from ..submitter import Submitter


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_outputs")
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


# Tests for tasks initializations
def test_task_init_1():
    """ task with mandatory arguments only"""
    nn = fun_addtwo()
    assert isinstance(nn, TaskBase)
    assert nn.name == "fun_addtwo"
    assert hasattr(nn, "__call__")


def test_task_init_1a():
    with pytest.raises(TypeError):
        fun_addtwo("NA")


def test_task_init_2():
    """ task with a name and inputs"""
    nn = fun_addtwo(name="NA", a=3)
    # adding NA to the name of the variable
    assert getattr(nn.inputs, "a") == 3
    assert nn.state is None


@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, states_ind, states_val",
    [("a", "NA.a", ["NA.a"], [{"NA.a": 0}, {"NA.a": 1}], [{"NA.a": 3}, {"NA.a": 5}])],
)
@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_init_3(
    splitter, state_splitter, state_rpn, states_ind, states_val, input_type
):
    """ task with inputs and splitter"""
    a_in = [3, 5]
    if input_type == "array":
        a_in = np.array(a_in)

    nn = fun_addtwo(name="NA", a=a_in).split(splitter=splitter)

    assert np.allclose(nn.inputs.a, [3, 5])
    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == states_ind
    assert nn.state.states_val == states_val


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
    """ task with inputs and splitter"""
    a_in, b_in = [3, 5], [10, 20]
    if input_type == "array":
        a_in, b_in = np.array(a_in), np.array(b_in)
    elif input_type == "mixed":
        a_in = np.array(a_in)
    nn = fun_addvar(name="NA", a=a_in, b=b_in).split(splitter=splitter)

    assert np.allclose(nn.inputs.a, [3, 5])
    assert np.allclose(nn.inputs.b, [10, 20])
    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == states_ind
    assert nn.state.states_val == states_val


def test_task_init_4():
    """ task with interface and inputs. splitter set using split method"""
    nn = fun_addtwo(name="NA", a=[3, 5])
    nn.split(splitter="a")
    assert np.allclose(nn.inputs.a, [3, 5])

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert nn.state.states_val == [{"NA.a": 3}, {"NA.a": 5}]


def test_task_init_4a():
    """ task with a splitter and inputs set in the split method"""
    nn = fun_addtwo(name="NA")
    nn.split(splitter="a", a=[3, 5])
    assert np.allclose(nn.inputs.a, [3, 5])

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert nn.state.states_val == [{"NA.a": 3}, {"NA.a": 5}]


def test_task_init_4b():
    """ updating splitter using overwrite=True"""
    nn = fun_addtwo(name="NA")
    nn.split(splitter="b", a=[3, 5])
    nn.split(splitter="a", overwrite=True)
    assert np.allclose(nn.inputs.a, [3, 5])

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert nn.state.states_val == [{"NA.a": 3}, {"NA.a": 5}]


def test_task_init_4c():
    """ trying to set splitter twice without using overwrite"""
    nn = fun_addtwo(name="NA").split(splitter="b", a=[3, 5])
    with pytest.raises(Exception) as excinfo:
        nn.split(splitter="a")
    assert "splitter has been already set" in str(excinfo.value)

    assert nn.state.splitter == "NA.b"


def test_task_init_4d():
    """ trying to set the same splitter twice without using overwrite
        if the splitter is the same, the exception shouldn't be raised
    """
    nn = fun_addtwo(name="NA").split(splitter="a", a=[3, 5])
    nn.split(splitter="a")
    assert nn.state.splitter == "NA.a"


def test_task_init_5():
    """ task with inputs, splitter and combiner"""
    nn = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[3, 5], b=[1, 2])
        .combine("b")
    )

    assert nn.state.splitter == ["NA.a", "NA.b"]
    assert nn.state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert nn.state.combiner == ["NA.b"]

    assert nn.state.splitter_final == "NA.a"
    assert nn.state.splitter_rpn_final == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert nn.state.states_val == [
        {"NA.a": 3, "NA.b": 1},
        {"NA.a": 3, "NA.b": 2},
        {"NA.a": 5, "NA.b": 1},
        {"NA.a": 5, "NA.b": 2},
    ]

    assert nn.state.final_combined_ind_mapping == {0: [0, 1], 1: [2, 3]}


def test_task_init_5a():
    """ updating combiner using overwrite=True"""
    nn = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[3, 5], b=[1, 2])
        .combine("b")
    )
    nn.combine("a", overwrite=True)

    assert nn.state.splitter == ["NA.a", "NA.b"]
    assert nn.state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert nn.state.combiner == ["NA.a"]

    assert nn.state.splitter_final == "NA.b"
    assert nn.state.splitter_rpn_final == ["NA.b"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert nn.state.states_val == [
        {"NA.a": 3, "NA.b": 1},
        {"NA.a": 3, "NA.b": 2},
        {"NA.a": 5, "NA.b": 1},
        {"NA.a": 5, "NA.b": 2},
    ]

    assert nn.state.final_combined_ind_mapping == {0: [0, 2], 1: [1, 3]}


def test_task_init_5b():
    """ updating combiner without using overwrite"""
    nn = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[3, 5], b=[1, 2])
        .combine("b")
    )
    with pytest.raises(Exception) as excinfo:
        nn.combine("a")
    assert "combiner has been already set" in str(excinfo.value)

    assert nn.state.combiner == ["NA.b"]


def test_task_init_5c():
    """ trying to set the same combiner twice without using overwrite
        if the combiner is the same, the exception shouldn't be raised
    """
    nn = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[3, 5], b=[1, 2])
        .combine("b")
    )
    nn.combine("b")

    assert nn.state.splitter == ["NA.a", "NA.b"]
    assert nn.state.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert nn.state.combiner == ["NA.b"]

    assert nn.state.splitter_final == "NA.a"
    assert nn.state.splitter_rpn_final == ["NA.a"]


def test_task_init_6():
    """ task with splitter, but the input is an empty list"""
    nn = fun_addtwo(name="NA", a=[])
    nn.split(splitter="a")
    assert nn.inputs.a == []

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == []
    assert nn.state.states_val == []


def test_task_init_7(tmpdir):
    """ task with a dictionary of files as an input, checking checksum"""
    file1 = tmpdir.join("file1.txt")
    with open(file1, "w") as f:
        f.write("hello")

    file2 = tmpdir.join("file2.txt")
    with open(file2, "w") as f:
        f.write("from pydra\n")

    nn1 = fun_file_list(name="NA", filename_list=[file1, file2])
    output_dir1 = nn1.output_dir

    # changing the content of the file
    file2 = tmpdir.join("file2.txt")
    with open(file2, "w") as f:
        f.write("from pydra")

    nn2 = fun_file_list(name="NA", filename_list=[file1, file2])
    output_dir2 = nn2.output_dir

    # the checksum should be different - content of file2 is different
    assert output_dir1.name != output_dir2.name


def test_task_init_8():
    """ task without setting the input, the value should be set to attr.NOTHING"""
    nn = fun_addtwo(name="NA")
    assert nn.inputs.a is attr.NOTHING


def test_task_init_9():
    """ task without setting the input, but using the default avlue from function"""
    nn1 = fun_addvar_default(name="NA", a=2)
    assert nn1.inputs.b == 1

    nn2 = fun_addvar_default(name="NA", a=2, b=1)
    assert nn2.inputs.b == 1
    # both tasks should have the same checksum
    assert nn1.checksum == nn2.checksum


def test_task_error():
    func = fun_div(name="div", a=1, b=0)
    with pytest.raises(ZeroDivisionError):
        func()
    assert (func.output_dir / "_error.pklz").exists()


def test_odir_init():
    """ checking if output_dir is available for a task without init
        before running the task
    """
    nn = fun_addtwo(name="NA", a=3)
    assert nn.output_dir


# Tests for tasks without state (i.e. no splitter)


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1(plugin_dask_opt, tmpdir):
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=3)
    nn.cache_dir = tmpdir
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5
    # checking the return_inputs option, either is return_inputs is True, or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = nn.result(return_inputs=True)
    results_verb_val = nn.result(return_inputs="val")
    assert results_verb[0] == results_verb_val[0] == {"NA.a": 3}
    assert results_verb[1].output.out == results_verb_val[1].output.out == 5
    # checking the return_inputs option return_inputs="ind"
    # it should give indices of inputs (instead of values) for each element
    results_verb_ind = nn.result(return_inputs="ind")
    assert results_verb_ind[0] == {"NA.a": None}
    assert results_verb_ind[1].output.out == 5

    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_1_call():
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=3)
    nn()
    # checking the results
    results = nn.result()
    assert results.output.out == 5
    # checking the output_dir
    assert nn.output_dir.exists()


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1_call_subm(plugin_dask_opt, tmpdir):
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=3)
    nn.cache_dir = tmpdir
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin_dask_opt) as sub:
        nn(submitter=sub)

    # checking the results
    results = nn.result()
    assert results.output.out == 5
    # checking the output_dir
    assert nn.output_dir.exists()


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_1_call_plug(plugin_dask_opt, tmpdir):
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=3)
    nn.cache_dir = tmpdir
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    nn(plugin=plugin_dask_opt)

    # checking the results
    results = nn.result()
    assert results.output.out == 5
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_1_call_updateinp():
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=30)
    # updating input when calling the node
    nn(a=3)

    # checking the results
    results = nn.result()
    assert results.output.out == 5
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_2(plugin, tmpdir):
    """ task with a list as an input, but no splitter"""
    nn = moment(name="NA", n=3, lst=[2, 3, 4])
    nn.cache_dir = tmpdir
    assert np.allclose(nn.inputs.n, [3])
    assert np.allclose(nn.inputs.lst, [2, 3, 4])
    assert nn.state is None

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 33
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_3(plugin, tmpdir):
    """ task with a dictionary as an input"""
    nn = fun_dict(name="NA", d={"a": "ala", "b": "bala"})
    nn.cache_dir = tmpdir
    assert nn.inputs.d == {"a": "ala", "b": "bala"}

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == "a:ala_b:bala"
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_4(plugin, tmpdir):
    """ task with a dictionary as an input"""
    file1 = tmpdir.join("file.txt")
    with open(file1, "w") as f:
        f.write("hello from pydra\n")

    nn = fun_file(name="NA", filename=file1)
    nn.cache_dir = tmpdir

    with Submitter(plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == "hello from pydra\n"
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_5(tmpdir):
    """ task with a dictionary of files as an input"""
    file1 = tmpdir.join("file1.txt")
    with open(file1, "w") as f:
        f.write("hello")

    file2 = tmpdir.join("file2.txt")
    with open(file2, "w") as f:
        f.write("from pydra\n")

    nn = fun_file_list(name="NA", filename_list=[file1, file2])

    nn()

    # checking the results
    results = nn.result()
    assert results.output.out == "hello from pydra\n"
    # checking the output_dir
    assert nn.output_dir.exists()


def test_task_nostate_6():
    """ checking if the function gets the None value"""
    nn = fun_addvar_none(name="NA", a=2, b=None)
    assert nn.inputs.b is None
    nn()
    assert nn.result().output.out == 2


def test_task_nostate_6a_exception():
    """ checking if the function gets the attr.Nothing value"""
    nn = fun_addvar_none(name="NA", a=2)
    assert nn.inputs.b is attr.NOTHING
    with pytest.raises(TypeError) as excinfo:
        nn()
    assert "unsupported" in str(excinfo.value)


def test_task_nostate_7():
    """ using the default value from the function for b input"""
    nn = fun_addvar_default(name="NA", a=2)
    assert nn.inputs.b == 1
    nn()
    assert nn.result().output.out == 3


# Testing caching for tasks without states


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachedir(plugin_dask_opt, tmpdir):
    """ task with provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_task_nostate")
    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachedir_relativepath(tmpdir, plugin_dask_opt):
    """ task with provided cache_dir as relative path"""
    cwd = tmpdir.chdir()
    cache_dir = "test_task_nostate"
    tmpdir.mkdir(cache_dir)

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5

    shutil.rmtree(cache_dir)


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_nostate_cachelocations(plugin_dask_opt, tmpdir):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn2)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if the second task didn't run the interface again
    assert nn.output_dir.exists()
    assert not nn2.output_dir.exists()


def test_task_nostate_cachelocations_forcererun(plugin, tmpdir):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn2, rerun=True)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if the second task rerun the interface
    assert nn.output_dir.exists()
    assert nn2.output_dir.exists()


def test_task_nostate_cachelocations_nosubmitter(tmpdir):
    """
    Two identical tasks (that are run without submitter!) with provided cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    nn()

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    nn2()

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if the second task didn't run the interface again
    assert nn.output_dir.exists()
    assert not nn2.output_dir.exists()


def test_task_nostate_cachelocations_nosubmitter_forcererun(tmpdir):
    """
    Two identical tasks (that are run without submitter!) with provided cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    nn()

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    nn2(rerun=True)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if the second task run the interface again
    assert nn.output_dir.exists()
    assert nn2.output_dir.exists()


def test_task_nostate_cachelocations_updated(plugin, tmpdir):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations in init,
     that is later overwritten in Submitter.__call__;
    the cache_locations passed to call doesn't exist so the second task should run again
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir1 = tmpdir.mkdir("test_task_nostate1")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    # updating cache location to non-existing dir
    with Submitter(plugin=plugin) as sub:
        sub(nn2, cache_locations=cache_dir1)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if both tasks run interface
    assert nn.output_dir.exists()
    assert nn2.output_dir.exists()


# Tests for tasks with states (i.e. with splitter)


@pytest.mark.flaky(reruns=2)  # when dask
@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_state_1(plugin_dask_opt, input_type, tmpdir):
    """ task with the simplest splitter"""
    a_in = [3, 5]
    if input_type == "array":
        a_in = np.array(a_in)

    nn = fun_addtwo(name="NA").split(splitter="a", a=a_in)
    nn.cache_dir = tmpdir

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert (nn.inputs.a == np.array([3, 5])).all()

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]

    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = nn.result(return_inputs=True)
    results_verb_val = nn.result(return_inputs="val")
    for i, res in enumerate(expected):
        assert (results_verb[i][0], results_verb[i][1].output.out) == res
        assert (results_verb_val[i][0], results_verb_val[i][1].output.out) == res

    # checking the return_inputs option return_inputs="ind"
    # it should give indices of inputs (instead of values) for each element
    results_verb_ind = nn.result(return_inputs="ind")
    expected_ind = [({"NA.a": 0}, 5), ({"NA.a": 1}, 7)]
    for i, res in enumerate(expected_ind):
        assert (results_verb_ind[i][0], results_verb_ind[i][1].output.out) == res

    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_1a(plugin, tmpdir):
    """ task with the simplest splitter (inputs set separately)"""
    nn = fun_addtwo(name="NA")
    nn.split(splitter="a")
    nn.inputs.a = [3, 5]
    nn.cache_dir = tmpdir

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert (nn.inputs.a == np.array([3, 5])).all()

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


def test_task_state_singl_1(plugin, tmpdir):
    """ Tasks with two inputs and a splitter (no combiner)
        one input is a single value, the other is in the splitter and combiner
    """
    nn = fun_addvar(name="NA").split(splitter="a", a=[3, 5], b=10)
    nn.cache_dir = tmpdir

    assert nn.inputs.a == [3, 5]
    assert nn.inputs.b == 10
    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.state.splitter_final == "NA.a"
    assert nn.state.splitter_rpn_final == ["NA.a"]

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    expected = [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 10}, 15)]
    results = nn.result()
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


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
    tmpdir,
):
    """ Tasks with two inputs and a splitter (no combiner)"""
    a_in, b_in = [3, 5], [10, 20]
    if input_type == "array":
        a_in, b_in = np.array(a_in), np.array(b_in)
    elif input_type == "mixed":
        a_in = np.array(a_in)
    nn = fun_addvar(name="NA").split(splitter=splitter, a=a_in, b=b_in)
    nn.cache_dir = tmpdir

    assert (nn.inputs.a == np.array([3, 5])).all()
    assert (nn.inputs.b == np.array([10, 20])).all()
    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn
    assert nn.state.splitter_final == state_splitter
    assert nn.state.splitter_rpn_final == state_rpn

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]

    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = nn.result(return_inputs=True)
    results_verb_val = nn.result(return_inputs="val")
    for i, res in enumerate(expected):
        assert (results_verb[i][0], results_verb[i][1].output.out) == res
        assert (results_verb_val[i][0], results_verb_val[i][1].output.out) == res

    # checking the return_inputs option return_inputs="ind"
    # it should give indices of inputs (instead of values) for each element
    results_verb_ind = nn.result(return_inputs="ind")
    for i, res in enumerate(expected_ind):
        assert (results_verb_ind[i][0], results_verb_ind[i][1].output.out) == res

    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_3(plugin, tmpdir):
    """ task with the simplest splitter, the input is an empty list"""
    nn = fun_addtwo(name="NA").split(splitter="a", a=[])
    nn.cache_dir = tmpdir

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.inputs.a == []

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = []
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]
    # checking the output_dir
    assert nn.output_dir == []


@pytest.mark.parametrize("input_type", ["list", "array"])
def test_task_state_4(plugin, input_type, tmpdir):
    """ task with a list as an input, and a simple splitter """
    lst_in = [[2, 3, 4], [1, 2, 3]]
    if input_type == "array":
        lst_in = np.array(lst_in)
    nn = moment(name="NA", n=3, lst=lst_in).split(splitter="lst")
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, 3)
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == "NA.lst"

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking that split is done across dim 0
    el_0 = nn.state.states_val[0]["NA.lst"]
    if input_type == "list":
        assert el_0 == [2, 3, 4]
    elif input_type == "array":
        assert isinstance(el_0, np.ndarray)
        assert (el_0 == [2, 3, 4]).all()

    # checking the results
    results = nn.result()
    for i, expected in enumerate([33, 12]):
        assert results[i].output.out == expected
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_4a(plugin, tmpdir):
    """ task with a tuple as an input, and a simple splitter """
    nn = moment(name="NA", n=3, lst=[(2, 3, 4), (1, 2, 3)]).split(splitter="lst")
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, 3)
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == "NA.lst"

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    for i, expected in enumerate([33, 12]):
        assert results[i].output.out == expected
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_5(plugin, tmpdir):
    """ task with a list as an input, and the variable is part of the scalar splitter"""
    nn = moment(name="NA", n=[1, 3], lst=[[2, 3, 4], [1, 2, 3]]).split(
        splitter=("n", "lst")
    )
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, [1, 3])
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == ("NA.n", "NA.lst")

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    for i, expected in enumerate([3, 12]):
        assert results[i].output.out == expected
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_5_exception(plugin, tmpdir):
    """ task with a list as an input, and the variable is part of the scalar splitter
        the shapes are not matching, so exception should be raised
    """
    nn = moment(name="NA", n=[1, 3, 3], lst=[[2, 3, 4], [1, 2, 3]]).split(
        splitter=("n", "lst")
    )
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, [1, 3, 3])
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == ("NA.n", "NA.lst")

    with pytest.raises(Exception) as excinfo:
        with Submitter(plugin=plugin) as sub:
            sub(nn)
    assert "shape" in str(excinfo.value)


def test_task_state_6(plugin, tmpdir):
    """ ask with a list as an input, and the variable is part of the outer splitter """
    nn = moment(name="NA", n=[1, 3], lst=[[2, 3, 4], [1, 2, 3]]).split(
        splitter=["n", "lst"]
    )
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, [1, 3])
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == ["NA.n", "NA.lst"]

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    for i, expected in enumerate([3, 2, 33, 12]):
        assert results[i].output.out == expected
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_6a(plugin, tmpdir):
    """ ask with a tuple as an input, and the variable is part of the outer splitter """
    nn = moment(name="NA", n=[1, 3], lst=[(2, 3, 4), (1, 2, 3)]).split(
        splitter=["n", "lst"]
    )
    nn.cache_dir = tmpdir

    assert np.allclose(nn.inputs.n, [1, 3])
    assert np.allclose(nn.inputs.lst, [[2, 3, 4], [1, 2, 3]])
    assert nn.state.splitter == ["NA.n", "NA.lst"]

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    for i, expected in enumerate([3, 2, 33, 12]):
        assert results[i].output.out == expected
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_state_comb_1(plugin_dask_opt, tmpdir):
    """ task with the simplest splitter and combiner"""
    nn = fun_addtwo(name="NA").split(a=[3, 5], splitter="a").combine(combiner="a")
    nn.cache_dir = tmpdir

    assert (nn.inputs.a == np.array([3, 5])).all()

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.state.combiner == ["NA.a"]
    assert nn.state.splitter_final is None
    assert nn.state.splitter_rpn_final == []

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    assert nn.state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert nn.state.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    # checking the results
    results = nn.result()
    # fully combined (no nested list)
    combined_results = [res.output.out for res in results]
    assert combined_results == [5, 7]

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    expected_ind = [({"NA.a": 0}, 5), ({"NA.a": 1}, 7)]
    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = nn.result(return_inputs=True)
    results_verb_val = nn.result(return_inputs="val")
    for i, res in enumerate(expected):
        assert (results_verb[i][0], results_verb[i][1].output.out) == res
        assert (results_verb_val[i][0], results_verb_val[i][1].output.out) == res
    # checking the return_inputs option return_inputs="ind"
    # it should give indices of inputs (instead of values) for each element
    results_verb_ind = nn.result(return_inputs="ind")
    for i, res in enumerate(expected_ind):
        assert (results_verb_ind[i][0], results_verb_ind[i][1].output.out) == res

    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


@pytest.mark.parametrize(
    "splitter, combiner, state_splitter, state_rpn, state_combiner, state_combiner_all, "
    "state_splitter_final, state_rpn_final, expected, expected_val",
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
            [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
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
            [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
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
            [
                [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 10}, 15)],
                [({"NA.a": 3, "NA.b": 20}, 23), ({"NA.a": 5, "NA.b": 20}, 25)],
            ],
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
            [
                [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 3, "NA.b": 20}, 23)],
                [({"NA.a": 5, "NA.b": 10}, 15), ({"NA.a": 5, "NA.b": 20}, 25)],
            ],
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
            [
                ({"NA.a": 3, "NA.b": 10}, 13),
                ({"NA.a": 3, "NA.b": 20}, 23),
                ({"NA.a": 5, "NA.b": 10}, 15),
                ({"NA.a": 5, "NA.b": 20}, 25),
            ],
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
    expected_val,
    tmpdir,
):
    """ Tasks with scalar and outer splitters and  partial or full combiners"""
    nn = (
        fun_addvar(name="NA")
        .split(a=[3, 5], b=[10, 20], splitter=splitter)
        .combine(combiner=combiner)
    )
    nn.cache_dir = tmpdir

    assert (nn.inputs.a == np.array([3, 5])).all()

    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn
    assert nn.state.combiner == state_combiner

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    assert nn.state.splitter_final == state_splitter_final
    assert nn.state.splitter_rpn_final == state_rpn_final
    assert set(nn.state.current_combiner_all) == set(state_combiner_all)

    # checking the results
    results = nn.result()
    # checking the return_inputs option, either return_inputs is True or "val",
    # it should give values of inputs that corresponds to the specific element
    results_verb = nn.result(return_inputs=True)

    if nn.state.splitter_rpn_final:
        for i, res in enumerate(expected):
            assert [res.output.out for res in results[i]] == res
        # results_verb
        for i, res_l in enumerate(expected_val):
            for j, res in enumerate(res_l):
                assert (results_verb[i][j][0], results_verb[i][j][1].output.out) == res
    # if the combiner is full expected is "a flat list"
    else:
        assert [res.output.out for res in results] == expected
        for i, res in enumerate(expected_val):
            assert (results_verb[i][0], results_verb[i][1].output.out) == res

    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_comb_singl_1(plugin, tmpdir):
    """ Tasks with two inputs;
     one input is a single value, the other is in the splitter and combiner
     """
    nn = fun_addvar(name="NA").split(splitter="a", a=[3, 5], b=10).combine(combiner="a")
    nn.cache_dir = tmpdir

    assert nn.inputs.a == [3, 5]
    assert nn.inputs.b == 10
    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.state.combiner == ["NA.a"]
    assert nn.state.splitter_final == None
    assert nn.state.splitter_rpn_final == []

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    expected = ({}, [13, 15])
    results = nn.result()
    # full combiner, no nested list
    combined_results = [res.output.out for res in results]
    assert combined_results == expected[1]
    # checking the output_dir
    assert nn.output_dir
    for odir in nn.output_dir:
        assert odir.exists()


def test_task_state_comb_3(plugin, tmpdir):
    """ task with the simplest splitter, the input is an empty list"""
    nn = fun_addtwo(name="NA").split(splitter="a", a=[]).combine(combiner=["a"])
    nn.cache_dir = tmpdir

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.inputs.a == []

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = []
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]
    # checking the output_dir
    assert nn.output_dir == []


def test_task_state_comb_order():
    """ tasks with an outer splitter and various combiner;
        showing the order of results
    """

    # single combiner "a" - will create two lists, first one for b=3, second for b=5
    nn_a = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[10, 20], b=[3, 5])
        .combine(combiner="a")
    )
    assert nn_a.state.combiner == ["NA.a"]

    results_a = nn_a()
    combined_results_a = [[res.output.out for res in res_l] for res_l in results_a]
    assert combined_results_a == [[13, 23], [15, 25]]

    # single combiner "b" - will create two lists, first one for a=10, second for a=20
    nn_b = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[10, 20], b=[3, 5])
        .combine(combiner="b")
    )
    assert nn_b.state.combiner == ["NA.b"]

    results_b = nn_b()
    combined_results_b = [[res.output.out for res in res_l] for res_l in results_b]
    assert combined_results_b == [[13, 15], [23, 25]]

    # combiner with both fields ["a", "b"] - will create one list
    nn_ab = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[10, 20], b=[3, 5])
        .combine(combiner=["a", "b"])
    )
    assert nn_ab.state.combiner == ["NA.a", "NA.b"]

    results_ab = nn_ab()
    # full combiner, no nested list
    combined_results_ab = [res.output.out for res in results_ab]
    assert combined_results_ab == [13, 15, 23, 25]

    # combiner with both fields ["b", "a"] - will create the same list as nn_ab
    # no difference in the order for setting combiner
    nn_ba = (
        fun_addvar(name="NA")
        .split(splitter=["a", "b"], a=[10, 20], b=[3, 5])
        .combine(combiner=["b", "a"])
    )
    assert nn_ba.state.combiner == ["NA.b", "NA.a"]

    results_ba = nn_ba()
    combined_results_ba = [res.output.out for res in results_ba]
    assert combined_results_ba == [13, 15, 23, 25]


# Testing caching for tasks with states


@pytest.mark.flaky(reruns=2)  # when dask
def test_task_state_cachedir(plugin_dask_opt, tmpdir):
    """ task with a state and provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_task_nostate")
    nn = fun_addtwo(name="NA", cache_dir=cache_dir).split(splitter="a", a=[3, 5])

    assert nn.state.splitter == "NA.a"
    assert (nn.inputs.a == np.array([3, 5])).all()

    with Submitter(plugin=plugin_dask_opt) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


def test_task_state_cachelocations(plugin, tmpdir):
    """
    Two identical tasks with a state and cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir).split(splitter="a", a=[3, 5])
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(
        name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir
    ).split(splitter="a", a=[3, 5])
    with Submitter(plugin=plugin) as sub:
        sub(nn2)

    # checking the results
    results2 = nn2.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2[i].output.out == res[1]

    assert all([dir.exists() for dir in nn.output_dir])
    assert not any([dir.exists() for dir in nn2.output_dir])


def test_task_state_cachelocations_forcererun(plugin, tmpdir):
    """
    Two identical tasks with a state and cache_dir;
    the second task has cache_locations,
    but submitter is called with rerun=True, so should recompute
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir).split(splitter="a", a=[3, 5])
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(
        name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir
    ).split(splitter="a", a=[3, 5])
    with Submitter(plugin=plugin) as sub:
        sub(nn2, rerun=True)

    # checking the results
    results2 = nn2.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2[i].output.out == res[1]

    # both workflows should be run
    assert all([dir.exists() for dir in nn.output_dir])
    assert all([dir.exists() for dir in nn2.output_dir])


def test_task_state_cachelocations_updated(plugin, tmpdir):
    """
    Two identical tasks with states and cache_dir;
    the second task has cache_locations in init,
     that is later overwritten in Submitter.__call__;
    the cache_locations from call doesn't exist so the second task should run again
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir1 = tmpdir.mkdir("test_task_nostate1")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", cache_dir=cache_dir).split(splitter="a", a=[3, 5])
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", cache_dir=cache_dir2, cache_locations=cache_dir).split(
        splitter="a", a=[3, 5]
    )
    with Submitter(plugin=plugin) as sub:
        sub(nn2, cache_locations=cache_dir1)

    # checking the results
    results2 = nn2.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results2[i].output.out == res[1]

    # both workflows should be run
    assert all([dir.exists() for dir in nn.output_dir])
    assert all([dir.exists() for dir in nn2.output_dir])
