import os, shutil
import numpy as np
import pytest, pdb

from ..core import TaskBase
from ..submitter import Submitter
from ..task import to_task

Plugins = ["cf"]


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_outputs")
    os.makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


@to_task
def fun_addtwo(a):
    import time

    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@to_task
def fun_addvar(a, b):
    return a + b


@to_task
def fun_addvar4(a, b, c, d):
    return a + b + c + d


@to_task
def moment(lst, n):
    return sum([i ** n for i in lst]) / len(lst)


@to_task
def fun_div(a, b):
    return a / b


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


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, states_ind, states_val",
    [("a", "NA.a", ["NA.a"], [{"NA.a": 0}, {"NA.a": 1}], [{"NA.a": 3}, {"NA.a": 5}])],
)
def test_task_init_3(splitter, state_splitter, state_rpn, states_ind, states_val):
    """ task with inputs and splitter"""
    nn = fun_addtwo(name="NA", a=[3, 5]).split(splitter=splitter)

    assert np.allclose(nn.inputs.a, [3, 5])
    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == states_ind
    assert nn.state.states_val == states_val


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
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
def test_task_init_3a(splitter, state_splitter, state_rpn, states_ind, states_val):
    """ task with inputs and splitter"""
    nn = fun_addvar(name="NA", a=[3, 5], b=[10, 20]).split(splitter=splitter)

    assert np.allclose(nn.inputs.a, [3, 5])
    assert np.allclose(nn.inputs.b, [10, 20])
    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn

    nn.state.prepare_states(nn.inputs)
    assert nn.state.states_ind == states_ind
    assert nn.state.states_val == states_val


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
def test_task_init_4():
    """ task with interface and inputs. splitter set using split method"""
    nn = fun_addtwo(name="NA", a=[3, 5])
    nn.split(splitter="a")
    assert np.allclose(nn.inputs.a, [3, 5])

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    nn.state.states_ind = [{"NA.a": 0}, {"NA.a": 1}]
    nn.state.states_val = [{"NA.a": 3}, {"NA.a": 5}]


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
def test_task_init_4a():
    """ task with a splitter and inputs set in the split method"""
    nn = fun_addtwo(name="NA")
    nn.split(splitter="a", a=[3, 5])
    assert np.allclose(nn.inputs.a, [3, 5])

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]

    nn.state.prepare_states(nn.inputs)
    nn.state.states_ind = [{"NA.a": 0}, {"NA.a": 1}]
    nn.state.states_val = [{"NA.a": 3}, {"NA.a": 5}]


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
def test_task_init_4b():
    """ trying to set splitter twice"""
    nn = fun_addtwo(name="NA").split(splitter="a", a=[3, 5])
    with pytest.raises(Exception) as excinfo:
        nn.split(splitter="a")
    assert str(excinfo.value) == "splitter has been already set"


def test_task_error():
    func = fun_div(name="div", a=1, b=0)
    with pytest.raises(ZeroDivisionError):
        func()
    assert (func.output_dir / "_error.pklz").exists()


# Tests for tasks without state (i.e. no splitter)


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_1(plugin):
    """ task without splitter"""
    nn = fun_addtwo(name="NA", a=3)
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_2(plugin):
    """ task with a list as an input, but no splitter"""
    nn = moment(name="NA", n=3, lst=[2, 3, 4])
    assert np.allclose(nn.inputs.n, [3])
    assert np.allclose(nn.inputs.lst, [2, 3, 4])
    assert nn.state is None

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 33


# Testing caching for tasks without states


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_cachedir(plugin, tmpdir):
    """ task with provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_task_nostate")
    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_cachedir_relativepath(tmpdir, plugin):
    """ task with provided cache_dir as relative path"""
    cwd = tmpdir.chdir()
    cache_dir = "test_task_nostate"
    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    assert np.allclose(nn.inputs.a, [3])
    assert nn.state is None

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    assert results.output.out == 5

    shutil.rmtree(cache_dir)


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_cachelocations(plugin, tmpdir):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations and should not recompute the results
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn2)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if the second task didn't run the interface again
    assert nn.output_dir.exists()
    assert not nn2.output_dir.exists()


@pytest.mark.parametrize("plugin", Plugins)
def test_task_nostate_cachelocations_updated(plugin, tmpdir):
    """
    Two identical tasks with provided cache_dir;
    the second task has cache_locations in init that is later overwritten in run;
    the cache_locations from run doesn't exist so the second task should run again
    """
    cache_dir = tmpdir.mkdir("test_task_nostate")
    cache_dir1 = tmpdir.mkdir("test_task_nostate1")
    cache_dir2 = tmpdir.mkdir("test_task_nostate2")

    nn = fun_addtwo(name="NA", a=3, cache_dir=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn)

    nn2 = fun_addtwo(name="NA", a=3, cache_dir=cache_dir2, cache_locations=cache_dir)
    with Submitter(plugin=plugin) as sub:
        sub(nn2, cache_locations=cache_dir1)

    # checking the results
    results2 = nn2.result()
    assert results2.output.out == 5

    # checking if both tasks run interface
    assert nn.output_dir.exists()
    assert nn2.output_dir.exists()


# Tests for tasks with states (i.e. with splitter)


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_1(plugin):
    """ task with the simplest splitter"""
    nn = fun_addtwo(name="NA").split(splitter="a", a=[3, 5])

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


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_1a(plugin):
    """ task with the simplest splitter (inputs set separately)"""
    nn = fun_addtwo(name="NA")
    nn.split(splitter="a")
    nn.inputs.a = [3, 5]

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


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize(
    "splitter, state_splitter, state_rpn, expected",
    [
        (
            ("a", "b"),
            ("NA.a", "NA.b"),
            ["NA.a", "NA.b", "."],
            [({"NA.a": 3, "NA.b": 10}, 13), ({"NA.a": 5, "NA.b": 20}, 25)],
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
        ),
    ],
)
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_2(plugin, splitter, state_splitter, state_rpn, expected):
    """ Tasks with two inputs and a splitter (no combiner)"""
    nn = fun_addvar(name="NA").split(splitter=splitter, a=[3, 5], b=[10, 20])

    assert nn.inputs.a == [3, 5]
    assert nn.inputs.b == [10, 20]
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


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_comb_1(plugin):
    """ task with the simplest splitter and combiner"""
    nn = fun_addtwo(name="NA").split(a=[3, 5], splitter="a").combine(combiner="a")

    assert (nn.inputs.a == np.array([3, 5])).all()

    assert nn.state.splitter == "NA.a"
    assert nn.state.splitter_rpn == ["NA.a"]
    assert nn.state.combiner == ["NA.a"]
    assert nn.state.splitter_final is None
    assert nn.state.splitter_rpn_final == []

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    assert nn.state.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert nn.state.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    # checking the results
    results = nn.result()

    combined_results = [[res.output.out for res in res_l] for res_l in results]
    expected = [({}, [5, 7])]
    for i, res in enumerate(expected):
        assert combined_results[i] == res[1]


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize(
    "splitter, combiner, state_splitter, state_rpn, state_combiner, state_combiner_all, "
    "state_splitter_final, state_rpn_final, expected",
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
            [({}, [13, 25])],
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
            [({}, [13, 25])],
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
            [({"NA.b": 10}, [13, 15]), ({"NA.b": 20}, [23, 25])],
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
            [({"NA.a": 3}, [13, 23]), ({"NA.a": 5}, [15, 25])],
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
            [({}, [13, 23, 15, 25])],
        ),
    ],
)
@pytest.mark.parametrize("plugin", Plugins)
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
):
    """ Tasks with scalar and outer splitters and  partial or full combiners"""
    nn = (
        fun_addvar(name="NA")
        .split(a=[3, 5], b=[10, 20], splitter=splitter)
        .combine(combiner=combiner)
    )

    assert (nn.inputs.a == np.array([3, 5])).all()

    assert nn.state.splitter == state_splitter
    assert nn.state.splitter_rpn == state_rpn
    assert nn.state.combiner == state_combiner

    assert nn.state.splitter_final == state_splitter_final
    assert nn.state.splitter_rpn_final == state_rpn_final
    assert set(nn.state.right_combiner_all) == set(state_combiner_all)

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()

    combined_results = [[res.output.out for res in res_l] for res_l in results]
    for i, res in enumerate(expected):
        assert combined_results[i] == res[1]


# Testing caching for tasks with states


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_cachedir(plugin, tmpdir):
    """ task with a state and provided cache_dir using pytest tmpdir"""
    cache_dir = tmpdir.mkdir("test_task_nostate")
    nn = fun_addtwo(name="NA", cache_dir=cache_dir).split(splitter="a", a=[3, 5])

    assert nn.state.splitter == "NA.a"
    assert (nn.inputs.a == np.array([3, 5])).all()

    with Submitter(plugin=plugin) as sub:
        sub(nn)

    # checking the results
    results = nn.result()
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.xfail(reason="TODO: output_dir.exists check doesn't work when splitter")
@pytest.mark.parametrize("plugin", Plugins)
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

    # TODO: this doesnt work properly when splitter
    # checking if the second task didn't run the interface again
    assert nn.output_dir.exists()
    assert not nn2.output_dir.exists()


@pytest.mark.xfail(reason="WIP: state doesnt work yet")
@pytest.mark.xfail(reason="TODO: output_dir.exists check doesn't work when splitter")
@pytest.mark.parametrize("plugin", Plugins)
def test_task_state_cachelocations_updated(plugin, tmpdir):
    """
    Two identical tasks with states and cache_dir;
    the second task has cache_locations in init that is later overwritten in run;
    the cache_locations from run doesn't exist so the second task should run again
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

    # TODO: this doesnt work properly when splitter
    # checking if both tasks run interface
    assert nn.output_dir.exists()
    assert nn2.output_dir.exists()
