import os
import pytest, pdb

from nipype.utils.filemanip import save_json, makedirs, to_str

from ..submitter import Submitter
from ..task import to_task


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_satra_example")
    makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)

Plugins = ["serial", "cf"]


# @python_app
# def double(x):
#       return x*2
#
# # doubled_x is an AppFuture
# doubled_x = double().split('x', x=range(10))
#
# # Check status of doubled_x, this will print True if the result is available, else false
# print(doubled_x.done())


# TODO: for some reason node doesn't work when result is 0 (in this function if x=0)
# TODO: need more debugging...
@to_task
def double(x):
    return x*2


@pytest.mark.parametrize("plugin", Plugins)
def test_1(plugin, change_dir):
    doubled_x = double(name="double", workingdir="test_1_{}".format(plugin))\
            .split("x", x=[1, 2, 3])

    assert doubled_x.state.splitter == "double.x"

    sub = Submitter(plugin=plugin, runnable=doubled_x)
    sub.run()
    sub.close()
    print(doubled_x.done)

    # checking the results
    results = doubled_x.result()
    expected = [({"double.x": 1}, 2), ({"double.x": 2}, 4), ({"double.x": 3}, 6)]
    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
def test_1a(plugin, change_dir):
    doubled_x = double(name="double", workingdir="test_1a_{}".format(plugin),
                       splitter="x", x=[1, 2, 3])

    assert doubled_x.state.splitter == "double.x"

    sub = Submitter(plugin=plugin, runnable=doubled_x)
    sub.run()
    sub.close()
    print(doubled_x.done)

    # checking the results
    results = doubled_x.result()
    expected = [({"double.x": 1}, 2), ({"double.x": 2}, 4), ({"double.x": 3}, 6)]
    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


# @python_app
# def multiply(x, y):
#       return x*y
#
# # multiply_x is an AppFuture
# multiple_x = multiply().split(['x', 'y'], x=range(10), y=range(10))
#
# # Check status of doubled_x, this will print True if the result is available, else false
# print(multiply_x.done())


@to_task
def multiply(x, y):
      return x*y

@pytest.mark.parametrize("plugin", Plugins)
def test_2(plugin, change_dir):
    # multiply_x is an AppFuture
    multiple_x = multiply(name="mult", workingdir="test_2_{}".format(plugin))\
        .split(['x', 'y'], x=[1, 2], y=[1, 2])

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({"mult.x": 1, "mult.y": 1}, 1), ({"mult.x": 1, "mult.y": 2}, 2),
                ({"mult.x": 2, "mult.y": 1}, 2), ({"mult.x": 2, "mult.y": 2}, 4)]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


# # multiply_x is an AppFuture
# multiple_x = multiply().split(['x', 'y'], x=range(10), y=range(10)).combine('x')

@pytest.mark.parametrize("plugin", Plugins)
def test_3(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_3_{}".format(plugin))\
        .split(['x', 'y'], x=[1, 2], y=[1, 2])\
        .combine("x")

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]
    assert multiple_x.state.combiner == ["mult.x"]

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({"mult.y": 1}, [1, 2]), ({"mult.y": 2}, [2, 4])]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
def test_3a(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_3a_{}".format(plugin),
                          splitter=['x', 'y'], x=[1, 2], y=[1, 2],
                          combiner="x")

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]
    assert multiple_x.state.combiner == ["mult.x"]

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({"mult.y": 1}, [1, 2]), ({"mult.y": 2}, [2, 4])]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]



# # multiply_x is an AppFuture
# multiple_x = multiply().split(['x', 'y'], x=range(10), y=range(10)).combine(['y', 'x'])
# this would return a nested list of lists as a result.

@pytest.mark.parametrize("plugin", Plugins)
def test_4(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_4_{}".format(plugin))\
        .split(['x', 'y'], x=[1, 2], y=[1, 2])\
        .combine(["y", "x"])

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]
    assert multiple_x.state.combiner == ["mult.y", "mult.x"]

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({}, [1, 2, 2, 4])]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]



# in the following example, note parentheses instead of brackets. this is our syntax for synchronized parallelism
# # multiply_x is an AppFuture
# multiple_x = multiply().split(('x', 'y'), x=range(10), y=range(10)).combine('x')
# this will return 0 * 0, 1 * 1, and so on. and we can have any combination thereof.

@pytest.mark.parametrize("plugin", Plugins)
def test_5(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_5_{}".format(plugin))\
        .split(('x', 'y'), x=[1, 2], y=[1, 2])

    assert multiple_x.state.splitter == ("mult.x", "mult.y")

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({"mult.x": 1, "mult.y": 1}, 1), ({"mult.x": 2, "mult.y": 2}, 4)]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
def test_6(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_6_{}".format(plugin))\
        .split(('x', 'y'), x=[1, 2], y=[1, 2])\
        .combine(["x"])

    assert multiple_x.state.splitter == ("mult.x", "mult.y")
    assert multiple_x.state.combiner == ["mult.x"]

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({}, [1, 4])]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]



# TODO: have to use workflow
# # multiply_x is an AppFuture
# multiple_x = multiply().split(('x', 'y'), x=range(10), y=range(10))
# add_x = add2(x='multiply.out').combine('multiply.x')

@pytest.mark.parametrize("plugin", Plugins)
@pytest.mark.xfail(reason="wip: Workflow")
def test_7(plugin, change_dir):
    multiple_x = multiply(name="mult", workingdir="test_7_{}".format(plugin))\
        .split(('x', 'y'), x=[1, 2], y=[1, 2])

    add_x = add2(x='multiply.out').combine('multiply.x')

    sub = Submitter(plugin=plugin, runnable=multiple_x)
    sub.run()
    sub.close()

    # checking the results
    results = multiple_x.result()
    expected = [({"mult.x": 1, "mult.y": 1}, 1), ({"mult.x": 2, "mult.y": 2}, 4)]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]
