import pytest

from ..submitter import Submitter
from ..task import to_task
from ..node import Workflow


Plugins = ["cf"]


# @python_app
# def double(x):
#       return x*2
#
# # doubled_x is an AppFuture
# doubled_x = double().split('x', x=range(10))
#
# # Check status of doubled_x, this will print True if the result is available, else false
# print(doubled_x.done())


@to_task
def double(x):
    return x * 2


@pytest.mark.parametrize("plugin", Plugins)
def test_1(plugin):
    doubled_x = double(name="double").split("x", x=list(range(3)))

    assert doubled_x.state.splitter == "double.x"

    with Submitter(plugin=plugin) as sub:
        sub.run(doubled_x)

    # checking the results
    results = doubled_x.result()
    expected = [({"double.x": 0}, 0), ({"double.x": 1}, 2), ({"double.x": 2}, 4)]
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


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
    return x * y


@pytest.mark.parametrize("plugin", Plugins)
def test_2(plugin):
    # multiply_x is an AppFuture
    multiple_x = multiply(name="mult").split(["x", "y"], x=[1, 2], y=[1, 2])

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]

    with Submitter(plugin=plugin) as sub:
        sub.run(multiple_x)

    # checking the results
    results = multiple_x.result()
    expected = [
        ({"mult.x": 1, "mult.y": 1}, 1),
        ({"mult.x": 1, "mult.y": 2}, 2),
        ({"mult.x": 2, "mult.y": 1}, 2),
        ({"mult.x": 2, "mult.y": 2}, 4),
    ]

    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


# # multiply_x is an AppFuture
# multiple_x = multiply().split(['x', 'y'], x=range(10), y=range(10)).combine('x')


@pytest.mark.parametrize("plugin", Plugins)
def test_3(plugin):
    multiple_x = (
        multiply(name="mult").split(["x", "y"], x=[1, 2], y=[1, 2]).combine("x")
    )

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]
    assert multiple_x.state.combiner == ["mult.x"]

    with Submitter(plugin=plugin) as sub:
        sub.run(multiple_x)

    # checking the results
    expected = [({"mult.y": 1}, [1, 2]), ({"mult.y": 2}, [2, 4])]
    results = multiple_x.result()
    combined_results = [[res.output.out for res in res_l] for res_l in results]
    for i, res in enumerate(expected):
        assert combined_results[i] == res[1]


# # multiply_x is an AppFuture
# multiple_x = multiply().split(['x', 'y'], x=range(10), y=range(10)).combine(['y', 'x'])
# this would return a nested list of lists as a result.


@pytest.mark.parametrize("plugin", Plugins)
def test_4(plugin):
    multiple_x = (
        multiply(name="mult").split(["x", "y"], x=[1, 2], y=[1, 2]).combine(["y", "x"])
    )

    assert multiple_x.state.splitter == ["mult.x", "mult.y"]
    assert multiple_x.state.combiner == ["mult.y", "mult.x"]

    with Submitter(plugin=plugin) as sub:
        sub.run(multiple_x)

    # checking the results
    expected = [({}, [1, 2, 2, 4])]
    results = multiple_x.result()
    combined_results = [[res.output.out for res in res_l] for res_l in results]
    for i, res in enumerate(expected):
        assert combined_results[i] == res[1]


# in the following example, note parentheses instead of brackets. this is our syntax for synchronized parallelism
# # multiply_x is an AppFuture
# multiple_x = multiply().split(('x', 'y'), x=range(10), y=range(10)).combine('x')
# this will return 0 * 0, 1 * 1, and so on. and we can have any combination thereof.


@pytest.mark.parametrize("plugin", Plugins)
def test_5(plugin):
    multiple_x = multiply(name="mult").split(("x", "y"), x=[1, 2], y=[1, 2])

    assert multiple_x.state.splitter == ("mult.x", "mult.y")

    with Submitter(plugin=plugin) as sub:
        sub.run(multiple_x)

    # checking the results
    expected = [({"mult.x": 1, "mult.y": 1}, 1), ({"mult.x": 2, "mult.y": 2}, 4)]
    results = multiple_x.result()
    for i, res in enumerate(expected):
        assert results[i].output.out == res[1]


@pytest.mark.parametrize("plugin", Plugins)
def test_6(plugin):
    multiple_x = (
        multiply(name="mult").split(("x", "y"), x=[1, 2], y=[1, 2]).combine(["x"])
    )

    assert multiple_x.state.splitter == ("mult.x", "mult.y")
    assert multiple_x.state.combiner == ["mult.x"]

    with Submitter(plugin=plugin) as sub:
        sub.run(multiple_x)

    # checking the results
    expected = [({}, [1, 4])]
    results = multiple_x.result()
    combined_results = [[res.output.out for res in res_l] for res_l in results]
    for i, res in enumerate(expected):
        assert combined_results[i] == res[1]


# TODO: have to use workflow
# # multiply_x is an AppFuture
# multiple_x = multiply().split(('x', 'y'), x=range(10), y=range(10))
# add_x = add2(x='multiply.out').combine('multiply.x')


@to_task
def add2(x):
    return x + 2


@pytest.mark.xfail(reason="finish after futures")
@pytest.mark.parametrize("plugin", Plugins)
def test_7(plugin):
    """Test workflow with workflow level splitters and combiners"""
    wf = Workflow(name="test7", input_spec=["x", "y"])
    wf.add(multiply(name="mult", inputs=dict(x=wf.inputs.x, y=wf.inputs.y)))
    wf.add(add2(name="add2", x=wf.mult.output.out))

    wf.split(("x", "y"), x=[1, 2], y=[1, 2])
    wf.combine("x")
    wf.set_output([("out", wf.add2.output.out)])

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    results = wf.result()
    expected = [({"test7.x": 1, "test7.y": 1}, 1), ({"test7.x": 2, "test.y": 2}, 4)]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]


@pytest.mark.xfail(reason="finish after futures")
@pytest.mark.parametrize("plugin", Plugins)
def test_8(plugin):
    """Test workflow with node level splitters and combiners"""
    wf = Workflow(name="test7", input_spec=["x", "y"])
    wf.add(multiply(name="mult").split(("x", "y"), x=wf.inputs.x, y=wf.inputs.y))
    wf.add(add2(name="add2", x=wf.mult.result.out).combine("x"))
    wf.set_output([("out", wf.add2.result.out)])

    with Submitter(plugin=plugin) as sub:
        sub.run(wf)

    # checking the results
    results = wf.result()
    expected = [({"mult.x": 1, "mult.y": 1}, 1), ({"mult.x": 2, "mult.y": 2}, 4)]

    for i, res in enumerate(expected):
        assert results["out"][i][0] == res[0]
        assert results["out"][i][1] == res[1]
