import sys, os
import numpy as np
from pathlib import Path

from nipype.utils.filemanip import save_json, makedirs, to_str
from nipype.interfaces import fsl
from nipype import Function

from ..node import Node, Workflow
from ..auxiliary import CurrentInterface
from ..submitter import Submitter

import pytest
import pdb

TEST_DATA_DIR = Path(os.getenv('PYDRA_TEST_DATA', '/nonexistent/path'))
DS114_DIR = TEST_DATA_DIR / 'ds000114'

python35_only = pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python>3.4")


@pytest.fixture(scope="module")
def change_dir(request):
    orig_dir = os.getcwd()
    test_dir = os.path.join(orig_dir, "test_outputs_combine")
    makedirs(test_dir, exist_ok=True)
    os.chdir(test_dir)

    def move2orig():
        os.chdir(orig_dir)

    request.addfinalizer(move2orig)


Plugins = ["serial"]
Plugins = ["serial", "mp", "cf", "dask"]


def fun_addtwo(a):
    import time
    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2

_interf_addtwo = Function(function=fun_addtwo, input_names=["a"], output_names=["out"])
interf_addtwo = CurrentInterface(interface=_interf_addtwo, name="addtwo")


def fun_addvar(b, c):
    return b + c

_interf_addvar = Function(function=fun_addvar, input_names=["b", "c"], output_names=["out"])
interf_addvar = CurrentInterface(interface=_interf_addvar, name="addvar")


def fun_addvar3(a, b, c):
    return a + b + c

_interf_addvar3 = Function(function=fun_addvar3, input_names=["a", "b", "c"], output_names=["out"])
interf_addvar3 = CurrentInterface(interface=_interf_addvar3, name="addvar3")


def fun_addvar4(a, b, c, d):
    return a + b + c + d

_interf_addvar4 = Function(function=fun_addvar4, input_names=["a", "b", "c", "d"], output_names=["out"])
interf_addvar4 = CurrentInterface(interface=_interf_addvar4, name="addvar4")


def fun_sumlist(a):
    return sum(a)

_interf_sumlist = Function(function=fun_sumlist, input_names=["a"], output_names=["out"])
interf_sumlist = CurrentInterface(interface=_interf_sumlist, name="sumlist")


# initializing nodes, setting containers

def test_node_combine_1():
    """Node with mandatory arguments only"""
    nn = Node(name="NA", interface=interf_addtwo)
    assert nn.splitter is None
    assert nn.combiner is None


def test_node_combine_2():
    """Node with interface, inputs and splitter"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]}, splitter="a")
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()
    assert nn.state._splitter == "NA.a"
    assert nn.combiner is None


def test_node_combine_3():
    """Node with interface, inputs, splitter and combiner"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a", combiner="a")
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()
    assert nn.state._splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_3a():
    """Node with interface, inputs, splitter and combiner (with node name)"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a", combiner="NA.a")
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()
    assert nn.state._splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_3b():
    """Node with interface, inputs and combiner (without splitter)"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addtwo, inputs={"a": 3},
                  combiner="a")
    assert str(excinfo.value) ==\
           "splitter has to be set before setting combiner"


def test_node_combine_4():
    """Node with interface,  two inputs, splitter and combiner"""
    nn = Node(name="NA", interface=interf_addvar,
              inputs={"a": [3, 5], "b": [10, 20]},
              splitter=["a", "b"], combiner=["a", "b"])
    assert nn.splitter == ["NA.a", "NA.b"]
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()
    assert (nn.inputs["NA.b"] == np.array([10, 20])).all()
    assert nn.state._splitter == ["NA.a", "NA.b"]
    assert nn.combiner == ["NA.a", "NA.b"]


def test_node_combine_4a():
    """Node with interface,  two inputs, splitter and combiner
        (different order for combiner)
    """
    nn = Node(name="NA", interface=interf_addvar,
              inputs={"a": [3, 5], "b": [10, 20]},
              splitter=["a", "b"], combiner=["b", "a"])
    assert nn.splitter == ["NA.a", "NA.b"]
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()
    assert (nn.inputs["NA.b"] == np.array([10, 20])).all()
    assert nn.state._splitter == ["NA.a", "NA.b"]
    assert nn.combiner == ["NA.b", "NA.a"]


def test_node_combine_5():
    """Node with interface, inputs, splitter, combiner set later"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a")
    nn.combiner = "a"
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_5a():
    """Node with interface, inputs, splitter, combiner set later (using node name)"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a")
    nn.combiner = "NA.a"
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_5b():
    """Node with interface, inputs, splitter, setting combiner twice"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a", combiner="a")
    with pytest.raises(Exception) as excinfo:
        nn.combiner = "a"
    assert str(excinfo.value) == "combiner is already set"


def test_node_combine_6():
    """Node with interface, inputs, splitter, combiner set by combine method"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a")
    nn.combine(combiner="a")
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_6a():
    """Node with interface, inputs, splitter and combiner set in one chain"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]})
    nn.split(splitter="a").combine(combiner="a")
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]


def test_node_combine_6b():
    """Node with interface, inputs, splitter, setting combiner twice"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a", combiner="a")
    with pytest.raises(Exception) as excinfo:
        nn.combine(combiner="a")
    assert str(excinfo.value) == "combiner is already set"


def test_node_combine_6c():
    """Node with interface, inputs, combiner set by combine method (no splitter)"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]})
    with pytest.raises(Exception) as excinfo:
        nn.combine(combiner="a")
    assert str(excinfo.value) ==\
           "splitter has to be set before setting combiner"


def test_node_combine_7():
    """Node with interface, inputs, combiner set by combine method (no splitter)"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]}, splitter="a")
    with pytest.raises(Exception) as excinfo:
        nn.combine(combiner="b")
    assert str(excinfo.value) ==\
           "element NA.b of combiner is not found in the splitter NA.a"


# testing prepare state inputs

def test_node_combine_8():
    """Node with interface, inputs, splitter and combiner"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]},
              splitter="a", combiner="a")
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]

    nn.prepare_state_input()
    assert nn.state._input_for_axis == [["NA.a"]]


def test_node_combine_9():
    """Node with interface,  two inputs, splitter and combiner"""
    nn = Node(name="NA", interface=interf_addvar,
              inputs={"a": [3, 5], "b": [10, 20]},
              splitter=["a", "b"], combiner=["a", "b"])
    assert nn.splitter == ["NA.a", "NA.b"]
    assert nn.combiner == ["NA.a", "NA.b"]

    nn.prepare_state_input()
    assert nn.state._input_for_axis == [["NA.a"], ["NA.b"]]

# running nodes with combiner

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_10(plugin, change_dir):
    """simplest combiner in init, running the node"""
    nn = Node(name="NA", interface=interf_addtwo, workingdir="test_nd10_{}".format(plugin),
              output_names=["out"], splitter="a", combiner="a", inputs={"a": [3, 5]})

    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_10a(plugin, change_dir):
    """simplest combiner by combine, running the node"""
    nn = Node(name="NA", interface=interf_addtwo, workingdir="test_nd10a_{}".format(plugin),
              output_names=["out"], inputs={"a": [3, 5]})
    nn.split(splitter="a").combine(combiner="a")
    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_10b(plugin, change_dir):
    """simplest combiner in init, running the node"""
    nn = Node(name="NA", interface=interf_addtwo, workingdir="test_nd10b_{}".format(plugin),
              output_names=["out"], splitter="a", inputs={"a": [3, 5]})
    nn.combiner = "a"

    assert nn.splitter == "NA.a"
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]




@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_11(plugin, change_dir):
    """Node with interface, inputs and scalar splitter, running interface"""
    nn = Node(name="NA", interface=interf_addvar, workingdir="test_nd11_{}".format(plugin),
              output_names=["out"], splitter=("b", "c"), combiner="b",
              inputs={"b": [3, 5], "c": [2, 1]})

    assert nn.splitter == ("NA.b", "NA.c")
    assert nn.combiner == ["NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [[({"NA.b": 3, "NA.c": 2}, 5), ({"NA.b": 5, "NA.c": 1}, 6)]]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_11a(plugin, change_dir):
    """Node with interface, inputs and scalar splitter
    trying to combine with both elements of scalar splitter
    """
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar, workingdir="test_nd11a_{}".format(plugin),
                  output_names=["out"], splitter=("b", "c"), combiner=["b", "c"],
                  inputs={"b": [3, 5], "c": [2, 1]})
    assert "already removed" in str(excinfo.value)


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_12(plugin, change_dir):
    """Node with interface, inputs and vector splitter, running interface"""
    nn = Node(name="NA", interface=interf_addvar, workingdir="test_nd12_{}".format(plugin),
              output_names=["out"], splitter=["b", "c"], combiner="b",
              inputs={"b": [3, 5], "c": [2, 1]})

    assert nn.splitter == ["NA.b", "NA.c"]
    assert nn.combiner == ["NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.b": 5, "NA.c": 1}, 6), ({"NA.b": 3, "NA.c": 1}, 4)],
        [({"NA.b": 3, "NA.c": 2}, 5), ({"NA.b": 5, "NA.c": 2}, 7)]
    ]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]


# run node with combiner (more complicated splitter)

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_13(plugin, change_dir):
    """two outer splitters, one combiner"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd13_{}".format(plugin),
              output_names=["out"], splitter=["a", ["b", "c"]], combiner="a",
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ["NA.b", "NA.c"]]
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5)],
        [({"NA.a": 1, "NA.b": 1, "NA.c": 2}, 4), ({"NA.a": 2, "NA.b": 1, "NA.c": 2}, 5),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 2}, 6)],
        [({"NA.a": 1, "NA.b": 2, "NA.c": 1}, 4), ({"NA.a": 2, "NA.b": 2, "NA.c": 1}, 5),
         ({"NA.a": 3, "NA.b": 2, "NA.c": 1}, 6)],
        [({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5), ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6),
         ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['NA.b:1_NA.c:1', 'NA.b:1_NA.c:2', 'NA.b:2_NA.c:1', 'NA.b:2_NA.c:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_13a(plugin, change_dir):
    """two outer splitters, two combiners"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd13a_{}".format(plugin),
              output_names=["out"], splitter=["a", ["b", "c"]], combiner=["a", "b"],
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ["NA.b", "NA.c"]]
    assert nn.combiner == ["NA.a", "NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5), ({"NA.a": 1, "NA.b": 2, "NA.c": 1}, 4),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 1}, 5), ({"NA.a": 3, "NA.b": 2, "NA.c": 1}, 6)],
        [({"NA.a": 1, "NA.b": 1, "NA.c": 2}, 4), ({"NA.a": 2, "NA.b": 1, "NA.c": 2}, 5),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 2}, 6), ({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6), ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['NA.c:1', 'NA.c:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_13b(plugin, change_dir):
    """two outer splitters, three combiners"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd13b_{}".format(plugin),
              output_names=["out"], splitter=["a", ["b", "c"]], combiner=["a", "b", "c"],
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ["NA.b", "NA.c"]]
    assert nn.combiner == ["NA.a", "NA.b", "NA.c"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5), ({"NA.a": 1, "NA.b": 2, "NA.c": 1}, 4),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 1}, 5), ({"NA.a": 3, "NA.b": 2, "NA.c": 1}, 6),
         ({"NA.a": 1, "NA.b": 1, "NA.c": 2}, 4), ({"NA.a": 2, "NA.b": 1, "NA.c": 2}, 5),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 2}, 6), ({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6), ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state=[""]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14(plugin, change_dir):
    """outer and scalar splitter, one combiner (from outer part)"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd14_{}".format(plugin),
              output_names=["out"], splitter=["a", ("b", "c")], combiner="a",
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ("NA.b", "NA.c")]
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5)],
         [({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5), ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6),
         ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['NA.b:1_NA.c:1', 'NA.b:2_NA.c:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14a(plugin, change_dir):
    """outer and scalar splitter, two combiners (one from outer and one from scalar part)"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd14a_{}".format(plugin),
              output_names=["out"], splitter=["a", ("b", "c")], combiner=["a", "b"],
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ("NA.b", "NA.c")]
    assert nn.combiner == ["NA.a", "NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5), ({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6), ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['']
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14b(plugin, change_dir):
    """outer and scalar splitter, two combiners (second one from scalar part - should be the same)"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd14b_{}".format(plugin),
              output_names=["out"], splitter=["a", ("b", "c")], combiner=["a", "c"],
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ("NA.b", "NA.c")]
    assert nn.combiner == ["NA.a", "NA.c"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4),
         ({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5), ({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5),
         ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6), ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['']
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14c(plugin, change_dir):
    """outer and scalar splitter, two combiners (second one from scalar part - should be the same)"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd14c_{}".format(plugin),
                  output_names=["out"], splitter=["a", ("b", "c")], combiner=["a", "b", "c"],
                  inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})
    assert "already removed" in str(excinfo.value)


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14d(plugin, change_dir):
    """outer and scalar splitter, one combiner (from outer part)"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd14d_{}".format(plugin),
              output_names=["out"], splitter=["a", ("b", "c")], combiner="b",
              inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ["NA.a", ("NA.b", "NA.c")]
    assert nn.combiner == ["NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1}, 3), ({"NA.a": 1, "NA.b": 2, "NA.c": 2}, 5)],
        [({"NA.a": 2, "NA.b": 1, "NA.c": 1}, 4), ({"NA.a": 2, "NA.b": 2, "NA.c": 2}, 6)],
        [({"NA.a": 3, "NA.b": 1, "NA.c": 1}, 5), ({"NA.a": 3, "NA.b": 2, "NA.c": 2}, 7)],
    ]
    expected_state = ['NA.a:1', 'NA.a:2', 'NA.a:3']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_14e(plugin, change_dir):
    """outer and scalar splitter, two combiners (second one from scalar part - should be the same)"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar3, output_names=["out"],
                  workingdir="test_nd14e_{}".format(plugin),
                  splitter=["a", ("b", "c")], combiner=["b", "c"],
                  inputs={"a": [1, 2, 3], "b": [1, 2], "c": [1, 2]})
    assert "already removed" in str(excinfo.value)


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_15(plugin, change_dir):
    """scalar and outer splitter, one combiner (from scalar part)"""
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd15_{}".format(plugin),
              output_names=["out"], splitter=("a", ["b", "c"]), combiner="a",
              inputs={"a": [[11, 12], [21, 22]], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ("NA.a", ["NA.b", "NA.c"])
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 11, "NA.b": 1, "NA.c": 1}, 13), ({"NA.a": 12, "NA.b": 1, "NA.c": 2}, 15),
         ({"NA.a": 21, "NA.b": 2, "NA.c": 1}, 24), ({"NA.a": 22, "NA.b": 2, "NA.c": 2}, 26)],
    ]
    expected_state = ['']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_15a(plugin, change_dir):
    """scalar and outer splitter, two combiners (from scalar and outer parts)"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar3, output_names=["out"],
                  workingdir="test_nd15a_{}".format(plugin),
                  splitter=("a", ["b", "c"]), combiner=["a", "b"],
                  inputs={"a": [[11, 12], [21, 22]], "b": [1, 2], "c": [1, 2]})
    assert "already removed" in str(excinfo.value)


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_15b(plugin, change_dir):
    """scalar and outer splitter, one combiner (from outer part)
    We assumed that this is OK, but will eliminate also a
    """
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd15b_{}".format(plugin),
              output_names=["out"], splitter=("a", ["b", "c"]), combiner="b",
              inputs={"a": [[11, 12], [21, 22]], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ("NA.a", ["NA.b", "NA.c"])
    assert nn.combiner == ["NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 11, "NA.b": 1, "NA.c": 1}, 13), ({"NA.a": 21, "NA.b": 2, "NA.c": 1}, 24)],
        [({"NA.a": 12, "NA.b": 1, "NA.c": 2}, 15), ({"NA.a": 22, "NA.b": 2, "NA.c": 2}, 26)],
    ]
    expected_state = ['NA.c:1', 'NA.c:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_15c(plugin, change_dir):
    """scalar and outer splitter, one combiner (from outer part)
    We assumed that this is OK, but will eliminate also a
    """
    nn = Node(name="NA", interface=interf_addvar3, workingdir="test_nd15c_{}".format(plugin),
              output_names=["out"], splitter=("a", ["b", "c"]), combiner="c",
              inputs={"a": [[11, 12], [21, 22]], "b": [1, 2], "c": [1, 2]})

    assert nn.splitter == ("NA.a", ["NA.b", "NA.c"])
    assert nn.combiner == ["NA.c"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 11, "NA.b": 1, "NA.c": 1}, 13), ({"NA.a": 12, "NA.b": 1, "NA.c": 2}, 15)],
        [({"NA.a": 21, "NA.b": 2, "NA.c": 1}, 24), ({"NA.a": 22, "NA.b": 2, "NA.c": 2}, 26)],
    ]
    expected_state = ['NA.b:1', 'NA.b:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_15d(plugin, change_dir):
    """scalar and outer splitter, one combiner (from outer part)
    We assumed that this is OK, but will eliminate also a
    """
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar3, output_names=["out"],
                  workingdir="test_nd15d_{}".format(plugin),
                  splitter=("a", ["b", "c"]), combiner=["c", "a"],
                  inputs={"a": [[11, 12], [21, 22]], "b": [1, 2], "c": [1, 2]})
    assert "already removed" in str(excinfo.value)


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_16(plugin, change_dir):
    """scalar and outer splitter, one combiner"""
    nn = Node(name="NA", interface=interf_addvar4, workingdir="test_nd16_{}".format(plugin),
              output_names=["out"], splitter=(["a", "b"], ["c", "d"]), combiner="a",
              inputs={"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1,2]})

    assert nn.splitter == (["NA.a", "NA.b"], ["NA.c", "NA.d"])
    assert nn.combiner == ["NA.a"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1, "NA.d": 1}, 4), ({"NA.a": 2, "NA.b": 1, "NA.c": 2, "NA.d": 1}, 6)],
        [({"NA.a": 1, "NA.b": 2, "NA.c": 1, "NA.d": 2}, 6), ({"NA.a": 2, "NA.b": 2, "NA.c": 2, "NA.d": 2}, 8)],
    ]
    expected_state = ['NA.b:1_NA.d:1', 'NA.b:2_NA.d:2']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_16a(plugin, change_dir):
    """scalar and outer splitter, two combiners"""
    nn = Node(name="NA", interface=interf_addvar4, workingdir="test_nd16a_{}".format(plugin),
              output_names=["out"], splitter=(["a", "b"], ["c", "d"]), combiner=["a", "b"],
              inputs={"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1,2]})

    assert nn.splitter == (["NA.a", "NA.b"], ["NA.c", "NA.d"])
    assert nn.combiner == ["NA.a", "NA.b"]

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [
        [({"NA.a": 1, "NA.b": 1, "NA.c": 1, "NA.d": 1}, 4), ({"NA.a": 2, "NA.b": 1, "NA.c": 2, "NA.d": 1}, 6),
         ({"NA.a": 1, "NA.b": 2, "NA.c": 1, "NA.d": 2}, 6), ({"NA.a": 2, "NA.b": 2, "NA.c": 2, "NA.d": 2}, 8)],
    ]
    expected_state = ['']

    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    nn.result["out"].sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in nn.result["out"]]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert nn.result["out"][i][1][j][0] == res[0]
            assert nn.result["out"][i][1][j][1] == res[1]
            assert nn.result["out"][i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_combine_16b(plugin, change_dir):
    """scalar and outer splitter, two combiner from the same axis (exception)"""
    with pytest.raises(Exception) as excinfo:
        nn = Node(name="NA", interface=interf_addvar4, output_names=["out"],
                  workingdir="test_nd16b_{}".format(plugin),
                  splitter=(["a", "b"], ["c", "d"]), combiner=["a", "c"],
                  inputs={"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1,2]})
    assert "already removed" in str(excinfo.value)


# # tests for workflows


@python35_only
def test_workflow_combine_0(plugin="serial"):
    """workflow (without run) with one node with a splitter"""
    wf = Workflow(name="wf0", workingdir="test_wf0_{}".format(plugin))
    # defining a node with splitter and inputs first
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["a"])
    na.split(splitter="a", inputs={"a": [3, 5]}).combine(combiner="a")
    # one of the way of adding nodes to the workflow
    wf.add_nodes([na])
    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[0].combiner == ["NA.a"]
    assert (wf.nodes[0].inputs['NA.a'] == np.array([3, 5])).all()
    assert len(wf.graph.nodes) == 1


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_1(plugin, change_dir):
    """workflow with one node with a splitter and a combiner"""
    wf = Workflow(name="wf1", workingdir="test_wf1_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"],
              splitter="a", combiner="a", inputs={"a": [3, 5]})
    wf.add_nodes([na])

    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[0].combiner == ["NA.a"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_1a(plugin, change_dir):
    """workflow with one node with a splitter and a combiner (checking results of the wf)"""
    wf = Workflow(name="wf1", workingdir="test_wf1a_{}".format(plugin),
                  wf_output_names=[("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]}).combine(combiner="a")
    wf.add_nodes([na])

    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[0].combiner == ["NA.a"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_2(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner"""
    wf = Workflow(name="wf2", workingdir="test_wf2_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]}).combine(combiner="a")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[0].combiner == ["NA.a"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expected_B = [({}, 12)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        for key in res[0].keys():
            assert (wf.nodes[1].result["out"][i][0][key] == res[0][key]).all()
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        for key in res[0].keys():
            assert (wf.nodes[1].result["out"][i][0][key] == res[0][key]).all()
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_3(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner (different than splitter)"""
    wf = Workflow(name="wf3", workingdir="test_wf3_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar, workingdir="na", output_names=["out"])
    na.split(splitter=["b", "c"], inputs={"b": [3, 5, 7], "c": [10, 20]}).combine(combiner="b")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == ["NA.b", "NA.c"]
    assert wf.nodes[0].combiner == ["NA.b"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.b": 3, "NA.c": 10}, 13), ({"NA.b": 5, "NA.c": 10}, 15), ({"NA.b": 7, "NA.c": 10}, 17)],
                [({"NA.b": 3, "NA.c": 20}, 23), ({"NA.b": 5, "NA.c": 20}, 25), ({"NA.b": 7, "NA.c": 20}, 27)]]
    expected_state = ["NA.c:10", "NA.c:20"]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


    expectedB = [({"NA.c": 10}, 45), ({"NA.c": 20}, 75)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]


    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_3a(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner (different than splitter)"""
    wf = Workflow(name="wf3a", workingdir="test_wf3a_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar, workingdir="na", output_names=["out"])
    na.split(splitter=["b", "c"], inputs={"b": [3, 5, 7], "c": [10, 20]}).combine(combiner="c")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == ["NA.b", "NA.c"]
    assert wf.nodes[0].combiner == ["NA.c"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.b": 3, "NA.c": 10}, 13), ({"NA.b": 3, "NA.c": 20}, 23)],
                [({"NA.b": 5, "NA.c": 10}, 15), ({"NA.b": 5, "NA.c": 20}, 25)],
                [({"NA.b": 7, "NA.c": 10}, 17), ({"NA.b": 7, "NA.c": 20}, 27)]]
    expected_state = ["NA.b:3", "NA.b:5", "NA.b:7"]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({"NA.b": 3}, 36), ({"NA.b": 5}, 40), ({"NA.b": 7}, 44)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_3b(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner (both axes)"""
    wf = Workflow(name="wf3b", workingdir="test_wf3b_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar, workingdir="na", output_names=["out"])
    na.split(splitter=["b", "c"], inputs={"b": [3, 5, 7], "c": [10, 20]}).combine(combiner=["b", "c"])

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == ["NA.b", "NA.c"]
    assert wf.nodes[0].combiner == ["NA.b", "NA.c"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.b": 3, "NA.c": 10}, 13), ({"NA.b": 3, "NA.c": 20}, 23),
                 ({"NA.b": 5, "NA.c": 10}, 15), ({"NA.b": 5, "NA.c": 20}, 25),
                 ({"NA.b": 7, "NA.c": 10}, 17), ({"NA.b": 7, "NA.c": 20}, 27)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({}, 120)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_4(plugin, change_dir):
    """ using add(node) method
        using wf.connect to connect two nodes
    """
    wf = Workflow(name="wf4", workingdir="test_wf4_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    wf.add(na)
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # explicit splitter with a variable from the previous node
    # providing inputs with b
    nb.split(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
    wf.add(nb)
    # connect method as it is in the current version
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expected_B = [({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]



# # using split after add method


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_5(plugin, change_dir):
    """using split_node and combiner_node methods for one node"""
    wf = Workflow(name="wf5", workingdir="test_wf5_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na)
    # using the split_node method after add (using splitter for the last added node as default)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.combine_node(combiner="a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_5a(plugin, change_dir):
    """using split_node and combine_node methods for one node in one chain"""
    wf = Workflow(name="wf5a", workingdir="test_wf5a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na).split_node(splitter="a", inputs={"a": [3, 5]}).combine_node(combiner="a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_6(plugin, change_dir):
    """using split_node and combine_node methods for two nodes (using last added node as default)"""
    wf = Workflow(name="wf6", workingdir="test_wf6_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}).combine_node(combiner="c")
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_6a(plugin, change_dir):
    """using split_node and combine_node methods for two nodes (specifying the node)"""
    wf = Workflow(name="wf6a", workingdir="test_wf6a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split method after add (specifying the node)
    wf.add(na)
    wf.add(nb)
    wf.split_node(splitter="a", inputs={"a": [3, 5]}, node=na)
    # TODO: should we se ("b", "c") instead?? shold I forget "NA.a" value?
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}, node=nb)
    wf.combine_node(combiner="c", node=nb)
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_6b(plugin, change_dir):
    """using split_node and combine_node methods for two nodes (specifying the node),
    using kwarg arg instead of connect
    """
    wf = Workflow(name="wf6b", workingdir="test_wf6b_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])

    wf.add(na)
    wf.add(nb, b="NA.out")
    wf.split_node(splitter="a", inputs={"a": [3, 5]}, node=na)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}, node=nb)
    wf.combine_node(combiner="c", node=nb)

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


# combiner from previous node

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_7(plugin, change_dir):
    """using split_node and combine_node methods for two nodes,
        using combiner from previous node, scalar splitter
    """
    wf = Workflow(name="wf7", workingdir="test_wf7_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}).combine_node(combiner="NA.a")
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_7a(plugin, change_dir):
    """using split_node and combine_node methods for two nodes
        using combiner from previous node, outer splitter
    """
    wf = Workflow(name="wf7a", workingdir="test_wf7a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=["NA.a", "c"], inputs={"c": [2, 1]}).combine_node(combiner="NA.a")
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 1}, 6), ({"NA.a": 5, "NB.c": 1}, 8)],
                 [({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 2}, 9)]]
    expectedB_state = ["NB.c:1", "NB.c:2"]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


# splitters with three input

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_8(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner (different than splitter)"""
    wf = Workflow(name="wf8", workingdir="test_wf8_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar3, workingdir="na", output_names=["out"])
    na.split(splitter=[("a", "b"), "c"], inputs={"a": [1, 2], "b": [3, 5], "c": [10, 20, 30]})\
        .combine(combiner="b")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == [("NA.a", "NA.b"), "NA.c"]
    assert wf.nodes[0].combiner == ["NA.b"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 1, "NA.b": 3, "NA.c": 10}, 14), ({"NA.a": 2, "NA.b": 5, "NA.c": 10}, 17)],
                [({"NA.a": 1, "NA.b": 3, "NA.c": 20}, 24), ({"NA.a": 2, "NA.b": 5, "NA.c": 20}, 27)],
                [({"NA.a": 1, "NA.b": 3, "NA.c": 30}, 34), ({"NA.a": 2, "NA.b": 5, "NA.c": 30}, 37)]]
    expected_state = ["NA.c:10", "NA.c:20", "NA.c:30"]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({"NA.c": 10}, 31), ({"NA.c": 20}, 51), ({"NA.c": 30}, 71)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_9(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner,
        having a "partial combined" input from a scalar splitter
    """
    wf = Workflow(name="wf9", workingdir="test_wf9_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar3, workingdir="na", output_names=["out"])
    na.split(splitter=(["a", "b"], "c"), inputs={"a": [1, 2], "b": [3, 5, 7],
                                                 "c": [[10, 20, 30], [100, 200, 300]]})\
        .combine(combiner="a")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == (["NA.a", "NA.b"], "NA.c")
    assert wf.nodes[0].combiner == ["NA.a"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 1, "NA.b": 3, "NA.c": 10}, 14), ({"NA.a": 2, "NA.b": 3, "NA.c": 100}, 105)],
                [({"NA.a": 1, "NA.b": 5, "NA.c": 20}, 26), ({"NA.a": 2, "NA.b": 5, "NA.c": 200}, 207)],
                [({"NA.a": 1, "NA.b": 7, "NA.c": 30}, 38), ({"NA.a": 2, "NA.b": 7, "NA.c": 300}, 309)]]
    expected_state = ["NA.b:3", "NA.b:5", "NA.b:7"]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({"NA.b": 3}, 119), ({"NA.b": 5}, 233), ({"NA.b": 7}, 347)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_9a(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner,
        having a "partial combined" input from a scalar splitter
        (different combiner)
    """
    wf = Workflow(name="wf9a", workingdir="test_wf9a_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar3, workingdir="na", output_names=["out"])
    na.split(splitter=(["a", "b"], "c"), inputs={"a": [1, 2], "b": [3, 5, 7],
                                                 "c": [[10, 20, 30], [100, 200, 300]]})\
        .combine(combiner="b")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == (["NA.a", "NA.b"], "NA.c")
    assert wf.nodes[0].combiner == ["NA.b"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 1, "NA.b": 3, "NA.c": 10}, 14), ({"NA.a": 1, "NA.b": 5, "NA.c": 20}, 26),
                 ({"NA.a": 1, "NA.b": 7, "NA.c": 30}, 38)],
                [({"NA.a": 2, "NA.b": 3, "NA.c": 100}, 105), ({"NA.a": 2, "NA.b": 5, "NA.c": 200}, 207),
                 ({"NA.a": 2, "NA.b": 7, "NA.c": 300}, 309)]]
    expected_state = ["NA.a:1", "NA.a:2"]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({"NA.a": 1}, 78), ({"NA.a": 2}, 621)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_9b(plugin, change_dir):
    """workflow with two nodes, first one with splitter and combiner,
        having a "partial combined" input from a scalar splitter
        (different combiner)
    """
    wf = Workflow(name="wf9b", workingdir="test_wf9b_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addvar3, workingdir="na", output_names=["out"])
    na.split(splitter=(["a", "b"], "c"), inputs={"a": [1, 2], "b": [3, 5, 7],
                                                 "c": [[10, 20, 30], [100, 200, 300]]})\
        .combine(combiner="c")

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_sumlist, workingdir="nb", output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "a")

    assert wf.nodes[0].splitter == (["NA.a", "NA.b"], "NA.c")
    assert wf.nodes[0].combiner == ["NA.c"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 1, "NA.b": 3, "NA.c": 10}, 14), ({"NA.a": 1, "NA.b": 5, "NA.c": 20}, 26),
                 ({"NA.a": 1, "NA.b": 7, "NA.c": 30}, 38), ({"NA.a": 2, "NA.b": 3, "NA.c": 100}, 105),
                 ({"NA.a": 2, "NA.b": 5, "NA.c": 200}, 207), ({"NA.a": 2, "NA.b": 7, "NA.c": 300}, 309)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]

    expectedB = [({}, 699)]
    key_sort = list(expectedB[0][0].keys())
    expectedB.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expectedB):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]



# tests for a workflow that have its own input


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_10(plugin, change_dir):
    """using inputs for workflow and connect_workflow"""
    # adding inputs to the workflow directly
    wf = Workflow(name="wf10", inputs={"wfa": [3, 5]}, workingdir="test_wf10_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na)
    # connecting the node with inputs from the workflow
    wf.connect_wf_input("wfa", "NA", "a")
    wf.split_node(splitter="a").combine_node(combiner="a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [[({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]]
    expected_state = [""]
    key_sort = list(expected[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expected]
    results = wf.nodes[0].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expected):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expected_state[i]


# # testing if _NA in splitter works, using interfaces in add


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_11(plugin, change_dir):
    """using add(interface) method and splitter from previous nodes
        combiner in the second node
    """
    wf = Workflow(name="wf11", workingdir="test_wf11_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addtwo, workingdir="na",
           output_names=["out"]).split_node(splitter="a", inputs={"a": [3, 5]})
    # _NA means that I'm using splitter from the NA node, it's the same as ("NA.a", "b")
    wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
           output_names=["out"]).split_node(splitter=("_NA", "c"), inputs={"c": [2, 1]})\
        .combine_node(combiner="c")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


@pytest.mark.xfail(reason="combiner from previus node doesn't work in this case")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_combine_11a(plugin, change_dir):
    """using add(interface) method and splitter from previous nodes
        combiner in the second node
    """
    wf = Workflow(name="wf11a", workingdir="test_wf11a_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addtwo, workingdir="na",
           output_names=["out"]).split_node(splitter="a", inputs={"a": [3, 5]})
    # _NA means that I'm using splitter from the NA node, it's the same as ("NA.a", "b")
    wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
           output_names=["out"]).split_node(splitter=("_NA", "c"), inputs={"c": [2, 1]})\
        .combine_node(combiner="NA.a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expectedB = [[({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]]
    expectedB_state = [""]
    key_sort = list(expectedB[0][0][0].keys())
    [exp.sort(key=lambda t: [t[0][key] for key in key_sort]) for exp in expectedB]
    results = wf.nodes[1].result["out"]
    results.sort()
    [res[1].sort(key=lambda t: [t[0][key] for key in key_sort]) for res in results]
    for i, res_comb in enumerate(expectedB):
        for j, res in enumerate(res_comb):
            assert results[i][1][j][0] == res[0]
            assert results[i][1][j][1] == res[1]
            assert results[i][0] == expectedB_state[i]


# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_10(plugin, change_dir):
#     """using add(interface) method and scalar splitter from previous nodes"""
#     wf = Workflow(name="wf10", workingdir="test_wf10_{}".format(plugin))
#     wf.add(name="NA", runnable=interf_addvar, workingdir="na",
#            output_names=["out"]).split_node(
#         splitter=("b", "c"), inputs={"b": [3, 5], "c": [0, 10]})
#     # _NA means that I'm using splitter from the NA node, it's the same as (("NA.a", NA.b), "b")
#     wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
#            output_names=["out"]).split_node(splitter=("_NA", "c"), inputs={"c": [2, 1]})
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 5, "NA.c": 10}, 15)]
#     key_sort = list(expected[0][0].keys())
#     expected.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected):
#         assert wf.nodes[0].result["out"][i][0] == res[0]
#         assert wf.nodes[0].result["out"][i][1] == res[1]
#
#     expected_B = [({"NA.b": 3, "NA.c": 0, "NB.c": 2}, 5), ({"NA.b": 5, "NA.c": 10, "NB.c": 1}, 16)]
#     key_sort = list(expected_B[0][0].keys())
#     expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected_B):
#         assert wf.nodes[1].result["out"][i][0] == res[0]
#         assert wf.nodes[1].result["out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_10a(plugin, change_dir):
#     """using add(interface) method and vector splitter from previous nodes"""
#     wf = Workflow(name="wf10a", workingdir="test_wf10a_{}".format(plugin))
#     wf.add(name="NA", runnable=interf_addvar, workingdir="na",
#            output_names=["out"]).split_node(
#             splitter=["b", "c"], inputs={"b": [3, 5], "c": [0, 10]})
#     # _NA means that I'm using splitter from the NA node, it's the same as (["NA.a", NA.b], "b")
#     wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
#            output_names=["out"]).split_node(
#             splitter=("_NA", "c"), inputs={"c": [[2, 1], [0, 0]]})
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 3, "NA.c": 10}, 13),
#                 ({"NA.b": 5, "NA.c": 0}, 5), ({"NA.b": 5, "NA.c": 10}, 15)]
#     key_sort = list(expected[0][0].keys())
#     expected.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected):
#         assert wf.nodes[0].result["out"][i][0] == res[0]
#         assert wf.nodes[0].result["out"][i][1] == res[1]
#
#     expected_B = [({"NA.b": 3, "NA.c": 0, "NB.c": 2}, 5),
#                   ({"NA.b": 3, "NA.c": 10, "NB.c": 1}, 14),
#                   ({"NA.b": 5, "NA.c": 0, "NB.c": 0}, 5),
#                   ({"NA.b": 5, "NA.c": 10, "NB.c": 0}, 15)]
#     key_sort = list(expected_B[0][0].keys())
#     expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected_B):
#         assert wf.nodes[1].result["out"][i][0] == res[0]
#         assert wf.nodes[1].result["out"][i][1] == res[1]
#
#
# # TODO: this test started sometimes failing for mp and cf
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_11(plugin, change_dir):
#     """using add(interface) method and vector splitter from previous two nodes"""
#     wf = Workflow(name="wf11", workingdir="test_wf11_{}".format(plugin))
#     wf.add(name="NA", runnable=interf_addvar, workingdir="na",
#            output_names=["out"]).split_node(
#             splitter=("b", "c"), inputs={"b": [3, 5],"c": [0, 10]})
#     wf.add(name="NB", runnable=interf_addtwo, workingdir="nb",
#            output_names=["out"]).split_node(splitter="a", inputs={"a": [2, 1]})
#     # _NA, _NB means that I'm using splitters from the NA/NB nodes, it's the same as [("NA.a", NA.b), "NB.a"]
#     wf.add(name="NC", runnable=interf_addvar, workingdir="nc", b="NA.out", c="NB.out",
#            output_names=["out"]).split_node(splitter=["_NA", "_NB"])  # TODO: this should eb default?
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 5, "NA.c": 10}, 15)]
#     key_sort = list(expected[0][0].keys())
#     expected.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected):
#         assert wf.nodes[0].result["out"][i][0] == res[0]
#         assert wf.nodes[0].result["out"][i][1] == res[1]
#
#     expected_C = [({"NA.b": 3,"NA.c": 0,"NB.a": 1}, 6),
#                   ({"NA.b": 3,"NA.c": 0,"NB.a": 2}, 7),
#                   ({"NA.b": 5,"NA.c": 10,"NB.a": 1}, 18),
#                   ({"NA.b": 5,"NA.c": 10,"NB.a": 2}, 19)]
#     key_sort = list(expected_C[0][0].keys())
#     expected_C.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.nodes[2].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected_C):
#         assert wf.nodes[2].result["out"][i][0] == res[0]
#         assert wf.nodes[2].result["out"][i][1] == res[1]
#
#
# # checking workflow.result
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_12(plugin, change_dir):
#     """testing if wf.result works (the same workflow as in test_workflow_6)"""
#     wf = Workflow(name="wf12", workingdir="test_wf12_{}".format(plugin),
#         wf_output_names=[("NA", "out", "NA_out"), ("NB", "out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
#     nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
#     # using the split methods after add (using splitter for the last added nodes as default)
#     wf.add(na)
#     wf.split_node(splitter="a", inputs={"a": [3, 5]})
#     wf.add(nb)
#     wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
#     wf.connect("NA", "out", "NB", "b")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     # checking if workflow.results is the same as results of nodes
#     assert wf.result["NA_out"] == wf.nodes[0].result["out"]
#     assert wf.result["out"] == wf.nodes[1].result["out"]
#
#     # checking values of workflow.result
#     expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
#     key_sort = list(expected[0][0].keys())
#     expected.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.result["NA_out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#
#     assert wf.is_complete
#     for i, res in enumerate(expected):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1] == res[1]
#
#     expected_B = [({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]
#     key_sort = list(expected_B[0][0].keys())
#     expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected_B):
#         assert wf.result["out"][i][0] == res[0]
#         assert wf.result["out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_12a(plugin, change_dir):
#     """testing if wf.result raises exceptione (the same workflow as in test_workflow_6)"""
#     wf = Workflow(name="wf12a",workingdir="test_wf12a_{}".format(plugin),
#         wf_output_names=[("NA", "out", "wf_out"), ("NB", "out", "wf_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
#     nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
#     # using the split methods after add (using splitter for the last added nodes as default)
#     wf.add(na)
#     wf.split_node(splitter="a", inputs={"a": [3, 5]})
#     wf.add(nb)
#     wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
#     wf.connect("NA", "out", "NB", "b")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     # wf_out can't be used twice
#     with pytest.raises(Exception) as exinfo:
#         sub.run()
#     assert str(exinfo.value) == "the key wf_out is already used in workflow.result"
#
#
# # tests for a workflow that have its own input and splitter
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_13(plugin, change_dir):
#     """using inputs for workflow and connect_wf_input"""
#     wf = Workflow(name="wf13", inputs={"wfa": [3, 5]},splitter="wfa",
#         workingdir="test_wf13_{}".format(plugin),
#         wf_output_names=[("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
#     wf.add(na)
#     wf.connect_wf_input("wfa", "NA", "a")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"wf13.wfa": 3}, [({"NA.a": 3}, 5)]), ({'wf13.wfa': 5}, [({"NA.a": 5}, 7)])]
#     for i, res in enumerate(expected):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
#         assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_13a(plugin, change_dir):
#     """using inputs for workflow and connect_wf_input (the node has 2 inputs)"""
#     wf = Workflow(name="wf13a", inputs={"wfa": [3, 5]}, splitter="wfa",
#         workingdir="test_wf13a_{}".format(plugin),
#         wf_output_names=[("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addvar, workingdir="na", splitter="c",
#               inputs={"c": [10, 20]}, output_names=["out"])
#     wf.add(na)
#     wf.connect_wf_input("wfa", "NA", "b")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"wf13a.wfa": 3}, [({"NA.b": 3, "NA.c": 10}, 13),
#                                     ({"NA.b": 3, "NA.c": 20}, 23)]),
#                 ({'wf13a.wfa': 5}, [({"NA.b": 5, "NA.c": 10}, 15),
#                                     ({"NA.b": 5, "NA.c": 20}, 25)])]
#     for i, res in enumerate(expected):
#         assert wf.result["NA_out"][i][0] == res[0]
#         for j in range(len(res[1])):
#             assert wf.result["NA_out"][i][1][j][0] == res[1][j][0]
#             assert wf.result["NA_out"][i][1][j][1] == res[1][j][1]
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_13b(plugin, change_dir):
#     """using inputs for workflow and connect_wf_input, using wf.split(splitter)"""
#     wf = Workflow(name="wf13b", inputs={"wfa": [3, 5]},
#         workingdir="test_wf13b_{}".format(plugin),
#         wf_output_names=[("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na",
#               output_names=["out"])
#     wf.add(na).split(splitter="wfa")
#     wf.connect_wf_input("wfa", "NA", "a")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"wf13b.wfa": 3}, [({"NA.a": 3}, 5)]),
#                 ({'wf13b.wfa': 5}, [({"NA.a": 5}, 7)])]
#     for i, res in enumerate(expected):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
#         assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]
#
#
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_13c(plugin, change_dir):
#     """using inputs for workflow and connect_wf_input, using wf.split(splitter, inputs)"""
#     wf = Workflow(name="wf13c", workingdir="test_wf13c_{}".format(plugin),
#         wf_output_names=[("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na",
#               output_names=["out"])
#     wf.add(na).split(splitter="wfa", inputs={"wfa": [3, 5]})
#     wf.connect_wf_input("wfa", "NA", "a")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"wf13c.wfa": 3}, [({"NA.a": 3}, 5)]), ({'wf13c.wfa': 5}, [({"NA.a": 5}, 7)])]
#     for i, res in enumerate(expected):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
#         assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]
#
#
# # workflow as a node
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_14(plugin, change_dir):
#     """workflow with a workflow as a node (no splitter)"""
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", inputs={"a": 3},
#               output_names=["out"])
#     wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
#     wfa.add(na)
#
#     wf = Workflow(name="wf14", workingdir="test_wf14_{}".format(plugin),
#             wf_output_names=[("wfa", "NA_out", "wfa_out")])
#     wf.add(wfa)
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"NA.a": 3}, 5)]
#     for i, res in enumerate(expected):
#         assert wf.result["wfa_out"][i][0] == res[0]
#         assert wf.result["wfa_out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_14a(plugin, change_dir):
#     """workflow with a workflow as a node (no splitter, using connect_wf_input in wfa)"""
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na",
#               output_names=["out"])
#     wfa = Workflow(name="wfa", workingdir="test_wfa", inputs={"a": 3},
#             wf_output_names=[("NA", "out", "NA_out")])
#     wfa.add(na)
#     wfa.connect_wf_input("a", "NA", "a")
#
#     wf = Workflow(name="wf14a", workingdir="test_wf14a_{}".format(plugin),
#             wf_output_names=[("wfa", "NA_out", "wfa_out")])
#     wf.add(wfa)
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"NA.a": 3}, 5)]
#     for i, res in enumerate(expected):
#         assert wf.result["wfa_out"][i][0] == res[0]
#         assert wf.result["wfa_out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_14b(plugin, change_dir):
#     """workflow with a workflow as a node (no splitter, using connect_wf_input in wfa and wf)"""
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
#     wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
#     wfa.add(na)
#     wfa.connect_wf_input("a", "NA", "a")
#
#     wf = Workflow(name="wf14b", workingdir="test_wf14b_{}".format(plugin),
#             wf_output_names=[("wfa", "NA_out", "wfa_out")], inputs={"a": 3})
#     wf.add(wfa)
#     wf.connect_wf_input("a", "wfa", "a")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"NA.a": 3}, 5)]
#     for i, res in enumerate(expected):
#         assert wf.result["wfa_out"][i][0] == res[0]
#         assert wf.result["wfa_out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_15(plugin, change_dir):
#     """workflow with a workflow as a node with splitter (like 14 but with a splitter)"""
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na",
#               inputs={"a": [3, 5]}, splitter="a", output_names=["out"])
#     wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
#     wfa.add(na)
#
#     wf = Workflow(name="wf15", workingdir="test_wf15_{}".format(plugin),
#             wf_output_names=[("wfa", "NA_out", "wfa_out")])
#     wf.add(wfa)
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
#     for i, res in enumerate(expected):
#         assert wf.result["wfa_out"][i][0] == res[0]
#         assert wf.result["wfa_out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_16(plugin, change_dir):
#     """workflow with two nodes, and one is a workflow (no splitter)"""
#     wf = Workflow(name="wf16", workingdir="test_wf16_{}".format(plugin),
#             wf_output_names=[("wfb", "NB_out"), ("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na", inputs={"a": 3},
#               output_names=["out"])
#     wf.add(na)
#
#     # the second node does not have explicit splitter (but keeps the splitter from the NA node)
#     nb = Node(name="NB", interface=interf_addvar, workingdir="nb",
#               output_names=["out"])
#     wfb = Workflow(name="wfb", workingdir="test_wfb", inputs={"c": 10},
#             wf_output_names=[("NB", "out", "NB_out")])
#     wfb.add(nb)
#     wfb.connect_wf_input("b", "NB", "b")
#     wfb.connect_wf_input("c", "NB", "c")
#
#     wf.add(wfb)
#     wf.connect("NA", "out", "wfb", "b")
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#     expected_A = [({"NA.a": 3}, 5)]
#     for i, res in enumerate(expected_A):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1] == res[1]
#
#     # TODO (res): the naming rememebrs only the node, doesnt remember that a came from NA...
#     # the naming should have names with workflows??
#     expected_B = [({"NB.b": 5, "NB.c": 10}, 15)]
#     for i, res in enumerate(expected_B):
#         assert wf.result["NB_out"][i][0] == res[0]
#         assert wf.result["NB_out"][i][1] == res[1]
#
#
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_workflow_16a(plugin, change_dir):
#     """workflow with two nodes, and one is a workflow (with splitter)"""
#     wf = Workflow(name="wf16a", workingdir="test_wf16a_{}".format(plugin),
#         wf_output_names=[("wfb", "NB_out"), ("NA", "out", "NA_out")])
#     na = Node(name="NA", interface=interf_addtwo, workingdir="na",
#               output_names=["out"])
#     na.split(splitter="a", inputs={"a": [3, 5]})
#     wf.add(na)
#     # the second node does not have explicit splitter (but keeps the splitter from the NA node)
#     nb = Node(name="NB", interface=interf_addvar, workingdir="nb",
#               output_names=["out"])
#     wfb = Workflow(name="wfb", workingdir="test_wfb", inputs={"c": 10},
#         wf_output_names=[("NB", "out", "NB_out")])
#     wfb.add(nb)
#     wfb.connect_wf_input("b", "NB", "b")
#     wfb.connect_wf_input("c", "NB", "c")
#
#     # adding 2 nodes and create a connection (as it is now)
#     wf.add(wfb)
#     wf.connect("NA", "out", "wfb", "b")
#     assert wf.nodes[0].splitter == "NA.a"
#
#     sub = Submitter(runnable=wf, plugin=plugin)
#     sub.run()
#     sub.close()
#
#     assert wf.is_complete
#
#     expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
#     for i, res in enumerate(expected_A):
#         assert wf.result["NA_out"][i][0] == res[0]
#         assert wf.result["NA_out"][i][1] == res[1]
#
#     # TODO (res): the naming rememebrs only the node, doesnt remember that came from NA...
#     # the naming should have names with workflows??
#     expected_B = [({"NB.b": 5, "NB.c": 10}, 15), ({"NB.b": 7, "NB.c": 10}, 17)]
#     key_sort = list(expected_B[0][0].keys())
#     expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
#     wf.result["NB_out"].sort(key=lambda t: [t[0][key] for key in key_sort])
#     for i, res in enumerate(expected_B):
#         assert wf.result["NB_out"][i][0] == res[0]
#         assert wf.result["NB_out"][i][1] == res[1]
#
#
# # testing CurrentInterface that is a temporary wrapper for current interfaces
# T1_file = "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
# T1_file_list = [
#     "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
#     "/Users/dorota/nipype_workshop/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz"
#     ]
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_node_1(change_dir, plugin):
#     """Node with a current interface and inputs, no splitter, running interface"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#     nn = Node(
#         name="NA",
#         inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
#         interface=interf_bet,
#         workingdir="test_cnd1_{}".format(plugin),
#         output_names=["out_file"])
#
#     sub = Submitter(plugin=plugin, runnable=nn)
#     sub.run()
#     sub.close()
#     # TODO (res): nodes only returns relative path
#     assert "out_file" in nn.output.keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_node_2(change_dir, plugin):
#     """Node with a current interface and splitter"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     in_file_l = [
#         str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
#         str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
#     ]
#     nn = Node(
#         name="NA",
#         inputs={"in_file": in_file_l},
#         splitter="in_file",
#         interface=interf_bet,
#         write_state=False,
#         workingdir="test_cnd2_{}".format(plugin),
#         output_names=["out_file"])
#
#     sub = Submitter(plugin=plugin, runnable=nn)
#     sub.run()
#     sub.close()
#
#     assert "out_file" in nn.output.keys()
#     assert "NA.in_file:0" in nn.output["out_file"].keys()
#     assert "NA.in_file:1" in nn.output["out_file"].keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_1(change_dir, plugin):
#     """Wf with a current interface, no splitter"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     nn = Node(
#         name="fsl",
#         inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
#         interface=interf_bet,
#         workingdir="nn",
#         output_names=["out_file"],
#         write_state=False)
#
#     wf = Workflow(
#         workingdir="test_cwf_1_{}".format(plugin),
#         name="cw1",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         write_state=False)
#     wf.add_nodes([nn])
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_1a(change_dir, plugin):
#     """Wf with a current interface, no splitter"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     nn = Node(
#         name="fsl",
#         inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
#         interface=interf_bet,
#         workingdir="nn",
#         output_names=["out_file"],
#         write_state=False)
#
#     wf = Workflow(
#         workingdir="test_cwf_1a_{}".format(plugin),
#         name="cw1",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         write_state=False)
#     wf.add(runnable=nn)
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_1b(change_dir, plugin):
#     """Wf with a current interface, no splitter; using wf.add(nipype CurrentInterface)"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     wf = Workflow(
#         workingdir="test_cwf_1b_{}".format(plugin),
#         name="cw1",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         write_state=False)
#     wf.add(
#         runnable=interf_bet,
#         name="fsl",
#         workingdir="nn",
#         output_names=["out_file"],
#         write_state=False,
#         inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")})
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_1c(change_dir, plugin):
#     """Wf with a current interface, no splitter; using wf.add(nipype interface) """
#
#     wf = Workflow(
#         workingdir="test_cwf_1c_{}".format(plugin),
#         name="cw1",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         write_state=False)
#     wf.add(
#         runnable=fsl.BET(),
#         name="fsl",
#         workingdir="nn",
#         output_names=["out_file"],
#         write_state=False,
#         inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")})
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_2(change_dir, plugin):
#     """Wf with a current interface and splitter"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     in_file_l = [
#         str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
#         str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
#     ]
#
#     nn = Node(
#         name="fsl",
#         interface=interf_bet,
#         write_state=False,
#         workingdir="nn",
#         output_names=["out_file"])
#
#     wf = Workflow(
#         workingdir="test_cwf_2_{}".format(plugin),
#         name="cw2",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         inputs={"in_file": in_file_l},
#         splitter="in_file",
#         write_state=False)
#     wf.add_nodes([nn])
#     wf.connect_wf_input("in_file", "fsl", "in_file")
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#     assert 'cw2.in_file:0' in wf.output["fsl_out"].keys()
#     assert 'cw2.in_file:1' in wf.output["fsl_out"].keys()
#
#
# @pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
# @pytest.mark.parametrize("plugin", Plugins)
# @python35_only
# def test_current_wf_2a(change_dir, plugin):
#     """Wf with a current interface and splitter"""
#     interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
#
#     in_file_l = [
#         str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
#         str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
#     ]
#
#     nn = Node(
#         name="fsl",
#         interface=interf_bet,
#         write_state=False,
#         workingdir="nn",
#         output_names=["out_file"],
#         inputs={"in_file": in_file_l},
#         splitter="in_file")
#
#     wf = Workflow(
#         workingdir="test_cwf_2a_{}".format(plugin),
#         name="cw2a",
#         wf_output_names=[("fsl", "out_file", "fsl_out")],
#         write_state=False)
#     wf.add_nodes([nn])
#     # wf.connect_wf_input("in_file", "fsl", "in_file")
#
#     sub = Submitter(plugin=plugin, runnable=wf)
#     sub.run()
#     sub.close()
#
#     assert "fsl_out" in wf.output.keys()
#     assert 'fsl.in_file:0' in wf.output["fsl_out"].keys()
#     assert 'fsl.in_file:1' in wf.output["fsl_out"].keys()
