import sys
import os
import time
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
    test_dir = os.path.join(orig_dir, "test_outputs")
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


def fun_addvar4(a, b, c, d):
    return a + b + c + d

_interf_addvar4 = Function(function=fun_addvar4, input_names=["a", "b", "c", "d"], output_names=["out"])
interf_addvar4 = CurrentInterface(interface=_interf_addvar4, name="addvar4")


def test_node_1():
    """Node with mandatory arguments only"""
    nn = Node(name="NA", interface=interf_addtwo)
    assert nn.splitter is None
    assert nn.inputs == {}
    assert nn.state._splitter is None


def test_node_2():
    """Node with interface and inputs"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": 3})
    assert nn.splitter is None
    # adding NA to the name of the variable
    assert nn.inputs == {"NA.a": 3}
    assert nn.state._splitter is None


def test_node_3():
    """Node with interface, inputs and splitter"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]}, splitter="a")
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()

    assert nn.state._splitter == "NA.a"

    nn.prepare_state_input()
    assert nn.state.state_values([0]) == {"NA.a": 3}
    assert nn.state.state_values([1]) == {"NA.a": 5}


def test_node_4():
    """Node with interface and inputs. splitter set using split method"""
    nn = Node(name="NA", interface=interf_addtwo, inputs={"a": [3, 5]})
    nn.split(splitter="a")
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()

    nn.prepare_state_input()
    assert nn.state._splitter == "NA.a"
    assert nn.state.state_values([0]) == {"NA.a": 3}
    assert nn.state.state_values([1]) == {"NA.a": 5}


def test_node_4a():
    """Node with interface, splitter and inputs set with the split method"""
    nn = Node(name="NA", interface=interf_addtwo)
    nn.split(splitter="a", inputs={"a": [3, 5]})
    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()

    assert nn.state._splitter == "NA.a"
    nn.prepare_state_input()
    assert nn.state.state_values([0]) == {"NA.a": 3}
    assert nn.state.state_values([1]) == {"NA.a": 5}


def test_node_4b():
    """Node with interface and inputs. trying to set splitter twice"""
    nn = Node(name="NA", splitter="a", interface=interf_addtwo, inputs={"a": [3, 5]})
    with pytest.raises(Exception) as excinfo:
        nn.split(splitter="a")
    assert str(excinfo.value) == "splitter is already set"


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_5(plugin, change_dir):
    """Node with interface and inputs, no splitter, running interface"""
    nn = Node(name="NA", inputs={"a": 3}, interface=interf_addtwo,
        workingdir="test_nd5_{}".format(plugin), output_names=["out"])

    assert (nn.inputs["NA.a"] == np.array([3])).all()

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [({}, 5)]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    nn.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])

    for i, res in enumerate(expected):
        assert nn.result["out"][i][0] == res[0]
        assert nn.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_6(plugin, change_dir):
    """Node with interface, inputs and the simplest splitter, running interface"""
    nn = Node(name="NA", interface=interf_addtwo, workingdir="test_nd6_{}".format(plugin),
              output_names=["out"])
    nn.split(splitter="a", inputs={"a": [3, 5]})

    assert nn.splitter == "NA.a"
    assert (nn.inputs["NA.a"] == np.array([3, 5])).all()

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    nn.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])

    for i, res in enumerate(expected):
        assert nn.result["out"][i][0] == res[0]
        assert nn.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_7(plugin, change_dir):
    """Node with interface, inputs and scalar splitter, running interface"""
    nn = Node(name="NA", interface=interf_addvar, workingdir="test_nd7_{}".format(plugin),
              output_names=["out"])
    # scalar splitter
    nn.split(splitter=("b", "c"), inputs={"b": [3, 5], "c": [2, 1]})

    assert nn.splitter == ("NA.b", "NA.c")
    assert (nn.inputs["NA.b"] == np.array([3, 5])).all()
    assert (nn.inputs["NA.c"] == np.array([2, 1])).all()

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking the results
    expected = [({"NA.b": 3, "NA.c": 2}, 5), ({"NA.b": 5, "NA.c": 1}, 6)]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    nn.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])

    for i, res in enumerate(expected):
        assert nn.result["out"][i][0] == res[0]
        assert nn.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_8(plugin, change_dir):
    """Node with interface, inputs and vector splitter, running interface"""
    nn = Node(name="NA", interface=interf_addvar, workingdir="test_nd8_{}".format(plugin),
              output_names=["out"])
    # [] for outer product
    nn.split(splitter=["b", "c"], inputs={"b": [3, 5], "c": [2, 1]})

    assert nn.splitter == ["NA.b", "NA.c"]
    assert (nn.inputs["NA.b"] == np.array([3, 5])).all()
    assert (nn.inputs["NA.c"] == np.array([2, 1])).all()

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    # checking teh results
    expected = [({"NA.b": 3, "NA.c": 1}, 4), ({"NA.b": 3, "NA.c": 2}, 5),
                ({"NA.b": 5, "NA.c": 1}, 6), ({"NA.b": 5, "NA.c": 2}, 7)]
    # to be sure that there is the same order (not sure if node itself should keep the order)
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    nn.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert nn.result["out"][i][0] == res[0]
        assert nn.result["out"][i][1] == res[1]

#
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_node_9(plugin, change_dir):
    """scalar and outer splitter, one combiner (from scalar part)"""
    nn = Node(name="NA", interface=interf_addvar4, workingdir="test_nd16_{}".format(plugin),
              output_names=["out"], splitter=(["a", "b"], ["c", "d"]),
              inputs={"a": [1, 2], "b": [1, 2], "c": [1, 2], "d": [1,2]})

    assert nn.splitter == (["NA.a", "NA.b"], ["NA.c", "NA.d"])

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()


# tests for workflows


@python35_only
def test_workflow_0(plugin="serial"):
    """workflow (without run) with one node with a splitter"""
    wf = Workflow(name="wf0", workingdir="test_wf0_{}".format(plugin))
    # defining a node with splitter and inputs first
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["a"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    # one of the way of adding nodes to the workflow
    wf.add_nodes([na])
    assert wf.nodes[0].splitter == "NA.a"
    assert (wf.nodes[0].inputs['NA.a'] == np.array([3, 5])).all()
    assert len(wf.graph.nodes) == 1


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_1(plugin, change_dir):
    """workflow with one node with a splitter"""
    wf = Workflow(name="wf1", workingdir="test_wf1_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    wf.add_nodes([na])

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_2(plugin, change_dir):
    """workflow with two nodes, second node without splitter"""
    wf = Workflow(name="wf2", workingdir="test_wf2_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_addvar, inputs={"c": 10}, workingdir="nb",
              output_names=["out"])

    # adding 2 nodes and create a connection (as it is now)
    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "b")
    assert wf.nodes[0].splitter == "NA.a"

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected_A[0][0].keys())
    expected_A.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_A):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    # results from NB keeps the "state input" from the first node
    # two elements as in NA
    expected_B = [({"NA.a": 3}, 15), ({"NA.a": 5}, 17)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    #output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]



@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_2a(plugin, change_dir):
    """workflow with two nodes, second node with a scalar splitter"""
    wf = Workflow(name="wf2", workingdir="test_wf2a_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # explicit scalar splitter between "a" from NA and b
    nb.split(splitter=("NA.a", "c"), inputs={"c": [2, 1]})

    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "b")

    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[1].splitter == ("NA.a", "NB.c")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected_A[0][0].keys())
    expected_A.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_A):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    # two elements (scalar splitter)
    expected_B = [({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    # output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_2b(plugin):
    """workflow with two nodes, second node with a vector splitter"""
    wf = Workflow(name="wf2", workingdir="test_wf2b_{}".format(plugin),
                  wf_output_names=[("NB", "out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # outer splitter
    nb.split(splitter=["NA.a", "c"], inputs={"c": [2, 1]})

    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "b")

    assert wf.nodes[0].splitter == "NA.a"
    assert wf.nodes[1].splitter == ["NA.a", "NB.c"]

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected_A[0][0].keys())
    expected_A.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_A):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    # four elements (outer product)
    expected_B = [({"NA.a": 3, "NB.c": 1}, 6), ({"NA.a": 3, "NB.c": 2}, 7),
                  ({"NA.a": 5, "NB.c": 1}, 8), ({"NA.a": 5, "NB.c": 2}, 9)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]

    # output of the wf
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


# using add method to add nodes

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_3(plugin, change_dir):
    """using add(node) method"""
    wf = Workflow(name="wf3", workingdir="test_wf3_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    # using add method (as in the Satra's example) with a node
    wf.add(na)

    assert wf.nodes[0].splitter == "NA.a"

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_3a(plugin, change_dir):
    """using add(interface) method"""
    wf = Workflow(name="wf3a", workingdir="test_wf3a_{}".format(plugin))
    # using the add method with an interface
    wf.add(interf_addtwo, workingdir="na", splitter="a", inputs={"a": [3, 5]}, name="NA",
           output_names=["out"])

    assert wf.nodes[0].splitter == "NA.a"

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_3b(plugin, change_dir):
    """using add (function) method"""
    wf = Workflow(name="wf3b", workingdir="test_wf3b_{}".format(plugin))
    # using the add method with a function
    wf.add(fun_addtwo, input_names=["a"], workingdir="na", splitter="a",
           inputs={"a": [3, 5]}, name="NA", output_names=["out"])

    assert wf.nodes[0].splitter == "NA.a"

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_4(plugin, change_dir):
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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_4a(plugin, change_dir):
    """ using add(node) method with kwarg arg to connect nodes (instead of wf.connect) """
    wf = Workflow(name="wf4a", workingdir="test_wf4a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    wf.add(na)


    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # explicit splitter with a variable from the previous node
    nb.split(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
    # instead of "connect", using kwrg argument in the add method as in the example
    wf.add(nb, b="NA.out")

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


#
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_4b(plugin, change_dir):
    """ using add(interf) method
        using wf.connect to connect two nodes
    """
    wf = Workflow(name="wf4b", workingdir="test_wf4b_{}".format(plugin))
    wf.add(runnable=interf_addtwo, name="NA", workingdir="na", output_names=["out"],
           splitter="a", inputs={"a": [3, 5]})
    wf.add(runnable=interf_addvar, name="NB", workingdir="nb", output_names=["out"],
           splitter=("NA.a", "c"), inputs={"c": [2, 1]})
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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_4c(plugin, change_dir):
    """ using add(interf) method (split after add)
        using wf.connect to connect two nodes
    """
    wf = Workflow(name="wf4c", workingdir="test_wf4c_{}".format(plugin))
    wf.add(runnable=interf_addtwo, name="NA", workingdir="na", output_names=["out"],
           inputs={"a": [3, 5]})
    wf.add(runnable=interf_addvar, name="NB", workingdir="nb", output_names=["out"],
           inputs={"c": [2, 1]})
    wf.split_node(splitter="a", node="NA")
    wf.split_node(splitter=("NA.a", "c"), node="NB")

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_4d(plugin, change_dir):
    """ using add(interf) method (split after add)
        using wf.connect to connect two nodes
    """
    wf = Workflow(name="wf4d", workingdir="test_wf4d_{}".format(plugin))
    wf.add(runnable=interf_addtwo, name="NA", workingdir="na", output_names=["out"],
           inputs={"a": [3, 5]})
    wf.add(runnable=interf_addvar, name="NB", workingdir="nb", output_names=["out"],
           inputs={"c": [2, 1]})

    wf.connect("NA", "out", "NB", "b")

    wf.split_node(splitter="a", node="NA")
    wf.split_node(splitter=("NA.a", "c"), node="NB")

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


# using split after add method


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_5(plugin, change_dir):
    """using a split method for one node"""
    wf = Workflow(name="wf5", workingdir="test_wf5_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na)
    # using the split method after add (using splitter for the last added node as default)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_5a(plugin, change_dir):
    """using a split method for one node (using add and split in one chain)"""
    wf = Workflow(name="wf5a", workingdir="test_wf5a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na).split_node(splitter="a", inputs={"a": [3, 5]})

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_6(plugin, change_dir):
    """using a split method for two nodes (using last added node as default)"""
    wf = Workflow(name="wf6", workingdir="test_wf6_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_6a(plugin, change_dir):
    """using a split method for two nodes (specifying the node)"""
    wf = Workflow(name="wf6a", workingdir="test_wf6a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split method after add (specifying the node)
    wf.add(na)
    wf.add(nb)
    wf.split_node(splitter="a", inputs={"a": [3, 5]}, node=na)
    # TODO: should we se ("b", "c") instead?? shold I forget "NA.a" value?
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}, node=nb)
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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_6b(plugin, change_dir):
    """using a split method for two nodes (specifying the node), using kwarg arg instead of connect"""
    wf = Workflow(name="wf6b", workingdir="test_wf6b_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])

    wf.add(na)
    wf.add(nb, b="NA.out")
    wf.split_node(splitter="a", inputs={"a": [3, 5]}, node=na)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]}, node=nb)

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


# tests for a workflow that have its own input


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_7(plugin, change_dir):
    """using inputs for workflow and connect_workflow"""
    # adding inputs to the workflow directly
    wf = Workflow(name="wf7", inputs={"wfa": [3, 5]}, workingdir="test_wf7_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])

    wf.add(na)
    # connecting the node with inputs from the workflow
    wf.connect_wf_input("wfa", "NA", "a")
    wf.split_node(splitter="a")

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_7a(plugin, change_dir):
    """using inputs for workflow and kwarg arg in add (instead of connect)"""
    wf = Workflow(name="wf7a", inputs={"wfa": [3, 5]}, workingdir="test_wf7a_{}".format(plugin))
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    # using kwrg argument in the add method (instead of connect or connect_wf_input
    wf.add(na, a="wfa")
    wf.split_node(splitter="a")

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_8(plugin, change_dir):
    """using inputs for workflow and connect_wf_input for the second node"""
    wf = Workflow(name="wf8", workingdir="test_wf8_{}".format(plugin), inputs={"c": 10})
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])

    wf.add_nodes([na, nb])
    wf.connect("NA", "out", "NB", "b")
    wf.connect_wf_input("c", "NB", "c")
    assert wf.nodes[0].splitter == "NA.a"

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected_A[0][0].keys())
    expected_A.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_A):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expected_B = [({"NA.a": 3}, 15), ({"NA.a": 5}, 17)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]


# testing if _NA in splitter works, using interfaces in add


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_9(plugin, change_dir):
    """using add(interface) method and splitter from previous nodes"""
    wf = Workflow(name="wf9", workingdir="test_wf9_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addtwo, workingdir="na",
           output_names=["out"]).split_node(splitter="a", inputs={"a": [3, 5]})
    # _NA means that I'm using splitter from the NA node, it's the same as ("NA.a", "b")
    wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
           output_names=["out"]).split_node(splitter=("_NA", "c"), inputs={"c": [2, 1]})

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


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_10(plugin, change_dir):
    """using add(interface) method and scalar splitter from previous nodes"""
    wf = Workflow(name="wf10", workingdir="test_wf10_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addvar, workingdir="na",
           output_names=["out"]).split_node(
        splitter=("b", "c"), inputs={"b": [3, 5], "c": [0, 10]})
    # _NA means that I'm using splitter from the NA node, it's the same as (("NA.a", NA.b), "b")
    wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
           output_names=["out"]).split_node(splitter=("_NA", "c"), inputs={"c": [2, 1]})

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 5, "NA.c": 10}, 15)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expected_B = [({"NA.b": 3, "NA.c": 0, "NB.c": 2}, 5), ({"NA.b": 5, "NA.c": 10, "NB.c": 1}, 16)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_10a(plugin, change_dir):
    """using add(interface) method and vector splitter from previous nodes"""
    wf = Workflow(name="wf10a", workingdir="test_wf10a_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addvar, workingdir="na",
           output_names=["out"]).split_node(
            splitter=["b", "c"], inputs={"b": [3, 5], "c": [0, 10]})
    # _NA means that I'm using splitter from the NA node, it's the same as (["NA.a", NA.b], "b")
    wf.add(name="NB", runnable=interf_addvar, workingdir="nb", b="NA.out",
           output_names=["out"]).split_node(
            splitter=("_NA", "c"), inputs={"c": [[2, 1], [0, 0]]})

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 3, "NA.c": 10}, 13),
                ({"NA.b": 5, "NA.c": 0}, 5), ({"NA.b": 5, "NA.c": 10}, 15)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expected_B = [({"NA.b": 3, "NA.c": 0, "NB.c": 2}, 5),
                  ({"NA.b": 3, "NA.c": 10, "NB.c": 1}, 14),
                  ({"NA.b": 5, "NA.c": 0, "NB.c": 0}, 5),
                  ({"NA.b": 5, "NA.c": 10, "NB.c": 0}, 15)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[1].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.nodes[1].result["out"][i][0] == res[0]
        assert wf.nodes[1].result["out"][i][1] == res[1]


# TODO: this test started sometimes failing for mp and cf
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_11(plugin, change_dir):
    """using add(interface) method and vector splitter from previous two nodes"""
    wf = Workflow(name="wf11", workingdir="test_wf11_{}".format(plugin))
    wf.add(name="NA", runnable=interf_addvar, workingdir="na",
           output_names=["out"]).split_node(
            splitter=("b", "c"), inputs={"b": [3, 5],"c": [0, 10]})
    wf.add(name="NB", runnable=interf_addtwo, workingdir="nb",
           output_names=["out"]).split_node(splitter="a", inputs={"a": [2, 1]})
    # _NA, _NB means that I'm using splitters from the NA/NB nodes, it's the same as [("NA.a", NA.b), "NB.a"]
    wf.add(name="NC", runnable=interf_addvar, workingdir="nc", b="NA.out", c="NB.out",
           output_names=["out"]).split_node(splitter=["_NA", "_NB"])  # TODO: this should eb default?

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    expected = [({"NA.b": 3, "NA.c": 0}, 3), ({"NA.b": 5, "NA.c": 10}, 15)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[0].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected):
        assert wf.nodes[0].result["out"][i][0] == res[0]
        assert wf.nodes[0].result["out"][i][1] == res[1]

    expected_C = [({"NA.b": 3,"NA.c": 0,"NB.a": 1}, 6),
                  ({"NA.b": 3,"NA.c": 0,"NB.a": 2}, 7),
                  ({"NA.b": 5,"NA.c": 10,"NB.a": 1}, 18),
                  ({"NA.b": 5,"NA.c": 10,"NB.a": 2}, 19)]
    key_sort = list(expected_C[0][0].keys())
    expected_C.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.nodes[2].result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_C):
        assert wf.nodes[2].result["out"][i][0] == res[0]
        assert wf.nodes[2].result["out"][i][1] == res[1]


# checking workflow.result


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_12(plugin, change_dir):
    """testing if wf.result works (the same workflow as in test_workflow_6)"""
    wf = Workflow(name="wf12", workingdir="test_wf12_{}".format(plugin),
        wf_output_names=[("NA", "out", "NA_out"), ("NB", "out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    # checking if workflow.results is the same as results of nodes
    assert wf.result["NA_out"] == wf.nodes[0].result["out"]
    assert wf.result["out"] == wf.nodes[1].result["out"]

    # checking values of workflow.result
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    key_sort = list(expected[0][0].keys())
    expected.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.result["NA_out"].sort(key=lambda t: [t[0][key] for key in key_sort])

    assert wf.is_complete
    for i, res in enumerate(expected):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1] == res[1]

    expected_B = [({"NA.a": 3, "NB.c": 2}, 7), ({"NA.a": 5, "NB.c": 1}, 8)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    wf.result["out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    for i, res in enumerate(expected_B):
        assert wf.result["out"][i][0] == res[0]
        assert wf.result["out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_12a(plugin, change_dir):
    """testing if wf.result raises exceptione (the same workflow as in test_workflow_6)"""
    wf = Workflow(name="wf12a",workingdir="test_wf12a_{}".format(plugin),
        wf_output_names=[("NA", "out", "wf_out"), ("NB", "out", "wf_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb", output_names=["out"])
    # using the split methods after add (using splitter for the last added nodes as default)
    wf.add(na)
    wf.split_node(splitter="a", inputs={"a": [3, 5]})
    wf.add(nb)
    wf.split_node(splitter=("NA.a", "c"), inputs={"c": [2, 1]})
    wf.connect("NA", "out", "NB", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    # wf_out can't be used twice
    with pytest.raises(Exception) as exinfo:
        sub.run()
    assert str(exinfo.value) == "the key wf_out is already used in workflow.result"


# tests for a workflow that have its own input and splitter

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_13(plugin, change_dir):
    """using inputs for workflow and connect_wf_input"""
    wf = Workflow(name="wf13", inputs={"wfa": [3, 5]}, splitter="wfa",
        workingdir="test_wf13_{}".format(plugin),
        wf_output_names=[("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    wf.add(na)
    wf.connect_wf_input("wfa", "NA", "a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({"wf13.wfa": 3}, [({}, 5)]), ({'wf13.wfa': 5}, [({}, 7)])]
    for i, res in enumerate(expected):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
        assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_13a(plugin, change_dir):
    """using inputs for workflow and connect_wf_input (the node has 2 inputs)"""
    wf = Workflow(name="wf13a", inputs={"wfa": [3, 5]}, splitter="wfa",
        workingdir="test_wf13a_{}".format(plugin),
        wf_output_names=[("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addvar, workingdir="na", splitter="c",
              inputs={"c": [10, 20]}, output_names=["out"])
    wf.add(na)
    wf.connect_wf_input("wfa", "NA", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({"wf13a.wfa": 3}, [({"NA.c": 10}, 13),
                                    ({"NA.c": 20}, 23)]),
                ({'wf13a.wfa': 5}, [({"NA.c": 10}, 15),
                                    ({"NA.c": 20}, 25)])]
    for i, res in enumerate(expected):
        assert wf.result["NA_out"][i][0] == res[0]
        for j in range(len(res[1])):
            assert wf.result["NA_out"][i][1][j][0] == res[1][j][0]
            assert wf.result["NA_out"][i][1][j][1] == res[1][j][1]

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_13b(plugin, change_dir):
    """using inputs for workflow and connect_wf_input, using wf.split(splitter)"""
    wf = Workflow(name="wf13b", inputs={"wfa": [3, 5]},
        workingdir="test_wf13b_{}".format(plugin),
        wf_output_names=[("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              output_names=["out"])
    wf.add(na).split(splitter="wfa")
    wf.connect_wf_input("wfa", "NA", "a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({"wf13b.wfa": 3}, [({}, 5)]),
                ({'wf13b.wfa': 5}, [({}, 7)])]
    for i, res in enumerate(expected):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
        assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]




@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_13c(plugin, change_dir):
    """using inputs for workflow and connect_wf_input, using wf.split(splitter, inputs)"""
    wf = Workflow(name="wf13c", workingdir="test_wf13c_{}".format(plugin),
        wf_output_names=[("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              output_names=["out"])
    wf.add(na).split(splitter="wfa", inputs={"wfa": [3, 5]})
    wf.connect_wf_input("wfa", "NA", "a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({"wf13c.wfa": 3}, [({}, 5)]), ({'wf13c.wfa': 5}, [({}, 7)])]
    for i, res in enumerate(expected):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1][0][0] == res[1][0][0]
        assert wf.result["NA_out"][i][1][0][1] == res[1][0][1]


# workflow as a node

@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_14(plugin, change_dir):
    """workflow with a workflow as a node (no splitter)"""
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", inputs={"a": 3},
              output_names=["out"])
    wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
    wfa.add(na)

    wf = Workflow(name="wf14", workingdir="test_wf14_{}".format(plugin),
            wf_output_names=[("wfa", "NA_out", "wfa_out")])
    wf.add(wfa)

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({}, 5)]
    for i, res in enumerate(expected):
        assert wf.result["wfa_out"][i][0] == res[0]
        assert wf.result["wfa_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_14a(plugin, change_dir):
    """workflow with a workflow as a node (no splitter, using connect_wf_input in wfa)"""
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              output_names=["out"])
    wfa = Workflow(name="wfa", workingdir="test_wfa", inputs={"a": 3},
            wf_output_names=[("NA", "out", "NA_out")])
    wfa.add(na)
    wfa.connect_wf_input("a", "NA", "a")

    wf = Workflow(name="wf14a", workingdir="test_wf14a_{}".format(plugin),
            wf_output_names=[("wfa", "NA_out", "wfa_out")])
    wf.add(wfa)

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({}, 5)]
    for i, res in enumerate(expected):
        assert wf.result["wfa_out"][i][0] == res[0]
        assert wf.result["wfa_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_14b(plugin, change_dir):
    """workflow with a workflow as a node (no splitter, using connect_wf_input in wfa and wf)"""
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", output_names=["out"])
    wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
    wfa.add(na)
    wfa.connect_wf_input("a", "NA", "a")

    wf = Workflow(name="wf14b", workingdir="test_wf14b_{}".format(plugin),
            wf_output_names=[("wfa", "NA_out", "wfa_out")], inputs={"a": 3})
    wf.add(wfa)
    wf.connect_wf_input("a", "wfa", "a")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({}, 5)]
    for i, res in enumerate(expected):
        assert wf.result["wfa_out"][i][0] == res[0]
        assert wf.result["wfa_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_15(plugin, change_dir):
    """workflow with a workflow as a node with splitter (like 14 but with a splitter)"""
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              inputs={"a": [3, 5]}, splitter="a", output_names=["out"])
    wfa = Workflow(name="wfa", workingdir="test_wfa", wf_output_names=[("NA", "out", "NA_out")])
    wfa.add(na)

    wf = Workflow(name="wf15", workingdir="test_wf15_{}".format(plugin),
            wf_output_names=[("wfa", "NA_out", "wfa_out")])
    wf.add(wfa)

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected):
        assert wf.result["wfa_out"][i][0] == res[0]
        assert wf.result["wfa_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_16(plugin, change_dir):
    """workflow with two nodes, and one is a workflow (no splitter)"""
    wf = Workflow(name="wf16", workingdir="test_wf16_{}".format(plugin),
            wf_output_names=[("wfb", "NB_out"), ("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na", inputs={"a": 3},
              output_names=["out"])
    wf.add(na)

    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb",
              output_names=["out"])
    wfb = Workflow(name="wfb", workingdir="test_wfb", inputs={"c": 10},
            wf_output_names=[("NB", "out", "NB_out")])
    wfb.add(nb)
    wfb.connect_wf_input("b", "NB", "b")
    wfb.connect_wf_input("c", "NB", "c")

    wf.add(wfb)
    wf.connect("NA", "out", "wfb", "b")

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    assert wf.is_complete
    expected_A = [({}, 5)]
    for i, res in enumerate(expected_A):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1] == res[1]

    # TODO (res): the naming rememebrs only the node, doesnt remember that a came from NA...
    # the naming should have names with workflows??
    expected_B = [({}, 15)]
    for i, res in enumerate(expected_B):
        assert wf.result["NB_out"][i][0] == res[0]
        assert wf.result["NB_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_16a(plugin, change_dir):
    """workflow with two nodes, and one is a workflow (with splitter)"""
    wf = Workflow(name="wf16a", workingdir="test_wf16a_{}".format(plugin),
        wf_output_names=[("wfb", "NB_out"), ("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              output_names=["out"])
    na.split(splitter="a", inputs={"a": [3, 5]})
    wf.add(na)
    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb",
              output_names=["out"])
    wfb = Workflow(name="wfb", workingdir="test_wfb", inputs={"c": 10},
        wf_output_names=[("NB", "out", "NB_out")])
    wfb.add(nb)
    wfb.connect_wf_input("b", "NB", "b")
    wfb.connect_wf_input("c", "NB", "c")

    # adding 2 nodes and create a connection (as it is now)
    wf.add(wfb)
    wf.connect("NA", "out", "wfb", "b")
    assert wf.nodes[0].splitter == "NA.a"

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()
    # wf.result for NB_out doeant have state in results
    assert wf.is_complete

    expected_A = [({"NA.a": 3}, 5), ({"NA.a": 5}, 7)]
    for i, res in enumerate(expected_A):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1] == res[1]

    # TODO (res): the naming remembers only the node, doesnt remember that came from NA...
    # TODO (res): because nb doesnt have splitter, but only wfb, and wf.result reads from node results,
    # TODO ...  : we dont have any state values here. probably should change it that wf can see wfb.b values
    #TODO (res): compare wf.result and wfb.rsult (wfb has to many var in state_Dict)
    # the naming should have names with workflows??
    expected_B = [({}, 15), ({}, 17)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    if wf.result["NB_out"][0][0]: #has some state valuse
        wf.result["NB_out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    else: # dictionary empty so sorting regarding the values
        wf.result["NB_out"].sort()
    for i, res in enumerate(expected_B):
        assert wf.result["NB_out"][i][0] == res[0]
        assert wf.result["NB_out"][i][1] == res[1]


@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_workflow_16b(plugin, change_dir):
    """workflow with two nodes, and one is a workflow (with splitter)
        the sae as 16b, but using indices instead of values (write_state=False)
    """
    wf = Workflow(name="wf16b", workingdir="test_wf16b_{}".format(plugin),
        wf_output_names=[("wfb", "NB_out"), ("NA", "out", "NA_out")])
    na = Node(name="NA", interface=interf_addtwo, workingdir="na",
              output_names=["out"], write_state=False)
    na.split(splitter="a", inputs={"a": [3, 5]})
    wf.add(na)
    # the second node does not have explicit splitter (but keeps the splitter from the NA node)
    nb = Node(name="NB", interface=interf_addvar, workingdir="nb",
              output_names=["out"])
    wfb = Workflow(name="wfb", workingdir="test_wfb", inputs={"c": 10},
        wf_output_names=[("NB", "out", "NB_out")], write_state=False)
    wfb.add(nb)
    wfb.connect_wf_input("b", "NB", "b")
    wfb.connect_wf_input("c", "NB", "c")

    # adding 2 nodes and create a connection (as it is now)
    wf.add(wfb)
    wf.connect("NA", "out", "wfb", "b")
    assert wf.nodes[0].splitter == "NA.a"

    sub = Submitter(runnable=wf, plugin=plugin)
    sub.run()
    sub.close()

    # wf.result for NB_out doeant have state in results
    assert wf.is_complete

    expected_A = [({"NA.a": "0"}, 5), ({"NA.a": "1"}, 7)]
    for i, res in enumerate(expected_A):
        assert wf.result["NA_out"][i][0] == res[0]
        assert wf.result["NA_out"][i][1] == res[1]

    # TODO (res): the naming remembers only the node, doesnt remember that came from NA...
    # TODO (res): because nb doesnt have splitter, but only wfb, and wf.result reads from node results,
    # TODO ...  : we dont have any state values here. probably should change it that wf can see wfb.b values
    #TODO (res): compare wf.result and wfb.rsult (wfb has to many var in state_Dict)
    # the naming should have names with workflows??
    expected_B = [({}, 15), ({}, 17)]
    key_sort = list(expected_B[0][0].keys())
    expected_B.sort(key=lambda t: [t[0][key] for key in key_sort])
    if wf.result["NB_out"][0][0]: #has some state valuse
        wf.result["NB_out"].sort(key=lambda t: [t[0][key] for key in key_sort])
    else: # dictionary empty so sorting regarding the values
        wf.result["NB_out"].sort()
    for i, res in enumerate(expected_B):
        assert wf.result["NB_out"][i][0] == res[0]
        assert wf.result["NB_out"][i][1] == res[1]



# testing CurrentInterface that is a temporary wrapper for current interfaces
T1_file = "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"
T1_file_list = [
    "/Users/dorota/nipype_workshop/data/ds000114/sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz",
    "/Users/dorota/nipype_workshop/data/ds000114/sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz"
    ]

@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_node_1(change_dir, plugin):
    """Node with a current interface and inputs, no splitter, running interface"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")
    nn = Node(
        name="NA",
        inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
        interface=interf_bet,
        workingdir="test_cnd1_{}".format(plugin),
        output_names=["out_file"])

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()
    # TODO (res): nodes only returns relative path
    assert "out_file" in nn.output.keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_node_2(change_dir, plugin):
    """Node with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
    ]
    nn = Node(
        name="NA",
        inputs={"in_file": in_file_l},
        splitter="in_file",
        interface=interf_bet,
        write_state=False,
        workingdir="test_cnd2_{}".format(plugin),
        output_names=["out_file"])

    sub = Submitter(plugin=plugin, runnable=nn)
    sub.run()
    sub.close()

    assert "out_file" in nn.output.keys()
    assert "NA.in_file:0" in nn.output["out_file"].keys()
    assert "NA.in_file:1" in nn.output["out_file"].keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_1(change_dir, plugin):
    """Wf with a current interface, no splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    nn = Node(
        name="fsl",
        inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
        interface=interf_bet,
        workingdir="nn",
        output_names=["out_file"],
        write_state=False)

    wf = Workflow(
        workingdir="test_cwf_1_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False)
    wf.add_nodes([nn])

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_1a(change_dir, plugin):
    """Wf with a current interface, no splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    nn = Node(
        name="fsl",
        inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")},
        interface=interf_bet,
        workingdir="nn",
        output_names=["out_file"],
        write_state=False)

    wf = Workflow(
        workingdir="test_cwf_1a_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False)
    wf.add(runnable=nn)

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_1b(change_dir, plugin):
    """Wf with a current interface, no splitter; using wf.add(nipype CurrentInterface)"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    wf = Workflow(
        workingdir="test_cwf_1b_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False)
    wf.add(
        runnable=interf_bet,
        name="fsl",
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
        inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")})

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_1c(change_dir, plugin):
    """Wf with a current interface, no splitter; using wf.add(nipype interface) """

    wf = Workflow(
        workingdir="test_cwf_1c_{}".format(plugin),
        name="cw1",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False)
    wf.add(
        runnable=fsl.BET(),
        name="fsl",
        workingdir="nn",
        output_names=["out_file"],
        write_state=False,
        inputs={"in_file": str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz")})

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_2(change_dir, plugin):
    """Wf with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
    ]

    nn = Node(
        name="fsl",
        interface=interf_bet,
        write_state=False,
        workingdir="nn",
        output_names=["out_file"])

    wf = Workflow(
        workingdir="test_cwf_2_{}".format(plugin),
        name="cw2",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        inputs={"in_file": in_file_l},
        splitter="in_file",
        write_state=False)
    wf.add_nodes([nn])
    wf.connect_wf_input("in_file", "fsl", "in_file")

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()
    assert 'cw2.in_file:0' in wf.output["fsl_out"].keys()
    assert 'cw2.in_file:1' in wf.output["fsl_out"].keys()


@pytest.mark.skipif(not DS114_DIR.exists(), reason="Missing $PYDRA_TEST_DATA/ds000114")
@pytest.mark.parametrize("plugin", Plugins)
@python35_only
def test_current_wf_2a(change_dir, plugin):
    """Wf with a current interface and splitter"""
    interf_bet = CurrentInterface(interface=fsl.BET(), name="fsl_interface")

    in_file_l = [
        str(DS114_DIR / "sub-01/ses-test/anat/sub-01_ses-test_T1w.nii.gz"),
        str(DS114_DIR / "sub-02/ses-test/anat/sub-02_ses-test_T1w.nii.gz")
    ]

    nn = Node(
        name="fsl",
        interface=interf_bet,
        write_state=False,
        workingdir="nn",
        output_names=["out_file"],
        inputs={"in_file": in_file_l},
        splitter="in_file")

    wf = Workflow(
        workingdir="test_cwf_2a_{}".format(plugin),
        name="cw2a",
        wf_output_names=[("fsl", "out_file", "fsl_out")],
        write_state=False)
    wf.add_nodes([nn])
    # wf.connect_wf_input("in_file", "fsl", "in_file")

    sub = Submitter(plugin=plugin, runnable=wf)
    sub.run()
    sub.close()

    assert "fsl_out" in wf.output.keys()
    assert 'fsl.in_file:0' in wf.output["fsl_out"].keys()
    assert 'fsl.in_file:1' in wf.output["fsl_out"].keys()
