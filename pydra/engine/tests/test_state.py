import sys
import numpy as np

from ..state import State
from ..node import Node, Workflow
from ..auxiliary import CurrentInterface

from nipype import Function

import pytest, pdb
python35_only = pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python>3.4")

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


def test_state_1():
    nd = Node(name="NA", interface=interf_addtwo, splitter="a", combiner="a",
              inputs={"a": np.array([3, 5])})
    st = State(node=nd)

    assert st._splitter == "NA.a"
    assert st._splitter_rpn == ["NA.a"]
    assert st.splitter_comb is None
    assert st._splitter_rpn_comb == []

    st.prepare_state_input()

    expected_axis_for_input = {"NA.a": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["NA.a"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_1a():
    wf = Workflow(name="wf", workingdir="wf_test", splitter="a", combiner="a",
                  inputs={"a": np.array([3, 5])})
    st = State(node=wf)

    assert st._splitter == "wf.a"
    assert st._splitter_rpn == ["wf.a"]
    assert st.splitter_comb is None
    assert st._splitter_rpn_comb == []

    st.prepare_state_input()

    expected_axis_for_input = {"wf.a": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["wf.a"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_2():
    nd = Node(name="NA", interface=interf_addvar, splitter=("a", "b"), combiner="a",
              inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
    st = State(node=nd)

    assert st._splitter == ("NA.a", "NA.b")
    assert st._splitter_rpn == ["NA.a", "NA.b", "."]
    assert st.splitter_comb is None
    assert st._splitter_rpn_comb == []

    st.prepare_state_input()

    expected_axis_for_input = {"NA.a": [0], "NA.b": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["NA.a", "NA.b"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_2a():
    wf = Workflow(name="wf", workingdir="wf_test", splitter=("a", "b"), combiner="a",
              inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
    st = State(node=wf)

    assert st._splitter == ("wf.a", "wf.b")
    assert st._splitter_rpn == ["wf.a", "wf.b", "."]
    assert st.splitter_comb is None
    assert st._splitter_rpn_comb == []

    st.prepare_state_input()

    expected_axis_for_input = {"wf.a": [0], "wf.b": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["wf.a", "wf.b"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_3():
    nd = Node(name="NA", interface=interf_addvar, splitter=["a", "b"], combiner="a",
              inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
    st = State(node=nd)

    assert st._splitter == ["NA.a", "NA.b"]
    assert st._splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st.splitter_comb == "NA.b"
    assert st._splitter_rpn_comb == ["NA.b"]

    st.prepare_state_input()

    expected_axis_for_input = {"NA.a": [0], "NA.b": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    assert st._input_for_axis == [["NA.a"], ["NA.b"]]


    expected_axis_for_input_comb = {"NA.b": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["NA.b"]]


def test_state_4():
    nd = Node(name="NA", interface=_interf_addvar3, splitter=["a", ("b", "c")], combiner="b",
              inputs={"a": np.array([3, 5]), "b": np.array([3, 5]), "c": np.array([3, 5])})
    st = State(node=nd)

    assert st._splitter == ["NA.a", ("NA.b", "NA.c")]
    assert st._splitter_rpn == ["NA.a", "NA.b", "NA.c", ".", "*"]
    assert st.splitter_comb == "NA.a"
    assert st._splitter_rpn_comb == ["NA.a"]

    st.prepare_state_input()

    expected_axis_for_input = {"NA.a": [0], "NA.b": [1], "NA.c": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    assert st._input_for_axis == [["NA.a"], ["NA.b", "NA.c"]]


    expected_axis_for_input_comb = {"NA.a": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["NA.a"]]


def test_state_5():
    nd = Node(name="NA", interface=_interf_addvar3, splitter=("a", ["b", "c"]), combiner="b",
              inputs={"a": np.array([[3, 5], [3, 5]]), "b": np.array([3, 5]),
                      "c": np.array([3, 5])})
    st = State(node=nd)

    assert st._splitter == ("NA.a", ["NA.b", "NA.c"])
    assert st._splitter_rpn == ["NA.a", "NA.b", "NA.c", "*", "."]
    assert st.splitter_comb == "NA.c"
    assert st._splitter_rpn_comb == ["NA.c"]

    st.prepare_state_input()

    expected_axis_for_input = {"NA.a": [0, 1], "NA.b": [0], "NA.c": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    expected_input_for_axis = [["NA.a", "NA.b"], ["NA.a", "NA.c"]]
    for (i, exp_l) in enumerate(expected_input_for_axis):
        exp_l.sort()
        st._input_for_axis[i].sort()
    assert st._input_for_axis[i] == exp_l


    expected_axis_for_input_comb = {"NA.c": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["NA.c"]]
