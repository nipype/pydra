import sys
import numpy as np

from ..state import State

import pytest

python35_only = pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python>3.4")


def test_state_1():
    st = State(node_name="test", mapper="a", combiner="a")
    st.prepare_state_input({"a": np.array([3,5])})

    assert st._mapper == "a"
    assert st._mapper_rpn == ["a"]
    assert st.mapper_comb is None
    assert st._mapper_rpn_comb == []

    expected_axis_for_input = {"a": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["a"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_2():
    st = State(node_name="test", mapper=("a", "b"), combiner="a")
    st.prepare_state_input({"a": np.array([3,5]), "b": np.array([3,5])})

    assert st._mapper == ("a", "b")
    assert st._mapper_rpn == ["a", "b", "."]
    assert st.mapper_comb is None
    assert st._mapper_rpn_comb == []

    expected_axis_for_input = {"a": [0], "b": [0]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 1
    assert st._input_for_axis == [["a", "b"]]


    expected_axis_for_input_comb = {}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 0
    assert st._input_for_axis_comb == []


def test_state_3():
    st = State(node_name="test", mapper=["a", "b"], combiner="a")
    st.prepare_state_input({"a": np.array([3,5]), "b": np.array([3,5])})

    assert st._mapper == ["a", "b"]
    assert st._mapper_rpn == ["a", "b", "*"]
    assert st.mapper_comb == "b"
    assert st._mapper_rpn_comb == ["b"]

    expected_axis_for_input = {"a": [0], "b": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    assert st._input_for_axis == [["a"], ["b"]]


    expected_axis_for_input_comb = {"b": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["b"]]


def test_state_4():
    st = State(node_name="test", mapper=["a", ("b", "c")], combiner="b")
    st.prepare_state_input({"a": np.array([3,5]), "b": np.array([3,5]), "c": np.array([3,5])})

    assert st._mapper == ["a", ("b", "c")]
    assert st._mapper_rpn == ["a", "b", "c", ".", "*"]
    assert st.mapper_comb == "a"
    assert st._mapper_rpn_comb == ["a"]

    expected_axis_for_input = {"a": [0], "b": [1], "c": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    assert st._input_for_axis == [["a"], ["b", "c"]]


    expected_axis_for_input_comb = {"a": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["a"]]


def test_state_5():
    st = State(node_name="test", mapper=("a", ["b", "c"]), combiner="b")
    st.prepare_state_input({"a": np.array([[3, 5], [3,5]]), "b": np.array([3,5]), "c": np.array([3,5])})

    assert st._mapper == ("a", ["b", "c"])
    assert st._mapper_rpn == ["a", "b", "c", "*", "."]
    assert st.mapper_comb == "c"
    assert st._mapper_rpn_comb == ["c"]

    expected_axis_for_input = {"a": [0, 1], "b": [0], "c": [1]}
    for key, val in expected_axis_for_input.items():
        assert st._axis_for_input[key] == val
    assert st._ndim == 2
    expected_input_for_axis = [["a", "b"], ["a", "c"]]
    for (i, exp_l) in enumerate(expected_input_for_axis):
        exp_l.sort()
        st._input_for_axis[i].sort()
    assert st._input_for_axis[i] == exp_l


    expected_axis_for_input_comb = {"c": [0]}
    for key, val in expected_axis_for_input_comb.items():
        assert st._axis_for_input_comb[key] == val
    assert st._ndim_comb == 1
    assert st._input_for_axis_comb == [["c"]]
