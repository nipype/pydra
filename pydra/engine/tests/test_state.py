import sys
import numpy as np

from ..state import State
from ..node import Node, Workflow
from ..task import to_task

from nipype import Function

import pytest, pdb
python35_only = pytest.mark.skipif(sys.version_info < (3, 5), reason="requires Python>3.4")


@to_task
def fun_addtwo(a):
    import time
    time.sleep(1)
    if a == 3:
        time.sleep(2)
    return a + 2


@to_task
def fun_addvar(b, c):
    return b + c


@to_task
def fun_addvar3(a, b, c):
    return a + b + c



@pytest.mark.parametrize("inputs, splitter, ndim, states_ind, states_val, input_for_groups, "
                         "groups_stack", [
    ({"NA.a": [3, 5]}, "a", 1, [{'NA.a': 0}, {'NA.a': 1}],
     [{'NA.a': 3}, {'NA.a': 5}], {0: ["NA.a"]}, [[0]]),
    ({"NA.a": [3, 5], "NA.b": ["str1", "str2"]}, ("a", "b"), 1,
     [{'NA.a': 0, 'NA.b': 0}, {'NA.a': 1, 'NA.b': 1}],
     [{'NA.a': 3, 'NA.b': "str1"}, {'NA.a': 5, 'NA.b': "str2"}],
     {0: ["NA.a", "NA.b"]}, [[0]]),
    ({"NA.a": [3, 5], "NA.b": ["str1", "str2"]}, ["a", "b"], 2,
     [{'NA.a': 0, 'NA.b': 0}, {'NA.a': 0, 'NA.b': 1},
      {'NA.a': 1, 'NA.b': 0}, {'NA.a': 1, 'NA.b': 1}],
     [{'NA.a': 3, 'NA.b': "str1"}, {'NA.a': 3, 'NA.b': "str2"},
      {'NA.a': 5, 'NA.b': "str1"}, {'NA.a': 5, 'NA.b': "str2"}],
     {0: ["NA.a"], 1: ["NA.b"]}, [[0, 1]]),
    ({"NA.a": [3, 5], "NA.b": ["str1", "str2"], "NA.c": [10, 20]}, [("a", "c"), "b"], 2,
     [{'NA.a': 0, 'NA.b': 0, "NA.c": 0}, {'NA.a': 0, 'NA.b': 1, "NA.c": 0},
      {'NA.a': 1, 'NA.b': 0, "NA.c": 1}, {'NA.a': 1, 'NA.b': 1, "NA.c": 1}],
     [{'NA.a': 3, 'NA.b': "str1", "NA.c": 10}, {'NA.a': 3, 'NA.b': "str2", "NA.c": 10},
      {'NA.a': 5, 'NA.b': "str1", "NA.c": 20}, {'NA.a': 5, 'NA.b': "str2", "NA.c": 20}],
      {0: ["NA.a", "NA.c"], 1: ["NA.b"]}, [[0, 1]])
])
def test_state_1(inputs, splitter, ndim, states_ind, states_val, input_for_groups, groups_stack):
    st = State(name="NA", splitter=splitter)
    st.prepare_states_ind(inputs)
    assert st.states_ind == states_ind
    st.prepare_states_val(inputs)
    assert st.states_val == states_val
    assert st.input_for_groups == input_for_groups
    assert st.ndim == ndim
    assert st.groups_stack == groups_stack


def test_state_merge_1():
    st1 = State(name="NA", splitter="a")#, inputs={"a": [3, 5]})
    st2 = State(name="NB", others={st1: "b"})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a"]

    st2.prepare_states(inputs={"NA.a": [3, 5]})
    assert st2.states_ind == [{'NA.a': 0}, {'NA.a': 1}]
    assert st2.states_val == [{'NA.a': 3}, {'NA.a': 5}]


def test_state_merge_1a():
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="_NA", others={st1: "b"})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a"]

    st2.prepare_states(inputs={"NA.a": [3, 5]})
    assert st2.states_ind == [{'NA.a': 0}, {'NA.a': 1}]
    assert st2.states_val == [{'NA.a': 3}, {'NA.a': 5}]


@pytest.mark.xfail(reason="should check if I'm not using partial splitter from previous node")
def test_state_merge_1b():
    st1 = State(name="NA", splitter="a")
    with pytest.raises(Exception):
        st2 = State(name="NB", splitter="NA.a", others={st1: "b"})


def test_state_merge_2():
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["_NA", "a"], others={st1: "b"})
    st1.prepare_states(inputs={"NA.a": [3, 5]})

    assert st2.splitter == ["_NA", "NB.a"]
    assert st2.splitter_rpn == ["NA.a", "NB.a", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [1, 2]})
    assert st2.states_ind == [{'NA.a': 0, "NB.a": 0}, {'NA.a': 0, "NB.a": 1},
                              {'NA.a': 1, "NB.a": 0}, {'NA.a': 1, "NB.a": 1}]
    assert st2.states_val == [{'NA.a': 3, "NB.a": 1}, {'NA.a': 3, "NB.a": 2},
                              {'NA.a': 5, "NB.a": 1}, {'NA.a': 5, "NB.a": 2}]


def test_state_merge_2a():
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["_NA", "a"], others={st1: "b"})
    st1.prepare_states(inputs={"NA.a": [3, 5]})

    assert st2.splitter == ["_NA", "NB.a"]
    assert st2.splitter_rpn == ["NA.a", "NB.a", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [1, 2], "NB.s": 1})
    assert st2.states_ind == [{'NA.a': 0, "NB.a": 0}, {'NA.a': 0, "NB.a": 1},
                              {'NA.a': 1, "NB.a": 0}, {'NA.a': 1, "NB.a": 1}]
    assert st2.states_val == [{'NA.a': 3, "NB.a": 1}, {'NA.a': 3, "NB.a": 2},
                              {'NA.a': 5, "NB.a": 1}, {'NA.a': 5, "NB.a": 2}]



def test_state_merge_3():
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(name="NC", others={st1: "b", st2: "c"})

    assert st3.splitter == ["_NB", "_NA"]
    assert st3.splitter_rpn == ["NB.a", "NA.a", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [30, 50]})
    assert st3.states_ind == [{'NA.a': 0, "NB.a": 0}, {'NA.a': 1, "NB.a": 0},
                              {'NA.a': 0, "NB.a": 1}, {'NA.a': 1, "NB.a": 1}]
    assert st3.states_val == [{'NA.a': 3, "NB.a": 30}, {'NA.a': 5, "NB.a": 30},
                              {'NA.a': 3, "NB.a": 50}, {'NA.a': 5, "NB.a": 50}]


def test_state_merge_4():
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", others={st1: "a"})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a", "NA.b", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20]})
    assert st2.states_ind == [{'NA.a': 0, "NA.b": 0}, {'NA.a': 0, "NA.b": 1},
                              {'NA.a': 1, "NA.b": 0}, {'NA.a': 1, "NA.b": 1}]
    assert st2.states_val == [{'NA.a': 3, "NA.b": 10}, {'NA.a': 3, "NA.b": 20},
                              {'NA.a': 5, "NA.b": 10}, {'NA.a': 5, "NA.b": 20}]


def test_state_merge_5():
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", splitter="a")
    st3 = State(name="NC", others={st1: "a", st2: "b"})
    assert st3.splitter == ["_NB", "_NA"]
    assert st3.splitter_rpn == ["NB.a", "NA.a", "NA.b", "*", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.a": [600, 700]})
    assert st3.states_ind == [{'NB.a': 0, 'NA.a': 0, "NA.b": 0}, {'NB.a': 0, 'NA.a': 0, "NA.b": 1},
                              {'NB.a': 0, 'NA.a': 1, "NA.b": 0}, {'NB.a': 0, 'NA.a': 1, "NA.b": 1},
                              {'NB.a': 1, 'NA.a': 0, "NA.b": 0}, {'NB.a': 1, 'NA.a': 0, "NA.b": 1},
                              {'NB.a': 1, 'NA.a': 1, "NA.b": 0}, {'NB.a': 1, 'NA.a': 1, "NA.b": 1}]
    assert st3.states_val == [{'NB.a': 600, 'NA.a': 3, "NA.b": 10}, {'NB.a': 600, 'NA.a': 3, "NA.b": 20},
                              {'NB.a': 600, 'NA.a': 5, "NA.b": 10}, {'NB.a': 600, 'NA.a': 5, "NA.b": 20},
                              {'NB.a': 700, 'NA.a': 3, "NA.b": 10}, {'NB.a': 700, 'NA.a': 3, "NA.b": 20},
                              {'NB.a': 700, 'NA.a': 5, "NA.b": 10}, {'NB.a': 700, 'NA.a': 5, "NA.b": 20}]


@pytest.mark.xfail(reason="wip")
def test_state_combine_1():
    st = State(name="NA", splitter="a", combiner="a")
    assert st.splitter == "NA.a"
    assert st.splitter_rpn == ["NA.a"]
    #pdb.set_trace()

    # st2.prepare_states(inputs={"NA.a": [3, 5]})
    # assert st2.states_ind == [{'NA.a': 0}, {'NA.a': 1}]
    # assert st2.states_val == [{'NA.a': 3}, {'NA.a': 5}]



# def test_state_merge_inner_1():
#     st1 = State(name="NA", splitter="a", inputs={"a": [3, 5]})
#     st2 = State(name="NB", others=[{"state": st1, "in_field": "b"}])
#     assert st2.splitter == "NA.a"
#     st2.prepare_states_ind()


        # assert st._splitter_rpn == ["NA.a"]
    # assert st.splitter_comb is None
    # assert st._splitter_rpn_comb == []
    #
    # st.prepare_state_input()
    #
    # expected_axis_for_input = {"NA.a": [0]}
    # for key, val in expected_axis_for_input.items():
    #     assert st._axis_for_input[key] == val
    # assert st._ndim == 1
    # assert st._input_for_axis == [["NA.a"]]
    #
    #
    # expected_axis_for_input_comb = {}
    # for key, val in expected_axis_for_input_comb.items():
    #     assert st._axis_for_input_comb[key] == val
    # assert st._ndim_comb == 0
    # assert st._input_for_axis_comb == []



# def test_state_1():
#     nd = Node(name="NA", interface=fun_addtwo(), splitter="a", combiner="a",
#               inputs={"a": np.array([3, 5])})
#     st = State(node=nd)
#
#     assert st._splitter == "NA.a"
#     assert st._splitter_rpn == ["NA.a"]
#     assert st.splitter_comb is None
#     assert st._splitter_rpn_comb == []
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"NA.a": [0]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 1
#     assert st._input_for_axis == [["NA.a"]]
#
#
#     expected_axis_for_input_comb = {}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 0
#     assert st._input_for_axis_comb == []
#
#
# def test_state_1a():
#     wf = Workflow(name="wf", workingdir="wf_test", splitter="a", combiner="a",
#                   inputs={"a": np.array([3, 5])})
#     st = State(node=wf)
#
#     assert st._splitter == "wf.a"
#     assert st._splitter_rpn == ["wf.a"]
#     assert st.splitter_comb is None
#     assert st._splitter_rpn_comb == []
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"wf.a": [0]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 1
#     assert st._input_for_axis == [["wf.a"]]
#
#
#     expected_axis_for_input_comb = {}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 0
#     assert st._input_for_axis_comb == []
#
#
# def test_state_2():
#     nd = Node(name="NA", interface=fun_addvar(), splitter=("a", "b"), combiner="a",
#               inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
#     st = State(node=nd)
#
#     assert st._splitter == ("NA.a", "NA.b")
#     assert st._splitter_rpn == ["NA.a", "NA.b", "."]
#     assert st.splitter_comb is None
#     assert st._splitter_rpn_comb == []
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"NA.a": [0], "NA.b": [0]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 1
#     assert st._input_for_axis == [["NA.a", "NA.b"]]
#
#
#     expected_axis_for_input_comb = {}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 0
#     assert st._input_for_axis_comb == []
#
#
# def test_state_2a():
#     wf = Workflow(name="wf", workingdir="wf_test", splitter=("a", "b"), combiner="a",
#               inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
#     st = State(node=wf)
#
#     assert st._splitter == ("wf.a", "wf.b")
#     assert st._splitter_rpn == ["wf.a", "wf.b", "."]
#     assert st.splitter_comb is None
#     assert st._splitter_rpn_comb == []
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"wf.a": [0], "wf.b": [0]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 1
#     assert st._input_for_axis == [["wf.a", "wf.b"]]
#
#
#     expected_axis_for_input_comb = {}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 0
#     assert st._input_for_axis_comb == []
#
#
# def test_state_3():
#     nd = Node(name="NA", interface=fun_addvar(), splitter=["a", "b"], combiner="a",
#               inputs={"a": np.array([3, 5]), "b": np.array([3, 5])})
#     st = State(node=nd)
#
#     assert st._splitter == ["NA.a", "NA.b"]
#     assert st._splitter_rpn == ["NA.a", "NA.b", "*"]
#     assert st.splitter_comb == "NA.b"
#     assert st._splitter_rpn_comb == ["NA.b"]
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"NA.a": [0], "NA.b": [1]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 2
#     assert st._input_for_axis == [["NA.a"], ["NA.b"]]
#
#
#     expected_axis_for_input_comb = {"NA.b": [0]}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 1
#     assert st._input_for_axis_comb == [["NA.b"]]
#
#
# def test_state_4():
#     nd = Node(name="NA", interface=fun_addvar3(), splitter=["a", ("b", "c")], combiner="b",
#               inputs={"a": np.array([3, 5]), "b": np.array([3, 5]), "c": np.array([3, 5])})
#     st = State(node=nd)
#
#     assert st._splitter == ["NA.a", ("NA.b", "NA.c")]
#     assert st._splitter_rpn == ["NA.a", "NA.b", "NA.c", ".", "*"]
#     assert st.splitter_comb == "NA.a"
#     assert st._splitter_rpn_comb == ["NA.a"]
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"NA.a": [0], "NA.b": [1], "NA.c": [1]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 2
#     assert st._input_for_axis == [["NA.a"], ["NA.b", "NA.c"]]
#
#
#     expected_axis_for_input_comb = {"NA.a": [0]}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 1
#     assert st._input_for_axis_comb == [["NA.a"]]
#
#
# def test_state_5():
#     nd = Node(name="NA", interface=fun_addvar3(), splitter=("a", ["b", "c"]), combiner="b",
#               inputs={"a": np.array([[3, 5], [3, 5]]), "b": np.array([3, 5]),
#                       "c": np.array([3, 5])})
#     st = State(node=nd)
#
#     assert st._splitter == ("NA.a", ["NA.b", "NA.c"])
#     assert st._splitter_rpn == ["NA.a", "NA.b", "NA.c", "*", "."]
#     assert st.splitter_comb == "NA.c"
#     assert st._splitter_rpn_comb == ["NA.c"]
#
#     st.prepare_state_input()
#
#     expected_axis_for_input = {"NA.a": [0, 1], "NA.b": [0], "NA.c": [1]}
#     for key, val in expected_axis_for_input.items():
#         assert st._axis_for_input[key] == val
#     assert st._ndim == 2
#     expected_input_for_axis = [["NA.a", "NA.b"], ["NA.a", "NA.c"]]
#     for (i, exp_l) in enumerate(expected_input_for_axis):
#         exp_l.sort()
#         st._input_for_axis[i].sort()
#     assert st._input_for_axis[i] == exp_l
#
#
#     expected_axis_for_input_comb = {"NA.c": [0]}
#     for key, val in expected_axis_for_input_comb.items():
#         assert st._axis_for_input_comb[key] == val
#     assert st._ndim_comb == 1
#     assert st._input_for_axis_comb == [["NA.c"]]
