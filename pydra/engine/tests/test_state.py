import pytest

from ..state import State
from ..helpers_state import PydraStateError


@pytest.mark.parametrize(
    "inputs, splitter, ndim, states_ind, states_val, group_for_inputs, " "groups_stack",
    [
        (
            {"NA.a": [3, 5]},
            "a",
            1,
            [{"NA.a": 0}, {"NA.a": 1}],
            [{"NA.a": 3}, {"NA.a": 5}],
            {"NA.a": 0},
            [[0]],
        ),
        (
            {"NA.a": [3, 5], "NA.b": ["str1", "str2"]},
            ("a", "b"),
            1,
            [{"NA.a": 0, "NA.b": 0}, {"NA.a": 1, "NA.b": 1}],
            [{"NA.a": 3, "NA.b": "str1"}, {"NA.a": 5, "NA.b": "str2"}],
            {"NA.a": 0, "NA.b": 0},
            [[0]],
        ),
        (
            {"NA.a": [3, 5], "NA.b": ["str1", "str2"]},
            ["a", "b"],
            2,
            [
                {"NA.a": 0, "NA.b": 0},
                {"NA.a": 0, "NA.b": 1},
                {"NA.a": 1, "NA.b": 0},
                {"NA.a": 1, "NA.b": 1},
            ],
            [
                {"NA.a": 3, "NA.b": "str1"},
                {"NA.a": 3, "NA.b": "str2"},
                {"NA.a": 5, "NA.b": "str1"},
                {"NA.a": 5, "NA.b": "str2"},
            ],
            {"NA.a": 0, "NA.b": 1},
            [[0, 1]],
        ),
        (
            {"NA.a": [3, 5], "NA.b": "str1"},
            "a",
            1,
            [{"NA.a": 0}, {"NA.a": 1}],
            [{"NA.a": 3}, {"NA.a": 5}],
            {"NA.a": 0},
            [[0]],
        ),
        (
            {"NA.a": [3, 5], "NA.b": ["str1", "str2"], "NA.c": [10, 20]},
            [("a", "c"), "b"],
            2,
            [
                {"NA.a": 0, "NA.b": 0, "NA.c": 0},
                {"NA.a": 0, "NA.b": 1, "NA.c": 0},
                {"NA.a": 1, "NA.b": 0, "NA.c": 1},
                {"NA.a": 1, "NA.b": 1, "NA.c": 1},
            ],
            [
                {"NA.a": 3, "NA.b": "str1", "NA.c": 10},
                {"NA.a": 3, "NA.b": "str2", "NA.c": 10},
                {"NA.a": 5, "NA.b": "str1", "NA.c": 20},
                {"NA.a": 5, "NA.b": "str2", "NA.c": 20},
            ],
            {"NA.a": 0, "NA.c": 0, "NA.b": 1},
            [[0, 1]],
        ),
    ],
)
def test_state_1(
    inputs, splitter, ndim, states_ind, states_val, group_for_inputs, groups_stack
):
    """ single state: testing groups, prepare_states and prepare_inputs"""
    st = State(name="NA", splitter=splitter)
    assert st.splitter == st.current_splitter
    assert st.splitter_rpn == st.current_splitter_rpn
    assert st.prev_state_splitter is None
    assert st.prev_state_combiner_all == []

    st.prepare_states(inputs)
    assert st.group_for_inputs_final == group_for_inputs
    assert st.groups_stack_final == groups_stack

    assert st.states_ind == states_ind
    assert st.states_val == states_val

    st.prepare_inputs()
    assert st.inputs_ind == states_ind


def test_state_2_err():
    with pytest.raises(PydraStateError) as exinfo:
        st = State("NA", splitter={"a"})
    assert "splitter has to be a string, a tuple or a list" == str(exinfo.value)


def test_state_3_err():
    with pytest.raises(PydraStateError) as exinfo:
        st = State("NA", splitter=["a", "b"], combiner=("a", "b"))
    assert "combiner has to be a string or a list" == str(exinfo.value)


def test_state_4_err():
    st = State("NA", splitter="a", combiner=["a", "b"])
    with pytest.raises(PydraStateError) as exinfo:
        st.combiner_validation()
    assert "all combiners have to be in the splitter" in str(exinfo.value)


def test_state_5_err():
    st = State("NA", combiner="a")
    with pytest.raises(PydraStateError) as exinfo:
        st.combiner_validation()
    assert "splitter has to be set before" in str(exinfo.value)


def test_state_connect_1():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs
        no explicit splitter for the second state
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", other_states={"NA": (st1, "b")})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a"]
    assert st2.prev_state_splitter == st2.splitter
    assert st2.prev_state_splitter_rpn == st2.splitter_rpn
    assert st2.current_splitter is None
    assert st2.current_splitter_rpn == []

    st2.prepare_states(inputs={"NA.a": [3, 5]})
    assert st2.group_for_inputs_final == {"NA.a": 0}
    assert st2.groups_stack_final == [[0]]
    assert st2.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert st2.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.b": 0}, {"NB.b": 1}]


def test_state_connect_1a():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs
        the second state has explicit splitter from the first one (the prev-state part)
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="_NA", other_states={"NA": (st1, "b")})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a"]

    st2.prepare_states(inputs={"NA.a": [3, 5]})
    assert st2.group_for_inputs_final == {"NA.a": 0}
    assert st2.groups_stack_final == [[0]]
    assert st2.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert st2.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.b": 0}, {"NB.b": 1}]


def test_state_connect_1b_exception():
    """can't provide explicitly NA.a (should be _NA)"""
    st1 = State(name="NA", splitter="a", other_states={})
    st2 = State(name="NB", splitter="NA.a")
    with pytest.raises(PydraStateError) as excinfo:
        st2.splitter_validation()
    assert "consider using _NA" in str(excinfo.value)


@pytest.mark.parametrize("splitter2, other_states2", [("_NA", {}), ("_N", {"NA": ()})])
def test_state_connect_1c_exception(splitter2, other_states2):
    """can't ask for splitter from node that is not connected"""
    st1 = State(name="NA", splitter="a")
    with pytest.raises(PydraStateError) as excinfo:
        st2 = State(name="NB", splitter=splitter2, other_states=other_states2)
        st2.splitter_validation()


def test_state_connect_2():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs
         the second state has explicit splitter that contains
         splitter from the first node and a new field (the prev-state and current part)
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["_NA", "a"], other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", "NB.a"]
    assert st2.splitter_rpn == ["NA.a", "NB.a", "*"]
    assert st2.prev_state_splitter == "_NA"
    assert st2.prev_state_splitter_rpn == ["NA.a"]
    assert st2.current_splitter == "NB.a"
    assert st2.current_splitter_rpn == ["NB.a"]

    st2.update_connections()
    st2.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [1, 2]})
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st2.groups_stack_final == [[0, 1]]
    assert st2.keys_final == ["NA.a", "NB.a"]

    assert st2.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NB.a": 1},
        {"NA.a": 3, "NB.a": 2},
        {"NA.a": 5, "NB.a": 1},
        {"NA.a": 5, "NB.a": 2},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0, "NB.a": 0},
        {"NB.b": 0, "NB.a": 1},
        {"NB.b": 1, "NB.a": 0},
        {"NB.b": 1, "NB.a": 1},
    ]


def test_state_connect_2a():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs
         the second state has explicit splitter that contains
         splitter from the first node and a new field;
         adding an additional scalar field that is not part of the splitter
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["_NA", "a"], other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", "NB.a"]
    assert st2.splitter_rpn == ["NA.a", "NB.a", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [1, 2], "NB.s": 1})
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st2.groups_stack_final == [[0, 1]]
    assert st2.keys_final == ["NA.a", "NB.a"]

    assert st2.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NB.a": 1},
        {"NA.a": 3, "NB.a": 2},
        {"NA.a": 5, "NB.a": 1},
        {"NA.a": 5, "NB.a": 2},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0, "NB.a": 0},
        {"NB.b": 0, "NB.a": 1},
        {"NB.b": 1, "NB.a": 0},
        {"NB.b": 1, "NB.a": 1},
    ]


def test_state_connect_2b():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs
        the second state has explicit splitter with a new field (the current part)
        splitter from the first node (the prev-state part) has to be added
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a", other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", "NB.a"]
    assert st2.splitter_rpn == ["NA.a", "NB.a", "*"]
    assert st2.current_splitter == "NB.a"
    assert st2.prev_state_splitter == "_NA"

    st2.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [1, 2]})
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st2.groups_stack_final == [[0, 1]]
    assert st2.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NB.a": 1},
        {"NA.a": 3, "NB.a": 2},
        {"NA.a": 5, "NB.a": 1},
        {"NA.a": 5, "NB.a": 2},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0, "NB.a": 0},
        {"NB.b": 0, "NB.a": 1},
        {"NB.b": 1, "NB.a": 0},
        {"NB.b": 1, "NB.a": 1},
    ]


def test_state_connect_3():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs
        the third state connected to two previous states;
        splitter from the previous states (the prev-state part) has to be added
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(name="NC", other_states={"NA": (st1, "b"), "NB": (st2, "c")})

    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NB.a", "*"]
    assert st3.prev_state_splitter == st3.splitter
    assert st3.prev_state_splitter_rpn == st3.splitter_rpn
    assert st3.prev_state_splitter_rpn_compact == ["_NA", "_NB", "*"]
    assert st3.current_splitter is None
    assert st3.current_splitter_rpn == []

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [30, 50]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st3.groups_stack_final == [[0, 1]]
    assert st3.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NB.a": 30},
        {"NA.a": 3, "NB.a": 50},
        {"NA.a": 5, "NB.a": 30},
        {"NA.a": 5, "NB.a": 50},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.b": 0, "NC.c": 0},
        {"NC.b": 0, "NC.c": 1},
        {"NC.b": 1, "NC.c": 0},
        {"NC.b": 1, "NC.c": 1},
    ]


def test_state_connect_3a():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs
        the third state connected to two previous states;
        the third state has explicit splitter that contains splitters from previous states
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC",
        splitter=["_NA", "_NB"],
        other_states={"NA": (st1, "b"), "NB": (st2, "c")},
    )

    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NB.a", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [30, 50]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st3.groups_stack_final == [[0, 1]]
    assert st3.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NB.a": 30},
        {"NA.a": 3, "NB.a": 50},
        {"NA.a": 5, "NB.a": 30},
        {"NA.a": 5, "NB.a": 50},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.b": 0, "NC.c": 0},
        {"NC.b": 0, "NC.c": 1},
        {"NC.b": 1, "NC.c": 0},
        {"NC.b": 1, "NC.c": 1},
    ]


def test_state_connect_3b():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs
        the third state connected to two previous states;
        the third state has explicit splitter that contains splitter only from the first state.
        splitter from the second state has to be added (partial prev-state part)
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC", splitter="_NB", other_states={"NA": (st1, "b"), "NB": (st2, "c")}
    )

    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NB.a", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [30, 50]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.a": 1}
    assert st3.groups_stack_final == [[0, 1]]
    assert st3.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NB.a": 30},
        {"NA.a": 3, "NB.a": 50},
        {"NA.a": 5, "NB.a": 30},
        {"NA.a": 5, "NB.a": 50},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.b": 0, "NC.c": 0},
        {"NC.b": 0, "NC.c": 1},
        {"NC.b": 1, "NC.c": 0},
        {"NC.b": 1, "NC.c": 1},
    ]


def test_state_connect_4():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs
        the third state connected to two previous states;
        the third state has explicit scalar(!) splitter that contains two previous states
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC",
        splitter=("_NA", "_NB"),
        other_states={"NA": (st1, "b"), "NB": (st2, "c")},
    )

    assert st3.splitter == ("_NA", "_NB")
    assert st3.splitter_rpn == ["NA.a", "NB.a", "."]
    assert st3.prev_state_splitter == st3.splitter
    assert st3.prev_state_splitter_rpn == st3.splitter_rpn
    assert st3.prev_state_splitter_rpn_compact == ["_NA", "_NB", "."]
    assert st3.current_splitter is None
    assert st3.current_splitter_rpn == []

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [30, 50]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.a": 0}
    assert st3.groups_stack_final == [[0]]
    assert st3.states_ind == [{"NA.a": 0, "NB.a": 0}, {"NA.a": 1, "NB.a": 1}]
    assert st3.states_val == [{"NA.a": 3, "NB.a": 30}, {"NA.a": 5, "NB.a": 50}]

    st3.prepare_inputs()
    assert st3.inputs_ind == [{"NC.b": 0, "NC.c": 0}, {"NC.b": 1, "NC.c": 1}]


def test_state_connect_5():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs,
        the first state has outer splitter,
        the second state has no explicit splitter
    """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", other_states={"NA": (st1, "a")})
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a", "NA.b", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20]})
    assert st2.group_for_inputs_final == {"NA.a": 0, "NA.b": 1}
    assert st2.groups_stack_final == [[0, 1]]
    assert st2.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.a": 0}, {"NB.a": 1}, {"NB.a": 2}, {"NB.a": 3}]


def test_state_connect_6():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs,
        the first state has outer splitter,
        the third state has explicit splitter with splitters from previous states
     """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC",
        splitter=["_NA", "_NB"],
        other_states={"NA": (st1, "a"), "NB": (st2, "b")},
    )
    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NA.b", "*", "NB.a", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.a": [600, 700]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NA.b": 1, "NB.a": 2}
    assert st3.groups_stack_final == [[0, 1, 2]]

    assert st3.states_ind == [
        {"NA.a": 0, "NA.b": 0, "NB.a": 0},
        {"NA.a": 0, "NA.b": 0, "NB.a": 1},
        {"NA.a": 0, "NA.b": 1, "NB.a": 0},
        {"NA.a": 0, "NA.b": 1, "NB.a": 1},
        {"NA.a": 1, "NA.b": 0, "NB.a": 0},
        {"NA.a": 1, "NA.b": 0, "NB.a": 1},
        {"NA.a": 1, "NA.b": 1, "NB.a": 0},
        {"NA.a": 1, "NA.b": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NA.b": 10, "NB.a": 600},
        {"NA.a": 3, "NA.b": 10, "NB.a": 700},
        {"NA.a": 3, "NA.b": 20, "NB.a": 600},
        {"NA.a": 3, "NA.b": 20, "NB.a": 700},
        {"NA.a": 5, "NA.b": 10, "NB.a": 600},
        {"NA.a": 5, "NA.b": 10, "NB.a": 700},
        {"NA.a": 5, "NA.b": 20, "NB.a": 600},
        {"NA.a": 5, "NA.b": 20, "NB.a": 700},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.a": 0, "NC.b": 0},
        {"NC.a": 0, "NC.b": 1},
        {"NC.a": 1, "NC.b": 0},
        {"NC.a": 1, "NC.b": 1},
        {"NC.a": 2, "NC.b": 0},
        {"NC.a": 2, "NC.b": 1},
        {"NC.a": 3, "NC.b": 0},
        {"NC.a": 3, "NC.b": 1},
    ]


def test_state_connect_6a():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs,
        the first state has outer splitter,
        the third state has no explicit splitter
    """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", splitter="a")
    st3 = State(name="NC", other_states={"NA": (st1, "a"), "NB": (st2, "b")})
    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NA.b", "*", "NB.a", "*"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.a": [600, 700]})
    assert st3.group_for_inputs_final == {"NA.a": 0, "NA.b": 1, "NB.a": 2}
    assert st3.groups_stack_final == [[0, 1, 2]]
    assert st3.states_ind == [
        {"NA.a": 0, "NA.b": 0, "NB.a": 0},
        {"NA.a": 0, "NA.b": 0, "NB.a": 1},
        {"NA.a": 0, "NA.b": 1, "NB.a": 0},
        {"NA.a": 0, "NA.b": 1, "NB.a": 1},
        {"NA.a": 1, "NA.b": 0, "NB.a": 0},
        {"NA.a": 1, "NA.b": 0, "NB.a": 1},
        {"NA.a": 1, "NA.b": 1, "NB.a": 0},
        {"NA.a": 1, "NA.b": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NA.b": 10, "NB.a": 600},
        {"NA.a": 3, "NA.b": 10, "NB.a": 700},
        {"NA.a": 3, "NA.b": 20, "NB.a": 600},
        {"NA.a": 3, "NA.b": 20, "NB.a": 700},
        {"NA.a": 5, "NA.b": 10, "NB.a": 600},
        {"NA.a": 5, "NA.b": 10, "NB.a": 700},
        {"NA.a": 5, "NA.b": 20, "NB.a": 600},
        {"NA.a": 5, "NA.b": 20, "NB.a": 700},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.a": 0, "NC.b": 0},
        {"NC.a": 0, "NC.b": 1},
        {"NC.a": 1, "NC.b": 0},
        {"NC.a": 1, "NC.b": 1},
        {"NC.a": 2, "NC.b": 0},
        {"NC.a": 2, "NC.b": 1},
        {"NC.a": 3, "NC.b": 0},
        {"NC.a": 3, "NC.b": 1},
    ]


def test_state_connect_innerspl_1():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs,
        the second state has an inner splitter, full splitter provided
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["_NA", "b"], other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", "NB.b"]
    assert st2.splitter_rpn == ["NA.a", "NB.b", "*"]
    assert st2.prev_state_splitter == "_NA"
    assert st2.prev_state_splitter_rpn == ["NA.a"]
    assert st2.prev_state_splitter_rpn_compact == ["_NA"]
    assert st2.current_splitter == "NB.b"
    assert st2.current_splitter_rpn == ["NB.b"]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.other_states["NA"][1] == "b"
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.b": 1}
    assert st2.groups_stack_final == [[0], [1]]

    assert st2.states_ind == [
        {"NA.a": 0, "NB.b": 0},
        {"NA.a": 0, "NB.b": 1},
        {"NA.a": 0, "NB.b": 2},
        {"NA.a": 1, "NB.b": 3},
        {"NA.a": 1, "NB.b": 4},
        {"NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NB.b": 1},
        {"NA.a": 3, "NB.b": 10},
        {"NA.a": 3, "NB.b": 100},
        {"NA.a": 5, "NB.b": 2},
        {"NA.a": 5, "NB.b": 20},
        {"NA.a": 5, "NB.b": 200},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0},
        {"NB.b": 1},
        {"NB.b": 2},
        {"NB.b": 3},
        {"NB.b": 4},
        {"NB.b": 5},
    ]


def test_state_connect_innerspl_1a():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs,
        the second state has an inner splitter,
        splitter from the first state (the prev-state part) has to be added
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="b", other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", "NB.b"]
    assert st2.splitter_rpn == ["NA.a", "NB.b", "*"]
    assert st2.prev_state_splitter == "_NA"
    assert st2.prev_state_splitter_rpn == ["NA.a"]
    assert st2.prev_state_splitter_rpn_compact == ["_NA"]
    assert st2.current_splitter == "NB.b"
    assert st2.current_splitter_rpn == ["NB.b"]

    assert st2.other_states["NA"][1] == "b"

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.b": 1}
    assert st2.groups_stack_final == [[0], [1]]

    assert st2.states_ind == [
        {"NA.a": 0, "NB.b": 0},
        {"NA.a": 0, "NB.b": 1},
        {"NA.a": 0, "NB.b": 2},
        {"NA.a": 1, "NB.b": 3},
        {"NA.a": 1, "NB.b": 4},
        {"NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NB.b": 1},
        {"NA.a": 3, "NB.b": 10},
        {"NA.a": 3, "NB.b": 100},
        {"NA.a": 5, "NB.b": 2},
        {"NA.a": 5, "NB.b": 20},
        {"NA.a": 5, "NB.b": 200},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0},
        {"NB.b": 1},
        {"NB.b": 2},
        {"NB.b": 3},
        {"NB.b": 4},
        {"NB.b": 5},
    ]


def test_state_connect_innerspl_1b():
    """incorrect splitter - the current & prev-state parts in scalar splitter"""
    with pytest.raises(PydraStateError):
        st1 = State(name="NA", splitter="a")
        st2 = State(name="NB", splitter=("_NA", "b"), other_states={"NA": (st1, "b")})


def test_state_connect_innerspl_2():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs,
        the second state has one inner splitter and one 'normal' splitter
        only the current part of the splitter provided (the prev-state has to be added)
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["c", "b"], other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", ["NB.c", "NB.b"]]
    assert st2.splitter_rpn == ["NA.a", "NB.c", "NB.b", "*", "*"]
    assert st2.prev_state_splitter == "_NA"
    assert st2.prev_state_splitter_rpn == ["NA.a"]
    assert st2.prev_state_splitter_rpn_compact == ["_NA"]
    assert st2.current_splitter == ["NB.c", "NB.b"]
    assert st2.current_splitter_rpn == ["NB.c", "NB.b", "*"]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]], "NB.c": [13, 17]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.other_states["NA"][1] == "b"
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.c": 1, "NB.b": 2}
    assert st2.groups_stack_final == [[0], [1, 2]]

    assert st2.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.b": 0},
        {"NB.c": 0, "NB.b": 1},
        {"NB.c": 0, "NB.b": 2},
        {"NB.c": 0, "NB.b": 3},
        {"NB.c": 0, "NB.b": 4},
        {"NB.c": 0, "NB.b": 5},
        {"NB.c": 1, "NB.b": 0},
        {"NB.c": 1, "NB.b": 1},
        {"NB.c": 1, "NB.b": 2},
        {"NB.c": 1, "NB.b": 3},
        {"NB.c": 1, "NB.b": 4},
        {"NB.c": 1, "NB.b": 5},
    ]


def test_state_connect_innerspl_2a():
    """ two 'connected' states: testing groups, prepare_states and prepare_inputs,
        the second state has one inner splitter and one 'normal' splitter
        only the current part of the splitter provided (different order!),

    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["b", "c"], other_states={"NA": (st1, "b")})

    assert st2.splitter == ["_NA", ["NB.b", "NB.c"]]
    assert st2.splitter_rpn == ["NA.a", "NB.b", "NB.c", "*", "*"]
    assert st2.other_states["NA"][1] == "b"

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]], "NB.c": [13, 17]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.c": 2, "NB.b": 1}
    assert st2.groups_stack_final == [[0], [1, 2]]

    assert st2.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5},
    ]

    assert st2.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.b": 0},
        {"NB.c": 1, "NB.b": 0},
        {"NB.c": 0, "NB.b": 1},
        {"NB.c": 1, "NB.b": 1},
        {"NB.c": 0, "NB.b": 2},
        {"NB.c": 1, "NB.b": 2},
        {"NB.c": 0, "NB.b": 3},
        {"NB.c": 1, "NB.b": 3},
        {"NB.c": 0, "NB.b": 4},
        {"NB.c": 1, "NB.b": 4},
        {"NB.c": 0, "NB.b": 5},
        {"NB.c": 1, "NB.b": 5},
    ]


def test_state_connect_innerspl_3():
    """ three serially 'connected' states: testing groups, prepare_states and prepare_inputs,
        the second state has one inner splitter and one 'normal' splitter
        the prev-state parts of the splitter have to be added
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["c", "b"], other_states={"NA": (st1, "b")})
    st3 = State(name="NC", splitter="d", other_states={"NB": (st2, "a")})

    assert st3.splitter == ["_NB", "NC.d"]
    assert st3.splitter_rpn == ["NA.a", "NB.c", "NB.b", "*", "*", "NC.d", "*"]
    assert st3.prev_state_splitter == "_NB"
    assert st3.prev_state_splitter_rpn == ["NA.a", "NB.c", "NB.b", "*", "*"]
    assert st3.prev_state_splitter_rpn_compact == ["_NB"]
    assert st3.current_splitter == "NC.d"
    assert st3.current_splitter_rpn == ["NC.d"]

    st3.prepare_states(
        inputs={
            "NA.a": [3, 5],
            "NB.b": [[1, 10, 100], [2, 20, 200]],
            "NB.c": [13, 17],
            "NC.d": [33, 77],
        },
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.c": 1, "NB.b": 2, "NC.d": 3}
    assert st3.groups_stack_final == [[0], [1, 2, 3]]

    assert st2.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200},
    ]

    assert st3.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0, "NC.d": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 0, "NC.d": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1, "NC.d": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1, "NC.d": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2, "NC.d": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2, "NC.d": 1},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3, "NC.d": 0},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3, "NC.d": 1},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4, "NC.d": 0},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4, "NC.d": 1},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5, "NC.d": 0},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5, "NC.d": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0, "NC.d": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0, "NC.d": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1, "NC.d": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1, "NC.d": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2, "NC.d": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2, "NC.d": 1},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3, "NC.d": 0},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3, "NC.d": 1},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4, "NC.d": 0},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4, "NC.d": 1},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5, "NC.d": 0},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5, "NC.d": 1},
    ]
    assert st3.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1, "NC.d": 33},
        {"NB.c": 13, "NA.a": 3, "NB.b": 1, "NC.d": 77},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10, "NC.d": 33},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10, "NC.d": 77},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100, "NC.d": 33},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100, "NC.d": 77},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2, "NC.d": 33},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2, "NC.d": 77},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20, "NC.d": 33},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20, "NC.d": 77},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200, "NC.d": 33},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200, "NC.d": 77},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1, "NC.d": 33},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1, "NC.d": 77},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10, "NC.d": 33},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10, "NC.d": 77},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100, "NC.d": 33},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100, "NC.d": 77},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2, "NC.d": 33},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2, "NC.d": 77},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20, "NC.d": 33},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20, "NC.d": 77},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200, "NC.d": 33},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200, "NC.d": 77},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.a": 0, "NC.d": 0},
        {"NC.a": 0, "NC.d": 1},
        {"NC.a": 1, "NC.d": 0},
        {"NC.a": 1, "NC.d": 1},
        {"NC.a": 2, "NC.d": 0},
        {"NC.a": 2, "NC.d": 1},
        {"NC.a": 3, "NC.d": 0},
        {"NC.a": 3, "NC.d": 1},
        {"NC.a": 4, "NC.d": 0},
        {"NC.a": 4, "NC.d": 1},
        {"NC.a": 5, "NC.d": 0},
        {"NC.a": 5, "NC.d": 1},
        {"NC.a": 6, "NC.d": 0},
        {"NC.a": 6, "NC.d": 1},
        {"NC.a": 7, "NC.d": 0},
        {"NC.a": 7, "NC.d": 1},
        {"NC.a": 8, "NC.d": 0},
        {"NC.a": 8, "NC.d": 1},
        {"NC.a": 9, "NC.d": 0},
        {"NC.a": 9, "NC.d": 1},
        {"NC.a": 10, "NC.d": 0},
        {"NC.a": 10, "NC.d": 1},
        {"NC.a": 11, "NC.d": 0},
        {"NC.a": 11, "NC.d": 1},
    ]


def test_state_connect_innerspl_4():
    """ three'connected' states: testing groups, prepare_states and prepare_inputs,
        the third one connected to two previous, only the current part of splitter provided
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter=["b", "c"])
    st3 = State(
        name="NC", splitter="d", other_states={"NA": (st1, "e"), "NB": (st2, "f")}
    )

    assert st3.splitter == [["_NA", "_NB"], "NC.d"]
    assert st3.splitter_rpn == ["NA.a", "NB.b", "NB.c", "*", "*", "NC.d", "*"]
    assert st3.other_states["NA"][1] == "e"
    assert st3.other_states["NB"][1] == "f"

    st3.prepare_states(
        inputs={
            "NA.a": [3, 5],
            "NB.b": [10, 20],
            "NB.c": [13, 17],
            "NC.e": [30, 50],
            "NC.f": [[23, 27], [33, 37]],
            "NC.d": [1, 2],
        },
        cont_dim={"NC.f": 2},  # will be treated as 2d container
    )
    assert st3.group_for_inputs_final == {"NA.a": 0, "NB.c": 2, "NB.b": 1, "NC.d": 3}
    assert st3.groups_stack_final == [[0, 1, 2, 3]]
    assert st3.states_ind == [
        {"NA.a": 0, "NB.b": 0, "NB.c": 0, "NC.d": 0},
        {"NA.a": 0, "NB.b": 0, "NB.c": 0, "NC.d": 1},
        {"NA.a": 0, "NB.b": 0, "NB.c": 1, "NC.d": 0},
        {"NA.a": 0, "NB.b": 0, "NB.c": 1, "NC.d": 1},
        {"NA.a": 0, "NB.b": 1, "NB.c": 0, "NC.d": 0},
        {"NA.a": 0, "NB.b": 1, "NB.c": 0, "NC.d": 1},
        {"NA.a": 0, "NB.b": 1, "NB.c": 1, "NC.d": 0},
        {"NA.a": 0, "NB.b": 1, "NB.c": 1, "NC.d": 1},
        {"NA.a": 1, "NB.b": 0, "NB.c": 0, "NC.d": 0},
        {"NA.a": 1, "NB.b": 0, "NB.c": 0, "NC.d": 1},
        {"NA.a": 1, "NB.b": 0, "NB.c": 1, "NC.d": 0},
        {"NA.a": 1, "NB.b": 0, "NB.c": 1, "NC.d": 1},
        {"NA.a": 1, "NB.b": 1, "NB.c": 0, "NC.d": 0},
        {"NA.a": 1, "NB.b": 1, "NB.c": 0, "NC.d": 1},
        {"NA.a": 1, "NB.b": 1, "NB.c": 1, "NC.d": 0},
        {"NA.a": 1, "NB.b": 1, "NB.c": 1, "NC.d": 1},
    ]

    assert st3.states_val == [
        {"NA.a": 3, "NB.b": 10, "NB.c": 13, "NC.d": 1},
        {"NA.a": 3, "NB.b": 10, "NB.c": 13, "NC.d": 2},
        {"NA.a": 3, "NB.b": 10, "NB.c": 17, "NC.d": 1},
        {"NA.a": 3, "NB.b": 10, "NB.c": 17, "NC.d": 2},
        {"NA.a": 3, "NB.b": 20, "NB.c": 13, "NC.d": 1},
        {"NA.a": 3, "NB.b": 20, "NB.c": 13, "NC.d": 2},
        {"NA.a": 3, "NB.b": 20, "NB.c": 17, "NC.d": 1},
        {"NA.a": 3, "NB.b": 20, "NB.c": 17, "NC.d": 2},
        {"NA.a": 5, "NB.b": 10, "NB.c": 13, "NC.d": 1},
        {"NA.a": 5, "NB.b": 10, "NB.c": 13, "NC.d": 2},
        {"NA.a": 5, "NB.b": 10, "NB.c": 17, "NC.d": 1},
        {"NA.a": 5, "NB.b": 10, "NB.c": 17, "NC.d": 2},
        {"NA.a": 5, "NB.b": 20, "NB.c": 13, "NC.d": 1},
        {"NA.a": 5, "NB.b": 20, "NB.c": 13, "NC.d": 2},
        {"NA.a": 5, "NB.b": 20, "NB.c": 17, "NC.d": 1},
        {"NA.a": 5, "NB.b": 20, "NB.c": 17, "NC.d": 2},
    ]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.e": 0, "NC.f": 0, "NC.d": 0},
        {"NC.e": 0, "NC.f": 0, "NC.d": 1},
        {"NC.e": 0, "NC.f": 1, "NC.d": 0},
        {"NC.e": 0, "NC.f": 1, "NC.d": 1},
        {"NC.e": 0, "NC.f": 2, "NC.d": 0},
        {"NC.e": 0, "NC.f": 2, "NC.d": 1},
        {"NC.e": 0, "NC.f": 3, "NC.d": 0},
        {"NC.e": 0, "NC.f": 3, "NC.d": 1},
        {"NC.e": 1, "NC.f": 0, "NC.d": 0},
        {"NC.e": 1, "NC.f": 0, "NC.d": 1},
        {"NC.e": 1, "NC.f": 1, "NC.d": 0},
        {"NC.e": 1, "NC.f": 1, "NC.d": 1},
        {"NC.e": 1, "NC.f": 2, "NC.d": 0},
        {"NC.e": 1, "NC.f": 2, "NC.d": 1},
        {"NC.e": 1, "NC.f": 3, "NC.d": 0},
        {"NC.e": 1, "NC.f": 3, "NC.d": 1},
    ]


def test_state_combine_1():
    """ single state with splitter and combiner"""
    st = State(name="NA", splitter="a", combiner="a")
    assert st.splitter == "NA.a"
    assert st.splitter_rpn == ["NA.a"]
    assert st.current_combiner == st.current_combiner_all == st.combiner == ["NA.a"]
    assert st.prev_state_combiner == st.prev_state_combiner_all == []
    assert st.splitter_final == None
    assert st.splitter_rpn_final == []

    st.prepare_states(inputs={"NA.a": [3, 5]})
    assert st.group_for_inputs_final == {}
    assert st.groups_stack_final == []
    assert st.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert st.states_val == [{"NA.a": 3}, {"NA.a": 5}]

    st.prepare_inputs()
    assert st.inputs_ind == [{"NA.a": 0}, {"NA.a": 1}]


def test_state_connect_combine_1():
    """two connected states; outer splitter and combiner in the first one"""
    st1 = State(name="NA", splitter=["a", "b"], combiner="a")
    st2 = State(name="NB", other_states={"NA": (st1, "c")})

    assert st1.splitter == ["NA.a", "NA.b"]
    assert st1.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st1.splitter_rpn_final == ["NA.b"]
    assert st1.splitter_final == "NA.b"
    assert st1.combiner == st1.current_combiner == st1.current_combiner_all == ["NA.a"]
    assert st1.prev_state_combiner_all == st1.prev_state_combiner == []
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.b"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20]})
    assert st2.group_for_inputs_final == {"NA.b": 0}
    assert st2.groups_stack_final == [[0]]
    assert st1.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st1.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]
    assert st1.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]
    assert st1.keys_final == ["NA.b"]
    assert st1.final_combined_ind_mapping == {0: [0, 2], 1: [1, 3]}

    assert st2.states_ind == [{"NA.b": 0}, {"NA.b": 1}]
    assert st2.states_val == [{"NA.b": 10}, {"NA.b": 20}]
    assert st2.keys_final == ["NA.b"]
    assert st2.final_combined_ind_mapping == {0: [0], 1: [1]}

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.c": 0}, {"NB.c": 1}]


def test_state_connect_combine_2():
    """
    two connected states; outer splitter and combiner in the first one;
    additional splitter in the second node
    """
    st1 = State(name="NA", splitter=["a", "b"], combiner="a")
    st2 = State(name="NB", splitter="d", other_states={"NA": (st1, "c")})

    assert st1.splitter == ["NA.a", "NA.b"]
    assert st1.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st1.combiner == ["NA.a"]
    assert st1.splitter_final == "NA.b"
    assert st1.splitter_rpn_final == ["NA.b"]

    assert st2.splitter == ["_NA", "NB.d"]
    assert st2.splitter_rpn == ["NA.b", "NB.d", "*"]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.c": [90, 150], "NB.d": [0, 1]}
    )
    assert st2.group_for_inputs_final == {"NA.b": 0, "NB.d": 1}
    assert st2.groups_stack_final == [[0, 1]]
    assert st1.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st1.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]
    assert st1.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]
    assert st1.keys_final == ["NA.b"]
    assert st1.final_combined_ind_mapping == {0: [0, 2], 1: [1, 3]}

    assert st2.states_ind == [
        {"NA.b": 0, "NB.d": 0},
        {"NA.b": 0, "NB.d": 1},
        {"NA.b": 1, "NB.d": 0},
        {"NA.b": 1, "NB.d": 1},
    ]
    assert st2.states_val == [
        {"NA.b": 10, "NB.d": 0},
        {"NA.b": 10, "NB.d": 1},
        {"NA.b": 20, "NB.d": 0},
        {"NA.b": 20, "NB.d": 1},
    ]
    assert st2.keys_final == ["NA.b", "NB.d"]
    assert st2.final_combined_ind_mapping == {0: [0], 1: [1], 2: [2], 3: [3]}

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.d": 0},
        {"NB.c": 0, "NB.d": 1},
        {"NB.c": 1, "NB.d": 0},
        {"NB.c": 1, "NB.d": 1},
    ]


def test_state_connect_combine_3():
    """
    two connected states; outer splitter and combiner in the first one;
    additional splitter in the second node
    """
    st1 = State(name="NA", splitter=["a", "b"], combiner="a")
    st2 = State(name="NB", splitter="d", combiner="d", other_states={"NA": (st1, "c")})

    assert st1.splitter == ["NA.a", "NA.b"]
    assert st1.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st1.combiner == ["NA.a"]
    assert st1.splitter_final == "NA.b"
    assert st1.splitter_rpn_final == ["NA.b"]

    assert st2.splitter == ["_NA", "NB.d"]
    assert st2.splitter_rpn == ["NA.b", "NB.d", "*"]
    assert st2.splitter_rpn_final == ["NA.b"]
    assert st2.prev_state_combiner_all == st2.prev_state_combiner == []
    assert st2.current_combiner_all == st2.current_combiner == st2.combiner == ["NB.d"]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.c": [90, 150], "NB.d": [0, 1]}
    )
    assert st2.group_for_inputs_final == {"NA.b": 0}
    assert st2.groups_stack_final == [[0]]

    assert st1.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st1.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]
    assert st1.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]
    assert st1.keys_final == ["NA.b"]
    assert st1.final_combined_ind_mapping == {0: [0, 2], 1: [1, 3]}

    assert st2.states_ind == [
        {"NA.b": 0, "NB.d": 0},
        {"NA.b": 0, "NB.d": 1},
        {"NA.b": 1, "NB.d": 0},
        {"NA.b": 1, "NB.d": 1},
    ]
    assert st2.states_val == [
        {"NA.b": 10, "NB.d": 0},
        {"NA.b": 10, "NB.d": 1},
        {"NA.b": 20, "NB.d": 0},
        {"NA.b": 20, "NB.d": 1},
    ]
    assert st2.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]
    assert st2.keys_final == ["NA.b"]
    assert st2.final_combined_ind_mapping == {0: [0, 1], 1: [2, 3]}

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.d": 0},
        {"NB.c": 0, "NB.d": 1},
        {"NB.c": 1, "NB.d": 0},
        {"NB.c": 1, "NB.d": 1},
    ]


def test_state_connect_innerspl_combine_1():
    """one previous node and one inner splitter (and inner splitter combiner);
    only current part provided - the prev-state part had to be added"""
    st1 = State(name="NA", splitter="a")
    st2 = State(
        name="NB", splitter=["c", "b"], combiner=["b"], other_states={"NA": (st1, "b")}
    )

    assert st2.splitter == ["_NA", ["NB.c", "NB.b"]]
    assert st2.splitter_rpn == ["NA.a", "NB.c", "NB.b", "*", "*"]
    assert st2.splitter_final == ["NA.a", "NB.c"]
    assert st2.splitter_rpn_final == ["NA.a", "NB.c", "*"]
    assert st2.prev_state_combiner_all == st2.prev_state_combiner == []
    assert st2.current_combiner_all == st2.current_combiner == st2.combiner == ["NB.b"]
    # TODO: i think at the end I should merge [0] and [1], because there are no inner splitters anymore
    # TODO: didn't include it in my code...
    # assert st2.groups_stack_final == [[0, 1]]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]], "NB.c": [13, 17]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.c": 1}
    assert st2.groups_stack_final == [[0], [1]]

    # NOW TODO: checking st2.states_ind_final!!!
    assert st2.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200},
    ]
    assert st2.states_ind_final == [
        {"NB.c": 0, "NA.a": 0},
        {"NB.c": 1, "NA.a": 0},
        {"NB.c": 0, "NA.a": 1},
        {"NB.c": 1, "NA.a": 1},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.b": 0},
        {"NB.c": 0, "NB.b": 1},
        {"NB.c": 0, "NB.b": 2},
        {"NB.c": 0, "NB.b": 3},
        {"NB.c": 0, "NB.b": 4},
        {"NB.c": 0, "NB.b": 5},
        {"NB.c": 1, "NB.b": 0},
        {"NB.c": 1, "NB.b": 1},
        {"NB.c": 1, "NB.b": 2},
        {"NB.c": 1, "NB.b": 3},
        {"NB.c": 1, "NB.b": 4},
        {"NB.c": 1, "NB.b": 5},
    ]


def test_state_connect_innerspl_combine_2():
    """ two 'connected' state, the second has inner and normal splitter,
        and 'normal' combiner
        only the current part of the splitter provided,
        the prev-state part has to be added
        """
    st1 = State(name="NA", splitter="a")
    st2 = State(
        name="NB", splitter=["c", "b"], combiner=["c"], other_states={"NA": (st1, "b")}
    )

    assert st2.splitter == ["_NA", ["NB.c", "NB.b"]]
    assert st2.splitter_rpn == ["NA.a", "NB.c", "NB.b", "*", "*"]
    assert st2.splitter_final == ["NA.a", "NB.b"]
    assert st2.splitter_rpn_final == ["NA.a", "NB.b", "*"]

    st2.prepare_states(
        inputs={"NA.a": [3, 5], "NB.b": [[1, 10, 100], [2, 20, 200]], "NB.c": [13, 17]},
        cont_dim={"NB.b": 2},  # will be treated as 2d container
    )
    assert st2.group_for_inputs_final == {"NA.a": 0, "NB.b": 1}
    assert st2.groups_stack_final == [[0], [1]]
    assert st2.states_ind == [
        {"NB.c": 0, "NA.a": 0, "NB.b": 0},
        {"NB.c": 0, "NA.a": 0, "NB.b": 1},
        {"NB.c": 0, "NA.a": 0, "NB.b": 2},
        {"NB.c": 0, "NA.a": 1, "NB.b": 3},
        {"NB.c": 0, "NA.a": 1, "NB.b": 4},
        {"NB.c": 0, "NA.a": 1, "NB.b": 5},
        {"NB.c": 1, "NA.a": 0, "NB.b": 0},
        {"NB.c": 1, "NA.a": 0, "NB.b": 1},
        {"NB.c": 1, "NA.a": 0, "NB.b": 2},
        {"NB.c": 1, "NA.a": 1, "NB.b": 3},
        {"NB.c": 1, "NA.a": 1, "NB.b": 4},
        {"NB.c": 1, "NA.a": 1, "NB.b": 5},
    ]
    assert st2.states_val == [
        {"NB.c": 13, "NA.a": 3, "NB.b": 1},
        {"NB.c": 13, "NA.a": 3, "NB.b": 10},
        {"NB.c": 13, "NA.a": 3, "NB.b": 100},
        {"NB.c": 13, "NA.a": 5, "NB.b": 2},
        {"NB.c": 13, "NA.a": 5, "NB.b": 20},
        {"NB.c": 13, "NA.a": 5, "NB.b": 200},
        {"NB.c": 17, "NA.a": 3, "NB.b": 1},
        {"NB.c": 17, "NA.a": 3, "NB.b": 10},
        {"NB.c": 17, "NA.a": 3, "NB.b": 100},
        {"NB.c": 17, "NA.a": 5, "NB.b": 2},
        {"NB.c": 17, "NA.a": 5, "NB.b": 20},
        {"NB.c": 17, "NA.a": 5, "NB.b": 200},
    ]
    assert st2.states_ind_final == [
        {"NA.a": 0, "NB.b": 0},
        {"NA.a": 0, "NB.b": 1},
        {"NA.a": 0, "NB.b": 2},
        {"NA.a": 1, "NB.b": 3},
        {"NA.a": 1, "NB.b": 4},
        {"NA.a": 1, "NB.b": 5},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.c": 0, "NB.b": 0},
        {"NB.c": 0, "NB.b": 1},
        {"NB.c": 0, "NB.b": 2},
        {"NB.c": 0, "NB.b": 3},
        {"NB.c": 0, "NB.b": 4},
        {"NB.c": 0, "NB.b": 5},
        {"NB.c": 1, "NB.b": 0},
        {"NB.c": 1, "NB.b": 1},
        {"NB.c": 1, "NB.b": 2},
        {"NB.c": 1, "NB.b": 3},
        {"NB.c": 1, "NB.b": 4},
        {"NB.c": 1, "NB.b": 5},
    ]


def test_state_connect_combine_prevst_1():
    """ two 'connected' states,
        the first one has the simplest splitter,
        the second has combiner from the first state
        (i.e. from the prev-state part of the splitter),
    """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", other_states={"NA": (st1, "b")}, combiner="NA.a")
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a"]
    assert (
        st2.combiner
        == st2.prev_state_combiner
        == st2.prev_state_combiner_all
        == ["NA.a"]
    )
    assert st2.current_combiner == st2.current_combiner_all == []
    assert st2.splitter_rpn_final == []

    st2.prepare_states(inputs={"NA.a": [3, 5]})
    assert st2.group_for_inputs_final == {}
    assert st2.groups_stack_final == []
    assert st2.states_ind == [{"NA.a": 0}, {"NA.a": 1}]
    assert st2.states_val == [{"NA.a": 3}, {"NA.a": 5}]
    assert st2.states_ind_final == []

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.b": 0}, {"NB.b": 1}]


def test_state_connect_combine_prevst_2():
    """ two 'connected' states,
        the first one has outer splitter,
        the second has combiner from the first state
        (i.e. from the prev-state part of the splitter),
    """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", other_states={"NA": (st1, "b")}, combiner="NA.a")
    assert st2.splitter == "_NA"
    assert st2.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st2.combiner == ["NA.a"]
    assert st2.prev_state_combiner_all == st2.prev_state_combiner == ["NA.a"]
    assert st2.current_combiner_all == st2.current_combiner == []
    assert st2.splitter_rpn_final == ["NA.b"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20]})
    assert st2.group_for_inputs_final == {"NA.b": 0}
    assert st2.groups_stack_final == [[0]]
    assert st2.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]
    assert st2.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]

    st2.prepare_inputs()
    assert st2.inputs_ind == [{"NB.b": 0}, {"NB.b": 1}, {"NB.b": 2}, {"NB.b": 3}]


def test_state_connect_combine_prevst_3():
    """ three serially 'connected' states,
        the first one has outer splitter,
        the third one has combiner from the first state
        (i.e. from the prev-state part of the splitter),
    """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(name="NB", other_states={"NA": (st1, "b")})
    st3 = State(name="NC", other_states={"NB": (st2, "c")}, combiner="NA.a")
    assert st3.splitter == "_NB"
    assert st3.splitter_rpn == ["NA.a", "NA.b", "*"]
    assert st3.combiner == ["NA.a"]
    assert st3.splitter_rpn_final == ["NA.b"]

    st3.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20]})
    assert st3.group_for_inputs_final == {"NA.b": 0}
    assert st3.groups_stack_final == [[0]]

    assert st3.states_ind == [
        {"NA.a": 0, "NA.b": 0},
        {"NA.a": 0, "NA.b": 1},
        {"NA.a": 1, "NA.b": 0},
        {"NA.a": 1, "NA.b": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NA.b": 10},
        {"NA.a": 3, "NA.b": 20},
        {"NA.a": 5, "NA.b": 10},
        {"NA.a": 5, "NA.b": 20},
    ]
    assert st3.states_ind_final == [{"NA.b": 0}, {"NA.b": 1}]

    st3.prepare_inputs()
    assert st3.inputs_ind == [{"NC.c": 0}, {"NC.c": 1}, {"NC.c": 2}, {"NC.c": 3}]


def test_state_connect_combine_prevst_4():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs,
         the first two states have the simplest splitters,
         the third state has only the prev-state part of splitter,
         the third state has also combiner from the prev-state part
      """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC",
        splitter=["_NA", "_NB"],
        combiner=["NA.a"],
        other_states={"NA": (st1, "a"), "NB": (st2, "b")},
    )
    assert st3.splitter == ["_NA", "_NB"]
    assert st3.splitter_rpn == ["NA.a", "NB.a", "*"]
    assert st3.splitter_rpn_final == ["NB.a"]
    assert (
        st3.prev_state_combiner_all
        == st3.prev_state_combiner
        == st3.combiner
        == ["NA.a"]
    )
    assert st3.current_combiner_all == st3.current_combiner == []

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [600, 700]})
    assert st3.group_for_inputs_final == {"NB.a": 0}
    assert st3.groups_stack_final == [[0]]

    assert st3.states_ind == [
        {"NA.a": 0, "NB.a": 0},
        {"NA.a": 0, "NB.a": 1},
        {"NA.a": 1, "NB.a": 0},
        {"NA.a": 1, "NB.a": 1},
    ]
    assert st3.states_val == [
        {"NA.a": 3, "NB.a": 600},
        {"NA.a": 3, "NB.a": 700},
        {"NA.a": 5, "NB.a": 600},
        {"NA.a": 5, "NB.a": 700},
    ]
    assert st3.states_ind_final == [{"NB.a": 0}, {"NB.a": 1}]

    st3.prepare_inputs()
    assert st3.inputs_ind == [
        {"NC.a": 0, "NC.b": 0},
        {"NC.a": 0, "NC.b": 1},
        {"NC.a": 1, "NC.b": 0},
        {"NC.a": 1, "NC.b": 1},
    ]


def test_state_connect_combine_prevst_5():
    """ three 'connected' states: testing groups, prepare_states and prepare_inputs,
        the first two states have the simplest splitters,
        the third state has scalar splitter in the prev-state part,
        the third state has also combiner from the prev-state part
     """
    st1 = State(name="NA", splitter="a")
    st2 = State(name="NB", splitter="a")
    st3 = State(
        name="NC",
        splitter=("_NA", "_NB"),
        combiner=["NA.a"],
        other_states={"NA": (st1, "a"), "NB": (st2, "b")},
    )
    assert st3.splitter == ("_NA", "_NB")
    assert st3.splitter_rpn == ["NA.a", "NB.a", "."]
    st3.set_input_groups()
    assert st3.splitter_rpn_final == []
    assert set(st3.prev_state_combiner_all) == {"NA.a", "NB.a"}
    assert st3.prev_state_combiner == st3.combiner == ["NA.a"]
    assert st3.current_combiner_all == st3.current_combiner == []

    st3.prepare_states(inputs={"NA.a": [3, 5], "NB.a": [600, 700]})
    assert st3.group_for_inputs_final == {}
    assert st3.groups_stack_final == []

    assert st3.states_ind == [{"NA.a": 0, "NB.a": 0}, {"NA.a": 1, "NB.a": 1}]
    assert st3.states_val == [{"NA.a": 3, "NB.a": 600}, {"NA.a": 5, "NB.a": 700}]
    assert st3.states_ind_final == []

    st3.prepare_inputs()
    assert st3.inputs_ind == [{"NC.a": 0, "NC.b": 0}, {"NC.a": 1, "NC.b": 1}]


def test_state_connect_combine_prevst_6():
    """ two 'connected' states,
        the first one has outer splitter, the second has an additional current splitter,
        the second also has combiner from the first state
        (i.e. from the prev-state part of the splitter),
    """
    st1 = State(name="NA", splitter=["a", "b"])
    st2 = State(
        name="NB", splitter="c", other_states={"NA": (st1, "b")}, combiner="NA.a"
    )
    assert st2.splitter == ["_NA", "NB.c"]
    assert st2.splitter_rpn == ["NA.a", "NA.b", "*", "NB.c", "*"]
    assert st2.combiner == ["NA.a"]
    assert st2.prev_state_combiner_all == st2.prev_state_combiner == ["NA.a"]
    assert st2.current_combiner_all == st2.current_combiner == []
    assert st2.splitter_rpn_final == ["NA.b", "NB.c", "*"]

    st2.prepare_states(inputs={"NA.a": [3, 5], "NA.b": [10, 20], "NB.c": [0, 1]})
    assert st2.group_for_inputs_final == {"NA.b": 0, "NB.c": 1}
    assert st2.groups_stack_final == [[0, 1]]

    assert st2.states_ind == [
        {"NA.a": 0, "NA.b": 0, "NB.c": 0},
        {"NA.a": 0, "NA.b": 0, "NB.c": 1},
        {"NA.a": 0, "NA.b": 1, "NB.c": 0},
        {"NA.a": 0, "NA.b": 1, "NB.c": 1},
        {"NA.a": 1, "NA.b": 0, "NB.c": 0},
        {"NA.a": 1, "NA.b": 0, "NB.c": 1},
        {"NA.a": 1, "NA.b": 1, "NB.c": 0},
        {"NA.a": 1, "NA.b": 1, "NB.c": 1},
    ]
    assert st2.states_val == [
        {"NA.a": 3, "NA.b": 10, "NB.c": 0},
        {"NA.a": 3, "NA.b": 10, "NB.c": 1},
        {"NA.a": 3, "NA.b": 20, "NB.c": 0},
        {"NA.a": 3, "NA.b": 20, "NB.c": 1},
        {"NA.a": 5, "NA.b": 10, "NB.c": 0},
        {"NA.a": 5, "NA.b": 10, "NB.c": 1},
        {"NA.a": 5, "NA.b": 20, "NB.c": 0},
        {"NA.a": 5, "NA.b": 20, "NB.c": 1},
    ]
    assert st2.states_ind_final == [
        {"NA.b": 0, "NB.c": 0},
        {"NA.b": 0, "NB.c": 1},
        {"NA.b": 1, "NB.c": 0},
        {"NA.b": 1, "NB.c": 1},
    ]

    st2.prepare_inputs()
    assert st2.inputs_ind == [
        {"NB.b": 0, "NB.c": 0},
        {"NB.b": 0, "NB.c": 1},
        {"NB.b": 1, "NB.c": 0},
        {"NB.b": 1, "NB.c": 1},
        {"NB.b": 2, "NB.c": 0},
        {"NB.b": 2, "NB.c": 1},
        {"NB.b": 3, "NB.c": 0},
        {"NB.b": 3, "NB.c": 1},
    ]


@pytest.mark.parametrize(
    "splitter, other_states, expected_splitter, expected_prevst, expected_current",
    [
        (None, {"NA": (State(name="NA", splitter="a"), "b")}, "_NA", "_NA", None),
        (
            "b",
            {"NA": (State(name="NA", splitter="a"), "b")},
            ["_NA", "CN.b"],
            "_NA",
            "CN.b",
        ),
        (
            ("b", "c"),
            {"NA": (State(name="NA", splitter="a"), "b")},
            ["_NA", ("CN.b", "CN.c")],
            "_NA",
            ("CN.b", "CN.c"),
        ),
        (
            None,
            {
                "NA": (State(name="NA", splitter="a"), "a"),
                "NB": (State(name="NB", splitter="a"), "b"),
            },
            ["_NA", "_NB"],
            ["_NA", "_NB"],
            None,
        ),
        (
            "b",
            {
                "NA": (State(name="NA", splitter="a"), "a"),
                "NB": (State(name="NB", splitter="a"), "b"),
            },
            [["_NA", "_NB"], "CN.b"],
            ["_NA", "_NB"],
            "CN.b",
        ),
        (
            ["_NA", "b"],
            {
                "NA": (State(name="NA", splitter="a"), "a"),
                "NB": (State(name="NB", splitter="a"), "b"),
            },
            [["_NB", "_NA"], "CN.b"],
            ["_NB", "_NA"],
            "CN.b",
        ),
    ],
)
def test_connect_splitters(
    splitter, other_states, expected_splitter, expected_prevst, expected_current
):
    st = State(name="CN", splitter=splitter, other_states=other_states)
    st.set_input_groups()
    assert st.splitter == expected_splitter
    assert st.prev_state_splitter == expected_prevst
    assert st.current_splitter == expected_current


@pytest.mark.parametrize(
    "splitter, other_states",
    [
        (("_NA", "b"), {"NA": (State(name="NA", splitter="a"), "b")}),
        (["b", "_NA"], {"NA": (State(name="NA", splitter="a"), "b")}),
        (
            ["_NB", ["_NA", "b"]],
            {
                "NA": (State(name="NA", splitter="a"), "a"),
                "NB": (State(name="NB", splitter="a"), "b"),
            },
        ),
    ],
)
def test_connect_splitters_exception_1(splitter, other_states):
    with pytest.raises(PydraStateError) as excinfo:
        st = State(name="CN", splitter=splitter, other_states=other_states)
    assert "prev-state and current splitters are mixed" in str(excinfo.value)


def test_connect_splitters_exception_2():
    st = State(
        name="CN",
        splitter="_NB",
        other_states={"NA": (State(name="NA", splitter="a"), "b")},
    )
    with pytest.raises(PydraStateError) as excinfo:
        st.set_input_groups()
    assert "can't ask for splitter from NB" in str(excinfo.value)


def test_connect_splitters_exception_3():
    with pytest.raises(PydraStateError) as excinfo:
        st = State(
            name="CN",
            splitter="_NB",
            other_states=["NA", (State(name="NA", splitter="a"), "b")],
        )
    assert "other states has to be a dictionary" == str(excinfo.value)
