from .. import helpers_state as hlpst

import pytest


# TODO: feature?
class other_states_to_tests:
    def __init__(
        self,
        splitter,
        splitter_final=None,
        keys_final=None,
        ind_l=None,
        ind_l_final=None,
    ):
        self.splitter = splitter
        if splitter_final:
            self.splitter_final = splitter_final
        else:
            self.splitter_final = splitter
        self.other_states = {}
        self.keys_final = keys_final
        self.name = "NA"
        self.ind_l = ind_l
        if ind_l_final:
            self.ind_l_final = ind_l_final
        else:
            self.ind_l_final = ind_l


@pytest.mark.parametrize(
    "splitter, keys_exp, groups_exp, grstack_exp",
    [
        ("a", ["a"], {"a": 0}, [[0]]),
        (["a"], ["a"], {"a": 0}, [[0]]),
        (("a",), ["a"], {"a": 0}, [[0]]),
        (("a", "b"), ["a", "b"], {"a": 0, "b": 0}, [[0]]),
        (["a", "b"], ["a", "b"], {"a": 0, "b": 1}, [[0, 1]]),
        ([["a", "b"]], ["a", "b"], {"a": 0, "b": 1}, [[0, 1]]),
        ((["a", "b"],), ["a", "b"], {"a": 0, "b": 1}, [[0, 1]]),
        ((["a", "b"], "c"), ["a", "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]]),
        ([("a", "b"), "c"], ["a", "b", "c"], {"a": 0, "b": 0, "c": 1}, [[0, 1]]),
        ([["a", "b"], "c"], ["a", "b", "c"], {"a": 0, "b": 1, "c": 2}, [[0, 1, 2]]),
        (
            (["a", "b"], ["c", "d"]),
            ["a", "b", "c", "d"],
            {"a": 0, "b": 1, "c": 0, "d": 1},
            [[0, 1]],
        ),
    ],
)
def test_splits_groups(splitter, keys_exp, groups_exp, grstack_exp):
    splitter_rpn = hlpst.splitter2rpn(splitter)
    keys_f, groups_f, grstack_f, _ = hlpst.splits_groups(splitter_rpn)

    assert set(keys_f) == set(keys_exp)
    assert groups_f == groups_exp
    assert grstack_f == grstack_exp


@pytest.mark.parametrize(
    "splitter, combiner, combiner_all_exp,"
    "keys_final_exp, groups_final_exp, grstack_final_exp",
    [
        ("a", ["a"], ["a"], [], {}, []),
        (["a"], ["a"], ["a"], [], {}, []),
        (("a",), ["a"], ["a"], [], {}, []),
        (("a", "b"), ["a"], ["a", "b"], [], {}, [[]]),
        (("a", "b"), ["b"], ["a", "b"], [], {}, [[]]),
        (["a", "b"], ["b"], ["b"], ["a"], {"a": 0}, [[0]]),
        (["a", "b"], ["a"], ["a"], ["b"], {"b": 0}, [[0]]),
        ((["a", "b"], "c"), ["a"], ["a", "c"], ["b"], {"b": 0}, [[0]]),
        ((["a", "b"], "c"), ["b"], ["b", "c"], ["a"], {"a": 0}, [[0]]),
        ((["a", "b"], "c"), ["a"], ["a", "c"], ["b"], {"b": 0}, [[0]]),
        ((["a", "b"], "c"), ["c"], ["a", "b", "c"], [], {}, [[]]),
        ([("a", "b"), "c"], ["a"], ["a", "b"], ["c"], {"c": 0}, [[0]]),
        ([("a", "b"), "c"], ["b"], ["a", "b"], ["c"], {"c": 0}, [[0]]),
        ([("a", "b"), "c"], ["c"], ["c"], ["a", "b"], {"a": 0, "b": 0}, [[0]]),
        ([[("a", "b"), "c"]], ["c"], ["c"], ["a", "b"], {"a": 0, "b": 0}, [[0]]),
        (([("a", "b"), "c"],), ["c"], ["c"], ["a", "b"], {"a": 0, "b": 0}, [[0]]),
    ],
)
def test_splits_groups_comb(
    splitter,
    combiner,
    keys_final_exp,
    groups_final_exp,
    grstack_final_exp,
    combiner_all_exp,
):
    splitter_rpn = hlpst.splitter2rpn(splitter)
    keys_final, groups_final, grstack_final, combiner_all = hlpst.splits_groups(
        splitter_rpn, combiner
    )
    assert keys_final == keys_final_exp
    assert groups_final == groups_final_exp
    assert grstack_final == grstack_final_exp

    assert combiner_all == combiner_all_exp


@pytest.mark.parametrize(
    "splitter, rpn",
    [
        ("a", ["a"]),
        (("a", "b"), ["a", "b", "."]),
        (["a", "b"], ["a", "b", "*"]),
        (["a", ("b", "c")], ["a", "b", "c", ".", "*"]),
        ([("a", "b"), "c"], ["a", "b", ".", "c", "*"]),
        (["a", ["b", ["c", "d"]]], ["a", "b", "c", "d", "*", "*", "*"]),
        (["a", ("b", ["c", "d"])], ["a", "b", "c", "d", "*", ".", "*"]),
        ((["a", "b"], "c"), ["a", "b", "*", "c", "."]),
        ((["a", "b"], ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
        (([["a", "b"]], ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
        (((["a", "b"],), ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
        ([("a", "b"), ("c", "d")], ["a", "b", ".", "c", "d", ".", "*"]),
    ],
)
def test_splitter2rpn(splitter, rpn):
    assert hlpst.splitter2rpn(splitter) == rpn


@pytest.mark.parametrize(
    "splitter, rpn",
    [
        ((("a", "b"), "c"), ["a", "b", ".", "c", "."]),
        (("a", "b", "c"), ["a", "b", ".", "c", "."]),
        ([["a", "b"], "c"], ["a", "b", "*", "c", "*"]),
        (["a", "b", "c"], ["a", "b", "*", "c", "*"]),
    ],
)
def test_splitter2rpn_2(splitter, rpn):
    assert hlpst.splitter2rpn(splitter) == rpn


@pytest.mark.parametrize(
    "splitter, rpn",
    [
        ("a", ["a"]),
        (("a", "b"), ["a", "b", "."]),
        (["a", "b"], ["a", "b", "*"]),
        (["a", ("b", "c")], ["a", "b", "c", ".", "*"]),
        ([("a", "b"), "c"], ["a", "b", ".", "c", "*"]),
        (["a", ["b", ["c", "d"]]], ["a", "b", "c", "d", "*", "*", "*"]),
        (["a", ("b", ["c", "d"])], ["a", "b", "c", "d", "*", ".", "*"]),
        ((["a", "b"], "c"), ["a", "b", "*", "c", "."]),
        ((["a", "b"], ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
        ([("a", "b"), ("c", "d")], ["a", "b", ".", "c", "d", ".", "*"]),
    ],
)
def test_rpn2splitter(splitter, rpn):
    assert hlpst.rpn2splitter(rpn) == splitter


@pytest.mark.parametrize(
    "splitter, other_states, rpn",
    [
        (
            ["a", "_NA"],
            {"NA": (other_states_to_tests(("b", "c")), "d")},
            ["a", "NA.b", "NA.c", ".", "*"],
        ),
        (
            ["_NA", "c"],
            {"NA": (other_states_to_tests(("a", "b")), "d")},
            ["NA.a", "NA.b", ".", "c", "*"],
        ),
        (
            ["a", ("b", "_NA")],
            {"NA": (other_states_to_tests(["c", "d"]), "d")},
            ["a", "b", "NA.c", "NA.d", "*", ".", "*"],
        ),
    ],
)
def test_splitter2rpn_wf_splitter_1(splitter, other_states, rpn):
    assert hlpst.splitter2rpn(splitter, other_states=other_states) == rpn


@pytest.mark.parametrize(
    "splitter, other_states, rpn",
    [
        (
            ["a", "_NA"],
            {"NA": (other_states_to_tests(("b", "c")), "d")},
            ["a", "_NA", "*"],
        ),
        (
            ["_NA", "c"],
            {"NA": (other_states_to_tests(("a", "b")), "d")},
            ["_NA", "c", "*"],
        ),
        (
            ["a", ("b", "_NA")],
            {"NA": (other_states_to_tests(["c", "d"]), "d")},
            ["a", "b", "_NA", ".", "*"],
        ),
    ],
)
def test_splitter2rpn_wf_splitter_3(splitter, other_states, rpn):
    assert (
        hlpst.splitter2rpn(splitter, other_states=other_states, state_fields=False)
        == rpn
    )


@pytest.mark.parametrize(
    "splitter, splitter_changed",
    [
        ("a", "Node.a"),
        (["a", ("b", "c")], ["Node.a", ("Node.b", "Node.c")]),
        (("a", ["b", "c"]), ("Node.a", ["Node.b", "Node.c"])),
    ],
)
def test_addname_splitter(splitter, splitter_changed):
    assert hlpst.add_name_splitter(splitter, "Node") == splitter_changed


@pytest.mark.parametrize(
    "splitter_rpn, input_to_remove, final_splitter_rpn",
    [
        (["a", "b", "."], ["b", "a"], []),
        (["a", "b", "*"], ["b"], ["a"]),
        (["a", "b", "c", ".", "*"], ["b", "c"], ["a"]),
        (["a", "b", "c", ".", "*"], ["a"], ["b", "c", "."]),
        (["a", "b", ".", "c", "*"], ["a", "b"], ["c"]),
        (["a", "b", "c", "d", "*", "*", "*"], ["c"], ["a", "b", "d", "*", "*"]),
        (["a", "b", "c", "d", "*", "*", "*"], ["a"], ["b", "c", "d", "*", "*"]),
        (["a", "b", "c", "d", "*", ".", "*"], ["a"], ["b", "c", "d", "*", "."]),
        (["a", "b", "*", "c", "."], ["a", "c"], ["b"]),
        (["a", "b", "*", "c", "d", "*", "."], ["a", "c"], ["b", "d", "."]),
        (["a", "b", ".", "c", "d", ".", "*"], ["a", "b"], ["c", "d", "."]),
    ],
)
def test_remove_inp_from_splitter_rpn(
    splitter_rpn, input_to_remove, final_splitter_rpn
):
    assert (
        hlpst.remove_inp_from_splitter_rpn(splitter_rpn, input_to_remove)
        == final_splitter_rpn
    )


@pytest.mark.parametrize(
    "group_for_inputs, input_for_groups, ndim",
    [
        ({"a": 0, "b": 0}, {0: ["a", "b"]}, 1),
        ({"a": 0, "b": 1}, {0: ["a"], 1: ["b"]}, 2),
    ],
)
def test_groups_to_input(group_for_inputs, input_for_groups, ndim):
    res = hlpst.converter_groups_to_input(group_for_inputs)
    assert res[0] == input_for_groups
    assert res[1] == ndim
