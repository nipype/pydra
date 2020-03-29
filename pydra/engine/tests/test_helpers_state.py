from .. import helpers_state as hlpst

import pytest


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
        (("a", "b"), ["a", "b"], {"a": 0, "b": 0}, [[0]]),
        (["a", "b"], ["a", "b"], {"a": 0, "b": 1}, [[0, 1]]),
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
    "splitter, cont_dim, values, keys, splits",
    [
        ("a", None, [(0,), (1,)], ["a"], [{"a": 1}, {"a": 2}]),
        (
            ("a", "v"),
            None,
            [(0, 0), (1, 1)],
            ["a", "v"],
            [{"a": 1, "v": "a"}, {"a": 2, "v": "b"}],
        ),
        (
            ["a", "v"],
            None,
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            ["a", "v"],
            [
                {"a": 1, "v": "a"},
                {"a": 1, "v": "b"},
                {"a": 2, "v": "a"},
                {"a": 2, "v": "b"},
            ],
        ),
        (
            ("a", "v", "c"),
            None,
            [((0, 0), 0), ((1, 1), 1)],
            ["a", "v", "c"],
            [{"a": 1, "c": 3, "v": "a"}, {"a": 2, "c": 4, "v": "b"}],
        ),
        (
            (("a", "v"), "c"),
            None,
            [((0, 0), 0), ((1, 1), 1)],
            ["a", "v", "c"],
            [{"a": 1, "c": 3, "v": "a"}, {"a": 2, "c": 4, "v": "b"}],
        ),
        (
            ("a", ("v", "c")),
            None,
            [(0, (0, 0)), (1, (1, 1))],
            ["a", "v", "c"],
            [{"a": 1, "c": 3, "v": "a"}, {"a": 2, "c": 4, "v": "b"}],
        ),
        (
            ["a", "v", "c"],
            None,
            [
                ((0, 0), 0),
                ((0, 0), 1),
                ((0, 1), 0),
                ((0, 1), 1),
                ((1, 0), 0),
                ((1, 0), 1),
                ((1, 1), 0),
                ((1, 1), 1),
            ],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "a", "c": 4},
                {"a": 1, "v": "b", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 3},
                {"a": 2, "v": "a", "c": 4},
                {"a": 2, "v": "b", "c": 3},
                {"a": 2, "v": "b", "c": 4},
            ],
        ),
        (
            [["a", "v"], "c"],
            None,
            [
                ((0, 0), 0),
                ((0, 0), 1),
                ((0, 1), 0),
                ((0, 1), 1),
                ((1, 0), 0),
                ((1, 0), 1),
                ((1, 1), 0),
                ((1, 1), 1),
            ],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "a", "c": 4},
                {"a": 1, "v": "b", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 3},
                {"a": 2, "v": "a", "c": 4},
                {"a": 2, "v": "b", "c": 3},
                {"a": 2, "v": "b", "c": 4},
            ],
        ),
        (
            ["a", ["v", "c"]],
            None,
            [
                (0, (0, 0)),
                (0, (0, 1)),
                (0, (1, 0)),
                (0, (1, 1)),
                (1, (0, 0)),
                (1, (0, 1)),
                (1, (1, 0)),
                (1, (1, 1)),
            ],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "a", "c": 4},
                {"a": 1, "v": "b", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 3},
                {"a": 2, "v": "a", "c": 4},
                {"a": 2, "v": "b", "c": 3},
                {"a": 2, "v": "b", "c": 4},
            ],
        ),
        (
            [("a", "v"), "c"],
            None,
            [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "a", "c": 4},
                {"a": 2, "v": "b", "c": 3},
                {"a": 2, "v": "b", "c": 4},
            ],
        ),
        (
            ["a", ("v", "c")],
            None,
            [(0, (0, 0)), (0, (1, 1)), (1, (0, 0)), (1, (1, 1))],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 3},
                {"a": 2, "v": "b", "c": 4},
            ],
        ),
        # TODO: check if it's ok
        (
            (("a", "v"), ("c", "z")),
            None,
            [((0, 0), (0, 0)), ((1, 1), (1, 1))],
            ["a", "v", "c", "z"],
            [{"a": 1, "v": "a", "c": 3, "z": 7}, {"a": 2, "v": "b", "c": 4, "z": 8}],
        ),
        (
            (["a", "v"], ["c", "z"]),
            None,
            [((0, 0), (0, 0)), ((0, 1), (0, 1)), ((1, 0), (1, 0)), ((1, 1), (1, 1))],
            ["a", "v", "c", "z"],
            [
                {"a": 1, "v": "a", "c": 3, "z": 7},
                {"a": 1, "v": "b", "c": 3, "z": 8},
                {"a": 2, "v": "a", "c": 4, "z": 7},
                {"a": 2, "v": "b", "c": 4, "z": 8},
            ],
        ),
        (
            [("a", "v"), ("c", "z")],
            None,
            [((0, 0), (0, 0)), ((0, 0), (1, 1)), ((1, 1), (0, 0)), ((1, 1), (1, 1))],
            ["a", "v", "c", "z"],
            [
                {"a": 1, "v": "a", "c": 3, "z": 7},
                {"a": 1, "v": "a", "c": 4, "z": 8},
                {"a": 2, "v": "b", "c": 3, "z": 7},
                {"a": 2, "v": "b", "c": 4, "z": 8},
            ],
        ),
        (
            (["a", "v"], "x"),
            {"x": 2},  # input x is treated as 2d container, so x will be flatten
            [((0, 0), 0), ((0, 1), 1), ((1, 0), 2), ((1, 1), 3)],
            ["a", "v", "x"],
            [
                {"a": 1, "v": "a", "x": 10},
                {"a": 1, "v": "b", "x": 100},
                {"a": 2, "v": "a", "x": 20},
                {"a": 2, "v": "b", "x": 200},
            ],
        ),
    ],
)
def test_splits_1b(splitter, cont_dim, values, keys, splits):
    inputs = {
        "a": [1, 2],
        "v": ["a", "b"],
        "c": [3, 4],
        "z": [7, 8],
        "x": [[10, 100], [20, 200]],
    }
    splitter_rpn = hlpst.splitter2rpn(splitter)
    values_out, keys_out = hlpst.splits(splitter_rpn, inputs, cont_dim=cont_dim)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(
        hlpst.map_splits(
            hlpst.iter_splits(value_list, keys_out), inputs, cont_dim=cont_dim
        )
    )
    assert splits_out == splits


@pytest.mark.parametrize(
    "splitter, cont_dim, inputs, mismatch",
    [
        ((["a", "v"], "c"), None, {"a": [1, 2], "v": ["a", "b"], "c": [3, 4]}, True),
        (
            (["a", "v"], "c"),
            {"c": 2},  # c is treated as 2d container
            {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], [5, 6]]},
            False,
        ),
        (
            (["a", "v"], "c"),
            None,
            {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], [5]]},
            True,
        ),
    ],
)
def test_splits_1c(splitter, cont_dim, inputs, mismatch):
    splitter_rpn = hlpst.splitter2rpn(splitter)
    if mismatch:
        with pytest.raises(ValueError):
            hlpst.splits(splitter_rpn, inputs, cont_dim=cont_dim)
    else:
        hlpst.splits(splitter_rpn, inputs, cont_dim=cont_dim)


@pytest.mark.parametrize(
    "splitter, cont_dim, values, keys, shapes, splits",
    [
        (
            (["a", "v"], "c"),
            {"c": 2},
            [((0, 0), 0), ((0, 1), 1), ((1, 0), 2), ((1, 1), 3)],
            ["a", "v", "c"],
            {"a": (2,), "v": (2,), "c": (2, 2)},
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 5},
                {"a": 2, "v": "b", "c": 6},
            ],
        ),
        (
            ("c", ["a", "v"]),
            {"c": 2},
            [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))],
            ["c", "a", "v"],
            {"a": (2,), "v": (2,), "c": (2, 2)},
            [
                {"a": 1, "v": "a", "c": 3},
                {"a": 1, "v": "b", "c": 4},
                {"a": 2, "v": "a", "c": 5},
                {"a": 2, "v": "b", "c": 6},
            ],
        ),
    ],
)
def test_splits_1d(splitter, cont_dim, values, keys, shapes, splits):
    inputs = {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], [5, 6]]}
    splitter_rpn = hlpst.splitter2rpn(splitter)
    values_out, keys_out = hlpst.splits(splitter_rpn, inputs, cont_dim=cont_dim)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(
        hlpst.map_splits(
            hlpst.iter_splits(value_list, keys_out), inputs, cont_dim=cont_dim
        )
    )
    assert splits_out == splits


@pytest.mark.parametrize(
    "splitter, values, keys, splits",
    [
        (
            (("a", "v"), "c"),
            [((0, 0), 0), ((1, 1), 1)],
            ["a", "v", "c"],
            [{"a": 1, "v": "a", "c": [3, 4]}, {"a": 2, "v": "b", "c": 5}],
        ),
        (
            [("a", "v"), "c"],
            [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
            ["a", "v", "c"],
            [
                {"a": 1, "v": "a", "c": [3, 4]},
                {"a": 1, "v": "a", "c": 5},
                {"a": 2, "v": "b", "c": [3, 4]},
                {"a": 2, "v": "b", "c": 5},
            ],
        ),
    ],
)
def test_splits_1e(splitter, values, keys, splits):
    # dj?: not sure if I like that this example works
    # c - is like an inner splitter
    inputs = {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], 5]}
    splitter_rpn = hlpst.splitter2rpn(splitter)
    values_out, keys_out = hlpst.splits(splitter_rpn, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(hlpst.map_splits(hlpst.iter_splits(value_list, keys_out), inputs))
    assert splits_out == splits


@pytest.mark.parametrize(
    "splitter_rpn, inner_inputs, values, keys, splits",
    [
        (
            ["NB.b"],
            {
                "NB.b": other_states_to_tests(
                    splitter="NA.a", keys_final=["NA.a"], ind_l=[(0,), (1,)]
                )
            },
            [(0, 0), (0, 1), (1, 2), (1, 3)],
            ["NA.a", "NB.b"],
            [
                {"NA.a": "a1", "NB.b": "b11"},
                {"NA.a": "a1", "NB.b": "b12"},
                {"NA.a": "a2", "NB.b": "b21"},
                {"NA.a": "a2", "NB.b": "b22"},
            ],
        )
    ],
)
# TODO: adding more?
def test_splits_2(splitter_rpn, inner_inputs, values, keys, splits):
    inputs = {
        "NA.a": ["a1", "a2"],
        "NA.b": ["b1", "b2"],
        "NB.b": [["b11", "b12"], ["b21", "b22"]],
        # needed?
        # "c": ["c1", "c2"],
        # "NB.d": [
        #     [["d111", "d112"], ["d121", "d122"]],
        #     [["d211", "d212"], ["d221", "d222"]],
        # ],
    }
    cont_dim = {"NB.b": 2}  # will be treated as 2d container
    values_out, keys_out = hlpst.splits(
        splitter_rpn, inputs, inner_inputs=inner_inputs, cont_dim=cont_dim
    )
    value_list = list(values_out)
    assert keys == keys_out
    splits_out = list(
        hlpst.map_splits(
            hlpst.iter_splits(value_list, keys_out), inputs, cont_dim=cont_dim
        )
    )
    assert splits_out == splits


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
