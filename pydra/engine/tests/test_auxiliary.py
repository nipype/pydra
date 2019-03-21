from .. import auxiliary as aux

import numpy as np
import pytest, pdb


class other_splitters_to_tests:
    def __init__(self, splitter, splitter_final=None, keys_final=None, ind_l=None, ind_l_final=None):
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


@pytest.mark.parametrize("splitter, keys_exp, groups_exp, grstack_exp", [
    ("a", ['a'], {"a": 0}, [[0]]),
    (("a", "b"), ['a', "b"], {"a": 0, "b": 0}, [[0]]),
    (["a", "b"], ['a', "b"], {"a": 0, "b": 1}, [[0, 1]]),
    ((["a", "b"], "c"), ['a', "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]]),
    ([("a", "b"), "c"], ['a', "b", "c"], {"a": 0, "b": 0, "c": 1}, [[0, 1]])
])
def test_splits_groups(splitter, keys_exp, groups_exp, grstack_exp):
    splitter_rpn = aux.splitter2rpn(splitter)
    keys, groups, grstack, keys_f, groups_f, grstack_f, _ = aux._splits_groups(splitter_rpn)
    assert keys == keys_exp
    assert groups == groups_exp
    assert grstack == grstack_exp
    # final values should be the same
    assert keys_f == keys_exp
    assert groups_f == groups_exp
    assert grstack_f == grstack_exp


@pytest.mark.parametrize("splitter, combiner, combiner_all_exp,"
                         "keys_exp, groups_exp, grstack_exp, "
                         "keys_final_exp, groups_final_exp, grstack_final_exp", [
    ("a", ["a"], ["a"],
     ["a"], {"a": 0}, [[0]],
     [],    {},       []),

    (("a", "b"), ["a"], ["a", "b"],
     ["a", "b"], {"a": 0, "b": 0}, [[0]],
     [],         {},               [[]]),
    (("a", "b"), ["b"], ["a", "b"],
     ["a", "b"], {"a": 0, "b": 0}, [[0]],
     [],         {},               [[]]),

    (["a", "b"], ["b"], ["b"],
     ["a", "b"], {"a": 0, "b": 1}, [[0, 1]],
     ["a"],      {"a": 0},         [[0]]),
    (["a", "b"], ["a"], ["a"],
     ["a", "b"], {"a": 0, "b": 1}, [[0, 1]],
     ["b"],      {"b": 0},         [[0]]),

    ((["a", "b"], "c"), ["a"], ["a", "c"],
     ["a", "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]],
     ["b"],           {"b": 0},                      [[0]]),
    ((["a", "b"], "c"), ["b"], ["b", "c"],
     ["a", "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]],
     ["a"],           {"a": 0},                      [[0]]),
    ((["a", "b"], "c"), ["a"], ["a", "c"],
     ["a", "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]],
     ["b"], {"b": 0}, [[0]]),
    ((["a", "b"], "c"), ["c"], ["a", "b", "c"],
     ["a", "b", "c"], {"a": 0, "b": 1, "c": [0, 1]}, [[0, 1]],
     [],              {},                            [[]]),

    ([("a", "b"), "c"], ["a"], ["a", "b"],
     ["a", "b", "c"], {"a": 0, "b": 0, "c": 1}, [[0, 1]],
     ["c"],           {"c": 0},                 [[0]]),
    ([("a", "b"), "c"], ["b"], ["a", "b"],
     ["a", "b", "c"], {"a": 0, "b": 0, "c": 1}, [[0, 1]],
     ["c"],           {"c": 0},                 [[0]]),
    ([("a", "b"), "c"], ["c"], ["c"],
     ["a", "b", "c"], {"a": 0, "b": 0, "c": 1}, [[0, 1]],
     ["a", "b"],      {"a": 0, "b": 0},         [[0]]),

])
def test_splits_groups_comb(splitter, combiner, keys_exp, groups_exp, grstack_exp,
                            keys_final_exp, groups_final_exp, grstack_final_exp,
                            combiner_all_exp):
    splitter_rpn = aux.splitter2rpn(splitter)
    keys, groups, grstack, keys_final, groups_final, grstack_final, combiner_all \
        = aux._splits_groups(splitter_rpn, combiner)
    assert keys == keys_exp
    assert groups == groups_exp
    assert grstack == grstack_exp

    assert keys_final == keys_final_exp
    assert groups_final == groups_final_exp
    assert grstack_final == grstack_final_exp

    assert combiner_all == combiner_all_exp

# TODO: I think this test should be removed, proper connections are done in the state
# @pytest.mark.parametrize("splitter, other_splitters, keys_exp, groups_exp, grstack_exp", [
#     ("_NA", {"NA": (other_splitters_to_tests(splitter="NA.a", keys_final=["NA.a"]), "a")},
#      ['NA.a'], {"NA.a": 0}, [[0]]),
#     ("_NA", {"NA": (other_splitters_to_tests(splitter=["NA.a", "NA.b"], keys_final=["NA.a", "NA.b"]), "a")},
#      ['NA.a', "NA.b"], {"NA.a": 0, "NA.b": 1}, [[0, 1]]),
#     (["_NA", "NB.a"],
#      {"NA": (other_splitters_to_tests(splitter=["NA.a", "NA.b"], keys_final=["NA.a", "NA.b"]), "NB.a")},
#      ['NA.a', "NA.b", "NB.a"], {"NA.a": 0, "NA.b": 1, "NB.a": 2}, [[0, 1], [2]]),
#     # (["_NA", "NB.b"],
#     #  {"NA": (other_splitters_to_tests(splitter=["NA.a", "NA.b"], keys_final=["NA.a", "NA.b"]), "NB.a")},
#     #  ['NA.a', "NA.b", "NB.b"], {"NA.a": 0, "NA.b": 1, "NB.b": 2}, [[0, 1, 2]]),
# ])
# def test_splits_groups_connect(splitter, other_splitters, keys_exp, groups_exp, grstack_exp):
#     splitter_rpn = aux.splitter2rpn(splitter, other_splitters)
#     inner_inputs = {k: v for _, (v, k) in other_splitters.items()}
#     keys, groups, grstack, keys_f, groups_f, grstack_f, _ = aux._splits_groups(splitter_rpn, inner_inputs=inner_inputs)
#     assert keys == keys_exp
#     assert groups == groups_exp
#     assert grstack == grstack_exp
#     # final values should be the same
#     assert keys_f == keys_exp
#     assert groups_f == groups_exp
#     assert grstack_f == grstack_exp


@pytest.mark.parametrize("splitter, values, keys, splits", [
   ("a", [(0,), (1,)], ['a'], [{'a': 1}, {'a': 2}]),
    (("a", "v"), [(0, 0), (1, 1)], ['a', 'v'],
     [{'a': 1, 'v': 'a'}, {'a': 2, 'v': 'b'}]),
    (["a", "v"], [(0, 0), (0, 1), (1, 0), (1, 1)], ['a', 'v'],
     [{'a': 1, 'v': 'a'}, {'a': 1, 'v': 'b'}, {'a': 2, 'v': 'a'},
      {'a': 2, 'v': 'b'}]),
    (("a", "v", "c"), [((0, 0), 0), ((1, 1), 1)], ['a', 'v', 'c'],
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    ((("a", "v"), "c"), [((0, 0), 0), ((1, 1), 1)], ['a', 'v', 'c'],
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    (("a", ("v", "c")), [(0, (0, 0)), (1, (1, 1))], ['a', 'v', 'c'],
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    (["a", "v", "c"],
     [((0, 0), 0), ((0, 0), 1), ((0, 1), 0), ((0, 1), 1),
      ((1, 0), 0), ((1, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    ([["a", "v"], "c"],
     [((0, 0), 0), ((0, 0), 1), ((0, 1), 0), ((0, 1), 1),
      ((1, 0), 0), ((1, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    (["a", ["v", "c"]],
     [(0, (0, 0)), (0, (0, 1)), (0, (1, 0)), (0, (1, 1)),
      (1, (0, 0)), (1, (0, 1)), (1, (1, 0)), (1, (1, 1))],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    ([("a", "v"), "c"],
     [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    (["a", ("v", "c")],
     [(0, (0, 0)), (0, (1, 1)), (1, (0, 0)), (1, (1, 1))],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    # TODO: check if it's ok
    ((("a", "v"), ("c", "z")),
     [((0, 0), (0, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'],
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ((["a", "v"], ["c", "z"]),
     [((0, 0), (0, 0)), ((0, 1), (0, 1)),
      ((1, 0), (1, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'],
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 1, 'v': 'b', 'c': 3, 'z': 8},
      {'a': 2, 'v': 'a', 'c': 4, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ([("a", "v"), ("c", "z")],
     [((0, 0), (0, 0)), ((0, 0), (1, 1)),
      ((1, 1), (0, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'],
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 1, 'v': 'a', 'c': 4, 'z': 8},
      {'a': 2, 'v': 'b', 'c': 3, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ((["a", "v"], "x"),
     [((0, 0), 0), ((0, 1), 1),
      ((1, 0), 2), ((1, 1), 3)],
     ['a', 'v', 'x'],
     [{'a': 1, 'v': 'a', 'x': 10},
      {'a': 1, 'v': 'b', 'x': 100},
      {'a': 2, 'v': 'a', 'x': 20},
      {'a': 2, 'v': 'b', 'x': 200}]),
])
def test_splits_1b(splitter, values, keys, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c": [3, 4], "z": [7, 8], "x": [[10, 100], [20, 200]]}
    splitter_rpn = aux.splitter2rpn(splitter)
    values_out, keys_out, _, _ = aux._splits(splitter_rpn, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
    assert splits_out == splits


@pytest.mark.parametrize(
    "splitter, inputs, mismatch",
    [
        ((["a", "v"], "c"), {"a": [1, 2], "v": ["a", "b"], "c": [3, 4]}, True),
        (
            (["a", "v"], "c"),
            {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], [5, 6]]},
            False,
        ),
        ((["a", "v"], "c"), {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], [5]]}, True),
    ],
)
def test_splits_1c(splitter, inputs, mismatch):
    splitter_rpn = aux.splitter2rpn(splitter)
    if mismatch:
        with pytest.raises(ValueError):
            aux._splits(splitter_rpn, inputs)
    else:
        aux._splits(splitter_rpn, inputs)


@pytest.mark.parametrize("splitter, values, keys, shapes, splits", [
    ((["a", "v"], "c"),
     [((0, 0), 0), ((0, 1), 1), ((1, 0), 2), ((1, 1), 3)],
     ['a', 'v', 'c'],
     {'a': (2,), 'v': (2,), 'c': (2, 2)},
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 5}, {'a': 2, 'v': 'b', 'c': 6}]),
    (("c", ["a", "v"]),
     [(0, (0, 0)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1))],
     ['c', 'a', 'v'],
     {'a': (2,), 'v': (2,), 'c': (2, 2)},
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 5}, {'a': 2, 'v': 'b', 'c': 6}]),
])
def test_splits_1d(splitter, values, keys, shapes, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c": [[3, 4], [5, 6]]}
    splitter_rpn = aux.splitter2rpn(splitter)
    values_out, keys_out, shapes_out, _ = aux._splits(splitter_rpn, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    assert shapes == shapes_out
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out), inputs))
    assert splits_out == splits


@pytest.mark.parametrize("splitter, values, keys, splits", [
    ((("a", "v"), "c"),
     [((0, 0), 0), ((1, 1), 1)],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': [3, 4]}, {'a': 2, 'v': 'b', 'c': 5}]),
    ([("a", "v"), "c"],
     [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'],
     [{'a': 1, 'v': 'a', 'c': [3, 4]}, {'a': 1, 'v': 'a', 'c': 5},
      {'a': 2, 'v': 'b', 'c': [3, 4]}, {'a': 2, 'v': 'b', 'c': 5}]),
    ])
def test_splits_1e(splitter, values, keys, splits):
    # dj?: not sure if I like that this example works
    # c - is like an inner splitter
    inputs = {"a": [1, 2], "v": ["a", "b"], "c": [[3, 4], 5]}
    splitter_rpn = aux.splitter2rpn(splitter)
    values_out, keys_out, _, _ = aux._splits(splitter_rpn, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
    assert splits_out == splits


@pytest.mark.parametrize("splitter_rpn, inner_inputs, values, keys, splits", [
    # (["NA.a", "NA.b", "*"],
    #  {"c": other_splitters_to_tests(splitter=["NA.a", "NA.b"], keys_final=["NA.a", "NA.b"],
    #                                 ind_l_final=[(0, 0), (0, 1), (1, 0), (1, 1)])},
    #  [(0, 0), (0, 1), (1, 0), (1, 1)], ["NA.a", "NA.b"],
    #  [{"NA.a": "a1", "NA.b": "b1"}, {"NA.a": "a1", "NA.b": "b2"},
    #   {"NA.a": "a2", "NA.b": "b1"}, {"NA.a": "a2", "NA.b": "b2"}]),
    (["NA.a", "NB.b", "*"],
     {"NB.b": other_splitters_to_tests(splitter="NA.a", keys_final=["NA.a"], ind_l=[(0,), (1,)])},
     [(0, 0), (0, 1), (1, 2), (1, 3)], ["NA.a", "NB.b"],
     [{"NA.a": "a1", "NB.b": "b11"}, {"NA.a": "a1", "NB.b": "b12"},
      {"NA.a": "a2", "NB.b": "b21"}, {"NA.a": "a2", "NB.b": "b22"}]),

    # [(0, 0), (0, 1), (1, 2), (1, 3)],
     # ["a", "b"],
     # [{"a": "a1", "b": "b11"}, {"a": "a1", "b": "b12"},
     #  {"a": "a2", "b": "b21"}, {"a": "a2", "b": "b22"}]),
    # (["c", "a", ".", "b", "*"], ["b"], [((0, 0), 0), ((0, 0), 1), ((1, 1), 2), ((1, 1), 3)],
    #  ["c", "a", "b"],
    #  [{"a": "a1", "b": "b11", "c": "c1"}, {"a": "a1", "b": "b12", "c": "c1"},
    #   {"a": "a2", "b": "b21", "c": "c2"}, {"a": "a2", "b": "b22", "c": "c2"}]),
    # (["c", "a", "*", "d", "*"], ["d"],
    #  [((0, 0), 0), ((0, 0), 1), ((0, 1), 2), ((0, 1), 3),
    #   ((1, 0), 4), ((1, 0), 5), ((1, 1), 6), ((1, 1), 7)],
    #  ["c", "a", "d"],
    #  [{"a": "a1", "c": "c1", "d": "d111"}, {"a": "a1", "c": "c1", "d": "d112"},
    #   {"a": "a2", "c": "c1", "d": "d121"}, {"a": "a2", "c": "c1", "d": "d122"},
    #   {"a": "a1", "c": "c2", "d": "d211"}, {"a": "a1", "c": "c2", "d": "d212"},
    #   {"a": "a2", "c": "c2", "d": "d221"}, {"a": "a2", "c": "c2", "d": "d222"}])
    # no idea how this should work TODO!
    #(["a", "b", "*", "c", "*"], ["b"])
    ])
@pytest.mark.xfail(reason="fix inner_inputs!")
#TODO: check how it works with previous (with and without inner_input)
def test_splits_2(splitter_rpn, inner_inputs, values, keys, splits):
    inputs = {"NA.a": ["a1", "a2"], "NA.b": ["b1", "b2"],
              "NB.b": [["b11", "b12"], ["b21", "b22"]], "c": ["c1", "c2"],
              "d": [[["d111", "d112"], ["d121", "d122"]], [["d211", "d212"], ["d221", "d222"]]]}
    #inner_inputs = {k: v for _, (v, k) in other_splitters.items()}
    values_out, keys_out, _, _ = aux._splits(splitter_rpn, inputs, inner_inputs=inner_inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
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
    assert aux.splitter2rpn(splitter) == rpn


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
    assert aux.splitter2rpn(splitter) == rpn


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
    assert aux.rpn2splitter(rpn) == splitter


@pytest.mark.parametrize(
    "rpn, keys_remove, new_rpn",
    [
        (["a"], ["a"], []),
        (["a"], [], ["a"]),
        (["a", "b", "*"], ["a"], ["b"]),
        (["a", "b", "*"], ["b"], ["a"]),
        (["a", "b", "*"], ["a", "b"], []),
        (["a", "b", "*"], ["b", "a"], []),
        (["a", "b", "c", ".", "*"], ["a"], ["b", "c", "."]),
        (["a", "b", "c", "d", "*", "*", "*"], ["b"], ["a", "c", "d", "*", "*"]),
        (["a", "b", "c", "d", "*", "*", "*"], ["b", "d"], ["a", "c", "*"]),
    ],
)
@pytest.mark.skip(reason="for now using removing_inputs_rpn,"
                         "will probably remove removing_inputs_rpn")
def test_removing_inputs_rpn(rpn, keys_remove, new_rpn):
    assert new_rpn == aux.removing_inputs_rpn(rpn, keys_remove)


@pytest.mark.parametrize(
    "rpn, keys_remove",
    [
        (["a", "b", "."], ["a"]),
        (["a", "b", "."], ["b"]),
        (["a", "b", "c", ".", "*"], ["b"]),
    ],
)
@pytest.mark.xfail(reason="for now using removing_inputs_rpn")
def test_removing_inputs_rpn_exception(rpn, keys_remove):
    with pytest.raises(Exception):
        aux.remove_inp_from_splitter_rpn(rpn, keys_remove)


@pytest.mark.xfail(reason="fix the class")
@pytest.mark.parametrize(
    "splitter, other_splitters, rpn",
    [
        (
            ["a", "_NA"],
            {"NA": (other_splitters_to_tests(("b", "c")), "d")},
            ["a", "NA.b", "NA.c", ".", "*"],
        ),
        (
            ["_NA", "c"],
            {"NA": (other_splitters_to_tests(("a", "b")), "d")},
            ["NA.a", "NA.b", ".", "c", "*"],
        ),
        (
            ["a", ("b", "_NA")],
            {"NA": (other_splitters_to_tests(["c", "d"]), "d")},
            ["a", "b", "NA.c", "NA.d", "*", ".", "*"],
        ),
    ],
)
def test_splitter2rpn_wf_splitter_1(splitter, other_splitters, rpn):
    assert aux.splitter2rpn(splitter, other_splitters=other_splitters) == rpn


@pytest.mark.parametrize(
    "splitter, other_splitters, rpn",
    [
        (
            ["a", "_NA"],
            {"NA": (other_splitters_to_tests(("b", "c")), "d")},
            ["a", "_NA", "*"],
        ),
        (
            ["_NA", "c"],
            {"NA": (other_splitters_to_tests(("a", "b")), "d")},
            ["_NA", "c", "*"],
        ),
        (
            ["a", ("b", "_NA")],
            {"NA": (other_splitters_to_tests(["c", "d"]), "d")},
            ["a", "b", "_NA", ".", "*"],
        ),
    ],
)
@pytest.mark.xfail(reason="fix the class")
def test_splitter2rpn_wf_splitter_3(splitter, other_splitters, rpn):
    assert aux.splitter2rpn(splitter, other_splitters=other_splitters, state_fields=False) == rpn


@pytest.mark.parametrize("splitter, splitter_changed",[
    ("a", "Node.a"),
    (["a", ("b", "c")], ["Node.a", ("Node.b", "Node.c")]),
    (("a", ["b", "c"]), ("Node.a", ["Node.b", "Node.c"]))
])
def test_change_splitter(splitter, splitter_changed):
    assert aux.change_splitter(splitter, "Node") == splitter_changed


@pytest.mark.parametrize("inputs, rpn, expected", [
    ({"a": np.array([1, 2])}, ["a"], {"a": [0]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4])},
     ["a", "b", "."], {"a": [0],"b": [0]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4, 1])},
     ["a", "b", "*"], {"a": [0],"b": [1]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4]), "c": np.array([1, 2, 3])},
     ["a", "b", ".", "c", "*"], {"a": [0], "b": [0], "c": [1]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4]), "c": np.array([1, 2, 3])},
     ["c", "a", "b", ".", "*"], {"a": [1], "b": [1], "c": [0]}),
    ({"a": np.array([[1, 2], [1, 2]]), "b": np.array([[3, 4], [3, 3]]),
      "c": np.array([1, 2, 3])},
     ["a", "b", ".", "c", "*"], {"a": [0, 1], "b": [0, 1], "c": [2]}),
    ({"a": np.array([[1, 2], [1, 2]]), "b": np.array([[3, 4], [3, 3]]),
      "c": np.array([1, 2, 3])},
     ["c", "a", "b", ".", "*"], {"a": [1, 2], "b": [1, 2], "c": [0]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 3]), "c": np.array([[1, 2], [3, 4]])},
     ["a", "b", "*", "c", "."], {"a": [0], "b": [1], "c": [0, 1]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4, 5]), "c": np.array([1, 2]),
      "d": np.array([1, 2, 3])},
     ["a", "b", "*", "c", "d", "*", "."], {"a":[0], "b": [1], "c": [0], "d": [1]}),
    ({"a": np.array([1, 2]), "b": np.array([3, 4]), "c": np.array([1, 2, 3]),
      "d": np.array([1, 2, 3])},
     ["a", "b", ".", "c", "d", ".", "*"], {"a": [0], "b": [0], "c": [1], "d": [1]})
])
def test_splitting_axis(inputs, rpn, expected):
    res = aux.splitting_axis(inputs, rpn)[0]
    print(res)
    for key in inputs.keys():
        assert res[key] == expected[key]


def test_splitting_axis_error():
    with pytest.raises(Exception):
        aux.splitting_axis({"a": np.array([1, 2]), "b": np.array([3, 4, 5])}, ["a", "b", "."])


@pytest.mark.parametrize("inputs, axis_inputs, ndim, expected", [
    ({"a": np.array([1, 2])}, {"a": [0]}, 1, [["a"]]),
    ({"a": np.array([1, 2]), "b": np.array([3, 4])},
     {"a": [0], "b": [0]}, 1, [["a", "b"]]),
    ({"a": np.array([1, 2]), "b": np.array([3, 4, 1])},
     {"a": [0], "b": [1]}, 2, [["a"], ["b"]]),
    ({"a": np.array([1, 2]), "b": np.array([3, 4]), "c": np.array([1, 2, 3])},
     {"a": [0], "b": [0], "c": [1]}, 2, [["a", "b"], ["c"]]),
    ({"a": np.array([1, 2]), "b": np.array([3, 4]), "c": np.array([1, 2, 3])},
     {"a": [1], "b": [1], "c": [0]}, 2, [["c"], ["a", "b"]]),
    ({"a": np.array([[1, 2], [1, 2]]), "b": np.array([[3, 4], [3, 3]]),
      "c": np.array([1, 2, 3])},
     {"a": [0, 1], "b": [0, 1], "c": [2]}, 3, [["a", "b"], ["a", "b"], ["c"]]),
    ({"a": np.array([[1, 2], [1, 2]]), "b": np.array([[3, 4], [3, 3]]),
      "c": np.array([1, 2, 3])},
     {"a": [1, 2], "b": [1, 2], "c": [0]}, 3, [["c"], ["a", "b"], ["a", "b"]])
])
def test_converting_axis2input(inputs, axis_inputs, ndim, expected):
    assert aux.converting_axis2input(state_inputs=inputs, axis_for_input=axis_inputs,
                                     ndim=ndim)[0] == expected


@pytest.mark.parametrize("rpn, expected, ndim", [
    (["a"], {"a": [0]}, 1),
    (["a", "b", "."], {"a": [0],"b": [0]}, 1),
    (["a", "b", "*"], {"a": [0],"b": [1]}, 2),
    (["a", "b", ".", "c", "*"], {"a": [0], "b": [0], "c": [1]}, 2),
    (["c", "a", "b", ".", "*"], {"a": [1], "b": [1], "c": [0]}, 2),
    (["a", "b", ".", "c", "*"], {"a": [0], "b": [0], "c": [1]}, 2),
    (["c", "a", "b", ".", "*"], {"a": [1], "b": [1], "c": [0]}, 2),
    (["a", "b", "*", "c", "."], {"a": [0], "b": [1], "c": [0, 1]}, 2),
    (["a", "b", "*", "c", "d", "*", "."], {"a":[0], "b": [1], "c": [0], "d": [1]}, 2),
    (["a", "b", ".", "c", "d", ".", "*"], {"a": [0], "b": [0], "c": [1], "d": [1]}, 2)
])
def test_matching_input_from_splitter(rpn, expected, ndim):
    res = aux.matching_input_from_splitter(rpn)
    print(res)
    for key in expected.keys():
        assert res[0][key] == expected[key]
    assert res[1] == ndim


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
        aux.remove_inp_from_splitter_rpn(splitter_rpn, input_to_remove)
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
    res = aux.converter_groups_to_input(group_for_inputs)
    assert res[0] == input_for_groups
    assert res[1] == ndim


@pytest.mark.parametrize("splitter, other_splitters, expected_splitter, expected_left, expected_right", [
    (None, {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")},
     "_NA", "_NA", None),
    ("b", {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")},
     ["_NA", "b"], "_NA", "b"),
    (("b", "c"), {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")},
     ["_NA", ("b", "c")], "_NA", ("b", "c")),
    (None,
     {"NA": (other_splitters_to_tests(splitter="NA.a"), "a"), "NB": (other_splitters_to_tests(splitter="NB.a"), "b")},
     ["_NA", "_NB"], ["_NA", "_NB"], None),
    ("b",
     {"NA": (other_splitters_to_tests(splitter="NA.a"), "a"), "NB": (other_splitters_to_tests(splitter="NB.a"), "b")},
     [["_NA", "_NB"], "b"], ["_NA", "_NB"], "b"),
    (["_NA", "b"],
     {"NA": (other_splitters_to_tests(splitter="NA.a"), "a"), "NB": (other_splitters_to_tests(splitter="NB.a"), "b")},
     [["_NB", "_NA"], "b"], ["_NB", "_NA"], "b"),
])
def test_connect_splitters(splitter, other_splitters, expected_splitter, expected_left, expected_right):
    updated_splitter, left_splitter, right_splitter = \
        aux.connect_splitters(splitter, other_splitters)
    assert updated_splitter == expected_splitter
    assert left_splitter == expected_left
    assert right_splitter == expected_right


@pytest.mark.parametrize("splitter, other_splitters", [
    ("_NB", {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")}),
    (("_NA", "b"), {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")}),
    (["b", "_NA"], {"NA": (other_splitters_to_tests(splitter="NA.a"), "b")}),
    (["_NB", ["_NA", "b"]],
     {"NA": (other_splitters_to_tests(splitter="NA.a"), "a"), "NB": (other_splitters_to_tests(splitter="NB.a"), "b")}),
])
def test_connect_splitters_exception(splitter, other_splitters):
    with pytest.raises(Exception):
        aux.connect_splitters(splitter, other_splitters)
