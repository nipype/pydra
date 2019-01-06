from .. import auxiliary as aux

import numpy as np
import pytest


@pytest.mark.parametrize("splitter, values, keys, groups, fgroup, splits", [
    (("a", "v"), [(0, 0), (1, 1)], ['a', 'v'], {'a': 0, 'v': 0},
     0,
     [{'a': 1, 'v': 'a'}, {'a': 2, 'v': 'b'}]),
    (["a", "v"], [(0, 0), (0, 1), (1, 0), (1, 1)], ['a', 'v'],
     {'a': 0, 'v': 1}, [0, 1],
     [{'a': 1, 'v': 'a'}, {'a': 1, 'v': 'b'}, {'a': 2, 'v': 'a'},
      {'a': 2, 'v': 'b'}]),
    (("a", "v", "c"), [((0, 0), 0), ((1, 1), 1)], ['a', 'v', 'c'],
     {'a': 0, 'v': 0, 'c': 0}, 0,
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    ((("a", "v"), "c"), [((0, 0), 0), ((1, 1), 1)], ['a', 'v', 'c'],
     {'a': 0, 'v': 0, 'c': 0}, 0,
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    (("a", ("v", "c")), [(0, (0, 0)), (1, (1, 1))], ['a', 'v', 'c'],
     {'a': 0, 'v': 0, 'c': 0}, 0,
     [{'a': 1, 'c': 3, 'v': 'a'}, {'a': 2, 'c': 4, 'v': 'b'}]),
    (["a", "v", "c"],
     [((0, 0), 0), ((0, 0), 1), ((0, 1), 0), ((0, 1), 1),
      ((1, 0), 0), ((1, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'], {'a': 0, 'v': 1, 'c': 2}, [0, 1, 2],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    ([["a", "v"], "c"],
     [((0, 0), 0), ((0, 0), 1), ((0, 1), 0), ((0, 1), 1),
      ((1, 0), 0), ((1, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'], {'a': 0, 'v': 1, 'c': 2}, [0, 1, 2],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    (["a", ["v", "c"]],
     [(0, (0, 0)), (0, (0, 1)), (0, (1, 0)), (0, (1, 1)),
      (1, (0, 0)), (1, (0, 1)), (1, (1, 0)), (1, (1, 1))],
     ['a', 'v', 'c'], {'a': 2, 'c': 1, 'v': 0}, [2, 0, 1],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 1, 'v': 'b', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    ([("a", "v"), "c"],
     [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'], {'a': 0, 'v': 0, 'c': 1}, [0, 1],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'a', 'c': 4},
      {'a': 2, 'v': 'b', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    (["a", ("v", "c")],
     [(0, (0, 0)), (0, (1, 1)), (1, (0, 0)), (1, (1, 1))],
     ['a', 'v', 'c'], {'v': 0, 'c': 0, 'a': 1}, [1, 0],
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 3}, {'a': 2, 'v': 'b', 'c': 4}]),
    ((("a", "v"), ("c", "z")),
     [((0, 0), (0, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'], {'v': 0, 'c': 0, 'a': 0, 'z': 0}, 0,
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ((["a", "v"], ["c", "z"]),
     [((0, 0), (0, 0)), ((0, 1), (0, 1)),
      ((1, 0), (1, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'], {'a': 0, 'v': 1, 'c': 2, 'z': 3}, [0, 1],
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 1, 'v': 'b', 'c': 3, 'z': 8},
      {'a': 2, 'v': 'a', 'c': 4, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ([("a", "v"), ("c", "z")],
     [((0, 0), (0, 0)), ((0, 0), (1, 1)),
      ((1, 1), (0, 0)), ((1, 1), (1, 1))],
     ['a', 'v', 'c', 'z'], {'a': 0, 'v': 0, 'c': 1, 'z': 1}, [0, 1],
     [{'a': 1, 'v': 'a', 'c': 3, 'z': 7},
      {'a': 1, 'v': 'a', 'c': 4, 'z': 8},
      {'a': 2, 'v': 'b', 'c': 3, 'z': 7},
      {'a': 2, 'v': 'b', 'c': 4, 'z': 8}]),
    ])
def test_splits_1b(splitter, values, keys, groups, fgroup, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c": [3, 4], "z": [7, 8]}
    values_out, keys_out, groups_out, finalgrp_out = aux._splits(splitter, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    assert groups == groups_out
    assert fgroup == finalgrp_out
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
    assert splits_out == splits


@pytest.mark.parametrize("splitter, inputs, mismatch", [
    ((["a", "v"], "c"), {"a": [1, 2], "v": ['a', 'b'], "c": [3, 4]}, True),
    ((["a", "v"], "c"), {"a": [1, 2], "v": ['a', 'b'], "c": [[3, 4], [5, 6]]},
     False),
    ((["a", "v"], "c"), {"a": [1, 2], "v": ['a', 'b'], "c": [[3, 4], [5]]},
     True),
    ])
def test_splits_1c(splitter, inputs, mismatch):
    if mismatch:
        with pytest.raises(ValueError):
            aux._splits(splitter, inputs)
    else:
        aux._splits(splitter, inputs)


@pytest.mark.parametrize("splitter, values, keys, groups, splits", [
    ((["a", "v"], "c"),
     [((0, 0), 0), ((0, 1), 1), ((1, 0), 2), ((1, 1), 3)],
     ['a', 'v', 'c'], {'a': 0, 'v': 1, 'c': [0, 1]},
     [{'a': 1, 'v': 'a', 'c': 3}, {'a': 1, 'v': 'b', 'c': 4},
      {'a': 2, 'v': 'a', 'c': 5}, {'a': 2, 'v': 'b', 'c': 6}]),
    ])
def test_splits_1d(splitter, values, keys, groups, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c": [[3, 4], [5, 6]]}
    values_out, keys_out, groups_out = aux._splits(splitter, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    assert groups == groups_out
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
    assert splits_out == splits

@pytest.mark.parametrize("splitter, values, keys, groups, splits", [
    ((("a", "v"), "c"),
     [((0, 0), 0), ((1, 1), 1)],
     ['a', 'v', 'c'], {'a': 0, 'v': 0, 'c': 0},
     [{'a': 1, 'v': 'a', 'c': [3, 4]}, {'a': 2, 'v': 'b', 'c': 5}]),
    ([("a", "v"), "c"],
     [((0, 0), 0), ((0, 0), 1), ((1, 1), 0), ((1, 1), 1)],
     ['a', 'v', 'c'], {'a': 0, 'v': 0, 'c': 1},
     [{'a': 1, 'v': 'a', 'c': [3, 4]}, {'a': 1, 'v': 'a', 'c': 5},
      {'a': 2, 'v': 'b', 'c': [3, 4]}, {'a': 2, 'v': 'b', 'c': 5}]),
    ])
def test_splits_1e(splitter, values, keys, groups, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c": [[3, 4], 5]}
    values_out, keys_out, groups_out = aux._splits(splitter, inputs)
    value_list = list(values_out)
    assert keys == keys_out
    assert values == value_list
    assert groups == groups_out
    splits_out = list(aux.map_splits(aux.iter_splits(value_list, keys_out),
                                     inputs))
    assert splits_out == splits


@pytest.mark.parametrize("splitter, splits", [
    (("a", "v", "c"), [{'c': 3, 'v': 'a', 'a': 1},
                      {'c': 4, 'v': 'b', 'a': 2}]),
    (["a", "v", "c"], [{'a': 1, 'v': 'a', 'c': 3},
                      {'a': 1, 'v': 'a', 'c': 4},
                      {'a': 1, 'v': 'b', 'c': 3},
                      {'a': 1, 'v': 'b', 'c': 4},
                      {'a': 2, 'v': 'a', 'c': 3},
                      {'a': 2, 'v': 'a', 'c': 4},
                      {'a': 2, 'v': 'b', 'c': 3},
                      {'a': 2, 'v': 'b', 'c': 4}]),
    (["a", ("v", "c")], [{'a': 1, 'v': 'a', 'c': 3},
                      {'a': 1, 'v': 'b', 'c': 4},
                      {'a': 2, 'v': 'a', 'c': 3},
                      {'a': 2, 'v': 'b', 'c': 4}])])
def test_splits_2(splitter, splits):
    inputs = {"a": [1, 2], "v": ['a', 'b'], "c":[3, 4]}
    splits_out = list(aux.map_splits(aux.splits(splitter, inputs),
                                     inputs))
    assert list(splits_out[0].keys()) == ["a", "v", "c"]
    assert splits_out == splits


@pytest.mark.parametrize("splitter, rpn", [
    ("a", ["a"]),
    (("a", "b"), ["a", "b", "."]),
    (["a", "b"], ["a", "b", "*"]),
    (["a", ("b", "c")], ["a", "b", "c", ".", "*"]),
    ([("a", "b"), "c"], ["a", "b", ".", "c", "*"]),
    (["a", ["b", ["c", "d"]]], ["a", "b", "c", "d", "*", "*", "*"]),
    (["a", ("b", ["c", "d"])], ["a", "b", "c", "d", "*", ".", "*"]),
    ((["a", "b"], "c"), ["a", "b", "*", "c", "."]),
    ((["a", "b"], ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
    ([("a", "b"), ("c", "d")], ["a", "b", ".", "c", "d", ".", "*"])
])
def test_splitter2rpn(splitter, rpn):
    assert aux.splitter2rpn(splitter) == rpn


@pytest.mark.parametrize("splitter, rpn", [
    ("a", ["a"]),
    (("a", "b"), ["a", "b", "."]),
    (["a", "b"], ["a", "b", "*"]),
    (["a", ("b", "c")], ["a", "b", "c", ".", "*"]),
    ([("a", "b"), "c"], ["a", "b", ".", "c", "*"]),
    (["a", ["b", ["c", "d"]]], ["a", "b", "c", "d", "*", "*", "*"]),
    (["a", ("b", ["c", "d"])], ["a", "b", "c", "d", "*", ".", "*"]),
    ((["a", "b"], "c"), ["a", "b", "*", "c", "."]),
    ((["a", "b"], ["c", "d"]), ["a", "b", "*", "c", "d", "*", "."]),
    ([("a", "b"), ("c", "d")], ["a", "b", ".", "c", "d", ".", "*"])
])
def test_rpn2splitter(splitter, rpn):
    assert aux.rpn2splitter(rpn) == splitter

@pytest.mark.xfail
@pytest.mark.parametrize("splitter, other_splitters, rpn",[
    (["a", "_NA"], {"NA": ("b", "c")}, ["a", "NA.b", "NA.c", ".", "*"]),
    (["_NA", "c"], {"NA": ("a", "b")}, ["NA.a", "NA.b", ".", "c", "*"]),
    (["a", ("b", "_NA")], {"NA": ["c", "d"]}, ["a", "b", "NA.c", "NA.d", "*", ".", "*"])
])
def test_splitter2rpn_wf_splitter(splitter, other_splitters, rpn):
    assert aux.splitter2rpn(splitter, other_splitters=other_splitters) == rpn


@pytest.mark.xfail
@pytest.mark.parametrize("splitter, splitter_changed",[
    ("a", "Node.a"),
    (["a", ("b", "c")], ["Node.a", ("Node.b", "Node.c")]),
    (("a", ["b", "c"]), ("Node.a", ["Node.b", "Node.c"]))
])
def test_change_splitter(splitter, splitter_changed):
    assert aux.change_splitter(splitter, "Node") == splitter_changed


@pytest.mark.xfail
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


@pytest.mark.xfail
def test_splitting_axis_error():
    with pytest.raises(Exception):
        aux.splitting_axis({"a": np.array([1, 2]), "b": np.array([3, 4, 5])}, ["a", "b", "."])


@pytest.mark.xfail
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


@pytest.mark.xfail
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


@pytest.mark.parametrize("splitter_rpn, input_to_remove, final_splitter_rpn", [
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
    (["a", "b", ".", "c", "d", ".", "*"], ["a", "b"], ["c", "d", "."])
])
def test_remove_inp_from_splitter_rpn(splitter_rpn, input_to_remove, final_splitter_rpn):
    assert aux.remove_inp_from_splitter_rpn(splitter_rpn, input_to_remove) ==\
           final_splitter_rpn
