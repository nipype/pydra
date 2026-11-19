"""Arrays whose dtype holds references are digested by content at any depth (pydra issue
#942, RFC 0004 section 6.2)."""

import numpy as np
import pytest

from pydra.utils import digest as dg
from pydra.utils.digest import digest
from pydra.utils.hash import hash_function

STRING_DTYPE = getattr(getattr(np, "dtypes", None), "StringDType", None)
needs_stringdtype = pytest.mark.skipif(
    STRING_DTYPE is None, reason="numpy without StringDType"
)


def as_list(a):
    return [a]


def as_tuple(a):
    return (a, 1)


def as_dict_value(a):
    return {"k": a}


def as_nested(a):
    return {"k": [(a,)]}


WRAPS = [as_list, as_tuple, as_dict_value, as_nested]


def _fresh(text):
    """An equal string that is not the same object"""
    return "".join(list(text))


@needs_stringdtype
@pytest.mark.parametrize("wrap", WRAPS)
def test_nested_stringdtype_digests_by_content(wrap):
    a = np.array(["a" * 60], dtype=STRING_DTYPE())
    b = np.array(["b" * 60], dtype=STRING_DTYPE())
    same = np.array([_fresh("a" * 60)], dtype=STRING_DTYPE())
    assert digest(wrap(a)) != digest(wrap(b))
    assert digest(wrap(a)) == digest(wrap(same))
    assert hash_function(wrap(a)) != hash_function(wrap(b))
    assert digest(wrap(a)) != digest(wrap(a.reshape(1, 1)))


@pytest.mark.parametrize("wrap", WRAPS)
def test_nested_structured_object_field_digests_by_content(wrap):
    dt = np.dtype([("x", object), ("n", "<i4")])
    a = np.array([(_fresh("ab" * 30), 1)], dtype=dt)
    same = np.array([(_fresh("ab" * 30), 1)], dtype=dt)
    other = np.array([(_fresh("ac" * 30), 1)], dtype=dt)
    assert digest(wrap(a)) == digest(wrap(same))
    assert digest(wrap(a)) != digest(wrap(other))
    assert hash_function(wrap(a)) == hash_function(wrap(same))


@pytest.mark.parametrize("wrap", WRAPS)
def test_nested_object_array_digests_by_content(wrap):
    a = np.array([_fresh("ab" * 30), {"k": 1}], dtype=object)
    same = np.array([_fresh("ab" * 30), {"k": 1}], dtype=object)
    other = np.array([_fresh("ab" * 30), {"k": 2}], dtype=object)
    assert digest(wrap(a)) == digest(wrap(same))
    assert digest(wrap(a)) != digest(wrap(other))


def test_object_array_shape_is_part_of_the_digest():
    flat = np.array(["a", "b", "c", "d", "e", "f"], dtype=object)
    assert digest([flat.reshape(2, 3)]) != digest([flat.reshape(3, 2)])
    assert digest(flat.reshape(2, 3)) != digest(flat.reshape(3, 2))


def test_reference_arrays_are_the_same_at_every_depth():
    dt = np.dtype([("x", object)])
    a = np.array([(_fresh("abc"),)], dtype=dt)
    assert hash_function(a) == hash_function(np.array([(_fresh("abc"),)], dtype=dt))
    assert hash_function([a]) != hash_function(a)
    assert digest(a).startswith("py@1:")


def test_nested_reference_array_golden():
    dt = np.dtype([("x", object), ("n", "<i4")])
    a = np.array([("abc", 1), ("de", 2)], dtype=dt)
    assert digest({"k": [a]}) == GOLDEN_STRUCTURED


GOLDEN_STRUCTURED = "py@1:2bdc44d3397c54c72b944ddff87662c6"
