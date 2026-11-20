"""Arrays whose dtype holds references are digested by content at any depth (pydra issue
#942, RFC 0004 section 6.2)."""

import numpy as np
import pytest

from pydra.utils import digest as dg
from pydra.utils.digest import digest
from pydra.utils.hash import Cache, hash_function, hash_object

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


def test_shared_cache_keeps_the_element_temporaries_alive():
    """One ``Cache`` serves every input of a task; it is keyed by ``id``, so the element
    lists made while hashing an array must outlive the call"""
    dt = np.dtype([("x", object), ("n", "<i4")])
    cache = Cache()
    arrays = [  # held, so that only the element temporaries can lose their ``id``
        np.array([(_fresh(f"value-{i}"), i)], dtype=dt) for i in range(3000)
    ]
    hashes = {hash_object(a, cache=cache) for a in arrays}
    assert len(hashes) == 3000


@pytest.mark.parametrize("wrap", [lambda a: a] + WRAPS)
def test_masked_reference_array_digests_its_mask(wrap):
    hidden = np.ma.array(["x", "a"], mask=[True, False], dtype=object)
    shown_none = np.ma.array([None, "a"], mask=[False, False], dtype=object)
    shown = np.ma.array(["x", "a"], mask=[False, False], dtype=object)
    assert digest(wrap(hidden)) != digest(wrap(shown_none))
    assert digest(wrap(hidden)) != digest(wrap(shown))
    assert digest(wrap(hidden)) == digest(
        wrap(np.ma.array(["x", "a"], mask=[True, False], dtype=object))
    )


def test_masked_plain_array_digests_its_mask():
    masked = np.ma.array([999999, 2], mask=[True, False])
    other = np.ma.array([5, 2], mask=[True, False])
    unmasked = np.ma.array([999999, 2])
    assert digest([masked]) != digest([unmasked])
    assert hash_function(masked) != hash_function(unmasked)
    assert hash_function(masked) != hash_function(np.ma.array([5, 2]))
    assert hash_function(other) != hash_function(np.ma.array([5, 2]))
