"""Merging routes keyed on their gates (RFC 0002 section 2.3(c)).

``select(n)`` builds a Python task class with inputs ``when_0 … when_{n-1}`` (``bool``)
and ``value_0 … value_{n-1}``, every one of them ``absent="accept"``. It returns
``value_k`` for the first ``k`` whose ``when_k`` is true, whatever that value is (a
routed branch's content ``Absent`` reaches the reader with its cause), and
``Absent("not-routed")`` if none is. The module is itself callable::

    from pydra.compose import select

    merged = workflow.add(
        select(2)(when_0=pos.out, value_0=up.out, when_1=neg.out, value_1=down.out)
    )
"""

import functools
import importlib
import sys
import types
import typing as ty

from pydra.compose import python

__all__ = ["select"]


def _select(**inputs: ty.Any) -> ty.Any:
    """The value of the first route whose gate is true, else ``Absent("not-routed")``"""
    from pydra.engine.absent import NOT_ROUTED, Absent

    for k in range(len(inputs) // 2):
        if inputs[f"when_{k}"] is True:
            return inputs[f"value_{k}"]
    return Absent(NOT_ROUTED, source="select")


@functools.cache
def select(n: int) -> "type[python.Task]":
    """The task class merging ``n`` routes on their gates

    Parameters
    ----------
    n : int
        The number of routes, at least 1

    Returns
    -------
    type[python.Task]
        A Python task class with inputs ``when_0 … when_{n-1}`` and
        ``value_0 … value_{n-1}`` and output ``out``; one class per arity
    """
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"select(n) needs a positive int, not {n!r}")
    inputs: dict[str, python.arg] = {}
    for k in range(n):
        inputs[f"when_{k}"] = python.arg(type=bool, absent="accept")
    for k in range(n):
        inputs[f"value_{k}"] = python.arg(type=ty.Any, absent="accept")
    return python.define(
        _select,
        inputs=inputs,
        outputs={"out": ty.Any},
        name=f"Select{n}",
    )


class _CallableModule(types.ModuleType):
    """Lets ``pydra.compose.select`` be called as ``select(n)``.

    ``pydra.compose`` is a namespace package that other distributions extend, so it has
    no ``__init__`` that could bind the function there; the module's class is swapped
    instead. ``from pydra.compose.select import select`` also works.
    """

    def __call__(self, n: int) -> "type[python.Task]":
        return select(n)

    def __reduce__(self):
        # pickled by reference, as a plain module is
        return importlib.import_module, (self.__name__,)


sys.modules[__name__].__class__ = _CallableModule
