"""The runtime reach check of ``pydracli check-reach`` (RFC 0004 section 4.2).

A development tool: it runs a task's function once under `sys.setprofile` and lists the
user functions that ran but that the code walk of `pydra.utils.hash` did not reach, which
are the places where ``getattr``, ``importlib`` or a registry hides a dependency. It is
never used while a workflow runs."""

import sys
import types
import typing as ty
from collections.abc import Mapping

from pydra.utils.hash import (
    Cache,
    _class_members,
    _is_unowned_python,
    _package_override,
    _path_is_library,
    _module_is_library,
    check_code_hash,
    code_hash_default,
)
from pydra.utils.general import get_fields

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task


def _codes(code: types.CodeType) -> ty.Iterator[types.CodeType]:
    """``code`` and the code objects nested in it (inner functions, lambdas, comprehensions)"""
    yield code
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            yield from _codes(const)


def reached_codes(
    func: types.FunctionType, code_hash: str | None = None
) -> set[types.CodeType]:
    """The code objects of the definitions the walk of ``func`` reaches at the
    ``code_hash`` tier, with those nested in them"""
    from pydra.utils.hash import _CodeWalk

    cache = Cache(code_hash=check_code_hash(code_hash or code_hash_default()))
    walk = _CodeWalk(func, cache, defaults=True)
    for _ in walk:
        pass
    codes: set[types.CodeType] = set()
    for obj in walk.definitions:
        if isinstance(obj, type):
            funcs = [f for _, f in _class_members(obj)]
        else:
            funcs = [obj]
        for f in funcs:
            codes.update(_codes(f.__code__))
    return codes


def _is_user_code(module: str, file: str) -> bool:
    """Whether code of ``module`` in ``file`` is walked as user code (the rule of
    `pydra.utils.hash.is_library` for a function)"""
    override = _package_override(module)
    if override is None:
        override = _path_is_library(file)
    if override is None:
        override = _module_is_library(module, file)
    return not override or _is_unowned_python(module)


def _task_function(task: "type[Task]") -> types.FunctionType:
    field = get_fields(task).get(getattr(task, "_executor_name", None) or "")
    func = getattr(field, "default", None)
    if not isinstance(func, types.FunctionType):
        raise TypeError(f"{task!r} is not a Python task class")
    return func


def check_reach(task: "type[Task]", inputs: Mapping[str, ty.Any]) -> list[str]:
    """The user functions that ran when the function of ``task`` was called with
    ``inputs`` but that the code walk did not reach, each as
    ``"<module>.<qualname> (<file>:<line>)"``, sorted. The function is called directly,
    in this process, and its outputs are discarded."""
    func = _task_function(task)
    reached = reached_codes(func, getattr(task, "_code_hash", None))
    ran: dict[types.CodeType, str] = {}

    def profiler(frame: types.FrameType, event: str, arg: ty.Any) -> None:
        if event != "call":
            return
        code = frame.f_code
        if code in reached or code in ran or code.co_name == "<module>":
            return
        module = frame.f_globals.get("__name__", "")
        if _is_user_code(module, code.co_filename):
            ran[code] = (
                f"{module}.{code.co_qualname} ({code.co_filename}:{code.co_firstlineno})"
            )

    previous = sys.getprofile()
    sys.setprofile(profiler)
    try:
        func(**inputs)
    finally:
        sys.setprofile(previous)
    return sorted(ran.values())
