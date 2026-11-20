"""The runtime reach check of ``pydracli check-reach`` (RFC 0004 section 4.2).

A development tool: it runs a Python task once under `sys.setprofile` and lists the
user functions that ran but that the code walk of `pydra.utils.hash` did not reach,
which are the places where ``getattr``, ``importlib`` or a registry hides a
dependency. It is never used while a workflow runs."""

import sys
import threading
import types
import typing as ty
from collections.abc import Mapping

from pydra.utils.general import asdict
from pydra.utils.hash import is_user_code, reached_definitions

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task


def _describe(code: types.CodeType, globals_: Mapping[str, ty.Any]) -> str:
    module = globals_.get("__name__", "")
    return f"{module}.{code.co_qualname} ({code.co_filename}:{code.co_firstlineno})"


def reach_report(
    task: "type[Task]", inputs: Mapping[str, ty.Any]
) -> tuple[list[str], list[str]]:
    """``(missed, by_name)``: the user functions that ran when ``task`` was built from
    ``inputs`` (validated and coerced as a task validates them) and its function called
    with the resolved values, but that the code walk did not reach, and those among
    them that the walk keys by name only at the ``function`` and ``module`` tiers, each
    as ``"<module>.<qualname> (<file>:<line>)"``, sorted. Everything runs in this
    process; code run in a thread is profiled, code run in another process is not.

    Raises:
        TypeError: ``task`` is not a Python task class
        Exception: whatever building the task or the function raises"""
    if not (
        isinstance(task, type) and getattr(task, "_executor_name", None) == "function"
    ):
        raise TypeError(f"{task!r} is not a Python task class")
    instance = task(**inputs)
    func = instance.function
    values = asdict(instance)
    del values["function"]
    keyed, named = reached_definitions(func, values, getattr(task, "_code_hash", None))
    missed: dict[types.CodeType, str] = {}
    by_name: dict[types.CodeType, str] = {}
    under_name: dict[int, int] = {}  # per thread, the depth of the frames of named code

    def profiler(frame: types.FrameType, event: str, arg: ty.Any) -> None:
        code = frame.f_code
        thread = threading.get_ident()
        if event == "return":
            if code in named and under_name.get(thread):
                under_name[thread] -= 1
            return
        if event != "call":
            return
        inside = under_name.get(thread, 0) > 0
        if code in named:
            under_name[thread] = under_name.get(thread, 0) + 1
        if code in keyed or code in missed or code in by_name:
            return
        if code.co_name == "<module>" or not is_user_code(code, frame.f_globals):
            return
        text = _describe(code, frame.f_globals)
        if code in named or inside:
            by_name[code] = text
        else:
            missed[code] = text

    previous = sys.getprofile()
    thread_previous = getattr(threading, "_profile_hook", None)
    set_all = getattr(threading, "setprofile_all_threads", None)
    (set_all or threading.setprofile)(profiler)
    sys.setprofile(profiler)
    try:
        func(**values)
    finally:
        (set_all or threading.setprofile)(thread_previous)
        sys.setprofile(previous)
    return sorted(missed.values()), sorted(by_name.values())


def check_reach(task: "type[Task]", inputs: Mapping[str, ty.Any]) -> list[str]:
    """The user functions that ran but that the code walk did not reach (see
    `reach_report`)"""
    return reach_report(task, inputs)[0]
