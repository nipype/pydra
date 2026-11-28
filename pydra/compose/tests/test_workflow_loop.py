"""``workflow.loop``: the task class of a runtime while-loop (RFC 0002 section 2.3(d),
R11). These tests cover the class the factory generates and its construction checks;
the runner's growth of a loop is tested in ``pydra/engine/tests/test_loop_runner.py``.
"""

import typing as ty

import pytest

from pydra.compose import python, workflow
from pydra.engine.workflow import Workflow
from pydra.utils.general import get_fields, is_workflow

# -- bodies ------------------------------------------------------------------------------------------


@python.define(outputs=["x", "done"])
def Halve(x: int, target: int) -> tuple[int, bool]:
    y = x // 2
    return y, y <= target


@python.define(outputs=["y", "done"])
def NoCarry(x: int) -> tuple[int, bool]:
    return x, False


@python.define(outputs=["x", "label", "done", "small"])
def Labelled(x: int, n: int = 0) -> tuple[int, str, bool, bool]:
    return x - 1, str(x), x <= 1, x < 5


@python.define(outputs=["x", "rounds"])
def Reserved(x: int) -> tuple[int, bool]:
    return x, True


def _halver(divisor: int):
    """Two bodies named ``Body`` whose functions differ only in their source"""
    if divisor == 2:

        @python.define(outputs=["x", "done"])
        def Body(x: int) -> tuple[int, bool]:
            return x // 2, x < 2

    else:

        @python.define(outputs=["x", "done"])
        def Body(x: int) -> tuple[int, bool]:
            return x // 3, x < 2

    return Body


# -- the generated class -----------------------------------------------------------------------------


def test_loop_class_copies_body_fields():
    Loop = workflow.loop(Labelled, until="done", carry={"x": "x"}, index="n")
    assert is_workflow(Loop(x=3))
    inputs = {f.name: f.type for f in get_fields(Loop) if f.name != "constructor"}
    assert inputs == {"x": int}  # the index is the loop's, not an input
    outputs = {f.name: f.type for f in get_fields(Loop.Outputs)}
    assert outputs == {
        "x": int,
        "label": str,
        "done": bool,
        "small": bool,
        "rounds": int,
        "converged": bool,
        "stopped": str,
    }
    # one class per set of parameters
    assert workflow.loop(Labelled, until="done", carry={"x": "x"}, index="n") is Loop


def test_loop_construct_has_round_zero_only():
    wf = workflow.loop(Halve, until="done", max_rounds=50)(x=16, target=1).construct()
    assert wf.node_names == ["r0"]


def test_loop_construct_is_never_memoised():
    """The runner grows the constructed workflow, so no two frames may share one, not even
    after another caller has cached a construction of the same task"""
    task = workflow.loop(Halve, until="done")(x=16, target=1)
    assert task.construct() is not task.construct()
    cached = Workflow.construct(task)  # caches, as a graph export does
    assert task.construct() is not cached
    assert task.construct() is not task.construct()


# -- the loop job's checksum -------------------------------------------------------------------------


def test_loop_checksum_covers_body_and_parameters():
    def checksum(body=Halve, **kwargs):
        kwargs.setdefault("until", "done")
        inputs = {"x": 16, "target": 1} if body is Halve else {"x": 16}
        return workflow.loop(body, **kwargs)(**inputs)._checksum

    base = checksum()
    assert checksum() == base
    assert checksum(max_rounds=5) != base
    assert checksum(max_rounds=5) != checksum(max_rounds=6)
    # same name, different function source
    assert checksum(_halver(2)) != checksum(_halver(3))
    assert checksum(_halver(2)) == checksum(_halver(2))

    def labelled(**kwargs):
        kwargs.setdefault("until", "done")
        kwargs.setdefault("carry", {"x": "x"})
        return workflow.loop(Labelled, **kwargs)(x=3)._checksum

    assert labelled() != labelled(until="small")
    assert labelled(index="n") != labelled()
    assert labelled(carry={"n": "x"}, index=None) != labelled()


# -- construction checks -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"until": "x"}, ValueError, "until"),
        ({"until": "missing"}, ValueError, "until"),
        ({"index": "missing"}, ValueError, "index"),
        # a carried input cannot be the index
        ({"index": "x", "carry": {"x": "x"}}, ValueError, "index"),
        ({"max_rounds": 0}, ValueError, "max_rounds"),
        ({"max_rounds": True}, ValueError, "max_rounds"),
        ({"carry": {"z": "x"}}, ValueError, "carry"),
        ({"carry": {"x": "nope"}}, ValueError, "carry"),
        ({"carry": {"x": "label"}}, TypeError, "carry"),
    ],
)
def test_loop_refuses_bad_parameters(kwargs, error, match):
    kwargs.setdefault("until", "done")
    with pytest.raises(error, match=match) as exc:
        workflow.loop(Labelled, **kwargs)
    assert "Labelled" in str(exc.value)


def test_empty_carry_without_index_refused():
    """Nothing carried and no index: every round would be the same computation"""
    with pytest.raises(ValueError, match="carr"):
        workflow.loop(NoCarry, until="done")
    with pytest.raises(ValueError, match="carr"):
        workflow.loop(Labelled, until="done", carry={})
    # a retry or poll loop names an index
    workflow.loop(Labelled, until="done", carry={}, index="n")


def test_reserved_outputs_refused():
    with pytest.raises(ValueError, match="rounds"):
        workflow.loop(Reserved, until="rounds", carry={"x": "x"})


def test_body_must_be_a_task_class():
    with pytest.raises(TypeError, match="class"):
        workflow.loop(Halve(target=1).split(x=[1, 2]), until="done")
    with pytest.raises(TypeError, match="class"):
        workflow.loop(len, until="done")


def test_until_typed_bool_only():
    @python.define(outputs=["x", "done"])
    def Loose(x: int) -> tuple[int, ty.Any]:
        return x, True

    with pytest.raises(ValueError, match="bool"):
        workflow.loop(Loose, until="done")
