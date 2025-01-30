import pytest
import random
import typing as ty
import inspect
import re
import ast
from pydra.design import python
from pydra.engine.specs import PythonDef, PythonOutputs
from pydra.engine.helpers import list_fields, attrs_values
from pydra.utils.hash import bytes_repr


def test_task_equivalence():
    """testing equivalence of tasks created in different ways"""

    def add_two(a: int) -> int:
        return a + 2

    @python.define
    class Canonical(PythonDef["Canonical.Outputs"]):

        a: ty.Any

        class Outputs(PythonOutputs):
            out: ty.Any

        @staticmethod
        def function(a: int) -> int:
            return a + 2

    canonical = Canonical(a=3)

    decorated1 = python.define(add_two)(a=3)

    @python.define
    def addtwo(a: int) -> int:
        return a + 2

    decorated2 = addtwo(a=3)

    assert canonical._compute_hashes()[1] == decorated1._compute_hashes()[1]
    assert canonical._compute_hashes()[1] == decorated2._compute_hashes()[1]

    c_outputs = canonical()
    d1_outputs = decorated1()
    d2_outputs = decorated2()

    assert attrs_values(c_outputs) == attrs_values(d1_outputs)
    assert attrs_values(c_outputs) == attrs_values(d2_outputs)


def test_annotation_equivalence_1():
    """testing various ways of annotation: one output, only types provided"""

    def direct(a: int) -> int:
        return a + 2

    Direct = python.define(direct)

    @python.define(outputs={"out": int})
    def Partial(a: int):
        return a + 2

    @python.define(inputs={"a": int}, outputs={"out": int})
    def Indirect(a):
        return a + 2

    assert list_fields(Direct) == list_fields(Partial)
    assert list_fields(Direct) == list_fields(Indirect)

    assert list_fields(Direct.Outputs) == list_fields(Partial.Outputs)
    assert list_fields(Direct.Outputs) == list_fields(Indirect.Outputs)

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert attrs_values(Direct(a)) == attrs_values(Partial(a))
    assert attrs_values(Direct(a)) == attrs_values(Indirect(a))

    # checking if the annotation is properly converted to output_spec if used in task
    assert list_fields(Direct.Outputs)[0] == python.out(name="out", type=int)


def test_annotation_equivalence_2():
    """testing various ways of annotation: multiple outputs, using a tuple for output annot."""

    def direct(a: int) -> (int, float):
        return a + 2, a + 2.0

    @python.define(outputs={"out": (int, float)})
    def partial(a: int):
        return a + 2, a + 2.0

    @python.define(inputs={"a": int})
    def indirect(a) -> tuple[int, float]:
        return a + 2, a + 2.0

    # checking if the annotations are equivalent
    assert direct.__annotations__ == partial.__annotations__
    assert direct.__annotations__ == indirect.__annotations__

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert direct(a) == partial(a)
    assert direct(a) == indirect(a)

    # checking if the annotation is properly converted to output_spec if used in task
    task_direct = python.define(direct)()
    assert task_direct.output_spec.fields == [("out1", int), ("out2", float)]


def test_annotation_equivalence_3():
    """testing various ways of annotation: using dictionary for output annot."""

    @python.define(outputs=["out1"])
    def direct(a: int) -> int:
        return a + 2

    @python.define(inputs={"return": {"out1": int}})
    def partial(a: int):
        return a + 2

    @python.define(inputs={"a": int}, outputs={"out1": int})
    def indirect(a):
        return a + 2

    # checking if the annotations are equivalent
    assert direct.__annotations__ == partial.__annotations__
    assert direct.__annotations__ == indirect.__annotations__

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert direct(a) == partial(a)
    assert direct(a) == indirect(a)

    # checking if the annotation is properly converted to output_spec if used in task
    task_direct = python.define(direct)()
    assert task_direct.output_spec.fields[0] == ("out1", int)


def test_annotation_equivalence_4():
    """testing various ways of annotation: using ty.NamedTuple for the output"""

    @python.define(outputs=["sum", "sub"])
    def direct(a: int) -> tuple[int, int]:
        return a + 2, a - 2

    @python.define(outputs={"sum": int, "sub": int})
    def partial(a: int):
        return a + 2, a - 2

    @python.define(inputs={"a": int}, outputs={"sum": int, "sub": int})
    def indirect(a):
        return a + 2, a - 2

    # checking if the annotations are equivalent
    assert (
        direct.__annotations__["return"].__annotations__
        == partial.__annotations__["return"].__annotations__
        == indirect.__annotations__["return"].__annotations__
    )
    assert (
        direct.__annotations__["return"].__name__
        == partial.__annotations__["return"].__name__
        == indirect.__annotations__["return"].__name__
    )

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert direct(a) == partial(a)
    assert direct(a) == indirect(a)

    # checking if the annotation is properly converted to output_spec if used in task
    task_direct = python.define(direct)()
    assert list_fields(task_direct.Outputs) == [
        python.arg(name="sum", type=int),
        python.arg(name="sub", type=int),
    ]


def test_invalid_annotation():
    with pytest.raises(TypeError):

        @python.define(inputs={"b": int})
        def addtwo(a):
            return a + 2


def test_annotated_task():

    def square(in_val: float):
        return in_val**2

    res = square(in_val=2.0)()
    assert res.output.out == 4.0


def test_return_annotated_task():

    @python.define(inputs={"in_val": float}, outputs={"squared": float})
    def square(in_val):
        return in_val**2

    res = square(in_val=2.0)()
    assert res.output.squared == 4.0


def test_return_halfannotated_annotated_task():

    @python.define(inputs={"in_val": float}, outputs={"out": float})
    def square(in_val):
        return in_val**2

    res = square(in_val=2.0)()
    assert res.output.out == 4.0


def test_return_annotated_task_multiple_output():

    @python.define(inputs={"in_val": float}, outputs={"squared": float, "cubed": float})
    def square(in_val):
        return in_val**2, in_val**3

    res = square(in_val=2.0)()
    assert res.output.squared == 4.0
    assert res.output.cubed == 8.0


def test_return_halfannotated_task_multiple_output():

    @python.define(inputs={"in_val": float}, outputs=(float, float))
    def square(in_val):
        return in_val**2, in_val**3

    res = square(in_val=2.0)()
    assert res.output.out1 == 4.0
    assert res.output.out2 == 8.0
