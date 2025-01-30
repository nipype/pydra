import pytest
import random
import typing as ty
from pydra.design.base import Field
from pydra.design import python
from pydra.engine.specs import PythonDef, PythonOutputs
from pydra.engine.helpers import list_fields, attrs_values


def non_func_fields(defn: PythonDef) -> list[Field]:
    return [f for f in list_fields(defn) if f.name != "function"]


def non_func_values(defn: PythonDef) -> dict:
    return {n: v for n, v in attrs_values(defn).items() if n != "function"}


def hashes(defn: PythonDef) -> dict[str, str]:
    return defn._compute_hashes()[1]


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

    assert (
        non_func_values(c_outputs)
        == non_func_values(d1_outputs)
        == non_func_values(d2_outputs)
    )


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

    assert non_func_fields(Direct) == non_func_fields(Partial)
    assert non_func_fields(Direct) == non_func_fields(Indirect)

    assert list_fields(Direct.Outputs) == list_fields(Partial.Outputs)
    assert list_fields(Direct.Outputs) == list_fields(Indirect.Outputs)

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert non_func_values(Direct(a=a)) == non_func_values(Partial(a=a))
    assert non_func_values(Direct(a=a)) == non_func_values(Indirect(a=a))

    # checking if the annotation is properly converted to output_spec if used in task
    assert list_fields(Direct.Outputs)[0] == python.out(name="out", type=int, order=0)


def test_annotation_equivalence_2():
    """testing various ways of annotation: multiple outputs, using a tuple for output annot."""

    def direct(a: int) -> tuple[int, float]:
        return a + 2, a + 2.0

    Direct = python.define(direct, outputs=["out1", "out2"])

    @python.define(outputs={"out1": int, "out2": float})
    def Partial(a: int):
        return a + 2, a + 2.0

    @python.define(inputs={"a": int}, outputs=["out1", "out2"])
    def Indirect(a) -> tuple[int, float]:
        return a + 2, a + 2.0

    # checking if the annotations are equivalent
    assert (
        non_func_fields(Direct) == non_func_fields(Partial) == non_func_fields(Indirect)
    )

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert hashes(Direct(a=a)) == hashes(Partial(a=a)) == hashes(Indirect(a=a))

    # checking if the annotation is properly converted to output_spec if used in task
    assert list_fields(Direct.Outputs) == [
        python.out(name="out1", type=int, order=0),
        python.out(name="out2", type=float, order=1),
    ]


def test_annotation_equivalence_3():
    """testing various ways of annotation: using dictionary for output annot."""

    def direct(a: int) -> int:
        return a + 2

    Direct = python.define(direct, outputs=["out1"])

    @python.define(outputs={"out1": int})
    def Partial(a: int):
        return a + 2

    @python.define(inputs={"a": int}, outputs={"out1": int})
    def Indirect(a):
        return a + 2

    # checking if the annotations are equivalent
    assert (
        non_func_fields(Direct) == non_func_fields(Partial) == non_func_fields(Indirect)
    )

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert hashes(Direct(a=a)) == hashes(Partial(a=a)) == hashes(Indirect(a=a))

    # checking if the annotation is properly converted to output_spec if used in task
    assert list_fields(Direct.Outputs)[0] == python.out(name="out1", type=int, order=0)


def test_annotation_equivalence_4():
    """testing various ways of annotation: using ty.NamedTuple for the output"""

    @python.define(outputs=["sum", "sub"])
    def Direct(a: int) -> tuple[int, int]:
        return a + 2, a - 2

    @python.define(outputs={"sum": int, "sub": int})
    def Partial(a: int):
        return a + 2, a - 2

    @python.define(inputs={"a": int}, outputs={"sum": int, "sub": int})
    def Indirect(a):
        return a + 2, a - 2

    # checking if the annotations are equivalent
    assert (
        list_fields(Direct.Outputs)
        == list_fields(Partial.Outputs)
        == list_fields(Indirect.Outputs)
    )
    assert (
        list_fields(Direct.Outputs)
        == list_fields(Partial.Outputs)
        == list_fields(Indirect.Outputs)
    )

    # Run functions to ensure behavior is unaffected
    a = random.randint(0, (1 << 32) - 3)
    assert Direct(a=a) == Partial(a=a)
    assert Direct(a=a) == Indirect(a=a)

    # checking if the annotation is properly converted to output_spec if used in task
    assert list_fields(Direct.Outputs) == [
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
