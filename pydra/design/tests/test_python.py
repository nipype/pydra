from operator import attrgetter
import typing as ty
from decimal import Decimal
import attrs
import pytest
from pydra.engine.helpers import list_fields
from pydra.engine.specs import PythonSpec
from pydra.design import python
from pydra.engine.task import PythonTask


sort_key = attrgetter("name")


def test_interface_wrap_function():
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleSpec = python.define(func)

    assert issubclass(SampleSpec, PythonSpec)
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [python.out(name="out", type=float)]
    spec = SampleSpec(a=1)
    result = spec()
    assert result.output.out == 2.0
    with pytest.raises(TypeError):
        SampleSpec(a=1.5)


def test_interface_wrap_function_with_default():
    def func(a: int, k: float = 2.0) -> float:
        """Sample function with inputs and outputs"""
        return a * k

    SampleSpec = python.define(func)

    assert issubclass(SampleSpec, PythonSpec)
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, default=func),
        python.arg(name="k", type=float, default=2.0),
    ]
    assert outputs == [python.out(name="out", type=float)]
    assert SampleSpec(a=1)().output.out == 2.0
    assert SampleSpec(a=10, k=3.0)().output.out == 30.0


def test_interface_wrap_function_overrides():
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleSpec = python.define(
        func,
        inputs={"a": python.arg(help_string="The argument to be doubled")},
        outputs={"b": python.out(help_string="the doubled output", type=Decimal)},
    )

    assert issubclass(SampleSpec, PythonSpec)
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="The argument to be doubled"),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [
        python.out(name="b", type=Decimal, help_string="the doubled output"),
    ]
    outputs = SampleSpec.Outputs(b=Decimal(2.0))
    assert isinstance(outputs.b, Decimal)


def test_interface_wrap_function_types():
    def func(a: int) -> int:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleSpec = python.define(
        func,
        inputs={"a": float},
        outputs={"b": float},
    )

    assert issubclass(SampleSpec, PythonSpec)
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=float),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [python.out(name="b", type=float)]
    intf = SampleSpec(a=1)
    assert isinstance(intf.a, float)
    outputs = SampleSpec.Outputs(b=2.0)
    assert isinstance(outputs.b, float)


def test_decorated_function_interface():
    @python.define(outputs=["c", "d"])
    def SampleSpec(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        return a + b, a * b

    assert issubclass(SampleSpec, PythonSpec)
    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="b", type=float),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float),
        python.out(name="d", type=float),
    ]
    assert attrs.fields(SampleSpec).function.default.__name__ == "SampleSpec"
    SampleSpec.Outputs(c=1.0, d=2.0)


def test_interface_with_function_implicit_outputs_from_return_stmt():
    @python.define
    def SampleSpec(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        c = a + b
        d = a * b
        return c, d

    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="b", type=float),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float),
        python.out(name="d", type=float),
    ]
    assert attrs.fields(SampleSpec).function.default.__name__ == "SampleSpec"
    SampleSpec.Outputs(c=1.0, d=2.0)


def test_interface_with_function_docstr():
    @python.define(outputs=["c", "d"])
    def SampleSpec(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        :param a: First input to be inputted
        :param b: Second input
        :return c: Sum of a and b
        :return d: product of a and b
        """
        return a + b, a * b

    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="product of a and b"),
    ]
    assert attrs.fields(SampleSpec).function.default.__name__ == "SampleSpec"


def test_interface_with_function_google_docstr():
    @python.define(outputs=["c", "d"])
    def SampleSpec(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        Args:
            a: First input
                to be inputted
            b: Second input

        Returns:
            c: Sum of a and b
            d: Product of a and b
        """
        return a + b, a * b

    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert attrs.fields(SampleSpec).function.default.__name__ == "SampleSpec"


def test_interface_with_function_numpy_docstr():
    @python.define(
        outputs=["c", "d"]
    )  # Could potentiall read output names from doc-string instead
    def SampleSpec(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        Parameters
        ----------
        a: int
            First input
            to be inputted
        b: float
            Second input

        Returns
        -------
        c : int
            Sum of a and b
        d : float
            Product of a and b
        """
        return a + b, a * b

    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert attrs.fields(SampleSpec).function.default.__name__ == "SampleSpec"


def test_interface_with_class():
    @python.define
    class SampleSpec:
        """Sample class for testing

        Args:
            a: First input
                to be inputted
            b: Second input
        """

        a: int
        b: float = 2.0

        class Outputs:
            """
            Args:
                c: Sum of a and b
                d: Product of a and b
            """

            c: float
            d: float

        @staticmethod
        def function(a, b):
            return a + b, a * b

    assert issubclass(SampleSpec, PythonSpec)
    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, default=2.0, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleSpec.function.__name__ == "function"
    SampleSpec(a=1)
    SampleSpec(a=1, b=2.0)
    SampleSpec.Outputs(c=1.0, d=2.0)


def test_interface_with_inheritance():
    @python.define
    class SampleSpec(PythonSpec["SampleSpec.Outputs"]):
        """Sample class for testing

        Args:
            a: First input
                to be inputted
            b: Second input
        """

        a: int
        b: float

        class Outputs:
            """
            Args:
                c: Sum of a and b
                d: Product of a and b
            """

            c: float
            d: float

        @staticmethod
        def function(a, b):
            return a + b, a * b

    assert issubclass(SampleSpec, PythonSpec)


def test_interface_with_class_no_auto_attribs():
    @python.define(auto_attribs=False)
    class SampleSpec:
        a: int = python.arg(help_string="First input to be inputted")
        b: float = python.arg(help_string="Second input")

        x: int

        class Outputs:
            c: float = python.out(help_string="Sum of a and b")
            d: float = python.out(help_string="Product of a and b")

            y: str

        @staticmethod
        def function(a, b):
            return a + b, a * b

    assert SampleSpec.Task is PythonTask
    inputs = sorted(list_fields(SampleSpec), key=sort_key)
    outputs = sorted(list_fields(SampleSpec.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleSpec).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleSpec.function.__name__ == "function"
    SampleSpec(a=1, b=2.0)
    SampleSpec.Outputs(c=1.0, d=2.0)
    with pytest.raises(TypeError):
        SampleSpec(a=1, b=2.0, x=3)
    with pytest.raises(TypeError):
        SampleSpec.Outputs(c=1.0, d=2.0, y="hello")


def test_interface_invalid_wrapped1():
    with pytest.raises(ValueError):

        @python.define(inputs={"a": python.arg()})
        class SampleSpec(PythonSpec["SampleSpec.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1


def test_interface_invalid_wrapped2():
    with pytest.raises(ValueError):

        @python.define(outputs={"b": python.out()})
        class SampleSpec(PythonSpec["SampleSpec.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1
