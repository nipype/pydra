from operator import attrgetter
import typing as ty
from decimal import Decimal
import attrs
import pytest
from pydra.design import list_fields, Interface
from pydra.design import python
from pydra.engine.task import FunctionTask


sort_key = attrgetter("name")


def test_interface_wrap_function():
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleInterface = python.interface(func)

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [python.out(name="out", type=float)]


def test_interface_wrap_function_with_default():
    def func(a: int, k: float = 2.0) -> float:
        """Sample function with inputs and outputs"""
        return a * k

    SampleInterface = python.interface(func)

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, default=func),
        python.arg(name="k", type=float, default=2.0),
    ]
    assert outputs == [python.out(name="out", type=float)]


def test_interface_wrap_function_overrides():
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleInterface = python.interface(
        func,
        inputs={"a": python.arg(help_string="The argument to be doubled")},
        outputs={"b": python.out(help_string="the doubled output", type=Decimal)},
    )

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="The argument to be doubled"),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [
        python.out(name="b", type=Decimal, help_string="the doubled output"),
    ]


def test_interface_wrap_function_types():
    def func(a: int) -> int:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleInterface = python.interface(
        func,
        inputs={"a": float},
        outputs={"b": float},
    )

    assert issubclass(SampleInterface, Interface)
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=float),
        python.arg(name="function", type=ty.Callable, default=func),
    ]
    assert outputs == [python.out(name="b", type=float)]


def test_decorated_function_interface():
    @python.interface(outputs=["c", "d"])
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        return a + b, a * b

    assert issubclass(SampleInterface, Interface)
    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="b", type=float),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float),
        python.out(name="d", type=float),
    ]


def test_interface_with_function_implicit_outputs_from_return_stmt():
    @python.interface
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        c = a + b
        d = a * b
        return c, d

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="b", type=float),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float),
        python.out(name="d", type=float),
    ]


def test_interface_with_function_docstr():
    @python.interface(outputs=["c", "d"])
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        :param a: First input to be inputted
        :param b: Second input
        :return c: Sum of a and b
        :return d: product of a and b
        """
        return a + b, a * b

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="product of a and b"),
    ]


def test_interface_with_function_google_docstr():
    @python.interface(outputs=["c", "d"])
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
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

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]


def test_interface_with_function_numpy_docstr():
    @python.interface(
        outputs=["c", "d"]
    )  # Could potentiall read output names from doc-string instead
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
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

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]


def test_interface_with_class():
    @python.interface
    class SampleInterface:
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

    assert issubclass(SampleInterface, Interface)
    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, default=2.0, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "function"


def test_interface_with_inheritance():
    @python.interface
    class SampleInterface(Interface["SampleInterface.Outputs"]):
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

    assert issubclass(SampleInterface, Interface)


def test_interface_with_class_no_auto_attribs():
    @python.interface(auto_attribs=False)
    class SampleInterface:
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

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(list_fields(SampleInterface), key=sort_key)
    outputs = sorted(list_fields(SampleInterface.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help_string="First input to be inputted"),
        python.arg(name="b", type=float, help_string="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            default=attrs.fields(SampleInterface).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help_string="Sum of a and b"),
        python.out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "function"


def test_interface_invalid_wrapped1():
    with pytest.raises(ValueError):

        @python.interface(inputs={"a": python.arg()})
        class SampleInterface(Interface["SampleInterface.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1


def test_interface_invalid_wrapped2():
    with pytest.raises(ValueError):

        @python.interface(outputs={"b": python.out()})
        class SampleInterface(Interface["SampleInterface.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1
