from operator import attrgetter
import typing as ty
from decimal import Decimal
import attrs
import pytest
from pydra.engine.helpers import list_fields
from pydra.engine.specs import PythonDef, PythonOutputs
from pydra.design import python


sort_key = attrgetter("name")


def test_interface_wrap_function(tmp_path):
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleDef = python.define(func)

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, hash_eq=True, default=func),
    ]
    assert outputs == [python.out(name="out", type=float)]
    definition = SampleDef(a=1)
    outputs = definition(cache_dir=tmp_path)
    assert outputs.out == 2.0
    with pytest.raises(TypeError):
        SampleDef(a=1.5)


def test_interface_wrap_function_with_default():
    def func(a: int, k: float = 2.0) -> float:
        """Sample function with inputs and outputs"""
        return a * k

    SampleDef = python.define(func)

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="function", type=ty.Callable, hash_eq=True, default=func),
        python.arg(name="k", type=float, default=2.0),
    ]
    assert outputs == [python.out(name="out", type=float)]
    assert SampleDef(a=1)().out == 2.0
    assert SampleDef(a=10, k=3.0)().out == 30.0


def test_interface_wrap_function_overrides():
    def func(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleDef = python.define(
        func,
        inputs={"a": python.arg(help="The argument to be doubled")},
        outputs={"b": python.out(help="the doubled output", type=Decimal)},
    )

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="The argument to be doubled"),
        python.arg(name="function", type=ty.Callable, hash_eq=True, default=func),
    ]
    assert outputs == [
        python.out(name="b", type=Decimal, help="the doubled output"),
    ]
    outputs = SampleDef.Outputs(b=Decimal(2.0))
    assert isinstance(outputs.b, Decimal)


def test_interface_wrap_function_types():
    def func(a: int) -> int:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleDef = python.define(
        func,
        inputs={"a": float},
        outputs={"b": float},
    )

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=float),
        python.arg(name="function", type=ty.Callable, hash_eq=True, default=func),
    ]
    assert outputs == [python.out(name="b", type=float)]
    intf = SampleDef(a=1)
    assert isinstance(intf.a, float)
    outputs = SampleDef.Outputs(b=2.0)
    assert isinstance(outputs.b, float)


def test_decorated_function_interface():
    @python.define(outputs=["c", "d"])
    def SampleDef(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        return a + b, a * b

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int),
        python.arg(name="b", type=float),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float),
        python.out(name="d", type=float),
    ]
    assert attrs.fields(SampleDef).function.default.__name__ == "SampleDef"
    SampleDef.Outputs(c=1.0, d=2.0)


def test_interface_with_function_docstr():
    @python.define(outputs=["c", "d"])
    def SampleDef(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        :param a: First input to be inputted
        :param b: Second input
        :return c: Sum of a and b
        :return d: product of a and b
        """
        return a + b, a * b

    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="First input to be inputted"),
        python.arg(name="b", type=float, help="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help="Sum of a and b"),
        python.out(name="d", type=float, help="product of a and b"),
    ]
    assert attrs.fields(SampleDef).function.default.__name__ == "SampleDef"


def test_interface_with_function_google_docstr():
    @python.define(outputs=["c", "d"])
    def SampleDef(a: int, b: float) -> tuple[float, float]:
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

    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="First input to be inputted"),
        python.arg(name="b", type=float, help="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help="Sum of a and b"),
        python.out(name="d", type=float, help="Product of a and b"),
    ]
    assert attrs.fields(SampleDef).function.default.__name__ == "SampleDef"


def test_interface_with_function_numpy_docstr():
    @python.define(
        outputs=["c", "d"]
    )  # Could potentiall read output names from doc-string instead
    def SampleDef(a: int, b: float) -> tuple[float, float]:
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

    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="First input to be inputted"),
        python.arg(name="b", type=float, help="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help="Sum of a and b"),
        python.out(name="d", type=float, help="Product of a and b"),
    ]
    assert attrs.fields(SampleDef).function.default.__name__ == "SampleDef"


def test_interface_with_class():
    @python.define
    class SampleDef(PythonDef["SampleDef.Outputs"]):
        """Sample class for testing

        Args:
            a: First input
                to be inputted
            b: Second input
        """

        a: int
        b: float = 2.0

        class Outputs(PythonOutputs):
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

    assert issubclass(SampleDef, PythonDef)
    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="First input to be inputted"),
        python.arg(name="b", type=float, default=2.0, help="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help="Sum of a and b"),
        python.out(name="d", type=float, help="Product of a and b"),
    ]
    assert SampleDef.function.__name__ == "function"
    SampleDef(a=1)
    SampleDef(a=1, b=2.0)
    SampleDef.Outputs(c=1.0, d=2.0)


def test_interface_with_inheritance():
    @python.define
    class SampleDef(PythonDef["SampleDef.Outputs"]):
        """Sample class for testing

        Args:
            a: First input
                to be inputted
            b: Second input
        """

        a: int
        b: float

        class Outputs(PythonOutputs):
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

    assert issubclass(SampleDef, PythonDef)


def test_interface_with_class_no_auto_attribs():
    @python.define(auto_attribs=False)
    class SampleDef(PythonDef["SampleDef.Outputs"]):
        a: int = python.arg(help="First input to be inputted")
        b: float = python.arg(help="Second input")

        x: int

        class Outputs(PythonOutputs):
            c: float = python.out(help="Sum of a and b")
            d: float = python.out(help="Product of a and b")

            y: str

        @staticmethod
        def function(a, b):
            return a + b, a * b

    inputs = sorted(list_fields(SampleDef), key=sort_key)
    outputs = sorted(list_fields(SampleDef.Outputs), key=sort_key)
    assert inputs == [
        python.arg(name="a", type=int, help="First input to be inputted"),
        python.arg(name="b", type=float, help="Second input"),
        python.arg(
            name="function",
            type=ty.Callable,
            hash_eq=True,
            default=attrs.fields(SampleDef).function.default,
        ),
    ]
    assert outputs == [
        python.out(name="c", type=float, help="Sum of a and b"),
        python.out(name="d", type=float, help="Product of a and b"),
    ]
    assert SampleDef.function.__name__ == "function"
    SampleDef(a=1, b=2.0)
    SampleDef.Outputs(c=1.0, d=2.0)
    with pytest.raises(TypeError):
        SampleDef(a=1, b=2.0, x=3)
    with pytest.raises(TypeError):
        SampleDef.Outputs(c=1.0, d=2.0, y="hello")


def test_interface_invalid_wrapped1():
    with pytest.raises(ValueError):

        @python.define(inputs={"a": python.arg()})
        class SampleDef(PythonDef["SampleDef.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1


def test_interface_invalid_wrapped2():
    with pytest.raises(ValueError):

        @python.define(outputs={"b": python.out()})
        class SampleDef(PythonDef["SampleDef.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1
