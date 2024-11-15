from operator import attrgetter
import pytest
from pydra import design
from decimal import Decimal
from pydra.design.python import arg, out, interface
from pydra.engine.task import FunctionTask


def test_interface_wrap_function():
    def sample_interface(a: int) -> float:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleInterface = interface(
        sample_interface,
        inputs={"a": arg(help_string="The argument to be doubled")},
        outputs={"b": out(help_string="the doubled output", type=Decimal)},
    )

    assert issubclass(SampleInterface, design.Interface)
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="The argument to be doubled"),
    ]
    assert outputs == [
        out(name="b", type=Decimal, help_string="the doubled output"),
    ]


def test_interface_wrap_function_types():
    def sample_interface(a: int) -> int:
        """Sample function with inputs and outputs"""
        return a * 2

    SampleInterface = interface(
        sample_interface,
        inputs={"a": float},
        outputs={"b": float},
    )

    assert issubclass(SampleInterface, design.Interface)
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [arg(name="a", type=float)]
    assert outputs == [out(name="b", type=float)]


def test_decorated_function_interface():
    @design.python.interface(outputs=["c", "d"])
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        return a + b, a * b

    assert issubclass(SampleInterface, design.Interface)
    assert SampleInterface.Task is FunctionTask
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int),
        arg(name="b", type=float),
    ]
    assert outputs == [
        out(name="c", type=float),
        out(name="d", type=float),
    ]
    assert SampleInterface.function.__name__ == "SampleInterface"


def test_interface_with_function_implicit_outputs_from_return_stmt():
    @design.python.interface
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing"""
        c = a + b
        d = a * b
        return c, d

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int),
        arg(name="b", type=float),
    ]
    assert outputs == [
        out(name="c", type=float),
        out(name="d", type=float),
    ]
    assert SampleInterface.function.__name__ == "SampleInterface"


def test_interface_with_function_docstr():
    @design.python.interface(outputs=["c", "d"])
    def SampleInterface(a: int, b: float) -> tuple[float, float]:
        """Sample function for testing

        :param a: First input to be inputted
        :param b: Second input
        :return c: Sum of a and b
        :return d: product of a and b
        """
        return a + b, a * b

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="First input to be inputted"),
        arg(name="b", type=float, help_string="Second input"),
    ]
    assert outputs == [
        out(name="c", type=float, help_string="Sum of a and b"),
        out(name="d", type=float, help_string="product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "SampleInterface"


def test_interface_with_function_google_docstr():
    @design.python.interface(outputs=["c", "d"])
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
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="First input to be inputted"),
        arg(name="b", type=float, help_string="Second input"),
    ]
    assert outputs == [
        out(name="c", type=float, help_string="Sum of a and b"),
        out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "SampleInterface"


def test_interface_with_function_numpy_docstr():
    @design.python.interface(
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
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="First input to be inputted"),
        arg(name="b", type=float, help_string="Second input"),
    ]
    assert outputs == [
        out(name="c", type=float, help_string="Sum of a and b"),
        out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "SampleInterface"


def test_interface_with_class():
    @design.python.interface
    class SampleInterface:
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

    assert issubclass(SampleInterface, design.Interface)
    assert SampleInterface.Task is FunctionTask
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="First input to be inputted"),
        arg(name="b", type=float, help_string="Second input"),
    ]
    assert outputs == [
        out(name="c", type=float, help_string="Sum of a and b"),
        out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "function"


def test_interface_with_inheritance():
    @design.python.interface
    class SampleInterface(design.Interface["SampleInterface.Outputs"]):
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

    assert issubclass(SampleInterface, design.Interface)


def test_interface_with_class_no_auto_attribs():
    @design.python.interface(auto_attribs=False)
    class SampleInterface:
        a: int = arg(help_string="First input to be inputted")
        b: float = arg(help_string="Second input")

        x: int

        class Outputs:
            c: float = out(help_string="Sum of a and b")
            d: float = out(help_string="Product of a and b")

            y: str

        @staticmethod
        def function(a, b):
            return a + b, a * b

    assert SampleInterface.Task is FunctionTask
    inputs = sorted(design.list_fields(SampleInterface), key=attrgetter("name"))
    outputs = sorted(
        design.list_fields(SampleInterface.Outputs), key=attrgetter("name")
    )
    assert inputs == [
        arg(name="a", type=int, help_string="First input to be inputted"),
        arg(name="b", type=float, help_string="Second input"),
    ]
    assert outputs == [
        out(name="c", type=float, help_string="Sum of a and b"),
        out(name="d", type=float, help_string="Product of a and b"),
    ]
    assert SampleInterface.function.__name__ == "function"


def test_interface_invalid_wrapped1():
    with pytest.raises(ValueError):

        @design.python.interface(inputs={"a": arg()})
        class SampleInterface(design.Interface["SampleInterface.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1


def test_interface_invalid_wrapped2():
    with pytest.raises(ValueError):

        @design.python.interface(outputs={"b": out()})
        class SampleInterface(design.Interface["SampleInterface.Outputs"]):
            a: int

            class Outputs:
                b: float

            @staticmethod
            def function(a):
                return a + 1
