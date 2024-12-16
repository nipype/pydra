Hashing and Caching
===================

Work in progress....







## Overrides




## wrap_function_types

def func(a: int) -> int:
    """Sample function with inputs and outputs"""
    return a * 2

SampleSpec = python.define(
    func,
    inputs={"a": float},
    outputs={"b": float},
)







## implicit_outputs_from_return_stmt

@python.define
def SampleSpec(a: int, b: float) -> tuple[float, float]:
    """Sample function for testing"""
    c = a + b
    d = a * b
    return c, d


## Function docstr


@python.define(outputs=["c", "d"])
def SampleSpec(a: int, b: float) -> tuple[float, float]:
    """Sample function for testing

    :param a: First input to be inputted
    :param b: Second input
    :return c: Sum of a and b
    :return d: product of a and b
    """
    return a + b, a * b

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


## Canonical (dataclass-style) form

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



## With inheritance

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


## no_auto_attribs

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
