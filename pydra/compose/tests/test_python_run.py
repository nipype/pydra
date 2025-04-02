import typing as ty
import os
import pytest
from pathlib import Path
import glob as glob
from pydra.compose import python
from pydra.utils.general import task_fields, task_help
from pydra.utils.general import default_run_cache_root
from pydra.utils.typing import (
    MultiInputObj,
    MultiOutputObj,
)


@python.define
def FunAddTwo(a):
    return a + 2


def test_output():
    nn = FunAddTwo(a=3)
    outputs = nn()
    assert outputs.out == 5


def test_python_output():
    @python.define(outputs=["output"])
    def TestFunc(a: int, b: float = 0.1) -> float:
        """
        Parameters
        ----------
        a : int
            first input
        b : float
            second input

        Returns
        -------
        output : float
            sum of a and b
        """
        return a + b

    funky = TestFunc(a=1)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "function")
    assert getattr(funky, "a") == 1
    assert getattr(funky, "b") == 0.1
    assert getattr(funky, "function") is not None
    assert set(f.name for f in task_fields(funky.Outputs)) == {"output"}

    outputs = funky()
    assert hasattr(outputs, "output")
    assert outputs.output == 1.1

    assert os.path.exists(
        default_run_cache_root / f"python-{funky._hash}" / "_result.pklz"
    )
    funky()  # should not recompute
    funky.a = 2
    outputs = funky()
    assert outputs.output == 2.1

    help = task_help(funky)
    assert help == [
        "-------------------------------",
        "Help for Python task 'TestFunc'",
        "-------------------------------",
        "",
        "Inputs:",
        "- a: int",
        "    first input",
        "- b: float; default = 0.1",
        "    second input",
        "- function: Callable[]; default = TestFunc()",
        "",
        "Outputs:",
        "- output: float",
        "    sum of a and b",
        "",
    ]


def test_python_output_dictreturn(tmp_path: Path):
    """Test mapping from returned dictionary to output definition."""

    @python.define(outputs={"sum": int, "mul": int | None})
    def TestFunc(a: int, b: int):
        return dict(sum=a + b, diff=a - b)

    task = TestFunc(a=2, b=3)
    outputs = task(cache_root=tmp_path)

    # Part of the annotation and returned, should be exposed to output.
    assert outputs.sum == 5

    # Part of the annotation but not returned, should be coalesced to None
    assert outputs.mul is None

    # Not part of the annotation, should be discarded.
    assert not hasattr(outputs, "diff")


def test_python_output_multreturn():
    """the function has two elements in the return statement"""

    @python.define(outputs={"fractional": float, "integer": int})
    def TestFunc(
        a: float,
    ):
        import math

        return math.modf(a)[0], int(math.modf(a)[1])

    funky = TestFunc(a=3.5)
    assert hasattr(funky, "a")
    assert hasattr(funky, "function")
    assert getattr(funky, "a") == 3.5
    assert getattr(funky, "function") is not None
    assert set(f.name for f in task_fields(funky.Outputs)) == {"fractional", "integer"}

    outputs = funky()
    assert os.path.exists(
        default_run_cache_root / f"python-{funky._hash}" / "_result.pklz"
    )
    assert hasattr(outputs, "fractional")
    assert outputs.fractional == 0.5
    assert hasattr(outputs, "integer")
    assert outputs.integer == 3

    help = task_help(funky)
    assert help == [
        "-------------------------------",
        "Help for Python task 'TestFunc'",
        "-------------------------------",
        "",
        "Inputs:",
        "- a: float",
        "- function: Callable[]; default = TestFunc()",
        "",
        "Outputs:",
        "- fractional: float",
        "- integer: int",
        "",
    ]


def test_python_func_1():
    """the function with annotated input (float)"""

    @python.define
    def TestFunc(a: float):
        return a

    funky = TestFunc(a=3.5)
    assert getattr(funky, "a") == 3.5


def test_python_func_2():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def TestFunc(a: int):
        return a

    with pytest.raises(TypeError):
        TestFunc(a=3.5)


def test_python_func_2a():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def TestFunc(a: int):
        return a

    funky = TestFunc()
    with pytest.raises(TypeError):
        funky.a = 3.5


def test_python_func_3():
    """the function with annotated input (list)"""

    @python.define
    def TestFunc(a: list):
        return sum(a)

    funky = TestFunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_python_func_3a():
    """the function with annotated input (list of floats)"""

    @python.define
    def TestFunc(a: ty.List[float]):
        return sum(a)

    funky = TestFunc(a=[1.0, 3.5])
    assert getattr(funky, "a") == [1.0, 3.5]


def test_python_func_3b():
    """the function with annotated input
    (list of floats - int and float provided, should be fine)
    """

    @python.define
    def TestFunc(a: ty.List[float]):
        return sum(a)

    funky = TestFunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_python_func_3c_excep():
    """the function with annotated input
    (list of ints - int and float provided, should raise an error)
    """

    @python.define
    def TestFunc(a: ty.List[int]):
        return sum(a)

    with pytest.raises(TypeError):
        TestFunc(a=[1, 3.5])


def test_python_func_4():
    """the function with annotated input (dictionary)"""

    @python.define
    def TestFunc(a: dict):
        return sum(a.values())

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_python_func_4a():
    """the function with annotated input (dictionary of floats)"""

    @python.define
    def TestFunc(a: ty.Dict[str, float]):
        return sum(a.values())

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_python_func_4b_excep():
    """the function with annotated input (dictionary of ints, but float provided)"""

    @python.define
    def TestFunc(a: ty.Dict[str, int]):
        return sum(a.values())

    with pytest.raises(TypeError):
        TestFunc(a={"el1": 1, "el2": 3.5})


def test_python_func_5():
    """the function with annotated more complex input type (ty.List in ty.Dict)
    the validator should simply check if values of dict are lists
    so no error for 3.5
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.List]):
        return sum(a["el1"])

    funky = TestFunc(a={"el1": [1, 3.5]})
    assert getattr(funky, "a") == {"el1": [1, 3.5]}


def test_python_func_5a_except():
    """the function with annotated more complex input type (ty.Dict in ty.Dict)
    list is provided as a dict value (instead a dict), so error is raised
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Dict[str, float]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        TestFunc(a={"el1": [1, 3.5]})


def test_python_func_6():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Union[float, int]]):
        return sum(a["el1"])

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_python_func_6a_excep():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union and raise an error for 3.5
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Union[str, int]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        TestFunc(a={"el1": 1, "el2": 3.5})


def test_python_func_7():
    """the function with annotated input (float)
    the task has a splitter, so list of float is provided
    it should work, the validator tries to guess if this is a field with a splitter
    """

    @python.define
    def TestFunc(a: float):
        return a

    funky = TestFunc().split("a", a=[3.5, 2.1])
    assert getattr(funky, "a") == [3.5, 2.1]


def test_python_func_7a_excep():
    """the function with annotated input (int) and splitter
    list of float provided - should raise an error (list of int would be fine)
    """

    @python.define
    def TestFunc(a: int):
        return a

    with pytest.raises(TypeError):
        TestFunc(a=[3.5, 2.1]).split("a")


def test_python_func_8():
    """the function with annotated input as MultiInputObj
    a single value is provided and should be converted to a list
    """

    @python.define
    def TestFunc(a: MultiInputObj):
        return len(a)

    funky = TestFunc(a=3.5)
    assert getattr(funky, "a") == [3.5]
    outputs = funky()
    assert outputs.out == 1


def test_python_func_8a():
    """the function with annotated input as MultiInputObj
    a 1-el list is provided so shouldn't be changed
    """

    @python.define
    def TestFunc(a: MultiInputObj):
        return len(a)

    funky = TestFunc(a=[3.5])
    assert getattr(funky, "a") == [3.5]
    outputs = funky()
    assert outputs.out == 1


def test_python_func_8b():
    """the function with annotated input as MultiInputObj
    a single value is provided after initial. the task
    (input should still be converted to a list)
    """

    @python.define
    def TestFunc(a: MultiInputObj):
        return len(a)

    funky = TestFunc()
    # setting a after init
    funky.a = 3.5
    assert getattr(funky, "a") == [3.5]
    outputs = funky()
    assert outputs.out == 1


def test_python_output_multreturn_exception():
    """function has two elements in the return statement,
    but three element provided in the task - should raise an error
    """

    @python.define(outputs={"fractional": float, "integer": int, "who_knows": int})
    def TestFunc(
        a: float,
    ):
        import math

        return math.modf(a)

    funky = TestFunc(a=3.5)
    with pytest.raises(Exception) as excinfo:
        funky()
    assert "expected 3 elements" in str(excinfo.value)


def test_halfpython_output(tmp_path):

    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    @python.define
    def TestFunc(a, b) -> int:
        return a + b

    funky = TestFunc(a=10, b=20)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "function")
    assert getattr(funky, "a") == 10
    assert getattr(funky, "b") == 20
    assert getattr(funky, "function") is not None
    assert set(f.name for f in task_fields(funky.Outputs)) == {"out"}

    outputs = funky(cache_root=cache_root)
    assert hasattr(outputs, "out")
    assert outputs.out == 30

    assert Path(cache_root / f"python-{funky._hash}" / "_result.pklz").exists()

    funky(cache_root=cache_root)  # should not recompute
    funky.a = 11
    assert not Path(cache_root / f"python-{funky._hash}").exists()
    outputs = funky(cache_root=cache_root)
    assert outputs.out == 31
    help = task_help(funky)

    assert help == [
        "-------------------------------",
        "Help for Python task 'TestFunc'",
        "-------------------------------",
        "",
        "Inputs:",
        "- a: Any",
        "- b: Any",
        "- function: Callable[]; default = TestFunc()",
        "",
        "Outputs:",
        "- out: int",
        "",
    ]


def test_halfpython_output_multreturn(tmp_path):

    cache_root = tmp_path / "cache"
    cache_root.mkdir()

    @python.define(outputs=["out1", "out2"])
    def TestFunc(a, b) -> tuple[int, int]:
        return a + 1, b + 1

    funky = TestFunc(a=10, b=20)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "function")
    assert getattr(funky, "a") == 10
    assert getattr(funky, "b") == 20
    assert getattr(funky, "function") is not None
    assert set(f.name for f in task_fields(funky.Outputs)) == {"out1", "out2"}

    outputs = funky(cache_root=cache_root)
    assert hasattr(outputs, "out1")
    assert outputs.out1 == 11

    assert Path(cache_root / f"python-{funky._hash}" / "_result.pklz").exists()

    funky(cache_root=cache_root)  # should not recompute
    funky.a = 11
    assert not Path(cache_root / f"python-{funky._hash}" / "_result.pklz").exists()
    outputs = funky(cache_root=cache_root)
    assert outputs.out1 == 12
    help = task_help(funky)

    assert help == [
        "-------------------------------",
        "Help for Python task 'TestFunc'",
        "-------------------------------",
        "",
        "Inputs:",
        "- a: Any",
        "- b: Any",
        "- function: Callable[]; default = TestFunc()",
        "",
        "Outputs:",
        "- out1: int",
        "- out2: int",
        "",
    ]


def test_notpython_output():
    @python.define
    def NoAnnots(c, d):
        return c + d

    no_annots = NoAnnots(c=17, d=3.2)
    assert hasattr(no_annots, "c")
    assert hasattr(no_annots, "d")
    assert hasattr(no_annots, "function")

    outputs = no_annots()
    assert hasattr(outputs, "out")
    assert outputs.out == 20.2


def test_notpython_output_returnlist():
    @python.define
    def NoAnnots(c, d):
        return [c, d]

    no_annots = NoAnnots(c=17, d=3.2)
    outputs = no_annots()
    assert hasattr(outputs, "out")
    assert outputs.out == [17, 3.2]


def test_halfpython_output_multrun_returnlist():
    @python.define(outputs=["out1", "out2"])
    def NoAnnots(c, d) -> tuple[list, float]:
        return [c, d], c + d

    no_annots = NoAnnots(c=17, d=3.2)
    outputs = no_annots()

    assert hasattr(outputs, "out1")
    assert hasattr(outputs, "out2")
    assert outputs.out1 == [17, 3.2]
    assert outputs.out2 == 20.2


def test_notpython_output_multreturn():
    """no annotation and multiple values are returned
    all elements should be returned as a tuple and set to "out"
    """

    @python.define
    def NoAnnots(c, d):
        return c + d, c - d

    no_annots = NoAnnots(c=17, d=3.2)
    assert hasattr(no_annots, "c")
    assert hasattr(no_annots, "d")
    assert hasattr(no_annots, "function")

    outputs = no_annots()
    assert hasattr(outputs, "out")
    assert outputs.out == (20.2, 13.8)


def test_input_spec_func_1():
    """the function w/o annotated, but input_spec is used"""

    @python.define(inputs={"a": python.arg(type=float, help="input a")})
    def TestFunc(a):
        return a

    funky = TestFunc(a=3.5)
    assert funky.a == 3.5


def test_input_spec_func_1a_except():
    """the function w/o annotated, but input_spec is used
    a TypeError is raised (float is provided instead of int)
    """

    @python.define(inputs={"a": python.arg(type=int, help="input a")})
    def TestFunc(a):
        return a

    with pytest.raises(TypeError):
        TestFunc(a=3.5)


def test_input_spec_func_1b_except():
    """the function w/o annotated, but input_spec is used
    metadata checks raise an error
    """

    with pytest.raises(
        TypeError, match="got an unexpected keyword argument 'position'"
    ):

        @python.define(inputs={"a": python.arg(type=float, position=1, help="input a")})
        def TestFunc(a):
            return a


def test_input_spec_func_1d_except():
    """the function w/o annotated, but input_spec is used
    input_spec doesn't contain 'a' input, an error is raised
    """

    @python.define
    def TestFunc(a):
        return a

    funky = TestFunc()
    with pytest.raises(ValueError, match="Mandatory field 'a' is not set"):
        funky()


def test_input_spec_func_2():
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    """

    @python.define(inputs={"a": python.arg(type=float, help="input a")})
    def TestFunc(a: int):
        return a

    funky = TestFunc(a=3.5)
    assert funky.a == 3.5


def test_input_spec_func_2a():
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    using the shorter syntax
    """

    @python.define(inputs={"a": python.arg(type=float, help="input a")})
    def TestFunc(a: int):
        return a

    funky = TestFunc(a=3.5)
    assert funky.a == 3.5


def test_input_spec_func_3():
    """the function w/o annotated, but input_spec is used
    additional keys (allowed_values) are used in metadata
    """

    @python.define(
        inputs={
            "a": python.arg(
                type=int,
                help="input a",
                allowed_values=[0, 1, 2],
            )
        }
    )
    def TestFunc(a):
        return a

    funky = TestFunc(a=2)
    assert funky.a == 2


def test_input_spec_func_3a_except():
    """the function w/o annotated, but input_spec is used
    allowed_values is used in metadata and the ValueError is raised
    """

    @python.define(
        inputs={
            "a": python.arg(
                type=int,
                help="input a",
                allowed_values=[0, 1, 2],
            )
        }
    )
    def TestFunc(a):
        return a

    with pytest.raises(ValueError, match="value of a has to be"):
        TestFunc(a=3)


def test_input_spec_func_4():
    """the function with a default value for b
    but b is set as mandatory in the input_spec, so error is raised if not provided
    """

    @python.define(
        inputs={
            "a": python.arg(type=int, help="input a"),
            "b": python.arg(type=int, help="input b"),
        }
    )
    def TestFunc(a, b):
        return a + b

    funky = TestFunc(a=2)
    with pytest.raises(Exception, match="Mandatory field 'b' is not set"):
        funky()


def test_input_spec_func_4a():
    """the function with a default value for b and metadata in the input_spec
    has a different default value, so value from the function is overwritten
    """

    @python.define(
        inputs={
            "a": python.arg(type=int, help="input a"),
            "b": python.arg(type=int, help="input b", default=10),
        }
    )
    def TestFunc(a, b=1):
        return a + b

    funky = TestFunc(a=2)
    outputs = funky()
    assert outputs.out == 12


def test_input_spec_func_5():
    """the python.Task with input_spec, a input has MultiInputObj type
    a single value is provided and should be converted to a list
    """

    @python.define(inputs={"a": python.arg(type=MultiInputObj, help="input a")})
    def TestFunc(a):
        return len(a)

    funky = TestFunc(a=3.5)
    assert funky.a == MultiInputObj([3.5])
    outputs = funky()
    assert outputs.out == 1


def test_output_spec_func_1():
    """the function w/o annotated, but output_spec is used"""

    @python.define(outputs={"out1": python.out(type=float, help="output")})
    def TestFunc(a):
        return a

    funky = TestFunc(a=3.5)
    outputs = funky()
    assert outputs.out1 == 3.5


def test_output_spec_func_1a_except():
    """the function w/o annotated, but output_spec is used
    float returned instead of int - TypeError
    """

    @python.define(outputs={"out1": python.out(type=int, help="output")})
    def TestFunc(a):
        return a

    funky = TestFunc(a=3.5)
    with pytest.raises(TypeError):
        funky()


def test_output_spec_func_2():
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    """

    @python.define(outputs={"out1": python.out(type=float, help="output")})
    def TestFunc(a) -> int:
        return a

    funky = TestFunc(a=3.5)
    outputs = funky()
    assert outputs.out1 == 3.5


def test_output_spec_func_2a():
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    using a shorter syntax
    """

    @python.define(outputs={"out1": python.out(type=float, help="output")})
    def TestFunc(a) -> int:
        return a

    funky = TestFunc(a=3.5)
    outputs = funky()
    assert outputs.out1 == 3.5


def test_output_spec_func_3():
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 2-el list, so converter doesn't do anything
    """

    @python.define(outputs={"out_list": python.out(type=MultiOutputObj, help="output")})
    def TestFunc(a, b):
        return [a, b]

    funky = TestFunc(a=3.5, b=1)
    outputs = funky()
    assert outputs.out_list == [3.5, 1]


def test_output_spec_func_4():
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 1el list, so converter return the element
    """

    @python.define(outputs={"out_list": python.out(type=MultiOutputObj, help="output")})
    def TestFunc(a):
        return [a]

    funky = TestFunc(a=3.5)
    outputs = funky()
    assert outputs.out_list == 3.5


def test_functask_callable(tmpdir):
    # no submitter or worker
    foo = FunAddTwo(a=1)
    outputs = foo()
    assert outputs.out == 3

    # worker
    bar = FunAddTwo(a=2)
    outputs = bar(worker="cf", cache_root=tmpdir)
    assert outputs.out == 4
