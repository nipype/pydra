import typing as ty
import os
import sys
import attr
import pytest
import cloudpickle as cp
from pathlib import Path
import json
import glob as glob
from pydra.design import python
from pydra.utils.messenger import FileMessenger, PrintMessenger, collect_messages
from ..task import AuditFlag, ShellTask
from pydra.engine.specs import argstr_formatting
from pydra.engine.helpers import list_fields, print_help
from .utils import BasicWorkflow
from pydra.utils import default_run_cache_dir
from pydra.utils.typing import (
    MultiInputObj,
    MultiOutputObj,
)
from ..specs import (
    ShellDef,
)
from fileformats.generic import File
from pydra.utils.hash import hash_function


no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


@python.define
def FunAddTwo(a):
    return a + 2


def test_output():
    nn = FunAddTwo(a=3)
    outputs = nn()
    assert outputs.out == 5


def test_numpy():
    """checking if mark.task works for numpy functions"""
    np = pytest.importorskip("numpy")
    FFT = python.define(inputs={"a": np.ndarray}, outputs={"out": np.ndarray})(
        np.fft.fft
    )

    arr = np.array([[1, 10], [2, 20]])
    fft = FFT(a=arr)
    outputs = fft()
    assert np.allclose(np.fft.fft(arr), outputs.out)


@pytest.mark.xfail(reason="cp.dumps(func) depends on the system/setup, TODO!!")
def test_checksum():
    nn = FunAddTwo(a=3)
    assert (
        nn.checksum
        == "PythonTask_abb4e7cc03b13d0e73884b87d142ed5deae6a312275187a9d8df54407317d7d3"
    )


def test_annotated_func():
    @python.define(outputs=["out_out"])
    def testfunc(a: int, b: float = 0.1) -> float:
        return a + b

    funky = testfunc(a=1)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "function")
    assert getattr(funky, "a") == 1
    assert getattr(funky, "b") == 0.1
    assert getattr(funky, "function") is not None
    assert set(f.name for f in list_fields(funky.Outputs)) == {"out_out"}

    outputs = funky()
    assert hasattr(outputs, "out_out")
    assert outputs.out_out == 1.1

    assert os.path.exists(
        default_run_cache_dir / f"python-{funky._hash}" / "_result.pklz"
    )
    funky()  # should not recompute
    funky.a = 2
    outputs = funky()
    assert outputs.out_out == 2.1

    help = print_help(funky)
    assert help == [
        "Help for PythonTask",
        "Input Parameters:",
        "- a: int",
        "- b: float (default: 0.1)",
        "- _func: bytes",
        "Output Parameters:",
        "- out_out: float",
    ]


def test_annotated_func_dictreturn():
    """Test mapping from returned dictionary to output definition."""

    @python.define(outputs={"sum": int, "mul": ty.Optional[int]})
    def testfunc(a: int, b: int):
        return dict(sum=a + b, diff=a - b)

    task = testfunc(a=2, b=3)
    outputs = task()

    # Part of the annotation and returned, should be exposed to output.
    assert outputs.sum == 5

    # Part of the annotation but not returned, should be coalesced to None
    assert outputs.mul is None

    # Not part of the annotation, should be discarded.
    assert not hasattr(outputs, "diff")


def test_annotated_func_multreturn():
    """the function has two elements in the return statement"""

    @python.define
    def testfunc(
        a: float,
    ) -> ty.NamedTuple("Output", [("fractional", float), ("integer", int)]):
        import math

        return math.modf(a)[0], int(math.modf(a)[1])

    funky = testfunc(a=3.5)
    assert hasattr(funky, "a")
    assert hasattr(funky, "_func")
    assert getattr(funky, "a") == 3.5
    assert getattr(funky, "_func") is not None
    assert set(funky.output_names) == {"fractional", "integer"}
    assert funky.__class__.__name__ + "_" + funky.hash == funky.checksum

    outputs = funky()
    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    assert hasattr(result, "output")
    assert hasattr(outputs, "fractional")
    assert outputs.fractional == 0.5
    assert hasattr(outputs, "integer")
    assert outputs.integer == 3

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for PythonTask",
        "Input Parameters:",
        "- a: float",
        "- _func: bytes",
        "Output Parameters:",
        "- fractional: float",
        "- integer: int",
    ]


def test_annotated_input_func_1():
    """the function with annotated input (float)"""

    @python.define
    def testfunc(a: float):
        return a

    funky = testfunc(a=3.5)
    assert getattr(funky, "a") == 3.5


def test_annotated_input_func_2():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def testfunc(a: int):
        return a

    with pytest.raises(TypeError):
        testfunc(a=3.5)


def test_annotated_input_func_2a():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def testfunc(a: int):
        return a

    funky = testfunc()
    with pytest.raises(TypeError):
        funky.a = 3.5


def test_annotated_input_func_3():
    """the function with annotated input (list)"""

    @python.define
    def testfunc(a: list):
        return sum(a)

    funky = testfunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_annotated_input_func_3a():
    """the function with annotated input (list of floats)"""

    @python.define
    def testfunc(a: ty.List[float]):
        return sum(a)

    funky = testfunc(a=[1.0, 3.5])
    assert getattr(funky, "a") == [1.0, 3.5]


def test_annotated_input_func_3b():
    """the function with annotated input
    (list of floats - int and float provided, should be fine)
    """

    @python.define
    def testfunc(a: ty.List[float]):
        return sum(a)

    funky = testfunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_annotated_input_func_3c_excep():
    """the function with annotated input
    (list of ints - int and float provided, should raise an error)
    """

    @python.define
    def testfunc(a: ty.List[int]):
        return sum(a)

    with pytest.raises(TypeError):
        testfunc(a=[1, 3.5])


def test_annotated_input_func_4():
    """the function with annotated input (dictionary)"""

    @python.define
    def testfunc(a: dict):
        return sum(a.values())

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4a():
    """the function with annotated input (dictionary of floats)"""

    @python.define
    def testfunc(a: ty.Dict[str, float]):
        return sum(a.values())

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4b_excep():
    """the function with annotated input (dictionary of ints, but float provided)"""

    @python.define
    def testfunc(a: ty.Dict[str, int]):
        return sum(a.values())

    with pytest.raises(TypeError):
        testfunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_5():
    """the function with annotated more complex input type (ty.List in ty.Dict)
    the validator should simply check if values of dict are lists
    so no error for 3.5
    """

    @python.define
    def testfunc(a: ty.Dict[str, ty.List]):
        return sum(a["el1"])

    funky = testfunc(a={"el1": [1, 3.5]})
    assert getattr(funky, "a") == {"el1": [1, 3.5]}


def test_annotated_input_func_5a_except():
    """the function with annotated more complex input type (ty.Dict in ty.Dict)
    list is provided as a dict value (instead a dict), so error is raised
    """

    @python.define
    def testfunc(a: ty.Dict[str, ty.Dict[str, float]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        testfunc(a={"el1": [1, 3.5]})


def test_annotated_input_func_6():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union
    """

    @python.define
    def testfunc(a: ty.Dict[str, ty.Union[float, int]]):
        return sum(a["el1"])

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_6a_excep():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union and raise an error for 3.5
    """

    @python.define
    def testfunc(a: ty.Dict[str, ty.Union[str, int]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        testfunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_7():
    """the function with annotated input (float)
    the task has a splitter, so list of float is provided
    it should work, the validator tries to guess if this is a field with a splitter
    """

    @python.define
    def testfunc(a: float):
        return a

    funky = testfunc().split("a", a=[3.5, 2.1])
    assert getattr(funky, "a") == [3.5, 2.1]


def test_annotated_input_func_7a_excep():
    """the function with annotated input (int) and splitter
    list of float provided - should raise an error (list of int would be fine)
    """

    @python.define
    def testfunc(a: int):
        return a

    with pytest.raises(TypeError):
        testfunc(a=[3.5, 2.1]).split("a")


def test_annotated_input_func_8():
    """the function with annotated input as MultiInputObj
    a single value is provided and should be converted to a list
    """

    @python.define
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc(a=3.5)
    assert getattr(funky, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_input_func_8a():
    """the function with annotated input as MultiInputObj
    a 1-el list is provided so shouldn't be changed
    """

    @python.define
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc(a=[3.5])
    assert getattr(funky, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_input_func_8b():
    """the function with annotated input as MultiInputObj
    a single value is provided after initial. the task
    (input should still be converted to a list)
    """

    @python.define
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc()
    # setting a after init
    funky.a = 3.5
    assert getattr(funky, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_func_multreturn_exception():
    """function has two elements in the return statement,
    but three element provided in the definition - should raise an error
    """

    @python.define
    def testfunc(
        a: float,
    ) -> ty.NamedTuple(
        "Output", [("fractional", float), ("integer", int), ("who_knows", int)]
    ):
        import math

        return math.modf(a)

    funky = testfunc(a=3.5)
    with pytest.raises(Exception) as excinfo:
        funky()
    assert "expected 3 elements" in str(excinfo.value)


def test_halfannotated_func():
    @python.define
    def testfunc(a, b) -> int:
        return a + b

    funky = testfunc(a=10, b=20)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "_func")
    assert getattr(funky, "a") == 10
    assert getattr(funky, "b") == 20
    assert getattr(funky, "_func") is not None
    assert set(funky.output_names) == {"out"}
    assert funky.__class__.__name__ + "_" + funky.hash == funky.checksum

    outputs = funky()
    assert hasattr(result, "output")
    assert hasattr(outputs, "out")
    assert outputs.out == 30

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert outputs.out == 31
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for PythonTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: bytes",
        "Output Parameters:",
        "- out: int",
    ]


def test_halfannotated_func_multreturn():
    @python.define
    def testfunc(a, b) -> (int, int):
        return a + 1, b + 1

    funky = testfunc(a=10, b=20)
    assert hasattr(funky, "a")
    assert hasattr(funky, "b")
    assert hasattr(funky, "_func")
    assert getattr(funky, "a") == 10
    assert getattr(funky, "b") == 20
    assert getattr(funky, "_func") is not None
    assert set(funky.output_names) == {"out1", "out2"}
    assert funky.__class__.__name__ + "_" + funky.hash == funky.checksum

    outputs = funky()
    assert hasattr(result, "output")
    assert hasattr(outputs, "out1")
    assert outputs.out1 == 11

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert outputs.out1 == 12
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for PythonTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: bytes",
        "Output Parameters:",
        "- out1: int",
        "- out2: int",
    ]


def test_notannotated_func():
    @python.define
    def no_annots(c, d):
        return c + d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask, "c")
    assert hasattr(natask, "d")
    assert hasattr(natask, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(outputs, "out")
    assert outputs.out == 20.2


def test_notannotated_func_returnlist():
    @python.define
    def no_annots(c, d):
        return [c, d]

    natask = no_annots(c=17, d=3.2)
    result = natask._run()
    assert hasattr(outputs, "out")
    assert outputs.out == [17, 3.2]


def test_halfannotated_func_multrun_returnlist():
    @python.define
    def no_annots(c, d) -> (list, float):
        return [c, d], c + d

    natask = no_annots(c=17, d=3.2)
    result = natask._run()

    assert hasattr(outputs, "out1")
    assert hasattr(outputs, "out2")
    assert outputs.out1 == [17, 3.2]
    assert outputs.out2 == 20.2


def test_notannotated_func_multreturn():
    """no annotation and multiple values are returned
    all elements should be returned as a tuple and set to "out"
    """

    @python.define
    def no_annots(c, d):
        return c + d, c - d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask, "c")
    assert hasattr(natask, "d")
    assert hasattr(natask, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(outputs, "out")
    assert outputs.out == (20.2, 13.8)


def test_input_spec_func_1():
    """the function w/o annotated, but input_spec is used"""

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=float, metadata={"help": "input a"}))],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky, "a") == 3.5


def test_input_spec_func_1a_except():
    """the function w/o annotated, but input_spec is used
    a TypeError is raised (float is provided instead of int)
    """

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=int, metadata={"help": "input a"}))],
        bases=(FunctionDef,),
    )
    with pytest.raises(TypeError):
        testfunc(a=3.5, input_spec=my_input_spec)


def test_input_spec_func_1b_except():
    """the function w/o annotated, but input_spec is used
    metadata checks raise an error
    """

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(type=float, metadata={"position": 1, "help": "input a"}),
            )
        ],
        bases=(FunctionDef,),
    )
    with pytest.raises(AttributeError, match="only these keys are supported"):
        testfunc(a=3.5, input_spec=my_input_spec)


def test_input_spec_func_1d_except():
    """the function w/o annotated, but input_spec is used
    input_spec doesn't contain 'a' input, an error is raised
    """

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(name="Input", fields=[], bases=(FunctionDef,))
    funky = testfunc(a=3.5, input_spec=my_input_spec)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        funky()


def test_input_spec_func_2():
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    """

    @python.define
    def testfunc(a: int):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=float, metadata={"help": "input a"}))],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky, "a") == 3.5


def test_input_spec_func_2a():
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    using the shorter syntax
    """

    @python.define
    def testfunc(a: int):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", float, {"help": "input a"})],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky, "a") == 3.5


def test_input_spec_func_3():
    """the function w/o annotated, but input_spec is used
    additional keys (allowed_values) are used in metadata
    """

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int,
                    metadata={"help": "input a", "allowed_values": [0, 1, 2]},
                ),
            )
        ],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    assert getattr(funky, "a") == 2


def test_input_spec_func_3a_except():
    """the function w/o annotated, but input_spec is used
    allowed_values is used in metadata and the ValueError is raised
    """

    @python.define
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int,
                    metadata={"help": "input a", "allowed_values": [0, 1, 2]},
                ),
            )
        ],
        bases=(FunctionDef,),
    )

    with pytest.raises(ValueError, match="value of a has to be"):
        testfunc(a=3, input_spec=my_input_spec)


def test_input_spec_func_4():
    """the function with a default value for b
    but b is set as mandatory in the input_spec, so error is raised if not provided
    """

    @python.define
    def testfunc(a, b=1):
        return a + b

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(type=int, metadata={"help": "input a", "mandatory": True}),
            ),
            (
                "b",
                attr.ib(type=int, metadata={"help": "input b", "mandatory": True}),
            ),
        ],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    with pytest.raises(Exception, match="b is mandatory"):
        funky()


def test_input_spec_func_4a():
    """the function with a default value for b and metadata in the input_spec
    has a different default value, so value from the function is overwritten
    """

    @python.define
    def testfunc(a, b=1):
        return a + b

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(type=int, metadata={"help": "input a", "mandatory": True}),
            ),
            ("b", attr.ib(type=int, default=10, metadata={"help": "input b"})),
        ],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    res = funky()
    assert res.output.out == 12


def test_input_spec_func_5():
    """the PythonTask with input_spec, a input has MultiInputObj type
    a single value is provided and should be converted to a list
    """

    @python.define
    def testfunc(a):
        return len(a)

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=MultiInputObj, metadata={"help": "input a"}))],
        bases=(FunctionDef,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky, "a") == MultiInputObj([3.5])
    res = funky()
    assert res.output.out == 1


def test_output_spec_func_1():
    """the function w/o annotated, but output_spec is used"""

    @python.define
    def testfunc(a):
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=float, metadata={"help": "output"}))],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_1a_except():
    """the function w/o annotated, but output_spec is used
    float returned instead of int - TypeError
    """

    @python.define
    def testfunc(a):
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=int, metadata={"help": "output"}))],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    with pytest.raises(TypeError):
        funky()


def test_output_spec_func_2():
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    """

    @python.define
    def testfunc(a) -> int:
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=float, metadata={"help": "output"}))],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_2a():
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    using a shorter syntax
    """

    @python.define
    def testfunc(a) -> int:
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", float, {"help": "output"})],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_3():
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 2-el list, so converter doesn't do anything
    """

    @python.define
    def testfunc(a, b):
        return [a, b]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_list",
                attr.ib(type=MultiOutputObj, metadata={"help": "output"}),
            )
        ],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, b=1, output_spec=my_output_spec)
    res = funky()
    assert res.output.out_list == [3.5, 1]


def test_output_spec_func_4():
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 1el list, so converter return the element
    """

    @python.define
    def testfunc(a):
        return [a]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1el",
                attr.ib(type=MultiOutputObj, metadata={"help": "output"}),
            )
        ],
        bases=(BaseDef,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out_1el == 3.5


def test_exception_func():
    @python.define
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """checking if None is properly returned as the result"""

    @python.define
    def fun_none(x):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out is None


def test_result_none_2():
    """checking if None is properly set for all outputs"""

    @python.define
    def fun_none(x) -> (ty.Any, ty.Any):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out1 is None
    assert res.output.out2 is None


def test_audit_prov(
    tmpdir,
):
    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    # printing the audit message
    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    funky.cache_dir = tmpdir
    funky()

    # saving the audit message into the file
    funky = testfunc(a=2, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    funky.cache_dir = tmpdir
    funky()
    # this should be the default loctaion
    message_path = tmpdir / funky.checksum / "messages"
    assert (tmpdir / funky.checksum / "messages").exists()

    collect_messages(tmpdir / funky.checksum, message_path, ld_op="compact")
    assert (tmpdir / funky.checksum / "messages.jsonld").exists()


def test_audit_task(tmpdir):
    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    from glob import glob

    funky = testfunc(a=2, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    funky.cache_dir = tmpdir
    funky()
    message_path = tmpdir / funky.checksum / "messages"

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)
            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "testfunc" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert None is data["Label"]
            if "AssociatedWith" in data:
                assert None is data["AssociatedWith"]

    # assert any(json_content)


def test_audit_shellcommandtask(tmpdir):
    args = "-l"
    shelly = ShellTask(
        name="shelly",
        executable="ls",
        args=args,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )

    from glob import glob

    shelly.cache_dir = tmpdir
    shelly()
    message_path = tmpdir / shelly.checksum / "messages"
    # go through each jsonld file in message_path and check if the label field exists

    command_content = []

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)

            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "shelly" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert data["Label"] is None

            if "Command" in data:
                command_content.append(True)
                assert "ls -l" == data["Command"]

    assert any(command_content)


def test_audit_shellcommandtask_file(tmp_path):
    # sourcery skip: use-fstring-for-concatenation
    import glob
    import shutil

    # create test.txt file with "This is a test" in it in the tmpdir
    # create txt file in cwd
    with open("test.txt", "w") as f:
        f.write("This is a test")

    with open("test2.txt", "w") as f:
        f.write("This is a test")

    # copy the test.txt file to the tmpdir
    shutil.copy("test.txt", tmp_path)
    shutil.copy("test2.txt", tmp_path)

    cmd = "cat"
    file_in = File(tmp_path / "test.txt")
    file_in_2 = File(tmp_path / "test2.txt")
    test_file_hash = hash_function(file_in)
    test_file_hash_2 = hash_function(file_in_2)
    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "in_file",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 1,
                        "argstr": "",
                        "help": "text",
                        "mandatory": True,
                    },
                ),
            ),
            (
                "in_file_2",
                attr.ib(
                    type=File,
                    metadata={
                        "position": 2,
                        "argstr": "",
                        "help": "text",
                        "mandatory": True,
                    },
                ),
            ),
        ],
        bases=(ShellDef,),
    )
    shelly = ShellTask(
        name="shelly",
        in_file=file_in,
        in_file_2=file_in_2,
        input_spec=my_input_spec,
        executable=cmd,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    shelly.cache_dir = tmp_path
    results = shelly()
    message_path = tmp_path / shelly.checksum / "messages"
    for file in glob.glob(str(message_path) + "/*.jsonld"):
        with open(file) as x:
            data = json.load(x)
            if "@type" in data:
                if data["@type"] == "input":
                    if data["Label"] == "in_file":
                        assert data["AtLocation"] == str(file_in)
                        assert data["digest"] == test_file_hash
                    if data["Label"] == "in_file_2":
                        assert data["AtLocation"] == str(file_in_2)
                        assert data["digest"] == test_file_hash_2


def test_audit_shellcommandtask_version(tmpdir):
    import subprocess as sp

    version_cmd = sp.run("less --version", shell=True, stdout=sp.PIPE).stdout.decode(
        "utf-8"
    )
    version_cmd = version_cmd.splitlines()[0]
    cmd = "less"
    shelly = ShellTask(
        name="shelly",
        executable=cmd,
        args="test_task.py",
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )

    import glob

    shelly.cache_dir = tmpdir
    shelly()
    message_path = tmpdir / shelly.checksum / "messages"
    # go through each jsonld file in message_path and check if the label field exists
    version_content = []
    for file in glob.glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)
            if "AssociatedWith" in data:
                if version_cmd in data["AssociatedWith"]:
                    version_content.append(True)

    assert any(version_content)


def test_audit_prov_messdir_1(
    tmpdir,
):
    """customized messenger dir"""

    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    # printing the audit message
    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    funky.cache_dir = tmpdir
    funky()

    # saving the audit message into the file
    funky = testfunc(a=2, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    # user defined path
    message_path = tmpdir / funky.checksum / "my_messages"
    funky.cache_dir = tmpdir
    # providing messenger_dir for audit
    funky.audit.messenger_args = dict(message_dir=message_path)
    funky()
    assert (tmpdir / funky.checksum / "my_messages").exists()

    collect_messages(tmpdir / funky.checksum, message_path, ld_op="compact")
    assert (tmpdir / funky.checksum / "messages.jsonld").exists()


def test_audit_prov_messdir_2(
    tmpdir,
):
    """customized messenger dir in init"""

    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    # printing the audit message
    funky = testfunc(a=1, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())
    funky.cache_dir = tmpdir
    funky()

    # user defined path (doesn't depend on checksum, can be defined before init)
    message_path = tmpdir / "my_messages"
    # saving the audit message into the file
    funky = testfunc(
        a=2,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    funky.cache_dir = tmpdir
    # providing messenger_dir for audit
    funky()
    assert (tmpdir / "my_messages").exists()

    collect_messages(tmpdir, message_path, ld_op="compact")
    assert (tmpdir / "messages.jsonld").exists()


def test_audit_prov_wf(
    tmpdir,
):
    """FileMessenger for wf"""

    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    wf = Workflow(
        name="wf",
        input_spec=["x"],
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    wf.add(testfunc(name="testfunc", a=wf.lzin.x))
    wf.set_output([("out", wf.testfunc.lzout.out)])
    wf.x = 2

    wf(plugin="cf")
    # default path
    message_path = tmpdir / wf.checksum / "messages"
    assert message_path.exists()

    collect_messages(tmpdir / wf.checksum, message_path, ld_op="compact")
    assert (tmpdir / wf.checksum / "messages.jsonld").exists()


def test_audit_all(
    tmpdir,
):
    @python.define
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    funky = testfunc(a=2, audit_flags=AuditFlag.ALL, messengers=FileMessenger())
    message_path = tmpdir / funky.checksum / "messages"
    funky.cache_dir = tmpdir
    funky.audit.messenger_args = dict(message_dir=message_path)
    funky()
    from glob import glob

    assert len(glob(str(tmpdir / funky.checksum / "proc*.log"))) == 1
    assert len(glob(str(message_path / "*.jsonld"))) == 7

    # commented out to speed up testing
    collect_messages(tmpdir / funky.checksum, message_path, ld_op="compact")
    assert (tmpdir / funky.checksum / "messages.jsonld").exists()


@no_win
def test_shell_cmd(tmpdir):
    cmd = ["echo", "hail", "pydra"]

    # all args given as executable
    shelly = ShellTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    shelly = ShellTask(executable=cmd[0], args=cmd[1:])
    assert shelly.definition.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.return_code == 0
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"


def test_functask_callable(tmpdir):
    # no submitter or plugin
    foo = FunAddTwo(a=1)
    res = foo()
    assert res.output.out == 3
    assert foo.plugin is None

    # plugin
    bar = FunAddTwo(a=2)
    res = bar(plugin="cf")
    assert res.output.out == 4
    assert bar.plugin is None

    foo2 = FunAddTwo(a=3)
    foo2.plugin = "cf"
    res = foo2()
    assert res.output.out == 5
    assert foo2.plugin == "cf"


def test_taskhooks_1(tmpdir, capsys):
    foo = FunAddTwo(name="foo", a=1, cache_dir=tmpdir)
    assert foo.hooks
    # ensure all hooks are defined
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    def myhook(task, *args):
        print("I was called")

    foo.hooks.pre_run = myhook
    foo()
    captured = capsys.readouterr()
    assert "I was called\n" in captured.out
    del captured

    # setting unknown hook should not be allowed
    with pytest.raises(AttributeError):
        foo.hooks.mid_run = myhook

    # set all hooks
    foo.hooks.post_run = myhook
    foo.hooks.pre_run_task = myhook
    foo.hooks.post_run_task = myhook
    foo.a = 2  # ensure not pre-cached
    foo()
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured

    # hooks are independent across tasks by default
    bar = FunAddTwo(name="bar", a=3, cache_dir=tmpdir)
    assert bar.hooks is not foo.hooks
    # but can be shared across tasks
    bar.hooks = foo.hooks
    # and workflows
    wf = BasicWorkflow()
    wf.tmpdir = tmpdir
    wf.hooks = bar.hooks
    assert foo.hooks == bar.hooks == wf.hooks

    wf(plugin="cf")
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured

    # reset all hooks
    foo.hooks.reset()
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None


def test_taskhooks_2(tmpdir, capsys):
    """checking order of the hooks; using task's attributes"""
    foo = FunAddTwo(name="foo", a=1, cache_dir=tmpdir)

    def myhook_prerun(task, *args):
        print(f"i. prerun hook was called from {task.name}")

    def myhook_prerun_task(task, *args):
        print(f"ii. prerun task hook was called {task.name}")

    def myhook_postrun_task(task, *args):
        print(f"iii. postrun task hook was called {task.name}")

    def myhook_postrun(task, *args):
        print(f"iv. postrun hook was called {task.name}")

    foo.hooks.pre_run = myhook_prerun
    foo.hooks.post_run = myhook_postrun
    foo.hooks.pre_run_task = myhook_prerun_task
    foo.hooks.post_run_task = myhook_postrun_task
    foo()

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking the order of the hooks
    assert "i. prerun hook" in hook_messages[0]
    assert "ii. prerun task hook" in hook_messages[1]
    assert "iii. postrun task hook" in hook_messages[2]
    assert "iv. postrun hook" in hook_messages[3]


def test_taskhooks_3(tmpdir, capsys):
    """checking results in the post run hooks"""
    foo = FunAddTwo(name="foo", a=1, cache_dir=tmpdir)

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook, the result is {outputs.out}")

    def myhook_postrun(task, result, *args):
        print(f"postrun hook, the result is {outputs.out}")

    foo.hooks.post_run = myhook_postrun
    foo.hooks.post_run_task = myhook_postrun_task
    foo()

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking that the postrun hooks have access to results
    assert "postrun task hook, the result is 3" in hook_messages[0]
    assert "postrun hook, the result is 3" in hook_messages[1]


def test_taskhooks_4(tmpdir, capsys):
    """task raises an error: postrun task should be called, postrun shouldn't be called"""
    foo = FunAddTwo(name="foo", a="one", cache_dir=tmpdir)

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook was called, result object is {result}")

    def myhook_postrun(task, result, *args):
        print("postrun hook should not be called")

    foo.hooks.post_run = myhook_postrun
    foo.hooks.post_run_task = myhook_postrun_task

    with pytest.raises(Exception):
        foo()

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # only post run task hook should be called
    assert len(hook_messages) == 1
    assert "postrun task hook was called" in hook_messages[0]


def test_traceback(tmpdir):
    """checking if the error raised in a function is properly returned;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def fun_error(x):
        raise Exception("Error from the function")

    task = fun_error(name="error", cache_dir=tmpdir).split("x", x=[3, 4])

    with pytest.raises(Exception, match="from the function") as exinfo:
        task()

    # getting error file from the error message
    error_file_match = str(exinfo.value).split("here: ")[-1].split("_error.pklz")[0]
    error_file = Path(error_file_match) / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    # reading error message from the pickle file
    error_tb = cp.loads(error_file.read_bytes())["error message"]
    # the error traceback should be a list and should point to a specific line in the function
    assert isinstance(error_tb, list)
    assert "in fun_error" in error_tb[-2]


def test_traceback_wf(tmpdir):
    """checking if the error raised in a function is properly returned by a workflow;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def fun_error(x):
        raise Exception("Error from the function")

    wf = Workflow(name="wf", input_spec=["x"], cache_dir=tmpdir).split("x", x=[3, 4])
    wf.add(fun_error(name="error", x=wf.lzin.x))
    wf.set_output([("out", wf.error.lzout.out)])

    with pytest.raises(Exception, match="Task error raised an error") as exinfo:
        wf()

    # getting error file from the error message
    error_file_match = str(exinfo.value).split("here: ")[-1].split("_error.pklz")[0]
    error_file = Path(error_file_match) / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    # reading error message from the pickle file
    error_tb = cp.loads(error_file.read_bytes())["error message"]
    # the error traceback should be a list and should point to a specific line in the function
    assert isinstance(error_tb, list)
    assert "in fun_error" in error_tb[-2]


def test_rerun_errored(tmpdir, capfd):
    """Test rerunning a task containing errors.
    Only the errored tasks should be rerun"""

    @python.define
    def pass_odds(x):
        if x % 2 == 0:
            print(f"x%2 = {x % 2} (error)\n")
            raise Exception("even error")
        else:
            print(f"x%2 = {x % 2}\n")
            return x

    task = pass_odds(name="pass_odds", cache_dir=tmpdir).split("x", x=[1, 2, 3, 4, 5])

    with pytest.raises(Exception, match="even error"):
        task()
    with pytest.raises(Exception, match="even error"):
        task()

    out, err = capfd.readouterr()
    stdout_lines = out.splitlines()

    tasks_run = 0
    errors_found = 0

    for line in stdout_lines:
        if "x%2" in line:
            tasks_run += 1
        if "(error)" in line:
            errors_found += 1

    # There should have been 5 messages of the form "x%2 = XXX" after calling task() the first time
    # and another 2 messagers after calling the second time
    assert tasks_run == 7
    assert errors_found == 4


@attr.s(auto_attribs=True)
class A:
    x: int


def test_object_input():
    """Test function tasks with object inputs"""

    @python.define
    def testfunc(a: A):
        return a.x

    outputs = testfunc(a=A(x=7))()
    assert outputs.out == 7


def test_argstr_formatting():
    @attr.define
    class Inputs:
        a1_field: str
        b2_field: float
        c3_field: ty.Dict[str, str]
        d4_field: ty.List[str]

    inputs = Inputs("1", 2.0, {"c": "3"}, ["4"])
    assert (
        argstr_formatting(
            "{a1_field} {b2_field:02f} -test {c3_field[c]} -me {d4_field[0]}",
            inputs,
        )
        == "1 2.000000 -test 3 -me 4"
    )
