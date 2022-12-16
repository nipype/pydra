import typing as ty
import os, sys
import attr
import pytest
import cloudpickle as cp
from pathlib import Path
import re
import json
import glob as glob
from ... import mark
from ..core import Workflow
from ..task import AuditFlag, ShellCommandTask, DockerTask, SingularityTask
from ...utils.messenger import FileMessenger, PrintMessenger, collect_messages
from .utils import gen_basic_wf, use_validator, Submitter
from ..specs import (
    MultiInputObj,
    MultiOutputObj,
    SpecInfo,
    FunctionSpec,
    BaseSpec,
    ShellSpec,
    File,
)
from ..helpers import hash_file

no_win = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="docker/singularity command not adjusted for windows",
)


@mark.task
def funaddtwo(a):
    return a + 2


def test_output():
    nn = funaddtwo(a=3)
    res = nn._run()
    assert res.output.out == 5


def test_name_conflict():
    """raise error if task name conflicts with a class attribute or method"""
    with pytest.raises(ValueError) as excinfo1:
        nn = funaddtwo(name="split", a=3)
    assert "Cannot use names of attributes or methods" in str(excinfo1.value)
    with pytest.raises(ValueError) as excinfo2:
        nn = funaddtwo(name="checksum", a=3)
    assert "Cannot use names of attributes or methods" in str(excinfo2.value)


def test_numpy(use_validator):
    """checking if mark.task works for numpy functions"""
    np = pytest.importorskip("numpy")
    fft = mark.annotate({"a": np.ndarray, "return": np.ndarray})(np.fft.fft)
    fft = mark.task(fft)()
    arr = np.array([[1, 10], [2, 20]])
    fft.inputs.a = arr
    res = fft()
    assert np.allclose(np.fft.fft(arr), res.output.out)


@pytest.mark.xfail(reason="cp.dumps(func) depends on the system/setup, TODO!!")
def test_checksum():
    nn = funaddtwo(a=3)
    assert (
        nn.checksum
        == "FunctionTask_abb4e7cc03b13d0e73884b87d142ed5deae6a312275187a9d8df54407317d7d3"
    )


def test_annotated_func(use_validator):
    @mark.task
    def testfunc(
        a: int, b: float = 0.1
    ) -> ty.NamedTuple("Output", [("out_out", float)]):
        return a + b

    funky = testfunc(a=1)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 1
    assert getattr(funky.inputs, "b") == 0.1
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == {"out_out"}
    # assert funky.inputs.hash == '17772c3aec9540a8dd3e187eecd2301a09c9a25c6e371ddd86e31e3a1ecfeefa'
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out_out")
    assert result.output.out_out == 1.1

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    funky.result()  # should not recompute
    funky.inputs.a = 2
    # assert funky.checksum == '537d25885fd2ea5662b7701ba02c132c52a9078a3a2d56aa903a777ea90e5536'
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out_out == 2.1

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: int",
        "- b: float (default: 0.1)",
        "- _func: str",
        "Output Parameters:",
        "- out_out: float",
    ]


def test_annotated_func_multreturn(use_validator):
    """the function has two elements in the return statement"""

    @mark.task
    def testfunc(
        a: float,
    ) -> ty.NamedTuple("Output", [("fractional", float), ("integer", int)]):
        import math

        return math.modf(a)[0], int(math.modf(a)[1])

    funky = testfunc(a=3.5)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 3.5
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == {"fractional", "integer"}
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")
    assert hasattr(result, "output")
    assert hasattr(result.output, "fractional")
    assert result.output.fractional == 0.5
    assert hasattr(result.output, "integer")
    assert result.output.integer == 3

    help = funky.help(returnhelp=True)
    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: float",
        "- _func: str",
        "Output Parameters:",
        "- fractional: float",
        "- integer: int",
    ]


def test_annotated_input_func_1(use_validator):
    """the function with annotated input (float)"""

    @mark.task
    def testfunc(a: float):
        return a

    funky = testfunc(a=3.5)
    assert getattr(funky.inputs, "a") == 3.5


def test_annotated_input_func_2(use_validator):
    """the function with annotated input (int, but float provided)"""

    @mark.task
    def testfunc(a: int):
        return a

    with pytest.raises(TypeError):
        funky = testfunc(a=3.5)


def test_annotated_input_func_2a(use_validator):
    """the function with annotated input (int, but float provided)"""

    @mark.task
    def testfunc(a: int):
        return a

    funky = testfunc()
    with pytest.raises(TypeError):
        funky.inputs.a = 3.5


def test_annotated_input_func_3(use_validator):
    """the function with annotated input (list)"""

    @mark.task
    def testfunc(a: list):
        return sum(a)

    funky = testfunc(a=[1, 3.5])
    assert getattr(funky.inputs, "a") == [1, 3.5]


def test_annotated_input_func_3a():
    """the function with annotated input (list of floats)"""

    @mark.task
    def testfunc(a: ty.List[float]):
        return sum(a)

    funky = testfunc(a=[1.0, 3.5])
    assert getattr(funky.inputs, "a") == [1.0, 3.5]


def test_annotated_input_func_3b(use_validator):
    """the function with annotated input
    (list of floats - int and float provided, should be fine)
    """

    @mark.task
    def testfunc(a: ty.List[float]):
        return sum(a)

    funky = testfunc(a=[1, 3.5])
    assert getattr(funky.inputs, "a") == [1, 3.5]


def test_annotated_input_func_3c_excep(use_validator):
    """the function with annotated input
    (list of ints - int and float provided, should raise an error)
    """

    @mark.task
    def testfunc(a: ty.List[int]):
        return sum(a)

    with pytest.raises(TypeError):
        funky = testfunc(a=[1, 3.5])


def test_annotated_input_func_4(use_validator):
    """the function with annotated input (dictionary)"""

    @mark.task
    def testfunc(a: dict):
        return sum(a.values())

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky.inputs, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4a(use_validator):
    """the function with annotated input (dictionary of floats)"""

    @mark.task
    def testfunc(a: ty.Dict[str, float]):
        return sum(a.values())

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky.inputs, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4b_excep(use_validator):
    """the function with annotated input (dictionary of ints, but float provided)"""

    @mark.task
    def testfunc(a: ty.Dict[str, int]):
        return sum(a.values())

    with pytest.raises(TypeError):
        funky = testfunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_5(use_validator):
    """the function with annotated more complex input type (ty.List in ty.Dict)
    the validator should simply check if values of dict are lists
    so no error for 3.5
    """

    @mark.task
    def testfunc(a: ty.Dict[str, ty.List[int]]):
        return sum(a["el1"])

    funky = testfunc(a={"el1": [1, 3.5]})
    assert getattr(funky.inputs, "a") == {"el1": [1, 3.5]}


def test_annotated_input_func_5a_except(use_validator):
    """the function with annotated more complex input type (ty.Dict in ty.Dict)
    list is provided as a dict value (instead a dict), so error is raised
    """

    @mark.task
    def testfunc(a: ty.Dict[str, ty.Dict[str, float]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        funky = testfunc(a={"el1": [1, 3.5]})


def test_annotated_input_func_6(use_validator):
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union
    """

    @mark.task
    def testfunc(a: ty.Dict[str, ty.Union[float, int]]):
        return sum(a["el1"])

    funky = testfunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky.inputs, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_6a_excep(use_validator):
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union and raise an error for 3.5
    """

    @mark.task
    def testfunc(a: ty.Dict[str, ty.Union[str, int]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        funky = testfunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_7(use_validator):
    """the function with annotated input (float)
    the task has a splitter, so list of float is provided
    it should work, the validator tries to guess if this is a field with a splitter
    """

    @mark.task
    def testfunc(a: float):
        return a

    funky = testfunc(a=[3.5, 2.1]).split("a")
    assert getattr(funky.inputs, "a") == [3.5, 2.1]


def test_annotated_input_func_7a_excep(use_validator):
    """the function with annotated input (int) and splitter
    list of float provided - should raise an error (list of int would be fine)
    """

    @mark.task
    def testfunc(a: int):
        return a

    with pytest.raises(TypeError):
        funky = testfunc(a=[3.5, 2.1]).split("a")


def test_annotated_input_func_8():
    """the function with annotated input as MultiInputObj
    a single value is provided and should be converted to a list
    """

    @mark.task
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc(a=3.5)
    assert getattr(funky.inputs, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_input_func_8a():
    """the function with annotated input as MultiInputObj
    a 1-el list is provided so shouldn't be changed
    """

    @mark.task
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc(a=[3.5])
    assert getattr(funky.inputs, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_input_func_8b():
    """the function with annotated input as MultiInputObj
    a single value is provided after initial. the task
    (input should still be converted to a list)
    """

    @mark.task
    def testfunc(a: MultiInputObj):
        return len(a)

    funky = testfunc()
    # setting a after init
    funky.inputs.a = 3.5
    assert getattr(funky.inputs, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_annotated_func_multreturn_exception(use_validator):
    """function has two elements in the return statement,
    but three element provided in the spec - should raise an error
    """

    @mark.task
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
    @mark.task
    def testfunc(a, b) -> int:
        return a + b

    funky = testfunc(a=10, b=20)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 10
    assert getattr(funky.inputs, "b") == 20
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == {"out"}
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == 30

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.inputs.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out == 31
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: str",
        "Output Parameters:",
        "- out: int",
    ]


def test_halfannotated_func_multreturn():
    @mark.task
    def testfunc(a, b) -> (int, int):
        return a + 1, b + 1

    funky = testfunc(a=10, b=20)
    assert hasattr(funky.inputs, "a")
    assert hasattr(funky.inputs, "b")
    assert hasattr(funky.inputs, "_func")
    assert getattr(funky.inputs, "a") == 10
    assert getattr(funky.inputs, "b") == 20
    assert getattr(funky.inputs, "_func") is not None
    assert set(funky.output_names) == {"out1", "out2"}
    assert funky.__class__.__name__ + "_" + funky.inputs.hash == funky.checksum

    result = funky()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out1")
    assert result.output.out1 == 11

    assert os.path.exists(funky.cache_dir / funky.checksum / "_result.pklz")

    funky.result()  # should not recompute
    funky.inputs.a = 11
    assert funky.result() is None
    funky()
    result = funky.result()
    assert result.output.out1 == 12
    help = funky.help(returnhelp=True)

    assert help == [
        "Help for FunctionTask",
        "Input Parameters:",
        "- a: _empty",
        "- b: _empty",
        "- _func: str",
        "Output Parameters:",
        "- out1: int",
        "- out2: int",
    ]


def test_notannotated_func():
    @mark.task
    def no_annots(c, d):
        return c + d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask.inputs, "c")
    assert hasattr(natask.inputs, "d")
    assert hasattr(natask.inputs, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == 20.2


def test_notannotated_func_returnlist():
    @mark.task
    def no_annots(c, d):
        return [c, d]

    natask = no_annots(c=17, d=3.2)
    result = natask._run()
    assert hasattr(result.output, "out")
    assert result.output.out == [17, 3.2]


def test_halfannotated_func_multrun_returnlist():
    @mark.task
    def no_annots(c, d) -> (list, float):
        return [c, d], c + d

    natask = no_annots(c=17, d=3.2)
    result = natask._run()

    assert hasattr(result.output, "out1")
    assert hasattr(result.output, "out2")
    assert result.output.out1 == [17, 3.2]
    assert result.output.out2 == 20.2


def test_notannotated_func_multreturn():
    """no annotation and multiple values are returned
    all elements should be returned as a tuple and set to "out"
    """

    @mark.task
    def no_annots(c, d):
        return c + d, c - d

    natask = no_annots(c=17, d=3.2)
    assert hasattr(natask.inputs, "c")
    assert hasattr(natask.inputs, "d")
    assert hasattr(natask.inputs, "_func")

    result = natask._run()
    assert hasattr(result, "output")
    assert hasattr(result.output, "out")
    assert result.output.out == (20.2, 13.8)


def test_input_spec_func_1(use_validator):
    """the function w/o annotated, but input_spec is used"""

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=float, metadata={"help_string": "input a"}))],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky.inputs, "a") == 3.5


def test_input_spec_func_1a_except(use_validator):
    """the function w/o annotated, but input_spec is used
    a TypeError is raised (float is provided instead of int)
    """

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=int, metadata={"help_string": "input a"}))],
        bases=(FunctionSpec,),
    )
    with pytest.raises(TypeError):
        funky = testfunc(a=3.5, input_spec=my_input_spec)


def test_input_spec_func_1b_except(use_validator):
    """the function w/o annotated, but input_spec is used
    metadata checks raise an error
    """

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(type=float, metadata={"position": 1, "help_string": "input a"}),
            )
        ],
        bases=(FunctionSpec,),
    )
    with pytest.raises(AttributeError, match="only these keys are supported"):
        funky = testfunc(a=3.5, input_spec=my_input_spec)


def test_input_spec_func_1d_except(use_validator):
    """the function w/o annotated, but input_spec is used
    input_spec doesn't contain 'a' input, an error is raised
    """

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(name="Input", fields=[], bases=(FunctionSpec,))
    funky = testfunc(a=3.5, input_spec=my_input_spec)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        funky()


def test_input_spec_func_2(use_validator):
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    """

    @mark.task
    def testfunc(a: int):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", attr.ib(type=float, metadata={"help_string": "input a"}))],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky.inputs, "a") == 3.5


def test_input_spec_func_2a(use_validator):
    """the function with annotation, and the task has input_spec,
    input_spec changes the type of the input (so error is not raised)
    using the shorter syntax
    """

    @mark.task
    def testfunc(a: int):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[("a", float, {"help_string": "input a"})],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky.inputs, "a") == 3.5


def test_input_spec_func_3(use_validator):
    """the function w/o annotated, but input_spec is used
    additional keys (allowed_values) are used in metadata
    """

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int,
                    metadata={"help_string": "input a", "allowed_values": [0, 1, 2]},
                ),
            )
        ],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    assert getattr(funky.inputs, "a") == 2


def test_input_spec_func_3a_except(use_validator):
    """the function w/o annotated, but input_spec is used
    allowed_values is used in metadata and the ValueError is raised
    """

    @mark.task
    def testfunc(a):
        return a

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int,
                    metadata={"help_string": "input a", "allowed_values": [0, 1, 2]},
                ),
            )
        ],
        bases=(FunctionSpec,),
    )

    with pytest.raises(ValueError, match="value of a has to be"):
        funky = testfunc(a=3, input_spec=my_input_spec)


def test_input_spec_func_4(use_validator):
    """the function with a default value for b
    but b is set as mandatory in the input_spec, so error is raised if not provided
    """

    @mark.task
    def testfunc(a, b=1):
        return a + b

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int, metadata={"help_string": "input a", "mandatory": True}
                ),
            ),
            (
                "b",
                attr.ib(
                    type=int, metadata={"help_string": "input b", "mandatory": True}
                ),
            ),
        ],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    with pytest.raises(Exception, match="b is mandatory"):
        funky()


def test_input_spec_func_4a(use_validator):
    """the function with a default value for b and metadata in the input_spec
    has a different default value, so value from the function is overwritten
    """

    @mark.task
    def testfunc(a, b=1):
        return a + b

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            (
                "a",
                attr.ib(
                    type=int, metadata={"help_string": "input a", "mandatory": True}
                ),
            ),
            ("b", attr.ib(type=int, default=10, metadata={"help_string": "input b"})),
        ],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=2, input_spec=my_input_spec)
    res = funky()
    assert res.output.out == 12


def test_input_spec_func_5():
    """the FunctionTask with input_spec, a input has MultiInputObj type
    a single value is provided and should be converted to a list
    """

    @mark.task
    def testfunc(a):
        return len(a)

    my_input_spec = SpecInfo(
        name="Input",
        fields=[
            ("a", attr.ib(type=MultiInputObj, metadata={"help_string": "input a"}))
        ],
        bases=(FunctionSpec,),
    )

    funky = testfunc(a=3.5, input_spec=my_input_spec)
    assert getattr(funky.inputs, "a") == [3.5]
    res = funky()
    assert res.output.out == 1


def test_output_spec_func_1(use_validator):
    """the function w/o annotated, but output_spec is used"""

    @mark.task
    def testfunc(a):
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=float, metadata={"help_string": "output"}))],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_1a_except(use_validator):
    """the function w/o annotated, but output_spec is used
    float returned instead of int - TypeError
    """

    @mark.task
    def testfunc(a):
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=int, metadata={"help_string": "output"}))],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    with pytest.raises(TypeError):
        res = funky()


def test_output_spec_func_2(use_validator):
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    """

    @mark.task
    def testfunc(a) -> int:
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", attr.ib(type=float, metadata={"help_string": "output"}))],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_2a(use_validator):
    """the function w/o annotated, but output_spec is used
    output_spec changes the type of the output (so error is not raised)
    using a shorter syntax
    """

    @mark.task
    def testfunc(a) -> int:
        return a

    my_output_spec = SpecInfo(
        name="Output",
        fields=[("out1", float, {"help_string": "output"})],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out1 == 3.5


def test_output_spec_func_3(use_validator):
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 2-el list, so converter doesn't do anything
    """

    @mark.task
    def testfunc(a, b):
        return [a, b]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_list",
                attr.ib(type=MultiOutputObj, metadata={"help_string": "output"}),
            )
        ],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, b=1, output_spec=my_output_spec)
    res = funky()
    assert res.output.out_list == [3.5, 1]


def test_output_spec_func_4(use_validator):
    """the function w/o annotated, but output_spec is used
    MultiOutputObj is used, output is a 1el list, so converter return the element
    """

    @mark.task
    def testfunc(a):
        return [a]

    my_output_spec = SpecInfo(
        name="Output",
        fields=[
            (
                "out_1el",
                attr.ib(type=MultiOutputObj, metadata={"help_string": "output"}),
            )
        ],
        bases=(BaseSpec,),
    )

    funky = testfunc(a=3.5, output_spec=my_output_spec)
    res = funky()
    assert res.output.out_1el == 3.5


def test_exception_func():
    @mark.task
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """checking if None is properly returned as the result"""

    @mark.task
    def fun_none(x):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out is None


def test_result_none_2():
    """checking if None is properly set for all outputs"""

    @mark.task
    def fun_none(x) -> (ty.Any, ty.Any):
        return None

    task = fun_none(name="none", x=3)
    res = task()
    assert res.output.out1 is None
    assert res.output.out2 is None


def test_audit_prov(tmpdir, use_validator):
    @mark.task
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
    @mark.task
    def testfunc(a: int, b: float = 0.1) -> ty.NamedTuple("Output", [("out", float)]):
        return a + b

    from glob import glob

    funky = testfunc(a=2, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    funky.cache_dir = tmpdir
    funky()
    message_path = tmpdir / funky.checksum / "messages"

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file, "r") as f:
            data = json.load(f)
            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "testfunc" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert None == data["Label"]
            if "AssociatedWith" in data:
                assert None == data["AssociatedWith"]

    # assert any(json_content)


def test_audit_shellcommandtask(tmpdir):
    args = "-l"
    shelly = ShellCommandTask(
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
        with open(file, "r") as f:
            data = json.load(f)

            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "shelly" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert data["Label"] == None

            if "Command" in data:
                command_content.append(True)
                assert "ls -l" == data["Command"]

    assert any(command_content)


def test_audit_shellcommandtask_file(tmpdir):
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
    shutil.copy("test.txt", tmpdir)
    shutil.copy("test2.txt", tmpdir)

    cmd = "cat"
    file_in = tmpdir / "test.txt"
    file_in_2 = tmpdir / "test2.txt"
    test_file_hash = hash_file(file_in)
    test_file_hash_2 = hash_file(file_in_2)
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
                        "help_string": "text",
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
                        "help_string": "text",
                        "mandatory": True,
                    },
                ),
            ),
        ],
        bases=(ShellSpec,),
    )
    shelly = ShellCommandTask(
        name="shelly",
        in_file=file_in,
        in_file_2=file_in_2,
        input_spec=my_input_spec,
        executable=cmd,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    shelly.cache_dir = tmpdir
    shelly()
    message_path = tmpdir / shelly.checksum / "messages"
    for file in glob.glob(str(message_path) + "/*.jsonld"):
        with open(file, "r") as x:
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
    shelly = ShellCommandTask(
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
        with open(file, "r") as f:
            data = json.load(f)
            if "AssociatedWith" in data:
                if version_cmd in data["AssociatedWith"]:
                    version_content.append(True)

    assert any(version_content)


def test_audit_prov_messdir_1(tmpdir, use_validator):
    """customized messenger dir"""

    @mark.task
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


def test_audit_prov_messdir_2(tmpdir, use_validator):
    """customized messenger dir in init"""

    @mark.task
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


def test_audit_prov_wf(tmpdir, use_validator):
    """FileMessenger for wf"""

    @mark.task
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
    wf.inputs.x = 2

    wf(plugin="cf")
    # default path
    message_path = tmpdir / wf.checksum / "messages"
    assert message_path.exists()

    collect_messages(tmpdir / wf.checksum, message_path, ld_op="compact")
    assert (tmpdir / wf.checksum / "messages.jsonld").exists()


def test_audit_all(tmpdir, use_validator):
    @mark.task
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
    shelly = ShellCommandTask(name="shelly", executable=cmd)
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    shelly = ShellCommandTask(executable=cmd[0], args=cmd[1:])
    assert shelly.inputs.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    res = shelly._run()
    assert res.output.return_code == 0
    assert res.output.stdout == " ".join(cmd[1:]) + "\n"


def test_container_cmds(tmpdir):
    containy = DockerTask(name="containy", executable="pwd")
    with pytest.raises(AttributeError) as excinfo:
        containy.cmdline
    assert "mandatory" in str(excinfo.value)
    containy.inputs.image = "busybox"
    assert containy.cmdline


@no_win
def test_docker_cmd(tmpdir):
    docky = DockerTask(name="docky", executable="pwd", image="busybox")
    assert (
        docky.cmdline
        == f"docker run --rm -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    )
    docky.inputs.container_xargs = ["--rm", "-it"]
    assert (
        docky.cmdline
        == f"docker run --rm -it -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    )
    # TODO: we probably don't want to support container_path
    # docky.inputs.bindings = [
    #     ("/local/path", "/container/path", "ro"),
    #     ("/local2", "/container2", None),
    # ]
    # assert docky.cmdline == (
    #     "docker run --rm -it -v /local/path:/container/path:ro"
    #     f" -v /local2:/container2:rw -v {docky.output_dir}:/output_pydra:rw -w /output_pydra busybox pwd"
    # )


@no_win
def test_singularity_cmd(tmpdir):
    # todo how this should be done?
    image = "library://sylabsed/linux/alpine"
    singu = SingularityTask(name="singi", executable="pwd", image=image)
    assert (
        singu.cmdline
        == f"singularity exec -B {singu.output_dir}:/output_pydra:rw --pwd /output_pydra {image} pwd"
    )
    # TODO: we probably don't want to support container_path
    # singu.inputs.bindings = [
    #     ("/local/path", "/container/path", "ro"),
    #     ("/local2", "/container2", None),
    # ]
    # assert singu.cmdline == (
    #     "singularity exec -B /local/path:/container/path:ro"
    #     f" -B /local2:/container2:rw -B {singu.output_dir}:/output_pydra:rw --pwd /output_pydra {image} pwd"
    # )


def test_functask_callable(tmpdir):
    # no submitter or plugin
    foo = funaddtwo(a=1)
    res = foo()
    assert res.output.out == 3
    assert foo.plugin is None

    # plugin
    bar = funaddtwo(a=2)
    res = bar(plugin="cf")
    assert res.output.out == 4
    assert bar.plugin is None

    foo2 = funaddtwo(a=3)
    foo2.plugin = "cf"
    res = foo2()
    assert res.output.out == 5
    assert foo2.plugin == "cf"


def test_taskhooks_1(tmpdir, capsys):
    foo = funaddtwo(name="foo", a=1, cache_dir=tmpdir)
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
    foo.inputs.a = 2  # ensure not pre-cached
    foo()
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured

    # hooks are independent across tasks by default
    bar = funaddtwo(name="bar", a=3, cache_dir=tmpdir)
    assert bar.hooks is not foo.hooks
    # but can be shared across tasks
    bar.hooks = foo.hooks
    # and workflows
    wf = gen_basic_wf()
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
    foo = funaddtwo(name="foo", a=1, cache_dir=tmpdir)

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
    foo = funaddtwo(name="foo", a=1, cache_dir=tmpdir)

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook, the result is {result.output.out}")

    def myhook_postrun(task, result, *args):
        print(f"postrun hook, the result is {result.output.out}")

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
    foo = funaddtwo(name="foo", a="one", cache_dir=tmpdir)

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook was called, result object is {result}")

    def myhook_postrun(task, result, *args):
        print(f"postrun hook should not be called")

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

    @mark.task
    def fun_error(x):
        raise Exception("Error from the function")

    task = fun_error(name="error", x=[3, 4], cache_dir=tmpdir).split("x")

    with pytest.raises(Exception, match="from the function") as exinfo:
        res = task()

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

    @mark.task
    def fun_error(x):
        raise Exception("Error from the function")

    wf = Workflow(name="wf", input_spec=["x"], x=[3, 4], cache_dir=tmpdir).split("x")
    wf.add(fun_error(name="error", x=wf.lzin.x))
    wf.set_output([("out", wf.error.lzout.out)])

    with pytest.raises(Exception, match="Task error raised an error") as exinfo:
        res = wf()

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

    @mark.task
    def pass_odds(x):
        if x % 2 == 0:
            print(f"x%2 = {x % 2} (error)\n")
            raise Exception("even error")
        else:
            print(f"x%2 = {x % 2}\n")
            return x

    task = pass_odds(name="pass_odds", x=[1, 2, 3, 4, 5], cache_dir=tmpdir).split("x")

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

    @mark.task
    def testfunc(a: A):
        return a.x

    result = testfunc(a=A(x=7))()
    assert result.output.out == 7
