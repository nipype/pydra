import typing as ty
import os
import sys
import attrs
import shutil
import pytest
import cloudpickle as cp
from pathlib import Path
import json
import glob as glob
from pydra.compose import python, shell, workflow
from pydra.compose.shell.templating import argstr_formatting
from pydra.utils.messenger import FileMessenger, PrintMessenger, collect_messages
from pydra.engine.audit import AuditFlag
from pydra.engine.hooks import TaskHooks
from pydra.utils.general import task_fields, task_help
from pydra.engine.submitter import Submitter
from pydra.engine.job import Job
from pydra.utils.general import default_run_cache_dir
from pydra.utils.typing import (
    MultiInputObj,
    MultiOutputObj,
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
        nn._checksum
        == "PythonTask_abb4e7cc03b13d0e73884b87d142ed5deae6a312275187a9d8df54407317d7d3"
    )


def test_annotated_func():
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
        default_run_cache_dir / f"python-{funky._hash}" / "_result.pklz"
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


def test_annotated_func_dictreturn(tmp_path: Path):
    """Test mapping from returned dictionary to output definition."""

    @python.define(outputs={"sum": int, "mul": int | None})
    def TestFunc(a: int, b: int):
        return dict(sum=a + b, diff=a - b)

    task = TestFunc(a=2, b=3)
    outputs = task(cache_dir=tmp_path)

    # Part of the annotation and returned, should be exposed to output.
    assert outputs.sum == 5

    # Part of the annotation but not returned, should be coalesced to None
    assert outputs.mul is None

    # Not part of the annotation, should be discarded.
    assert not hasattr(outputs, "diff")


def test_annotated_func_multreturn():
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
        default_run_cache_dir / f"python-{funky._hash}" / "_result.pklz"
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


def test_annotated_input_func_1():
    """the function with annotated input (float)"""

    @python.define
    def TestFunc(a: float):
        return a

    funky = TestFunc(a=3.5)
    assert getattr(funky, "a") == 3.5


def test_annotated_input_func_2():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def TestFunc(a: int):
        return a

    with pytest.raises(TypeError):
        TestFunc(a=3.5)


def test_annotated_input_func_2a():
    """the function with annotated input (int, but float provided)"""

    @python.define
    def TestFunc(a: int):
        return a

    funky = TestFunc()
    with pytest.raises(TypeError):
        funky.a = 3.5


def test_annotated_input_func_3():
    """the function with annotated input (list)"""

    @python.define
    def TestFunc(a: list):
        return sum(a)

    funky = TestFunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_annotated_input_func_3a():
    """the function with annotated input (list of floats)"""

    @python.define
    def TestFunc(a: ty.List[float]):
        return sum(a)

    funky = TestFunc(a=[1.0, 3.5])
    assert getattr(funky, "a") == [1.0, 3.5]


def test_annotated_input_func_3b():
    """the function with annotated input
    (list of floats - int and float provided, should be fine)
    """

    @python.define
    def TestFunc(a: ty.List[float]):
        return sum(a)

    funky = TestFunc(a=[1, 3.5])
    assert getattr(funky, "a") == [1, 3.5]


def test_annotated_input_func_3c_excep():
    """the function with annotated input
    (list of ints - int and float provided, should raise an error)
    """

    @python.define
    def TestFunc(a: ty.List[int]):
        return sum(a)

    with pytest.raises(TypeError):
        TestFunc(a=[1, 3.5])


def test_annotated_input_func_4():
    """the function with annotated input (dictionary)"""

    @python.define
    def TestFunc(a: dict):
        return sum(a.values())

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4a():
    """the function with annotated input (dictionary of floats)"""

    @python.define
    def TestFunc(a: ty.Dict[str, float]):
        return sum(a.values())

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_4b_excep():
    """the function with annotated input (dictionary of ints, but float provided)"""

    @python.define
    def TestFunc(a: ty.Dict[str, int]):
        return sum(a.values())

    with pytest.raises(TypeError):
        TestFunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_5():
    """the function with annotated more complex input type (ty.List in ty.Dict)
    the validator should simply check if values of dict are lists
    so no error for 3.5
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.List]):
        return sum(a["el1"])

    funky = TestFunc(a={"el1": [1, 3.5]})
    assert getattr(funky, "a") == {"el1": [1, 3.5]}


def test_annotated_input_func_5a_except():
    """the function with annotated more complex input type (ty.Dict in ty.Dict)
    list is provided as a dict value (instead a dict), so error is raised
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Dict[str, float]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        TestFunc(a={"el1": [1, 3.5]})


def test_annotated_input_func_6():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Union[float, int]]):
        return sum(a["el1"])

    funky = TestFunc(a={"el1": 1, "el2": 3.5})
    assert getattr(funky, "a") == {"el1": 1, "el2": 3.5}


def test_annotated_input_func_6a_excep():
    """the function with annotated more complex input type (ty.Union in ty.Dict)
    the validator should unpack values from the Union and raise an error for 3.5
    """

    @python.define
    def TestFunc(a: ty.Dict[str, ty.Union[str, int]]):
        return sum(a["el1"])

    with pytest.raises(TypeError):
        TestFunc(a={"el1": 1, "el2": 3.5})


def test_annotated_input_func_7():
    """the function with annotated input (float)
    the task has a splitter, so list of float is provided
    it should work, the validator tries to guess if this is a field with a splitter
    """

    @python.define
    def TestFunc(a: float):
        return a

    funky = TestFunc().split("a", a=[3.5, 2.1])
    assert getattr(funky, "a") == [3.5, 2.1]


def test_annotated_input_func_7a_excep():
    """the function with annotated input (int) and splitter
    list of float provided - should raise an error (list of int would be fine)
    """

    @python.define
    def TestFunc(a: int):
        return a

    with pytest.raises(TypeError):
        TestFunc(a=[3.5, 2.1]).split("a")


def test_annotated_input_func_8():
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


def test_annotated_input_func_8a():
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


def test_annotated_input_func_8b():
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


def test_annotated_func_multreturn_exception():
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


def test_halfannotated_func(tmp_path):

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

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

    outputs = funky(cache_dir=cache_dir)
    assert hasattr(outputs, "out")
    assert outputs.out == 30

    assert Path(cache_dir / f"python-{funky._hash}" / "_result.pklz").exists()

    funky(cache_dir=cache_dir)  # should not recompute
    funky.a = 11
    assert not Path(cache_dir / f"python-{funky._hash}").exists()
    outputs = funky(cache_dir=cache_dir)
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


def test_halfannotated_func_multreturn(tmp_path):

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

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

    outputs = funky(cache_dir=cache_dir)
    assert hasattr(outputs, "out1")
    assert outputs.out1 == 11

    assert Path(cache_dir / f"python-{funky._hash}" / "_result.pklz").exists()

    funky(cache_dir=cache_dir)  # should not recompute
    funky.a = 11
    assert not Path(cache_dir / f"python-{funky._hash}" / "_result.pklz").exists()
    outputs = funky(cache_dir=cache_dir)
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


def test_notannotated_func():
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


def test_notannotated_func_returnlist():
    @python.define
    def NoAnnots(c, d):
        return [c, d]

    no_annots = NoAnnots(c=17, d=3.2)
    outputs = no_annots()
    assert hasattr(outputs, "out")
    assert outputs.out == [17, 3.2]


def test_halfannotated_func_multrun_returnlist():
    @python.define(outputs=["out1", "out2"])
    def NoAnnots(c, d) -> tuple[list, float]:
        return [c, d], c + d

    no_annots = NoAnnots(c=17, d=3.2)
    outputs = no_annots()

    assert hasattr(outputs, "out1")
    assert hasattr(outputs, "out2")
    assert outputs.out1 == [17, 3.2]
    assert outputs.out2 == 20.2


def test_notannotated_func_multreturn():
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


def test_exception_func():
    @python.define
    def raise_exception(c, d):
        raise Exception()

    bad_funk = raise_exception(c=17, d=3.2)
    assert pytest.raises(Exception, bad_funk)


def test_result_none_1():
    """checking if None is properly returned as the result"""

    @python.define
    def FunNone(x):
        return None

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out is None


def test_result_none_2():
    """checking if None is properly set for all outputs"""

    @python.define(outputs=["out1", "out2"])
    def FunNone(x) -> tuple[ty.Any, ty.Any]:
        return None  # Do we actually want this behaviour?

    task = FunNone(x=3)
    outputs = task()
    assert outputs.out1 is None
    assert outputs.out2 is None


def test_audit_prov(
    tmpdir,
):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_dir=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # saving the audit message into the file
    funky = TestFunc(a=2)
    funky(cache_dir=tmpdir, audit_flags=AuditFlag.PROV, messengers=FileMessenger())
    # this should be the default loctaion
    message_path = tmpdir / funky._checksum / "messages"
    assert (tmpdir / funky._checksum / "messages").exists()

    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()


def test_audit_task(tmpdir):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    from glob import glob

    funky = TestFunc(a=2)
    funky(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / funky._checksum / "messages"

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)
            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "main" in data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert None is data["Label"]
            if "AssociatedWith" in data:
                assert None is data["AssociatedWith"]

    # assert any(json_content)


def test_audit_shellcommandtask(tmpdir):
    Shelly = shell.define("ls -l<long=True>")

    from glob import glob

    shelly = Shelly()

    shelly(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / shelly._checksum / "messages"
    # go through each jsonld file in message_path and check if the label field exists

    command_content = []

    for file in glob(str(message_path) + "/*.jsonld"):
        with open(file) as f:
            data = json.load(f)

            if "@type" in data:
                if "AssociatedWith" in data:
                    assert "main" == data["Label"]

            if "@type" in data:
                if data["@type"] == "input":
                    assert data["Label"] is None

            if "Command" in data:
                command_content.append(True)
                assert "ls -l" == data["Command"]

    assert any(command_content)


def test_audit_shellcommandtask_file(tmp_path):
    # sourcery skip: use-fstring-for-concatenation
    # create test.txt file with "This is a test" in it in the tmpdir
    # create txt file in cwd
    test1_file = tmp_path / "test.txt"
    test2_file = tmp_path / "test2.txt"
    with open(test1_file, "w") as f:
        f.write("This is a test")

    with open(test2_file, "w") as f:
        f.write("This is a test")

    cmd = "cat"
    file_in = File(test1_file)
    file_in_2 = File(test2_file)
    test_file_hash = hash_function(file_in)
    test_file_hash_2 = hash_function(file_in_2)
    Shelly = shell.define(
        cmd,
        inputs={
            "in_file": shell.arg(
                type=File,
                position=1,
                argstr="",
                help="text",
            ),
            "in_file_2": shell.arg(
                type=File,
                position=2,
                argstr="",
                help="text",
            ),
        },
    )
    shelly = Shelly(
        in_file=file_in,
        in_file_2=file_in_2,
    )
    shelly(
        cache_dir=tmp_path,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmp_path / shelly._hash / "messages"
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
    cmd = "less test_task.py"
    Shelly = shell.define(cmd)
    shelly = Shelly()

    import glob

    shelly(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    message_path = tmpdir / shelly._checksum / "messages"
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

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_dir=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # saving the audit message into the file
    funky = TestFunc(a=2)
    # user defined path
    message_path = tmpdir / funky._checksum / "my_messages"
    # providing messenger_dir for audit
    funky_task = Job(
        task=funky,
        submitter=Submitter(
            cache_dir=tmpdir, audit_flags=AuditFlag.PROV, messengers=FileMessenger()
        ),
        name="funky",
    )
    funky_task.audit.messenger_args = dict(message_dir=message_path)
    funky_task.run()
    assert (tmpdir / funky._checksum / "my_messages").exists()

    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()


def test_audit_prov_messdir_2(
    tmpdir,
):
    """customized messenger dir in init"""

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    # printing the audit message
    funky = TestFunc(a=1)
    funky(cache_dir=tmpdir, audit_flags=AuditFlag.PROV, messengers=PrintMessenger())

    # user defined path (doesn't depend on checksum, can be defined before init)
    message_path = tmpdir / "my_messages"
    # saving the audit message into the file
    funky = TestFunc(a=2)
    # providing messenger_dir for audit
    funky(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    assert (tmpdir / "my_messages").exists()

    collect_messages(tmpdir, message_path, ld_op="compact")
    assert (tmpdir / "messages.jsonld").exists()


def test_audit_prov_wf(
    tmpdir,
):
    """FileMessenger for wf"""

    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    @workflow.define
    def Workflow(x: int):
        test_func = workflow.add(TestFunc(a=x))
        return test_func.out

    wf = Workflow(x=2)

    wf(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.PROV,
        messengers=FileMessenger(),
    )
    # default path
    message_path = tmpdir / wf._checksum / "messages"
    assert message_path.exists()

    collect_messages(tmpdir / wf._checksum, message_path, ld_op="compact")
    assert (tmpdir / wf._checksum / "messages.jsonld").exists()


def test_audit_all(
    tmpdir,
):
    @python.define(outputs={"out": float})
    def TestFunc(a: int, b: float = 0.1):
        return a + b

    funky = TestFunc(a=2)
    message_path = tmpdir / funky._checksum / "messages"

    funky(
        cache_dir=tmpdir,
        audit_flags=AuditFlag.ALL,
        messengers=FileMessenger(),
        messenger_args=dict(message_dir=message_path),
    )
    from glob import glob

    assert len(glob(str(tmpdir / funky._checksum / "proc*.log"))) == 1
    assert len(glob(str(message_path / "*.jsonld"))) == 7

    # commented out to speed up testing
    collect_messages(tmpdir / funky._checksum, message_path, ld_op="compact")
    assert (tmpdir / funky._checksum / "messages.jsonld").exists()


@no_win
def test_shell_cmd(tmpdir):
    cmd = ["echo", "hail", "pydra"]

    # all args given as executable
    Shelly = shell.define(" ".join(cmd))
    shelly = Shelly()
    assert shelly.cmdline == " ".join(cmd)
    outputs = shelly()
    assert outputs.stdout == " ".join(cmd[1:]) + "\n"

    # separate command into exec + args
    Shelly = shell.define(
        cmd[0], inputs=[shell.arg(name=a, default=a) for a in cmd[1:]]
    )
    shelly = Shelly()
    assert shelly.executable == "echo"
    assert shelly.cmdline == " ".join(cmd)
    outputs = shelly()
    assert outputs.return_code == 0
    assert outputs.stdout == " ".join(cmd[1:]) + "\n"


def test_functask_callable(tmpdir):
    # no submitter or worker
    foo = FunAddTwo(a=1)
    outputs = foo()
    assert outputs.out == 3

    # worker
    bar = FunAddTwo(a=2)
    outputs = bar(worker="cf", cache_dir=tmpdir)
    assert outputs.out == 4


def test_taskhooks_1(tmpdir: Path, capsys):
    cache_dir = tmpdir / "cache"
    cache_dir.mkdir()

    foo = Job(task=FunAddTwo(a=1), submitter=Submitter(cache_dir=tmpdir), name="foo")
    assert foo.hooks
    # ensure all hooks are defined
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    def myhook(task, *args):
        print("I was called")

    FunAddTwo(a=1)(cache_dir=cache_dir, hooks=TaskHooks(pre_run=myhook))
    captured = capsys.readouterr()
    assert "I was called\n" in captured.out
    del captured

    # setting unknown hook should not be allowed
    with pytest.raises(AttributeError):
        foo.hooks.mid_run = myhook

    # reset all hooks
    foo.hooks.reset()
    for attr in ("pre_run", "post_run", "pre_run_task", "post_run_task"):
        hook = getattr(foo.hooks, attr)
        assert hook() is None

    # clear cache
    shutil.rmtree(cache_dir)
    cache_dir.mkdir()

    # set all hooks
    FunAddTwo(a=1)(
        cache_dir=cache_dir,
        hooks=TaskHooks(
            pre_run=myhook,
            post_run=myhook,
            pre_run_task=myhook,
            post_run_task=myhook,
        ),
    )
    captured = capsys.readouterr()
    assert captured.out.count("I was called\n") == 4
    del captured


def test_taskhooks_2(tmpdir, capsys):
    """checking order of the hooks; using task's attributes"""

    def myhook_prerun(task, *args):
        print(f"i. prerun hook was called from {task.name}")

    def myhook_prerun_task(task, *args):
        print(f"ii. prerun task hook was called {task.name}")

    def myhook_postrun_task(task, *args):
        print(f"iii. postrun task hook was called {task.name}")

    def myhook_postrun(task, *args):
        print(f"iv. postrun hook was called {task.name}")

    FunAddTwo(a=1)(
        cache_dir=tmpdir,
        hooks=TaskHooks(
            pre_run=myhook_prerun,
            post_run=myhook_postrun,
            pre_run_task=myhook_prerun_task,
            post_run_task=myhook_postrun_task,
        ),
    )

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking the order of the hooks
    assert "i. prerun hook" in hook_messages[0]
    assert "ii. prerun task hook" in hook_messages[1]
    assert "iii. postrun task hook" in hook_messages[2]
    assert "iv. postrun hook" in hook_messages[3]


def test_taskhooks_3(tmpdir, capsys):
    """checking results in the post run hooks"""
    foo = Job(task=FunAddTwo(a=1), name="foo", submitter=Submitter(cache_dir=tmpdir))

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook, the result is {result.outputs.out}")

    def myhook_postrun(task, result, *args):
        print(f"postrun hook, the result is {result.outputs.out}")

    foo.hooks.post_run = myhook_postrun
    foo.hooks.post_run_task = myhook_postrun_task
    foo.run()

    captured = capsys.readouterr()
    hook_messages = captured.out.strip().split("\n")
    # checking that the postrun hooks have access to results
    assert "postrun task hook, the result is 3" in hook_messages[0]
    assert "postrun hook, the result is 3" in hook_messages[1]


def test_taskhooks_4(tmpdir, capsys):
    """task raises an error: postrun task should be called, postrun shouldn't be called"""

    def myhook_postrun_task(task, result, *args):
        print(f"postrun task hook was called, result object is {result}")

    def myhook_postrun(task, result, *args):
        print("postrun hook should not be called")

    with pytest.raises(Exception):
        FunAddTwo(a="one")(
            cache_dir=tmpdir,
            hooks=TaskHooks(post_run=myhook_postrun, post_run_task=myhook_postrun_task),
        )

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
    def FunError(x):
        raise Exception("Error from the function")

    with pytest.raises(Exception, match="Error from the function") as exinfo:
        with Submitter(worker="cf", cache_dir=tmpdir) as sub:
            sub(FunError(x=3), raise_errors=True)

    # getting error file from the error message
    error_file_match = (
        str(exinfo.value.__notes__[0]).split("here: ")[-1].split("_error.pklz")[0]
    )
    error_file = Path(error_file_match) / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    # reading error message from the pickle file
    error_tb = cp.loads(error_file.read_bytes())["error message"]
    # the error traceback should be a list and should point to a specific line in the function
    assert isinstance(error_tb, list)
    assert "in FunError" in error_tb[-2]


def test_traceback_wf(tmp_path: Path):
    """checking if the error raised in a function is properly returned by a workflow;
    checking if there is an error filename in the error message that contains
    full traceback including the line in the python function
    """

    @python.define
    def FunError(x):
        raise Exception("Error from the function")

    @workflow.define
    def Workflow(x_list):
        fun_error = workflow.add(FunError().split(x=x_list), name="fun_error")
        return fun_error.out

    wf = Workflow(x_list=[3, 4])
    with pytest.raises(RuntimeError, match="Job 'fun_error.*, errored") as exinfo:
        with Submitter(worker="cf", cache_dir=tmp_path) as sub:
            sub(wf, raise_errors=True)

    # getting error file from the error message
    output_dir_match = Path(
        str(exinfo.value).split("See output directory for details: ")[-1].strip()
    )
    assert output_dir_match.exists()
    error_file = output_dir_match / "_error.pklz"
    # checking if the file exists
    assert error_file.exists()
    assert "in FunError" in str(exinfo.value)


@pytest.mark.flaky(reruns=3)
def test_rerun_errored(tmp_path, capfd):
    """Test rerunning a task containing errors.
    Only the errored tasks should be rerun"""

    @python.define
    def PassOdds(x):
        if x % 2 == 0:
            print(f"x={x} -> x%2 = {bool(x % 2)} (even error)\n")
            raise Exception("even error")
        else:
            print(f"x={x} -> x%2 = {bool(x % 2)}\n")
            return x

    pass_odds = PassOdds().split("x", x=[1, 2, 3, 4, 5])

    with pytest.raises(Exception):
        pass_odds(cache_dir=tmp_path, worker="cf", n_procs=3)
    with pytest.raises(Exception):
        pass_odds(cache_dir=tmp_path, worker="cf", n_procs=3)

    out, err = capfd.readouterr()
    stdout_lines = out.splitlines()

    tasks_run = 0
    errors_found = 0

    for line in stdout_lines:
        if "-> x%2" in line:
            tasks_run += 1
        if "(even error)" in line:
            errors_found += 1

    # There should have been 5 messages of the form "x%2 = XXX" after calling task() the first time
    # and another 2 messagers after calling the second time
    assert tasks_run == 7
    assert errors_found == 4


@attrs.define(auto_attribs=True)
class A:
    x: int


def test_object_input():
    """Test function tasks with object inputs"""

    @python.define
    def TestFunc(a: A):
        return a.x

    outputs = TestFunc(a=A(x=7))()
    assert outputs.out == 7


def test_argstr_formatting():
    @shell.define
    class Defn(shell.Task["Defn.Outputs"]):
        a1_field: str
        b2_field: float
        c3_field: ty.Dict[str, str]
        d4_field: ty.List[str] = shell.arg(sep=" ")
        executable = "dummy"

        class Outputs(shell.Outputs):
            pass

    values = dict(a1_field="1", b2_field=2.0, c3_field={"c": "3"}, d4_field=["4"])
    assert (
        argstr_formatting(
            "{a1_field} {b2_field:02f} -test {c3_field[c]} -me {d4_field[0]}",
            values,
        )
        == "1 2.000000 -test 3 -me 4"
    )
