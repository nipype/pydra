from operator import attrgetter
from copy import copy
import pytest
import attrs
from pydra.engine.workflow.base import Workflow
from pydra.engine.workflow.lazy import LazyInField, LazyOutField
import typing as ty
from pydra.design import shell, python, workflow
from pydra.engine.helpers import list_fields
from pydra.engine.specs import WorkflowSpec, WorkflowOutputs
from fileformats import video, image

# NB: We use PascalCase for interfaces and workflow functions as it is translated into a class


@python.define
def Add(a, b):
    return a + b


@python.define
def Mul(a, b):
    return a * b


@python.define(outputs=["divided"])
def Divide(x, y):
    return x / y


@python.define
def Sum(x: list[float]) -> float:
    return sum(x)


def a_converter(value):
    if value is attrs.NOTHING:
        return value
    return float(value)


def test_workflow():

    @workflow.define
    def MyTestWorkflow(a, b):
        add = workflow.add(Add(a=a, b=b))
        mul = workflow.add(Mul(a=add.out, b=b))
        return mul.out

    constructor = MyTestWorkflow().constructor
    assert constructor.__name__ == "MyTestWorkflow"

    # The constructor function is included as a part of the specification so it is
    # included in the hash by default and can be overridden if needed. Not 100% sure
    # if this is a good idea or not
    assert list_fields(MyTestWorkflow) == [
        workflow.arg(name="a"),
        workflow.arg(name="b"),
        workflow.arg(name="constructor", type=ty.Callable, default=constructor),
    ]
    assert list_fields(MyTestWorkflow.Outputs) == [
        workflow.out(name="out"),
    ]
    workflow_spec = MyTestWorkflow(a=1, b=2.0)
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.a == 1
    assert wf.inputs.b == 2.0
    assert wf.outputs.out == LazyOutField(node=wf["Mul"], field="out", type=ty.Any)

    # Nodes are named after the specs by default
    assert list(wf.node_names) == ["Add", "Mul"]


def test_shell_workflow():

    @workflow.define
    def MyTestShellWorkflow(
        input_video: video.Mp4,
        watermark: image.Png,
        watermark_dims: tuple[int, int] = (10, 10),
    ) -> video.Mp4:

        add_watermark = workflow.add(
            shell.define(
                "ffmpeg -i <in_video> -i <watermark:image/png> "
                "-filter_complex <filter> <out|out_video>"
            )(
                in_video=input_video,
                watermark=watermark,
                filter="overlay={}:{}".format(*watermark_dims),
            ),
            name="add_watermark",
        )
        output_video = workflow.add(
            shell.define(
                "HandBrakeCLI -i <in_video> -o <out|out_video> "
                "--width <width:int> --height <height:int>",
                # By default any input/output specified with a flag (e.g. -i <in_video>)
                # is considered optional, i.e. of type `FsObject | None`, and therefore
                # won't be used by default. By overriding this with non-optional types,
                # the fields are specified as being required.
                inputs={"in_video": video.Mp4},
                outputs={"out_video": video.Mp4},
            )(in_video=add_watermark.out_video, width=1280, height=720),
            name="resize",
        ).out_video

        return output_video  # test implicit detection of output name

    constructor = MyTestShellWorkflow().constructor
    assert constructor.__name__ == "MyTestShellWorkflow"
    assert list_fields(MyTestShellWorkflow) == [
        workflow.arg(name="input_video", type=video.Mp4),
        workflow.arg(name="watermark", type=image.Png),
        workflow.arg(name="watermark_dims", type=tuple[int, int], default=(10, 10)),
        workflow.arg(name="constructor", type=ty.Callable, default=constructor),
    ]
    assert list_fields(MyTestShellWorkflow.Outputs) == [
        workflow.out(name="output_video", type=video.Mp4),
    ]
    input_video = video.Mp4.mock("input.mp4")
    watermark = image.Png.mock("watermark.png")
    workflow_spec = MyTestShellWorkflow(
        input_video=input_video,
        watermark=watermark,
    )
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.input_video == input_video
    assert wf.inputs.watermark == watermark
    assert wf.outputs.output_video == LazyOutField(
        node=wf["resize"], field="out_video", type=video.Mp4, type_checked=True
    )
    assert list(wf.node_names) == ["add_watermark", "resize"]


def test_workflow_canonical():
    """Test class-based workflow definition"""

    # NB: We use PascalCase (i.e. class names) as it is translated into a class

    @workflow.define
    class MyTestWorkflow(WorkflowSpec["MyTestWorkflow.Outputs"]):

        a: int
        b: float = workflow.arg(
            help_string="A float input",
            converter=a_converter,
        )

        @staticmethod
        def constructor(a, b):
            add = workflow.add(Add(a=a, b=b))
            mul = workflow.add(Mul(a=add.out, b=b))
            return mul.out

        @workflow.outputs
        class Outputs(WorkflowOutputs):
            out: float

    constructor = MyTestWorkflow().constructor
    assert constructor.__name__ == "constructor"

    # The constructor function is included as a part of the specification so it is
    # included in the hash by default and can be overridden if needed. Not 100% sure
    # if this is a good idea or not
    assert sorted(list_fields(MyTestWorkflow), key=attrgetter("name")) == [
        workflow.arg(name="a", type=int),
        workflow.arg(
            name="b", type=float, help_string="A float input", converter=a_converter
        ),
        workflow.arg(name="constructor", type=ty.Callable, default=constructor),
    ]
    assert list_fields(MyTestWorkflow.Outputs) == [
        workflow.out(name="out", type=float),
    ]
    workflow_spec = MyTestWorkflow(a=1, b=2.0)
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.a == 1
    assert wf.inputs.b == 2.0
    assert wf.outputs.out == LazyOutField(node=wf["Mul"], field="out", type=ty.Any)

    # Nodes are named after the specs by default
    assert list(wf.node_names) == ["Add", "Mul"]


def test_workflow_lazy():

    @workflow.define(lazy=["input_video", "watermark"])
    def MyTestShellWorkflow(
        input_video: video.Mp4,
        watermark: image.Png,
        watermark_dims: tuple[int, int] = (10, 10),
    ) -> video.Mp4:

        add_watermark = workflow.add(
            shell.define(
                "ffmpeg -i <in_video> -i <watermark:image/png> "
                "-filter_complex <filter> <out|out_video>"
            )(
                in_video=input_video,
                watermark=watermark,
                filter="overlay={}:{}".format(*watermark_dims),
            ),
            name="add_watermark",
        )
        output_video = workflow.add(
            shell.define(
                "HandBrakeCLI -i <in_video> -o <out|out_video> "
                "--width <width:int> --height <height:int>",
                # By default any input/output specified with a flag (e.g. -i <in_video>)
                # is considered optional, i.e. of type `FsObject | None`, and therefore
                # won't be used by default. By overriding this with non-optional types,
                # the fields are specified as being required.
                inputs={"in_video": video.Mp4},
                outputs={"out_video": video.Mp4},
            )(in_video=add_watermark.out_video, width=1280, height=720),
            name="resize",
        ).out_video

        return output_video  # test implicit detection of output name

    input_video = video.Mp4.mock("input.mp4")
    watermark = image.Png.mock("watermark.png")
    workflow_spec = MyTestShellWorkflow(
        input_video=input_video,
        watermark=watermark,
    )
    wf = Workflow.construct(workflow_spec)
    assert wf["add_watermark"].inputs.in_video == LazyInField(
        workflow=wf, field="input_video", type=video.Mp4, type_checked=True
    )
    assert wf["add_watermark"].inputs.watermark == LazyInField(
        workflow=wf, field="watermark", type=image.Png, type_checked=True
    )


def test_direct_access_of_workflow_object():

    @python.define(inputs={"x": float}, outputs={"z": float})
    def Add(x, y):
        return x + y

    def Mul(x, y):
        return x * y

    @workflow.define(outputs=["out1", "out2"])
    def MyTestWorkflow(a: int, b: float) -> tuple[float, float]:
        """A test workflow demonstration a few alternative ways to set and connect nodes

        Args:
            a: An integer input
            b: A float input

        Returns:
            out1: The first output
            out2: The second output
        """

        wf = workflow.this()

        add = wf.add(Add(x=a, y=b), name="addition")
        mul = wf.add(python.define(Mul, outputs={"out": float})(x=add.z, y=b))
        divide = wf.add(Divide(x=wf["addition"].lzout.z, y=mul.out), name="division")

        # Alter one of the inputs to a node after it has been initialised
        wf["Mul"].inputs.y *= 2

        return mul.out, divide.divided

    assert list_fields(MyTestWorkflow) == [
        workflow.arg(name="a", type=int, help_string="An integer input"),
        workflow.arg(name="b", type=float, help_string="A float input"),
        workflow.arg(
            name="constructor", type=ty.Callable, default=MyTestWorkflow().constructor
        ),
    ]
    assert list_fields(MyTestWorkflow.Outputs) == [
        workflow.out(name="out1", type=float, help_string="The first output"),
        workflow.out(name="out2", type=float, help_string="The second output"),
    ]
    workflow_spec = MyTestWorkflow(a=1, b=2.0)
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.a == 1
    assert wf.inputs.b == 2.0
    assert wf.outputs.out1 == LazyOutField(
        node=wf["Mul"], field="out", type=float, type_checked=True
    )
    assert wf.outputs.out2 == LazyOutField(
        node=wf["division"], field="divided", type=ty.Any
    )
    assert list(wf.node_names) == ["addition", "Mul", "division"]


def test_workflow_set_outputs_directly():

    @workflow.define(outputs={"out1": float, "out2": float})
    def MyTestWorkflow(a: int, b: float):

        wf = workflow.this()
        add = wf.add(Add(a=a, b=b))
        wf.add(Mul(a=add.out, b=b))

        # Set the outputs of the workflow directly instead of returning them them in
        # a tuple
        wf.outputs.out2 = add.out  # Using the returned lzout outputs
        wf.outputs.out1 = wf["Mul"].lzout.out  # accessing the lzout outputs via getitem

        # no return is used when the outputs are set directly

    assert list_fields(MyTestWorkflow) == [
        workflow.arg(name="a", type=int),
        workflow.arg(name="b", type=float),
        workflow.arg(
            name="constructor", type=ty.Callable, default=MyTestWorkflow().constructor
        ),
    ]
    assert list_fields(MyTestWorkflow.Outputs) == [
        workflow.out(name="out1", type=float),
        workflow.out(name="out2", type=float),
    ]
    workflow_spec = MyTestWorkflow(a=1, b=2.0)
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.a == 1
    assert wf.inputs.b == 2.0
    assert wf.outputs.out1 == LazyOutField(node=wf["Mul"], field="out", type=ty.Any)
    assert wf.outputs.out2 == LazyOutField(node=wf["Add"], field="out", type=ty.Any)
    assert list(wf.node_names) == ["Add", "Mul"]


def test_workflow_split_combine1():

    @python.define
    def Mul(x: float, y: float) -> float:
        return x * y

    @python.define
    def Sum(x: list[float]) -> float:
        return sum(x)

    @workflow.define
    def MyTestWorkflow(a: list[int], b: list[float]) -> list[float]:
        mul = workflow.add(Mul()).split(x=a, y=b).combine("x")
        sum = workflow.add(Sum(x=mul.out))
        return sum.out

    wf = Workflow.construct(MyTestWorkflow(a=[1, 2, 3], b=[1.0, 10.0, 100.0]))
    assert wf["Mul"].splitter == ["Mul.x", "Mul.y"]
    assert wf["Mul"].combiner == ["Mul.x"]
    assert wf.outputs.out == LazyOutField(
        node=wf["Sum"], field="out", type=list[float], type_checked=True
    )


def test_workflow_split_combine2():

    @python.define
    def Mul(x: float, y: float) -> float:
        return x * y

    @python.define
    def Add(x: float, y: float) -> float:
        return x + y

    @workflow.define
    def MyTestWorkflow(a: list[int], b: list[float], c: float) -> list[float]:
        mul = workflow.add(Mul()).split(x=a, y=b)
        add = workflow.add(Add(x=mul.out, y=c)).combine("Mul.x")
        sum = workflow.add(Sum(x=add.out))
        return sum.out

    wf = Workflow.construct(MyTestWorkflow(a=[1, 2, 3], b=[1.0, 10.0, 100.0], c=2.0))
    assert wf["Mul"].splitter == ["Mul.x", "Mul.y"]
    assert wf["Mul"].combiner == []
    assert wf["Add"].splitter == "_Mul"
    assert wf["Add"].combiner == ["Mul.x"]
    assert wf.outputs.out == LazyOutField(
        node=wf["Sum"], field="out", type=list[float], type_checked=True
    )


def test_workflow_split_after_access_fail():
    """It isn't possible to split/combine a node after one of its outputs has been type
    checked as this changes the type of the outputs and renders the type checking
    invalid
    """

    @python.define
    def Add(x: float, y: float) -> float:
        return x + y

    @python.define
    def Mul(x: float, y: float) -> float:
        return x * y

    @workflow.define
    def MyTestWorkflow(a: list[int], b: list[float]) -> list[float]:

        add = workflow.add(Add())
        mul = workflow.add(Mul(x=add.out, y=2.0))  # << Add.out is accessed here
        add.split(x=a, y=b).combine("x")
        return mul.out

    with pytest.raises(RuntimeError, match="Outputs .* have already been accessed"):
        Workflow.construct(MyTestWorkflow(a=[1, 2, 3], b=[1.0, 10.0, 100.0]))


def test_nested_workflow():
    """Simple test of a nested workflow"""

    @python.define
    def Add(x: float, y: float) -> float:
        return x + y

    @python.define
    def Mul(x: float, y: float) -> float:
        return x * y

    @python.define
    def Divide(x: float, y: float) -> float:
        return x / y

    @python.define
    def Power(x: float, y: float) -> float:
        return x**y

    @workflow.define
    def NestedWorkflow(a: float, b: float, c: float) -> float:
        pow = workflow.add(Power(x=a, y=c))
        add = workflow.add(Add(x=pow.out, y=b))
        return add.out

    @workflow.define
    def MyTestWorkflow(a: int, b: float, c: float) -> float:
        div = workflow.add(Divide(x=a, y=b))
        nested = workflow.add(NestedWorkflow(a=div.out, b=b, c=c))
        return nested.out

    wf = Workflow.construct(MyTestWorkflow(a=1, b=10.0, c=2.0))
    assert wf.inputs.a == 1
    assert wf.inputs.b == 10.0
    assert wf.inputs.c == 2.0
    assert wf.outputs.out == LazyOutField(
        node=wf["NestedWorkflow"], field="out", type=float, type_checked=True
    )
    assert list(wf.node_names) == ["Divide", "NestedWorkflow"]
    nwf_spec = copy(wf["NestedWorkflow"]._spec)
    nwf_spec.a = 100.0
    nwf = Workflow.construct(nwf_spec)
    nwf.inputs.a == 100.0
    nwf.inputs.b == 10.0
    nwf.inputs.c == 2.0
    nwf.outputs.out == LazyOutField(node=nwf["Add"], field="out", type=float)
    assert list(nwf.node_names) == ["Power", "Add"]


def test_recursively_nested_conditional_workflow():
    """More complex nested workflow example demonstrating conditional branching at run
    time"""

    @python.define
    def Add(x: float, y: float) -> float:
        return x + y

    @python.define
    def Subtract(x: float, y: float) -> float:
        return x - y

    @workflow.define
    def RecursiveNestedWorkflow(a: float, depth: int) -> float:
        add = workflow.add(Add(x=a, y=1))
        decrement_depth = workflow.add(Subtract(x=depth, y=1))
        if depth > 0:
            out_node = workflow.add(
                RecursiveNestedWorkflow(a=add.out, depth=decrement_depth.out)
            )
        else:
            out_node = add
        return out_node.out

    wf = Workflow.construct(RecursiveNestedWorkflow(a=1, depth=3))
    assert wf.inputs.a == 1
    assert wf.inputs.depth == 3
    assert wf.outputs.out == LazyOutField(
        node=wf["RecursiveNestedWorkflow"], field="out", type=float, type_checked=True
    )
