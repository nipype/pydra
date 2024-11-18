from pydra.engine.workflow import Workflow
from pydra.engine.specs import LazyField
import typing as ty
from pydra.design import shell, python, workflow, list_fields
from fileformats import video, image


def test_workflow():

    @workflow.define
    def MyTestWorkflow(a, b):

        @python.define
        def Add(a, b):
            return a + b

        @python.define
        def Mul(a, b):
            return a * b

        add = workflow.add(Add(a=a, b=b))
        mul = workflow.add(Mul(a=add.out, b=b))
        return mul.out

    constructor = MyTestWorkflow().constructor
    assert constructor.__name__ == "MyTestWorkflow"
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
    assert wf.outputs.out == LazyField(name="Mul", field="out", type=ty.Any)
    assert list(wf.node_names) == ["Add", "Mul"]


def test_shell_workflow():

    @workflow.define
    def MyTestShellWorkflow(input_video: video.Mp4, watermark: image.Png) -> video.Mp4:

        add_watermark = workflow.add(
            shell.define(
                "ffmpeg -i <in_video> -i <watermark:image/png> -filter_complex <filter> <out|out_video>"
            )(in_video=input_video, watermark=watermark, filter="overlay=10:10"),
            name="add_watermark",
        )
        output_video = workflow.add(
            shell.define(
                (
                    "HandBrakeCLI -i <in_video> -o <out|out_video> "
                    "--width <width:int> --height <height:int>"
                ),
                # this specifies that this output is required even though it has a flag,
                # optional inputs and outputs are of type * | None
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
    assert wf.outputs.output_video == LazyField(
        name="resize", field="out_video", type=video.Mp4
    )
    assert list(wf.node_names) == ["add_watermark", "resize"]


def test_workflow_alt_syntax():

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

        @python.define(inputs={"x": float}, outputs={"out": float})
        def Add(x, y):
            return x + y

        def Mul(x, y):
            return x * y

        @python.define(outputs=["divided"])
        def Divide(x, y):
            return x / y

        wf = workflow.this()

        add = wf.add(Add(x=a, y=b), name="addition")
        mul = wf.add(python.define(Mul, outputs={"out": float})(x=add.out, y=b))
        divide = wf.add(Divide(x=wf["addition"].lzout.out, y=mul.out), name="division")

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
    assert wf.outputs.out1 == LazyField(name="Mul", field="out", type=float)
    assert wf.outputs.out2 == LazyField(name="division", field="divided", type=ty.Any)
    assert list(wf.node_names) == ["addition", "Mul", "division"]


def test_workflow_set_outputs_directly():

    @workflow.define(outputs={"out1": float, "out2": float})
    def MyTestWorkflow(a: int, b: float):

        @python.define
        def Add(a, b):
            return a + b

        @python.define
        def Mul(a, b):
            return a * b

        wf = workflow.this()

        add = wf.add(Add(a=a, b=b))
        wf.add(Mul(a=add.out, b=b))

        wf.outputs.out2 = add.out  # Using the returned lzout outputs
        wf.outputs.out1 = wf["Mul"].lzout.out  # accessing the lzout outputs via getitem

        # no return required when the outputs are set directly

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
    assert wf.outputs.out1 == LazyField(name="Mul", field="out", type=ty.Any)
    assert wf.outputs.out2 == LazyField(name="Add", field="out", type=ty.Any)
    assert list(wf.node_names) == ["Add", "Mul"]
