from pydra.engine.workflow import Workflow
from pydra.engine.specs import LazyField
import typing as ty
from pydra.design import shell, python, workflow, list_fields


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
