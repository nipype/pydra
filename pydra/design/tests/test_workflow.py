from pydra.engine.workflow import Workflow
from pydra.engine.specs import LazyField
import typing as ty
from pydra.design import shell, python, workflow, list_fields


def test_workflow():

    @workflow.define
    def MyTestWorkflow(a: int, b: float) -> float:

        @python.define
        def Add(a, b):
            return a + b

        @python.define
        def Mul(a, b):
            return a * b

        add = workflow.add(Add(a=a, b=b))
        mul = workflow.add(Mul(a=add.out, b=b))
        return mul.out

    assert list_fields(MyTestWorkflow) == [
        workflow.arg(name="a", type=int),
        workflow.arg(name="b", type=float),
        workflow.arg(
            name="constructor", type=ty.Callable, default=MyTestWorkflow().constructor
        ),
    ]
    assert list_fields(MyTestWorkflow.Outputs) == [
        workflow.out(name="out", type=float),
    ]
    workflow_spec = MyTestWorkflow(a=1, b=2.0)
    wf = Workflow.construct(workflow_spec)
    assert wf.inputs.a == 1
    assert wf.inputs.b == 2.0
    assert wf.outputs.out == LazyField(name="Mul", field="out", type=ty.Any)
    assert list(wf.node_names) == ["Add", "Mul"]
