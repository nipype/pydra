from pydra.design import python, workflow
from pydra.engine.submitter import Submitter


@python.define
def Add(x: float, y: float) -> float:
    return x + y


@python.define
def Subtract(x: float, y: float) -> float:
    return x - y


@python.define
def Divide(x: float, y: float) -> float:
    return x / y


@workflow.define
def UnsafeWorkflow(a: float, b: float, c: float) -> float:
    add = workflow.add(Add(x=a, y=b))
    divide = workflow.add(Divide(x=add.out, y=c))
    subtract = workflow.add(Subtract(x=divide.out, y=b))
    return subtract.out


# This workflow will fail because we are trying to divide by 0
failing_workflow = UnsafeWorkflow(a=10, b=5).split(c=[3, 2, 0])

with Submitter() as sub:
    result = sub(failing_workflow)
