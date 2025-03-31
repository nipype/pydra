from pydra.compose import python, workflow
from pydra.utils import print_help, show_workflow


# Example python tasks
@python.define
def Add(a, b):
    return a + b


@python.define
def Mul(a, b):
    return a * b


@workflow.define
class CanonicalWorkflowTask(workflow.Task["CanonicalWorkflowTask.Outputs"]):

    @staticmethod
    def a_converter(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    a: int
    b: float = workflow.arg(
        help="A float input",
        converter=a_converter,
    )

    @staticmethod
    def constructor(a, b):
        add = workflow.add(Add(a=a, b=b))
        mul = workflow.add(Mul(a=add.out, b=b))
        return mul.out

    class Outputs(workflow.Outputs):
        out: float


print_help(CanonicalWorkflowTask)
show_workflow(CanonicalWorkflowTask)
