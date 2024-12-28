from pydra.design import workflow, python


# Example python task specifications
@python.define
def Add(a, b):
    return a + b


@python.define
def Mul(a, b):
    return a * b


@python.define
def Sum(x: list[float]) -> float:
    return sum(x)


@workflow.define
def MySplitWorkflow(a: list[int], b: list[float]) -> list[float]:
    # Multiply over all combinations of the elements of a and b, then combine the results
    # for each a element into a list over each b element
    mul = workflow.add(Mul()).split(x=a, y=b).combine("x")
    # Sume the multiplications across all all b elements for each a element
    sum = workflow.add(Sum(x=mul.out))
    return sum.out
