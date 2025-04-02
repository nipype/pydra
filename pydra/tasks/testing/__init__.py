from pydra.compose import python, workflow


@python.define
def Add(x: float, y: float) -> float:
    return x + y


@python.define
def Divide(x: float, y: float) -> float:
    return x / y


@python.define
def SafeDivide(x: float, y: float) -> float:
    if y == 0:
        return float("nan")
    return x / y


@python.define
def Subtract(x: float, y: float) -> float:
    return x - y


@workflow.define
def UnsafeDivisionWorkflow(a: float, b: float, denominator: float) -> float:
    """Adds 'a' and 'b' together, divides by 'denominator', and then subtracts 'b' from
    the output. Division by 0 is not guarded against so the workflow will fail if
    the value passed to the 'denominator' parameter is 0.

    Parameters
    ----------
    a : float
        The first number to add.
    b : float
        The second number to add.
    denominator : float
        The number to divide the sum of 'a' and 'b' by.

    Returns
    -------
    out : float
        The result of subtracting 'b' from the result of dividing the sum of 'a' and
        'b' by 'denominator'.
    """
    add = workflow.add(Add(x=a, y=b))
    divide = workflow.add(Divide(x=add.out, y=denominator))
    subtract = workflow.add(Subtract(x=divide.out, y=b))
    return subtract.out


@workflow.define
def SafeDivisionWorkflow(a: float, b: float, denominator: float) -> float:
    """Adds 'a' and 'b' together, divides by 'denominator', and then subtracts 'b' from
    the output. Division by 0 is not guarded against so the workflow will fail if
    the value passed to the 'denominator' parameter is 0.

    Parameters
    ----------
    a : float
        The first number to add.
    b : float
        The second number to add.
    denominator : float
        The number to divide the sum of 'a' and 'b' by.

    Returns
    -------
    out : float
        The result of subtracting 'b' from the result of dividing the sum of 'a' and
        'b' by 'denominator'.
    """
    add = workflow.add(Add(x=a, y=b))
    divide = workflow.add(SafeDivide(x=add.out, y=denominator))
    subtract = workflow.add(Subtract(x=divide.out, y=b))
    return subtract.out


@python.define
def TenToThePower(p: int) -> int:
    return 10**p
