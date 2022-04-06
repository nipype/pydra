""" Decorators to apply to functions used in Pydra workflows """
from functools import wraps


def annotate(annotation):
    """
    Update the annotation of a function.

    Example
    -------
    >>> import pydra
    >>> @pydra.mark.annotate({'a': int, 'return': float})
    ... def square(a):
    ...     return a ** 2.0

    """
    import inspect

    def decorate(func):
        sig = inspect.signature(func)
        unknown = set(annotation) - set(sig.parameters) - {"return"}
        if unknown:
            raise TypeError(f"Cannot annotate unknown parameters: {tuple(unknown)}")
        func.__annotations__.update(annotation)
        return func

    return decorate


def task(func):
    """
    Promote a function to a :class:`~pydra.engine.task.FunctionTask`.

    Example
    -------
    >>> import pydra
    >>> @pydra.mark.task
    ... def square(a: int) -> float:
    ...     return a ** 2.0

    """
    from ..engine.task import FunctionTask

    @wraps(func)
    def decorate(**kwargs):
        return FunctionTask(func=func, **kwargs)

    return decorate
