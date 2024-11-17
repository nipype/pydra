import typing as ty
import inspect
import attrs
from pydra.engine.task import FunctionTask
from .base import (
    Arg,
    Out,
    collate_fields,
    make_task_spec,
    TaskSpec,
    parse_doc_string,
    extract_inputs_and_outputs_from_function,
    check_explicit_fields_are_none,
    get_fields_from_class,
)


__all__ = ["arg", "out", "define"]


@attrs.define
class arg(Arg):
    pass


@attrs.define
class out(Out):
    pass


def define(
    wrapped: type | ty.Callable | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
) -> TaskSpec:
    """
    Create an interface for a function or a class.

    Parameters
    ----------
    wrapped : type | callable | None
        The function or class to create an interface for.
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The inputs to the function or class.
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The outputs of the function or class.
    auto_attribs : bool
        Whether to use auto_attribs mode when creating the class.
    """

    def make(wrapped: ty.Callable | type) -> TaskSpec:
        if inspect.isclass(wrapped):
            klass = wrapped
            function = klass.function
            name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = get_fields_from_class(
                klass, arg, out, auto_attribs
            )
        else:
            if not inspect.isfunction(wrapped):
                raise ValueError(
                    f"wrapped must be a class or a function, not {wrapped!r}"
                )
            klass = None
            function = wrapped
            input_helps, output_helps = parse_doc_string(function.__doc__)
            inferred_inputs, inferred_outputs = (
                extract_inputs_and_outputs_from_function(function, arg, inputs, outputs)
            )
            name = function.__name__

            parsed_inputs, parsed_outputs = collate_fields(
                arg_type=arg,
                out_type=out,
                inputs=inferred_inputs,
                outputs=inferred_outputs,
                input_helps=input_helps,
                output_helps=output_helps,
            )

        parsed_inputs["function"] = arg(
            name="function", type=ty.Callable, default=function
        )

        interface = make_task_spec(
            FunctionTask,
            parsed_inputs,
            parsed_outputs,
            name=name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
        )

        return interface

    if wrapped is not None:
        if not isinstance(wrapped, (ty.Callable, type)):
            raise ValueError(f"wrapped must be a class or a callable, not {wrapped!r}")
        return make(wrapped)
    return make
