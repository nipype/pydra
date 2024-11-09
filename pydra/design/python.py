import typing as ty
import inspect
import attrs
from pydra.engine.task import FunctionTask
from .base import (
    Arg,
    Out,
    collate_fields,
    make_interface,
    Interface,
    parse_doc_string,
    extract_inputs_and_outputs_from_function,
    check_explicit_fields_are_none,
    get_fields_from_class,
)


__all__ = ["arg", "out", "interface"]


@attrs.define
class arg(Arg):
    pass


@attrs.define
class out(Out):
    pass


def interface(
    wrapped: ty.Callable | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    auto_attribs: bool = True,
) -> Interface:

    def make(
        wrapped: ty.Callable | type | None = None,
    ) -> Interface:

        if inspect.isclass(wrapped):
            klass = wrapped
            function = klass.function
            name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = get_fields_from_class(
                klass, arg, out, auto_attribs
            )
        else:
            klass = None
            function = wrapped
            input_helps, output_helps = parse_doc_string(function.__doc__)
            inferred_inputs, inferred_outputs = (
                extract_inputs_and_outputs_from_function(function, inputs, outputs)
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
        interface = make_interface(
            FunctionTask, parsed_inputs, parsed_outputs, name=name, klass=klass
        )
        # Set the function in the created class
        interface.function = function
        return interface

    if wrapped is not None:
        if not isinstance(wrapped, ty.Callable):
            raise ValueError(f"wrapped must be a callable, not {wrapped!r}")
        return make(wrapped)
    return make
