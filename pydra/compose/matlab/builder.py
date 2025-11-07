import typing as ty
import inspect
import re
from typing import dataclass_transform
from . import field
from .task import Task, Outputs
from pydra.compose.base import (
    ensure_field_objects,
    build_task_class,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(field.arg,),
)
def define(
    wrapped: type | ty.Callable | None = None,
    /,
    inputs: list[str | field.arg] | dict[str, field.arg | type] | None = None,
    outputs: list[str | field.out] | dict[str, field.out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    name: str | None = None,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> "Task":
    """
    Create an interface for a function or a class.

    Parameters
    ----------
    wrapped : type | callable | None
        The function or class to create an interface for.
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The inputs to the function or class.
    outputs : list[str | base.Out] | dict[str, base.Out | type] | type | None
        The outputs of the function or class.
    auto_attribs : bool
        Whether to use auto_attribs mode when creating the class.
    name: str | None
        The name of the returned class
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    Task
        The task class for the Python function
    """

    def make(wrapped: ty.Callable | type) -> Task:
        if inspect.isclass(wrapped):
            klass = wrapped
            function = klass.function
            class_name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                Task,
                Outputs,
                klass,
                field.arg,
                field.out,
                auto_attribs,
                skip_fields=["function"],
            )
        else:
            if not isinstance(wrapped, str):
                raise ValueError(
                    f"wrapped must be a class or a string containing a MATLAB snipped, not {wrapped!r}"
                )
                klass = None
                input_helps, output_helps = {}, {}

                function_name, inferred_inputs, inferred_outputs = (
                    parse_matlab_function(
                        wrapped,
                        inputs=inputs,
                        outputs=outputs,
                    )
                )

                parsed_inputs, parsed_outputs = ensure_field_objects(
                    arg_type=field.arg,
                    out_type=field.out,
                    inputs=inferred_inputs,
                    outputs=inferred_outputs,
                    input_helps=input_helps,
                    output_helps=output_helps,
                )

                if name:
                    class_name = name
                else:
                    class_name = function_name
                    class_name = re.sub(r"[^\w]", "_", class_name)
                if class_name[0].isdigit():
                    class_name = f"_{class_name}"

            # Add in fields from base classes
            parsed_inputs.update({n: getattr(Task, n) for n in Task.BASE_ATTRS})
            parsed_outputs.update({n: getattr(Outputs, n) for n in Outputs.BASE_ATTRS})

            function = wrapped

        parsed_inputs["function"] = field.arg(
            name="function",
            type=str,
            default=function,
            help=Task.FUNCTION_HELP,
        )

        defn = build_task_class(
            Task,
            Outputs,
            parsed_inputs,
            parsed_outputs,
            name=class_name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
            xor=xor,
        )

        return defn

    if wrapped is not None:
        if not isinstance(wrapped, (str, type)):
            raise ValueError(f"wrapped must be a class or a string, not {wrapped!r}")
        return make(wrapped)
    return make


def parse_matlab_function(
    function: str,
    inputs: list[str | field.arg] | dict[str, field.arg | type] | None = None,
    outputs: list[str | field.out] | dict[str, field.out | type] | type | None = None,
) -> tuple[str, dict[str, field.arg], dict[str, field.out]]:
    """
    Parse a MATLAB function string to extract inputs and outputs.

    Parameters
    ----------
    function : str
        The MATLAB function string.
    inputs : list or dict, optional
        The inputs to the function.
    outputs : list or dict, optional
        The outputs of the function.

    Returns
    -------
    tuple
        A tuple containing the function name, inferred inputs, and inferred outputs.
    """
    raise NotImplementedError
