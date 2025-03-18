import typing as ty
import inspect
from typing import dataclass_transform
import attrs
from pydra.design.base import (
    Arg,
    Out,
    ensure_field_objects,
    make_task_def,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)

if ty.TYPE_CHECKING:
    from pydra.engine.specs import PythonDef

__all__ = ["arg", "out", "define"]


@attrs.define
class arg(Arg):
    """Argument of a Python task definition

    Parameters
    ----------
    help: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    type: type, optional
        The type of the field, by default it is Any
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    """

    pass


@attrs.define
class out(Out):
    """Output of a Python task definition

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    help: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    position : int
        The position of the output in the output list, allows for tuple unpacking of
        outputs
    """

    pass


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(out,),
)
def outputs(wrapped):
    """Decorator to specify the output fields of a shell command is a dataclass-style type"""
    return wrapped


@dataclass_transform(
    kw_only_default=True,
    field_specifiers=(arg,),
)
def define(
    wrapped: type | ty.Callable | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> "PythonDef":
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
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    PythonDef
        The task definition class for the Python function
    """
    from pydra.engine.specs import PythonDef, PythonOutputs

    def make(wrapped: ty.Callable | type) -> PythonDef:
        if inspect.isclass(wrapped):
            klass = wrapped
            function = klass.function
            name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                PythonDef,
                PythonOutputs,
                klass,
                arg,
                out,
                auto_attribs,
                skip_fields=["function"],
            )
        else:
            if not isinstance(wrapped, ty.Callable):
                raise ValueError(
                    f"wrapped must be a class or a function, not {wrapped!r}"
                )
            klass = None
            function = wrapped
            input_helps, output_helps = parse_doc_string(function.__doc__)
            inferred_inputs, inferred_outputs = extract_function_inputs_and_outputs(
                function, arg, inputs, outputs
            )
            name = function.__name__

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=arg,
                out_type=out,
                inputs=inferred_inputs,
                outputs=inferred_outputs,
                input_helps=input_helps,
                output_helps=output_helps,
            )
        if "function" in parsed_inputs:
            raise ValueError(
                "The argument 'function' is reserved for a field to hold the function "
                "to be wrapped"
            )

        parsed_inputs["function"] = arg(
            name="function", type=ty.Callable, default=function, hash_eq=True
        )

        defn = make_task_def(
            PythonDef,
            PythonOutputs,
            parsed_inputs,
            parsed_outputs,
            name=name,
            klass=klass,
            bases=bases,
            outputs_bases=outputs_bases,
            xor=xor,
        )

        return defn

    if wrapped is not None:
        if not isinstance(wrapped, (ty.Callable, type)):
            raise ValueError(f"wrapped must be a class or a callable, not {wrapped!r}")
        return make(wrapped)
    return make
