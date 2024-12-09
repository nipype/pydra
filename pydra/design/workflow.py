import typing as ty
import inspect
import attrs
from .base import (
    Arg,
    Out,
    ensure_field_objects,
    make_task_spec,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)

if ty.TYPE_CHECKING:
    from pydra.engine.workflow.base import Workflow
    from pydra.engine.specs import TaskSpec, Outputs, WorkflowSpec


__all__ = ["define", "add", "this", "arg", "out"]


@attrs.define
class arg(Arg):
    """Argument of a workflow task spec

    Parameters
    ----------
    help_string: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    xor: list, optional
        Names of the inputs that are mutually exclusive with the field.
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
    lazy: bool, optional
        If True the input field is not required at construction time but is passed straight
        through to the tasks, by default it is False
    """

    lazy: bool = False


@attrs.define
class out(Out):
    """Output of a workflow task spec

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    help_string: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    """

    pass


def define(
    wrapped: type | ty.Callable | None = None,
    /,
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    lazy: list[str] | None = None,
    auto_attribs: bool = True,
) -> "WorkflowSpec":
    """
    Create an interface for a function or a class. Can be used either as a decorator on
    a constructor function or the "canonical" dataclass-form of a task specification.

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

    Returns
    -------
    TaskSpec
        The interface for the function or class.
    """
    from pydra.engine.core import WorkflowTask
    from pydra.engine.specs import TaskSpec, WorkflowSpec, WorkflowOutputs

    if lazy is None:
        lazy = []

    def make(wrapped: ty.Callable | type) -> TaskSpec:
        if inspect.isclass(wrapped):
            klass = wrapped
            constructor = klass.constructor
            name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                klass, arg, out, auto_attribs
            )
        else:
            if not inspect.isfunction(wrapped):
                raise ValueError(
                    f"wrapped must be a class or a function, not {wrapped!r}"
                )
            klass = None
            constructor = wrapped
            input_helps, output_helps = parse_doc_string(constructor.__doc__)
            inferred_inputs, inferred_outputs = extract_function_inputs_and_outputs(
                constructor, arg, inputs, outputs
            )
            name = constructor.__name__

            parsed_inputs, parsed_outputs = ensure_field_objects(
                arg_type=arg,
                out_type=out,
                inputs=inferred_inputs,
                outputs=inferred_outputs,
                input_helps=input_helps,
                output_helps=output_helps,
            )

        parsed_inputs["constructor"] = arg(
            name="constructor", type=ty.Callable, default=constructor
        )
        for inpt_name in lazy:
            parsed_inputs[inpt_name].lazy = True

        interface = make_task_spec(
            WorkflowSpec,
            WorkflowOutputs,
            WorkflowTask,
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


def this() -> "Workflow":
    """Get the workflow currently being constructed.

    Returns
    -------
    Workflow
        The workflow currently being constructed.
    """
    from pydra.engine.workflow.base import Workflow

    return Workflow.under_construction


OutSpecType = ty.TypeVar("OutSpecType", bound="Outputs")


def add(task_spec: "TaskSpec[OutSpecType]", name: str = None) -> OutSpecType:
    """Add a node to the workflow currently being constructed

    Parameters
    ----------
    task_spec : TaskSpec
        The specification of the task to add to the workflow as a node
    name : str, optional
        The name of the node, by default it will be the name of the task specification
        class

    Returns
    -------
    Outputs
        The outputs specification of the node
    """
    return this().add(task_spec, name=name)
