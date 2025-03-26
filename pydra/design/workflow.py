import typing as ty
import inspect
from typing import dataclass_transform
import attrs
from pydra.design.base import (
    Arg,
    Out,
    ensure_field_objects,
    make_task,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)

if ty.TYPE_CHECKING:
    from pydra.engine.core import Workflow
    from pydra.engine.specs import Task, TaskOutputs, WorkflowTask
    from pydra.engine.environments import Environment
    from pydra.engine.specs import TaskHooks


__all__ = ["define", "add", "this", "arg", "out"]


@attrs.define
class arg(Arg):
    """Argument of a workflow task

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
    lazy: bool, optional
        If True the input field is not required at construction time but is passed straight
        through to the tasks, by default it is False
    """

    pass


@attrs.define
class out(Out):
    """Output of a workflow task

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
    lazy: list[str] | None = None,
    auto_attribs: bool = True,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> "WorkflowTask":
    """
    Create an interface for a function or a class. Can be used either as a decorator on
    a constructor function or the "canonical" dataclass-form of a task.

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
    Task
        The interface for the function or class.
    """
    from pydra.engine.specs import Task, WorkflowTask, WorkflowOutputs

    if lazy is None:
        lazy = []

    def make(wrapped: ty.Callable | type) -> Task:
        if inspect.isclass(wrapped):
            klass = wrapped
            constructor = klass.constructor
            name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                WorkflowTask,
                WorkflowOutputs,
                klass,
                arg,
                out,
                auto_attribs,
                skip_fields=["constructor"],
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

        if "constructor" in parsed_inputs:
            raise ValueError(
                "The argument 'constructor' is reserved and cannot be used as an "
                "argument name"
            )

        parsed_inputs["constructor"] = arg(
            name="constructor", type=ty.Callable, hash_eq=True, default=constructor
        )
        for inpt_name in lazy:
            parsed_inputs[inpt_name].lazy = True

        defn = make_task(
            WorkflowTask,
            WorkflowOutputs,
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


def this() -> "Workflow":
    """Get the workflow currently being constructed.

    Returns
    -------
    Workflow
        The workflow currently being constructed.
    """
    from pydra.engine.core import Workflow

    return Workflow.under_construction()


OutputsType = ty.TypeVar("OutputsType", bound="TaskOutputs")


def add(
    task: "Task[OutputsType]",
    name: str | None = None,
    environment: "Environment | None" = None,
    hooks: "TaskHooks | None" = None,
) -> OutputsType:
    """Add a node to the workflow currently being constructed

    Parameters
    ----------
    task : Task
        The definition of the task to add to the workflow as a node
    name : str, optional
        The name of the node, by default it will be the name of the task
        class
    environment : Environment, optional
        The environment to run the task in, such as the Docker or Singularity container,
        by default it will be the "native"
    hooks : TaskHooks, optional
        The hooks to run before or after the task, by default no hooks will be run

    Returns
    -------
    Outputs
        The outputs of the node
    """
    return this().add(task, name=name, environment=environment, hooks=hooks)


U = ty.TypeVar("U")


def cast(field: ty.Any, new_type: type[U]) -> U:
    """Cast a lazy field to a new type. Note that the typing in the signature is a white
    lie, as the return field is actually a LazyField as placeholder for the object of
    type U.

    Parameters
    ----------
    field : LazyField[T]
        The field to cast
    new_type : type[U]
        The new type to cast the field to

    Returns
    -------
    LazyField[U]
        A copy of the lazy field with the new type
    """
    return attrs.evolve(
        field,
        type=new_type,
        cast_from=field._cast_from if field._cast_from else field._type,
    )
