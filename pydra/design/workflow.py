import typing as ty
import inspect
import attrs
from pydra.engine.task import FunctionTask
from pydra.engine.core import Workflow, AuditFlag
from .base import (
    Arg,
    Out,
    collate_with_helps,
    make_task_spec,
    TaskSpec,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    get_fields_from_class,
)


__all__ = ["arg", "out", "define", "this", "add", "Node", "WorkflowSpec"]


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
    if lazy is None:
        lazy = []

    def make(wrapped: ty.Callable | type) -> TaskSpec:
        if inspect.isclass(wrapped):
            klass = wrapped
            constructor = klass.constructor
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
            constructor = wrapped
            input_helps, output_helps = parse_doc_string(constructor.__doc__)
            inferred_inputs, inferred_outputs = extract_function_inputs_and_outputs(
                constructor, arg, inputs, outputs
            )
            name = constructor.__name__

            parsed_inputs, parsed_outputs = collate_with_helps(
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


OutputType = ty.TypeVar("OutputType")


class WorkflowSpec(TaskSpec[OutputType]):

    under_construction: Workflow = None

    def __construct__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir: ty.Any | None = None,
        cache_locations: ty.Any | None = None,
        cont_dim: ty.Any | None = None,
        messenger_args: ty.Any | None = None,
        messengers: ty.Any | None = None,
        rerun: bool = False,
        propagate_rerun: bool = True,
    ) -> OutputType:
        wf = self.under_construction = Workflow(
            name=type(self).__name__,
            inputs=self,
            audit_flags=audit_flags,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            cont_dim=cont_dim,
            messenger_args=messenger_args,
            messengers=messengers,
            rerun=rerun,
            propagate_rerun=propagate_rerun,
        )

        try:
            output_fields = self.construct(**attrs.asdict(self))
        finally:
            self.under_construction = None

        wf.outputs = self.Outputs(
            **dict(
                zip(
                    (f.name for f in attrs.fields(self.Outputs)),
                    output_fields,
                )
            )
        )
        return wf


def this() -> Workflow:
    return WorkflowSpec.under_construction


def add(task_spec: TaskSpec[OutputType]) -> OutputType:
    return this().add(task_spec)


@attrs.define
class Node:
    task_spec: TaskSpec
    splitter: str | list[str] | None = None
