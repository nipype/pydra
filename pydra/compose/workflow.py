import typing as ty
import inspect
from typing import dataclass_transform
import attrs
from pydra.compose import base
from pydra.compose.base import (
    ensure_field_objects,
    build_task_class,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)
from pydra.utils.general import attrs_values
from pydra.utils.typing import StateArray

if ty.TYPE_CHECKING:
    from pydra.engine.workflow import Workflow
    from pydra.engine.job import Job
    from pydra.engine.lazy import LazyOutField
    from pydra.engine.graph import DiGraph
    from pydra.engine.submitter import NodeExecution
    from pydra.environments.base import Environment
    from pydra.engine.hooks import TaskHooks


__all__ = ["define", "add", "this", "arg", "out", "Task", "Outputs", "cast"]


@attrs.define
class arg(base.Arg):
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
class out(base.Out):
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
    inputs: list[str | arg] | dict[str, arg | type] | None = None,
    outputs: list[str | out] | dict[str, out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    lazy: list[str] | None = None,
    auto_attribs: bool = True,
    name: str | None = None,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
) -> "Task":
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
    name: str | None
        The name of the returned class
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    Task
        The interface for the function or class.
    """

    if lazy is None:
        lazy = []

    def make(wrapped: ty.Callable | type) -> Task:
        if inspect.isclass(wrapped):
            klass = wrapped
            constructor = klass.constructor
            class_name = klass.__name__
            check_explicit_fields_are_none(klass, inputs, outputs)
            parsed_inputs, parsed_outputs = extract_fields_from_class(
                Task,
                Outputs,
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

            class_name = constructor.__name__ if name is None else name

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
    from pydra.engine.workflow import Workflow

    return Workflow.under_construction()


OutputsType = ty.TypeVar("OutputsType", bound="Outputs")


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


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class WorkflowOutputs(base.Outputs):

    @classmethod
    def _from_job(cls, job: "Job[WorkflowTask]") -> ty.Self:
        """Collect the outputs of a workflow job from the outputs of the nodes in the

        Parameters
        ----------
        job : Job[WorfklowDef]
            The job whose outputs are being collected.

        Returns
        -------
        outputs : Outputs
            The outputs of the job
        """

        workflow: "Workflow" = job.return_values["workflow"]
        exec_graph: "DiGraph[NodeExecution]" = job.return_values["exec_graph"]

        # Check for errors in any of the workflow nodes
        if errored := [n for n in exec_graph.nodes if n.errored]:
            errors = []
            for node in errored:
                for node_task in node.errored.values():
                    result = node_task.result()
                    if result.errors:
                        time_of_crash = result.errors["time of crash"]
                        error_message = "\n".join(result.errors["error message"])
                    else:
                        time_of_crash = "UNKNOWN-TIME"
                        error_message = "NOT RETRIEVED"
                    errors.append(
                        f"Job {node.name!r} failed @ {time_of_crash} running "
                        f"{node._task} with the following errors:\n{error_message}"
                        "\nTo inspect, please load the pickled job object from here: "
                        f"{result.cache_dir}/_job.pklz"
                    )
            raise RuntimeError(
                f"Workflow {job!r} failed with errors:\n\n" + "\n\n".join(errors)
            )

        # Retrieve values from the output fields
        values = {}
        lazy_field: LazyOutField
        for name, lazy_field in attrs_values(workflow.outputs).items():
            val_out = lazy_field._get_value(workflow=workflow, graph=exec_graph)
            if isinstance(val_out, StateArray):
                val_out = list(val_out)  # implicitly combine state arrays
            values[name] = val_out

        # Set the values in the outputs object
        outputs = super()._from_job(job)
        outputs = attrs.evolve(outputs, **values)
        outputs._cache_dir = job.cache_dir
        return outputs


WorkflowOutputsType = ty.TypeVar("OutputType", bound=WorkflowOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class WorkflowTask(base.Task[WorkflowOutputsType]):

    _executor_name = "constructor"

    RESERVED_FIELD_NAMES = base.Task.RESERVED_FIELD_NAMES + ("construct",)

    _constructed = attrs.field(default=None, init=False, repr=False, eq=False)

    def _run(self, job: "Job[WorkflowTask]", rerun: bool) -> None:
        """Run the workflow."""
        job.submitter.expand_workflow(job, rerun)

    async def _run_async(self, job: "Job[WorkflowTask]", rerun: bool) -> None:
        """Run the workflow asynchronously."""
        await job.submitter.expand_workflow_async(job, rerun)

    def construct(self) -> "Workflow":
        from pydra.engine.workflow import Workflow

        if self._constructed is not None:
            return self._constructed
        self._constructed = Workflow.construct(self)
        return self._constructed


# Alias WorkflowTask to Task so we can refer to it as workflow.Task
Task = WorkflowTask
Outputs = WorkflowOutputs
