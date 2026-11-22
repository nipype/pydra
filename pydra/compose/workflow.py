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
from pydra.utils.hash import check_code_hash
from pydra.utils.typing import StateArray, is_lazy

if ty.TYPE_CHECKING:
    from pydra.engine.workflow import Workflow
    from pydra.engine.job import Job
    from pydra.engine.lazy import LazyField, LazyOutField
    from pydra.engine.graph import DiGraph
    from pydra.engine.runner import NodeExecution
    from pydra.environments.base import Environment
    from pydra.engine.hooks import TaskHooks
    from pydra.utils.resources import ComputeResources


__all__ = [
    "define",
    "add",
    "this",
    "arg",
    "out",
    "Task",
    "Outputs",
    "cast",
    "loop",
    "LoopOutputs",
]


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
    code_hash: str | None = None,
    checksum: ty.Callable[["Task", dict[str, ty.Any]], str] | None = None,
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
    code_hash: str, optional
        How far the checksum follows the code the constructor reaches. ``"function"``
        hashes its own code and the values it reads (closure cells and globals);
        ``"module"`` adds the functions and classes it reaches in its own module;
        ``"user"`` adds all reachable code that is not library code (the standard
        library, installed distributions and pydra; see ``PYDRA_USER_PACKAGES`` in
        `pydra.utils.hash`). Library code is identified by its qualified name. By
        default the ``PYDRA_CODE_HASH`` environment variable, otherwise ``"user"``.
    checksum: callable, optional
        A function of the task and its inputs, ``checksum(task, inputs) -> str``, used
        as the task's ``__checksum__``: its return value is the job checksum and
        neither the constructor nor the input values are hashed. It must identify the task as
        well as its inputs, as jobs with equal checksums share one result and run once

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
            checksum=checksum,
        )

        defn.__doc__ = constructor.__doc__
        defn._code_hash = None if code_hash is None else check_code_hash(code_hash)
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
    resources: "ComputeResources | None" = None,
    *,
    when: "LazyField[bool] | None" = None,
    unless: "LazyField[bool] | None" = None,
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
    resources : ComputeResources, optional
        The resources to request for the node, overriding those set when the task was
        defined
    when : LazyField[bool], optional
        A gate: each element of the node runs only where it is true, and otherwise
        completes with every output ``Absent("not-routed")`` (RFC 0002 section 2.3(b))
    unless : LazyField[bool], optional
        A gate: each element of the node runs only where it is false

    Returns
    -------
    Outputs
        The outputs of the node
    """
    return this().add(
        task,
        name=name,
        environment=environment,
        hooks=hooks,
        resources=resources,
        when=when,
        unless=unless,
    )


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


class LoopOutputs:
    """The outputs of :func:`loop`: each output of the body from the last round that ran,
    ``rounds`` (int, the number of rounds that ran) and ``converged`` (bool, the last round
    that ran reported ``until`` true)"""

    def __init__(self, **fields: ty.Any):
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        return f"LoopOutputs({', '.join(self.__dict__)})"


def loop(
    body: "type[Task]",
    *,
    until: str,
    max_rounds: int,
    index: str | None = None,
    name: str | None = None,
    **inputs: ty.Any,
) -> LoopOutputs:
    """Add a bounded loop to the workflow currently being constructed, unrolled into
    ``max_rounds`` nodes ``<name>.0 … <name>.<max_rounds - 1>`` (RFC 0002 section 2.3(d))

    Round 0 takes ``inputs``. Each later round takes, for every input of ``body`` that is
    also an output of it, the previous round's output, and the rest of ``inputs``
    unchanged; it runs unless the previous round's ``until`` output is true. Rounds after
    the one that reported ``until`` true complete absent and are never dispatched.

    With ``index=None`` the body is taken to be a pure fixed-point iteration: a round
    whose inputs equal those of an earlier round has its checksum and is served that
    round's result. A body that carries nothing forward, such as a retry or a poll, must
    name an input in ``index``, which is given the round number so every round has its
    own checksum.

    Parameters
    ----------
    body : type[Task]
        The task class of one round
    until : str
        The name of a ``bool`` output of ``body``; the loop stops after the first round
        where it is true
    max_rounds : int
        The number of rounds unrolled, at least 1
    index : str, optional
        An input of ``body`` given the round number, 0 for the first round
    name : str, optional
        The prefix of the node names, by default the name of ``body``
    **inputs
        The inputs of the first round, and of every round for inputs ``body`` does not
        carry forward

    Returns
    -------
    LoopOutputs
        Each output of ``body`` from the last round that ran, chosen on the rounds'
        ``until`` outputs rather than on presence, so a content ``Absent`` that round
        returned is returned as such; plus ``rounds`` and ``converged``
    """
    from pydra.compose.select import select
    from pydra.utils.general import get_fields

    if name is None:
        name = body.__name__
    if (
        not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or max_rounds < 1
    ):
        raise ValueError(
            f"max_rounds of loop {name!r} must be an int >= 1, not {max_rounds!r}"
        )
    outputs = {f.name: f.type for f in get_fields(body.Outputs)}
    if outputs.get(until) is not bool:
        raise ValueError(
            f"until={until!r} of loop {name!r} must name a bool output of {body.__name__}, "
            f"whose outputs are {outputs}"
        )
    in_names = {f.name for f in get_fields(body)}
    if index is not None and index not in in_names:
        raise ValueError(
            f"index={index!r} of loop {name!r} must name an input of {body.__name__}, "
            f"whose inputs are {sorted(in_names)}"
        )
    for reserved in ("rounds", "converged"):
        if reserved in outputs:
            raise ValueError(
                f"{body.__name__} has an output named {reserved!r}, which loop {name!r} "
                "adds itself"
            )
    carried = sorted((in_names & set(outputs)) - {index})

    rounds = []
    for k in range(max_rounds):
        round_inputs = dict(inputs)
        if index is not None:
            round_inputs[index] = k
        if rounds:
            round_inputs.update({f: getattr(rounds[-1], f) for f in carried})
        rounds.append(
            add(
                body(**round_inputs),
                name=f"{name}.{k}",
                unless=getattr(rounds[-1], until) if rounds else None,
            )
        )

    # The last round that ran is the first whose until is true, or else the last round:
    # a round after the one reporting true never ran, and its until is an Absent.
    gates = [getattr(r, until) for r in rounds[:-1]] + [True]

    def last_round(field: str, values: list[ty.Any]) -> ty.Any:
        merge = select(max_rounds)(
            **{f"when_{k}": g for k, g in enumerate(gates)},
            **{f"value_{k}": v for k, v in enumerate(values)},
        )
        return add(merge, name=f"{name}.{field}").out

    selected = {
        f: cast(last_round(f, [getattr(r, f) for r in rounds]), tp)
        for f, tp in outputs.items()
    }
    selected["rounds"] = cast(last_round("rounds", list(range(1, max_rounds + 1))), int)
    selected["converged"] = cast(
        last_round("converged", [getattr(r, until) for r in rounds]), bool
    )
    return LoopOutputs(**selected)


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
            if is_lazy(lazy_field):
                val_out = lazy_field._get_value(workflow=workflow, graph=exec_graph)
            else:
                val_out = lazy_field  # handle non-lazy inputs that are passed through
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
        job.submitter._expand(job, rerun)

    async def _run_async(self, job: "Job[WorkflowTask]", rerun: bool) -> None:
        """Run the workflow asynchronously."""
        await job.submitter._expand_async(job, rerun)

    def construct(self) -> "Workflow":
        from pydra.engine.workflow import Workflow

        if self._constructed is not None:
            return self._constructed
        self._constructed = Workflow.construct(self)
        return self._constructed


# Alias WorkflowTask to Task so we can refer to it as workflow.Task
Task = WorkflowTask
Outputs = WorkflowOutputs
