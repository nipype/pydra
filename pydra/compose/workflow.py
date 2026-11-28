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


__all__ = ["define", "add", "this", "arg", "out", "Task", "Outputs", "cast", "loop"]


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
        `pydra.utils.hash`). Library code is identified by its qualified name and the
        version of the distribution that owns it. By
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
        cls._raise_node_errors(job, exec_graph)

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

    @staticmethod
    def _raise_node_errors(job: "Job", exec_graph: "DiGraph[NodeExecution]") -> None:
        """Raise the errors of the nodes of a workflow job's graph, if any"""
        # A block of a partial node that could not be expanded has no element, so its
        # error is reported before any read of the node
        if block_errors := [e for n in exec_graph.nodes for e in n._block_errors]:
            raise RuntimeError(
                f"Workflow {job!r} failed with errors:\n\n" + "\n\n".join(block_errors)
            )
        # Check for errors in any of the workflow nodes
        if errored := [n for n in exec_graph.nodes if n.errored]:
            errors = []
            for node in errored:
                for node_task in node.errored.values():
                    if isinstance(node_task, Exception):
                        errors.append(
                            f"Job {node.name!r} could not be resolved: {node_task!r}"
                        )
                        continue
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


WorkflowOutputsType = ty.TypeVar("OutputType", bound=WorkflowOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class WorkflowTask(base.Task[WorkflowOutputsType]):
    _executor_name = "constructor"
    #: the loop's parameters on a class :func:`loop` generates, None otherwise
    _loop = None

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


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class LoopOutputs(WorkflowOutputs):
    """The outputs of a loop: the body's outputs from the last round instantiated, plus
    ``rounds``, ``converged`` and ``stopped`` (see :func:`loop`)"""

    @classmethod
    def _from_job(cls, job: "Job[LoopTask]") -> ty.Self:
        workflow: "Workflow" = job.return_values["workflow"]
        exec_graph: "DiGraph[NodeExecution]" = job.return_values["exec_graph"]
        state = job.return_values["loop"]
        cls._raise_node_errors(job, exec_graph)
        last = state.last.node.lzout
        values = {
            n: getattr(last, n)._get_value(workflow=workflow, graph=exec_graph)
            for n in state.spec.body_outputs
        }
        values.update(rounds=state.ran, converged=state.until, stopped=state.stopped)
        outputs = super(WorkflowOutputs, cls)._from_job(job)
        outputs = attrs.evolve(outputs, **values)
        outputs._cache_dir = job.cache_dir
        return outputs


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class LoopTask(WorkflowTask[LoopOutputs]):
    """A workflow task whose graph the runner grows one round at a time (see
    :func:`loop`)"""

    def construct(self) -> "Workflow":
        """A new workflow holding round ``r0``, on every call: the runner adds rounds to
        it, so it is never shared, cached or memoised"""
        from pydra.engine.workflow import Workflow

        Workflow.clear_cache(self)
        return Workflow.construct(self, dont_cache=True)


_loop_classes: dict = {}


def loop(
    body: "type[base.Task]",
    until: str = "done",
    carry: ty.Mapping[str, str] | None = None,
    index: str | None = None,
    max_rounds: int | None = None,
    environment: "Environment | None" = None,
    hooks: "TaskHooks | None" = None,
    resources: "ComputeResources | None" = None,
) -> "type[LoopTask]":
    """The task class of a loop over ``body``, whose rounds are instantiated at run time
    (RFC 0002 section 2.3(d))

    The loop is one node, which can be split, gated and read; round ``r<k+1>`` is added
    when round ``k`` returns a false ``until``, its ``carry`` inputs reading round ``k``'s
    outputs. The loop stops when ``until`` is an ``Absent`` or true, when a round repeats an
    earlier round's checksum (``index=None``), or at ``max_rounds``; a failed round fails
    it. Its outputs are the last round's, plus ``rounds``, ``converged`` and ``stopped``.

    Parameters
    ----------
    body : type[Task]
        The task class of one round, with no splitter of its own
    until : str
        The name of a ``bool`` output of ``body``
    carry : Mapping[str, str], optional
        Maps an input of ``body`` to the output of ``body`` that feeds it in the next
        round; by default the inputs and outputs that share a name. A pair whose types do
        not match, a name that matches nothing, and an empty carry without ``index`` are
        refused
    index : str, optional
        An input of ``body`` given the round number, 0 for ``r0``; required for a body that
        carries nothing forward
    max_rounds : int, optional
        The most rounds instantiated
    environment, hooks, resources : optional
        Given to each round, as to a node added with :func:`add`

    Returns
    -------
    type[LoopTask]
        The loop's task class, one per set of parameters
    """
    from pydra.engine.lazy import LazyOutField
    from pydra.engine.loop import LOOP_OUTPUTS, LoopConstructor, LoopSpec
    from pydra.utils.general import get_fields

    if not (inspect.isclass(body) and issubclass(body, base.Task)):
        raise TypeError(
            f"The body of a loop must be a task class, not {body!r}: a body that splits "
            "is a workflow that combines before its 'until' output"
        )
    what = f"loop over {body.__name__}"
    executor = getattr(body, "_executor_name", None)
    in_fields = {f.name: f for f in get_fields(body) if f.name != executor}
    out_fields = {f.name: f for f in get_fields(body.Outputs)}
    if until not in out_fields or out_fields[until].type is not bool:
        raise ValueError(
            f"until={until!r} of the {what} must name an output typed bool; its outputs "
            f"are {({n: f.type for n, f in out_fields.items()})}"
        )
    if reserved := [n for n in LOOP_OUTPUTS if n in out_fields]:
        raise ValueError(
            f"{body.__name__} has outputs {reserved}, which the {what} adds itself"
        )
    if max_rounds is not None and (
        not isinstance(max_rounds, int)
        or isinstance(max_rounds, bool)
        or max_rounds < 1
    ):
        raise ValueError(
            f"max_rounds of the {what} must be None or an int >= 1, not {max_rounds!r}"
        )
    if index is not None and index not in in_fields:
        raise ValueError(
            f"index={index!r} of the {what} must name an input; its inputs are "
            f"{sorted(in_fields)}"
        )
    if carry is None:
        carry = {n: n for n in in_fields if n in out_fields and n != index}
    carry = dict(carry)
    if unknown := [n for n in carry if n not in in_fields]:
        raise ValueError(
            f"carry of the {what} names {unknown}, which are not inputs; its inputs are "
            f"{sorted(in_fields)}"
        )
    if unknown := [o for o in carry.values() if o not in out_fields]:
        raise ValueError(
            f"carry of the {what} names {unknown}, which are not outputs; its outputs "
            f"are {sorted(out_fields)}"
        )
    if index is not None and index in carry:
        raise ValueError(
            f"index={index!r} of the {what} is also carried: the round number and a "
            "carried value cannot feed one input"
        )
    if not carry and index is None:
        raise ValueError(
            f"The {what} carries nothing and has no index, so every round would be the "
            "same computation: name the carried inputs in carry, or the input given the "
            "round number in index (a retry or poll loop)"
        )
    converters = attrs.fields_dict(body)
    for inp, outp in carry.items():
        probe = LazyOutField(node=None, field=outp, type=out_fields[outp].type)
        try:
            converters[inp].converter(probe)
        except TypeError as exc:
            raise TypeError(
                f"carry of the {what} feeds output {outp!r} "
                f"({out_fields[outp].type}) to input {inp!r} ({in_fields[inp].type}): "
                f"{exc}"
            ) from None

    spec = LoopSpec(
        body=body,
        until=until,
        carry=tuple(sorted(carry.items())),
        index=index,
        max_rounds=max_rounds,
        environment=environment,
        hooks=hooks,
        resources=resources,
    )
    key = (spec, environment, hooks, resources)
    try:
        return _loop_classes[key]
    except TypeError:  # an unhashable environment, hooks or resources: not kept
        key = None
    except KeyError:
        pass
    inputs = {
        n: arg(
            name=n,
            type=f.type,
            default=f.default,
            help=f.help,
            absent=f.absent,
            hash_eq=f.hash_eq,
        )
        for n, f in in_fields.items()
        if n != index
    }
    outputs = {n: out(name=n, type=f.type, help=f.help) for n, f in out_fields.items()}
    outputs["rounds"] = out(name="rounds", type=int)
    outputs["converged"] = out(name="converged", type=bool)
    outputs["stopped"] = out(name="stopped", type=str)
    inputs["constructor"] = arg(
        name="constructor",
        type=ty.Callable,
        hash_eq=True,
        default=LoopConstructor(spec),
    )
    klass = build_task_class(
        LoopTask,
        LoopOutputs,
        inputs,
        outputs,
        name=f"{body.__name__}Loop",
    )
    klass.__doc__ = f"A runtime loop over {body.__name__} until {until!r}"
    klass._code_hash = None
    klass._loop = spec
    if key is not None:
        _loop_classes[key] = klass
    return klass
