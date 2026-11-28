import typing as ty
import warnings
import inspect
from typing import dataclass_transform
import attrs
from pydra.utils.general import get_fields, asdict
from pydra.utils.hash import check_code_hash
from pydra.compose import base
from pydra.compose.base import (
    ensure_field_objects,
    build_task_class,
    parse_doc_string,
    extract_function_inputs_and_outputs,
    check_explicit_fields_are_none,
    extract_fields_from_class,
)

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.utils.resources import ComputeResources

__all__ = ["arg", "out", "define", "Task", "Outputs", "units_of", "batched_inputs"]


@attrs.define
class arg(base.Arg):
    """Argument of a Python task

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
class out(base.Out):
    """Output of a Python task

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
    field_specifiers=(arg,),
)
def define(
    wrapped: type | ty.Callable | None = None,
    /,
    inputs: list[str | arg] | dict[str, arg | type] | None = None,
    outputs: list[str | out] | dict[str, out | type] | type | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    auto_attribs: bool = True,
    name: str | None = None,
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
    version: str | None = None,
    resources: "ComputeResources | None" = None,
    code_hash: str | None = None,
    checksum: ty.Callable[["Task", dict[str, ty.Any]], str] | None = None,
    batched: ty.Callable[..., ty.Sequence[ty.Any]] | None = None,
    units: ty.Callable[..., float | None] | None = None,
    batch: int | None = None,
) -> "Task":
    """Create an interface for a function or a class.

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
    version: str, optional
        Declared version of the task: the checksum uses the function's module, name and
        this version in place of the code it reaches, so change it whenever a code change
        changes results. The values the code reads are still hashed. A
        `pydra.engine.store.DeclaredVersionWarning` is issued when a stored result of the
        same version was computed from other code (``reuse="exact"`` then recomputes it);
        ``pydracli check-versions`` checks versions against a committed table of digests.
        A function defined in ``__main__`` or inside another function is identified only by
        its name, version and the values it reads; ``define`` warns (``UserWarning``) when
        such a function is given a version.
    resources: ComputeResources, optional
        What the task asks of a scheduler worker (Slurm, SGE); never part of the checksum
    code_hash: str, optional
        How far the checksum follows the code the function reaches. ``"function"``
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
        neither the function nor the input values are hashed. It must identify the task as
        well as its inputs, as jobs with equal checksums share one result and run once
    batched: callable, optional
        The task over several jobs in one call, for a warm pool worker (RFC 0005 section
        4.4): ``batched(**inputs)`` takes each input as a list with one element per job (a
        provided field as the one state they share) and returns, per job, its return value
        or the exception it raised. Its results must equal the function's on each job
        alone; its code joins the task's key. The function runs wherever jobs are not
        batched.
    units: callable, optional
        A job's size in the task's own units (seconds of audio, voxels), called with the
        job's inputs its signature names; a number or None. A pool orders a batched call's
        jobs by it and fits memory against it. Never part of the checksum.
    batch: int, optional
        The most jobs one batched call takes, capped by a measured memory bound. Without
        it a pool's `MeasuredFit` probes the size call by call; `DeclaredFit` runs one job
        per call. Never part of the checksum.

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

            class_name = function.__name__ if name is None else name

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
            resources=resources,
            checksum=checksum,
        )

        defn.__doc__ = function.__doc__
        if version is not None and (
            function.__module__ == "__main__" or "<locals>" in function.__qualname__
        ):
            warnings.warn(
                f"version={version!r} identifies {function.__module__}."
                f"{function.__qualname__}, which another function of the same name, "
                "version and values read would share, so their results would be "
                "reused as one",
                UserWarning,
                stacklevel=2,
            )
        defn._version = version
        defn._code_hash = None if code_hash is None else check_code_hash(code_hash)
        for what, value in (("batched", batched), ("units", units)):
            if value is not None and not callable(value):
                raise TypeError(f"{what} must be callable, not {value!r}")
        if batch is not None and (
            isinstance(batch, bool) or not isinstance(batch, int) or batch < 1
        ):
            raise ValueError(f"batch must be a positive int, not {batch!r}")
        defn._batched = batched
        defn._units = units
        defn._batch = batch
        return defn

    if wrapped is not None:
        if not isinstance(wrapped, (ty.Callable, type)):
            raise ValueError(f"wrapped must be a class or a callable, not {wrapped!r}")
        return make(wrapped)
    return make


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class PythonOutputs(base.Outputs):
    @classmethod
    def _from_job(cls, job: "Job[PythonTask]") -> ty.Self:
        """Collect the outputs of a job from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        job : Job[Task]
            The job whose outputs are being collected.
        outputs_dict : dict[str, ty.Any]
            The outputs of the job, as a dictionary

        Returns
        -------
        outputs : Outputs
            The outputs of the job in dataclass
        """
        outputs = super()._from_job(job)
        for name, val in job.return_values.items():
            setattr(outputs, name, val)
        return outputs


PythonOutputsType = ty.TypeVar("PythonOutputsType", bound=PythonOutputs)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class PythonTask(base.Task[PythonOutputsType]):
    _executor_name = "function"
    # The task over several jobs in one call, and a job's size, set by `define` (RFC 0005
    # section 4.4). Class attributes, never fields; the batched callable's code enters the
    # task's key (`pydra.utils.identity.field_digests`).
    _batched: ty.ClassVar[ty.Callable[..., ty.Sequence[ty.Any]] | None] = None
    _units: ty.ClassVar[ty.Callable[..., float | None] | None] = None
    _batch: ty.ClassVar[int | None] = None  # the declared most jobs of one batched call

    def _run(self, job: "Job[PythonTask]", rerun: bool = True) -> None:
        element = getattr(job, "batched_element", None)
        if element is not None:
            # this job's element of a batched call that already ran (`batched_inputs`)
            (returned,) = element
            if isinstance(returned, BaseException):
                raise returned
        else:
            # Prepare the inputs to the function
            inputs = asdict(self)
            del inputs["function"]
            # A provided field is handed the state its provider loaded from the spec
            inputs.update(getattr(job, "provided_values", None) or {})
            # Run the actual function
            returned = self.function(**inputs)
        # Collect the outputs and save them into the job.return_values dictionary
        return_names = [f.name for f in get_fields(self.Outputs)]
        if returned is None:
            job.return_values = {nm: None for nm in return_names}
        elif not return_names:
            raise ValueError(
                f"No output fields were specified, but the function returned {returned}"
            )
        elif len(return_names) == 1:
            # if only one element in the fields, everything should be returned together
            job.return_values[return_names[0]] = returned
        elif isinstance(returned, tuple) and len(return_names) == len(returned):
            job.return_values.update(zip(return_names, returned))
        elif isinstance(returned, dict):
            job.return_values.update(
                {key: returned[key] for key in return_names if key in returned}
            )
        else:
            raise RuntimeError(
                f"expected {len(return_names)} elements, but {returned} were returned"
            )


def units_of(task: PythonTask) -> float | None:
    """The task's size in its own units: its class's ``units`` called with the inputs its
    signature names (all of them when it takes ``**kwargs``); None without one."""
    units = getattr(type(task), "_units", None)
    if units is None:
        return None
    inputs = asdict(task)
    inputs.pop("function", None)
    params = inspect.signature(units).parameters.values()
    if not any(p.kind is p.VAR_KEYWORD for p in params):
        names = {p.name for p in params}
        inputs = {k: v for k, v in inputs.items() if k in names}
    value = units(**inputs)
    return None if value is None else float(value)


def batched_inputs(
    tasks: ty.Sequence[PythonTask], provided: dict[str, ty.Any]
) -> dict[str, ty.Any]:
    """The keyword arguments of a batched call of ``tasks`` (one task class): each input as a
    list over the tasks, and each provided field as the one state ``provided`` holds."""
    inputs: dict[str, ty.Any] = {}
    for task in tasks:
        values = asdict(task)
        del values["function"]
        for name, value in values.items():
            if name not in provided:
                inputs.setdefault(name, []).append(value)
    inputs.update(provided)
    return inputs


# Alias ShellTask to Task so we can refer to it by shell.Task
Task = PythonTask
Outputs = PythonOutputs
