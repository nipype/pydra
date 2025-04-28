import typing as ty
import inspect
from typing import dataclass_transform
import attrs
from pydra.utils.general import get_fields, asdict
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

__all__ = ["arg", "out", "define", "Task", "Outputs"]


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
        )

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

    def _run(self, job: "Job[PythonTask]", rerun: bool = True) -> None:
        # Prepare the inputs to the function
        inputs = asdict(self)
        del inputs["function"]
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


# Alias ShellTask to Task so we can refer to it by shell.Task
Task = PythonTask
Outputs = PythonOutputs
