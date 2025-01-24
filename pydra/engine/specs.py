"""Task I/O definitions."""

from pathlib import Path
import re
from copy import copy
import os
from operator import attrgetter
import inspect
import itertools
import platform
import shlex
from collections import Counter
import typing as ty
from glob import glob
from copy import deepcopy
from typing_extensions import Self
import attrs
import cloudpickle as cp
from fileformats.generic import FileSet
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils.typing import TypeParser
from .helpers import (
    attrs_fields,
    attrs_values,
    is_lazy,
    list_fields,
    position_sort,
    ensure_list,
    parse_format_string,
)
from .helpers_file import template_update, template_update_single
from . import helpers_state as hlpst
from . import lazy
from pydra.utils.hash import hash_function, Cache
from pydra.utils.typing import StateArray
from pydra.design.base import Field, Arg, Out, RequirementSet, EMPTY
from pydra.design import shell
from pydra.engine.lazy import LazyInField, LazyOutField

if ty.TYPE_CHECKING:
    from pydra.engine.core import Task
    from pydra.engine.graph import DiGraph
    from pydra.engine.submitter import NodeExecution
    from pydra.engine.core import Workflow
    from pydra.engine.state import StateIndex
    from pydra.engine.environments import Environment
    from pydra.engine.workers import Worker


DefType = ty.TypeVar("DefType", bound="TaskDef")


def is_set(value: ty.Any) -> bool:
    """Check if a value has been set."""
    return value not in (attrs.NOTHING, EMPTY)


class TaskOutputs:
    """Base class for all output definitions"""

    RESERVED_FIELD_NAMES = ("inputs",)

    @property
    def inputs(self):
        """The inputs object associated with a lazy-outputs object"""
        return self._get_node().inputs

    @classmethod
    def _from_defaults(cls) -> Self:
        """Create an output object from the default values of the fields"""
        return cls(
            **{
                f.name: (
                    f.default.factory()
                    if isinstance(f.default, attrs.Factory)
                    else f.default
                )
                for f in attrs_fields(cls)
            }
        )

    def _get_node(self):
        try:
            return self._node
        except AttributeError:
            raise AttributeError(
                f"{self} outputs object is not a lazy output of a workflow node"
            ) from None

    def __getitem__(self, name_or_index: str | int) -> ty.Any:
        """Return the value for the given attribute

        Parameters
        ----------
        name : str
            the name of the attribute to return

        Returns
        -------
        Any
            the value of the attribute
        """
        if isinstance(name_or_index, int):
            return list(self)[name_or_index]
        try:
            return getattr(self, name_or_index)
        except AttributeError:
            raise KeyError(
                f"{self} doesn't have an attribute {name_or_index}"
            ) from None

    def __iter__(self) -> ty.Generator[ty.Any, None, None]:
        """Iterate through all the values in the definition, allows for tuple unpacking"""
        fields = sorted(attrs_fields(self), key=attrgetter("order"))
        for field in fields:
            yield getattr(self, field.name)


OutputsType = ty.TypeVar("OutputType", bound=TaskOutputs)


@attrs.define(kw_only=True, auto_attribs=False)
class TaskDef(ty.Generic[OutputsType]):
    """Base class for all task definitions"""

    # The following fields are used to store split/combine state information
    _splitter = attrs.field(default=None, init=False, repr=False)
    _combiner = attrs.field(default=None, init=False, repr=False)
    _cont_dim = attrs.field(default=None, init=False, repr=False)
    _hashes = attrs.field(default=None, init=False, eq=False, repr=False)

    RESERVED_FIELD_NAMES = ("split", "combine")

    def __call__(
        self,
        cache_dir: os.PathLike | None = None,
        worker: "str | ty.Type[Worker] | Worker" = "debug",
        environment: "Environment | None" = None,
        rerun: bool = False,
        cache_locations: ty.Iterable[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        **kwargs: ty.Any,
    ) -> OutputsType:
        """Create a task from this definition and execute it to produce a result.

        Parameters
        ----------
        cache_dir : os.PathLike, optional
            Cache directory where the working directory/results for the task will be
            stored, by default None
        worker : str or Worker, optional
            The worker to use, by default "cf"
        environment: Environment, optional
            The execution environment to use, by default None
        rerun : bool, optional
            Whether to force the re-computation of the task results even if existing
            results are found, by default False
        cache_locations : list[os.PathLike], optional
            Alternate cache locations to check for pre-computed results, by default None
        audit_flags : AuditFlag, optional
            Auditing configuration, by default AuditFlag.NONE
        messengers : list, optional
            Messengers, by default None
        messenger_args : dict, optional
            Messenger arguments, by default None
        **kwargs : dict
            Keyword arguments to pass on to the worker initialisation

        Returns
        -------
        OutputsType or list[OutputsType]
            The output interface of the task, or in the case of split tasks, a list of
            output interfaces
        """
        from pydra.engine.submitter import (  # noqa: F811
            Submitter,
            WORKER_KWARG_FAIL_NOTE,
        )

        try:
            with Submitter(
                audit_flags=audit_flags,
                cache_dir=cache_dir,
                cache_locations=cache_locations,
                messenger_args=messenger_args,
                messengers=messengers,
                rerun=rerun,
                environment=environment,
                worker=worker,
                **kwargs,
            ) as sub:
                result = sub(self)
        except TypeError as e:
            if hasattr(e, "__notes__") and WORKER_KWARG_FAIL_NOTE in e.__notes__:
                if match := re.match(
                    r".*got an unexpected keyword argument '(\w+)'", str(e)
                ):
                    if match.group(1) in self:
                        e.add_note(
                            f"Note that the unrecognised argument, {match.group(1)!r}, is "
                            f"an input of the task definition {self!r} that has already been "
                            f"parameterised (it is being called to execute it)"
                        )
            raise
        if result.errored:
            if isinstance(self, WorkflowDef) or self._splitter:
                raise RuntimeError(f"Workflow {self} failed with errors:")
            else:
                errors = result.errors
                raise RuntimeError(
                    f"Task {self} failed @ {errors['time of crash']} with following errors:\n"
                    + "\n".join(errors["error message"])
                )
        return result.outputs

    def split(
        self,
        splitter: ty.Union[str, ty.List[str], ty.Tuple[str, ...], None] = None,
        /,
        overwrite: bool = False,
        cont_dim: ty.Optional[dict] = None,
        **inputs,
    ) -> Self:
        """
        Run this task parametrically over lists of split inputs.

        Parameters
        ----------
        splitter : str or list[str] or tuple[str] or None
            the fields which to split over. If splitting over multiple fields, lists of
            fields are interpreted as outer-products and tuples inner-products. If None,
            then the fields to split are taken from the keyword-arg names.
        overwrite : bool, optional
            whether to overwrite an existing split on the node, by default False
        cont_dim : dict, optional
            Container dimensions for specific inputs, used in the splitter.
            If input name is not in cont_dim, it is assumed that the input values has
            a container dimension of 1, so only the most outer dim will be used for splitting.
        **inputs
            fields to split over, will be automatically wrapped in a StateArray object
            and passed to the node inputs

        Returns
        -------
        self : TaskBase
            a reference to the task
        """
        if self._splitter and not overwrite:
            raise ValueError(
                f"Cannot overwrite existing splitter {self._splitter} on {self}, "
                "set 'overwrite=True' to do so"
            )
        if splitter:
            unwraped_split = list(hlpst.unwrap_splitter(splitter))
            if duplicated := [f for f, c in Counter(unwraped_split).items() if c > 1]:
                raise ValueError(f"Splitter fields {duplicated} are duplicated")
            split_names = set(
                s for s in unwraped_split if not s.startswith("_") and "." not in s
            )
            input_names = set(inputs)
            if missing_inputs := list(split_names - input_names):
                raise ValueError(
                    f"Splitter fields {missing_inputs} need to be provided as a keyword "
                    f"arguments to the split method (provided {list(inputs)})"
                )
            if unrecognised_inputs := list(input_names - split_names):
                raise ValueError(
                    f"Provided inputs {unrecognised_inputs} are not present in the "
                    f"splitter {splitter}"
                )
        else:
            # If no splitter is provided, use the names of the inputs as combinatorial splitter
            split_names = splitter = list(inputs)
        for field_name in cont_dim or []:
            if field_name not in split_names:
                raise ValueError(
                    f"Container dimension for {field_name} is provided but the field "
                    f"is not present in the inputs"
                )
        split_inputs = {}
        for name, value in inputs.items():
            if isinstance(value, lazy.LazyField):
                split_val = value.split(splitter)
            elif isinstance(value, ty.Iterable) and not isinstance(
                value, (ty.Mapping, str)
            ):
                split_val = StateArray(value)
            else:
                raise TypeError(f"Could not split {value} as it is not a sequence type")
            split_inputs[name] = split_val
        split_def = attrs.evolve(self, **split_inputs)
        split_def._splitter = splitter
        split_def._cont_dim = cont_dim
        return split_def

    def combine(
        self,
        combiner: ty.Union[ty.List[str], str],
        overwrite: bool = False,
    ) -> Self:
        """
        Combine inputs parameterized by one or more previous tasks.

        Parameters
        ----------
        combiner : list[str] or str
            the field or list of inputs to be combined (i.e. not left split) after the
            task has been run
        overwrite : bool
            whether to overwrite an existing combiner on the node
        **kwargs : dict[str, Any]
            values for the task that will be "combined" before they are provided to the
            node

        Returns
        -------
        self : Self
            a reference to the outputs object
        """
        if self._combiner and not overwrite:
            raise ValueError(
                f"Attempting to overwrite existing combiner {self._combiner} on {self}, "
                "set 'overwrite=True' to do so"
            )
        if isinstance(combiner, str):
            combiner = [combiner]
        local_names = set(c for c in combiner if "." not in c and not c.startswith("_"))
        if unrecognised := local_names - set(self):
            raise ValueError(
                f"Combiner fields {unrecognised} are not present in the task definition"
            )
        combined_def = copy(self)
        combined_def._combiner = combiner
        return combined_def

    def __iter__(self) -> ty.Generator[str, None, None]:
        """Iterate through all the names in the definition"""
        return (
            f.name
            for f in list_fields(self)
            if not (f.name.startswith("_") or f.name in self.RESERVED_FIELD_NAMES)
        )

    def __getitem__(self, name: str) -> ty.Any:
        """Return the value for the given attribute, resolving any templates

        Parameters
        ----------
        name : str
            the name of the attribute to return

        Returns
        -------
        Any
            the value of the attribute
        """
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(f"{self} doesn't have an attribute {name}") from None

    @property
    def _hash(self):
        hsh, self._hashes = self._compute_hashes()
        return hsh

    def _hash_changes(self):
        """Detects any changes in the hashed values between the current inputs and the
        previously calculated values"""
        _, new_hashes = self._compute_hashes()
        return [k for k, v in new_hashes.items() if v != self._hashes[k]]

    def _compute_hashes(self) -> ty.Tuple[bytes, ty.Dict[str, bytes]]:
        """Compute a basic hash for any given set of fields."""
        inp_dict = {}
        for field in list_fields(self):
            if isinstance(field, Out):
                continue  # Skip output fields
            # removing values that are not set from hash calculation
            if getattr(self, field.name) is attrs.NOTHING:
                continue
            if getattr(field, "container_path", False):
                continue
            inp_dict[field.name] = getattr(self, field.name)
        hash_cache = Cache()
        field_hashes = {
            k: hash_function(v, cache=hash_cache) for k, v in inp_dict.items()
        }
        return hash_function(sorted(field_hashes.items())), field_hashes

    def _resolve_lazy_inputs(
        self,
        workflow_inputs: "WorkflowDef",
        graph: "DiGraph[NodeExecution]",
        state_index: "StateIndex | None" = None,
    ) -> Self:
        """Resolves lazy fields in the task definition by replacing them with their
        actual values.

        Parameters
        ----------
        workflow : Workflow
            The workflow the task is part of
        graph : DiGraph[NodeExecution]
            The execution graph of the workflow
        state_index : StateIndex, optional
            The state index for the workflow, by default None

        Returns
        -------
        Self
            The task definition with all lazy fields resolved
        """
        resolved = {}
        for name, value in attrs_values(self).items():
            if isinstance(value, LazyInField):
                resolved[name] = value.get_value(workflow_inputs)
            elif isinstance(value, LazyOutField):
                resolved[name] = value.get_value(graph, state_index)
        return attrs.evolve(self, **resolved)

    def _check_rules(self):
        """Check if all rules are satisfied."""

        field: Arg
        errors = []
        for field in list_fields(self):
            value = getattr(self, field.name)

            if is_lazy(value):
                continue

            if value is attrs.NOTHING:
                errors.append(f"Mandatory field {field.name!r} is not set")

            # Collect alternative fields associated with this field.
            if field.xor:
                mutually_exclusive = {name: getattr(self, name) for name in field.xor}
                are_set = [
                    f"{n}={v!r}" for n, v in mutually_exclusive.items() if v is not None
                ]
                if len(are_set) > 1:
                    errors.append(
                        f"Mutually exclusive fields {field.xor} are set together: "
                        + ", ".join(are_set)
                    )
                elif not are_set:
                    errors.append(
                        f"At least one of the mutually exclusive fields {field.xor} "
                        f"should be set"
                    )

            # Raise error if any required field is unset.
            if (
                value is not None
                and field.requires
                and not any(rs.satisfied(self) for rs in field.requires)
            ):
                if len(field.requires) > 1:
                    qualification = (
                        " at least one of the following requirements to be satisfied: "
                    )
                else:
                    qualification = ""
                errors.append(
                    f"{field.name!r} requires{qualification} {[str(r) for r in field.requires]}"
                )
        if errors:
            raise ValueError(
                f"Found the following errors in task {self} definition:\n"
                + "\n".join(errors)
            )

    @classmethod
    def _check_arg_refs(cls, inputs: list[Arg], outputs: list[Out]) -> None:
        """
        Checks if all fields referenced in requirements and xor are present in the inputs
        are valid field names
        """
        field: Field
        input_names = set(inputs)
        for field in itertools.chain(inputs.values(), outputs.values()):
            if unrecognised := (
                set([r.name for s in field.requires for r in s]) - input_names
            ):
                raise ValueError(
                    "'Unrecognised' field names in referenced in the requirements "
                    f"of {field} " + str(list(unrecognised))
                )
        for inpt in inputs.values():
            if unrecognised := set(inpt.xor) - input_names:
                raise ValueError(
                    "'Unrecognised' field names in referenced in the xor "
                    f"of {inpt} " + str(list(unrecognised))
                )

    def _check_resolved(self):
        """Checks that all the fields in the definition have been resolved"""
        if lazy_values := [n for n, v in attrs_values(self).items() if is_lazy(v)]:
            raise ValueError(
                f"Cannot execute {self} because the following fields "
                f"still have lazy values {lazy_values}"
            )


@attrs.define(kw_only=True)
class Runtime:
    """Represent run time metadata."""

    rss_peak_gb: ty.Optional[float] = None
    """Peak in consumption of physical RAM."""
    vms_peak_gb: ty.Optional[float] = None
    """Peak in consumption of virtual memory."""
    cpu_peak_percent: ty.Optional[float] = None
    """Peak in cpu consumption."""


@attrs.define(kw_only=True)
class Result(ty.Generic[OutputsType]):
    """Metadata regarding the outputs of processing."""

    output_dir: Path
    outputs: OutputsType | None = None
    runtime: Runtime | None = None
    errored: bool = False

    def __getstate__(self):
        state = attrs_values(self)
        if state["outputs"] is not None:
            state["outputs"] = cp.dumps(state["outputs"])
        return state

    def __setstate__(self, state):
        if state["outputs"] is not None:
            state["outputs"] = cp.loads(state["outputs"])
        for name, val in state.items():
            setattr(self, name, val)

    def get_output_field(self, field_name):
        """Used in get_values in Workflow

        Parameters
        ----------
        field_name : `str`
            Name of field in LazyField object
        """
        if field_name == "all_":
            return attrs_values(self.outputs)
        else:
            return getattr(self.outputs, field_name)

    @property
    def errors(self):
        if self.errored:
            error_file = self.output_dir / "_error.pklz"
            if error_file.exists():
                with open(error_file, "rb") as f:
                    return cp.load(f)
        return None


@attrs.define(kw_only=True)
class RuntimeSpec:
    """
    Specification for a task.

    From CWL::

        InlineJavascriptRequirement
        SchemaDefRequirement
        DockerRequirement
        SoftwareRequirement
        InitialWorkDirRequirement
        EnvVarRequirement
        ShellCommandRequirement
        ResourceRequirement

        InlineScriptRequirement

    """

    outdir: ty.Optional[str] = None
    container: ty.Optional[str] = "shell"
    network: bool = False


class PythonOutputs(TaskOutputs):

    @classmethod
    def _from_task(cls, task: "Task[PythonDef]") -> Self:
        """Collect the outputs of a task from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        task : Task[PythonDef]
            The task whose outputs are being collected.
        outputs_dict : dict[str, ty.Any]
            The outputs of the task, as a dictionary

        Returns
        -------
        outputs : Outputs
            The outputs of the task in dataclass
        """
        outputs = cls._from_defaults()
        for name, val in task.return_values.items():
            setattr(outputs, name, val)
        return outputs


PythonOutputsType = ty.TypeVar("OutputType", bound=PythonOutputs)


class PythonDef(TaskDef[PythonOutputsType]):

    def _run(self, task: "Task[PythonDef]") -> None:
        # Prepare the inputs to the function
        inputs = attrs_values(self)
        del inputs["function"]
        # Run the actual function
        returned = self.function(**inputs)
        # Collect the outputs and save them into the task.return_values dictionary
        task.return_values = {f.name: f.default for f in attrs.fields(self.Outputs)}
        return_names = list(task.return_values)
        if returned is None:
            task.return_values = {nm: None for nm in return_names}
        elif len(task.return_values) == 1:
            # if only one element in the fields, everything should be returned together
            task.return_values = {list(task.return_values)[0]: returned}
        elif isinstance(returned, tuple) and len(return_names) == len(returned):
            task.return_values = dict(zip(return_names, returned))
        elif isinstance(returned, dict):
            task.return_values = {key: returned.get(key, None) for key in return_names}
        else:
            raise RuntimeError(
                f"expected {len(return_names)} elements, but {returned} were returned"
            )


class WorkflowOutputs(TaskOutputs):

    @classmethod
    def _from_task(cls, task: "Task[WorkflowDef]") -> Self:
        """Collect the outputs of a workflow task from the outputs of the nodes in the

        Parameters
        ----------
        task : Task[WorfklowDef]
            The task whose outputs are being collected.

        Returns
        -------
        outputs : Outputs
            The outputs of the task
        """
        outputs = cls._from_defaults()
        # collecting outputs from tasks
        output_wf = {}
        lazy_field: lazy.LazyOutField
        workflow: "Workflow" = task.return_values["workflow"]
        exec_graph: "DiGraph[NodeExecution]" = task.return_values["exec_graph"]
        nodes_dict = {n.name: n for n in exec_graph.nodes}
        for name, lazy_field in attrs_values(workflow.outputs).items():
            try:
                val_out = lazy_field.get_value(exec_graph)
                output_wf[name] = val_out
            except (ValueError, AttributeError):
                output_wf[name] = None
                node: "NodeExecution" = nodes_dict[lazy_field.name]
                # checking if the tasks has predecessors that raises error
                if isinstance(node.errored, list):
                    raise ValueError(f"Tasks {node._errored} raised an error")
                else:
                    err_files = [(t.output_dir / "_error.pklz") for t in node.tasks]
                    err_files = [f for f in err_files if f.exists()]
                    if not err_files:
                        raise
                    raise ValueError(
                        f"Task {lazy_field.name} raised an error, full crash report is "
                        f"here: "
                        + (
                            str(err_files[0])
                            if len(err_files) == 1
                            else "\n" + "\n".join(str(f) for f in err_files)
                        )
                    )
        return attrs.evolve(outputs, **output_wf)


WorkflowOutputsType = ty.TypeVar("OutputType", bound=WorkflowOutputs)


@attrs.define(kw_only=True)
class WorkflowDef(TaskDef[WorkflowOutputsType]):

    RESERVED_FIELD_NAMES = TaskDef.RESERVED_FIELD_NAMES + ("construct",)

    _constructed = attrs.field(default=None, init=False)

    def _run(self, task: "Task[WorkflowDef]") -> None:
        """Run the workflow."""
        task.submitter.expand_workflow(task)

    async def _run_async(self, task: "Task[WorkflowDef]") -> None:
        """Run the workflow asynchronously."""
        await task.submitter.expand_workflow_async(task)

    def construct(self) -> "Workflow":
        from pydra.engine.core import Workflow

        if self._constructed is not None:
            return self._constructed
        self._constructed = Workflow.construct(self)
        return self._constructed


RETURN_CODE_HELP = """The process' exit code."""
STDOUT_HELP = """The standard output stream produced by the command."""
STDERR_HELP = """The standard error stream produced by the command."""


class ShellOutputs(TaskOutputs):
    """Output definition of a generic shell process."""

    return_code: int = shell.out(help=RETURN_CODE_HELP)
    stdout: str = shell.out(help=STDOUT_HELP)
    stderr: str = shell.out(help=STDERR_HELP)

    @classmethod
    def _from_task(cls, task: "Task[ShellDef]") -> Self:
        """Collect the outputs of a shell process from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        inputs : ShellDef
            The input definition of the shell process.
        output_dir : Path
            The directory where the process was run.
        stdout : str
            The standard output of the process.
        stderr : str
            The standard error of the process.
        return_code : int
            The exit code of the process.

        Returns
        -------
        outputs : ShellOutputs
            The outputs of the shell process
        """
        outputs = cls._from_defaults()
        fld: shell.out
        for fld in list_fields(cls):
            if fld.name in ["return_code", "stdout", "stderr"]:
                resolved_value = task.return_values[fld.name]
            # Get the corresponding value from the inputs if it exists, which will be
            # passed through to the outputs, to permit manual overrides
            elif isinstance(fld, shell.outarg) and is_set(
                getattr(task.definition, fld.name)
            ):
                resolved_value = task.inputs[fld.name]
            elif is_set(fld.default):
                resolved_value = cls._resolve_default_value(fld, task.output_dir)
            else:
                resolved_value = cls._resolve_value(fld, task)
            # Set the resolved value
            setattr(outputs, fld.name, resolved_value)
        return outputs

    @classmethod
    def _resolve_default_value(cls, fld: shell.out, output_dir: Path) -> ty.Any:
        """Resolve path and glob expr default values relative to the output dir"""
        default = fld.default
        if fld.type is Path:
            assert isinstance(default, Path)
            if not default.is_absolute():
                default = output_dir.joinpath(default)
            if "*" not in str(default):
                if default.exists():
                    return default
                else:
                    raise AttributeError(f"file {default} does not exist")
            else:
                all_files = [Path(el) for el in glob(default.expanduser())]
                if len(all_files) > 1:
                    return all_files
                elif len(all_files) == 1:
                    return all_files[0]
                else:
                    raise AttributeError(f"no file matches {default.name}")
        return default

    @classmethod
    def _required_fields_satisfied(cls, fld: shell.out, inputs: "ShellDef") -> bool:
        """checking if all fields from the requires and template are set in the input
        if requires is a list of list, checking if at least one list has all elements set
        """

        if not fld.requires:
            return True

        requirements: list[RequirementSet]
        if fld.requires:
            requirements = deepcopy(fld.requires)
        else:
            requirements = [RequirementSet()]

        # if the output has output_file_template field, add in all input fields from
        # the template to requires
        if isinstance(fld, shell.outarg) and fld.path_template:
            # if a template is a function it has to be run first with the inputs as the only arg
            if callable(fld.path_template):
                template = fld.path_template(inputs)
            inp_fields = re.findall(r"{(\w+)(?:\:[^\}]+)?}", template)
            for req in requirements:
                req += inp_fields

        # Check to see if any of the requirement sets are satisfied
        return any(rs.satisfied(inputs) for rs in requirements)

    @classmethod
    def _resolve_value(
        cls,
        fld: "shell.out",
        task: "Task[DefType]",
    ) -> ty.Any:
        """Collect output file if metadata specified."""
        from pydra.design import shell

        if not cls._required_fields_satisfied(fld, task.definition):
            return None
        elif isinstance(fld, shell.outarg) and fld.path_template:
            return template_update_single(
                fld,
                definition=task.definition,
                output_dir=task.output_dir,
                spec_type="output",
            )
        elif fld.callable:
            callable_ = fld.callable
            if isinstance(fld.callable, staticmethod):
                # In case callable is defined as a static method,
                # retrieve the function wrapped in the descriptor.
                callable_ = fld.callable.__func__
            call_args = inspect.getfullargspec(callable_)
            call_args_val = {}
            for argnm in call_args.args:
                if argnm == "field":
                    call_args_val[argnm] = fld
                elif argnm == "output_dir":
                    call_args_val[argnm] = task.output_dir
                elif argnm == "inputs":
                    call_args_val[argnm] = task.inputs
                elif argnm == "stdout":
                    call_args_val[argnm] = task.return_values["stdout"]
                elif argnm == "stderr":
                    call_args_val[argnm] = task.return_values["stderr"]
                else:
                    try:
                        call_args_val[argnm] = task.inputs[argnm]
                    except KeyError as e:
                        e.add_note(
                            f"arguments of the callable function from {fld.name} "
                            f"has to be in inputs or be field or output_dir, "
                            f"but {argnm} is used"
                        )
                        raise
            return callable_(**call_args_val)
        else:
            raise Exception(
                f"Metadata for '{fld.name}', does not not contain any of the required fields "
                f'("callable", "output_file_template" or "value"): {fld}.'
            )


ShellOutputsType = ty.TypeVar("OutputType", bound=ShellOutputs)


class ShellDef(TaskDef[ShellOutputsType]):

    arguments: ty.List[str] = shell.arg(
        default=attrs.Factory(list),
        sep=" ",
        help="Additional free-form arguments to append to the end of the command.",
    )

    RESERVED_FIELD_NAMES = TaskDef.RESERVED_FIELD_NAMES + ("cmdline",)

    def _run(self, task: "Task[ShellDef]") -> None:
        """Run the shell command."""
        task.return_values = task.environment.execute(task)

    @property
    def cmdline(self) -> str:
        """The equivalent command line that would be submitted if the task were run on
        the current working directory."""
        # checking the inputs fields before returning the command line
        self._check_resolved()
        # Skip the executable, which can be a multi-part command, e.g. 'docker run'.
        cmd_args = self._command_args()
        cmdline = cmd_args[0]
        for arg in cmd_args[1:]:
            # If there are spaces in the arg, and it is not enclosed by matching
            # quotes, add quotes to escape the space. Not sure if this should
            # be expanded to include other special characters apart from spaces
            if " " in arg:
                cmdline += " '" + arg + "'"
            else:
                cmdline += " " + arg
        return cmdline

    def _command_args(
        self,
        output_dir: Path | None = None,
        input_updates: dict[str, ty.Any] | None = None,
        root: Path | None = None,
    ) -> list[str]:
        """Get command line arguments"""
        if output_dir is None:
            output_dir = Path.cwd()
        self._check_resolved()
        inputs = attrs_values(self)
        modified_inputs = template_update(self, output_dir=output_dir)
        if input_updates:
            inputs.update(input_updates)
        inputs.update(modified_inputs)
        pos_args = []  # list for (position, command arg)
        self._positions_provided = []
        for field in list_fields(self):
            name = field.name
            value = inputs[name]
            if value is None:
                continue
            if name == "executable":
                pos_args.append(self._command_shelltask_executable(field, value))
            elif name == "arguments":
                continue
            elif name == "args":
                pos_val = self._command_shelltask_args(field, value)
                if pos_val:
                    pos_args.append(pos_val)
            else:
                if name in modified_inputs:
                    pos_val = self._command_pos_args(
                        field=field,
                        value=value,
                        inputs=inputs,
                        root=root,
                        output_dir=output_dir,
                    )
                else:
                    pos_val = self._command_pos_args(
                        field=field,
                        value=value,
                        output_dir=output_dir,
                        inputs=inputs,
                        root=root,
                    )
                if pos_val:
                    pos_args.append(pos_val)

        # Sort command and arguments by position
        cmd_args = position_sort(pos_args)
        # pos_args values are each a list of arguments, so concatenate lists after sorting
        command_args = sum(cmd_args, [])
        command_args += self.arguments
        return command_args

    def _command_shelltask_executable(
        self, field: shell.arg, value: ty.Any
    ) -> tuple[int, ty.Any]:
        """Returning position and value for executable ShellTask input"""
        pos = 0  # executable should be the first el. of the command
        assert value
        return pos, ensure_list(value, tuple2list=True)

    def _command_shelltask_args(
        self, field: shell.arg, value: ty.Any
    ) -> tuple[int, ty.Any]:
        """Returning position and value for args ShellTask input"""
        pos = -1  # assuming that args is the last el. of the command
        if value is None:
            return None
        else:
            return pos, ensure_list(value, tuple2list=True)

    def _command_pos_args(
        self,
        field: shell.arg,
        value: ty.Any,
        inputs: dict[str, ty.Any],
        output_dir: Path,
        root: Path | None = None,
    ) -> tuple[int, ty.Any]:
        """
        Checking all additional input fields, setting pos to None, if position not set.
        Creating a list with additional parts of the command that comes from
        the specific field.
        """
        if field.argstr is None and field.formatter is None:
            # assuming that input that has no argstr is not used in the command,
            # or a formatter is not provided too.
            return None
        if field.position is not None:
            if not isinstance(field.position, int):
                raise Exception(
                    f"position should be an integer, but {field.position} given"
                )
            # checking if the position is not already used
            if field.position in self._positions_provided:
                raise Exception(
                    f"{field.name} can't have provided position, {field.position} is already used"
                )

            self._positions_provided.append(field.position)

            # Shift non-negatives up to allow executable to be 0
            # Shift negatives down to allow args to be -1
            field.position += 1 if field.position >= 0 else -1

        if value and isinstance(value, str):
            if root:  # values from templates
                value = value.replace(str(output_dir), f"{root}{output_dir}")

        if field.readonly and value is not None:
            raise Exception(f"{field.name} is read only, the value can't be provided")
        elif value is None and not field.readonly and field.formatter is None:
            return None

        cmd_add = []
        # formatter that creates a custom command argument
        # it can take the value of the field, all inputs, or the value of other fields.
        if field.formatter:
            call_args = inspect.getfullargspec(field.formatter)
            call_args_val = {}
            for argnm in call_args.args:
                if argnm == "field":
                    call_args_val[argnm] = value
                elif argnm == "inputs":
                    call_args_val[argnm] = inputs
                else:
                    if argnm in inputs:
                        call_args_val[argnm] = inputs[argnm]
                    else:
                        raise AttributeError(
                            f"arguments of the formatter function from {field.name} "
                            f"has to be in inputs or be field or output_dir, "
                            f"but {argnm} is used"
                        )
            cmd_el_str = field.formatter(**call_args_val)
            cmd_el_str = cmd_el_str.strip().replace("  ", " ")
            if cmd_el_str != "":
                cmd_add += split_cmd(cmd_el_str)
        elif field.type is bool and "{" not in field.argstr:
            # if value is simply True the original argstr is used,
            # if False, nothing is added to the command.
            if value is True:
                cmd_add.append(field.argstr)
        else:
            if (
                field.argstr.endswith("...")
                and isinstance(value, ty.Iterable)
                and not isinstance(value, (str, bytes))
            ):
                field.argstr = field.argstr.replace("...", "")
                # if argstr has a more complex form, with "{input_field}"
                if "{" in field.argstr and "}" in field.argstr:
                    argstr_formatted_l = []
                    for val in value:
                        argstr_f = argstr_formatting(
                            field.argstr, self, value_updates={field.name: val}
                        )
                        argstr_formatted_l.append(f" {argstr_f}")
                    cmd_el_str = field.sep.join(argstr_formatted_l)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    cmd_el_str = field.sep.join(
                        [f" {field.argstr} {val}" for val in value]
                    )
            else:
                # in case there are ... when input is not a list
                field.argstr = field.argstr.replace("...", "")
                if isinstance(value, ty.Iterable) and not isinstance(
                    value, (str, bytes)
                ):
                    cmd_el_str = field.sep.join([str(val) for val in value])
                    value = cmd_el_str
                # if argstr has a more complex form, with "{input_field}"
                if "{" in field.argstr and "}" in field.argstr:
                    cmd_el_str = field.argstr.replace(f"{{{field.name}}}", str(value))
                    cmd_el_str = argstr_formatting(cmd_el_str, self.definition)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    if value:
                        cmd_el_str = f"{field.argstr} {value}"
                    else:
                        cmd_el_str = ""
            if cmd_el_str:
                cmd_add += split_cmd(cmd_el_str)
        return field.position, cmd_add

    def _get_bindings(self, root: str | None = None) -> dict[str, tuple[str, str]]:
        """Return bindings necessary to run task in an alternative root.

        This is primarily intended for contexts when a task is going
        to be run in a container with mounted volumes.

        Arguments
        ---------
        root: str

        Returns
        -------
        bindings: dict
          Mapping from paths in the host environment to the target environment
        """

        if root is None:
            return {}
        else:
            self._prepare_bindings(root=root)
            return self.bindings

    def _prepare_bindings(self, root: str):
        """Prepare input files to be passed to the task

        This updates the ``bindings`` attribute of the current task to make files available
        in an ``Environment``-defined ``root``.
        """
        fld: Arg
        for fld in attrs_fields(self):
            if TypeParser.contains_type(FileSet, fld.type):
                fileset: FileSet = self[fld.name]
                if not isinstance(fileset, FileSet):
                    raise NotImplementedError(
                        "Generating environment bindings for nested FileSets are not "
                        "yet supported"
                    )
                copy = fld.copy_mode == FileSet.CopyMode.copy

                host_path, env_path = fileset.parent, Path(f"{root}{fileset.parent}")

                # Default to mounting paths as read-only, but respect existing modes
                old_mode = self.bindings.get(host_path, ("", "ro"))[1]
                self.bindings[host_path] = (env_path, "rw" if copy else old_mode)

                # Provide in-container paths without type-checking
                self.inputs_mod_root[fld.name] = tuple(
                    env_path / rel for rel in fileset.relative_fspaths
                )

    def _generated_output_names(self, stdout: str, stderr: str):
        """Returns a list of all outputs that will be generated by the task.
        Takes into account the task input and the requires list for the output fields.
        TODO: should be in all Output specs?
        """
        # checking the input (if all mandatory fields are provided, etc.)
        self._check_rules()
        output_names = ["return_code", "stdout", "stderr"]
        for fld in list_fields(self):
            # assuming that field should have either default or metadata, but not both
            if is_set(fld.default):
                output_names.append(fld.name)
            elif is_set(self.Outputs._resolve_output_value(fld, stdout, stderr)):
                output_names.append(fld.name)
        return output_names

    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.adjacent


def donothing(*args: ty.Any, **kwargs: ty.Any) -> None:
    return None


@attrs.define(kw_only=True)
class TaskHook:
    """Callable task hooks."""

    pre_run_task: ty.Callable = donothing
    post_run_task: ty.Callable = donothing
    pre_run: ty.Callable = donothing
    post_run: ty.Callable = donothing

    def __setattr__(self, attr, val):
        if attr not in ["pre_run_task", "post_run_task", "pre_run", "post_run"]:
            raise AttributeError("Cannot set unknown hook")
        super().__setattr__(attr, val)

    def reset(self):
        for val in ["pre_run_task", "post_run_task", "pre_run", "post_run"]:
            setattr(self, val, donothing)


def split_cmd(cmd: str):
    """Splits a shell command line into separate arguments respecting quotes

    Parameters
    ----------
    cmd : str
        Command line string or part thereof

    Returns
    -------
    str
        the command line string split into process args
    """
    # Check whether running on posix or Windows system
    on_posix = platform.system() != "Windows"
    args = shlex.split(cmd, posix=on_posix)
    cmd_args = []
    for arg in args:
        match = re.match("(['\"])(.*)\\1$", arg)
        if match:
            cmd_args.append(match.group(2))
        else:
            cmd_args.append(arg)
    return cmd_args


def argstr_formatting(
    argstr: str, inputs: dict[str, ty.Any], value_updates: dict[str, ty.Any] = None
):
    """formatting argstr that have form {field_name},
    using values from inputs and updating with value_update if provided
    """
    # if there is a value that has to be updated (e.g. single value from a list)
    # getting all fields that should be formatted, i.e. {field_name}, ...
    if value_updates:
        inputs = copy(inputs)
        inputs.update(value_updates)
    inp_fields = parse_format_string(argstr)
    val_dict = {}
    for fld_name in inp_fields:
        fld_value = inputs[fld_name]
        fld_attr = getattr(attrs.fields(type(inputs)), fld_name)
        if fld_value is None or (
            fld_value is False
            and fld_attr.type is not bool
            and TypeParser.matches_type(fld_attr.type, ty.Union[Path, bool])
        ):
            # if value is NOTHING, nothing should be added to the command
            val_dict[fld_name] = ""
        else:
            val_dict[fld_name] = fld_value

    # formatting string based on the val_dict
    argstr_formatted = argstr.format(**val_dict)
    # removing extra commas and spaces after removing the field that have NOTHING
    argstr_formatted = (
        argstr_formatted.replace("[ ", "[")
        .replace(" ]", "]")
        .replace("[,", "[")
        .replace(",]", "]")
        .strip()
    )
    return argstr_formatted
