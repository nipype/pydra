"""Task I/O definitions."""

from pathlib import Path
import re
from copy import copy
import inspect
import itertools
import platform
import shlex
import typing as ty
from glob import glob
from copy import deepcopy
from typing_extensions import Self
import attrs
import cloudpickle as cp
from fileformats.generic import File
from pydra.engine.audit import AuditFlag
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
from .helpers_file import template_update
from pydra.utils.hash import hash_function, Cache
from pydra.design.base import Field, Arg, Out, RequirementSet, EMPTY
from pydra.design import shell

if ty.TYPE_CHECKING:
    from pydra.engine.core import Task
    from pydra.engine.task import ShellTask


def is_set(value: ty.Any) -> bool:
    """Check if a value has been set."""
    return value not in (attrs.NOTHING, EMPTY)


class TaskOutputs:
    """Base class for all output definitions"""

    RESERVED_FIELD_NAMES = ("inputs", "split", "combine")

    @classmethod
    def from_task(cls, task: "Task") -> Self:
        """Collect the outputs of a task from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        task : Task
            The task whose outputs are being collected.

        Returns
        -------
        outputs : Outputs
            The outputs of the task
        """
        return cls(
            **{
                f.name: task.output_.get(f.name, attrs.NOTHING)
                for f in attrs_fields(cls)
            }
        )

    @property
    def inputs(self):
        """The inputs object associated with a lazy-outputs object"""
        return self._get_node().inputs

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
            fields to split over, will automatically be wrapped in a StateArray object
            and passed to the node inputs

        Returns
        -------
        self : TaskBase
            a reference to the task
        """
        self._get_node().split(
            splitter, overwrite=overwrite, cont_dim=cont_dim, **inputs
        )
        return self

    def combine(
        self,
        combiner: ty.Union[ty.List[str], str],
        overwrite: bool = False,  # **kwargs
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
        self._get_node().combine(combiner, overwrite=overwrite)
        return self

    def _get_node(self):
        try:
            return self._node
        except AttributeError:
            raise AttributeError(
                f"{self} outputs object is not a lazy output of a workflow node"
            )

    def __iter__(self) -> ty.Generator[str, None, None]:
        """Iterate through all the names in the definition"""
        return (f.name for f in list_fields(self))

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


OutputsType = ty.TypeVar("OutputType", bound=TaskOutputs)


class TaskDef(ty.Generic[OutputsType]):
    """Base class for all task definitions"""

    Task: "ty.Type[core.Task]"

    RESERVED_FIELD_NAMES = ()

    def __call__(
        self,
        name: str | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        inputs: ty.Text | File | dict[str, ty.Any] | None = None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        rerun=False,
        **kwargs,
    ) -> "Result[OutputsType]":
        """Create a task from this definition and execute it to produce a result.

        Parameters
        ----------
        name : str, optional
            The name of the task, by default None
        audit_flags : AuditFlag, optional
            Auditing configuration, by default AuditFlag.NONE
        cache_dir : os.PathLike, optional
            Cache directory, by default None
        cache_locations : list[os.PathLike], optional
            Cache locations, by default None
        inputs : str or File or dict, optional
            Inputs for the task, by default None
        cont_dim : dict, optional
            Container dimensions for specific inputs, by default None
        messenger_args : dict, optional
            Messenger arguments, by default None
        messengers : list, optional
            Messengers, by default None
        rerun : bool, optional
            Whether to rerun the task, by default False
        **kwargs
            Additional keyword arguments to pass to the task

        Returns
        -------
        Result
            The result of the task
        """
        self._check_rules()
        task = self.Task(
            spec=self,
            name=name,
            audit_flags=audit_flags,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            inputs=inputs,
            cont_dim=cont_dim,
            messenger_args=messenger_args,
            messengers=messengers,
            rerun=rerun,
        )
        return task(**kwargs)

    def __iter__(self) -> ty.Generator[str, None, None]:
        """Iterate through all the names in the definition"""
        return (f.name for f in list_fields(self))

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
        for field in attrs_fields(self):
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

    def _retrieve_values(self, wf, state_index=None):
        """Parse output results."""
        temp_values = {}
        for field in attrs_fields(self):
            value = getattr(self, field.name)
            if is_lazy(value):
                temp_values[field.name] = value.get_value(wf, state_index=state_index)
        for field, val in temp_values.items():
            setattr(self, field, val)

    def _check_rules(self):
        """Check if all rules are satisfied."""

        field: Arg
        for field in list_fields(self):
            value = getattr(self, field.name)

            if is_lazy(value):
                continue

            # Collect alternative fields associated with this field.
            if field.xor:
                alternative_fields = {
                    name: getattr(self, name)
                    for name in field.xor
                    if name != field.name
                }
                set_alternatives = {n: v for n, v in alternative_fields.items() if v}

                # Raise error if no field in mandatory alternative group is set.
                if not is_set(value):
                    if set_alternatives:
                        continue
                    message = f"{field.name} is mandatory and unset."
                    if alternative_fields:
                        raise AttributeError(
                            message[:-1]
                            + f", and no alternative provided in {list(alternative_fields)}."
                        )
                    else:
                        raise AttributeError(message)

                # Raise error if multiple alternatives are set.
                elif set_alternatives:
                    raise AttributeError(
                        f"{field.name} is mutually exclusive with {set_alternatives}"
                    )

            # Raise error if any required field is unset.
            if (
                value
                and field.requires
                and not any(rs.satisfied(self) for rs in field.requires)
            ):
                if len(field.requires) > 1:
                    qualification = (
                        " at least one of the following requirements to be satisfied: "
                    )
                else:
                    qualification = ""
                raise ValueError(
                    f"{field.name!r} requires{qualification} {[str(r) for r in field.requires]}"
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
        """Checks that all the fields in the spec have been resolved"""
        if has_lazy_values := [n for n, v in attrs_values(self).items() if is_lazy(v)]:
            raise ValueError(
                f"Cannot execute {self} because the following fields "
                f"still have lazy values {has_lazy_values}"
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

    output: OutputsType | None = None
    runtime: Runtime | None = None
    errored: bool = False

    def __getstate__(self):
        state = attrs_values(self)
        if state["output"] is not None:
            state["output"] = cp.dumps(state["output"])
        return state

    def __setstate__(self, state):
        if state["output"] is not None:
            state["output"] = cp.loads(state["output"])
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
            return attrs_values(self.output)
        else:
            return getattr(self.output, field_name)


@attrs.define(kw_only=True)
class RuntimeDef:
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
    pass


PythonOutputsType = ty.TypeVar("OutputType", bound=PythonOutputs)


class PythonDef(TaskDef[PythonOutputsType]):
    pass


class WorkflowOutputs(TaskOutputs):
    pass


WorkflowOutputsType = ty.TypeVar("OutputType", bound=WorkflowOutputs)


class WorkflowDef(TaskDef[WorkflowOutputsType]):
    pass


RETURN_CODE_HELP = """The process' exit code."""
STDOUT_HELP = """The standard output stream produced by the command."""
STDERR_HELP = """The standard error stream produced by the command."""


class ShellOutputs(TaskOutputs):
    """Output definition of a generic shell process."""

    return_code: int = shell.out(help_string=RETURN_CODE_HELP)
    stdout: str = shell.out(help_string=STDOUT_HELP)
    stderr: str = shell.out(help_string=STDERR_HELP)

    @classmethod
    def from_task(
        cls,
        task: "ShellTask",
    ) -> Self:
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

        outputs = super().from_task(task)

        fld: shell.out
        for fld in list_fields(cls):
            if fld.name in ["return_code", "stdout", "stderr"]:
                continue
            # Get the corresponding value from the inputs if it exists, which will be
            # passed through to the outputs, to permit manual overrides
            if isinstance(fld, shell.outarg) and is_set(getattr(task.spec, fld.name)):
                resolved_value = task.inputs[fld.name]
            elif is_set(fld.default):
                resolved_value = cls._resolve_default_value(fld, task.output_dir)
            else:
                resolved_value = task.resolve_value(fld, outputs.stdout, outputs.stderr)
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


ShellOutputsType = ty.TypeVar("OutputType", bound=ShellOutputs)


class ShellDef(TaskDef[ShellOutputsType]):

    RESERVED_FIELD_NAMES = ("cmdline",)

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
        return sum(cmd_args, [])

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
                    cmd_el_str = argstr_formatting(cmd_el_str, self.spec)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    if value:
                        cmd_el_str = f"{field.argstr} {value}"
                    else:
                        cmd_el_str = ""
            if cmd_el_str:
                cmd_add += split_cmd(cmd_el_str)
        return field.position, cmd_add


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


from pydra.engine import core  # noqa: E402
