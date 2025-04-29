from __future__ import annotations
import typing as ty
import re
import glob
import inspect
import shlex
import platform
from pathlib import Path
from copy import copy, deepcopy
import attrs
from fileformats.generic import FileSet, File
from pydra.utils.general import (
    attrs_values,
    get_fields,
    ensure_list,
    position_sort,
)
from pydra.utils.typing import (
    is_fileset_or_union,
    state_array_support,
    is_optional,
    optional_type,
    is_multi_input,
    MultiOutputObj,
    MultiOutputFile,
)
from pydra.compose import base
from pydra.compose.base.field import RequirementSet
from pydra.compose.base.helpers import is_set
from . import field
from .templating import (
    template_update,
    template_update_single,
    argstr_formatting,
    fields_in_formatter,
    parse_format_string,
)

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job

TaskType = ty.TypeVar("TaskType", bound="Task")


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class ShellOutputs(base.Outputs):
    """Output task of a generic shell process."""

    BASE_ATTRS = ["return_code", "stdout", "stderr"]
    RETURN_CODE_HELP = """The process' exit code."""
    STDOUT_HELP = """The standard output stream produced by the command."""
    STDERR_HELP = """The standard error stream produced by the command."""

    return_code: int = field.out(name="return_code", type=int, help=RETURN_CODE_HELP)
    stdout: str = field.out(name="stdout", type=str, help=STDOUT_HELP)
    stderr: str = field.out(name="stderr", type=str, help=STDERR_HELP)

    @classmethod
    def _from_job(cls, job: "Job[Task]") -> ty.Self:
        """Collect the outputs of a shell process from a combination of the provided inputs,
        the objects in the output directory, and the stdout and stderr of the process.

        Parameters
        ----------
        inputs : Task
            The input task of the shell process.
        cache_dir : Path
            The directory where the process was run.
        stdout : str
            The standard output of the process.
        stderr : str
            The standard error of the process.
        return_code : int
            The exit code of the process.

        Returns
        -------
        outputs : Outputs
            The outputs of the shell process
        """
        outputs = super()._from_job(job)
        fld: field.out
        for fld in get_fields(cls):
            if fld.name in ["return_code", "stdout", "stderr"]:
                resolved_value = job.return_values[fld.name]
            # Get the corresponding value from the inputs if it exists, which will be
            # passed through to the outputs, to permit manual overrides
            elif isinstance(fld, field.outarg) and isinstance(
                job.inputs[fld.name], Path
            ):
                resolved_value = job.inputs[fld.name]
            elif is_set(fld.default):
                resolved_value = cls._resolve_default_value(fld, job.cache_dir)
            else:
                resolved_value = cls._resolve_value(fld, job)
            # Set the resolved value
            try:
                setattr(outputs, fld.name, resolved_value)
            except FileNotFoundError:
                if is_optional(fld.type):
                    setattr(outputs, fld.name, None)
                else:
                    raise ValueError(
                        f"file system path(s) provided to mandatory field {fld.name!r}, "
                        f"'{resolved_value}', does not exist, this is likely due to an "
                        f"error in the {job.name!r} job"
                    )
        return outputs

    @classmethod
    def _resolve_default_value(cls, fld: field.out, cache_dir: Path) -> ty.Any:
        """Resolve path and glob expr default values relative to the output dir"""
        default = fld.default
        if fld.type is Path:
            assert isinstance(default, Path)
            if not default.is_absolute():
                default = cache_dir.joinpath(default)
            if "*" not in str(default):
                if default.exists():
                    return default
                else:
                    raise FileNotFoundError(f"file {default} does not exist")
            else:
                all_files = [Path(el) for el in glob(default.expanduser())]
                if len(all_files) > 1:
                    return all_files
                elif len(all_files) == 1:
                    return all_files[0]
                else:
                    raise FileNotFoundError(f"no file matches {default.name}")
        return default

    @classmethod
    def _required_fields_satisfied(cls, fld: field.out, inputs: "Task") -> bool:
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
        if isinstance(fld, field.outarg) and fld.path_template:
            # if a template is a function it has to be run first with the inputs as the only arg
            if callable(fld.path_template):
                template = fld.path_template(inputs)
            else:
                template = fld.path_template
            inp_fields = re.findall(r"{(\w+)(?:\:[^\}]+)?}", template)
            for req in requirements:
                req += inp_fields

        # Check to see if any of the requirement sets are satisfied
        return any(rs.satisfied(inputs) for rs in requirements)

    @classmethod
    def _resolve_value(
        cls,
        fld: "field.out",
        job: "Job[TaskType]",
    ) -> ty.Any:
        """Collect output file if metadata specified."""

        if not cls._required_fields_satisfied(fld, job.task):
            return None
        if isinstance(fld, field.outarg) and fld.path_template:
            return template_update_single(
                fld,
                task=job.task,
                cache_dir=job.cache_dir,
                spec_type="output",
            )
        assert fld.callable, (
            f"Output field '{fld.name}', does not not contain any of the required fields "
            f'("callable", "output_file_template" or "value"): {fld}.'
        )
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
            elif argnm == "cache_dir":
                call_args_val[argnm] = job.cache_dir
            elif argnm == "executable":
                call_args_val[argnm] = job.task.executable
            elif argnm == "inputs":
                call_args_val[argnm] = job.inputs
            elif argnm == "stdout":
                call_args_val[argnm] = job.return_values["stdout"]
            elif argnm == "stderr":
                call_args_val[argnm] = job.return_values["stderr"]
            elif argnm == "self":
                pass  # If the callable is a class
            else:
                try:
                    call_args_val[argnm] = job.inputs[argnm]
                except KeyError as e:
                    e.add_note(
                        f"arguments of the callable function from {fld.name!r} "
                        f"has to be in inputs or be field or cache_dir, "
                        f"but {argnm!r} is used"
                    )
                    raise
        return callable_(**call_args_val)


ShellOutputsType = ty.TypeVar("OutputType", bound=ShellOutputs)


@state_array_support
def append_args_converter(value: ty.Any) -> list[str]:
    """Convert additional arguments to a list of strings."""
    if isinstance(value, str):
        return shlex.split(value)
    if not isinstance(value, ty.Sequence):
        return [value]
    return list(value)


@attrs.define(kw_only=True, auto_attribs=False, eq=False, repr=False)
class ShellTask(base.Task[ShellOutputsType]):

    _executor_name = "executable"

    BASE_ATTRS = ("append_args",)

    EXECUTABLE_HELP = (
        "the first part of the command, can be a string, "
        "e.g. 'ls', or a list, e.g. ['ls', '-l', 'dirname']"
    )

    append_args: list[str | File] = field.arg(
        name="append_args",
        default=attrs.Factory(list),
        converter=append_args_converter,
        type=list[str | File],
        sep=" ",
        help="Additional free-form arguments to append to the end of the command.",
    )

    RESERVED_FIELD_NAMES = base.Task.RESERVED_FIELD_NAMES + ("cmdline",)

    def _run(self, job: "Job[ShellTask]", rerun: bool = True) -> None:
        """Run the shell command."""
        job.return_values = job.environment.execute(job)

    @property
    def cmdline(self) -> str:
        """The equivalent command line that would be submitted if the job were run on
        the current working directory."""
        # Skip the executable, which can be a multi-part command, e.g. 'docker run'.
        values = attrs_values(self)
        values.update(template_update(self, cache_dir=Path.cwd()))
        cmd_args = self._command_args(values=values)
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

    def _command_args(self, values: dict[str, ty.Any]) -> list[str]:
        """Get command line arguments"""
        self._check_resolved()
        self._check_rules()
        # Drop none/empty values and optional path fields that are set to false
        values = copy(values)  # Create a copy so we can drop items from the dictionary
        for fld in get_fields(self):
            fld_value = values[fld.name]
            if fld_value is None or (is_multi_input(fld.type) and fld_value == []):
                del values[fld.name]
            if is_fileset_or_union(fld.type) and type(fld_value) is bool:
                del values[fld.name]
        # Drop special fields that are added separately
        del values["executable"]
        del values["append_args"]
        # Add executable
        pos_args = [
            self._command_shelltask_executable(fld, self.executable),
        ]  # list for (position, command arg)
        positions_provided = [0]
        fields = {f.name: f for f in get_fields(self)}
        for field_name in values:
            pos_val = self._command_pos_args(
                fld=fields[field_name],
                values=values,
                positions_provided=positions_provided,
            )
            if pos_val:
                pos_args.append(pos_val)
        # Sort command and arguments by position
        cmd_args = position_sort(pos_args)
        # pos_args values are each a list of arguments, so concatenate lists after sorting
        command_args = sum(cmd_args, [])
        # Append additional arguments to the end of the command
        command_args += self.append_args
        return command_args

    def _command_shelltask_executable(
        self, fld: field.arg, value: ty.Any
    ) -> tuple[int, ty.Any]:
        """Returning position and value for executable Task input"""
        pos = 0  # executable should be the first el. of the command
        assert value
        return pos, ensure_list(value, tuple2list=True)

    def _command_shelltask_args(
        self, fld: field.arg, value: ty.Any
    ) -> tuple[int, ty.Any]:
        """Returning position and value for args Task input"""
        pos = -1  # assuming that args is the last el. of the command
        if value is None:
            return None
        else:
            return pos, ensure_list(value, tuple2list=True)

    def _command_pos_args(
        self,
        fld: field.arg,
        values: dict[str, ty.Any],
        positions_provided: list[str],
    ) -> tuple[int, ty.Any]:
        """
        Checking all additional input fields, setting pos to None, if position not set.
        Creating a list with additional parts of the command that comes from
        the specific field.

        Parameters
        ----------
        """
        if fld.argstr is None and fld.formatter is None:
            # assuming that input that has no argstr is not used in the command,
            # or a formatter is not provided too.
            return None
        if fld.position is not None:
            if not isinstance(fld.position, int):
                raise Exception(
                    f"position should be an integer, but {fld.position} given"
                )
            # checking if the position is not already used
            if fld.position in positions_provided:
                raise Exception(
                    f"{fld.name} can't have provided position, {fld.position} is already used"
                )

            positions_provided.append(fld.position)

        value = values[fld.name]

        if fld.readonly and type(value) is not bool and value is not attrs.NOTHING:
            raise Exception(f"{fld.name} is read only, the value can't be provided")
        elif value is None and not fld.readonly and fld.formatter is None:
            return None

        cmd_add = []
        # formatter that creates a custom command argument
        # it can take the value of the field, all inputs, or the value of other fields.
        tp = optional_type(fld.type) if is_optional(fld.type) else fld.type
        if fld.formatter:
            call_args = inspect.getfullargspec(fld.formatter)
            call_args_val = {}
            for argnm in call_args.args:
                if argnm == "field":
                    call_args_val[argnm] = fld
                elif argnm == "inputs":
                    call_args_val[argnm] = values
                else:
                    if argnm in values:
                        call_args_val[argnm] = values[argnm]
                    else:
                        raise AttributeError(
                            f"arguments of the formatter function from {fld.name} "
                            f"has to be in inputs or be field, but {argnm} is used"
                        )
            cmd_el_str = fld.formatter(**call_args_val)
            cmd_el_str = cmd_el_str.strip().replace("  ", " ")
            if cmd_el_str != "":
                cmd_add += split_cmd(cmd_el_str)
        elif tp is bool and "{" not in fld.argstr:
            # if value is simply True the original argstr is used,
            # if False, nothing is added to the command.
            if value is True:
                cmd_add.append(fld.argstr)
        elif is_multi_input(tp) or tp is MultiOutputObj or tp is MultiOutputFile:
            # if the field is MultiInputObj, it is used to create a list of arguments
            for val in value or []:
                split_values = copy(values)
                split_values[fld.name] = val
                cmd_add += self._format_arg(fld, split_values)
        else:
            cmd_add += self._format_arg(fld, values)
        return fld.position, cmd_add

    def _format_arg(self, fld: field.arg, values: dict[str, ty.Any]) -> list[str]:
        """Returning arguments used to specify the command args for a single inputs"""
        value = values[fld.name]
        if (
            fld.argstr.endswith("...")
            and isinstance(value, ty.Iterable)
            and not isinstance(value, (str, bytes))
        ):
            argstr = fld.argstr.replace("...", "")
            # if argstr has a more complex form, with "{input_field}"
            if "{" in argstr and "}" in argstr:
                argstr_formatted_l = []
                for val in value:
                    split_values = copy(values)
                    split_values[fld.name] = val
                    argstr_f = argstr_formatting(argstr, split_values)
                    argstr_formatted_l.append(f" {argstr_f}")
                cmd_el_str = fld.sep.join(argstr_formatted_l)
            else:  # argstr has a simple form, e.g. "-f", or "--f"
                cmd_el_str = fld.sep.join([f" {argstr} {val}" for val in value])
        else:
            # in case there are ... when input is not a list
            argstr = fld.argstr.replace("...", "")
            if isinstance(value, ty.Iterable) and not isinstance(value, (str, bytes)):
                cmd_el_str = fld.sep.join([str(val) for val in value])
                value = cmd_el_str
            # if argstr has a more complex form, with "{input_field}"
            if "{" in argstr and "}" in argstr:
                cmd_el_str = argstr.replace(f"{{{fld.name}}}", str(value))
                cmd_el_str = argstr_formatting(cmd_el_str, values)
            else:  # argstr has a simple form, e.g. "-f", or "--f"
                if value:
                    cmd_el_str = f"{argstr} {value}"
                else:
                    cmd_el_str = ""
        return split_cmd(cmd_el_str)

    def _rule_violations(self) -> list[str]:

        errors = super()._rule_violations()
        # if there is a value that has to be updated (e.g. single value from a list)
        # getting all fields that should be formatted, i.e. {field_name}, ...
        fields = get_fields(self)
        available_template_names = [f.name for f in fields] + ["field", "inputs"]
        for fld in fields:
            if fld.argstr:
                if unrecognised := [
                    f
                    for f in parse_format_string(fld.argstr)
                    if f not in available_template_names
                ]:
                    errors.append(
                        f"Unrecognised field names in the argstr of {fld.name} "
                        f"({fld.argstr}): {unrecognised}"
                    )
            if getattr(fld, "path_template", None):
                if unrecognised := [
                    f
                    for f in fields_in_formatter(fld.path_template)
                    if f not in available_template_names
                ]:
                    errors.append(
                        f"Unrecognised field names in the path_template of {fld.name} "
                        f"({fld.path_template}): {unrecognised}"
                    )

        return errors

    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.adjacent


def split_cmd(cmd: str | None):
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
    if cmd is None:
        return []
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


# Alias ShellTask to Task so we can refer to it by shell.Task
Task = ShellTask
Outputs = ShellOutputs
