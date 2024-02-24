"""
Implement processing nodes.

.. admonition :: Notes:

    * Environment specs

        1. neurodocker json
        2. singularity file+hash
        3. docker hash
        4. conda env
        5. niceman config
        6. environment variables

    * Monitors/Audit

        1. internal monitor
        2. external monitor
        3. callbacks

    * Resuming

        1. internal tracking
        2. external tracking (DMTCP)

    * Provenance

        1. Local fragments
        2. Remote server

    * Isolation

        1. Working directory
        2. File (copy to local on write)
        3. read only file system

    * `Original implementation
      <https://colab.research.google.com/drive/1RRV1gHbGJs49qQB1q1d5tQEycVRtuhw6>`__

"""

from __future__ import annotations

import platform
import re
import attr
import inspect
import typing as ty
import shlex
from pathlib import Path
import warnings
import cloudpickle as cp
from fileformats.core import FileSet, DataType
from .core import TaskBase, is_lazy
from ..utils.messenger import AuditFlag
from .specs import (
    BaseSpec,
    SpecInfo,
    ShellSpec,
    ShellOutSpec,
    attr_fields,
)
from .helpers import (
    ensure_list,
    position_sort,
    argstr_formatting,
    output_from_inputfields,
    parse_copyfile,
)
from .helpers_file import template_update
from ..utils.typing import TypeParser
from .environments import Native


class FunctionTask(TaskBase):
    """Wrap a Python callable as a task element."""

    def __init__(
        self,
        func: ty.Callable,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Optional[ty.Union[SpecInfo, BaseSpec]] = None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[ty.Union[SpecInfo, BaseSpec]] = None,
        rerun=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        func : :obj:`callable`
            A Python executable function.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        cache_locations : :obj:`list` of :obj:`os.pathlike`
            List of alternative cache locations.
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        cont_dim : :obj:`dict`, or `None`
            Container dimensions for input fields,
            if any of the container should be treated as a container
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.

        """
        if input_spec is None:
            fields = []
            for val in inspect.signature(func).parameters.values():
                if val.default is not inspect.Signature.empty:
                    val_dflt = val.default
                else:
                    val_dflt = attr.NOTHING
                if isinstance(val.annotation, ty.TypeVar):
                    raise NotImplementedError(
                        "Template types are not currently supported in task signatures "
                        f"(found in '{val.name}' field of '{name}' task), "
                        "see https://github.com/nipype/pydra/issues/672"
                    )
                fields.append(
                    (
                        val.name,
                        attr.ib(
                            default=val_dflt,
                            type=val.annotation,
                            metadata={
                                "help_string": f"{val.name} parameter from {func.__name__}"
                            },
                        ),
                    )
                )
            fields.append(("_func", attr.ib(default=cp.dumps(func), type=bytes)))
            input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
        else:
            input_spec.fields.append(
                ("_func", attr.ib(default=cp.dumps(func), type=bytes))
            )
        self.input_spec = input_spec
        if name is None:
            name = func.__name__
        super().__init__(
            name,
            inputs=kwargs,
            cont_dim=cont_dim,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            rerun=rerun,
        )
        if output_spec is None:
            name = "Output"
            fields = [("out", ty.Any)]
            if "return" in func.__annotations__:
                return_info = func.__annotations__["return"]
                # # e.g. python annotation: fun() -> ty.NamedTuple("Output", [("out", float)])
                # # or pydra decorator: @pydra.mark.annotate({"return": ty.NamedTuple(...)})
                #

                if (
                    hasattr(return_info, "__name__")
                    and getattr(return_info, "__annotations__", None)
                    and not issubclass(return_info, DataType)
                ):
                    name = return_info.__name__
                    fields = list(return_info.__annotations__.items())
                # e.g. python annotation: fun() -> {"out": int}
                # or pydra decorator: @pydra.mark.annotate({"return": {"out": int}})
                elif isinstance(return_info, dict):
                    fields = list(return_info.items())
                # e.g. python annotation: fun() -> (int, int)
                # or pydra decorator: @pydra.mark.annotate({"return": (int, int)})
                elif isinstance(return_info, tuple):
                    fields = [(f"out{i}", t) for i, t in enumerate(return_info, 1)]
                # e.g. python annotation: fun() -> int
                # or pydra decorator: @pydra.mark.annotate({"return": int})
                else:
                    fields = [("out", return_info)]
            output_spec = SpecInfo(name=name, fields=fields, bases=(BaseSpec,))

        self.output_spec = output_spec

    def _run_task(self, environment=None):
        inputs = attr.asdict(self.inputs, recurse=False)
        del inputs["_func"]
        self.output_ = None
        output = cp.loads(self.inputs._func)(**inputs)
        output_names = [el[0] for el in self.output_spec.fields]
        if output is None:
            self.output_ = {nm: None for nm in output_names}
        elif len(output_names) == 1:
            # if only one element in the fields, everything should be returned together
            self.output_ = {output_names[0]: output}
        elif isinstance(output, tuple) and len(output_names) == len(output):
            self.output_ = dict(zip(output_names, output))
        elif isinstance(output, dict):
            self.output_ = {key: output.get(key, None) for key in output_names}
        else:
            raise RuntimeError(
                f"expected {len(self.output_spec.fields)} elements, "
                f"but {output} were returned"
            )


class ShellCommandTask(TaskBase):
    """Wrap a shell command as a task element."""

    input_spec = None
    output_spec = None

    def __init__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        cont_dim=None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
        environment=Native(),
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        cont_dim : :obj:`dict`, or `None`
            Container dimensions for input fields,
            if any of the container should be treated as a container
        messenger_args :
            TODO
        messengers :
            TODO
        name : :obj:`str`
            Name of this task.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """

        # using default name for task if no name provided
        if name is None:
            name = "ShellTask_noname"

        # using provided spec, class attribute or setting the default SpecInfo
        self.input_spec = (
            input_spec
            or self.input_spec
            or SpecInfo(name="Inputs", fields=[], bases=(ShellSpec,))
        )
        self.output_spec = (
            output_spec
            or self.output_spec
            or SpecInfo(name="Output", fields=[], bases=(ShellOutSpec,))
        )
        self.output_spec = output_from_inputfields(self.output_spec, self.input_spec)

        for special_inp in ["executable", "args"]:
            if hasattr(self, special_inp):
                if special_inp not in kwargs:
                    kwargs[special_inp] = getattr(self, special_inp)
                elif kwargs[special_inp] != getattr(self, special_inp):
                    warnings.warn(
                        f"you are changing the executable from {getattr(self, special_inp)} "
                        f"to {kwargs[special_inp]}"
                    )

        super().__init__(
            name=name,
            inputs=kwargs,
            cont_dim=cont_dim,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            rerun=rerun,
        )
        self.strip = strip
        self.environment = environment
        self.bindings = {}
        self.inputs_mod_root = {}

    def get_bindings(self, root: str | None = None) -> dict[str, tuple[str, str]]:
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

    def command_args(self, root=None):
        """Get command line arguments"""
        if is_lazy(self.inputs):
            raise Exception("can't return cmdline, self.inputs has LazyFields")
        if self.state:
            raise NotImplementedError

        modified_inputs = template_update(self.inputs, output_dir=self.output_dir)
        for field_name, field_value in modified_inputs.items():
            setattr(self.inputs, field_name, field_value)

        pos_args = []  # list for (position, command arg)
        self._positions_provided = []
        for field in attr_fields(self.inputs):
            name, meta = field.name, field.metadata
            if (
                getattr(self.inputs, name) is attr.NOTHING
                and not meta.get("readonly")
                and not meta.get("formatter")
            ):
                continue
            if name == "executable":
                pos_args.append(self._command_shelltask_executable(field))
            elif name == "args":
                pos_val = self._command_shelltask_args(field)
                if pos_val:
                    pos_args.append(pos_val)
            else:
                if name in modified_inputs:
                    pos_val = self._command_pos_args(field, root=root)
                else:
                    pos_val = self._command_pos_args(field)
                if pos_val:
                    pos_args.append(pos_val)

        # Sort command and arguments by position
        cmd_args = position_sort(pos_args)
        # pos_args values are each a list of arguments, so concatenate lists after sorting
        return sum(cmd_args, [])

    def _field_value(self, field, check_file=False):
        """
        Checking value of the specific field, if value is not set, None is returned.
        check_file has no effect, but subclasses can use it to validate or modify
        filenames.
        """
        value = getattr(self.inputs, field.name)
        if value == attr.NOTHING:
            value = None
        return value

    def _command_shelltask_executable(self, field):
        """Returning position and value for executable ShellTask input"""
        pos = 0  # executable should be the first el. of the command
        value = self._field_value(field)
        if value is None:
            raise ValueError("executable has to be set")
        return pos, ensure_list(value, tuple2list=True)

    def _command_shelltask_args(self, field):
        """Returning position and value for args ShellTask input"""
        pos = -1  # assuming that args is the last el. of the command
        value = self._field_value(field, check_file=True)
        if value is None:
            return None
        else:
            return pos, ensure_list(value, tuple2list=True)

    def _command_pos_args(self, field, root=None):
        """
        Checking all additional input fields, setting pos to None, if position not set.
        Creating a list with additional parts of the command that comes from
        the specific field.
        """
        argstr = field.metadata.get("argstr", None)
        formatter = field.metadata.get("formatter", None)
        if argstr is None and formatter is None:
            # assuming that input that has no argstr is not used in the command,
            # or a formatter is not provided too.
            return None
        pos = field.metadata.get("position", None)
        if pos is not None:
            if not isinstance(pos, int):
                raise Exception(f"position should be an integer, but {pos} given")
            # checking if the position is not already used
            if pos in self._positions_provided:
                raise Exception(
                    f"{field.name} can't have provided position, {pos} is already used"
                )

            self._positions_provided.append(pos)

            # Shift non-negatives up to allow executable to be 0
            # Shift negatives down to allow args to be -1
            pos += 1 if pos >= 0 else -1

        value = self._field_value(field, check_file=True)

        if value:
            if field.name in self.inputs_mod_root:
                value = self.inputs_mod_root[field.name]
            elif root:  # values from templates
                value = value.replace(str(self.output_dir), f"{root}{self.output_dir}")

        if field.metadata.get("readonly", False) and value is not None:
            raise Exception(f"{field.name} is read only, the value can't be provided")
        elif (
            value is None
            and not field.metadata.get("readonly", False)
            and formatter is None
        ):
            return None

        inputs_dict = attr.asdict(self.inputs, recurse=False)

        cmd_add = []
        # formatter that creates a custom command argument
        # it can take the value of the field, all inputs, or the value of other fields.
        if "formatter" in field.metadata:
            call_args = inspect.getfullargspec(field.metadata["formatter"])
            call_args_val = {}
            for argnm in call_args.args:
                if argnm == "field":
                    call_args_val[argnm] = value
                elif argnm == "inputs":
                    call_args_val[argnm] = inputs_dict
                else:
                    if argnm in inputs_dict:
                        call_args_val[argnm] = inputs_dict[argnm]
                    else:
                        raise AttributeError(
                            f"arguments of the formatter function from {field.name} "
                            f"has to be in inputs or be field or output_dir, "
                            f"but {argnm} is used"
                        )
            cmd_el_str = field.metadata["formatter"](**call_args_val)
            cmd_el_str = cmd_el_str.strip().replace("  ", " ")
            if cmd_el_str != "":
                cmd_add += split_cmd(cmd_el_str)
        elif field.type is bool:
            # if value is simply True the original argstr is used,
            # if False, nothing is added to the command.
            if value is True:
                cmd_add.append(argstr)
        else:
            sep = field.metadata.get("sep", " ")
            if (
                argstr.endswith("...")
                and isinstance(value, ty.Iterable)
                and not isinstance(value, (str, bytes))
            ):
                argstr = argstr.replace("...", "")
                # if argstr has a more complex form, with "{input_field}"
                if "{" in argstr and "}" in argstr:
                    argstr_formatted_l = []
                    for val in value:
                        argstr_f = argstr_formatting(
                            argstr, self.inputs, value_updates={field.name: val}
                        )
                        argstr_formatted_l.append(f" {argstr_f}")
                    cmd_el_str = sep.join(argstr_formatted_l)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    cmd_el_str = sep.join([f" {argstr} {val}" for val in value])
            else:
                # in case there are ... when input is not a list
                argstr = argstr.replace("...", "")
                if isinstance(value, ty.Iterable) and not isinstance(
                    value, (str, bytes)
                ):
                    cmd_el_str = sep.join([str(val) for val in value])
                    value = cmd_el_str
                # if argstr has a more complex form, with "{input_field}"
                if "{" in argstr and "}" in argstr:
                    cmd_el_str = argstr.replace(f"{{{field.name}}}", str(value))
                    cmd_el_str = argstr_formatting(cmd_el_str, self.inputs)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    if value:
                        cmd_el_str = f"{argstr} {value}"
                    else:
                        cmd_el_str = ""
            if cmd_el_str:
                cmd_add += split_cmd(cmd_el_str)
        return pos, cmd_add

    @property
    def cmdline(self):
        """Get the actual command line that will be submitted
        Returns a list if the task has a state.
        """
        if is_lazy(self.inputs):
            raise Exception("can't return cmdline, self.inputs has LazyFields")
        # checking the inputs fields before returning the command line
        self.inputs.check_fields_input_spec()
        if self.state:
            raise NotImplementedError
        # Skip the executable, which can be a multi-part command, e.g. 'docker run'.
        cmdline = self.command_args()[0]
        for arg in self.command_args()[1:]:
            # If there are spaces in the arg, and it is not enclosed by matching
            # quotes, add quotes to escape the space. Not sure if this should
            # be expanded to include other special characters apart from spaces
            if " " in arg:
                cmdline += " '" + arg + "'"
            else:
                cmdline += " " + arg
        return cmdline

    def _run_task(self, environment=None):
        if environment is None:
            environment = self.environment
        self.output_ = environment.execute(self)

    def _prepare_bindings(self, root: str):
        """Prepare input files to be passed to the task

        This updates the ``bindings`` attribute of the current task to make files available
        in an ``Environment``-defined ``root``.
        """
        for fld in attr_fields(self.inputs):
            if TypeParser.contains_type(FileSet, fld.type):
                fileset = getattr(self.inputs, fld.name)
                copy = parse_copyfile(fld)[0] == FileSet.CopyMode.copy

                host_path, env_path = fileset.parent, Path(f"{root}{fileset.parent}")

                # Default to mounting paths as read-only, but respect existing modes
                old_mode = self.bindings.get(host_path, ("", "ro"))[1]
                self.bindings[host_path] = (env_path, "rw" if copy else old_mode)

                # Provide in-container paths without type-checking
                self.inputs_mod_root[fld.name] = tuple(
                    env_path / rel for rel in fileset.relative_fspaths
                )

    DEFAULT_COPY_COLLATION = FileSet.CopyCollation.adjacent


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
