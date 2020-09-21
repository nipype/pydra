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
import attr
import cloudpickle as cp
import inspect
import typing as ty
from pathlib import Path
import warnings

from .core import TaskBase, is_lazy
from ..utils.messenger import AuditFlag
from .specs import (
    File,
    BaseSpec,
    SpecInfo,
    ShellSpec,
    ShellOutSpec,
    ContainerSpec,
    DockerSpec,
    SingularitySpec,
    attr_fields,
)
from .helpers import (
    ensure_list,
    execute,
    position_adjustment,
    argstr_formatting,
    output_from_inputfields,
)
from .helpers_file import template_update, is_local_file


class FunctionTask(TaskBase):
    """Wrap a Python callable as a task element."""

    def __init__(
        self,
        func: ty.Callable,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        cache_locations=None,
        input_spec: ty.Optional[ty.Union[SpecInfo, BaseSpec]] = None,
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
            fields.append(("_func", attr.ib(default=cp.dumps(func), type=str)))
            input_spec = SpecInfo(name="Inputs", fields=fields, bases=(BaseSpec,))
        else:
            input_spec.fields.append(
                ("_func", attr.ib(default=cp.dumps(func), type=str))
            )
        self.input_spec = input_spec
        if name is None:
            name = func.__name__
        super().__init__(
            name,
            inputs=kwargs,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            cache_locations=cache_locations,
            rerun=rerun,
        )
        if output_spec is None:
            if "return" not in func.__annotations__:
                output_spec = SpecInfo(
                    name="Output", fields=[("out", ty.Any)], bases=(BaseSpec,)
                )
            else:
                return_info = func.__annotations__["return"]
                # e.g. python annotation: fun() -> ty.NamedTuple("Output", [("out", float)])
                # or pydra decorator: @pydra.mark.annotate({"return": ty.NamedTuple(...)})
                if hasattr(return_info, "__name__") and hasattr(
                    return_info, "__annotations__"
                ):
                    output_spec = SpecInfo(
                        name=return_info.__name__,
                        fields=list(return_info.__annotations__.items()),
                        bases=(BaseSpec,),
                    )
                # e.g. python annotation: fun() -> {"out": int}
                # or pydra decorator: @pydra.mark.annotate({"return": {"out": int}})
                elif isinstance(return_info, dict):
                    output_spec = SpecInfo(
                        name="Output",
                        fields=list(return_info.items()),
                        bases=(BaseSpec,),
                    )
                # e.g. python annotation: fun() -> (int, int)
                # or pydra decorator: @pydra.mark.annotate({"return": (int, int)})
                elif isinstance(return_info, tuple):
                    output_spec = SpecInfo(
                        name="Output",
                        fields=[
                            ("out{}".format(n + 1), t)
                            for n, t in enumerate(return_info)
                        ],
                        bases=(BaseSpec,),
                    )
                # e.g. python annotation: fun() -> int
                # or pydra decorator: @pydra.mark.annotate({"return": int})
                else:
                    output_spec = SpecInfo(
                        name="Output", fields=[("out", return_info)], bases=(BaseSpec,)
                    )
        self.output_spec = output_spec

    def _run_task(self):
        inputs = attr.asdict(self.inputs)
        del inputs["_func"]
        self.output_ = None
        output = cp.loads(self.inputs._func)(**inputs)
        output_names = [el[0] for el in self.output_spec.fields]
        if output is None:
            self.output_ = {nm: None for nm in output_names}
        else:
            if len(output_names) == 1:
                # if only one element in the fields, everything should be returned together
                self.output_ = {output_names[0]: output}
            else:
                if isinstance(output, tuple) and len(output_names) == len(output):
                    self.output_ = dict(zip(output_names, output))
                else:
                    raise Exception(
                        f"expected {len(self.output_spec.fields)} elements, "
                        f"but {output} were returned"
                    )


class ShellCommandTask(TaskBase):
    """Wrap a shell command as a task element."""

    input_spec = None
    output_spec = None

    def __new__(cls, container_info=None, *args, **kwargs):
        if not container_info:
            return super().__new__(cls)

        if len(container_info) == 3:
            type_cont, image, bind = container_info
        elif len(container_info) == 2:
            type_cont, image, bind = container_info + (None,)
        else:
            raise Exception(
                f"container_info has to have 2 or 3 elements, but {container_info} provided"
            )

        if type_cont == "docker":
            return DockerTask(image=image, bindings=bind, *args, **kwargs)
        elif type_cont == "singularity":
            return SingularityTask(image=image, bindings=bind, *args, **kwargs)
        else:
            raise Exception(
                f"first element of container_info has to be "
                f"docker or singularity, but {container_info[0]} provided"
            )

    def __init__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        name=None,
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
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
                        f"you are changing the executable from {getattr(self, special_inp)} to {kwargs[special_inp]}"
                    )

        super().__init__(
            name=name,
            inputs=kwargs,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            rerun=rerun,
        )
        self.strip = strip

    @property
    def command_args(self):
        """Get command line arguments, returns a list if task has a state"""
        if is_lazy(self.inputs):
            raise Exception("can't return cmdline, self.inputs has LazyFields")
        if self.state:
            command_args_list = []
            self.state.prepare_states(self.inputs)
            for ii, el in enumerate(self.state.states_ind):
                command_args_list.append(self._command_args_single(el, ind=ii))
            return command_args_list
        else:
            return self._command_args_single(self.inputs)

    def _command_args_single(self, state_ind, ind=None):
        """Get command line arguments for a single state"""
        pos_args = []  # list for (position, command arg)
        self._positions_provided = []
        for f in attr_fields(self.inputs):
            # these inputs will eb used in container_args
            if isinstance(self, ContainerTask) and f.name in [
                "container",
                "image",
                "container_xargs",
                "bindings",
            ]:
                continue
            elif f.name == "executable":
                pos_args.append(
                    self._command_shelltask_executable(
                        field=f, state_ind=state_ind, ind=ind
                    )
                )
            elif f.name == "args":
                pos_val = self._command_shelltask_args(
                    field=f, state_ind=state_ind, ind=ind
                )
                if pos_val:
                    pos_args.append(pos_val)
            else:
                pos_val = self._command_pos_args(field=f, state_ind=state_ind, ind=ind)
                if pos_val:
                    pos_args.append(pos_val)

        # sorted elements of the command
        cmd_args = position_adjustment(pos_args)
        return cmd_args

    def _field_value(self, field, state_ind, ind, check_file=False):
        """
        Checking value of the specific field, if value is not set, None is returned.
        If state_ind and ind, taking a specific element of the field.
        If check_file is True, checking if field is a a local file
        and settings bindings if needed.
        """
        name = f"{self.name}.{field.name}"
        if self.state and name in state_ind:
            value = getattr(self.inputs, field.name)[state_ind[name]]
        else:
            value = getattr(self.inputs, field.name)
        if value is attr.NOTHING or value is None:
            return None
        if check_file:
            if is_local_file(field):
                value = str(value)
                # changing path to the cpath (the directory should be mounted)
                if getattr(self, "bind_paths", None):
                    lpath = Path(value)
                    cdir = self.bind_paths(ind=ind)[lpath.parent][0]
                    cpath = cdir.joinpath(lpath.name)
                    value = str(cpath)
        return value

    def _command_shelltask_executable(self, field, state_ind, ind):
        """Returining position and value for executable ShellTask input"""
        pos = 0  # executable should be the first el. of the command
        value = self._field_value(field=field, state_ind=state_ind, ind=ind)
        if value is None:
            raise Exception("executable has to be set")
        return pos, ensure_list(value, tuple2list=True)

    def _command_shelltask_args(self, field, state_ind, ind):
        """Returining position and value for args ShellTask input"""
        pos = -1  # assuming that args is the last el. of the command
        value = self._field_value(
            field=field, state_ind=state_ind, ind=ind, check_file=True
        )
        if value is None:
            return None
        else:
            return pos, ensure_list(value, tuple2list=True)

    def _command_pos_args(self, field, state_ind, ind):
        """
        Checking all additional input fields, setting pos to None, if position not set.
        Creating a list with additional parts of the command that comes from
        the specific field.
        """
        argstr = field.metadata.get("argstr", None)
        if argstr is None:
            # assuming that input that has no arstr is not used in the command
            return None
        pos = field.metadata.get("position", None)
        if pos is None:
            # position will be set at the end
            pass
        elif not isinstance(pos, int):
            raise Exception(f"position should be an integer, but {pos} given")
        elif pos == 0:
            raise Exception(f"position can't be 0")
        elif pos < 0:  # position -1 is for args
            pos = pos - 1
        # checking if the position is not already used
        elif pos in self._positions_provided:
            raise Exception(
                f"{field.name} can't have provided position, {pos} is already used"
            )
        self._positions_provided.append(pos)
        value = self._field_value(
            field=field, state_ind=state_ind, ind=ind, check_file=True
        )
        if field.metadata.get("readonly", False) and value is not None:
            raise Exception(f"{field.name} is read only, the value can't be provided")
        elif value is None and not field.metadata.get("readonly", False):
            return None

        cmd_add = []
        if field.type is bool:
            # if value is simply True the original argstr is used,
            # if False, nothing is added to the command
            if value is True:
                cmd_add.append(argstr)
        else:
            sep = field.metadata.get("sep", " ")
            if argstr.endswith("...") and isinstance(value, list):
                argstr = argstr.replace("...", "")
                # if argstr has a more complex form, with "{input_field}"
                if "{" in argstr and "}" in argstr:
                    argstr_formatted_l = []
                    for val in value:
                        argstr_f = argstr_formatting(
                            argstr, self.inputs, value_updates={field.name: val}
                        )
                        argstr_formatted_l.append(argstr_f)
                    cmd_el_str = sep.join(argstr_formatted_l)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    cmd_el_str = sep.join([f" {argstr} {val}" for val in value])
            else:
                # in case there are ... when input is not a list
                argstr = argstr.replace("...", "")
                if isinstance(value, list):
                    cmd_el_str = sep.join([str(val) for val in value])
                    value = cmd_el_str
                # if argstr has a more complex form, with "{input_field}"
                if "{" in argstr and "}" in argstr:
                    cmd_el_str = argstr_formatting(argstr, self.inputs)
                else:  # argstr has a simple form, e.g. "-f", or "--f"
                    if value:
                        cmd_el_str = f"{argstr} {value}"
                    else:
                        cmd_el_str = ""
            # removing double spacing
            cmd_el_str = cmd_el_str.strip().replace("  ", " ")
            if cmd_el_str:
                cmd_add += cmd_el_str.split(" ")
        return pos, cmd_add

    @property
    def cmdline(self):
        """ Get the actual command line that will be submitted
            Returns a list if the task has a state.
        """
        if is_lazy(self.inputs):
            raise Exception("can't return cmdline, self.inputs has LazyFields")
        # checking the inputs fields before returning the command line
        self.inputs.check_fields_input_spec()
        orig_inputs = attr.asdict(self.inputs)
        modified_inputs = template_update(self.inputs, output_dir=self.output_dir)
        if modified_inputs is not None:
            self.inputs = attr.evolve(self.inputs, **modified_inputs)
        if isinstance(self, ContainerTask):
            if self.state:
                cmdline = []
                for con, com in zip(self.container_args, self.command_args):
                    cmdline.append(" ".join(con + com))
            else:
                cmdline = " ".join(self.container_args + self.command_args)
        else:
            if self.state:
                cmdline = []
                for el in self.command_args:
                    cmdline.append(" ".join(el))
            else:
                cmdline = " ".join(self.command_args)

        self.inputs = attr.evolve(self.inputs, **orig_inputs)
        return cmdline

    def _run_task(self):
        self.output_ = None
        if isinstance(self, ContainerTask):
            args = self.container_args + self.command_args
        else:
            args = self.command_args
        if args:
            # removing empty strings
            args = [str(el) for el in args if el not in ["", " "]]
            keys = ["return_code", "stdout", "stderr"]
            values = execute(args, strip=self.strip)
            self.output_ = dict(zip(keys, values))
            if self.output_["return_code"]:
                if self.output_["stderr"]:
                    raise RuntimeError(self.output_["stderr"])
                else:
                    raise RuntimeError(self.output_["stdout"])


class ContainerTask(ShellCommandTask):
    """Extend shell command task for containerized execution."""

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_cpath="/output_pydra",
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_cpath : :obj:`str`
            Output path within the container filesystem.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if input_spec is None:
            input_spec = SpecInfo(name="Inputs", fields=[], bases=(ContainerSpec,))
        self.output_cpath = Path(output_cpath)
        super().__init__(
            name=name,
            input_spec=input_spec,
            output_spec=output_spec,
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            cache_dir=cache_dir,
            strip=strip,
            rerun=rerun,
            **kwargs,
        )

    def container_check(self, container_type):
        """Get container-specific CLI arguments."""
        if self.inputs.container is None:
            raise AttributeError("Container software is not specified")
        elif self.inputs.container != container_type:
            raise AttributeError(
                f"Container type should be {container_type}, but {self.inputs.container} given"
            )
        if self.inputs.image is attr.NOTHING:
            raise AttributeError("Container image is not specified")

    def bind_paths(self, ind=None):
        """Return bound mount points: ``dict(lpath: (cpath, mode))``."""
        bind_paths = {}
        output_dir_cpath = None
        if self.inputs.bindings is None:
            self.inputs.bindings = []
        if ind is None:
            output_dir = self.output_dir
        else:
            output_dir = self.output_dir[ind]
        for binding in self.inputs.bindings:
            if len(binding) == 3:
                lpath, cpath, mode = binding
            elif len(binding) == 2:
                lpath, cpath, mode = binding + ["rw"]
            else:
                raise Exception(
                    f"binding should have length 2, 3, or 4, it has {len(binding)}"
                )
            if Path(lpath) == output_dir:
                output_dir_cpath = cpath
            if mode is None:
                mode = "rw"  # default
            bind_paths[Path(lpath)] = (Path(cpath), mode)
        # output_dir is added to the bindings if not part of self.inputs.bindings
        if not output_dir_cpath:
            bind_paths[output_dir] = (self.output_cpath, "rw")
        return bind_paths

    def binds(self, opt, ind=None):
        """
        Specify mounts to bind from local filesystems to container and working directory.

        Uses py:meth:`binds_paths`

        """
        bargs = []
        for (key, val) in self.bind_paths(ind).items():
            bargs.extend([opt, f"{key}:{val[0]}:{val[1]}"])
        return bargs


class DockerTask(ContainerTask):
    """Extend shell command task for containerized execution with the Docker Engine."""

    init = False

    def __init__(
        self,
        name,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_cpath="/output_pydra",
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_cpath : :obj:`str`
            Output path within the container filesystem.
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if not self.init:
            if input_spec is None:
                input_spec = SpecInfo(name="Inputs", fields=[], bases=(DockerSpec,))

            super().__init__(
                name=name,
                input_spec=input_spec,
                output_spec=output_spec,
                audit_flags=audit_flags,
                messengers=messengers,
                messenger_args=messenger_args,
                cache_dir=cache_dir,
                strip=strip,
                output_cpath=output_cpath,
                rerun=rerun,
                **kwargs,
            )
            self.inputs.container_xargs = ["--rm"]
            self.init = True

    @property
    def container_args(self):
        """Get container-specific CLI arguments, returns a list if the task has a state"""
        if is_lazy(self.inputs):
            raise Exception("can't return container_args, self.inputs has LazyFields")
        self.container_check("docker")
        if self.state:
            self.state.prepare_states(self.inputs)
            cargs_list = []
            for ii, el in enumerate(self.state.states_ind):
                if f"{self.name}.image" in el:
                    cargs_list.append(
                        self._container_args_single(
                            self.inputs.image[el[f"{self.name}.image"]], ind=ii
                        )
                    )
                else:
                    cargs_list.append(
                        self._container_args_single(self.inputs.image, ind=ii)
                    )
            return cargs_list
        else:
            return self._container_args_single(self.inputs.image)

    def _container_args_single(self, image, ind=None):
        cargs = ["docker", "run"]
        if self.inputs.container_xargs is not None:
            cargs.extend(self.inputs.container_xargs)

        cargs.extend(self.binds("-v", ind))
        cargs.extend(["-w", str(self.output_cpath)])
        cargs.append(image)

        return cargs


class SingularityTask(ContainerTask):
    """Extend shell command task for containerized execution with Singularity."""

    init = False

    def __init__(
        self,
        audit_flags: AuditFlag = AuditFlag.NONE,
        cache_dir=None,
        input_spec: ty.Optional[SpecInfo] = None,
        messenger_args=None,
        messengers=None,
        output_spec: ty.Optional[SpecInfo] = None,
        rerun=False,
        strip=False,
        **kwargs,
    ):
        """
        Initialize this task.

        Parameters
        ----------
        name : :obj:`str`
            Name of this task.
        audit_flags : :obj:`pydra.utils.messenger.AuditFlag`
            Auditing configuration
        cache_dir : :obj:`os.pathlike`
            Cache directory
        input_spec : :obj:`pydra.engine.specs.SpecInfo`
            Specification of inputs.
        messenger_args :
            TODO
        messengers :
            TODO
        output_spec : :obj:`pydra.engine.specs.BaseSpec`
            Specification of inputs.
        strip : :obj:`bool`
            TODO

        """
        if not self.init:
            if input_spec is None:
                input_spec = SpecInfo(
                    name="Inputs", fields=[], bases=(SingularitySpec,)
                )
            super().__init__(
                input_spec=input_spec,
                output_spec=output_spec,
                audit_flags=audit_flags,
                messengers=messengers,
                messenger_args=messenger_args,
                cache_dir=cache_dir,
                strip=strip,
                rerun=rerun,
                **kwargs,
            )
            self.init = True

    @property
    def container_args(self):
        """Get container-specific CLI arguments."""
        if is_lazy(self.inputs):
            raise Exception("can't return container_args, self.inputs has LazyFields")
        self.container_check("singularity")
        if self.state:
            self.state.prepare_states(self.inputs)
            cargs_list = []
            for ii, el in enumerate(self.state.states_ind):
                if f"{self.name}.image" in el:
                    cargs_list.append(
                        self._container_args_single(
                            self.inputs.image[el[f"{self.name}.image"]], ind=ii
                        )
                    )
                else:
                    cargs_list.append(
                        self._container_args_single(self.inputs.image, ind=ii)
                    )
            return cargs_list
        else:
            return self._container_args_single(self.inputs.image)

    def _container_args_single(self, image, ind=None):
        cargs = ["singularity", "exec"]

        if self.inputs.container_xargs is not None:
            cargs.extend(self.inputs.container_xargs)
        cargs.append(image)

        # insert bindings before image
        idx = len(cargs) - 1
        cargs[idx:-1] = self.binds("-B", ind)
        return cargs
