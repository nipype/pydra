"""Task I/O specifications."""

import os
from pathlib import Path
import re
import inspect
import typing as ty
from glob import glob
from typing_extensions import Self
import attrs
from fileformats.generic import File
from pydra.engine.audit import AuditFlag
from pydra.utils.typing import TypeParser, MultiOutputObj
from .helpers import attrs_fields, attrs_values, is_lazy, list_fields, ensure_list
from .helpers_file import template_update_single
from pydra.utils.hash import hash_function, Cache
from pydra.design.base import Arg, Out
from pydra.design import shell


def is_set(value: ty.Any) -> bool:
    """Check if a value has been set."""
    return value is not attrs.NOTHING


class OutSpec:
    """Base class for all output specifications"""

    RESERVED_FIELD_NAMES = ("split", "combine")

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
        self._node.split(splitter, overwrite=overwrite, cont_dim=cont_dim, **inputs)
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
        self._node.combine(combiner, overwrite=overwrite)
        return self


OutSpecType = ty.TypeVar("OutputType", bound=OutSpec)


class TaskSpec(ty.Generic[OutSpecType]):
    """Base class for all task specifications"""

    Task: "ty.Type[core.Task]"

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
    ):
        self._check_rules()
        task = self.Task(
            self,
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

            # Collect alternative fields associated with this field.
            alternative_fields = {
                name: getattr(self, name) for name in field.xor if name != field.name
            }
            set_alternatives = {
                n: v for n, v in alternative_fields.items() if is_set(v)
            }

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

            # Collect required fields associated with this field.
            required_fields = {
                name: is_set(getattr(self, name))
                for name in field.requires
                if name != field.name
            }

            # Raise error if any required field is unset.
            if not all(required_fields.values()):
                unset_required_fields = [
                    name for name, is_set_ in required_fields.items() if not is_set_
                ]
                raise AttributeError(f"{field.name} requires {unset_required_fields}")


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
class Result:
    """Metadata regarding the outputs of processing."""

    output: ty.Optional[ty.Any] = None
    runtime: ty.Optional[Runtime] = None
    errored: bool = False

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["output"] is not None:
            fields = tuple((el.name, el.type) for el in attrs_fields(state["output"]))
            state["output_spec"] = (state["output"].__class__.__name__, fields)
            state["output"] = attrs.asdict(state["output"], recurse=False)
        return state

    def __setstate__(self, state):
        if "output_spec" in state:
            spec = list(state["output_spec"])
            del state["output_spec"]
            klass = attrs.make_class(
                spec[0], {k: attrs.field(type=v) for k, v in list(spec[1])}
            )
            state["output"] = klass(**state["output"])
        self.__dict__.update(state)

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


class PythonOutSpec(OutSpec):
    pass


class PythonSpec(TaskSpec):
    pass


class WorkflowOutSpec(OutSpec):
    pass


class WorkflowSpec(TaskSpec):
    pass


@attrs.define(kw_only=True)
class ShellOutSpec(OutSpec):
    """Output specification of a generic shell process."""

    return_code: int
    """The process' exit code."""
    stdout: str
    """The process' standard output."""
    stderr: str
    """The process' standard input."""

    RESERVED_FIELD_NAMES = ("split", "combine", "return_code", "stdout", "stderr")

    @classmethod
    def collect_outputs(
        self,
        inputs: "ShellSpec",
        output_dir: Path,
        return_code: int,
        stdout: str,
        stderr: str,
    ) -> Self:

        outputs = Self(return_code=return_code, stdout=stdout, stderr=stderr)
        fld: shell.out
        for fld in list_fields(self):
            if not TypeParser.is_subclass(
                fld.type,
                (
                    os.PathLike,
                    MultiOutputObj,
                    int,
                    float,
                    bool,
                    str,
                    list,
                ),
            ):
                raise TypeError(
                    f"Support for {fld.type} type, required for '{fld.name}' in {self}, "
                    "has not been implemented in collect_additional_output"
                )
            # Get the corresponding value from the inputs if it exists, which will be
            # passed through to the outputs, to permit manual overrides
            if isinstance(fld, shell.outarg) and is_set(getattr(inputs, fld.name)):
                resolved_value = getattr(inputs, fld.name)
            elif is_set(fld.default):
                resolved_value = self._resolve_default_value(fld, output_dir)
            else:
                if fld.type in [int, float, bool, str, list] and not fld.callable:
                    raise AttributeError(
                        f"{fld.type} has to have a callable in metadata"
                    )
                resolved_value = self._generate_implicit_value(
                    fld, inputs, output_dir, outputs, stdout, stderr
                )
            # Set the resolved value
            setattr(outputs, fld.name, resolved_value)
        return outputs

    def _generated_output_names(
        self, inputs: "ShellSpec", output_dir: Path, stdout: str, stderr: str
    ):
        """Returns a list of all outputs that will be generated by the task.
        Takes into account the task input and the requires list for the output fields.
        TODO: should be in all Output specs?
        """
        # checking the input (if all mandatory fields are provided, etc.)
        inputs._check_rules()
        output_names = ["return_code", "stdout", "stderr"]
        for fld in list_fields(self):
            # assuming that field should have either default or metadata, but not both
            if is_set(fld.default):
                output_names.append(fld.name)
            elif is_set(
                self._generate_implicit_value(fld, inputs, output_dir, stdout, stderr)
            ):
                output_names.append(fld.name)
        return output_names

    def _resolve_default_value(self, fld: shell.out, output_dir: Path) -> ty.Any:
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

    def _generate_implicit_value(
        self,
        fld: shell.out,
        inputs: "ShellSpec",
        output_dir: Path,
        stdout: str,
        stderr: str,
    ) -> ty.Any:
        """Collect output file if metadata specified."""
        if not self._required_fields_set(fld, inputs):
            return attrs.NOTHING
        elif isinstance(fld, shell.outarg) and fld.path_template:
            return template_update_single(
                fld, inputs=inputs, output_dir=output_dir, spec_type="output"
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
                    call_args_val[argnm] = output_dir
                elif argnm == "inputs":
                    call_args_val[argnm] = inputs
                elif argnm == "stdout":
                    call_args_val[argnm] = stdout
                elif argnm == "stderr":
                    call_args_val[argnm] = stderr
                else:
                    try:
                        call_args_val[argnm] = getattr(inputs, argnm)
                    except AttributeError:
                        raise AttributeError(
                            f"arguments of the callable function from {fld.name} "
                            f"has to be in inputs or be field or output_dir, "
                            f"but {argnm} is used"
                        )
            return callable_(**call_args_val)
        else:
            raise Exception(
                f"Metadata for '{fld.name}', does not not contain any of the required fields "
                f'("callable", "output_file_template" or "value"): {fld}.'
            )

    def _required_fields_set(self, fld: shell.out, inputs: "ShellSpec") -> bool:
        """checking if all fields from the requires and template are set in the input
        if requires is a list of list, checking if at least one list has all elements set
        """

        if not fld.requires:
            return True

        # if requires is a list of list it is treated as el[0] OR el[1] OR...
        if all([isinstance(el, list) for el in fld.requires]):
            field_required_OR = fld.requires
        # if requires is a list of tuples/strings - I'm creating a 1-el nested list
        elif all([isinstance(el, (str, tuple)) for el in fld.requires]):
            field_required_OR = [fld.requires]
        else:
            raise Exception(
                f"requires field can be a list of list, or a list "
                f"of strings/tuples, but {fld.metadata['requires']} "
                f"provided for {fld.name}"
            )

        for field_required in field_required_OR:
            # if the output has output_file_template field,
            # adding all input fields from the template to requires
            if isinstance(fld, shell.outarg) and self.path_template:
                # if a template is a function it has to be run first with the inputs as the only arg
                if callable(self.path_template):
                    template = self.path_template(inputs)
                inp_fields = re.findall(r"{(\w+)(?:\:[^\}]+)?}", template)
                field_required += [
                    el[1:-1] for el in inp_fields if el[1:-1] not in field_required
                ]

        # it's a flag, of the field from the list is not in input it will be changed to False
        required_found = True
        for field_required in field_required_OR:
            required_found = True
            # checking if the input fields from requires have set values
            for inp in field_required:
                if isinstance(inp, str):  # name of the input field
                    if not hasattr(inputs, inp):
                        raise Exception(
                            f"{inp} is not a valid input field, can't be used in requires"
                        )
                    elif getattr(inputs, inp) in [attrs.NOTHING, None]:
                        required_found = False
                        break
                elif isinstance(inp, tuple):  # (name, allowed values)
                    inp, allowed_val = inp[0], ensure_list(inp[1])
                    if not hasattr(inputs, inp):
                        raise Exception(
                            f"{inp} is not a valid input field, can't be used in requires"
                        )
                    elif getattr(inputs, inp) not in allowed_val:
                        required_found = False
                        break
                else:
                    raise Exception(
                        f"each element of the requires element should be a string or a tuple, "
                        f"but {inp} is found in {field_required}"
                    )
            # if the specific list from field_required_OR has all elements set, no need to check more
            if required_found:
                break
        return required_found


class ShellSpec(TaskSpec):
    pass


def donothing(*args, **kwargs):
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


from pydra.engine import core  # noqa: E402
