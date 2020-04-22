"""Task I/O specifications."""
import attr
from pathlib import Path
import typing as ty


def attr_fields(x):
    return x.__attrs_attrs__


class File:
    """An :obj:`os.pathlike` object, designating a file."""


class Directory:
    """An :obj:`os.pathlike` object, designating a folder."""


@attr.s(auto_attribs=True, kw_only=True)
class SpecInfo:
    """Base data structure for metadata of specifications."""

    name: str
    """A name for the specification."""
    fields: ty.List[ty.Tuple] = attr.ib(factory=list)
    """List of names of fields (inputs or outputs)."""
    bases: ty.Tuple[ty.Type] = attr.ib(factory=tuple)
    """Keeps track of this specification inheritance."""


@attr.s(auto_attribs=True, kw_only=True)
class BaseSpec:
    """The base dataclass specs for all inputs and outputs."""

    def collect_additional_outputs(self, input_spec, inputs, output_dir):
        """Get additional outputs."""
        return {}

    @property
    def hash(self):
        """Compute a basic hash for any given set of fields."""
        from .helpers import hash_value, hash_function

        inp_dict = {}
        for field in attr_fields(self):
            if field.name in ["_graph_checksums", "bindings"] or field.metadata.get(
                "output_file_template"
            ):
                continue
            # removing values that are notset from hash calculation
            if getattr(self, field.name) is attr.NOTHING:
                continue
            inp_dict[field.name] = hash_value(
                value=getattr(self, field.name), tp=field.type, metadata=field.metadata
            )
        inp_hash = hash_function(inp_dict)
        if hasattr(self, "_graph_checksums"):
            return hash_function((inp_hash, self._graph_checksums))
        else:
            return inp_hash

    def retrieve_values(self, wf, state_index=None):
        """Get values contained by this spec."""
        temp_values = {}
        for field in attr_fields(self):
            value = getattr(self, field.name)
            if isinstance(value, LazyField):
                value = value.get_value(wf, state_index=state_index)
                temp_values[field.name] = value
        for field, value in temp_values.items():
            setattr(self, field, value)

    def check_metadata(self):
        """Check contained metadata."""

    def check_fields_input_spec(self):
        """Check input fields."""

    def template_update(self):
        """Update template."""

    def copyfile_input(self, output_dir):
        """Copy the file pointed by a :class:`File` input."""


@attr.s(auto_attribs=True, kw_only=True)
class Runtime:
    """Represent run time metadata."""

    rss_peak_gb: ty.Optional[float] = None
    """Peak in consumption of physical RAM."""
    vms_peak_gb: ty.Optional[float] = None
    """Peak in consumption of virtual memory."""
    cpu_peak_percent: ty.Optional[float] = None
    """Peak in cpu consumption."""


@attr.s(auto_attribs=True, kw_only=True)
class Result:
    """Metadata regarding the outputs of processing."""

    output: ty.Optional[ty.Any] = None
    runtime: ty.Optional[Runtime] = None
    errored: bool = False

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["output"] is not None:
            fields = tuple((el.name, el.type) for el in attr_fields(state["output"]))
            state["output_spec"] = (state["output"].__class__.__name__, fields)
            state["output"] = attr.asdict(state["output"])
        return state

    def __setstate__(self, state):
        if "output_spec" in state:
            spec = list(state["output_spec"])
            del state["output_spec"]
            klass = attr.make_class(
                spec[0], {k: attr.ib(type=v) for k, v in list(spec[1])}
            )
            state["output"] = klass(**state["output"])
        self.__dict__.update(state)


@attr.s(auto_attribs=True, kw_only=True)
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


@attr.s(auto_attribs=True, kw_only=True)
class ShellSpec(BaseSpec):
    """Specification for a process invoked from a shell."""

    executable: ty.Union[str, ty.List[str]] = attr.ib(
        metadata={
            "help_string": "the first part of the command, can be a string, "
            "e.g. 'ls', or a list, e.g. ['ls', '-l', 'dirname']"
        }
    )
    args: ty.Union[str, ty.List[str], None] = attr.ib(
        None,
        metadata={
            "help_string": "the last part of the command, can be a string, "
            "e.g. <file_name>, or a list"
        },
    )

    def retrieve_values(self, wf, state_index=None):
        """Parse output results."""
        temp_values = {}
        for field in attr_fields(self):
            # retrieving values that do not have templates
            if not field.metadata.get("output_file_template"):
                value = getattr(self, field.name)
                if isinstance(value, LazyField):
                    value = value.get_value(wf, state_index=state_index)
                    temp_values[field.name] = value
        for field, value in temp_values.items():
            value = path_to_string(value)
            setattr(self, field, value)

    def check_metadata(self):
        """
        Check the metadata for fields in input_spec and fields.

        Also sets the default values when available and needed.

        """
        supported_keys = {
            "allowed_values",
            "argstr",
            "container_path",
            "copyfile",
            "help_string",
            "mandatory",
            "output_field_name",
            "output_file_template",
            "position",
            "requires",
            "separate_ext",
            "xor",
        }
        # special inputs, don't have to follow rules for standard inputs
        special_input = ["_func", "_graph_checksums"]

        fields = [fld for fld in attr_fields(self) if fld.name not in special_input]
        for fld in fields:
            mdata = fld.metadata
            # checking keys from metadata
            if set(mdata.keys()) - supported_keys:
                raise Exception(
                    f"only these keys are supported {supported_keys}, but "
                    f"{set(mdata.keys()) - supported_keys} provided"
                )
            # checking if the help string is provided (required field)
            if "help_string" not in mdata:
                raise Exception(f"{fld.name} doesn't have help_string field")

            # assuming that fields with output_file_template shouldn't have default
            if not fld.default == attr.NOTHING and mdata.get("output_file_template"):
                raise Exception(
                    "default value should not be set together with output_file_template"
                )
            # not allowing for default if the field is mandatory
            if not fld.default == attr.NOTHING and mdata.get("mandatory"):
                raise Exception(
                    "default value should not be set when the field is mandatory"
                )
            # setting default if value not provided and default is available
            if getattr(self, fld.name) is None:
                if not fld.default == attr.NOTHING:
                    setattr(self, fld.name, fld.default)

    def check_fields_input_spec(self):
        """
        Check fields from input spec based on the medatada.

        e.g., if xor, requires are fulfilled, if value provided when mandatory.

        """
        fields = attr_fields(self)
        names = []
        require_to_check = {}
        for fld in fields:
            mdata = fld.metadata
            # checking if the mandatory field is provided
            if getattr(self, fld.name) is attr.NOTHING:
                if mdata.get("mandatory"):
                    raise Exception(f"{fld.name} is mandatory, but no value provided")
                else:
                    continue
            names.append(fld.name)

            # checking if fields meet the xor and requires are
            if "xor" in mdata:
                if [el for el in mdata["xor"] if el in names]:
                    raise Exception(
                        f"{fld.name} is mutually exclusive with {mdata['xor']}"
                    )

            if "requires" in mdata:
                if [el for el in mdata["requires"] if el not in names]:
                    # will check after adding all fields to names
                    require_to_check[fld.name] = mdata["requires"]

            if fld.type is File:
                self._file_check(fld)

        for nm, required in require_to_check.items():
            required_notfound = [el for el in required if el not in names]
            if required_notfound:
                raise Exception(f"{nm} requires {required_notfound}")

            # TODO: types might be checked here
        self._type_checking()

    def _file_check(self, field):
        file = Path(getattr(self, field.name))
        if not file.exists():
            raise Exception(f"the file from the {field.name} input does not exist")

    def _type_checking(self):
        """Use fld.type to check the types TODO.

        This may be done through attr validators.

        """
        fields = attr_fields(self)
        allowed_keys = ["min_val", "max_val", "range", "enum"]  # noqa
        for fld in fields:
            # TODO
            pass


@attr.s(auto_attribs=True, kw_only=True)
class ShellOutSpec(BaseSpec):
    """Output specification of a generic shell process."""

    return_code: int
    """The process' exit code."""
    stdout: ty.Union[File, str]
    """The process' standard output."""
    stderr: ty.Union[File, str]
    """The process' standard input."""

    def collect_additional_outputs(self, input_spec, inputs, output_dir):
        """Collect additional outputs from shelltask output_spec."""
        additional_out = {}
        for fld in attr_fields(self):
            if fld.name not in ["return_code", "stdout", "stderr"]:
                if fld.type is File:
                    # assuming that field should have either default or metadata, but not both
                    if (
                        fld.default is None or fld.default == attr.NOTHING
                    ) and not fld.metadata:  # TODO: is it right?
                        raise Exception("File has to have default value or metadata")
                    elif not fld.default == attr.NOTHING:
                        additional_out[fld.name] = self._field_defaultvalue(
                            fld, output_dir
                        )
                    elif fld.metadata:
                        additional_out[fld.name] = self._field_metadata(
                            fld, inputs, output_dir
                        )
                else:
                    raise Exception("not implemented")
        return additional_out

    def _field_defaultvalue(self, fld, output_dir):
        """Collect output file if the default value specified."""
        if not isinstance(fld.default, (str, Path)):
            raise Exception(
                f"{fld.name} is a File, so default value "
                f"should be a string or a Path, "
                f"{fld.default} provided"
            )
        default = fld.default
        if isinstance(default, str):
            default = Path(default)

        default = output_dir / default

        if "*" not in default.name:
            if default.exists():
                return default
            else:
                raise Exception(f"file {default} does not exist")
        else:
            all_files = list(Path(default.parent).expanduser().glob(default.name))
            if len(all_files) > 1:
                return all_files
            elif len(all_files) == 1:
                return all_files[0]
            else:
                raise Exception(f"no file matches {default.name}")

    def _field_metadata(self, fld, inputs, output_dir):
        """Collect output file if metadata specified."""
        if "value" in fld.metadata:
            return output_dir / fld.metadata["value"]
        elif "output_file_template" in fld.metadata:
            sfx_tmpl = (output_dir / fld.metadata["output_file_template"]).suffixes
            if sfx_tmpl:
                # removing suffix from input field if template has it's own suffix
                inputs_templ = {
                    k: v.split(".")[0]
                    for k, v in inputs.__dict__.items()
                    if isinstance(v, str)
                }
            else:
                inputs_templ = {
                    k: v for k, v in inputs.__dict__.items() if isinstance(v, str)
                }
            out_path = output_dir / fld.metadata["output_file_template"].format(
                **inputs_templ
            )
            return out_path

        elif "callable" in fld.metadata:
            return fld.metadata["callable"](fld.name, output_dir)
        else:
            raise Exception("not implemented")


@attr.s(auto_attribs=True, kw_only=True)
class ContainerSpec(ShellSpec):
    """Refine the generic command-line specification to container execution."""

    image: ty.Union[File, str] = attr.ib(
        metadata={"help_string": "image", "mandatory": True}
    )
    """The image to be containerized."""
    container: ty.Union[File, str, None] = attr.ib(
        metadata={"help_string": "container"}
    )
    """The container."""
    container_xargs: ty.Optional[ty.List[str]] = attr.ib(
        default=None, metadata={"help_string": "todo"}
    )
    """Execution arguments to run the image."""
    bindings: ty.Optional[
        ty.List[
            ty.Tuple[
                Path,  # local path
                Path,  # container path
                ty.Optional[str],  # mount mode
            ]
        ]
    ] = attr.ib(default=None, metadata={"help_string": "bindings"})
    """Mount points to be bound into the container."""

    def _file_check(self, field):
        file = Path(getattr(self, field.name))
        if field.metadata.get("container_path"):
            # if the path is in a container the input should be treated as a str (hash as a str)
            # field.type = "str"
            # setattr(self, field.name, str(file))
            pass
        # if this is a local path, checking if the path exists
        elif file.exists():
            if self.bindings is None:
                self.bindings = []
            self.bindings.append((file.parent, f"/pydra_inp_{field.name}", "ro"))
        else:
            raise Exception(
                f"the file from {field.name} input does not exist, "
                f"if the file comes from the container, "
                f"use field.metadata['container_path']=True"
            )


@attr.s(auto_attribs=True, kw_only=True)
class DockerSpec(ContainerSpec):
    """Particularize container specifications to the Docker engine."""

    container: str = attr.ib("docker", metadata={"help_string": "container"})


@attr.s(auto_attribs=True, kw_only=True)
class SingularitySpec(ContainerSpec):
    """Particularize container specifications to Singularity."""

    container: str = attr.ib("singularity", metadata={"help_string": "container type"})


class LazyField:
    """Lazy fields implement promises."""

    def __init__(self, node, attr_type):
        """Initialize a lazy field."""
        self.name = node.name
        if attr_type == "input":
            self.fields = [field[0] for field in node.input_spec.fields]
        elif attr_type == "output":
            self.fields = node.output_names
        else:
            raise ValueError("LazyField: Unknown attr_type: {}".format(attr_type))
        self.attr_type = attr_type
        self.field = None

    def __getattr__(self, name):
        if name in self.fields or name == "all_":
            self.field = name
            return self
        if name in dir(self):
            return self.__getattribute__(name)
        raise AttributeError(
            "Task {0} has no {1} attribute {2}".format(self.name, self.attr_type, name)
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["name"] = self.name
        state["fields"] = self.fields
        state["field"] = self.field
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self):
        return "LF('{0}', '{1}')".format(self.name, self.field)

    def get_value(self, wf, state_index=None):
        """Return the value of a lazy field."""
        if self.attr_type == "input":
            return getattr(wf.inputs, self.field)
        elif self.attr_type == "output":
            node = getattr(wf, self.name)
            result = node.result(state_index=state_index)
            if isinstance(result, list):
                if len(result) and isinstance(result[0], list):
                    results_new = []
                    for res_l in result:
                        if self.field == "all_":
                            res_l_new = [attr.asdict(res.output) for res in res_l]
                        else:
                            res_l_new = [
                                getattr(res.output, self.field) for res in res_l
                            ]
                        results_new.append(res_l_new)
                    return results_new
                else:
                    if self.field == "all_":
                        return [attr.asdict(res.output) for res in result]
                    else:
                        return [getattr(res.output, self.field) for res in result]
            else:
                if self.field == "all_":
                    return attr.asdict(result.output)
                else:
                    return getattr(result.output, self.field)


def donothing(*args, **kwargs):
    return None


@attr.s(auto_attribs=True, kw_only=True)
class TaskHook:
    """Callable task hooks."""

    pre_run_task: ty.Callable = donothing
    post_run_task: ty.Callable = donothing
    pre_run: ty.Callable = donothing
    post_run: ty.Callable = donothing

    def __setattr__(cls, attr, val):
        if attr not in ["pre_run_task", "post_run_task", "pre_run", "post_run"]:
            raise AttributeError("Cannot set unknown hook")
        super().__setattr__(attr, val)

    def reset(self):
        for val in ["pre_run_task", "post_run_task", "pre_run", "post_run"]:
            setattr(self, val, donothing)


def path_to_string(value):
    """Convert paths to strings."""
    if isinstance(value, Path):
        value = str(value)
    elif isinstance(value, list) and len(value) and isinstance(value[0], Path):
        value = [str(val) for val in value]
    return value
