"""Task I/O specifications."""
import attr
from pathlib import Path
import typing as ty
import inspect
import re

from .helpers_file import template_update_single


def attr_fields(x):
    return x.__attrs_attrs__


class File:
    """An :obj:`os.pathlike` object, designating a file."""


class Directory:
    """An :obj:`os.pathlike` object, designating a folder."""


class MultiInputObj:
    """A ty.List[ty.Any] object, converter changes a single values to a list"""

    @classmethod
    def converter(cls, value):
        from .helpers import ensure_list

        return ensure_list(value)


class MultiOutputObj:
    """A ty.List[ty.Any] object, converter changes an 1-el list to the single value"""

    @classmethod
    def converter(cls, value):
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        else:
            return value


@attr.s(auto_attribs=True, kw_only=True)
class SpecInfo:
    """Base data structure for metadata of specifications."""

    name: str
    """A name for the specification."""
    fields: ty.List[ty.Tuple] = attr.ib(factory=list)
    """List of names of fields (can be inputs or outputs)."""
    bases: ty.Tuple[ty.Type] = attr.ib(factory=tuple)
    """Keeps track of specification inheritance.
       Should be a tuple containing at least one BaseSpec """


@attr.s(auto_attribs=True, kw_only=True)
class BaseSpec:
    """The base dataclass specs for all inputs and outputs."""

    def __setattr__(self, name, value):
        """changing settatr, so the converter and validator is run
        if input is set after __init__
        """
        if inspect.stack()[1][3] == "__init__":  # or name.startswith("_"):
            super().__setattr__(name, value)
        else:
            tp = attr.fields_dict(self.__class__)[name].type
            # if the type has a converter, e.g., MultiInputObj
            if hasattr(tp, "converter"):
                value = tp.converter(value)
            super().__setattr__(name, value)
            # validate all fields that have set a validator
            attr.validate(self)

    def collect_additional_outputs(self, inputs, output_dir):
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
                    raise AttributeError(
                        f"{fld.name} is mandatory, but no value provided"
                    )
                else:
                    continue
            names.append(fld.name)

            # checking if fields meet the xor and requires are
            if "xor" in mdata:
                if [el for el in mdata["xor"] if (el in names and el != fld.name)]:
                    raise AttributeError(
                        f"{fld.name} is mutually exclusive with {mdata['xor']}"
                    )

            if "requires" in mdata:
                if [el for el in mdata["requires"] if el not in names]:
                    # will check after adding all fields to names
                    require_to_check[fld.name] = mdata["requires"]

            if fld.type is File:
                self._file_check_n_bindings(fld)

        for nm, required in require_to_check.items():
            required_notfound = [el for el in required if el not in names]
            if required_notfound:
                raise AttributeError(f"{nm} requires {required_notfound}")

    def _file_check_n_bindings(self, field):
        """for tasks without container, this is simple check if the file exists"""
        file = Path(getattr(self, field.name))
        if not file.exists():
            raise AttributeError(f"the file from the {field.name} input does not exist")

    def check_metadata(self):
        """Check contained metadata."""

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

    def get_output_field(self, field_name):
        """Used in get_values in Workflow

        Parameters
        ----------
        field_name : `str`
            Name of field in LazyField object
        """
        if field_name == "all_":
            return attr.asdict(self.output)
        else:
            return getattr(self.output, field_name)


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
class FunctionSpec(BaseSpec):
    """Specification for a process invoked from a shell."""

    def check_metadata(self):
        """
        Check the metadata for fields in input_spec and fields.

        Also sets the default values when available and needed.

        """
        supported_keys = {
            "allowed_values",
            "copyfile",
            "help_string",
            "mandatory",
            # "readonly", #likely not needed
            # "output_field_name", #likely not needed
            # "output_file_template", #likely not needed
            "requires",
            "keep_extension",
            "xor",
            "sep",
        }
        # special inputs, don't have to follow rules for standard inputs
        special_input = ["_func", "_graph_checksums"]

        fields = [fld for fld in attr_fields(self) if fld.name not in special_input]
        for fld in fields:
            mdata = fld.metadata
            # checking keys from metadata
            if set(mdata.keys()) - supported_keys:
                raise AttributeError(
                    f"only these keys are supported {supported_keys}, but "
                    f"{set(mdata.keys()) - supported_keys} provided"
                )
            # checking if the help string is provided (required field)
            if "help_string" not in mdata:
                raise AttributeError(f"{fld.name} doesn't have help_string field")
            # not allowing for default if the field is mandatory
            if not fld.default == attr.NOTHING and mdata.get("mandatory"):
                raise AttributeError(
                    "default value should not be set when the field is mandatory"
                )
            # setting default if value not provided and default is available
            if getattr(self, fld.name) is None:
                if not fld.default == attr.NOTHING:
                    setattr(self, fld.name, fld.default)


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
            "readonly",
            "output_field_name",
            "output_file_template",
            "position",
            "requires",
            "keep_extension",
            "xor",
            "sep",
        }
        # special inputs, don't have to follow rules for standard inputs
        special_input = ["_func", "_graph_checksums"]

        fields = [fld for fld in attr_fields(self) if fld.name not in special_input]
        for fld in fields:
            mdata = fld.metadata
            # checking keys from metadata
            if set(mdata.keys()) - supported_keys:
                raise AttributeError(
                    f"only these keys are supported {supported_keys}, but "
                    f"{set(mdata.keys()) - supported_keys} provided"
                )
            # checking if the help string is provided (required field)
            if "help_string" not in mdata:
                raise AttributeError(f"{fld.name} doesn't have help_string field")
            # assuming that fields with output_file_template shouldn't have default
            if fld.default not in [attr.NOTHING, True, False] and mdata.get(
                "output_file_template"
            ):
                raise AttributeError(
                    "default value should not be set together with output_file_template"
                )
            # not allowing for default if the field is mandatory
            if not fld.default == attr.NOTHING and mdata.get("mandatory"):
                raise AttributeError(
                    "default value should not be set when the field is mandatory"
                )
            # setting default if value not provided and default is available
            if getattr(self, fld.name) is None:
                if not fld.default == attr.NOTHING:
                    setattr(self, fld.name, fld.default)


@attr.s(auto_attribs=True, kw_only=True)
class ShellOutSpec:
    """Output specification of a generic shell process."""

    return_code: int
    """The process' exit code."""
    stdout: ty.Union[File, str]
    """The process' standard output."""
    stderr: ty.Union[File, str]
    """The process' standard input."""

    def collect_additional_outputs(self, inputs, output_dir):
        """Collect additional outputs from shelltask output_spec."""
        additional_out = {}
        for fld in attr_fields(self):
            if fld.name not in ["return_code", "stdout", "stderr"]:
                if fld.type is File:
                    # assuming that field should have either default or metadata, but not both
                    if (
                        fld.default is None or fld.default == attr.NOTHING
                    ) and not fld.metadata:  # TODO: is it right?
                        raise AttributeError(
                            "File has to have default value or metadata"
                        )
                    elif fld.default != attr.NOTHING:
                        additional_out[fld.name] = self._field_defaultvalue(
                            fld, output_dir
                        )
                    elif fld.metadata:
                        additional_out[fld.name] = self._field_metadata(
                            fld, inputs, output_dir
                        )
                else:
                    raise Exception("not implemented (collect_additional_output)")
        return additional_out

    def generated_output_names(self, inputs, output_dir):
        """ Returns a list of all outputs that will be generated by the task.
            Takes into account the task input and the requires list for the output fields.
            TODO: should be in all Output specs?
        """
        # checking the input (if all mandatory fields are provided, etc.)
        inputs.check_fields_input_spec()
        output_names = ["return_code", "stdout", "stderr"]
        for fld in attr_fields(self):
            if fld.name not in ["return_code", "stdout", "stderr"]:
                if fld.type is File:
                    # assuming that field should have either default or metadata, but not both
                    if (
                        fld.default is None or fld.default == attr.NOTHING
                    ) and not fld.metadata:  # TODO: is it right?
                        raise AttributeError(
                            "File has to have default value or metadata"
                        )
                    elif fld.default != attr.NOTHING:
                        output_names.append(fld.name)
                    elif (
                        fld.metadata
                        and self._field_metadata(fld, inputs, output_dir)
                        != attr.NOTHING
                    ):
                        output_names.append(fld.name)
                else:
                    raise Exception("not implemented (collect_additional_output)")
        return output_names

    def _field_defaultvalue(self, fld, output_dir):
        """Collect output file if the default value specified."""
        if not isinstance(fld.default, (str, Path)):
            raise AttributeError(
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
                raise AttributeError(f"file {default} does not exist")
        else:
            all_files = list(Path(default.parent).expanduser().glob(default.name))
            if len(all_files) > 1:
                return all_files
            elif len(all_files) == 1:
                return all_files[0]
            else:
                raise AttributeError(f"no file matches {default.name}")

    def _field_metadata(self, fld, inputs, output_dir):
        """Collect output file if metadata specified."""
        if self._check_requires(fld, inputs) is False:
            return attr.NOTHING

        if "value" in fld.metadata:
            return output_dir / fld.metadata["value"]
        # this block is only run if "output_file_template" is provided in output_spec
        # if the field is set in input_spec with output_file_template,
        # than the field already should have value
        elif "output_file_template" in fld.metadata:
            inputs_templ = attr.asdict(inputs)
            value = template_update_single(
                fld, inputs_templ, output_dir=output_dir, spec_type="output"
            )
            return Path(value)
        elif "callable" in fld.metadata:
            return fld.metadata["callable"](fld.name, output_dir)
        else:
            raise Exception("(_field_metadata) is not a current valid metadata key.")

    def _check_requires(self, fld, inputs):
        """ checking if all fields from the requires and template are set in the input
            if requires is a list of list, checking if at least one list has all elements set
        """
        from .helpers import ensure_list

        if "requires" in fld.metadata:
            # if requires is a list of list it is treated as el[0] OR el[1] OR...
            if all([isinstance(el, list) for el in fld.metadata["requires"]]):
                field_required_OR = fld.metadata["requires"]
            # if requires is a list of tuples/strings - I'm creating a 1-el nested list
            elif all([isinstance(el, (str, tuple)) for el in fld.metadata["requires"]]):
                field_required_OR = [fld.metadata["requires"]]
            else:
                raise Exception(
                    f"requires field can be a list of list, or a list "
                    f"of strings/tuples, but {fld.metadata['requires']} "
                    f"provided for {fld.name}"
                )
        else:
            field_required_OR = [[]]

        for field_required in field_required_OR:
            # if the output has output_file_template field, adding all input fields from the template to requires
            if "output_file_template" in fld.metadata:
                inp_fields = re.findall("{\w+}", fld.metadata["output_file_template"])
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
                    elif getattr(inputs, inp) in [attr.NOTHING, None]:
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

        if required_found:
            return True
        else:
            return False


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

    def _file_check_n_bindings(self, field):
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
            raise ValueError(f"LazyField: Unknown attr_type: {attr_type}")
        self.attr_type = attr_type
        self.field = None

    def __getattr__(self, name):
        if name in self.fields or name == "all_":
            self.field = name
            return self
        if name in dir(self):
            return self.__getattribute__(name)
        raise AttributeError(
            f"Task {self.name} has no {self.attr_type} attribute {name}"
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
        return f"LF('{self.name}', '{self.field}')"

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
                        res_l_new = []
                        for res in res_l:
                            if res.errored:
                                raise ValueError("Error from get_value")
                            else:
                                res_l_new.append(res.get_output_field(self.field))
                        results_new.append(res_l_new)
                    return results_new
                else:
                    results_new = []
                    for res in result:
                        if res.errored:
                            raise ValueError("Error from get_value")
                        else:
                            results_new.append(res.get_output_field(self.field))
                    return results_new
            else:
                if result.errored:
                    raise ValueError("Error from get_value")
                return result.get_output_field(self.field)


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
