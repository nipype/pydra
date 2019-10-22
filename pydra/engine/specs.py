import dataclasses as dc
from pathlib import Path
import os
import typing as ty


class File(Path):
    pass


class Directory(Path):
    pass


@dc.dataclass
class SpecInfo:
    name: str
    fields: ty.List[ty.Tuple] = dc.field(default_factory=list)
    bases: ty.Tuple[dc.dataclass] = dc.field(default_factory=tuple)


@dc.dataclass(order=True)
class BaseSpec:
    """The base dataclass specs for all inputs and outputs"""

    def collect_additional_outputs(self, input_spec, inputs, output_dir):
        return {}

    def check_input_spec(self, update_template=False):
        pass  # TODO

    @property
    def hash(self):
        """Compute a basic hash for any given set of fields"""
        from .helpers import hash_function, hash_file

        inp_dict = {
            field.name: hash_file(getattr(self, field.name))
            if field.type == File
            else getattr(self, field.name)
            for field in dc.fields(self)
            if field.name not in ["_graph_checksums"]
        }
        inp_hash = hash_function(inp_dict)
        if hasattr(self, "_graph_checksums"):
            return hash_function((inp_hash, self._graph_checksums))
        else:
            return inp_hash

    def retrieve_values(self, wf, state_index=None):
        temp_values = {}
        for field in dc.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, LazyField):
                value = value.get_value(wf, state_index=state_index)
                temp_values[field.name] = value
        for field, value in temp_values.items():
            setattr(self, field, value)

    def check_input_spec(self, update_template=False):
        pass  # TODO


@dc.dataclass
class Runtime:
    rss_peak_gb: ty.Optional[float] = None
    vms_peak_gb: ty.Optional[float] = None
    cpu_peak_percent: ty.Optional[float] = None


@dc.dataclass
class Result:
    output: ty.Optional[ty.Any] = None
    runtime: ty.Optional[Runtime] = None
    errored: bool = False

    def __getstate__(self):
        state = self.__dict__.copy()
        if state["output"] is not None:
            fields = tuple((el.name, el.type) for el in dc.fields(state["output"]))
            state["output_spec"] = (state["output"].__class__.__name__, fields)
            state["output"] = dc.asdict(state["output"])
        return state

    def __setstate__(self, state):
        if "output_spec" in state:
            spec = list(state["output_spec"])
            del state["output_spec"]
            klass = dc.make_dataclass(spec[0], list(spec[1]))
            state["output"] = klass(**state["output"])
        self.__dict__.update(state)


@dc.dataclass
class RuntimeSpec:
    outdir: ty.Optional[str] = None
    container: ty.Optional[str] = "shell"
    network: bool = False
    """
    from CWL:
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


@dc.dataclass
class ShellSpec(BaseSpec):
    executable: ty.Union[str, ty.List[str]] = dc.field(
        metadata={
            "help_string": "the first part of the command, can be a string, "
            "e.g. 'ls', or a list, e.g. ['ls', '-l', 'dirname']"
        }
    )
    args: ty.Union[str, ty.List[str]] = dc.field(
        metadata={
            "help_string": "the last part of the command, can be a string, "
            "e.g. <file_name>, or a list"
        }
    )

    def check_input_spec(self, update_template=False):
        supported_keys = [
            "mandatory",
            "xor",
            "requires",
            "default_value",
            "position",
            "help_string",
            "output_file_template",
            "output_field_name",
            "separate_ext",
            "argstr",
            "allowed_values",
        ]

        fields = dc.fields(self)
        names = []
        require_to_check = {}
        for fld in fields:
            mdata = fld.metadata
            if (
                not isinstance(fld.default, dc._MISSING_TYPE)
                and mdata.get("default_value")
                and mdata.get("default_value") != fld.default
            ):
                raise Exception(
                    "field.default and metadata[default_value] are both set and differ"
                )
            if (
                not isinstance(fld.default, dc._MISSING_TYPE)
                or mdata.get("default_value")
            ) and mdata.get("output_file_template"):
                raise Exception(
                    "default value should not be set together with output_file_template"
                )
            if (
                not isinstance(fld.default, dc._MISSING_TYPE)
                or mdata.get("default_value")
            ) and mdata.get("mandatory"):
                raise Exception(
                    "default value should not be set when the field is mandatory"
                )

            if dc.asdict(self)[fld.name] is None or update_template:
                if mdata.get("mandatory"):
                    raise Exception(f"{fld.name} is mandatory, but no value provided")
                elif mdata.get("default_value"):
                    setattr(self, fld.name, mdata["default_value"])
                elif mdata.get("output_file_template"):
                    if fld.type is str:
                        # TODO: this has to be done after LF fields are replaced with the final values
                        value = fld.metadata["output_file_template"].format(
                            **self.__dict__
                        )
                    elif fld.type is tuple:  # TODO tu tez trzeba wywalic
                        name, ext = os.path.splitext(
                            fld.metadata["output_file_template"][0].format(
                                **self.__dict__
                            )
                        )
                        value = f"{name}{fld.metadata['output_file_template'][1]}{ext}"
                    else:
                        raise Exception(
                            "output names should be a string or a tuple of two strings"
                        )
                    setattr(self, fld.name, value)
                else:
                    continue
            names.append(fld.name)

            if set(mdata.keys()) - set(supported_keys):
                raise Exception(
                    f"only these keys are supported {supported_keys}, but "
                    f"{set(mdata.keys()) - set(supported_keys)} provided"
                )
            # checking if the help string is provided (required field)
            if "help_string" not in mdata:
                raise Exception(f"{fld.name} doesn't have help_string field")
            # checking if field has set value if mandatory=True
            # or (set value or default) if not mandatory

            if "xor" in mdata:
                if [el for el in mdata["xor"] if el in names]:
                    raise Exception(
                        f"{fld.name} is mutually exclusive with {mdata['xor']}"
                    )

            if "requires" in mdata:
                if [el for el in mdata["requires"] if el not in names]:
                    # will check after adding all fields to names
                    require_to_check[fld.name] = mdata["requires"]

            # TODO: types might be checked here
            self._type_checking(fld)

        for nm, required in require_to_check.items():
            required_notfound = [el for el in required if el not in names]
            if required_notfound:
                raise Exception(f"{nm} requires {required_notfound}")

    def _type_checking(self, field):

        allowed_keys = ["min_val", "max_val", "range", "enum"]
        # TODO
        pass


@dc.dataclass
class ShellOutSpec(BaseSpec):
    return_code: int
    stdout: ty.Union[File, str]
    stderr: ty.Union[File, str]

    def collect_additional_outputs(self, input_spec, inputs, output_dir):
        """collecting additional outputs from shelltask output_spec"""
        additional_out = {}
        for fld in dc.fields(self):
            if fld.name not in ["return_code", "stdout", "stderr"]:
                if fld.type is File:
                    # assuming that field should have either default or metadata, but not both
                    if (
                        not fld.default and isinstance(fld.default, dc._MISSING_TYPE)
                    ) or (
                        not isinstance(fld.default, dc._MISSING_TYPE) and fld.metadata
                    ):
                        raise Exception("File has to have default value or metadata")
                    elif not isinstance(fld.default, dc._MISSING_TYPE):
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
        """collecting output file if the default value specified"""
        if not isinstance(fld.default, (str, Path)):
            raise Exception(
                f"{fld.name} is a File, so default value "
                f"should be string or Path, "
                f"{fld.default} provided"
            )
        if isinstance(fld.default, str):
            fld.default = Path(fld.default)

        fld.default = output_dir / fld.default

        if "*" not in fld.default.name:
            if fld.default.exists():
                return fld.default
            else:
                raise Exception(f"file {fld.default.name} does not exist")
        else:
            all_files = list(
                Path(fld.default.parent).expanduser().glob(fld.default.name)
            )
            if len(all_files) > 1:
                return all_files
            elif len(all_files) == 1:
                return all_files[0]
            else:
                raise Exception(f"no file matches {fld.default.name}")

    def _field_metadata(self, fld, inputs, output_dir):
        """collecting output file if metadata specified"""
        if "value" in fld.metadata:
            return output_dir / fld.metadata["value"]
        elif "output_file_template" in fld.metadata:
            return output_dir / fld.metadata["output_file_template"].format(
                **inputs.__dict__
            )
        elif "callable" in fld.metadata:
            return fld.metadata["callable"](fld.name, output_dir)
        else:
            raise Exception("not implemented")


@dc.dataclass
class ContainerSpec(ShellSpec):
    image: ty.Union[File, str] = dc.field(metadata={"help_string": "image"})
    container: ty.Union[File, str, None] = dc.field(
        metadata={"help_string": "container"}
    )
    container_xargs: ty.Optional[ty.List[str]] = dc.field(
        default=None, metadata={"help_string": "todo"}
    )
    bindings: ty.Optional[
        ty.List[
            ty.Tuple[
                Path,  # local path
                Path,  # container path
                ty.Optional[str],  # mount mode
            ]
        ]
    ] = dc.field(default=None, metadata={"help_string": "bindings"})


@dc.dataclass
class DockerSpec(ContainerSpec):
    container: str = dc.field(default="docker", metadata={"help_string": "container"})


@dc.dataclass
class SingularitySpec(ContainerSpec):
    container: str = "singularity"


class LazyField:
    def __init__(self, node, attr_type):
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
        if self.attr_type == "input":
            return getattr(wf.inputs, self.field)
        elif self.attr_type == "output":
            node = getattr(wf, self.name)
            result = node.result(state_index=state_index)
            if isinstance(result, list):
                if isinstance(result[0], list):
                    results_new = []
                    for res_l in result:
                        if self.field == "all_":
                            res_l_new = [dc.asdict(res.output) for res in res_l]
                        else:
                            res_l_new = [
                                getattr(res.output, self.field) for res in res_l
                            ]
                        results_new.append(res_l_new)
                    return results_new
                else:
                    if self.field == "all_":
                        return [dc.asdict(res.output) for res in result]
                    else:
                        return [getattr(res.output, self.field) for res in result]
            else:
                if self.field == "all_":
                    return dc.asdict(result.output)
                else:
                    return getattr(result.output, self.field)


@dc.dataclass
class TaskHook:
    """Callable task hooks"""

    def none(*args, **kwargs):
        return None

    pre_run_task: ty.Callable = none
    post_run_task: ty.Callable = none
    pre_run: ty.Callable = none
    post_run: ty.Callable = none

    def __setattr__(cls, attr, val):
        if not hasattr(cls, attr):
            raise AttributeError("Cannot set unknown hook")
        super().__setattr__(attr, val)

    def reset(self):
        self.__dict__ = TaskHook().__dict__
