"""Task I/O specifications."""

from pathlib import Path
import typing as ty

# import inspect
# import re
# import os
from pydra.engine.audit import AuditFlag

# from glob import glob
import attrs
from typing_extensions import Self

# from fileformats.core import FileSet
from fileformats.generic import (
    File,
    # Directory,
)
from .helpers import attr_fields

# from .helpers_file import template_update_single
# from pydra.utils.hash import hash_function, Cache

# from pydra.utils.misc import add_exc_note


# @attrs.define(auto_attribs=True, kw_only=True)
# class SpecInfo:
#     """Base data structure for metadata of specifications."""

#     name: str
#     """A name for the specification."""
#     fields: ty.List[ty.Tuple] = attrs.field(factory=list)
#     """List of names of fields (can be inputs or outputs)."""
#     bases: ty.Sequence[ty.Type["BaseSpec"]] = attrs.field(factory=tuple)
#     """Keeps track of specification inheritance.
#        Should be a tuple containing at least one BaseSpec """


# @attrs.define(auto_attribs=True, kw_only=True)
# class BaseSpec:
#     """The base dataclass specs for all inputs and outputs."""

#     def collect_additional_outputs(self, inputs, output_dir, outputs):
#         """Get additional outputs."""
#         return {}

#     @property
#     def hash(self):
#         hsh, self._hashes = self._compute_hashes()
#         return hsh

#     def hash_changes(self):
#         """Detects any changes in the hashed values between the current inputs and the
#         previously calculated values"""
#         _, new_hashes = self._compute_hashes()
#         return [k for k, v in new_hashes.items() if v != self._hashes[k]]

#     def _compute_hashes(self) -> ty.Tuple[bytes, ty.Dict[str, bytes]]:
#         """Compute a basic hash for any given set of fields."""
#         inp_dict = {}
#         for field in attr_fields(
#             self, exclude_names=("_graph_checksums", "bindings", "files_hash")
#         ):
#             if field.metadata.get("output_file_template"):
#                 continue
#             # removing values that are not set from hash calculation
#             if getattr(self, field.name) is attrs.NOTHING:
#                 continue
#             if "container_path" in field.metadata:
#                 continue
#             inp_dict[field.name] = getattr(self, field.name)
#         hash_cache = Cache()
#         field_hashes = {
#             k: hash_function(v, cache=hash_cache) for k, v in inp_dict.items()
#         }
#         if hasattr(self, "_graph_checksums"):
#             field_hashes["_graph_checksums"] = self._graph_checksums
#         return hash_function(sorted(field_hashes.items())), field_hashes

#     def retrieve_values(self, wf, state_index: ty.Optional[int] = None):
#         """Get values contained by this spec."""
#         retrieved_values = {}
#         for field in attr_fields(self):
#             value = getattr(self, field.name)
#             if isinstance(value, LazyField):
#                 retrieved_values[field.name] = value.get_value(
#                     wf, state_index=state_index
#                 )
#         for field, val in retrieved_values.items():
#             setattr(self, field, val)

#     def check_fields_input_spec(self):
#         """
#         Check fields from input spec based on the medatada.

#         e.g., if xor, requires are fulfilled, if value provided when mandatory.

#         """
#         fields = attr_fields(self)

#         for field in fields:
#             field_is_mandatory = bool(field.metadata.get("mandatory"))
#             field_is_unset = getattr(self, field.name) is attrs.NOTHING

#             if field_is_unset and not field_is_mandatory:
#                 continue

#             # Collect alternative fields associated with this field.
#             alternative_fields = {
#                 name: getattr(self, name) is not attrs.NOTHING
#                 for name in field.metadata.get("xor", [])
#                 if name != field.name
#             }
#             alternatives_are_set = any(alternative_fields.values())

#             # Raise error if no field in mandatory alternative group is set.
#             if field_is_unset:
#                 if alternatives_are_set:
#                     continue
#                 message = f"{field.name} is mandatory and unset."
#                 if alternative_fields:
#                     raise AttributeError(
#                         message[:-1]
#                         + f", but no alternative provided by {list(alternative_fields)}."
#                     )
#                 else:
#                     raise AttributeError(message)

#             # Raise error if multiple alternatives are set.
#             elif alternatives_are_set:
#                 set_alternative_fields = [
#                     name for name, is_set in alternative_fields.items() if is_set
#                 ]
#                 raise AttributeError(
#                     f"{field.name} is mutually exclusive with {set_alternative_fields}"
#                 )

#             # Collect required fields associated with this field.
#             required_fields = {
#                 name: getattr(self, name) is not attrs.NOTHING
#                 for name in field.metadata.get("requires", [])
#                 if name != field.name
#             }

#             # Raise error if any required field is unset.
#             if not all(required_fields.values()):
#                 unset_required_fields = [
#                     name for name, is_set in required_fields.items() if not is_set
#                 ]
#                 raise AttributeError(f"{field.name} requires {unset_required_fields}")

#     def check_metadata(self):
#         """Check contained metadata."""

#     def template_update(self):
#         """Update template."""

#     def copyfile_input(self, output_dir):
#         """Copy the file pointed by a :class:`File` input."""


@attrs.define(auto_attribs=True, kw_only=True)
class Runtime:
    """Represent run time metadata."""

    rss_peak_gb: ty.Optional[float] = None
    """Peak in consumption of physical RAM."""
    vms_peak_gb: ty.Optional[float] = None
    """Peak in consumption of virtual memory."""
    cpu_peak_percent: ty.Optional[float] = None
    """Peak in cpu consumption."""


@attrs.define(auto_attribs=True, kw_only=True)
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
            return attrs.asdict(self.output, recurse=False)
        else:
            return getattr(self.output, field_name)


@attrs.define(auto_attribs=True, kw_only=True)
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


# @attrs.define(auto_attribs=True, kw_only=True)
# class FunctionSpec(BaseSpec):
#     """Specification for a process invoked from a shell."""

#     def check_metadata(self):
#         """
#         Check the metadata for fields in input_spec and fields.

#         Also sets the default values when available and needed.

#         """
#         supported_keys = {
#             "allowed_values",
#             "copyfile",
#             "help_string",
#             "mandatory",
#             # "readonly", #likely not needed
#             # "output_field_name", #likely not needed
#             # "output_file_template", #likely not needed
#             "requires",
#             "keep_extension",
#             "xor",
#             "sep",
#         }
#         for fld in attr_fields(self, exclude_names=("_func", "_graph_checksums")):
#             mdata = fld.metadata
#             # checking keys from metadata
#             if set(mdata.keys()) - supported_keys:
#                 raise AttributeError(
#                     f"only these keys are supported {supported_keys}, but "
#                     f"{set(mdata.keys()) - supported_keys} provided"
#                 )
#             # checking if the help string is provided (required field)
#             if "help_string" not in mdata:
#                 raise AttributeError(f"{fld.name} doesn't have help_string field")
#             # not allowing for default if the field is mandatory
#             if not fld.default == attrs.NOTHING and mdata.get("mandatory"):
#                 raise AttributeError(
#                     f"default value ({fld.default!r}) should not be set when the field "
#                     f"('{fld.name}') in {self}) is mandatory"
#                 )
#             # setting default if value not provided and default is available
#             if getattr(self, fld.name) is None:
#                 if not fld.default == attrs.NOTHING:
#                     setattr(self, fld.name, fld.default)


# @attrs.define(auto_attribs=True, kw_only=True)
# class ShellSpec(BaseSpec):
#     """Specification for a process invoked from a shell."""

#     executable: ty.Union[str, ty.List[str]] = attrs.field(
#         metadata={
#             "help_string": "the first part of the command, can be a string, "
#             "e.g. 'ls', or a list, e.g. ['ls', '-l', 'dirname']"
#         }
#     )
#     args: ty.Union[str, ty.List[str], None] = attrs.field(
#         default=None,
#         metadata={
#             "help_string": "the last part of the command, can be a string, "
#             "e.g. <file_name>, or a list"
#         },
#     )

#     def retrieve_values(self, wf, state_index=None):
#         """Parse output results."""
#         temp_values = {}
#         for field in attr_fields(self):
#             # retrieving values that do not have templates
#             if not field.metadata.get("output_file_template"):
#                 value = getattr(self, field.name)
#                 if isinstance(value, LazyField):
#                     temp_values[field.name] = value.get_value(
#                         wf, state_index=state_index
#                     )
#         for field, val in temp_values.items():
#             value = path_to_string(value)
#             setattr(self, field, val)

#     def check_metadata(self):
#         """
#         Check the metadata for fields in input_spec and fields.

#         Also sets the default values when available and needed.

#         """
#         from pydra.utils.typing import TypeParser

#         supported_keys = {
#             "allowed_values",
#             "argstr",
#             "container_path",
#             "copyfile",
#             "help_string",
#             "mandatory",
#             "readonly",
#             "output_field_name",
#             "output_file_template",
#             "position",
#             "requires",
#             "keep_extension",
#             "xor",
#             "sep",
#             "formatter",
#             "_output_type",
#         }

#         for fld in attr_fields(self, exclude_names=("_func", "_graph_checksums")):
#             mdata = fld.metadata
#             # checking keys from metadata
#             if set(mdata.keys()) - supported_keys:
#                 raise AttributeError(
#                     f"only these keys are supported {supported_keys}, but "
#                     f"{set(mdata.keys()) - supported_keys} provided for '{fld.name}' "
#                     f"field in {self}"
#                 )
#             # checking if the help string is provided (required field)
#             if "help_string" not in mdata:
#                 raise AttributeError(
#                     f"{fld.name} doesn't have help_string field in {self}"
#                 )
#             # assuming that fields with output_file_template shouldn't have default
#             if mdata.get("output_file_template"):
#                 if not any(
#                     TypeParser.matches_type(fld.type, t) for t in OUTPUT_TEMPLATE_TYPES
#                 ):
#                     raise TypeError(
#                         f"Type of '{fld.name}' should be one of {OUTPUT_TEMPLATE_TYPES} "
#                         f"(not {fld.type}) because it has a value for output_file_template "
#                         f"({mdata['output_file_template']!r})"
#                     )
#                 if fld.default not in [attrs.NOTHING, True, False]:
#                     raise AttributeError(
#                         f"default value ({fld.default!r}) should not be set together with "
#                         f"output_file_template ({mdata['output_file_template']!r}) for "
#                         f"'{fld.name}' field in {self}"
#                     )
#             # not allowing for default if the field is mandatory
#             if not fld.default == attrs.NOTHING and mdata.get("mandatory"):
#                 raise AttributeError(
#                     f"default value ({fld.default!r}) should not be set when the field "
#                     f"('{fld.name}') in {self}) is mandatory"
#                 )
#             # setting default if value not provided and default is available
#             if getattr(self, fld.name) is None:
#                 if not fld.default == attrs.NOTHING:
#                     setattr(self, fld.name, fld.default)


# @attrs.define(auto_attribs=True, kw_only=True)
# class ShellOutSpec:
#     """Output specification of a generic shell process."""

#     return_code: int
#     """The process' exit code."""
#     stdout: str
#     """The process' standard output."""
#     stderr: str
#     """The process' standard input."""

#     def collect_additional_outputs(self, inputs, output_dir, outputs):
#         from pydra.utils.typing import TypeParser

#         """Collect additional outputs from shelltask output_spec."""
#         additional_out = {}
#         for fld in attr_fields(self, exclude_names=("return_code", "stdout", "stderr")):
#             if not TypeParser.is_subclass(
#                 fld.type,
#                 (
#                     os.PathLike,
#                     MultiOutputObj,
#                     int,
#                     float,
#                     bool,
#                     str,
#                     list,
#                 ),
#             ):
#                 raise TypeError(
#                     f"Support for {fld.type} type, required for '{fld.name}' in {self}, "
#                     "has not been implemented in collect_additional_output"
#                 )
#             # assuming that field should have either default or metadata, but not both
#             input_value = getattr(inputs, fld.name, attrs.NOTHING)
#             if fld.metadata and "callable" in fld.metadata:
#                 fld_out = self._field_metadata(fld, inputs, output_dir, outputs)
#             elif fld.type in [int, float, bool, str, list]:
#                 raise AttributeError(f"{fld.type} has to have a callable in metadata")
#             elif input_value:  # Map input value through to output
#                 fld_out = input_value
#             elif fld.default != attrs.NOTHING:
#                 fld_out = self._field_defaultvalue(fld, output_dir)
#             else:
#                 raise AttributeError("File has to have default value or metadata")
#             if TypeParser.contains_type(FileSet, fld.type):
#                 label = f"output field '{fld.name}' of {self}"
#                 fld_out = TypeParser(fld.type, label=label).coerce(fld_out)
#             additional_out[fld.name] = fld_out
#         return additional_out

#     def generated_output_names(self, inputs, output_dir):
#         """Returns a list of all outputs that will be generated by the task.
#         Takes into account the task input and the requires list for the output fields.
#         TODO: should be in all Output specs?
#         """
#         # checking the input (if all mandatory fields are provided, etc.)
#         inputs.check_fields_input_spec()
#         output_names = ["return_code", "stdout", "stderr"]
#         for fld in attr_fields(self, exclude_names=("return_code", "stdout", "stderr")):
#             if fld.type not in [File, MultiOutputFile, Directory]:
#                 raise Exception("not implemented (collect_additional_output)")
#             # assuming that field should have either default or metadata, but not both
#             if (
#                 fld.default in (None, attrs.NOTHING) and not fld.metadata
#             ):  # TODO: is it right?
#                 raise AttributeError("File has to have default value or metadata")
#             elif fld.default != attrs.NOTHING:
#                 output_names.append(fld.name)
#             elif (
#                 fld.metadata
#                 and self._field_metadata(
#                     fld, inputs, output_dir, outputs=None, check_existance=False
#                 )
#                 != attrs.NOTHING
#             ):
#                 output_names.append(fld.name)
#         return output_names

#     def _field_defaultvalue(self, fld, output_dir):
#         """Collect output file if the default value specified."""
#         if not isinstance(fld.default, (str, Path)):
#             raise AttributeError(
#                 f"{fld.name} is a File, so default value "
#                 f"should be a string or a Path, "
#                 f"{fld.default} provided"
#             )
#         default = fld.default
#         if isinstance(default, str):
#             default = Path(default)

#         default = output_dir / default
#         if "*" not in str(default):
#             if default.exists():
#                 return default
#             else:
#                 raise AttributeError(f"file {default} does not exist")
#         else:
#             all_files = [Path(el) for el in glob(str(default.expanduser()))]
#             if len(all_files) > 1:
#                 return all_files
#             elif len(all_files) == 1:
#                 return all_files[0]
#             else:
#                 raise AttributeError(f"no file matches {default.name}")

#     def _field_metadata(
#         self, fld, inputs, output_dir, outputs=None, check_existance=True
#     ):
#         """Collect output file if metadata specified."""
#         if self._check_requires(fld, inputs) is False:
#             return attrs.NOTHING

#         if "value" in fld.metadata:
#             return output_dir / fld.metadata["value"]
#         # this block is only run if "output_file_template" is provided in output_spec
#         # if the field is set in input_spec with output_file_template,
#         # than the field already should have value
#         elif "output_file_template" in fld.metadata:
#             value = template_update_single(
#                 fld, inputs=inputs, output_dir=output_dir, spec_type="output"
#             )

#             if fld.type is MultiOutputFile and type(value) is list:
#                 # TODO: how to deal with mandatory list outputs
#                 ret = []
#                 for val in value:
#                     val = Path(val)
#                     if check_existance and not val.exists():
#                         ret.append(attrs.NOTHING)
#                     else:
#                         ret.append(val)
#                 return ret
#             else:
#                 val = Path(value)
#                 # checking if the file exists
#                 if check_existance and not val.exists():
#                     # if mandatory raise exception
#                     if "mandatory" in fld.metadata:
#                         if fld.metadata["mandatory"]:
#                             raise Exception(
#                                 f"mandatory output for variable {fld.name} does not exist"
#                             )
#                     return attrs.NOTHING
#                 return val
#         elif "callable" in fld.metadata:
#             callable_ = fld.metadata["callable"]
#             if isinstance(callable_, staticmethod):
#                 # In case callable is defined as a static method,
#                 # retrieve the function wrapped in the descriptor.
#                 callable_ = callable_.__func__
#             call_args = inspect.getfullargspec(callable_)
#             call_args_val = {}
#             for argnm in call_args.args:
#                 if argnm == "field":
#                     call_args_val[argnm] = fld
#                 elif argnm == "output_dir":
#                     call_args_val[argnm] = output_dir
#                 elif argnm == "inputs":
#                     call_args_val[argnm] = inputs
#                 elif argnm == "stdout":
#                     call_args_val[argnm] = outputs["stdout"]
#                 elif argnm == "stderr":
#                     call_args_val[argnm] = outputs["stderr"]
#                 else:
#                     try:
#                         call_args_val[argnm] = getattr(inputs, argnm)
#                     except AttributeError:
#                         raise AttributeError(
#                             f"arguments of the callable function from {fld.name} "
#                             f"has to be in inputs or be field or output_dir, "
#                             f"but {argnm} is used"
#                         )
#             return callable_(**call_args_val)
#         else:
#             raise Exception(
#                 f"Metadata for '{fld.name}', does not not contain any of the required fields "
#                 f'("callable", "output_file_template" or "value"): {fld.metadata}.'
#             )

#     def _check_requires(self, fld, inputs):
#         """checking if all fields from the requires and template are set in the input
#         if requires is a list of list, checking if at least one list has all elements set
#         """
#         from .helpers import ensure_list

#         if "requires" in fld.metadata:
#             # if requires is a list of list it is treated as el[0] OR el[1] OR...
#             required_fields = ensure_list(fld.metadata["requires"])
#             if all([isinstance(el, list) for el in required_fields]):
#                 field_required_OR = required_fields
#             # if requires is a list of tuples/strings - I'm creating a 1-el nested list
#             elif all([isinstance(el, (str, tuple)) for el in required_fields]):
#                 field_required_OR = [required_fields]
#             else:
#                 raise Exception(
#                     f"requires field can be a list of list, or a list "
#                     f"of strings/tuples, but {fld.metadata['requires']} "
#                     f"provided for {fld.name}"
#                 )
#         else:
#             field_required_OR = [[]]

#         for field_required in field_required_OR:
#             # if the output has output_file_template field,
#             # adding all input fields from the template to requires
#             if "output_file_template" in fld.metadata:
#                 template = fld.metadata["output_file_template"]
#                 # if a template is a function it has to be run first with the inputs as the only arg
#                 if callable(template):
#                     template = template(inputs)
#                 inp_fields = re.findall(r"{\w+}", template)
#                 field_required += [
#                     el[1:-1] for el in inp_fields if el[1:-1] not in field_required
#                 ]

#         # it's a flag, of the field from the list is not in input it will be changed to False
#         required_found = True
#         for field_required in field_required_OR:
#             required_found = True
#             # checking if the input fields from requires have set values
#             for inp in field_required:
#                 if isinstance(inp, str):  # name of the input field
#                     if not hasattr(inputs, inp):
#                         raise Exception(
#                             f"{inp} is not a valid input field, can't be used in requires"
#                         )
#                     elif getattr(inputs, inp) in [attrs.NOTHING, None]:
#                         required_found = False
#                         break
#                 elif isinstance(inp, tuple):  # (name, allowed values)
#                     inp, allowed_val = inp[0], ensure_list(inp[1])
#                     if not hasattr(inputs, inp):
#                         raise Exception(
#                             f"{inp} is not a valid input field, can't be used in requires"
#                         )
#                     elif getattr(inputs, inp) not in allowed_val:
#                         required_found = False
#                         break
#                 else:
#                     raise Exception(
#                         f"each element of the requires element should be a string or a tuple, "
#                         f"but {inp} is found in {field_required}"
#                     )
#             # if the specific list from field_required_OR has all elements set, no need to check more
#             if required_found:
#                 break

#         if required_found:
#             return True
#         else:
#             return False


# @attrs.define
# class LazyInterface:
#     _task: "core.Task" = attrs.field()
#     _attr_type: str

#     def __getattr__(self, name):
#         if name in ("_task", "_attr_type", "_field_names"):
#             raise AttributeError(f"{name} hasn't been set yet")
#         if name not in self._field_names:
#             raise AttributeError(
#                 f"Task '{self._task.name}' has no {self._attr_type} attribute '{name}', "
#                 "available: '" + "', '".join(self._field_names) + "'"
#             )
#         type_ = self._get_type(name)
#         splits = self._get_task_splits()
#         combines = self._get_task_combines()
#         if combines and self._attr_type == "output":
#             # Add in any scalar splits referencing upstream splits, i.e. "_myupstreamtask",
#             # "_myarbitrarytask"
#             combined_upstreams = set()
#             if self._task.state:
#                 for scalar in LazyField.normalize_splitter(
#                     self._task.state.splitter, strip_previous=False
#                 ):
#                     for field in scalar:
#                         if field.startswith("_"):
#                             node_name = field[1:]
#                             if any(c.split(".")[0] == node_name for c in combines):
#                                 combines.update(
#                                     f for f in scalar if not f.startswith("_")
#                                 )
#                                 combined_upstreams.update(
#                                     f[1:] for f in scalar if f.startswith("_")
#                                 )
#             if combines:
#                 # Wrap type in list which holds the combined items
#                 type_ = ty.List[type_]
#                 # Iterate through splits to remove any splits which are removed by the
#                 # combiner
#                 for splitter in copy(splits):
#                     remaining = tuple(
#                         s
#                         for s in splitter
#                         if not any(
#                             (x in combines or x.split(".")[0] in combined_upstreams)
#                             for x in s
#                         )
#                     )
#                     if remaining != splitter:
#                         splits.remove(splitter)
#                         if remaining:
#                             splits.add(remaining)
#         # Wrap the type in a nested StateArray type
#         if splits:
#             type_ = StateArray[type_]
#         lf_klass = LazyInField if self._attr_type == "input" else LazyOutField
#         return lf_klass[type_](
#             name=self._task.name,
#             field=name,
#             type=type_,
#             splits=splits,
#         )

#     def _get_task_splits(self) -> ty.Set[ty.Tuple[ty.Tuple[str, ...], ...]]:
#         """Returns the states over which the inputs of the task are split"""
#         splitter = self._task.state.splitter if self._task.state else None
#         splits = set()
#         if splitter:
#             # Ensure that splits is of tuple[tuple[str, ...], ...] form
#             splitter = LazyField.normalize_splitter(splitter)
#             if splitter:
#                 splits.add(splitter)
#         for inpt in attrs.asdict(self._task.inputs, recurse=False).values():
#             if isinstance(inpt, LazyField):
#                 splits.update(inpt.splits)
#         return splits

#     def _get_task_combines(self) -> ty.Set[ty.Union[str, ty.Tuple[str, ...]]]:
#         """Returns the states over which the outputs of the task are combined"""
#         combiner = (
#             self._task.state.combiner
#             if self._task.state is not None
#             else getattr(self._task, "fut_combiner", None)
#         )
#         return set(combiner) if combiner else set()


# class LazyIn(LazyInterface):
#     _attr_type = "input"

#     def _get_type(self, name):
#         attr = next(t for n, t in self._task.input_spec.fields if n == name)
#         if attr is None:
#             return ty.Any
#         elif inspect.isclass(attr):
#             return attr
#         else:
#             return attr.type

#     @property
#     def _field_names(self):
#         return [field[0] for field in self._task.input_spec.fields]


# class LazyOut(LazyInterface):
#     _attr_type = "output"

#     def _get_type(self, name):
#         try:
#             type_ = next(f[1] for f in self._task.output_spec.fields if f[0] == name)
#         except StopIteration:
#             type_ = ty.Any
#         else:
#             if not inspect.isclass(type_):
#                 try:
#                     type_ = type_.type  # attrs _CountingAttribute
#                 except AttributeError:
#                     pass  # typing._SpecialForm
#         return type_

#     @property
#     def _field_names(self):
#         return self._task.output_names + ["all_"]


def donothing(*args, **kwargs):
    return None


@attrs.define(auto_attribs=True, kw_only=True)
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


def path_to_string(value):
    """Convert paths to strings."""
    if isinstance(value, Path):
        value = str(value)
    elif isinstance(value, list) and len(value) and isinstance(value[0], Path):
        value = [str(val) for val in value]
    return value


class OutputsSpec:
    """Base class for all output specifications"""

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


OutputType = ty.TypeVar("OutputType", bound=OutputsSpec)


class TaskSpec(ty.Generic[OutputType]):
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
        self._check_for_unset_values()
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

    def _check_for_unset_values(self):
        if unset := [
            k
            for k, v in attrs.asdict(self, recurse=False).items()
            if v is attrs.NOTHING
        ]:
            raise ValueError(
                f"The following values {unset} in the {self!r} interface need to be set "
                "before the workflow can be constructed"
            )


from pydra.engine import core  # noqa: E402
