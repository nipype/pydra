import typing as ty
import types
import inspect
import re
import enum
from pathlib import Path
from copy import copy
import attrs.validators
from attrs.converters import default_if_none
from fileformats.generic import File
from pydra.utils.typing import TypeParser, is_optional, is_fileset_or_union

# from pydra.utils.misc import get_undefined_symbols
from pydra.engine.helpers import from_list_if_single, ensure_list
from pydra.engine.specs import (
    LazyField,
    MultiInputObj,
    MultiInputFile,
    MultiOutputObj,
    MultiOutputFile,
)
from pydra.engine.core import Task, AuditFlag

__all__ = [
    "Field",
    "Arg",
    "Out",
    "TaskSpec",
    "collate_with_helps",
    "make_task_spec",
    "list_fields",
]


class _Empty(enum.Enum):

    EMPTY = enum.auto()

    def __repr__(self):
        return "EMPTY"

    def __bool__(self):
        return False


EMPTY = _Empty.EMPTY  # To provide a blank placeholder for the default field


def is_type(_, __, val: ty.Any) -> bool:
    """check that the value is a type or generic"""
    return inspect.isclass(val) or ty.get_origin(val)


@attrs.define(kw_only=True)
class Field:
    """Base class for input and output fields to task specifications

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    help_string: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    """

    name: str | None = None
    type: ty.Type[ty.Any] = attrs.field(
        validator=is_type, default=ty.Any, converter=default_if_none(ty.Any)
    )
    help_string: str = ""
    requires: list | None = None
    converter: ty.Callable | None = None
    validator: ty.Callable | None = None


@attrs.define(kw_only=True)
class Arg(Field):
    """Base class for input fields of task specifications

    Parameters
    ----------
    help_string: str
        A short description of the input field.
    default : Any, optional
        the default value for the argument
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    xor: list, optional
        Names of the inputs that are mutually exclusive with the field.
    copy_mode: File.CopyMode, optional
        The mode of copying the file, by default it is File.CopyMode.any
    copy_collation: File.CopyCollation, optional
        The collation of the file, by default it is File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition, optional
        The extension decomposition of the file, by default it is
        File.ExtensionDecomposition.single
    readonly: bool, optional
        If True the input field can’t be provided by the user but it aggregates other
        input fields (for example the fields with argstr: -o {fldA} {fldB}), by default
        it is False
    type: type, optional
        The type of the field, by default it is Any
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    """

    default: ty.Any = EMPTY
    allowed_values: list | None = None
    xor: list | None = None
    copy_mode: File.CopyMode = File.CopyMode.any
    copy_collation: File.CopyCollation = File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition = File.ExtensionDecomposition.single
    readonly: bool = False


@attrs.define(kw_only=True)
class Out(Field):
    """Base class for output fields of task specifications

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    help_string: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    """

    pass


OutputType = ty.TypeVar("OutputType")


class TaskSpec(ty.Generic[OutputType]):
    """Base class for all task specifications"""

    Task: ty.Type[Task]

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
        if unset := [k for k, v in attrs.asdict(self).items() if v is attrs.NOTHING]:
            raise ValueError(
                f"The following values in the {self!r} interface need to be set before it "
                f"can be executed: {unset}"
            )


def get_fields_from_class(
    klass: type,
    arg_type: type[Arg],
    out_type: type[Out],
    auto_attribs: bool,
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Parse the input and output fields from a class"""

    input_helps, _ = parse_doc_string(klass.__doc__)

    def get_fields(klass, field_type, auto_attribs, helps) -> dict[str, Field]:
        """Get the fields from a class"""
        fields_dict = {}
        # Get fields defined in base classes if present
        for field in list_fields(klass):
            fields_dict[field.name] = field
        type_hints = ty.get_type_hints(klass)
        for atr_name in dir(klass):
            if atr_name in ["Task", "Outputs"] or atr_name.startswith("__"):
                continue
            try:
                atr = getattr(klass, atr_name)
            except Exception:
                continue
            if isinstance(atr, Field):
                atr.name = atr_name
                fields_dict[atr_name] = atr
                if atr_name in type_hints:
                    atr.type = type_hints[atr_name]
                if not atr.help_string:
                    atr.help_string = helps.get(atr_name, "")
            elif atr_name in type_hints:
                if atr_name in fields_dict:
                    fields_dict[atr_name].type = type_hints[atr_name]
                elif auto_attribs:
                    fields_dict[atr_name] = field_type(
                        name=atr_name,
                        type=type_hints[atr_name],
                        default=atr,
                        help_string=helps.get(atr_name, ""),
                    )
        if auto_attribs:
            for atr_name, type_ in type_hints.items():
                if atr_name not in list(fields_dict) + ["Task", "Outputs"]:
                    fields_dict[atr_name] = field_type(
                        name=atr_name, type=type_, help_string=helps.get(atr_name, "")
                    )
        return fields_dict

    inputs = get_fields(klass, arg_type, auto_attribs, input_helps)

    try:
        outputs_klass = klass.Outputs
    except AttributeError:
        raise AttributeError(
            f"Nested Outputs class not found in {klass.__name__}"
        ) from None
    output_helps, _ = parse_doc_string(outputs_klass.__doc__)
    outputs = get_fields(outputs_klass, out_type, auto_attribs, output_helps)

    return inputs, outputs


def make_task_spec(
    task_type: type[Task],
    inputs: dict[str, Arg],
    outputs: dict[str, Out],
    klass: type | None = None,
    name: str | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
):
    if name is None and klass is not None:
        name = klass.__name__
    outputs_klass = type(
        "Outputs",
        tuple(outputs_bases),
        {
            o.name: attrs.field(
                converter=make_converter(o, f"{name}.Outputs"),
                metadata={PYDRA_ATTR_METADATA: o},
                **_get_default(o),
            )
            for o in outputs.values()
        },
    )
    outputs_klass.__annotations__.update((o.name, o.type) for o in outputs.values())
    outputs_klass = attrs.define(auto_attribs=False, kw_only=True)(outputs_klass)

    if klass is None or not issubclass(klass, TaskSpec):
        if name is None:
            raise ValueError("name must be provided if klass is not")
        bases = tuple(bases)
        if not any(issubclass(b, TaskSpec) for b in bases):
            bases = bases + (TaskSpec,)
        if klass is not None:
            bases += tuple(c for c in klass.__mro__ if c not in bases + (object,))
        klass = types.new_class(
            name=name,
            bases=bases,
            kwds={},
            exec_body=lambda ns: ns.update(
                {"Task": task_type, "Outputs": outputs_klass}
            ),
        )
    else:
        # Ensure that the class has it's own annotaitons dict so we can modify it without
        # messing up other classes
        klass.__annotations__ = copy(klass.__annotations__)
        klass.Task = task_type
        klass.Outputs = outputs_klass
    # Now that we have saved the attributes in lists to be
    for arg in inputs.values():
        # If an outarg input then the field type should be Path not a FileSet
        if isinstance(arg, Out) and is_fileset_or_union(arg.type):
            if getattr(arg, "path_template", False):
                if is_optional(arg.type):
                    field_type = Path | bool | None
                    # Will default to None and not be inserted into the command
                else:
                    field_type = Path | bool
                    arg.default = True
            elif is_optional(arg.type):
                field_type = Path | None
            else:
                field_type = Path
        else:
            field_type = arg.type
        setattr(
            klass,
            arg.name,
            attrs.field(
                converter=make_converter(arg, klass.__name__, field_type),
                validator=make_validator(arg, klass.__name__),
                metadata={PYDRA_ATTR_METADATA: arg},
                on_setattr=attrs.setters.convert,
                **_get_default(arg),
            ),
        )
        klass.__annotations__[arg.name] = field_type

    # Create class using attrs package, will create attributes for all columns and
    # parameters
    attrs_klass = attrs.define(auto_attribs=False, kw_only=True)(klass)

    return attrs_klass


def collate_with_helps(
    arg_type: type[Arg],
    out_type: type[Out],
    doc_string: str | None = None,
    inputs: dict[str, Arg | type] | None = None,
    outputs: dict[str, Out | type] | None = None,
    input_helps: dict[str, str] | None = None,
    output_helps: dict[str, str] | None = None,
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Assign help strings to the appropriate inputs and outputs"""

    for input_name, arg in list(inputs.items()):
        if isinstance(arg, Arg):
            if arg.name is None:
                arg.name = input_name
            elif arg.name != input_name:
                raise ValueError(
                    "Name of the argument must be the same as the key in the "
                    f"dictionary. The argument name is {arg.name} and the key "
                    f"is {input_name}"
                )
            else:
                arg.name = input_name
            if not arg.help_string:
                arg.help_string = input_helps.get(input_name, "")
        else:
            inputs[input_name] = arg_type(
                type=arg,
                name=input_name,
                help_string=input_helps.get(input_name, ""),
            )

    for output_name, out in list(outputs.items()):
        if isinstance(out, Out):
            if out.name is None:
                out.name = output_name
            elif out.name != output_name:
                raise ValueError(
                    "Name of the argument must be the same as the key in the "
                    f"dictionary. The argument name is {out.name} and the key "
                    f"is {output_name}"
                )
            else:
                out.name = output_name
            if not out.help_string:
                out.help_string = output_helps.get(output_name, "")
        else:
            outputs[output_name] = out_type(
                type=out,
                name=output_name,
                help_string=output_helps.get(output_name, ""),
            )

    return inputs, outputs


def make_converter(
    field: Field, interface_name: str, field_type: ty.Type | None = None
):
    if field_type is None:
        field_type = field.type
    checker_label = f"'{field.name}' field of {interface_name} interface"
    type_checker = TypeParser[field_type](
        field_type, label=checker_label, superclass_auto_cast=True
    )
    converters = []
    if field.type in (MultiInputObj, MultiInputFile):
        converters.append(ensure_list)
    elif field.type in (MultiOutputObj, MultiOutputFile):
        converters.append(from_list_if_single)
    if field.converter:
        converters.append(field.converter)
    if converters:
        converters.append(type_checker)
        converter = attrs.converters.pipe(*converters)
    else:
        converter = type_checker
    return converter


def make_validator(field: Field, interface_name: str):
    validators = []
    if field.allowed_values:
        validators.append(allowed_values_validator)
    if isinstance(field.validator, ty.Iterable):
        validators.extend(field.validator)
    elif field.validator:
        validators.append(field.validator)
    if len(validators) > 1:
        return validators
    elif validators:
        return validators[0]
    return None


def allowed_values_validator(_, attribute, value):
    """checking if the values is in allowed_values"""
    allowed = attribute.metadata[PYDRA_ATTR_METADATA].allowed_values
    if value is attrs.NOTHING or isinstance(value, LazyField):
        pass
    elif value not in allowed:
        raise ValueError(
            f"value of {attribute.name} has to be from {allowed}, but {value} provided"
        )


def extract_function_inputs_and_outputs(
    function: ty.Callable,
    arg_type: type[Arg],
    inputs: list[str | Arg] | dict[str, Arg | type] | None = None,
    outputs: list[str | Out] | dict[str, Out | type] | type | None = None,
) -> tuple[dict[str, type | Arg], dict[str, type | Out]]:
    """Extract input output types and output names from the function source if they
    aren't explicitly"""
    # if undefined_symbols := get_undefined_symbols(
    #     function, exclude_signature_type_hints=True, ignore_decorator=True
    # ):
    #     raise ValueError(
    #         f"The following symbols are not defined within the scope of the function "
    #         f"{function!r}, {undefined_symbols}. Ensure that all imports are "
    #         "defined within the function scope so it is portable"
    #     )
    sig = inspect.signature(function)
    type_hints = ty.get_type_hints(function)
    input_types = {}
    input_defaults = {}
    for p in sig.parameters.values():
        input_types[p.name] = type_hints.get(p.name, ty.Any)
        if p.default is not inspect.Parameter.empty:
            input_defaults[p.name] = p.default
    if inputs:
        if not isinstance(inputs, dict):
            raise ValueError(
                f"Input names ({inputs}) should not be provided when "
                "wrapping/decorating a function as "
            )
        for inpt_name, type_ in input_types.items():
            try:
                inpt = inputs[inpt_name]
            except KeyError:
                inputs[inpt_name] = type_
            else:
                if isinstance(inpt, Arg) and inpt.type is ty.Any:
                    inpt.type = type_
    else:
        inputs = input_types
    for inpt_name, default in input_defaults.items():
        inpt = inputs[inpt_name]
        if isinstance(inpt, arg_type) and inpt.default is EMPTY:
            inpt.default = default
        else:
            inputs[inpt_name] = arg_type(type=inpt, default=default)
    return_type = type_hints.get("return", ty.Any)
    if outputs is None:
        src = inspect.getsource(function).strip()
        return_lines = re.findall(r"\n\s+return .*$", src)
        if len(return_lines) == 1 and src.endswith(return_lines[0]):
            implicit_outputs = [
                o.strip()
                for o in re.match(r"\s*return\s+(.*)", return_lines[0])
                .group(1)
                .split(",")
            ]
            if len(implicit_outputs) > 1 and all(
                re.match(r"^\w+$", o) for o in implicit_outputs
            ):
                outputs = implicit_outputs
    if outputs and len(outputs) > 1:
        if return_type is not ty.Any:
            if ty.get_origin(return_type) is not tuple:
                raise ValueError(
                    f"Multiple outputs specified ({outputs}) but non-tuple "
                    f"return value {return_type}"
                )
            return_types = ty.get_args(return_type)
        if len(return_types) != len(outputs):
            raise ValueError(
                f"Length of the outputs ({outputs}) does not match that "
                f"of the return types ({return_types})"
            )
        output_types = dict(zip(outputs, return_types))
        if isinstance(outputs, dict):
            for output_name, output in outputs.items():
                if isinstance(output, Out) and output.type is ty.Any:
                    output.type = output_types[output_name]
        else:
            outputs = output_types

    elif outputs:
        if isinstance(outputs, dict):
            output_name, output = next(iter(outputs.items()))
        elif isinstance(outputs, list):
            output_name = outputs[0]
            output = ty.Any
        if isinstance(output, Out):
            if output.type is ty.Any:
                output.type = return_type
        elif output is ty.Any:
            output = return_type
        outputs = {output_name: output}
    else:
        outputs = {"out": return_type}
    return inputs, outputs


def parse_doc_string(doc_str: str) -> tuple[dict[str, str], dict[str, str] | list[str]]:
    """Parse the docstring to pull out the description of the parameters/args and returns

    Parameters
    -----------
    doc_string
        the doc string to parse

    Returns
    -------
    input_helps
        the documentation for each of the parameter/args of the class/function
    output_helps
        the documentation for each of the return values of the class function, if no
        names are provided then the help strings are returned as a list
    """
    input_helps = {}
    output_helps = {}
    if doc_str is None:
        return input_helps, output_helps
    for param, param_help in re.findall(r":param (\w+): (.*)", doc_str):
        input_helps[param] = param_help
    for return_val, return_help in re.findall(r":return (\w+): (.*)", doc_str):
        output_helps[return_val] = return_help
    google_args_match = re.match(
        r".*\n\s+Args:\n(.*)", doc_str, flags=re.DOTALL | re.MULTILINE
    )
    google_returns_match = re.match(
        r".*\n\s+Returns:\n(.*)", doc_str, flags=re.DOTALL | re.MULTILINE
    )
    if google_args_match:
        args_str = google_args_match.group(1)
        for arg_str in split_block(args_str):
            arg_name, arg_help = arg_str.split(":", maxsplit=1)
            arg_name = arg_name.strip()
            arg_help = white_space_re.sub(" ", arg_help).strip()
            input_helps[arg_name] = arg_help
    if google_returns_match:
        returns_str = google_returns_match.group(1)
        for return_str in split_block(returns_str):
            return_name, return_help = return_str.split(":", maxsplit=1)
            return_name = return_name.strip()
            return_help = white_space_re.sub(" ", return_help).strip()
            output_helps[return_name] = return_help
    numpy_args_match = re.match(
        r".*\n\s+Parameters\n\s*---------- *\n(.*)",
        doc_str,
        flags=re.DOTALL | re.MULTILINE,
    )
    numpy_returns_match = re.match(
        r".*\n\s+Returns\n\s+------- *\n(.*)", doc_str, flags=re.DOTALL | re.MULTILINE
    )
    if numpy_args_match:
        args_str = numpy_args_match.group(1)
        for arg_str in split_block(args_str):
            arg_decl, arg_help = arg_str.split("\n", maxsplit=1)
            arg_name = arg_decl.split(":")[0].strip()
            arg_help = white_space_re.sub(" ", arg_help).strip()
            input_helps[arg_name] = arg_help
    if numpy_returns_match:
        returns_str = numpy_returns_match.group(1)
        for return_str in split_block(returns_str):
            return_decl, return_help = return_str.split("\n", maxsplit=1)
            return_name = return_decl.split(":")[0].strip()
            return_help = white_space_re.sub(" ", return_help).strip()
            output_helps[return_name] = return_help
    return input_helps, output_helps


def split_block(string: str) -> ty.Generator[str, None, None]:
    """Split a block of text into groups lines"""
    indent_re = re.compile(r"^\s*")
    leading_indent = indent_re.match(string).group()
    leading_indent_len = len(leading_indent)
    block = ""
    for line in string.split("\n"):
        if not line.strip():
            break
        indent_len = len(indent_re.match(line).group())
        if block and indent_len == leading_indent_len:
            yield block.strip()
            block = ""
        block += line + "\n"
        if indent_len < leading_indent_len:
            raise ValueError(
                f"Indentation block is not consistent in docstring:\n{string}"
            )
    if block:
        yield block.strip()


def list_fields(interface: TaskSpec) -> list[Field]:
    if not attrs.has(interface):
        return []
    return [
        f.metadata[PYDRA_ATTR_METADATA]
        for f in attrs.fields(interface)
        if PYDRA_ATTR_METADATA in f.metadata
    ]


def check_explicit_fields_are_none(klass, inputs, outputs):
    if inputs is not None:
        raise ValueError(
            f"inputs should not be provided to `python.task` ({inputs}) "
            f"explicitly when decorated a class ({klass})"
        )
    if outputs is not None:
        raise ValueError(
            f"outputs should not be provided to `python.task` ({outputs}) "
            f"explicitly when decorated a class ({klass})"
        )


def _get_default(field: Field) -> dict[str, ty.Any]:
    if not hasattr(field, "default"):
        return {"factory": nothing_factory}
    if field.default is not EMPTY:
        return {"default": field.default}
    if is_optional(field.type):
        return {"default": None}
    return {"factory": nothing_factory}


def nothing_factory():
    return attrs.NOTHING


white_space_re = re.compile(r"\s+")

PYDRA_ATTR_METADATA = "__PYDRA_METADATA__"
