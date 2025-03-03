import typing as ty
import types
import inspect
import re
import enum
from pathlib import Path
from copy import copy
from typing import Self
import attrs.validators
from attrs.converters import default_if_none
from fileformats.generic import File
from pydra.utils.typing import TypeParser, is_optional, is_fileset_or_union, is_type
from pydra.engine.helpers import (
    from_list_if_single,
    ensure_list,
    PYDRA_ATTR_METADATA,
    list_fields,
    is_lazy,
)
from pydra.utils.typing import (
    MultiInputObj,
    MultiInputFile,
    MultiOutputObj,
    MultiOutputFile,
)
from pydra.utils.hash import hash_function


if ty.TYPE_CHECKING:
    from pydra.engine.specs import TaskDef, TaskOutputs


__all__ = [
    "Field",
    "Arg",
    "Out",
    "ensure_field_objects",
    "make_task_def",
]


class _Empty(enum.Enum):

    NO_DEFAULT = enum.auto()

    def __repr__(self):
        return "NO_DEFAULT"

    def __bool__(self):
        return False


NO_DEFAULT = _Empty.NO_DEFAULT  # To provide a blank placeholder for the default field


def convert_default_value(value: ty.Any, self_: "Field") -> ty.Any:
    """Ensure the default value has been coerced into the correct type"""
    if value is NO_DEFAULT or isinstance(value, attrs.Factory):
        return value
    if self_.type is ty.Callable and isinstance(value, ty.Callable):
        return value
    return TypeParser[self_.type](self_.type, label=self_.name)(value)


def allowed_values_converter(value: ty.Iterable[str] | None) -> list[str] | None:
    """Ensure the allowed_values field is a list of strings or None"""
    if value is None:
        return None
    return list(value)


@attrs.define
class Requirement:
    """Define a requirement for a task input field

    Parameters
    ----------
    name : str
        The name of the input field that is required
    allowed_values : list[str], optional
        The allowed values for the input field that is required, if not provided any
        value is allowed
    """

    name: str
    allowed_values: list[str] | None = attrs.field(
        default=None, converter=allowed_values_converter
    )

    def satisfied(self, inputs: "TaskDef") -> bool:
        """Check if the requirement is satisfied by the inputs"""
        value = getattr(inputs, self.name)
        field = {f.name: f for f in list_fields(inputs)}[self.name]
        if value is attrs.NOTHING or field.type is bool and value is False:
            return False
        if self.allowed_values is None:
            return True
        return value in self.allowed_values

    @classmethod
    def parse(cls, value: ty.Any) -> Self:
        if isinstance(value, Requirement):
            return value
        elif isinstance(value, str):
            return Requirement(value)
        else:
            name, allowed_values = value
            if isinstance(allowed_values, str) or not isinstance(
                allowed_values, ty.Collection
            ):
                raise ValueError(
                    f"allowed_values must be a collection of strings, not {allowed_values}"
                )
            return Requirement(name, allowed_values)

    def __str__(self):
        if not self.allowed_values:
            return self.name
        return f"{self.name}(" + ",".join(repr(v) for v in self.allowed_values) + ")"


def requirements_converter(value: ty.Any) -> list[Requirement]:
    """Ensure the requires field is a list of Requirement objects"""
    if isinstance(value, Requirement):
        return [value]
    elif isinstance(value, (str, tuple)):
        return [Requirement.parse(value)]
    return [Requirement.parse(v) for v in value]


@attrs.define
class RequirementSet:
    """Define a set of requirements for a task input field, all of which must be satisfied"""

    requirements: list[Requirement] = attrs.field(
        factory=list,
        converter=requirements_converter,
    )

    def satisfied(self, inputs: "TaskDef") -> bool:
        """Check if all the requirements are satisfied by the inputs"""
        return all(req.satisfied(inputs) for req in self.requirements)

    def __str__(self):
        if len(self.requirements) == 1:
            return str(self.requirements[0])
        return "+".join(str(r) for r in self.requirements)

    def __iter__(self):
        return iter(self.requirements)

    def __iadd__(self, other: "RequirementSet | list[Requirement]") -> "RequirementSet":
        self.requirements.extend(requirements_converter(other))
        return self


def requires_converter(
    value: (
        str
        | ty.Collection[
            Requirement | str | ty.Collection[str | tuple[str, ty.Collection[ty.Any]]]
        ]
    ),
) -> list[RequirementSet]:
    """Ensure the requires field is a tuple of tuples"""
    if isinstance(value, (str, tuple, Requirement)):
        return [RequirementSet(value)]
    return [RequirementSet(v) for v in value]


@attrs.define(kw_only=True)
class Field:
    """Base class for input and output fields to task definitions

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
    type: type, optional
        The type of the field, by default it is Any
        from name to field, by default it is None
    default : Any, optional
        the default value for the field, by default it is NO_DEFAULT
    help: str, optional
        A short description of the input field.
    requires: str | list[str | list[str] | Requirement], optional
        The input fields that are required to be provided, along with the optional allowed
        values, that are required together with the field. Can be provided
        as a single name, a collection of names, a collection of collections of names,
        or a collection of collection of name/allowed values pairs.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    hash_eq: bool, optional
        Whether to use the hash of the value for equality comparison, by default it is False
    """

    name: str | None = None
    type: ty.Type[ty.Any] = attrs.field(
        validator=is_type, default=ty.Any, converter=default_if_none(ty.Any)
    )
    default: ty.Any = attrs.field(
        default=NO_DEFAULT,
        converter=attrs.Converter(convert_default_value, takes_self=True),
    )
    help: str = ""
    requires: list[RequirementSet] = attrs.field(
        factory=list, converter=requires_converter
    )
    converter: ty.Callable[..., ty.Any] | None = None
    validator: ty.Callable[..., bool] | None = None
    hash_eq: bool = False

    def requirements_satisfied(self, inputs: "TaskDef") -> bool:
        """Check if all the requirements are satisfied by the inputs"""
        return any(req.satisfied(inputs) for req in self.requires)

    @property
    def mandatory(self):
        return self.default is NO_DEFAULT


@attrs.define(kw_only=True)
class Arg(Field):
    """Base class for input fields of task definitions

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    default : Any, optional
        the default value for the field, by default it is NO_DEFAULT
    help: str
        A short description of the input field.
    allowed_values: list, optional
        List of allowed values for the field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    xor: list[str], optional
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
    """

    allowed_values: tuple = attrs.field(default=(), converter=tuple)
    xor: tuple[str] = attrs.field(default=(), converter=tuple)
    copy_mode: File.CopyMode = File.CopyMode.any
    copy_collation: File.CopyCollation = File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition = File.ExtensionDecomposition.single
    readonly: bool = False


@attrs.define(kw_only=True)
class Out(Field):
    """Base class for output fields of task definitions

    Parameters
    ----------
    name: str, optional
        The name of the field, used when specifying a list of fields instead of a mapping
        from name to field, by default it is None
    type: type, optional
        The type of the field, by default it is Any
    default : Any, optional
        the default value for the field, by default it is NO_DEFAULT
    help: str, optional
        A short description of the input field.
    requires: list, optional
        Names of the inputs that are required together with the field.
    converter: callable, optional
        The converter for the field passed through to the attrs.field, by default it is None
    validator: callable | iterable[callable], optional
        The validator(s) for the field passed through to the attrs.field, by default it is None
    """

    pass


def extract_fields_from_class(
    spec_type: type["TaskDef"],
    outputs_type: type["TaskOutputs"],
    klass: type,
    arg_type: type[Arg],
    out_type: type[Out],
    auto_attribs: bool,
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Extract the input and output fields from an existing class

    Parameters
    ----------
    klass : type
        The class to extract the fields from
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    auto_attribs : bool
        Whether to assume that all attribute annotations should be interpreted as
        fields or not

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields extracted from the class
    outputs : dict[str, Out]
        The output fields extracted from the class
    """

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
                if not atr.help:
                    atr.help = helps.get(atr_name, "")
            elif atr_name in type_hints:
                if atr_name in fields_dict:
                    fields_dict[atr_name].type = type_hints[atr_name]
                elif auto_attribs:
                    fields_dict[atr_name] = field_type(
                        name=atr_name,
                        type=type_hints[atr_name],
                        default=atr,
                        help=helps.get(atr_name, ""),
                    )
        if auto_attribs:
            for atr_name, type_ in type_hints.items():
                if atr_name not in list(fields_dict) + ["Task", "Outputs"]:
                    fields_dict[atr_name] = field_type(
                        name=atr_name, type=type_, help=helps.get(atr_name, "")
                    )
        return fields_dict

    if not issubclass(klass, spec_type):
        raise ValueError(
            f"When using the canonical form for {spec_type.__module__.split('.')[-1]} "
            f"tasks, {klass} must inherit from {spec_type}"
        )

    inputs = get_fields(klass, arg_type, auto_attribs, input_helps)

    try:
        outputs_klass = klass.Outputs
    except AttributeError:
        raise AttributeError(
            f"Nested Outputs class not found in {klass.__name__}"
        ) from None
    if not issubclass(outputs_klass, outputs_type):
        raise ValueError(
            f"When using the canonical form for {outputs_type.__module__.split('.')[-1]} "
            f"task outputs {outputs_klass}, you must inherit from {outputs_type}"
        )

    output_helps, _ = parse_doc_string(outputs_klass.__doc__)
    outputs = get_fields(outputs_klass, out_type, auto_attribs, output_helps)

    return inputs, outputs


def make_task_def(
    spec_type: type["TaskDef"],
    out_type: type["TaskOutputs"],
    inputs: dict[str, Arg],
    outputs: dict[str, Out],
    klass: type | None = None,
    name: str | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
):
    """Create a task definition class and its outputs definition class from the
    input and output fields provided to the decorator/function.

    Modifies the class so that its attributes are converted from pydra fields to attrs fields
    and then calls `attrs.define` to create an attrs class (dataclass-like).
    on

    Parameters
    ----------
    task_type : type
        The type of the task to be created
    inputs : dict[str, Arg]
        The input fields of the task
    outputs : dict[str, Out]
        The output fields of the task
    klass : type, optional
        The class to be decorated, by default None
    name : str, optional
        The name of the class, by default
    bases : ty.Sequence[type], optional
        The base classes for the task definition class, by default ()
    outputs_bases : ty.Sequence[type], optional
        The base classes for the outputs definition class, by default ()

    Returns
    -------
    klass : type
        The class created using the attrs package
    """

    spec_type._check_arg_refs(inputs, outputs)

    # Check that the field attributes are valid after all fields have been set
    # (especially the type)
    for inpt in inputs.values():
        attrs.validate(inpt)
    for outpt in outputs.values():
        attrs.validate(outpt)

    if name is None and klass is not None:
        name = klass.__name__
    if reserved_names := [n for n in inputs if n in spec_type.RESERVED_FIELD_NAMES]:
        raise ValueError(
            f"{reserved_names} are reserved and cannot be used for {spec_type} field names"
        )
    outputs_klass = make_outputs_spec(out_type, outputs, outputs_bases, name)
    if klass is None:
        if name is None:
            raise ValueError("name must be provided if klass is not")
        bases = tuple(bases)
        # Ensure that TaskDef is a base class
        if not any(issubclass(b, spec_type) for b in bases):
            bases = bases + (spec_type,)
        # If building from a decorated class (as opposed to dynamically from a function
        # or shell-template), add any base classes not already in the bases tuple
        if klass is not None:
            bases += tuple(c for c in klass.__mro__ if c not in bases + (object,))
        # Create a new class with the TaskDef as a base class
        klass = types.new_class(
            name=name,
            bases=bases,
            kwds={},
            exec_body=lambda ns: ns.update({"Outputs": outputs_klass}),
        )
    else:
        # Ensure that the class has it's own annotations dict so we can modify it without
        # messing up other classes
        klass.__annotations__ = copy(klass.__annotations__)
        klass.Outputs = outputs_klass
    # Now that we have saved the attributes in lists to be
    for arg in inputs.values():
        # If an outarg input then the field type should be Path not a FileSet
        attrs_kwargs = _get_attrs_kwargs(arg)
        if isinstance(arg, Out) and is_fileset_or_union(arg.type):
            if getattr(arg, "path_template", False):
                if is_optional(arg.type):
                    field_type = Path | bool | None
                    attrs_kwargs = {"default": None}
                else:
                    field_type = Path | bool
                    attrs_kwargs = {"default": True}  # use the template by default
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
                **attrs_kwargs,
            ),
        )
        klass.__annotations__[arg.name] = field_type

    # Create class using attrs package, will create attributes for all columns and
    # parameters
    attrs_klass = attrs.define(auto_attribs=False, kw_only=True, eq=False)(klass)

    return attrs_klass


def make_outputs_spec(
    spec_type: type["TaskOutputs"],
    outputs: dict[str, Out],
    bases: ty.Sequence[type],
    spec_name: str,
) -> type["TaskOutputs"]:
    """Create an outputs definition class and its outputs definition class from the
    output fields provided to the decorator/function.

    Creates a new class with attrs fields and then calls `attrs.define` to create an
    attrs class (dataclass-like).

    Parameters
    ----------
    outputs : dict[str, Out]
        The output fields of the task
    bases : ty.Sequence[type], optional
        The base classes for the outputs definition class, by default ()
    spec_name : str
        The name of the task definition class the outputs are for

    Returns
    -------
    klass : type
        The class created using the attrs package
    """
    from pydra.engine.specs import TaskOutputs

    if not any(issubclass(b, spec_type) for b in bases):
        if out_spec_bases := [b for b in bases if issubclass(b, TaskOutputs)]:
            raise ValueError(
                f"Cannot make {spec_type} output definition from {out_spec_bases} bases"
            )
        outputs_bases = bases + (spec_type,)
    if reserved_names := [n for n in outputs if n in spec_type.RESERVED_FIELD_NAMES]:
        raise ValueError(
            f"{reserved_names} are reserved and cannot be used for {spec_type} field names"
        )
    # Add in any fields in base classes that haven't already been converted into attrs
    # fields (e.g. stdout, stderr and return_code)
    for base in outputs_bases:
        base_outputs = {
            n: o
            for n, o in base.__dict__.items()
            if isinstance(o, Out) and n not in outputs
        }
        for name, field in base_outputs.items():
            field.name = name
            field.type = base.__annotations__.get(name, ty.Any)
        outputs.update(base_outputs)
    assert all(o.name == n for n, o in outputs.items())
    outputs_klass = type(
        spec_name + "Outputs",
        tuple(outputs_bases),
        {
            n: attrs.field(
                converter=make_converter(o, f"{spec_name}.Outputs"),
                metadata={PYDRA_ATTR_METADATA: o},
                **_get_attrs_kwargs(o),
            )
            for n, o in outputs.items()
        },
    )
    outputs_klass.__annotations__.update((o.name, o.type) for o in outputs.values())
    outputs_klass = attrs.define(auto_attribs=False, kw_only=True, eq=False)(
        outputs_klass
    )
    return outputs_klass


def ensure_field_objects(
    arg_type: type[Arg],
    out_type: type[Out],
    doc_string: str | None = None,
    inputs: dict[str, Arg | type] | None = None,
    outputs: dict[str, Out | type] | None = None,
    input_helps: dict[str, str] | None = None,
    output_helps: dict[str, str] | None = None,
) -> tuple[dict[str, Arg], dict[str, Out]]:
    """Converts dicts containing input/output types into input/output, including any
    help strings to the appropriate inputs and outputs

    Parameters
    ----------
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    doc_string : str, optional
        The docstring of the function or class
    inputs : dict[str, Arg | type], optional
        The inputs to the function or class
    outputs : dict[str, Out | type], optional
        The outputs of the function or class
    input_helps : dict[str, str], optional
        The help strings for the inputs
    output_helps : dict[str, str], optional
        The help strings for the outputs

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields with help strings added
    outputs : dict[str, Out]
        The output fields with help strings added
    """

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
            if not arg.help:
                arg.help = input_helps.get(input_name, "")
        elif is_type(arg):
            inputs[input_name] = arg_type(
                type=arg,
                name=input_name,
                help=input_helps.get(input_name, ""),
            )
        elif isinstance(arg, dict):
            arg_kwds = copy(arg)
            if "help" not in arg_kwds:
                arg_kwds["help"] = input_helps.get(input_name, "")
            inputs[input_name] = arg_type(
                name=input_name,
                **arg_kwds,
            )
        else:
            raise ValueError(
                f"Input {input_name} must be an instance of {Arg}, a type, or a dictionary "
                f" of keyword arguments to pass to {Arg}, not {arg}"
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
            if not out.help:
                out.help = output_helps.get(output_name, "")
        elif inspect.isclass(out) or ty.get_origin(out):
            outputs[output_name] = out_type(
                type=out,
                name=output_name,
                help=output_helps.get(output_name, ""),
            )
        elif isinstance(out, dict):
            out_kwds = copy(out)
            if "help" not in out_kwds:
                out_kwds["help"] = output_helps.get(output_name, "")
            outputs[output_name] = out_type(
                name=output_name,
                **out_kwds,
            )
        elif isinstance(out, ty.Callable) and hasattr(out_type, "callable"):
            outputs[output_name] = out_type(
                name=output_name,
                type=ty.get_type_hints(out).get("return", ty.Any),
                callable=out,
                help=re.split(r"\n\s*\n", out.__doc__)[0] if out.__doc__ else "",
            )
        else:
            raise ValueError(
                f"Unrecognised value provided to outputs ({arg}), can be either {out_type} "
                "type" + (" or callable" if hasattr(out_type, "callable") else "")
            )

    return inputs, outputs


def make_converter(
    field: Field, interface_name: str, field_type: ty.Type | None = None
) -> ty.Callable[..., ty.Any]:
    """Makes an attrs converter for the field, combining type checking with any explicit
    converters

    Parameters
    ----------
    field : Field
        The field to make the converter for
    interface_name : str
        The name of the interface the field is part of
    field_type : type, optional
        The type of the field, by default None

    Returns
    -------
    converter : callable
        The converter for the field
    """
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


def make_validator(field: Field, interface_name: str) -> ty.Callable[..., None] | None:
    """Makes an attrs validator for the field, combining allowed values and any explicit
    validators

    Parameters
    ----------
    field : Field
        The field to make the validator for
    interface_name : str
        The name of the interface the field is part of

    Returns
    -------
    validator : callable
        The validator for the field
    """
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
    if value is attrs.NOTHING or is_lazy(value):
        pass
    elif value is None and is_optional(attribute.type):
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
    aren't explicitly

    Parameters
    ----------
    function : callable
        The function to extract the inputs and outputs from
    arg_type : type
        The type of the input fields
    out_type : type
        The type of the output fields
    inputs : list[str | Arg] | dict[str, Arg | type] | None
        The inputs to the function
    outputs : list[str | Out] | dict[str, Out | type] | type | None
        The outputs of the function

    Returns
    -------
    inputs : dict[str, Arg]
        The input fields extracted from the function
    outputs : dict[str, Out]
        The output fields extracted from the function
    """
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
    has_varargs = False
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL or p.kind is p.VAR_KEYWORD:
            has_varargs = True
            continue
        input_types[p.name] = type_hints.get(p.name, ty.Any)
        if p.default is not inspect.Parameter.empty:
            input_defaults[p.name] = p.default
    if inputs:
        if not isinstance(inputs, dict):
            raise ValueError(
                f"Input names ({inputs}) should not be provided when "
                "wrapping/decorating a function as "
            )
        if not has_varargs:
            if unrecognised := set(inputs) - set(input_types):
                raise ValueError(
                    f"Unrecognised input names ({unrecognised}) not present in the signature "
                    f"of the function {function!r}"
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
        if isinstance(inpt, arg_type):
            if inpt.default is NO_DEFAULT:
                inpt.default = default
        elif inspect.isclass(inpt) or ty.get_origin(inpt):
            inputs[inpt_name] = arg_type(type=inpt, default=default)
        else:
            raise ValueError(
                f"Unrecognised input type ({inpt}) for input {inpt_name} with default "
                f"value {default}"
            )
    return_type = type_hints.get("return", ty.Any)
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
        else:
            output_types = {o: ty.Any for o in outputs}
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


def _get_attrs_kwargs(field: Field) -> dict[str, ty.Any]:
    kwargs = {}
    if field.default is not NO_DEFAULT:
        kwargs["default"] = field.default
    # elif is_optional(field.type):
    #     kwargs["default"] = None
    else:
        kwargs["factory"] = nothing_factory
    if field.hash_eq:
        kwargs["eq"] = hash_function
    return kwargs


def nothing_factory():
    return attrs.NOTHING


# def set_none_default_if_optional(field: Field) -> None:
#     if is_optional(field.type) and field.default is NO_DEFAULT:
#         field.default = None


white_space_re = re.compile(r"\s+")
