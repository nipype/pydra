import typing as ty
import types
from pathlib import Path
from copy import copy
from pydra.utils.typing import TypeParser, is_optional, is_fileset_or_union
import attrs
from .task import Task, Outputs
from pydra.utils.hash import hash_function
from pydra.utils.general import (
    from_list_if_single,
    ensure_list,
    PYDRA_ATTR_METADATA,
)
from pydra.utils.typing import (
    MultiInputObj,
    MultiInputFile,
    MultiOutputObj,
    MultiOutputFile,
    is_lazy,
)
from .field import Field, Arg, Out
from .helpers import sanitize_xor


def build_task_class(
    spec_type: type["Task"],
    out_type: type["Outputs"],
    inputs: dict[str, Arg],
    outputs: dict[str, Out],
    klass: type | None = None,
    name: str | None = None,
    bases: ty.Sequence[type] = (),
    outputs_bases: ty.Sequence[type] = (),
    xor: ty.Sequence[str | None] | ty.Sequence[ty.Sequence[str | None]] = (),
):
    """Create a task class and its outputs class from the
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
        The base classes for the task class, by default ()
    outputs_bases : ty.Sequence[type], optional
        The base classes for the outputs class, by default ()
    xor: Sequence[str | None] | Sequence[Sequence[str | None]], optional
        Names of args that are exclusive mutually exclusive, which must include
        the name of the current field. If this list includes None, then none of the
        fields need to be set.

    Returns
    -------
    klass : type
        The class created using the attrs package
    """
    xor = sanitize_xor(xor)
    spec_type._check_arg_refs(inputs, outputs, xor)

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
    outputs_klass = build_outputs_class(out_type, outputs, outputs_bases, name)
    if klass is None:
        if name is None:
            raise ValueError("name must be provided if klass is not")
        bases = tuple(bases)
        # Ensure that Task is a base class
        if not any(issubclass(b, spec_type) for b in bases):
            bases = bases + (spec_type,)
        # If building from a decorated class (as opposed to dynamically from a function
        # or shell-template), add any base classes not already in the bases tuple
        if klass is not None:
            bases += tuple(c for c in klass.__mro__ if c not in bases + (object,))
        # Create a new class with the Task as a base class
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
                    if arg.mandatory:  # provide default if one is not provided
                        attrs_kwargs["default"] = True if arg.requires else None
                        del attrs_kwargs["factory"]
                else:
                    field_type = Path | bool
                    if arg.mandatory:  # provide default if one is not provided
                        attrs_kwargs["default"] = True  # use the template by default
                        del attrs_kwargs["factory"]
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
        # Store the xor sets for the class
        klass._xor = xor
        klass.__annotations__[arg.name] = field_type

    # Create class using attrs package, will create attributes for all columns and
    # parameters
    attrs_klass = attrs.define(auto_attribs=False, kw_only=True, eq=False, repr=False)(
        klass
    )

    return attrs_klass


def build_outputs_class(
    spec_type: type["Outputs"],
    outputs: dict[str, Out],
    bases: ty.Sequence[type],
    spec_name: str,
) -> type["Outputs"]:
    """Create an outputs class and its outputs class from the
    output fields provided to the decorator/function.

    Creates a new class with attrs fields and then calls `attrs.define` to create an
    attrs class (dataclass-like).

    Parameters
    ----------
    outputs : dict[str, Out]
        The output fields of the task
    bases : ty.Sequence[type], optional
        The base classes for the outputs class, by default ()
    spec_name : str
        The name of the task class the outputs are for

    Returns
    -------
    klass : type
        The class created using the attrs package
    """

    if not any(issubclass(b, spec_type) for b in bases):
        if out_spec_bases := [b for b in bases if issubclass(b, Outputs)]:
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
    if field_type in (MultiInputObj, MultiInputFile):
        converters.append(ensure_list)
    elif field_type in (MultiOutputObj, MultiOutputFile):
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


def _get_attrs_kwargs(field: Field) -> dict[str, ty.Any]:
    kwargs = {}
    if not field.mandatory:
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
