import typing as ty
import enum
from typing import Self
import attrs.validators
from attrs.converters import default_if_none
from fileformats.core import to_mime
from fileformats.generic import File, FileSet
from pydra.utils.typing import (
    TypeParser,
    is_truthy_falsy,
    is_type,
    is_union,
)
from pydra.utils.general import get_fields, wrap_text
import attrs

if ty.TYPE_CHECKING:
    from .task import Task


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
    if isinstance(self_, Out) and TypeParser.contains_type(FileSet, self_.type):
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

    def satisfied(self, inputs: "Task") -> bool:
        """Check if the requirement is satisfied by the inputs"""
        value = getattr(inputs, self.name)
        field = {f.name: f for f in get_fields(inputs)}[self.name]
        if value is None or field.type is bool and value is False:
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
        elif isinstance(value, tuple):
            name, allowed_values = value
            if isinstance(allowed_values, str) or not isinstance(
                allowed_values, ty.Collection
            ):
                raise ValueError(
                    f"allowed_values must be a collection of strings, not {allowed_values}"
                )
            return Requirement(name, allowed_values)
        else:
            raise ValueError(
                f"Requirements must be a input field name, a tuple of an input field "
                f"name and allowed values or a Requirement object, not {value!r}"
            )

    def __str__(self):
        if not self.allowed_values:
            return self.name
        return f"{self.name}(" + ",".join(repr(v) for v in self.allowed_values) + ")"


def requirements_converter(value: ty.Any) -> list[Requirement]:
    """Ensure the requires field is a list of Requirement objects"""
    if isinstance(value, Requirement):
        return [value]
    elif isinstance(value, (str, tuple)):
        try:
            return [Requirement.parse(value)]
        except ValueError as e:
            e.add_note(
                f"Parsing requirements specification {value!r} as a single requirement"
            )
            raise e
    try:
        return [Requirement.parse(v) for v in value]
    except ValueError as e:
        e.add_note(
            f"Parsing requirements specification {value!r} as a set of concurrent "
            "requirements (i.e. logical AND)"
        )
        raise e


@attrs.define
class RequirementSet:
    """Define a set of requirements for a task input field, all of which must be satisfied"""

    requirements: list[Requirement] = attrs.field(
        factory=list,
        converter=requirements_converter,
    )

    def satisfied(self, inputs: "Task") -> bool:
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
        try:
            return [RequirementSet(value)]
        except ValueError as e:
            e.add_note(
                f"Parsing requirements set specification {value!r} as a single requirement set"
            )
            raise e
    try:
        return [RequirementSet(v) for v in value]
    except ValueError as e:
        e.add_note(
            f"Parsing requirements set specification {value!r} as a set of alternative "
            "requirements (i.e. logical OR)"
        )
        raise e


@attrs.define(kw_only=True)
class Field:
    """Base class for input and output fields to tasks

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

    def requirements_satisfied(self, inputs: "Task") -> bool:
        """Check if all the requirements are satisfied by the inputs"""
        return any(req.satisfied(inputs) for req in self.requires)

    @property
    def mandatory(self):
        return self.default is NO_DEFAULT

    @requires.validator
    def _requires_validator(self, _, value):
        if value and not is_truthy_falsy(self.type):
            raise ValueError(
                f"Fields with requirements must be of optional (i.e. in union "
                f"with None) or truthy/falsy type, not type {self.type} ({self!r})"
            )

    def markdown_listing(
        self, line_width: int = 79, help_indent: int = 4, **kwargs
    ) -> str:
        """Get the listing for the field in markdown-like format

        Parameters
        ----------
        line_width: int
            The maximum line width for the output, by default it is 79
        help_indent: int
            The indentation for the help text, by default it is 4

        Returns
        -------
        str
            The listing for the field in markdown-like format
        """

        def type_to_str(type_: ty.Type[ty.Any]) -> str:
            if type_ is type(None):
                return "None"
            if is_union(type_):
                return " | ".join(
                    type_to_str(t) for t in ty.get_args(type_) if t is not None
                )
            try:
                type_str = to_mime(type_, official=False)
            except Exception:
                if origin := ty.get_origin(type_):
                    type_str = f"{origin.__name__}[{', '.join(map(type_to_str, ty.get_args(type_)))}]"
                else:
                    try:
                        type_str = type_.__name__
                    except AttributeError:
                        type_str = str(type_)
            return type_str

        s = f"- {self.name}: {type_to_str(self.type)}"
        if isinstance(self.default, attrs.Factory):
            s += f"; default-factory = {self.default.factory.__name__}()"
        elif callable(self.default):
            s += f"; default = {self.default.__name__}()"
        elif not self.mandatory:
            s += f"; default = {self.default!r}"
        if self._additional_descriptors(**kwargs):
            s += f" ({', '.join(self._additional_descriptors(**kwargs))})"
        if self.help:
            s += f"\n{wrap_text(self.help, width=line_width, indent_size=help_indent)}"
        return s

    def _additional_descriptors(self, **kwargs) -> list[str]:
        """Get additional descriptors for the field"""
        return []

    def __lt__(self, other: "Field") -> bool:
        """Compare two fields based on their position"""
        return self.name < other.name


@attrs.define(kw_only=True)
class Arg(Field):
    """Base class for input fields of tasks

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
    requires: list, optional
        Names of the inputs that are required together with the field.
    allowed_values: Sequence, optional
        List of allowed values for the field.
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

    allowed_values: frozenset = attrs.field(factory=frozenset, converter=frozenset)
    copy_mode: File.CopyMode = File.CopyMode.any
    copy_collation: File.CopyCollation = File.CopyCollation.any
    copy_ext_decomp: File.ExtensionDecomposition = File.ExtensionDecomposition.single
    readonly: bool = False

    def _additional_descriptors(self, **kwargs) -> list[str]:
        """Get additional descriptors for the field"""
        descriptors = super()._additional_descriptors(**kwargs)
        if self.allowed_values:
            descriptors.append(f"allowed_values={self.allowed_values}")
        return descriptors


@attrs.define(kw_only=True, slots=False)
class Out(Field):
    """Base class for output fields of tasks

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
