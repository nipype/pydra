import itertools
import inspect
from pathlib import Path
import collections.abc
import os
from copy import copy
import sys
import types
import typing as ty
import logging
import attrs
from fileformats import field, core, generic
from pydra.utils.general import add_exc_note
from pydra.utils.mount_identifier import MountIndentifier

try:
    from typing import get_origin, get_args
except ImportError:
    # Python < 3.8
    from typing import get_origin, get_args  # type: ignore

if sys.version_info >= (3, 10):
    UNION_TYPES = (ty.Union, types.UnionType)
else:
    UNION_TYPES = (ty.Union,)

if ty.TYPE_CHECKING:
    from pydra.engine.lazy import LazyField

logger = logging.getLogger("pydra")

NO_GENERIC_ISSUBCLASS = sys.version_info.major == 3 and sys.version_info.minor < 10

if NO_GENERIC_ISSUBCLASS:
    from typing_utils import issubtype

try:
    import numpy
except ImportError:
    HAVE_NUMPY = False
else:
    HAVE_NUMPY = True

T = ty.TypeVar("T")
TypeOrAny = ty.Union[type, ty.Any]


# These are special types that are checked for in the construction of input/output specs
# and special converters inserted into the attrs fields.


class MultiInputObj(list, ty.Generic[T]):
    pass


MultiInputFile = MultiInputObj[generic.File]


# Since we can't create a NewType from a type union, we add a dummy type to the union
# so we can detect the MultiOutput in the input/output task creation
class MultiOutputType:
    pass


MultiOutputObj = ty.Union[list, object, MultiOutputType]
MultiOutputFile = ty.Union[generic.File, ty.List[generic.File], MultiOutputType]

OUTPUT_TEMPLATE_TYPES = (
    Path,
    ty.List[Path],
    ty.Union[Path, bool],
    ty.Union[ty.List[Path], bool],
    ty.List[ty.List[Path]],
)


class StateArray(ty.List[T]):
    """an array of values from, or to be split over in an array of nodes (see TaskBase.split()),
    multiple nodes of the same task. Used in type-checking to differentiate between list
    types and values for multiple nodes
    """

    def __repr__(self):
        return f"{type(self).__name__}(" + ", ".join(repr(i) for i in self) + ")"


class TypeParser(ty.Generic[T]):
    """A callable which can be used as a converter for attrs.fields to check whether an
    object or LazyField matches the specified field type, or can be
    coerced into it (given the criteria passed on initialisation of the checker).
    Nested container type are expanded and each of their type args are checked/coerced
    against corresponding parts of the object.

    Parameters
    ----------
    tp : type
        the type objects will be coerced to
    coercible: Iterable[ty.Tuple[type or Any, type or Any]], optional
        limits coercing between the pairs of types where they appear within the
        tree of more complex nested container types. If None, then all types are
        coercible except explicitly excluded
    not_coercible: Iterable[ty.Tuple[type or Any, type or Any]], optional
        excludes the limits coercing between the pairs of types where they appear within
        the tree of more complex nested container types. Overrides 'coercible' to enable
        you to carve out exceptions, such as TypeParser(list, coercible=[(ty.Iterable, list)],
        not_coercible=[(str, list)])
    superclass_auto_cast : bool
        Allow lazy fields to pass the type check if their types are superclasses of the
        specified pattern (instead of matching or being subclasses of the pattern)
    label : str
        the label to be used to identify the type parser in error messages. Especially
        useful when TypeParser is used as a converter in attrs.fields
    match_any_of_union : bool
        match if any of the options in the union are a subclass (but not necessarily all)
    """

    tp: ty.Type[T]
    coercible: ty.List[ty.Tuple[TypeOrAny, TypeOrAny]]
    not_coercible: ty.List[ty.Tuple[TypeOrAny, TypeOrAny]]
    superclass_auto_cast: bool
    label: str
    match_any_of_union: bool

    COERCIBLE_DEFAULT: ty.Tuple[ty.Tuple[type, type], ...] = (
        (
            (ty.Sequence, ty.Sequence),
            (ty.Sequence, collections.abc.Set),
            (collections.abc.Set, ty.Sequence),
            (ty.Mapping, ty.Mapping),
            (Path, os.PathLike),
            (str, os.PathLike),
            (os.PathLike, Path),
            (os.PathLike, str),
            (ty.Any, MultiInputObj),
            (int, float),
            (field.Integer, float),
            (int, field.Decimal),
        )
        + tuple((f, f.primitive) for f in field.Singular.subclasses() if f.primitive)
        + tuple((f.primitive, f) for f in field.Singular.subclasses() if f.primitive)
    )

    if HAVE_NUMPY:
        COERCIBLE_DEFAULT += (
            (numpy.integer, int),
            (numpy.floating, float),
            (numpy.bool_, bool),
            (numpy.integer, float),
            (numpy.character, str),
            (numpy.complexfloating, complex),
            (numpy.bytes_, bytes),
            (numpy.ndarray, ty.Sequence),
            (ty.Sequence, numpy.ndarray),
        )

    NOT_COERCIBLE_DEFAULT = ((str, ty.Sequence), (ty.Sequence, str))

    def __init__(
        self,
        tp,
        coercible: ty.Optional[
            ty.Iterable[ty.Tuple[TypeOrAny, TypeOrAny]]
        ] = COERCIBLE_DEFAULT,
        not_coercible: ty.Optional[
            ty.Iterable[ty.Tuple[TypeOrAny, TypeOrAny]]
        ] = NOT_COERCIBLE_DEFAULT,
        superclass_auto_cast: bool = False,
        label: str = "",
        match_any_of_union: bool = False,
    ):
        def expand_pattern(t):
            """Recursively expand the type arguments of the target type in nested tuples"""
            if t is inspect._empty:
                return None
            origin = get_origin(t)
            if origin is None:
                return t
            args = get_args(t)
            if not args or args == (Ellipsis,):  # Not sure Ellipsis by itself is valid
                # If no args were provided, or those arguments were an ellipsis
                assert isinstance(origin, type)
                return origin
            if origin not in UNION_TYPES + (type,) and not issubclass(
                origin, ty.Iterable
            ):
                raise TypeError(
                    f"TypeParser doesn't know how to handle args ({args}) for {origin} "
                    f"types{self.label_str}"
                )
            return (origin, [expand_pattern(a) for a in args])

        self.label = label
        self.tp = tp
        self.coercible = (
            list(coercible) if coercible is not None else [(ty.Any, ty.Any)]
        )
        self.not_coercible = list(not_coercible) if not_coercible is not None else []
        self.pattern = expand_pattern(tp)
        self.superclass_auto_cast = superclass_auto_cast
        self.match_any_of_union = match_any_of_union

    def __call__(self, obj: ty.Any) -> T:
        """Attempts to coerce the object to the specified type, unless the value is
        a LazyField where the type of the field is just checked instead or an
        attrs.NOTHING where it is simply returned.

        Parameters
        ----------
        obj : ty.Any
            the object to coerce/check-type

        Returns
        -------
        T
            the coerced object

        Raises
        ------
        TypeError
            if the coercion is not possible, or not specified by the
            `coercible`/`not_coercible` parameters, then a TypeError is raised
        """

        coerced: T
        if obj is attrs.NOTHING:
            coerced = attrs.NOTHING  # type: ignore[assignment]
        elif is_lazy(obj):
            try:
                self.check_type(obj._type)
            except TypeError as e:
                if self.superclass_auto_cast:
                    try:
                        # Check whether the type of the lazy field isn't a superclass of
                        # the type to check against, and if so, allow it due to permissive
                        # typing rules.
                        TypeParser(obj._type, match_any_of_union=True).check_type(
                            self.tp
                        )
                    except TypeError:
                        raise TypeError(
                            f"Incorrect type for lazy field{self.label_str}: {obj._type!r} "
                            f"is not a subclass or superclass of {self.tp} (and will not "
                            "be able to be coerced to one that is)"
                        ) from e
                    else:
                        logger.info(
                            "Connecting lazy field %s to %s%s via permissive typing that "
                            "allows super-to-sub type connections",
                            obj,
                            self.tp,
                            self.label_str,
                        )
                else:
                    raise TypeError(
                        f"Incorrect type for lazy field{self.label_str}: {obj._type!r} "
                        f"is not a subclass of {self.tp} (and will not be able to be "
                        "coerced to one that is)"
                    ) from e
            coerced = obj  # type: ignore
            if obj._type is not ty.Any:
                # Used to check whether the type of the field can be changed
                obj._type_checked = True
        elif isinstance(obj, StateArray):
            coerced = StateArray(self(o) for o in obj)  # type: ignore[assignment]
        else:
            try:
                coerced = self.coerce(obj)
            except TypeError as e:
                if obj is None:
                    raise TypeError(
                        f"Mandatory field{self.label_str} of type {self.tp} was not "
                        "provided a value (i.e. a value that wasn't None) "
                    ) from None
                raise TypeError(
                    f"Incorrect type for field{self.label_str}: {obj!r} is not of type "
                    f"{self.tp} (and cannot be coerced to it)"
                ) from e
        return coerced

    def coerce(self, object_: ty.Any) -> T:
        """Attempts to coerce the given object to the type of the specified type"""
        if self.pattern is None:
            return object_

        def expand_and_coerce(obj, pattern: ty.Union[type, tuple]):
            """Attempt to expand the object along the lines of the coercion pattern"""
            if obj is attrs.NOTHING:
                return attrs.NOTHING
            if not isinstance(pattern, tuple):
                return coerce_basic(obj, pattern)
            origin, pattern_args = pattern
            if origin == MultiInputObj:
                return coerce_multi_input(obj, pattern_args)
            if origin in UNION_TYPES:
                return coerce_union(obj, pattern_args)
            if origin is type:
                return coerce_type(obj, pattern_args)
            if not self.is_instance(obj, origin):
                self.check_coercible(obj, origin)
                type_ = origin
            else:
                type_ = type(obj)
            if issubclass(type_, ty.Mapping):
                return coerce_mapping(obj, type_, pattern_args)
            try:
                obj_args = list(obj)
            except TypeError as e:
                msg = (
                    f" (part of coercion from {object_!r} to {self.pattern}"
                    if obj is not object_
                    else ""
                )
                raise TypeError(
                    f"Could not coerce to {type_} as {obj!r} is not iterable{msg}{self.label_str}"
                ) from e
            if issubclass(origin, tuple):
                return coerce_tuple(type_, obj_args, pattern_args)
            if issubclass(origin, ty.Iterable):
                return coerce_sequence(type_, obj_args, pattern_args)
            assert False, f"Coercion from {obj!r} to {pattern} is not handled"

        def coerce_basic(obj, pattern):
            """Coerce an object to a "basic types" like `int`, `float`, `bool`, `Path`
            and `File` in contrast to compound types like `list[int]`,
            `dict[str, str]` and `dict[str, list[int]]`"""
            if self.is_instance(obj, pattern):
                return obj
            self.check_coercible(obj, pattern)
            return coerce_obj(obj, pattern)

        def coerce_union(obj, pattern_args):
            """Coerce an object into the first type in a Union construct that it is
            coercible into"""
            reasons = []
            for arg in pattern_args:
                try:
                    return expand_and_coerce(obj, arg)
                except TypeError as e:
                    reasons.append(e)
            raise TypeError(
                f"Could not coerce object, {obj!r}, to any of the union types "
                f"{pattern_args}{self.label_str}:\n\n"
                + "\n\n".join(f"{a} -> {e}" for a, e in zip(pattern_args, reasons))
            )

        def coerce_multi_input(obj, pattern_args):
            # Attempt to coerce the object into arg type of the MultiInputObj first,
            # and if that fails, try to coerce it into a list of the arg type
            try:
                return coerce_sequence(list, obj, pattern_args)
            except TypeError as e1:
                try:
                    return [expand_and_coerce(obj, pattern_args[0])]
                except TypeError as e2:
                    raise TypeError(
                        f"Could not coerce object ({obj!r}) to MultiInputObj[{pattern_args[0]}] "
                        f"either as sequence of {pattern_args[0]} ({e1}) or a single {pattern_args[0]} "
                        f"object to be wrapped in a list {e2}"
                    ) from e2

        def coerce_mapping(
            obj: ty.Mapping, type_: ty.Type[ty.Mapping], pattern_args: list
        ):
            """Coerce a mapping (e.g. dict)"""
            key_pattern, val_pattern = pattern_args
            items: ty.Iterable[ty.Tuple[ty.Any, ty.Any]]
            try:
                items = obj.items()
            except AttributeError as e:
                msg = (
                    f" (part of coercion from {object_} to {self.pattern}"
                    if obj is not object_
                    else ""
                )
                raise TypeError(
                    f"Could not coerce to {type_} as {obj} is not a mapping type{msg}{self.label_str}"
                ) from e
            return coerce_obj(
                {
                    expand_and_coerce(k, key_pattern): expand_and_coerce(v, val_pattern)
                    for k, v in items
                },
                type_,
            )

        def coerce_tuple(
            type_: ty.Type[ty.Sequence],
            obj_args: list,
            pattern_args: list,
        ):
            """coerce to a tuple object"""
            if pattern_args[-1] is Ellipsis:
                pattern_args = itertools.chain(  # type: ignore[assignment]
                    pattern_args[:-2], itertools.repeat(pattern_args[-2])
                )
            elif len(pattern_args) != len(obj_args):
                raise TypeError(
                    f"Incorrect number of items in tuple, expected "
                    f"{len(pattern_args)}, got {len(obj_args)}{self.label_str}"
                )
            return coerce_obj(
                [expand_and_coerce(o, p) for o, p in zip(obj_args, pattern_args)], type_
            )

        def coerce_sequence(
            type_: ty.Type[ty.Sequence], obj_args: list, pattern_args: list
        ):
            """Coerce a non-tuple sequence object (e.g. list, ...)"""
            assert len(pattern_args) == 1
            return coerce_obj(
                [expand_and_coerce(o, pattern_args[0]) for o in obj_args], type_
            )

        def coerce_type(type_: ty.Type[ty.Any], pattern_args: ty.List[ty.Type[ty.Any]]):
            if not any(issubclass(type_, t) for t in pattern_args):
                raise TypeError(
                    f"{type_} is not one of the specified types {pattern_args}{self.label_str}"
                )
            return type_

        def coerce_obj(obj, type_):
            """Attempt to do the innermost (i.e. non-nested) coercion and fail with
            helpful message
            """
            try:
                return type_(obj)
            except (TypeError, ValueError) as e:
                msg = (
                    f" (part of coercion from {object_} to {self.pattern}"
                    if obj is not object_
                    else ""
                )
                raise TypeError(
                    f"Cannot coerce {obj!r} into {type_}{msg}{self.label_str}"
                ) from e

        return expand_and_coerce(object_, self.pattern)

    def check_type(self, type_: ty.Type[ty.Any]):
        """Checks the given type to see whether it matches or is a subtype of the
        specified type or whether coercion rule is specified between the types

        Parameters
        ----------
        type_ : ty.Type[ty.Any]
            the type to check whether it is coercible into the specified type

        Raises
        ------
        TypeError
            if the type is not either the specified type, a sub-type or coercible to it
        """
        if self.pattern is None or type_ is ty.Any:
            return
        if self.is_subclass(type_, StateArray):
            args = get_args(type_)
            if not args:
                raise TypeError("Splits without any type arguments are invalid")
            if len(args) > 1:
                raise TypeError(
                    f"Splits with more than one type argument ({args}) are invalid{self.label_str}"
                )
            return self.check_type(args[0])

        def expand_and_check(tp, pattern: ty.Union[type, tuple]):
            """Attempt to expand the object along the lines of the coercion pattern"""
            if not isinstance(pattern, tuple):
                return check_basic(tp, pattern)
            pattern_origin, pattern_args = pattern
            if pattern_origin in UNION_TYPES:
                return check_union(tp, pattern_args)
            tp_origin = get_origin(tp)
            if tp_origin is None:
                if issubclass(tp, pattern_origin):
                    raise TypeError(
                        f"Type {tp} wasn't declared with type args required to match pattern "
                        f"{pattern_args}, when matching {type_} to {self.pattern}"
                    )
                raise TypeError(
                    f"{tp} doesn't match pattern {pattern}, when matching {type_} to "
                    f"{self.pattern}{self.label_str}"
                )
            tp_args = get_args(tp)
            self.check_type_coercible(tp_origin, pattern_origin)
            if issubclass(pattern_origin, ty.Mapping):
                return check_mapping(tp_args, pattern_args)
            if issubclass(pattern_origin, tuple):
                if not issubclass(tp_origin, tuple):
                    assert len(tp_args) == 1
                    tp_args += (Ellipsis,)
                return check_tuple(tp_args, pattern_args)
            return check_sequence(tp_args, pattern_args)

        def check_basic(tp, target):
            # Note that we are deliberately more permissive than typical type-checking
            # here, allowing parents of the target type as well as children,
            # to avoid users having to cast from loosely typed tasks to strict ones
            if self.match_any_of_union and get_origin(tp) is ty.Union:
                reasons = []
                tp_args = get_args(tp)
                for tp_arg in tp_args:
                    if self.is_subclass(tp_arg, target):
                        return
                    try:
                        self.check_coercible(tp_arg, target)
                    except TypeError as e:
                        reasons.append(e)
                    else:
                        return
                if reasons:
                    raise TypeError(
                        f"Cannot coerce any union args {tp_arg} to {target}"
                        f"{self.label_str}:\n\n"
                        + "\n\n".join(f"{a} -> {e}" for a, e in zip(tp_args, reasons))
                    )
            if not self.is_subclass(tp, target):
                self.check_type_coercible(tp, target)

        def check_union(tp, pattern_args):
            if get_origin(tp) in UNION_TYPES:
                tp_args = get_args(tp)
                for tp_arg in tp_args:
                    reasons = []
                    for pattern_arg in pattern_args:
                        try:
                            expand_and_check(tp_arg, pattern_arg)
                        except TypeError as e:
                            reasons.append(e)
                        else:
                            reasons = []
                            break
                    if self.match_any_of_union and len(reasons) < len(tp_args):
                        # Just need one of the union args to match
                        return
                    if reasons:
                        determiner = "any" if self.match_any_of_union else "all"
                        raise TypeError(
                            f"Cannot coerce {tp} to ty.Union["
                            f"{', '.join(str(a) for a in pattern_args)}]{self.label_str}, "
                            f"because {tp_arg} cannot be coerced to {determiner} of its args:\n\n"
                            + "\n\n".join(
                                f"{a} -> {e}" for a, e in zip(pattern_args, reasons)
                            )
                        )
                return
            reasons = []
            for pattern_arg in pattern_args:
                try:
                    return expand_and_check(tp, pattern_arg)
                except TypeError as e:
                    reasons.append(e)
            raise TypeError(
                f"Cannot coerce {tp} to any of the union types:\n\n"
                + "\n\n".join(f"{a} -> {e}" for a, e in zip(pattern_args, reasons))
            )

        def check_mapping(tp_args, pattern_args):
            key_pattern, val_pattern = pattern_args
            key_tp, val_tp = tp_args
            expand_and_check(key_tp, key_pattern)
            expand_and_check(val_tp, val_pattern)

        def check_tuple(tp_args, pattern_args):
            if pattern_args[-1] is Ellipsis:
                if len(pattern_args) == 1:  # matches anything
                    return
                if tp_args[-1] is Ellipsis:
                    return expand_and_check(tp_args[0], pattern_args[0])
                for arg in tp_args:
                    expand_and_check(arg, pattern_args[0])
                return
            elif tp_args[-1] is Ellipsis:
                for pattern_arg in pattern_args:
                    expand_and_check(tp_args[0], pattern_arg)
                return
            if len(tp_args) != len(pattern_args):
                raise TypeError(
                    f"Wrong number of type arguments in tuple {tp_args}  compared to pattern "
                    f"{pattern_args} in attempting to match {type_} to {self.pattern}{self.label_str}"
                )
            for t, p in zip(tp_args, pattern_args):
                expand_and_check(t, p)

        def check_sequence(tp_args, pattern_args):
            assert len(pattern_args) == 1
            if tp_args[-1] is Ellipsis:
                tp_args = tp_args[:-1]
                if not tp_args:
                    raise TypeError(
                        "Generic ellipsis type arguments not specific enough to match "
                        f"{pattern_args} in attempting to match {type_} to "
                        f"{self.pattern}{self.label_str}"
                    )
            for arg in tp_args:
                expand_and_check(arg, pattern_args[0])

        try:
            return expand_and_check(type_, self.pattern)
        except TypeError as e:
            # Special handling for MultiInputObjects (which are annoying)
            if not isinstance(self.pattern, tuple) or self.pattern[0] != MultiInputObj:
                raise e
            # Attempt to coerce the object into arg type of the MultiInputObj first,
            # and if that fails, try to coerce it into a list of the arg type
            inner_type_parser = copy(self)
            inner_type_parser.pattern = self.pattern[1][0]
            try:
                inner_type_parser.check_type(type_)
            except TypeError:
                add_exc_note(
                    e,
                    "Also failed to coerce to the arg-type of the MultiInputObj "
                    f"({self.pattern[1][0]})",
                )
                raise e

    def check_coercible(self, source: ty.Any, target: ty.Union[type, ty.Any]):
        """Checks whether the source object is coercible to the target type given the coercion
        rules defined in the `coercible` and `not_coercible` attrs

        Parameters
        ----------
        source : object
            the object to be coerced
        target : type or typing.Any
            the target type for the object to be coerced to

        Raises
        ------
        TypeError
            If the object cannot be coerced into the target type depending on the explicit
            inclusions and exclusions set in the `coercible` and `not_coercible` member attrs
        """
        if (
            isinstance(source, ty.Sequence)
            and issubclass(target, core.FileSet)
            and all(isinstance(p, os.PathLike) for p in source)
        ):
            return True
        self.check_type_coercible(type(source), target, source_repr=repr(source))

    def check_type_coercible(
        self,
        source: ty.Union[type, ty.Any],
        target: ty.Union[type, ty.Any],
        source_repr: ty.Optional[str] = None,
    ):
        """Checks whether the source type is coercible to the target type
        given the coercion rules defined in the `coercible` and `not_coercible` attrs

        Parameters
        ----------
        source : type or typing.Any
            source type to be coerced
        target : type or typing.Any
            target type for the source to be coerced to

        Raises
        ------
        TypeError
            If the source type cannot be coerced into the target type depending on the
            explicit inclusions and exclusions set in the `coercible` and `not_coercible`
            member attrs
        """
        if source_repr is None:
            source_repr = repr(source)
        # Short-circuit the basic cases where the source and target are the same
        if source is target:
            return
        if self.superclass_auto_cast and self.is_subclass(target, source):
            logger.info(
                "Attempting to coerce %s into %s due to super-to-sub class coercion "
                "being permitted",
                source,
                target,
            )
            return
        source_origin = get_origin(source)
        if source_origin is not None:
            source = source_origin

        def matches_criteria(criteria):
            return [
                (src, tgt)
                for src, tgt in criteria
                if self.is_subclass(source, src) and self.is_subclass(target, tgt)
            ]

        def type_name(t):
            try:
                return t.__name__
            except AttributeError:
                return t._name  # typing generics for Python < 3.10

        if not matches_criteria(self.coercible):
            raise TypeError(
                f"Cannot coerce {source_repr} into {target}{self.label_str} as the "
                "coercion doesn't match any of the explicit inclusion criteria: "
                + ", ".join(
                    f"{type_name(s)} -> {type_name(t)}" for s, t in self.coercible
                )
            )
        matches_not_coercible = matches_criteria(self.not_coercible)
        if matches_not_coercible:
            raise TypeError(
                f"Cannot coerce {source_repr} into {target}{self.label_str} as it is explicitly "
                "excluded by the following coercion criteria: "
                + ", ".join(
                    f"{type_name(s)} -> {type_name(t)}"
                    for s, t in matches_not_coercible
                )
            )

    @classmethod
    def matches(cls, obj: ty.Type[ty.Any], target: ty.Type[ty.Any], **kwargs) -> bool:
        """Returns true if the provided type matches the pattern of the TypeParser

        Parameters
        ----------
        type_ : type
            the type to check
        target : type
            the target type to check against
        **kwargs : dict[str, Any], optional
            passed on to TypeParser.__init__

        Returns
        -------
        matches : bool
            whether the type matches the target type factoring in sub-classes and coercible
            pairs
        """
        parser = cls(target, **kwargs)
        try:
            parser.coerce(obj)
        except TypeError:
            return False
        return True

    @classmethod
    def matches_type(
        cls, type_: ty.Type[ty.Any], target: ty.Type[ty.Any], **kwargs
    ) -> bool:
        """Returns true if the provided type matches the pattern of the TypeParser

        Parameters
        ----------
        type_ : type
            the type to check
        target : type
            the target type to check against
        **kwargs : dict[str, Any], optional
            passed on to TypeParser.__init__

        Returns
        -------
        matches : bool
            whether the type matches the target type factoring in sub-classes and coercible
            pairs
        """
        parser = cls(target, **kwargs)
        try:
            parser.check_type(type_)
        except TypeError:
            return False
        return True

    @classmethod
    def is_instance(
        cls,
        obj: object,
        candidates: ty.Union[ty.Type[ty.Any], ty.Sequence[ty.Type[ty.Any]], None],
    ) -> bool:
        """Checks whether the object is an instance of cls or that cls is typing.Any,
        extending the built-in isinstance to check nested type args

        Parameters
        ----------
        obj: object
            the object to check whether it is an instance of one of the candidates
        candidates : type or ty.Iterable[type]
            the candidate types to check the object against
        """
        if candidates is None:
            candidates = [type(None)]
        if not isinstance(candidates, ty.Sequence):
            candidates = [candidates]
        for candidate in candidates:
            if candidate is ty.Any:
                return True
            # Handle ty.Type[*] candidates
            if ty.get_origin(candidate) is type:
                return inspect.isclass(obj) and cls.is_subclass(
                    obj, ty.get_args(candidate)[0]
                )
            if NO_GENERIC_ISSUBCLASS:
                if inspect.isclass(obj):
                    return candidate is type
                if issubtype(type(obj), candidate) or (
                    type(obj) is dict and candidate is ty.Mapping  # noqa: E721
                ):
                    return True
            else:
                if isinstance(obj, candidate):
                    return True
        return False

    @classmethod
    def is_subclass(
        cls,
        klass: ty.Type[ty.Any],
        candidates: ty.Union[ty.Type[ty.Any], ty.Sequence[ty.Type[ty.Any]]],
        any_ok: bool = False,
    ) -> bool:
        """Checks whether the class a is either the same as b, a subclass of b or b is
        typing.Any, extending built-in issubclass to check nested type args

        Parameters
        ----------
        klass : type
            the klass to check whether it is a subclass of one of the candidates
        candidates : type or ty.Iterable[type]
            the candidate types to check the object against
        any_ok : bool
            whether klass=typing.Any should return True or False
        """
        if klass is None:
            # Implicitly convert None to NoneType, like in other typing
            klass = type(None)
        if not isinstance(candidates, ty.Sequence):
            candidates = [candidates]
        if ty.Any in candidates:
            return True
        if klass is ty.Any:
            return any_ok

        origin = get_origin(klass)
        args = get_args(klass)

        for candidate in candidates:
            if candidate is None:
                candidate = type(None)
            candidate_origin = get_origin(candidate)
            candidate_args = get_args(candidate)
            # Handle ty.Type[*] types in klass and candidates
            if origin is type and (candidate is type or candidate_origin is type):
                if candidate is type:
                    return True
                return cls.is_subclass(args[0], candidate_args[0])
            elif origin is type or candidate_origin is type:
                return False
            if NO_GENERIC_ISSUBCLASS:
                if klass is type and candidate is not type:
                    return False
                if issubtype(klass, candidate) or (
                    klass is dict and candidate is ty.Mapping
                ):
                    return True
            else:
                if origin in UNION_TYPES:
                    union_args = (
                        candidate_args
                        if candidate_origin in UNION_TYPES
                        else (candidate,)
                    )
                    matches = all(
                        any(cls.is_subclass(a, c) for c in union_args) for a in args
                    )
                    if matches:
                        return True
                else:
                    if candidate_args and candidate_origin not in UNION_TYPES:
                        if (
                            origin
                            and issubclass(origin, candidate_origin)  # type: ignore[arg-type]
                            and len(args) == len(candidate_args)
                            and all(
                                issubclass(a, c) for a, c in zip(args, candidate_args)
                            )
                        ):
                            return True
                    else:
                        if issubclass(origin if origin else klass, candidate):
                            return True
        return False

    @classmethod
    def contains_type(cls, target: ty.Type[ty.Any], type_: ty.Type[ty.Any]):
        """Checks a potentially nested type for sub-classes of the target type

        Parameters
        ----------
        target : type
            the target type to check for sub-classes of
        type_: type
            the type to check for nested types that are sub-classes of target
        """
        if cls.is_subclass(type_, target):
            return True
        if type_ in (str, bytes, int, bool, float):  # shortcut primitive types
            return False
        type_args = get_args(type_)
        if not type_args:
            return False
        type_origin = get_origin(type_)
        if type_origin in UNION_TYPES:
            for type_arg in type_args:
                if cls.contains_type(target, type_arg):
                    return True
            return False
        if cls.is_subclass(type_origin, ty.Mapping):
            type_key, type_val = type_args
            return cls.contains_type(target, type_key) or cls.contains_type(
                target, type_val
            )
        if cls.is_subclass(type_, (ty.Sequence, MultiOutputObj)):
            if type_args[-1] == Ellipsis:
                type_args = type_args[:-1]
            return any(cls.contains_type(target, a) for a in type_args)
        return False

    @classmethod
    def apply_to_instances(
        cls,
        target_type: ty.Type[ty.Any],
        func: ty.Callable,
        value: ty.Any,
        cache: ty.Optional[ty.Dict[int, ty.Any]] = None,
    ) -> ty.Any:
        """Applies a function to all instances of the given type that are potentially
        nested within the given value, caching previously computed modifications to
        handle repeated elements

        Parameters
        ----------
        target_type : type
            the target type to apply the function to
        func : callable
            the callable object (e.g. function) to apply to the instances
        value : Any
            the value to copy files from (if required)
        cache: dict, optional
            guards against multiple references to the same objects by keeping a cache of
            the modified
        """
        if (
            not cls.is_instance(value, (target_type, ty.Mapping, ty.Sequence))
            or target_type is not str
            and cls.is_instance(value, str)
        ):
            return value
        if cache is None:
            cache = {}
        obj_id = id(value)
        try:
            return cache[obj_id]
        except KeyError:
            pass
        if cls.is_instance(value, target_type):
            modified = func(value)
        elif cls.is_instance(value, ty.Mapping):
            modified = type(value)(  # type: ignore
                (
                    cls.apply_to_instances(target_type, func, key),
                    cls.apply_to_instances(target_type, func, val),
                )
                for (key, val) in value.items()
            )
        else:
            assert cls.is_instance(value, (ty.Sequence, MultiOutputObj))
            args = [cls.apply_to_instances(target_type, func, val) for val in value]
            modified = type(value)(args)  # type: ignore
        cache[obj_id] = modified
        return modified

    @classmethod
    def get_item_type(
        cls, sequence_type: ty.Type[ty.Sequence[T]]
    ) -> ty.Union[ty.Type[T], ty.Any]:
        """Return the type of the types of items in a sequence type

        Parameters
        ----------
        sequence_type: type[Sequence]
            the type to find the type of the items of

        Returns
        -------
        item_type: type or None
            the type of the items
        """
        if not TypeParser.is_subclass(sequence_type, ty.Sequence):
            raise TypeError(
                f"Cannot get item type from {sequence_type}, as it is not a sequence type"
            )
        args = get_args(sequence_type)
        if not args:
            return ty.Any
        if len(args) > 1 and not (len(args) == 2 and args[-1] == Ellipsis):
            raise TypeError(
                f"Cannot get item type from {sequence_type}, as it has multiple "
                f"item types: {args}"
            )
        return args[0]

    @classmethod
    def strip_splits(cls, type_: ty.Type[ty.Any]) -> ty.Tuple[ty.Type, int]:
        """Strips any StateArray types from the outside of the specified type and returns
        the stripped type and the depth it was found at

        Parameters
        ----------
        type_ : ty.Type[ty.Any]
            the type to list the nested sequences of

        Returns
        -------
        inner_type : type
            the inner type once all outer sequences are stripped
        depth : int
            the number of splits outside the inner_type
        """
        depth = 0
        while cls.is_subclass(type_, StateArray) and not cls.is_subclass(type_, str):
            origin = get_origin(type_)
            # If type is a union, pick the first sequence type in the union
            if origin in UNION_TYPES:
                for tp in get_args(type_):
                    if cls.is_subclass(tp, ty.Sequence):
                        type_ = tp
                        break
            type_ = cls.get_item_type(type_)
            depth += 1
        return type_, depth

    @property
    def label_str(self):
        return f" in {self.label} " if self.label else ""

    get_origin = staticmethod(get_origin)
    get_args = staticmethod(get_args)


def is_union(type_: type, args: list[type] = None) -> bool:
    """Checks whether a type is a Union, in either ty.Union[T, U] or T | U form

    Parameters
    ----------
    type_ : type
        the type to check
    args : list[type], optional
        required arguments of the union to check, by default (None) any args will match

    Returns
    -------
    is_union : bool
        whether the type is a Union type
    """
    if ty.get_origin(type_) in UNION_TYPES:
        if args is not None:
            return ty.get_args(type_) == args
        return True
    return False


def is_optional(type_: type) -> bool:
    """Check if the type is Optional, i.e. a union containing None"""
    if is_union(type_):
        return any(a is type(None) or is_optional(a) for a in ty.get_args(type_))
    return False


def is_container(type_: type) -> bool:
    """Check if the type is a container, i.e. a list, tuple, or MultiOutputObj"""
    origin = ty.get_origin(type_)
    if origin is ty.Union:
        return all(is_container(a) for a in ty.get_args(type_))
    tp = origin if origin else type_
    return inspect.isclass(tp) and issubclass(tp, ty.Container)


def is_truthy_falsy(type_: type) -> bool:
    """Check if the type is a truthy type, i.e. not None, bool, or typing.Any"""
    return (
        type_ in (ty.Any, bool, int, str)
        or is_optional(type_)
        or is_container(type_)
        or hasattr(type_, "__bool__")
        or hasattr(type_, "__len__")
    )


def optional_type(type_: type) -> type:
    """Gets the non-None args of an optional type (i.e. a union with a None arg)"""
    if is_optional(type_):
        non_none = [a for a in ty.get_args(type_) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
        return ty.Union[tuple(non_none)]
    return type_


def is_multi_input(type_: type) -> bool:
    """Check if the type is a MultiInputObj"""
    type_ = optional_type(type_)
    return MultiInputObj in (type_, ty.get_origin(type_))


def is_fileset_or_union(type_: type, allow_none: bool | None = None) -> bool:
    """Check if the type is a FileSet or a Union containing a FileSet

    Parameters
    ----------
    type_ : type
        the type to check
    allow_none : bool, optional
        whether to allow None as a valid type, by default None. If None, then None
        is not allowed at the outer layer, but is allowed within a Union

    Returns
    -------
    is_fileset : bool
        whether the type is a FileSet or a Union containing a FileSet
    """
    return is_subclass_or_union(type_, core.FileSet, allow_none=allow_none)


def is_subclass_or_union(
    type_: type, reference: type, allow_none: bool | None = None
) -> bool:
    """Check if the type is a subclass of given reference or a Union containing
    that reference type

    Parameters
    ----------
    type_ : type
        the type to check
    reference : type
        the reference type to check whether the type is a sub-class of or not
    allow_none : bool, optional
        whether to allow None as a valid type, by default None. If None, then None
        is not allowed at the outer layer, but is allowed within a Union

    Returns
    -------
    bool
        whether the type is a FileSet or a Union containing a FileSet
    """
    if type_ is None and allow_none:
        return True
    if is_union(type_):
        return any(
            is_subclass_or_union(
                t, reference, allow_none=allow_none or allow_none is None
            )
            for t in ty.get_args(type_)
        )
    elif not inspect.isclass(type_):
        return False
    return issubclass(type_, reference)


def is_type(*args: ty.Any) -> bool:
    """check that the value is a type or generic"""
    if len(args) == 3:  # attrs validator
        val = args[2]
    elif len(args) != 1:
        raise TypeError(f"is_type() takes 1 or 3 arguments, not {args}")
    else:
        val = args[0]
    return inspect.isclass(val) or ty.get_origin(val)


T = ty.TypeVar("T")
U = ty.TypeVar("U")


def state_array_support(
    function: ty.Callable[T, U],
) -> ty.Callable[T | StateArray[T], U | StateArray[U]]:
    """
    Decorator to convert a allow a function to accept and return StateArray objects,
    where the function is applied to each element of the StateArray.
    """

    def state_array_wrapper(
        value: "T | StateArray[T] | LazyField[T]",
    ) -> "U | StateArray[U] | LazyField[U]":
        if is_lazy(value):
            return value
        if isinstance(value, StateArray):
            return StateArray(function(v) for v in value)
        return function(value)

    return state_array_wrapper


def is_lazy(obj):
    """Check whether an object is a lazy field or has any attribute that is a Lazy Field"""
    from pydra.engine.lazy import LazyField

    return isinstance(obj, LazyField)


def copy_nested_files(
    value: ty.Any,
    dest_dir: os.PathLike,
    supported_modes: generic.FileSet.CopyMode = generic.FileSet.CopyMode.any,
    **kwargs,
) -> ty.Any:
    """Copies all "file-sets" found within the nested value (e.g. dict, list,...) into the
    destination directory. If no nested file-sets are found then the original value is
    returned. Note that multiple nested file-sets (e.g. a list) will to have unique names
    names (i.e. not differentiated by parent directories) otherwise there will be a path
    clash in the destination directory.

    Parameters
    ----------
    value : Any
        the value to copy files from (if required)
    dest_dir : os.PathLike
        the destination directory to copy the files to
    **kwargs
        passed directly onto FileSet.copy()
    """
    from pydra.utils.typing import TypeParser  # noqa

    cache: ty.Dict[generic.FileSet, generic.FileSet] = {}

    # Set to keep track of file paths that have already been copied
    # to allow FileSet.copy to avoid name clashes
    clashes_to_avoid = set()

    def copy_fileset(fileset: generic.FileSet):
        try:
            return cache[fileset]
        except KeyError:
            pass
        supported = supported_modes
        if any(MountIndentifier.on_cifs(p) for p in fileset.fspaths):
            supported -= generic.FileSet.CopyMode.symlink
        if not all(
            MountIndentifier.on_same_mount(p, dest_dir) for p in fileset.fspaths
        ):
            supported -= generic.FileSet.CopyMode.hardlink
        cp_kwargs = {}

        cp_kwargs.update(kwargs)
        copied = fileset.copy(
            dest_dir=dest_dir,
            supported_modes=supported,
            avoid_clashes=clashes_to_avoid,  # this prevents fname clashes between filesets
            **kwargs,
        )
        cache[fileset] = copied
        return copied

    return TypeParser.apply_to_instances(generic.FileSet, copy_fileset, value)
