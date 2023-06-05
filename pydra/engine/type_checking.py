import itertools
import inspect
from pathlib import Path
import os
import typing as ty
import attr
from .specs import (
    LazyField,
    gathered,
    MultiInputObj,
    MultiInputFile,
    MultiOutputObj,
)


T = ty.TypeVar("T")
TypeOrAny = ty.Union[type, ty.Any]


class TypeChecker(ty.Generic[T]):
    """A callable which can be used as a converter for attrs.fields to check whether an
    object or LazyField matches the specified field type, or can be
    coerced into it (given the criteria passed on initialisation of the checker).
    Nested container type are expanded and each of their type args are checked/coerced
    against corresponding parts of the object.

    Parameters
    ----------
    tp : type
        the type objects will be coerced to
    coercible: Iterable[tuple[type or Any, type or Any]], optional
        limits coercing between the pairs of types where they appear within the
        tree of more complex nested container types. If None, then all types are
        coercible except explicitly excluded
    not_coercible: Iterable[tuple[type or Any, type or Any]], optional
        excludes the limits coercing between the pairs of types where they appear within
        the tree of more complex nested container types. Overrides 'coercible' to enable
        you to carve out exceptions, such as
            TypeChecker(list, coercible=[(ty.Iterable, list)], not_coercible=[(str, list)])
    """

    tp: ty.Type[T]
    coercible: list[tuple[TypeOrAny, TypeOrAny]]
    not_coercible: list[tuple[TypeOrAny, TypeOrAny]]

    COERCIBLE_DEFAULT = (
        (ty.Sequence, ty.Sequence),
        (ty.Mapping, ty.Mapping),
        (Path, os.PathLike),
        (str, os.PathLike),
        (os.PathLike, Path),
        (os.PathLike, str),
        (ty.Any, MultiInputObj),
        (ty.Union[os.PathLike, str], MultiInputFile),
        (ty.Sequence, MultiOutputObj),
        (int, float),
    )

    NOT_COERCIBLE_DEFAULT = ((str, ty.Sequence), (ty.Sequence, str))

    def __init__(
        self,
        tp,
        coercible: ty.Optional[
            ty.Iterable[tuple[TypeOrAny, TypeOrAny]]
        ] = COERCIBLE_DEFAULT,
        not_coercible: ty.Optional[
            ty.Iterable[tuple[TypeOrAny, TypeOrAny]]
        ] = NOT_COERCIBLE_DEFAULT,
    ):
        def expand_pattern(t):
            """Recursively expand the type arguments of the target type in nested tuples"""
            if t is inspect._empty:
                return None
            origin = ty.get_origin(t)
            if origin is None:
                return t
            args = ty.get_args(t)
            if not args or args == (Ellipsis,):
                assert isinstance(origin, type)
                return origin
            return (origin, [expand_pattern(a) for a in args])

        self.tp = tp
        self.coercible = (
            list(coercible) if coercible is not None else [(ty.Any, ty.Any)]
        )
        self.not_coercible = list(not_coercible) if not_coercible is not None else []
        self.pattern = expand_pattern(tp)

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
        if obj is attr.NOTHING:
            coerced = attr.NOTHING  # type: ignore[assignment]
        elif isinstance(obj, LazyField):
            if obj.attr_type == "output":
                self.check_type(obj.type)
            coerced = obj  # type: ignore[assignment]
        elif isinstance(obj, gathered):
            coerced = gathered(self(o) for o in obj)  # type: ignore[assignment]
        else:
            coerced = self.coerce(obj)
        return coerced

    def coerce(self, object_: ty.Any) -> T:
        """Attempts to coerce the given object to the type of the specified type"""
        if self.pattern is None:
            return object_

        def expand_and_coerce(obj, pattern: ty.Union[type | tuple]):
            """Attempt to expand the object along the lines of the coercion pattern"""
            if obj is attr.NOTHING:
                return attr.NOTHING
            if not isinstance(pattern, tuple):
                return coerce_basic(obj, pattern)
            origin, pattern_args = pattern
            if origin is ty.Union:
                return coerce_union(obj, pattern_args)
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
                    f" (part of coercion from {object_} to {self.pattern}"
                    if obj is not object_
                    else ""
                )
                raise TypeError(
                    f"Could not coerce to {type_} as {obj} is not iterable{msg}"
                ) from e
            if issubclass(origin, ty.Tuple):
                return coerce_tuple(type_, obj_args, pattern_args)
            return coerce_sequence(type_, obj_args, pattern_args)

        def coerce_basic(obj, pattern):
            """Coerce an object to a "basic types" like `int`, `float`, `bool`, `Path`
            and `File` in contrast to compound types like `list[int]`,
            `dict[str, str]` and `dict[str, list[int]]`"""
            if self.is_instance(obj, pattern):
                return obj
            self.check_coercible(obj, pattern)
            return coerce_to_type(obj, pattern)

        def coerce_union(obj, pattern_args):
            """Coerce an object into the first type in a Union construct that it is
            coercible into"""
            reasons = []
            for arg in pattern_args:
                try:
                    return expand_and_coerce(obj, arg)
                except Exception as e:
                    reasons.append(e)
            raise TypeError(
                f"Could not coerce {obj} to any of the union types:\n\n"
                + "\n\n".join(f"{a} -> {e}" for a, e in zip(pattern_args, reasons))
            )

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
                    f"Could not coerce to {type_} as {obj} is not a mapping type{msg}"
                ) from e
            return coerce_to_type(
                (
                    (
                        expand_and_coerce(k, key_pattern),
                        expand_and_coerce(v, val_pattern),
                    )
                    for k, v in items
                ),
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
                    f"{len(pattern_args)}, got {len(obj_args)}"
                )
            return coerce_to_type(
                [expand_and_coerce(o, p) for o, p in zip(obj_args, pattern_args)], type_
            )

        def coerce_sequence(
            type_: ty.Type[ty.Sequence], obj_args: list, pattern_args: list
        ):
            """Coerce a non-tuple sequence object (e.g. list, ...)"""
            assert len(pattern_args) == 1
            return coerce_to_type(
                [expand_and_coerce(o, pattern_args[0]) for o in obj_args], type_
            )

        def coerce_to_type(obj, type_):
            """Attempt to do the innermost (i.e. non-nested) coercion and fail with
            helpful message
            """
            try:
                return type_(obj)
            except TypeError as e:
                msg = (
                    f" (part of coercion from {object_} to {self.pattern}"
                    if obj is not object_
                    else ""
                )
                raise TypeError(f"Cannot coerce {obj} into {type_}{msg}") from e

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
        if self.pattern is None:
            return

        def expand_and_check(tp, pattern: ty.Union[type | tuple]):
            """Attempt to expand the object along the lines of the coercion pattern"""
            if not isinstance(pattern, tuple):
                return check_basic(tp, pattern)
            pattern_origin, pattern_args = pattern
            if pattern_origin is ty.Union:
                return check_union(tp, pattern_args)
            tp_origin = ty.get_origin(tp)
            if tp_origin is None:
                if issubclass(tp, pattern_origin):
                    raise TypeError(
                        f"Type {tp} wasn't declared with type args required to match pattern "
                        f"{pattern_args}, when matching {type_} to {self.pattern}"
                    )
                raise TypeError(
                    f"{tp} doesn't match pattern {pattern}, when matching {type_} to "
                    f"{self.pattern}"
                )
            tp_args = ty.get_args(tp)
            self.check_coercible(tp_origin, pattern_origin)
            if issubclass(pattern_origin, ty.Mapping):
                return check_mapping(tp_args, pattern_args)
            if issubclass(pattern_origin, ty.Tuple):
                if not issubclass(tp_origin, ty.Tuple):
                    assert len(tp_args) == 1
                    tp_args += (Ellipsis,)
                return check_tuple(tp_args, pattern_args)
            return check_sequence(tp_args, pattern_args)

        def check_basic(tp, pattern):
            if not self.is_or_subclass(tp, pattern):
                self.check_coercible(tp, pattern)

        def check_union(tp, pattern_args):
            reasons = []
            for arg in pattern_args:
                try:
                    return expand_and_check(tp, arg)
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
                if len(tp_args) == 1:
                    raise TypeError(
                        "Generic ellipsis type arguments not specific enough to match "
                        f"{pattern_args} in attempting to match {type_} to {self.pattern}"
                    )
                if tp_args[-1] is Ellipsis:
                    return expand_and_check(tp_args[0], pattern_args[0])
                for arg in tp_args:
                    expand_and_check(arg, pattern_args[0])
                return
            if len(tp_args) != len(pattern_args):
                raise TypeError(
                    f"Wrong number of type arguments in tuple {tp_args}  compared to pattern "
                    f"{pattern_args} in attempting to match {type_} to {self.pattern}"
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
                        f"{pattern_args} in attempting to match {type_} to {self.pattern}"
                    )
            for arg in tp_args:
                expand_and_check(arg, pattern_args[0])

        return expand_and_check(type_, self.pattern)

    def check_coercible(self, source: object | type, target: type | ty.Any):
        """Checks whether the source object or type is coercible to the target type
        given the coercion rules defined in the `coercible` and `not_coercible` attrs

        Parameters
        ----------
        source : object or type
            source object or type to be coerced
        target : type or ty.Any
            target type for the source to be coerced to

        Raises
        ------
        TypeError
            If the source type cannot be coerced into the target type depending on the
            explicit inclusions and exclusions set in the `coercible` and `not_coercible`
            member attrs
        """

        source_origin = ty.get_origin(source)
        if source_origin is not None:
            source = source_origin

        source_check = (
            self.is_or_subclass if inspect.isclass(source) else self.is_instance
        )

        def matches(criteria):
            return [
                (src, tgt)
                for src, tgt in criteria
                if source_check(source, src) and self.is_or_subclass(target, tgt)
            ]

        if not matches(self.coercible):
            raise TypeError(
                f"Cannot coerce {repr(source)} into {target} as the coercion doesn't match "
                f"any of the explicit inclusion criteria: "
                + ", ".join(f"{s.__name__} -> {t.__name__}" for s, t in self.coercible)
            )
        matches_not_coercible = matches(self.not_coercible)
        if matches_not_coercible:
            raise TypeError(
                f"Cannot coerce {repr(source)} into {target} as it is explicitly "
                "excluded by the following coercion criteria: "
                + ", ".join(
                    f"{s.__name__} -> {t.__name__}" for s, t in matches_not_coercible
                )
            )

    @staticmethod
    def is_instance(obj, cls):
        """Checks whether the object is an instance of cls or that cls is typing.Any"""
        return cls is ty.Any or isinstance(obj, cls)

    @staticmethod
    def is_or_subclass(a, b):
        """Checks whether the class a is either the same as b, a subclass of b or b is
        typing.Any"""
        origin = ty.get_origin(a)
        if origin is not None:
            a = origin
        return a is b or b is ty.Any or issubclass(a, b)
