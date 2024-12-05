import typing as ty
from typing_extensions import Self
import attrs
from pydra.utils.typing import StateArray
from . import node

if ty.TYPE_CHECKING:
    from .base import Workflow


T = ty.TypeVar("T")

TypeOrAny = ty.Union[type, ty.Any]


@attrs.define(auto_attribs=True, kw_only=True)
class LazyField(ty.Generic[T]):
    """Lazy fields implement promises."""

    node: node.Node
    field: str
    type: TypeOrAny
    # Set of splitters that have been applied to the lazy field. Note that the splitter
    # specifications are transformed to a tuple[tuple[str, ...], ...] form where the
    # outer tuple is the outer product, the inner tuple are inner products (where either
    # product can be of length==1)
    splits: ty.FrozenSet[ty.Tuple[ty.Tuple[str, ...], ...]] = attrs.field(
        factory=frozenset, converter=frozenset
    )
    cast_from: ty.Optional[ty.Type[ty.Any]] = None
    # type_checked will be set to False after it is created but defaults to True here for
    # ease of testing
    type_checked: bool = True

    def __bytes_repr__(self, cache):
        yield type(self).__name__.encode()
        yield self.name.encode()
        yield self.field.encode()

    def cast(self, new_type: TypeOrAny) -> Self:
        """ "casts" the lazy field to a new type

        Parameters
        ----------
        new_type : type
            the type to cast the lazy-field to

        Returns
        -------
        cast_field : LazyField
            a copy of the lazy field with the new type
        """
        return type(self)[new_type](
            name=self.name,
            field=self.field,
            type=new_type,
            splits=self.splits,
            cast_from=self.cast_from if self.cast_from else self.type,
        )

    # def split(self, splitter: Splitter) -> Self:
    #     """ "Splits" the lazy field over an array of nodes by replacing the sequence type
    #     of the lazy field with StateArray to signify that it will be "split" across

    #     Parameters
    #     ----------
    #     splitter : str or ty.Tuple[str, ...] or ty.List[str]
    #         the splitter to append to the list of splitters
    #     """
    #     from pydra.utils.typing import (
    #         TypeParser,
    #     )  # pylint: disable=import-outside-toplevel

    #     splits = self.splits | set([LazyField.normalize_splitter(splitter)])
    #     # Check to see whether the field has already been split over the given splitter
    #     if splits == self.splits:
    #         return self

    #     # Modify the type of the lazy field to include the split across a state-array
    #     inner_type, prev_split_depth = TypeParser.strip_splits(self.type)
    #     assert prev_split_depth <= 1
    #     if inner_type is ty.Any:
    #         type_ = StateArray[ty.Any]
    #     elif TypeParser.matches_type(inner_type, list):
    #         item_type = TypeParser.get_item_type(inner_type)
    #         type_ = StateArray[item_type]
    #     else:
    #         raise TypeError(
    #             f"Cannot split non-sequence field {self}  of type {inner_type}"
    #         )
    #     if prev_split_depth:
    #         type_ = StateArray[type_]
    #     return type(self)[type_](
    #         name=self.name,
    #         field=self.field,
    #         type=type_,
    #         splits=splits,
    #     )

    # # def combine(self, combiner: str | list[str]) -> Self:

    def _apply_cast(self, value):
        """\"Casts\" the value from the retrieved type if a cast has been applied to
        the lazy-field"""
        from pydra.utils.typing import TypeParser

        if self.cast_from:
            assert TypeParser.matches(value, self.cast_from)
            value = self.type(value)
        return value


@attrs.define(auto_attribs=True, kw_only=True)
class LazyInField(LazyField[T]):

    attr_type = "input"

    def __eq__(self, other):
        return (
            isinstance(other, LazyInField)
            and self.field == other.field
            and self.type == other.type
            and self.splits == other.splits
        )

    def get_value(self, wf: "Workflow", state_index: ty.Optional[int] = None) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        wf : Workflow
            the workflow the lazy field references
        state_index : int, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        from pydra.utils.typing import (
            TypeParser,
        )  # pylint: disable=import-outside-toplevel

        value = getattr(wf.inputs, self.field)
        if TypeParser.is_subclass(self.type, StateArray) and not wf._pre_split:
            _, split_depth = TypeParser.strip_splits(self.type)

            def apply_splits(obj, depth):
                if depth < 1:
                    return obj
                return StateArray[self.type](apply_splits(i, depth - 1) for i in obj)

            value = apply_splits(value, split_depth)
        value = self._apply_cast(value)
        return value


class LazyOutField(LazyField[T]):
    attr_type = "output"

    def get_value(self, wf: "Workflow", state_index: ty.Optional[int] = None) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        wf : Workflow
            the workflow the lazy field references
        state_index : int, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        from pydra.utils.typing import (
            TypeParser,
        )  # pylint: disable=import-outside-toplevel

        node = getattr(wf, self.name)
        result = node.result(state_index=state_index)
        if result is None:
            raise RuntimeError(
                f"Could not find results of '{node.name}' node in a sub-directory "
                f"named '{node.checksum}' in any of the cache locations.\n"
                + "\n".join(str(p) for p in set(node.cache_locations))
                + f"\n\nThis is likely due to hash changes in '{self.name}' node inputs. "
                f"Current values and hashes: {node.inputs}, "
                f"{node.inputs._hashes}\n\n"
                "Set loglevel to 'debug' in order to track hash changes "
                "throughout the execution of the workflow.\n\n "
                "These issues may have been caused by `bytes_repr()` methods "
                "that don't return stable hash values for specific object "
                "types across multiple processes (see bytes_repr() "
                '"singledispatch "function in pydra/utils/hash.py).'
                "You may need to write specific `bytes_repr()` "
                "implementations (see `pydra.utils.hash.register_serializer`) or a "
                "`__bytes_repr__()` dunder methods to handle one or more types in "
                "your interface inputs."
            )
        _, split_depth = TypeParser.strip_splits(self.type)

        def get_nested_results(res, depth: int):
            if isinstance(res, list):
                if not depth:
                    val = [r.get_output_field(self.field) for r in res]
                else:
                    val = StateArray[self.type](
                        get_nested_results(res=r, depth=depth - 1) for r in res
                    )
            else:
                if res.errored:
                    raise ValueError(
                        f"Cannot retrieve value for {self.field} from {self.name} as "
                        "the node errored"
                    )
                val = res.get_output_field(self.field)
                if depth and not wf._pre_split:
                    assert isinstance(val, ty.Sequence) and not isinstance(val, str)
                    val = StateArray[self.type](val)
            return val

        value = get_nested_results(result, depth=split_depth)
        value = self._apply_cast(value)
        return value
