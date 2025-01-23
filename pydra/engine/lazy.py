import typing as ty
import abc
from typing_extensions import Self
import attrs
from pydra.utils.typing import StateArray
from pydra.utils.hash import hash_single
from . import node

if ty.TYPE_CHECKING:
    from .graph import DiGraph
    from .submitter import NodeExecution
    from .core import Task, Workflow


T = ty.TypeVar("T")

TypeOrAny = ty.Union[type, ty.Any]


@attrs.define(kw_only=True)
class LazyField(ty.Generic[T], metaclass=abc.ABCMeta):
    """Lazy fields implement promises."""

    field: str
    type: TypeOrAny
    cast_from: ty.Optional[ty.Type[ty.Any]] = None
    type_checked: bool = False

    def __bytes_repr__(self, cache):
        yield type(self).__name__.encode() + b"("
        yield from bytes(hash_single(self.source, cache))
        yield b"field=" + self.field.encode()
        yield b"type=" + bytes(hash_single(self.type, cache))
        yield b"cast_from=" + bytes(hash_single(self.cast_from, cache))
        yield b")"

    def _apply_cast(self, value):
        """\"Casts\" the value from the retrieved type if a cast has been applied to
        the lazy-field"""
        from pydra.utils.typing import TypeParser

        if self.cast_from:
            assert TypeParser.matches(value, self.cast_from)
            value = self.type(value)
        return value


@attrs.define(kw_only=True)
class LazyInField(LazyField[T]):

    workflow: "Workflow" = attrs.field()

    attr_type = "input"

    def __eq__(self, other):
        return (
            isinstance(other, LazyInField)
            and self.field == other.field
            and self.type == other.type
        )

    @property
    def source(self):
        return self.workflow

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

    def cast(self, new_type: TypeOrAny) -> Self:
        """ "casts" the lazy field to a new type

        Parameters
        ----------
        new_type : type
            the type to cast the lazy-field to

        Returns
        -------
        cast_field : LazyInField
            a copy of the lazy field with the new type
        """
        return type(self)[new_type](
            workflow=self.workflow,
            field=self.field,
            type=new_type,
            cast_from=self.cast_from if self.cast_from else self.type,
        )


@attrs.define(kw_only=True)
class LazyOutField(LazyField[T]):

    node: node.Node
    attr_type = "output"

    @property
    def name(self) -> str:
        return self.node.name

    def get_value(
        self, graph: "DiGraph[NodeExecution]", state_index: ty.Optional[int] = None
    ) -> ty.Any:
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

        task = graph.node(self.node.name).task(state_index)
        _, split_depth = TypeParser.strip_splits(self.type)

        def get_nested(task: "Task", depth: int):
            if isinstance(task, StateArray):
                val = [get_nested(task=t, depth=depth - 1) for t in task]
                if depth:
                    val = StateArray[self.type](val)
            else:
                if task.errored:
                    raise ValueError(
                        f"Cannot retrieve value for {self.field} from {self.name} as "
                        "the node errored"
                    )
                res = task.result()
                if res is None:
                    raise RuntimeError(
                        f"Could not find results of '{task.name}' node in a sub-directory "
                        f"named '{{{task.checksum}}}' in any of the cache locations.\n"
                        + "\n".join(str(p) for p in set(task.cache_locations))
                        + f"\n\nThis is likely due to hash changes in '{task.name}' node inputs. "
                        f"Current values and hashes: {task.inputs}, "
                        f"{task.definition._hash}\n\n"
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
                val = res.get_output_field(self.field)
            return val

        value = get_nested(task, depth=split_depth)
        value = self._apply_cast(value)
        return value

    @property
    def source(self):
        return self.node

    def cast(self, new_type: TypeOrAny) -> Self:
        """ "casts" the lazy field to a new type

        Parameters
        ----------
        new_type : type
            the type to cast the lazy-field to

        Returns
        -------
        cast_field : LazyOutField
            a copy of the lazy field with the new type
        """
        return type(self)[new_type](
            node=self.node,
            field=self.field,
            type=new_type,
            cast_from=self.cast_from if self.cast_from else self.type,
        )
