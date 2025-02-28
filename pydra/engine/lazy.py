import typing as ty
import abc
import attrs
from pydra.utils.typing import StateArray
from pydra.utils.hash import hash_single
from . import node

if ty.TYPE_CHECKING:
    from .submitter import DiGraph, NodeExecution
    from .core import Task, Workflow
    from .specs import TaskDef
    from .state import StateIndex


T = ty.TypeVar("T")
DefType = ty.TypeVar("DefType", bound="TaskDef")

TypeOrAny = ty.Union[type, ty.Any]


@attrs.define(kw_only=True)
class LazyField(ty.Generic[T], metaclass=abc.ABCMeta):
    """Lazy fields implement promises."""

    _field: str
    _type: TypeOrAny
    _cast_from: ty.Optional[ty.Type[ty.Any]] = None
    _type_checked: bool = False

    def __bytes_repr__(self, cache):
        yield type(self).__name__.encode() + b"("
        yield from bytes(hash_single(self.source, cache))
        yield b"field=" + self._field.encode()
        yield b"type=" + bytes(hash_single(self._type, cache))
        yield b"cast_from=" + bytes(hash_single(self._cast_from, cache))
        yield b")"

    def _apply_cast(self, value):
        """\"Casts\" the value from the retrieved type if a cast has been applied to
        the lazy-field"""
        from pydra.utils.typing import TypeParser

        if self._cast_from:
            assert TypeParser.matches(value, self._cast_from)
            value = self._type(value)
        return value

    def _get_value(
        self,
        workflow: "Workflow",
        graph: "DiGraph[NodeExecution]",
        state_index: "StateIndex | None" = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : StateIndex, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        raise NotImplementedError("LazyField is an abstract class")


@attrs.define(kw_only=True)
class LazyInField(LazyField[T]):

    _workflow: "Workflow" = attrs.field()

    _attr_type = "input"

    def __eq__(self, other):
        return (
            isinstance(other, LazyInField)
            and self._field == other._field
            and self._type == other._type
        )

    def __repr__(self):
        return f"{type(self).__name__}(field={self._field!r}, type={self._type})"

    @property
    def _source(self):
        return self._workflow

    def _get_value(
        self,
        workflow: "Workflow",
        graph: "DiGraph[NodeExecution]",
        state_index: "StateIndex | None" = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : StateIndex, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        value = workflow.inputs[self._field]
        value = self._apply_cast(value)
        return value


@attrs.define(kw_only=True)
class LazyOutField(LazyField[T]):

    _node: node.Node
    _attr_type = "output"

    def __repr__(self):
        return (
            f"{type(self).__name__}(node={self._node.name!r}, "
            f"field={self._field!r}, type={self._type})"
        )

    def _get_value(
        self,
        workflow: "Workflow",
        graph: "DiGraph[NodeExecution]",
        state_index: "StateIndex | None" = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : StateIndex, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        from pydra.engine.state import StateIndex

        if state_index is None:
            state_index = StateIndex()

        node_exec = graph.node(self._node.name)
        task = node_exec.task(state_index)
        split_depth = node_exec.node.state.depth() if node_exec.node.state else 0

        def get_nested(task: "Task[DefType]", depth: int):
            if depth:
                assert isinstance(task, StateArray)
                val = [get_nested(task=t, depth=depth - 1) for t in task]
                if depth:
                    val = StateArray[self._type](val)
            else:
                if task.errored:
                    raise ValueError(
                        f"Cannot retrieve value for {self._field} from {self._node.name} as "
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
                val = res.get_output_field(self._field)
                val = self._apply_cast(val)
            return val

        value = get_nested(task, depth=split_depth)
        return value

    @property
    def _source(self):
        return self._node
