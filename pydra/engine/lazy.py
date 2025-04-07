import typing as ty
import abc
import attrs
from typing import Self
from pydra.utils.typing import StateArray
from pydra.utils.hash import hash_single
from pydra.engine import node

if ty.TYPE_CHECKING:
    from pydra.engine.submitter import DiGraph, NodeExecution
    from pydra.engine.job import Job
    from pydra.engine.workflow import Workflow
    from pydra.compose.base import Task


T = ty.TypeVar("T")
TaskType = ty.TypeVar("TaskType", bound="Task")

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
        yield b"source=" + bytes(hash_single(self._source, cache))
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
        state_index: int | None = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : int, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """
        raise NotImplementedError("LazyField is an abstract class")

    def split(self) -> Self:
        """ "Splits" the lazy field over an array of nodes by replacing the sequence type
        of the lazy field with StateArray to signify that it will be "split" across
        """
        from pydra.utils.typing import (
            TypeParser,
        )  # pylint: disable=import-outside-toplevel

        # Modify the type of the lazy field to include the split across a state-array
        inner_type, prev_split_depth = TypeParser.strip_splits(self._type)
        assert prev_split_depth <= 1
        if inner_type is ty.Any:
            type_ = StateArray[ty.Any]
        elif TypeParser.matches_type(inner_type, list):
            item_type = TypeParser.get_item_type(inner_type)
            type_ = StateArray[item_type]
        else:
            raise TypeError(
                f"Cannot split non-sequence field {self}  of type {inner_type}"
            )
        if prev_split_depth:
            type_ = StateArray[
                type_
            ]  # FIXME: This nesting of StateArray is probably unnecessary
        return attrs.evolve(self, type=type_)


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
        state_index: int | None = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : int, optional
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
        state_index: int | None = None,
    ) -> ty.Any:
        """Return the value of a lazy field.

        Parameters
        ----------
        workflow: Workflow
            the workflow object
        graph: DiGraph[NodeExecution]
            the graph representing the execution state of the workflow
        state_index : int, optional
            the state index of the field to access

        Returns
        -------
        value : Any
            the resolved value of the lazy-field
        """

        def retrieve_from_job(job: "Job[TaskType]") -> ty.Any:
            if job.errored:
                raise ValueError(
                    f"Cannot retrieve value for {self._field!r} from {self._node.name} as "
                    "the node errored"
                )
            res = job.result()
            if res is None:
                raise RuntimeError(
                    f"Could not find results of '{job.name}' node in a sub-directory "
                    f"named '{{{job.checksum}}}' in any of the cache locations.\n"
                    + "\n".join(str(p) for p in set(job.readonly_caches))
                    + f"\n\nThis is likely due to hash changes in '{job.name}' node inputs. "
                    f"Current values and hashes: {job.inputs}, "
                    f"{job.task._hash}\n\n"
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

        # Get the execution node that the value is coming from
        upstream_node = graph.node(self._node.name)

        if not upstream_node._tasks:  # No jobs, return empty state array
            return StateArray()
        if not upstream_node.state:  # Return the singular job
            value = retrieve_from_job(upstream_node._tasks[None])
            if state_index is not None:
                return value[state_index]
            return value
        if upstream_node.state.combiner:

            # No state remains after the combination, return all values in a list
            if not upstream_node.state.ind_l_final:
                return [retrieve_from_job(j) for j in upstream_node.tasks]

            # Group the values of the tasks into list before returning
            def group_values(index: int) -> list:
                # Get a slice of the tasks that match the given index of the state array of the
                # combined values
                final_index = set(upstream_node.state.states_ind_final[index].items())
                return [
                    retrieve_from_job(upstream_node._tasks[i])
                    for i, ind in enumerate(upstream_node.state.states_ind)
                    if set(ind.items()).issuperset(final_index)
                ]

            if state_index is None:  # return all groups if no index is given
                return StateArray(
                    group_values(i) for i in range(len(upstream_node.state.ind_l_final))
                )
            return group_values(state_index)  # select the group that matches the index
        if state_index is None:  # return all jobs in a state array
            return StateArray(retrieve_from_job(j) for j in upstream_node.tasks)
        # Select the job that matches the index
        return retrieve_from_job(upstream_node._tasks[state_index])

    @property
    def _source(self):
        return self._node
