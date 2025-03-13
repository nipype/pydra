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

    def split(self) -> "LazyField":
        """ "Splits" the lazy field over an array of nodes by replacing the sequence type
        of the lazy field with StateArray to signify that it will be "split" across
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

    def split(self) -> "LazyField":
        """ "Splits" the lazy field over an array of nodes by replacing the sequence type
        of the lazy field with StateArray to signify that it will be "split" across
        """
        from ..utils.typing import TypeParser  # pylint: disable=import-outside-toplevel

        assert not isinstance(self, LazyInField)

        if not TypeParser.matches_type(self.type, list):
            raise TypeError(
                f"Cannot split non-sequence field {self}  of type {self.type}"
            )

        return type(self)(
            name=self.name,
            field=self.field,
            type=StateArray[TypeParser.get_item_type(self.type)],
        )


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
        state = self._node.state
        jobs = graph.node(self._node.name).get_jobs(state_index)

        def retrieve_from_job(job: "Task[DefType]") -> ty.Any:
            if job.errored:
                raise ValueError(
                    f"Cannot retrieve value for {self._field} from {self._node.name} as "
                    "the node errored"
                )
            res = job.result()
            if res is None:
                raise RuntimeError(
                    f"Could not find results of '{job.name}' node in a sub-directory "
                    f"named '{{{job.checksum}}}' in any of the cache locations.\n"
                    + "\n".join(str(p) for p in set(job.cache_locations))
                    + f"\n\nThis is likely due to hash changes in '{job.name}' node inputs. "
                    f"Current values and hashes: {job.inputs}, "
                    f"{job.definition._hash}\n\n"
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

        if not isinstance(jobs, StateArray):  # single job
            return retrieve_from_job(jobs)
        elif not state or not state.depth(before_combine=True):
            assert len(jobs) == 1
            return retrieve_from_job(jobs[0])
        # elif state.combiner and state.keys_final:
        #     # We initialise it here rather than using a defaultdict to ensure the order
        #     # of the keys matches how it is defined in the state so we can return the
        #     # values in the correct order
        #     sorted_values = {frozenset(i.items()): [] for i in state.states_ind_final}
        #     # Iterate through the jobs and append the values to the correct final state
        #     # key
        #     for job in jobs:
        #         state_key = frozenset(
        #             (key, state.states_ind[job.state_index][key])
        #             for key in state.keys_final
        #         )
        #         sorted_values[state_key].append(retrieve_from_job(job))
        #     return StateArray(sorted_values.values())
        # else:
        return [retrieve_from_job(j) for j in jobs]

    def split(self) -> "LazyField":
        """ "Splits" the lazy field over an array of nodes by replacing the sequence type
        of the lazy field with StateArray to signify that it will be "split" across
        """
        from ..utils.typing import TypeParser  # pylint: disable=import-outside-toplevel

        # Modify the type of the lazy field to include the split across a state-array
        inner_type, prev_split_depth = TypeParser.strip_splits(self.type)
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
        return type(self)[type_](
            name=self.name,
            field=self.field,
            type=type_,
        )

    @property
    def _source(self):
        return self._node
