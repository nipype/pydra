"""The graph runner: the state of a workflow's execution graph, and the rule by which
its jobs are released, completed and marked unrunnable. Drivers (see
:class:`~pydra.engine.submitter.Submitter`) loop over a :class:`GraphRunner`, taking the
jobs it hands out, running them and reporting each outcome."""

import itertools
import typing as ty
from traceback import format_exc, format_exception
import re
from collections import defaultdict
from datetime import datetime
import logging
import attrs
from pydra.engine.graph import DiGraph
from pydra.utils.general import attrs_values
from pydra.engine.lazy import LazyField, LazyOutField
from pydra.engine.job import Job
from pydra.engine.state import (
    Entry,
    PydraStateError,
    compatible,
    expand_rpn,
    flatten,
    is_sequence,
    remove_inp_from_splitter_rpn,
    rpn_terms,
    scalar_operands,
)
from pydra.utils.general import ensure_list
from pydra.utils.typing import StateArray

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task, Outputs
    from pydra.engine.node import Node
    from pydra.engine.result import Result
    from pydra.engine.hooks import TaskHooks
    from pydra.engine.store import ResultStore
    from pydra.engine.workflow import Workflow
    from pydra.engine.submitter import Submitter
    from pydra.environments.base import Environment


TaskType = ty.TypeVar("TaskType", bound="Task")
OutputType = ty.TypeVar("OutputType", bound="Outputs")

OnError = ty.Literal["raise_at_end", "fail_fast"]


@attrs.frozen
class JobErrorEntry:
    """The error of one job of a workflow, as plain picklable values

    Attributes
    ----------
    job_name : str
        The name of the job
    state_index : int | None
        The index of the job's element within its split, None if it has none
    exception_type : str
        The qualified name of the exception's class, ``module.Class``
    message : str
        ``str()`` of the exception
    traceback : str
        The formatted traceback
    phase : str
        ``"run"`` if the job ran and errored, ``"resolution"`` if its inputs could
        not be resolved from its upstream results, so it never ran, ``"expansion"`` if
        a block of a partial node could not be expanded, so it has no element
    state_key : tuple | None
        The key of the element on a partial node, whose ``state_index`` is then None
    block : tuple | None
        The key of the block that could not be expanded (phase ``"expansion"``)
    """

    job_name: str
    state_index: int | None
    exception_type: str
    message: str
    traceback: str
    phase: str = "run"
    state_key: tuple | None = None
    block: tuple | None = None


class _AllElements:
    """Reads every element of a node (a whole read)"""

    def __repr__(self):
        return "ALL"


ALL = _AllElements()


@attrs.frozen
class GroupRead:
    """Reads the elements of one combined group of a partial node"""

    key: tuple


@attrs.frozen
class Block:
    """A block of a partial node that failed before it was expanded, so has no element.
    It is a key of ``NodeExecution.unrunnable``

    Attributes
    ----------
    source : str
        The node the failure came from: the node itself for its own block, else the
        upstream node whose failed block it inherits
    key : tuple
        The key of the block in ``source``
    fields : tuple
        The coordinates fixed by the block, as ``(path, coordinate)`` pairs in the path
        space of the node that holds it
    """

    source: str
    key: tuple
    fields: tuple

    def __str__(self):
        return f"{self.source}[block {_key_text(self.key)}]"


def _key_text(key: tuple) -> str:
    return ",".join(str(k) for k in key)


def element_label(
    name: str, state_index: int | None, state_key: tuple | None = None
) -> str:
    """The name of an element in messages: ``B[1,0]`` on a partial node (its key),
    ``B(3)`` on any other split node, ``B`` without a split"""
    if state_key is not None:
        return f"{name}[{_key_text(state_key)}]"
    if state_index is not None:
        return f"{name}({state_index})"
    return name


class _BlockState:
    __slots__ = ("fields", "status")

    def __init__(self, fields: dict):
        self.fields = fields
        self.status = "pending"  # "expanded", "failed" or "errored"


class _Partial:
    """The structure of a partial node, worked out from its splitter and its
    upstream nodes once they have started

    Attributes
    ----------
    inner : dict
        Each of the node's split inputs that is an upstream output, ``"<node>.<input>"``
        mapped to its lazy field
    rpn : list
        The splitter in RPN without the prev-state term of each inner source
        (``state.py`` ``prepare_states_ind``), which fixes the element order
    sources : list
        The nodes whose final keys make up a block key: the inner sources, then the
        partial upstream nodes of prev-state terms
    linked : set
        The sources linked through the node's ``State`` (an inner source that is
        neither stateless nor combined to nothing)
    elem_paths, group_paths, final_paths : list
        The paths of an element key, of a group key (None without a combiner), and of
        the keys the node's readers see
    units : list
        ``(inner path, source paths)`` of each inner input still split in the final
        keys: an inner position means something only beside its source's coordinates
    prev_fields : dict
        The node's non-split inputs read through each prev-state term
    group_by_block : bool
        Whether the groups of a block are known from the block alone, so that a block
        with no element still has its groups
    """

    def __init__(self):
        self.inner: dict[str, LazyOutField] = {}
        self.rpn: list[str] = []
        self.sources: list[str] = []
        self.linked: set[str] = set()
        self.elem_paths: list[tuple] = []
        self.group_paths: list[tuple] | None = None
        self.final_paths: list[tuple] = []
        self.units: list[tuple[tuple, list[tuple]]] = []
        self.prev_fields: dict[str, list[str]] = {}
        self.group_by_block = False


def _dedupe(paths: ty.Iterable[tuple]) -> list[tuple]:
    return list(dict.fromkeys(paths))


class NodeExecution(ty.Generic[TaskType]):
    """A wrapper around a workflow node containing the execution state of the tasks that
    are generated from it.

    A node is classified once its upstream nodes have started (``mode``):

    * ``"static"``: its elements are known when it starts, indexed by int as before;
    * ``"partial"``: it splits over an upstream output (or reads a partial node through
      its state), so it is expanded one *block* per upstream element as each succeeds.
      Its elements are keyed by ``state_key``, a tuple of coordinates that never
      changes; ``state_index`` is filled in when the node is *sealed* (every block
      expanded);
    * ``"barrier"``: a shape that cannot be expanded per block, which waits for every
      upstream node to be done as before;
    * ``"error"``: a combiner that cannot apply, reported as a resolution error.
    """

    name: str
    node: "Node"
    submitter: "Submitter"

    # List of tasks that were completed successfully
    successful: dict[ty.Any, "Job[TaskType]"]
    # List of tasks that failed
    # (an element whose inputs could not be resolved is mapped to the exception)
    errored: dict[ty.Any, "Job[TaskType] | Exception"]
    # Elements that couldn't be run due to upstream errors, mapped to the upstream
    # nodes whose errored or unrunnable elements they read (keyed by None if the whole
    # node couldn't be started, by a ``Block`` for a block that failed unexpanded)
    unrunnable: dict[ty.Any, list["NodeExecution"]]
    # List of tasks that are queued
    queued: dict[ty.Any, "Job[TaskType]"]
    # List of tasks that are queued
    running: dict[ty.Any, tuple["Job[TaskType]", datetime]]
    # Elements that are waiting on upstream elements, mapped to the upstream nodes and
    # what of them they read (a range of all of a static node's indices, or ``ALL``,
    # stands for reading all of its elements)
    blocked: dict[ty.Any, list[tuple["NodeExecution", ty.Any]]] | None

    # Jobs of the elements, created when they are released or first read
    _tasks: dict[ty.Any, "Job[TaskType] | None"] | None

    workflow: "Workflow"

    graph: DiGraph["NodeExecution"] | None

    # Descriptions of input-resolution errors not yet collected by ``Submitter``
    _resolution_errors: list[str]
    _resolution_entries: "list[JobErrorEntry]"
    # The progress of the predecessors when the blocked elements were last checked
    _progress: tuple | None

    def __init__(
        self,
        node: "Node",
        submitter: "Submitter",
        workflow: "Workflow",
    ):
        self.name = node.name
        self.node = node
        self.submitter = submitter
        # Initialize the state dictionaries
        self._tasks = None
        self.blocked = None
        self.successful = {}
        self.errored = {}
        self.queued = {}
        self.running = {}  # Not used in logic, but may be useful for progress tracking
        self.unrunnable = defaultdict(list)
        self.workflow = workflow
        self.graph = None
        self._resolution_errors = []
        self._resolution_entries = []
        self._progress = None
        self._mode: str | None = None
        self._mode_error: Exception | None = None
        self._node_error: Exception | None = None
        self._partial: _Partial | None = None
        # partial nodes only
        self._blocks: dict[tuple, _BlockState] | None = None
        self._inherited: set = set()
        self._block_errors: list[str] = []
        self._vals: dict[tuple, dict] = {}
        self._inputs_ind: dict[tuple, dict] = {}
        self._groups: dict[tuple, list[tuple]] = {}
        self._group_of: dict[tuple, tuple] = {}
        self._closed: set[tuple] = set()
        self._static_entries: dict[str, list] = {}
        self._version = 0
        self._sealed = False
        self._rank: dict[tuple, int] = {}
        self._keys_by_rank: list[tuple] = []
        self._legacy: "bool | Exception | None" = None
        self._legacy_groups: list[tuple] = []
        self._reg_token: tuple | None = None

    @property
    def state(self):
        return self.node.state

    def __repr__(self):
        return (
            f"NodeExecution(name={self.name!r}, blocked={list(self.blocked or ())}, "
            f"queued={list(self.queued)}, running={list(self.running)}, "
            f"successful={list(self.successful)}, errored={list(self.errored)}, "
            f"unrunnable={list(self.unrunnable)})"
        )

    @property
    def inputs(self) -> "Node.Inputs":
        return self.node.inputs

    @property
    def _task(self) -> "Node":
        return self.node._task

    @property
    def is_partial(self) -> bool:
        return self._mode == "partial"

    @property
    def tasks(self) -> list["Job[TaskType]"]:
        """The jobs of all elements, in state order, creating any that do not exist
        yet (all upstream elements they read must have succeeded)"""
        if self._tasks is None:
            raise RuntimeError("Tasks have not been generated")
        indices = sorted(self._tasks) if self.is_partial else self._tasks
        return [self.job(index) for index in indices]

    def job(self, index: ty.Any) -> "Job[TaskType]":
        """The job of one element of the node, created on first use. The upstream
        elements that its lazy inputs read must have completed by then.

        Parameters
        ----------
        index : int, tuple or None
            The state index of the element (its key on a partial node; an int on a
            sealed partial node is its flat index), None if the node has no state
        """
        if self.is_partial and isinstance(index, int):
            index = self._keys_by_rank[index]
        if (job := self._tasks[index]) is None:
            if self.is_partial:
                state_index, state_key = self._rank.get(index), index
            else:
                state_index, state_key = index, None
            job = self._tasks[index] = Job(
                task=self._split_task(index),
                submitter=self.submitter,
                environment=self.node._environment,
                name=self.node.name,
                hooks=self.node._hooks,
                resources=self.node._resources,
                state_index=state_index,
                state_key=state_key,
            )
        return job

    # ------------------------------------------------------------------------------
    # Classification

    @property
    def mode(self) -> str | None:
        """``"static"``, ``"partial"``, ``"barrier"`` or ``"error"`` once classified
        (when every upstream node has started), else None"""
        return self._mode

    def _source_paths(self, name: str) -> list[tuple]:
        """The paths of the final keys of an upstream node"""
        node = self.graph.node(name)
        if node._classify() == "partial":
            return node._partial.final_paths
        if not node.state:
            return []
        return [(n,) for n in node.state.keys_final]

    def _source_names(self, name: str) -> set[str]:
        return {p[-1] for p in self._source_paths(name)}

    def _classify(self) -> str:
        """Classify the node; its upstream nodes must have started"""
        if self._mode is not None:
            return self._mode
        self._mode = "static"
        state = self.state
        if not state:
            return self._mode
        graph = self.graph
        rpn = list(state.splitter_rpn_compact)
        inner = {
            name: value
            for name, value in self.node.state_values.items()
            if name in state.current_splitter_rpn and isinstance(value, LazyOutField)
        }
        inner_sources = list(dict.fromkeys(v._node.name for v in inner.values()))
        prev_terms = [
            t
            for t in rpn_terms(rpn)
            if t.startswith("_") and t[1:] in state.other_states
        ]
        partial_prevs = [
            t[1:]
            for t in prev_terms
            if t[1:] not in inner_sources and graph.node(t[1:])._classify() == "partial"
        ]
        if not inner and not partial_prevs:
            return self._mode
        self._mode = self._classify_partial(
            rpn, inner, inner_sources, prev_terms, partial_prevs
        )
        return self._mode

    def _classify_partial(self, rpn, inner, inner_sources, prev_terms, partial_prevs):
        state = self.state
        graph = self.graph
        spec = _Partial()
        spec.inner = inner
        spec.linked = {u for u in inner_sources if u in state.other_states}
        spec.rpn = remove_inp_from_splitter_rpn(rpn, [f"_{u}" for u in spec.linked])
        spec.sources = inner_sources + partial_prevs
        # (3) a container dimension set by the user on an inner input
        user_ndim = self.node._task._container_ndim or {}
        if any(user_ndim.get(name.split(".", 1)[1], 1) > 1 for name in inner):
            return "barrier"
        # (5) a non-split input read from an inner source, directly or through an
        # upstream node that shares the source's coordinates
        reduced_terms = set(rpn_terms(spec.rpn))
        for name, value in attrs_values(self.node._task).items():
            if not isinstance(value, LazyOutField) or f"{self.name}.{name}" in inner:
                continue
            source = graph.node(value._node.name)
            if value._node.name in inner_sources or (
                source.state
                and source.state.splitter_rpn_final
                and f"_{source.name}" not in reduced_terms
            ):
                return "barrier"
        source_names = set()
        for u in inner_sources:
            source_names |= self._source_names(u)
        for t in prev_terms:
            if t[1:] not in inner_sources and source_names & self._source_names(t[1:]):
                return "barrier"

        # (1) and (2): ``.`` pairs only the node's own static terms, or the inner inputs
        # of one source
        def tag(term):
            if term in inner:
                return "inner:" + inner[term]._node.name
            if term.startswith("_") and term[1:] in partial_prevs:
                return "partial:" + term[1:]
            return "static"

        for left, right in scalar_operands(spec.rpn, tag):
            if not (
                left == right
                and len(left) == 1
                and not next(iter(left)).startswith("partial:")
            ):
                return "barrier"

        def term_paths(term):
            if term in inner:
                u = inner[term]._node.name
                base = (
                    [("_" + u,) + p for p in self._source_paths(u)]
                    if u in spec.linked
                    else []
                )
                return base + [(term,)]
            if term.startswith("_"):
                return [(term,) + p for p in self._source_paths(term[1:])]
            return [(term,)]

        spec.elem_paths = _dedupe(p for t in rpn_terms(spec.rpn) for p in term_paths(t))
        units = []
        for term, value in inner.items():
            u = value._node.name
            base = (
                [("_" + u,) + p for p in self._source_paths(u)]
                if u in spec.linked
                else []
            )
            units.append(((term,), base))
        for t in rpn_terms(spec.rpn):
            if t.startswith("_") and t[1:] in partial_prevs:
                prefix = (t,)
                for j, srcs in graph.node(t[1:])._partial.units:
                    units.append((prefix + j, [prefix + s for s in srcs]))
        for u in spec.linked:
            if graph.node(u)._classify() == "partial":
                prefix = ("_" + u,)
                for j, srcs in graph.node(u)._partial.units:
                    units.append((prefix + j, [prefix + s for s in srcs]))
        combiner = list(state.combiner or [])
        if combiner:
            keep = {
                u
                for u in spec.linked
                if all(n in combiner for n, v in inner.items() if v._node.name == u)
            }
            group_rpn = remove_inp_from_splitter_rpn(
                rpn, [f"_{u}" for u in spec.linked if u not in keep]
            )
            candidates = _dedupe(
                p
                for t in rpn_terms(group_rpn)
                if not (t in inner and t in combiner)
                for p in term_paths(t)
            )
            every = set(spec.elem_paths) | set(candidates)
            for name in combiner:
                matches = {p for p in every if p[-1] == name}
                if len(matches) != 1:
                    return "barrier"
            spec.group_paths = [p for p in candidates if p[-1] not in combiner]
            removed = {p for p in every if p[-1] in combiner}
            final_units = []
            for j, srcs in units:
                if j in removed:
                    continue
                if any(s in removed for s in srcs):
                    if len(j) == 1:
                        self._mode_error = PydraStateError(
                            f"Node {self.name!r} combines {combiner} but keeps its split "
                            f"input {j[-1]!r}: combining the block axis without its "
                            "inner input has no element order"
                        )
                        return "error"
                    return "barrier"
                final_units.append((j, srcs))
            spec.units = final_units
            spec.final_paths = spec.group_paths
            inner_paths = {j for j, _ in units}
            spec.group_by_block = not (inner_paths & set(spec.group_paths))
        else:
            spec.units = units
            spec.final_paths = spec.elem_paths
        for t in rpn_terms(spec.rpn):
            if t.startswith("_"):
                spec.prev_fields[t[1:]] = [
                    f
                    for f in state.other_states[t[1:]][1]
                    if f"{self.name}.{f}" not in inner
                ]
        self._partial = spec
        return "partial"

    # ------------------------------------------------------------------------------
    # Starting

    def start(self) -> None:
        """Prepare the state of the node and record, for each of its elements, the
        upstream elements it reads"""
        if self.state:
            values = {}
            for name, value in self.node.state_values.items():
                if name in self.node.state.current_splitter_rpn:
                    if name in ("*", "."):
                        continue
                    if isinstance(value, LazyField):
                        values[name] = value._get_value(
                            workflow=self.workflow, graph=self.graph
                        )
                    else:
                        values[name] = value
            if self._mode == "barrier":
                try:
                    self._check_barrier(values)
                except PydraStateError as exc:
                    self._fail_node(exc)
                    return
            self.state.prepare_states(values)
            self.state.prepare_inputs()
            indices = range(len(self.state.states_ind))
        else:
            indices = [None]
        self._tasks = dict.fromkeys(indices)
        self.blocked = {}
        for index in indices:
            upstream = self._upstream(index)
            for node, inds in upstream:
                if isinstance(inds, range):
                    missing = [] if len(inds) == len(node._tasks) else list(inds)
                elif isinstance(inds, list):
                    missing = [i for i in inds if i not in node._tasks]
                else:
                    missing = []
                if missing:
                    raise RuntimeError(
                        f"Element {index} of node {self.name!r} reads elements {missing} "
                        f"of node {node.name!r}, which has elements {list(node._tasks)}"
                    )
            self.blocked[index] = upstream

    def _check_barrier(self, values: dict) -> None:
        """Refuse what today's ``State`` gets silently wrong: a ragged or mixed inner
        input, an inner source that is combined, or a partial upstream node whose
        elements today's ``State`` cannot index (raises ``PydraStateError``)"""
        for name, value in self.node.state_values.items():
            if name not in values or not isinstance(value, LazyOutField):
                continue
            source = self.graph.node(value._node.name)
            if (
                source.state
                and source.state.combiner
                and source.state.splitter_rpn_final
            ):
                raise PydraStateError(
                    f"Node {self.name!r} splits {name!r} over the output of combined node "
                    f"{source.name!r}, which only a partial node can index"
                )
            items = list(values[name]) if is_sequence(values[name]) else []
            kinds = {len(i) if is_sequence(i) else None for i in items}
            if len(kinds) > 1:
                raise PydraStateError(
                    f"Node {self.name!r} splits {name!r} over upstream values of unequal "
                    f"lengths {sorted(kinds, key=str)}, which only a partial node can index"
                )
        for name in self.state.other_states:
            node = self.graph.node(name)
            if node.is_partial:
                if (error := node._legacy_prepare()) is not True:
                    raise PydraStateError(
                        f"Node {self.name!r} reads partial node {name!r} through a splitter "
                        f"that cannot be expanded per element: {error}"
                    )

    def _fail_node(self, exc: Exception) -> None:
        """Record an error of the whole node, which runs no element"""
        self._node_error = exc
        self.errored[None] = exc
        self._tasks = {}
        self.blocked = {}
        try:
            raise exc
        except Exception:
            trace = format_exc()
        self._resolution_errors.append(
            f"Node {self.name!r}, its state could not be prepared, errored:\n{trace}"
        )
        self._resolution_entries.append(
            error_entry(self.name, None, exc, phase="resolution")
        )

    def _start_partial(self) -> None:
        """Start a partial node: no element is known yet; blocks are registered as their
        source keys appear"""
        spec = self._partial
        self._tasks = {}
        self.blocked = {}
        self._blocks = {}
        for term in rpn_terms(spec.rpn):
            if term in spec.inner:
                continue
            if term.startswith("_"):
                name = term[1:]
                if name in spec.sources:
                    continue
                source = self.graph.node(name)
                self._static_entries[term] = [
                    Entry(
                        {(term, n): c for n, c in ind.items()},
                        {},
                        {name: g},
                    )
                    for g, ind in enumerate(source.state.states_ind_final)
                ]
            else:
                value = self.node.state_values[term]
                if isinstance(value, LazyField):
                    value = value._get_value(workflow=self.workflow, graph=self.graph)
                ndim = (self.state.container_ndim or {}).get(term, 1)
                items = list(flatten(ensure_list(value), max_depth=ndim))
                self._static_entries[term] = [
                    Entry({(term,): i}, {term: v}, {}) for i, v in enumerate(items)
                ]
        self._register()

    # ------------------------------------------------------------------------------
    # The final keys a node shows its readers

    def _final_keys(self) -> list:
        """The final keys of the node that exist now"""
        if self.is_partial:
            if self._partial.group_paths is None:
                return sorted(self._tasks)
            return sorted(g for g in self._groups if self._group_closed(g))
        if not self.state:
            return [None]
        return list(range(len(self.state.states_ind_final)))

    def _final_fields(self, key) -> dict:
        """The coordinate of each final path of one final key"""
        if self.is_partial:
            return dict(zip(self._partial.final_paths, key))
        if key is None or not self.state:
            return {}
        return {(n,): c for n, c in self.state.states_ind_final[key].items()}

    def _final_status(self, key) -> str:
        """``"ok"``, ``"failed"`` or ``"pending"`` for the elements of one final key"""
        spec = self._read_indices(key)
        if _has_failed(self, spec):
            return "failed"
        if _has_succeeded(self, spec):
            return "ok"
        return "pending"

    def _failed_blocks(self) -> list[tuple[ty.Any, dict]]:
        """Each failed or errored block of the node, own or inherited, with the fields it
        fixes in the node's path space"""
        if not self.is_partial or self._blocks is None:
            return []
        own = [
            (("block", k), b.fields)
            for k, b in self._blocks.items()
            if b.status in ("failed", "errored")
        ]
        return own + [
            (("from", k.source, k.key), dict(k.fields))
            for k in self.unrunnable
            if isinstance(k, Block) and k.source != self.name
        ]

    # ------------------------------------------------------------------------------
    # Blocks

    def _register(self) -> None:
        """Register a block for each new combination of source keys, and a failed
        block for each failed block of a partial source"""
        spec = self._partial
        token = tuple(self.graph.node(n)._token() for n in spec.sources)
        if token == self._reg_token:
            return
        self._reg_token = token
        lists = []
        for name in spec.sources:
            source = self.graph.node(name)
            if name in spec.linked or source.is_partial:
                lists.append(source._final_keys())
            else:
                lists.append([None])
        for combo in itertools.product(*lists):
            if combo in self._blocks:
                continue
            fields = {}
            for name, key in zip(spec.sources, combo):
                source = self.graph.node(name)
                for p, c in source._final_fields(key).items():
                    fields[("_" + name,) + p] = c
            self._blocks[combo] = _BlockState(fields)
            self._version += 1
        for name in spec.sources:
            source = self.graph.node(name)
            for ident, fields in source._failed_blocks():
                if (name, ident) in self._inherited:
                    continue
                self._inherited.add((name, ident))
                prefixed = tuple((("_" + name,) + p, c) for p, c in fields.items())
                block = Block(source=name, key=ident[1:], fields=prefixed)
                self.unrunnable[block] = [source]
                self._version += 1

    def _expand_ready(self) -> None:
        """Expand each pending block whose inner sources have succeeded, and fail each
        whose inner source has failed"""
        spec = self._partial
        for bkey in sorted(self._blocks, key=_sort_key):
            block = self._blocks[bkey]
            if block.status != "pending":
                continue
            statuses = []
            for name, key in zip(spec.sources, bkey):
                if name not in {v._node.name for v in spec.inner.values()}:
                    continue
                source = self.graph.node(name)
                if name in spec.linked or source.is_partial:
                    statuses.append((source, source._final_status(key)))
                else:
                    statuses.append((source, source._final_status(None)))
            if failed := [s for s, status in statuses if status == "failed"]:
                block.status = "failed"
                self.unrunnable[Block(self.name, bkey, tuple(block.fields.items()))] = (
                    failed
                )
                self._version += 1
            elif all(status == "ok" for _, status in statuses):
                self._expand(bkey, block)

    def _expand(self, bkey: tuple, block: _BlockState) -> None:
        spec = self._partial
        entries = dict(self._static_entries)
        skeleton = dict(self._static_entries)
        positions = dict(zip(spec.sources, bkey))
        try:
            for term in rpn_terms(spec.rpn):
                if term in spec.inner:
                    lf = spec.inner[term]
                    u = lf._node.name
                    key = positions[u]
                    value = lf._get_value(
                        workflow=self.workflow, graph=self.graph, state_index=key
                    )
                    items = list(value) if is_sequence(value) else [value]
                    base = {p: c for p, c in block.fields.items() if p[0] == "_" + u}
                    terms = {u: key} if u in spec.linked else {}
                    entries[term] = [
                        Entry({**base, (term,): j}, {term: v}, terms)
                        for j, v in enumerate(items)
                    ]
                    skeleton[term] = [Entry(dict(base), {}, terms)]
                elif term.startswith("_") and term[1:] in spec.sources:
                    name = term[1:]
                    base = {p: c for p, c in block.fields.items() if p[0] == term}
                    entries[term] = skeleton[term] = [
                        Entry(base, {}, {name: positions[name]})
                    ]
            elements = expand_rpn(spec.rpn, entries)
        except Exception as exc:
            block.status = "errored"
            self._version += 1
            label = f"{self.name}[block {_key_text(bkey)}]"
            try:
                raise exc
            except Exception:
                trace = format_exc()
            self._block_errors.append(
                f"Job {label!r}, could not be expanded, errored:\n{trace}"
            )
            self._resolution_errors.append(self._block_errors[-1])
            self._resolution_entries.append(
                error_entry(self.name, None, exc, phase="expansion", block=bkey)
            )
            return
        block.status = "expanded"
        self._version += 1
        if spec.group_paths is not None and spec.group_by_block:
            for entry in expand_rpn(spec.rpn, skeleton):
                self._groups.setdefault(
                    tuple(entry.fields[p] for p in spec.group_paths), []
                )
        for entry in elements:
            key = tuple(entry.fields[p] for p in spec.elem_paths)
            self._tasks[key] = None
            self._vals[key] = entry.values
            self._inputs_ind[key] = {
                f"{self.name}.{field}": entry.terms[name]
                for name, fields in spec.prev_fields.items()
                if name in entry.terms
                for field in fields
            }
            if spec.group_paths is not None:
                group = tuple(entry.fields[p] for p in spec.group_paths)
                self._groups.setdefault(group, []).append(key)
                self._group_of[key] = group
            self.blocked[key] = self._upstream(key)

    # ------------------------------------------------------------------------------
    # Settled, sealed, groups

    def _block_sources(self) -> list["NodeExecution"]:
        return [self.graph.node(n) for n in self._partial.sources]

    @property
    def settled(self) -> bool:
        """Whether every element the node will have is known: its sources are settled,
        every source key has a registered block, and no block is pending. A node that is
        not partial is settled once started"""
        if not self.is_partial:
            return bool(self.started)
        if self._blocks is None:
            return False
        self._register()
        if any(b.status == "pending" for b in self._blocks.values()):
            return False
        return all(s.settled for s in self._block_sources())

    @property
    def sealed(self) -> bool:
        """Whether the node is settled with every block expanded and every source
        sealed; its elements then get their flat ``state_index``"""
        if not self.is_partial:
            return bool(self.started)
        if self._sealed:
            return True
        if not self.settled or any(
            b.status != "expanded" for b in self._blocks.values()
        ):
            return False
        if any(isinstance(k, Block) for k in self.unrunnable):
            return False
        if not all(s.sealed for s in self._block_sources()):
            return False
        self._sealed = True
        self._keys_by_rank = sorted(self._tasks)
        self._rank = {k: i for i, k in enumerate(self._keys_by_rank)}
        for key, job in self._tasks.items():
            if job is not None:
                job.state_index = self._rank[key]
        return True

    def _may_produce(self, assignment: dict) -> bool:
        """Whether an element whose final coordinates agree with ``assignment`` may still
        appear (conservative: True when it cannot be ruled out)"""
        if not self.is_partial:
            return not self.started
        if self._blocks is None:
            return True
        self._register()
        for block in self._blocks.values():
            if block.status == "pending" and compatible(block.fields, assignment):
                return True
        for source in self._block_sources():
            prefix = "_" + source.name
            sub = {p[1:]: c for p, c in assignment.items() if p[0] == prefix}
            if source._may_produce(sub) or source._has_open_group(sub):
                return True
        return False

    def _has_open_group(self, assignment: dict) -> bool:
        if not self.is_partial or self._partial.group_paths is None:
            return False
        paths = self._partial.group_paths
        return any(
            not self._group_closed(g) and compatible(dict(zip(paths, g)), assignment)
            for g in self._groups
        )

    def _group_closed(self, group: tuple) -> bool:
        if group in self._closed:
            return True
        if self._may_produce(dict(zip(self._partial.group_paths, group))):
            return False
        self._closed.add(group)
        return True

    def _group_failed(self, group: tuple) -> bool:
        if any(
            m in self.errored or m in self.unrunnable
            for m in self._groups.get(group, ())
        ):
            return True
        assignment = dict(zip(self._partial.group_paths, group))
        return any(
            compatible(fields, assignment) for _, fields in self._failed_blocks()
        )

    def _blocks_summary(self) -> str:
        """The blocks of a partial node that are not expanded, for ``stuck()``"""
        if not self.is_partial or self._blocks is None:
            return ""
        by_status = defaultdict(list)
        for key, block in self._blocks.items():
            if block.status != "expanded":
                by_status[block.status].append(key)
        unsettled = [s.name for s in self._block_sources() if not s.settled]
        return "".join(
            f", {status} blocks={keys}" for status, keys in sorted(by_status.items())
        ) + (f", unsettled sources={unsettled}" if unsettled else "")

    def _token(self) -> tuple:
        """The progress of the node as its readers see it"""
        token = (len(self.successful), len(self.errored), len(self.unrunnable))
        if self.is_partial:
            token += (
                self._version,
                tuple(s._token() for s in self._block_sources()),
            )
        return token

    # ------------------------------------------------------------------------------
    # Today's ``State`` for a barrier reader of a partial node

    def _legacy_prepare(self) -> "bool | Exception":
        """Prepare the node's ``State`` as today's barrier path would, for a barrier
        reader, once the node is sealed. Returns True, or the reason it cannot be
        (a ragged or combined source, or an order today's ``State`` gets wrong)"""
        if self._legacy is not None:
            return self._legacy
        if not self.sealed:
            self._legacy = PydraStateError(f"node {self.name!r} is not sealed")
            return self._legacy
        try:
            for source in self._block_sources():
                if (
                    source.is_partial
                    and (error := source._legacy_prepare()) is not True
                ):
                    raise PydraStateError(str(error))
            values = {}
            for name, value in self.node.state_values.items():
                if name in self.state.current_splitter_rpn and name not in ("*", "."):
                    if isinstance(value, LazyField):
                        values[name] = value._get_value(
                            workflow=self.workflow, graph=self.graph
                        )
                    else:
                        values[name] = value
            self._check_barrier(values)
            self.state.prepare_states(values)
            self.state.prepare_inputs()
            own = [k for k in self._vals[self._keys_by_rank[0]]] if self._tasks else []
            legacy = [{k: v[k] for k in own if k in v} for v in self.state.states_val]
            if legacy != [self._vals[k] for k in self._keys_by_rank]:
                raise PydraStateError(
                    f"today's State orders the elements of {self.name!r} differently"
                )
            if self._partial.group_paths is not None:
                groups = sorted(self._groups)
                mine = [sorted(self._rank[m] for m in self._groups[g]) for g in groups]
                theirs = [
                    sorted(self.state.final_combined_ind_mapping[i])
                    for i in range(len(self.state.final_combined_ind_mapping))
                ]
                if self.state.ind_l_final and mine != theirs:
                    raise PydraStateError(
                        f"today's State groups the elements of {self.name!r} differently"
                    )
                self._legacy_groups = groups
        except Exception as exc:
            self._legacy = exc
        else:
            self._legacy = True
        return self._legacy

    # ------------------------------------------------------------------------------
    # Reading

    def read_value(self, retrieve: ty.Callable[["Job"], ty.Any], state_index: ty.Any):
        """The value of a partial node's output for a reader (see
        ``LazyOutField._get_value``): one element, one group, or all of them once
        sealed"""
        spec = self._partial
        if isinstance(state_index, int):
            if spec.group_paths is None:
                state_index = self._keys_by_rank[state_index]
            else:
                state_index = self._legacy_groups[state_index]
        if state_index is None:
            if not self.sealed:
                raise RuntimeError(
                    f"Cannot read every element of node {self.name!r}: it is not sealed "
                    "(an upstream element has not completed, or has failed)"
                )
            if spec.group_paths is None:
                return StateArray(retrieve(self.job(k)) for k in self._keys_by_rank)
            if not spec.group_paths:
                return [retrieve(self.job(k)) for k in self._keys_by_rank]
            return StateArray(
                [retrieve(self.job(m)) for m in sorted(self._groups[g])]
                for g in sorted(self._groups)
            )
        if spec.group_paths is None:
            return retrieve(self.job(state_index))
        return [retrieve(self.job(m)) for m in sorted(self._groups[state_index])]

    @property
    def started(self) -> bool:
        return (
            self.successful
            or self.errored
            or self.unrunnable
            or self.queued
            or self.blocked is not None
        )

    @property
    def done(self) -> bool:
        self.update_status()
        if not self.started:
            return False
        if self.is_partial and not self.settled:
            return False
        # Check to see if any previously queued tasks have completed
        return not (self.queued or self.blocked or self.running)

    @property
    def has_errored(self) -> bool:
        self.update_status()
        return bool(self.errored)

    def update_status(self) -> None:
        """Updates the status of the tasks in the node."""
        if not self.started:
            return
        # Check to see if any previously queued tasks have completed
        for index, job in list(self.queued.items()):
            try:
                is_done = job.done
            except ValueError:
                errored = True
                is_done = False
            else:
                errored = False
            if is_done:
                self.successful[index] = self.queued.pop(index)
            elif job.errored or errored:
                self.errored[index] = self.queued.pop(index)
            elif not job._awaiting_run and job.run_start_time:
                self.running[index] = (
                    self.queued.pop(index),
                    job.run_start_time,
                )
        # Check to see if any previously running tasks have completed
        for index, (job, _) in list(self.running.items()):
            try:
                is_done = job.done
            except ValueError:  # raised by Job.done if the job has errored
                errored = True
                is_done = False
            else:
                errored = False
            if is_done:
                self.successful[index] = self.running.pop(index)[0]
            elif job.errored or errored:
                self.errored[index] = self.running.pop(index)[0]

    @property
    def all_failed(self) -> bool:
        pending = self.is_partial and not self.settled
        return (self.unrunnable or self.errored or self._block_errors) and not (
            self.successful or self.blocked or self.queued or pending
        )

    def _lazy_reads(self, index: ty.Any) -> ty.Iterator[tuple[str, LazyField, ty.Any]]:
        """The lazy inputs of one element that are read from upstream (i.e. are not
        state values), with the state index each is read at

        Parameters
        ----------
        index : int, tuple or None
            The state index of the element (its key on a partial node), None if the
            node has no state
        """
        if self.is_partial:
            vals = self._vals[index]
            inputs_ind = self._inputs_ind[index]
        else:
            vals = self.state.states_val[index] if index is not None else {}
            inputs_ind = self.state.inputs_ind[index] if index is not None else {}
        for name, value in attrs_values(self.node._task).items():
            state_key = f"{self.name}.{name}"
            if isinstance(value, LazyField) and state_key not in vals:
                yield name, value, inputs_ind.get(state_key)

    def _split_task(self, index: ty.Any) -> "Task[OutputType]":
        """Resolve the task of one element of the node from its state values and the
        upstream values its lazy inputs read

        Parameters
        ----------
        index : int, tuple or None
            The state index of the element (its key on a partial node), None if the
            node has no state
        """
        resolved = {}
        if index is not None:
            if self.is_partial:
                vals = self._vals[index]
            else:
                vals = self.state.states_val[index]
            for name in self.node.input_names:
                if (state_key := f"{self.name}.{name}") in vals:
                    resolved[name] = vals[state_key]
        for name, value, state_index in self._lazy_reads(index):
            resolved[name] = value._get_value(
                workflow=self.workflow, graph=self.graph, state_index=state_index
            )
        return attrs.evolve(self.node._task, **resolved)

    def _upstream(self, index: ty.Any) -> list[tuple["NodeExecution", ty.Any]]:
        """The upstream nodes, and what of them, that ``_split_task`` reads to resolve
        one element

        Parameters
        ----------
        index : int, tuple or None
            The state index of the element (its key on a partial node), None if the
            node has no state
        """
        upstream = []
        for _, value, state_index in self._lazy_reads(index):
            if isinstance(value, LazyOutField):
                node = self.graph.node(value._node.name)
                upstream.append((node, node._read_indices(state_index)))
        return upstream

    def _read_indices(self, state_index: ty.Any) -> ty.Any:
        """What of this node a downstream lazy field reads at the given state index
        (see ``LazyOutField._get_value``)

        Parameters
        ----------
        state_index : int, tuple or None
            The state index the lazy field is read at, None to read all elements, which
            are returned as a range (``ALL`` on a partial node) so that readers need not
            store every index
        """
        if self.is_partial:
            spec = self._partial
            if state_index is None:
                return ALL
            if isinstance(state_index, int):  # a barrier reader, after the seal
                if spec.group_paths is None:
                    return [self._keys_by_rank[state_index]]
                state_index = self._legacy_groups[state_index]
            if spec.group_paths is None:
                return [state_index]
            if not spec.group_paths:
                return ALL
            return GroupRead(state_index)
        if not self.state:
            return [None]
        if state_index is None or (self.state.combiner and not self.state.ind_l_final):
            return range(len(self.state.states_ind))
        return list(self.state.final_combined_ind_mapping[state_index])

    def get_runnable_tasks(self, graph: DiGraph) -> list["Job[TaskType]"]:
        """For a given node, release the elements whose upstream elements have
        completed successfully, and mark unrunnable the elements that read an errored
        or unrunnable upstream element.

        Parameters
        ----------
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object

        Returns
        -------
        runnable : list[Job]
            List of jobs that are ready to run
        """
        predecessors: list[NodeExecution] = graph.predecessors[self.node.name]
        # ``done`` refreshes a predecessor's status, so it is read for every predecessor
        # before any ``errored``/``unrunnable`` check: one that has just failed is then
        # seen as errored rather than as done-and-successful
        preds_done = [p.done for p in predecessors]
        if not self.started:
            # a predecessor that failed as a whole fails this node before it is
            # classified (classifying reads the predecessors' prepared states)
            if whole := [
                p
                for p in predecessors
                if (p._tasks is None and p.unrunnable) or p._node_error is not None
            ]:
                self.unrunnable = {None: whole}
                self.blocked = {}
            elif all(p.started for p in predecessors):
                mode = self._classify()
                if mode == "barrier" and (
                    unrunnable := [
                        p
                        for p in predecessors
                        if p.errored or p.unrunnable or p._block_errors
                    ]
                ):
                    self.unrunnable = {None: unrunnable}
                    self.blocked = {}
                elif mode == "error":
                    self._fail_node(self._mode_error)
                elif mode == "partial":
                    self._start_partial()
                elif mode == "static" or all(preds_done):
                    self.start()
        if not self.started:
            return []
        progress = tuple(p._token() for p in predecessors)
        pending = self.is_partial and not self.settled
        if (self.blocked or pending) and progress != self._progress:
            self._progress = progress
            if self.is_partial:
                self._register()
                self._expand_ready()
            indices = sorted(self.blocked) if self.is_partial else list(self.blocked)
            for index in indices:
                upstream = self.blocked[index]
                if unrunnable := list(
                    dict.fromkeys(p for p, inds in upstream if _has_failed(p, inds))
                ):
                    del self.blocked[index]
                    self.unrunnable[index] = unrunnable
                elif all(_has_succeeded(p, inds) for p, inds in upstream):
                    del self.blocked[index]
                    self._release(index)
        if self.is_partial:
            self.sealed  # assigns the flat indices as soon as the node is sealed
        return list(self.queued.values())

    def _release(self, index: ty.Any) -> None:
        """Queue the job of an element whose upstream elements have all succeeded, or
        record the element as errored if its inputs cannot be resolved from them"""
        try:
            job = self.queued[index] = self.job(index)
            job._awaiting_run = True
        except Exception as exc:
            if self.is_partial:
                name = element_label(self.name, None, index)
                entry = error_entry(
                    self.name, None, exc, phase="resolution", state_key=index
                )
            else:
                name = element_label(self.name, index)
                entry = error_entry(self.name, index, exc, phase="resolution")
            self.errored[index] = exc
            self._resolution_errors.append(
                f"Job {name!r}, inputs could not be resolved, errored:\n{format_exc()}"
            )
            self._resolution_entries.append(entry)

    def pop_resolution_errors(
        self, entries: "list[JobErrorEntry] | None" = None
    ) -> list[str]:
        """The input-resolution errors recorded since the last call, and their records
        added to ``entries`` if given"""
        errors, self._resolution_errors = self._resolution_errors, []
        records, self._resolution_entries = self._resolution_entries, []
        if entries is not None:
            entries.extend(records)
        return errors


def _sort_key(key: tuple) -> tuple:
    """Orders block keys, whose positions may be None (a stateless source)"""
    return tuple((k is not None, k) for k in key)


def _has_failed(node: NodeExecution, indices: ty.Any) -> bool:
    """Whether any of the elements of a node that a reader reads has errored or is
    unrunnable"""
    if indices is ALL:
        return bool(node.errored or node.unrunnable or node._block_errors)
    if isinstance(indices, GroupRead):
        return node._group_failed(indices.key)
    if not node.is_partial and len(indices) == len(node._tasks):
        return bool(node.errored or node.unrunnable)
    return any(i in node.errored or i in node.unrunnable for i in indices)


def _has_succeeded(node: NodeExecution, indices: ty.Any) -> bool:
    """Whether the elements of a node that a reader reads have all succeeded"""
    if indices is ALL:
        if node.is_partial and not node.sealed:
            return False
        return len(node.successful) == len(node._tasks)
    if isinstance(indices, GroupRead):
        return node._group_closed(indices.key) and all(
            m in node.successful for m in node._groups.get(indices.key, ())
        )
    if not node.is_partial and len(indices) == len(node._tasks):
        return len(node.successful) == len(node._tasks)
    return all(i in node.successful for i in indices)


def job_error(job: "Job", exc: BaseException) -> str:
    """Describe the error of a job"""
    error_msg = "".join(format_exception(type(exc), exc, exc.__traceback__))
    if match := re.match(r'.*"""(.*)""".*', error_msg, flags=re.DOTALL | re.MULTILINE):
        error_msg = match.group(1)
    task_name = element_label(
        job.name,
        job.state_index if job.state_key is None else None,
        job.state_key,
    )
    return f"Job {task_name!r}, {job.task!r}, errored:{error_msg}"


def error_entry(
    job_name: str,
    state_index: int | None,
    exc: BaseException,
    phase: str = "run",
    state_key: tuple | None = None,
    block: tuple | None = None,
) -> JobErrorEntry:
    """The record of an exception"""
    exc_class = type(exc)
    return JobErrorEntry(
        job_name=job_name,
        state_index=state_index,
        exception_type=f"{exc_class.__module__}.{exc_class.__qualname__}",
        message=str(exc),
        traceback="".join(format_exception(exc_class, exc, exc.__traceback__)),
        phase=phase,
        state_key=state_key,
        block=block,
    )


@attrs.frozen
class Failed:
    """The outcome of a job that could not be run: the error that stopped it, as raised"""

    error: BaseException


Outcome = ty.Union["Result", Failed]  # NotRun joins in R7


@attrs.frozen
class Settled:
    """A job the runner completed itself, without dispatching it"""

    job: Job
    how: ty.Literal["stored", "shared", "gated", "absent", "workflow-stored"]
    result: "Result"


@attrs.frozen
class Unrunnable:
    """An element marked unrunnable because it reads a failed or unrunnable upstream"""

    node: str
    index: int | None
    upstream: list[str]


@attrs.frozen
class Round:
    """What one call of :meth:`GraphRunner.ready` decided

    Attributes
    ----------
    dispatch : list[Job]
        The jobs to run, each handed out once; ``job.rerun`` is already set
    settled : list[Settled]
        The jobs completed inside the call, never dispatched (empty until R7)
    unrunnable : list[Unrunnable]
        The elements marked unrunnable inside the call (empty until R7)
    retry_after : float | None
        Seconds until a waiting workflow lease is tried again (``None`` until R6)
    """

    dispatch: list[Job] = attrs.field(factory=list)
    settled: list[Settled] = attrs.field(factory=list)
    unrunnable: list[Unrunnable] = attrs.field(factory=list)
    retry_after: float | None = None


class GraphRunner:
    """The execution state of one workflow job, and the rule by which its jobs are
    released.

    A driver calls :meth:`ready` for the jobs to run, runs them however it chooses,
    reports each with :meth:`complete`, and repeats until :attr:`done`. The runner reads
    the store, but never runs a job.

    Parameters
    ----------
    task : Task
        The workflow task to expand
    store : ResultStore
        Where the results of the jobs are kept
    on_error : "raise_at_end" | "fail_fast"
        The mode of the driver, which reads :attr:`failed` to decide whether to stop
    rerun : bool
        Whether to force the recomputation of the results of the workflow's jobs; the
        outer job runs with it and the inner jobs with it if ``propagate_rerun``
    propagate_rerun : bool
        Whether ``rerun`` reaches the inner jobs
    environment : Environment, optional
        Reserved: the environment of each job is the one of its node
    hooks : TaskHooks, optional
        Reserved: the hooks of each job are the ones of its node
    lookup_before_dispatch : bool
        Reserved: has no effect until the store is read for every job (R7)
    submitter : Submitter, optional
        Transitional: the submitter the jobs are created for (a job reads its audit,
        cache root and store from it). Without it a ``debug`` submitter over ``store``
        is created. R6 and R7 remove it
    job : Job, optional
        The workflow job that the runner expands, whose :meth:`result` it reads
    """

    def __init__(
        self,
        task: "Task",
        store: "ResultStore",
        *,
        on_error: OnError,
        rerun: bool = False,
        propagate_rerun: bool = True,
        environment: "Environment | None" = None,
        hooks: "TaskHooks | None" = None,
        lookup_before_dispatch: bool = False,
        submitter: "Submitter | None" = None,
        job: "Job | None" = None,
    ):
        self.task = task
        self.store = store
        self.on_error = on_error
        self.rerun = rerun
        self.propagate_rerun = propagate_rerun
        self.environment = environment
        self.hooks = hooks
        self.lookup_before_dispatch = lookup_before_dispatch
        if submitter is None:
            from pydra.engine.submitter import Submitter

            submitter = Submitter(worker="debug", store=store, telemetry=False)
        self.submitter = submitter
        self.job = job
        self.workflow: "Workflow" = task.construct()
        self.exec_graph: DiGraph[NodeExecution] = self.workflow.execution_graph(
            submitter=submitter
        )
        self._handed: set[Job] = set()
        self._errors: list[str] = []
        self._entries: list[JobErrorEntry] = []
        self._failed = False
        self._first_error: BaseException | None = None
        self._collect_resolution_errors()

    @property
    def _inner_rerun(self) -> bool:
        return self.rerun and self.propagate_rerun

    def _collect_resolution_errors(self) -> None:
        for node in self.exec_graph.nodes:
            records: list[JobErrorEntry] = []
            self._errors.extend(node.pop_resolution_errors(records))
            self._entries.extend(records)
            if records:
                self._failed = True

    def _runnable(self) -> list[Job]:
        return self._runnable_jobs(self.exec_graph)

    @staticmethod
    def _runnable_jobs(graph: "DiGraph[NodeExecution]") -> list[Job]:
        """The jobs of all the nodes that are released, in node order"""
        jobs: list[Job] = []
        not_started = set()
        node: NodeExecution
        for node in graph.sorted_nodes:
            if node.done:
                continue
            # since the list is sorted (breadth-first) we can stop
            # when we find a job that depends on any job that is already in jobs
            preds = set(graph.predecessors[node.name])
            if preds.intersection(not_started):
                break
            if not node.started:
                not_started.add(node)
            jobs.extend(node.get_runnable_tasks(graph))
        return jobs

    def ready(self, limit: int | None = None) -> Round:
        """Release the jobs whose upstream jobs have completed

        Parameters
        ----------
        limit : int, optional
            The most jobs this call hands out

        Returns
        -------
        Round
            The jobs to dispatch, each handed out once, with ``job.rerun`` set
        """
        candidates = [j for j in self._runnable() if j not in self._handed]
        if candidates and not self._inner_rerun:
            candidates = self._read_succeeded(candidates)
        if limit is not None and len(candidates) > limit:
            if limit > 0:
                logger.info(
                    "Reducing number of tasks to run concurrently from %d to %d",
                    len(candidates),
                    limit,
                )
            candidates = candidates[: max(limit, 0)]
        for job in candidates:
            self._handed.add(job)
            job.rerun = self._inner_rerun
        self._collect_resolution_errors()
        return Round(dispatch=candidates)

    def _read_succeeded(self, jobs: list[Job]) -> list[Job]:
        dispatch = self._read_succeeded_jobs(self.store, jobs)
        self._handed.update(j for j in jobs if j not in dispatch)
        return dispatch

    @staticmethod
    def _read_succeeded_jobs(store: "ResultStore", jobs: list[Job]) -> list[Job]:
        """Complete, without their leases, the jobs whose slots carry a success record,
        and return the other jobs, which are to be dispatched.

        Each job costs one existence check of its own record
        (`PickleStore.read_succeeded`). A job read this way is complete: its result is
        cached on it and it is not dispatched, so it takes no lease. A job without a
        record (its slot has errored, is running, is in a read-only cache or was written
        before records existed), or whose result the store's ``lookup`` does not return,
        returns errored or returns from another context, is dispatched and runs under its
        lease as before. A store without records dispatches every job.
        """
        read = getattr(store, "read_succeeded", None)
        if read is None:
            return jobs
        dispatch = []
        for job in jobs:
            result = read(job.checksum, job.context, job.task)
            if result is None:
                dispatch.append(job)
                continue
            job.hooks.pre_run(job)
            job._cached_result = result
            job._awaiting_run = False
        return dispatch

    def complete(self, job: Job, outcome: Outcome) -> None:
        """Record the outcome of a job that was handed out

        Parameters
        ----------
        job : Job
            A job from :meth:`ready`
        outcome : Result | Failed
            What came of running it. A :class:`Failed` is recorded as the job's error and
            the elements that read the job become unrunnable
        """
        job._awaiting_run = False
        if isinstance(outcome, Failed):
            exc = outcome.error
            job._errored = True
            self._failed = True
            if self._first_error is None:
                self._first_error = exc
            self._errors.append(job_error(job, exc))
            self._entries.append(
                error_entry(
                    job.name,
                    job.state_index if job.state_key is None else None,
                    exc,
                    state_key=job.state_key,
                )
            )
        elif getattr(outcome, "errored", False):
            self._failed = True
        self._collect_resolution_errors()

    def stuck(self) -> str | None:
        """A diagnosis of why nothing can progress, or None if the graph is done"""
        if self.done:
            return None
        wf = self.workflow
        exec_graph = self.exec_graph
        not_done = "\n".join(
            (
                f"{n.name}: started={bool(n.started)}, "
                f"blocked={list(n.blocked or ())}, queued={list(n.queued)}"
                + n._blocks_summary()
            )
            for n in exec_graph.nodes
            if not n.done
        )
        msg = (
            "Something has gone wrong when retrieving the predecessor "
            f"results. Not able to get any more tasks but he following "
            f"nodes of the {wf.name!r} workflow are not done:"
            f"\n{not_done}\n\n"
        )
        not_done = [n for n in exec_graph.nodes if not n.done]
        msg += "\n" + ", ".join(
            f"{t.name}: {t.done}" for t in not_done[0].queued.values()
        )
        # Get blocked tasks and the predecessors they are blocked on
        outstanding: dict[Job, list[Job]] = {
            t: [p for p in exec_graph.predecessors[t.name] if not p.done]
            for t in exec_graph.sorted_nodes
        }

        for job, blocked_on in outstanding.items():
            if not blocked_on:
                continue
            msg += f"- '{job.name}' node blocked due to\n"
            for pred in blocked_on:
                msg += f"    - '{pred.name}' node is not done\n"
            msg += "\n"
        return msg

    @property
    def done(self) -> bool:
        """Whether every node of the graph has completed"""
        return all(n.done for n in self.exec_graph.nodes)

    @property
    def entries(self) -> list[JobErrorEntry]:
        """The record of each errored job and of each element whose inputs could not be
        resolved, in the order of failure"""
        return self._entries

    @property
    def errors(self) -> list[str]:
        """A description of each entry"""
        return self._errors

    @property
    def failed(self) -> bool:
        """Whether any job has failed so far"""
        return self._failed

    @property
    def first_error(self) -> BaseException | None:
        """The first exception reported by a :class:`Failed`, as raised"""
        return self._first_error

    def result(self) -> "Result":
        """The result of the workflow job, read from the store"""
        if self.job is None:
            raise RuntimeError("The runner was created without a workflow job")
        return self.job.result()
