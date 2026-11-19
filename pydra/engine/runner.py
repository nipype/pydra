"""The graph runner: the state of a workflow's execution graph, and the rule by which
its jobs are released, completed and marked unrunnable. Drivers (see
:class:`~pydra.engine.submitter.Submitter`) loop over a :class:`GraphRunner`, taking the
jobs it hands out, running them and reporting each outcome."""

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
from pydra.engine.partial import (
    ALL,
    GroupRead,
    PartialExpansion,
    classify,
    key_text,
)
from pydra.engine.state import PydraStateError, is_sequence

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


def element_label(
    name: str, state_index: int | None, state_key: tuple | None = None
) -> str:
    """The name of an element in messages: ``B[1,0]`` on a partial node (its key),
    ``B(3)`` on any other split node, ``B`` without a split"""
    if state_key is not None:
        return f"{name}[{key_text(state_key)}]"
    if state_index is not None:
        return f"{name}({state_index})"
    return name


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
        # the blocks, keys and groups of a partial node
        self._partial: PartialExpansion | None = None

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
            index = self._partial.keys_by_rank[index]
        if (job := self._tasks[index]) is None:
            if self.is_partial:
                state_index, state_key = self._partial.rank.get(index), index
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

    def _classify(self) -> str:
        """Classify the node (see :func:`~pydra.engine.partial.classify`); its upstream
        nodes must have started"""
        if self._mode is None:
            self._mode = "static"  # while the upstream nodes are classified
            mode, detail = classify(self)
            if mode == "partial":
                self._partial = detail
            elif mode == "error":
                self._mode_error = detail
            self._mode = mode
        return self._mode

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

    # ------------------------------------------------------------------------------
    # What a node shows its readers (partial nodes delegate to ``self._partial``)

    def _final_keys(self) -> list:
        """The final keys of the node that exist now"""
        if self.is_partial:
            return self._partial.final_keys()
        if not self.state:
            return [None]
        return list(range(len(self.state.states_ind_final)))

    def _final_fields(self, key) -> dict:
        """The coordinate of each final path of one final key"""
        if self.is_partial:
            return self._partial.final_fields(key)
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
        """Each failed or errored block of a partial node, own or inherited, with the
        fields it fixes in the node's path space"""
        return self._partial.failed if self.is_partial else []

    @property
    def _block_errors(self) -> list[str]:
        """The errors of the blocks of a partial node that could not be expanded"""
        return self._partial.block_errors if self._partial is not None else []

    @property
    def settled(self) -> bool:
        """Whether every element the node will have is known: its sources are settled,
        every source key has a registered block, and no block is pending. A node that is
        not partial is settled once started"""
        if not self.is_partial:
            return bool(self.started)
        return self._partial.settled()

    @property
    def sealed(self) -> bool:
        """Whether the node is settled with every block expanded and every source
        sealed; the elements of a partial node then get their flat ``state_index``"""
        if not self.is_partial:
            return bool(self.started)
        return self._partial.sealed()

    def _may_produce(self, assignment: dict) -> bool:
        """Whether an element whose final coordinates agree with ``assignment`` may still
        appear (conservative: True when it cannot be ruled out)"""
        if not self.is_partial:
            return not self.started
        return self._partial.may_produce(assignment)

    def _has_open_group(self, assignment: dict) -> bool:
        return self.is_partial and self._partial.has_open_group(assignment)

    def _blocks_summary(self) -> str:
        """The blocks of a partial node that are not expanded, for ``stuck()``"""
        return self._partial.summary() if self.is_partial else ""

    def _token(self) -> tuple:
        """The progress of the node as its readers see it"""
        token = (len(self.successful), len(self.errored), len(self.unrunnable))
        if self.is_partial:
            token += self._partial.token()
        return token

    def _legacy_prepare(self) -> "bool | Exception":
        """Prepare a partial node's ``State`` as today's barrier path would, for a
        barrier reader (see ``PartialExpansion.legacy_prepare``)"""
        return self._partial.legacy_prepare()

    def read_value(self, retrieve: ty.Callable[["Job"], ty.Any], state_index: ty.Any):
        """The value of a partial node's output for a reader (see
        ``LazyOutField._get_value``): one element, one group, or all of them once
        sealed"""
        return self._partial.read_value(retrieve, state_index)

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
            vals = self._partial.vals[index]
            inputs_ind = self._partial.inputs_ind[index]
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
                vals = self._partial.vals[index]
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
            return self._partial.read_spec(state_index)
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
                    self._partial.start()
                elif mode == "static" or all(preds_done):
                    self.start()
        if not self.started:
            return []
        progress = tuple(p._token() for p in predecessors)
        pending = self.is_partial and not self.settled
        if (self.blocked or pending) and progress != self._progress:
            self._progress = progress
            if self.is_partial:
                self._partial.register()
                self._partial.expand_ready()
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


def _has_failed(node: NodeExecution, indices: ty.Any) -> bool:
    """Whether any of the elements of a node that a reader reads has errored or is
    unrunnable"""
    if indices is ALL:
        return bool(node.errored or node.unrunnable or node._block_errors)
    if isinstance(indices, GroupRead):
        return node._partial.group_failed(indices.key)
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
        return node._partial.group_succeeded(indices.key)
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
