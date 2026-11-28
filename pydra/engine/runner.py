"""The graph runner: the state of a workflow's execution graph, and the rule by which
its jobs are released, completed and marked unrunnable. Drivers (see
:class:`~pydra.engine.submitter.Submitter`) loop over a :class:`GraphRunner`, taking the
jobs it hands out, running them and reporting each outcome."""

import asyncio
import concurrent.futures
import typing as ty
from traceback import format_exc, format_exception
import re
from collections import defaultdict
from datetime import datetime
import logging
import time
from uuid import uuid4
import attrs
import cloudpickle as cp
from pydra.engine.graph import DiGraph
from pydra.engine.progress import (
    UNSET,
    TrackedSet,
    changed,
    changes,
    tracked,
)
from pydra.utils.general import attrs_values
from pydra.engine.lazy import LazyField, LazyOutField
from pathlib import Path
from pydra.engine.job import Job
from pydra.engine.absent import (
    WHOLE,
    Absent,
    AbsentNotAccepted,
    absence_identity,
    contains_absent,
    resolving,
    task_definition,
)
from pydra.engine.lease import LeaseLost
from pydra.engine.partial import (
    ALL,
    GroupRead,
    PartialExpansion,
    classify,
    key_text,
)
from pydra.engine.result import Result
from pydra.engine.state import PydraStateError, is_sequence
from pydra.utils.general import get_fields, is_workflow
from pydra.utils.typing import StateArray

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task, Outputs
    from pydra.engine.node import Node
    from pydra.engine.plan import Plan
    from pydra.engine.store import ResultStore
    from pydra.engine.workflow import Workflow
    from pydra.engine.submitter import Submitter


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
    exception_type : str or None
        The qualified name of the exception's class, ``module.Class``; None for a
        ``"stopped"`` entry, which records no exception
    message : str
        ``str()`` of the exception
    traceback : str
        The formatted traceback
    phase : str
        ``"run"`` if the job ran and errored, ``"resolution"`` if its inputs could
        not be resolved from its upstream results, so it never ran, ``"expansion"`` if
        a block of a partial node could not be expanded, so it has no element,
        ``"lease"`` if the lease of a workflow job could not be taken, ``"commit"`` if
        its result could not be committed (the lease was lost), ``"stopped"`` if the
        job was handed out but the run ended before it completed (its message names
        the failure that stopped the run)
    state_key : tuple | None
        The key of the element on a partial node, whose ``state_index`` is then None
    block : tuple | None
        The key of the block that could not be expanded (phase ``"expansion"``)
    """

    job_name: str
    state_index: int | None
    exception_type: str | None
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


class Logged(dict):
    """A dict that also lists each key set, in order (``log``, append-only), so that a
    reader can take the keys set since it last looked without scanning the dict"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log: list = list(self)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        try:
            self.log.append(key)
        except AttributeError:  # being unpickled or copied: items come before state
            self.log = [key]


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
    # The nested workflows that hold the node, outermost first
    path: tuple[str, ...]

    graph: DiGraph["NodeExecution"] | None

    # The jobs completed absent and not yet collected by the runner
    _absent: list["Job[TaskType]"]
    # The Absent a splitter input resolved to, which leaves the node no elements
    absent: "Absent | None"
    # The node whose state it inherits that absence from, None if its own splitter input
    absent_inherits: str | None

    # Descriptions of input-resolution errors not yet collected by ``Submitter``
    _resolution_errors: list[str]
    _resolution_entries: "list[JobErrorEntry]"
    # The progress of the predecessors when the blocked elements were last checked
    _progress: tuple | None

    # the tables whose changes are progress for `GraphRunner.ready` (`progress.changed`)
    _TRACKED = frozenset(
        {
            "blocked",
            "successful",
            "errored",
            "unrunnable",
            "queued",
            "_mode",
            "_partial",
        }
    )

    def __setattr__(self, name, value):
        if name in self._TRACKED:
            value = tracked(value)
            if self.__dict__.get(name, UNSET) != value:
                changed()
        super().__setattr__(name, value)

    def __init__(
        self,
        node: "Node",
        submitter: "Submitter",
        workflow: "Workflow",
        path: tuple[str, ...] = (),
    ):
        self.name = node.name
        self.path = path
        self.node = node
        self.submitter = submitter
        # Initialize the state dictionaries
        self._tasks = None
        self.blocked = None
        self.successful = Logged()
        self.errored = Logged()
        self.queued = {}
        self.running = {}  # Not used in logic, but may be useful for progress tracking
        self.unrunnable = Logged()
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
        self._absent = []
        self.absent = None
        self.absent_inherits = None

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
            job = self._tasks[index] = self._new_job(index)
        return job

    def _new_job(self, index: ty.Any) -> "Job[TaskType]":
        """The job of an element, formed without being kept (:meth:`job` keeps it)"""
        if self.is_partial:
            state_index, state_key = self._partial.rank.get(index), index
        else:
            state_index, state_key = index, None
        task, propagated, accepted = self._resolve(index)
        job = Job(
            task=task,
            submitter=self.submitter,
            environment=self.node._environment,
            name=self.node.name,
            hooks=self.node._hooks,
            resources=self.node._resources,
            state_index=state_index,
            state_key=state_key,
        )
        job.workflow_path = self.path
        job._accepted_absent = accepted
        if propagated is not None:
            # its identity is the absence record: the task's checksum is never computed
            job._propagated_absent, identity = propagated
            job._unsuffixed_checksum = job._checksum = identity
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
    # Whole-node absence (RFC 0002 section 2.3(a))

    def _inherit_absent(self) -> bool:
        """Make the node absent as a whole if its state inherits that of a wholly absent
        node; whether it did"""
        for upstream in (self.state.other_states or {}) if self.state else ():
            if (inherited := self.graph.node(upstream).absent) is not None:
                self._become_absent(inherited, upstream)
                return True
        return False

    def _become_absent(self, absent: Absent, inherits: str | None = None) -> None:
        """Make the node absent as a whole: it has no elements, is never classified,
        expanded, routed, placed or looked up, and every read of it is ``absent``. Only
        called before the node has an element (a partial node that has expanded no
        block)"""
        self.absent = absent
        self.absent_inherits = inherits
        self._mode = "static"
        self._partial = None
        self._tasks = {}
        self.blocked = {}

    # ------------------------------------------------------------------------------
    # Starting

    def start(self) -> None:
        """Prepare the state of the node and record, for each of its elements, the
        upstream elements it reads"""
        indices: range | list[int | None]
        if self.state:
            self._inherit_absent()
            values = {}
            for name, value in self.node.state_values.items():
                if self.absent is not None:
                    break
                if name in self.node.state.current_splitter_rpn:
                    if name in ("*", "."):
                        continue
                    if isinstance(value, LazyField):
                        values[name] = value._get_value(
                            workflow=self.workflow, graph=self.graph
                        )
                    else:
                        values[name] = value
                    if isinstance(values[name], Absent):
                        # absent as a whole: no elements, and readers get the Absent
                        self.absent = values[name]
                        break
                    if isinstance(value, LazyOutField):
                        _check_splittable(self.name, name, values[name])
            if self.absent is not None:
                indices = []
            elif self._mode == "barrier":
                try:
                    self._check_barrier(values)
                    self.state.prepare_states(values)
                    self.state.prepare_inputs()
                except PydraStateError as exc:
                    self._fail_node(exc)
                    return
                except Exception as exc:  # today's State cannot prepare this shape
                    error = PydraStateError(
                        f"Node {self.name!r} could not prepare its state: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    error.__cause__ = exc
                    self._fail_node(error)
                    return
                indices = range(len(self.state.states_ind))
            else:
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
        # A job handed to an executor is not polled: it is classified when its outcome
        # is reported to ``GraphRunner.complete``, which holds the result on the job
        for index, job in list(self.queued.items()):
            status = self._reported_status(job)
            if status is None:
                continue
            if status == "done":
                self.successful[index] = self.queued.pop(index)
            elif status == "errored":
                self.errored[index] = self.queued.pop(index)
            elif job.run_start_time:
                self.running[index] = (
                    self.queued.pop(index),
                    job.run_start_time,
                )
        for index, (job, _) in list(self.running.items()):
            status = self._reported_status(job)
            if status == "done":
                self.successful[index] = self.running.pop(index)[0]
            elif status == "errored":
                self.errored[index] = self.running.pop(index)[0]

    @staticmethod
    def _reported_status(job: "Job") -> str | None:
        """``"done"`` or ``"errored"`` for a job whose outcome is known, ``"pending"``
        for one that is not handed out, ``None`` for one still awaiting its report"""
        held = job._held
        if held is not None:
            return "errored" if held.errored else "done"
        if job.errored:
            return "errored"
        if job._awaiting_run:
            return None
        try:
            return "done" if job.done else "pending"
        except ValueError:  # raised by Job.done if the job has errored
            return "errored"

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
        return self._resolve(index)[0]

    def _resolve(
        self, index: ty.Any
    ) -> tuple[
        "Task[OutputType]", tuple[Absent, str] | None, list[tuple[str, int, str]]
    ]:
        """Resolve the task of one element, applying each input field's ``absent`` mode
        (RFC 0002 section 2.3(a)). A whole-value ``Absent`` propagates under ``"drop"``
        too, as it has no elements to drop.

        Parameters
        ----------
        index : int, tuple or None
            The state index of the element (its key on a partial node), None if the
            node has no state

        Returns
        -------
        task : Task
            The element's task, a ``"drop"`` field's combined value without its absent
            elements and a ``"fill"`` field's with each replaced by the fill
        propagated : tuple[Absent, str] or None
            The ``Absent`` the element completes with, of the first propagating field in
            declaration order, and the job's identity
            (:func:`~pydra.engine.absent.absence_identity`); None if it runs
        accepted : list[tuple[str, int, str]]
            The ``(field, element index, cause)`` of every accepted, replaced or filled
            ``Absent``

        Raises
        ------
        AbsentNotAccepted
            An ``Absent`` is left inside the value of a field that does not accept it.
        """
        resolved = {}
        combined = set()
        reads = {}
        with resolving():
            if index is not None:
                if self.is_partial:
                    vals = self._partial.vals[index]
                else:
                    vals = self.state.states_val[index]
                for name in self.node.input_names:
                    if (state_key := f"{self.name}.{name}") in vals:
                        resolved[name] = vals[state_key]
            for name, value, state_index in self._lazy_reads(index):
                reads[name] = (value, state_index)
                resolved[name] = value._get_value(
                    workflow=self.workflow, graph=self.graph, state_index=state_index
                )
                if self._reads_combined(value, state_index):
                    combined.add(name)
            task = attrs.evolve(self.node._task, **resolved)
        propagated, accepted, replaced = None, [], {}
        modes = {}
        for field in get_fields(task):
            value = getattr(task, field.name)
            mode = modes[field.name] = field.absent or "propagate"
            if isinstance(value, Absent):
                if mode == "drop" and field.name in combined:
                    # a combined read of a wholly absent node: nothing is left
                    accepted.append((field.name, WHOLE, value.cause))
                    replaced[field.name] = []
                    continue
                if mode == "fill":
                    accepted.append((field.name, WHOLE, value.cause))
                    replaced[field.name] = field.fill
                    continue
                if mode != "accept":  # a single value has no elements to drop
                    propagated = (field.name, WHOLE, value)
                    break
                accepted.append((field.name, WHOLE, value.cause))
            elif field.name in combined and isinstance(value, (list, StateArray)):
                where = [i for i, v in enumerate(value) if isinstance(v, Absent)]
                if not where:
                    continue
                if mode == "propagate":
                    propagated = (field.name, where[0], value[where[0]])
                    break
                accepted.extend((field.name, i, value[i].cause) for i in where)
                if mode == "drop":
                    replaced[field.name] = type(value)(
                        v for v in value if not isinstance(v, Absent)
                    )
                elif mode == "fill":
                    replaced[field.name] = type(value)(
                        field.fill if isinstance(v, Absent) else v for v in value
                    )
        if propagated is not None:
            name, element, absent = propagated
            producer = self._producer(reads.get(name), element)
            identity = absence_identity(
                task_definition(task), producer, (name, element, absent.cause)
            )
            return task, (absent, identity), []
        if replaced:
            with resolving():
                task = attrs.evolve(task, **replaced)
        for name, mode in modes.items():
            if mode != "accept" and contains_absent(getattr(task, name)):
                raise AbsentNotAccepted(
                    f"{self.name}.{name} holds an Absent; declare absent='accept' (or "
                    "'fill') on the field, or read the value element by element"
                )
        return task, None, accepted

    def _producer(
        self, read: tuple[LazyField, int | None] | None, element: int
    ) -> str | None:
        """The unsuffixed checksum of the upstream job a propagated ``Absent`` was read
        from (the element of a combined read), None if it was not read from one"""
        if read is None or not isinstance(read[0], LazyOutField):
            return None
        value, state_index = read
        upstream = self.graph.node(value._node.name)
        indices = upstream._read_elements(upstream._read_indices(state_index))
        if element != WHOLE and element < len(indices):
            position = indices[element]
        elif indices:
            position = indices[0]
        else:
            return None
        return upstream.job(position).unsuffixed_checksum

    def _reads_combined(self, value: LazyField, state_index: int | None) -> bool:
        """Whether a lazy input reads a combined value, a list of upstream elements,
        rather than one element's value"""
        if not isinstance(value, LazyOutField):
            return False
        upstream = self.graph.node(value._node.name)
        return bool(upstream.state) and (
            bool(upstream.state.combiner) or state_index is None
        )

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
        if self.absent is not None:  # absent as a whole: it has no elements
            return range(0)
        if self.is_partial:
            return self._partial.read_spec(state_index)
        if not self.state:
            return [None]
        if state_index is None or (self.state.combiner and not self.state.ind_l_final):
            return range(len(self.state.states_ind))
        return list(self.state.final_combined_ind_mapping[state_index])

    def _read_elements(self, spec: ty.Any) -> list:
        """The elements of this node a read spec from :meth:`_read_indices` names"""
        if spec is ALL:
            if self.is_partial and self._partial.is_sealed:
                return list(self._partial.keys_by_rank)
            return sorted(self._tasks) if self.is_partial else list(self._tasks)
        if isinstance(spec, GroupRead):
            return sorted(self._partial.groups.get(spec.key, ()))
        return list(spec)

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
                self.unrunnable = Logged({None: whole})
                self.blocked = {}
            elif all(p.started for p in predecessors) and self._inherit_absent():
                pass  # absent as a whole, in any mode: it has no elements
            elif all(p.started for p in predecessors):
                mode = self._classify()
                if mode == "barrier" and (
                    unrunnable := [
                        p
                        for p in predecessors
                        if p.errored or p.unrunnable or p._block_errors
                    ]
                ):
                    self.unrunnable = Logged({None: unrunnable})
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
            if self.is_partial and not self._tasks and self._inherit_absent():
                return []
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
        record the element as errored if its inputs cannot be resolved from them. An
        element whose propagating input holds an ``Absent`` completes absent at once: it
        is never looked up, shared, dispatched or committed."""
        try:
            job = self.job(index)
            if job._propagated_absent is not None:
                self._complete_absent(index, job)
                return
            self.queued[index] = job
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

    def _complete_absent(self, index: ty.Any, job: "Job[TaskType]") -> None:
        """Complete a job with the ``Absent`` it propagates as every output"""
        absent = job._propagated_absent
        outputs = job.task.Outputs(
            **{f.name: absent for f in get_fields(job.task.Outputs)}
        )
        job._held = Result(
            cache_dir=Path(job.cache_root),
            outputs=outputs,
            task=job.task,
            stored=False,
        )
        job._awaiting_run = False
        self.successful[index] = job
        self._absent.append(job)

    def pop_absent(self) -> list["Job[TaskType]"]:
        """The jobs completed absent since the last call"""
        jobs, self._absent = self._absent, []
        return jobs

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


def _check_splittable(node: str, state_key: str, value: ty.Any) -> None:
    """Refuse an upstream output a splitter splits over that is not a sequence, such as a
    reference an external driver completed the upstream job with (RFC 0002 section 3.5)
    """
    if hasattr(value, "__len__") and hasattr(value, "__getitem__"):
        return
    field = state_key.rpartition(".")[2]
    raise TypeError(
        f"Node {node!r} splits over {field!r}, whose upstream value is a "
        f"{type(value).__name__} ({value!r}): a value a splitter splits over must be a "
        "real sequence, not a reference to one"
    )


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


class RecordedError(RuntimeError):
    """An error recorded by a worker that could not be read back as the exception raised:
    its message is the traceback's last line, :attr:`exception_type` the type's
    qualified name (None if not recorded)"""

    def __init__(self, message: str, exception_type: str | None = None):
        super().__init__(message)
        self.exception_type = exception_type


def result_error(job: "Job", result: "Result") -> BaseException:
    """The error of an errored result as its worker recorded it (``_error.pklz``): the
    exception itself where it was pickled, else a `RecordedError`; a `RuntimeError`
    naming the missing record when there is none. The recorded traceback is kept as
    ``recorded_traceback``."""
    recorded = result.errors or {}
    error = recorded.get("error message")
    if isinstance(error, BaseException):
        return error
    if isinstance(error, str):
        error = [error]
    if error:
        text = "".join(error)
        try:
            exc = cp.loads(recorded["exception pickle"])
        except Exception:  # not recorded, or its class cannot be loaded here
            exc = None
        if not isinstance(exc, BaseException):
            last = text.rstrip().splitlines()[-1] if text.strip() else ""
            exc = RecordedError(last, recorded.get("error type"))
        exc.recorded_traceback = text  # type: ignore[attr-defined]
        return exc
    return RuntimeError(
        f"Job {job.name!r} errored and left no error record in {str(job.cache_dir)!r}"
    )


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
    exception_type = exc.exception_type if isinstance(exc, RecordedError) else None
    return JobErrorEntry(
        job_name=job_name,
        state_index=state_index,
        exception_type=exception_type
        or f"{exc_class.__module__}.{exc_class.__qualname__}",
        message=str(exc),
        traceback=getattr(exc, "recorded_traceback", None)
        or "".join(format_exception(exc_class, exc, exc.__traceback__)),
        phase=phase,
        state_key=state_key,
        block=block,
    )


def job_entry(job: "Job", exc: BaseException, phase: str = "run") -> JobErrorEntry:
    """The record of an exception of a job: by its ``state_key`` on a partial node, by
    its ``state_index`` on any other"""
    if job.state_key is not None:
        return error_entry(job.name, None, exc, phase=phase, state_key=job.state_key)
    return error_entry(job.name, job.state_index, exc, phase=phase)


@attrs.frozen
class Failed:
    """The outcome of a job that could not be run: the error that stopped it, as raised"""

    error: BaseException


@attrs.frozen
class NotRun:
    """The outcome of a job that did not run and did not fail: cancelled, never started, or
    held by infrastructure (``held``). The job stays pending and its readers stay blocked
    until :meth:`GraphRunner.retry` hands it out again; no error is recorded.

    Attributes
    ----------
    reason : str
        Why the job did not run
    held : bool
        Whether the cause was infrastructure (a held queue or environment)
    """

    reason: str
    held: bool


# None: the worker returned no result, which the store holds
Outcome = ty.Union["Result", Failed, NotRun, None]


@attrs.frozen
class Settled:
    """A job the runner completed itself, without dispatching it

    Attributes
    ----------
    job : Job
        The job
    how : str
        ``"stored"`` (its result was read from the store), ``"shared"`` (it has the
        checksum of a job that was handed out, whose outcome it takes),
        ``"workflow-stored"`` (a workflow job whose stored result was used, so it was not
        expanded), ``"absent"`` (a propagating input held an ``Absent``, which is every
        output of its result; it was never looked up, shared or committed)
    result : Result
        Its result
    """

    job: Job
    how: ty.Literal["stored", "shared", "absent", "workflow-stored"]
    result: "Result"


@attrs.frozen
class Unrunnable:
    """An element marked unrunnable because it reads a failed or unrunnable upstream

    Attributes
    ----------
    node : str
        The node's workflow path and name, ``"/"``-joined (``"inner[0]/B"``)
    index : int | tuple | Block | None
        The element's state index (its ``state_key`` on a partial node, or the
        :class:`~pydra.engine.partial.Block` that failed before it was expanded), None
        for the whole node
    upstream : list[str]
        The upstream nodes, named as ``node``, whose failed or unrunnable elements it reads
    """

    node: str
    index: int | None
    upstream: list[str]


@attrs.frozen
class AbsentNode:
    """A node absent as a whole: a splitter input resolved to an ``Absent``, or its state
    inherits that of such a node (RFC 0002 section 2.3(a)). It has no elements, so no job
    of it is dispatched or settled; this is its record.

    Attributes
    ----------
    node : str
        The node's workflow path and name, ``"/"``-joined
    absent : Absent
        The ``Absent`` every read of it receives
    inherits : str | None
        The node, named as ``node``, whose absence its state inherits; None for the node
        whose own splitter input is the ``Absent``
    """

    node: str
    absent: Absent
    inherits: str | None


def node_label(node: "NodeExecution") -> str:
    """The name of a node with the path of the nested workflows that hold it"""
    return "/".join((*node.path, node.name))


@attrs.frozen
class Round:
    """What one call of :meth:`GraphRunner.ready` decided

    Attributes
    ----------
    dispatch : list[Job]
        The jobs to run, each handed out once, grouped by node in node order (workflow
        path, then name); ``job.rerun`` and ``job.attempt`` are already set
    settled : list[Settled]
        The jobs completed by the runner since the last call, never dispatched
    absent_nodes : list[AbsentNode]
        The nodes found absent as a whole since the last call, each reported once
    unrunnable : list[Unrunnable]
        The elements marked unrunnable since the last call, each reported once
    retry_after : float | None
        Seconds until a waiting workflow lease is tried again, ``None`` if none waits
    """

    dispatch: list[Job] = attrs.field(factory=list)
    settled: list[Settled] = attrs.field(factory=list)
    unrunnable: list[Unrunnable] = attrs.field(factory=list)
    absent_nodes: list[AbsentNode] = attrs.field(factory=list)
    retry_after: float | None = None


class _Frame:
    """The expansion of one workflow job: its lease, its execution graph and the errors
    of its nodes. The root frame is the runner's own task, with a job only when the
    runner is given one."""

    def __setattr__(self, name, value):
        if name == "state" and self.__dict__.get(name, UNSET) != value:
            changed()  # progress for `GraphRunner.ready`
        super().__setattr__(name, value)

    def __init__(self, job, parent, path, owned):
        self.job: Job | None = job
        self.parent: "_Frame | None" = parent
        self.path: tuple[str, ...] = path
        # whether the runner leases, checks and commits ``job``
        self.owned: bool = owned
        self.state: str = "leasing" if owned else "expanding"
        self.lease: ty.Any = None
        self.result: "Result | None" = None
        self.workflow: "Workflow | None" = None
        self.graph: "DiGraph[NodeExecution] | None" = None
        self.retry_at: float | None = None
        self.errors: list[str] = []
        self.entries: list[JobErrorEntry] = []

    @property
    def nodes_done(self) -> bool:
        return all(n.done for n in self.graph.nodes)


class GraphRunner:
    """The execution state of one workflow job, and the rule by which its jobs are
    released.

    A driver calls :meth:`ready` for the jobs to run, runs them however it chooses,
    reports each with :meth:`complete`, and repeats until :attr:`done`. The runner reads
    the store, but never runs a leaf job.

    The runner expands nested workflows itself (:meth:`ready` returns the leaf jobs of
    every level), leasing, checking and committing each workflow job; a lease another
    process holds is tried again after :attr:`Round.retry_after`. A driver ends with
    :meth:`close`, which releases the leases of unfinished workflow jobs.

    Every job the runner completes itself (read from the store, shared with a job of the
    same checksum, or a stored workflow job) is reported in :attr:`Round.settled`; a driver
    never receives a twin to dispatch.
    Every element marked unrunnable is reported once in :attr:`Round.unrunnable`. An
    outcome is a ``Result``, a :class:`Failed` or a :class:`NotRun`; a run whose only
    remaining jobs are ``NotRun`` is :attr:`done`, lists them in :attr:`not_run` and is
    incomplete, not failed.

    The runner is not thread-safe: every call is made from the driver's thread.

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
    lookup_before_dispatch : bool
        Whether to look each job without a success record up in the store (``lookup``)
        before handing it out, so that a driver that does not run jobs through
        :meth:`Job.run` is given only jobs with no acceptable result
    submitter : Submitter, optional
        The submitter the jobs are created for (a job reads its audit,
        cache root and store from it). Without it a ``debug`` submitter over ``store``
        is created
    job : Job, optional
        The workflow job that the runner expands, whose :meth:`result` it reads
    own_job : bool
        Whether the runner leases, checks and commits ``job`` (the default); False
        if the caller holds its lease and commits it, as :meth:`Job.run` does
    """

    def __init__(
        self,
        task: "Task",
        store: "ResultStore",
        *,
        on_error: OnError,
        rerun: bool = False,
        propagate_rerun: bool = True,
        lookup_before_dispatch: bool = False,
        submitter: "Submitter | None" = None,
        job: "Job | None" = None,
        own_job: bool = True,
    ):
        self.task = task
        self.store = store
        self.on_error = on_error
        self.rerun = rerun
        self.propagate_rerun = propagate_rerun
        self.lookup_before_dispatch = lookup_before_dispatch
        if submitter is None:
            from pydra.engine.submitter import Submitter

            submitter = Submitter(worker="debug", store=store, telemetry=False)
        self.submitter = submitter
        self.job = job
        # the jobs handed out or completed by the runner, never considered again
        self._handed: set[Job] = TrackedSet()
        self._done_prefix: dict[int, int] = {}
        # the jobs dispatched whose outcome has not been reported
        self._in_flight: set[Job] = set()
        # the released jobs read from the store without a result to serve
        self._checked: set[Job] = TrackedSet()
        self._not_run: dict[Job, NotRun] = {}
        # the jobs completed by the runner and the elements marked unrunnable, not yet
        # reported by ``ready``
        self._settled: list[Settled] = []
        self._reported_unrunnable: set[tuple[int, int | None]] = set()
        self._reported_absent_nodes: set[int] = set()
        # the same-checksum table: the job handed out for each checksum until its outcome
        # is reported, the twins held behind it, and the outcome once reported
        self._first: dict[str, Job] = {}
        self._twins: dict[str, list[Job]] = defaultdict(list)
        self._waiting_twins: set[Job] = set()
        self._finished: dict[str, Outcome] = {}
        # whether the last ``ready`` found nothing to do and nothing awaits an outcome
        self._quiescent = False
        # releases of workflow leases run off the driver's thread (`release_off_thread`)
        self._release_pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._releases: list[concurrent.futures.Future] = []
        self._failed = False
        self._first_error: BaseException | None = None
        self._root_error: BaseException | None = None
        self._root = _Frame(job, None, (), owned=job is not None and own_job)
        self._frames: list[_Frame] = [self._root]
        # the frames not finished, pruned once per pass of ``ready``
        self._open: list[_Frame] = [self._root]
        # the frames waiting to retry their lease
        self._waiting: set[_Frame] = set()
        # the frame each job belongs to, and the frame of each workflow job
        self._frame_of: dict[Job, _Frame] = {}
        self._child: dict[Job, _Frame] = {}
        if self._root.owned:
            job.rerun = rerun
            try:
                self._prepare(self._root)
            except Exception as exc:
                self._root.state = "finished"
                self._root_error = exc
                self._fail_root(self._root, exc)
        else:
            self._expand(self._root, task)

    @property
    def workflow(self) -> "Workflow | None":
        return self._root.workflow

    @property
    def exec_graph(self) -> "DiGraph[NodeExecution] | None":
        return self._root.graph

    @property
    def _errors(self) -> list[str]:
        return self._root.errors

    @property
    def _entries(self) -> "list[JobErrorEntry]":
        return self._root.entries

    @property
    def _inner_rerun(self) -> bool:
        return self.rerun and self.propagate_rerun

    # Workflow jobs

    def _prepare(self, frame: _Frame) -> None:
        """Run the ``pre_run`` hook of a workflow job and make its lease"""
        job = frame.job
        job.hooks.pre_run(job)
        frame.lease = job._acquire()

    def _expand(self, frame: _Frame, task: "Task") -> None:
        frame.workflow = task.construct()
        frame.graph = frame.workflow.execution_graph(
            submitter=self.submitter, path=frame.path
        )
        frame.state = "expanding"
        if frame.job is not None:
            frame.job.return_values = {
                "workflow": frame.workflow,
                "exec_graph": frame.graph,
            }
        self._collect_resolution_errors(frame)

    def _enter(self, job: Job, parent: _Frame) -> None:
        """Start the handling of a released workflow job, once"""
        if job in self._child or job in self._handed:
            return
        job.rerun = self._inner_rerun
        if not job.rerun and self._read_succeeded([job]) == []:
            # stored with a success record: complete without a lease
            self._settled.append(Settled(job, "workflow-stored", job._cached_result))
            return
        if job.state_key is not None:  # an element of a partial node, by its key
            label = f"{job.name}[{key_text(job.state_key)}]"
        elif job.state_index is not None:
            label = f"{job.name}[{job.state_index}]"
        else:
            label = job.name
        frame = _Frame(job, parent, parent.path + (label,), owned=True)
        self._child[job] = frame
        self._frames.append(frame)
        self._open.append(frame)
        try:
            self._prepare(frame)
        except Exception as exc:  # a hook or the store refused: an entry, not a raise
            self._fail(frame, exc, "run")

    def _try_lease(self, frame: _Frame) -> None:
        now = time.monotonic()
        if frame.retry_at is not None and now < frame.retry_at:
            return
        try:
            delay = frame.lease.try_take()
        except Exception as exc:  # LeaseTimeout, StorageUnsupported
            self._fail(frame, exc, "lease")
            return
        if delay is not None:
            frame.retry_at = now + delay
            self._waiting.add(frame)
            return
        job = frame.job
        rerun = self.rerun if frame.parent is None else job.rerun
        if not rerun:
            try:
                result = job.result()
            except Exception as exc:
                self._fail(frame, exc, "run")
                return
            if result is not None and not result.errored:
                job._cached_result = result
                job._held = result
                self._release_lease(frame)
                self._succeeded(frame)
                self._settled.append(Settled(job, "workflow-stored", result))
                return
        if frame.parent is not None or job.attempt is None:
            # the outer job has its attempt from its creation (`Submitter.__call__`); a
            # job that can resume keeps its stable work directory under a fresh attempt
            job.attempt = uuid4().hex
        try:
            frame.result = job._start_workflow(frame.lease)
            self._expand(frame, job.task)
        except Exception as exc:
            self._finish(frame, exc)

    def _release_lease(self, frame: _Frame) -> None:
        lease, frame.lease = frame.lease, None
        if lease is not None:
            if self._release_pool is not None:
                self._releases.append(
                    self._release_pool.submit(lease.__exit__, None, None, None)
                )
            else:
                lease.__exit__(None, None, None)
        for other in self._waiting:  # a lease waited for here may be free now
            other.retry_at = None
        self._waiting.clear()

    def release_off_thread(self) -> None:
        """Release workflow leases on a helper thread from here on, so that a release the
        storage refuses for a while (``RELEASE_PATIENCE``) does not block the driver's
        event loop. The driver then awaits :meth:`released` once it has called
        :meth:`close`."""
        if self._release_pool is None:
            self._release_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pydra-lease-release"
            )

    async def released(self) -> None:
        """Wait, without blocking the event loop, until every lease release begun on the
        helper thread has ended; later releases happen in the calling thread"""
        pool, self._release_pool = self._release_pool, None
        pending, self._releases = self._releases, []
        try:
            for future in pending:
                try:
                    await asyncio.wrap_future(future)
                except Exception as exc:
                    logger.warning("A workflow lease could not be released: %s", exc)
        finally:  # interrupted meanwhile: the rest are waited for here
            concurrent.futures.wait(pending)
            if pool is not None:
                pool.shutdown(wait=False)

    def _succeeded(self, frame: _Frame) -> None:
        frame.state = "finished"
        frame.job._awaiting_run = False

    def _fail(self, frame: _Frame, exc: BaseException, phase: str) -> None:
        """Record the error of a workflow job that stops it (never raised) and release
        its lease; its readers become unrunnable"""
        job = frame.job
        frame.state = "finished"
        job._errored = True
        job._awaiting_run = False
        self._release_lease(frame)
        owner = frame.parent or frame
        self._failed = True
        if self._first_error is None:
            self._first_error = exc
        owner.errors.append(job_error(job, exc))
        owner.entries.append(job_entry(job, exc, phase=phase))

    def _finish(self, frame: _Frame, error: BaseException | None = None) -> None:
        """Commit the result of a workflow job whose nodes are done (or that failed
        while expanding)"""
        job = frame.job
        # an element released in this pass of ``ready`` may have failed to resolve
        self._collect_resolution_errors(frame)
        if error is None and not frame.errors and not frame.nodes_done:
            frame.state = "finished"  # interrupted: nothing is built or stored
            self._release_lease(frame)
            return
        if error is None and frame.errors:
            from pydra.engine.submitter import workflow_error

            error = workflow_error(job, frame.errors, frame.entries)
        try:
            error = job._finish_workflow(frame.lease, frame.result, error)
        except LeaseLost as exc:
            self._fail(frame, exc, "commit")
            return
        except Exception as exc:
            self._fail(frame, exc, "run")
            return
        if error is None:
            job._held = frame.result
            self._release_lease(frame)
            job._complete_workflow(frame.result)
            self._forget(job)
            self._succeeded(frame)
        elif frame.parent is None:
            frame.state = "finished"
            self._release_lease(frame)
            if not frame.errors:  # the workflow itself could not be built
                self._root_error = error
                self._fail_root(frame, error)
        else:
            self._fail(frame, error, "run")

    def _fail_root(self, frame: _Frame, exc: BaseException) -> None:
        self._failed = True
        if self._first_error is None:
            self._first_error = exc
        frame.errors.append(job_error(frame.job, exc))
        frame.entries.append(job_entry(frame.job, exc))

    def _settle_frames(self) -> None:
        self._open = [f for f in self._open if f.state != "finished"]
        for frame in reversed(list(self._open)):
            if frame.state == "expanding" and frame.owned and frame.nodes_done:
                self._finish(frame)

    def _name_stopped(self, frame: _Frame) -> None:
        """Add an entry, naming the failure that stopped the workflow, for each element
        of its nodes that was handed out and has not completed. An element that
        completed has its own outcome, and one never handed out (blocked behind an
        unfinished element, or released but held back by the ``ready`` limit) is not
        an entry: the driver reports it as unrunnable."""
        cause = frame.entries[0] if frame.entries else None
        if cause is None:
            return
        who = element_label(cause.job_name, cause.state_index, cause.state_key)
        first = cause.message.strip().splitlines()[0] if cause.message.strip() else ""
        for node in frame.graph.nodes:
            node.update_status()
            unfinished = [
                *node.queued.items(),
                *((i, job) for i, (job, _) in node.running.items()),
            ]
            for index, job in unfinished:
                if job not in self._handed and job not in self._child:
                    continue
                if job in self._not_run or job in self._waiting_twins:
                    continue  # it has its outcome, or was never dispatched
                frame.entries.append(
                    JobErrorEntry(
                        job_name=node.name,
                        state_index=None if node.is_partial else index,
                        state_key=index if node.is_partial else None,
                        exception_type=None,
                        message=f"stopped by {who}: {first}",
                        traceback="",
                        phase="stopped",
                    )
                )

    def close(self) -> None:
        """Release what the runner holds: a workflow job with errors of its own is
        committed as errored, any other that is not finished gives up its lease and
        stores nothing (its unfinished nodes did not run, so no result is built).
        Idempotent; a driver calls it however the run ends"""
        self._collect_resolution_errors()
        for frame in reversed(list(self._frames)):
            if frame.state == "finished" or not frame.owned:
                continue
            if frame.state == "expanding" and frame.errors:
                self._name_stopped(frame)
                self._finish(frame)
            else:
                frame.state = "finished"
                self._release_lease(frame)

    # Rounds

    def _collect_resolution_errors(self, frame: _Frame | None = None) -> None:
        for fr in [frame] if frame is not None else list(self._open):
            if fr.graph is None:
                continue
            for node in fr.graph.nodes:
                records: list[JobErrorEntry] = []
                fr.errors.extend(node.pop_resolution_errors(records))
                fr.entries.extend(records)
                if records:
                    self._failed = True

    def _runnable(self) -> list[Job]:
        leaves: list[Job] = []
        i = 0
        while i < len(self._open):  # frames made meanwhile are visited too
            frame = self._open[i]
            i += 1
            if frame.state == "leasing":
                self._try_lease(frame)
            if frame.state != "expanding":
                continue
            window = self._window(frame.graph, self._done_upto(frame.graph))
            jobs = [j for node in window for j in node.get_runnable_tasks(frame.graph)]
            for node in window:  # a node's absent elements are made as it releases
                for job in node.pop_absent():
                    self._handed.add(job)
                    self._frame_of[job] = frame
                    self._settled.append(Settled(job, "absent", job._held))
            for job in jobs:
                if is_workflow(job.task):
                    self._enter(job, frame)
                else:
                    self._frame_of[job] = frame
                    leaves.append(job)
        return leaves

    def _progress(self) -> int:
        """The count of changes to the state a pass of :meth:`ready` reads: the jobs
        handed out, the frames' states, the nodes' tables and modes, and a partial node's
        blocks, groups and seal (`~pydra.engine.progress.changed`). ``ready`` repeats its
        pass until a pass hands out a job or leaves the count unchanged."""
        return changes()

    def _done_upto(self, graph: "DiGraph[NodeExecution]") -> int:
        """The number of leading nodes of ``graph`` (in node order) that are done"""
        nodes = graph.sorted_nodes
        i = self._done_prefix.get(id(graph), 0)
        while i < len(nodes) and nodes[i].done:
            i += 1
        self._done_prefix[id(graph)] = i
        return i

    @staticmethod
    def _window(
        graph: "DiGraph[NodeExecution]", start: int = 0
    ) -> list["NodeExecution"]:
        """The nodes a pass of :meth:`ready` can change, in node order: from ``start``, the
        nodes not done up to the first whose predecessors include a node not started"""
        window: list[NodeExecution] = []
        not_started = set()
        for node in graph.sorted_nodes[start:]:
            if node.done:
                continue
            if not_started.intersection(graph.predecessors[node.name]):
                break
            if not node.started:
                not_started.add(node)
            window.append(node)
        return window

    @staticmethod
    def _runnable_jobs(graph: "DiGraph[NodeExecution]", start: int = 0) -> list[Job]:
        """The jobs of all the nodes that are released, in node order, from ``start``"""
        jobs: list[Job] = []
        for node in GraphRunner._window(graph, start):
            jobs.extend(node.get_runnable_tasks(graph))
        return jobs

    def _retry_after(self) -> float | None:
        now = time.monotonic()
        waits = [
            max(0.0, (f.retry_at if f.retry_at is not None else now) - now)
            for f in self._open
            if f.state == "leasing"
        ]
        return min(waits) if waits else None

    def ready(self, limit: int | None = None) -> Round:
        """Release the jobs whose upstream jobs have completed

        Each released element is taken through the store (a success record, then
        ``lookup`` when ``lookup_before_dispatch``) and the same-checksum table: a twin of
        a job whose outcome is known is settled at once, a twin of a job handed out and
        not yet reported is held and settled when that job's outcome is reported, and the
        rest are handed out, each once, with a fresh ``job.attempt``.

        Parameters
        ----------
        limit : int, optional
            The most jobs this call hands out, taken by whole node groups: groups are handed
            out in node order while they fit and only the group that crosses the limit is
            sliced. None for every ready job.

        Returns
        -------
        Round
            The jobs to dispatch, the jobs the runner completed itself, the elements newly
            marked unrunnable, and the seconds until a waiting workflow lease is tried again
        """
        while True:
            before = self._progress()
            self._settle_frames()
            candidates = [j for j in self._runnable() if j not in self._handed]
            candidates = self._serve(candidates)
            if candidates or self._progress() == before:
                break
        # grouped by node, in node order; the sort is stable, so elements keep their order
        candidates.sort(key=lambda j: (j.workflow_path, j.name))
        dispatch: list[Job] = []
        held_back = 0
        for job in candidates:
            first = self._first.get(job.checksum)
            if first is not None and first is not job:
                # a twin of a job handed out whose outcome is not reported: held
                self._handed.add(job)
                self._waiting_twins.add(job)
                self._twins[job.checksum].append(job)
                continue
            if limit is not None and len(dispatch) >= limit:
                held_back += 1  # released, handed out by a later call
                continue
            self._first[job.checksum] = job
            dispatch.append(job)
        if held_back and limit:
            logger.info(
                "Reducing number of tasks to run concurrently from %d to %d",
                len(dispatch) + held_back,
                limit,
            )
        for job in dispatch:
            self._handed.add(job)
            self._in_flight.add(job)
            job.rerun = self._inner_rerun
            job.attempt = uuid4().hex
        self._collect_resolution_errors()
        settled, self._settled = self._settled, []
        unrunnable = self._new_unrunnable()
        absent_nodes = self._new_absent_nodes()
        retry_after = self._retry_after()
        self._quiescent = (
            not (dispatch or settled or unrunnable or absent_nodes)
            and retry_after is None
            and not self._in_flight
        )
        return Round(
            dispatch=dispatch,
            settled=settled,
            unrunnable=unrunnable,
            absent_nodes=absent_nodes,
            retry_after=retry_after,
        )

    def _serve(self, jobs: list[Job]) -> list[Job]:
        """Settle the released jobs the store holds and the twins of jobs whose outcome is
        known; return the others"""
        if jobs and not self._inner_rerun:
            jobs = self._read_store(jobs)
        rest = []
        for job in jobs:
            outcome = self._finished.get(job.checksum)
            if outcome is None:
                rest.append(job)
            else:
                self._handed.add(job)
                self._share(job, outcome)
        return rest

    def _read_store(self, jobs: list[Job]) -> list[Job]:
        """Settle as ``stored`` the jobs whose slots carry a success record, and, under
        ``lookup_before_dispatch``, those the store's ``lookup`` returns an acceptable
        result for; return the others. A job read once without a result is not read
        again while the ``ready`` limit holds it back."""
        rest = []
        for job in jobs:
            if job in self._checked:
                rest.append(job)
                continue
            result = self.store.read_succeeded(job.checksum, job.context, job.task)
            if result is None and self.lookup_before_dispatch:
                result = self._lookup(job)
            if result is None:
                self._checked.add(job)
                rest.append(job)
                continue
            self._serve_stored(job, result)
            self._handed.add(job)
            self._settled.append(Settled(job, "stored", result))
        return rest

    @staticmethod
    def _serve_stored(job: Job, result: "Result") -> None:
        """Complete ``job`` with ``result``, read from the store: it takes no lease and is
        not dispatched"""
        job.hooks.pre_run(job)
        job._cached_result = result
        job._awaiting_run = False

    def _lookup(self, job: Job) -> "Result | None":
        try:
            result = self.store.lookup(job.checksum, job.context, task=job.task)
        except Exception as exc:  # unreadable: dispatched, as a job without a record
            logger.debug("Looking up %s before dispatch: %s", job.checksum, exc)
            return None
        if result is None or result.errored:
            return None
        return result

    def _share(self, job: Job, outcome: Outcome) -> None:
        """Complete a twin with the outcome of the job of its checksum that was handed
        out. A failure is recorded once, for that job; the twin is errored with it."""
        job._awaiting_run = False
        if isinstance(outcome, Failed):
            job._errored = True
            result = job.result()
        else:
            result = outcome
            job._held = outcome
            if outcome is not None and outcome.errored:
                job._errored = True
        self._settled.append(Settled(job, "shared", result))

    def _new_unrunnable(self) -> list[Unrunnable]:
        """The elements marked unrunnable that have not been reported yet"""
        found = []
        for frame in self._frames:
            if frame.graph is None:
                continue
            for node in frame.graph.nodes:
                for index, upstream in list(node.unrunnable.items()):
                    key = (id(node), index)
                    if key in self._reported_unrunnable:
                        continue
                    self._reported_unrunnable.add(key)
                    found.append(
                        Unrunnable(
                            node=node_label(node),
                            index=index,
                            upstream=[node_label(p) for p in upstream],
                        )
                    )
        return found

    def _new_absent_nodes(self) -> list[AbsentNode]:
        """The nodes found absent as a whole that have not been reported yet"""
        found = []
        for frame in self._frames:
            if frame.graph is None:
                continue
            for node in frame.graph.sorted_nodes:
                if node.absent is None or id(node) in self._reported_absent_nodes:
                    continue
                self._reported_absent_nodes.add(id(node))
                inherits = None
                if node.absent_inherits is not None:
                    inherits = node_label(frame.graph.node(node.absent_inherits))
                found.append(AbsentNode(node_label(node), node.absent, inherits))
        return found

    def _read_succeeded(self, jobs: list[Job]) -> list[Job]:
        dispatch = self._read_succeeded_jobs(self.store, jobs)
        self._handed.update(j for j in jobs if j not in dispatch)
        return dispatch

    @staticmethod
    def _read_succeeded_jobs(store: "ResultStore", jobs: list[Job]) -> list[Job]:
        """Complete, without their leases, the jobs whose slots carry a success record (one existence
        check each, `PickleStore.read_succeeded`), and return the others, to be dispatched under
        their leases: a job without a record, or whose result the store's ``lookup`` does not
        return, returns errored or returns from another context. A store without records
        dispatches every job.
        """
        dispatch = []
        for job in jobs:
            result = store.read_succeeded(job.checksum, job.context, job.task)
            if result is None:
                dispatch.append(job)
                continue
            GraphRunner._serve_stored(job, result)
        return dispatch

    def complete(self, job: Job, outcome: Outcome) -> None:
        """Record the outcome of a job that was handed out

        Parameters
        ----------
        job : Job
            A job from :meth:`ready`
        outcome : Result | Failed | NotRun
            What came of running it. A :class:`Failed` is recorded as the job's error and
            the elements that read the job become unrunnable. A ``Result`` is held on the
            job as given (whatever attributes it carries stay on it), so readers use it
            without a store read; an errored one is a failure of the run. A
            :class:`NotRun` leaves the job pending, its readers blocked and no error
            recorded, until :meth:`retry`. The twins held behind the job take a
            ``Result`` or ``Failed`` as theirs. None when the worker returned no result.
        """
        self._in_flight.discard(job)
        self._quiescent = False
        if isinstance(outcome, NotRun):
            self._not_run[job] = outcome
            return
        job._awaiting_run = False
        frame = self._frame_of.get(job, self._root)
        if isinstance(outcome, Failed):
            exc = outcome.error
            job._errored = True
            self._failed = True
            if self._first_error is None:
                self._first_error = exc
            frame.errors.append(job_error(job, exc))
            frame.entries.append(job_entry(job, exc))
        else:
            job._held = outcome
            if outcome is not None and outcome.errored:
                exc = result_error(job, outcome)
                job._errored = True
                self._failed = True
                frame.errors.append(job_error(job, exc))
                frame.entries.append(job_entry(job, exc))
            else:
                self._forget(job)
        if self._first.get(job.checksum) is job:
            del self._first[job.checksum]
            self._finished[job.checksum] = outcome
            for twin in self._twins.pop(job.checksum, ()):
                self._waiting_twins.discard(twin)
                self._share(twin, outcome)
        self._collect_resolution_errors()

    def _forget(self, job: Job) -> None:
        """Tell a store that keeps a job's outcome in its work directory until the runner
        holds it (``forget_committed``, RFC 0003 section 5.4) that it is held now"""
        try:
            self.store.forget_committed(job)
        except Exception as exc:  # left for the store's collection
            logger.debug("Forgetting the outcome file of %s: %s", job.checksum, exc)

    def retry(self, job: Job) -> None:
        """Hand a job reported :class:`NotRun` out again at the next :meth:`ready`, with
        a new attempt

        Raises
        ------
        ValueError
            The job's last outcome was not ``NotRun``
        """
        if job not in self._not_run:
            raise ValueError(
                f"Job {job.name!r} was not reported NotRun, so it is not handed out again"
            )
        del self._not_run[job]
        self._handed.discard(job)
        self._quiescent = False
        frame = self._frame_of.get(job)
        if (
            frame is not None and frame.graph is not None
        ):  # its node may be done no more
            self._done_prefix.pop(id(frame.graph), None)

    @property
    def not_run(self) -> list[tuple[Job, NotRun]]:
        """The jobs whose last outcome is :class:`NotRun`, with it"""
        return list(self._not_run.items())

    def upstream(self, job: Job) -> dict[str, list[tuple[Job, ty.Any]]]:
        """For each input of a job read from upstream nodes, the upstream jobs it was
        resolved from and their indices: the state index of an element of a static node,
        the ``state_key`` of an element of a partial node. The index is None for an
        upstream node without a state and for an input that reads the upstream whole
        (combined, or the sequence a splitter splits over).

        Parameters
        ----------
        job : Job
            A leaf job handed out by :meth:`ready`
        """
        frame = self._frame_of.get(job)
        if frame is None:
            frame = self._child[job].parent if job in self._child else self._root
        node: NodeExecution = frame.graph.node(job.name)
        if node.is_partial:
            index = job.state_key
            split, inputs_ind = (
                node._partial.vals[index],
                node._partial.inputs_ind[index],
            )
            inner = node._partial.inner_reads(index)
        else:
            index = job.state_index
            split = node.state.states_val[index] if index is not None else {}
            inputs_ind = node.state.inputs_ind[index] if index is not None else {}
            inner = {}
        found: dict[str, list[tuple[Job, ty.Any]]] = {}
        for name, value in attrs_values(node.node._task).items():
            if not isinstance(value, LazyOutField):
                continue
            up: NodeExecution = frame.graph.node(value._node.name)
            state_key = f"{node.name}.{name}"
            if state_key in inner:  # split per upstream element (a partial node)
                indices = up._read_indices(inner[state_key])
            elif state_key in split:  # the sequence the node's splitter splits over
                indices = up._read_indices(None)
            else:
                indices = up._read_indices(inputs_ind.get(state_key))
            whole = isinstance(indices, range) or indices is ALL
            found[name] = [
                (up.job(i), None if whole else i) for i in up._read_elements(indices)
            ]
        return found

    def plan(self, executors: "ty.Mapping[str, ty.Any] | None" = None) -> "Plan":
        """What the run would do now, without running, leasing or writing anything

        No job is released, handed out or settled, no lease is taken and no live node or
        its ``State`` is written, so a planned run ends as an unplanned one. A workflow job
        waiting for its lease is ``waiting``, one not yet tried is ``deferred``. Nodes that
        are done are not listed. A job is formed only from upstream elements the runner has
        accepted, never from one in flight. The store is asked once per node
        (``recorded`` or ``expected_hits``) and the router once per node, with a real job when
        one can be formed.

        Parameters
        ----------
        executors : Mapping[str, Executor], optional
            The executors by the names the router returns, by default the submitter's
            workers. Each is asked where the node's unfinished elements would run
            (``place(jobs)``, None for no answer), carried as :attr:`PlanNode.placement`

        Returns
        -------
        Plan
            One :class:`PlanNode` per node still to do, in release order, nested
            workflows that are expanded by their own nodes
        """
        from pydra.engine.plan import build_plan

        return build_plan(self, executors)

    def stuck(self) -> str | None:
        """A diagnosis of why nothing can progress, or None if the graph is done or a
        workflow job is waiting for its lease"""
        if self.done or any(f.state == "leasing" for f in self._frames):
            return None
        msg = ""
        for frame in self._frames:
            if frame.state == "expanding" and not frame.nodes_done:
                msg += self._diagnose(frame)
        return msg or "Nothing can progress, but the run is not done"

    @staticmethod
    def _diagnose(frame: _Frame) -> str:
        wf = frame.workflow
        exec_graph = frame.graph
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
        """Whether every node of the graph has completed (and, if the runner owns the
        workflow job, it is committed), or nothing else can progress and jobs remain
        :class:`NotRun` (an incomplete run: see :attr:`not_run`)"""
        root = self._root
        if root.owned:
            finished = root.state == "finished"
        else:
            finished = self._done_upto(root.graph) == len(root.graph.sorted_nodes)
        return finished or (bool(self._not_run) and self._quiescent)

    @property
    def entries(self) -> list[JobErrorEntry]:
        """The record of each errored job and of each element whose inputs could not be
        resolved, in the order of failure"""
        return self._root.entries

    @property
    def errors(self) -> list[str]:
        """A description of each entry"""
        return self._root.errors

    @property
    def failed(self) -> bool:
        """Whether any job has failed so far, at any level"""
        return self._failed

    @property
    def first_error(self) -> BaseException | None:
        """The first exception reported by a :class:`Failed`, as raised"""
        return self._first_error

    @property
    def root_error(self) -> BaseException | None:
        """The exception that stopped the runner's own workflow job (building the
        workflow), as raised; its nodes' errors are in :attr:`errors`"""
        return self._root_error

    def result(self) -> "Result":
        """The result of the workflow job, read from the store"""
        if self.job is None:
            raise RuntimeError("The runner was created without a workflow job")
        return self.job.result()
