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

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from pydra.compose.base import Task, Outputs
    from pydra.engine.node import Node
    from pydra.engine.result import Result
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
    exception_type : str
        The qualified name of the exception's class, ``module.Class``
    message : str
        ``str()`` of the exception
    traceback : str
        The formatted traceback
    phase : str
        ``"run"`` if the job ran and errored, ``"resolution"`` if its inputs could
        not be resolved from its upstream results, so it never ran
    """

    job_name: str
    state_index: int | None
    exception_type: str
    message: str
    traceback: str
    phase: str = "run"


class NodeExecution(ty.Generic[TaskType]):
    """A wrapper around a workflow node containing the execution state of the tasks that
    are generated from it"""

    name: str
    node: "Node"
    submitter: "Submitter"

    # List of tasks that were completed successfully
    successful: dict[int, list["Job[TaskType]"]]
    # List of tasks that failed
    # (an element whose inputs could not be resolved is mapped to the exception)
    errored: dict[int, "Job[TaskType] | Exception"]
    # Elements that couldn't be run due to upstream errors, mapped to the upstream
    # nodes whose errored or unrunnable elements they read (keyed by None if the whole
    # node couldn't be started)
    unrunnable: dict[int | None, list["NodeExecution"]]
    # List of tasks that are queued
    queued: dict[int, "Job[TaskType]"]
    # List of tasks that are queued
    running: dict[int, tuple["Job[TaskType]", datetime]]
    # Elements that are waiting on upstream elements, mapped to the upstream nodes and
    # the indices of their elements that they read
    # (a range of all of an upstream node's indices stands for reading all of its elements)
    blocked: (
        dict[int | None, list[tuple["NodeExecution", range | list[int | None]]]] | None
    )

    # Jobs of the elements, created when they are released or first read
    _tasks: dict[int | None, "Job[TaskType] | None"] | None

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

    @property
    def state(self):
        return self.node.state

    def __repr__(self):
        return (
            f"NodeExecution(name={self.name!r}, blocked={list(self.blocked)}, "
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
    def tasks(self) -> list["Job[TaskType]"]:
        """The jobs of all elements, in state order, creating any that do not exist
        yet (all upstream elements they read must have succeeded)"""
        if self._tasks is None:
            raise RuntimeError("Tasks have not been generated")
        return [self.job(index) for index in self._tasks]

    def job(self, index: int | None) -> "Job[TaskType]":
        """The job of one element of the node, created on first use. The upstream
        elements that its lazy inputs read must have completed by then.

        Parameters
        ----------
        index : int or None
            The state index of the element, None if the node has no state
        """
        if (job := self._tasks[index]) is None:
            job = self._tasks[index] = Job(
                task=self._split_task(index),
                submitter=self.submitter,
                environment=self.node._environment,
                name=self.node.name,
                hooks=self.node._hooks,
                resources=self.node._resources,
                state_index=index,
            )
        return job

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
                else:
                    missing = [i for i in inds if i not in node._tasks]
                if missing:
                    raise RuntimeError(
                        f"Element {index} of node {self.name!r} reads elements {missing} "
                        f"of node {node.name!r}, which has elements {list(node._tasks)}"
                    )
            self.blocked[index] = upstream

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
                self.successful[job.state_index] = self.queued.pop(index)
            elif job.errored or errored:
                self.errored[job.state_index] = self.queued.pop(index)
            elif not job._awaiting_run and job.run_start_time:
                self.running[job.state_index] = (
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
                self.successful[job.state_index] = self.running.pop(index)[0]
            elif job.errored or errored:
                self.errored[job.state_index] = self.running.pop(index)[0]

    @property
    def all_failed(self) -> bool:
        return (self.unrunnable or self.errored) and not (
            self.successful or self.blocked or self.queued
        )

    def _lazy_reads(
        self, index: int | None
    ) -> ty.Iterator[tuple[str, LazyField, int | None]]:
        """The lazy inputs of one element that are read from upstream (i.e. are not
        state values), with the state index each is read at

        Parameters
        ----------
        index : int or None
            The state index of the element, None if the node has no state
        """
        vals = self.state.states_val[index] if index is not None else {}
        inputs_ind = self.state.inputs_ind[index] if index is not None else {}
        for name, value in attrs_values(self.node._task).items():
            state_key = f"{self.name}.{name}"
            if isinstance(value, LazyField) and state_key not in vals:
                yield name, value, inputs_ind.get(state_key)

    def _split_task(self, index: int | None) -> "Task[OutputType]":
        """Resolve the task of one element of the node from its state values and the
        upstream values its lazy inputs read

        Parameters
        ----------
        index : int or None
            The state index of the element, None if the node has no state
        """
        resolved = {}
        if index is not None:
            vals = self.state.states_val[index]
            for name in self.node.input_names:
                if (state_key := f"{self.name}.{name}") in vals:
                    resolved[name] = vals[state_key]
        for name, value, state_index in self._lazy_reads(index):
            resolved[name] = value._get_value(
                workflow=self.workflow, graph=self.graph, state_index=state_index
            )
        return attrs.evolve(self.node._task, **resolved)

    def _upstream(
        self, index: int | None
    ) -> list[tuple["NodeExecution", list[int | None]]]:
        """The upstream nodes, and the indices of their elements, that ``_split_task``
        reads to resolve one element

        Parameters
        ----------
        index : int or None
            The state index of the element, None if the node has no state
        """
        upstream = []
        for _, value, state_index in self._lazy_reads(index):
            if isinstance(value, LazyOutField):
                node = self.graph.node(value._node.name)
                upstream.append((node, node._read_indices(state_index)))
        return upstream

    def _read_indices(self, state_index: int | None) -> range | list[int | None]:
        """The indices of the elements of this node that a downstream lazy field reads
        at the given state index (see ``LazyOutField._get_value``)

        Parameters
        ----------
        state_index : int or None
            The state index the lazy field is read at, None to read all elements, which
            are returned as a range so that readers need not store every index
        """
        if not self.state:
            return [None]
        if state_index is None or (self.state.combiner and not self.state.ind_l_final):
            return range(len(self.state.states_ind))
        return list(self.state.final_combined_ind_mapping[state_index])

    @property
    def _splits_upstream_output(self) -> bool:
        """Whether the node's own splitter splits over an upstream output, in which case
        its state can only be prepared once the upstream nodes are done"""
        return bool(self.state) and any(
            isinstance(value, LazyOutField)
            for name, value in self.node.state_values.items()
            if name in self.state.current_splitter_rpn
        )

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
            resplit = self._splits_upstream_output
            if unrunnable := [
                p
                for p in predecessors
                if (p._tasks is None and p.unrunnable)
                or (resplit and (p.errored or p.unrunnable))
            ]:
                self.unrunnable = {None: unrunnable}
                self.blocked = {}
            elif all(
                d if resplit else p.started for p, d in zip(predecessors, preds_done)
            ):
                self.start()
        progress = tuple(
            (len(p.successful), len(p.errored), len(p.unrunnable)) for p in predecessors
        )
        if self.blocked and progress != self._progress:
            self._progress = progress
            for index, upstream in list(self.blocked.items()):
                if unrunnable := list(
                    dict.fromkeys(p for p, inds in upstream if _has_failed(p, inds))
                ):
                    del self.blocked[index]
                    self.unrunnable[index] = unrunnable
                elif all(_has_succeeded(p, inds) for p, inds in upstream):
                    del self.blocked[index]
                    self._release(index)
        return list(self.queued.values())

    def _release(self, index: int | None) -> None:
        """Queue the job of an element whose upstream elements have all succeeded, or
        record the element as errored if its inputs cannot be resolved from them"""
        try:
            job = self.queued[index] = self.job(index)
            job._awaiting_run = True
        except Exception as exc:
            name = self.name if index is None else f"{self.name}({index})"
            self.errored[index] = exc
            self._resolution_errors.append(
                f"Job {name!r}, inputs could not be resolved, errored:\n{format_exc()}"
            )
            self._resolution_entries.append(
                error_entry(self.name, index, exc, phase="resolution")
            )

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


def _has_failed(node: NodeExecution, indices: range | list[int | None]) -> bool:
    """Whether any of the elements of a node at the given indices has errored or is
    unrunnable"""
    if len(indices) == len(node._tasks):
        return bool(node.errored or node.unrunnable)
    return any(i in node.errored or i in node.unrunnable for i in indices)


def _has_succeeded(node: NodeExecution, indices: range | list[int | None]) -> bool:
    """Whether the elements of a node at the given indices have all succeeded"""
    if len(indices) == len(node._tasks):
        return len(node.successful) == len(node._tasks)
    return all(i in node.successful for i in indices)


def job_error(job: "Job", exc: BaseException) -> str:
    """Describe the error of a job"""
    error_msg = "".join(format_exception(type(exc), exc, exc.__traceback__))
    if match := re.match(r'.*"""(.*)""".*', error_msg, flags=re.DOTALL | re.MULTILINE):
        error_msg = match.group(1)
    task_name = job.name
    if job.state_index is not None:
        task_name += f"({job.state_index})"
    return f"Job {task_name!r}, {job.task!r}, errored:{error_msg}"


def error_entry(
    job_name: str, state_index: int | None, exc: BaseException, phase: str = "run"
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
    )


@attrs.frozen
class Failed:
    """The outcome of a job that could not be run: the error that stopped it, as raised"""

    error: BaseException


# None: the worker returned no result, which the store holds
Outcome = ty.Union["Result", Failed, None]


@attrs.frozen
class Round:
    """What one call of :meth:`GraphRunner.ready` decided

    Attributes
    ----------
    dispatch : list[Job]
        The jobs to run, each handed out once; ``job.rerun`` is already set
    """

    dispatch: list[Job] = attrs.field(factory=list)


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
    submitter : Submitter, optional
        The submitter the jobs are created for (a job reads its audit,
        cache root and store from it). Without it a ``debug`` submitter over ``store``
        is created
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
        submitter: "Submitter | None" = None,
        job: "Job | None" = None,
    ):
        self.task = task
        self.store = store
        self.on_error = on_error
        self.rerun = rerun
        self.propagate_rerun = propagate_rerun
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
            The most jobs this call hands out; None for every ready job

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
        outcome : Result | Failed | None
            What came of running it. A :class:`Failed` is recorded as the job's error and
            the elements that read the job become unrunnable; None when the worker
            returned no result
        """
        job._awaiting_run = False
        if isinstance(outcome, Failed):
            exc = outcome.error
            job._errored = True
            self._failed = True
            if self._first_error is None:
                self._first_error = exc
            self._errors.append(job_error(job, exc))
            self._entries.append(error_entry(job.name, job.state_index, exc))
        elif outcome is not None and outcome.errored:
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
