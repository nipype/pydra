"""Handle execution backends."""

import asyncio
import typing as ty
import pickle
import os
from pathlib import Path
from tempfile import mkdtemp
from copy import copy
from datetime import datetime
from collections import defaultdict
from .workers import Worker, WORKERS
from .graph import DiGraph
from .helpers import (
    get_open_loop,
    list_fields,
)
from pydra.utils.hash import PersistentCache
from .state import StateIndex
from pydra.utils.typing import StateArray
from .audit import Audit
from .core import Task
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils import default_run_cache_dir
from pydra.design import workflow
import logging

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from .node import Node
    from .specs import TaskDef, TaskOutputs, WorkflowDef, TaskHooks, Result
    from .environments import Environment
    from .state import State

DefType = ty.TypeVar("DefType", bound="TaskDef")
OutputType = ty.TypeVar("OutputType", bound="TaskOutputs")

# Used to flag development mode of Audit
develop = False

WORKER_KWARG_FAIL_NOTE = "Attempting to instantiate worker submitter"


class Submitter:
    """Send a task to the execution backend.

    Parameters
    ----------
    cache_dir : os.PathLike, optional
        Cache directory where the working directory/results for the task will be
        stored, by default None
    worker : str or Worker, optional
        The worker to use, by default "cf"
    environment: Environment, optional
        The execution environment to use, by default None
    rerun : bool, optional
        Whether to force the re-computation of the task results even if existing
        results are found, by default False
    cache_locations : list[os.PathLike], optional
        Alternate cache locations to check for pre-computed results, by default None
    audit_flags : AuditFlag, optional
        Auditing configuration, by default AuditFlag.NONE
    messengers : list, optional
        Messengers, by default None
    messenger_args : dict, optional
        Messenger arguments, by default None
    clean_stale_locks : bool, optional
        Whether to clean stale lock files, i.e. lock files that were created before the
        start of the current run. Don't set if using a global cache where there are
        potentially multiple workflows that are running concurrently. By default (None),
        lock files will be cleaned if the *debug* worker is used
    **kwargs : dict
        Keyword arguments to pass on to the worker initialisation
    """

    cache_dir: os.PathLike
    worker: Worker
    environment: "Environment | None"
    rerun: bool
    cache_locations: list[os.PathLike]
    audit_flags: AuditFlag
    messengers: ty.Iterable[Messenger]
    messenger_args: dict[str, ty.Any]
    clean_stale_locks: bool
    run_start_time: datetime | None

    def __init__(
        self,
        cache_dir: os.PathLike | None = None,
        worker: str | ty.Type[Worker] | Worker | None = "debug",
        environment: "Environment | None" = None,
        rerun: bool = False,
        cache_locations: list[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        clean_stale_locks: bool | None = None,
        **kwargs,
    ):

        if worker is None:
            worker = "debug"

        from . import check_latest_version

        if Task._etelemetry_version_data is None:
            Task._etelemetry_version_data = check_latest_version()

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        if cache_dir is None:
            cache_dir = default_run_cache_dir
        cache_dir = Path(cache_dir).resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_dir = cache_dir
        self.cache_locations = cache_locations
        self.environment = environment
        self.rerun = rerun
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        if isinstance(worker, Worker):
            self._worker = worker
            self.worker_name = worker.plugin_name
        else:
            if isinstance(worker, str):
                self.worker_name = worker
                try:
                    worker_cls = WORKERS[self.worker_name]
                except KeyError:
                    raise NotImplementedError(
                        f"No worker for '{self.worker_name}' plugin"
                    )
            else:
                try:
                    self.worker_name = worker.plugin_name
                except AttributeError:
                    raise ValueError(
                        "Worker class must have a 'plugin_name' str attribute"
                    )
                worker_cls = worker
            try:
                self._worker = worker_cls(**kwargs)
            except TypeError as e:
                e.add_note(WORKER_KWARG_FAIL_NOTE)
                raise
        self.run_start_time = None
        self.clean_stale_locks = (
            clean_stale_locks
            if clean_stale_locks is not None
            else (self.worker_name == "debug")
        )
        self.worker_kwargs = kwargs
        self._worker.loop = self.loop

    @property
    def worker(self):
        if self._worker is None:
            raise RuntimeError(
                "Cannot access worker of unpickeld submitter (typically in subprocess)"
            )
        return self._worker

    def __call__(
        self,
        task_def: "TaskDef[OutputType]",
        hooks: "TaskHooks | None" = None,
        raise_errors: bool | None = None,
    ) -> "Result[OutputType]":
        """Submitter run function.

        Parameters
        ----------
        task_def : :obj:`~pydra.engine.specs.TaskDef`
            The task definition to run
        hooks : :obj:`~pydra.engine.specs.TaskHooks`, optional
            Task hooks, callable functions called as the task is setup and torn down,
            by default no functions are called at the hooks
        raise_errors : bool, optional
            Whether to raise errors, by default True if the 'debug' worker is used,
            otherwise False

        Returns
        -------
        result : Any
            The result of the task
        """
        if raise_errors is None:
            raise_errors = self.worker_name == "debug"
        if not isinstance(raise_errors, bool):
            raise TypeError(
                f"'raise_errors' must be a boolean or None, not {type(raise_errors)}"
            )

        task_def._check_rules()
        # If the outer task is split, create an implicit workflow to hold the split nodes
        if task_def._splitter:
            from pydra.engine.specs import TaskDef

            output_types = {o.name: list[o.type] for o in list_fields(task_def.Outputs)}

            @workflow.define(outputs=output_types)
            def Split(defn: TaskDef, output_types: dict):
                node = workflow.add(defn)
                return tuple(getattr(node, o) for o in output_types)

            task_def = Split(defn=task_def, output_types=output_types)

        elif task_def._combiner:
            raise ValueError(
                f"Task {self} is marked for combining, but not splitting. "
                "Use the `split` method to split the task before combining."
            )
        task = Task(
            task_def,
            submitter=self,
            name="main",
            environment=self.environment,
            hooks=hooks,
        )
        try:
            self.run_start_time = datetime.now()
            if self.worker.is_async:  # Only workflow tasks can be async
                self.loop.run_until_complete(
                    self.worker.run_async(task, rerun=self.rerun)
                )
            else:
                self.worker.run(task, rerun=self.rerun)
        except Exception as e:
            msg = (
                f"Full crash report for {type(task_def).__name__!r} task is here: "
                + str(task.output_dir / "_error.pklz")
            )
            if raise_errors:
                e.add_note(msg)
                raise e
            else:
                logger.error("\nTask execution failed\n" + msg)
        finally:
            self.run_start_time = None
        PersistentCache().clean_up()
        result = task.result()
        if result is None:
            if task.lockfile.exists():
                raise RuntimeError(
                    f"Task {task} has a lockfile, but no result was found. "
                    "This may be due to another submission that is currently running, or the hard "
                    "interrupt (e.g. a debugging abortion) interrupting a previous run. "
                    f"In the case of an interrupted run, please remove {str(task.lockfile)!r} "
                    "and resubmit."
                )
            raise RuntimeError(f"Task {task} has no result in {str(task.output_dir)!r}")
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries or those that should not be pickled
        # When unpickled (in another process) the submitter can't be called
        state["loop"] = None
        state["_worker"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the loop and worker
        self.loop = get_open_loop()
        self._worker = WORKERS[self.worker_name](**self.worker_kwargs)
        self.worker.loop = self.loop

    def expand_workflow(self, workflow_task: "Task[WorkflowDef]") -> None:
        """Expands and executes a workflow task synchronously. Typically only used during
        debugging and testing, as the asynchronous version is more efficient.

        Parameters
        ----------
        task : :obj:`~pydra.engine.core.Task[WorkflowDef]`
            Workflow Task object

        """
        # Construct the workflow
        wf = workflow_task.definition.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        tasks = self.get_runnable_tasks(exec_graph)
        while tasks or any(not n.done for n in exec_graph.nodes):
            for task in tasks:
                self.worker.run(task, rerun=self.rerun)
            tasks = self.get_runnable_tasks(exec_graph)
        workflow_task.return_values = {"workflow": wf, "exec_graph": exec_graph}

    async def expand_workflow_async(self, workflow_task: "Task[WorkflowDef]") -> None:
        """
        Expand and execute a workflow task asynchronously.

        Parameters
        ----------
        task : :obj:`~pydra.engine.core.Task[WorkflowDef]`
            Workflow Task object
        """
        wf = workflow_task.definition.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        # keep track of pending futures
        task_futures = set()
        tasks = self.get_runnable_tasks(exec_graph)
        while tasks or task_futures or any(not n.done for n in exec_graph.nodes):
            if not tasks and not task_futures:
                # it's possible that task_futures is empty, but not able to get any
                # tasks from graph_copy (using get_runnable_tasks)
                # this might be related to some delays saving the files
                # so try to get_runnable_tasks for another minute
                ii = 0
                while not tasks and any(not n.done for n in exec_graph.nodes):
                    tasks = self.get_runnable_tasks(exec_graph)
                    ii += 1
                    # don't block the event loop!
                    await asyncio.sleep(1)
                    if ii > 10:
                        not_done = "\n".join(
                            (
                                f"{n.name}: started={bool(n.started)}, "
                                f"blocked={list(n.blocked)}, queued={list(n.queued)}"
                            )
                            for n in exec_graph.nodes
                            if not n.done
                        )
                        msg = (
                            "Something has gone wrong when retrieving the predecessor "
                            f"results. Not able to get any more tasks but he following "
                            f"nodes of the {wf.name!r} workflow are not done:\n{not_done}\n\n"
                        )
                        not_done = [n for n in exec_graph.nodes if not n.done]
                        msg += "\n" + ", ".join(
                            f"{t.name}: {t.done}" for t in not_done[0].queued.values()
                        )
                        # Get blocked tasks and the predecessors they are blocked on
                        outstanding: dict[Task[DefType], list[Task[DefType]]] = {
                            t: [
                                p for p in exec_graph.predecessors[t.name] if not p.done
                            ]
                            for t in exec_graph.sorted_nodes
                        }

                        hashes_have_changed = False
                        for task, blocked_on in outstanding.items():
                            if not blocked_on:
                                continue
                            msg += f"- '{task.name}' node blocked due to\n"
                            for pred in blocked_on:
                                if (
                                    pred.checksum
                                    != wf.inputs._graph_checksums[pred.name]
                                ):
                                    msg += (
                                        f"    - hash changes in '{pred.name}' node inputs. "
                                        f"Current values and hashes: {pred.inputs}, "
                                        f"{pred.inputs._hashes}\n"
                                    )
                                    hashes_have_changed = True
                                elif pred not in outstanding:
                                    msg += (
                                        f"    - undiagnosed issues in '{pred.name}' node, "
                                        "potentially related to file-system access issues "
                                    )
                            msg += "\n"
                        if hashes_have_changed:
                            msg += (
                                "Set loglevel to 'debug' in order to track hash changes "
                                "throughout the execution of the workflow.\n\n "
                                "These issues may have been caused by `bytes_repr()` methods "
                                "that don't return stable hash values for specific object "
                                "types across multiple processes (see bytes_repr() "
                                '"singledispatch "function in pydra/utils/hash.py).'
                                "You may need to write specific `bytes_repr()` "
                                "implementations (see `pydra.utils.hash.register_serializer`) "
                                "or `__bytes_repr__()` dunder methods to handle one "
                                "or more types in your interface inputs."
                            )
                        raise RuntimeError(msg)
            for task in tasks:
                if task.is_async:
                    await self.worker.run_async(task, rerun=self.rerun)
                else:
                    task_futures.add(self.worker.run(task, rerun=self.rerun))
            task_futures = await self.worker.fetch_finished(task_futures)
            tasks = self.get_runnable_tasks(exec_graph)
        workflow_task.return_values = {"workflow": wf, "exec_graph": exec_graph}

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """
        Close submitter.

        Do not close previously queued loop.

        """
        self.worker.close()
        if self._own_loop:
            self.loop.close()

    def _check_locks(self, tasks: list[Task]) -> None:
        """Check for stale lock files and remove them."""
        if self.clean_stale_locks:
            for task in tasks:
                start_time = task.run_start_time
                if start_time and start_time < self.run_start_time:
                    task.lockfile.unlink()

    def get_runnable_tasks(self, graph: DiGraph) -> list["Task[DefType]"]:
        """Parse a graph and return all runnable tasks.

        Parameters
        ----------
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object

        Returns
        -------
        tasks : list of :obj:`~pydra.engine.core.Task`
            List of runnable tasks
        following_err : dict[NodeToExecute, list[str]]
            Dictionary of tasks that are blocked by errored tasks
        """
        tasks = []
        not_started = set()
        node: NodeExecution
        for node in graph.sorted_nodes:
            if node.done:
                continue
            # since the list is sorted (breadth-first) we can stop
            # when we find a task that depends on any task that is already in tasks
            if set(graph.predecessors[node.name]).intersection(not_started):
                break
            # Record if the node has not been started
            if not node.started:
                not_started.add(node)
            tasks.extend(node.get_runnable_tasks(graph))
        self._check_locks(tasks)
        return tasks

    @property
    def cache_dir(self):
        """Get the location of the cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, location):
        if location is not None:
            self._cache_dir = Path(location).resolve()
            self._cache_dir.mkdir(parents=False, exist_ok=True)
        else:
            self._cache_dir = mkdtemp()
            self._cache_dir = Path(self._cache_dir).resolve()


class NodeExecution(ty.Generic[DefType]):
    """A wrapper around a workflow node containing the execution state of the tasks that
    are generated from it"""

    name: str
    node: "Node"
    submitter: Submitter

    # List of tasks that were completed successfully
    successful: dict[StateIndex | None, list["Task[DefType]"]]
    # List of tasks that failed
    errored: dict[StateIndex | None, "Task[DefType]"]
    # List of tasks that couldn't be run due to upstream errors
    unrunnable: dict[StateIndex | None, list["Task[DefType]"]]
    # List of tasks that are queued
    queued: dict[StateIndex | None, "Task[DefType]"]
    # List of tasks that are queued
    running: dict[StateIndex | None, tuple["Task[DefType]", datetime]]
    # List of tasks that are blocked on other tasks to complete before they can be run
    blocked: dict[StateIndex | None, "Task[DefType]"]

    _tasks: dict[StateIndex | None, "Task[DefType]"] | None

    workflow_inputs: "WorkflowDef"

    graph: DiGraph["NodeExecution"] | None

    def __init__(
        self,
        node: "Node",
        submitter: Submitter,
        workflow_inputs: "WorkflowDef",
    ):
        self.name = node.name
        self.node = node
        self.submitter = submitter
        # Initialize the state dictionaries
        self._tasks = None
        self.blocked = {}
        self.successful = {}
        self.errored = {}
        self.queued = {}
        self.running = {}  # Not used in logic, but may be useful for progress tracking
        self.unrunnable = defaultdict(list)
        self.state_names = self.node.state.names if self.node.state else []
        self.workflow_inputs = workflow_inputs
        self.graph = None

    @property
    def inputs(self) -> "Node.Inputs":
        return self.node.inputs

    @property
    def _definition(self) -> "Node":
        return self.node._definition

    @property
    def state(self) -> "State":
        return self.node.state

    @property
    def tasks(self) -> ty.Iterable["Task[DefType]"]:
        if self._tasks is None:
            self._tasks = {t.state_index: t for t in self._generate_tasks()}
        return self._tasks.values()

    def task(self, index: StateIndex = StateIndex()) -> "Task | list[Task[DefType]]":
        """Get a task object for a given state index."""
        self.tasks  # Ensure tasks are loaded
        try:
            return self._tasks[index]
        except KeyError:
            if not index:
                return StateArray(self._tasks.values())
            raise

    @property
    def started(self) -> bool:
        return (
            self.successful
            or self.errored
            or self.unrunnable
            or self.queued
            or self.blocked
        )

    @property
    def done(self) -> bool:
        self.update_status()
        if not self.started:
            return False
        # Check to see if any previously queued tasks have completed
        return not (self.queued or self.blocked or self.running)

    def update_status(self) -> None:
        """Updates the status of the tasks in the node."""
        if not self.started:
            return
        # Check to see if any previously queued tasks have completed
        for index, task in list(self.queued.items()):
            try:
                is_done = task.done
            except ValueError:
                errored = True
                is_done = False
            else:
                errored = False
            if is_done:
                self.successful[task.state_index] = self.queued.pop(index)
            elif task.errored or errored:
                self.errored[task.state_index] = self.queued.pop(index)
            elif task.run_start_time:
                self.running[task.state_index] = (
                    self.queued.pop(index),
                    task.run_start_time,
                )
        # Check to see if any previously running tasks have completed
        for index, (task, start_time) in list(self.running.items()):
            if task.done:
                self.successful[task.state_index] = self.running.pop(index)[0]
            elif task.errored:
                self.errored[task.state_index] = self.running.pop(index)[0]

    @property
    def all_failed(self) -> bool:
        return (self.unrunnable or self.errored) and not (
            self.successful or self.blocked or self.queued
        )

    def _generate_tasks(self) -> ty.Iterable["Task[DefType]"]:
        if not self.node.state:
            yield Task(
                definition=self.node._definition._resolve_lazy_inputs(
                    workflow_inputs=self.workflow_inputs,
                    graph=self.graph,
                ),
                submitter=self.submitter,
                environment=self.node._environment,
                hooks=self.node._hooks,
                name=self.node.name,
            )
        else:
            for index, split_defn in self.node._split_definition().items():
                yield Task(
                    definition=split_defn._resolve_lazy_inputs(
                        workflow_inputs=self.workflow_inputs,
                        graph=self.graph,
                        state_index=index,
                    ),
                    submitter=self.submitter,
                    environment=self.node._environment,
                    name=self.node.name,
                    hooks=self.node._hooks,
                    state_index=index,
                )

    def get_runnable_tasks(self, graph: DiGraph) -> list["Task[DefType]"]:
        """For a given node, check to see which tasks have been successfully run, are ready
        to run, can't be run due to upstream errors, or are blocked on other tasks to complete.

        Parameters
        ----------
        node : :obj:`~pydra.engine.node.Node`
            The node object to get the tasks for
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object


        Returns
        -------
        runnable : list[NodeExecution]
            List of tasks that are ready to run
        """
        runnable: list["Task[DefType]"] = []
        self.tasks  # Ensure tasks are loaded
        if not self.started:
            self.blocked = copy(self._tasks)
        # Check to see if any blocked tasks are now runnable/unrunnable
        for index, task in list(self.blocked.items()):
            pred: NodeExecution
            is_runnable = True
            for pred in graph.predecessors[self.node.name]:
                if index not in pred.successful:
                    is_runnable = False
                    if index in pred.errored:
                        self.unrunnable[index].append(self.blocked.pop(index))
                    if index in pred.unrunnable:
                        self.unrunnable[index].extend(pred.unrunnable[index])
                        self.blocked.pop(index)
                    break
            if is_runnable:
                runnable.append(self.blocked.pop(index))
        self.queued.update({t.state_index: t for t in runnable})
        return runnable


async def prepare_runnable(runnable):
    return runnable.pickle_task()


def _list_blocked_tasks(graph):
    """Generates a list of tasks that can't be run and predecessors that are blocking
    them to help debugging of broken workflows"""
    blocked = []
    for tsk in graph.sorted_nodes:
        blocking = []
        for pred in graph.predecessors[tsk.name]:
            if not pred.done:
                matching_name = []
                for cache_loc in tsk.cache_locations:
                    for tsk_work_dir in cache_loc.iterdir():
                        if (tsk_work_dir / "_task.pklz").exists():
                            with open(tsk_work_dir / "_task.pklz", "rb") as f:
                                saved_tsk = pickle.load(f)
                            if saved_tsk.name == pred.name:
                                matching_name.append(
                                    f"{saved_tsk.name} ({tsk_work_dir.name})"
                                )
                blocking.append((pred, ", ".join(matching_name)))
        if blocking:
            blocked.append(
                f"\n{tsk.name} ({tsk.checksum}) is blocked by "
                + "; ".join(
                    f"{pred.name} ({pred.checksum}), which matches names of [{matching}]"
                    for pred, matching in blocking
                )
            )
    return blocked
