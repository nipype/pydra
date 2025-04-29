"""Handle execution backends."""

import asyncio
import typing as ty
import re
import os
from pathlib import Path
from traceback import format_exc
from tempfile import mkdtemp
from copy import copy, deepcopy
from datetime import datetime
from collections import defaultdict
import attrs
import logging
from pydra.engine.graph import DiGraph
from pydra.utils.general import (
    get_fields,
    attrs_values,
)
from pydra.utils.hash import PersistentCache
from pydra.engine.lazy import LazyField
from pydra.engine.audit import Audit
from pydra.engine.job import Job
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils.general import default_run_cache_root
from pydra.compose import workflow
from pydra.engine.state import State
from pydra.workers.base import Worker
from pydra.compose.base import Task, Outputs

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from pydra.engine.node import Node
    from pydra.engine.result import Result
    from pydra.engine.hooks import TaskHooks
    from pydra.engine.workflow import Workflow
    from pydra.environments.base import Environment


TaskType = ty.TypeVar("TaskType", bound="Task")
OutputType = ty.TypeVar("OutputType", bound="Outputs")

# Used to flag development mode of Audit
develop = False

WORKER_KWARG_FAIL_NOTE = "Attempting to instantiate worker submitter"


class Submitter:
    """Send a job to the execution backend.

    Parameters
    ----------
    cache_root : os.PathLike, optional
        Cache directory where the working directory/results for the job will be
        stored, by default None
    worker : str or Worker, optional
        The worker to use, by default "cf"
    environment: Environment, optional
        The execution environment to use, by default None
    readonly_caches : list[os.PathLike], optional
        Alternate cache locations to check for pre-computed results, by default None
    max_concurrent: int | float, optional
        Maximum number of concurrent tasks to run, by default float("inf") (unlimited)
    audit_flags : AuditFlag, optional
        Configure provenance tracking. available flags: :class:`~pydra.utils.messenger.AuditFlag`
        Default is no provenance tracking.
    messenger : :class:`Messenger` or :obj:`list` of :class:`Messenger` or None
        Messenger(s) used by Audit. Saved in the `audit` attribute.
        See available flags at :class:`~pydra.utils.messenger.Messenger`.
    messengers_args : dict[str, Any], optional
        Argument(s) used by `messegner`. Saved in the `audit` attribu
    clean_stale_locks : bool, optional
        Whether to clean stale lock files, i.e. lock files that were created before the
        start of the current run. Don't set if using a global cache where there are
        potentially multiple workflows that are running concurrently. By default (None),
        lock files will be cleaned if the *debug* worker is used
    **kwargs : dict
        Keyword arguments to pass on to the worker initialisation
    """

    cache_root: os.PathLike
    worker: Worker
    environment: "Environment | None"
    readonly_caches: list[os.PathLike]
    audit_flags: AuditFlag
    messengers: ty.Iterable[Messenger]
    messenger_args: dict[str, ty.Any]
    max_concurrent: int | float
    clean_stale_locks: bool
    run_start_time: datetime | None
    propagate_rerun: bool

    def __init__(
        self,
        /,
        cache_root: os.PathLike | None = None,
        worker: str | ty.Type[Worker] | Worker | None = "debug",
        environment: "Environment | None" = None,
        readonly_caches: list[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        max_concurrent: int | float = float("inf"),
        propagate_rerun: bool = True,
        clean_stale_locks: bool | None = None,
        **kwargs,
    ):

        from pydra.environments import native

        if worker is None:
            worker = "debug"

        from pydra.utils.etelemetry import check_latest_version

        if Job._etelemetry_version_data is None:
            Job._etelemetry_version_data = check_latest_version()

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
            develop=develop,
        )
        if cache_root is None:
            cache_root = default_run_cache_root
        cache_root = Path(cache_root).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        self.cache_root = cache_root
        self.readonly_caches = readonly_caches
        self.propagate_rerun = propagate_rerun
        if max_concurrent < 1 or (
            isinstance(max_concurrent, float) and max_concurrent != float("inf")
        ):
            raise ValueError(
                "'max_concurrent' arg must be a positive integer or float('inf'), "
                f"not {max_concurrent}"
            )
        self.max_concurrent = max_concurrent
        self.environment = (
            environment if environment is not None else native.Environment()
        )
        self.loop = get_open_loop()
        self._own_loop = not self.loop.is_running()
        if not isinstance(worker, Worker):
            if isinstance(worker, str):
                worker_cls = Worker.plugin(worker)
            elif issubclass(worker, Worker):
                worker_cls = worker
            else:
                raise TypeError(
                    "Worker must be a Worker object, name of a worker or a Worker "
                    f"class, not {worker}"
                )
            try:
                worker = worker_cls(**kwargs)
            except TypeError as e:
                e.add_note(WORKER_KWARG_FAIL_NOTE)
                raise
        self.worker = worker
        self.run_start_time = None
        self.clean_stale_locks = (
            clean_stale_locks
            if clean_stale_locks is not None
            else (self.worker.plugin_name() == "debug")
        )
        self.worker_kwargs = kwargs
        self.worker.loop = self.loop

    def __call__(
        self,
        task: "Task[OutputType]",
        hooks: "TaskHooks | None" = None,
        raise_errors: bool | None = None,
        rerun: bool = False,
    ) -> "Result[OutputType]":
        """Submitter run function.

        Parameters
        ----------
        task : :obj:`~pydra.compose.base.Task`
            The task to run
        hooks : :obj:`~pydra.engine.hooks.TaskHooks`, optional
            Job hooks, callable functions called as the job is setup and torn down,
            by default no functions are called at the hooks
        raise_errors : bool, optional
            Whether to raise errors, by default True if the 'debug' worker is used,
            otherwise False
        rerun : bool, optional
            Whether to force the re-computation of the job results even if existing
            results are found, by default False
        propagate_rerun : bool, optional
            Whether to propagate the rerun flag to all tasks in the workflow, by default True

        Returns
        -------
        result : Any
            The result of the job
        """
        from pydra.environments.base import Environment

        if raise_errors is None:
            raise_errors = self.worker.plugin_name() == "debug"
        if not isinstance(raise_errors, bool):
            raise TypeError(
                f"'raise_errors' must be a boolean or None, not {type(raise_errors)}"
            )

        task._check_rules()
        # If the outer job is split, create an implicit workflow to hold the split nodes
        if task._splitter:

            state = State(
                name="outer_split",
                splitter=deepcopy(task._splitter),
                combiner=deepcopy(task._combiner),
                container_ndim=deepcopy(task._container_ndim),
            )

            def wrap_type(tp):
                tp = state.nest_output_type(tp)
                tp = state.combine_state_arrays(tp)
                return tp

            output_types = {o.name: wrap_type(o.type) for o in get_fields(task.Outputs)}

            @workflow.define(outputs=output_types)
            def Split(defn: Task, output_types: dict, environment: Environment | None):
                node = workflow.add(defn, environment=environment, hooks=hooks)
                return tuple(getattr(node, o) for o in output_types)

            task = Split(
                defn=task, output_types=output_types, environment=self.environment
            )

            environment = None
        elif task._combiner:
            raise ValueError(
                f"Job {self} is marked for combining, but not splitting. "
                "Use the `split` method to split the job before combining."
            )
        else:
            environment = self.environment

        job = Job(
            task,
            submitter=self,
            name="main",
            environment=environment,
            hooks=hooks,
        )
        try:
            self.run_start_time = datetime.now()
            self.submit(job, rerun=rerun)
        except Exception as exc:
            error_msg = (
                f"Full crash report for {type(task).__name__!r} job is here: "
                + str(job.cache_dir / "_error.pklz")
            )
            exc.add_note(error_msg)
            if raise_errors or not job.result():
                raise exc
            else:
                logger.error("\nTask execution failed\n%s", error_msg)
        finally:
            self.run_start_time = None
        PersistentCache().clean_up()
        result = job.result()
        if result is None:
            if job.lockfile.exists():
                raise RuntimeError(
                    f"Job {job} has a lockfile, but no result was found. "
                    "This may be due to another submission that is currently running, or the hard "
                    "interrupt (e.g. a debugging abortion) interrupting a previous run. "
                    f"In the case of an interrupted run, please remove {str(job.lockfile)!r} "
                    "and resubmit."
                )
            raise RuntimeError(f"Job {job} has no result in {str(job.cache_dir)!r}")
        return result

    def submit(self, job: "Job[TaskType]", rerun: bool = False) -> None:
        """Submit a job to the worker.

        Parameters
        ----------
        job : :obj:`~pydra.engine.job.Job`
            The job to submit
        rerun : bool, optional
            Whether to force the re-computation of the job results even if existing
            results are found, by default False
        """
        if self.worker.is_async:  # Only workflow tasks can be async
            self.loop.run_until_complete(self.worker.submit(job, rerun=rerun))
        else:
            self.worker.run(job, rerun=rerun)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries or those that should not be pickled
        # When unpickled (in another process) the submitter can't be called
        state["loop"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the loop and worker
        self.loop = get_open_loop()
        self.worker.loop = self.loop

    def expand_workflow(self, workflow_task: "Job[workflow.Task]", rerun: bool) -> None:
        """Expands and executes a workflow job synchronously. Typically only used during
        debugging and testing, as the asynchronous version is more efficient.

        Parameters
        ----------
        job : :obj:`~pydra.engine.job.Job[workflow.Task]`
            Workflow Job object

        """
        # Construct the workflow
        wf = workflow_task.task.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        workflow_task.return_values = {"workflow": wf, "exec_graph": exec_graph}
        tasks = self.get_runnable_tasks(exec_graph)
        while tasks or any(not n.done for n in exec_graph.nodes):
            for job in tasks:
                self.worker.run(job, rerun=rerun and self.propagate_rerun)
            tasks = self.get_runnable_tasks(exec_graph)

    async def expand_workflow_async(
        self, workflow_task: "Job[workflow.Task]", rerun: bool
    ) -> None:
        """
        Expand and execute a workflow job asynchronously.

        Parameters
        ----------
        job : :obj:`~pydra.engine.job.Job[workflow.Task]`
            Workflow Job object
        """
        wf = workflow_task.task.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        workflow_task.return_values = {"workflow": wf, "exec_graph": exec_graph}
        # keep track of pending futures
        task_futures = set()
        futured: dict[str, Job[TaskType]] = {}
        tasks = self.get_runnable_tasks(exec_graph)
        errors = []
        try:
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
                                f"nodes of the {wf.name!r} workflow are not done:"
                                f"\n{not_done}\n\n"
                            )
                            not_done = [n for n in exec_graph.nodes if not n.done]
                            msg += "\n" + ", ".join(
                                f"{t.name}: {t.done}"
                                for t in not_done[0].queued.values()
                            )
                            # Get blocked tasks and the predecessors they are blocked on
                            outstanding: dict[Job[TaskType], list[Job[TaskType]]] = {
                                t: [
                                    p
                                    for p in exec_graph.predecessors[t.name]
                                    if not p.done
                                ]
                                for t in exec_graph.sorted_nodes
                            }

                            hashes_have_changed = False
                            for job, blocked_on in outstanding.items():
                                if not blocked_on:
                                    continue
                                msg += f"- '{job.name}' node blocked due to\n"
                                for pred in blocked_on:
                                    if (
                                        pred.checksum
                                        != wf.inputs._graph_checksums[pred.name]
                                    ):
                                        msg += (
                                            f"    - hash changes in '{pred.name}' node "
                                            f"inputs. Current values and hashes: "
                                            f"{pred.inputs}, {pred.inputs._hashes}\n"
                                        )
                                        hashes_have_changed = True
                                    elif pred not in outstanding:
                                        msg += (
                                            f"    - undiagnosed issues in '{pred.name}' "
                                            "node, potentially related to file-system "
                                            "access issues "
                                        )
                                msg += "\n"
                            if hashes_have_changed:
                                msg += (
                                    "Set loglevel to 'debug' in order to track hash "
                                    "changes throughout the execution of the workflow."
                                    "\n\n These issues may have been caused by "
                                    "`bytes_repr()` methods that don't return stable "
                                    "hash values for specific object types across "
                                    "multiple processes (see bytes_repr() "
                                    '"singledispatch "function in pydra/utils/hash.py).'
                                    "You may need to write specific `bytes_repr()` "
                                    "implementations (see `pydra.utils.hash.register_serializer`) "
                                    "or `__bytes_repr__()` dunder methods to handle one "
                                    "or more types in your interface inputs."
                                )
                            raise RuntimeError(msg)
                for job in tasks:
                    if job.is_async:  # Only workflows at this stage
                        await self.worker.submit(
                            job, rerun=rerun and self.propagate_rerun
                        )
                    elif job.checksum not in futured:
                        asyncio_task = asyncio.Task(
                            self.worker.run(job, rerun=rerun and self.propagate_rerun),
                            name=job.checksum,
                        )
                        task_futures.add(asyncio_task)
                        futured[job.checksum] = job
                task_futures, completed = await self.fetch_finished(task_futures)
                for task_future in completed:
                    try:
                        task_future.result()
                    except Exception:
                        error_msg = format_exc()
                        if match := re.match(
                            r'.*"""(.*)""".*',
                            error_msg,
                            flags=re.DOTALL | re.MULTILINE,
                        ):
                            error_msg = match.group(1)
                        job = futured[task_future.get_name()]
                        task_name = job.name
                        if job.state_index is not None:
                            task_name += f"({job.state_index})"
                        errors.append(
                            f"Job {task_name!r}, {job.task!r}, errored:{error_msg}"
                        )
                tasks = self.get_runnable_tasks(exec_graph)
        finally:
            if errors:
                all_errors = "\n\n".join(errors)
                raise RuntimeError(
                    f"Workflow job {workflow_task} failed with errors"
                    f":\n\n{all_errors}\n\nSee output directory for details: {workflow_task.cache_dir}"
                )

    async def fetch_finished(
        self, futures
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        """
        Awaits asyncio's :class:`asyncio.Task` until one is finished.

        Parameters
        ----------
        futures : set of asyncio awaitables
            Job execution coroutines or asyncio :class:`asyncio.Task`

        Returns
        -------
        pending : set
            Pending asyncio :class:`asyncio.Task`.
        done: set
            Completed asyncio :class:`asyncio.Task`

        """
        done = set()
        try:
            done, pending = await asyncio.wait(
                futures, return_when=asyncio.FIRST_COMPLETED
            )
        except ValueError:
            # nothing pending!
            pending = set()
        logger.debug(f"Tasks finished: {len(done)}")
        return pending, done

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

    def _check_locks(self, tasks: list[Job]) -> None:
        """Check for stale lock files and remove them."""
        if self.clean_stale_locks:
            for job in tasks:
                start_time = job.run_start_time
                if start_time and start_time < self.run_start_time:
                    job.lockfile.unlink()

    def get_runnable_tasks(self, graph: DiGraph) -> list["Job[TaskType]"]:
        """Parse a graph and return all runnable tasks.

        Parameters
        ----------
        graph : :obj:`~pydra.engine.graph.DiGraph`
            Graph object

        Returns
        -------
        tasks : list of :obj:`~pydra.engine.job.Job`
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
            # when we find a job that depends on any job that is already in tasks
            preds = set(graph.predecessors[node.name])
            if preds.intersection(not_started):
                break
            # Record if the node has not been started
            if not node.started:
                not_started.add(node)
            tasks.extend(node.get_runnable_tasks(graph))
        self._check_locks(tasks)
        if len(tasks) > self.max_concurrent:
            logger.info(
                "Reducing number of tasks to run concurrently from %d to %d",
                len(tasks),
                self.max_concurrent,
            )
            tasks = tasks[: self.max_concurrent]
        return tasks

    @property
    def cache_root(self):
        """Get the location of the cache directory."""
        return self._cache_root

    @cache_root.setter
    def cache_root(self, location):
        if location is not None:
            self._cache_root = Path(location).resolve()
            self._cache_root.mkdir(parents=False, exist_ok=True)
        else:
            self._cache_root = mkdtemp()
            self._cache_root = Path(self._cache_root).resolve()


class NodeExecution(ty.Generic[TaskType]):
    """A wrapper around a workflow node containing the execution state of the tasks that
    are generated from it"""

    name: str
    node: "Node"
    submitter: Submitter

    # List of tasks that were completed successfully
    successful: dict[int, list["Job[TaskType]"]]
    # List of tasks that failed
    errored: dict[int, "Job[TaskType]"]
    # List of tasks that couldn't be run due to upstream errors
    unrunnable: dict[int, list["Job[TaskType]"]]
    # List of tasks that are queued
    queued: dict[int, "Job[TaskType]"]
    # List of tasks that are queued
    running: dict[int, tuple["Job[TaskType]", datetime]]
    # List of tasks that are blocked on other tasks to complete before they can be run
    blocked: dict[int, "Job[TaskType]"] | None

    _tasks: dict[int | None, "Job[TaskType]"] | None

    workflow: "Workflow"

    graph: DiGraph["NodeExecution"] | None

    def __init__(
        self,
        node: "Node",
        submitter: Submitter,
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
    def tasks(self) -> ty.Generator["Job[TaskType]", None, None]:
        if self._tasks is None:
            raise RuntimeError("Tasks have not been generated")
        return self._tasks.values()

    def start(self) -> None:
        """Prepare the execution node so that it can be processed"""
        self._tasks = {}
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
            # Generate the tasks
            for index, split_defn in enumerate(self._split_task()):
                self._tasks[index] = Job(
                    task=split_defn,
                    submitter=self.submitter,
                    environment=self.node._environment,
                    name=self.node.name,
                    hooks=self.node._hooks,
                    state_index=index,
                )
        else:
            self._tasks[None] = Job(
                task=self._resolve_lazy_inputs(task=self.node._task),
                submitter=self.submitter,
                environment=self.node._environment,
                hooks=self.node._hooks,
                name=self.node.name,
            )
        self.blocked = copy(self._tasks)

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
            elif job.run_start_time:
                self.running[job.state_index] = (
                    self.queued.pop(index),
                    job.run_start_time,
                )
        # Check to see if any previously running tasks have completed
        for index, (job, _) in list(self.running.items()):
            if job.done:
                self.successful[job.state_index] = self.running.pop(index)[0]
            elif job.errored:
                self.errored[job.state_index] = self.running.pop(index)[0]

    @property
    def all_failed(self) -> bool:
        return (self.unrunnable or self.errored) and not (
            self.successful or self.blocked or self.queued
        )

    def _resolve_lazy_inputs(
        self,
        task: "Task",
        state_index: int | None = None,
    ) -> "Task":
        """Resolves lazy fields in the task by replacing them with their
        actual values calculated by upstream jobs.

        Parameters
        ----------
        task : Task
            The task to resolve the lazy fields of
        state_index : int, optional
            The state index for the workflow, by default None

        Returns
        -------
        Task
            The task with all lazy fields resolved
        """
        resolved = {}
        for name, value in attrs_values(task).items():
            if isinstance(value, LazyField):
                resolved[name] = value._get_value(
                    workflow=self.workflow, graph=self.graph, state_index=state_index
                )
        return attrs.evolve(task, **resolved)

    def _split_task(self) -> dict[int, "Task[OutputType]"]:
        """Split the task into the different states it will be run over

        Parameters
        ----------
        values : dict[str, Any]
            The values to use for the split
        """
        # TODO: doesn't work properly for more cmplicated wf (check if still an issue)
        if not self.node.state:
            return {None: self.node._task}
        split_defs = []
        for index, vals in zip(self.node.state.inputs_ind, self.node.state.states_val):
            resolved = {}
            for inpt_name in set(self.node.input_names):
                value = getattr(self._task, inpt_name)
                state_key = f"{self.node.name}.{inpt_name}"
                try:
                    resolved[inpt_name] = vals[state_key]
                except KeyError:
                    if isinstance(value, LazyField):
                        resolved[inpt_name] = value._get_value(
                            workflow=self.workflow,
                            graph=self.graph,
                            state_index=index.get(state_key),
                        )
            split_defs.append(attrs.evolve(self.node._task, **resolved))
        return split_defs

    def get_runnable_tasks(self, graph: DiGraph) -> list["Job[TaskType]"]:
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
        runnable: list["Job[TaskType]"] = []
        predecessors: list["Job[TaskType]"] = graph.predecessors[self.node.name]

        # If there is a split, we need to wait for all predecessor nodes to finish
        # In theory, if the current splitter splits an already split state we should
        # only need to wait for the direct predecessor jobs to finish, however, this
        # would require a deep refactor of the State class as we need the whole state
        # in order to assign consistent state indices across the new split

        # FIXME: The branch for handling partially completed/errored/unrunnable
        # predecessor nodes can't be used until the State class can be partially
        # initialised with lazy-fields.
        if True:  # self.node.splitter:
            if unrunnable := [p for p in predecessors if p.errored or p.unrunnable]:
                self.unrunnable = {None: unrunnable}
                self.blocked = {}
                assert self.done
            else:
                if all(p.done for p in predecessors):
                    if not self.started:
                        self.start()
                    if self.node.state is None:
                        inds = [None]
                    else:
                        inds = list(range(len(self.node.state.states_ind)))
                    if self.blocked:
                        for i in inds:
                            runnable.append(self.blocked.pop(i))
        else:
            if not self.started:
                self.start()

            # Check to see if any blocked tasks are now runnable/unrunnable
            for index, job in list(self.blocked.items()):
                pred: NodeExecution
                is_runnable = True
                states_ind = (
                    list(self.node.state.states_ind[index].items())
                    if self.node.state
                    else []
                )
                for pred in predecessors:
                    if pred.node.state:
                        pred_states_ind = {
                            (k, i)
                            for k, i in states_ind
                            if k.startswith(pred.name + ".")
                        }
                        pred_inds = [
                            i
                            for i, ind in enumerate(pred.node.state.states_ind)
                            if set(ind.items()).issuperset(pred_states_ind)
                        ]
                    else:
                        pred_inds = [None]
                    if not all(i in pred.successful for i in pred_inds):
                        is_runnable = False
                        blocked = True
                        if pred_errored := [
                            pred.errored[i] for i in pred_inds if i in pred.errored
                        ]:
                            self.unrunnable[index].extend(pred_errored)
                            blocked = False
                        if pred_unrunnable := [
                            pred.unrunnable[i]
                            for i in pred_inds
                            if i in pred.unrunnable
                        ]:
                            self.unrunnable[index].extend(pred_unrunnable)
                            blocked = False
                        if not blocked:
                            del self.blocked[index]
                        break
                if is_runnable:
                    runnable.append(self.blocked.pop(index))
        self.queued.update({t.state_index: t for t in runnable})
        return list(self.queued.values())


async def prepare_runnable(runnable):
    return runnable.pickle_task()


# def _list_blocked_tasks(graph):
#     """Generates a list of tasks that can't be run and predecessors that are blocking
#     them to help debugging of broken workflows"""
#     blocked = []
#     for tsk in graph.sorted_nodes:
#         blocking = []
#         for pred in graph.predecessors[tsk.name]:
#             if not pred.done:
#                 matching_name = []
#                 for cache_loc in tsk.readonly_caches:
#                     for tsk_work_dir in cache_loc.iterdir():
#                         if (tsk_work_dir / "_job.pklz").exists():
#                             with open(tsk_work_dir / "_job.pklz", "rb") as f:
#                                 saved_tsk = pickle.load(f)
#                             if saved_tsk.name == pred.name:
#                                 matching_name.append(
#                                     f"{saved_tsk.name} ({tsk_work_dir.name})"
#                                 )
#                 blocking.append((pred, ", ".join(matching_name)))
#         if blocking:
#             blocked.append(
#                 f"\n{tsk.name} ({tsk.checksum}) is blocked by "
#                 + "; ".join(
#                     f"{pred.name} ({pred.checksum}), which matches names of [{matching}]"
#                     for pred, matching in blocking
#                 )
#             )
#     return blocked


def get_open_loop():
    """
    Get current event loop.

    If the loop is closed, a new
    loop is created and set as the current event loop.

    Returns
    -------
    loop : :obj:`asyncio.EventLoop`
        The current event loop

    """
    if os.name == "nt":
        loop = asyncio.ProactorEventLoop()  # for subprocess' pipes on Windows
    else:
        try:
            loop = asyncio.get_event_loop()
        # in case RuntimeError: There is no current event loop in thread 'MainThread'
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
    return loop
