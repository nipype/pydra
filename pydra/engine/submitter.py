"""Handle execution backends."""

import asyncio
import typing as ty
import re
import os
from pathlib import Path
from traceback import format_exc
from tempfile import mkdtemp
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import attrs
import logging
from pydra.engine.graph import DiGraph
from pydra.utils.general import (
    get_fields,
    attrs_values,
    is_workflow,
)
from pydra.utils.hash import PersistentCache
from pydra.engine.lazy import LazyField, LazyOutField
from pydra.engine.audit import Audit
from pydra.engine.job import Job
from pydra.engine.store import PickleStore, ResultStore
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils.general import default_run_cache_root
from pydra.compose import workflow
from pydra.engine.state import State
from pydra.workers.base import Worker
from pydra.utils.routing import DEFAULT_WORKER, Router, default_router
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

OnError = ty.Literal["raise_at_end", "fail_fast"]


class Submitter:
    """Send a job to the execution backend.

    Parameters
    ----------
    cache_root : os.PathLike, optional
        Cache directory where the working directory/results for the job will be
        stored, by default None
    worker : str or Worker, optional
        The worker to use, by default "cf". It runs every job the router does not send
        elsewhere, and is available to the router under the name ``"default"``
    workers : dict[str, str or Worker], optional
        Further workers by name (each a worker name, class or object), for the router to
        choose between. The name ``"default"`` is reserved for ``worker``
    router : callable, optional
        Called with each :class:`~pydra.engine.job.Job` at dispatch, returns the name of
        the worker to run it on (or ``None`` for ``worker``). By default every job goes
        to ``worker``. See :func:`pydra.utils.routing.tag_router`
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
    store : ResultStore, optional
        Where jobs run and their results are kept, by default a
        :class:`~pydra.engine.store.PickleStore` over ``cache_root`` and
        ``readonly_caches``
    reuse : str, optional
        Which stored results the default store reuses: ``"exact"`` (default) only
        those produced in the same execution context (environment checksum, platform,
        Python version), ``"any"`` those produced in any context. Set the policy on
        the store itself when passing ``store``
    on_error : str, optional
        What to do when a job errors: "fail_fast" stops at the first error and raises
        it, "raise_at_end" keeps running every element that does not read an errored
        or unrunnable upstream element, and raises the errors together at the end. By
        default (None), the mode of the loop that expands the workflow: "fail_fast"
        where the router sends the workflow to a synchronous worker such as *debug*,
        otherwise "raise_at_end". The ``on_error`` attribute then holds the mode for
        a workflow sent to ``worker``
    **kwargs : dict
        Keyword arguments to pass on to the worker initialisation
    """

    cache_root: os.PathLike
    worker: Worker
    workers: dict[str, Worker]
    router: Router
    environment: "Environment | None"
    readonly_caches: list[os.PathLike]
    audit_flags: AuditFlag
    messengers: ty.Iterable[Messenger]
    messenger_args: dict[str, ty.Any]
    max_concurrent: int | float
    clean_stale_locks: bool
    store: ResultStore
    run_start_time: datetime | None
    propagate_rerun: bool
    on_error: OnError
    # The error summary of the last run, one entry per errored job
    errors: list[str]

    ON_ERROR_MODES = ty.get_args(OnError)

    def __init__(
        self,
        /,
        cache_root: os.PathLike | None = None,
        worker: str | ty.Type[Worker] | Worker | None = "debug",
        workers: "dict[str, str | ty.Type[Worker] | Worker] | None" = None,
        router: Router | None = None,
        environment: "Environment | None" = None,
        readonly_caches: list[os.PathLike] | None = None,
        audit_flags: AuditFlag = AuditFlag.NONE,
        messengers: ty.Iterable[Messenger] | None = None,
        messenger_args: dict[str, ty.Any] | None = None,
        max_concurrent: int | float = float("inf"),
        propagate_rerun: bool = True,
        clean_stale_locks: bool | None = None,
        store: ResultStore | None = None,
        reuse: str | None = None,
        on_error: OnError | None = None,
        **kwargs,
    ):

        from pydra.environments import native

        if worker is None:
            worker = "debug"
        if workers is None:
            workers = {}
        if DEFAULT_WORKER in workers:
            raise ValueError(
                f"{DEFAULT_WORKER!r} is reserved for the 'worker' argument, "
                "name the extra worker something else"
            )
        if router is None:
            router = default_router
        elif not callable(router):
            raise TypeError(f"'router' must be callable, not {router!r}")

        Job.check_etelemetry()

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
        if store is None:
            store = PickleStore(cache_root, readonly_caches, reuse=reuse or "exact")
        elif reuse is not None:
            raise ValueError(
                "'reuse' applies to the default store, set it on the store passed as "
                "'store' instead"
            )
        self.store = store
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
        worker = self._make_worker(worker, kwargs)
        self.worker = worker
        self.router = router
        self.workers = {DEFAULT_WORKER: worker}
        for name, spec in workers.items():
            self.workers[name] = self._make_worker(spec, {})
        self.run_start_time = None
        self.clean_stale_locks = (
            clean_stale_locks
            if clean_stale_locks is not None
            else (self.worker.plugin_name() == "debug")
        )
        if on_error is not None and on_error not in self.ON_ERROR_MODES:
            raise ValueError(
                f"'on_error' must be one of {self.ON_ERROR_MODES}, not {on_error!r}"
            )
        self._on_error_arg = on_error
        self.on_error = self._loop_on_error(self.worker.is_async)
        self.errors = []
        self.worker_kwargs = kwargs
        for each in self.workers.values():
            each.loop = self.loop

    @staticmethod
    def _make_worker(
        worker: "str | ty.Type[Worker] | Worker", kwargs: dict[str, ty.Any]
    ) -> Worker:
        """Instantiate a worker from its name or class, or pass an object through."""
        if isinstance(worker, Worker):
            return worker
        if isinstance(worker, str):
            worker_cls = Worker.plugin(worker)
        elif isinstance(worker, type) and issubclass(worker, Worker):
            worker_cls = worker
        else:
            raise TypeError(
                "Worker must be a Worker object, name of a worker or a Worker "
                f"class, not {worker}"
            )
        try:
            return worker_cls(**kwargs)
        except TypeError as e:
            e.add_note(WORKER_KWARG_FAIL_NOTE)
            raise

    def _loop_on_error(self, is_async: bool) -> OnError:
        """The on_error mode of an expansion loop: the one passed to the submitter, or
        the default of the synchronous or asynchronous loop."""
        if self._on_error_arg is not None:
            return self._on_error_arg
        return "raise_at_end" if is_async else "fail_fast"

    def worker_for(self, job: "Job[TaskType]") -> Worker:
        """The worker the router sends a job to."""
        name = self.router(job)
        if name is None:
            name = DEFAULT_WORKER
        try:
            return self.workers[name]
        except KeyError:
            raise ValueError(
                f"The router sent job {job.name!r} to {name!r}, which is not one of "
                f"the submitter's workers: {sorted(self.workers)}"
            ) from None

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
        self.errors = []
        try:
            self.run_start_time = datetime.now()
            self.submit(job, rerun=rerun)
        except Exception as exc:
            self.errors = getattr(exc, "errors", [])
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
        worker = self.worker_for(job)
        if worker.is_async:  # Only workflow tasks can be async
            self.loop.run_until_complete(worker.submit(job, rerun=rerun))
        else:
            worker.run(job, rerun=rerun)

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the unpicklable entries or those that should not be pickled
        # When unpickled (in another process) the submitter can't be called
        state["loop"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the loop and workers
        self.loop = get_open_loop()
        for worker in self.workers.values():
            worker.loop = self.loop

    def expand_workflow(self, wf_job: "Job[workflow.Task]", rerun: bool) -> None:
        """Expands and executes a workflow job synchronously. Typically only used during
        debugging and testing, as the asynchronous version is more efficient.

        Parameters
        ----------
        job : :obj:`~pydra.engine.job.Job[workflow.Task]`
            Workflow Job object

        """
        # Construct the workflow
        wf = wf_job.task.construct()
        # Generate the execution graph
        exec_graph = wf.execution_graph(submitter=self)
        wf_job.return_values = {"workflow": wf, "exec_graph": exec_graph}
        tasks = self.get_runnable_tasks(exec_graph)
        on_error = self._loop_on_error(is_async=False)
        errors = []
        while tasks or any(not n.done for n in exec_graph.nodes):
            for job in tasks:
                try:
                    if is_workflow(job.task):  # expanded here, not placed by the router
                        job.run(rerun=rerun and self.propagate_rerun)
                    else:
                        self.submit(job, rerun=rerun and self.propagate_rerun)
                except Exception:
                    if on_error == "fail_fast":
                        raise
                    # The job may have failed before it could save a result
                    job._errored = True
                    errors.append(job_error(job))
            tasks = self.get_runnable_tasks(exec_graph)
        if errors:
            raise workflow_error(wf_job, errors)

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
        on_error = self._loop_on_error(is_async=True)
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
                    if job.checksum not in futured:
                        job_rerun = rerun and self.propagate_rerun
                        if is_workflow(job.task):
                            # expanded in this loop, not placed by the router
                            coroutine = job.run_async(rerun=job_rerun)
                        elif (worker := self.worker_for(job)).is_async:
                            coroutine = worker.run(job, rerun=job_rerun)
                        else:  # a synchronous worker, run in this loop
                            coroutine = _run_blocking(worker, job, job_rerun)
                        asyncio_task = asyncio.Task(coroutine, name=job.checksum)
                        task_futures.add(asyncio_task)
                        futured[job.checksum] = job
                task_futures, completed = await self.fetch_finished(task_futures)
                for task_future in completed:
                    try:
                        task_future.result()
                    except Exception:
                        job = futured[task_future.get_name()]
                        # The job may have failed before it could save a result (e.g. if
                        # the worker was unable to run it), in which case it would
                        # otherwise remain queued indefinitely
                        job._errored = True
                        errors.append(job_error(job))
                if errors and on_error == "fail_fast":
                    break
                tasks = self.get_runnable_tasks(exec_graph)
        finally:
            # Cancel any outstanding jobs if exiting early (e.g. due to an exception)
            # so they aren't left running unattended on the event loop, which may be
            # shared with other code (e.g. in Jupyter notebooks)
            for task_future in task_futures:
                task_future.cancel()
            if errors:
                raise workflow_error(workflow_task, errors)

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
        for worker in self.workers.values():
            worker.close()
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
    blocked: dict[int | None, list[tuple["NodeExecution", list[int | None]]]] | None

    # Jobs of the elements, created when they are released or first read
    _tasks: dict[int | None, "Job[TaskType] | None"] | None

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
                if missing := [i for i in inds if i not in node._tasks]:
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
            elif job.run_start_time:
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

    def _read_indices(self, state_index: int | None) -> list[int | None]:
        """The indices of the elements of this node that a downstream lazy field reads
        at the given state index (see ``LazyOutField._get_value``)

        Parameters
        ----------
        state_index : int or None
            The state index the lazy field is read at, None to read all elements
        """
        if not self.state:
            return [None]
        if state_index is None or (self.state.combiner and not self.state.ind_l_final):
            return list(range(len(self.state.states_ind)))
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
        for index, upstream in list((self.blocked or {}).items()):
            if unrunnable := list(
                dict.fromkeys(
                    p
                    for p, inds in upstream
                    if any(i in p.errored or i in p.unrunnable for i in inds)
                )
            ):
                del self.blocked[index]
                self.unrunnable[index] = unrunnable
            elif all(i in p.successful for p, inds in upstream for i in inds):
                del self.blocked[index]
                self.queued[index] = self.job(index)
        return list(self.queued.values())


def job_error(job: "Job") -> str:
    """Describe the error of a job, called while its exception is being handled"""
    error_msg = format_exc()
    if match := re.match(r'.*"""(.*)""".*', error_msg, flags=re.DOTALL | re.MULTILINE):
        error_msg = match.group(1)
    task_name = job.name
    if job.state_index is not None:
        task_name += f"({job.state_index})"
    return f"Job {task_name!r}, {job.task!r}, errored:{error_msg}"


def workflow_error(wf_job: "Job[workflow.Task]", errors: list[str]) -> RuntimeError:
    """The error raised at the end of a workflow whose jobs errored"""
    all_errors = "\n\n".join(errors)
    error = RuntimeError(
        f"Workflow job {wf_job} failed with errors:\n\n{all_errors}\n\n"
        f"See output directory for details: {wf_job.cache_dir}"
    )
    error.errors = errors
    return error


async def _run_blocking(worker: Worker, job: "Job", rerun: bool) -> "Result":
    """Run a job on a synchronous worker from within the event loop."""
    return worker.run(job, rerun=rerun)


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
