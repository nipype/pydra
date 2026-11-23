"""Handle execution backends."""

import asyncio
import typing as ty
import warnings
import os
from pathlib import Path
from tempfile import mkdtemp
from copy import deepcopy
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
from pydra.engine.graph import DiGraph  # noqa: F401
from pydra.utils.general import (
    get_fields,
    is_workflow,
)
from pydra.utils.hash import PersistentCache
from pydra.engine.audit import Audit
from pydra.engine.drivers import drive, drive_inline
from pydra.engine.executor import InlineExecutor, WorkerExecutor, no_result_diagnosis
from pydra.engine.job import Job
from pydra.engine.runner import (  # noqa: F401
    Failed,
    GraphRunner,
    JobErrorEntry,
    NodeExecution,
    OnError,
)
from pydra.engine.store import (
    PickleStore,
    ResultStore,
    ReusePolicy,
)
from pydra.utils.messenger import AuditFlag, Messenger
from pydra.utils.general import default_run_cache_root
from pydra.compose import workflow
from pydra.engine.state import State
from pydra.workers.base import Worker
from pydra.utils.routing import DEFAULT_WORKER, Router, default_router
from pydra.compose.base import Task, Outputs

logger = logging.getLogger("pydra.submitter")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result
    from pydra.engine.hooks import TaskHooks
    from pydra.environments.base import Environment


TaskType = ty.TypeVar("TaskType", bound="Task")
OutputType = ty.TypeVar("OutputType", bound="Outputs")


WORKER_KWARG_FAIL_NOTE = "Attempting to instantiate worker submitter"


class WorkflowError(RuntimeError):
    """Raised when jobs of a workflow errored

    Attributes
    ----------
    errors : list[str]
        A description of each errored job, and of each element whose inputs could not
        be resolved
    entries : list[JobErrorEntry]
        A record of each errored job and of each element whose inputs could not be
        resolved, in the order of failure; see ``JobErrorEntry.phase``
    """

    def __init__(
        self,
        message: str,
        errors: list[str],
        entries: "list[JobErrorEntry] | None" = None,
    ):
        super().__init__(message)
        self.errors = errors
        self.entries = list(entries) if entries is not None else []

    def __reduce__(self):
        return (type(self), (self.args[0], self.errors, self.entries))


class Submitter:
    """Send a job to the execution backend.

    Parameters
    ----------
    cache_root : os.PathLike, optional
        Cache directory where the working directory/results for the job will be
        stored, by default None. Loading from a cache unpickles its files, so
        pointing Pydra at a cache trusts everyone who can write to it with code
        execution as the reading user: only use caches whose writers you trust
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
        Alternate cache locations to check for pre-computed results, by default None.
        These are unpickled like ``cache_root``, so they trust everyone who can write
        to them with code execution as the reading user: only list caches whose
        writers you trust
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
        Deprecated and ignored. A lease whose holder stopped renewing it is taken over
        once it is older than ``stale_after`` (see
        :class:`~pydra.engine.lease.LeaseSettings`), whichever host left it
    store : ResultStore, optional
        Where jobs run and their results are kept, by default a
        :class:`~pydra.engine.store.PickleStore` over ``cache_root`` and
        ``readonly_caches``
    reuse : str | ReusePolicy, optional
        Which stored results the default store reuses, given the execution context
        recorded beside each (see :func:`~pydra.engine.context.job_context`):
        ``"consistent"`` (default) those produced under the same Python major.minor,
        platform, environment and versions of the distributions the task reaches,
        ``"exact"`` only those produced in an equal context (including the full Python
        version and every installed distribution), ``"any"`` those produced in any
        context, a :class:`~pydra.engine.store.ReusePolicy` those whose context it
        accepts. A result whose context holds an unknown version is reused by no
        other process under ``"consistent"`` or ``"exact"``. Set the policy on the
        store itself when passing ``store``
    on_error : str, optional
        What to do when a job errors: "fail_fast" stops at the first error and raises
        it, "raise_at_end" keeps running every element that does not read an errored
        or unrunnable upstream element, and raises the errors together at the end. By
        default (None), the mode of the loop that expands the workflow: "fail_fast"
        for a synchronous ``worker`` such as *debug*, otherwise "raise_at_end" (workflows
        are always expanded by ``worker``, never routed, so the ``on_error`` attribute
        holds the mode used). Under "fail_fast" jobs that are already running
        on an asynchronous worker are not stopped (cancelling their futures does not
        stop the executor's processes), so they finish after the call returns, and
        their lock files remain until they do or until the pool is torn down with
        ``shutdown(cancel_futures=True)``
    telemetry : bool
        Whether to check for a newer version of pydra, once per session. The
        ``NO_ET`` environment variable also turns the check off
    chdir : bool
        Whether a job runs with the process working directory set to its cache
        directory, by default True. With ``chdir=False`` Python and workflow tasks run in
        the caller's working directory and the process directory never changes, so
        several jobs can share one process. Shell tasks run their command with the
        cache directory as its working directory, and relative outputs of shell tasks
        are resolved under it. A relative path a Python function writes lands in the
        caller's directory, not the job's cache directory, and is not moved: another job
        can overwrite it, so a cache hit can return a different file's content (a warning
        is emitted when a ``FileSet`` output lies outside the cache directory). The
        ``pre_run_task``/``post_run_task`` hooks see the caller's directory, and audit
        messages without an explicit ``message_dir`` go to ``<cache_dir>/messages``
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
    telemetry: bool
    # The error summary of the last call, one entry per errored job. It is per-instance
    # state reset by each call, so it is only meaningful when calls are not concurrent
    errors: list[str]
    # The records of the errored jobs of the last call, see ``errors``
    entries: list[JobErrorEntry]

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
        reuse: str | ReusePolicy | None = None,
        on_error: OnError | None = None,
        telemetry: bool = True,
        chdir: bool = True,
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

        self.chdir = chdir
        self.telemetry = telemetry
        if telemetry:
            Job.check_etelemetry()

        self.audit = Audit(
            audit_flags=audit_flags,
            messengers=messengers,
            messenger_args=messenger_args,
        )
        store_root = getattr(store, "cache_root", None)
        if store is not None:
            if readonly_caches is not None:
                raise ValueError(
                    "readonly_caches cannot be combined with store, "
                    "give them to the store instead"
                )
            if cache_root is None:
                cache_root = store_root
            elif (
                store_root is not None
                and Path(cache_root).resolve() != Path(store_root).resolve()
            ):
                raise ValueError(
                    f"cache_root {str(cache_root)!r} differs from the store's "
                    f"cache_root {str(store_root)!r}"
                )
        if cache_root is None:
            cache_root = default_run_cache_root()
        cache_root = Path(cache_root).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        self.cache_root = cache_root
        self.readonly_caches = readonly_caches
        if store is None:
            store = PickleStore(
                cache_root, readonly_caches, reuse=reuse or "consistent"
            )
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
        worker.plugin_name()  # a worker class must name its plugin
        self.worker = worker
        self.router = router
        self.workers = {DEFAULT_WORKER: worker}
        for name, spec in workers.items():
            self.workers[name] = self._make_worker(spec, {})
        self._sync_pool = None
        self.run_start_time = None
        if clean_stale_locks is not None:
            warnings.warn(
                "Submitter(clean_stale_locks=...) is deprecated and has no effect: a "
                "lease whose holder stopped renewing it is taken over after "
                "LeaseSettings.stale_after",
                DeprecationWarning,
                stacklevel=2,
            )
        self.clean_stale_locks = False
        if on_error is not None and on_error not in self.ON_ERROR_MODES:
            raise ValueError(
                f"'on_error' must be one of {self.ON_ERROR_MODES}, not {on_error!r}"
            )
        self._on_error_arg = on_error
        self.on_error = self._loop_on_error(self.worker.is_async)
        self.errors = []
        self.entries = []
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

    def route_name(self, job: "Job[TaskType]") -> str:
        """The name of the worker a job runs on.

        Only leaf jobs are routed: a workflow job is expanded by the submitter's own
        ``worker``, whatever its resources, and its nodes are placed individually.
        """
        if is_workflow(job.task):
            return DEFAULT_WORKER
        name = self.router(job)
        if name is None:
            name = DEFAULT_WORKER
        if name not in self.workers:
            raise ValueError(
                f"The router sent job {job.name!r} to {name!r}, which is not one of "
                f"the submitter's workers: {sorted(self.workers)}"
            )
        return name

    def worker_for(self, job: "Job[TaskType]") -> Worker:
        """The worker a job runs on, see :meth:`route_name`"""
        return self.workers[self.route_name(job)]

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
        self.entries = []
        try:
            self.run_start_time = datetime.now()
            self.submit(job, rerun=rerun)
        except Exception as exc:
            self.errors = exc.errors if isinstance(exc, WorkflowError) else []
            self.entries = exc.entries if isinstance(exc, WorkflowError) else []
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
            raise RuntimeError(no_result_diagnosis(job))
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
        state["_sync_pool"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore the loop and workers
        self.loop = get_open_loop()
        for worker in self.workers.values():
            worker.loop = self.loop

    def _limit(self, in_flight: int = 0) -> int | None:
        """The most jobs to hand out in one round given those already running, None if
        unlimited"""
        if self.max_concurrent == float("inf"):
            return None
        return max(0, int(self.max_concurrent) - in_flight)

    def _runner(
        self, wf_job: "Job[workflow.Task]", rerun: bool, is_async: bool
    ) -> GraphRunner:
        runner = GraphRunner(
            wf_job.task,
            self.store,
            on_error=self._loop_on_error(is_async=is_async),
            rerun=rerun,
            propagate_rerun=self.propagate_rerun,
            submitter=self,
            job=wf_job,
        )
        wf_job.return_values = {
            "workflow": runner.workflow,
            "exec_graph": runner.exec_graph,
        }
        return runner

    def expand_workflow(self, wf_job: "Job[workflow.Task]", rerun: bool) -> None:
        """Deprecated: expand and execute a workflow job synchronously. Use a
        :class:`~pydra.engine.runner.GraphRunner` with
        :func:`~pydra.engine.drivers.drive_inline`"""
        warnings.warn(
            "Submitter.expand_workflow is deprecated, use GraphRunner with drive_inline",
            DeprecationWarning,
            stacklevel=2,
        )
        self._expand(wf_job, rerun)

    def _expand(self, wf_job: "Job[workflow.Task]", rerun: bool) -> None:
        """Expand and execute a workflow job synchronously, in the calling thread"""
        runner = self._runner(wf_job, rerun, is_async=False)
        if not any(w.is_async for w in self.workers.values()):
            drive_inline(
                runner,
                InlineExecutor(self.worker, self.worker_for),
                on_error=runner.on_error,
                limit=self._limit(),
            )
        else:  # some job may run on a worker that is not inline
            self.loop.run_until_complete(self._drive(runner))
            if runner.on_error == "fail_fast" and runner.first_error is not None:
                raise runner.first_error
        if runner.errors:
            raise workflow_error(wf_job, runner.errors, runner.entries)

    async def expand_workflow_async(
        self, workflow_task: "Job[workflow.Task]", rerun: bool
    ) -> None:
        """Deprecated: expand and execute a workflow job asynchronously. Use a
        :class:`~pydra.engine.runner.GraphRunner` with the asynchronous driver"""
        warnings.warn(
            "Submitter.expand_workflow_async is deprecated, use GraphRunner with drive",
            DeprecationWarning,
            stacklevel=2,
        )
        await self._expand_async(workflow_task, rerun)

    async def _expand_async(
        self, workflow_task: "Job[workflow.Task]", rerun: bool
    ) -> None:
        """
        Expand and execute a workflow job asynchronously.

        Parameters
        ----------
        job : :obj:`~pydra.engine.job.Job[workflow.Task]`
            Workflow Job object
        """
        runner = self._runner(workflow_task, rerun, is_async=self.worker.is_async)
        try:
            await self._drive(runner)
        finally:
            if runner.errors:
                if (
                    not self.worker.is_async
                    and runner.on_error == "fail_fast"
                    and runner.first_error is not None
                ):  # a nested workflow of a synchronous run raises the task's exception
                    raise runner.first_error
                raise workflow_error(workflow_task, runner.errors, runner.entries)

    async def _drive(self, runner: GraphRunner) -> None:
        """Run the runner's jobs on executors for the submitter's workers, on the
        running loop"""
        executors = {
            name: WorkerExecutor(
                worker,
                blocking=None if worker.is_async else self._run_blocking,
            )
            for name, worker in self.workers.items()
        }
        await drive(
            runner,
            executors,
            self.route_name,
            max_concurrent=self.max_concurrent,
            on_error=runner.on_error,
        )

    def get_runnable_tasks(self, graph: DiGraph) -> list["Job[TaskType]"]:
        """Deprecated: the jobs of the released nodes of an execution graph. Use
        :meth:`GraphRunner.ready`"""
        warnings.warn(
            "Submitter.get_runnable_tasks is deprecated, use GraphRunner.ready",
            DeprecationWarning,
            stacklevel=2,
        )
        return GraphRunner._runnable_jobs(graph)

    def read_succeeded(self, jobs: list["Job[TaskType]"]) -> list["Job[TaskType]"]:
        """Deprecated: complete the jobs with a success record and return the others.
        Use :meth:`GraphRunner.ready`"""
        warnings.warn(
            "Submitter.read_succeeded is deprecated, use GraphRunner.ready",
            DeprecationWarning,
            stacklevel=2,
        )
        return GraphRunner._read_succeeded_jobs(self.store, jobs)

    @staticmethod
    def pop_resolution_errors(
        exec_graph: DiGraph["NodeExecution"],
        entries: "list[JobErrorEntry] | None" = None,
    ) -> list[str]:
        """Deprecated: the errors of elements whose inputs could not be resolved, not
        collected before. Use :attr:`GraphRunner.errors`"""
        warnings.warn(
            "Submitter.pop_resolution_errors is deprecated, use GraphRunner.errors",
            DeprecationWarning,
            stacklevel=2,
        )
        return [e for n in exec_graph.nodes for e in n.pop_resolution_errors(entries)]

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        """
        Close submitter.

        Do not close previously queued loop.

        """
        errors = []
        if self._sync_pool is not None:
            self._sync_pool.shutdown()
            self._sync_pool = None
        for worker in {id(w): w for w in self.workers.values()}.values():
            try:
                worker.close()
            except Exception as e:
                errors.append(e)
        try:
            if self._own_loop:
                self.loop.close()
        finally:
            if errors:
                raise errors[0]

    async def _run_blocking(self, worker: Worker, job: "Job", rerun: bool) -> "Result":
        """Run a job on a synchronous worker without blocking the event loop.

        Synchronous jobs run one at a time on a single helper thread, so dispatch,
        result collection and the polling of other workers carry on meanwhile. As
        ``Job.run`` changes the process's working directory, a job should not rely on
        it while sibling jobs of an in-process worker run.
        """
        if self._sync_pool is None:
            self._sync_pool = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="pydra-sync-worker"
            )
        return await asyncio.get_running_loop().run_in_executor(
            self._sync_pool, lambda: worker.run(job, rerun=rerun)
        )

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


def workflow_error(
    wf_job: "Job[workflow.Task]",
    errors: list[str],
    entries: "list[JobErrorEntry] | None" = None,
) -> WorkflowError:
    """The error raised at the end of a workflow whose jobs errored"""
    all_errors = "\n\n".join(errors)
    return WorkflowError(
        f"Workflow job {wf_job} failed with errors:\n\n{all_errors}\n\n"
        f"See output directory for details: {wf_job.cache_dir}",
        errors,
        entries,
    )


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
