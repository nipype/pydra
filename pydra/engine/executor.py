"""Executors: where the jobs a :class:`~pydra.engine.runner.GraphRunner` releases run"""

import asyncio
import typing as ty
import attrs
from pydra.engine.lease import LeaseSettings
from pydra.engine.runner import Failed, NotRun
from pydra.utils.general import is_workflow
from pydra.utils.retry import Backoff

READ_BACK = Backoff(first=0.05, cap=1.0)
"""How `WorkerExecutor` polls for a result a worker reported written, until its
``visibility_timeout``: quick for a local store, at most once a second for shared storage."""

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.runner import Outcome
    from pydra.workers.base import Worker


@attrs.define(eq=False)
class Handle:
    """The jobs handed to an executor in one ``submit`` call

    Attributes
    ----------
    executor : Executor
        The executor the jobs were handed to
    jobs : tuple[Job, ...]
        The jobs of the call
    open : set[Job]
        The jobs whose outcome has not been reported yet
    """

    executor: "Executor"
    jobs: "tuple[Job, ...]"
    open: "set[Job]"


@ty.runtime_checkable
class Executor(ty.Protocol):
    """Where the jobs a runner releases run

    Every job submitted is reported exactly once through the callable given to
    :meth:`bind`, as a ``Result``, a :class:`~pydra.engine.runner.Failed` or a
    :class:`~pydra.engine.runner.NotRun` (a job the worker cancelled). The callable may
    be invoked from any thread. :meth:`cancel` is called only as the run ends, after
    ``runner.close()``: the jobs of a cancelled handle are not reported, and are recorded
    ``"stopped"`` where their workflow failed. Cancelling in the middle of a run is not
    supported.
    """

    batches: bool
    checks_store: ty.ClassVar[bool]
    inline: ty.ClassVar[bool]
    # whether the driver feeds it one job at a time under ``fail_fast``
    serial: bool

    def bind(self, report: "ty.Callable[[Job, Outcome], None]") -> None:
        """Set the callable that receives the outcome of each job"""

    def submit(self, jobs: "ty.Sequence[Job]") -> Handle:
        """Start the jobs; an executor with ``batches = False`` gets one per call"""

    def cancel(self, handle: Handle) -> None:
        """Stop the jobs of the handle that have not finished, as the run ends; they are
        not reported"""

    def close(self) -> None:
        """Release what the executor holds"""


class InlineExecutor:
    """Runs each job to completion in the caller's thread, before ``submit`` returns

    Parameters
    ----------
    worker : Worker
        A synchronous worker, whose ``run`` executes a job
    worker_for : Callable[[Job], Worker], optional
        Chooses the worker of each job (the submitter's router); an exception it raises
        is the job's failure. Without it every job runs on ``worker``
    """

    batches = False
    checks_store = True
    inline = True
    serial = True

    def __init__(
        self, worker: "Worker", worker_for: "ty.Callable[[Job], Worker] | None" = None
    ):
        self.worker = worker
        self.worker_for = worker_for
        self._report: "ty.Callable[[Job, Outcome], None] | None" = None

    def bind(self, report: "ty.Callable[[Job, Outcome], None]") -> None:
        """Set the callable that receives the outcome of each job"""
        self._report = report

    def submit(self, jobs: "ty.Sequence[Job]") -> Handle:
        """Run the jobs in order and report each outcome; an exception a job raises is
        reported as :class:`~pydra.engine.runner.Failed` and does not stop the rest, a
        ``CancelledError`` the worker raises as :class:`~pydra.engine.runner.NotRun`"""
        if self._report is None:
            raise RuntimeError("bind() a report callable before submitting jobs")
        handle = Handle(self, tuple(jobs), set(jobs))
        for job in handle.jobs:
            try:
                worker = self.worker_for(job) if self.worker_for else self.worker
                outcome: "Outcome" = worker.run(job, rerun=job.rerun)
            except asyncio.CancelledError as exc:
                outcome = cancelled(exc)
            except Exception as exc:
                outcome = Failed(exc)
            handle.open.discard(job)
            self._report(job, outcome)
        return handle

    def cancel(self, handle: Handle) -> None:
        """Nothing is left running once ``submit`` has returned"""
        handle.open.clear()

    def close(self) -> None:
        """Nothing to release"""


def cancelled(exc: BaseException) -> NotRun:
    """The outcome of a job its worker cancelled: not run, and not held by infrastructure"""
    return NotRun(reason=f"cancelled by its worker: {exc!r}", held=False)


def no_result_diagnosis(job: "Job") -> str:
    """Why a job that its worker reported finished has no result in the store"""
    if job.lockfile.exists():
        return (
            f"Job {job} has a lease, but no result was found. "
            "This may be due to another submission that is currently running. "
            f"A lease left by an interrupted run ({str(job.lockfile)!r}) is "
            "taken over once it has not been renewed for "
            "LeaseSettings.stale_after; resubmit then."
        )
    return f"Job {job} has no result in {str(job.cache_dir)!r}"


class WorkerExecutor:
    """Runs jobs on a :class:`~pydra.workers.base.Worker`, on the driver's event loop

    ``worker.run(job)`` is scheduled for each job, in one go for the jobs of a
    ``submit`` call, and the job is reported when it ends. A worker that returns
    ``True`` instead of a result (slurm, sge) is read back with ``job.result()``,
    retried with backoff for up to ``visibility_timeout`` seconds, then reported
    :class:`~pydra.engine.runner.Failed`. A synchronous worker runs on a helper thread.
    A worker declaring ``batches`` (slurm, sge) is given the jobs of a ``submit`` call
    together through ``submit_many(jobs)``, so it can place them in arrays. One declaring
    ``reports_done`` is also passed ``done(job, outcome)``, so a job is reported when it
    ends rather than when the round does.

    Parameters
    ----------
    worker : Worker
        The worker that runs the jobs
    visibility_timeout : float, optional
        Seconds a result the worker reported written may take to show; by default
        `LeaseSettings.visibility_timeout` of the default settings
    blocking : Callable[[Worker, Job, bool], Awaitable], optional
        Runs a job of a synchronous worker off the loop; by default the loop's own
        thread pool, the submitter passes its single helper thread
    """

    batches = False
    checks_store = True
    inline = False

    @property
    def serial(self) -> bool:
        """Whether the worker is synchronous: its jobs run one at a time on a helper
        thread, which the driver feeds one at a time under ``fail_fast``"""
        return not self.worker.is_async

    def __init__(
        self,
        worker: "Worker",
        *,
        visibility_timeout: float | None = None,
        blocking: "ty.Callable[..., ty.Awaitable] | None" = None,
    ):
        self.worker = worker
        self.visibility_timeout = (
            LeaseSettings().visibility_timeout
            if visibility_timeout is None
            else visibility_timeout
        )
        self.blocking = blocking
        self.batches = worker.batches
        self._reports_done = worker.reports_done
        self._report: "ty.Callable[[Job, Outcome], None] | None" = None
        self._tasks: "dict[Job, asyncio.Task]" = {}
        # the submit_many calls of each handle
        self._submissions: "dict[Handle, set[asyncio.Task]]" = {}
        self._pending: "dict[Job, asyncio.Future]" = {}

    def bind(self, report: "ty.Callable[[Job, Outcome], None]") -> None:
        """Set the callable that receives the outcome of each job"""
        self._report = report

    def submit(self, jobs: "ty.Sequence[Job]") -> Handle:
        """Schedule every job on the running loop before yielding to it"""
        if self._report is None:
            raise RuntimeError("bind() a report callable before submitting jobs")
        handle = Handle(self, tuple(jobs), set(jobs))
        loop = asyncio.get_running_loop()
        if self.batches:
            placed = [j for j in handle.jobs if not is_workflow(j.task)]
            for job in placed:
                self._pending[job] = loop.create_future()
            if placed:
                task = loop.create_task(self._submit_many(placed, handle))
                self._submissions.setdefault(handle, set()).add(task)
        for job in handle.jobs:
            task = loop.create_task(self._run(job, handle), name=job.checksum)
            self._tasks[job] = task
        return handle

    async def _submit_many(self, jobs: "list[Job]", handle: Handle) -> None:
        """Give the worker a round; each job's future is set as the job ends"""

        def ended(job: "Job", outcome: "bool | BaseException") -> None:
            future = self._pending.get(job)
            if future is not None and not future.done():
                if isinstance(outcome, asyncio.CancelledError):
                    future.set_result(cancelled(outcome))  # the worker cancelled it
                elif isinstance(outcome, Exception):
                    future.set_exception(outcome)
                else:
                    future.set_result(outcome)

        try:
            if self._reports_done:
                outcomes = await self.worker.submit_many(jobs, done=ended)
            else:
                outcomes = await self.worker.submit_many(jobs)
            if len(outcomes) != len(jobs):
                raise RuntimeError(
                    f"{type(self.worker).__name__}.submit_many returned "
                    f"{len(outcomes)} outcomes for {len(jobs)} jobs"
                )
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if task not in self._submissions.get(handle, ()) or (
                task is not None and task.cancelling()
            ):
                self._forget_submission(handle, task)
                raise  # cancelled by cancel() or by the loop: not reported
            outcomes = [exc] * len(jobs)  # raised by the worker: its jobs did not run
        except Exception as exc:
            outcomes = [exc] * len(jobs)
        finally:
            self._forget_submission(handle, asyncio.current_task())
        for job, outcome in zip(jobs, outcomes):
            ended(job, outcome)

    def _forget_submission(self, handle: Handle, task) -> None:
        tasks = self._submissions.get(handle)
        if tasks is not None:
            tasks.discard(task)
            if not tasks:
                del self._submissions[handle]

    def cancel(self, handle: Handle) -> None:
        """Cancel the jobs of the handle that are still running; they are not reported
        (the driver cancels only as the run ends)"""
        for submission in self._submissions.pop(handle, ()):
            submission.cancel()
        for job in list(handle.open):
            task = self._tasks.pop(job, None)
            if task is not None:
                task.cancel()
            self._pending.pop(job, None)
        handle.open.clear()

    def close(self) -> None:
        """Nothing to release: the worker belongs to the submitter"""

    async def _run(self, job: "Job", handle: Handle) -> None:
        try:
            outcome = await self._execute(job)
        except asyncio.CancelledError as exc:
            task = asyncio.current_task()
            if job not in handle.open or (task is not None and task.cancelling()):
                raise  # cancelled by cancel() or by the loop: not reported
            outcome = cancelled(exc)  # raised by the worker: not run
        except Exception as exc:
            outcome = Failed(exc)
        self._tasks.pop(job, None)
        self._pending.pop(job, None)
        if job not in handle.open:  # cancelled while the outcome was being read
            return
        handle.open.discard(job)
        self._report(job, outcome)

    async def _execute(self, job: "Job") -> "Outcome":
        if is_workflow(job.task):  # expanded on this loop, not placed on the worker
            outcome = await job.run_async(rerun=job.rerun)
        elif job in self._pending:  # the worker has the whole round
            outcome = await self._pending[job]
        elif self.worker.is_async:
            outcome = await self.worker.run(job, rerun=job.rerun)
        elif self.blocking is not None:
            outcome = await self.blocking(self.worker, job, job.rerun)
        else:
            outcome = await asyncio.get_running_loop().run_in_executor(
                None, lambda: self.worker.run(job, rerun=job.rerun)
            )
        if outcome is True:  # the result exists only in the store
            outcome = await self._read_back(job)
        return outcome

    async def _read_back(self, job: "Job") -> "Outcome":
        tries = READ_BACK.tries(self.visibility_timeout)
        while True:
            # reported complete: under a store with work directories a result not found
            # is an errored one naming the work directory (RFC 0003 section 7.2)
            result = job.result(settled=True)
            if result is not None:
                return result
            wait = tries.next_wait()
            if wait is None:
                return Failed(RuntimeError(no_result_diagnosis(job)))
            await asyncio.sleep(wait)
