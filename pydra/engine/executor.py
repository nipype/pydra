"""Executors: where the jobs a :class:`~pydra.engine.runner.GraphRunner` releases run"""

import typing as ty
import attrs
from pydra.engine.runner import Failed

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.runner import Outcome
    from pydra.workers.base import Worker


@attrs.define(eq=False)
class Handle:
    """The jobs handed to an executor in one :meth:`InlineExecutor.submit` call

    Attributes
    ----------
    executor : InlineExecutor
        The executor the jobs were handed to
    jobs : tuple[Job, ...]
        The jobs of the call
    open : set[Job]
        The jobs whose outcome has not been reported yet
    """

    executor: "InlineExecutor"
    jobs: "tuple[Job, ...]"
    open: "set[Job]"


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
        reported as :class:`~pydra.engine.runner.Failed` and does not stop the rest"""
        if self._report is None:
            raise RuntimeError("bind() a report callable before submitting jobs")
        handle = Handle(self, tuple(jobs), set(jobs))
        for job in handle.jobs:
            try:
                worker = self.worker_for(job) if self.worker_for else self.worker
                outcome: "Outcome" = worker.run(job, rerun=job.rerun)
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
