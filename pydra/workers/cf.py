import os
import attrs
import typing as ty
import cloudpickle as cp
import concurrent.futures as cf
import logging
from pydra.engine.job import Job
from pydra.workers import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


def get_available_cpus():
    """
    Return the number of CPUs available to the current process or, if that is not
    available, the total number of CPUs on the system.

    Returns
    -------
    n_proc : :obj:`int`
        The number of available CPUs.
    """
    # Will not work on some systems or if psutil is not installed.
    # See https://psutil.readthedocs.io/en/latest/#psutil.Process.cpu_affinity
    try:
        import psutil

        return len(psutil.Process().cpu_affinity())
    except (AttributeError, ImportError, NotImplementedError):
        pass

    # Not available on all systems, including macOS.
    # See https://docs.python.org/3/library/os.html#os.sched_getaffinity
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    # Last resort
    return os.cpu_count()


@attrs.define
class ConcurrentFuturesWorker(base.Worker):
    """A worker to execute in parallel using Python's concurrent futures."""

    n_procs: int = attrs.field(factory=get_available_cpus)
    pool: cf.ProcessPoolExecutor = attrs.field(
        eq=False, init=False, hash=False, repr=False
    )

    @pool.default
    def _pool_default(self) -> cf.ProcessPoolExecutor:
        return cf.ProcessPoolExecutor(self.n_procs)

    def __getstate__(self) -> dict[str, ty.Any]:
        """Return state for pickling."""
        state = super().__getstate__()
        del state["pool"]
        return state

    def __setstate__(self, state: dict[str, ty.Any]) -> None:
        """Set state from pickling."""
        super().__setstate__(state)
        self.pool = cf.ProcessPoolExecutor(self.n_procs)

    async def run(
        self,
        job: Job[base.TaskType],
        rerun: bool = False,
    ) -> "Result":
        """Run a job."""
        assert self.loop, "No event loop available to submit tasks"
        job_pkl = cp.dumps(job)
        return await self.loop.run_in_executor(
            self.pool, self.uncloudpickle_and_run, job_pkl, rerun
        )

    @classmethod
    def uncloudpickle_and_run(cls, job_pkl: bytes, rerun: bool) -> "Result":
        """Unpickle and run a job."""
        job: Job[base.TaskType] = cp.loads(job_pkl)
        return job.run(rerun=rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        self.pool.shutdown()


# Alias so it can be referred to as cf.Worker
Worker = ConcurrentFuturesWorker
