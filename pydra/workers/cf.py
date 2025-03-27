import asyncio
import os
import typing as ty
import cloudpickle as cp
import concurrent.futures as cf
import logging
from pydra.engine.job import Job
from . import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


class Worker(base.Worker):
    """A worker to execute in parallel using Python's concurrent futures."""

    plugin_name = "cf"

    n_procs: int
    loop: asyncio.AbstractEventLoop
    pool: cf.ProcessPoolExecutor

    def __init__(self, n_procs: int | None = None):
        """Initialize Worker."""
        super().__init__()
        self.n_procs = get_available_cpus() if n_procs is None else n_procs
        # added cpu_count to verify, remove once confident and let PPE handle
        self.pool = cf.ProcessPoolExecutor(self.n_procs)
        # self.loop = asyncio.get_event_loop()
        logger.debug("Initialize ConcurrentFuture")

    async def run(
        self,
        job: Job[base.TaskType],
        rerun: bool = False,
    ) -> "Result":
        """Run a job."""
        assert self.loop, "No event loop available to submit tasks"
        return await self.loop.run_in_executor(
            self.pool, self.uncloudpickle_and_run, cp.dumps(job), rerun
        )

    @classmethod
    def uncloudpickle_and_run(cls, job_pkl: bytes, rerun: bool) -> "Result":
        """Unpickle and run a job."""
        job: Job[base.TaskType] = cp.loads(job_pkl)
        return job.run(rerun=rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        self.pool.shutdown()


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
