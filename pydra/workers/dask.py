import typing as ty
import logging
from pydra.engine.job import Job
from . import base

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


class Worker(base.Worker):
    """A worker to execute in parallel using Dask.distributed.
    This is an experimental implementation with limited testing.
    """

    plugin_name = "dask"

    def __init__(self, **kwargs):
        """Initialize Worker."""
        super().__init__()
        try:
            from dask.distributed import Client  # noqa: F401
        except ImportError:
            logger.critical("Please instiall Dask distributed.")
            raise
        self.client = None
        self.client_args = kwargs
        logger.debug("Initialize Dask")

    async def run(
        self,
        job: "Job[base.TaskType]",
        rerun: bool = False,
    ) -> "Result":
        from dask.distributed import Client

        async with Client(**self.client_args, asynchronous=True) as client:
            return await client.submit(job.run, rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        pass
