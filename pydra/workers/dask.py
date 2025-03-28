import typing as ty
import logging
import attrs
from pydra.engine.job import Job
from . import base

try:
    from dask.distributed import Client
except ImportError:
    pass

logger = logging.getLogger("pydra.worker")

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


@attrs.define
class Worker(base.Worker):
    """A worker to execute in parallel using Dask.distributed.
    This is an experimental implementation with limited testing.
    """

    client_args: ty.Dict[str, ty.Any] = attrs.field(factory=dict)

    async def run(
        self,
        job: "Job[base.TaskType]",
        rerun: bool = False,
    ) -> "Result":

        async with Client(**self.client_args, asynchronous=True) as client:
            return await client.submit(job.run, rerun)

    def close(self):
        """Finalize the internal pool of tasks."""
        pass
