import typing as ty
import logging
from pydra.engine.job import Job
from . import base

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


logger = logging.getLogger("pydra.worker")


class Worker(base.Worker):
    """A worker to execute linearly."""

    plugin_name: str = "debug"

    def __init__(self, **kwargs):
        """Initialize worker."""
        logger.debug("Initialize SerialWorker")

    def run(
        self,
        job: "Job[base.TaskType]",
        rerun: bool = False,
    ) -> "Result":
        """Run a job."""
        return job.run(rerun=rerun)

    def close(self):
        """Return whether the job is finished."""

    async def fetch_finished(self, futures):
        raise NotImplementedError("DebugWorker does not support async execution")
