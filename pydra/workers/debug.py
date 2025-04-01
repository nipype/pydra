import typing as ty
import logging
from pydra.engine.job import Job
from pydra.workers import base

if ty.TYPE_CHECKING:
    from pydra.engine.result import Result


logger = logging.getLogger("pydra.worker")


class DebugWorker(base.Worker):
    """A worker to execute linearly."""

    def run(
        self,
        job: "Job[base.TaskType]",
        rerun: bool = False,
    ) -> "Result":
        """Run a job."""
        return job.run(rerun=rerun)

    def close(self):
        """Return whether the job is finished."""


# Alias so it can be referred to as debug.Worker
Worker = DebugWorker
