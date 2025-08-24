import typing as ty
import logging
from pydra.compose import shell
from pydra.environments import base

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Native(base.Environment):
    """
    Native environment, i.e. the tasks are executed in the current shell environment.
    """

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        cmd_args = job.task._command_args(values=job.inputs)
        return base.read_and_display(*cmd_args)


# Alias so it can be referred to as native.Environment
Environment = Native
