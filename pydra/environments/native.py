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
        keys = ["return_code", "stdout", "stderr"]
        cmd_args = job.task._command_args(values=job.inputs)
        values = base.execute(cmd_args)
        output = dict(zip(keys, values))
        if output["return_code"]:
            msg = f"Error running '{job.name}' job with {cmd_args}:"
            if output["stderr"]:
                msg += "\n\nstderr:\n" + output["stderr"]
            if output["stdout"]:
                msg += "\n\nstdout:\n" + output["stdout"]
            raise RuntimeError(msg)
        return output


# Alias so it can be referred to as native.Environment
Environment = Native
