import typing as ty
import logging
from pydra.compose import shell
from pydra.compose.shell.version import check_in_worker
from pydra.environments import base

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Native(base.Environment):
    """
    Native environment, i.e. the tasks are executed in the current shell environment.
    """

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        # The tool a worker finds must be the one the job was keyed by (RFC 0004 5.2)
        check_in_worker(job.task, job.record.get("job", {}).get("tool_version"))
        keys = ["return_code", "stdout", "stderr"]
        cmd_args = job.task._command_args(values=job.inputs)
        job.cache_dir.mkdir(parents=True, exist_ok=True)
        values = base.execute(cmd_args, cwd=job.cache_dir)
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
