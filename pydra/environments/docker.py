import typing as ty
import logging
from pydra.compose import shell
from pydra.environments import base

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Docker(base.Container):
    """Docker environment."""

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        docker_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts, values = self.get_bindings(job=job, root=self.root)

        docker_args = [
            "docker",
            "run",
            *self.xargs,
        ]
        docker_args.extend(
            " ".join(
                [f"-v {key}:{val[0]}:{val[1]}" for (key, val) in mounts.items()]
            ).split()
        )
        docker_args.extend(["-w", f"{self.root}{job.cache_dir}"])

        job.cache_dir.mkdir(exist_ok=True)
        return base.read_and_display(
            *(docker_args + [docker_img] + job.task._command_args(values=values)),
        )


# Alias so it can be referred to as docker.Environment
Environment = Docker
