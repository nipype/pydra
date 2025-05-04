import typing as ty
import logging
from pydra.compose import shell
from pydra.environments import base

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Singularity(base.Container):
    """Singularity environment."""

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        singularity_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts, values = self.get_bindings(job=job, root=self.root)

        # todo adding xargsy etc
        singularity_args = [
            "singularity",
            "exec",
            *self.xargs,
        ]
        singularity_args.extend(
            " ".join(
                [f"-B {key}:{val[0]}:{val[1]}" for (key, val) in mounts.items()]
            ).split()
        )
        singularity_args.extend(
            ["--pwd", f"{self.root.rstrip('/')}{job.cache_dir.absolute()}"]
        )
        job.cache_dir.mkdir(exist_ok=True)
        return base.read_and_display(
            singularity_args
            + [singularity_img]
            + job.task._command_args(values=values),
        )


# Alias so it can be referred to as singularity.Environment
Environment = Singularity
