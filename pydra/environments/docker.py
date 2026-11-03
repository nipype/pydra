import typing as ty
import logging
from pydra.compose import shell
from pydra.environments import base

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Docker(base.Container):
    """Docker environment."""

    def _resolve_digest(self) -> str | None:
        """The local image ID (``docker image inspect``), pulling the image first if it
        is not present. It is not a registry ``RepoDigest``, so it differs between
        architectures for the same tag."""
        image = f"{self.image}:{self.tag}"
        inspect = ["docker", "image", "inspect", "--format", "{{.Id}}", image]
        try:
            rc, out, _ = base.execute(inspect, strip=True)
            if rc != 0:
                if base.execute(["docker", "pull", "--quiet", image])[0] != 0:
                    return None
                rc, out, _ = base.execute(inspect, strip=True)
        except OSError:
            return None
        return out if rc == 0 and out else None

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        docker_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts, arg_values = self.get_bindings(job=job, root=self.root)

        docker_args = [
            "docker",
            "run",
            *self.xargs,
        ]
        for key, val in mounts.items():
            docker_args.extend(["-v", f"{key}:{val[0]}:{val[1]}"])
        docker_args.extend(["-w", f"{self.root}{job.cache_dir}"])
        keys = ["return_code", "stdout", "stderr"]

        job.cache_dir.mkdir(exist_ok=True)
        values = base.execute(
            docker_args + [docker_img] + job.task._command_args(values=arg_values),
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stdout"] + "\n" + output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output


# Alias so it can be referred to as docker.Environment
Environment = Docker
