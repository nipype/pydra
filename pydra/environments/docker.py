import typing as ty
import logging
from pydra.compose import shell
from pydra.environments import base, images

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


class Docker(base.Container):
    """Docker environment."""

    def _ref_key(self) -> ty.Hashable:
        repository, digest = images.split_reference(self.image)
        return ("docker", images.normal(repository), digest, self.tag, self.identity)

    def resolve(self) -> base.ResolvedRef:
        """Resolve the image once: the reference given, the local image store, then
        the registry (RFC 0004 section 5.3). The image runs by the returned ``run``.

        Output of ``docker image inspect`` that cannot be read raises ``ImageResolutionError``,
        even under ``resolve_images="best-effort"``.

        ``identity=`` is the author's assertion and is keyed as written (``docker@1:``
        followed by it, whatever it is). A ``docker-local@1:<Id>`` key names an image
        in this host's image store: it is reused on this host, and on workers that load
        the same image, and a worker without the image fails.
        """
        repository, digest = images.split_reference(self.image)
        carried = {"repository": repository}
        if self.identity is not None:
            run = self.image if digest else f"{self.image}:{self.tag}"
            return base.ResolvedRef(f"docker@1:{self.identity}", run, carried)
        if digest:
            return base.ResolvedRef(f"docker@1:{digest}", self.image, carried)
        tagged = f"{self.image}:{self.tag}"
        local = self._inspect(tagged)
        if local is not None:
            image_id, repo_digests = local
            mine = [
                d
                for d in repo_digests
                if images.normal(d.partition("@")[0]) == images.normal(repository)
            ]
            recorded = {"image_id": image_id}
            if len(mine) == 1:
                found = mine[0].partition("@")[2]
                return base.ResolvedRef(
                    f"docker@1:{found}", f"{repository}@{found}", carried, recorded
                )
            return base.ResolvedRef(
                f"docker-local@1:{image_id}", image_id, carried, recorded
            )
        answer = images.registry_manifest(repository, self.tag)
        if answer is None:
            return base.ResolvedRef(f"unresolved:docker:{tagged}", tagged, carried)
        found, platform_digest = answer
        recorded = (
            {} if platform_digest is None else {"platform_digest": platform_digest}
        )
        return base.ResolvedRef(
            f"docker@1:{found}", f"{repository}@{found}", carried, recorded
        )

    def _inspect(self, tagged: str) -> tuple[str, list[str]] | None:
        """``(Id, RepoDigests)`` of the image in the local store, ``None`` if Docker or
        the image is not there. It never pulls."""
        import json

        try:
            rc, out, _ = base.execute(
                [
                    "docker",
                    "image",
                    "inspect",
                    "--format",
                    "{{.Id}}\t{{json .RepoDigests}}",
                    tagged,
                ],
                strip=True,
            )
        except OSError:
            return None
        if rc != 0 or not out:
            return None
        image_id, _, digests = out.partition("\t")
        try:
            repo_digests = json.loads(digests or "null") or []
        except ValueError as e:
            raise base.ImageResolutionError(
                f"'docker image inspect' of {tagged!r} gave an unreadable answer: {out!r}"
            ) from e
        return image_id, repo_digests

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        if job.environment is not self:
            raise base.ImageResolutionError(
                f"{self!r} was asked to run a job built with the environment "
                f"{job.environment!r}, whose reference is the one carried with it"
            )
        # the reference the submitting process resolved and carried with the job
        docker_img = job.resolved.run
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

        job.cache_dir.mkdir(parents=True, exist_ok=True)
        values = base.execute(
            docker_args + [docker_img] + job.task._command_args(values=arg_values),
            cwd=job.cache_dir,
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
