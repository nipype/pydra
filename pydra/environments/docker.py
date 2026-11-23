import json
import os
import typing as ty
import logging
from pathlib import Path
import attrs
from pydra.compose import shell
from pydra.environments import base, images

logger = logging.getLogger("pydra")

SLEEP_FOREVER = str(2**31 - 1)
"""The keep-alive container's ``sleep``: the largest value ``sleep`` accepts everywhere (a signed
32-bit count of seconds)."""

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.resident import Device


@attrs.define
class Docker(base.Container):
    """Docker environment.

    Parameters
    ----------
    read_only : bool
        A warm container's root file system is read-only, with a ``tmpfs`` on ``/tmp``
    """

    read_only: bool = attrs.field(default=False, kw_only=True)

    def _warm_handle(
        self,
        ref: base.ResolvedRef,
        roots: frozenset[Path],
        device: "Device | None",
        writable: frozenset[Path],
        scratch: Path | None,
    ) -> "WarmDocker":
        return WarmDocker(self, ref, roots, device, writable, scratch)

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


def user_flag() -> list[str]:
    """``--user <uid>:<gid>`` of this process, so files a container writes are owned
    by the submitting user; nothing where there are no POSIX ids"""
    if not hasattr(os, "getuid"):
        return []
    return ["--user", f"{os.getuid()}:{os.getgid()}"]


def split_entrypoint(xargs: list[str]) -> tuple[list[str], list[str] | None]:
    """``xargs`` without an ``--entrypoint`` option, and the entrypoint it names
    (``[]`` for an empty one), or ``None`` when it names none"""
    rest, entrypoint, i = [], None, 0
    while i < len(xargs):
        arg = xargs[i]
        if arg == "--entrypoint" and i + 1 < len(xargs):
            entrypoint, i = xargs[i + 1], i + 2
            continue
        if arg.startswith("--entrypoint="):
            entrypoint = arg.partition("=")[2]
        else:
            rest.append(arg)
        i += 1
    if entrypoint is None:
        return rest, None
    return rest, [entrypoint] if entrypoint else []


class WarmDocker(base.WarmContainer):
    """One ``docker run -d`` of the resolved reference, with the roots bound and a
    sleeper as entrypoint, then a ``docker exec`` per job (RFC 0005 section 3.1). Each
    exec runs the image's entrypoint (or the one ``xargs`` names) before the
    command, as a cold ``docker run`` does."""

    container_id: str | None = None
    entrypoint: list[str] = []

    def _start(self) -> None:
        env: Docker = self.environment
        xargs, override = split_entrypoint(env.xargs)
        cmd = ["docker", "run", "-d", "--rm", "--init", "--name", self.name]
        cmd.extend(["--label", f"{base.OWNER_LABEL}={base.owner_label()}"])
        cmd.extend(["--label", f"{base.UID_LABEL}={base.uid()}"])
        cmd.extend(user_flag())
        devices = base.visible_devices(self.device)
        if devices is not None:
            cmd.extend(["--gpus", "all" if devices == "all" else f'"device={devices}"'])
        if env.read_only:
            cmd.append("--read-only")
        for source, mounted, mode in self._mounts():
            cmd.extend(["-v", f"{source}:{mounted}:{mode}"])
        cmd.extend(xargs)
        cmd.extend(["--entrypoint", "sleep", self.ref.run, SLEEP_FOREVER])
        rc, out, err = base.execute(cmd, strip=True)
        if rc or not out:
            raise RuntimeError(
                f"'docker run -d' of {self.ref.run!r} failed ({rc}): {err}{out}"
            )
        self.container_id = out.splitlines()[-1]
        self.entrypoint = override if override is not None else self._entrypoint()

    def _entrypoint(self) -> list[str]:
        """The image's ``ENTRYPOINT``, read once the container has pulled it"""
        rc, out, err = base.execute(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{json .Config.Entrypoint}}",
                self.ref.run,
            ],
            strip=True,
        )
        if rc:
            raise RuntimeError(
                f"'docker image inspect' of {self.ref.run!r} failed ({rc}): {err}"
            )
        value = json.loads(out or "null")
        if isinstance(value, str):
            return [value]
        return list(value or [])

    def _exec_command(
        self, job: "Job", env: dict[str, str], args: list[str]
    ) -> list[str]:
        cmd = ["docker", "exec", "-w", self.environment.mounted(job.cache_dir)]
        for name, value in env.items():
            cmd.extend(["-e", f"{name}={value}"])
        return cmd + [self.container_id] + self.entrypoint + args

    def _stop(self) -> None:
        # removes the container after its processes are killed, and waits; by name
        # after a start that failed before its id was read
        base.execute(["docker", "rm", "-f", self.container_id or self.name])

    def _reap(self) -> None:
        """Remove this user's warm containers whose owner has ended in this host's
        pid namespace (`base.label_owner_dead`); never another user's"""
        rc, out, _ = base.execute(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                f"label={base.UID_LABEL}={base.uid()}",
                "--filter",
                f"label={base.OWNER_LABEL}",
                "--format",
                f'{{{{.ID}}}}\t{{{{.Label "{base.OWNER_LABEL}"}}}}',
            ],
            strip=True,
        )
        if rc:
            return
        for line in out.splitlines():
            container, _, label = line.partition("\t")
            if container and base.label_owner_dead(label):
                logger.info("Removing the orphaned warm container %s", container)
                base.execute(["docker", "rm", "-f", container])


# Alias so it can be referred to as docker.Environment
Environment = Docker
