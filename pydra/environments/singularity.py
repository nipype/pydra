import shutil
import typing as ty
import logging
from pathlib import Path
import attrs
from pydra.compose import shell
from pydra.environments import base, images

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job
    from pydra.engine.resident import Device


def local_image(resolved: base.ResolvedRef) -> str:
    """The image to run for ``resolved``: a ``docker://`` digest's local SIF
    (`images.local_sif`), so a run never reaches a registry; a keyed ``.sif`` file
    after its digest here is compared with the key; anything else as carried"""
    expected = resolved.carried.get("sif_digest")
    if expected is not None:
        try:
            found = images.sif_digest(resolved.run)
        except OSError as e:
            raise base.ImageResolutionError(
                f"The image {resolved.run!r} the job was keyed by cannot be read "
                f"here: {e}"
            ) from e
        if found != expected:
            raise base.ImageResolutionError(
                f"The image {resolved.run!r} has the digest {found} here, but the "
                f"job was keyed by {expected}"
            )
    image = resolved.run
    _, digest = images.split_reference(image.partition("://")[2])
    if image.startswith("docker://") and digest:
        # exec of a docker:// URI asks the registry on every run, cached or not
        image = str(images.local_sif(image, digest))
    return image


def runtime() -> str:
    """``apptainer`` where it is installed, otherwise ``singularity``"""
    return "apptainer" if shutil.which("apptainer") else "singularity"


@attrs.define
class Singularity(base.Container):
    """Singularity/Apptainer environment.

    ``image`` is a ``docker://`` URI, a path to a ``.sif`` file, or another URI that
    Singularity pulls (``library://``, ``shub://``, ...)."""

    def _warm_handle(
        self,
        ref: base.ResolvedRef,
        roots: frozenset[Path],
        device: "Device | None",
        writable: frozenset[Path],
        scratch: Path | None,
    ) -> "WarmSingularity":
        return WarmSingularity(self, ref, roots, device, writable, scratch)

    def _ref_key(self) -> ty.Hashable:
        return ("singularity", self.image, self.tag, self.identity)

    def resolve(self) -> base.ResolvedRef:
        """Resolve the image once (RFC 0004 section 5.3). The image runs by ``run``.

        * ``docker://``: a digest given or ``identity=`` is keyed with no lookup,
          otherwise the registry is asked once per tag per run; the key is the Docker
          key of the same image (``docker@1:<digest>``) and the reference carried is
          ``docker://<repository>@<digest>``; a worker pulls it once into a local SIF
          (`images.local_sif`) and runs that, so a run never reaches a registry. A tag
          in the URI takes the place of ``tag``.
        * a ``.sif`` path is keyed by the sha256 of its content (``sif@1:sha256:<hex>``)
          and runs by its absolute path; before ``exec`` a worker compares the digest
          its own stat cache gives for the file (a stat once read) with the key. The key
          is exact for that file but not portable: a site that wants to share results
          names the ``docker://`` digest.
        * any other URI, a missing file or a directory is unresolved, and runs as before.
        """
        scheme, sep, rest = self.image.partition("://")
        if sep and scheme == "docker":
            return self._resolve_docker(rest)
        if sep:
            run = f"{self.image}:{self.tag}"
            return base.ResolvedRef(f"unresolved:singularity:{run}", run)
        if self.identity is not None:
            return base.ResolvedRef(f"sif@1:{self.identity}", self.image)
        path = Path(self.image)
        if not path.is_file():
            return base.ResolvedRef(f"unresolved:singularity:{self.image}", self.image)
        digest = images.sif_digest(path)
        run = str(path.resolve())  # exec runs in the job's directory
        return base.ResolvedRef(
            f"sif@1:{digest}", run, {"sif_digest": digest}, {"path": run}
        )

    def _resolve_docker(self, reference: str) -> base.ResolvedRef:
        repository, digest = images.split_reference(reference)
        name = reference.partition("@")[0]
        tag = name[len(repository) + 1 :] or self.tag
        carried = {"repository": repository}
        if digest:
            run = f"docker://{repository}@{digest}"
            return base.ResolvedRef(f"docker@1:{self.identity or digest}", run, carried)
        if self.identity is not None:
            run = f"docker://{repository}:{tag}"
            return base.ResolvedRef(f"docker@1:{self.identity}", run, carried)
        answer = images.registry_manifest(repository, tag)
        if answer is None:
            run = f"docker://{repository}:{tag}"
            return base.ResolvedRef(f"unresolved:singularity:{run}", run, carried)
        found, platform_digest = answer
        recorded = (
            {} if platform_digest is None else {"platform_digest": platform_digest}
        )
        return base.ResolvedRef(
            f"docker@1:{found}", f"docker://{repository}@{found}", carried, recorded
        )

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        if job.environment is not self:
            raise base.ImageResolutionError(
                f"{self!r} was asked to run a job built with the environment "
                f"{job.environment!r}, whose reference is the one carried with it"
            )
        # the reference the submitting process resolved and carried with the job
        image = local_image(job.resolved)
        # mounting all input locations
        mounts, values = self.get_bindings(job=job, root=self.root)

        # todo adding xargsy etc
        singularity_args = [
            "singularity",
            "exec",
            *self.xargs,
        ]
        for key, val in mounts.items():
            singularity_args.extend(["-B", f"{key}:{val[0]}:{val[1]}"])
        singularity_args.extend(
            ["--pwd", f"{self.root.rstrip('/')}{job.cache_dir.absolute()}"]
        )
        keys = ["return_code", "stdout", "stderr"]

        job.cache_dir.mkdir(parents=True, exist_ok=True)
        values = base.execute(
            singularity_args + [image] + job.task._command_args(values=values),
            cwd=job.cache_dir,
        )
        output = dict(zip(keys, values))
        # success is the exit code: Singularity writes progress, such as a cold pull's
        # "INFO: Converting OCI blobs to SIF format", to stderr
        if output["return_code"]:
            raise RuntimeError(
                f"singularity exited with exit code {output['return_code']}:\n"
                f"{output['stderr']}{output['stdout']}"
            )
        return output


class WarmSingularity(base.WarmContainer):
    """One ``instance start -c --no-home`` of the image the job would run cold (a ``docker://``
    digest's local SIF, never the URI), with the roots bound, then an ``exec`` in the instance per
    job (RFC 0005 section 3.1), with ``HOME`` and ``TMPDIR`` set by ``env``. The instance's name
    carries its owner's pid and host."""

    devices: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executable = runtime()

    def _start(self) -> None:
        image = local_image(self.ref)
        cmd = [self.executable, "instance", "start", "-c", "--no-home"]
        self.devices = base.visible_devices(self.device)
        if self.devices is not None:
            cmd.append("--nv")
        for source, mounted, mode in self._mounts():
            cmd.extend(["-B", f"{source}:{mounted}:{mode}"])
        cmd.extend(self.environment.xargs)
        cmd.extend([image, self.name])
        rc, out, err = base.execute(cmd)
        if rc:
            raise RuntimeError(
                f"'{self.executable} instance start' of {image!r} failed ({rc}): "
                f"{err}{out}"
            )

    def _exec_command(
        self, job: "Job", env: dict[str, str], args: list[str]
    ) -> list[str]:
        env = dict(env)
        if self.devices is not None and self.devices != "all":
            env["CUDA_VISIBLE_DEVICES"] = self.devices
        return [
            self.executable,
            "exec",
            "--pwd",
            self.environment.mounted(job.cache_dir),
            f"instance://{self.name}",
            "env",
            *(f"{name}={value}" for name, value in env.items()),
            *args,
        ]

    def _stop(self) -> None:
        base.execute([self.executable, "instance", "stop", self.name])

    def _reap(self) -> None:
        import json

        rc, out, _ = base.execute([self.executable, "instance", "list", "--json"])
        if rc:
            return
        try:
            instances = json.loads(out or "{}").get("instances") or []
        except (ValueError, AttributeError):
            return
        for instance in instances:
            name = instance.get("instance", "")
            parts = name.split("-", 4)
            if name.startswith("pydra-warm-") and len(parts) == 5:
                pid, host = parts[2], parts[4]
                if pid.isdigit() and base.owner_dead(host, int(pid)):
                    logger.info("Stopping the orphaned warm instance %s", name)
                    base.execute([self.executable, "instance", "stop", name])


# Alias so it can be referred to as singularity.Environment
Environment = Singularity
