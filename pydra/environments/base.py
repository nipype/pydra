import typing as ty
import os
from copy import copy
from pydra.utils.general import execute
from pathlib import Path
import logging
from fileformats.generic import FileSet
from pydra.utils.general import list_fields
from pydra.utils.typing import TypeParser

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.core import Job
    from pydra.engine.specs import ShellTask


class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is assumed that the environment, including container images,
    are available and are not removed at the end
    TODO: add setup and teardown methods
    """

    def setup(self):
        pass

    def execute(self, job: "Job[ShellTask]") -> dict[str, ty.Any]:
        """
        Execute the job in the environment.

        Parameters
        ----------
        job : TaskBase
            the job to execute

        Returns
        -------
        output: dict[str, Any]
            Output of the job.
        """
        raise NotImplementedError

    def teardown(self):
        pass


class Native(Environment):
    """
    Native environment, i.e. the tasks are executed in the current python environment.
    """

    def execute(self, job: "Job[ShellTask]") -> dict[str, ty.Any]:
        keys = ["return_code", "stdout", "stderr"]
        cmd_args = job.task._command_args(values=job.inputs)
        values = execute(cmd_args)
        output = dict(zip(keys, values))
        if output["return_code"]:
            msg = f"Error running '{job.name}' job with {cmd_args}:"
            if output["stderr"]:
                msg += "\n\nstderr:\n" + output["stderr"]
            if output["stdout"]:
                msg += "\n\nstdout:\n" + output["stdout"]
            raise RuntimeError(msg)
        return output


class Container(Environment):
    """
    Base class for container environments used by Docker and Singularity.

    Parameters
    ----------
    image : str
        Name of the container image
    tag : str
        Tag of the container image
    root : str
        Base path for mounting host directories into the container
    xargs : Union[str, List[str]]
        Extra arguments to be passed to the container
    """

    def __init__(self, image, tag="latest", root="/mnt/pydra", xargs=None):
        self.image = image
        self.tag = tag
        if xargs is None:
            xargs = []
        elif isinstance(xargs, str):
            xargs = xargs.split()
        self.xargs = xargs
        self.root = root

    def bind(self, loc, mode="ro"):
        loc_abs = Path(loc).absolute()
        return f"{loc_abs}:{self.root}{loc_abs}:{mode}"

    def get_bindings(
        self, job: "Job", root: str | None = None
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[Path, ...]]]:
        """Return bindings necessary to run job in an alternative root.

        This is primarily intended for contexts when a job is going
        to be run in a container with mounted volumes.

        Arguments
        ---------
        root: str, optional


        Returns
        -------
        bindings: dict
          Mapping from paths in the host environment to the target environment
        """
        from pydra.compose import shell

        bindings: dict[str, tuple[str, str]] = {}
        value_updates: dict[str, tuple[Path, ...]] = {}
        if root is None:
            return bindings
        fld: shell.arg
        for fld in list_fields(job.task):
            if TypeParser.contains_type(FileSet, fld.type):
                value: FileSet | None = job.inputs[fld.name]
                if not value:
                    continue

                copy_file = fld.copy_mode == FileSet.CopyMode.copy

                def map_path(fileset: os.PathLike | FileSet) -> Path:
                    host_path, env_path = fileset.parent, Path(
                        f"{root}{fileset.parent}"
                    )

                    # Default to mounting paths as read-only, but respect existing modes
                    bindings[host_path] = (
                        env_path,
                        "rw" if copy_file or isinstance(fld, shell.outarg) else "ro",
                    )
                    return (
                        env_path / fileset.name
                        if isinstance(fileset, os.PathLike)
                        else tuple(env_path / rel for rel in fileset.relative_fspaths)
                    )

                # Provide updated in-container paths to the command to be run. If a
                # fs-object, which resolves to a single path, just pass in the name of
                # that path relative to the location in the mount point in the container.
                # If it is a more complex file-set with multiple paths, then it is converted
                # into a tuple of paths relative to the base of the fileset.
                if TypeParser.matches(value, os.PathLike | FileSet):
                    value_updates[fld.name] = map_path(value)
                elif TypeParser.matches(value, ty.Sequence[FileSet | os.PathLike]):
                    mapped_value = []
                    for val in value:
                        mapped_val = map_path(val)
                        if isinstance(mapped_val, tuple):
                            mapped_value.extend(mapped_val)
                        else:
                            mapped_value.append(mapped_val)
                    value_updates[fld.name] = mapped_value
                else:
                    logger.debug(
                        "No support for generating bindings for %s types " "(%s)",
                        type(value),
                        value,
                    )

        # Add the cache directory to the list of mounts
        bindings[job.cache_dir] = (
            f"{self.root.rstrip('/')}{job.cache_dir.absolute()}",
            "rw",
        )

        # Update values with the new paths
        values = copy(job.inputs)
        values.update(value_updates)

        return bindings, values


class Docker(Container):
    """Docker environment."""

    def execute(self, job: "Job[ShellTask]") -> dict[str, ty.Any]:
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
        docker_args.extend(["-w", f"{self.root}{job.output_dir}"])
        keys = ["return_code", "stdout", "stderr"]

        job.output_dir.mkdir(exist_ok=True)
        values = execute(
            docker_args + [docker_img] + job.task._command_args(values=values),
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output


class Singularity(Container):
    """Singularity environment."""

    def execute(self, job: "Job[ShellTask]") -> dict[str, ty.Any]:
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
            ["--pwd", f"{self.root.rstrip('/')}{job.output_dir.absolute()}"]
        )
        keys = ["return_code", "stdout", "stderr"]

        job.output_dir.mkdir(exist_ok=True)
        values = execute(
            singularity_args
            + [singularity_img]
            + job.task._command_args(values=values),
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output
