import typing as ty
import os
from .helpers import execute
from pathlib import Path
from fileformats.generic import FileSet
from pydra.engine.helpers import list_fields
from pydra.utils.typing import TypeParser


if ty.TYPE_CHECKING:
    from pydra.engine.core import Task
    from pydra.engine.specs import ShellDef


class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is assumed that the environment, including container images,
    are available and are not removed at the end
    TODO: add setup and teardown methods
    """

    def setup(self):
        pass

    def execute(self, task: "Task[ShellDef]") -> dict[str, ty.Any]:
        """
        Execute the task in the environment.

        Parameters
        ----------
        task : TaskBase
            the task to execute

        Returns
        -------
        output: dict[str, Any]
            Output of the task.
        """
        raise NotImplementedError

    def teardown(self):
        pass


class Native(Environment):
    """
    Native environment, i.e. the tasks are executed in the current python environment.
    """

    def execute(self, task: "Task[ShellDef]") -> dict[str, ty.Any]:
        keys = ["return_code", "stdout", "stderr"]
        values = execute(task.definition._command_args())
        output = dict(zip(keys, values))
        if output["return_code"]:
            msg = f"Error running '{task.name}' task with {task.definition._command_args()}:"
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
        self, task: "Task", root: str | None = None
    ) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[Path, ...]]]:
        """Return bindings necessary to run task in an alternative root.

        This is primarily intended for contexts when a task is going
        to be run in a container with mounted volumes.

        Arguments
        ---------
        root: str, optional


        Returns
        -------
        bindings: dict
          Mapping from paths in the host environment to the target environment
        """
        from pydra.design import shell

        bindings: dict[str, tuple[str, str]] = {}
        input_updates: dict[str, tuple[Path, ...]] = {}
        if root is None:
            return bindings
        fld: shell.arg
        for fld in list_fields(task.definition):
            if TypeParser.contains_type(FileSet, fld.type):
                fileset: FileSet | None = task.inputs[fld.name]
                if not fileset:
                    continue
                if not isinstance(fileset, (os.PathLike, FileSet)):
                    raise NotImplementedError(
                        f"No support for generating bindings for {type(fileset)} types "
                        f"({fileset})"
                    )
                copy = fld.copy_mode == FileSet.CopyMode.copy

                host_path, env_path = fileset.parent, Path(f"{root}{fileset.parent}")

                # Default to mounting paths as read-only, but respect existing modes
                bindings[host_path] = (
                    env_path,
                    "rw" if copy or isinstance(fld, shell.outarg) else "ro",
                )

                # Provide updated in-container paths to the command to be run. If a
                # fs-object, which resolves to a single path, just pass in the name of
                # that path relative to the location in the mount point in the container.
                # If it is a more complex file-set with multiple paths, then it is converted
                # into a tuple of paths relative to the base of the fileset.
                input_updates[fld.name] = (
                    env_path / fileset.name
                    if isinstance(fileset, os.PathLike)
                    else tuple(env_path / rel for rel in fileset.relative_fspaths)
                )

        # Add the cache directory to the list of mounts
        bindings[task.cache_dir] = (f"{self.root}/{task.cache_dir}", "rw")

        return bindings, input_updates


class Docker(Container):
    """Docker environment."""

    def execute(self, task: "Task[ShellDef]") -> dict[str, ty.Any]:
        docker_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts, input_updates = self.get_bindings(task=task, root=self.root)

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
        docker_args.extend(["-w", f"{self.root}{task.output_dir}"])
        keys = ["return_code", "stdout", "stderr"]

        values = execute(
            docker_args
            + [docker_img]
            + task.definition._command_args(
                root=self.root, input_updates=input_updates
            ),
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

    def execute(self, task: "Task[ShellDef]") -> dict[str, ty.Any]:
        singularity_img = f"{self.image}:{self.tag}"
        # mounting all input locations
        mounts, input_updates = self.get_bindings(task=task, root=self.root)

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
        singularity_args.extend(["--pwd", f"{self.root}{task.output_dir}"])
        keys = ["return_code", "stdout", "stderr"]

        values = execute(
            singularity_args
            + [singularity_img]
            + task.definition._command_args(
                root=self.root, input_updates=input_updates
            ),
        )
        output = dict(zip(keys, values))
        if output["return_code"]:
            if output["stderr"]:
                raise RuntimeError(output["stderr"])
            else:
                raise RuntimeError(output["stdout"])
        return output
