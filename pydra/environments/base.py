import typing as ty
import os
from copy import copy
import attrs
import subprocess as sp
from pathlib import Path
import logging
from fileformats.generic import FileSet
from pydra.compose import shell
from pydra.utils.general import get_fields, get_plugin_classes
from pydra.utils.typing import TypeParser
import pydra.environments

logger = logging.getLogger("pydra")

if ty.TYPE_CHECKING:
    from pydra.engine.job import Job


@attrs.define
class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is assumed that the environment, including container images,
    are available and are not removed at the end
    TODO: add setup and teardown methods
    """

    has_entrypoint = False

    def setup(self):
        pass

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
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

    @classmethod
    def available_plugins(cls) -> ty.Dict[str, ty.Type["Environment"]]:
        """Return all installed worker types"""
        return get_plugin_classes(pydra.environments, "Environment")

    @classmethod
    def plugin(cls, plugin_name: str) -> ty.Type["Environment"]:
        """Return a worker class by name."""
        try:
            return cls.available_plugins()[plugin_name.replace("-", "_")]
        except KeyError:
            raise ValueError(
                f"No environment matches {plugin_name!r}, check if there is a "
                f"plugin package called 'pydra-environments-{plugin_name}' that needs to be "
                "installed."
            )

    @classmethod
    def plugin_name(cls) -> str:
        """Return the name of the plugin."""
        try:
            plugin_name = cls._plugin_name
        except AttributeError:
            parts = cls.__module__.split(".")
            if parts[:-1] != ["pydra", "environments"]:
                raise ValueError(
                    f"Cannot infer plugin name of Environment (({cls}) from module path, "
                    f"as it isn't installed within `pydra.environments` ({cls.__module__}). "
                    "Please set the `_plugin_name` attribute on the class explicitly."
                )
            plugin_name = parts[-1]
        return plugin_name.replace("_", "-")


def split_if_str(s) -> list[str]:
    if isinstance(s, str):
        return s.split()
    elif not isinstance(s, list):
        return list(s)
    return s


@attrs.define
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

    has_entrypoint = True

    image: str
    tag: str = "latest"
    root: str = "/mnt/pydra"
    xargs: list[str] = attrs.field(factory=list, converter=split_if_str)

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

        bindings: dict[str, tuple[str, str]] = {}
        value_updates: dict[str, tuple[Path, ...]] = {}
        if root is None:
            return bindings
        fld: shell.arg
        for fld in get_fields(job.task):
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
        bindings[job.cache_root] = (
            f"{self.root.rstrip('/')}{job.cache_root.absolute()}",
            "rw",
        )

        # Update values with the new paths
        values = copy(job.inputs)
        values.update(value_updates)

        return bindings, values


def read_and_display(
    *cmd: str, strip: bool = False, hide_display: bool = False
) -> dict[str, int | str]:
    """Capture a process' standard output.

    Parameters
    ----------
    cmd : str
        The command to execute, as a list of strings.
    strip : bool, optional
        If True, the output will be stripped of leading and trailing whitespace.
    hide_display : bool, optional
        If True, the output will not be displayed.

    Returns
    -------
        dict[str, Any]
            A dictionary containing the return code, standard output, and standard error.

    Raises
    ------
        RuntimeError
            If the return code is not 0, a RuntimeError is raised with a formatted
            error message.
    """
    try:
        process = sp.run(cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    except Exception:
        # TODO editing some tracing?
        raise

    stdout = process.stdout.decode("utf-8")
    if strip:
        stdout = stdout.strip()
    stderr = process.stderr.decode("utf-8")
    if process.returncode:
        msg = f"Error executing command {' '.join(cmd)!r} (code: {process.returncode}):"
        if stdout:
            msg += "\n\nstderr:\n" + stderr
        if stdout:
            msg += "\n\nstdout:\n" + stdout
        raise RuntimeError(msg)
    return {"return_code": process.returncode, "stdout": stdout, "stderr": stderr}
