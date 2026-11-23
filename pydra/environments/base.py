import typing as ty
import os
import json
import hashlib
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


class ImageResolutionError(RuntimeError):
    """An environment reference (a container image, Lmod modules) could not be resolved
    and ``resolve_images="require"``, or a worker cannot run the reference it was given
    """


@attrs.frozen
class ResolvedRef:
    """The reference an environment resolved once, in the submitting process.

    Attributes
    ----------
    key : str
        What identity records: ``docker@1:sha256:<digest>``, ``docker-local@1:<Id>``,
        ``sif@1:sha256:<hex>``, ``lmod@1:<full module names>``, or for a reference that
        could not be resolved ``unresolved:<kind>:<reference>`` (never in the form of a
        resolved key)
    run : str
        The reference the worker executes
    carried : dict
        What travels with the job but is not keyed, e.g. the repository to pull from
    recorded : dict
        Facts about what was resolved that the context records but neither keys nor
        compares (RFC 0004 section 3.2): a local image id, the per-platform manifest
        digest, the Lmod tree, ``MODULEPATH``, dependency modules
    """

    key: str
    run: str
    carried: dict = attrs.field(factory=dict, eq=False)
    recorded: dict = attrs.field(factory=dict, eq=False)

    @property
    def unresolved(self) -> bool:
        return self.key.startswith("unresolved:")


@attrs.define
class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is assumed that the environment, including container images,
    are available and are not removed at the end
    TODO: add setup and teardown methods
    """

    resolves: ty.ClassVar[bool] = False
    """Whether a shell job resolves the environment's reference once, carries it and
    runs it (`resolve`), rather than keying the environment's checksum"""

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

    def checksum(self) -> str:
        """Identity of the environment as declared, as a hex digest. A job keys the
        reference its environment resolves (`resolve`), which for a container is not
        this: a checksum never reaches Docker, a registry or the file system for it."""
        return hashlib.sha256(
            json.dumps(
                [type(self).__qualname__, self._identity()],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

    def _identity(self) -> ty.Any:
        """The JSON-serialisable value the checksum is derived from."""
        return attrs.asdict(self)

    def _ref_key(self) -> ty.Hashable | None:
        """What `resolve` depends on, or ``None`` when it should not be memoised. A
        ``Submitter`` resolves each distinct key once per run."""
        return None

    def resolve(self) -> ResolvedRef:
        """Resolve the environment's reference, once, in the submitting process. The
        job keys ``key`` and the worker executes ``run``. By default the environment's
        checksum is the key and there is nothing to run by reference."""
        return ResolvedRef(key=self.checksum(), run="")

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
    identity : str, optional
        User-supplied identity of the image (e.g. its digest), keyed as written instead
        of contacting a registry or hashing the image
    """

    image: str
    tag: str = "latest"
    root: str = "/mnt/pydra"
    xargs: list[str] = attrs.field(factory=list, converter=split_if_str)
    identity: str | None = None

    resolves: ty.ClassVar[bool] = True

    def _identity(self) -> ty.Any:
        # what was declared: resolving reaches a container runtime, the network or the
        # image file, and the job keys the resolved reference (`resolve`) instead
        if self.identity is not None:
            return {"identity": self.identity}
        return {"image": self.image, "tag": self.tag}

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


def execute(
    cmd: ty.Sequence[str],
    strip: bool = False,
    **kwargs: ty.Any,
) -> tuple[int, str, str]:
    """
    Run the event loop with coroutine.

    Uses :func:`read_and_display_async` unless a loop is
    already running, in which case :func:`read_and_display`
    is used.

    Parameters
    ----------
    cmd : :obj:`list` or :obj:`tuple`
        The command line to be executed.
    strip : :obj:`bool`
        Whether to strip the output strings. Default is ``False``.
    kwargs : keyword arguments
        Additional keyword arguments passed to the subprocess call.

    """
    rc, stdout, stderr = read_and_display(*cmd, strip=strip, **kwargs)
    return rc, stdout, stderr


def read_and_display(
    *cmd: str, strip: bool = False, **kwargs: ty.Any
) -> tuple[int, str, str]:
    """Capture a process' standard output."""
    try:
        process: sp.CompletedProcess = sp.run(
            cmd, stdout=sp.PIPE, stderr=sp.PIPE, **kwargs
        )
    except Exception:
        # TODO editing some tracing?
        raise

    if strip:
        return (
            process.returncode,
            process.stdout.decode("utf-8").strip(),
            process.stderr.decode("utf-8"),
        )
    else:
        return (
            process.returncode,
            process.stdout.decode("utf-8"),
            process.stderr.decode("utf-8"),
        )
