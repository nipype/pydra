import typing as ty
import os
import json
import hashlib
import shutil
import tempfile
import threading
import uuid
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
    from pydra.engine.resident import Device


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


EXECUTION_FIELDS = frozenset(
    {"warm_requested", "max_jobs", "max_mount_roots", "read_only"}
)
"""Fields that say how an environment runs jobs, not what runs: never in a checksum"""


@attrs.define
class Environment:
    """
    Base class for environments that are used to execute tasks.
    Right now it is assumed that the environment, including container images,
    are available and are not removed at the end. `setup` and `teardown` are called
    when a warm environment (`warm`) starts and closes.

    Parameters
    ----------
    warm : bool
        The author's statement that the tool keeps no state between invocations in
        the environment, so one started environment may serve many jobs (`warm`).
        Stored as ``warm_requested``, as ``warm`` is the method
    max_jobs : int, optional
        Restart a warm environment after this many jobs; ``None`` never restarts it
    max_mount_roots : int
        The most binding roots a warm environment is started with; a job needing more
        runs cold
    """

    warm_requested: bool = attrs.field(default=False, kw_only=True, alias="warm")
    max_jobs: int | None = attrs.field(default=None, kw_only=True)
    max_mount_roots: int = attrs.field(default=16, kw_only=True)

    resolves: ty.ClassVar[bool] = False
    """Whether a shell job resolves the environment's reference once, carries it and
    runs it (`resolve`), rather than keying the environment's checksum"""

    def setup(self):
        pass

    def warm(
        self,
        roots: ty.Sequence[Path] = (),
        device: "Device | None" = None,
        ref: ResolvedRef | None = None,
    ) -> "WarmEnvironment | None":
        """A started environment that runs many jobs, or ``None`` when the environment
        is not warmable or was not declared ``warm=True``.

        Parameters
        ----------
        roots : sequence of Path
            The host directories bound into the environment
        device : Device, optional
            The device its jobs use; a ``cuda`` device adds the runtime's GPU flags
        ref : ResolvedRef, optional
            The reference the jobs carry; a worker passes it and never resolves. When
            omitted the environment resolves it here
        """
        if not self.warm_requested:
            return None
        handle = self._warm_handle(
            self.resolve() if ref is None else ref, frozenset(roots), device
        )
        if handle is not None:
            handle.start()
        return handle

    def _warm_handle(
        self, ref: ResolvedRef, roots: frozenset[Path], device: "Device | None"
    ) -> "WarmEnvironment | None":
        """The unstarted handle of a warmable environment, ``None`` if not warmable"""
        return None

    def binding_roots(self, job: "Job") -> frozenset[Path]:
        """The top-level host directories a warm environment must bind to run ``job``"""
        return frozenset()

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
        """The JSON-serialisable value the checksum is derived from: the fields, less
        the ones that say how jobs run (`EXECUTION_FIELDS`)."""
        return {
            k: v for k, v in attrs.asdict(self).items() if k not in EXECUTION_FIELDS
        }

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

    def binding_roots(self, job: "Job") -> frozenset[Path]:
        """The top-level directories of the host paths `get_bindings` gives ``job``"""
        mounts, _ = self.get_bindings(job=job, root=self.root)
        return frozenset(top_root(Path(host)) for host in mounts)

    def mounted(self, host: Path) -> str:
        """Where the host path ``host`` appears inside the container"""
        return f"{self.root.rstrip('/')}{Path(host).absolute()}"

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


def top_root(path: Path) -> Path:
    """The top-level directory of ``path`` (``/data`` for ``/data/sub-01/anat``), the
    unit a warm environment binds"""
    path = Path(path).absolute()
    return Path(*path.parts[:2]) if len(path.parts) > 1 else path


def visible_devices(device: "Device | None") -> str | None:
    """The devices a ``cuda`` device names: the worker's ``CUDA_VISIBLE_DEVICES``, else
    the device's index, else ``"all"``; ``None`` for no device or a CPU"""
    if device is None or device.kind != "cuda":
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return visible
    return "all" if device.index is None else str(device.index)


class WarmEnvironment:
    """One started environment that runs many jobs (RFC 0005 section 3.1).

    A subclass starts the environment in `_start`, runs one job in `_execute` and
    stops it in `_stop`. The environment's ``setup`` and ``teardown`` are called on
    start and close. Warmth is execution detail: a job's checksum, record and context
    are those of a cold run, and so are its outputs.

    Attributes
    ----------
    environment : Environment
        The environment it was started from
    ref : ResolvedRef
        The reference it runs; it runs only jobs keyed by the same reference
    key : ResidentKey
        ``kind="environment"``, the environment's plugin name and the resolved key
    roots : frozenset[Path]
        The binding roots it was started with
    device : Device, optional
        The device it was started for
    jobs_run : int
        The jobs it has run
    """

    def __init__(
        self,
        environment: Environment,
        ref: ResolvedRef,
        roots: frozenset[Path] = frozenset(),
        device: "Device | None" = None,
    ):
        from pydra.engine.resident import ResidentKey

        self.environment = environment
        self.ref = ref
        self.roots = frozenset(roots)
        self.device = device
        self.key = ResidentKey("environment", environment.plugin_name(), ref.key)
        self.jobs_run = 0
        self.started = False
        self.closed = False

    def start(self) -> None:
        """Start the environment: `Environment.setup`, then the runtime"""
        self.environment.setup()
        try:
            self._start()
        except BaseException:
            self.closed = True
            self.environment.teardown()
            self._cleanup()
            raise
        self.started = True

    def covers(self, job: "Job") -> bool:
        """Whether ``job`` can run here: the same environment, keyed by the same
        reference, with its bindings inside `roots`"""
        if self.closed or not self.started or job.environment != self.environment:
            return False
        if job.resolves_reference:
            resolved = job.resolved
            if (resolved.key, resolved.run) != (self.ref.key, self.ref.run):
                return False
        return self.environment.binding_roots(job) <= self.roots

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        """Run ``job`` as `Environment.execute` would; it must be `covers`-ed"""
        if not self.covers(job):
            raise ImageResolutionError(
                f"The warm environment {self.key} cannot run the job {job.name!r}: "
                "another environment or reference, or a binding outside "
                f"{sorted(map(str, self.roots))}"
            )
        self.jobs_run += 1
        return self._execute(job)

    def close(self) -> None:
        """Stop the environment and wait for it, then `Environment.teardown`"""
        if self.closed:
            return
        self.closed = True
        try:
            if self.started:
                self._stop()
        finally:
            try:
                self.environment.teardown()
            finally:
                self._cleanup()

    def __enter__(self) -> "WarmEnvironment":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _start(self) -> None:
        pass

    def _execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        raise NotImplementedError

    def _stop(self) -> None:
        pass

    def _cleanup(self) -> None:
        pass


class WarmContainer(WarmEnvironment):
    """A started container. A scratch directory of its own is bound with `roots`, and
    each job runs with ``HOME`` and ``TMPDIR`` in a directory of the scratch that is
    removed when the job ends, so no job sees what another wrote there. Subclasses
    give the start, exec and stop commands."""

    def __init__(self, environment, ref, roots=frozenset(), device=None):
        super().__init__(environment, ref, roots, device)
        self.scratch = Path(tempfile.mkdtemp(prefix="pydra-warm-")).absolute()
        self.roots = self.roots | {top_root(self.scratch)}
        self.name = f"pydra-warm-{uuid.uuid4().hex[:12]}"

    def _cleanup(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _binds(self) -> list[tuple[str, str]]:
        """``(host root, container path)`` of each root, in a stable order"""
        return [(str(r), self.environment.mounted(r)) for r in sorted(self.roots)]

    def _execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        job_dir = self.scratch / uuid.uuid4().hex
        home, tmp = job_dir / "home", job_dir / "tmp"
        home.mkdir(parents=True, mode=0o700)
        tmp.mkdir(mode=0o700)
        try:
            _, values = self.environment.get_bindings(
                job=job, root=self.environment.root
            )
            job.cache_dir.mkdir(parents=True, exist_ok=True)
            env = {
                "HOME": self.environment.mounted(home),
                "TMPDIR": self.environment.mounted(tmp),
            }
            cmd = self._exec_command(job, env, job.task._command_args(values=values))
            output = dict(
                zip(
                    ["return_code", "stdout", "stderr"], execute(cmd, cwd=job.cache_dir)
                )
            )
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
        if output["return_code"]:
            raise RuntimeError(
                f"{self.key.name} exited with exit code {output['return_code']}:\n"
                f"{output['stderr']}{output['stdout']}"
            )
        return output

    def _exec_command(
        self, job: "Job", env: dict[str, str], args: list[str]
    ) -> list[str]:
        raise NotImplementedError


class WarmSession:
    """One environment's warm handle in a process, with the restart rules of RFC 0005
    section 3.1: start on the first job; restart once when a job needs a binding root
    the handle lacks, up to the environment's ``max_mount_roots``, and after
    ``max_jobs`` jobs; otherwise, and for a job keyed by another reference, run the
    job through the environment's cold `Environment.execute`. Correctness never
    depends on warmth.

    Parameters
    ----------
    environment : Environment
        The environment whose jobs it runs
    device : Device, optional
        The device the handle starts for; the process's device by default
    roots : sequence of Path
        Directories bound at every start besides the jobs' (a worker class's mounts,
        the store root, the work root)

    Attributes
    ----------
    starts : int
        How many times a handle was started
    cold_runs : int
        How many jobs ran cold
    """

    def __init__(
        self,
        environment: Environment,
        device: "Device | None" = None,
        roots: ty.Sequence[Path] = (),
    ):
        if device is None:
            from pydra.engine.resident import process_device

            device = process_device()
        self.environment = environment
        self.device = device
        self.base_roots = frozenset(top_root(Path(r)) for r in roots)
        self.handle: WarmEnvironment | None = None
        self.starts = 0
        self.cold_runs = 0
        self.unwarmable = False

    def accepts(self, job: "Job") -> bool:
        """Whether ``job``'s environment is this session's"""
        return job.environment == self.environment

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        """Run ``job`` warm when the rules allow it, otherwise cold"""
        handle = self._handle_for(job)
        if handle is None:
            self.cold_runs += 1
            return job.environment.execute(job)
        return handle.execute(job)

    def _handle_for(self, job: "Job") -> WarmEnvironment | None:
        env = self.environment
        if self.unwarmable or not env.warm_requested or not self.accepts(job):
            return None
        ref = job.resolved if job.resolves_reference else env.resolve()
        handle = self.handle
        if handle is not None and (handle.ref.key, handle.ref.run) != (
            ref.key,
            ref.run,
        ):
            return None  # one reference per session
        needed = env.binding_roots(job)
        if handle is None:
            roots = self.base_roots | needed
        else:
            spent = env.max_jobs is not None and handle.jobs_run >= env.max_jobs
            if not spent and handle.covers(job):
                return handle
            roots = handle.roots | needed
        if len(roots) > env.max_mount_roots:
            logger.debug(
                "%s needs %d binding roots, more than max_mount_roots=%d: runs cold",
                job.name,
                len(roots),
                env.max_mount_roots,
            )
            return None
        self.close()
        try:
            handle = env.warm(roots, self.device, ref=ref)
        except Exception as e:
            logger.warning(
                "The warm environment %s did not start, its jobs run cold: %s",
                ref.key,
                e,
            )
            self.unwarmable = True
            return None
        if handle is None:
            self.unwarmable = True
            return None
        self.handle = handle
        self.starts += 1
        return handle if handle.covers(job) else None

    def close(self) -> None:
        """Close the handle, if any"""
        handle, self.handle = self.handle, None
        if handle is not None:
            handle.close()

    def __enter__(self) -> "WarmSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_SERVING: list[WarmSession] = []
_SERVING_LOCK = threading.Lock()


class serving:
    """Within the block, shell jobs this process runs whose environment is a
    session's run through that session (`run_shell`): ``with serving(session):
    sub(task)``. The sessions are closed on exit unless ``close=False``."""

    def __init__(self, *sessions: WarmSession, close: bool = True):
        self.sessions = sessions
        self.close = close

    def __enter__(self) -> "serving":
        with _SERVING_LOCK:
            _SERVING.extend(self.sessions)
        return self

    def __exit__(self, *exc) -> None:
        with _SERVING_LOCK:
            for session in self.sessions:
                _SERVING.remove(session)
        if self.close:
            for session in self.sessions:
                session.close()


def run_shell(job: "Job[shell.Task]") -> dict[str, ty.Any]:
    """Run a shell job: through a serving session that accepts it, otherwise through
    its environment's `Environment.execute`"""
    if _SERVING:
        with _SERVING_LOCK:
            sessions = list(_SERVING)
        for session in sessions:
            if session.accepts(job):
                return session.execute(job)
    return job.environment.execute(job)
