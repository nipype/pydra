import typing as ty
import atexit
import os
import re
import signal
import socket
import json
import hashlib
import shutil
import tempfile
import threading
import time
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
from pydra.utils.storage import WARM_PREFIX
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
        *,
        writable: ty.Sequence[Path] = (),
        scratch: Path | None = None,
    ) -> "WarmEnvironment | None":
        """A started environment that runs many jobs, or ``None`` when the environment
        is not warmable or was not declared ``warm=True``.

        Parameters
        ----------
        roots : sequence of Path
            The host directories bound into the environment, read-only
        device : Device, optional
            The device its jobs use; a ``cuda`` device adds the runtime's GPU flags
        ref : ResolvedRef, optional
            The reference the jobs carry; a worker passes it and never resolves. When
            omitted the environment resolves it here
        writable : sequence of Path
            The roots also bound writable (a cache, store or work root); each is also
            in ``roots``
        scratch : Path, optional
            The directory under which the environment keeps its per-job ``HOME`` and
            its ``/tmp``; it must lie inside a writable root. The system's temporary
            directory by default
        """
        if not self.warm_requested:
            return None
        writable = frozenset(Path(r).absolute() for r in writable)
        handle = self._warm_handle(
            self.resolve() if ref is None else ref,
            frozenset(Path(r).absolute() for r in roots) | writable,
            device,
            writable,
            scratch,
        )
        if handle is not None:
            handle.start()
        return handle

    def _warm_handle(
        self,
        ref: ResolvedRef,
        roots: frozenset[Path],
        device: "Device | None",
        writable: frozenset[Path],
        scratch: Path | None,
    ) -> "WarmEnvironment | None":
        """The unstarted handle of a warmable environment, ``None`` if not warmable"""
        return None

    def binding_roots(self, job: "Job") -> frozenset[Path]:
        """The top-level host directories a warm environment must bind to run ``job``"""
        return frozenset()

    def writable_roots(self, job: "Job") -> frozenset[Path]:
        """Those of `binding_roots` a cold run of ``job`` binds writable"""
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

    def writable_roots(self, job: "Job") -> frozenset[Path]:
        """The top-level directories of the paths `get_bindings` binds writable"""
        mounts, _ = self.get_bindings(job=job, root=self.root)
        return frozenset(
            top_root(Path(host)) for host, (_, mode) in mounts.items() if mode == "rw"
        )

    @property
    def ro_root(self) -> str:
        """Where a warm container binds its roots read-only"""
        return f"{self.root.rstrip('/')}-ro"

    def mounted(self, host: Path, mode: str = "rw") -> str:
        """Where the host path ``host`` appears inside the container: under `root`, or
        under `ro_root` for ``mode="ro"`` in a warm container"""
        prefix = self.ro_root if mode == "ro" else self.root.rstrip("/")
        return f"{prefix}{Path(host).absolute()}"

    @staticmethod
    def binding_mode(fld: ty.Any) -> str:
        """``"rw"`` for a field whose files a job writes (copied in, or an output),
        ``"ro"`` otherwise"""
        copy_file = getattr(fld, "copy_mode", None) == FileSet.CopyMode.copy
        return "rw" if copy_file or isinstance(fld, shell.outarg) else "ro"

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

                mode = self.binding_mode(fld)

                def map_path(fileset: os.PathLike | FileSet) -> Path:
                    host_path, env_path = fileset.parent, Path(
                        f"{root}{fileset.parent}"
                    )

                    # Default to mounting paths as read-only, but respect existing modes
                    bindings[host_path] = (env_path, mode)
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


OWNER_LABEL = "pydra.warm.owner"
"""The label naming the process that started a warm container (`owner_label`)"""

UID_LABEL = "pydra.warm.uid"
"""The label naming the uid that started a warm container, which a listing filters on"""


def owner() -> tuple[str, int]:
    """``(host, pid)`` of this process, as a warm container's owner records it; the
    host has only letters, digits and ``-``"""
    host = re.sub(r"[^A-Za-z0-9-]", "-", socket.gethostname()) or "host"
    return host, os.getpid()


def uid() -> str:
    """This process's uid, ``""`` where there are no POSIX ids"""
    return str(os.getuid()) if hasattr(os, "getuid") else ""


def pid_namespace() -> str:
    """The identity of this process's pid namespace (the inode of
    ``/proc/self/ns/pid``), ``""`` where it cannot be read (macOS, Windows)"""
    try:
        return str(os.stat("/proc/self/ns/pid").st_ino)
    except OSError:
        return ""


def owner_label() -> str:
    """``<host>:<uid>:<pid namespace>:<pid>`` of this process, ``-`` for a pid
    namespace that cannot be read"""
    host, pid = owner()
    return f"{host}:{uid()}:{pid_namespace() or '-'}:{pid}"


def label_owner_dead(label: str) -> bool:
    """Whether the process an `owner_label` names has ended and was this user's, on
    this host, in this pid namespace. Where the namespace cannot be read (``-``) the
    host and uid must match only. A label in any other form is never dead."""
    parts = label.split(":")
    if len(parts) != 4:
        return False
    host, owner_uid, namespace, pid = parts
    if owner_uid != uid() or namespace != (pid_namespace() or "-"):
        return False
    return pid.isdigit() and owner_dead(host, int(pid))


def owner_dead(host: str, pid: int) -> bool:
    """Whether the owner ``host``:``pid`` is a process of this host that has ended"""
    if host != owner()[0]:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


_OPEN: "set[WarmEnvironment]" = set()
_OPEN_LOCK = threading.RLock()  # re-entrant: the SIGTERM handler may interrupt a holder
_HOOKED = False


def close_open_environments() -> None:
    """Close every warm environment this process has open, logging errors; run at exit
    and on ``SIGTERM`` (`_hook`)"""
    with _OPEN_LOCK:
        handles = list(_OPEN)
    for handle in handles:
        try:
            handle.close()
        except Exception as e:  # pragma: no cover - logged
            logger.warning("Closing the warm environment %s failed: %s", handle.key, e)


def _on_sigterm(signum, frame) -> None:
    close_open_environments()
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def _hook() -> None:
    """Close open warm environments at exit, and on ``SIGTERM`` when the process has
    no handler of its own (a pool worker's handler closes its resident state)"""
    global _HOOKED
    if _HOOKED:
        return
    _HOOKED = True
    atexit.register(close_open_environments)
    if threading.current_thread() is not threading.main_thread():
        return
    try:
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, None):
            signal.signal(signal.SIGTERM, _on_sigterm)
    except (ValueError, OSError, AttributeError):
        pass


class WarmEnvironment:
    """One started environment that runs many jobs (RFC 0005 section 3.1).

    A subclass starts the environment in `_start`, runs one job in `_execute` and
    stops it in `_stop`. The environment's ``setup`` and ``teardown`` are called on
    start and close. Warmth is execution detail: a job's checksum, record and context
    are those of a cold run, and so are its outputs. Open handles are closed at exit
    and on ``SIGTERM`` (`close_open_environments`).

    Attributes
    ----------
    environment : Environment
        The environment it was started from
    ref : ResolvedRef
        The reference it runs; it runs only jobs keyed by the same reference
    key : ResidentKey
        ``kind="environment"``, the environment's plugin name and the resolved key
    roots : frozenset[Path]
        The binding roots it was started with, bound read-only
    writable : frozenset[Path]
        Those of `roots` also bound writable
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
        writable: frozenset[Path] = frozenset(),
        scratch: Path | None = None,
    ):
        from pydra.engine.resident import ResidentKey

        self.environment = environment
        self.ref = ref
        self.writable = frozenset(writable)
        self.roots = frozenset(roots) | self.writable
        self.device = device
        self.key = ResidentKey("environment", environment.plugin_name(), ref.key)
        self.jobs_run = 0
        self.started = False
        self.closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the environment: `Environment.setup`, then the runtime. A start that
        fails removes what it may have left running"""
        try:
            self.environment.setup()
        except BaseException:
            self.closed = True
            self._cleanup()
            raise
        with _OPEN_LOCK:
            _OPEN.add(self)
        _hook()
        try:
            self._start()
        except BaseException:
            self.closed = True
            try:
                self._stop()
            except Exception as e:
                logger.debug("Removing a failed warm start of %s: %s", self.key, e)
            finally:
                self._finish()
            raise
        self.started = True

    @property
    def busy(self) -> bool:
        """Whether a job runs in it now"""
        return self._lock.locked()

    def covers(self, job: "Job") -> bool:
        """Whether ``job`` can run here: the same environment, keyed by the same
        reference, with its bindings inside `roots` and its writable bindings inside
        `writable`"""
        if self.closed or not self.started or job.environment != self.environment:
            return False
        if job.resolves_reference:
            resolved = job.resolved
            if (resolved.key, resolved.run) != (self.ref.key, self.ref.run):
                return False
        env = self.environment
        return (
            env.binding_roots(job) <= self.roots
            and env.writable_roots(job) <= self.writable
        )

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        """Run ``job`` as `Environment.execute` would; it must be `covers`-ed. Jobs
        run one at a time"""
        with self._lock:
            if not self.covers(job):
                raise ImageResolutionError(
                    f"The warm environment {self.key} cannot run the job "
                    f"{job.name!r}: another environment or reference, or a binding "
                    f"outside {sorted(map(str, self.roots))}"
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
            self._finish()

    def _finish(self) -> None:
        with _OPEN_LOCK:
            _OPEN.discard(self)
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
    """A started container. Every root is bound read-only under the environment's
    ``ro_root``, and a writable root also writable under its ``root``, so a job's
    inputs are read-only and its working directory and outputs writable, as a cold
    run binds them. A scratch directory (under ``scratch``, inside a writable root)
    holds each job's ``HOME``, removed when the job ends, and the container's
    ``/tmp``, emptied when each job ends, so no job sees what another wrote there.
    The container is labelled with its owner (`owner`); containers whose owner has
    ended on this host are removed at the next start (`_reap`). Subclasses give the
    start, exec, stop and reap commands."""

    def __init__(
        self,
        environment,
        ref,
        roots=frozenset(),
        device=None,
        writable=frozenset(),
        scratch=None,
    ):
        super().__init__(environment, ref, roots, device, writable, scratch)
        host, pid = owner()
        token = uuid.uuid4().hex[:12]
        self.name = f"pydra-warm-{pid}-{token}-{host}"
        parent = Path(scratch if scratch is not None else tempfile.gettempdir())
        self.scratch = parent.absolute() / f"{WARM_PREFIX}{token}"
        self.tmp = self.scratch / "tmp"
        root = top_root(self.scratch)
        self.roots = self.roots | {root}
        self.writable = self.writable | {root}

    def start(self) -> None:
        self.tmp.mkdir(parents=True, mode=0o700)
        try:
            self._reap()
        except Exception as e:
            logger.debug("Reaping orphaned warm containers failed: %s", e)
        super().start()

    def _cleanup(self) -> None:
        shutil.rmtree(self.scratch, ignore_errors=True)

    def _mounts(self) -> list[tuple[str, str, str]]:
        """``(host, container path, mode)`` of each bind, in a stable order: every
        root read-only, the writable roots writable, and the scratch ``/tmp``"""
        env = self.environment
        mounts = [(str(r), env.mounted(r, "ro"), "ro") for r in sorted(self.roots)]
        mounts += [(str(r), env.mounted(r), "rw") for r in sorted(self.writable)]
        mounts.append((str(self.tmp), "/tmp", "rw"))
        return mounts

    def _values(self, job: "Job") -> dict[str, ty.Any]:
        """The job's inputs with its paths in the container: read-only fields under
        ``ro_root``, writable ones (outputs, copies) under ``root``"""
        env = self.environment
        _, values = env.get_bindings(job=job, root=env.root)
        _, ro_values = env.get_bindings(job=job, root=env.ro_root)
        for fld in get_fields(job.task):
            if fld.name in ro_values and env.binding_mode(fld) == "ro":
                values[fld.name] = ro_values[fld.name]
        return values

    def _empty_tmp(self) -> None:
        for entry in self.tmp.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    def _execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        home = self.scratch / "jobs" / uuid.uuid4().hex / "home"
        home.mkdir(parents=True, mode=0o700)
        try:
            values = self._values(job)
            job.cache_dir.mkdir(parents=True, exist_ok=True)
            env = {"HOME": self.environment.mounted(home), "TMPDIR": "/tmp"}
            cmd = self._exec_command(job, env, job.task._command_args(values=values))
            output = dict(
                zip(
                    ["return_code", "stdout", "stderr"], execute(cmd, cwd=job.cache_dir)
                )
            )
        finally:
            shutil.rmtree(home.parent, ignore_errors=True)
            self._empty_tmp()
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

    def _reap(self) -> None:
        """Remove the warm containers whose owner has ended on this host"""


class WarmSession:
    """One environment's warm handle in a process (RFC 0005 section 3.1): started on the first
    job, restarted once when a job needs a binding root it lacks, or lacks writable, and after
    ``max_jobs`` jobs. A job needing more than ``max_mount_roots`` roots (scratch included), or
    keyed by another reference, runs through the cold `Environment.execute`. One job runs at a
    time.

    Parameters
    ----------
    environment : Environment
        The environment whose jobs it runs
    device : Device, optional
        The device the handle starts for; the process's device by default
    roots : sequence of Path
        Directories bound read-only at every start besides the jobs' (a worker
        class's mounts)
    writable : sequence of Path
        Directories bound writable at every start (the store root, the work root)
    scratch : Path, optional
        Where the handle keeps its scratch directory; the first job's cache root by
        default
    retry : bool
        Whether a failed start is tried again by a later job: `prepare` then raises
        the start's error (a pool worker types it a load fault), and a job whose
        start fails in `execute` runs cold alone. By default a failed start runs every
        later job of the session cold

    Attributes
    ----------
    starts : int
        How many times a handle was started
    cold_runs : int
        How many jobs ran cold
    start_s : float, optional
        The seconds the latest start took
    """

    def __init__(
        self,
        environment: Environment,
        device: "Device | None" = None,
        roots: ty.Sequence[Path] = (),
        writable: ty.Sequence[Path] = (),
        scratch: Path | None = None,
        *,
        retry: bool = False,
    ):
        if device is None:
            from pydra.engine.resident import process_device

            device = process_device()
        self.retry = retry
        self.environment = environment
        self.device = device
        self.base_writable = frozenset(top_root(Path(r)) for r in writable)
        self.base_roots = frozenset(top_root(Path(r)) for r in roots)
        self.base_roots |= self.base_writable
        self.scratch = scratch
        self.handle: WarmEnvironment | None = None
        self.starts = 0
        self.cold_runs = 0
        self.unwarmable = False
        self.start_s: float | None = None
        self._lock = threading.RLock()

    def accepts(self, job: "Job") -> bool:
        """Whether ``job``'s environment is this session's"""
        return job.environment == self.environment

    @property
    def busy(self) -> bool:
        """Whether a job runs in its handle now"""
        handle = self.handle
        return handle is not None and handle.busy

    def execute(self, job: "Job[shell.Task]") -> dict[str, ty.Any]:
        """Run ``job`` warm when the rules allow it, otherwise cold"""
        with self._lock:
            handle = self._handle_for(job)
            if handle is None:
                self.cold_runs += 1
            else:
                return handle.execute(job)
        return job.environment.execute(job)

    def prepare(self, jobs: ty.Sequence["Job"]) -> None:
        """Start the handle, or restart it once, for a group of jobs (a pool ticket):
        bound to the union of the binding roots of the jobs it accepts that are keyed
        by the first one's reference, so that the group needs no further restart. A
        handle that already covers them, is open and has jobs left before
        ``max_jobs`` is kept; a union beyond ``max_mount_roots`` starts nothing, and
        each job is then judged alone by `execute`. With ``retry``, a failed start
        raises its error."""
        with self._lock:
            env = self.environment
            if self.unwarmable or not env.warm_requested:
                return
            mine = [j for j in jobs if self.accepts(j)]
            if not mine:
                return
            ref = self._ref(mine[0])
            mine = [j for j in mine if self._ref(j) == ref]
            writable = frozenset().union(*(env.writable_roots(j) for j in mine))
            needed = frozenset().union(*(env.binding_roots(j) for j in mine))
            self._handle_for(mine[0], needed | writable, writable, raises=self.retry)

    def _ref(self, job: "Job") -> tuple[str, str]:
        ref = job.resolved if job.resolves_reference else self.environment.resolve()
        return ref.key, ref.run

    def _handle_for(
        self,
        job: "Job",
        needed: frozenset[Path] | None = None,
        writable: frozenset[Path] | None = None,
        raises: bool = False,
    ) -> WarmEnvironment | None:
        """The handle to run ``job`` on, started or restarted as the rules say, or
        ``None`` to run it cold. With ``needed`` (and ``writable``), the roots of a
        group of jobs (`prepare`): a handle covering them is kept and none is returned.
        With ``raises``, a failed start raises its error"""
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
        group = needed is not None
        if writable is None:
            writable = env.writable_roots(job)
        if needed is None:
            needed = env.binding_roots(job) | writable
        if handle is None:
            if self.scratch is None and isinstance(env, Container):
                self.scratch = Path(job.cache_root).absolute()
            roots, writable = self.base_roots | needed, self.base_writable | writable
            if isinstance(env, Container):
                scratch_root = top_root(
                    Path(self.scratch or tempfile.gettempdir()).absolute()
                )
                roots, writable = roots | {scratch_root}, writable | {scratch_root}
        else:
            spent = env.max_jobs is not None and handle.jobs_run >= env.max_jobs
            if group:
                if (
                    not spent
                    and not handle.closed
                    and needed <= handle.roots
                    and writable <= handle.writable
                ):
                    return None  # covered
            elif not spent and handle.covers(job):
                return handle
            roots, writable = handle.roots | needed, handle.writable | writable
        if len(roots) > env.max_mount_roots:
            logger.debug(
                "%s needs %d binding roots, more than max_mount_roots=%d: runs cold",
                job.name,
                len(roots),
                env.max_mount_roots,
            )
            return None
        self.close()
        began = time.monotonic()
        try:
            handle = env.warm(
                roots, self.device, ref=ref, writable=writable, scratch=self.scratch
            )
        except Exception as e:
            if raises:
                raise
            logger.warning(
                "The warm environment %s did not start, %s: %s",
                ref.key,
                "this job runs cold" if self.retry else "its jobs run cold",
                e,
            )
            self.unwarmable = not self.retry
            return None
        if handle is None:
            self.unwarmable = True
            return None
        self.handle = handle
        self.starts += 1
        self.start_s = time.monotonic() - began
        return handle if handle.covers(job) else None

    def close(self) -> None:
        """Close the handle, if any"""
        with self._lock:
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
